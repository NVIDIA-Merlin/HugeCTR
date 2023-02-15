/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <base/debug/logger.hpp>
#include <boost/algorithm/string.hpp>
#include <hps/database_backend_detail.hpp>
#include <hps/hier_parameter_server_base.hpp>
#include <hps/redis_backend.hpp>
#include <hps/redis_backend_detail.hpp>
#include <iostream>
#include <random>
#include <thread_pool.hpp>
#include <unordered_set>

// TODO: Remove me!
#pragma GCC diagnostic error "-Wconversion"

namespace HugeCTR {

inline std::string make_hkey(const std::string& table_name, const size_t part, const char suffix) {
  std::ostringstream os;
  // These curly brackets (`{` and `}`) are not a design choice. Instead, this will trigger Redis to
  // align node allocations for 'v' and 't'.
  os << "hps_et{" << table_name << "/p" << part << '}' << suffix;
  return os.str();
}

#ifdef HCTR_DEFINE_REDIS_VALUE_HKEY_
#error HCTR_DEFINE_REDIS_VALUE_HKEY_ should not be defined!
#endif
#define HCTR_DEFINE_REDIS_VALUE_HKEY_() \
  const std::string& hkey_v { make_hkey(table_name, part_index, 'v') }

#ifdef HCTR_DEFINE_REDIS_META_HKEY_
#error HCTR_DEFINE_REDIS_META_HKEY_ should not be defined!
#endif
#define HCTR_DEFINE_REDIS_META_HKEY_() \
  const std::string& hkey_m { make_hkey(table_name, part_index, 't') }

#define HCTR_RETHROW_REDIS_ERRORS_(...)                             \
  do {                                                              \
    try {                                                           \
      __VA_ARGS__;                                                  \
    } catch (sw::redis::Error & e) {                                \
      throw DatabaseBackendError(get_name(), part_index, e.what()); \
    }                                                               \
  } while (0)

#define HCTR_PRINT_REDIS_ERRORS_(...)                                                           \
  do {                                                                                          \
    try {                                                                                       \
      __VA_ARGS__;                                                                              \
    } catch (sw::redis::Error & e) {                                                            \
      HCTR_LOG_C(ERROR, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index, \
                 ": Unrecoverable error during refresh: ", e.what(), '\n');                     \
    }                                                                                           \
  } while (0)

template <typename Key>
RedisClusterBackend<Key>::RedisClusterBackend(const RedisClusterBackendParams& params)
    : Base(params) {
  HCTR_CHECK(params.num_node_connections > 0);
  HCTR_CHECK(params.num_partitions >= params.num_node_connections);

  // Put together cluster configuration.
  sw::redis::ConnectionOptions options;

  {
    std::string host = params.address;

    const std::string::size_type comma_pos = host.find(',');
    if (comma_pos != std::string::npos) {
      host = host.substr(0, comma_pos);
    }

    const std::string::size_type colon_pos = host.find(':');
    if (colon_pos == std::string::npos) {
      options.host = host;
    } else {
      options.host = host.substr(0, colon_pos);
      options.port = std::stoi(host.substr(colon_pos + 1));
    }
  }
  options.user = params.user_name;
  options.password = params.password;
  options.keep_alive = true;

  // Enable TLS/SSL support.
  options.tls.enabled = params.enable_tls;
  if (std::filesystem::is_directory(params.ca_certificate)) {
    options.tls.cacertdir = params.ca_certificate;
  } else {
    options.tls.cacert = params.ca_certificate;
  }
  options.tls.cert = params.client_certificate;
  options.tls.key = params.client_key;
  options.tls.sni = params.server_name_identification;

  sw::redis::ConnectionPoolOptions pool_options;
  pool_options.size = params.num_node_connections;

  // Connect to cluster.
  HCTR_LOG_C(INFO, WORLD, get_name(), ": Connecting via ", options.host, ':', options.port,
             "...\n");
  redis_ = std::make_unique<sw::redis::RedisCluster>(options, pool_options);
}

template <typename Key>
RedisClusterBackend<Key>::~RedisClusterBackend() {
  HCTR_LOG_C(INFO, WORLD, get_name(), ": Awaiting background worker to conclude...\n");
  background_worker_.await_idle();

  HCTR_LOG_C(INFO, WORLD, get_name(), ": Disconnecting...\n");
  redis_.reset();
}

template <typename Key>
size_t RedisClusterBackend<Key>::size(const std::string& table_name) const {
  if (this->params_.num_partitions == 1) {
    constexpr size_t part_index{0};
    HCTR_DEFINE_REDIS_VALUE_HKEY_();

    HCTR_RETHROW_REDIS_ERRORS_({ return redis_->hlen(hkey_v); });
  } else {
    std::atomic<size_t> joint_num_entries{0};

    const size_t num_partitions{this->params_.num_partitions};
    HCTR_HPS_DB_PARALLEL_FOR_EACH_PART_({
      HCTR_RETHROW_REDIS_ERRORS_({
        HCTR_DEFINE_REDIS_VALUE_HKEY_();

        joint_num_entries += redis_->hlen(hkey_v);
      });
    });

    return joint_num_entries;
  }
}

template <typename Key>
size_t RedisClusterBackend<Key>::contains(const std::string& table_name, const size_t num_keys,
                                          const Key* const keys,
                                          const std::chrono::nanoseconds& time_budget) const {
  const auto begin{std::chrono::high_resolution_clock::now()};

  const Key* const keys_end{&keys[num_keys]};
  const size_t max_batch_size{this->params_.max_batch_size};
  const size_t num_partitions{this->params_.num_partitions};

  size_t hit_count{0};
  size_t skip_count{0};

  if (num_keys == 0) {
    // Do nothing ;-).
  } else if (num_keys == 1 || num_partitions == 1) {
    const size_t part_index{num_partitions == 1 ? 0 : HCTR_HPS_KEY_TO_PART_INDEX_(*keys)};

    HCTR_RETHROW_REDIS_ERRORS_({
      HCTR_DEFINE_REDIS_VALUE_HKEY_();

      // Step through keys batch-by-batch.
      std::chrono::nanoseconds elapsed;
      for (const Key* k{keys}; k != keys_end;) {
        HCTR_HPS_DB_CHECK_TIME_BUDGET_(SEQUENTIAL_DIRECT, nullptr);

        const size_t hit_count_prev{hit_count};
        const size_t batch_size{std::min<size_t>(keys_end - k, max_batch_size)};
        HCTR_HPS_REDIS_CONTAINS_(SEQUENTIAL_DIRECT);

        HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                   ", batch ", (k - keys - 1) / max_batch_size, ": ", hit_count - hit_count_prev,
                   " / ", batch_size, " hits. Time: ", elapsed.count(), " / ", time_budget.count(),
                   " ns.\n");
      }
    });
  } else {
    std::atomic<size_t> joint_hit_count{0};
    std::atomic<size_t> joint_skip_count{0};

    HCTR_HPS_DB_PARALLEL_FOR_EACH_PART_({
      HCTR_DEFINE_REDIS_VALUE_HKEY_();

      size_t hit_count{0};

      HCTR_RETHROW_REDIS_ERRORS_({
        // Step through keys batch-by-batch.
        std::chrono::nanoseconds elapsed;
        size_t num_batches{0};
        for (const Key* k{keys}; k != keys_end; ++num_batches) {
          HCTR_HPS_DB_CHECK_TIME_BUDGET_(PARALLEL_DIRECT, nullptr);

          const size_t hit_count_prev{hit_count};
          size_t batch_size{0};
          HCTR_HPS_REDIS_CONTAINS_(PARALLEL_DIRECT);

          HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                     ", batch ", num_batches, ": ", hit_count - hit_count_prev, " / ", batch_size,
                     " hits. Time: ", elapsed.count(), " / ", time_budget.count(), " ns.\n");
        }
      });

      joint_hit_count += hit_count;
    });

    hit_count += joint_hit_count;
    skip_count += joint_skip_count;
  }

  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ": ", hit_count, " / ",
             num_keys - skip_count, " hits, ", skip_count, " skipped.\n");
  return hit_count;
}

template <typename Key>
size_t RedisClusterBackend<Key>::insert(const std::string& table_name, const size_t num_pairs,
                                        const Key* const keys, const char* const values,
                                        const uint32_t value_size, const size_t value_stride) {
  HCTR_CHECK(value_size <= value_stride);

  const Key* const keys_end{&keys[num_pairs]};
  const size_t max_batch_size{this->params_.max_batch_size};
  const size_t num_partitions{this->params_.num_partitions};

  size_t num_inserts{0};

  if (num_pairs == 0) {
    // Do nothing ;-).
  } else if (num_pairs == 1 || num_partitions == 1) {
    const size_t part_index{num_partitions == 1 ? 0 : HCTR_HPS_KEY_TO_PART_INDEX_(*keys)};

    HCTR_RETHROW_REDIS_ERRORS_({
      HCTR_DEFINE_REDIS_VALUE_HKEY_();
      HCTR_DEFINE_REDIS_META_HKEY_();

      std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kv_views;
      std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> km_views;
      kv_views.reserve(std::min(num_pairs, max_batch_size));
      km_views.reserve(std::min(num_pairs, max_batch_size));

      for (const Key* k{keys}; k != keys_end;) {
        const size_t prev_num_inserts{num_inserts};
        const size_t batch_size{std::min<size_t>(keys_end - k, max_batch_size)};
        size_t part_size;
        if (!HCTR_HPS_REDIS_INSERT_(SEQUENTIAL_DIRECT)) {
          break;
        }

        HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                   ", batch ", (k - keys - 1) / max_batch_size, ": Inserted ",
                   num_inserts - prev_num_inserts, " + updated ",
                   batch_size - num_inserts + prev_num_inserts, " = ", batch_size, " entries.\n");

        // Handle overflow situations.
        if (part_size > this->params_.overflow_margin) {
          resolve_overflow_(table_name, part_index, part_size);
        }
      }
    });
  } else {
    std::atomic<size_t> joint_num_inserts{0};

    HCTR_HPS_DB_PARALLEL_FOR_EACH_PART_({
      HCTR_DEFINE_REDIS_VALUE_HKEY_();
      HCTR_DEFINE_REDIS_META_HKEY_();

      size_t num_inserts{0};

      HCTR_RETHROW_REDIS_ERRORS_({
        std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kv_views;
        std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> km_views;
        kv_views.reserve(std::min(num_pairs / num_partitions, max_batch_size));
        km_views.reserve(std::min(num_pairs / num_partitions, max_batch_size));

        size_t num_batches{0};
        for (const Key* k{keys}; k != keys_end; ++num_batches) {
          const size_t prev_num_inserts{num_inserts};
          size_t batch_size{0};
          size_t part_size;
          if (!HCTR_HPS_REDIS_INSERT_(PARALLEL_DIRECT)) {
            break;
          }

          HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                     ", batch ", num_batches, ": Inserted ", num_inserts - prev_num_inserts,
                     " + updated ", kv_views.size() - num_inserts + prev_num_inserts, " = ",
                     kv_views.size(), " entries.\n");

          // Handle overflow situations.
          if (part_size > this->params_.overflow_margin) {
            resolve_overflow_(table_name, part_index, part_size);
          }
        }
      });

      joint_num_inserts += num_inserts;
    });

    num_inserts += joint_num_inserts;
  }

  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ": Inserted ", num_inserts,
             " + updated ", num_pairs - num_inserts, " = ", num_pairs, " entries.\n");
  return num_inserts;
}

template <typename Key>
size_t RedisClusterBackend<Key>::fetch(const std::string& table_name, const size_t num_keys,
                                       const Key* const keys, char* const values,
                                       const size_t value_stride,
                                       const DatabaseMissCallback& on_miss,
                                       const std::chrono::nanoseconds& time_budget) {
  const auto begin{std::chrono::high_resolution_clock::now()};

  const Key* const keys_end{&keys[num_keys]};
  const size_t max_batch_size{this->params_.max_batch_size};
  const size_t num_partitions{this->params_.num_partitions};

  size_t miss_count{0};
  size_t skip_count{0};

  if (num_keys == 0) {
    // Do nothing ;-).
  } else if (num_keys == 1 || num_partitions == 1) {
    const size_t part_index{num_partitions == 1 ? 0 : HCTR_HPS_KEY_TO_PART_INDEX_(*keys)};

    HCTR_RETHROW_REDIS_ERRORS_({
      HCTR_DEFINE_REDIS_VALUE_HKEY_();

      std::shared_ptr<std::vector<Key>> touched_keys;
      HCTR_HPS_REDIS_FETCH_DEFINE_V_VIEWS();
      std::vector<sw::redis::StringView> k_views;
      k_views.reserve(std::min(num_keys, max_batch_size));

      // Step through input batch-by-batch.
      std::chrono::nanoseconds elapsed;
      for (const Key* k{keys}; k != keys_end;) {
        HCTR_HPS_DB_CHECK_TIME_BUDGET_(SEQUENTIAL_DIRECT, on_miss);

        const size_t prev_miss_count{miss_count};
        const size_t batch_size{std::min<size_t>(keys_end - k, max_batch_size)};
        if (!HCTR_HPS_REDIS_FETCH_(SEQUENTIAL_DIRECT)) {
          break;
        }

        HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                   ", batch ", (k - keys - 1) / max_batch_size, ": ",
                   batch_size - miss_count + prev_miss_count, " / ", batch_size,
                   " hits. Time: ", elapsed.count(), " / ", time_budget.count(), " ns.\n");

        // Refresh metadata if required.
        if (touched_keys && !touched_keys->empty()) {
          queue_metadata_refresh_(table_name, part_index, std::move(touched_keys));
        }
      }
    });
  } else {
    std::atomic<size_t> joint_miss_count{0};
    std::atomic<size_t> joint_skip_count{0};

    HCTR_HPS_DB_PARALLEL_FOR_EACH_PART_({
      size_t miss_count{0};

      HCTR_RETHROW_REDIS_ERRORS_({
        HCTR_DEFINE_REDIS_VALUE_HKEY_();

        std::shared_ptr<std::vector<Key>> touched_keys;
        HCTR_HPS_REDIS_FETCH_DEFINE_V_VIEWS();
        std::vector<sw::redis::StringView> k_views;
        k_views.reserve(std::min(num_keys / num_partitions, max_batch_size));

        // Step through input batch-by-batch.
        std::chrono::nanoseconds elapsed;
        size_t num_batches{0};
        for (const Key* k{keys}; k != keys_end; ++num_batches) {
          HCTR_HPS_DB_CHECK_TIME_BUDGET_(PARALLEL_DIRECT, on_miss);

          const size_t prev_miss_count{miss_count};
          size_t batch_size{0};
          if (!HCTR_HPS_REDIS_FETCH_(PARALLEL_DIRECT)) {
            break;
          }

          HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                     ", batch ", num_batches, ": ", batch_size - miss_count + prev_miss_count,
                     " / ", batch_size, " hits. Time: ", elapsed.count(), " / ",
                     time_budget.count(), " ns.\n");
        }

        // Refresh metadata if required.
        if (touched_keys && !touched_keys->empty()) {
          queue_metadata_refresh_(table_name, part_index, std::move(touched_keys));
        }
      });

      joint_miss_count += miss_count;
    });

    miss_count += joint_miss_count;
    skip_count += joint_skip_count;
  }

  const size_t hit_count{num_keys - skip_count - miss_count};
  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ": ", hit_count, " / ",
             num_keys - skip_count, " hits; skipped ", skip_count, " keys.\n");
  return hit_count;
}

template <typename Key>
size_t RedisClusterBackend<Key>::fetch(const std::string& table_name, const size_t num_indices,
                                       const size_t* const indices, const Key* const keys,
                                       char* const values, const size_t value_stride,
                                       const DatabaseMissCallback& on_miss,
                                       const std::chrono::nanoseconds& time_budget) {
  const auto begin{std::chrono::high_resolution_clock::now()};

  const size_t* const indices_end{&indices[num_indices]};
  const size_t max_batch_size{this->params_.max_batch_size};
  const size_t num_partitions{this->params_.num_partitions};

  size_t miss_count{0};
  size_t skip_count{0};

  if (num_indices == 0) {
    // Do nothing ;-).
  } else if (num_indices == 1 || num_partitions == 1) {
    const size_t part_index{num_partitions == 1 ? 0 : HCTR_HPS_KEY_TO_PART_INDEX_(keys[*indices])};

    HCTR_RETHROW_REDIS_ERRORS_({
      HCTR_DEFINE_REDIS_VALUE_HKEY_();

      std::shared_ptr<std::vector<Key>> touched_keys;
      HCTR_HPS_REDIS_FETCH_DEFINE_V_VIEWS();
      std::vector<sw::redis::StringView> k_views;
      k_views.reserve(std::min(num_indices, max_batch_size));

      // Step through input batch-by-batch.
      std::chrono::nanoseconds elapsed;
      for (const size_t* i{indices}; i != indices_end;) {
        HCTR_HPS_DB_CHECK_TIME_BUDGET_(SEQUENTIAL_INDIRECT, on_miss);

        const size_t prev_miss_count{miss_count};
        const size_t batch_size{std::min<size_t>(indices_end - i, max_batch_size)};
        if (!HCTR_HPS_REDIS_FETCH_(SEQUENTIAL_INDIRECT)) {
          break;
        }

        HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                   ", batch ", (i - indices - 1) / max_batch_size, ": ",
                   batch_size - miss_count + prev_miss_count, " / ", batch_size,
                   " hits. Time: ", elapsed.count(), " / ", time_budget.count(), " ns.\n");
      }

      // Refresh metadata if required.
      if (touched_keys && !touched_keys->empty()) {
        queue_metadata_refresh_(table_name, part_index, std::move(touched_keys));
      }
    });
  } else {
    std::atomic<size_t> joint_miss_count{0};
    std::atomic<size_t> joint_skip_count{0};

    HCTR_HPS_DB_PARALLEL_FOR_EACH_PART_({
      size_t miss_count{0};

      HCTR_RETHROW_REDIS_ERRORS_({
        HCTR_DEFINE_REDIS_VALUE_HKEY_();

        std::shared_ptr<std::vector<Key>> touched_keys;
        HCTR_HPS_REDIS_FETCH_DEFINE_V_VIEWS();
        std::vector<sw::redis::StringView> k_views;
        k_views.reserve(std::min(num_indices / num_partitions, max_batch_size));

        // Step through input batch-by-batch.
        std::chrono::nanoseconds elapsed;
        size_t num_batches{0};
        for (const size_t* i{indices}; i != indices_end; ++num_batches) {
          HCTR_HPS_DB_CHECK_TIME_BUDGET_(PARALLEL_INDIRECT, on_miss);

          // Assemble query.
          const size_t prev_miss_count{miss_count};
          size_t batch_size{0};
          if (!HCTR_HPS_REDIS_FETCH_(PARALLEL_INDIRECT)) {
            break;
          }

          HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                     ", batch ", num_batches, ": ", batch_size - miss_count + prev_miss_count,
                     " / ", batch_size, " hits. Time: ", elapsed.count(), " / ",
                     time_budget.count(), " ns.\n");
        }

        // Refresh metadata if required.
        if (touched_keys && !touched_keys->empty()) {
          queue_metadata_refresh_(table_name, part_index, std::move(touched_keys));
        }
      });

      joint_miss_count += miss_count;
    });

    miss_count += joint_miss_count;
    skip_count += joint_skip_count;
  }

  const size_t hit_count{num_indices - skip_count - miss_count};
  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ": ", hit_count, " / ",
             num_indices - skip_count, " hits; skipped ", skip_count, " keys.\n");
  return hit_count;
}

template <typename Key>
size_t RedisClusterBackend<Key>::evict(const std::string& table_name) {
  const auto evict_part = [&](const size_t part_index) -> size_t {
    HCTR_DEFINE_REDIS_VALUE_HKEY_();
    HCTR_DEFINE_REDIS_META_HKEY_();

    HCTR_RETHROW_REDIS_ERRORS_({
      sw::redis::Pipeline pipe{redis_->pipeline(hkey_v, false)};
      pipe.hlen(hkey_v);
      pipe.del(hkey_v);
      pipe.del(hkey_m);

      sw::redis::QueuedReplies replies{pipe.exec()};
      return replies.get<long long>(0);
    });
  };

  size_t num_deletions{0};

  if (this->params_.num_partitions == 1) {
    num_deletions += evict_part(0);
  } else {
    std::atomic<size_t> joint_num_deletions{0};

    const size_t num_partitions{this->params_.num_partitions};
    HCTR_HPS_DB_PARALLEL_FOR_EACH_PART_({
      const size_t num_deletions{evict_part(part_index)};

      HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                 ": Erased ", num_deletions, " entries.\n");

      joint_num_deletions += num_deletions;
    });

    num_deletions += joint_num_deletions;
  }

  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ": Erased ", num_deletions,
             " entries.\n");
  return num_deletions;
}

template <typename Key>
size_t RedisClusterBackend<Key>::evict(const std::string& table_name, const size_t num_keys,
                                       const Key* const keys) {
  const Key* const keys_end{&keys[num_keys]};
  const size_t max_batch_size{this->params_.max_batch_size};
  const size_t num_partitions{this->params_.num_partitions};

  size_t num_deletions{0};

  if (num_keys == 0) {
    // Do nothing ;-).
  } else if (num_keys == 1 || num_partitions == 1) {
    const size_t part_index{num_partitions == 1 ? 0 : HCTR_HPS_KEY_TO_PART_INDEX_(*keys)};

    HCTR_RETHROW_REDIS_ERRORS_({
      HCTR_DEFINE_REDIS_VALUE_HKEY_();
      HCTR_DEFINE_REDIS_META_HKEY_();

      std::vector<sw::redis::StringView> k_views;
      k_views.reserve(std::min(num_keys, max_batch_size));

      for (const Key* k{keys}; k != keys_end;) {
        const size_t prev_num_deletions{num_deletions};
        const size_t batch_size{std::min<size_t>(keys_end - k, max_batch_size)};
        if (!HCTR_HPS_REDIS_EVICT_(SEQUENTIAL_DIRECT)) {
          break;
        }

        HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                   ", batch ", (k - keys - 1) / max_batch_size, ": Erased ",
                   num_deletions - prev_num_deletions, " / ", batch_size, " entries.\n");
      }
    });
  } else {
    std::atomic<size_t> joint_num_deletions{0};

    HCTR_HPS_DB_PARALLEL_FOR_EACH_PART_({
      size_t num_deletions{0};

      HCTR_RETHROW_REDIS_ERRORS_({
        HCTR_DEFINE_REDIS_VALUE_HKEY_();
        HCTR_DEFINE_REDIS_META_HKEY_();

        std::vector<sw::redis::StringView> k_views;
        k_views.reserve(std::min(num_keys / num_partitions, max_batch_size));

        size_t num_batches{0};
        for (const Key* k{keys}; k != keys_end; ++num_batches) {
          const size_t prev_num_deletions{num_deletions};
          size_t batch_size{0};
          if (!HCTR_HPS_REDIS_EVICT_(PARALLEL_DIRECT)) {
            break;
          }

          HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                     ", batch ", num_batches, ": Erased ", num_deletions - prev_num_deletions,
                     " / ", batch_size, " entries.\n");
        }
      });

      joint_num_deletions += num_deletions;
    });

    num_deletions += joint_num_deletions;
  }

  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ": Erased ", num_deletions,
             " / ", num_keys, " entries.\n");
  return num_deletions;
}

template <typename Key>
std::vector<std::string> RedisClusterBackend<Key>::find_tables(const std::string& model_name) {
  // Determine partition 0 key pattern.
  std::string table_name_pattern = HierParameterServerBase::make_tag_name(model_name, "", false);
  boost::replace_all(table_name_pattern, ".", "\\.");
  table_name_pattern += "*";
  // const std::string& hkey_pattern = make_hkey(table_name_pattern, 0, 'v');

  // Find suitable keys.
  std::vector<std::string> part_names;
  auto reply = redis_->command(sw::redis::cmd::keys, table_name_pattern);
  sw::redis::reply::to_array(*reply, std::back_inserter(part_names));

  // Turn partition numes into table names.
  std::unordered_set<std::string> unique_table_names;
  for (const std::string& part_name : part_names) {
    // Pattern: '{' << table_name << "/p" << partition << '}' << suffix;
    if (part_name.find("hps_et{") != 0) {
      continue;
    }

    size_t end = part_name.find_last_of('/');
    if (end == 0 || end == std::string::npos) {
      continue;
    }

    unique_table_names.emplace(part_name.substr(1, end - 1));
  }

  std::vector<std::string> table_names;
  table_names.insert(table_names.end(), unique_table_names.begin(), unique_table_names.end());
  return table_names;
}

template <typename Key>
std::vector<Key> RedisClusterBackend<Key>::keys(const std::string& table_name) const {
  std::vector<Key> keys;
  for (size_t part_index{0}; part_index < this->params_.num_partitions; ++part_index) {
    HCTR_DEFINE_REDIS_VALUE_HKEY_();
    redis_->hkeys(hkey_v, RedisKeyVectorInserter(keys));
  }
  return keys;
}

template <typename Key>
uint32_t RedisClusterBackend<Key>::value_size_for(const std::string& table_name) const {
  for (size_t part_index{0}; part_index < this->params_.num_partitions; ++part_index) {
    HCTR_DEFINE_REDIS_VALUE_HKEY_();

    // Find a valid key (if existing).
    std::vector<Key> keys;
    redis_->hkeys(hkey_v, RedisKeyVectorInserter(keys));
    if (keys.empty()) {
      continue;
    }

    // Fetch values until we got a "still" valid entry.
    for (const Key& key : keys) {
      const sw::redis::Optional<std::string> v_view{
          redis_->hget(hkey_v, {reinterpret_cast<const char*>(&key), sizeof(Key)})};
      if (v_view) {
        const uint32_t size{static_cast<uint32_t>(v_view->size())};
        HCTR_CHECK(size == v_view->size());
        return size;
      }
    }
  }

  return 0;
}

template <typename Key>
size_t RedisClusterBackend<Key>::dump_bin(const std::string& table_name, std::ofstream& file) {
  const size_t max_batch_size{this->params_.max_batch_size};
  const size_t num_partitions{this->params_.num_partitions};

  std::mutex mutex;

  uint32_t value_size{0};
  size_t num_entries{0};

  auto write_part = [&](const size_t part_index) {
    HCTR_RETHROW_REDIS_ERRORS_({
      HCTR_DEFINE_REDIS_VALUE_HKEY_();

      std::vector<Key> keys;
      redis_->hkeys(hkey_v, RedisKeyVectorInserter(keys));

      std::vector<sw::redis::StringView> k_views;
      k_views.reserve(std::min(keys.size(), max_batch_size));

      // Step through key views batch-by-batch.
      for (auto k_it{keys.begin()}; k_it != keys.end();) {
        const size_t batch_size{std::min<size_t>(keys.end() - k_it, max_batch_size)};

        k_views.clear();
        for (const auto batch_end{k_it + batch_size}; k_it != batch_end; ++k_it) {
          k_views.emplace_back(reinterpret_cast<const char*>(&*k_it), sizeof(Key));
        }

        const std::lock_guard lock(mutex);
        redis_->hmget(hkey_v, k_views.begin(), k_views.end(),
                      RedisBinFileInserter<Key>(k_views, value_size, file, num_entries));
      }
    });
  };

  if (num_partitions == 1) {
    write_part(0);
  } else {
    HCTR_HPS_DB_PARALLEL_FOR_EACH_PART_({ write_part(part_index); });
  }

  return num_entries;
}

template <typename Key>
size_t RedisClusterBackend<Key>::dump_sst(const std::string& table_name,
                                          rocksdb::SstFileWriter& file) {
  // Fetch all keys and establish global order (required by SST file writer!).
  std::vector<Key> keys{this->keys(table_name)};
  std::sort(keys.begin(), keys.end());

  // Step through key views batch-by-batch.
  const uint32_t value_size{value_size_for(table_name)};
  const size_t max_batch_size{this->params_.max_batch_size *
                              std::min<size_t>(this->params_.num_partitions, 2)};
  HCTR_CHECK(max_batch_size > 0);
  std::vector<char> values(max_batch_size * value_size);

  for (auto k_it{keys.begin()}; k_it != keys.end();) {
    const size_t batch_size{std::min<size_t>(keys.end() - k_it, max_batch_size)};

    // Read a batch.
    fetch(
        table_name, batch_size, &*k_it, values.data(), value_size,
        [&](const size_t index) { std::fill_n(&values[index * value_size], value_size, 0); },
        std::chrono::nanoseconds::zero());

    // Write to SST.
    for (const auto batch_end{k_it + batch_size}; k_it != batch_end; ++k_it) {
      HCTR_ROCKSDB_CHECK(file.Put({reinterpret_cast<const char*>(&*k_it), sizeof(Key)},
                                  {&values[(k_it - keys.begin()) * value_size], value_size}));
    }
  }

  return keys.size();
}

template <typename Key>
void RedisClusterBackend<Key>::resolve_overflow_(const std::string& table_name,
                                                 const size_t part_index, size_t part_size) {
  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
             " is overflowing (size = ", part_size, " > ", this->params_.overflow_margin,
             "). Attempting to resolve...\n");

  const size_t max_batch_size{this->params_.max_batch_size};

  HCTR_DEFINE_REDIS_VALUE_HKEY_();
  HCTR_DEFINE_REDIS_META_HKEY_();

  const auto delete_batch = [&](const std::vector<sw::redis::StringView>& k_views) {
    sw::redis::Pipeline pipe{redis_->pipeline(hkey_m, false)};
    pipe.hdel(hkey_m, k_views.begin(), k_views.end());
    pipe.hdel(hkey_v, k_views.begin(), k_views.end());
    pipe.hlen(hkey_v);

    sw::redis::QueuedReplies replies{pipe.exec()};
    part_size = std::max(replies.get<long long>(replies.size() - 1), 0LL);
  };

  switch (this->params_.overflow_policy) {
    case DatabaseOverflowPolicy_t::EvictRandom: {
      // Fetch all keys in partition.
      std::vector<Key> keys;
      keys.reserve(part_size);
      redis_->hkeys(hkey_m, RedisKeyVectorInserter(keys));

      part_size = keys.size();
      if (part_size <= this->overflow_resolution_margin_) {
        HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                   ": Overflow was already resolved by another process.\n");
        return;
      }

      // Randomly shuffle the keys.
      {
        std::random_device rd;
        std::default_random_engine gen(rd());
        std::shuffle(keys.begin(), keys.end(), gen);
      }

      // Delete entries in batches until overflow condition is no longer fulfilled.
      std::vector<sw::redis::StringView> k_views;
      k_views.reserve(std::min(part_size, max_batch_size));

      for (auto k_it{keys.begin()}; k_it != keys.end();) {
        const size_t batch_size{std::min<size_t>(keys.end() - k_it, max_batch_size)};

        HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                   " (size = ", part_size, "). Attempting to evict ", batch_size,
                   " RANDOM key/value pairs.\n");

        // Assemble and launch query.
        k_views.clear();
        for (const auto batch_end{k_it + batch_size}; k_it != batch_end; ++k_it) {
          k_views.emplace_back(reinterpret_cast<const char*>(&*k_it), sizeof(Key));
        }
        delete_batch(k_views);
        if (part_size <= this->overflow_resolution_margin_) {
          break;
        }
      }
    } break;

    case DatabaseOverflowPolicy_t::EvictLeastUsed: {
      // Fetch keys and parse all metadata.
      std::vector<std::pair<Key, long long>> keys_metas;
      keys_metas.reserve(part_size);
      redis_->hgetall(hkey_m, RedisKeyAccumulatorVectorInserter<Key>(keys_metas));

      part_size = keys_metas.size();
      if (part_size <= this->overflow_resolution_margin_) {
        HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                   ": Overflow was already resolved by another process.\n");
        return;
      }

      // Sort by ascending by number of accesses.
      std::sort(keys_metas.begin(), keys_metas.end(),
                [](const auto& km0, const auto& km1) { return km0.second < km1.second; });

      // Delete entries in batches until overflow condition is no longer fulfilled.
      auto km_it = keys_metas.begin();
      {
        std::vector<sw::redis::StringView> k_views;
        k_views.reserve(std::min(part_size, this->params_.max_batch_size));

        while (km_it != keys_metas.end()) {
          // Prepare query.
          const size_t batch_size{
              std::min<size_t>(keys_metas.end() - km_it, this->params_.max_batch_size)};

          HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                     " (size = ", part_size, "). Attempting to evict ", batch_size,
                     " LEAST USED key/value pairs.\n");

          // Assemble and launch query.
          k_views.clear();
          for (const auto batch_end{km_it + batch_size}; km_it != batch_end; ++km_it) {
            k_views.emplace_back(reinterpret_cast<const char*>(&km_it->first), sizeof(Key));
          }
          delete_batch(k_views);
          if (part_size <= this->overflow_resolution_margin_) {
            break;
          }
        }
      }

      // To simulate decay, reset remaining LFU counters to half the of the new minimum.
      if (km_it != keys_metas.end()) {
        const long long new_count{km_it->second / 2};

        auto touched_keys{std::make_shared<std::vector<Key>>()};
        touched_keys->reserve(keys_metas.end() - km_it);
        for (; km_it != keys_metas.end(); ++km_it) {
          touched_keys->emplace_back(km_it->first);
        }

        background_worker_.submit([this, table_name, part_index, touched_keys, new_count]() {
          refresh_metadata_lfu_set_(table_name, part_index, *touched_keys, new_count);
        });
      }
    } break;

    case DatabaseOverflowPolicy_t::EvictOldest: {
      // Fetch keys and metadata.
      std::vector<std::pair<Key, time_t>> keys_metas;
      keys_metas.reserve(part_size);
      redis_->hgetall(hkey_m, RedisKeyTimeVectorInserter<Key>(keys_metas));

      part_size = keys_metas.size();
      if (part_size <= this->overflow_resolution_margin_) {
        HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                   ": Overflow was already resolved by another process.\n");
        return;
      }

      // Sort by ascending by number of accesses.
      std::sort(keys_metas.begin(), keys_metas.end(),
                [](const auto& km0, const auto& km1) { return km0.second < km1.second; });

      // Delete entries in batches until overflow condition is no longer fulfilled.
      std::vector<sw::redis::StringView> k_views;
      k_views.reserve(std::min(part_size, this->params_.max_batch_size));

      for (auto km_it{keys_metas.begin()}; km_it != keys_metas.end();) {
        // Prepare query.
        k_views.clear();
        const size_t batch_size{
            std::min<size_t>(keys_metas.end() - km_it, this->params_.max_batch_size)};

        HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                   " (size = ", part_size, "). Attempting to evict ", batch_size,
                   " OLDEST key/value pairs.\n");

        // Assemble and launch query.
        for (const auto batch_end{km_it + batch_size}; km_it != batch_end; ++km_it) {
          k_views.emplace_back(reinterpret_cast<const char*>(&km_it->first), sizeof(Key));
        }
        delete_batch(k_views);
        if (part_size <= this->overflow_resolution_margin_) {
          break;
        }
      }
    } break;
  }

  HCTR_LOG_C(DEBUG, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
             ": Overflow resolution concluded!\n");
}

template <typename Key>
void RedisClusterBackend<Key>::refresh_metadata_lfu_inc_(const std::string& table_name,
                                                         const size_t part_index,
                                                         const std::vector<Key>& keys,
                                                         const long long amount) {
  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
             ": Refreshing LFU metadata of ", keys.size(), " entries (incrementing by ", amount,
             ").\n");

  HCTR_PRINT_REDIS_ERRORS_({
    HCTR_DEFINE_REDIS_META_HKEY_();

    // Step through input batch-by-batch.
    for (auto k_it{keys.begin()}; k_it != keys.end();) {
      const size_t batch_size{std::min<size_t>(keys.end() - k_it, this->params_.max_batch_size)};

      sw::redis::Pipeline pipe{redis_->pipeline(hkey_m, false)};
      for (const auto batch_end{k_it + batch_size}; k_it != batch_end; ++k_it) {
        pipe.hincrby(hkey_m, {reinterpret_cast<const char*>(&*k_it), sizeof(Key)}, amount);
      }
      pipe.exec();
    }
  });
}

template <typename Key>
void RedisClusterBackend<Key>::refresh_metadata_lfu_set_(const std::string& table_name,
                                                         const size_t part_index,
                                                         const std::vector<Key>& keys,
                                                         const long long amount) {
  const std::string rendered_amount{std::to_string(amount)};

  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
             ": Refreshing LFU metadata of ", keys.size(), " entries (assigning to ",
             rendered_amount, ").\n");

  HCTR_PRINT_REDIS_ERRORS_({
    HCTR_DEFINE_REDIS_META_HKEY_();

    std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> km_views;
    km_views.reserve(std::min(keys.size(), this->params_.max_batch_size));

    // Step through input batch-by-batch.
    for (auto k_it{keys.begin()}; k_it != keys.end();) {
      const size_t batch_size{std::min<size_t>(keys.end() - k_it, this->params_.max_batch_size)};

      km_views.clear();
      for (const auto batch_end{k_it + batch_size}; k_it != batch_end; ++k_it) {
        km_views.emplace_back(
            std::piecewise_construct,
            std::forward_as_tuple(reinterpret_cast<const char*>(&*k_it), sizeof(Key)),
            std::forward_as_tuple(rendered_amount.data(), rendered_amount.size()));
      }
      redis_->hset(hkey_m, km_views.begin(), km_views.end());
    }
  });
}

template <typename Key>
void RedisClusterBackend<Key>::refresh_metadata_lru_(const std::string& table_name,
                                                     const size_t part_index,
                                                     const std::vector<Key>& keys,
                                                     const time_t time) {
  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
             ": Refreshing LRU metadata of ", keys.size(), " entries (assigning to ", time, ").\n");

  HCTR_PRINT_REDIS_ERRORS_({
    HCTR_DEFINE_REDIS_META_HKEY_();

    std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> km_views;
    km_views.reserve(std::min(keys.size(), this->params_.max_batch_size));

    // Step through input batch-by-batch.
    for (auto k_it{keys.begin()}; k_it != keys.end();) {
      const size_t batch_size{std::min<size_t>(keys.end() - k_it, this->params_.max_batch_size)};

      km_views.clear();
      for (const auto batch_end{k_it + batch_size}; k_it != batch_end; ++k_it) {
        km_views.emplace_back(
            std::piecewise_construct,
            std::forward_as_tuple(reinterpret_cast<const char*>(&*k_it), sizeof(Key)),
            std::forward_as_tuple(reinterpret_cast<const char*>(&time), sizeof(time_t)));
      }
      redis_->hset(hkey_m, km_views.begin(), km_views.end());
    }
  });
}

template <typename Key>
void RedisClusterBackend<Key>::queue_metadata_refresh_(const std::string& table_name,
                                                       const size_t part_index,
                                                       std::shared_ptr<std::vector<Key>>&& keys) {
  switch (this->params_.overflow_policy) {
    case DatabaseOverflowPolicy_t::EvictRandom: {
    } break;

    case DatabaseOverflowPolicy_t::EvictLeastUsed: {
      background_worker_.submit([this, table_name, part_index, keys]() {
        refresh_metadata_lfu_inc_(table_name, part_index, *keys, 1);
      });
    } break;

    case DatabaseOverflowPolicy_t::EvictOldest: {
      const time_t now{std::time(nullptr)};
      background_worker_.submit([this, table_name, part_index, keys, now]() {
        refresh_metadata_lru_(table_name, part_index, *keys, now);
      });
    } break;
  }
}

template class RedisClusterBackend<unsigned int>;
template class RedisClusterBackend<long long>;

}  // namespace HugeCTR
