/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <iostream>
#include <optional>
#include <random>
#include <string_view>
#include <thread_pool.hpp>
#include <unordered_set>

// TODO: Remove me!
#pragma GCC diagnostic error "-Wconversion"

namespace HugeCTR {

inline std::string make_hkey(const std::string& table_name, const size_t partition,
                             const char suffix) {
  std::ostringstream os;
  // These curly brackets (`{` and `}`) are not a design choice. Instead, this will trigger Redis to
  // align node allocations for 'v' and 't'.
  os << "hps_et{" << table_name << "/p" << partition << '}' << suffix;
  return os.str();
}

#ifdef HCTR_REDIS_VALUE_HKEY
#error HCTR_REDIS_VALUE_HKEY should not be defined!
#else
#define HCTR_REDIS_VALUE_HKEY() const std::string& hkey_v = make_hkey(table_name, part, 'v')
#endif
#ifdef HCTR_REDIS_TIME_HKEY
#error HCTR_REDIS_TIME_HKEY should not be defined!
#else
#define HCTR_REDIS_TIME_HKEY() const std::string& hkey_t = make_hkey(table_name, part, 't')
#endif

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
  HCTR_LOG_S(INFO, WORLD) << get_name() << ": Connecting via " << options.host << ':'
                          << options.port << "..." << std::endl;
  redis_ = std::make_unique<sw::redis::RedisCluster>(options, pool_options);
}

template <typename Key>
RedisClusterBackend<Key>::~RedisClusterBackend() {
  HCTR_LOG_S(INFO, WORLD) << get_name() << ": Awaiting background worker to conclude..."
                          << std::endl;
  background_worker_.await_idle();

  HCTR_LOG_S(INFO, WORLD) << get_name() << ": Disconnecting..." << std::endl;
  redis_.reset();
}

template <typename Key>
size_t RedisClusterBackend<Key>::size(const std::string& table_name) const {
  size_t num_pairs;

  if (this->params_.num_partitions == 1) {
    // Precalc constants.
    static constexpr size_t part = 0;
    HCTR_REDIS_VALUE_HKEY();

    try {
      num_pairs = redis_->hlen(hkey_v);
    } catch (sw::redis::Error& e) {
      throw DatabaseBackendError(get_name(), part, e.what());
    }
  } else {
    std::atomic<size_t> joint_num_pairs{0};

    // Process partitions.
    std::vector<std::future<void>> tasks;
    tasks.reserve(this->params_.num_partitions);

    for (size_t part = 0; part < this->params_.num_partitions; ++part) {
      tasks.emplace_back(ThreadPool::get().submit([&, part]() {
        try {
          HCTR_REDIS_VALUE_HKEY();
          joint_num_pairs += redis_->hlen(hkey_v);
        } catch (sw::redis::Error& e) {
          throw DatabaseBackendError(get_name(), part, e.what());
        }
      }));
    }
    ThreadPool::await(tasks.begin(), tasks.end());

    num_pairs = joint_num_pairs;
  }

  return num_pairs;
}

template <typename Key>
size_t RedisClusterBackend<Key>::contains(const std::string& table_name, const size_t num_keys,
                                          const Key* const keys,
                                          const std::chrono::nanoseconds& time_budget) const {
  const auto begin = std::chrono::high_resolution_clock::now();

  size_t hit_count = 0;
  size_t ign_count = 0;

  switch (num_keys) {
    case 0: {
      // Do nothing ;-).
    } break;

    case 1: {
      // Precalc constants.
      const size_t part = HCTR_KEY_TO_DB_PART_INDEX(*keys);
      HCTR_REDIS_VALUE_HKEY();

      // Check time-budget.
      const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
      if (elapsed >= time_budget) {
        HCTR_LOG_S(WARNING, WORLD)
            << get_name() << " backend; Partition " << hkey_v << ": Timeout!" << std::endl;
        ign_count++;
        break;
      }

      // Launch query.
      try {
        if (redis_->hexists(hkey_v, {reinterpret_cast<const char*>(keys), sizeof(Key)})) {
          hit_count++;
        }
      } catch (sw::redis::Error& e) {
        throw DatabaseBackendError(get_name(), part, e.what());
      }
    } break;

    default: {
      // Precalc constants.
      const Key* const keys_end = &keys[num_keys];

      if (this->params_.num_partitions == 1) {
        // Precalc constants.
        static constexpr size_t part = 0;
        HCTR_REDIS_VALUE_HKEY();

        try {
          size_t num_batches = 0;
          for (const Key* k = keys; k != keys_end; num_batches++) {
            // Check time budget.
            const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
            if (elapsed >= time_budget) {
              HCTR_LOG_S(WARNING, WORLD)
                  << get_name() << " backend; Partition " << hkey_v << ": Timeout!" << std::endl;

              ign_count += keys_end - k;
              break;
            }

            // Prepare next batch and launch query.
            sw::redis::Pipeline pipe = redis_->pipeline(hkey_v, false);
            size_t batch_size = 0;
            for (; k != keys_end; k++) {
              pipe.hexists(hkey_v, {reinterpret_cast<const char*>(k), sizeof(Key)});
              if (++batch_size >= this->params_.max_get_batch_size) {
                ++k;
                break;
              }
            }
            sw::redis::QueuedReplies replies = pipe.exec();

            // Process results.
            size_t batch_hits = 0;
            for (size_t i = 0; i < replies.size(); i++) {
              if (replies.get<bool>(i)) {
                batch_hits++;
              }
            }

            HCTR_LOG_S(TRACE, WORLD)
                << get_name() << " backend; partition " << hkey_v << ", batch " << num_batches
                << ": " << batch_hits << " / " << batch_size << " hits. Time: " << elapsed.count()
                << " / " << time_budget.count() << " us." << std::endl;

            hit_count += batch_hits;
          }
        } catch (sw::redis::Error& e) {
          throw DatabaseBackendError(get_name(), part, e.what());
        }
      } else {
        std::atomic<size_t> joint_hit_count{0};
        std::atomic<size_t> joint_ign_count{0};

        // Process partitions.
        std::vector<std::future<void>> tasks;
        tasks.reserve(this->params_.num_partitions);

        for (size_t part = 0; part < this->params_.num_partitions; ++part) {
          tasks.emplace_back(ThreadPool::get().submit([&, part]() {
            size_t hit_count = 0;

            // Precalc constants.
            HCTR_REDIS_VALUE_HKEY();

            try {
              size_t num_batches = 0;
              for (const Key* k = keys; k != keys_end; num_batches++) {
                // Check time budget.
                const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
                if (elapsed >= time_budget) {
                  HCTR_LOG_S(WARNING, WORLD) << get_name() << " backend; Partition " << hkey_v
                                             << ": Timeout!" << std::endl;

                  size_t ign_count = 0;
                  for (; k != keys_end; k++) {
                    if (HCTR_KEY_TO_DB_PART_INDEX(*k) == part) {
                      ign_count++;
                    }
                  }
                  joint_ign_count += ign_count;
                  break;
                }

                // Prepare and launch query.
                sw::redis::Pipeline pipe = redis_->pipeline(hkey_v, false);
                size_t batch_size = 0;
                for (; k != keys_end; k++) {
                  if (HCTR_KEY_TO_DB_PART_INDEX(*k) == part) {
                    pipe.hexists(hkey_v, {reinterpret_cast<const char*>(k), sizeof(Key)});
                    if (++batch_size >= this->params_.max_get_batch_size) {
                      ++k;
                      break;
                    }
                  }
                }
                sw::redis::QueuedReplies replies = pipe.exec();

                // Process results.
                size_t batch_hits = 0;
                for (size_t i = 0; i < replies.size(); i++) {
                  if (replies.get<bool>(i)) {
                    batch_hits++;
                  }
                }

                HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; partition " << hkey_v
                                         << ", batch " << num_batches << ": " << batch_hits << " / "
                                         << batch_size << " hits. Time: " << elapsed.count()
                                         << " / " << time_budget.count() << " us." << std::endl;

                hit_count += batch_hits;
              }
            } catch (sw::redis::Error& e) {
              throw DatabaseBackendError(get_name(), part, e.what());
            }

            joint_hit_count += hit_count;
          }));
        }
        ThreadPool::await(tasks.begin(), tasks.end());

        hit_count += joint_hit_count;
        ign_count += joint_ign_count;
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": " << hit_count
                           << " / " << (num_keys - ign_count) << " hits, " << ign_count
                           << " ignored." << std::endl;
  return hit_count;
}

template <typename Key>
bool RedisClusterBackend<Key>::insert(const std::string& table_name, const size_t num_pairs,
                                      const Key* const keys, const char* const values,
                                      const size_t value_size) {
  size_t num_inserts = 0;

  switch (num_pairs) {
    case 0: {
      // Do nothing ;-).
    } break;

    case 1: {
      // Precalc constants.
      const size_t part = HCTR_KEY_TO_DB_PART_INDEX(*keys);
      HCTR_REDIS_VALUE_HKEY();
      HCTR_REDIS_TIME_HKEY();

      // Insert.
      try {
        const std::string_view k_view{reinterpret_cast<const char*>(keys), sizeof(Key)};
        const time_t now = std::time(nullptr);

        sw::redis::Pipeline pipe = redis_->pipeline(hkey_v, false);
        pipe.hset(hkey_v, k_view, {values, value_size});
        pipe.hset(hkey_t, k_view, {reinterpret_cast<const char*>(&now), sizeof(time_t)});
        pipe.exec();

        num_inserts++;

        // Overflow resolution.
        check_and_resolve_overflow_(part, hkey_v, hkey_t);
      } catch (sw::redis::Error& e) {
        throw DatabaseBackendError(get_name(), part, e.what());
      }
    } break;

    default: {
      // Precalc constants.
      const Key* const keys_end = &keys[num_pairs];

      if (this->params_.num_partitions == 1) {
        // Precalc constants.
        static constexpr size_t part = 0;
        HCTR_REDIS_VALUE_HKEY();
        HCTR_REDIS_TIME_HKEY();

        try {
          std::vector<std::pair<std::string_view, std::string_view>> v_views;
          std::vector<std::pair<std::string_view, std::string_view>> t_views;

          size_t num_batches = 0;
          for (const Key* k = keys; k != keys_end; num_batches++) {
            const time_t now = std::time(nullptr);

            // Prepare and launch query.
            t_views.clear();
            v_views.clear();
            for (; k != keys_end; k++) {
              v_views.emplace_back(
                  std::piecewise_construct,
                  std::forward_as_tuple(reinterpret_cast<const char*>(k), sizeof(Key)),
                  std::forward_as_tuple(&values[(k - keys) * value_size], value_size));
              t_views.emplace_back(
                  std::piecewise_construct,
                  std::forward_as_tuple(reinterpret_cast<const char*>(k), sizeof(Key)),
                  std::forward_as_tuple(reinterpret_cast<const char*>(&now), sizeof(time_t)));
              if (t_views.size() >= this->params_.max_set_batch_size) {
                ++k;
                break;
              }
            }
            if (t_views.empty()) {
              continue;
            }

            sw::redis::Pipeline pipe = redis_->pipeline(hkey_v, false);
            pipe.hmset(hkey_v, v_views.begin(), v_views.end());
            pipe.hmset(hkey_t, t_views.begin(), t_views.end());
            pipe.exec();

            HCTR_LOG_S(TRACE, WORLD)
                << get_name() << " backend; Partition " << hkey_v << ", batch " << num_batches
                << ": Inserted " << t_views.size() << " pairs." << std::endl;

            num_inserts += t_views.size();

            // Overflow resolution.
            check_and_resolve_overflow_(part, hkey_v, hkey_t);
          }
        } catch (sw::redis::Error& e) {
          throw DatabaseBackendError(get_name(), part, e.what());
        }
      } else {
        std::atomic<size_t> joint_num_inserts{0};

        // Process partitions.
        std::vector<std::future<void>> tasks;
        tasks.reserve(this->params_.num_partitions);

        for (size_t part = 0; part < this->params_.num_partitions; ++part) {
          tasks.emplace_back(ThreadPool::get().submit([&, part]() {
            size_t num_inserts = 0;

            // Precalc constants.
            HCTR_REDIS_VALUE_HKEY();
            HCTR_REDIS_TIME_HKEY();

            try {
              std::vector<std::pair<std::string_view, std::string_view>> v_views;
              std::vector<std::pair<std::string_view, std::string_view>> t_views;

              size_t num_batches = 0;
              for (const Key* k = keys; k != keys_end; num_batches++) {
                const time_t now = std::time(nullptr);

                // Prepare and launch query.
                t_views.clear();
                v_views.clear();
                for (; k != keys_end; k++) {
                  if (HCTR_KEY_TO_DB_PART_INDEX(*k) == part) {
                    v_views.emplace_back(
                        std::piecewise_construct,
                        std::forward_as_tuple(reinterpret_cast<const char*>(k), sizeof(Key)),
                        std::forward_as_tuple(&values[(k - keys) * value_size], value_size));
                    t_views.emplace_back(
                        std::piecewise_construct,
                        std::forward_as_tuple(reinterpret_cast<const char*>(k), sizeof(Key)),
                        std::forward_as_tuple(reinterpret_cast<const char*>(&now), sizeof(time_t)));
                    if (t_views.size() >= this->params_.max_set_batch_size) {
                      ++k;
                      break;
                    }
                  }
                }
                if (t_views.empty()) {
                  continue;
                }

                sw::redis::Pipeline pipe = redis_->pipeline(hkey_v, false);
                pipe.hmset(hkey_v, v_views.begin(), v_views.end());
                pipe.hmset(hkey_t, t_views.begin(), t_views.end());
                pipe.exec();

                HCTR_LOG_S(TRACE, WORLD)
                    << get_name() << " backend; Partition " << hkey_v << ", batch " << num_batches
                    << ": Inserted " << t_views.size() << " pairs." << std::endl;

                num_inserts += t_views.size();

                // Overflow resolution.
                check_and_resolve_overflow_(part, hkey_v, hkey_t);
              }
            } catch (sw::redis::Error& e) {
              throw DatabaseBackendError(get_name(), part, e.what());
            }

            joint_num_inserts += num_inserts;
          }));
        }
        ThreadPool::await(tasks.begin(), tasks.end());

        num_inserts += joint_num_inserts;
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": Inserted "
                           << num_inserts << " / " << num_pairs << " pairs." << std::endl;
  return true;
}

template <typename Key>
size_t RedisClusterBackend<Key>::fetch(const std::string& table_name, const size_t num_keys,
                                       const Key* const keys, const DatabaseHitCallback& on_hit,
                                       const DatabaseMissCallback& on_miss,
                                       const std::chrono::nanoseconds& time_budget) {
  const auto begin = std::chrono::high_resolution_clock::now();

  size_t hit_count = 0;
  size_t ign_count = 0;

  switch (num_keys) {
    case 0: {
      // Do nothing ;-).
    } break;

    case 1: {
      // Precalc constants.
      const size_t part = HCTR_KEY_TO_DB_PART_INDEX(*keys);
      HCTR_REDIS_VALUE_HKEY();

      // Check time budget.
      const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
      if (elapsed >= time_budget) {
        HCTR_LOG_S(WARNING, WORLD)
            << get_name() << " backend; Partition " << hkey_v << ": Timeout!" << std::endl;
        on_miss(0);
        ign_count++;
        break;
      }

      try {
        // Query.
        const std::optional<std::string>& v_opt =
            redis_->hget(hkey_v, {reinterpret_cast<const char*>(keys), sizeof(Key)});

        // Process result.
        if (v_opt) {
          on_hit(0, v_opt->data(), static_cast<uint32_t>(v_opt->size()));
          hit_count++;

          // Queue timestamp refresh.
          if (this->params_.refresh_time_after_fetch) {
            const Key k = *keys;
            const time_t now = std::time(nullptr);

            background_worker_.submit([this, table_name, part, k, now]() {
              HCTR_REDIS_TIME_HKEY();
              touch_(hkey_t, k, now);
            });
          }
        } else {
          on_miss(0);
        }
      } catch (sw::redis::Error& e) {
        throw DatabaseBackendError(get_name(), part, e.what());
      }
    } break;

    default: {
      // Precalc constants.
      const Key* const keys_end = &keys[num_keys];

      if (this->params_.num_partitions == 1) {
        // Precalc constants.
        constexpr size_t part = 0;
        HCTR_REDIS_VALUE_HKEY();

        try {
          std::vector<std::string_view> k_views;
          std::vector<std::optional<std::string>> v_opts;
          std::shared_ptr<std::vector<Key>> touched_keys;

          size_t num_batches = 0;
          for (const Key* k = keys; k != keys_end; num_batches++) {
            // Check time budget.
            const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
            if (elapsed >= time_budget) {
              HCTR_LOG_S(WARNING, WORLD)
                  << get_name() << " backend; Partition " << hkey_v << ": Timeout!" << std::endl;

              ign_count += keys_end - k;
              for (; k != keys_end; k++) {
                on_miss(k - keys);
              }
              break;
            }

            // Prepare query.
            size_t idx = k - keys;
            const Key* const batch_end = std::min(&k[this->params_.max_get_batch_size], keys_end);
            k_views.clear();
            k_views.reserve(batch_end - k);
            for (; k != batch_end; k++) {
              k_views.emplace_back(reinterpret_cast<const char*>(k), sizeof(Key));
            }

            // Launch query.
            v_opts.clear();
            v_opts.reserve(k_views.size());
            redis_->hmget(hkey_v, k_views.begin(), k_views.end(), std::back_inserter(v_opts));

            // Process results.
            for (const auto& v_opt : v_opts) {
              if (v_opt) {
                on_hit(idx, v_opt->data(), static_cast<uint32_t>(v_opt->size()));
                hit_count++;

                if (this->params_.refresh_time_after_fetch) {
                  if (!touched_keys) {
                    touched_keys = std::make_shared<std::vector<Key>>();
                  }
                  touched_keys->emplace_back(keys[idx]);
                }
              } else {
                on_miss(idx);
              }
              idx++;
            }

            // Refresh timestamps if desired.
            if (touched_keys && !touched_keys->empty()) {
              const time_t now = std::time(nullptr);

              background_worker_.submit([this, table_name, part, touched_keys, now]() {
                HCTR_REDIS_TIME_HKEY();
                touch_(hkey_t, touched_keys, now);
              });
              touched_keys.reset();
            }

            HCTR_LOG_S(TRACE, WORLD)
                << get_name() << " backend. Partition " << hkey_v << ", batch " << num_batches
                << ": Fetched " << v_opts.size() << " keys. Hits " << hit_count << '.' << std::endl;
          }
        } catch (sw::redis::Error& e) {
          throw DatabaseBackendError(get_name(), part, e.what());
        }
      } else {
        std::atomic<size_t> joint_hit_count{0};
        std::atomic<size_t> joint_ign_count{0};

        // Process partitions.
        std::vector<std::future<void>> tasks;
        tasks.reserve(this->params_.num_partitions);

        for (size_t part = 0; part < this->params_.num_partitions; ++part) {
          tasks.emplace_back(ThreadPool::get().submit([&, part]() {
            size_t hit_count = 0;

            // Precalc constants.
            HCTR_REDIS_VALUE_HKEY();

            try {
              std::vector<size_t> idx;
              std::vector<std::string_view> k_views;
              std::vector<std::optional<std::string>> v_opts;
              std::shared_ptr<std::vector<Key>> touched_keys;

              size_t num_batches = 0;
              for (const Key* k = keys; k != keys_end; num_batches++) {
                // Check time budget.
                const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
                if (elapsed >= time_budget) {
                  HCTR_LOG_S(WARNING, WORLD) << get_name() << " backend; Partition " << hkey_v
                                             << ": Timeout!" << std::endl;

                  size_t ign_count = 0;
                  for (; k != keys_end; k++) {
                    if (HCTR_KEY_TO_DB_PART_INDEX(*k) == part) {
                      on_miss(keys_end - k);
                      ign_count++;
                    }
                  }
                  joint_ign_count += ign_count;
                  break;
                }

                // Prepare query.
                k_views.clear();
                idx.clear();
                for (; k != keys_end; k++) {
                  if (HCTR_KEY_TO_DB_PART_INDEX(*k) == part) {
                    idx.emplace_back(k - keys);
                    k_views.emplace_back(reinterpret_cast<const char*>(k), sizeof(Key));
                    if (k_views.size() >= this->params_.max_get_batch_size) {
                      ++k;
                      break;
                    }
                  }
                }
                if (k_views.empty()) {
                  continue;
                }

                // Launch query.
                v_opts.clear();
                v_opts.reserve(k_views.size());
                redis_->hmget(hkey_v, k_views.begin(), k_views.end(), std::back_inserter(v_opts));

                // Process results.
                auto idx_it = idx.begin();
                for (const auto& v_opt : v_opts) {
                  if (v_opt) {
                    on_hit(*idx_it, v_opt->data(), static_cast<uint32_t>(v_opt->size()));
                    hit_count++;

                    if (this->params_.refresh_time_after_fetch) {
                      if (!touched_keys) {
                        touched_keys = std::make_shared<std::vector<Key>>();
                      }
                      touched_keys->emplace_back(keys[*idx_it]);
                    }
                  } else {
                    on_miss(*idx_it);
                  }
                  idx_it++;
                }
                HCTR_CHECK(idx_it == idx.end());

                // Refresh timestamps if desired.
                if (touched_keys && !touched_keys->empty()) {
                  const time_t now = std::time(nullptr);

                  background_worker_.submit([this, table_name, part, touched_keys, now]() {
                    HCTR_REDIS_TIME_HKEY();
                    touch_(hkey_t, touched_keys, now);
                  });
                  touched_keys.reset();
                }

                HCTR_LOG_S(TRACE, WORLD)
                    << get_name() << " backend. Partition " << hkey_v << ", batch " << num_batches
                    << ": Fetched " << v_opts.size() << " keys. Hits " << hit_count << '.'
                    << std::endl;
              }
            } catch (sw::redis::Error& e) {
              throw DatabaseBackendError(get_name(), part, e.what());
            }

            joint_hit_count += hit_count;
          }));
        }
        ThreadPool::await(tasks.begin(), tasks.end());

        hit_count += joint_hit_count;
        ign_count += joint_ign_count;
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": " << hit_count
                           << " / " << (num_keys - ign_count) << " hits, " << ign_count
                           << " ignored." << std::endl;
  return hit_count;
}

template <typename Key>
size_t RedisClusterBackend<Key>::fetch(const std::string& table_name, const size_t num_indices,
                                       const size_t* indices, const Key* const keys,
                                       const DatabaseHitCallback& on_hit,
                                       const DatabaseMissCallback& on_miss,
                                       const std::chrono::nanoseconds& time_budget) {
  const auto begin = std::chrono::high_resolution_clock::now();

  size_t hit_count = 0;
  size_t ign_count = 0;

  switch (num_indices) {
    case 0: {
      // Do nothing ;-).
    } break;

    case 1: {
      // Precalc constants.
      const Key k = keys[*indices];
      const size_t part = HCTR_KEY_TO_DB_PART_INDEX(k);
      HCTR_REDIS_VALUE_HKEY();

      // Check time budget.
      const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
      if (elapsed >= time_budget) {
        HCTR_LOG_S(WARNING, WORLD)
            << get_name() << " backend; Partition " << hkey_v << ": Timeout!" << std::endl;
        on_miss(*indices);
        ign_count++;
        break;
      }

      try {
        // Query.
        const std::optional<std::string>& v_opt =
            redis_->hget(hkey_v, {reinterpret_cast<const char*>(&k), sizeof(Key)});

        // Process result.
        if (v_opt) {
          on_hit(*indices, v_opt->data(), static_cast<uint32_t>(v_opt->size()));
          hit_count++;

          // Queue timestamp refresh.
          if (this->params_.refresh_time_after_fetch) {
            const time_t now = std::time(nullptr);

            background_worker_.submit([this, table_name, part, k, now]() {
              HCTR_REDIS_TIME_HKEY();
              touch_(hkey_t, k, now);
            });
          }
        } else {
          on_miss(*indices);
        }
      } catch (sw::redis::Error& e) {
        throw DatabaseBackendError(get_name(), part, e.what());
      }
    } break;

    default: {
      // Precalc constants.
      const size_t* const indices_end = &indices[num_indices];

      if (this->params_.num_partitions == 1) {
        // Precalc constants.
        constexpr size_t part = 0;
        HCTR_REDIS_VALUE_HKEY();

        std::vector<std::string_view> k_views;
        std::vector<std::optional<std::string>> v_opts;
        std::shared_ptr<std::vector<Key>> touched_keys;

        try {
          size_t num_batches = 0;
          for (const size_t* i = indices; i != indices_end; num_batches++) {
            // Check time budget.
            const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
            if (elapsed >= time_budget) {
              HCTR_LOG_S(WARNING, WORLD)
                  << get_name() << " backend; Partition " << hkey_v << ": Timeout!" << std::endl;

              ign_count += indices_end - i;
              for (; i != indices_end; i++) {
                on_miss(*i);
              }
              break;
            }

            // Prepare query.
            k_views.clear();
            const size_t* const batch_beg = i;
            const size_t* const batch_end =
                std::min(&i[this->params_.max_get_batch_size], indices_end);
            for (; i != batch_end; i++) {
              k_views.emplace_back(reinterpret_cast<const char*>(&keys[*i]), sizeof(Key));
              if (k_views.size() >= this->params_.max_get_batch_size) {
                ++i;
                break;
              }
            }
            if (k_views.empty()) {
              continue;
            }

            // Launch query.
            v_opts.clear();
            v_opts.reserve(k_views.size());
            redis_->hmget(hkey_v, k_views.begin(), k_views.end(), std::back_inserter(v_opts));

            // Process results.
            i = batch_beg;
            for (const auto& v_opt : v_opts) {
              if (v_opt) {
                on_hit(*i, v_opt->data(), static_cast<uint32_t>(v_opt->size()));
                hit_count++;

                if (this->params_.refresh_time_after_fetch) {
                  if (!touched_keys) {
                    touched_keys = std::make_shared<std::vector<Key>>();
                  }
                  touched_keys->emplace_back(keys[*i]);
                }
              } else {
                on_miss(*i);
              }
              i++;
            }
            HCTR_CHECK(i == batch_end);

            // Refresh timestamps if desired.
            if (touched_keys && !touched_keys->empty()) {
              const time_t now = std::time(nullptr);

              background_worker_.submit([this, table_name, part, touched_keys, now]() {
                HCTR_REDIS_TIME_HKEY();
                touch_(hkey_t, touched_keys, now);
              });
              touched_keys.reset();
            }

            HCTR_LOG_S(TRACE, WORLD)
                << get_name() << " backend. Partition " << hkey_v << ", batch " << num_batches
                << ": Fetched " << v_opts.size() << " keys. Hits " << hit_count << '.' << std::endl;
          }
        } catch (sw::redis::Error& e) {
          throw DatabaseBackendError(get_name(), part, e.what());
        }
      } else {
        std::atomic<size_t> joint_hit_count{0};
        std::atomic<size_t> joint_ign_count{0};

        // Process partitions.
        std::vector<std::future<void>> tasks;
        tasks.reserve(this->params_.num_partitions);

        for (size_t part = 0; part < this->params_.num_partitions; ++part) {
          tasks.emplace_back(ThreadPool::get().submit([&, part]() {
            size_t hit_count = 0;

            // Precalc constants.
            HCTR_REDIS_VALUE_HKEY();

            std::vector<size_t> idx;
            std::vector<std::string_view> k_views;
            std::vector<std::optional<std::string>> v_opts;
            std::shared_ptr<std::vector<Key>> touched_keys;

            try {
              size_t num_batches = 0;
              for (const size_t* i = indices; i != indices_end; num_batches++) {
                // Check time budget.
                const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
                if (elapsed >= time_budget) {
                  HCTR_LOG_S(WARNING, WORLD) << get_name() << " backend; Partition " << hkey_v
                                             << ": Timeout!" << std::endl;

                  size_t ign_count = 0;
                  for (; i != indices_end; i++) {
                    const Key& k = keys[*i];
                    if (HCTR_KEY_TO_DB_PART_INDEX(k) == part) {
                      on_miss(*i);
                      ign_count++;
                    }
                  }
                  joint_ign_count += ign_count;
                  break;
                }

                // Prepare query.
                k_views.clear();
                idx.clear();
                for (; i != indices_end; i++) {
                  const Key& k = keys[*i];
                  if (HCTR_KEY_TO_DB_PART_INDEX(k) == part) {
                    idx.emplace_back(*i);
                    k_views.emplace_back(reinterpret_cast<const char*>(&k), sizeof(Key));
                    if (k_views.size() >= this->params_.max_get_batch_size) {
                      ++i;
                      break;
                    }
                  }
                }
                if (k_views.empty()) {
                  continue;
                }

                // Launch query.
                v_opts.clear();
                v_opts.reserve(k_views.size());
                redis_->hmget(hkey_v, k_views.begin(), k_views.end(), std::back_inserter(v_opts));

                // Process results.
                auto idx_it = idx.begin();
                for (const auto& v_opt : v_opts) {
                  if (v_opt) {
                    on_hit(*idx_it, v_opt->data(), static_cast<uint32_t>(v_opt->size()));
                    hit_count++;

                    if (this->params_.refresh_time_after_fetch) {
                      if (!touched_keys) {
                        touched_keys = std::make_shared<std::vector<Key>>();
                      }
                      touched_keys->emplace_back(keys[*idx_it]);
                    }
                  } else {
                    on_miss(*idx_it);
                  }
                  idx_it++;
                }
                HCTR_CHECK(idx_it == idx.end());

                // Refresh timestamps if desired.
                if (touched_keys && !touched_keys->empty()) {
                  const time_t now = std::time(nullptr);

                  background_worker_.submit([this, table_name, part, touched_keys, now]() {
                    HCTR_REDIS_TIME_HKEY();
                    touch_(hkey_t, touched_keys, now);
                  });
                  touched_keys.reset();
                }

                HCTR_LOG_S(TRACE, WORLD)
                    << get_name() << " backend. Partition " << hkey_v << ", batch " << num_batches
                    << ": Fetched " << v_opts.size() << " keys. Hits " << hit_count << '.'
                    << std::endl;
              }
            } catch (sw::redis::Error& e) {
              throw DatabaseBackendError(get_name(), part, e.what());
            }

            joint_hit_count += hit_count;
          }));
        }
        ThreadPool::await(tasks.begin(), tasks.end());

        hit_count += joint_hit_count;
        ign_count += joint_ign_count;
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": " << hit_count
                           << " / " << (num_indices - ign_count) << " hits, " << ign_count
                           << " ignored." << std::endl;
  return hit_count;
}

template <typename Key>
size_t RedisClusterBackend<Key>::evict(const std::string& table_name) {
  size_t approx_num_keys;

  if (this->params_.num_partitions == 1) {
    static constexpr size_t part = 0;

    // Precalc constants.
    HCTR_REDIS_VALUE_HKEY();
    HCTR_REDIS_TIME_HKEY();

    try {
      approx_num_keys = redis_->hlen(hkey_v);

      // Delete the keys.
      sw::redis::Pipeline pipe = redis_->pipeline(hkey_v, false);
      pipe.del(hkey_v);
      pipe.del(hkey_t);
      pipe.exec();
    } catch (sw::redis::Error& e) {
      throw DatabaseBackendError(get_name(), part, e.what());
    }
  } else {
    std::atomic<size_t> joint_approx_num_keys{0};

    // Process partitions.
    std::vector<std::future<void>> tasks;
    tasks.reserve(this->params_.num_partitions);

    for (size_t part = 0; part < this->params_.num_partitions; ++part) {
      tasks.emplace_back(ThreadPool::get().submit([&, part]() {
        // Precalc constants.
        HCTR_REDIS_VALUE_HKEY();
        HCTR_REDIS_TIME_HKEY();

        try {
          const size_t approx_num_keys = redis_->hlen(hkey_v);

          // Delete the keys.
          sw::redis::Pipeline pipe = redis_->pipeline(hkey_v, false);
          pipe.del(hkey_v);
          pipe.del(hkey_t);
          pipe.exec();

          HCTR_LOG_S(TRACE, WORLD)
              << get_name() << " backend. Partition " << hkey_v << ": Erased approximately "
              << approx_num_keys << " pairs." << std::endl;

          joint_approx_num_keys += approx_num_keys;
        } catch (sw::redis::Error& e) {
          throw DatabaseBackendError(get_name(), part, e.what());
        }
      }));
    }
    ThreadPool::await(tasks.begin(), tasks.end());

    approx_num_keys = joint_approx_num_keys;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name
                           << " erased (approximately " << approx_num_keys << " pairs)."
                           << std::endl;
  return approx_num_keys;
}

template <typename Key>
size_t RedisClusterBackend<Key>::evict(const std::string& table_name, const size_t num_keys,
                                       const Key* const keys) {
  size_t hit_count = 0;

  switch (num_keys) {
    case 0: {
      // Do nothing ;-).
    } break;

    case 1: {
      // Precalc constants.
      const size_t part = HCTR_KEY_TO_DB_PART_INDEX(*keys);
      HCTR_REDIS_VALUE_HKEY();
      HCTR_REDIS_TIME_HKEY();

      try {
        const std::string_view k_view{reinterpret_cast<const char*>(keys), sizeof(Key)};

        // Erase.
        sw::redis::Pipeline pipe = redis_->pipeline(hkey_v, false);
        pipe.hdel(hkey_v, k_view);
        pipe.hdel(hkey_t, k_view);
        sw::redis::QueuedReplies replies = pipe.exec();
        HCTR_CHECK(replies.size() == 2);

        hit_count += std::max(replies.get<long long>(0), replies.get<long long>(1));
      } catch (sw::redis::Error& e) {
        throw DatabaseBackendError(get_name(), part, e.what());
      }
    } break;

    default: {
      // Precalc constants.
      const Key* const keys_end = &keys[num_keys];

      if (this->params_.num_partitions == 1) {
        static constexpr size_t part = 0;
        HCTR_REDIS_VALUE_HKEY();
        HCTR_REDIS_TIME_HKEY();

        try {
          std::vector<std::string_view> k_views;

          size_t num_batches = 0;
          for (const Key* k = keys; k != keys_end; num_batches++) {
            // Gather batch.
            const Key* const batch_end = std::min(&k[this->params_.max_set_batch_size], keys_end);
            k_views.reserve(batch_end - k);
            k_views.clear();
            for (; k != batch_end; k++) {
              k_views.emplace_back(reinterpret_cast<const char*>(k), sizeof(Key));
            }

            // Erase.
            sw::redis::Pipeline pipe = redis_->pipeline(hkey_v, false);
            pipe.hdel(hkey_v, k_views.begin(), k_views.end());
            pipe.hdel(hkey_t, k_views.begin(), k_views.end());
            sw::redis::QueuedReplies replies = pipe.exec();
            HCTR_CHECK(replies.size() == 2);

            const size_t batch_hits =
                std::max(replies.get<long long>(0), replies.get<long long>(1));

            HCTR_LOG_S(TRACE, WORLD)
                << get_name() << " backend; Partition " << hkey_v << ", batch " << num_batches
                << ": Erased " << batch_hits << " / " << k_views.size() << " pairs." << std::endl;

            hit_count += batch_hits;
          }
        } catch (sw::redis::Error& e) {
          throw DatabaseBackendError(get_name(), part, e.what());
        }
      } else {
        std::atomic<size_t> joint_hit_count{0};

        // Process partitions.
        std::vector<std::future<void>> tasks;
        tasks.reserve(this->params_.num_partitions);

        for (size_t part = 0; part < this->params_.num_partitions; ++part) {
          tasks.emplace_back(ThreadPool::get().submit([&, part]() {
            size_t hit_count = 0;

            // Precalc constants.
            HCTR_REDIS_VALUE_HKEY();
            HCTR_REDIS_TIME_HKEY();

            try {
              std::vector<std::string_view> k_views;

              size_t num_batches = 0;
              for (const Key* k = keys; k != keys_end; num_batches++) {
                // Gather batch.
                k_views.clear();
                for (; k != keys_end; k++) {
                  if (HCTR_KEY_TO_DB_PART_INDEX(*k) == part) {
                    k_views.emplace_back(reinterpret_cast<const char*>(k), sizeof(Key));
                    if (k_views.size() >= this->params_.max_set_batch_size) {
                      ++k;
                      break;
                    }
                  }
                }
                if (k_views.empty()) {
                  continue;
                }

                // Erase.
                sw::redis::Pipeline pipe = redis_->pipeline(hkey_v, false);
                pipe.hdel(hkey_v, k_views.begin(), k_views.end());
                pipe.hdel(hkey_t, k_views.begin(), k_views.end());
                sw::redis::QueuedReplies replies = pipe.exec();
                HCTR_CHECK(replies.size() == 2);

                const size_t batch_hits =
                    std::max(replies.get<long long>(0), replies.get<long long>(1));

                HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Partition " << hkey_v
                                         << ", batch " << num_batches << ": Erased " << batch_hits
                                         << " / " << k_views.size() << " pairs." << std::endl;

                hit_count += batch_hits;
              }
            } catch (sw::redis::Error& e) {
              throw DatabaseBackendError(get_name(), part, e.what());
            }

            joint_hit_count += hit_count;
          }));
        }
        ThreadPool::await(tasks.begin(), tasks.end());

        hit_count += joint_hit_count;
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": Erased "
                           << hit_count << " / " << num_keys << " pairs." << std::endl;
  return hit_count;
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
std::vector<Key> RedisClusterBackend<Key>::keys(const std::string& table_name) {
  std::vector<Key> k;

  if (this->params_.num_partitions == 1) {
    static constexpr size_t part = 0;
    HCTR_REDIS_TIME_HKEY();

    // Fetch keys.
    std::vector<std::string> k_views;
    try {
      redis_->hkeys(hkey_t, std::back_inserter(k_views));
    } catch (sw::redis::Error& e) {
      throw DatabaseBackendError(get_name(), part, e.what());
    }

    // Append to list.
    k.reserve(k_views.size());
    for (const std::string& k_view : k_views) {
      k.emplace_back(*reinterpret_cast<const Key*>(k_view.data()));
    }
  } else {
    std::mutex k_guard;

    // Process partitions.
    std::vector<std::future<void>> tasks;
    tasks.reserve(this->params_.num_partitions);

    for (size_t part = 0; part < this->params_.num_partitions; ++part) {
      tasks.emplace_back(ThreadPool::get().submit([&, part]() {
        HCTR_REDIS_TIME_HKEY();

        // Fetch keys.
        std::vector<std::string> k_views;
        try {
          redis_->hkeys(hkey_t, std::back_inserter(k_views));
        } catch (sw::redis::Error& e) {
          throw DatabaseBackendError(get_name(), part, e.what());
        }

        // Append to keys vector.
        const std::unique_lock lock(k_guard);

        k.reserve(k.size() + k_views.size());
        for (const std::string& k_view : k_views) {
          k.emplace_back(*reinterpret_cast<const Key*>(k_view.data()));
        }
      }));
      ThreadPool::await(tasks.begin(), tasks.end());
    }
  }

  return k;
}

template <typename Key>
void RedisClusterBackend<Key>::dump_bin(const std::string& table_name, std::ofstream& file) {
  std::vector<Key> k = keys(table_name);

  // TODO: Maybe not ideal to shuffle to undo partition sorting. Use a different data structure?
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(k.begin(), k.end(), gen);
  }

  // We just implement this as repeating queries.
  std::vector<std::string> v_views(this->params_.max_get_batch_size);
  std::atomic<uint32_t> first_value_size{0};

  for (auto k_it = k.begin(); k_it != k.end();) {
    // Read batch values.
    const auto batch_end = std::min(k_it + this->params_.max_get_batch_size, k.end());
    fetch(
        table_name, batch_end - k_it, &*k_it,
        [&](const size_t index, const char* const value, const size_t value_size) {
          if (!first_value_size) {
            HCTR_CHECK(k_it == k.begin());
            first_value_size = static_cast<uint32_t>(value_size);
          } else {
            HCTR_CHECK(value_size == first_value_size);
          }
          v_views[index].assign(value, value_size);
        },
        [&](const size_t index) { v_views[index].clear(); }, std::chrono::nanoseconds::max());

    // Write the value size if this is the first batch.
    if (k_it == k.begin() && first_value_size) {
      file.write(reinterpret_cast<const char*>(&first_value_size), sizeof(uint32_t));
    }

    // Write the batch.
    for (auto v_views_it = v_views.begin(); k_it != batch_end; ++k_it, ++v_views_it) {
      const std::string& value = *v_views_it;
      if (!value.empty()) {
        file.write(reinterpret_cast<const char*>(&*k_it), sizeof(Key));
        file.write(value.data(), value.size());
      }
    }
  }
}

template <typename Key>
void RedisClusterBackend<Key>::dump_sst(const std::string& table_name,
                                        rocksdb::SstFileWriter& file) {
  // Collect sorted set of all keys.
  std::vector<Key> k = keys(table_name);
  std::sort(k.begin(), k.end());

  // We just implement this as repeating queries.
  std::vector<std::string> v_views(this->params_.max_get_batch_size);

  for (auto k_it = k.begin(); k_it != k.end();) {
    // Read batch values.
    const auto batch_end = std::min(k_it + this->params_.max_get_batch_size, k.end());
    fetch(
        table_name, batch_end - k_it, &*k_it,
        [&](const size_t index, const char* const value, const size_t value_size) {
          v_views[index].assign(value, value_size);
        },
        [&](const size_t index) { v_views[index].clear(); }, std::chrono::nanoseconds::max());

    // Write the batch.
    rocksdb::Slice k_view{nullptr, sizeof(Key)};
    rocksdb::Slice v_view;
    for (auto v_views_it = v_views.begin(); k_it != batch_end; ++k_it, ++v_views_it) {
      k_view.data_ = reinterpret_cast<const char*>(&*k_it);
      v_view.data_ = v_views_it->data();
      v_view.size_ = v_views_it->size();
      HCTR_ROCKSDB_CHECK(file.Put(k_view, v_view));
    }
  }
}

template <typename Key>
void RedisClusterBackend<Key>::check_and_resolve_overflow_(const size_t part,
                                                           const std::string& hkey_v,
                                                           const std::string& hkey_t) {
  // Check overflow condition.
  size_t part_size = redis_->hlen(hkey_t);
  if (part_size <= this->params_.overflow_margin) {
    return;
  }
  HCTR_LOG_S(TRACE, WORLD) << get_name() << " partition " << hkey_v
                           << " is overflowing (size = " << part_size << " > "
                           << this->params_.overflow_margin << "). Attempting to resolve..."
                           << std::endl;

  // Select overflow resolution policy.
  switch (this->params_.overflow_policy) {
    case DatabaseOverflowPolicy_t::EvictOldest: {
      // Fetch keys and insert times.
      std::vector<std::pair<Key, time_t>> kt;
      {
        // Fetch from Redis.
        std::vector<std::pair<std::string, std::string>> kt_views;
        kt_views.reserve(part_size);
        redis_->hgetall(hkey_t, std::back_inserter(kt_views));

        // Sanity check.
        part_size = kt_views.size();
        if (part_size <= this->overflow_resolution_margin_) {
          return;
        }

        // Convert to native format.
        kt.reserve(part_size);
        for (const std::pair<std::string, std::string>& kt_view : kt_views) {
          kt.emplace_back(*reinterpret_cast<const Key*>(kt_view.first.data()),
                          *reinterpret_cast<const time_t*>(kt_view.second.data()));
        }
      }

      // Sort by ascending by time.
      std::sort(kt.begin(), kt.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });

      // Delete pairs in batches until overflow condition is no longer fulfilled.
      std::vector<std::string_view> k_views;
      k_views.reserve(this->params_.max_set_batch_size);

      for (auto kt_it = kt.begin(); kt_it != kt.end();) {
        // Collect a batch.
        const auto batch_end = std::min(kt_it + this->params_.max_set_batch_size, kt.end());
        k_views.clear();
        for (; kt_it != batch_end; ++kt_it) {
          k_views.emplace_back(reinterpret_cast<const char*>(&kt_it->first), sizeof(Key));
        }

        // Perform deletion.
        HCTR_LOG_S(TRACE, WORLD) << get_name() << " partition " << hkey_v
                                 << " (size = " << part_size << "). Attempting to evict "
                                 << k_views.size() << " OLD key/value pairs." << std::endl;

        sw::redis::Pipeline pipe = redis_->pipeline(hkey_v, false);
        pipe.hdel(hkey_v, k_views.begin(), k_views.end());
        pipe.hdel(hkey_t, k_views.begin(), k_views.end());
        pipe.exec();

        // Overflow resolved?
        part_size = redis_->hlen(hkey_t);
        if (part_size <= this->overflow_resolution_margin_) {
          break;
        }
      }
    } break;

    case DatabaseOverflowPolicy_t::EvictRandom: {
      // Fetch all keys in partition.
      std::vector<std::string> k_views;
      k_views.reserve(part_size);
      redis_->hkeys(hkey_t, std::back_inserter(k_views));

      // Sanity check.
      part_size = k_views.size();
      if (part_size <= this->overflow_resolution_margin_) {
        return;
      }

      // Shuffle the keys.
      {
        std::random_device rd;
        std::default_random_engine gen(rd());
        std::shuffle(k_views.begin(), k_views.end(), gen);
      }

      // Delete pairs in batches until overflow condition is no longer fulfilled.
      // Delete keys.
      const auto k_end = k_views.end();
      for (auto k_it = k_views.begin(); k_it != k_end;) {
        const auto batch_end = std::min(k_it + this->params_.max_set_batch_size, k_end);

        // Perform deletion.
        HCTR_LOG_S(TRACE, WORLD) << get_name() << " partition " << hkey_v
                                 << " (size = " << part_size << "). Attempting to evict "
                                 << k_views.size() << " RANDOM key/value pairs." << std::endl;

        sw::redis::Pipeline pipe = redis_->pipeline(hkey_v, false);
        pipe.hdel(hkey_v, k_it, batch_end);
        pipe.hdel(hkey_t, k_it, batch_end);
        pipe.exec();

        // Overflow resolved?
        part_size = redis_->hlen(hkey_t);
        if (part_size <= this->overflow_resolution_margin_) {
          break;
        }

        k_it = batch_end;
      }
    } break;

    default: {
      HCTR_LOG_S(WARNING, WORLD) << "Redis partition " << hkey_v << " (size = " << part_size
                                 << "), surpasses specified maximum size (="
                                 << this->params_.overflow_margin
                                 << "), but no compatible overflow policy (="
                                 << this->params_.overflow_policy << ") was selected!" << std::endl;
      return;
    } break;
  }

  HCTR_LOG_S(DEBUG, WORLD) << get_name() << " partition " << hkey_v
                           << " overflow resolution concluded!" << std::endl;
}

template <typename Key>
void RedisClusterBackend<Key>::touch_(const std::string& hkey_t, const Key& key,
                                      const time_t time) {
  HCTR_LOG_S(TRACE, WORLD) << get_name() << ": Touching key " << key << " of " << hkey_t << '.'
                           << std::endl;

  // Launch query.
  try {
    redis_->hset(hkey_t, {reinterpret_cast<const char*>(&key), sizeof(Key)},
                 {reinterpret_cast<const char*>(&time), sizeof(time_t)});
  } catch (sw::redis::Error& e) {
    HCTR_LOG_S(ERROR, WORLD) << get_name() << " partition " << hkey_t
                             << "; error during refresh: " << e.what() << '.' << std::endl;
  }
}

template <typename Key>
void RedisClusterBackend<Key>::touch_(const std::string& hkey_t,
                                      const std::shared_ptr<std::vector<Key>>& keys,
                                      const time_t time) {
  HCTR_LOG_S(TRACE, WORLD) << get_name() << ": Touching " << keys->size() << " keys of " << hkey_t
                           << '.' << std::endl;

  // Prepare query.
  std::vector<std::pair<std::string_view, std::string_view>> kt_views;
  kt_views.reserve(keys->size());
  for (const Key& k : *keys) {
    kt_views.emplace_back(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<const char*>(&k), sizeof(Key)),
        std::forward_as_tuple(reinterpret_cast<const char*>(&time), sizeof(time_t)));
  }

  // Launch query.
  try {
    redis_->hmset(hkey_t, kt_views.begin(), kt_views.end());
  } catch (sw::redis::Error& e) {
    HCTR_LOG_S(ERROR, WORLD) << get_name() << " partition " << hkey_t
                             << "; error touching refresh: " << e.what() << '.' << std::endl;
  }
}

template class RedisClusterBackend<unsigned int>;
template class RedisClusterBackend<long long>;

}  // namespace HugeCTR
