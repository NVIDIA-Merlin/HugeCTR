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
#include <hps/redis_backend.hpp>
#include <iostream>
#include <optional>
#include <random>
#include <string_view>
#include <thread_pool.hpp>
#include <unordered_set>

#define HCTR_USE_XXHASH
#ifdef HCTR_USE_XXHASH
#include <xxh3.h>
#define HCTR_HASH_OF_KEY(KEY) (XXH3_64bits((KEY), sizeof(TKey)))
#else
#define HCTR_HASH_OF_KEY(KEY) (static_cast<size_t>(*KEY))
#endif

// TODO: Remove me!
#pragma GCC diagnostic error "-Wconversion"

namespace HugeCTR {

const char REDIS_HKEY_VALUE_SUFFIX = 'v';
const char REDIS_HKEY_TIME_SUFFIX = 't';

std::string make_hash_key(const std::string& table_name, const size_t partition,
                          const char suffix) {
  static const char SEPARATOR = '/';
  std::ostringstream os;
  os << table_name << SEPARATOR << 'p' << partition << SEPARATOR << suffix;
  return os.str();
}

template <typename TKey>
RedisClusterBackend<TKey>::RedisClusterBackend(
    const std::string& address, const std::string& user_name, const std::string& password,
    const size_t num_partitions, const size_t max_get_batch_size, const size_t max_set_batch_size,
    const bool refresh_time_after_fetch, const size_t overflow_margin,
    const DatabaseOverflowPolicy_t overflow_policy, const double overflow_resolution_target)
    : TBase(refresh_time_after_fetch, overflow_margin, overflow_policy, overflow_resolution_target),
      // Can switch to std::range in C++20.
      num_partitions_(num_partitions),
      max_get_batch_size_(max_get_batch_size),
      max_set_batch_size_(max_set_batch_size),
      // TODO: Make this configurable?
      workers_("redis",
               std::min(num_partitions, static_cast<size_t>(std::thread::hardware_concurrency()))) {
  // Put together cluster configuration.
  sw::redis::ConnectionOptions options;

  const std::string::size_type comma_pos = address.find(',');
  const std::string host = comma_pos == std::string::npos ? address : address.substr(0, comma_pos);

  const std::string::size_type colon_pos = host.find(':');
  if (colon_pos == std::string::npos) {
    options.host = host;
  } else {
    options.host = host.substr(0, colon_pos);
    options.port = std::stoi(host.substr(colon_pos + 1));
  }
  options.user = user_name;
  options.password = password;
  options.keep_alive = true;

  sw::redis::ConnectionPoolOptions pool_options;
  pool_options.size = 1;

  // Connect to cluster.
  HCTR_LOG_S(INFO, WORLD) << get_name() << ": Connecting via " << options.host << ':'
                          << options.port << "..." << std::endl;
  redis_ = std::make_unique<sw::redis::RedisCluster>(options, pool_options);
}

template <typename TKey>
RedisClusterBackend<TKey>::~RedisClusterBackend() {
  HCTR_LOG_S(INFO, WORLD) << get_name() << ": Awaiting background worker to conclude..."
                          << std::endl;
  this->background_worker_.await_idle();

  HCTR_LOG_S(INFO, WORLD) << get_name() << ": Disconnecting..." << std::endl;
  redis_.reset();
}

template <typename TKey>
size_t RedisClusterBackend<TKey>::size(const std::string& table_name) const {
  std::atomic<size_t> joint_table_size(0);
  std::exception_ptr error;
  size_t error_part = -1;
  std::mutex error_guard;

  // Process partitions.
  std::vector<std::future<void>> tasks;
  tasks.reserve(num_partitions_);
  for (size_t part = 0; part < num_partitions_; part++) {
    tasks.emplace_back(workers_.submit([&, part]() {
      size_t table_size = 0;
      try {
        const std::string& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);

        table_size += redis_->hlen(hkey_kv);
      } catch (...) {
        std::unique_lock lock(error_guard);
        error = std::current_exception();
        error_part = part;
      }
      joint_table_size += table_size;
    }));
  }
  ThreadPool::await(tasks.begin(), tasks.end());
  const size_t table_size = static_cast<size_t>(joint_table_size);
  try {
    if (error) {
      std::rethrow_exception(error);
    }
  } catch (sw::redis::Error& e) {
    throw DatabaseBackendError(get_name(), error_part, e.what());
  }
  return table_size;
}

template <typename TKey>
size_t RedisClusterBackend<TKey>::contains(const std::string& table_name, const size_t num_keys,
                                           const TKey* const keys) const {
  size_t hit_count = 0;

  if (num_keys <= 0) {
    // Do nothing ;-).
  } else if (num_keys <= 1) {
    const size_t part = HCTR_HASH_OF_KEY(keys) % num_partitions_;
    try {
      const auto& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);

      if (redis_->hexists(hkey_kv, {reinterpret_cast<const char*>(keys), sizeof(TKey)})) {
        hit_count++;
      }
    } catch (sw::redis::Error& e) {
      throw DatabaseBackendError(get_name(), part, e.what());
    }
  } else {
    std::atomic<size_t> joint_hit_count(0);
    std::exception_ptr error;
    size_t error_part = -1;
    std::mutex error_guard;

    // Process partitions.
    std::vector<std::future<void>> tasks;
    tasks.reserve(num_partitions_);
    for (size_t part = 0; part < num_partitions_; part++) {
      tasks.emplace_back(workers_.submit([&, part]() {
        size_t hit_count = 0;
        try {
          const auto& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);

          sw::redis::Pipeline pipeline = redis_->pipeline(hkey_kv, false);
          size_t batch_size;

          size_t num_queries = 0;
          const TKey* const keys_end = &keys[num_keys];
          for (const TKey* k = keys; k != keys_end; num_queries++) {
            // Prepare and launch query.
            batch_size = 0;
            for (; k != keys_end && batch_size < max_set_batch_size_; k++) {
              if (HCTR_HASH_OF_KEY(k) % num_partitions_ == part) {
                pipeline.hexists(hkey_kv, {reinterpret_cast<const char*>(k), sizeof(TKey)});
                batch_size++;
              }
            }

            sw::redis::QueuedReplies replies = pipeline.exec();

            // Process results.
            for (size_t i = 0; i < replies.size(); i++) {
              if (replies.get<bool>(i)) {
                hit_count++;
              }
            }

            HCTR_LOG(TRACE, WORLD, "Redis partition %s, query %d: Found %d / %d keys.\n",
                     hkey_kv.c_str(), num_queries, hit_count, batch_size);
          }
        } catch (...) {
          std::unique_lock lock(error_guard);
          if (!error) {
            error = std::current_exception();
            error_part = part;
          }
        }
        joint_hit_count += hit_count;
      }));
    }
    ThreadPool::await(tasks.begin(), tasks.end());
    hit_count += static_cast<size_t>(joint_hit_count);
    try {
      if (error) {
        std::rethrow_exception(error);
      }
    } catch (sw::redis::Error& e) {
      throw DatabaseBackendError(get_name(), error_part, e.what());
    }
  }

  HCTR_LOG(DEBUG, WORLD, "%s backend. Table: %s. Found %d / %d keys.\n", get_name(),
           table_name.c_str(), hit_count, num_keys);
  return hit_count;
}

template <typename TKey>
bool RedisClusterBackend<TKey>::insert(const std::string& table_name, const size_t num_pairs,
                                       const TKey* const keys, const char* const values,
                                       const size_t value_size) {
  size_t num_inserts = 0;

  if (num_pairs <= 0) {
    // Do nothing ;-).
  } else if (num_pairs <= 1) {
    const size_t part = HCTR_HASH_OF_KEY(keys) % num_partitions_;
    try {
      const std::string& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);
      const std::string& hkey_kt = make_hash_key(table_name, part, REDIS_HKEY_TIME_SUFFIX);

      const time_t now = std::time(nullptr);

      // Insert.
      const std::string_view k_view{reinterpret_cast<const char*>(keys), sizeof(TKey)};
      redis_->hset(hkey_kv, k_view, {values, value_size});
      redis_->hset(hkey_kt, k_view, {reinterpret_cast<const char*>(&now), sizeof(time_t)});
      num_inserts++;

      // Queue overflow resolution.
      this->background_worker_.submit(
          [this, hkey_kv, hkey_kt]() { resolve_overflow_(hkey_kv, hkey_kt); });
    } catch (sw::redis::Error& e) {
      throw DatabaseBackendError(get_name(), part, e.what());
    }
  } else {
    std::atomic<size_t> joint_num_inserts(0);
    std::exception_ptr error;
    size_t error_part = -1;
    std::mutex error_guard;

    // Process partitions.
    std::vector<std::future<void>> tasks;
    tasks.reserve(num_partitions_);
    for (size_t part = 0; part < num_partitions_; part++) {
      tasks.emplace_back(workers_.submit([&, part]() {
        size_t num_inserts = 0;
        try {
          const std::string& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);
          const std::string& hkey_kt = make_hash_key(table_name, part, REDIS_HKEY_TIME_SUFFIX);

          std::vector<std::pair<std::string_view, std::string_view>> kv_views;
          std::vector<std::pair<std::string_view, std::string_view>> kt_views;

          size_t num_queries = 0;

          const TKey* const keys_end = &keys[num_pairs];
          for (const TKey* k = keys; k != keys_end; num_queries++) {
            const time_t now = std::time(nullptr);

            // Prepare and launch query.
            kt_views.clear();
            kv_views.clear();
            for (; k != keys_end && kv_views.size() < max_set_batch_size_; k++) {
              if (HCTR_HASH_OF_KEY(k) % num_partitions_ == part) {
                kv_views.emplace_back(
                    std::piecewise_construct,
                    std::forward_as_tuple(reinterpret_cast<const char*>(k), sizeof(TKey)),
                    std::forward_as_tuple(&values[(k - keys) * value_size], value_size));
                kt_views.emplace_back(
                    std::piecewise_construct,
                    std::forward_as_tuple(reinterpret_cast<const char*>(k), sizeof(TKey)),
                    std::forward_as_tuple(reinterpret_cast<const char*>(&now), sizeof(time_t)));
              }
            }
            if (kv_views.empty()) {
              continue;
            }

            // Insert.
            redis_->hmset(hkey_kv, kv_views.begin(), kv_views.end());
            redis_->hmset(hkey_kt, kt_views.begin(), kt_views.end());
            num_inserts += kv_views.size();

            HCTR_LOG_S(TRACE, WORLD)
                << get_name() << " backend. Partition " << hkey_kv << ", query " << num_queries
                << ": Inserted " << kv_views.size() << " pairs." << std::endl;

            // Queue overflow resolution.
            this->background_worker_.submit(
                [this, hkey_kv, hkey_kt]() { resolve_overflow_(hkey_kv, hkey_kt); });
          }
        } catch (...) {
          std::unique_lock lock(error_guard);
          error = std::current_exception();
          error_part = part;
        }
        joint_num_inserts += num_inserts;
      }));
    }
    ThreadPool::await(tasks.begin(), tasks.end());
    num_inserts += static_cast<size_t>(joint_num_inserts);
    try {
      if (error) {
        std::rethrow_exception(error);
      }
    } catch (sw::redis::Error& e) {
      throw DatabaseBackendError(get_name(), error_part, e.what());
    }
  }

  HCTR_LOG(DEBUG, WORLD, "%s backend. Table: %s. Inserted %d / %d pairs.\n", get_name(),
           table_name.c_str(), num_inserts, num_pairs);
  return true;
}

template <typename TKey>
size_t RedisClusterBackend<TKey>::fetch(const std::string& table_name, const size_t num_keys,
                                        const TKey* const keys, const DatabaseHitCallback& on_hit,
                                        const DatabaseMissCallback& on_miss) {
  size_t hit_count = 0;

  if (num_keys <= 0) {
    // Do nothing ;-).
  } else if (num_keys <= 1) {
    const size_t part = HCTR_HASH_OF_KEY(keys) % num_partitions_;
    try {
      const std::string& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);

      // Query.
      const auto& v_opt =
          redis_->hget(hkey_kv, {reinterpret_cast<const char*>(keys), sizeof(TKey)});

      // Process result.
      if (v_opt) {
        // Refresh timestamps if desired.
        if (this->refresh_time_after_fetch_) {
          const TKey k = *keys;
          const time_t now = std::time(nullptr);
          this->background_worker_.submit(
              [this, table_name, part, k, now]() { touch_(table_name, part, k, now); });
        }

        on_hit(0, v_opt->data(), v_opt->size());
        hit_count++;
      } else {
        on_miss(0);
      }
    } catch (sw::redis::Error& e) {
      throw DatabaseBackendError(get_name(), part, e.what());
    }
  } else {
    std::atomic<size_t> joint_hit_count(0);
    std::exception_ptr error;
    size_t error_part = -1;
    std::mutex error_guard;

    // Process partitions.
    std::vector<std::future<void>> tasks;
    tasks.reserve(num_partitions_);
    for (size_t part = 0; part < num_partitions_; part++) {
      tasks.emplace_back(workers_.submit([&, part]() {
        size_t hit_count = 0;
        try {
          const std::string& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);

          std::vector<size_t> indices;
          std::vector<std::string_view> k_views;
          std::vector<std::optional<std::string>> v_opts;
          std::shared_ptr<std::vector<TKey>> touched_keys;

          size_t num_queries = 0;

          const TKey* const keys_end = &keys[num_keys];
          for (const TKey* k = keys; k != keys_end; num_queries++) {
            // Prepare and launch query.
            indices.clear();
            k_views.clear();
            for (; k != keys_end; k++) {
              if (k_views.size() >= max_set_batch_size_) {
                break;
              }
              if (HCTR_HASH_OF_KEY(k) % num_partitions_ == part) {
                indices.push_back(k - keys);
                k_views.emplace_back(reinterpret_cast<const char*>(k), sizeof(TKey));
              }
            }
            if (k_views.empty()) {
              continue;
            }

            v_opts.clear();
            v_opts.reserve(k_views.size());
            redis_->hmget(hkey_kv, k_views.begin(), k_views.end(), std::back_inserter(v_opts));

            // Process results.
            auto indices_it = indices.begin();
            for (const auto& v_opt : v_opts) {
              if (v_opt) {
                on_hit(*indices_it, v_opt->data(), v_opt->size());
                hit_count++;

                if (this->refresh_time_after_fetch_) {
                  if (!touched_keys) {
                    touched_keys = std::make_shared<std::vector<TKey>>();
                  }
                  touched_keys->emplace_back(keys[*indices_it]);
                }
              } else {
                on_miss(*indices_it);
              }
              indices_it++;
            }
            HCTR_CHECK(indices_it == indices.end());

            // Refresh timestamps if desired.
            if (touched_keys && !touched_keys->empty()) {
              const time_t now = std::time(nullptr);
              this->background_worker_.submit([this, table_name, part, touched_keys, now]() {
                touch_(table_name, part, touched_keys, now);
              });
              touched_keys.reset();
            }

            HCTR_LOG_S(TRACE, WORLD)
                << get_name() << " backend. Partition " << hkey_kv << ", query " << num_queries
                << ": Fetched " << v_opts.size() << " keys. Hits " << hit_count << '.' << std::endl;
          }
        } catch (...) {
          std::unique_lock lock(error_guard);
          error = std::current_exception();
          error_part = part;
        }
        joint_hit_count += hit_count;
      }));
    }
    ThreadPool::await(tasks.begin(), tasks.end());
    hit_count += static_cast<size_t>(joint_hit_count);
    try {
      if (error) {
        std::rethrow_exception(error);
      }
    } catch (sw::redis::Error& e) {
      throw DatabaseBackendError(get_name(), error_part, e.what());
    }
  }

  HCTR_LOG(DEBUG, WORLD, "%s backend. Table: %s. Fetched %d / %d values.\n", get_name(),
           table_name.c_str(), hit_count, num_keys);
  return hit_count;
}

template <typename TKey>
size_t RedisClusterBackend<TKey>::fetch(const std::string& table_name, const size_t num_indices,
                                        const size_t* const indices, const TKey* const keys,
                                        const DatabaseHitCallback& on_hit,
                                        const DatabaseMissCallback& on_miss) {
  size_t hit_count = 0;

  if (num_indices <= 0) {
    // Do nothing ;-).
  } else if (num_indices <= 1) {
    const TKey& k = keys[*indices];
    const size_t part = HCTR_HASH_OF_KEY(&k) % num_partitions_;
    try {
      const auto& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);

      // Query.
      const auto& v_opt = redis_->hget(hkey_kv, {reinterpret_cast<const char*>(&k), sizeof(TKey)});

      // Process result.
      if (v_opt) {
        // Refresh timestamps if desired.
        if (this->refresh_time_after_fetch_) {
          const time_t now = std::time(nullptr);
          this->background_worker_.submit(
              [this, table_name, part, k, now]() { touch_(table_name, part, k, now); });
        }

        on_hit(*indices, v_opt->data(), v_opt->size());
        hit_count++;
      } else {
        on_miss(*indices);
      }
    } catch (sw::redis::Error& e) {
      throw DatabaseBackendError(get_name(), part, e.what());
    }
  } else {
    std::atomic<size_t> joint_hit_count(0);
    std::exception_ptr error;
    size_t error_part = -1;
    std::mutex error_guard;

    // Process partitions.
    std::vector<std::future<void>> tasks;
    tasks.reserve(num_partitions_);
    for (size_t part = 0; part < num_partitions_; part++) {
      tasks.emplace_back(workers_.submit([&, part]() {
        size_t hit_count = 0;
        try {
          const auto& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);

          std::vector<std::string_view> k_views;
          std::vector<std::optional<std::string>> v_opts;
          std::shared_ptr<std::vector<TKey>> touched_keys;

          size_t num_queries = 0;

          const size_t* const indices_end = &indices[num_indices];
          for (const size_t* i = indices; i != indices_end; num_queries++) {
            // Create and launch query.
            k_views.clear();
            for (const size_t* tmp_i = indices; tmp_i != indices_end; tmp_i++) {
              if (k_views.size() >= max_get_batch_size_) {
                break;
              }
              const TKey& k = keys[*tmp_i];
              if (HCTR_HASH_OF_KEY(&k) % num_partitions_ == part) {
                k_views.emplace_back(reinterpret_cast<const char*>(&k), sizeof(TKey));
              }
            }
            if (k_views.empty()) {
              continue;
            }

            v_opts.clear();
            v_opts.reserve(k_views.size());
            redis_->hmget(hkey_kv, k_views.begin(), k_views.end(), std::back_inserter(v_opts));

            // Process results.
            for (const auto& v_opt : v_opts) {
              if (v_opt) {
                on_hit(*i, v_opt->data(), v_opt->size());
                hit_count++;

                if (this->refresh_time_after_fetch_) {
                  if (!touched_keys) {
                    touched_keys = std::make_shared<std::vector<TKey>>();
                  }
                  touched_keys->emplace_back(keys[*i]);
                }
              } else {
                on_miss(*i);
              }
              i++;
            }

            // Refresh timestamps if desired.
            if (touched_keys && !touched_keys->empty()) {
              const time_t now = std::time(nullptr);
              this->background_worker_.submit([this, table_name, part, touched_keys, now]() {
                touch_(table_name, part, touched_keys, now);
              });
              touched_keys.reset();
            }

            HCTR_LOG_S(TRACE, WORLD)
                << get_name() << " backend. Partition " << hkey_kv << ", query " << num_queries
                << ": Fetched " << v_opts.size() << " keys. Hits " << hit_count << '.' << std::endl;
          }
        } catch (...) {
          std::unique_lock lock(error_guard);
          error = std::current_exception();
          error_part = part;
        }
        joint_hit_count += hit_count;
      }));
    }
    ThreadPool::await(tasks.begin(), tasks.end());
    hit_count += static_cast<size_t>(joint_hit_count);
    try {
      if (error) {
        std::rethrow_exception(error);
      }
    } catch (sw::redis::Error& e) {
      throw DatabaseBackendError(get_name(), error_part, e.what());
    }
  }

  HCTR_LOG(DEBUG, WORLD, "%s backend. Table: %s. Fetched %d / %d values.\n", get_name(),
           table_name.c_str(), hit_count, num_indices);
  return hit_count;
}

template <typename TKey>
size_t RedisClusterBackend<TKey>::evict(const std::string& table_name) {
  std::atomic<size_t> joint_hit_count(0);
  std::exception_ptr error;
  size_t error_part = -1;
  std::mutex error_guard;

  // Process partitions.
  std::vector<std::future<void>> tasks;
  tasks.reserve(num_partitions_);
  for (size_t part = 0; part < num_partitions_; part++) {
    tasks.emplace_back(workers_.submit([&, part]() {
      size_t hit_count;
      try {
        const auto& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);
        const auto& hkey_kt = make_hash_key(table_name, part, REDIS_HKEY_TIME_SUFFIX);

        hit_count = redis_->hlen(hkey_kv);

        // Delete the keys.
        redis_->del(hkey_kv);
        redis_->del(hkey_kt);

        HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend. Partition " << hkey_kv << ": Deleted "
                                 << hit_count << " keys." << std::endl;
      } catch (...) {
        std::unique_lock lock(error_guard);
        error = std::current_exception();
        error_part = part;
      }
      joint_hit_count += hit_count;
    }));
  }
  try {
    if (error) {
      std::rethrow_exception(error);
    }
  } catch (sw::redis::Error& e) {
    throw DatabaseBackendError(get_name(), error_part, e.what());
  }
  ThreadPool::await(tasks.begin(), tasks.end());
  const size_t hit_count = static_cast<size_t>(joint_hit_count);
  HCTR_LOG(DEBUG, WORLD, "%s backend. Table %s erased (%d pairs).\n", get_name(),
           table_name.c_str(), hit_count);
  return hit_count;
}

template <typename TKey>
size_t RedisClusterBackend<TKey>::evict(const std::string& table_name, const size_t num_keys,
                                        const TKey* const keys) {
  size_t hit_count = 0;

  if (num_keys <= 0) {
    // Do nothing ;-).
  } else if (num_keys <= 1) {
    const size_t part = HCTR_HASH_OF_KEY(keys) % num_partitions_;
    try {
      const auto& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);
      const auto& hkey_kt = make_hash_key(table_name, part, REDIS_HKEY_TIME_SUFFIX);

      const sw::redis::StringView k_view(reinterpret_cast<const char*>(keys), sizeof(TKey));

      hit_count += std::max(redis_->hdel(hkey_kv, k_view), redis_->hdel(hkey_kt, k_view));
    } catch (sw::redis::Error& e) {
      throw DatabaseBackendError(get_name(), part, e.what());
    }
  } else {
    std::atomic<size_t> joint_hit_count(0);
    std::exception_ptr error;
    size_t error_part = -1;
    std::mutex error_guard;

    // Process partitions.
    std::vector<std::future<void>> tasks;
    tasks.reserve(num_partitions_);
    for (size_t part = 0; part < num_partitions_; part++) {
      tasks.emplace_back(workers_.submit([&, part]() {
        size_t hit_count = 0;
        try {
          const auto& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);
          const auto& hkey_kt = make_hash_key(table_name, part, REDIS_HKEY_TIME_SUFFIX);

          std::vector<sw::redis::StringView> k_views;

          size_t num_queries = 0;

          const TKey* const keys_end = &keys[num_keys];
          for (const TKey* k = keys; k != keys_end; num_queries++) {
            k_views.clear();
            for (; k != keys_end && k_views.size() < max_set_batch_size_; k++) {
              if (HCTR_HASH_OF_KEY(k) % num_partitions_ == part) {
                k_views.emplace_back(reinterpret_cast<const char*>(k), sizeof(TKey));
              }
            }
            if (k_views.empty()) {
              continue;
            }

            hit_count += std::max(redis_->hdel(hkey_kv, k_views.begin(), k_views.end()),
                                  redis_->hdel(hkey_kt, k_views.begin(), k_views.end()));

            HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend. Partition " << hkey_kv
                                     << ", query " << num_queries << ": Deleted " << k_views.size()
                                     << " keys. Hits " << hit_count << '.' << std::endl;
          }
        } catch (...) {
          std::unique_lock lock(error_guard);
          error = std::current_exception();
          error_part = part;
        }
        joint_hit_count += hit_count;
      }));
    }
    ThreadPool::await(tasks.begin(), tasks.end());
    hit_count += static_cast<size_t>(joint_hit_count);
    try {
      if (error) {
        std::rethrow_exception(error);
      }
    } catch (sw::redis::Error& e) {
      throw DatabaseBackendError(get_name(), error_part, e.what());
    }
  }

  HCTR_LOG(DEBUG, WORLD, "%s backend. Table %s. %d / %d pairs erased.\n", get_name(),
           table_name.c_str(), hit_count, num_keys);
  return hit_count;
}

template <typename TKey>
void RedisClusterBackend<TKey>::resolve_overflow_(const std::string& hkey_kv,
                                                  const std::string& hkey_kt) {
  // Check overflow condition.
  size_t part_size = redis_->hlen(hkey_kt);
  if (part_size <= this->overflow_margin_) {
    return;
  }
  HCTR_LOG_S(TRACE, WORLD) << get_name() << " partition " << hkey_kv
                           << " is overflowing (size = " << part_size << " > "
                           << this->overflow_margin_ << "). Attempting to resolve..." << std::endl;

  // Select overflow resolution policy.
  if (this->overflow_policy_ == DatabaseOverflowPolicy_t::EvictOldest) {
    // Fetch keys and insert times.
    std::vector<std::pair<std::string, std::string>> kts;
    kts.reserve(part_size);
    redis_->hgetall(hkey_kt, std::back_inserter(kts));

    // Sanity check.
    part_size = kts.size();
    if (part_size <= this->overflow_resolution_target_) {
      return;
    }

    // Sort by ascending by time.
    for (const auto& kt : kts) {
      HCTR_CHECK_HINT(kt.second.size() == sizeof(time_t), "Value size mismatch!(%d <> %d)!",
                      kt.second.size(), sizeof(time_t));
    }
    std::sort(kts.begin(), kts.end(), [](const auto& kt0, const auto& kt1) {
      const time_t* t0 = reinterpret_cast<const time_t*>(kt0.second.data());
      const time_t* t1 = reinterpret_cast<const time_t*>(kt1.second.data());
      return *t0 < *t1;
    });

    std::vector<std::string_view> k;
    k.reserve(max_set_batch_size_);

    // Delete keys.
    const auto kt_end = kts.end();
    for (auto kt_it = kts.begin(); kt_it != kt_end;) {
      // Collect a batch.
      k.clear();
      for (; kt_it != kt_end && k.size() < max_set_batch_size_; kt_it++) {
        k.emplace_back(kt_it->first);
      }

      // Perform deletion.
      HCTR_LOG_S(TRACE, WORLD) << get_name() << " partition " << hkey_kv << " (size = " << part_size
                               << "). Attempting to evict " << k.size() << " OLD key/value pairs."
                               << std::endl;
      redis_->hdel(hkey_kv, k.begin(), k.end());
      redis_->hdel(hkey_kt, k.begin(), k.end());

      // Overflow resolved?
      part_size = redis_->hlen(hkey_kt);
      if (part_size <= this->overflow_resolution_target_) {
        kt_it = kt_end;
      }
    }
  } else if (this->overflow_policy_ == DatabaseOverflowPolicy_t::EvictRandom) {
    // Fetch all keys in partition.
    std::vector<std::string> k;
    k.reserve(part_size);
    redis_->hkeys(hkey_kt, std::back_inserter(k));

    // Sanity check.
    part_size = k.size();
    if (part_size <= this->overflow_resolution_target_) {
      return;
    }

    // Shuffle the keys.
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::shuffle(k.begin(), k.end(), gen);

    // Delete keys.
    const auto k_end = k.end();
    for (auto k_it = k.begin(); k_it != k_end;) {
      const auto k_next = std::min(k_it + max_set_batch_size_, k_end);

      // Perform deletion.
      HCTR_LOG_S(TRACE, WORLD) << get_name() << " partition " << hkey_kv << " (size = " << part_size
                               << "). Attempting to evict " << k.size()
                               << " RANDOM key/value pairs." << std::endl;
      redis_->hdel(hkey_kv, k_it, k_next);
      redis_->hdel(hkey_kt, k_it, k_next);

      // Overflow resolved?
      part_size = redis_->hlen(hkey_kt);
      k_it = (part_size <= this->overflow_resolution_target_) ? k_end : k_next;
    }
  } else {
    HCTR_LOG_S(WARNING, WORLD) << "Redis partition " << hkey_kv << " (size = " << part_size
                               << "), surpasses specified maximum size (=" << this->overflow_margin_
                               << "), but no compatible overflow policy (="
                               << this->overflow_policy_ << ") was selected!" << std::endl;
    return;
  }

  HCTR_LOG_S(DEBUG, WORLD) << get_name() << " partition " << hkey_kv
                           << " overflow resolution concluded!" << std::endl;
}

template <typename TKey>
void RedisClusterBackend<TKey>::touch_(const std::string& table_name, const size_t part,
                                       const TKey& key, const time_t time) {
  const std::string& hkey_kt = make_hash_key(table_name, part, REDIS_HKEY_TIME_SUFFIX);

  try {
    HCTR_LOG_S(TRACE, WORLD) << get_name() << ": Touching single key of " << hkey_kt << '.'
                             << std::endl;
    redis_->hset(hkey_kt, {reinterpret_cast<const char*>(&key), sizeof(TKey)},
                 {reinterpret_cast<const char*>(&time), sizeof(time_t)});
  } catch (sw::redis::Error& e) {
    HCTR_LOG_S(ERROR, WORLD) << get_name() << " partition " << hkey_kt
                             << "; error during refresh: " << e.what() << '.' << std::endl;
  }
}

template <typename TKey>
void RedisClusterBackend<TKey>::touch_(const std::string& table_name, const size_t part,
                                       const std::shared_ptr<std::vector<TKey>>& keys,
                                       const time_t time) {
  const std::string& hkey_kt = make_hash_key(table_name, part, REDIS_HKEY_TIME_SUFFIX);

  std::vector<std::pair<std::string_view, std::string_view>> kt_views;
  kt_views.reserve(keys->size());
  for (const TKey& k : *keys) {
    kt_views.emplace_back(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<const char*>(&k), sizeof(TKey)),
        std::forward_as_tuple(reinterpret_cast<const char*>(&time), sizeof(time_t)));
  }

  try {
    HCTR_LOG_S(TRACE, WORLD) << get_name() << ": Touching " << kt_views.size() << " keys of "
                             << hkey_kt << '.' << std::endl;
    redis_->hmset(hkey_kt, kt_views.begin(), kt_views.end());
  } catch (sw::redis::Error& e) {
    HCTR_LOG_S(ERROR, WORLD) << get_name() << " partition " << hkey_kt
                             << "; error touching refresh: " << e.what() << '.' << std::endl;
  }
}

template class RedisClusterBackend<unsigned int>;
template class RedisClusterBackend<long long>;

}  // namespace HugeCTR
