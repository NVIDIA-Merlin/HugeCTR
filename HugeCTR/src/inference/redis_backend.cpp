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
#include <inference/redis_backend.hpp>
#include <iostream>
#include <optional>
#include <random>
#include <unordered_set>

namespace HugeCTR {

const char* REDIS_HKEY_VALUE_SUFFIX = "v";
const char* REDIS_HKEY_TIME_SUFFIX = "t";

std::string make_hash_key(const std::string& table_name, const size_t partition,
                          const char* suffix) {
  static const char SEPARATOR = '/';
  std::stringstream ss;
  ss << table_name << SEPARATOR << 'p' << partition << SEPARATOR << suffix;
  return ss.str();
}

template <typename TKey>
RedisClusterBackend<TKey>::RedisClusterBackend(
    const std::string& address, const std::string& user_name, const std::string& password,
    const size_t num_partitions, const size_t max_get_batch_size, const size_t max_set_batch_size,
    const size_t overflow_margin, const DatabaseOverflowPolicy_t overflow_policy,
    const double overflow_resolution_target)
    : num_partitions_(num_partitions),
      // TODO: Make this configurable?
      thread_pool_(
          std::min(num_partitions, static_cast<size_t>(std::thread::hardware_concurrency()))),
      max_get_batch_size_(max_get_batch_size),
      max_set_batch_size_(max_set_batch_size),
      overflow_margin_(overflow_margin),
      overflow_policy_(overflow_policy),
      overflow_resolution_target_(hctr_safe_cast<size_t>(
          static_cast<double>(overflow_margin) * overflow_resolution_target + 0.5)) {
  HCTR_CHECK(overflow_resolution_target_ <= overflow_margin_);

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

  HCTR_LOG(INFO, WORLD, "Connecting to Redis cluster via %s:%d ...\n", options.host.c_str(),
           options.port);
  redis_ = std::make_unique<sw::redis::RedisCluster>(options, pool_options);
  HCTR_LOG(INFO, WORLD, "Connected to Redis database!\n");
}

template <typename TKey>
RedisClusterBackend<TKey>::~RedisClusterBackend() {
  HCTR_LOG(INFO, WORLD, "Disconnecting from Redis database...\n");
  redis_.reset();
  HCTR_LOG(INFO, WORLD, "Disconnected from Redis database!\n");
}

template <typename TKey>
size_t RedisClusterBackend<TKey>::contains(const std::string& table_name, const size_t num_keys,
                                           const TKey* const keys) const {
  size_t hit_count = 0;

  if (num_keys <= 0) {
    // Do nothing ;-).
  } else if (num_keys <= 1) {
    const size_t part = *keys % num_partitions_;
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
    for (size_t part = 0; part < num_partitions_; part++) {
      thread_pool_.submit([&, part]() {
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
            for (; k != keys_end; k++) {
              if (batch_size >= max_set_batch_size_) {
                break;
              }
              if (*k % num_partitions_ == part) {
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

            HCTR_LOG(DEBUG, WORLD, "Redis partition %s, query %d: Found %d / %d keys.\n",
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
      });
    }
    thread_pool_.await_idle();
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
    const size_t part = *keys % num_partitions_;
    try {
      const auto& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);
      const auto& hkey_kt = make_hash_key(table_name, part, REDIS_HKEY_TIME_SUFFIX);

      const time_t unix_time = time(nullptr);
      const sw::redis::StringView k_view(reinterpret_cast<const char*>(keys), sizeof(TKey));

      redis_->hset(hkey_kv, k_view, {values, value_size});
      redis_->hset(hkey_kt, k_view, {reinterpret_cast<const char*>(&unix_time), sizeof(time_t)});

      num_inserts++;
    } catch (sw::redis::Error& e) {
      throw DatabaseBackendError(get_name(), part, e.what());
    }
  } else {
    std::atomic<size_t> joint_num_inserts(0);
    std::exception_ptr error;
    size_t error_part = -1;
    std::mutex error_guard;

    // Process partitions.
    for (size_t part = 0; part < num_partitions_; part++) {
      thread_pool_.submit([&, part]() {
        size_t num_inserts = 0;
        try {
          const auto& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);
          const auto& hkey_kt = make_hash_key(table_name, part, REDIS_HKEY_TIME_SUFFIX);

          std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kv_views;
          std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kt_views;

          size_t num_queries = 0;

          const TKey* const keys_end = &keys[num_pairs];
          for (const TKey* k = keys; k != keys_end; num_queries++) {
            const time_t unix_time = time(nullptr);

            // Partition overflow handling.
            const size_t partition_size = redis_->hlen(hkey_kv);
            if (partition_size > overflow_margin_) {
              if (overflow_policy_ == DatabaseOverflowPolicy_t::EvictOldest) {
                // Fetch keys and insert times.
                std::vector<std::pair<std::string, std::string>> kt;
                kt.reserve(partition_size);
                redis_->hgetall(hkey_kt, std::back_inserter(kt));

                const size_t evict_amount = kt.size() - overflow_resolution_target_;
                if (evict_amount > 0) {
                  // Sort by ascending by time.
                  std::sort(kt.begin(), kt.end(), [](const auto& a, const auto& b) {
                    HCTR_CHECK_HINT(a.second.size() == sizeof(time_t), "Value size mismatch!");
                    HCTR_CHECK_HINT(b.second.size() == sizeof(time_t), "Value size mismatch!");
                    const time_t* t0 = reinterpret_cast<const time_t*>(a.second.data());
                    const time_t* t1 = reinterpret_cast<const time_t*>(b.second.data());
                    return *t0 < *t1;
                  });

                  // Take keys until eviction theshold has been reached.
                  std::vector<sw::redis::StringView> k;
                  k.reserve(evict_amount);
                  const auto kt_end = kt.begin() + evict_amount;
                  for (auto kt_it = kt.begin(); kt_it != kt_end; kt_it++) {
                    k.emplace_back(kt_it->first);
                  }

                  // Perform deletions.
                  HCTR_LOG(
                      INFO, WORLD,
                      "Redis partition %s (size = %d > %d). Overflow detected, evicting the %d "
                      "OLDEST key/value pairs!\n",
                      hkey_kv.c_str(), partition_size, overflow_margin_, evict_amount);

                  redis_->hdel(hkey_kv, k.begin(), k.end());
                  redis_->hdel(hkey_kt, k.begin(), k.end());
                }
              } else if (overflow_policy_ == DatabaseOverflowPolicy_t::EvictRandom) {
                // Fetch all keys in partition.
                std::vector<std::string> k;
                k.reserve(partition_size);
                redis_->hkeys(hkey_kt, std::back_inserter(k));

                const size_t evict_amount = k.size() - overflow_resolution_target_;
                if (evict_amount > 0) {
                  // Shuffle the keys.
                  std::random_device rd;
                  std::default_random_engine gen(rd());
                  std::shuffle(k.begin(), k.end(), gen);

                  // Perform deletions.
                  HCTR_LOG(INFO, WORLD,
                           "Redis partition %s (size = %d > %d). Overflow detected, evicting %d "
                           "RANDOM key/value pairs!\n",
                           hkey_kv.c_str(), partition_size, overflow_margin_, evict_amount);

                  redis_->hdel(hkey_kv, &k[0], &k[evict_amount]);
                  redis_->hdel(hkey_kt, &k[0], &k[evict_amount]);
                }
              } else {
                HCTR_LOG(
                    DEBUG, WORLD,
                    "Redis partition %s (size = %d), surpasses specified maximum size (= %d), but "
                    "no compatible overflow policy (=%d) was selected!\n",
                    hkey_kv.c_str(), partition_size, overflow_margin_, overflow_policy_);
              }
            }

            // Prepare and launch query.
            kt_views.clear();
            kv_views.clear();
            for (; k != keys_end; k++) {
              if (kv_views.size() >= max_set_batch_size_) {
                break;
              }
              if (*k % num_partitions_ == part) {
                kv_views.emplace_back(
                    std::piecewise_construct,
                    std::forward_as_tuple(reinterpret_cast<const char*>(k), sizeof(TKey)),
                    std::forward_as_tuple(&values[(k - keys) * value_size], value_size));
                kt_views.emplace_back(
                    std::piecewise_construct,
                    std::forward_as_tuple(reinterpret_cast<const char*>(k), sizeof(TKey)),
                    std::forward_as_tuple(reinterpret_cast<const char*>(&unix_time),
                                          sizeof(time_t)));
              }
            }
            if (kv_views.empty()) {
              continue;
            }

            redis_->hmset(hkey_kv, kv_views.begin(), kv_views.end());
            redis_->hmset(hkey_kt, kt_views.begin(), kt_views.end());
            num_inserts += kv_views.size();

            HCTR_LOG(DEBUG, WORLD, "Redis partition %s, query %d: Inserted %d pairs.\n",
                     hkey_kv.c_str(), num_queries, kv_views.size());
          }
        } catch (...) {
          std::unique_lock lock(error_guard);
          error = std::current_exception();
          error_part = part;
        }
        joint_num_inserts += num_inserts;
      });
    }
    thread_pool_.await_idle();
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
                                        const TKey* const keys, char* const values,
                                        const size_t value_size,
                                        MissingKeyCallback& missing_callback) const {
  size_t hit_count = 0;

  if (num_keys <= 0) {
    // Do nothing ;-).
  } else if (num_keys <= 1) {
    const size_t part = *keys % num_partitions_;
    try {
      const auto& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);

      const auto& v_opt =
          redis_->hget(hkey_kv, {reinterpret_cast<const char*>(keys), sizeof(TKey)});

      // Process result.
      if (v_opt) {
        HCTR_CHECK_HINT(v_opt->size() == value_size, "Redis return value size mismatch!");
        memcpy(values, v_opt->data(), value_size);
        hit_count++;
      } else {
        missing_callback(0);
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
    for (size_t part = 0; part < num_partitions_; part++) {
      thread_pool_.submit([&, part]() {
        size_t hit_count = 0;
        try {
          const auto& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);

          std::vector<size_t> i_buffer;
          std::vector<sw::redis::StringView> k_views;
          std::vector<sw::redis::OptionalString> v_views_opt;

          size_t num_queries = 0;

          const TKey* const keys_end = &keys[num_keys];
          for (const TKey* k = keys; k != keys_end; num_queries++) {
            // Prepare and launch query.
            i_buffer.clear();
            k_views.clear();
            for (; k != keys_end; k++) {
              if (k_views.size() >= max_set_batch_size_) {
                break;
              }
              if (*k % num_partitions_ == part) {
                i_buffer.push_back(k - keys);
                k_views.emplace_back(reinterpret_cast<const char*>(k), sizeof(TKey));
              }
            }
            if (k_views.empty()) {
              continue;
            }

            v_views_opt.clear();
            v_views_opt.reserve(k_views.size());
            redis_->hmget(hkey_kv, k_views.begin(), k_views.end(), std::back_inserter(v_views_opt));

            // Process results.
            auto i_buffer_it = i_buffer.begin();
            for (const auto& v_opt : v_views_opt) {
              if (v_opt) {
                HCTR_CHECK_HINT(v_opt->size() == value_size, "Redis return value size mismatch!");
                memcpy(&values[*i_buffer_it * value_size], v_opt->data(), value_size);
                hit_count++;
              } else {
                missing_callback(*i_buffer_it);
              }
              i_buffer_it++;
            }

            HCTR_LOG(DEBUG, WORLD, "Redis partition %s, query %d: Fetched %d keys. Hits %d.\n",
                     hkey_kv.c_str(), num_queries, v_views_opt.size(), hit_count);
          }
        } catch (...) {
          std::unique_lock lock(error_guard);
          error = std::current_exception();
          error_part = part;
        }
        joint_hit_count += hit_count;
      });
    }
    thread_pool_.await_idle();
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
                                        char* const values, const size_t value_size,
                                        MissingKeyCallback& missing_callback) const {
  size_t hit_count = 0;

  if (num_indices <= 0) {
    // Do nothing ;-).
  } else if (num_indices <= 1) {
    const TKey& k = keys[*indices];
    const size_t part = k % num_partitions_;
    try {
      const auto& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);

      const auto& v_opt = redis_->hget(hkey_kv, {reinterpret_cast<const char*>(&k), sizeof(TKey)});

      // Process result.
      if (v_opt) {
        HCTR_CHECK_HINT(v_opt->size() == value_size, "Redis return value size mismatch!");
        memcpy(&values[*indices * value_size], v_opt->data(), value_size);
        hit_count++;
      } else {
        missing_callback(*indices);
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
    for (size_t part = 0; part < num_partitions_; part++) {
      thread_pool_.submit([&, part]() {
        size_t hit_count = 0;
        try {
          const auto& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);

          std::vector<sw::redis::StringView> k_views;
          std::vector<sw::redis::OptionalString> v_views_opt;

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
              if (k % num_partitions_ == part) {
                k_views.emplace_back(reinterpret_cast<const char*>(&k), sizeof(TKey));
              }
            }
            if (k_views.empty()) {
              continue;
            }

            v_views_opt.clear();
            v_views_opt.reserve(k_views.size());
            redis_->hmget(hkey_kv, k_views.begin(), k_views.end(), std::back_inserter(v_views_opt));

            // Process results.
            for (const auto& v_opt : v_views_opt) {
              if (v_opt) {
                HCTR_CHECK_HINT(v_opt->size() == value_size, "Redis return value size mismatch!");
                memcpy(&values[*i * value_size], v_opt->data(), value_size);
                hit_count++;
              } else {
                missing_callback(*i);
              }
              i++;
            }
          }

          HCTR_LOG(DEBUG, WORLD, "Redis partition %s, query %d: Fetched %d keys. Hits %d.\n",
                   hkey_kv.c_str(), num_queries, v_views_opt.size(), hit_count);
        } catch (...) {
          std::unique_lock lock(error_guard);
          error = std::current_exception();
          error_part = part;
        }
        joint_hit_count += hit_count;
      });
    }
    thread_pool_.await_idle();
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
  std::exception_ptr error;
  size_t error_part = -1;
  std::mutex error_guard;

  auto fn = [&](const size_t part) -> size_t {
    size_t hit_count = 0;
    try {
      const auto& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);
      const auto& hkey_kt = make_hash_key(table_name, part, REDIS_HKEY_TIME_SUFFIX);

      hit_count += redis_->hlen(hkey_kv);

      // Delete the keys.
      redis_->del(hkey_kv);
      redis_->del(hkey_kt);

      HCTR_LOG(DEBUG, WORLD, "Redis partition %s: Deleted %d keys.\n", hkey_kv.c_str(), hit_count);
    } catch (...) {
      std::unique_lock lock(error_guard);
      error = std::current_exception();
      error_part = part;
    }
    return hit_count;
  };

  // Process partitions.
  size_t hit_count = 0;
  for (size_t part = 0; part < num_partitions_; part++) {
    hit_count += fn(part);
  }
  try {
    if (error) {
      std::rethrow_exception(error);
    }
  } catch (sw::redis::Error& e) {
    throw DatabaseBackendError(get_name(), error_part, e.what());
  }

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
    const size_t part = *keys % num_partitions_;
    try {
      const auto& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);
      const auto& hkey_kt = make_hash_key(table_name, part, REDIS_HKEY_TIME_SUFFIX);

      const sw::redis::StringView k_view(reinterpret_cast<const char*>(keys), sizeof(TKey));

      hit_count += redis_->hdel(hkey_kv, k_view) * 0;
      hit_count += redis_->hdel(hkey_kt, k_view);
    } catch (sw::redis::Error& e) {
      throw DatabaseBackendError(get_name(), part, e.what());
    }
  } else {
    std::exception_ptr error;
    size_t error_part = -1;
    std::mutex error_guard;

    auto fn = [&](const size_t part) -> size_t {
      size_t hit_count = 0;
      try {
        const auto& hkey_kv = make_hash_key(table_name, part, REDIS_HKEY_VALUE_SUFFIX);
        const auto& hkey_kt = make_hash_key(table_name, part, REDIS_HKEY_TIME_SUFFIX);

        std::vector<sw::redis::StringView> k_views;

        size_t num_queries = 0;

        const TKey* const keys_end = &keys[num_keys];
        for (const TKey* k = keys; k != keys_end; num_queries++) {
          k_views.clear();
          for (; k != keys_end; k++) {
            if (k_views.size() >= max_set_batch_size_) {
              break;
            }
            if (*k % num_partitions_ == part) {
              k_views.emplace_back(reinterpret_cast<const char*>(k), sizeof(TKey));
            }
          }
          if (k_views.empty()) {
            continue;
          }

          hit_count += redis_->hdel(hkey_kv, k_views.begin(), k_views.end()) * 0;
          hit_count += redis_->hdel(hkey_kt, k_views.begin(), k_views.end());

          HCTR_LOG(DEBUG, WORLD, "Redis partition %s, query %d: Deleted %d keys. Hits %d.\n",
                   hkey_kv.c_str(), num_queries, k_views.size(), hit_count);
        }
      } catch (...) {
        std::unique_lock lock(error_guard);
        error = std::current_exception();
        error_part = part;
      }
      return hit_count;
    };

    // Process partitions.
    for (size_t part = 0; part < num_partitions_; part++) {
      hit_count += fn(part);
    }
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

template class RedisClusterBackend<unsigned int>;
template class RedisClusterBackend<long long>;

}  // namespace HugeCTR