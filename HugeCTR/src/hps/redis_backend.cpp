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
#include <hps/hier_parameter_server_base.hpp>
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

#ifdef HCTR_KEY_TO_PART_INDEX
#error "HCTR_KEY_TO_PART_INDEX is not supposed to be defined at this point!"
#else
#define HCTR_USE_XXHASH
#ifdef HCTR_USE_XXHASH
#include <xxh3.h>
#define HCTR_KEY_TO_PART_INDEX(KEY) (XXH3_64bits(&(KEY), sizeof(TKey)) % num_partitions_)
#else
#define HCTR_KEY_TO_PART_INDEX(KEY) (KEY % num_partitions_)
#endif
#endif

namespace HugeCTR {

inline std::string make_hkey(const std::string& table_name, const size_t partition,
                             const char suffix) {
  static const char separator = '/';

  std::ostringstream os;
  os << table_name << separator << 'p' << partition << separator << suffix;
  return os.str();
}

#ifdef HCTR_REDIS_VALUE_HKEY
#error HCTR_REDIS_VALUE_HKEY should not be defined!
#else
#define HCTR_REDIS_VALUE_HKEY() (make_hkey(table_name, part, 'v'))
#endif
#ifdef HCTR_REDIS_TIME_HKEY
#error HCTR_REDIS_TIME_HKEY should not be defined!
#else
#define HCTR_REDIS_TIME_HKEY() (make_hkey(table_name, part, 't'))
#endif

template <typename TKey>
RedisClusterBackend<TKey>::RedisClusterBackend(
    const std::string& address, const std::string& user_name, const std::string& password,
    const size_t num_partitions, const size_t max_get_batch_size, const size_t max_set_batch_size,
    const bool refresh_time_after_fetch, const size_t overflow_margin,
    const DatabaseOverflowPolicy_t overflow_policy, const double overflow_resolution_target)
    : TBase(max_get_batch_size, max_set_batch_size, overflow_margin, overflow_policy,
            overflow_resolution_target),
      refresh_time_after_fetch_{refresh_time_after_fetch},
      // Can switch to std::range in C++20.
      num_partitions_{num_partitions} {
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
  std::atomic<size_t> joint_num_pairs{0};

  std::exception_ptr error;
  size_t error_part = -1;
  std::mutex error_guard;

  // Process partitions.
  std::vector<std::future<void>> tasks;
  tasks.reserve(num_partitions_);

  for (size_t part = 0; part < num_partitions_; part++) {
    tasks.emplace_back(ThreadPool::get().submit([&, part]() {
      // Precalc constants.
      const std::string& hkey = HCTR_REDIS_VALUE_HKEY();

      try {
        joint_num_pairs += redis_->hlen(hkey);
      } catch (...) {
        std::unique_lock lock(error_guard);
        error = std::current_exception();
        error_part = part;
      }
    }));
  }
  ThreadPool::await(tasks.begin(), tasks.end());
  const size_t num_pairs = static_cast<size_t>(joint_num_pairs);

  // Handle errors.
  try {
    if (error) {
      std::rethrow_exception(error);
    }
  } catch (sw::redis::Error& e) {
    throw DatabaseBackendError(get_name(), error_part, e.what());
  }

  return num_pairs;
}

template <typename TKey>
size_t RedisClusterBackend<TKey>::contains(const std::string& table_name, const size_t num_keys,
                                           const TKey* const keys,
                                           const std::chrono::microseconds& time_budget) const {
  const auto begin = std::chrono::high_resolution_clock::now();

  size_t hit_count = 0;
  size_t ign_count = 0;

  switch (num_keys) {
    case 0: {
      // Do nothing ;-).
    } break;
    case 1: {
      // Precalc constants.
      const size_t part = HCTR_KEY_TO_PART_INDEX(*keys);
      const std::string& hkey = HCTR_REDIS_VALUE_HKEY();

      // Check time-budget.
      const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
      if (elapsed >= time_budget) {
        HCTR_LOG_S(WARNING, WORLD)
            << get_name() << " backend; Partition " << hkey << ": Timeout!" << std::endl;
        ign_count++;
        break;
      }

      // Launch query.
      try {
        if (redis_->hexists(hkey, {reinterpret_cast<const char*>(keys), sizeof(TKey)})) {
          hit_count++;
        }
      } catch (sw::redis::Error& e) {
        throw DatabaseBackendError(get_name(), part, e.what());
      }
    } break;
    default: {
      std::atomic<size_t> joint_hit_count{0};
      std::atomic<size_t> joint_ign_count{0};

      std::exception_ptr error;
      size_t error_part = -1;
      std::mutex error_guard;

      // Precalc constants.
      const TKey* const keys_end = &keys[num_keys];

      // Process partitions.
      std::vector<std::future<void>> tasks;
      tasks.reserve(num_partitions_);

      for (size_t part = 0; part < num_partitions_; part++) {
        tasks.emplace_back(ThreadPool::get().submit([&, part]() {
          size_t hit_count = 0;

          // Precalc constants.
          const std::string& hkey = HCTR_REDIS_VALUE_HKEY();

          try {
            size_t num_batches = 0;
            for (const TKey* k = keys; k != keys_end; num_batches++) {
              // Check time budget.
              const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
              if (elapsed >= time_budget) {
                HCTR_LOG_S(WARNING, WORLD)
                    << get_name() << " backend; Partition " << hkey << ": Timeout!" << std::endl;

                size_t ign_count = 0;
                for (; k != keys_end; k++) {
                  if (HCTR_KEY_TO_PART_INDEX(*k) == part) {
                    ign_count++;
                  }
                }
                joint_ign_count += ign_count;
                break;
              }

              // Prepare and launch query.
              sw::redis::Pipeline pipeline = redis_->pipeline(hkey, false);

              size_t batch_size = 0;
              for (; k != keys_end; k++) {
                if (HCTR_KEY_TO_PART_INDEX(*k) == part) {
                  pipeline.hexists(hkey, {reinterpret_cast<const char*>(k), sizeof(TKey)});
                  if (++batch_size >= this->max_get_batch_size_) {
                    break;
                  }
                }
              }
              sw::redis::QueuedReplies replies = pipeline.exec();

              // Process results.
              size_t batch_hits = 0;
              for (size_t i = 0; i < replies.size(); i++) {
                if (replies.get<bool>(i)) {
                  batch_hits++;
                }
              }

              HCTR_LOG_S(TRACE, WORLD)
                  << get_name() << " backend; partition " << hkey << ", batch " << num_batches
                  << ": " << batch_hits << " / " << batch_size << " hits. Time: " << elapsed.count()
                  << " / " << time_budget.count() << " us." << std::endl;
              hit_count += batch_hits;
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
      ign_count += static_cast<size_t>(joint_ign_count);

      // Handle errors.
      try {
        if (error) {
          std::rethrow_exception(error);
        }
      } catch (sw::redis::Error& e) {
        throw DatabaseBackendError(get_name(), error_part, e.what());
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": " << hit_count
                           << " / " << (num_keys - ign_count) << " hits, " << ign_count
                           << " ignored." << std::endl;
  return hit_count;
}

template <typename TKey>
bool RedisClusterBackend<TKey>::insert(const std::string& table_name, const size_t num_pairs,
                                       const TKey* const keys, const char* const values,
                                       const size_t value_size) {
  size_t num_inserts = 0;

  switch (num_pairs) {
    case 0: {
      // Do nothing ;-).
    } break;
    case 1: {
      // Precalc constants.
      const size_t part = HCTR_KEY_TO_PART_INDEX(*keys);
      const std::string& hkey_v = HCTR_REDIS_VALUE_HKEY();
      const std::string& hkey_t = HCTR_REDIS_TIME_HKEY();

      // Insert.
      try {
        const std::string_view k_view{reinterpret_cast<const char*>(keys), sizeof(TKey)};
        const time_t now = std::time(nullptr);

        redis_->hset(hkey_v, k_view, {values, value_size});
        redis_->hset(hkey_t, k_view, {reinterpret_cast<const char*>(&now), sizeof(time_t)});
        num_inserts++;
      } catch (sw::redis::Error& e) {
        throw DatabaseBackendError(get_name(), part, e.what());
      }

      // Queue overflow resolution.
      this->background_worker_.submit(
          [this, hkey_v, hkey_t]() { check_and_resolve_overflow_(hkey_v, hkey_t); });
    } break;
    default: {
      std::atomic<size_t> joint_num_inserts{0};

      std::exception_ptr error;
      size_t error_part = -1;
      std::mutex error_guard;

      // Precalc constants.
      const TKey* const keys_end = &keys[num_pairs];

      // Process partitions.
      std::vector<std::future<void>> tasks;
      tasks.reserve(num_partitions_);

      for (size_t part = 0; part < num_partitions_; part++) {
        tasks.emplace_back(ThreadPool::get().submit([&, part]() {
          size_t num_inserts = 0;

          // Precalc constants.
          const std::string& hkey_v = HCTR_REDIS_VALUE_HKEY();
          const std::string& hkey_t = HCTR_REDIS_TIME_HKEY();

          try {
            std::vector<std::pair<std::string_view, std::string_view>> v_views;
            std::vector<std::pair<std::string_view, std::string_view>> t_views;

            size_t num_batches = 0;
            for (const TKey* k = keys; k != keys_end; num_batches++) {
              const time_t now = std::time(nullptr);

              // Prepare and launch query.
              t_views.clear();
              v_views.clear();
              for (; k != keys_end; k++) {
                if (HCTR_KEY_TO_PART_INDEX(*k) == part) {
                  v_views.emplace_back(
                      std::piecewise_construct,
                      std::forward_as_tuple(reinterpret_cast<const char*>(k), sizeof(TKey)),
                      std::forward_as_tuple(&values[(k - keys) * value_size], value_size));
                  t_views.emplace_back(
                      std::piecewise_construct,
                      std::forward_as_tuple(reinterpret_cast<const char*>(k), sizeof(TKey)),
                      std::forward_as_tuple(reinterpret_cast<const char*>(&now), sizeof(time_t)));
                  if (t_views.size() >= this->max_set_batch_size_) {
                    break;
                  }
                }
              }
              if (v_views.empty()) {
                continue;
              }

              redis_->hmset(hkey_v, v_views.begin(), v_views.end());
              redis_->hmset(hkey_t, t_views.begin(), t_views.end());
              num_inserts += t_views.size();

              // Queue overflow resolution.
              this->background_worker_.submit(
                  [this, hkey_v, hkey_t]() { check_and_resolve_overflow_(hkey_v, hkey_t); });

              HCTR_LOG_S(TRACE, WORLD)
                  << get_name() << " backend; Partition " << hkey_v << ", batch " << num_batches
                  << ": Inserted " << v_views.size() << " pairs." << std::endl;
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

      // Handle errors.
      try {
        if (error) {
          std::rethrow_exception(error);
        }
      } catch (sw::redis::Error& e) {
        throw DatabaseBackendError(get_name(), error_part, e.what());
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": Inserted "
                           << num_inserts << " / " << num_pairs << " pairs." << std::endl;
  return true;
}

template <typename TKey>
size_t RedisClusterBackend<TKey>::fetch(const std::string& table_name, const size_t num_keys,
                                        const TKey* const keys, const DatabaseHitCallback& on_hit,
                                        const DatabaseMissCallback& on_miss,
                                        const std::chrono::microseconds& time_budget) {
  const auto begin = std::chrono::high_resolution_clock::now();

  size_t hit_count = 0;
  size_t ign_count = 0;

  switch (num_keys) {
    case 0: {
      // Do nothing ;-).
    } break;
    case 1: {
      // Precalc constants.
      const size_t part = HCTR_KEY_TO_PART_INDEX(*keys);
      const std::string& hkey_v = HCTR_REDIS_VALUE_HKEY();

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
            redis_->hget(hkey_v, {reinterpret_cast<const char*>(keys), sizeof(TKey)});

        // Process result.
        if (v_opt) {
          on_hit(0, v_opt->data(), v_opt->size());
          hit_count++;

          // Queue timestamp refresh.
          if (this->refresh_time_after_fetch_) {
            const TKey k = *keys;
            const time_t now = std::time(nullptr);

            this->background_worker_.submit([this, table_name, part, k, now]() {
              const std::string& hkey_t = HCTR_REDIS_TIME_HKEY();
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
      std::atomic<size_t> joint_hit_count{0};
      std::atomic<size_t> joint_ign_count{0};

      std::exception_ptr error;
      size_t error_part = -1;
      std::mutex error_guard;

      // Precalc constants.
      const TKey* const keys_end = &keys[num_keys];

      // Process partitions.
      std::vector<std::future<void>> tasks;
      tasks.reserve(num_partitions_);

      for (size_t part = 0; part < num_partitions_; part++) {
        tasks.emplace_back(ThreadPool::get().submit([&, part]() {
          size_t hit_count = 0;

          // Precalc constants.
          const std::string& hkey_v = HCTR_REDIS_VALUE_HKEY();

          try {
            std::vector<size_t> i_vals;
            std::vector<std::string_view> k_views;
            std::vector<std::optional<std::string>> v_opts;
            std::shared_ptr<std::vector<TKey>> touched_keys;

            size_t num_batches = 0;
            for (const TKey* k = keys; k != keys_end; num_batches++) {
              // Check time budget.
              const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
              if (elapsed >= time_budget) {
                HCTR_LOG_S(WARNING, WORLD)
                    << get_name() << " backend; Partition " << hkey_v << ": Timeout!" << std::endl;

                size_t ign_count = 0;
                for (; k != keys_end; k++) {
                  if (HCTR_KEY_TO_PART_INDEX(*k) == part) {
                    on_miss(0);
                    ign_count++;
                  }
                }
                joint_ign_count += ign_count;
                break;
              }

              // Prepare query.
              k_views.clear();
              i_vals.clear();
              for (; k != keys_end; k++) {
                if (HCTR_KEY_TO_PART_INDEX(*k) == part) {
                  i_vals.emplace_back(k - keys);
                  k_views.emplace_back(reinterpret_cast<const char*>(k), sizeof(TKey));
                  if (k_views.size() >= this->max_get_batch_size_) {
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
              auto i_vals_it = i_vals.begin();
              for (const auto& v_opt : v_opts) {
                if (v_opt) {
                  on_hit(*i_vals_it, v_opt->data(), v_opt->size());
                  hit_count++;

                  if (this->refresh_time_after_fetch_) {
                    if (!touched_keys) {
                      touched_keys = std::make_shared<std::vector<TKey>>();
                    }
                    touched_keys->emplace_back(keys[*i_vals_it]);
                  }
                } else {
                  on_miss(*i_vals_it);
                }
                i_vals_it++;
              }
              HCTR_CHECK(i_vals_it == i_vals.end());

              // Refresh timestamps if desired.
              if (touched_keys && !touched_keys->empty()) {
                const time_t now = std::time(nullptr);

                this->background_worker_.submit([this, table_name, part, touched_keys, now]() {
                  const std::string& hkey_t = HCTR_REDIS_TIME_HKEY();
                  touch_(hkey_t, touched_keys, now);
                });
                touched_keys.reset();
              }

              HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend. Partition " << hkey_v
                                       << ", batch " << num_batches << ": Fetched " << v_opts.size()
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
      ign_count += static_cast<size_t>(joint_ign_count);

      // Handle errors.
      try {
        if (error) {
          std::rethrow_exception(error);
        }
      } catch (sw::redis::Error& e) {
        throw DatabaseBackendError(get_name(), error_part, e.what());
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": " << hit_count
                           << " / " << (num_keys - ign_count) << " hits, " << ign_count
                           << " ignored." << std::endl;
  return hit_count;
}

template <typename TKey>
size_t RedisClusterBackend<TKey>::fetch(const std::string& table_name, const size_t num_indices,
                                        const size_t* indices, const TKey* const keys,
                                        const DatabaseHitCallback& on_hit,
                                        const DatabaseMissCallback& on_miss,
                                        const std::chrono::microseconds& time_budget) {
  const auto begin = std::chrono::high_resolution_clock::now();

  size_t hit_count = 0;
  size_t ign_count = 0;

  switch (num_indices) {
    case 0: {
      // Do nothing ;-).
    } break;
    case 1: {
      // Precalc constants.
      const TKey k = keys[*indices];
      const size_t part = HCTR_KEY_TO_PART_INDEX(k);
      const std::string& hkey_v = HCTR_REDIS_VALUE_HKEY();

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
            redis_->hget(hkey_v, {reinterpret_cast<const char*>(&k), sizeof(TKey)});

        // Process result.
        if (v_opt) {
          on_hit(*indices, v_opt->data(), v_opt->size());
          hit_count++;

          // Queue timestamp refresh.
          if (this->refresh_time_after_fetch_) {
            const time_t now = std::time(nullptr);

            this->background_worker_.submit([this, table_name, part, k, now]() {
              const std::string& hkey_t = HCTR_REDIS_TIME_HKEY();
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
      std::atomic<size_t> joint_hit_count{0};
      std::atomic<size_t> joint_ign_count{0};

      std::exception_ptr error;
      size_t error_part = -1;
      std::mutex error_guard;

      // Precalc constants.
      const size_t* const indices_end = &indices[num_indices];

      // Process partitions.
      std::vector<std::future<void>> tasks;
      tasks.reserve(num_partitions_);

      for (size_t part = 0; part < num_partitions_; part++) {
        tasks.emplace_back(ThreadPool::get().submit([&, part]() {
          size_t hit_count = 0;

          // Precalc constants.
          const std::string& hkey_v = HCTR_REDIS_VALUE_HKEY();

          std::vector<size_t> i_vals;
          std::vector<std::string_view> k_views;
          std::vector<std::optional<std::string>> v_opts;
          std::shared_ptr<std::vector<TKey>> touched_keys;

          try {
            size_t num_batches = 0;
            for (const size_t* i = indices; i != indices_end; num_batches++) {
              // Check time budget.
              const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
              if (elapsed >= time_budget) {
                HCTR_LOG_S(WARNING, WORLD)
                    << get_name() << " backend; Partition " << hkey_v << ": Timeout!" << std::endl;

                size_t ign_count = 0;
                for (; i != indices_end; i++) {
                  const TKey& k = keys[*i];
                  if (HCTR_KEY_TO_PART_INDEX(k) == part) {
                    on_miss(*i);
                    ign_count++;
                  }
                }
                joint_ign_count += ign_count;
                break;
              }

              // Prepare query.
              k_views.clear();
              i_vals.clear();
              for (; i != indices_end; i++) {
                const TKey& k = keys[*i];
                if (HCTR_KEY_TO_PART_INDEX(k) == part) {
                  i_vals.emplace_back(*i);
                  k_views.emplace_back(reinterpret_cast<const char*>(&k), sizeof(TKey));
                  if (k_views.size() >= this->max_get_batch_size_) {
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
              auto i_vals_it = i_vals.begin();
              for (const auto& v_opt : v_opts) {
                if (v_opt) {
                  on_hit(*i_vals_it, v_opt->data(), v_opt->size());
                  hit_count++;

                  if (this->refresh_time_after_fetch_) {
                    if (!touched_keys) {
                      touched_keys = std::make_shared<std::vector<TKey>>();
                    }
                    touched_keys->emplace_back(keys[*i_vals_it]);
                  }
                } else {
                  on_miss(*i_vals_it);
                }
                i_vals_it++;
              }
              HCTR_CHECK(i_vals_it == i_vals.end());

              // Refresh timestamps if desired.
              if (touched_keys && !touched_keys->empty()) {
                const time_t now = std::time(nullptr);

                this->background_worker_.submit([this, table_name, part, touched_keys, now]() {
                  const std::string& hkey_t = HCTR_REDIS_TIME_HKEY();
                  touch_(hkey_t, touched_keys, now);
                });
                touched_keys.reset();
              }

              HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend. Partition " << hkey_v
                                       << ", batch " << num_batches << ": Fetched " << v_opts.size()
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
      ign_count += static_cast<size_t>(joint_ign_count);

      // Handle errors.
      try {
        if (error) {
          std::rethrow_exception(error);
        }
      } catch (sw::redis::Error& e) {
        throw DatabaseBackendError(get_name(), error_part, e.what());
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": " << hit_count
                           << " / " << (num_indices - ign_count) << " hits, " << ign_count
                           << " ignored." << std::endl;
  return hit_count;
}

template <typename TKey>
size_t RedisClusterBackend<TKey>::evict(const std::string& table_name) {
  std::atomic<size_t> joint_approx_num_keys{0};

  std::exception_ptr error;
  size_t error_part = -1;
  std::mutex error_guard;

  // Process partitions.
  std::vector<std::future<void>> tasks;
  tasks.reserve(num_partitions_);

  for (size_t part = 0; part < num_partitions_; part++) {
    tasks.emplace_back(ThreadPool::get().submit([&, part]() {
      size_t approx_num_keys = 0;

      // Precalc constants.
      const std::string& hkey_v = HCTR_REDIS_VALUE_HKEY();
      const std::string& hkey_t = HCTR_REDIS_TIME_HKEY();

      try {
        approx_num_keys += redis_->hlen(hkey_v);

        // Delete the keys.
        redis_->del(hkey_v);
        redis_->del(hkey_t);

        HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend. Partition " << hkey_v
                                 << ": Erased approximately " << approx_num_keys << " pairs."
                                 << std::endl;
      } catch (...) {
        std::unique_lock lock(error_guard);
        error = std::current_exception();
        error_part = part;
      }

      joint_approx_num_keys += approx_num_keys;
    }));
  }
  ThreadPool::await(tasks.begin(), tasks.end());
  const size_t approx_num_keys = static_cast<size_t>(joint_approx_num_keys);

  // Handle errors.
  try {
    if (error) {
      std::rethrow_exception(error);
    }
  } catch (sw::redis::Error& e) {
    throw DatabaseBackendError(get_name(), error_part, e.what());
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name
                           << " erased (approximately " << approx_num_keys << " pairs)."
                           << std::endl;
  return approx_num_keys;
}

template <typename TKey>
size_t RedisClusterBackend<TKey>::evict(const std::string& table_name, const size_t num_keys,
                                        const TKey* const keys) {
  size_t hit_count = 0;

  switch (num_keys) {
    case 0: {
      // Do nothing ;-).
    } break;
    case 1: {
      // Precalc constants.
      const size_t part = HCTR_KEY_TO_PART_INDEX(*keys);
      const std::string& hkey_v = HCTR_REDIS_VALUE_HKEY();
      const std::string& hkey_t = HCTR_REDIS_TIME_HKEY();

      try {
        // Erase.
        const std::string_view k_view{reinterpret_cast<const char*>(keys), sizeof(TKey)};
        hit_count += std::max(redis_->hdel(hkey_v, k_view), redis_->hdel(hkey_t, k_view));
      } catch (sw::redis::Error& e) {
        throw DatabaseBackendError(get_name(), part, e.what());
      }
    } break;
    default: {
      std::atomic<size_t> joint_hit_count{0};

      std::exception_ptr error;
      size_t error_part = -1;
      std::mutex error_guard;

      // Precalc constants.
      const TKey* const keys_end = &keys[num_keys];

      // Process partitions.
      std::vector<std::future<void>> tasks;
      tasks.reserve(num_partitions_);

      for (size_t part = 0; part < num_partitions_; part++) {
        tasks.emplace_back(ThreadPool::get().submit([&, part]() {
          size_t hit_count = 0;

          // Precalc constants.
          const std::string& hkey_v = HCTR_REDIS_VALUE_HKEY();
          const std::string& hkey_t = HCTR_REDIS_TIME_HKEY();

          try {
            std::vector<std::string_view> k_views;

            size_t num_batches = 0;
            for (const TKey* k = keys; k != keys_end; num_batches++) {
              // Prepare and launch query.
              k_views.clear();
              for (; k != keys_end; k++) {
                if (HCTR_KEY_TO_PART_INDEX(*k) == part) {
                  k_views.emplace_back(reinterpret_cast<const char*>(k), sizeof(TKey));
                  if (k_views.size() >= this->max_set_batch_size_) {
                    break;
                  }
                }
              }
              if (k_views.empty()) {
                continue;
              }

              const size_t batch_hits =
                  std::max(redis_->hdel(hkey_v, k_views.begin(), k_views.end()),
                           redis_->hdel(hkey_t, k_views.begin(), k_views.end()));

              HCTR_LOG_S(TRACE, WORLD)
                  << get_name() << " backend; Partition " << hkey_v << ", batch " << num_batches
                  << ": Erased " << batch_hits << " / " << k_views.size() << " pairs." << std::endl;
              hit_count += batch_hits;
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

      // Handle errors.
      try {
        if (error) {
          std::rethrow_exception(error);
        }
      } catch (sw::redis::Error& e) {
        throw DatabaseBackendError(get_name(), error_part, e.what());
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": Erased "
                           << hit_count << " / " << num_keys << " pairs." << std::endl;
  return hit_count;
}

template <typename TKey>
std::vector<std::string> RedisClusterBackend<TKey>::find_tables(const std::string& model_name) {
  // Determine partition 0 key pattern.
  std::string table_name_pattern = HierParameterServerBase::make_tag_name(model_name, "", false);
  boost::replace_all(table_name_pattern, ".", "\\.");
  table_name_pattern += "[a-zA-Z0-9_\\-]";
  const std::string& hkey_pattern = make_hkey(table_name_pattern, 0, 'v');

  // Find suitable keys.
  std::vector<std::string> part_names;
  auto reply = redis_->command(sw::redis::cmd::keys, hkey_pattern);
  sw::redis::reply::to_array(*reply, std::back_inserter(part_names));

  // Turn partition numes into table names.
  std::vector<std::string> table_names;
  std::transform(part_names.begin(), part_names.end(), std::back_inserter(table_names),
                 [](const std::string& part_name) {
                   // Pattern: table_name << '/' << 'p' << 0 << '/' << 'v';
                   //                         1      2     3     4      5
                   return part_name.substr(0, part_name.size() - 5);
                 });

  return table_names;
}

template <typename TKey>
void RedisClusterBackend<TKey>::check_and_resolve_overflow_(const std::string& hkey_v,
                                                            const std::string& hkey_t) {
  // Check overflow condition.
  size_t part_size = redis_->hlen(hkey_t);
  if (part_size <= this->overflow_margin_) {
    return;
  }
  HCTR_LOG_S(TRACE, WORLD) << get_name() << " partition " << hkey_v
                           << " is overflowing (size = " << part_size << " > "
                           << this->overflow_margin_ << "). Attempting to resolve..." << std::endl;

  // Select overflow resolution policy.
  switch (this->overflow_policy_) {
    case DatabaseOverflowPolicy_t::EvictOldest: {
      // Fetch keys and insert times.
      std::vector<std::pair<std::string, std::string>> kt_views;
      kt_views.reserve(part_size);
      redis_->hgetall(hkey_t, std::back_inserter(kt_views));

      // Sanity check.
      part_size = kt_views.size();
      if (part_size <= this->overflow_resolution_target_) {
        return;
      }

      // Sort by ascending by time.
      for (const auto& kt : kt_views) {
        HCTR_CHECK_HINT(kt.second.size() == sizeof(time_t), "Value size mismatch!(%d <> %d)!",
                        kt.second.size(), sizeof(time_t));
      }
      std::sort(kt_views.begin(), kt_views.end(), [](const auto& kt0, const auto& kt1) {
        const time_t t0 = *reinterpret_cast<const time_t*>(kt0.second.data());
        const time_t t1 = *reinterpret_cast<const time_t*>(kt1.second.data());
        return t0 < t1;
      });

      std::vector<std::string_view> k_views;
      k_views.reserve(this->max_set_batch_size_);

      // Delete items.
      const auto kt_end = kt_views.end();
      for (auto kt_it = kt_views.begin(); kt_it != kt_end;) {
        // Collect a batch.
        k_views.clear();
        for (; kt_it != kt_end; kt_it++) {
          k_views.emplace_back(kt_it->first);
          if (k_views.size() >= this->max_set_batch_size_) {
            break;
          }
        }

        // Perform deletion.
        HCTR_LOG_S(TRACE, WORLD) << get_name() << " partition " << hkey_v
                                 << " (size = " << part_size << "). Attempting to evict "
                                 << k_views.size() << " OLD key/value pairs." << std::endl;
        redis_->hdel(hkey_v, k_views.begin(), k_views.end());
        redis_->hdel(hkey_t, k_views.begin(), k_views.end());

        // Overflow resolved?
        part_size = redis_->hlen(hkey_t);
        if (part_size <= this->overflow_resolution_target_) {
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
      if (part_size <= this->overflow_resolution_target_) {
        return;
      }

      // Shuffle the keys.
      for (const auto& k_view : k_views) {
        HCTR_CHECK_HINT(k_view.size() == sizeof(time_t), "Value size mismatch!(%d <> %d)!",
                        k_view.size(), sizeof(time_t));
      }
      std::random_device rd;
      std::default_random_engine gen(rd());
      std::shuffle(k_views.begin(), k_views.end(), gen);

      // Delete keys.
      const auto k_end = k_views.end();
      for (auto k_it = k_views.begin(); k_it != k_end;) {
        const auto batch_end = std::min(k_it + this->max_set_batch_size_, k_end);

        // Perform deletion.
        HCTR_LOG_S(TRACE, WORLD) << get_name() << " partition " << hkey_v
                                 << " (size = " << part_size << "). Attempting to evict "
                                 << k_views.size() << " RANDOM key/value pairs." << std::endl;
        redis_->hdel(hkey_v, k_it, batch_end);
        redis_->hdel(hkey_t, k_it, batch_end);

        // Overflow resolved?
        part_size = redis_->hlen(hkey_t);
        if (part_size <= this->overflow_resolution_target_) {
          break;
        }
        k_it = batch_end;
      }
    } break;
    default: {
      HCTR_LOG_S(WARNING, WORLD) << "Redis partition " << hkey_v << " (size = " << part_size
                                 << "), surpasses specified maximum size (="
                                 << this->overflow_margin_
                                 << "), but no compatible overflow policy (="
                                 << this->overflow_policy_ << ") was selected!" << std::endl;
      return;
    } break;
  }

  HCTR_LOG_S(DEBUG, WORLD) << get_name() << " partition " << hkey_v
                           << " overflow resolution concluded!" << std::endl;
}

template <typename TKey>
void RedisClusterBackend<TKey>::touch_(const std::string& hkey_t, const TKey& key,
                                       const time_t time) {
  HCTR_LOG_S(TRACE, WORLD) << get_name() << ": Touching key " << key << " of " << hkey_t << '.'
                           << std::endl;

  // Launch query.
  try {
    redis_->hset(hkey_t, {reinterpret_cast<const char*>(&key), sizeof(TKey)},
                 {reinterpret_cast<const char*>(&time), sizeof(time_t)});
  } catch (sw::redis::Error& e) {
    HCTR_LOG_S(ERROR, WORLD) << get_name() << " partition " << hkey_t
                             << "; error during refresh: " << e.what() << '.' << std::endl;
  }
}

template <typename TKey>
void RedisClusterBackend<TKey>::touch_(const std::string& hkey_t,
                                       const std::shared_ptr<std::vector<TKey>>& keys,
                                       const time_t time) {
  HCTR_LOG_S(TRACE, WORLD) << get_name() << ": Touching " << keys->size() << " keys of " << hkey_t
                           << '.' << std::endl;

  // Prepare query.
  std::vector<std::pair<std::string_view, std::string_view>> kt_views;
  kt_views.reserve(keys->size());
  for (const TKey& k : *keys) {
    kt_views.emplace_back(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<const char*>(&k), sizeof(TKey)),
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
