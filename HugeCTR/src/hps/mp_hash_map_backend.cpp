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
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <boost/utility/string_view.hpp>
#include <hps/database_backend_detail.hpp>
#include <hps/hier_parameter_server_base.hpp>
#include <hps/mp_hash_map_backend.hpp>
#include <random>

// TODO: Remove me!
#pragma GCC diagnostic error "-Wconversion"

namespace HugeCTR {

#ifdef HCTR_HASH_MAP_BACKEND_CONTAINS_
#error HCTR_HASH_MAP_BACKEND_CONTAINS_ already defined. Potential naming conflict!
#else
#define HCTR_HASH_MAP_BACKEND_CONTAINS_(KEY) \
  hit_count += part.entries.find((KEY)) != part.entries.end()
#endif

#ifdef HCTR_HASH_MAP_BACKEND_INSERT_
#error HCTR_HASH_MAP_BACKEND_INSERT_ already defined. Potential naming conflict!
#else
#define HCTR_HASH_MAP_BACKEND_INSERT_(KEY, VALUES_PTR)                                   \
  do {                                                                                   \
    const auto& res = part.entries.try_emplace((KEY));                                   \
    Entry& entry = *res.first;                                                           \
                                                                                         \
    /* If new insertion. */                                                              \
    if (res.second) {                                                                    \
      /* If no free space, allocate another buffer, and fill pointer queue. */           \
      if (part.payload_slots.empty()) {                                                  \
        const size_t payload_size = meta_size + value_size;                              \
        const size_t num_payloads = part.allocation_rate / payload_size;                 \
        HCTR_CHECK(num_payloads > 0);                                                    \
                                                                                         \
        /* Get more memory. */                                                           \
        part.payload_pages.emplace_back(num_payloads* payload_size, sm_char_allocator_); \
        Page& page = part.payload_pages.back();                                          \
                                                                                         \
        /* Stock up slot references. */                                                  \
        part.payload_slots.reserve(num_payloads);                                        \
        for (auto nxt = page.end(); nxt != page.begin();) {                              \
          nxt -= payload_size;                                                           \
          part.payload_slots.emplace_back(reinterpret_cast<Payload*>(&*nxt));            \
        }                                                                                \
      }                                                                                  \
                                                                                         \
      /* Fetch pointer. */                                                               \
      entry.second = part.payload_slots.back();                                          \
      part.payload_slots.pop_back();                                                     \
    }                                                                                    \
                                                                                         \
    entry.second->last_access = now;                                                     \
    std::copy_n((VALUES_PTR), value_size, entry.second->value);                          \
    num_inserts++;                                                                       \
  } while (0)
#endif

#ifdef HCTR_HASH_MAP_BACKEND_FETCH_
#error HCTR_HASH_MAP_BACKEND_FETCH_ already defined. Potential naming conflict!
#else
#define HCTR_HASH_MAP_BACKEND_FETCH_(KEY, INDEX)                                             \
  do {                                                                                       \
    const auto& it = part.entries.find((KEY));                                               \
    if (it != part.entries.end()) {                                                          \
      const PayloadPtr& payload = it->second;                                                \
                                                                                             \
      /* Race-conditions here are deliberately ignored because insignificant in practice. */ \
      payload->last_access = now;                                                            \
      on_hit((INDEX), payload->value, part.value_size);                                      \
      hit_count++;                                                                           \
    } else {                                                                                 \
      on_miss((INDEX));                                                                      \
    }                                                                                        \
  } while (0)
#endif

#ifdef HCTR_HASH_MAP_BACKEND_EVICT_
#error HCTR_HASH_MAP_BACKEND_EVICT_ already defined. Potential naming conflict!
#else
#define HCTR_HASH_MAP_BACKEND_EVICT_(KEY)       \
  do {                                          \
    const auto& it = part.entries.find((KEY));  \
    if (it != part.entries.end()) {             \
      const PayloadPtr& payload = it->second;   \
                                                \
      /* Stash pointer and erase item. */       \
      part.payload_slots.emplace_back(payload); \
      part.entries.erase(it);                   \
      hit_count++;                              \
    }                                           \
  } while (0)
#endif

template <typename Key>
MultiProcessHashMapBackend<Key>::MultiProcessHashMapBackend(
    const size_t num_partitions, const size_t allocation_rate, const size_t sm_size,
    const std::string& sm_name, const std::chrono::nanoseconds& heart_beat_frequency,
    const bool auto_remove, const size_t max_get_batch_size, const size_t max_set_batch_size,
    const size_t overflow_margin, const DatabaseOverflowPolicy_t overflow_policy,
    const double overflow_resolution_target)
    : Base(max_get_batch_size, max_set_batch_size, overflow_margin, overflow_policy,
           overflow_resolution_target),
      num_partitions_{num_partitions},
      allocation_rate_{allocation_rate},
      sm_name_{sm_name},
      sm_segment_(boost::interprocess::open_or_create, sm_name.c_str(), sm_size),
      sm_char_allocator_{sm_segment_.get_allocator<char>()},
      sm_page_allocator_{sm_segment_.get_allocator<Page>()},
      sm_partition_allocator_{sm_segment_.get_allocator<Partition>()} {
  sm_ = sm_segment_.find_or_construct<SharedMemory>("sm")(
      overflow_margin, overflow_policy, overflow_resolution_target, heart_beat_frequency,
      auto_remove, sm_segment_);
  HCTR_CHECK(sm_);
  HCTR_CHECK(sm_->overflow_margin == overflow_margin);
  HCTR_CHECK(sm_->overflow_policy == overflow_policy);
  HCTR_CHECK(sm_->overflow_resolution_target == overflow_resolution_target);
  HCTR_CHECK(sm_->heart_beat_frequency == heart_beat_frequency);
  HCTR_CHECK(sm_->auto_remove == auto_remove);

  HCTR_LOG_S(INFO, WORLD) << "Connecting to shared memory '" << sm_name_ << "'..." << std::endl;

  // Ensure exclusive access.
  const boost::interprocess::scoped_lock lock(sm_->read_write_guard);

  // Sanity checks.
  const std::filesystem::space_info& si = std::filesystem::space("/dev/shm");
  HCTR_LOG_S(INFO, WORLD) << "Connected to shared memory '" << sm_name_
                          << "'; OS total = " << si.capacity << " bytes, OS available = " << si.free
                          << " bytes, HCTR allocated = " << sm_segment_.get_size()
                          << " bytes, HCTR free = " << sm_segment_.get_free_memory() << " bytes"
                          << "; other processes connected = " << is_process_connected_()
                          << std::endl;

  if (si.capacity < sm_segment_.get_size()) {
    HCTR_LOG_S(WARNING, WORLD) << "Shared memory (" << sm_segment_.get_size()
                               << " bytes) is larger than total shared memory capacity ("
                               << si.capacity
                               << " bytes). This might lead to esotheric runtime errors. Consider "
                                  "increasing OS shared memory size."
                               << std::endl;
  }

  if (si.free < allocation_rate_) {
    HCTR_LOG_S(WARNING, WORLD)
        << "Shared memory is (almost) full. Any further SHM allocation will definitely fail!"
        << std::endl;
  }

  // Start heart.
  heart_ = std::thread([&] {
    while (!heart_stop_signal_) {
      ++sm_->heart_beat;
      std::this_thread::sleep_for(sm_->heart_beat_frequency);
    }
  });
}

template <typename Key>
bool MultiProcessHashMapBackend<Key>::is_process_connected_() const {
  const uint64_t old_heart_beat = sm_->heart_beat;
  std::this_thread::sleep_for(sm_->heart_beat_frequency * 5);
  return sm_->heart_beat != old_heart_beat;
}

template <typename Key>
MultiProcessHashMapBackend<Key>::~MultiProcessHashMapBackend() {
  HCTR_LOG_S(INFO, WORLD) << "Disconnecting from shared memory '" << sm_name_ << "'." << std::endl;

  // Ensure exclusive access.
  const boost::interprocess::scoped_lock lock(sm_->read_write_guard);

  // Stop heart.
  heart_stop_signal_ = true;
  heart_.join();

  // Destroy SHM, if this was the last process and auto_remove is enabled.
  if (sm_->auto_remove && !is_process_connected_()) {
    HCTR_LOG_S(INFO, WORLD) << "Detached last process from shared memory '" << sm_name_
                            << "'. Auto remove in progress..." << std::endl;
    boost::interprocess::shared_memory_object::remove(sm_name_.c_str());
  }
}

template <typename Key>
size_t MultiProcessHashMapBackend<Key>::capacity(const std::string& table_name) const {
  const size_t part_cap = this->overflow_margin_;
  const size_t total_cap = part_cap * num_partitions_;
  return (total_cap > part_cap) ? total_cap : part_cap;
}

template <typename Key>
size_t MultiProcessHashMapBackend<Key>::size(const std::string& table_name) const {
  const boost::interprocess::sharable_lock lock(sm_->read_write_guard);

  // Locate the partitions.
  const auto& tables_it = sm_->tables.find({table_name.c_str(), sm_char_allocator_});
  if (tables_it == sm_->tables.end()) {
    return 0;
  }

  size_t num_keys = 0;
  for (const Partition& part : tables_it->second) {
    num_keys += part.entries.size();
  }
  return num_keys;
}

template <typename Key>
size_t MultiProcessHashMapBackend<Key>::contains(
    const std::string& table_name, size_t num_keys, const Key* keys,
    const std::chrono::nanoseconds& time_budget) const {
  const auto begin = std::chrono::high_resolution_clock::now();
  const boost::interprocess::sharable_lock lock(sm_->read_write_guard);

  // Locate partitions.
  const auto& tables_it = sm_->tables.find({table_name.c_str(), sm_char_allocator_});
  if (tables_it == sm_->tables.end()) {
    return Base::contains(table_name, num_keys, keys, time_budget);
  }
  const SharedVector<Partition>& parts = tables_it->second;

  size_t hit_count = 0;
  size_t ign_count = 0;

  switch (num_keys) {
    case 0: {
      // Nothing to do ;-).
    } break;
    case 1: {
      // Check time budget.
      const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
      if (elapsed >= time_budget) {
        HCTR_LOG_S(WARNING, WORLD)
            << get_name() << " backend; Table " << table_name << ": Timeout!" << std::endl;

        ign_count++;
        break;
      }

      // Check partition.
      const Partition& part = parts[HCTR_KEY_TO_DB_PART_INDEX(*keys)];
      HCTR_HASH_MAP_BACKEND_CONTAINS_(*keys);
    } break;
    default: {
      // Precalc constants.
      const Key* keys_end = &keys[num_keys];

      if (parts.size() == 1) {
        const Partition& part = parts.front();

        // Traverse through keys.
        size_t num_batches = 0;
        for (const Key* k = keys; k != keys_end; num_batches++) {
          // Check time budget.
          const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
          if (elapsed >= time_budget) {
            HCTR_LOG_S(WARNING, WORLD)
                << get_name() << " backend; Table " << table_name << ": Timeout!" << std::endl;

            ign_count += keys_end - k;
            break;
          }

          // Query next batch.
          const Key* const batch_end = std::min(&k[this->max_get_batch_size_], keys_end);
          for (; k != batch_end; k++) {
            HCTR_HASH_MAP_BACKEND_CONTAINS_(*k);
          }

          HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name
                                   << ", partition " << part.index << ", batch " << num_batches
                                   << ": " << hit_count << " hits. Time: " << elapsed.count()
                                   << " / " << time_budget.count() << " ns." << std::endl;
        }
      } else {
        std::atomic<size_t> joint_hit_count{0};
        std::atomic<size_t> joint_ign_count{0};

        // Process partitions.
        std::vector<std::future<void>> tasks;
        tasks.reserve(parts.size());

        for (const Partition& part : parts) {
          tasks.emplace_back(ThreadPool::get().submit([&]() {
            size_t hit_count = 0;

            size_t num_batches = 0;
            for (const Key* k = keys; k != keys_end; num_batches++) {
              // Check time budget.
              const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
              if (elapsed >= time_budget) {
                HCTR_LOG_S(WARNING, WORLD)
                    << get_name() << " backend; Table " << table_name << ": Timeout!" << std::endl;

                size_t ign_count = 0;
                for (; k != keys_end; k++) {
                  if (HCTR_KEY_TO_DB_PART_INDEX(*k) == part.index) {
                    ign_count++;
                  }
                }
                joint_ign_count += ign_count;
                break;
              }

              // Query next batch.
              size_t batch_size = 0;
              for (; k != keys_end; k++) {
                if (HCTR_KEY_TO_DB_PART_INDEX(*k) == part.index) {
                  HCTR_HASH_MAP_BACKEND_CONTAINS_(*k);
                  if (++batch_size >= this->max_get_batch_size_) {
                    ++k;
                    break;
                  }
                }
              }

              HCTR_LOG_S(TRACE, WORLD)
                  << get_name() << " backend; Table " << table_name << ", partition " << part.index
                  << ", batch " << num_batches << ": " << hit_count << " / " << batch_size
                  << " hits. Time: " << elapsed.count() << " / " << time_budget.count() << " ns."
                  << std::endl;
            }

            joint_hit_count += hit_count;
          }));
        }
        ThreadPool::await(tasks.begin(), tasks.end());
        hit_count += static_cast<size_t>(joint_hit_count);
        ign_count += static_cast<size_t>(joint_ign_count);
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": " << hit_count
                           << " / " << (num_keys - ign_count) << " hits, " << ign_count
                           << " ignored." << std::endl;
  return hit_count;
}

template <typename Key>
bool MultiProcessHashMapBackend<Key>::insert(const std::string& table_name, size_t num_pairs,
                                             const Key* keys, const char* values,
                                             size_t value_size) {
  const boost::interprocess::scoped_lock lock(sm_->read_write_guard);

  // Locate the partitions, or create them, if they do not exist yet.
  const auto& tables_it =
      sm_->tables.try_emplace({table_name.c_str(), sm_char_allocator_}, sm_partition_allocator_)
          .first;
  SharedVector<Partition>& parts = tables_it->second;
  if (parts.empty()) {
    HCTR_CHECK(value_size > 0 && value_size <= allocation_rate_);

    parts.reserve(num_partitions_);
    for (size_t i = 0; i < num_partitions_; i++) {
      parts.emplace_back(i, value_size, allocation_rate_, sm_segment_);
    }
  } else {
    HCTR_CHECK(parts.size() == num_partitions_);
  }

  size_t num_inserts = 0;

  switch (num_pairs) {
    case 0: {
      // Do nothing ;-).
    } break;
    case 1: {
      Partition& part = parts[HCTR_KEY_TO_DB_PART_INDEX(*keys)];
      HCTR_CHECK(part.value_size == value_size);

      // Check overflow condition.
      if (part.entries.size() >= this->overflow_margin_) {
        resolve_overflow_(table_name, part);
      }

      // Perform insertion.
      const time_t now = std::time(nullptr);
      HCTR_HASH_MAP_BACKEND_INSERT_(*keys, values);
    } break;
    default: {
      // Precalc constants.
      const Key* const keys_end = &keys[num_pairs];

      if (parts.size() == 1) {
        Partition& part = parts.front();
        HCTR_CHECK(part.value_size == value_size);

        // Step through batch-by-batch.
        for (const Key* k = keys; k != keys_end;) {
          // Check overflow condition.
          if (part.entries.size() >= this->overflow_margin_) {
            resolve_overflow_(table_name, part);
          }

          // Perform insertion.
          const time_t now = std::time(nullptr);

          const Key* const batch_end = std::min(&k[this->max_get_batch_size_], keys_end);
          for (; k != batch_end; k++) {
            HCTR_HASH_MAP_BACKEND_INSERT_(*k, &values[(k - keys) * value_size]);
          }
        }
      } else {
        std::atomic<size_t> joint_num_inserts{0};
        std::atomic<size_t> joint_ign_count{0};

        // Process partitions.
        std::vector<std::future<void>> tasks;
        tasks.reserve(num_partitions_);

        for (Partition& part : parts) {
          tasks.emplace_back(ThreadPool::get().submit([&]() {
            HCTR_CHECK(part.value_size == value_size);

            size_t num_inserts = 0;

            // Step through batch-by-batch.
            size_t num_batches = 0;
            for (const Key* k = keys; k != keys_end; num_batches++) {
              // Check overflow condition.
              if (part.entries.size() >= this->overflow_margin_) {
                resolve_overflow_(table_name, part);
              }

              // Perform insertion.
              const time_t now = std::time(nullptr);

              size_t batch_size = 0;
              for (; k != keys_end; k++) {
                if (HCTR_KEY_TO_DB_PART_INDEX(*k) == part.index) {
                  HCTR_HASH_MAP_BACKEND_INSERT_(*k, &values[(k - keys) * value_size]);
                  if (++batch_size >= this->max_set_batch_size_) {
                    ++k;
                    break;
                  }
                }
              }

              HCTR_LOG_S(TRACE, WORLD)
                  << get_name() << " backend; Table " << table_name << ", batch " << num_batches
                  << ": Inserted " << batch_size << " entries." << std::endl;
            }

            joint_num_inserts += num_inserts;
          }));
        }
        ThreadPool::await(tasks.begin(), tasks.end());
        num_inserts += static_cast<size_t>(joint_num_inserts);
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": Inserted "
                           << num_inserts << " / " << num_pairs
                           << " entries; free SM = " << sm_segment_.get_free_memory() << " bytes"
                           << std::endl;
  return true;
}

template <typename Key>
size_t MultiProcessHashMapBackend<Key>::fetch(const std::string& table_name, const size_t num_keys,
                                              const Key* const keys,
                                              const DatabaseHitCallback& on_hit,
                                              const DatabaseMissCallback& on_miss,
                                              const std::chrono::nanoseconds& time_budget) {
  const auto begin = std::chrono::high_resolution_clock::now();
  const boost::interprocess::sharable_lock lock(sm_->read_write_guard);

  // Locate the partitions.
  const auto& tables_it = sm_->tables.find({table_name.c_str(), sm_char_allocator_});
  if (tables_it == sm_->tables.end()) {
    return Base::fetch(table_name, num_keys, keys, on_hit, on_miss, time_budget);
  }
  const SharedVector<Partition>& parts = tables_it->second;

  size_t hit_count = 0;
  size_t ign_count = 0;

  switch (num_keys) {
    case 0: {
      // Nothing to do ;-).
    } break;
    case 1: {
      // Check time budget.
      const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
      if (elapsed >= time_budget) {
        HCTR_LOG_S(WARNING, WORLD)
            << get_name() << " backend; Table " << table_name << ": Timeout!" << std::endl;

        on_miss(0);
        ign_count++;
        break;
      }

      // Perform query.
      const Partition& part = parts[HCTR_KEY_TO_DB_PART_INDEX(*keys)];
      const time_t now = std::time(nullptr);

      HCTR_HASH_MAP_BACKEND_FETCH_(*keys, 0);
    } break;
    default: {
      // Precalc constants.
      const Key* const keys_end = &keys[num_keys];

      if (parts.size() == 1) {
        const Partition& part = parts.front();

        // Step through batch-by-batch.
        size_t num_batches = 0;
        for (const Key* k = keys; k != keys_end; num_batches++) {
          // Check time budget.
          const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
          if (elapsed >= time_budget) {
            HCTR_LOG_S(WARNING, WORLD)
                << get_name() << " backend; Table " << table_name << ": Timeout!" << std::endl;

            for (; k != keys_end; k++) {
              on_miss(k - keys);
              ign_count++;
            }
            break;
          }

          // Perform a bunch of queries.
          const size_t prev_hit_count = hit_count;
          const time_t now = std::time(nullptr);

          const Key* const batch_end = std::min(&k[this->max_get_batch_size_], keys_end);
          for (; k != batch_end; k++) {
            HCTR_HASH_MAP_BACKEND_FETCH_(*k, k - keys);
          }

          HCTR_LOG_S(TRACE, WORLD)
              << get_name() << " backend; Table " << table_name << ", partition " << part.index
              << ", batch " << num_batches << ": " << hit_count - prev_hit_count
              << " hits. Time: " << elapsed.count() << " / " << time_budget.count() << " ns."
              << std::endl;
        }
      } else {
        std::atomic<size_t> joint_hit_count{0};
        std::atomic<size_t> joint_ign_count{0};

        // Spawn threads.
        std::vector<std::future<void>> tasks;
        tasks.reserve(parts.size());

        for (const Partition& part : parts) {
          tasks.emplace_back(ThreadPool::get().submit([&]() {
            size_t hit_count = 0;

            // Traverse through keys, and fetch them one by one.
            size_t num_batches = 0;
            for (const Key* k = keys; k != keys_end; num_batches++) {
              // Check time budget.
              const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
              if (elapsed >= time_budget) {
                HCTR_LOG_S(WARNING, WORLD)
                    << get_name() << " backend; Table " << table_name << ": Timeout!" << std::endl;

                size_t ign_count = 0;
                for (; k != keys_end; k++) {
                  if (HCTR_KEY_TO_DB_PART_INDEX(*k) == part.index) {
                    on_miss(k - keys);
                    ign_count++;
                  }
                }
                joint_ign_count += ign_count;
                break;
              }

              // Perform a bunch of queries.
              const size_t prev_hit_count = hit_count;
              const time_t now = std::time(nullptr);

              size_t batch_size = 0;
              for (; k != keys_end; k++) {
                if (HCTR_KEY_TO_DB_PART_INDEX(*k) == part.index) {
                  HCTR_HASH_MAP_BACKEND_FETCH_(*k, k - keys);
                  if (++batch_size >= this->max_get_batch_size_) {
                    ++k;
                    break;
                  }
                }
              }

              HCTR_LOG_S(TRACE, WORLD)
                  << get_name() << " backend; Table " << table_name << ", partition " << part.index
                  << ", batch " << num_batches << ": " << (hit_count - prev_hit_count) << " / "
                  << batch_size << " hits. Time: " << elapsed.count() << " / "
                  << time_budget.count() << " ns." << std::endl;
            }

            joint_hit_count += hit_count;
          }));
        }
        ThreadPool::await(tasks.begin(), tasks.end());
        hit_count += static_cast<size_t>(joint_hit_count);
        ign_count += static_cast<size_t>(joint_ign_count);
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": " << hit_count
                           << " / " << (num_keys - ign_count) << " hits. Ignored: " << ign_count
                           << '.' << std::endl;
  return hit_count;
}

template <typename Key>
size_t MultiProcessHashMapBackend<Key>::fetch(const std::string& table_name,
                                              const size_t num_indices, const size_t* const indices,
                                              const Key* const keys,
                                              const DatabaseHitCallback& on_hit,
                                              const DatabaseMissCallback& on_miss,
                                              const std::chrono::nanoseconds& time_budget) {
  const auto begin = std::chrono::high_resolution_clock::now();
  const boost::interprocess::sharable_lock lock(sm_->read_write_guard);

  // Locate the partitions.
  const auto& tables_it = sm_->tables.find({table_name.c_str(), sm_char_allocator_});
  if (tables_it == sm_->tables.end()) {
    return Base::fetch(table_name, num_indices, indices, keys, on_hit, on_miss, time_budget);
  }
  const SharedVector<Partition>& parts = tables_it->second;

  size_t hit_count = 0;
  size_t ign_count = 0;

  switch (num_indices) {
    case 0: {
      // Nothing to do ;-).
    } break;
    case 1: {
      // Check time budget.
      const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
      if (elapsed >= time_budget) {
        HCTR_LOG_S(WARNING, WORLD)
            << get_name() << " backend; Table " << table_name << ": Timeout!" << std::endl;

        on_miss(*indices);
        ign_count++;
        break;
      }

      // Precalc constants.
      const Key& k = keys[*indices];
      const Partition& part = parts[HCTR_KEY_TO_DB_PART_INDEX(k)];

      // Perform query.
      const time_t now = std::time(nullptr);
      HCTR_HASH_MAP_BACKEND_FETCH_(k, *indices);
    } break;
    default: {
      // Precalc constants.
      const size_t* const indices_end = &indices[num_indices];

      if (parts.size() == 1) {
        const Partition& part = parts.front();

        // Step through batch-by-batch.
        size_t num_batches = 0;
        for (const size_t* i = indices; i != indices_end; num_batches++) {
          // Check time budget.
          const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
          if (elapsed >= time_budget) {
            HCTR_LOG_S(WARNING, WORLD)
                << get_name() << " backend; Table " << table_name << ": Timeout!" << std::endl;

            for (; i != indices_end; i++) {
              on_miss(*i);
              ign_count++;
            }
            break;
          }

          // Perform a bunch of queries.
          const size_t prev_hit_count = hit_count;
          const time_t now = std::time(nullptr);

          const size_t* const batch_end = std::min(&i[this->max_get_batch_size_], indices_end);
          for (; i != batch_end; i++) {
            HCTR_HASH_MAP_BACKEND_FETCH_(keys[*i], *i);
          }

          HCTR_LOG_S(TRACE, WORLD)
              << get_name() << " backend; Table " << table_name << ", partition " << part.index
              << ", batch " << num_batches << ": " << hit_count - prev_hit_count
              << " hits. Time: " << elapsed.count() << " / " << time_budget.count() << " ns."
              << std::endl;
        }
      } else {
        std::atomic<size_t> joint_hit_count{0};
        std::atomic<size_t> joint_ign_count{0};

        // Process partitions.
        std::vector<std::future<void>> tasks;
        tasks.reserve(parts.size());

        for (const Partition& part : parts) {
          tasks.emplace_back(ThreadPool::get().submit([&]() {
            size_t hit_count = 0;

            // Traverse through keys batch-wise.
            const size_t* i = indices;
            for (size_t num_batches = 0; i != indices_end; num_batches++) {
              // Check time budget.
              const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
              if (elapsed >= time_budget) {
                HCTR_LOG_S(WARNING, WORLD)
                    << get_name() << " backend; Table " << table_name << ": Timeout!" << std::endl;

                size_t ign_count = 0;
                for (; i != indices_end; i++) {
                  if (HCTR_KEY_TO_DB_PART_INDEX(keys[*i]) == part.index) {
                    on_miss(*i);
                    ign_count++;
                  }
                }
                joint_ign_count += ign_count;
                break;
              }

              // Step through batch 1 by 1 and fetch.
              const size_t prev_hit_count = hit_count;
              const time_t now = std::time(nullptr);

              size_t batch_size = 0;
              for (; i != indices_end; i++) {
                const Key& k = keys[*i];
                if (HCTR_KEY_TO_DB_PART_INDEX(k) == part.index) {
                  HCTR_HASH_MAP_BACKEND_FETCH_(k, *i);
                  if (++batch_size >= this->max_get_batch_size_) {
                    ++i;
                    break;
                  }
                }
              }

              HCTR_LOG_S(TRACE, WORLD)
                  << get_name() << " backend; Table " << table_name << ", partition " << part.index
                  << ", batch " << num_batches << ": " << (hit_count - prev_hit_count) << " / "
                  << batch_size << " hits. Time: " << elapsed.count() << " / "
                  << time_budget.count() << " ns." << std::endl;

              joint_hit_count += hit_count;
            }
          }));
        }
        ThreadPool::await(tasks.begin(), tasks.end());
        hit_count += static_cast<size_t>(joint_hit_count);
        ign_count += static_cast<size_t>(joint_ign_count);
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": " << hit_count
                           << " / " << (num_indices - ign_count) << " hits. Ignored: " << ign_count
                           << '.' << std::endl;
  return hit_count;
}

template <typename Key>
size_t MultiProcessHashMapBackend<Key>::evict(const std::string& table_name) {
  const boost::interprocess::scoped_lock lock(sm_->read_write_guard);

  // Locate the partitions.
  const auto& tables_it = sm_->tables.find({table_name.c_str(), sm_char_allocator_});
  if (tables_it == sm_->tables.end()) {
    return 0;
  }
  const SharedVector<Partition>& parts = tables_it->second;

  // Count items and erase.
  size_t hit_count = 0;
  for (const Partition& part : parts) {
    hit_count += part.entries.size();
  }
  sm_->tables.erase(tables_it);

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << " erased ("
                           << hit_count << " pairs)." << std::endl;
  return hit_count;
}

template <typename Key>
size_t MultiProcessHashMapBackend<Key>::evict(const std::string& table_name, const size_t num_keys,
                                              const Key* const keys) {
  const boost::interprocess::scoped_lock lock(sm_->read_write_guard);

  // Locate the partitions.
  const auto& tables_it = sm_->tables.find({table_name.c_str(), sm_char_allocator_});
  if (tables_it == sm_->tables.end()) {
    return 0;
  }
  SharedVector<Partition>& parts = tables_it->second;

  size_t hit_count = 0;

  switch (num_keys) {
    case 0: {
      // Nothing to do ;-).
    } break;
    case 1: {
      Partition& part = parts[HCTR_KEY_TO_DB_PART_INDEX(*keys)];
      HCTR_HASH_MAP_BACKEND_EVICT_(*keys);
    } break;
    default: {
      // Precalc constants.
      const Key* const keys_end = &keys[num_keys];

      if (parts.size() == 1) {
        Partition& part = parts.front();

        // Traverse through keys batch-wise.
        size_t num_batches = 0;
        for (const Key* k = keys; k != keys_end; num_batches++) {
          const size_t prev_hit_count = hit_count;

          // Step through batch 1 by 1 and delete.
          const Key* const batch_end = std::min(&k[this->max_set_batch_size_], keys_end);
          for (; k != batch_end; k++) {
            HCTR_HASH_MAP_BACKEND_EVICT_(*k);
          }

          HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Partition " << table_name << "/"
                                   << part.index << ", batch " << num_batches << ": Erased "
                                   << (hit_count - prev_hit_count) << " entries." << std::endl;
        }
      } else {
        std::atomic<size_t> joint_hit_count{0};

        // Process partitions.
        std::vector<std::future<void>> tasks;
        tasks.reserve(parts.size());

        for (Partition& part : parts) {
          tasks.emplace_back(ThreadPool::get().submit([&]() {
            size_t hit_count = 0;

            // Traverse through keys, batch-by-batch.
            size_t num_batches = 0;
            for (const Key* k = keys; k != keys_end; num_batches++) {
              const size_t prev_hit_count = hit_count;

              // Step through batch 1 by 1 and delete.
              size_t batch_size = 0;
              for (; k != keys_end; k++) {
                if (HCTR_KEY_TO_DB_PART_INDEX(*k) == part.index) {
                  HCTR_HASH_MAP_BACKEND_EVICT_(*k);
                  if (++batch_size >= this->max_set_batch_size_) {
                    ++k;
                    break;
                  }
                }
              }

              HCTR_LOG_S(TRACE, WORLD)
                  << get_name() << " backend; Partition " << table_name << "/" << part.index
                  << ", batch " << num_batches << ": Erased " << (hit_count - prev_hit_count)
                  << " / " << batch_size << " entries." << std::endl;
            }

            joint_hit_count += hit_count;
          }));
        }
        ThreadPool::await(tasks.begin(), tasks.end());
        hit_count += static_cast<size_t>(joint_hit_count);
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ". " << hit_count
                           << " / " << num_keys << " entries erased." << std::endl;
  return hit_count;
}

template <typename Key>
std::vector<std::string> MultiProcessHashMapBackend<Key>::find_tables(
    const std::string& model_name) {
  const std::string& tag_prefix = HierParameterServerBase::make_tag_name(model_name, "", false);

  const boost::interprocess::sharable_lock lock(sm_->read_write_guard);

  std::vector<std::string> matches;
  for (const auto& pair : sm_->tables) {
    if (pair.first.find(tag_prefix) == 0) {
      matches.push_back(pair.first);
    }
  }
  return matches;
}

template <typename Key>
void MultiProcessHashMapBackend<Key>::dump_bin(const std::string& table_name, std::ofstream& file) {
  const boost::interprocess::sharable_lock lock(sm_->read_write_guard);

  // Locate the partitions.
  const auto& tables_it = sm_->tables.find({table_name.c_str(), sm_char_allocator_});
  if (tables_it == sm_->tables.end()) {
    return;
  }
  const SharedVector<Partition>& parts = tables_it->second;

  // Store value size.
  const uint32_t value_size = parts.empty() ? 0 : parts[0].value_size;
  file.write(reinterpret_cast<const char*>(&value_size), sizeof(uint32_t));

  // Store values.
  for (const Partition& part : parts) {
    for (const Entry& entry : part.entries) {
      file.write(reinterpret_cast<const char*>(&entry.first), sizeof(Key));
      file.write(entry.second->value, value_size);
    }
  }
}

template <typename Key>
void MultiProcessHashMapBackend<Key>::dump_sst(const std::string& table_name,
                                               rocksdb::SstFileWriter& file) {
  const boost::interprocess::sharable_lock lock(sm_->read_write_guard);

  // Locate the partitions.
  const auto& tables_it = sm_->tables.find({table_name.c_str(), sm_char_allocator_});
  if (tables_it == sm_->tables.end()) {
    return;
  }
  const SharedVector<Partition>& parts = tables_it->second;

  // Sort keys by value.
  std::vector<const Entry*> entries;
  entries.reserve(
      std::accumulate(parts.begin(), parts.end(), 0,
                      [](const size_t a, const Partition& b) { return a + b.entries.size(); }));
  for (const Partition& part : parts) {
    for (const Entry& entry : part.entries) {
      entries.emplace_back(&entry);
    }
  }
  std::sort(entries.begin(), entries.end(),
            [](const auto& a, const auto& b) { return a->first < b->first; });

  // Iterate over pairs and insert.
  rocksdb::Slice k_view{nullptr, sizeof(Key)};
  rocksdb::Slice v_view{nullptr, parts.empty() ? 0 : parts[0].value_size};

  for (const Entry* const entry : entries) {
    k_view.data_ = reinterpret_cast<const char*>(&entry->first);
    v_view.data_ = entry->second->value;
    HCTR_ROCKSDB_CHECK(file.Put(k_view, v_view));
  }
}

template <typename Key>
size_t MultiProcessHashMapBackend<Key>::resolve_overflow_(const std::string& table_name,
                                                          Partition& part) {
  // Return if no overflow.
  if (part.entries.size() > this->overflow_margin_) {
    return 0;
  }

  size_t hit_count = 0;

  switch (this->overflow_policy_) {
    case DatabaseOverflowPolicy_t::EvictOldest: {
      // Fetch keys and insert times.
      std::vector<std::pair<Key, time_t>> kt;
      kt.reserve(part.entries.size());
      for (const auto& entry : part.entries) {
        kt.emplace_back(entry.first, entry.second->last_access);
      }

      // Sort by ascending by time.
      std::sort(kt.begin(), kt.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });

      // Call erase, until we reached the target amount.
      for (auto kt_it = kt.begin(); kt_it != kt.end();) {
        const auto& batch_end = std::min(kt_it + this->max_set_batch_size_, kt.end());

        HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Partition " << table_name << '/'
                                 << part.index << " is overflowing (size = " << part.entries.size()
                                 << " > " << this->overflow_margin_ << "): Attempting to evict "
                                 << (batch_end - kt_it) << " OLDEST key/value pairs!" << std::endl;

        for (; kt_it != batch_end; kt_it++) {
          HCTR_HASH_MAP_BACKEND_EVICT_(kt_it->first);
        }
        if (part.entries.size() <= this->overflow_resolution_target_) {
          break;
        }
      }
    } break;
    case DatabaseOverflowPolicy_t::EvictRandom: {
      // Fetch all keys.
      std::vector<Key> k;
      k.reserve(part.entries.size());
      for (const auto& pair : part.entries) {
        k.emplace_back(pair.first);
      }

      // Shuffle the keys.
      // TODO: This randomizer shoud fetch its seed from a central source.
      std::random_device rd;
      std::default_random_engine gen(rd());
      std::shuffle(k.begin(), k.end(), gen);

      // Delete items.
      for (auto k_it = k.begin(); k_it != k.end();) {
        const auto& batch_end = std::min(k_it + this->max_set_batch_size_, k.end());

        HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Partition " << table_name << "/"
                                 << part.index << " is overflowing (size = " << part.entries.size()
                                 << " > " << this->overflow_margin_ << "): Attempting to evict "
                                 << (batch_end - k_it) << " RANDOM key/value pairs!" << std::endl;

        // Call erase, until we reached the target amount.
        for (; k_it != batch_end; k_it++) {
          HCTR_HASH_MAP_BACKEND_EVICT_(*k_it);
        }
        if (part.entries.size() <= this->overflow_resolution_target_) {
          break;
        }
      }
    } break;
    default: {
      HCTR_LOG_S(WARNING, WORLD)
          << get_name() << " backend; Partition " << table_name << "/" << part.index
          << " is overflowing (size = " << part.entries.size() << " > " << this->overflow_margin_
          << "): Overflow cannot be resolved. No implementation for selected policy (="
          << this->overflow_policy_ << ")!" << std::endl;
    } break;
  }

  return hit_count;
}

template class MultiProcessHashMapBackend<unsigned int>;
template class MultiProcessHashMapBackend<long long>;

#ifdef HCTR_HASH_MAP_BACKEND_CONTAINS_
#undef HCTR_HASH_MAP_BACKEND_CONTAINS_
#else
#error HCTR_HASH_MAP_BACKEND_CONTAINS_ not defined. Sanity check!
#endif

#ifdef HCTR_HASH_MAP_BACKEND_INSERT_
#undef HCTR_HASH_MAP_BACKEND_INSERT_
#else
#error HCTR_HASH_MAP_BACKEND_INSERT_ not defined. Sanity check!
#endif

#ifdef HCTR_HASH_MAP_BACKEND_FETCH_
#undef HCTR_HASH_MAP_BACKEND_FETCH_
#else
#error HCTR_HASH_MAP_BACKEND_FETCH_ not defined. Sanity check!
#endif

#ifdef HCTR_HASH_MAP_BACKEND_EVICT_
#undef HCTR_HASH_MAP_BACKEND_EVICT_
#else
#error HCTR_HASH_MAP_BACKEND_EVICT_ not defined. Sanity check!
#endif

}  // namespace HugeCTR
