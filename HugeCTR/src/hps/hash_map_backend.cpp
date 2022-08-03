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

#include <algorithm>
#include <atomic>
#include <base/debug/logger.hpp>
#include <cstring>
#include <execution>
#include <hps/hash_map_backend.hpp>
#include <hps/hier_parameter_server_base.hpp>
#include <random>

#define HCTR_USE_XXHASH

#ifdef HCTR_USE_XXHASH
#include <xxh3.h>
#define HCTR_HASH_OF_KEY(KEY) (XXH3_64bits((KEY), sizeof(TKey)))
#else
#define HCTR_HASH_OF_KEY(KEY) (static_cast<size_t>(*KEY))
#endif
#define HCTR_PARTITION_OF_KEY(KEY) (HCTR_HASH_OF_KEY(KEY) % num_partitions_)

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
#define HCTR_HASH_MAP_BACKEND_INSERT_(KEY, VALUES_PTR)                                     \
  do {                                                                                     \
    const auto& res = part.entries.try_emplace((KEY));                                     \
    TEntry& entry = res.first->second;                                                     \
    entry.last_access = now;                                                               \
                                                                                           \
    /* If new insertion. */                                                                \
    if (res.second) {                                                                      \
      /* If no free space, allocate another buffer, and fill pointer queue. */             \
      if (part.value_ptrs.empty()) {                                                       \
        const size_t num_values = allocation_rate_ / value_size;                           \
        part.value_buffers.emplace_back(num_values* value_size);                           \
        std::vector<char>& value_buffer = part.value_buffers.back();                       \
                                                                                           \
        part.value_ptrs.reserve(num_values);                                               \
        for (auto it = value_buffer.end(); it != value_buffer.begin(); it -= value_size) { \
          part.value_ptrs.emplace_back(&*it);                                              \
        }                                                                                  \
      }                                                                                    \
                                                                                           \
      /* Fetch pointer. */                                                                 \
      entry.value = part.value_ptrs.back();                                                \
      part.value_ptrs.pop_back();                                                          \
    }                                                                                      \
                                                                                           \
    std::memcpy(entry.value, (VALUES_PTR), value_size);                                    \
    num_inserts++;                                                                         \
  } while (0)
#endif

#ifdef HCTR_HASH_MAP_BACKEND_FETCH_
#error HCTR_HASH_MAP_BACKEND_FETCH_ already defined. Potential naming conflict!
#else
#define HCTR_HASH_MAP_BACKEND_FETCH_(KEY, INDEX)                                             \
  do {                                                                                       \
    const auto& it = part.entries.find((KEY));                                               \
    if (it != part.entries.end()) {                                                          \
      TEntry& entry = it->second;                                                            \
                                                                                             \
      /* Race-conditions here are deliberately ignored because insignificant in practice. */ \
      entry.last_access = now;                                                               \
      on_hit((INDEX), entry.value, part.value_size);                                         \
      hit_count++;                                                                           \
    } else {                                                                                 \
      on_miss((INDEX));                                                                      \
    }                                                                                        \
  } while (0)
#endif

#ifdef HCTR_HASH_MAP_BACKEND_EVICT_
#error HCTR_HASH_MAP_BACKEND_EVICT_ already defined. Potential naming conflict!
#else
#define HCTR_HASH_MAP_BACKEND_EVICT_(KEY)        \
  do {                                           \
    const auto& it = part.entries.find((KEY));   \
    if (it != part.entries.end()) {              \
      const TEntry& entry = it->second;          \
                                                 \
      /* Stash pointer and erase item. */        \
      part.value_ptrs.emplace_back(entry.value); \
      part.entries.erase(it);                    \
      hit_count++;                               \
    }                                            \
  } while (0)
#endif

template <typename TKey>
HashMapBackend<TKey>::HashMapBackend(const size_t num_partitions, const size_t allocation_rate,
                                     const size_t overflow_margin,
                                     const DatabaseOverflowPolicy_t overflow_policy,
                                     const double overflow_resolution_target)
    : TBase(overflow_margin, overflow_policy, overflow_resolution_target),
      num_partitions_(num_partitions),
      allocation_rate_(allocation_rate) {}

template <typename TKey>
size_t HashMapBackend<TKey>::size(const std::string& table_name) const {
  std::shared_lock lock(read_write_guard_);

  // Locate the partitions.
  const auto& tables_it = tables_.find(table_name);
  if (tables_it == tables_.end()) {
    return 0;
  }
  const std::vector<TPartition>& parts = tables_it->second;

  // Iterate over partitions and accumulate sizes.
  size_t num_entries = 0;
  for (const TPartition& part : parts) {
    num_entries += part.entries.size();
  }
  return num_entries;
}

template <typename TKey>
size_t HashMapBackend<TKey>::contains(const std::string& table_name, const size_t num_keys,
                                      const TKey* const keys) const {
  std::shared_lock lock(read_write_guard_);

  // Locate the partitions.
  const auto& tables_it = tables_.find(table_name);
  if (tables_it == tables_.end()) {
    return TBase::contains(table_name, num_keys, keys);
  }
  const std::vector<TPartition>& parts = tables_it->second;

  size_t hit_count = 0;

  if (num_keys == 0) {
    // Nothing to do ;-).
  } else if (num_keys == 1) {
    const TPartition& part = parts[HCTR_PARTITION_OF_KEY(keys)];
    HCTR_HASH_MAP_BACKEND_CONTAINS_(*keys);
  } else {
    const TKey* keys_end = &keys[num_keys];

    if (parts.size() == 1) {
      const TPartition& part = parts.front();

      // Traverse through keys, and fetch them one by one.
      for (const TKey* k = keys; k != keys_end; k++) {
        HCTR_HASH_MAP_BACKEND_CONTAINS_(*k);
      }
    } else {
      std::atomic<size_t> joint_hit_count(0);

      // Spawn threads.
      std::for_each(std::execution::par, parts.begin(), parts.end(), [&](const TPartition& part) {
        size_t hit_count = 0;
        // Traverse through keys, and fetch them one by one.
        for (const TKey* k = keys; k != keys_end; k++) {
          if (HCTR_PARTITION_OF_KEY(k) == part.index) {
            HCTR_HASH_MAP_BACKEND_CONTAINS_(*k);
          }
        }
        joint_hit_count += hit_count;
      });

      hit_count += static_cast<size_t>(joint_hit_count);
    }
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend. Found " << hit_count << " / " << num_keys
                           << " keys." << std::endl;
  return hit_count;
}

template <typename TKey>
bool HashMapBackend<TKey>::insert(const std::string& table_name, const size_t num_pairs,
                                  const TKey* const keys, const char* const values,
                                  const size_t value_size) {
  std::unique_lock lock(read_write_guard_);

  // Locate the partitions, or create them, if they do not exist yet.
  const auto& tables_it = tables_.try_emplace(table_name).first;
  std::vector<TPartition>& parts = tables_it->second;
  if (parts.empty()) {
    HCTR_CHECK(value_size > 0 && value_size <= allocation_rate_);

    parts.reserve(num_partitions_);
    for (size_t i = 0; i < num_partitions_; i++) {
      parts.emplace_back(i, value_size);
    }
  } else {
    HCTR_CHECK(parts.size() == num_partitions_);
  }

  const time_t now = std::time(nullptr);
  size_t num_inserts = 0;

  if (num_pairs == 0) {
    // Nothing to do ;-).
  } else if (num_pairs == 1) {
    TPartition& part = parts[HCTR_PARTITION_OF_KEY(keys)];
    HCTR_CHECK(part.value_size == value_size);

    if (part.entries.size() >= this->overflow_margin_) {
      resolve_overflow_(table_name, part);
    }
    HCTR_HASH_MAP_BACKEND_INSERT_(*keys, values);
  } else {
    const TKey* const keys_end = &keys[num_pairs];

    if (parts.size() == 1) {
      TPartition& part = parts.front();
      HCTR_CHECK(part.value_size == value_size);

      // Traverse through keys/values, and insert them one by one.
      for (const TKey* k = keys; k != keys_end; k++) {
        if (part.entries.size() >= this->overflow_margin_) {
          resolve_overflow_(table_name, part);
        }
        HCTR_HASH_MAP_BACKEND_INSERT_(*k, &values[(k - keys) * value_size]);
      }
    } else {
      std::atomic<size_t> joint_num_inserts(0);

      // Spawn threads.
      std::for_each(std::execution::par, parts.begin(), parts.end(), [&](TPartition& part) {
        HCTR_CHECK(part.value_size == value_size);

        size_t num_inserts = 0;
        // Traverse through keys/values, and insert them one by one.
        for (const TKey* k = keys; k != keys_end; k++) {
          if (HCTR_PARTITION_OF_KEY(k) == part.index) {
            if (part.entries.size() >= this->overflow_margin_) {
              resolve_overflow_(table_name, part);
            }
            HCTR_HASH_MAP_BACKEND_INSERT_(*k, &values[(k - keys) * value_size]);
          }
        }
        joint_num_inserts += num_inserts;
      });

      num_inserts += static_cast<size_t>(joint_num_inserts);
    }
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend. Table: " << table_name << ". Inserted "
                           << num_inserts << " / " << num_pairs << " pairs." << std::endl;
  return true;
}

template <typename TKey>
size_t HashMapBackend<TKey>::fetch(const std::string& table_name, const size_t num_keys,
                                   const TKey* const keys, const DatabaseHitCallback& on_hit,
                                   const DatabaseMissCallback& on_miss) {
  std::shared_lock lock(read_write_guard_);

  // Locate the partition.
  const auto& tables_it = tables_.find(table_name);
  if (tables_it == tables_.end()) {
    return TBase::fetch(table_name, num_keys, keys, on_hit, on_miss);
  }
  std::vector<TPartition>& parts = tables_it->second;

  const time_t now = std::time(nullptr);
  size_t hit_count = 0;

  if (num_keys == 0) {
    // Nothing to do ;-).
  } else if (num_keys == 1) {
    TPartition& part = parts[HCTR_PARTITION_OF_KEY(keys)];
    HCTR_HASH_MAP_BACKEND_FETCH_(*keys, 0);
  } else {
    const TKey* const keys_end = &keys[num_keys];

    if (parts.size() == 1) {
      TPartition& part = parts.front();

      // Traverse through keys, and fetch them one by one.
      for (const TKey* k = keys; k != keys_end; k++) {
        HCTR_HASH_MAP_BACKEND_FETCH_(*k, k - keys);
      }
    } else {
      std::atomic<size_t> joint_hit_count(0);

      // Spawn threads.
      std::for_each(std::execution::par, parts.begin(), parts.end(), [&](TPartition& part) {
        size_t hit_count = 0;
        // Traverse through keys, and fetch them one by one.
        for (const TKey* k = keys; k != keys_end; k++) {
          if (HCTR_PARTITION_OF_KEY(k) == part.index) {
            HCTR_HASH_MAP_BACKEND_FETCH_(*k, k - keys);
          }
        }
        joint_hit_count += hit_count;
      });

      hit_count += static_cast<size_t>(joint_hit_count);
    }
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend. Table: " << table_name << ". Fetched "
                           << hit_count << " / " << num_keys << " values." << std::endl;
  return hit_count;
}

template <typename TKey>
size_t HashMapBackend<TKey>::fetch(const std::string& table_name, const size_t num_indices,
                                   const size_t* const indices, const TKey* const keys,
                                   const DatabaseHitCallback& on_hit,
                                   const DatabaseMissCallback& on_miss) {
  std::shared_lock lock(read_write_guard_);

  // Locate the partitions.
  const auto& tables_it = tables_.find(table_name);
  if (tables_it == tables_.end()) {
    return TBase::fetch(table_name, num_indices, indices, keys, on_hit, on_miss);
  }
  std::vector<TPartition>& parts = tables_it->second;

  const time_t now = std::time(nullptr);
  size_t hit_count = 0;

  if (num_indices == 0) {
    // Nothing to do ;-).
  } else if (num_indices == 1) {
    const TKey& k = keys[*indices];
    TPartition& part = parts[HCTR_PARTITION_OF_KEY(&k)];
    HCTR_HASH_MAP_BACKEND_FETCH_(k, *indices);
  } else {
    const size_t* const indices_end = &indices[num_indices];

    if (parts.size() == 1) {
      TPartition& part = parts.front();

      // Traverse through indices, lookup the keys, and fetch them one by one.
      for (const size_t* i = indices; i != indices_end; i++) {
        HCTR_HASH_MAP_BACKEND_FETCH_(keys[*i], *i);
      }
    } else {
      std::atomic<size_t> joint_hit_count(0);

      std::for_each(std::execution::par, parts.begin(), parts.end(), [&](TPartition& part) {
        size_t hit_count = 0;
        // Traverse through indices, lookup the keys, and fetch them one by one.
        for (const size_t* i = indices; i != indices_end; i++) {
          const TKey& k = keys[*i];
          if (HCTR_PARTITION_OF_KEY(&k) == part.index) {
            HCTR_HASH_MAP_BACKEND_FETCH_(k, *i);
          }
        }
        joint_hit_count += hit_count;
      });

      hit_count += static_cast<size_t>(joint_hit_count);
    }
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend. Table: " << table_name << ". Fetched "
                           << hit_count << " / " << num_indices << " values." << std::endl;
  return hit_count;
}

template <typename TKey>
size_t HashMapBackend<TKey>::evict(const std::string& table_name) {
  std::unique_lock lock(read_write_guard_);

  // Locate the partition.
  const auto& tables_it = tables_.find(table_name);
  if (tables_it == tables_.end()) {
    return 0;
  }
  const std::vector<TPartition>& parts = tables_it->second;

  // Count items and erase.
  size_t hit_count = 0;
  for (const TPartition& part : parts) {
    hit_count += part.entries.size();
  }
  tables_.erase(tables_it);

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend. Table " << table_name << " erased ("
                           << hit_count << " pairs)." << std::endl;
  return hit_count;
}

template <typename TKey>
size_t HashMapBackend<TKey>::evict(const std::string& table_name, const size_t num_keys,
                                   const TKey* const keys) {
  std::unique_lock lock(read_write_guard_);

  // Locate the partitions.
  const auto& tables_it = tables_.find(table_name);
  if (tables_it == tables_.end()) {
    return 0;
  }
  std::vector<TPartition>& parts = tables_it->second;

  size_t hit_count = 0;

  if (num_keys == 0) {
    // Nothing to do ;-).
  } else if (num_keys == 1) {
    TPartition& part = parts[HCTR_PARTITION_OF_KEY(keys)];
    HCTR_HASH_MAP_BACKEND_EVICT_(*keys);
  } else {
    const TKey* const keys_end = &keys[num_keys];

    if (parts.size() == 1) {
      TPartition& part = parts.front();

      // Traverse through keys, and delete them one by one.
      for (const TKey* k = keys; k != keys_end; k++) {
        HCTR_HASH_MAP_BACKEND_EVICT_(*k);
      }
    } else {
      std::atomic<size_t> joint_hit_count(0);

      std::for_each(std::execution::par, parts.begin(), parts.end(), [&](TPartition& part) {
        size_t hit_count = 0;
        // Traverse through keys, and delete them one by one.
        for (const TKey* k = keys; k != keys_end; k++) {
          if (HCTR_PARTITION_OF_KEY(k) == part.index) {
            HCTR_HASH_MAP_BACKEND_EVICT_(*k);
          }
        }
        joint_hit_count += hit_count;
      });

      hit_count += static_cast<size_t>(joint_hit_count);
    }
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend. Table " << table_name << ". " << hit_count
                           << " / " << num_keys << " pairs erased." << std::endl;
  return hit_count;
}

template <typename TKey>
std::vector<std::string> HashMapBackend<TKey>::find_tables(const std::string& model_name) {
  const std::string& tag_prefix = HierParameterServerBase::make_tag_name(model_name, "", false);

  std::vector<std::string> matches;
  for (const auto& pair : tables_) {
    if (pair.first.find(tag_prefix) == 0) {
      matches.push_back(pair.first);
    }
  }
  return matches;
}

template <typename TKey>
size_t HashMapBackend<TKey>::resolve_overflow_(const std::string& table_name, TPartition& part) {
  const size_t num_entries = part.entries.size();

  // Determine amount of entries that must be evicted.
  const size_t evict_amount = num_entries - this->overflow_resolution_target_;
  if (evict_amount <= 0) {
    HCTR_LOG_S(WARNING, WORLD) << get_name() << " backend. Table '" << table_name << "' p"
                               << part.index << " (size = " << num_entries << " > "
                               << this->overflow_margin_
                               << "). Overflow cannot be resolved. Evict amount (=" << evict_amount
                               << ") is negative!" << std::endl;
    return 0;
  }

  size_t hit_count = 0;

  // Select overflow resolution strategy.
  switch (this->overflow_policy_) {
    case DatabaseOverflowPolicy_t::EvictOldest: {
      // Fetch keys and insert times.
      std::vector<std::pair<TKey, time_t> > keys_times;
      keys_times.reserve(num_entries);
      for (const auto& entry : part.entries) {
        keys_times.emplace_back(entry.first, entry.second.last_access);
      }

      // Sort by ascending by time.
      std::sort(keys_times.begin(), keys_times.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });

      // Call erase, until we reached the target amount.
      HCTR_LOG_S(INFO, WORLD) << get_name() << " backend. Table '" << table_name << "' p"
                              << part.index << " (size = " << num_entries << " > "
                              << this->overflow_margin_ << "). Resolving overflow by evicting the "
                              << evict_amount << " OLDEST key/value pairs!" << std::endl;

      const auto& keys_times_end = keys_times.begin() + evict_amount;
      for (auto kt = keys_times.begin(); kt != keys_times_end; kt++) {
        HCTR_HASH_MAP_BACKEND_EVICT_(kt->first);
      }
    } break;
    case DatabaseOverflowPolicy_t::EvictRandom: {
      // Fetch all keys.
      std::vector<TKey> keys;
      keys.reserve(num_entries);
      for (const auto& entry : part.entries) {
        keys.emplace_back(entry.first);
      }

      // Shuffle the keys.
      // TODO: This randomizer shoud fetch its seed from a central source.
      std::random_device rd;
      std::default_random_engine gen(rd());
      std::shuffle(keys.begin(), keys.end(), gen);

      // Call erase, until we reached the target amount.
      HCTR_LOG_S(INFO, WORLD) << get_name() << " backend. Table '" << table_name << "' p"
                              << part.index << " (size = " << num_entries << " > "
                              << this->overflow_margin_ << "). Resolving overflow by evicting the "
                              << evict_amount << " RANDOM key/value pairs!" << std::endl;

      const auto& keys_end = keys.begin() + evict_amount;
      for (auto k = keys.begin(); k != keys_end; k++) {
        HCTR_HASH_MAP_BACKEND_EVICT_(*k);
      }
    } break;
    default: {
      HCTR_LOG_S(WARNING, WORLD) << get_name() << " backend. Table '" << table_name << "' p"
                                 << part.index << " (size = " << num_entries << " > "
                                 << this->overflow_margin_ << "). Overflow cannot be resolved. "
                                 << "No implementation for selected policy (="
                                 << this->overflow_policy_ << ")!" << std::endl;
    } break;
  }

  return hit_count;
}

template class HashMapBackend<unsigned int>;
template class HashMapBackend<long long>;

}  // namespace HugeCTR
