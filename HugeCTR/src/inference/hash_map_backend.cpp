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
#include <inference/hash_map_backend.hpp>
#include <random>

namespace HugeCTR {

template <typename TPartition>
HashMapBackendBase<TPartition>::HashMapBackendBase(const size_t overflow_margin,
                                                   const DatabaseOverflowPolicy_t overflow_policy,
                                                   const double overflow_resolution_target)
    : overflow_margin_(overflow_margin),
      overflow_policy_(overflow_policy),
      overflow_resolution_target_(hctr_safe_cast<size_t>(
          static_cast<double>(overflow_margin) * overflow_resolution_target + 0.5)) {
  HCTR_CHECK(overflow_resolution_target_ <= overflow_margin_);
}

template <typename TPartition>
void HashMapBackendBase<TPartition>::resolve_overflow_(const std::string& table_name,
                                                       const size_t part_idx, TPartition& part,
                                                       const size_t value_size) const {
  const size_t evict_amount = part.size() - overflow_resolution_target_;
  if (evict_amount <= 0) {
    HCTR_LOG(WARNING, WORLD,
             "%s backend. Table '%s' p%d (size = %d > %d). Overflow cannot be resolved. Evict "
             "amount (=%d) is negative!",
             this->get_name(), table_name.c_str(), part_idx, part.size(), overflow_margin_,
             evict_amount);
    return;
  }

  const size_t value_time_size = value_size + sizeof(time_t);

  if (overflow_policy_ == DatabaseOverflowPolicy_t::EvictOldest) {
    // Fetch keys and insert times.
    std::vector<std::pair<TKey, time_t>> keys_times;
    keys_times.reserve(part.size());
    for (const auto& pair : part) {
      HCTR_CHECK_HINT(pair.second.size() == value_time_size, "Value size mismatch!");
      keys_times.emplace_back(pair.first,
                              *reinterpret_cast<const time_t*>(&pair.second[value_size]));
    }

    // Sort by ascending by time.
    std::sort(keys_times.begin(), keys_times.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    // Call erase, until we reached the target amount.
    HCTR_LOG(INFO, WORLD,
             "%s backend. Table '%s' p%d (size = %d > %d). Resolving overflow by evicting the %d "
             "OLDEST key/value pairs!\n",
             this->get_name(), table_name.c_str(), part_idx, part.size(), overflow_margin_,
             evict_amount);

    const auto& keys_times_end = keys_times.begin() + evict_amount;
    for (auto kt = keys_times.begin(); kt != keys_times_end; kt++) {
      part.erase(kt->first);
    }
  } else if (overflow_policy_ == DatabaseOverflowPolicy_t::EvictRandom) {
    // Fetch all keys.
    std::vector<TKey> keys;
    keys.reserve(part.size());
    for (const auto& kt : part) {
      keys.emplace_back(kt.first);
    }

    // Shuffle the keys.
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::shuffle(keys.begin(), keys.end(), gen);

    // Call erase, until we reached the target amount.
    HCTR_LOG(INFO, WORLD,
             "%s backend. Table '%s' p%d (size = %d > %d). Resolving overflow by evicting the %d "
             "RANDOM key/value pairs!\n",
             this->get_name(), table_name.c_str(), part_idx, part.size(), overflow_margin_,
             evict_amount);

    for (const auto& k : keys) {
      part.erase(k);
    }
  } else {
    HCTR_LOG(WARNING, WORLD,
             "%s backend. Table '%s' p%d (size = %d > %d). Overflow cannot be resolved. "
             "No implementation for selected policy (=%d)!",
             this->get_name(), table_name.c_str(), part_idx, part.size(), overflow_margin_,
             overflow_policy_);
  }
}

#ifdef HCTR_HASH_MAP_BACKEND_INSERT_
#error HCTR_HASH_MAP_BACKEND_INSERT_ already defined. Potential naming conflict!
#else
#define HCTR_HASH_MAP_BACKEND_INSERT_(KEY, VALUES_PTR)                                       \
  do {                                                                                       \
    const auto& it = part.try_emplace((KEY), value_time_size).first;                         \
    HCTR_CHECK_HINT(it->second.size() == value_time_size, "Value size mismatch (%d <> %d)!", \
                    it->second.size(), value_time_size);                                     \
                                                                                             \
    memcpy(&it->second[0], (VALUES_PTR), value_size);                                        \
    memcpy(&it->second[value_size], &now_time, sizeof(time_t));                              \
    num_inserts++;                                                                           \
  } while (0)
#endif

#ifdef HCTR_HASH_MAP_BACKEND_FETCH_
#error HCTR_HASH_MAP_BACKEND_FETCH_ already defined. Potential naming conflict!
#else
#define HCTR_HASH_MAP_BACKEND_FETCH_(KEY, INDEX)                                               \
  do {                                                                                         \
    const auto& it = part.find((KEY));                                                         \
    if (it != part.end()) {                                                                    \
      HCTR_CHECK_HINT(it->second.size() == value_time_size, "Value size mismatch (%d <> %d)!", \
                      it->second.size(), value_time_size);                                     \
      memcpy(&values[(INDEX)*value_size], it->second.data(), value_size);                      \
      hit_count++;                                                                             \
    } else {                                                                                   \
      missing_callback((INDEX));                                                               \
    }                                                                                          \
  } while (0)
#endif

template <typename TPartition>
HashMapBackend<TPartition>::HashMapBackend(const size_t overflow_margin,
                                           const DatabaseOverflowPolicy_t overflow_policy,
                                           const double overflow_resolution_target)
    : TBase(overflow_margin, overflow_policy, overflow_resolution_target) {
  HCTR_LOG(INFO, WORLD, "Created blank database backend in local memory!\n");
}

template <typename TPartition>
size_t HashMapBackend<TPartition>::contains(const std::string& table_name, const size_t num_keys,
                                            const TKey* keys) const {
  // Locate the partition.
  const auto& tables_it = tables_.find(table_name);
  if (tables_it == tables_.end()) {
    return TBase::contains(table_name, num_keys, keys);
  }
  const TPartition& part = tables_it->second;

  size_t hit_count = 0;
  const TKey* const keys_end = &keys[num_keys];

  // Traverse through keys, search them and aggregate.
  for (; keys != keys_end; keys++) {
    hit_count += part.find(*keys) != part.end();
  }

  HCTR_LOG(DEBUG, WORLD, "%s backend. Table: %s. Found %d / %d keys.\n", get_name(),
           table_name.c_str(), hit_count, num_keys);
  return hit_count;
}

template <typename TPartition>
bool HashMapBackend<TPartition>::insert(const std::string& table_name, const size_t num_pairs,
                                        const TKey* const keys, const char* values,
                                        const size_t value_size) {
  // Locate the partition, or create it, if it does not exist yet.
  const auto& table_it = tables_.try_emplace(table_name).first;
  TPartition& part = table_it->second;

  const size_t value_time_size = value_size + sizeof(time_t);
  const time_t now_time = time(nullptr);
  size_t num_inserts = 0;
  const TKey* const keys_end = &keys[num_pairs];

  // Traverse through keys/values, and insert them one by one.
  for (const TKey* k = keys; k != keys_end; k++) {
    if (part.size() > this->overflow_margin_) {
      this->resolve_overflow_(table_name, 0, part, value_size);
    }
    HCTR_HASH_MAP_BACKEND_INSERT_(*k, values);
    values += value_size;
  }

  HCTR_LOG(DEBUG, WORLD, "%s backend. Table: %s. Inserted %d / %d pairs.\n", get_name(),
           table_name.c_str(), num_inserts, num_pairs);
  return true;
}

template <typename TPartition>
size_t HashMapBackend<TPartition>::fetch(const std::string& table_name, const size_t num_keys,
                                         const TKey* const keys, char* const values,
                                         const size_t value_size,
                                         MissingKeyCallback& missing_callback) const {
  // Locate the partition.
  const auto& tables_it = tables_.find(table_name);
  if (tables_it == tables_.end()) {
    return TBase::fetch(table_name, num_keys, keys, values, value_size, missing_callback);
  }
  const TPartition& part = tables_it->second;

  const size_t value_time_size = value_size + sizeof(time_t);
  size_t hit_count = 0;
  const TKey* const keys_end = &keys[num_keys];

  // Traverse through keys, and fetch them one by one.
  for (const TKey* k = keys; k != keys_end; k++) {
    HCTR_HASH_MAP_BACKEND_FETCH_(*k, k - keys);
  }

  HCTR_LOG(DEBUG, WORLD, "%s backend. Table: %s. Fetched %d / %d values.\n", get_name(),
           table_name.c_str(), hit_count, num_keys);
  return hit_count;
}

template <typename TPartition>
size_t HashMapBackend<TPartition>::fetch(const std::string& table_name, const size_t num_indices,
                                         const size_t* indices, const TKey* const keys,
                                         char* const values, const size_t value_size,
                                         MissingKeyCallback& missing_callback) const {
  // Locate the partition.
  const auto& tables_it = tables_.find(table_name);
  if (tables_it == tables_.end()) {
    return TBase::fetch(table_name, num_indices, indices, keys, values, value_size,
                        missing_callback);
  }
  const TPartition& part = tables_it->second;

  const size_t value_time_size = value_size + sizeof(time_t);
  size_t hit_count = 0;

  // Traverse through indices, lookup the keys, and fetch them one by one.
  const size_t* const indices_end = &indices[num_indices];
  for (; indices != indices_end; indices++) {
    HCTR_HASH_MAP_BACKEND_FETCH_(keys[*indices], *indices);
  }

  HCTR_LOG(DEBUG, WORLD, "%s backend. Table: %s. Fetched %d / %d values.\n", get_name(),
           table_name.c_str(), hit_count, num_indices);
  return hit_count;
}

template <typename TPartition>
size_t HashMapBackend<TPartition>::evict(const std::string& table_name) {
  // Locate the partition.
  const auto& tables_it = tables_.find(table_name);
  if (tables_it == tables_.end()) {
    return 0;
  }
  const TPartition& part = tables_it->second;

  // Count items and erase.
  const size_t hit_count = part.size();
  tables_.erase(table_name);

  HCTR_LOG(DEBUG, WORLD, "%s backend. Table %s erased (%d pairs).\n", get_name(),
           table_name.c_str(), hit_count);
  return hit_count;
}

template <typename TPartition>
size_t HashMapBackend<TPartition>::evict(const std::string& table_name, const size_t num_keys,
                                         const TKey* keys) {
  // Locate the partition.
  const auto& tables_it = tables_.find(table_name);
  if (tables_it == tables_.end()) {
    return 0;
  }
  TPartition& part = tables_it->second;

  size_t hit_count = 0;
  const TKey* const keys_end = &keys[num_keys];

  // Traverse through keys, and delete them one by one.
  for (; keys != keys_end; keys++) {
    hit_count += part.erase(*keys);
  }

  HCTR_LOG(DEBUG, WORLD, "%s backend. Table %s. %d / %d pairs erased.\n", get_name(),
           table_name.c_str(), hit_count, num_keys);
  return hit_count;
}

template class HCTR_DB_HASH_MAP_STL_(HashMapBackend, unsigned int);
template class HCTR_DB_HASH_MAP_PHM_(HashMapBackend, unsigned int);
template class HCTR_DB_HASH_MAP_STL_(HashMapBackend, long long);
template class HCTR_DB_HASH_MAP_PHM_(HashMapBackend, long long);

template <typename TPartition>
ParallelHashMapBackend<TPartition>::ParallelHashMapBackend(
    const size_t num_partitions, const size_t overflow_margin,
    const DatabaseOverflowPolicy_t overflow_policy, const double overflow_resolution_target)
    : TBase(overflow_margin, overflow_policy, overflow_resolution_target),
      num_partitions_(num_partitions) {
  HCTR_CHECK_HINT(num_partitions_ >= 1, "Number of partitions (%d) must be 1 or higher!",
                  num_partitions_);

  HCTR_LOG(INFO, WORLD,
           "Created parallel (%d partitions) blank database backend in local memory!\n",
           num_partitions_);
}

template <typename TPartition>
size_t ParallelHashMapBackend<TPartition>::contains(const std::string& table_name,
                                                    const size_t num_keys,
                                                    const TKey* const keys) const {
  // Locate the partitions.
  const auto& tables_it = tables_.find(table_name);
  if (tables_it == tables_.end()) {
    return TBase::contains(table_name, num_keys, keys);
  }
  const std::vector<TPartition>& parts = tables_it->second;
  HCTR_CHECK(parts.size() == num_partitions_);

  size_t hit_count;
  const TKey* keys_end = &keys[num_keys];

  // Traverse through keys, search them and aggregate.
  if (num_keys < num_partitions_) {
    hit_count = 0;
    for (const TKey* k = keys; k != keys_end; k++) {
      const TPartition& part = parts[*k % num_partitions_];
      hit_count += part.find(*k) != part.end();
    }
  } else {
    std::atomic<size_t> joint_hit_count(0);
    std::for_each(std::execution::par, parts.begin(), parts.end(), [&](const TPartition& part) {
      const size_t part_idx = &part - &parts[0];

      size_t hit_count = 0;
      for (const TKey* k = keys; k != keys_end; k++) {
        if (*k % num_partitions_ == part_idx) {
          hit_count += part.find(*k) != part.end();
        }
      }
      joint_hit_count += hit_count;
    });
    hit_count = static_cast<size_t>(joint_hit_count);
  }

  HCTR_LOG(DEBUG, WORLD, "%s backend. Found %d / %d keys.\n", get_name(), hit_count, num_keys);
  return hit_count;
}

template <typename TPartition>
bool ParallelHashMapBackend<TPartition>::insert(const std::string& table_name,
                                                const size_t num_pairs, const TKey* const keys,
                                                const char* const values, const size_t value_size) {
  // Locate the partitions, or create them, if they do not exist yet.
  const auto& tables_it = tables_.try_emplace(table_name, num_partitions_).first;
  std::vector<TPartition>& parts = tables_it->second;
  HCTR_CHECK(parts.size() == num_partitions_);

  const size_t value_time_size = value_size + sizeof(time_t);
  const time_t now_time = time(nullptr);
  size_t num_inserts;
  const TKey* const keys_end = &keys[num_pairs];

  // Traverse through keys/values, and insert them one by one.
  if (num_pairs < num_partitions_) {
    num_inserts = 0;
    for (const TKey* k = keys; k != keys_end; k++) {
      const size_t part_idx = *k % num_partitions_;
      TPartition& part = parts[part_idx];

      if (parts.size() > this->overflow_margin_) {
        this->resolve_overflow_(table_name, part_idx, part, value_size);
      }
      HCTR_HASH_MAP_BACKEND_INSERT_(*k, &values[(k - keys) * value_size]);
    }
  } else {
    std::atomic<size_t> joint_num_inserts(0);
    std::for_each(std::execution::par, parts.begin(), parts.end(), [&](TPartition& part) {
      const size_t part_idx = &part - &parts[0];

      size_t num_inserts = 0;
      for (const TKey* k = keys; k != keys_end; k++) {
        if (*k % num_partitions_ == part_idx) {
          if (part.size() > this->overflow_margin_) {
            this->resolve_overflow_(table_name, part_idx, part, value_size);
          }
          HCTR_HASH_MAP_BACKEND_INSERT_(*k, &values[(k - keys) * value_size]);
        }
      }
      joint_num_inserts += num_inserts;
    });
    num_inserts = static_cast<size_t>(joint_num_inserts);
  }

  HCTR_LOG(DEBUG, WORLD, "%s backend. Table: %s. Inserted %d / %d pairs.\n", get_name(),
           table_name.c_str(), num_inserts, num_pairs);
  return true;
}

template <typename TPartition>
size_t ParallelHashMapBackend<TPartition>::fetch(const std::string& table_name,
                                                 const size_t num_keys, const TKey* const keys,
                                                 char* const values, const size_t value_size,
                                                 MissingKeyCallback& missing_callback) const {
  // Locate the partitions.
  const auto& tables_it = tables_.find(table_name);
  if (tables_it == tables_.end()) {
    return TBase::fetch(table_name, num_keys, keys, values, value_size, missing_callback);
  }
  const std::vector<TPartition>& parts = tables_it->second;
  HCTR_CHECK(parts.size() == num_partitions_);

  const size_t value_time_size = value_size + sizeof(time_t);
  size_t hit_count;
  const TKey* const keys_end = &keys[num_keys];

  // Traverse through keys, and fetch them one by one.
  if (num_keys < num_partitions_) {
    hit_count = 0;
    for (const TKey* k = keys; k != keys_end; k++) {
      const size_t part_idx = *k % num_partitions_;
      const TPartition& part = parts[part_idx];
      HCTR_HASH_MAP_BACKEND_FETCH_(*k, k - keys);
    }
  } else {
    std::atomic<size_t> joint_hit_count(0);
    std::for_each(std::execution::par, parts.begin(), parts.end(), [&](const TPartition& part) {
      const size_t part_idx = &part - &parts[0];

      size_t hit_count = 0;
      for (const TKey* k = keys; k != keys_end; k++) {
        if (*k % num_partitions_ == part_idx) {
          HCTR_HASH_MAP_BACKEND_FETCH_(*k, k - keys);
        }
      }
      joint_hit_count += hit_count;
    });
    hit_count = static_cast<size_t>(joint_hit_count);
  }

  HCTR_LOG(DEBUG, WORLD, "%s backend. Table: %s. Fetched %d / %d values.\n", get_name(),
           table_name.c_str(), hit_count, num_keys);
  return hit_count;
}

template <typename TPartition>
size_t ParallelHashMapBackend<TPartition>::fetch(const std::string& table_name,
                                                 const size_t num_indices,
                                                 const size_t* const indices,
                                                 const TKey* const keys, char* const values,
                                                 const size_t value_size,
                                                 MissingKeyCallback& missing_callback) const {
  // Locate the partitions.
  const auto& tables_it = tables_.find(table_name);
  if (tables_it == tables_.end()) {
    return TBase::fetch(table_name, num_indices, indices, keys, values, value_size,
                        missing_callback);
  }
  const std::vector<TPartition>& parts = tables_it->second;
  HCTR_CHECK(parts.size() == num_partitions_);

  const size_t value_time_size = value_size + sizeof(time_t);
  size_t hit_count;
  const size_t* const indices_end = &indices[num_indices];

  // Traverse through indices, lookup the keys, and fetch them one by one.
  if (num_indices < num_partitions_) {
    hit_count = 0;
    for (const size_t* i = indices; i != indices_end; i++) {
      const TKey& k = keys[*i];
      const TPartition& part = parts[k % num_partitions_];
      HCTR_HASH_MAP_BACKEND_FETCH_(k, *i);
    }
  } else {
    std::atomic<size_t> joint_hit_count(0);
    std::for_each(std::execution::par, parts.begin(), parts.end(), [&](const TPartition& part) {
      const size_t part_idx = &part - &parts[0];

      size_t hit_count = 0;
      for (const size_t* i = indices; i != indices_end; i++) {
        const TKey& k = keys[*i];
        if (k % num_partitions_ == part_idx) {
          HCTR_HASH_MAP_BACKEND_FETCH_(k, *i);
        }
      }
      joint_hit_count += hit_count;
    });
    hit_count = static_cast<size_t>(joint_hit_count);
  }

  HCTR_LOG(DEBUG, WORLD, "%s backend. Table: %s. Fetched %d / %d values.\n", get_name(),
           table_name.c_str(), hit_count, num_indices);
  return hit_count;
}

template <typename TPartition>
size_t ParallelHashMapBackend<TPartition>::evict(const std::string& table_name) {
  // Locate the partitions.
  const auto& tables_it = tables_.find(table_name);
  if (tables_it == tables_.end()) {
    return 0;
  }
  const std::vector<TPartition>& parts = tables_it->second;
  HCTR_CHECK(parts.size() == num_partitions_);

  // Count items and erase.
  size_t hit_count = 0;
  for (const TPartition& part : parts) {
    hit_count += part.size();
  }
  tables_.erase(table_name);

  HCTR_LOG(DEBUG, WORLD, "%s backend. Table %s erased (%d pairs).\n", get_name(),
           table_name.c_str(), hit_count);
  return hit_count;
}

template <typename TPartition>
size_t ParallelHashMapBackend<TPartition>::evict(const std::string& table_name,
                                                 const size_t num_keys, const TKey* const keys) {
  // Locate the partitions.
  const auto& tables_it = tables_.find(table_name);
  if (tables_it == tables_.end()) {
    return 0;
  }
  std::vector<TPartition>& parts = tables_it->second;
  HCTR_CHECK(parts.size() == num_partitions_);

  size_t hit_count;
  const TKey* const keys_end = &keys[num_keys];

  // Traverse through keys, and delete them one by one.
  if (num_keys < num_partitions_) {
    hit_count = 0;
    for (const TKey* k = keys; k != keys_end; k++) {
      TPartition& part = parts[*k % num_partitions_];
      hit_count += part.erase(*k);
    }
  } else {
    std::atomic<size_t> joint_hit_count(0);
    std::for_each(std::execution::par, parts.begin(), parts.end(), [&](TPartition& part) {
      const size_t part_idx = &part - &parts[0];

      size_t hit_count = 0;
      for (const TKey* k = keys; k != keys_end; k++) {
        if (*k % num_partitions_ == part_idx) {
          hit_count += part.erase(*k);
        }
      }

      joint_hit_count += hit_count;
    });
    hit_count = static_cast<size_t>(joint_hit_count);
  }

  HCTR_LOG(DEBUG, WORLD, "%s backend. Table %s. %d / %d pairs erased.\n", get_name(),
           table_name.c_str(), hit_count, num_keys);
  return hit_count;
}

template class HCTR_DB_HASH_MAP_STL_(ParallelHashMapBackend, unsigned int);
template class HCTR_DB_HASH_MAP_PHM_(ParallelHashMapBackend, unsigned int);
template class HCTR_DB_HASH_MAP_STL_(ParallelHashMapBackend, long long);
template class HCTR_DB_HASH_MAP_PHM_(ParallelHashMapBackend, long long);

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

}  // namespace HugeCTR
