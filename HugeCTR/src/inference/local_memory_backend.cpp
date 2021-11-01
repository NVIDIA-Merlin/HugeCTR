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

#include <atomic>
#include <cassert>
#include <cstring>
#include <inference/local_memory_backend.hpp>
#include <iostream>

namespace HugeCTR {

#define DEBUG                                                  \
  std::cout << "[DEBUG]" << __FILE__ << "|" << __func__ << "|" \
            << "LINE" << __LINE__ << " "

template <typename TKey>
LocalMemoryBackend<TKey>::LocalMemoryBackend() : TBase() {
  DEBUG << "Created blank database backend in local memory!" << std::endl;
}

template <typename TKey>
const char* LocalMemoryBackend<TKey>::get_name() const {
  return "LocalMemory";
}

template <typename TKey>
size_t LocalMemoryBackend<TKey>::contains(const std::string& table_name, const size_t num_keys,
                                          const TKey* keys) const {
  // Locate the embedding table.
  const auto& table_it = tables_.find(table_name);
  if (table_it == tables_.end()) {
    return TBase::contains(table_name, num_keys, keys);
  }
  const TMap& table = table_it->second;

  size_t hit_count = 0;

  // Subsequently search for IDs and fill in values.
  const TKey* const keys_end = &keys[num_keys];
  for (; keys != keys_end; keys++) {
    const auto& it = table.find(*keys);
    if (it != table.end()) {
      hit_count++;
    }
  }

  DEBUG << get_name() << " backend. Contains " << hit_count << " / " << num_keys << " keys."
        << std::endl;
  return hit_count;
}

template <typename TKey>
bool LocalMemoryBackend<TKey>::insert(const std::string& table_name, const size_t num_pairs,
                                      const TKey* keys, const char* values,
                                      const size_t value_size) {
  // Locate the embedding table, or create it, if it does not exist yet.
  const auto& table_it = tables_.try_emplace(table_name).first;
  TMap& table = table_it->second;
  table.reserve(num_pairs);

  // Just traverse through the KV-pairs and insert them one by one.
  const TKey* const keys_end = &keys[num_pairs];
  for (; keys != keys_end; keys++) {
    const char* const values_next = &values[value_size];
    table.emplace(std::piecewise_construct, std::forward_as_tuple(*keys),
                  std::forward_as_tuple(values, values_next));
    values = values_next;
  }

  DEBUG << get_name() << " backend updated. Inserted " << num_pairs
        << " key-value pairs. Now containing: " << table.size() << " key-value pairs." << std::endl;
  return true;
}

template <typename TKey>
size_t LocalMemoryBackend<TKey>::fetch(const std::string& table_name, const size_t num_keys,
                                       const TKey* const keys, char* const values,
                                       const size_t value_size,
                                       MissingKeyCallback& missing_callback) const {
  // Locate the embedding table.
  const auto& table_it = tables_.find(table_name);
  if (table_it == tables_.end()) {
    return TBase::fetch(table_name, num_keys, keys, values, value_size, missing_callback);
  }
  const TMap& table = table_it->second;

  size_t hit_count = 0;

  // Subsequently search for IDs and fill in values.
  for (size_t i = 0; i < num_keys; i++) {
    const auto& it = table.find(keys[i]);
    if (it != table.end()) {
      assert(it->second.size() == value_size);
      memcpy(&values[i * value_size], it->second.data(), value_size);
      hit_count++;
    } else {
      missing_callback(i);
    }
  }

  DEBUG << get_name() << " backend. Fetched " << hit_count << " / " << num_keys << " pairs."
        << std::endl;
  return hit_count;
}

template <typename TKey>
size_t LocalMemoryBackend<TKey>::fetch(const std::string& table_name, const size_t num_indices,
                                       const size_t* indices, const TKey* const keys,
                                       char* const values, const size_t value_size,
                                       MissingKeyCallback& missing_callback) const {
  // Locate the embedding table.
  const auto& table_it = tables_.find(table_name);
  if (table_it == tables_.end()) {
    return TBase::fetch(table_name, num_indices, indices, keys, values, value_size,
                        missing_callback);
  }
  const TMap& table = table_it->second;

  size_t hit_count = 0;

  // Subsequently search for IDs and fill in values.
  const size_t* const indices_end = &indices[num_indices];
  for (; indices != indices_end; indices++) {
    const auto& it = table.find(keys[*indices]);
    if (it != table.end()) {
      assert(it->second.size() == value_size);
      memcpy(&values[*indices * value_size], it->second.data(), value_size);
      hit_count++;
    } else {
      missing_callback(*indices);
    }
  }

  DEBUG << get_name() << " backend. Fetched " << hit_count << " / " << num_indices << " pairs."
        << std::endl;
  return hit_count;
}

template <typename TKey>
size_t LocalMemoryBackend<TKey>::evict(const std::string& table_name) {
  // Locate the embedding table.
  const auto& table_it = tables_.find(table_name);
  if (table_it == tables_.end()) {
    return 0;
  }
  const size_t table_size = table_it->second.size();

  // Delete table.
  tables_.erase(table_name);

  return table_size;
}

template <typename TKey>
size_t LocalMemoryBackend<TKey>::evict(const std::string& table_name, const size_t num_keys,
                                       const TKey* keys) {
  // Locate the embedding table.
  const auto& table_it = tables_.find(table_name);
  if (table_it == tables_.end()) {
    return 0;
  }
  TMap& table = table_it->second;

  size_t hit_count = 0;

  // Delete items one by one.
  const TKey* const keys_end = &keys[num_keys];
  for (; keys != keys_end; keys++) {
    hit_count += table.erase(*keys);
  }

  DEBUG << get_name() << " backend updated. Now containing: " << table.size() << " key-value pairs."
        << std::endl;
  return hit_count;
}

template class LocalMemoryBackend<unsigned int>;
template class LocalMemoryBackend<long long>;

template <typename TKey>
ParallelLocalMemoryBackend<TKey>::ParallelLocalMemoryBackend(const size_t num_partitions)
    : TBase(), num_partitions_(num_partitions) {
  if (num_partitions_ < 1) {
    DEBUG << "Number of partitions cannot be below 1!" << std::endl;
    exit(EXIT_FAILURE);
  }
  DEBUG << "Created parallel (" << num_partitions_
        << " partitions) blank database backend in local memory!" << std::endl;
}

template <typename TKey>
const char* ParallelLocalMemoryBackend<TKey>::get_name() const {
  return "ParallelLocalMemory";
}

template <typename TKey>
size_t ParallelLocalMemoryBackend<TKey>::contains(const std::string& table_name,
                                                  const size_t num_keys,
                                                  const TKey* const keys) const {
  // Locate the embedding table.
  const auto& table_it = tables_.find(table_name);
  if (table_it == tables_.end()) {
    return TBase::contains(table_name, num_keys, keys);
  }
  const std::vector<TMap>& tables = table_it->second;

  std::atomic<size_t> total_hit_count(0);

  // TODO: Optimize for special cases.

  // Issue jobs to threads.
  std::vector<ThreadPoolResult> res;
  res.reserve(num_partitions_);
  for (size_t partition = 0; partition < num_partitions_; partition++) {
    auto fn = [&, partition](size_t, size_t) {
      const TMap& table = tables[partition];
      size_t hit_count = 0;

      // Subsequently search for IDs and fill in values.
      const TKey* keys_end = &keys[num_keys];
      for (const TKey* k = keys; k != keys_end; k++) {
        if (*k % num_partitions_ == partition) {
          const auto& it = table.find(*k);
          if (it != table.end()) {
            hit_count++;
          }
        }
      }

      total_hit_count += hit_count;
    };
    res.emplace_back(ThreadPool::get().post(fn));
  }
  ThreadPool::await(res);

  DEBUG << get_name() << " backend. Fetched " << total_hit_count << " / " << num_keys << " pairs."
        << std::endl;
  return total_hit_count;
}

template <typename TKey>
bool ParallelLocalMemoryBackend<TKey>::insert(const std::string& table_name, const size_t num_pairs,
                                              const TKey* const keys, const char* const values,
                                              const size_t value_size) {
  // Lookup table, or insert it, if it does not exist yet.
  const auto& table_it = tables_.try_emplace(table_name, num_partitions_).first;
  std::vector<TMap>& tables = table_it->second;
  if (tables.size() != num_partitions_) {
    DEBUG << "Recalled table list is not of length " << num_partitions_ << "!";
    exit(EXIT_FAILURE);
  }

  std::atomic<size_t> total_num_inserts(0);

  // Issue jobs to threads.
  std::vector<ThreadPoolResult> res;
  res.reserve(num_partitions_);
  for (size_t partition = 0; partition < num_partitions_; partition++) {
    auto fn = [&, partition](size_t, size_t) {
      TMap& table = tables[partition];
      table.reserve((num_pairs + num_partitions_ - 1) / num_partitions_);

      size_t num_inserts = 0;

      // Just traverse through the KV-pairs and insert them one by one.
      const TKey* const keys_end = &keys[num_pairs];
      for (const TKey* k = keys; k != keys_end; k++) {
        if (*k % num_partitions_ == partition) {
          const char* const v = &values[(k - keys) * value_size];
          table.emplace(std::piecewise_construct, std::forward_as_tuple(*k),
                        std::forward_as_tuple(v, &v[value_size]));
          num_inserts++;
        }
      }

      total_num_inserts += num_inserts;
    };
    res.emplace_back(ThreadPool::get().post(fn));
  }
  ThreadPool::await(res);

  DEBUG << get_name() << " backend updated. Inserted " << total_num_inserts << " / " << num_pairs
        << " key-value pairs." << std::endl;
  return true;
}

template <typename TKey>
size_t ParallelLocalMemoryBackend<TKey>::fetch(const std::string& table_name, const size_t num_keys,
                                               const TKey* const keys, char* const values,
                                               const size_t value_size,
                                               MissingKeyCallback& missing_callback) const {
  // Locate the embedding table.
  const auto& table_it = tables_.find(table_name);
  if (table_it == tables_.end()) {
    return TBase::fetch(table_name, num_keys, keys, values, value_size, missing_callback);
  }
  const std::vector<TMap>& tables = table_it->second;

  std::atomic<size_t> total_hit_count(0);

  // Issue jobs to threads.
  std::vector<ThreadPoolResult> res;
  res.reserve(num_partitions_);
  for (size_t partition = 0; partition < num_partitions_; partition++) {
    auto fn = [&, partition](size_t, size_t) {
      const TMap& table = tables[partition];
      size_t hit_count = 0;

      // Subsequently search for IDs and fill in values.
      for (size_t i = 0; i < num_keys; i++) {
        const TKey& k = keys[i];
        if (k % num_partitions_ == partition) {
          const auto& it = table.find(k);
          if (it != table.end()) {
            assert(it->second.size() == value_size);
            memcpy(&values[i * value_size], it->second.data(), value_size);
            hit_count++;
          } else {
            missing_callback(i);
          }
        }
      }

      total_hit_count += hit_count;
    };
    res.emplace_back(ThreadPool::get().post(fn));
  }
  ThreadPool::await(res);

  DEBUG << get_name() << " backend. Fetched " << total_hit_count << " / " << num_keys << " pairs."
        << std::endl;
  return total_hit_count;
}

template <typename TKey>
size_t ParallelLocalMemoryBackend<TKey>::fetch(const std::string& table_name,
                                               const size_t num_indices,
                                               const size_t* const indices, const TKey* const keys,
                                               char* const values, const size_t value_size,
                                               MissingKeyCallback& missing_callback) const {
  // Locate the embedding table.
  const auto& table_it = tables_.find(table_name);
  if (table_it == tables_.end()) {
    return TBase::fetch(table_name, num_indices, indices, keys, values, value_size,
                        missing_callback);
  }
  const std::vector<TMap>& tables = table_it->second;

  std::atomic<size_t> total_hit_count(0);

  // Issue jobs to threads.
  std::vector<ThreadPoolResult> res;
  res.reserve(num_partitions_);
  for (size_t partition = 0; partition < num_partitions_; partition++) {
    auto fn = [&, partition](size_t, size_t) {
      const TMap& table = tables[partition];
      size_t hit_count = 0;

      // Subsequently search for IDs and fill in values.
      const size_t* const indices_end = &indices[num_indices];
      for (const size_t* i = indices; i != indices_end; i++) {
        const TKey& k = keys[*i];
        if (k % num_partitions_ == partition) {
          const auto& it = table.find(k);
          if (it != table.end()) {
            assert(it->second.size() == value_size);
            memcpy(&values[*i * value_size], it->second.data(), value_size);
            hit_count++;
          } else {
            missing_callback(*i);
          }
        }
      }

      total_hit_count += hit_count;
    };
    res.emplace_back(ThreadPool::get().post(fn));
  }
  ThreadPool::await(res);

  DEBUG << get_name() << " backend. Fetched " << total_hit_count << " / " << num_indices
        << " pairs." << std::endl;
  return total_hit_count;
}

template <typename TKey>
size_t ParallelLocalMemoryBackend<TKey>::evict(const std::string& table_name) {
  // Locate the embedding table.
  const auto& table_it = tables_.find(table_name);
  if (table_it == tables_.end()) {
    return 0;
  }
  size_t table_size = 0;
  for (const TMap& table : table_it->second) {
    table_size += table.size();
  }

  // Erase table.
  tables_.erase(table_name);

  return table_size;
}

template <typename TKey>
size_t ParallelLocalMemoryBackend<TKey>::evict(const std::string& table_name, const size_t num_keys,
                                               const TKey* const keys) {
  // Locate the embedding table.
  const auto& table_it = tables_.find(table_name);
  if (table_it == tables_.end()) {
    return 0;
  }
  std::vector<TMap>& tables = table_it->second;

  std::atomic<size_t> total_hit_count(0);

  // Issue jobs to threads.
  std::vector<ThreadPoolResult> res;
  res.reserve(num_partitions_);
  for (size_t partition = 0; partition < num_partitions_; partition++) {
    auto fn = [&, partition](size_t, size_t) {
      TMap& table = tables[partition];
      size_t hit_count = 0;

      // Delete items one by one.
      const TKey* const keys_end = &keys[num_keys];
      for (const TKey* k = keys; k != keys_end; k++) {
        if (*k % num_partitions_ == partition) {
          hit_count += table.erase(*k);
        }
      }

      total_hit_count += hit_count;
    };
    res.emplace_back(ThreadPool::get().post(fn));
  }
  ThreadPool::await(res);

  DEBUG << get_name() << " backend updated." << std::endl;
  return total_hit_count;
}

template class ParallelLocalMemoryBackend<unsigned int>;
template class ParallelLocalMemoryBackend<long long>;

}  // namespace HugeCTR
