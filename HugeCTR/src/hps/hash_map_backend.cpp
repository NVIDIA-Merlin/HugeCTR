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
#include <hps/hash_map_backend_detail.hpp>
#include <hps/hier_parameter_server_base.hpp>
#include <random>

// TODO: Remove me!
#pragma GCC diagnostic error "-Wconversion"

namespace HugeCTR {

template <typename Key>
HashMapBackend<Key>::HashMapBackend(const HashMapBackendParams& params) : Base(params) {
  HCTR_LOG_C(DEBUG, WORLD, "Created blank database backend in local memory!\n");
}

template <typename Key>
size_t HashMapBackend<Key>::size(const std::string& table_name) const {
  const std::shared_lock lock(read_write_guard_);

  // Locate the partitions.
  const auto& tables_it{tables_.find(table_name)};
  if (tables_it == tables_.end()) {
    return 0;
  }
  const std::vector<Partition>& parts{tables_it->second};

  return std::accumulate(parts.begin(), parts.end(), UINT64_C(0),
                         [](const size_t a, const Partition& b) { return a + b.entries.size(); });
}

template <typename Key>
size_t HashMapBackend<Key>::contains(const std::string& table_name, const size_t num_keys,
                                     const Key* const keys,
                                     const std::chrono::nanoseconds& time_budget) const {
  const auto begin{std::chrono::high_resolution_clock::now()};
  const std::shared_lock lock(read_write_guard_);

  // Locate partitions.
  const auto& tables_it{tables_.find(table_name)};
  if (tables_it == tables_.end()) {
    return Base::contains(table_name, num_keys, keys, time_budget);
  }
  const std::vector<Partition>& parts{tables_it->second};

  const Key* const keys_end{&keys[num_keys]};
  const size_t num_partitions{parts.size()};
  const size_t max_batch_size{this->params_.max_batch_size};

  size_t hit_count{0};
  size_t skip_count{0};

  if (num_keys == 0) {
    // Do nothing ;-).
  } else if (num_keys == 1 || num_partitions == 1) {
    const size_t part_index{num_partitions == 1 ? 0 : HCTR_HPS_KEY_TO_PART_INDEX_(*keys)};
    const Partition& part{parts[part_index]};

    // Step through keys batch-by-batch.
    std::chrono::nanoseconds elapsed;
    for (const Key* k{keys}; k != keys_end;) {
      HCTR_HPS_DB_CHECK_TIME_BUDGET_(SEQUENTIAL_DIRECT, nullptr);

      const size_t prev_hit_count{hit_count};
      const size_t batch_size{std::min<size_t>(keys_end - k, max_batch_size)};
      HCTR_HPS_HASH_MAP_CONTAINS_(SEQUENTIAL_DIRECT);

      HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                 ", batch ", (k - keys - 1) / max_batch_size, ": ", hit_count - prev_hit_count,
                 " / ", batch_size, " hits. Time: ", elapsed.count(), " / ", time_budget.count(),
                 " ns.\n");
    }
  } else {
    std::atomic<size_t> joint_hit_count{0};
    std::atomic<size_t> joint_skip_count{0};

    HCTR_HPS_DB_PARALLEL_FOR_EACH_PART_({
      const Partition& part{parts[part_index]};

      size_t hit_count{0};

      // Step through keys batch-by-batch.
      std::chrono::nanoseconds elapsed;
      size_t num_batches{0};
      for (const Key* k{keys}; k != keys_end; ++num_batches) {
        HCTR_HPS_DB_CHECK_TIME_BUDGET_(PARALLEL_DIRECT, nullptr);

        const size_t prev_hit_count{hit_count};
        size_t batch_size{0};
        HCTR_HPS_HASH_MAP_CONTAINS_(PARALLEL_DIRECT);

        HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                   ", batch ", num_batches, ": ", hit_count - prev_hit_count, " / ", batch_size,
                   " hits. Time: ", elapsed.count(), " / ", time_budget.count(), " ns.\n");
      }

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
size_t HashMapBackend<Key>::insert(const std::string& table_name, const size_t num_pairs,
                                   const Key* const keys, const char* const values,
                                   const uint32_t value_size, const size_t value_stride) {
  HCTR_CHECK(value_size <= value_stride);

  const std::unique_lock lock(read_write_guard_);

  // Locate the partitions, or create them, if they do not exist yet.
  const auto& tables_it{tables_.try_emplace(table_name).first};
  std::vector<Partition>& parts{tables_it->second};
  if (parts.empty()) {
    HCTR_CHECK(value_size > 0 && value_size <= this->params_.allocation_rate);

    parts.reserve(this->params_.num_partitions);
    while (parts.size() < this->params_.num_partitions) {
      parts.emplace_back(value_size, this->params_);
    }
  }

  const Key* const keys_end{&keys[num_pairs]};
  const size_t num_partitions{parts.size()};
  const size_t max_batch_size{this->params_.max_batch_size};
  const DatabaseOverflowPolicy_t overflow_policy{this->params_.overflow_policy};

  size_t num_inserts{0};

  if (num_pairs == 0) {
    // Do nothing ;-).
  } else if (num_pairs == 1 || num_partitions == 1) {
    const size_t part_index{num_partitions == 1 ? 0 : HCTR_HPS_KEY_TO_PART_INDEX_(*keys)};
    Partition& part{parts[part_index]};
    HCTR_CHECK(part.value_size == value_size);

    // Step through batch-by-batch.
    for (const Key* k{keys}; k != keys_end;) {
      // Check overflow condition.
      if (part.entries.size() >= this->params_.overflow_margin) {
        resolve_overflow_(table_name, part_index, part);
      }

      // Perform insertion.
      const size_t prev_num_inserts{num_inserts};
      const size_t batch_size{std::min<size_t>(keys_end - k, max_batch_size)};
      HCTR_HPS_HASH_MAP_INSERT_(SEQUENTIAL_DIRECT);

      HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                 ", batch ", (k - keys - 1) / max_batch_size, ": Inserted ",
                 num_inserts - prev_num_inserts, " + updated ",
                 batch_size - num_inserts + prev_num_inserts, " = ", batch_size, " entries.\n");
    }
  } else {
    std::atomic<size_t> joint_num_inserts{0};

    HCTR_HPS_DB_PARALLEL_FOR_EACH_PART_({
      Partition& part{parts[part_index]};
      HCTR_CHECK(part.value_size == value_size);

      size_t num_inserts{0};

      // Step through batch-by-batch.
      size_t num_batches{0};
      for (const Key* k{keys}; k != keys_end; ++num_batches) {
        // Check overflow condition.
        if (part.entries.size() >= this->params_.overflow_margin) {
          resolve_overflow_(table_name, part_index, part);
        }

        // Perform insertion.
        const size_t prev_num_inserts{num_inserts};
        size_t batch_size{0};
        HCTR_HPS_HASH_MAP_INSERT_(PARALLEL_DIRECT);

        HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                   ", batch ", num_batches, ": Inserted ", num_inserts - prev_num_inserts,
                   " + updated ", batch_size - num_inserts + prev_num_inserts, " = ", batch_size,
                   " entries.\n");
      }

      joint_num_inserts += num_inserts;
    });

    num_inserts += joint_num_inserts;
  }

  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ": Inserted ", num_inserts,
             " + updated ", num_pairs - num_inserts, " = ", num_pairs, " entries.\n");
  return num_inserts;
}

template <typename Key>
size_t HashMapBackend<Key>::fetch(const std::string& table_name, const size_t num_keys,
                                  const Key* const keys, char* const values,
                                  const size_t value_stride, const DatabaseMissCallback& on_miss,
                                  const std::chrono::nanoseconds& time_budget) {
  const auto begin{std::chrono::high_resolution_clock::now()};
  const std::shared_lock lock(read_write_guard_);

  // Locate the partitions.
  const auto& tables_it{tables_.find(table_name)};
  if (tables_it == tables_.end()) {
    return Base::fetch(table_name, num_keys, keys, values, value_stride, on_miss, time_budget);
  }
  std::vector<Partition>& parts{tables_it->second};

  const Key* const keys_end{&keys[num_keys]};
  const size_t num_partitions{parts.size()};
  const size_t max_batch_size{this->params_.max_batch_size};
  const DatabaseOverflowPolicy_t overflow_policy{this->params_.overflow_policy};

  size_t miss_count{0};
  size_t skip_count{0};

  if (num_keys == 0) {
    // Do nothing ;-).
  } else if (num_keys == 1 || num_partitions == 1) {
    const size_t part_index{num_partitions == 1 ? 0 : HCTR_HPS_KEY_TO_PART_INDEX_(*keys)};
    Partition& part{parts[part_index]};
    HCTR_CHECK(part.value_size <= value_stride);

    // Step through input batch-by-batch.
    std::chrono::nanoseconds elapsed;
    for (const Key* k{keys}; k != keys_end;) {
      HCTR_HPS_DB_CHECK_TIME_BUDGET_(SEQUENTIAL_DIRECT, on_miss);

      const size_t prev_miss_count{miss_count};
      const size_t batch_size{std::min<size_t>(keys_end - k, max_batch_size)};
      HCTR_HPS_HASH_MAP_FETCH_(SEQUENTIAL_DIRECT);

      HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                 ", batch ", (k - keys - 1) / max_batch_size, ": ",
                 batch_size - miss_count + prev_miss_count, " / ", batch_size,
                 " hits. Time: ", elapsed.count(), " / ", time_budget.count(), " ns.\n");
    }
  } else {
    std::atomic<size_t> joint_miss_count{0};
    std::atomic<size_t> joint_skip_count{0};

    HCTR_HPS_DB_PARALLEL_FOR_EACH_PART_({
      Partition& part{parts[part_index]};
      HCTR_CHECK(part.value_size <= value_stride);

      size_t miss_count{0};

      // Step through input batch-by-batch.
      std::chrono::nanoseconds elapsed;
      size_t num_batches{0};
      for (const Key* k{keys}; k != keys_end; ++num_batches) {
        HCTR_HPS_DB_CHECK_TIME_BUDGET_(PARALLEL_DIRECT, on_miss);

        const size_t prev_miss_count{miss_count};
        size_t batch_size{0};
        HCTR_HPS_HASH_MAP_FETCH_(PARALLEL_DIRECT);

        HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                   ", batch ", num_batches, ": ", batch_size - miss_count + prev_miss_count, " / ",
                   batch_size, " hits. Time: ", elapsed.count(), " / ", time_budget.count(),
                   " ns.\n");
      }

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
size_t HashMapBackend<Key>::fetch(const std::string& table_name, const size_t num_indices,
                                  const size_t* const indices, const Key* const keys,
                                  char* const values, const size_t value_stride,
                                  const DatabaseMissCallback& on_miss,
                                  const std::chrono::nanoseconds& time_budget) {
  const auto begin{std::chrono::high_resolution_clock::now()};
  const std::shared_lock lock(read_write_guard_);

  // Locate the partitions.
  const auto& tables_it{tables_.find(table_name)};
  if (tables_it == tables_.end()) {
    return Base::fetch(table_name, num_indices, indices, keys, values, value_stride, on_miss,
                       time_budget);
  }
  std::vector<Partition>& parts{tables_it->second};

  const size_t* const indices_end{&indices[num_indices]};
  const size_t num_partitions{parts.size()};
  const size_t max_batch_size{this->params_.max_batch_size};
  const DatabaseOverflowPolicy_t overflow_policy{this->params_.overflow_policy};

  size_t miss_count{0};
  size_t skip_count{0};

  if (num_indices == 0) {
    // Do nothing ;-).
  } else if (num_indices == 1 || num_partitions == 1) {
    const size_t part_index{num_partitions == 1 ? 0 : HCTR_HPS_KEY_TO_PART_INDEX_(*keys)};
    Partition& part{parts[part_index]};
    HCTR_CHECK(part.value_size <= value_stride);

    // Step through input batch-by-batch.
    std::chrono::nanoseconds elapsed;
    for (const size_t* i{indices}; i != indices_end;) {
      HCTR_HPS_DB_CHECK_TIME_BUDGET_(SEQUENTIAL_INDIRECT, on_miss);

      const size_t prev_miss_count{miss_count};
      const size_t batch_size{std::min<size_t>(indices_end - i, max_batch_size)};
      HCTR_HPS_HASH_MAP_FETCH_(SEQUENTIAL_INDIRECT);

      HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                 ", batch ", (i - indices - 1) / max_batch_size, ": ",
                 batch_size - miss_count + prev_miss_count, " / ", batch_size,
                 " hits. Time: ", elapsed.count(), " / ", time_budget.count(), " ns.\n");
    }
  } else {
    std::atomic<size_t> joint_miss_count{0};
    std::atomic<size_t> joint_skip_count{0};

    HCTR_HPS_DB_PARALLEL_FOR_EACH_PART_({
      Partition& part{parts[part_index]};
      HCTR_CHECK(part.value_size <= value_stride);

      size_t miss_count{0};

      // Step through input batch-by-batch.
      std::chrono::nanoseconds elapsed;
      size_t num_batches{0};
      for (const size_t* i{indices}; i != indices_end; ++num_batches) {
        HCTR_HPS_DB_CHECK_TIME_BUDGET_(PARALLEL_INDIRECT, on_miss);

        const size_t prev_miss_count{miss_count};
        size_t batch_size{0};
        HCTR_HPS_HASH_MAP_FETCH_(PARALLEL_INDIRECT);

        HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                   ", batch ", num_batches, ": ", batch_size - miss_count + prev_miss_count, " / ",
                   batch_size, " hits. Time: ", elapsed.count(), " / ", time_budget.count(),
                   " ns.\n");
      }

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
size_t HashMapBackend<Key>::evict(const std::string& table_name) {
  const std::unique_lock lock(read_write_guard_);

  // Locate the partitions.
  const auto& tables_it{tables_.find(table_name)};
  if (tables_it == tables_.end()) {
    return 0;
  }
  const std::vector<Partition>& parts{tables_it->second};

  // Count items and erase.
  size_t num_deletions{0};
  for (const Partition& part : parts) {
    num_deletions += part.entries.size();
  }
  tables_.erase(tables_it);

  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ": Erased ", num_deletions,
             " entries.\n");
  return num_deletions;
}

template <typename Key>
size_t HashMapBackend<Key>::evict(const std::string& table_name, const size_t num_keys,
                                  const Key* const keys) {
  const std::unique_lock lock(read_write_guard_);

  // Locate the partitions.
  const auto& tables_it{tables_.find(table_name)};
  if (tables_it == tables_.end()) {
    return 0;
  }
  std::vector<Partition>& parts{tables_it->second};

  const Key* const keys_end{&keys[num_keys]};
  const size_t num_partitions{parts.size()};
  const size_t max_batch_size{this->params_.max_batch_size};

  size_t num_deletions{0};

  if (num_keys == 0) {
    // Do nothing ;-).
  } else if (num_keys == 1 || num_partitions == 1) {
    const size_t part_index{num_partitions == 1 ? 0 : HCTR_HPS_KEY_TO_PART_INDEX_(*keys)};
    Partition& part{parts[part_index]};

    // Step through input batch-by-batch.
    for (const Key* k{keys}; k != keys_end;) {
      const size_t batch_size{std::min<size_t>(keys_end - k, max_batch_size)};
      const size_t prev_num_deletions{num_deletions};
      HCTR_HPS_HASH_MAP_EVICT_(SEQUENTIAL_DIRECT);

      HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                 ", batch ", (k - keys - 1) / max_batch_size, ": Erased ",
                 num_deletions - prev_num_deletions, " entries.\n");
    }
  } else {
    std::atomic<size_t> joint_num_deletions{0};

    HCTR_HPS_DB_PARALLEL_FOR_EACH_PART_({
      Partition& part{parts[part_index]};

      size_t num_deletions{0};

      // Step through input batch-by-batch.
      size_t num_batches{0};
      for (const Key* k{keys}; k != keys_end; ++num_batches) {
        const size_t prev_num_deletions{num_deletions};
        size_t batch_size{0};
        HCTR_HPS_HASH_MAP_EVICT_(PARALLEL_DIRECT);

        HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                   ", batch ", num_batches, ": Erased ", num_deletions - prev_num_deletions, " / ",
                   batch_size, " entries.\n");
      }

      joint_num_deletions += num_deletions;
    });

    num_deletions += joint_num_deletions;
  }

  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ": Erased ", num_deletions,
             " / ", num_keys, " entries.\n");
  return num_deletions;
}

template <typename Key>
std::vector<std::string> HashMapBackend<Key>::find_tables(const std::string& model_name) {
  const std::string& tag_prefix{HierParameterServerBase::make_tag_name(model_name, "", false)};

  const std::shared_lock lock(read_write_guard_);

  std::vector<std::string> matches;
  for (const auto& pair : tables_) {
    if (pair.first.find(tag_prefix) == 0) {
      matches.push_back(pair.first);
    }
  }
  return matches;
}

template <typename Key>
size_t HashMapBackend<Key>::dump_bin(const std::string& table_name, std::ofstream& file) {
  const std::shared_lock lock(read_write_guard_);

  // Locate the partitions.
  const auto& tables_it{tables_.find(table_name)};
  if (tables_it == tables_.end()) {
    return 0;
  }
  const std::vector<Partition>& parts{tables_it->second};

  // Store value size.
  const uint32_t value_size{parts.empty() ? 0 : parts.front().value_size};
  file.write(reinterpret_cast<const char*>(&value_size), sizeof(uint32_t));

  // Store values.
  size_t num_entries{0};

  for (const Partition& part : parts) {
    for (const Entry& entry : part.entries) {
      file.write(reinterpret_cast<const char*>(&entry.first), sizeof(Key));
      file.write(entry.second.value, value_size);
    }
    num_entries += part.entries.size();
  }

  return num_entries;
}

template <typename Key>
size_t HashMapBackend<Key>::dump_sst(const std::string& table_name, rocksdb::SstFileWriter& file) {
  const std::shared_lock lock(read_write_guard_);

  // Locate the partitions.
  const auto& tables_it{tables_.find(table_name)};
  if (tables_it == tables_.end()) {
    return 0;
  }
  const std::vector<Partition>& parts{tables_it->second};

  // Sort keys by value.
  std::vector<const Entry*> entries;
  entries.reserve(
      std::accumulate(parts.begin(), parts.end(), UINT64_C(0),
                      [](const size_t a, const Partition& b) { return a + b.entries.size(); }));
  for (const Partition& part : parts) {
    for (const Entry& entry : part.entries) {
      entries.emplace_back(&entry);
    }
  }
  // TODO: Copy or ref? Chose ref because low memory footprint, but has worse cache locality.
  // Benchmark?
  std::sort(entries.begin(), entries.end(),
            [](const auto& a, const auto& b) { return a->first < b->first; });

  // Iterate over pairs and insert.
  rocksdb::Slice k_view{nullptr, sizeof(Key)};
  rocksdb::Slice v_view{nullptr, parts.empty() ? 0 : parts.front().value_size};

  for (const Entry* const entry : entries) {
    k_view.data_ = reinterpret_cast<const char*>(&entry->first);
    v_view.data_ = entry->second.value;
    HCTR_ROCKSDB_CHECK(file.Put(k_view, v_view));
  }

  return entries.size();
}

template <typename Key>
size_t HashMapBackend<Key>::resolve_overflow_(const std::string& table_name,
                                              const size_t part_index, Partition& part) {
  const size_t max_batch_size{this->params_.max_batch_size};

  size_t num_deletions{0};

  switch (this->params_.overflow_policy) {
    case DatabaseOverflowPolicy_t::EvictRandom: {
      // Fetch all keys.
      std::vector<Key> keys;
      keys.reserve(part.entries.size());
      for (const auto& pair : part.entries) {
        keys.emplace_back(pair.first);
      }

      // Shuffle the keys.
      {
        // TODO: This randomizer shoud fetch its seed from a central source.
        std::random_device rd;
        std::default_random_engine gen(rd());
        std::shuffle(keys.begin(), keys.end(), gen);
      }

      // Delete items.
      for (auto k_it{keys.begin()}; k_it != keys.end();) {
        const size_t batch_size{std::min<size_t>(keys.end() - k_it, max_batch_size)};

        HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                   " is overflowing (size = ", part.entries.size(), " > ",
                   this->params_.overflow_margin, "): Attempting to evict ", batch_size,
                   " RANDOM key/value pairs!\n");

        // Call erase, until we reached the target amount.
        for (const auto& batch_end{k_it + batch_size}; k_it != batch_end; ++k_it) {
          const Key* const k{&*k_it};
          HCTR_HPS_HASH_MAP_EVICT_K_();
        }
        if (part.entries.size() <= this->overflow_resolution_margin_) {
          break;
        }
      }
    } break;

    case DatabaseOverflowPolicy_t::EvictLeastUsed: {
      // Fetch keys and access counts.
      std::vector<std::pair<Key, uint64_t>> keys_metas;
      keys_metas.reserve(part.entries.size());
      for (const auto& entry : part.entries) {
        keys_metas.emplace_back(entry.first, entry.second.access_count);
      }

      // Sort by ascending by number of accesses.
      std::sort(keys_metas.begin(), keys_metas.end(),
                [](const auto& km0, const auto& km1) { return km0.second < km1.second; });

      // Call erase, until we reached the target amount.
      auto km_it = keys_metas.begin();
      while (km_it != keys_metas.end()) {
        const size_t batch_size{std::min<size_t>(keys_metas.end() - km_it, max_batch_size)};

        HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                   " is overflowing (size = ", part.entries.size(), " > ",
                   this->params_.overflow_margin, "): Attempting to evict ", batch_size,
                   " LEAST USED key/value pairs!\n");

        for (const auto& batch_end{km_it + batch_size}; km_it != batch_end; ++km_it) {
          const Key* const k{&km_it->first};
          HCTR_HPS_HASH_MAP_EVICT_K_();
        }
        if (part.entries.size() <= this->overflow_resolution_margin_) {
          break;
        }
      }

      // To avoid exploding LFU numbers that would prevent any new values from entering a steady
      // state, we do a normalization step on all existing values.
      //
      // 1) Since the conditions are again met, the next `km_it` value (if exists) points to the min
      // count. We subtract that count to `0` normalize the access counts. This means, that assuming
      // there is no fetch, the next value inserted has exactly the same chance to be evicted as the
      // least used existing value.
      //
      // 2) If the distribution changes, some values may become less popular. To avoid that a once
      // high value of a new less desired value prevents its eviction, we cut down all existing
      // access counts by dividing them by 2.
      //
      if (km_it != keys_metas.end()) {
        const uint64_t min_access_count{km_it->second};
        for (; km_it != keys_metas.end(); ++km_it) {
          km_it->second = (km_it->second - min_access_count) / 2;
        }
      }
    } break;

    case DatabaseOverflowPolicy_t::EvictOldest: {
      // Fetch keys and insert times.
      std::vector<std::pair<Key, time_t>> keys_metas;
      keys_metas.reserve(part.entries.size());
      for (const auto& entry : part.entries) {
        keys_metas.emplace_back(entry.first, entry.second.last_access);
      }

      // Sort by ascending by time.
      std::sort(keys_metas.begin(), keys_metas.end(),
                [](const auto& km0, const auto& km1) { return km0.second < km1.second; });

      // Call erase, until we reached the target amount.
      for (auto km_it{keys_metas.begin()}; km_it != keys_metas.end();) {
        const size_t batch_size{std::min<size_t>(keys_metas.end() - km_it, max_batch_size)};

        HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, '/', part_index,
                   " is overflowing (size = ", part.entries.size(), " > ",
                   this->params_.overflow_margin, "): Attempting to evict ", batch_size,
                   " OLDEST key/value pairs!\n");

        for (const auto& batch_end{km_it + batch_size}; km_it != batch_end; ++km_it) {
          const Key* const k{&km_it->first};
          HCTR_HPS_HASH_MAP_EVICT_K_();
        }
        if (part.entries.size() <= this->overflow_resolution_margin_) {
          break;
        }
      }
    } break;
  }

  return num_deletions;
}

template class HashMapBackend<unsigned int>;
template class HashMapBackend<long long>;

}  // namespace HugeCTR
