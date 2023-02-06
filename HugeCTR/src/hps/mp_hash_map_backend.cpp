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
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <boost/utility/string_view.hpp>
#include <hps/hash_map_backend_detail.hpp>
#include <hps/hier_parameter_server_base.hpp>
#include <hps/mp_hash_map_backend.hpp>
#include <random>

// TODO: Remove me!
#pragma GCC diagnostic error "-Wconversion"

namespace HugeCTR {

template <typename Key>
MultiProcessHashMapBackend<Key>::MultiProcessHashMapBackend(
    const MultiProcessHashMapBackendParams& params)
    : Base(params),
      sm_segment_(boost::interprocess::open_or_create, params.shared_memory_name.c_str(),
                  params.shared_memory_size),
      char_allocator_{sm_segment_.get_allocator<char>()},
      value_page_allocator_{sm_segment_.get_allocator<ValuePage>()},
      partition_allocator_{sm_segment_.get_allocator<Partition>()},
      sm_{sm_segment_.find_or_construct<SharedMemory>("sm")(params.heart_beat_frequency,
                                                            params.auto_remove, sm_segment_)} {
  HCTR_CHECK(sm_);
  HCTR_CHECK(sm_->heart_beat_frequency == params.heart_beat_frequency);
  HCTR_CHECK(sm_->auto_remove == params.auto_remove);

  HCTR_LOG_S(INFO, WORLD) << "Connecting to shared memory '" << params.shared_memory_name << "'..."
                          << std::endl;

  // Ensure exclusive access.
  const boost::interprocess::scoped_lock lock(sm_->read_write_guard);

  // Sanity checks.
  const std::filesystem::space_info& si = std::filesystem::space("/dev/shm");
  HCTR_LOG_S(INFO, WORLD) << "Connected to shared memory '" << params.shared_memory_name
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

  if (si.free < params.allocation_rate) {
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
  const uint64_t old_heart_beat{sm_->heart_beat};
  std::this_thread::sleep_for(sm_->heart_beat_frequency * 5);
  return sm_->heart_beat != old_heart_beat;
}

template <typename Key>
MultiProcessHashMapBackend<Key>::~MultiProcessHashMapBackend() {
  HCTR_LOG_S(INFO, WORLD) << "Disconnecting from shared memory '"
                          << this->params_.shared_memory_name << "'." << std::endl;

  // Ensure exclusive access.
  const boost::interprocess::scoped_lock lock(sm_->read_write_guard);

  // Stop heart.
  heart_stop_signal_ = true;
  heart_.join();

  // Destroy SHM, if this was the last process and auto_remove is enabled.
  if (sm_->auto_remove && !is_process_connected_()) {
    HCTR_LOG_S(INFO, WORLD) << "Detached last process from shared memory '"
                            << this->params_.shared_memory_name << "'. Auto remove in progress..."
                            << std::endl;
    boost::interprocess::shared_memory_object::remove(this->params_.shared_memory_name.c_str());
  }
}

template <typename Key>
size_t MultiProcessHashMapBackend<Key>::size(const std::string& table_name) const {
  const boost::interprocess::sharable_lock lock(sm_->read_write_guard);

  // Locate the partitions.
  const auto& tables_it{sm_->tables.find({table_name.c_str(), char_allocator_})};
  if (tables_it == sm_->tables.end()) {
    return 0;
  }
  const SharedVector<Partition>& parts{tables_it->second};

  return std::accumulate(parts.begin(), parts.end(), UINT64_C(0),
                         [](const size_t a, const Partition& b) { return a + b.entries.size(); });
}

template <typename Key>
size_t MultiProcessHashMapBackend<Key>::contains(
    const std::string& table_name, const size_t num_keys, const Key* const keys,
    const std::chrono::nanoseconds& time_budget) const {
  const auto begin{std::chrono::high_resolution_clock::now()};
  const boost::interprocess::sharable_lock lock(sm_->read_write_guard);

  // Locate partitions.
  const auto& tables_it{sm_->tables.find({table_name.c_str(), char_allocator_})};
  if (tables_it == sm_->tables.end()) {
    return Base::contains(table_name, num_keys, keys, time_budget);
  }
  const SharedVector<Partition>& parts{tables_it->second};

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
size_t MultiProcessHashMapBackend<Key>::insert(const std::string& table_name,
                                               const size_t num_pairs, const Key* const keys,
                                               const char* const values, const uint32_t value_size,
                                               const size_t value_stride) {
  HCTR_CHECK(value_size <= value_stride);

  const boost::interprocess::scoped_lock lock(sm_->read_write_guard);

  // Locate the partitions, or create them, if they do not exist yet.
  const auto& tables_it{
      sm_->tables.try_emplace({table_name.c_str(), char_allocator_}, partition_allocator_).first};
  SharedVector<Partition>& parts{tables_it->second};
  if (parts.empty()) {
    HCTR_CHECK(value_size > 0 && value_size <= this->params_.allocation_rate);

    parts.reserve(this->params_.num_partitions);
    while (parts.size() < this->params_.num_partitions) {
      parts.emplace_back(value_size, this->params_, sm_segment_);
    }
  }

  const Key* const keys_end{&keys[num_pairs]};
  const size_t num_partitions{parts.size()};
  const size_t max_batch_size{this->params_.max_batch_size};

  size_t num_inserts{0};

  if (num_pairs == 0) {
    // Do nothing ;-).
  } else if (num_pairs == 1 || num_partitions == 1) {
    const size_t part_index{num_partitions == 1 ? 0 : HCTR_HPS_KEY_TO_PART_INDEX_(*keys)};
    Partition& part{parts[part_index]};
    HCTR_CHECK(part.value_size == value_size);
    const DatabaseOverflowPolicy_t overflow_policy{part.overflow_policy};

    // Step through batch-by-batch.
    for (const Key* k{keys}; k != keys_end;) {
      // Check overflow condition.
      if (part.entries.size() >= part.overflow_margin) {
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
      const DatabaseOverflowPolicy_t overflow_policy{part.overflow_policy};

      size_t num_inserts{0};

      // Step through batch-by-batch.
      size_t num_batches{0};
      for (const Key* k{keys}; k != keys_end; ++num_batches) {
        // Check overflow condition.
        if (part.entries.size() >= part.overflow_margin) {
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
             " + updated ", num_pairs - num_inserts, " = ", num_pairs,
             " entries; free SM = ", sm_segment_.get_free_memory(), " bytes.\n");
  return num_inserts;
}

template <typename Key>
size_t MultiProcessHashMapBackend<Key>::fetch(const std::string& table_name, const size_t num_keys,
                                              const Key* const keys, char* const values,
                                              const size_t value_stride,
                                              const DatabaseMissCallback& on_miss,
                                              const std::chrono::nanoseconds& time_budget) {
  const auto begin{std::chrono::high_resolution_clock::now()};
  const boost::interprocess::sharable_lock lock(sm_->read_write_guard);

  // Locate the partitions.
  const auto& tables_it{sm_->tables.find({table_name.c_str(), char_allocator_})};
  if (tables_it == sm_->tables.end()) {
    return Base::fetch(table_name, num_keys, keys, values, value_stride, on_miss, time_budget);
  }
  SharedVector<Partition>& parts{tables_it->second};

  const Key* const keys_end{&keys[num_keys]};
  const size_t num_partitions{parts.size()};
  const size_t max_batch_size{this->params_.max_batch_size};

  size_t miss_count{0};
  size_t skip_count{0};

  if (num_keys == 0) {
    // Do nothing ;-).
  } else if (num_keys == 1 || num_partitions == 1) {
    const size_t part_index{num_partitions == 1 ? 0 : HCTR_HPS_KEY_TO_PART_INDEX_(*keys)};
    Partition& part{parts[part_index]};
    HCTR_CHECK(part.value_size <= value_stride);
    const DatabaseOverflowPolicy_t overflow_policy{part.overflow_policy};

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
      const DatabaseOverflowPolicy_t overflow_policy{part.overflow_policy};

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
size_t MultiProcessHashMapBackend<Key>::fetch(const std::string& table_name,
                                              const size_t num_indices, const size_t* const indices,
                                              const Key* const keys, char* const values,
                                              const size_t value_stride,
                                              const DatabaseMissCallback& on_miss,
                                              const std::chrono::nanoseconds& time_budget) {
  const auto begin{std::chrono::high_resolution_clock::now()};
  const boost::interprocess::sharable_lock lock(sm_->read_write_guard);

  // Locate the partitions.
  const auto& tables_it{sm_->tables.find({table_name.c_str(), char_allocator_})};
  if (tables_it == sm_->tables.end()) {
    return Base::fetch(table_name, num_indices, indices, keys, values, value_stride, on_miss,
                       time_budget);
  }
  SharedVector<Partition>& parts{tables_it->second};

  const size_t* const indices_end{&indices[num_indices]};
  const size_t num_partitions{parts.size()};
  const size_t max_batch_size{this->params_.max_batch_size};

  size_t miss_count{0};
  size_t skip_count{0};

  if (num_indices == 0) {
    // Do nothing ;-).
  } else if (num_indices == 1 || num_partitions == 1) {
    const size_t part_index{num_partitions == 1 ? 0 : HCTR_HPS_KEY_TO_PART_INDEX_(*keys)};
    Partition& part{parts[part_index]};
    HCTR_CHECK(part.value_size <= value_stride);
    const DatabaseOverflowPolicy_t overflow_policy{part.overflow_policy};

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
      const DatabaseOverflowPolicy_t overflow_policy{part.overflow_policy};

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
size_t MultiProcessHashMapBackend<Key>::evict(const std::string& table_name) {
  const boost::interprocess::scoped_lock lock(sm_->read_write_guard);

  // Locate the partitions.
  const auto& tables_it{sm_->tables.find({table_name.c_str(), char_allocator_})};
  if (tables_it == sm_->tables.end()) {
    return 0;
  }
  const SharedVector<Partition>& parts{tables_it->second};

  // Count items and erase.
  size_t num_deletions{0};
  for (const Partition& part : parts) {
    num_deletions += part.entries.size();
  }
  sm_->tables.erase(tables_it);

  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ": Erased ", num_deletions,
             " entries.\n");
  return num_deletions;
}

template <typename Key>
size_t MultiProcessHashMapBackend<Key>::evict(const std::string& table_name, const size_t num_keys,
                                              const Key* const keys) {
  const boost::interprocess::scoped_lock lock(sm_->read_write_guard);

  // Locate the partitions.
  const auto& tables_it{sm_->tables.find({table_name.c_str(), char_allocator_})};
  if (tables_it == sm_->tables.end()) {
    return 0;
  }
  SharedVector<Partition>& parts{tables_it->second};

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
      const size_t prev_num_deletions{num_deletions};
      const size_t batch_size{std::min<size_t>(keys_end - k, max_batch_size)};
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
std::vector<std::string> MultiProcessHashMapBackend<Key>::find_tables(
    const std::string& model_name) {
  const std::string& tag_prefix{HierParameterServerBase::make_tag_name(model_name, "", false)};

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
size_t MultiProcessHashMapBackend<Key>::dump_bin(const std::string& table_name,
                                                 std::ofstream& file) {
  const boost::interprocess::sharable_lock lock(sm_->read_write_guard);

  // Locate the partitions.
  const auto& tables_it{sm_->tables.find({table_name.c_str(), char_allocator_})};
  if (tables_it == sm_->tables.end()) {
    return 0;
  }
  const SharedVector<Partition>& parts{tables_it->second};

  // Store value size.
  const uint32_t value_size{parts.empty() ? 0 : parts.front().value_size};
  file.write(reinterpret_cast<const char*>(&value_size), sizeof(uint32_t));

  // Store values.
  size_t num_entries{0};

  for (const Partition& part : parts) {
    for (const Entry& entry : part.entries) {
      file.write(reinterpret_cast<const char*>(&entry.first), sizeof(Key));
      file.write(entry.second.value.get(), value_size);
    }
    num_entries += part.entries.size();
  }

  return num_entries;
}

template <typename Key>
size_t MultiProcessHashMapBackend<Key>::dump_sst(const std::string& table_name,
                                                 rocksdb::SstFileWriter& file) {
  const boost::interprocess::sharable_lock lock(sm_->read_write_guard);

  // Locate the partitions.
  const auto& tables_it{sm_->tables.find({table_name.c_str(), char_allocator_})};
  if (tables_it == sm_->tables.end()) {
    return 0;
  }
  const SharedVector<Partition>& parts{tables_it->second};

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
    v_view.data_ = entry->second.value.get();
    HCTR_ROCKSDB_CHECK(file.Put(k_view, v_view));
  }

  return entries.size();
}

template <typename Key>
size_t MultiProcessHashMapBackend<Key>::resolve_overflow_(const std::string& table_name,
                                                          const size_t part_index,
                                                          Partition& part) {
  const size_t max_batch_size{this->params_.max_batch_size};

  size_t num_deletions{0};

  switch (part.overflow_policy) {
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
                   " is overflowing (size = ", part.entries.size(), " > ", part.overflow_margin,
                   "): Attempting to evict ", batch_size, " RANDOM key/value pairs!\n");

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
                   " is overflowing (size = ", part.entries.size(), " > ", part.overflow_margin,
                   "): Attempting to evict ", batch_size, " LEAST USED key/value pairs!\n");

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
                   " is overflowing (size = ", part.entries.size(), " > ", part.overflow_margin,
                   "): Attempting to evict ", batch_size, " OLDEST key/value pairs!\n");

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

template class MultiProcessHashMapBackend<unsigned int>;
template class MultiProcessHashMapBackend<long long>;

}  // namespace HugeCTR
