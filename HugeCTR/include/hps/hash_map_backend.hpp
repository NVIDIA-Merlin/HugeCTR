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
#pragma once

#include <parallel_hashmap/phmap.h>

#include <condition_variable>
#include <deque>
#include <functional>
#include <hps/database_backend.hpp>
#include <shared_mutex>
#include <thread>
#include <thread_pool.hpp>
#include <unordered_map>
#include <vector>

namespace HugeCTR {

// TODO: Remove me!
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wconversion"

/**
 * \p DatabaseBackend implementation that stores key/value pairs in the local CPU memory.
 * that takes advantage of parallel processing capabilities.
 *
 * @tparam Key The data-type that is used for keys in this database.
 */
template <typename Key>
class HashMapBackend final : public VolatileBackend<Key> {
 public:
  using Base = VolatileBackend<Key>;

  HashMapBackend() = delete;
  DISALLOW_COPY_AND_MOVE(HashMapBackend);

  /**
   * Construct a new parallelized HashMapBackend object.
   * @param num_partitions The number of parallel partitions.
   * @param allocation_rate Number of additional bytes to allocate per allocation.
   * @param overflow_margin Margin at which further inserts will trigger overflow handling.
   * @param overflow_policy Policy to use in case an overflow has been detected.
   * @param overflow_resolution_target Target margin after applying overflow handling policy.
   */
  HashMapBackend(size_t num_partitions = 16, size_t allocation_rate = 256 * 1024 * 1024,
                 size_t max_get_batch_size = 10'000, size_t max_set_batch_size = 10'000,
                 size_t overflow_margin = std::numeric_limits<size_t>::max(),
                 DatabaseOverflowPolicy_t overflow_policy = DatabaseOverflowPolicy_t::EvictOldest,
                 double overflow_resolution_target = 0.8);

  bool is_shared() const override final { return false; }

  const char* get_name() const override { return "HashMapBackend"; }

  size_t capacity(const std::string& table_name) const override {
    const size_t part_cap = this->overflow_margin_;
    const size_t total_cap = part_cap * num_partitions_;
    return (total_cap > part_cap) ? total_cap : part_cap;
  }

  size_t size(const std::string& table_name) const override;

  size_t contains(const std::string& table_name, size_t num_keys, const Key* keys,
                  const std::chrono::nanoseconds& time_budget) const override;

  bool insert(const std::string& table_name, size_t num_pairs, const Key* keys, const char* values,
              size_t value_size) override;

  size_t fetch(const std::string& table_name, size_t num_keys, const Key* keys,
               const DatabaseHitCallback& on_hit, const DatabaseMissCallback& on_miss,
               const std::chrono::nanoseconds& time_budget) override;

  size_t fetch(const std::string& table_name, size_t num_indices, const size_t* indices,
               const Key* keys, const DatabaseHitCallback& on_hit,
               const DatabaseMissCallback& on_miss,
               const std::chrono::nanoseconds& time_budget) override;

  size_t evict(const std::string& table_name) override;

  size_t evict(const std::string& table_name, size_t num_keys, const Key* keys) override;

  std::vector<std::string> find_tables(const std::string& model_name) override;

  void dump_bin(const std::string& table_name, std::ofstream& file) override;

  void dump_sst(const std::string& table_name, rocksdb::SstFileWriter& file) override;

 protected:
  // Key metrics.
  static constexpr size_t key_size = sizeof(Key);

  // Data-structure that will be associated with every key.
  struct Payload final {
    time_t last_access;
    char value[1];
  };
  static constexpr size_t meta_size = sizeof(Payload) - sizeof(char[1]);
  static_assert(meta_size >= sizeof(time_t));
  using PayloadPtr = Payload*;
  using Entry = std::pair<const Key, PayloadPtr>;

  using Page = std::vector<char>;

  struct Partition final {
    const size_t index;
    const uint32_t value_size;

    // Pooled payload storage.
    std::vector<Page> payload_pages;
    std::vector<PayloadPtr> payload_slots;

    // Key -> Payload map.
    phmap::flat_hash_map<Key, PayloadPtr> entries;

    Partition() = delete;
    Partition(const size_t index, const uint32_t value_size)
        : index{index}, value_size{value_size} {}
  };

  const size_t num_partitions_;
  const size_t allocation_rate_;

  // Actual data.
  std::unordered_map<std::string, std::vector<Partition>> tables_;

  // Access control.
  mutable std::shared_mutex read_write_guard_;

  // Overflow resolution.
  size_t resolve_overflow_(const std::string& table_name, Partition& part);
};

// TODO: Remove me!
#pragma GCC diagnostic pop

}  // namespace HugeCTR