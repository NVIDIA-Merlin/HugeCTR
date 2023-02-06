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
#include <core/memory.hpp>
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

struct HashMapBackendParams final : public VolatileBackendParams {
  size_t allocation_rate{256L * 1024 *
                         1024};  // Number of additional bytes to allocate per allocation cycle.
};

/**
 * \p DatabaseBackend implementation that stores key/value pairs in the local CPU memory.
 * that takes advantage of parallel processing capabilities.
 *
 * @tparam Key The data-type that is used for keys in this database.
 */
template <typename Key>
class HashMapBackend final : public VolatileBackend<Key, HashMapBackendParams> {
 public:
  using Base = VolatileBackend<Key, HashMapBackendParams>;

  HCTR_DISALLOW_COPY_AND_MOVE(HashMapBackend);

  HashMapBackend() = delete;

  /**
   * Construct a new parallelized HashMapBackend object.
   */
  HashMapBackend(const HashMapBackendParams& params);

  bool is_shared() const override final { return false; }

  const char* get_name() const override { return "HashMapBackend"; }

  size_t size(const std::string& table_name) const override;

  size_t contains(const std::string& table_name, size_t num_keys, const Key* keys,
                  const std::chrono::nanoseconds& time_budget) const override;

  size_t insert(const std::string& table_name, size_t num_pairs, const Key* keys,
                const char* values, uint32_t value_size, size_t value_stride) override;

  size_t fetch(const std::string& table_name, size_t num_keys, const Key* keys, char* values,
               size_t value_stride, const DatabaseMissCallback& on_miss,
               const std::chrono::nanoseconds& time_budget) override;

  size_t fetch(const std::string& table_name, size_t num_indices, const size_t* indices,
               const Key* keys, char* values, size_t value_stride,
               const DatabaseMissCallback& on_miss,
               const std::chrono::nanoseconds& time_budget) override;

  size_t evict(const std::string& table_name) override;

  size_t evict(const std::string& table_name, size_t num_keys, const Key* keys) override;

  std::vector<std::string> find_tables(const std::string& model_name) override;

  size_t dump_bin(const std::string& table_name, std::ofstream& file) override;

  size_t dump_sst(const std::string& table_name, rocksdb::SstFileWriter& file) override;

 protected:
#if 1
  // Better performance on most systems.
  using CharAllocator = AlignedAllocator<char>;
  static constexpr size_t value_page_alignment{CharAllocator::alignment};
#else
  using CharAllocator = std::allocator<char>;
  // __STDCPP_DEFAULT_NEW_ALIGNMENT__ is actually 16 with current GCC. But we use sizeof(char) here
  // which equates to have no padding.
  static constexpr size_t value_page_alignment{sizeof(char)};
#endif
  static_assert(value_page_alignment > 0);

  using ValuePage = std::vector<char, CharAllocator>;
  using ValuePtr = char*;

  // Data-structure that will be associated with every key.
  struct Payload final {
    union {
      time_t last_access;
      uint64_t access_count;
    };
    ValuePtr value;
  };
  using Entry = std::pair<const Key, Payload>;

  struct Partition final {
    const uint32_t value_size;
    const size_t allocation_rate;

    // Pooled payload storage.
    std::vector<ValuePage> value_pages;
    std::vector<ValuePtr> value_slots;

    // Key -> Payload map.
    phmap::flat_hash_map<Key, Payload> entries;

    Partition() = delete;

    Partition(const uint32_t value_size, const HashMapBackendParams& params)
        : value_size{value_size}, allocation_rate{params.allocation_rate} {}
  };

  // Actual data.
  CharAllocator char_allocator_;
  std::unordered_map<std::string, std::vector<Partition>> tables_;

  // Access control.
  mutable std::shared_mutex read_write_guard_;

  // Overflow resolution.
  size_t resolve_overflow_(const std::string& table_name, size_t part_index, Partition& part);
};

// TODO: Remove me!
#pragma GCC diagnostic pop

}  // namespace HugeCTR