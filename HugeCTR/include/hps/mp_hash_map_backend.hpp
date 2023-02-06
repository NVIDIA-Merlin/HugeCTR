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
#pragma once

#include <boost/interprocess/containers/flat_map.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <boost/unordered_map.hpp>
#include <core/macro.hpp>
#include <hps/database_backend.hpp>

namespace HugeCTR {

// TODO: Remove me!
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wconversion"

struct MultiProcessHashMapBackendParams final : public VolatileBackendParams {
  size_t allocation_rate{256L * 1024 *
                         1024};  // Number of additional bytes to allocate per allocation cycle.
  size_t shared_memory_size{16L * 1024 * 1024 *
                            1024};  // Total amount of shared memory to reserve on startup.
  std::string shared_memory_name{
      "hctr_mp_hash_map_database"};  // Name of the shared memory to which we connect.
  std::chrono::nanoseconds heart_beat_frequency{std::chrono::milliseconds{
      100}};               // Frequency at which we tick up the heart-beat frequency counter.
  bool auto_remove{true};  // Remove SHM if this is the last process to detach from the SHM.
};

template <typename Key>
class MultiProcessHashMapBackend final
    : public VolatileBackend<Key, MultiProcessHashMapBackendParams> {
 public:
  using Base = VolatileBackend<Key, MultiProcessHashMapBackendParams>;

  HCTR_DISALLOW_COPY_AND_MOVE(MultiProcessHashMapBackend);

  MultiProcessHashMapBackend() = delete;

  MultiProcessHashMapBackend(const MultiProcessHashMapBackendParams& params);

  virtual ~MultiProcessHashMapBackend();

  bool is_shared() const override { return true; }

  const char* get_name() const override { return "MultiProcessHashMapBackend"; }

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
  using Segment = boost::interprocess::managed_shared_memory;
  template <typename T>
  using SegmentAllocator = boost::interprocess::allocator<T, Segment::segment_manager>;

  using SharedString =
      boost::interprocess::basic_string<char, std::char_traits<char>, SegmentAllocator<char>>;
  template <typename T>
  using SharedVector = boost::interprocess::vector<T, SegmentAllocator<T>>;

  template <typename K, typename V>
  using SharedMap = boost::unordered_map<K, V, boost::hash<K>, std::equal_to<K>,
                                         SegmentAllocator<std::pair<const K, V>>>;

  template <typename K, typename V>
  using SharedFlatMap =
      boost::interprocess::flat_map<K, V, std::less<K>, SegmentAllocator<std::pair<const K, V>>>;

 protected:
  static constexpr size_t value_page_alignment{1};

  using ValuePage = SharedVector<char>;
  using ValuePtr = boost::interprocess::offset_ptr<char>;

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
    uint32_t value_size;
    size_t allocation_rate;
    size_t overflow_margin;
    DatabaseOverflowPolicy_t overflow_policy;
    double overflow_resolution_target;

    // Pooled payload storage.
    SharedVector<ValuePage> value_pages;
    SharedVector<ValuePtr> value_slots;

    // Key -> Payload map.
    SharedFlatMap<Key, Payload> entries;

    Partition() = delete;

    Partition(const uint32_t value_size, const MultiProcessHashMapBackendParams& params,
              Segment& segment)
        : value_size{value_size},
          allocation_rate{params.allocation_rate},
          overflow_margin{params.overflow_margin},
          overflow_policy{params.overflow_policy},
          overflow_resolution_target{params.overflow_resolution_target},
          value_pages(segment.get_allocator<ValuePage>()),
          value_slots(segment.get_allocator<ValuePtr>()),
          entries(segment.get_allocator<Entry>()) {}
  };

  struct SharedMemory final {
    const std::chrono::nanoseconds heart_beat_frequency;
    const bool auto_remove;
    volatile std::atomic<uint64_t> heart_beat;

    // Access control.
    boost::interprocess::interprocess_sharable_mutex read_write_guard;

    // Actual data.
    SharedMap<SharedString, SharedVector<Partition>> tables;

    HCTR_DISALLOW_COPY_AND_MOVE(SharedMemory);

    SharedMemory() = delete;

    SharedMemory(const std::chrono::nanoseconds& heart_beat_frequency, const bool& auto_remove,
                 Segment& segment)
        : heart_beat_frequency{heart_beat_frequency},
          auto_remove{auto_remove},
          heart_beat{0},
          tables(segment.get_allocator<std::pair<const SharedString, SharedVector<Partition>>>()) {}
  };

  Segment sm_segment_;
  SegmentAllocator<char> char_allocator_;
  SegmentAllocator<ValuePage> value_page_allocator_;
  SegmentAllocator<Partition> partition_allocator_;
  SharedMemory* sm_;

  // Heart beat system.
  bool heart_stop_signal_ = false;
  std::thread heart_;
  bool is_process_connected_() const;

  // Overflow resolution.
  size_t resolve_overflow_(const std::string& table_name, size_t part_index, Partition& part);
};

// TODO: Remove me!
#pragma GCC diagnostic pop

}  // namespace HugeCTR