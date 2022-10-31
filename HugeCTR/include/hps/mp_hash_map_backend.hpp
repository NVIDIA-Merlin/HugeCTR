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

template <typename Key>
class MultiProcessHashMapBackend final : public VolatileBackend<Key> {
 public:
  using Base = VolatileBackend<Key>;

  MultiProcessHashMapBackend() = delete;
  DISALLOW_COPY_AND_MOVE(MultiProcessHashMapBackend);

  MultiProcessHashMapBackend(
      size_t num_partitions = 16, size_t allocation_rate = 256L * 1024L * 1024L,
      size_t sm_size = 16L * 1024L * 1024L * 1024L,
      const std::string& sm_name = "hctr_mp_hash_map_database",
      const std::chrono::nanoseconds& heart_beat_frequency = std::chrono::milliseconds{100},
      bool auto_remove = true, size_t max_get_batch_size = 10'000,
      size_t max_set_batch_size = 10'000,
      size_t overflow_margin = std::numeric_limits<size_t>::max(),
      DatabaseOverflowPolicy_t overflow_policy = DatabaseOverflowPolicy_t::EvictOldest,
      double overflow_resolution_target = 0.8);

  virtual ~MultiProcessHashMapBackend();

  bool is_shared() const override { return true; }

  const char* get_name() const override { return "MultiProcessHashMapBackend"; }

  size_t capacity(const std::string& table_name) const override;

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
  using PayloadPtr = boost::interprocess::offset_ptr<Payload>;
  using Entry = std::pair<const Key, PayloadPtr>;

  using Segment = boost::interprocess::managed_shared_memory;
  template <typename T>
  using SegmentAllocator = boost::interprocess::allocator<T, Segment::segment_manager>;

  using SharedString =
      boost::interprocess::basic_string<char, std::char_traits<char>, SegmentAllocator<char>>;
  template <typename T>
  using SharedVector = boost::interprocess::vector<T, SegmentAllocator<T>>;

  using Page = SharedVector<char>;

  template <typename K, typename V>
  using SharedMap = boost::unordered_map<K, V, boost::hash<K>, std::equal_to<K>,
                                         SegmentAllocator<std::pair<const K, V>>>;

  template <typename K, typename V>
  using SharedFlatMap =
      boost::interprocess::flat_map<K, V, std::less<K>, SegmentAllocator<std::pair<const K, V>>>;

  struct Partition final {
    size_t index;
    uint32_t value_size;
    size_t allocation_rate;

    // Pooled payload storage.
    SharedVector<Page> payload_pages;
    SharedVector<PayloadPtr> payload_slots;

    // Key -> Payload map.
    SharedFlatMap<Key, PayloadPtr> entries;

    Partition() = delete;
    Partition(const size_t index, const uint32_t value_size, const size_t allocation_rate,
              Segment& segment)
        : index{index},
          value_size{value_size},
          allocation_rate{allocation_rate},
          payload_pages(segment.get_allocator<Page>()),
          payload_slots(segment.get_allocator<PayloadPtr>()),
          entries(segment.get_allocator<Entry>()) {}
  };

  struct SharedMemory final {
    const size_t overflow_margin;
    const DatabaseOverflowPolicy_t overflow_policy;
    const double overflow_resolution_target;
    const std::chrono::nanoseconds heart_beat_frequency;
    const bool auto_remove;
    std::atomic<uint64_t> heart_beat;

    // Access control.
    boost::interprocess::interprocess_sharable_mutex read_write_guard;

    // Actual data.
    SharedMap<SharedString, SharedVector<Partition>> tables;

    SharedMemory() = delete;
    DISALLOW_COPY_AND_MOVE(SharedMemory);
    SharedMemory(const size_t overflow_margin, const DatabaseOverflowPolicy_t overflow_policy,
                 const double overflow_resolution_target,
                 const std::chrono::nanoseconds& heart_beat_frequency, const bool& auto_remove,
                 Segment& segment)
        : overflow_margin{overflow_margin},
          overflow_policy{overflow_policy},
          overflow_resolution_target{overflow_resolution_target},
          heart_beat_frequency{heart_beat_frequency},
          auto_remove{auto_remove},
          heart_beat{0},
          tables(segment.get_allocator<std::pair<const SharedString, SharedVector<Partition>>>()) {}
  };

  const size_t num_partitions_;
  const size_t allocation_rate_;
  const std::string sm_name_;
  Segment sm_segment_;
  SegmentAllocator<char> sm_char_allocator_;
  SegmentAllocator<Page> sm_page_allocator_;
  SegmentAllocator<Partition> sm_partition_allocator_;
  SharedMemory* sm_;

  // Heart beat system.
  bool heart_stop_signal_ = false;
  std::thread heart_;
  bool is_process_connected_() const;

  // Overflow resolution.
  size_t resolve_overflow_(const std::string& table_name, Partition& part);
};

// TODO: Remove me!
#pragma GCC diagnostic pop

}  // namespace HugeCTR