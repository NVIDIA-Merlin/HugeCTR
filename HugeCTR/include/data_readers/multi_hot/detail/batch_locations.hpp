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

#include <algorithm>
#include <data_readers/multi_hot/detail/batch_forward_iterator.hpp>
#include <memory>
#include <random>
#include <vector>

namespace HugeCTR {

template <typename T>
T round_up(T x, T y) {
  return ((x + y - 1) / y) * y;
}

struct BatchFileLocation {
  size_t i;
  size_t id;
  size_t offset;
  size_t size_bytes;
};

/**
 * @brief Interface for consuming batch locations.
 */
class IBatchLocations {
  friend class BatchForwardIterator;

 public:
  using iterator = BatchForwardIterator;

  virtual iterator begin() = 0;
  virtual iterator end() = 0;
  virtual size_t count() = 0;
  virtual size_t get_batch_size_bytes() const = 0;
  virtual std::vector<std::unique_ptr<IBatchLocations>> distribute(size_t n) const = 0;
  virtual std::vector<std::unique_ptr<IBatchLocations>> shard(
      size_t n, size_t min_batch_size_bytes) const = 0;

 private:
  virtual BatchFileLocation at(size_t i) = 0;
};

/**
 * @brief Provides the batch locations at computable offsets because batches are equal sizes
 */
class BatchLocations : public IBatchLocations {
 public:
  BatchLocations(size_t batch_size_bytes, size_t start_offset, size_t end_offset,
                 bool shuffle = false, unsigned long long seed = 0)
      : shard_size_bytes_(batch_size_bytes),
        batch_size_bytes_(batch_size_bytes),
        start_offset_(start_offset),
        end_offset_(end_offset),
        ids_(((end_offset - start_offset) + batch_size_bytes - 1) / batch_size_bytes),
        order_(ids_.size()) {
    std::iota(ids_.begin(), ids_.end(), 0);
    std::iota(order_.begin(), order_.end(), 0);

    if (shuffle) {
      std::mt19937 gen(seed);
      std::shuffle(ids_.begin(), ids_.end(), gen);
    }
  }

  size_t get_batch_size_bytes() const { return batch_size_bytes_; }

  std::vector<std::unique_ptr<IBatchLocations>> distribute(size_t n) const {
    std::vector<std::unique_ptr<IBatchLocations>> batch_locations;
    for (size_t i = 0; i < n; ++i) {
      auto other = new BatchLocations(*this);
      other->ids_.clear();
      other->order_.clear();
      batch_locations.emplace_back(other);
    }

    // round-robin distribute batches between threads
    for (size_t i = 0; i < ids_.size(); ++i) {
      auto other = static_cast<BatchLocations*>(batch_locations[i % n].get());
      other->ids_.emplace_back(ids_[i]);
      other->order_.emplace_back(order_[i]);
    }
    return batch_locations;
  }

  std::vector<std::unique_ptr<IBatchLocations>> shard(size_t n, size_t min_batch_size_bytes) const {
    size_t aligned_batch_size = round_up(batch_size_bytes_, min_batch_size_bytes);
    size_t shard_size_bytes = round_up(aligned_batch_size / n, min_batch_size_bytes);
    size_t required_shards = (aligned_batch_size + shard_size_bytes - 1) / shard_size_bytes;

    std::vector<std::unique_ptr<IBatchLocations>> batch_locations;
    for (size_t i = 0; i < required_shards; ++i) {
      auto other = new BatchLocations(*this);
      other->shard_size_bytes_ = shard_size_bytes;
      other->start_offset_ = start_offset_ + i * shard_size_bytes;
      batch_locations.emplace_back(other);
    }
    return batch_locations;
  }

  IBatchLocations::iterator begin() { return IBatchLocations::iterator(this, 0ul); }

  IBatchLocations::iterator end() { return IBatchLocations::iterator(this, ids_.size()); }

  size_t count() { return this->end() - this->begin(); }

 private:
  BatchFileLocation at(size_t i) {
    size_t batch_id = ids_[i % ids_.size()];
    BatchFileLocation batch;
    batch.i = order_[i % order_.size()];
    batch.id = batch_id;
    batch.offset = start_offset_ + batch_id * batch_size_bytes_;
    batch.offset = batch.offset >= end_offset_ ? SIZE_MAX : batch.offset;

    // size can be clamped by end of file, end of batch, or end of shard.
    size_t batch_end = (batch_id + 1) * batch_size_bytes_;
    size_t shard_end = batch.offset + shard_size_bytes_;
    size_t size = std::min(end_offset_, std::min(batch_end, shard_end)) - batch.offset;
    batch.size_bytes = batch.offset >= end_offset_ ? 0 : size;
    return batch;
  }

  size_t shard_size_bytes_;
  size_t batch_size_bytes_;
  size_t start_offset_;
  size_t end_offset_;
  std::vector<size_t> ids_;    // for shuffle
  std::vector<size_t> order_;  // global iteration order
};

}  // namespace HugeCTR