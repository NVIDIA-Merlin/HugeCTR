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

#include <gtest/gtest.h>

#include <data_readers/multi_hot/detail/batch_locations.hpp>

using namespace HugeCTR;

//#define round_up(x, y) ((((x) + ((y) - 1)) / (y)) * (y))

TEST(static_batch_locations, single_thread) {
  size_t num_batches = 51;
  size_t batch_size_bytes = 4;
  size_t start_offset = 0;
  size_t end_offset = 51 * batch_size_bytes;

  BatchLocations locations(batch_size_bytes, start_offset, end_offset);
  ASSERT_EQ(locations.count(), num_batches);

  auto it = locations.begin();

  size_t n_epochs = 5;

  for (size_t i = 0; i < num_batches * n_epochs; ++i) {
    auto location = *it;
    ASSERT_EQ(location.i, i % num_batches);
    ASSERT_EQ(location.id, i % num_batches);
    it++;
  }
}

TEST(static_batch_locations, multiple_threads) {
  size_t num_batches = 51;
  size_t batch_size_bytes = 4;
  size_t start_offset = 0;
  size_t end_offset = 51 * batch_size_bytes;
  size_t num_threads = 32;

  BatchLocations locations(batch_size_bytes, start_offset, end_offset);
  ASSERT_EQ(locations.count(), num_batches);

  auto thread_locations = locations.distribute(num_threads);
  ASSERT_EQ(thread_locations.size(), num_threads);

  std::vector<BatchForwardIterator> thread_iterators;
  for (auto& local_thread_locations : thread_locations) {
    thread_iterators.push_back(local_thread_locations->begin());
  }

  size_t n_epochs = 5;

  for (size_t epoch = 0; epoch < n_epochs; ++epoch) {
    for (size_t batch = 0; batch < num_batches; ++batch) {
      auto location = *thread_iterators[batch % num_threads];
      ASSERT_EQ(location.i, batch);
      ASSERT_EQ(location.id, batch);
      ASSERT_EQ(location.offset, batch * batch_size_bytes);
      thread_iterators[batch % num_threads]++;
    }
  }
}

TEST(static_batch_locations, single_thread_sharded) {
  size_t num_batches = 51;
  size_t batch_size_bytes = 5000;
  size_t start_offset = 0;
  size_t end_offset = 51 * batch_size_bytes - 1000;  // -1000 so shard 1 has no batch in last
  size_t num_shards = 3;
  size_t alignment = 4096;

  // expected
  size_t aligned_batch_size_bytes = round_up(batch_size_bytes, alignment);
  size_t shard_size = round_up(aligned_batch_size_bytes / num_shards, alignment);

  BatchLocations locations(batch_size_bytes, start_offset, end_offset);
  ASSERT_EQ(locations.count(), num_batches);

  auto sharded_locations = locations.shard(num_shards, alignment);
  ASSERT_EQ(sharded_locations.size(), 2);

  size_t n_epochs = 5;

  std::vector<BatchForwardIterator> shard_iterators;
  for (auto& shard_locations : sharded_locations) {
    shard_iterators.emplace_back(shard_locations->begin());
  }

  for (size_t epoch = 0; epoch < n_epochs; ++epoch) {
    for (size_t batch = 0; batch < num_batches; ++batch) {
      for (size_t shard = 0; shard < sharded_locations.size(); ++shard) {
        auto& __locations = sharded_locations[shard];
        auto it = shard_iterators[shard];
        shard_iterators[shard]++;

        ASSERT_EQ(__locations->count(), num_batches);

        auto location = *it;
        ASSERT_EQ(location.i, batch);
        ASSERT_EQ(location.id, batch);

        //        printf("batch: %zu, shard: %zu, offset: %zu, size: %zu\n", batch, shard,
        //        location.offset, location.shard_size_bytes);

        const size_t expected_offset = batch * batch_size_bytes + (shard * shard_size);
        ASSERT_EQ(location.offset, expected_offset >= end_offset ? SIZE_MAX : expected_offset);

        const size_t batch_end = (batch + 1) * batch_size_bytes;
        const size_t shard_end = batch * batch_size_bytes + (shard + 1) * shard_size;
        const size_t expected_size =
            expected_offset >= end_offset
                ? 0
                : std::min({end_offset, batch_end, shard_end}) - expected_offset;
        ASSERT_EQ(location.shard_size_bytes, expected_size);
      }
    }
  }
}

TEST(static_batch_locations, multiple_threads_sharded) {
  size_t num_batches = 51;
  size_t batch_size_bytes = 5000;
  size_t start_offset = 0;
  size_t end_offset = 51 * batch_size_bytes - 1000;  // -1000 so shard 1 has no batch in last
  size_t num_shards = 3;
  size_t alignment = 4096;
  size_t num_threads = 4;

  // expected
  size_t aligned_batch_size_bytes = round_up(batch_size_bytes, alignment);
  size_t shard_size = round_up(aligned_batch_size_bytes / num_shards, alignment);

  BatchLocations base_locations(batch_size_bytes, start_offset, end_offset);
  ASSERT_EQ(base_locations.count(), num_batches);

  auto sharded_locations = base_locations.shard(num_shards, alignment);
  ASSERT_EQ(sharded_locations.size(), 2);

  std::vector<std::vector<std::unique_ptr<IBatchLocations>>> shard_thread_locations(
      sharded_locations.size());
  std::vector<std::vector<BatchForwardIterator>> iterators(sharded_locations.size());

  for (size_t shard = 0; shard < sharded_locations.size(); ++shard) {
    auto thread_locations = sharded_locations[shard]->distribute(num_threads);
    for (auto& _locations : thread_locations) {
      iterators[shard].emplace_back(_locations->begin());
      //      printf("shard: %zu, thread_locations count: %zu\n", shard, _locations->count());
    }
    shard_thread_locations[shard] = std::move(thread_locations);
  }

  size_t n_epochs = 5;

  for (size_t epoch = 0; epoch < n_epochs; ++epoch) {
    for (size_t batch = 0; batch < num_batches; ++batch) {
      for (size_t shard = 0; shard < sharded_locations.size(); ++shard) {
        auto it = iterators[shard][batch % num_threads]++;

        auto location = *it;
        ASSERT_EQ(location.i, batch);
        ASSERT_EQ(location.id, batch);

        //        printf("thread: %zu, batch: %zu, shard: %zu, offset: %zu, size: %zu\n", batch %
        //        num_threads,
        //               batch, shard, location.offset, location.shard_size_bytes);

        const size_t expected_offset = batch * batch_size_bytes + (shard * shard_size);
        ASSERT_EQ(location.offset, expected_offset >= end_offset ? SIZE_MAX : expected_offset);

        const size_t batch_end = (batch + 1) * batch_size_bytes;
        const size_t shard_end = batch * batch_size_bytes + (shard + 1) * shard_size;
        const size_t expected_size =
            expected_offset >= end_offset
                ? 0
                : std::min({end_offset, batch_end, shard_end}) - expected_offset;
        ASSERT_EQ(location.shard_size_bytes, expected_size);
      }
    }
  }
}