/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "HugeCTR/include/heap.hpp"
#include <future>
#include <random>
#include "HugeCTR/include/csr_chunk.hpp"
#include "gtest/gtest.h"

using namespace HugeCTR;

TEST(heap, head_alloc_exceed_boundary) {
  EXPECT_THROW({ Heap<float> heap(33, 0.0f); }, internal_runtime_error);
}

TEST(heap, heap_basic_test) {
  unsigned int keys[5];
  float* chunks[5];

  Heap<float> heap(3, 0.0f);
  heap.free_chunk_checkout(&chunks[0], &keys[0]);
  heap.free_chunk_checkout(&chunks[1], &keys[1]);
  EXPECT_NE(chunks[0], chunks[1]);
  EXPECT_NE(keys[0], keys[1]);
  *chunks[1] = 20.0f;
  heap.chunk_write_and_checkin(keys[1]);
  *chunks[0] = 10.0f;
  heap.chunk_write_and_checkin(keys[0]);
  heap.free_chunk_checkout(&chunks[2], &keys[2]);
  EXPECT_TRUE(keys[2] != keys[0] && keys[2] != keys[1]);
  heap.data_chunk_checkout(&chunks[3], &keys[3]);
  EXPECT_TRUE((keys[3] == keys[0] && *chunks[3] == 10.0f) ||
              (keys[3] == keys[1] && *chunks[3] == 20.0f));
  heap.chunk_free_and_checkin(keys[3]);
  heap.data_chunk_checkout(&chunks[4], &keys[4]);
  EXPECT_NE(keys[3], keys[4]);
  EXPECT_TRUE((keys[4] == keys[0] && *chunks[4] == 10.0f) ||
              (keys[4] == keys[1] && *chunks[4] == 20.0f));
  heap.chunk_free_and_checkin(keys[3]);
  heap.chunk_write_and_checkin(keys[2]);
  heap.free_chunk_checkout(&chunks[2], &keys[2]);
}

TEST(heap, heap_csr_chunk_test) {
  const int num_devices = 4;
  const int batchsize = 2048;
  const int label_dim = 2;
  const int slot_num = 10;
  const int max_value_size = 2048 * 20;
  CSRChunk<long long> chunk(num_devices, batchsize, label_dim, slot_num, max_value_size);
  Heap<CSRChunk<long long>> csr_heap(32, chunk);
  unsigned int key = 0;
  CSRChunk<long long>* chunk_tmp = nullptr;
  csr_heap.free_chunk_checkout(&chunk_tmp, &key);
  const std::vector<CSR<long long>*>& csr_buffers = chunk_tmp->get_csr_buffers();
  csr_buffers[0]->reset();
}

TEST(heap, heap_multi_threads_test) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> interval_dis(0, 100);  // from 0ms to 100ms
  std::uniform_int_distribution<long> value_dis(0, INT32_MAX);

  const int TEST_ROUNDS = 2000;
  const int PARALLELISM = 8;

  int intervals[256];
  long values[256];
  for (int i = 0; i < 256; i++) {
    intervals[i] = interval_dis(gen);
    values[i] = value_dis(gen);
  }

  Heap<long> heap(32, 0l);

  std::future<void> write_futures[PARALLELISM];
  std::future<void> read_futures[PARALLELISM];

  std::atomic<long> write_sum(0l);
  std::atomic<size_t> write_index(0);
  for (int i = 0; i < PARALLELISM; i++) {
    write_futures[i] = std::async([&] {
      long* chunk;
      unsigned int key;
      size_t i;
      while ((i = write_index++) < TEST_ROUNDS) {
        heap.free_chunk_checkout(&chunk, &key);
        *chunk = values[i % 256];
        write_sum.fetch_add(*chunk);
        heap.chunk_write_and_checkin(key);
        if (intervals[i % 256] >= 80)
          std::this_thread::sleep_for(std::chrono::milliseconds(intervals[i % 256]));
      }
    });
  }

  std::atomic<long> read_sum(0l);
  std::atomic<size_t> read_index(0);
  for (int i = 0; i < PARALLELISM; i++) {
    read_futures[i] = std::async([&] {
      long* chunk;
      unsigned int key;
      size_t i;
      while ((i = read_index++) < TEST_ROUNDS) {
        heap.data_chunk_checkout(&chunk, &key);
        read_sum.fetch_add(*chunk);
        heap.chunk_free_and_checkin(key);
        if (intervals[(i + 128) % 256] < 20)
          std::this_thread::sleep_for(std::chrono::milliseconds(intervals[(i + 128) % 256]));
      }
    });
  }

  for (int i = 0; i < PARALLELISM; i++) {
    write_futures[i].wait();
    read_futures[i].wait();
  }
  EXPECT_EQ(write_sum, read_sum);
}