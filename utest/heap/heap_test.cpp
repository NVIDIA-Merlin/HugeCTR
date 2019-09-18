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
#include <thread>
#include "HugeCTR/include/csr_chunk.hpp"
#include "gtest/gtest.h"

using namespace HugeCTR;

TEST(heap, heap_basic_test) {
  float a = 10.f;
  Heap<float> heap(32, a);
  unsigned int key[] = {0, 0, 0, 0, 0};
  float* chunk[] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  heap.free_chunk_checkout(&chunk[0], &key[0]);
  std::cout << key[0] << std::endl;  // 1
  heap.free_chunk_checkout(&chunk[0], &key[1]);
  std::cout << key[1] << std::endl;  // 2
  heap.chunk_write_and_checkin(key[0]);
  heap.chunk_write_and_checkin(key[1]);
  heap.free_chunk_checkout(&chunk[0], &key[3]);
  std::cout << key[3] << std::endl;  // 4
  heap.data_chunk_checkout(&chunk[1], &key[4]);
  std::cout << key[4] << std::endl;  // 1
  heap.chunk_free_and_checkin(key[4]);
  heap.data_chunk_checkout(&chunk[1], &key[2]);
  std::cout << key[2] << std::endl;  // 2
  heap.chunk_free_and_checkin(key[2]);

  heap.chunk_write_and_checkin(key[1]);
  heap.free_chunk_checkout(&chunk[0], &key[0]);
  std::cout << key[0] << std::endl;  // 1
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
