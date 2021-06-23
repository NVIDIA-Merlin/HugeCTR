/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "HugeCTR/include/data_readers/heapex.hpp"

#include <future>
#include <random>

#include "HugeCTR/include/data_readers/csr_chunk.hpp"
#include "gtest/gtest.h"

using namespace HugeCTR;

TEST(heapex, head_alloc_exceed_boundary) {
  EXPECT_THROW({ HeapEx<float> heap(33, 0.0f); }, internal_runtime_error);
}

TEST(heapex, heapex_basic_test) {
  float* chunks[5];

  HeapEx<float> heapex(3, 0.0f);
  chunks[0] = heapex.checkout_free_chunk(0);
  chunks[1] = heapex.checkout_free_chunk(1);
  EXPECT_NE(chunks[0], chunks[1]);
  *chunks[1] = 20.0f;
  heapex.commit_data_chunk(1, false);
  *chunks[0] = 10.0f;
  heapex.commit_data_chunk(0, false);
  chunks[2] = heapex.checkout_free_chunk(2);
  chunks[3] = heapex.checkout_data_chunk();
  EXPECT_TRUE(*chunks[3] == 10.0f);
  heapex.return_free_chunk();
  chunks[4] = heapex.checkout_data_chunk();
  EXPECT_TRUE(*chunks[4] == 20.0f);
  heapex.return_free_chunk();
  heapex.commit_data_chunk(2, false);
}

TEST(heapex, heapex_csr_chunk_test) {
  const int num_devices = 4;
  const int batchsize = 2048;
  const int label_dim = 2;
  const int slot_num = 10;
  const DataReaderSparseParam param = {DataReaderSparse_t::Distributed, slot_num, slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  HeapEx<CSRChunk<long long>> csr_heapex(32, num_devices, batchsize, label_dim, params);
  CSRChunk<long long>* chunk_tmp = nullptr;
  chunk_tmp = csr_heapex.checkout_free_chunk(0);
  chunk_tmp->get_csr_buffer(0).reset();
}
