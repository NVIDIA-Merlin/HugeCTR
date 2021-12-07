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

#include <cuda_runtime.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/tensor2.hpp"
#include "HugeCTR/include/utils.cuh"

using namespace HugeCTR;
using namespace hybrid_embedding;

namespace {
template <typename dtype, typename emtype = float>
void data_test() {
  size_t batch_size = 4;
  size_t num_iterations = 2;
  std::vector<size_t> table_sizes{100, 10, 10, 20};
  std::vector<dtype> data_in{99, 3, 7, 19, 0,  0, 0, 0,  1, 1, 1, 1, 2, 2, 2, 2,
                             3,  3, 3, 3,  50, 2, 4, 10, 2, 2, 2, 2, 1, 1, 1, 1};
  std::vector<dtype> data_to_unique_categories_ref{
      99, 103, 117, 139, 0,  100, 110, 120, 1, 101, 111, 121, 2, 102, 112, 122,
      3,  103, 113, 123, 50, 102, 114, 130, 2, 102, 112, 122, 1, 101, 111, 121};

  Tensor2<dtype> d_data_in;
  // std::cout << "debug2" << std::endl;
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
  buff->reserve({batch_size * num_iterations * table_sizes.size()}, &d_data_in);
  buff->allocate();
  upload_tensor(data_in, d_data_in, 0);
  // std::cout << "debug3" << std::endl;
  Data<dtype> data(table_sizes, batch_size, num_iterations);
  // std::cout << "debug" << std::endl;
  data.data_to_unique_categories(d_data_in, 0);
  // std::cout << "debug1" << std::endl;
  std::vector<dtype> data_to_unique_categories_ret;
  download_tensor(data_to_unique_categories_ret, data.samples, 0);
  EXPECT_THAT(data_to_unique_categories_ret,
              ::testing::ElementsAreArray(data_to_unique_categories_ref));
};

}  // namespace

namespace HugeCTR {

namespace hybrid_embedding {

template <typename dtype>
void test_raw_data(dtype *d_raw_data, size_t num_samples, size_t num_tables, size_t num_iterations,
                   const std::vector<size_t> &table_sizes) {
  size_t num_elements = num_samples * num_tables * num_iterations;

  std::vector<dtype> h_raw_data(num_elements, (dtype)0);
  cudaStream_t stream = 0;
  CK_CUDA_THROW_(cudaMemcpyAsync(h_raw_data.data(), d_raw_data, num_elements * sizeof(dtype),
                                 cudaMemcpyDeviceToHost, stream));
  CK_CUDA_THROW_(cudaStreamSynchronize(stream));

  for (size_t iteration = 0; iteration < num_iterations; ++iteration) {
    for (size_t sample = 0; sample < num_samples; ++sample) {
      for (size_t embedding = 0; embedding < num_tables; ++embedding) {
        size_t category = (size_t)
            h_raw_data[iteration * num_samples * num_tables + sample * num_tables + embedding];
        EXPECT_TRUE(category < table_sizes[embedding]);
      }
    }
  }
}

template <typename dtype>
void test_samples(dtype *d_raw_data, Data<dtype> &data) {
  const size_t num_iterations = data.num_iterations;
  const size_t num_samples = data.batch_size;
  const size_t num_tables = data.table_sizes.size();

  size_t num_elements = num_iterations * num_samples * num_tables;

  const size_t num_categories = EmbeddingTableFunctors<dtype>::get_num_categories(data.table_sizes);
  std::vector<dtype> embedding_offsets;
  EmbeddingTableFunctors<dtype>::get_embedding_offsets(embedding_offsets, data.table_sizes);

  cudaStream_t stream = 0;
  std::vector<dtype> h_raw_data(num_elements, (dtype)0);
  CK_CUDA_THROW_(cudaMemcpyAsync(h_raw_data.data(), d_raw_data, num_elements * sizeof(dtype),
                                 cudaMemcpyDeviceToHost, stream));
  CK_CUDA_THROW_(cudaStreamSynchronize(stream));
  std::vector<dtype> h_samples;
  download_tensor(h_samples, data.samples, stream);

  for (size_t iteration = 0; iteration < num_iterations; ++iteration) {
    for (size_t sample = 0; sample < num_samples; ++sample) {
      for (size_t embedding = 0; embedding < num_tables; ++embedding) {
        size_t indx = iteration * num_samples * num_tables + sample * num_tables + embedding;
        size_t unique_category = (size_t)h_samples[indx];
        size_t category_samples = (size_t)unique_category - embedding_offsets[embedding];
        size_t category_data = (size_t)h_raw_data[indx];

        EXPECT_TRUE(category_samples == category_data);
        EXPECT_TRUE(unique_category < num_categories);
      }
    }
  }
}

template void test_raw_data<uint32_t>(uint32_t *d_raw_data, size_t num_samples, size_t num_tables,
                                      size_t num_iterations,
                                      const std::vector<size_t> &table_sizes);
template void test_raw_data<long long>(long long *d_raw_data, size_t num_samples, size_t num_tables,
                                       size_t num_iterations,
                                       const std::vector<size_t> &table_sizes);
template void test_samples<uint32_t>(uint32_t *d_raw_data, Data<uint32_t> &data);
template void test_samples<long long>(long long *d_raw_data, Data<long long> &data);
}  // namespace hybrid_embedding

}  // namespace HugeCTR

TEST(data_test, uint32) { data_test<uint32_t>(); };
TEST(data_test, long_long) { data_test<long long>(); };
