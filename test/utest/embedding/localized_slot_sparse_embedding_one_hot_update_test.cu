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

#include "HugeCTR/include/utils.hpp"
// #include "cub/cub/device/device_radix_sort.cuh"
// #include "cub/cub/device/device_scan.cuh"
// #include "HugeCTR/include/ft_topk/ft_topk.cuh"

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <common.hpp>
#include <cuda_utils.cuh>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "HugeCTR/include/embeddings/hybrid_embedding/statistics.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_one_hot.hpp"
#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"
#include "gtest/gtest.h"
#include "nvToolsExt.h"
#include "utest/embedding/embedding_test_utils.hpp"
#include "utest/embedding/hybrid_embedding/input_generator.hpp"
#include "utest/test_utils.h"

using namespace HugeCTR;
using namespace embedding_test;

template <typename TypeEmbeddingComp>
class GpuData {
 public:
  GpuData() {}
  ~GpuData() {}
  GpuData(const std::vector<size_t>& h_value_index, const size_t max_vocabulary_size,
          const size_t embedding_vec_size) {
    size_t num_samples = h_value_index.size();
    init_data(num_samples, max_vocabulary_size, embedding_vec_size);
    HCTR_LIB_THROW(cudaMemcpy(value_index.get_ptr(), h_value_index.data(),
                              sizeof(size_t) * num_samples, cudaMemcpyHostToDevice));
  }

  void init_data(const size_t num_samples, const size_t max_vocabulary_size,
                 const size_t embedding_vec_size) {
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();

    buf->reserve({num_samples}, &value_index);
    buf->reserve({max_vocabulary_size * embedding_vec_size}, &weights);
    buf->reserve({num_samples * embedding_vec_size}, &wgrad);

    const size_t max_top_categories = get_max_size_top_categories();
    buf->reserve({max_top_categories}, &top_categories);
    size_top_categories = 0;

    buf->allocate();
  }

  Tensor2<size_t> value_index;
  Tensor2<size_t> top_categories;
  size_t size_top_categories;

  Tensor2<TypeEmbeddingComp> wgrad;
  Tensor2<float> weights;

  void init_weights(size_t num_samples, size_t max_vocabulary_size, size_t embedding_vec_size,
                    const std::vector<TypeEmbeddingComp>& h_wgrad) {
    HCTR_LIB_THROW(cudaMemcpy(wgrad.get_ptr(), h_wgrad.data(),
                              sizeof(TypeEmbeddingComp) * num_samples * embedding_vec_size,
                              cudaMemcpyHostToDevice));
    HCTR_LIB_THROW(cudaMemset(weights.get_ptr(), 0.f,
                              sizeof(float) * max_vocabulary_size * embedding_vec_size));
  }
};

template <typename TypeEmbeddingComp>
void update_test(const std::vector<size_t>& value_index, size_t max_vocabulary_size,
                 size_t embedding_vec_size, const std::vector<TypeEmbeddingComp>& wgrad) {
  std::cout << "Starting embedding update test..." << std::endl;
  cudaStream_t stream = 0;

  // get number of sms
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, 0);

  // test sorting
  std::map<size_t, std::set<uint32_t>> ref_categorize;
  size_t num_samples = value_index.size();
  for (size_t i = 0; i < num_samples; ++i) {
    ref_categorize[value_index[i]].insert(i);
  }
  size_t num_unique_categories_ref = ref_categorize.size();

  std::vector<size_t> value_index_sort;
  std::vector<uint32_t> sample_id_sort;
  std::vector<uint32_t> sorted_sample_offset_category;

  GpuData<TypeEmbeddingComp> gpu_data(value_index, max_vocabulary_size, embedding_vec_size);

  // now for the update
  size_t weight_size = max_vocabulary_size * embedding_vec_size;
  std::vector<float> weights_test(weight_size, 0.0f);
  std::vector<float> weights_ref(weight_size, 0.0f);

  // ref weight update :
  for (auto const& pair : ref_categorize) {
    for (size_t j = 0; j < embedding_vec_size; ++j) {
      float sum_j = 0.f;
      for (auto const& sample_index : pair.second) {
        sum_j += (float)wgrad[sample_index * embedding_vec_size + j];
      }
      weights_ref[pair.first * embedding_vec_size + j] = -sum_j;
    }
  }
  // done with calculating ref weights

  // init wgrad and weights on gpu:
  gpu_data.init_weights(num_samples, max_vocabulary_size, embedding_vec_size, wgrad);

  std::cout << "performing atomic cached kernel..." << std::endl;
  SparseEmbeddingFunctors::opt_sgd_atomic_cached<TypeEmbeddingComp>(
      num_samples, embedding_vec_size, gpu_data.value_index.get_ptr(), 1.0f, 1.0f,
      gpu_data.wgrad.get_ptr(), gpu_data.weights.get_ptr(), gpu_data.top_categories.get_ptr(),
      gpu_data.size_top_categories, stream, true);

  std::cout << "done performing kernel, testing results.." << std::endl;
  HCTR_LIB_THROW(cudaMemcpy(weights_test.data(), gpu_data.weights.get_ptr(),
                            sizeof(float) * embedding_vec_size * max_vocabulary_size,
                            cudaMemcpyDeviceToHost));

  const float epsilon = 1.0e-4;
  double diff_ave = 0.0;

  size_t count_neq = 0;
  size_t count_all = 0;
  bool all_el_equal = true;
  for (auto const& pair : ref_categorize) {
    const size_t& category = pair.first;
    bool category_equal = true;
    for (size_t j = 0; j < embedding_vec_size; ++j) {
      size_t index = category * embedding_vec_size + j;
      float diff = weights_ref[index] - weights_test[index];
      diff = (diff > 0.f ? diff : -diff);
      diff_ave += (double)diff;
      all_el_equal = (all_el_equal && (diff < epsilon));
      category_equal = category_equal && (diff < epsilon);

      count_neq += (size_t)(diff >= epsilon);
      count_all++;
    }
    if (!category_equal) {
      std::cout << "Fail : the weights of category " << category << " are wrongly computed."
                << std::endl;
      std::cout << "Weight expected : " << weights_ref[category * embedding_vec_size]
                << "\t weight calculated : " << weights_test[category * embedding_vec_size]
                << std::endl;
    }
  }

  diff_ave /= (double)count_all;
  std::cout << "number of correct elements : " << count_all - count_neq << " out of  " << count_all
            << " = " << (double)(count_all - count_neq) / (double)count_all * 100.0 << " % "
            << std::endl;
  if (!all_el_equal) {
    std::cout << "average diff : " << diff_ave << std::endl;
    std::cout << "CPU : ";
    for (size_t i = 0; i < 10; ++i) {
      std::cout << '\t' << weights_ref[128 + i];
    }
    std::cout << std::endl;
    std::cout << "GPU : ";
    for (size_t i = 0; i < 10; ++i) {
      std::cout << '\t' << weights_test[128 + i];
    }
    std::cout << std::endl;
  }
  ASSERT_TRUE(all_el_equal && "not all embedding vector weights are updated correctly!");

  bool all_el_zero = true;
  for (size_t i = 0; i < max_vocabulary_size; ++i) {
    if (ref_categorize.find(i) == ref_categorize.end()) {
      for (size_t j = 0; j < embedding_vec_size; ++j) {
        all_el_zero = all_el_zero && (weights_test[i * embedding_vec_size + j] == 0.f);
      }
    }
  }
  ASSERT_TRUE(all_el_zero && "some embedding vectors that shouldn't be updated were modified!");

  std::cout << "Finished embedding update test SUCCESSFULLY!" << std::endl;
}

template <typename etype>
void setup_and_run_randomized_test(const int N_test, const int embedding_vec_size,
                                   const int num_samples) {
  std::vector<size_t> category_size{39884, 3, 63, 10};
  std::vector<size_t> category_offset(4);

  size_t max_vocabulary_size = 0;
  for (size_t i = 0; i < category_size.size(); ++i) {
    category_offset[i] = max_vocabulary_size;
    max_vocabulary_size += category_size[i];
  }

  std::vector<etype> wgrad(num_samples * embedding_vec_size, (etype)1.);

  for (int n = 0; n < N_test; ++n) {
    // create test input
    std::vector<size_t> value_index;
    for (int i = 0; i < num_samples; ++i) {
      int embedding = rand() % 4;
      size_t category = category_offset[embedding] + (size_t)rand() % category_size[embedding];
      value_index.push_back(category);
    }

    // perform test
    update_test<etype>(value_index, max_vocabulary_size, embedding_vec_size, wgrad);
  }
}

TEST(localized_one_hot_update_test, fp16_sgd_atomic_cached) {
  const int N_test = 5;
  const int embedding_vec_size = 128;
  const int num_samples = 64 * 1024;

  for (size_t multiplier = 1; multiplier < 32; multiplier *= 2) {
    setup_and_run_randomized_test<__half>(N_test, embedding_vec_size, num_samples);
  }
}

TEST(localized_one_hot_update_test, fp32_sgd_atomic_cached) {
  const int N_test = 5;
  const int embedding_vec_size = 128;
  const int num_samples = 64 * 1024;

  for (size_t multiplier = 1; multiplier < 32; multiplier *= 2) {
    setup_and_run_randomized_test<float>(N_test, embedding_vec_size, num_samples);
  }
}
