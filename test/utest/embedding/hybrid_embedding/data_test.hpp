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

#include <cuda_runtime.h>

#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

template <typename dtype>
std::shared_ptr<Data<dtype>> create_data_from_distribution(
    const std::vector<std::vector<double>> &distribution, const size_t batch_size,
    const size_t num_iterations) {
  std::vector<size_t> table_sizes(distribution.size());
  size_t num_categories = (size_t)0;
  for (size_t i = 0; i < distribution.size(); ++i) {
    table_sizes[i] = distribution[i].size();
    num_categories += table_sizes[i];
  }

  std::vector<double> acc_prob(num_categories);
  double sum_p = 0.;
  size_t category = (size_t)0;
  for (size_t embedding = 0; embedding < table_sizes.size(); ++embedding) {
    for (size_t em_category = 0; em_category < table_sizes[embedding]; ++em_category) {
      sum_p += distribution[embedding][em_category];
      acc_prob[category++] = sum_p;
    }
  }

  return std::make_shared<Data<dtype>>(table_sizes, batch_size, num_iterations);
}

template <typename dtype>
void test_raw_data(dtype *raw_data,
                   size_t num_samples, 
                   size_t num_tables, 
                   size_t num_iterations, 
                   const std::vector<size_t> &table_sizes);

template <typename dtype>
void test_samples(dtype *raw_data, Data<dtype> &data);

}

}