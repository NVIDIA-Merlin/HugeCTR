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

#include <algorithm>
#include <iostream>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/resource_manager.hpp"
#include "HugeCTR/include/tensor2.hpp"
#include "HugeCTR/include/utils.cuh"

namespace HugeCTR {

namespace hybrid_embedding {

template <typename dtype>
Model<dtype>::Model(const Model &model) {
  node_id = model.node_id;
  instance_id = model.instance_id;
  global_instance_id = model.global_instance_id;
  communication_type = model.communication_type;
  d_num_frequent = model.d_num_frequent;
  d_total_frequent_count = model.d_total_frequent_count;
  num_frequent = model.num_frequent;
  num_categories = model.num_categories;
  num_instances = model.num_instances;
  if (model.h_num_instances_per_node.size() > 0) {
    h_num_instances_per_node.resize(model.h_num_instances_per_node.size());
    for (size_t i = 0; i < model.h_num_instances_per_node.size(); ++i) {
      h_num_instances_per_node[i] = model.h_num_instances_per_node[i];
    }
  }
  num_instances_per_node = model.num_instances_per_node;
  category_location = model.category_location;
  frequent_categories = model.frequent_categories;
  infrequent_categories = model.infrequent_categories;
  if (model.h_frequent_model_table_offsets.size() > 0) {
    h_frequent_model_table_offsets = model.h_frequent_model_table_offsets;
  }
  if (model.h_infrequent_model_table_offsets.size() > 0) {
    h_infrequent_model_table_offsets = model.h_infrequent_model_table_offsets;
  }
}

template <typename dtype>
void Model<dtype>::init_params_and_reserve(CommunicationType communication_type_in,
                                           uint32_t global_instance_id_in,
                                           const std::vector<uint32_t> &num_instances_per_node_in,
                                           size_t num_categories_in,
                                           std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf) {
  // initialize model parameters and reserve memory
  communication_type = communication_type_in;
  global_instance_id = global_instance_id_in;
  h_num_instances_per_node = num_instances_per_node_in;
  num_categories = num_categories_in;
  num_instances = 0;
  for (size_t i = 0; i < h_num_instances_per_node.size(); ++i)
    num_instances += h_num_instances_per_node[i];

  const size_t num_nodes = h_num_instances_per_node.size();
  assert(num_nodes > 0);
  uint32_t sum_instances = (uint32_t)0;
  for (node_id = 0; node_id < num_nodes && global_instance_id >= sum_instances; ++node_id)
    sum_instances += h_num_instances_per_node[node_id];
  node_id--;

  // instance id within node
  instance_id = global_instance_id - (sum_instances - h_num_instances_per_node[node_id]);
  buf->reserve({1, 1}, &d_num_frequent);
  buf->reserve({1, 1}, &d_total_frequent_count);
  buf->reserve({h_num_instances_per_node.size(), 1}, &num_instances_per_node);
  buf->reserve({(size_t)(2 * (num_categories+1)), 1}, &category_location); // +1 for NULL category
  buf->reserve({(size_t)num_categories, 1}, &frequent_categories);
  buf->reserve({(size_t)num_categories, 1}, &infrequent_categories);
}

/// init_model calculates the optimal number of frequent categories
/// given the calibration of the all-to-all and all-reduce.
template <typename dtype>
void Model<dtype>::init_hybrid_model(const CalibrationData &calibration,
                                     Statistics<dtype> &statistics, const Data<dtype> &data,
                                     cudaStream_t stream) {
  // list the top categories sorted by count
  const Tensor2<dtype> &samples = data.samples;
  statistics.sort_categories_by_count(samples, stream);

  /* Calculate table offsets, i.e cumulative sum of the table sizes */
  std::vector<dtype> h_table_offsets(data.table_sizes.size() + 1);
  h_table_offsets[0] = 0;
  for (size_t i = 0; i < data.table_sizes.size(); i++) {
    h_table_offsets[i + 1] = h_table_offsets[i] + data.table_sizes[i];
  }
  upload_tensor(h_table_offsets, statistics.table_offsets, stream);

  // from the sorted count, determine the number of frequent categories
  //
  // If the calibration data is present, this is used to calculate the number
  // of frequent categories.  Otherwise use the threshold required by the
  // communication type.
  num_frequent = ModelInitializationFunctors<dtype>::calculate_num_frequent_categories(
      communication_type, num_instances, calibration, statistics, data, d_num_frequent.get_ptr(),
      stream);

  frequent_probability = ModelInitializationFunctors<dtype>::calculate_frequent_probability(
      statistics, num_frequent, d_total_frequent_count.get_ptr(), stream);

  dtype num_infrequent = num_categories - num_frequent;
  frequent_categories.reset_shape({(size_t)num_frequent, 1});
  infrequent_categories.reset_shape({(size_t)num_infrequent, 1});

  /* The categories are organized:
   *  - per instance (round-robin)
   *  - then per slot
   *  - and finally in decreasing order of frequency
   */
  statistics.calculate_frequent_and_infrequent_categories(
      frequent_categories.get_ptr(), infrequent_categories.get_ptr(), category_location.get_ptr(),
      num_frequent, num_infrequent, stream);
  /* Calculate frequent and infrequent table offsets */
  statistics.calculate_frequent_model_table_offsets(h_frequent_model_table_offsets,
                                                    frequent_categories, num_frequent, stream);
  statistics.calculate_infrequent_model_table_offsets(h_infrequent_model_table_offsets,
                                                      infrequent_categories, category_location,
                                                      global_instance_id, num_infrequent, stream);

  /* A synchronization is necessary to ensure that the host arrays have been copied */
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
}

template class Model<uint32_t>;
template class Model<long long>;

}  // namespace hybrid_embedding

}  // namespace HugeCTR
