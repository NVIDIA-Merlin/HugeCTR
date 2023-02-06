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

#include <cuda_runtime.h>

#include <common.hpp>
#include <embeddings/hybrid_embedding/calibration_data.hpp>
#include <embeddings/hybrid_embedding/data.hpp>
#include <embeddings/hybrid_embedding/statistics.hpp>
#include <embeddings/hybrid_embedding/utils.hpp>
#include <resource_manager.hpp>
#include <tensor2.hpp>
#include <vector>

namespace HugeCTR {

namespace hybrid_embedding {

// Depends on : Data, Statistics and CalibrationData

///
/// This class defines the hybrid embedding model:
///    it indicates which categories are frequent, which are infrequent
///    and it determines where the corresponding embedding vectors are stored.
///
/// Also the mlp network - nodes topology is defined here:
///    The node_id, instance_id where the current model instance is
///    associated with is stored. However, keep in mind that these are the only
///    differentiating variables inside this class that differ from other
///    instances. As this model describes the same distribution across the nodes
///    and gpu's (networks).
///
template <typename dtype>
struct Model {
 public:
  uint32_t node_id;
  uint32_t instance_id;
  uint32_t global_instance_id;

  CommunicationType communication_type;

  Tensor2<dtype> d_num_frequent;
  Tensor2<uint32_t> d_total_frequent_count;
  dtype num_frequent;
  dtype num_categories;
  double frequent_probability;

  uint32_t num_instances;
  std::vector<uint32_t> h_num_instances_per_node;
  Tensor2<uint32_t>
      num_instances_per_node;  // number of gpus for each node, .size() == number of nodes

  Tensor2<dtype> category_location;  // indicator category => location in embedding vector
  Tensor2<dtype> frequent_categories;
  std::vector<dtype> h_frequent_model_table_offsets;
  std::vector<dtype> h_infrequent_model_table_offsets;

  // constructors: overloaded for convenience / unit tests
  // copy constructor
  Model(const Model &model);
  ~Model(){};
  Model(CommunicationType communication_type_in, uint32_t global_instance_id_in,
        const std::vector<uint32_t> &num_instances_per_node_in, size_t num_categories_in) {
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
    init_params_and_reserve(communication_type_in, global_instance_id_in, num_instances_per_node_in,
                            num_categories_in, buf);
    buf->allocate();
  }
  Model(CommunicationType communication_type_in, uint32_t global_instance_id_in,
        const std::vector<uint32_t> &num_instances_per_node_in, size_t num_categories_in,
        std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf) {
    init_params_and_reserve(communication_type_in, global_instance_id_in, num_instances_per_node_in,
                            num_categories_in, buf);
  }

  void init_params_and_reserve(CommunicationType communication_type_in,
                               uint32_t global_instance_id_in,
                               const std::vector<uint32_t> &num_instances_per_node_in,
                               size_t num_categories_in,
                               std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf);
  void init_hybrid_model(const CalibrationData &calibration, Statistics<dtype> &statistics,
                         const Data<dtype> &data, Tensor2<dtype> &tmp_categories,
                         cudaStream_t stream);
};

}  // namespace hybrid_embedding

}  // namespace HugeCTR