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
#include <data_readers/async_reader/async_reader_common.hpp>
#include <embeddings/hybrid_embedding/data.hpp>
#include <embeddings/hybrid_embedding/frequent_embedding.hpp>
#include <embeddings/hybrid_embedding/hybrid_indices.hpp>
#include <embeddings/hybrid_embedding/infrequent_embedding.hpp>
#include <embeddings/hybrid_embedding/model.hpp>
#include <embeddings/hybrid_embedding/utils.hpp>
#include <gpu_resource.hpp>
#include <graph_wrapper.hpp>
#include <map>
#include <tensor2.hpp>
#include <vector>

namespace HugeCTR {
namespace hybrid_embedding {

template <typename dtype>
class BatchIndices {
 public:
  BatchIndices(std::vector<Model<dtype>>& models, std::vector<SparseTensor<dtype>> data_source,
               std::shared_ptr<ResourceManager>& resource_manager, size_t batch_size,
               std::vector<size_t>& slot_size_array, size_t max_num_frequent_categories,
               CommunicationType communication_type);

  void compute(int raw_device_id, size_t batch_size, cudaStream_t stream);

  FrequentEmbeddingCompression<dtype>& get_frequent(int raw_device_id) {
    return frequent_compression_[raw_device_id];
  }

  InfrequentEmbeddingSelection<dtype>& get_infrequent(int raw_device_id) {
    return infrequent_selection_[raw_device_id];
  }

 private:
  size_t num_slots_ = 0;
  std::shared_ptr<ResourceManager> resource_manager_;
  CommunicationType communication_type_;
  std::vector<Data<dtype>> data_;
  std::vector<FrequentEmbeddingCompression<dtype>> frequent_compression_;
  std::vector<InfrequentEmbeddingSelection<dtype>> infrequent_selection_;
};

}  // namespace hybrid_embedding
}  // namespace HugeCTR
