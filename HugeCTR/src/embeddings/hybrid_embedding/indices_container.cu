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

#include <data_readers/async_reader/split_label_dense_sparse.hpp>
#include <embeddings/hybrid_embedding/indices_container.hpp>
#include <tensor2.hpp>
#include <utils.cuh>

namespace HugeCTR {
namespace hybrid_embedding {

template <typename dtype>
BatchIndices<dtype>::BatchIndices(std::vector<Model<dtype>>& models,
                                  std::vector<SparseTensor<dtype>> data_sources,
                                  std::shared_ptr<ResourceManager>& resource_manager,
                                  size_t batch_size, std::vector<size_t>& slot_size_array,
                                  size_t max_num_frequent_categories,
                                  CommunicationType communication_type)
    : num_slots_(slot_size_array.size()),
      resource_manager_(resource_manager),
      communication_type_(communication_type) {
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); ++i) {
    CudaDeviceContext ctx(resource_manager_->get_local_gpu(i)->get_device_id());
    data_.emplace_back(data_sources[i].get_value_tensor(), slot_size_array, batch_size, 1);
  }

  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    CudaDeviceContext ctx(resource_manager_->get_local_gpu(i)->get_device_id());

    frequent_compression_.emplace_back(max_num_frequent_categories, data_[i], models[i]);
    infrequent_selection_.emplace_back(data_[i], models[i]);
  }
}

template <typename dtype>
void BatchIndices<dtype>::compute(int raw_device_id, size_t batch_size, cudaStream_t stream) {
  auto& local_gpu = resource_manager_->get_local_gpu(raw_device_id);
  auto& my_data = data_[raw_device_id];

  auto samples = my_data.samples;
  samples.reset_shape({batch_size, num_slots_});

  my_data.data_to_unique_categories(samples, stream);

  compute_indices(frequent_compression_[raw_device_id], infrequent_selection_[raw_device_id],
                  communication_type_, true, stream, local_gpu->get_sm_count());
}

template class BatchIndices<uint32_t>;
template class BatchIndices<long long>;

}  // namespace hybrid_embedding
}  // namespace HugeCTR
