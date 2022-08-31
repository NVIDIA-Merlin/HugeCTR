/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include "generic_lookup.cuh"
#include "model_backward.hpp"
#include "utils.cuh"
namespace embedding {

ModelBackward::ModelBackward(std::shared_ptr<CoreResourceManager> core, int num_gpus,
                             int num_local_embedding, const std::vector<int> &h_local_hotness_list,
                             const std::vector<int> &h_local_ev_size_list, int universal_batch_size)
    : core_(core), num_gpus_(num_gpus), num_local_embedding_(num_local_embedding) {
  std::vector<int> num_unique_key_list;
  for (int i = 0; i < num_local_embedding; ++i) {
    num_unique_key_list.push_back(h_local_hotness_list[i] * h_local_ev_size_list[i]);
  }

  int max_unique_key_ev_buffer_size =
      std::accumulate(num_unique_key_list.begin(), num_unique_key_list.end(), 0);
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());

  auto buffer_ptr = GetBuffer(core);
  grad_ev_ = buffer_ptr->reserve({universal_batch_size, max_unique_key_ev_buffer_size},
                                 DeviceType::GPU, TensorScalarType::Float32);
  buffer_ptr->allocate();
}

void ModelBackward::compute(const TensorList &model_comm_buffer, const Tensor &unique_dst_idx,
                            const Tensor &sorted_bucket_id_list,
                            const Tensor &sorted_bucket_id_offset, size_t num_unique_key,
                            const Tensor &d_local_ev_size_offset, int batch_size, int max_ev_size,
                            Tensor *grad_ev) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  auto stream = core_->get_local_gpu()->get_stream();
  int batch_size_per_gpu = batch_size / num_gpus_;

  DISPATCH_FLOAT_AND_HALF_FUNCTION(model_comm_buffer.dtype().type(), emb_t, [&] {
    const uint32_t *sorted_bucket_id_list_ptr = sorted_bucket_id_list.get<uint32_t>();
    const uint32_t *sorted_bucket_id_offset_ptr = sorted_bucket_id_offset.get<uint32_t>();
    const uint32_t *unique_dst_idx_ptr = unique_dst_idx.get<uint32_t>();
    const emb_t **model_comm_buffer_ptr = model_comm_buffer.get<emb_t>();
    const int *local_ev_offset_list_ptr = d_local_ev_size_offset.get<int>();
    float *grad_ev_ptr = grad_ev_.get<float>();

    auto multi_to_one_desc = make_MultiToOne<emb_t, float>(
        num_unique_key, [=] __device__(int i) { return sorted_bucket_id_offset_ptr[i]; },
        [=] __device__(int i) { return 1; },
        [=] __device__(int i) { return unique_dst_idx_ptr[i + 1] - unique_dst_idx_ptr[i]; },
        [=] __device__(int i) {
          int bucket_id = sorted_bucket_id_list_ptr[i];
          int i_lookup = bucket_id / batch_size;
          int batch_id = bucket_id % batch_size;
          int gpu_id = batch_id / batch_size_per_gpu;
          int local_batch_id = batch_id % batch_size_per_gpu;
          int ev_size = local_ev_offset_list_ptr[i_lookup + 1] - local_ev_offset_list_ptr[i_lookup];

          return model_comm_buffer_ptr[gpu_id] +
                 batch_size_per_gpu * local_ev_offset_list_ptr[i_lookup] + local_batch_id * ev_size;
        },
        [=] __device__(int i) { return grad_ev_ptr + unique_dst_idx_ptr[i]; });
    copy_multi_to_one(multi_to_one_desc, max_ev_size, stream);
  });
  *grad_ev = grad_ev_;
}

DPLocalReduce::DPLocalReduce(std::shared_ptr<CoreResourceManager> core, int num_gpus,
                             int num_local_embedding, const std::vector<int> &h_local_hotness_list,
                             const std::vector<int> &h_local_ev_size_list, int universal_batch_size)
    : core_(core), num_gpus_(num_gpus), num_local_embedding_(num_local_embedding) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  Device device{DeviceType::GPU};

  max_ev_size_ = *std::max_element(h_local_ev_size_list.begin(), h_local_ev_size_list.end());
  std::vector<int> num_unique_key_list;
  for (int i = 0; i < num_local_embedding; ++i) {
    num_unique_key_list.push_back(h_local_hotness_list[i] * h_local_ev_size_list[i]);
  }

  int max_unique_key_ev_buffer_size =
      std::accumulate(num_unique_key_list.begin(), num_unique_key_list.end(), 0);

  auto buffer_ptr = GetBuffer(core);
  grad_ev_ = buffer_ptr->reserve({universal_batch_size, max_unique_key_ev_buffer_size}, device,
                                 TensorScalarType::Float32);
  buffer_ptr->allocate();
}

void DPLocalReduce::compute(const Tensor &top_grad, const Tensor &unique_dst_idx,
                            const Tensor &sorted_bucket_id_list,
                            const Tensor &sorted_bucket_id_offset, size_t num_unique_key,
                            const Tensor &d_ev_size_offset, int batch_size, int max_ev_size,
                            Tensor *grad_ev) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  int batch_size_per_gpu = batch_size / num_gpus_;

  DISPATCH_FLOAT_AND_HALF_FUNCTION(top_grad.dtype().type(), emb_t, [&] {
    auto stream = core_->get_local_gpu()->get_stream();
    const uint32_t *sorted_bucket_id_list_ptr = sorted_bucket_id_list.get<uint32_t>();
    const uint32_t *sorted_bucket_id_offset_ptr = sorted_bucket_id_offset.get<uint32_t>();
    const uint32_t *unique_dst_idx_ptr = unique_dst_idx.get<uint32_t>();
    const int *d_ev_size_offset_ptr = d_ev_size_offset.get<int>();
    const emb_t *top_grad_ptr = top_grad.get<emb_t>();
    float *grad_ev_ptr = grad_ev_.get<float>();

    auto multi_to_one_desc = make_MultiToOne<emb_t, float>(
        num_unique_key, [=] __device__(int i) { return sorted_bucket_id_offset_ptr[i]; },
        [=] __device__(int i) { return 1; },
        [=] __device__(int i) { return unique_dst_idx_ptr[i + 1] - unique_dst_idx_ptr[i]; },
        [=] __device__(int i) {
          int bucket_id = sorted_bucket_id_list_ptr[i];
          int i_lookup = bucket_id / batch_size_per_gpu;
          int b = bucket_id % batch_size_per_gpu;
          int ev_size = d_ev_size_offset_ptr[i_lookup + 1] - d_ev_size_offset_ptr[i_lookup];

          return top_grad_ptr + batch_size_per_gpu * d_ev_size_offset_ptr[i_lookup] + b * ev_size;
        },
        [=] __device__(int i) { return grad_ev_ptr + unique_dst_idx_ptr[i]; });
    copy_multi_to_one(multi_to_one_desc, max_ev_size, stream);
  });

  *grad_ev = grad_ev_;
}
}  // namespace embedding
