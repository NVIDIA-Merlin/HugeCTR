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

#include <embedding/operators/generic_lookup.cuh>
#include <embedding/operators/index_calculation.hpp>  // FIXME: to introduce ReductionIndices
#include <embedding/operators/model_backward.hpp>     // FIXME: to introduce PartialReduceResult
#include <embedding/operators/multi_to_one_reduce.cuh>
#include <embedding/operators/weighted_model_backward.hpp>
#include <utils.cuh>
#include <utils.hpp>

namespace embedding {

WeightedModelBackward::WeightedModelBackward(std::shared_ptr<CoreResourceManager> core,
                                             int num_gpus, int num_local_embedding,
                                             const std::vector<int>& h_local_hotness_list,
                                             const std::vector<int>& h_local_ev_size_list,
                                             int universal_batch_size, int max_ev_size, int num_sms)
    : core_(core),
      num_gpus_(num_gpus),
      num_local_embedding_(num_local_embedding),
      max_ev_size_(max_ev_size),
      num_sms_(num_sms) {
  std::vector<int> num_unique_key_list;
  for (int i = 0; i < num_local_embedding; ++i) {
    num_unique_key_list.push_back(h_local_hotness_list[i] * h_local_ev_size_list[i]);
  }

  int max_unique_key_ev_buffer_size =
      std::accumulate(num_unique_key_list.begin(), num_unique_key_list.end(), 0);
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());

  int tensor_device_id = core_->get_device_id();
  core23::Device device(core23::DeviceType::GPU, tensor_device_id);
  core23::TensorParams params = core23::TensorParams();

  grad_ev_ = core23::Tensor(params.shape({universal_batch_size, max_unique_key_ev_buffer_size})
                                .device(device)
                                .data_type(core23::ScalarType::Float));

  partial_grad_ev_ = core23::Tensor(params.shape({num_sms_ * 4 * max_ev_size_})
                                        .device(device)
                                        .data_type(core23::ScalarType::Float));

  partial_key_ = core23::Tensor(
      params.shape({num_sms_ * 4}).device(device).data_type(core23::ScalarType::UInt32));

  partial_ev_length_ = core23::Tensor(
      params.shape({num_sms_ * 4}).device(device).data_type(core23::ScalarType::Int32));

  partial_dst_offset_array_ = core23::Tensor(
      params.shape({num_sms_ * 4}).device(device).data_type(core23::ScalarType::UInt32));
}

void WeightedModelBackward::compute(const core23::Tensor& model_comm_buffer,
                                    const core23::Tensor& unique_dst_idx,
                                    const core23::Tensor& sorted_bucket_id_list,
                                    const core23::Tensor& sorted_bucket_id_offset,
                                    uint64_t num_unique_key, const core23::Tensor& corrdinate_key,
                                    const core23::Tensor& coordinate_wgrad_dst_idx,
                                    const core23::Tensor& d_local_ev_size_offset, int batch_size,
                                    int max_ev_size, size_t num_model_key, core23::Tensor* grad_ev,
                                    const core23::Tensor& coordinate_sp_weight) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  auto stream = core_->get_local_gpu()->get_stream();
  int batch_size_per_gpu = batch_size / num_gpus_;

  cudaMemsetAsync(grad_ev_.data(), 0, grad_ev_.num_bytes(), stream);
  DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(model_comm_buffer.data_type().type(), emb_t, [&] {
    const uint32_t* unique_dst_idx_ptr = unique_dst_idx.data<uint32_t>();
    const emb_t** model_comm_buffer_ptr = static_cast<const emb_t**>(model_comm_buffer.data());
    const int* local_ev_offset_list_ptr = d_local_ev_size_offset.data<int>();
    const uint32_t* corrdinate_key_ptr = corrdinate_key.data<uint32_t>();
    const float* corrdinate_sp_weight_ptr = coordinate_sp_weight.data<float>();
    const uint32_t* sorted_bucket_id_list_ptr = sorted_bucket_id_list.data<uint32_t>();
    const uint32_t* coordinate_wgrad_dst_idx_ptr = coordinate_wgrad_dst_idx.data<uint32_t>();
    auto partial_grad_ev_ptr = partial_grad_ev_.data<float>();
    auto partial_key_ptr = partial_key_.data<uint32_t>();
    auto partial_ev_length_ptr = partial_ev_length_.data<int32_t>();
    auto partial_dst_offset_array_ptr = partial_dst_offset_array_.data<uint32_t>();
    float* grad_ev_ptr = grad_ev_.data<float>();

    auto multi_to_one_desc_first_stage = make_MultiToOne_reduce_weight<emb_t, float>(
        num_model_key, [=] __device__(int i) { return corrdinate_key_ptr[i]; },
        [=] __device__(int i) {
          uint32_t src_index = sorted_bucket_id_list_ptr[i];
          int embedding_id = src_index / batch_size;
          return local_ev_offset_list_ptr[embedding_id + 1] -
                 local_ev_offset_list_ptr[embedding_id];
        },
        [=] __device__(int i) {
          auto tmp_index = coordinate_wgrad_dst_idx_ptr[i];
          return unique_dst_idx_ptr[tmp_index + 1] - unique_dst_idx_ptr[tmp_index];
        },
        [=] __device__(int i) { return coordinate_wgrad_dst_idx_ptr[i]; },

        [=] __device__(int i) {
          uint32_t src_index = sorted_bucket_id_list_ptr[i];
          int embedding_id = src_index / batch_size;
          int batch_id = src_index % batch_size;
          int gpu_id = batch_id / batch_size_per_gpu;
          int local_batch_id = batch_id % batch_size_per_gpu;
          int ev_size =
              local_ev_offset_list_ptr[embedding_id + 1] - local_ev_offset_list_ptr[embedding_id];
          return model_comm_buffer_ptr[gpu_id] +
                 batch_size_per_gpu * local_ev_offset_list_ptr[embedding_id] +
                 local_batch_id * ev_size;
        },

        [=] __device__(int i) {
          auto tmp_index = coordinate_wgrad_dst_idx_ptr[i];
          return grad_ev_ptr + unique_dst_idx_ptr[tmp_index];
        },
        [=] __device__(int i) { return corrdinate_sp_weight_ptr[i]; });

    auto multi_to_one_desc_second_stage = make_MultiToOne_reduce_weight<float, float>(
        num_model_key, [=] __device__(int i) { return partial_key_ptr[i]; },
        [=] __device__(int i) { return partial_ev_length_ptr[i]; },
        [=] __device__(int i) {
          auto tmp_index = partial_dst_offset_array_ptr[i];
          return unique_dst_idx_ptr[tmp_index + 1] - unique_dst_idx_ptr[tmp_index];
        },
        [=] __device__(int i) { return 1; },

        [=] __device__(int i) { return partial_grad_ev_ptr + i * max_ev_size; },

        [=] __device__(int i) {
          auto tmp_index = partial_dst_offset_array_ptr[i];
          return grad_ev_ptr + unique_dst_idx_ptr[tmp_index];
        },
        [=] __device__(int i) { return 1.0; });

    multi_to_one_reduce_weight(multi_to_one_desc_first_stage, multi_to_one_desc_second_stage,
                               (float*)partial_grad_ev_.data(), (uint32_t*)partial_key_.data(),
                               (int*)partial_ev_length_.data(),
                               (uint32_t*)partial_dst_offset_array_.data(), num_sms_, max_ev_size,
                               stream);
  });

  *grad_ev = grad_ev_;
}
}  // namespace embedding
