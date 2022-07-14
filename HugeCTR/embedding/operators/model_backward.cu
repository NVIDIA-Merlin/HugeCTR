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
  CudaDeviceContext ctx(core_->get_device_id());

  auto buffer_ptr = GetBuffer(core);
  grad_ev_ = buffer_ptr->reserve({universal_batch_size, max_unique_key_ev_buffer_size},
                                 DeviceType::GPU, HugeCTR::TensorScalarType::Float32);
  buffer_ptr->allocate();
}

void ModelBackward::compute(const TensorList &model_comm_buffer, const Tensor &unique_dst_idx,
                            const Tensor &sorted_bucket_id_list,
                            const Tensor &sorted_bucket_id_offset, size_t num_unique_key,
                            const Tensor &d_local_ev_size_offset, int batch_size, Tensor *grad_ev) {
  CudaDeviceContext ctx(core_->get_device_id());
  int batch_size_per_gpu = batch_size / num_gpus_;

  DISPATCH_FLOAT_AND_HALF_FUNCTION(model_comm_buffer.dtype().type(), emb_t, [&] {
    auto stream = core_->get_local_gpu()->get_stream();

    ArrayView<uint32_t, RestrictPtrTraits, int32_t> sorted_bucket_id_ref{
        sorted_bucket_id_list.get(),
        static_cast<int32_t>(sorted_bucket_id_list.get_num_elements())};

    ArrayView<uint32_t, RestrictPtrTraits, int32_t> sorted_bucket_id_offset_ref{
        sorted_bucket_id_offset.get(), static_cast<int32_t>(num_unique_key) + 1};

    auto get_counter = [] __device__(int32_t index) -> uint32_t { return 1; };
    LambdaIterator<uint32_t, int32_t, decltype(get_counter)> counter_iter{
        get_counter, static_cast<int32_t>(num_unique_key)};

    auto get_output_idx = [] __device__(int32_t index) -> uint32_t { return index; };
    LambdaIterator<uint32_t, int32_t, decltype(get_output_idx)> output_counting_iter{
        get_output_idx, static_cast<int32_t>(num_unique_key)};

    RaggedModelBufferView<emb_t, RestrictPtrTraits, int32_t> model_comm_buffer_iterator{
        model_comm_buffer.get(), d_local_ev_size_offset.get<int>(), num_gpus_, batch_size};

    RaggedGradBufferView<float, RestrictPtrTraits, int32_t> grad_ev_iterator{
        grad_ev_.get(), unique_dst_idx.get<uint32_t>()};

    generic_lookup(sorted_bucket_id_ref, sorted_bucket_id_offset_ref, counter_iter,
                   output_counting_iter, model_comm_buffer_iterator, grad_ev_iterator, stream);
  });

  *grad_ev = grad_ev_;
}

DPLocalReduce::DPLocalReduce(std::shared_ptr<CoreResourceManager> core, int num_gpus,
                             int num_local_embedding, const std::vector<int> &h_local_hotness_list,
                             const std::vector<int> &h_local_ev_size_list, int universal_batch_size)
    : core_(core), num_gpus_(num_gpus), num_local_embedding_(num_local_embedding) {
  CudaDeviceContext ctx(core_->get_device_id());
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
                                 HugeCTR::TensorScalarType::Float32);
  buffer_ptr->allocate();
}

void DPLocalReduce::compute(const Tensor &top_grad, const Tensor &unique_dst_idx,
                            const Tensor &sorted_bucket_id_list,
                            const Tensor &sorted_bucket_id_offset, size_t num_unique_key,
                            const Tensor &d_ev_size_offset, int batch_size, Tensor *grad_ev) {
  CudaDeviceContext ctx(core_->get_device_id());
  int batch_size_per_gpu = batch_size / num_gpus_;

  DISPATCH_FLOAT_AND_HALF_FUNCTION(top_grad.dtype().type(), emb_t, [&] {
    auto stream = core_->get_local_gpu()->get_stream();

    ArrayView<uint32_t, RestrictPtrTraits, int64_t> sorted_bucket_id_ref{
        sorted_bucket_id_list.get(), sorted_bucket_id_list.get_num_elements()};

    ArrayView<uint32_t, RestrictPtrTraits, int32_t> sorted_bucket_id_offset_ref{
        sorted_bucket_id_offset.get(), static_cast<int32_t>(num_unique_key) + 1};

    auto get_counter = [] __device__(int32_t index) -> uint32_t { return 1; };
    LambdaIterator<uint32_t, int32_t, decltype(get_counter)> counter_iter{
        get_counter, static_cast<int32_t>(num_unique_key)};

    auto get_output_idx = [] __device__(int32_t index) -> uint32_t { return index; };
    LambdaIterator<uint32_t, int32_t, decltype(get_output_idx)> output_counting_iter{
        get_output_idx, static_cast<int32_t>(num_unique_key)};

    RaggedEmbForwardResultView<emb_t, RestrictPtrTraits, int32_t> top_grad_iterator{
        top_grad.get(), d_ev_size_offset.get<int>(), batch_size_per_gpu};

    RaggedGradBufferView<float, RestrictPtrTraits, int32_t> grad_ev_iterator{
        grad_ev_.get(), unique_dst_idx.get<uint32_t>()};

    generic_lookup(sorted_bucket_id_ref, sorted_bucket_id_offset_ref, counter_iter,
                   output_counting_iter, top_grad_iterator, grad_ev_iterator, stream);
  });

  *grad_ev = grad_ev_;
}
}  // namespace embedding
