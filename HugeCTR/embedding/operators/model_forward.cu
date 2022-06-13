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
#include "HugeCTR/include/utils.cuh"
#include "HugeCTR/include/utils.hpp"
#include "generic_lookup.cuh"
#include "model_forward.hpp"
namespace embedding {
using HugeCTR::CudaDeviceContext;

DPModelForward::DPModelForward(std::shared_ptr<CoreResourceManager> core, int num_gpus,
                               int num_embedding, int num_local_embedding)
    : core_(core),
      num_gpus_(num_gpus),
      num_embedding_(num_embedding),
      num_local_embedding_(num_local_embedding) {}

void DPModelForward::compute(const TensorList &dp_ev, const Tensor &dp_offset, const Tensor &dp_dst,
                             Tensor &output_buffer, const Tensor &d_local_ev_size_list,
                             const Tensor &d_local_combiner_list, const Tensor &d_ev_size_offset,
                             int batch_size) const {
  CudaDeviceContext ctx(core_->get_device_id());
  int batch_size_per_gpu = batch_size / core_->get_global_gpu_count();

  DISPATCH_FLOAT_AND_HALF_FUNCTION(output_buffer.dtype().type(), emb_t, [&] {
    auto stream = core_->get_local_gpu()->get_stream();
    auto counting_iter = [] __device__(int32_t index) { return index; };
    LambdaIterator<uint32_t, int32_t, decltype(counting_iter)> index_iter{
        counting_iter, static_cast<int32_t>(dp_ev.get_num_elements())};

    ArrayView<uint32_t, RestrictPtrTraits, int32_t> offset_iter{
        dp_offset.get(), batch_size_per_gpu * num_local_embedding_ + 1};

    auto get_counter = [batch_size_per_gpu, combiner_ptr = d_local_combiner_list.get<char>(),
                        offset_ptr =
                            dp_offset.get<uint32_t>()] __device__(int32_t index) -> uint32_t {
      int embedding_id = index / batch_size_per_gpu;
      return combiner_ptr[embedding_id] == static_cast<char>(Combiner::Average)
                 ? offset_ptr[index + 1] - offset_ptr[index]
                 : 1;
    };
    LambdaIterator<uint32_t, int32_t, decltype(get_counter)> counter_iter{
        get_counter, batch_size_per_gpu * num_local_embedding_};

    ArrayView<uint32_t, RestrictPtrTraits, int32_t> dst_iter{
        dp_dst.get(), batch_size_per_gpu * num_local_embedding_};

    RaggedLookupResultView<float, RestrictPtrTraits, int32_t> src_buffer_iter{
        dp_ev.get(), dp_offset.get<uint32_t>(), batch_size_per_gpu * num_local_embedding_ + 1,
        d_local_ev_size_list.get<int>(), batch_size_per_gpu};

    RaggedEmbForwardResultView<emb_t, RestrictPtrTraits, int32_t> dst_buffer_iter{
        output_buffer.get(), d_ev_size_offset.get<int>(), batch_size_per_gpu};

    generic_lookup(index_iter, offset_iter, counter_iter, dst_iter, src_buffer_iter,
                   dst_buffer_iter, stream);
  });
}

ModelForward::ModelForward(std::shared_ptr<CoreResourceManager> core, int num_gpus,
                           const std::vector<int> &local_embedding_list)
    : core_(core), num_gpus_(num_gpus), num_local_embedding_(local_embedding_list.size()) {}

void ModelForward::compute(const TensorList &mp_ev, const Tensor &model_offset,
                           TensorList &model_comm_buffer, const Tensor &d_local_ev_size_list,
                           const Tensor &d_local_ev_size_offset, int batch_size) {
  CudaDeviceContext ctx(core_->get_device_id());
  int batch_size_per_gpu = batch_size / core_->get_global_gpu_count();

  if (num_local_embedding_ > 0) {
    DISPATCH_FLOAT_AND_HALF_FUNCTION(model_comm_buffer.dtype().type(), emb_t, [&] {
      auto stream = core_->get_local_gpu()->get_stream();
      auto counting_iter = [] __device__(int32_t index) { return index; };
      LambdaIterator<uint32_t, int32_t, decltype(counting_iter)> index_iter{
          counting_iter, static_cast<int32_t>(mp_ev.get_num_elements())};

      ArrayView<uint32_t, RestrictPtrTraits, int32_t> offset_iter{
          model_offset.get(), batch_size * num_local_embedding_ + 1};

      auto get_counter = [] __device__(int32_t index) -> uint32_t { return 1; };
      LambdaIterator<uint32_t, int32_t, decltype(get_counter)> counter_iter{
          get_counter, batch_size * num_local_embedding_};

      LambdaIterator<uint32_t, int32_t, decltype(counting_iter)> dst_iter{
          counting_iter, batch_size * num_local_embedding_};

      RaggedLookupResultView<float, RestrictPtrTraits, int32_t> src_buffer_iter{
          mp_ev.get(), model_offset.get<uint32_t>(), batch_size * num_local_embedding_ + 1,
          d_local_ev_size_list.get<int>(), batch_size};

      RaggedModelBufferView<emb_t, RestrictPtrTraits, int32_t> dst_buffer_iter{
          model_comm_buffer.get(), d_local_ev_size_offset.get<int>(), num_gpus_, batch_size};
      generic_lookup(index_iter, offset_iter, counter_iter, dst_iter, src_buffer_iter,
                     dst_buffer_iter, stream);
    });
  }
}

}  // namespace embedding
