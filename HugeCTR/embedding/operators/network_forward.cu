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
#include "network_forward.hpp"
namespace embedding {
using namespace core;

NetworkForward::NetworkForward(std::shared_ptr<CoreResourceManager> core, int num_gpus)
    : core_(core), num_gpus_(num_gpus) {}

void NetworkForward::compute(const TensorList &network_comm_buffer, const Tensor &gpu_idx_offset,
                             const TensorList &global_ev_offset, const Tensor &network_idx,
                             const Tensor &network_offset, const Tensor &network_dst,
                             Tensor &output_buffer, const Tensor &d_ev_size_offset, int batch_size,
                             int max_ev_size) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  int batch_size_per_gpu = batch_size / num_gpus_;
  DISPATCH_FLOAT_AND_HALF_FUNCTION(network_comm_buffer.dtype().type(), emb_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION(output_buffer.dtype().type(), dst_emb_t, [&] {
      auto stream = core_->get_local_gpu()->get_stream();

      ArrayView<int, RestrictPtrTraits, int32_t> index_iter{
          network_idx.get(), static_cast<int32_t>(network_idx.get_num_elements())};

      ArrayView<int, RestrictPtrTraits, int32_t> offset_iter{
          network_offset.get(), static_cast<int32_t>(network_offset.get_num_elements())};

      ArrayView<int, RestrictPtrTraits, int32_t> dst_iter{
          network_dst.get(), static_cast<int32_t>(network_dst.get_num_elements())};

      RaggedNetworkBufferView<emb_t, RestrictPtrTraits, int32_t> src_buffer_iter{
          network_comm_buffer.get(), gpu_idx_offset.get<int>(), global_ev_offset.get<int>(),
          num_gpus_, batch_size};

      RaggedEmbForwardResultView<dst_emb_t, RestrictPtrTraits, int32_t> dst_buffer_iter{
          output_buffer.get(), d_ev_size_offset.get<int>(), batch_size_per_gpu};

      generic_lookup(index_iter, offset_iter, dst_iter, src_buffer_iter, dst_buffer_iter,
                     max_ev_size, stream);
    });
  });
}
}  // namespace embedding
