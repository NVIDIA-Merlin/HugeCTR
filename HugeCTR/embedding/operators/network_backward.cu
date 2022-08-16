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
#include "HugeCTR/embedding/common.hpp"
#include "HugeCTR/include/utils.hpp"
#include "generic_lookup.cuh"
#include "network_backward.hpp"
namespace embedding {

using namespace core;

void NetworkBackward::compute(const Tensor &top_grad, const Tensor &d_ev_size_offset,
                              const Tensor &gpu_idx_offset, const TensorList &global_ev_offset,
                              const Tensor &network_idx, const Tensor &network_offset,
                              const Tensor &network_dst, TensorList &network_comm_buffer,
                              int batch_size, int max_ev_size) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  int batch_size_per_gpu = batch_size / num_gpus_;

  DISPATCH_FLOAT_AND_HALF_FUNCTION(top_grad.dtype().type(), emb_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION(network_comm_buffer.dtype().type(), dst_emb_t, [&] {
      auto stream = core_->get_local_gpu()->get_stream();

      auto get_decompress_network_dst =
          [dst_ptr = network_dst.get<int>(), offset_ptr = network_offset.get<int>(),
           num_offset =
               static_cast<int>(network_offset.get_num_elements())] __device__(int32_t index) {
            int i = binary_search_index_lower_bound(offset_ptr, num_offset, index);
            return dst_ptr[i];
          };
      LambdaIterator<int, int32_t, decltype(get_decompress_network_dst)> index_iter(
          get_decompress_network_dst, static_cast<int>(network_idx.get_num_elements()));

      auto get_offset = [] __device__(int32_t index) { return index; };
      LambdaIterator<int, int32_t, decltype(get_offset)> offset_iter(
          get_offset, static_cast<int>(network_idx.get_num_elements() + 1));

      ArrayView<int, RestrictPtrTraits, int32_t> dst_iter{
          network_idx.get(), static_cast<int>(network_idx.get_num_elements())};

      RaggedEmbForwardResultView<emb_t, RestrictPtrTraits, int32_t> src_buffer_iter{
          top_grad.get(), d_ev_size_offset.get<int>(), batch_size_per_gpu};

      RaggedNetworkBufferView<dst_emb_t, RestrictPtrTraits, int32_t> dst_buffer_iter{
          network_comm_buffer.get(), gpu_idx_offset.get<int>(), global_ev_offset.get<int>(),
          num_gpus_, batch_size};
      generic_lookup(index_iter, offset_iter, dst_iter, src_buffer_iter, dst_buffer_iter,
                     max_ev_size, stream);
    });
  });
}
}  // namespace embedding
