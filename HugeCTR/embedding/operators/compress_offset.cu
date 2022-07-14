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
#include "compress_offset.hpp"

namespace embedding {


__global__ void compress_offset_kernel(const uint32_t* offset, int num, int stride,
                                       uint32_t* compressed_offset) {
  int thread_cnt = blockDim.x * blockDim.y;
  
  for (int tid = threadIdx.x + threadIdx.y * blockDim.x; tid < num; tid += thread_cnt) {
    compressed_offset[tid] = offset[tid * stride];
  }
}


CompressOffset::CompressOffset(std::shared_ptr<CoreResourceManager> core, int num_compressed_offset)
    : core_(core), num_compressed_offset_(num_compressed_offset) {
  CudaDeviceContext ctx(core_->get_device_id());

  auto buffer_ptr = GetBuffer(core);
  compressed_offset_ =
      buffer_ptr->reserve({num_compressed_offset}, DeviceType::GPU, TensorScalarType::UInt32);
  buffer_ptr->allocate();
}

void CompressOffset::compute(const Tensor& offset, int stride, Tensor* compressed_offset) {
  CudaDeviceContext ctx(core_->get_device_id());
  auto stream = core_->get_local_gpu()->get_stream();

  dim3 block_size(32, 8);

  compress_offset_kernel<<<1, block_size, 0, stream>>>(offset.get<uint32_t>(),
                                                       num_compressed_offset_, stride,
                                                       compressed_offset_.get<uint32_t>());

  *compressed_offset = compressed_offset_;
}

}