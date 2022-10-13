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
#include "compress_offset.hpp"
#include "generic_lookup.cuh"
namespace embedding {

__global__ void compress_offset_kernel(const uint32_t *offset, int num, int stride,
                                       uint32_t *compressed_offset) {
  int thread_cnt = blockDim.x * blockDim.y;

  for (int tid = threadIdx.x + threadIdx.y * blockDim.x; tid < num; tid += thread_cnt) {
    compressed_offset[tid] = offset[tid * stride];
  }
}

CompressOffset::CompressOffset(std::shared_ptr<CoreResourceManager> core, int num_compressed_offset)
    : core_(core), num_compressed_offset_(num_compressed_offset) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());

  auto buffer_ptr = GetBuffer(core);
  compressed_offset_ =
      buffer_ptr->reserve({num_compressed_offset}, DeviceType::GPU, TensorScalarType::UInt32);
  buffer_ptr->allocate();
}

void CompressOffset::compute(const Tensor &offset, int stride, Tensor *compressed_offset) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  auto stream = core_->get_local_gpu()->get_stream();

  dim3 block_size(32, 8);

  compress_offset_kernel<<<1, block_size, 0, stream>>>(
      offset.get<uint32_t>(), num_compressed_offset_, stride, compressed_offset_.get<uint32_t>());

  *compressed_offset = compressed_offset_;
}

AverageCombiner::AverageCombiner(std::shared_ptr<CoreResourceManager> core, int num_gpus,
                                 int num_local_embedding, const std::vector<int> &ev_size_list,
                                 int universal_batch_size)
    : core_(core), num_gpus_(num_gpus), num_local_embedding_(num_local_embedding) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());

  int num_ev_size_sum = std::accumulate(ev_size_list.begin(), ev_size_list.end(), 0);
  auto buffer_ptr = GetBuffer(core);
  // TODO: The float emb vec can be reduced to the scale of num of local embedding
  float_emb_vec_ = buffer_ptr->reserve({universal_batch_size / num_gpus, num_ev_size_sum},
                                       DeviceType::GPU, TensorScalarType::Float32);
  buffer_ptr->allocate();
}

void AverageCombiner::compute(const Tensor &bucket_range, const Tensor &top_grad,
                              const Tensor &d_local_embedding_list, const Tensor &d_combiner_list,
                              const Tensor &d_ev_size_offset, int batch_size, int max_ev_size) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  int gpu_id = core_->get_global_gpu_id();
  auto stream = core_->get_local_gpu()->get_stream();
  int batch_size_per_gpu = batch_size / num_gpus_;

  DISPATCH_INTEGRAL_FUNCTION(bucket_range.dtype().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION(top_grad.dtype().type(), emb_t, [&] {
      const offset_t *bucket_range_ptr = bucket_range.get<offset_t>();
      const int *local_embedding_ptr = d_local_embedding_list.get<int>();
      const int *d_ev_size_offset_ptr = d_ev_size_offset.get<int>();
      const emb_t *top_grad_ptr = top_grad.get<emb_t>();
      const char *combiner_ptr = d_combiner_list.get<char>();
      float *float_emb_vec_ptr = float_emb_vec_.get<float>();
      int gpu_id = core_->get_global_gpu_id();

      auto multi_to_one_desc = make_MultiToOne<emb_t, float>(
          batch_size_per_gpu * num_local_embedding_, [=] __device__(int i) { return i; },
          [=] __device__(int i) {
            int bid = i % batch_size_per_gpu;
            int lookup_id = local_embedding_ptr[i / batch_size_per_gpu];

            if (combiner_ptr[lookup_id] == static_cast<char>(Combiner::Average)) {
              int start = batch_size * lookup_id + gpu_id * batch_size_per_gpu + bid;
              return static_cast<int>(bucket_range_ptr[start + 1] - bucket_range_ptr[start]);
            } else {
              return 1;
            }
          },
          [=] __device__(int i) {
            int lookup_id = local_embedding_ptr[i / batch_size_per_gpu];
            return d_ev_size_offset_ptr[lookup_id + 1] - d_ev_size_offset_ptr[lookup_id];
          },
          [=] __device__(int i) {
            int bid = i % batch_size_per_gpu;
            int lookup_id = local_embedding_ptr[i / batch_size_per_gpu];

            int ev_offset = d_ev_size_offset_ptr[lookup_id] * batch_size_per_gpu;
            int ev_size = d_ev_size_offset_ptr[lookup_id + 1] - d_ev_size_offset_ptr[lookup_id];
            return top_grad_ptr + ev_offset + bid * ev_size;
          },
          [=] __device__(int i) {
            int bid = i % batch_size_per_gpu;
            int lookup_id = local_embedding_ptr[i / batch_size_per_gpu];

            int ev_offset = d_ev_size_offset_ptr[lookup_id] * batch_size_per_gpu;
            int ev_size = d_ev_size_offset_ptr[lookup_id + 1] - d_ev_size_offset_ptr[lookup_id];
            return float_emb_vec_ptr + ev_offset + bid * ev_size;
          });
      copy_multi_to_one(multi_to_one_desc, max_ev_size, stream);
    });
  });
}

}  // namespace embedding