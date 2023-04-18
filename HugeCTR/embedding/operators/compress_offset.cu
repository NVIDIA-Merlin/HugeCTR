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

#include <embedding/common.hpp>
#include <embedding/operators/compress_offset.hpp>
#include <embedding/operators/generic_lookup.cuh>
#include <utils.hpp>

namespace embedding {

template <typename offset_t>
__global__ void compress_offset_kernel(const offset_t *offset, int num, int stride,
                                       offset_t *compressed_offset) {
  CUDA_1D_KERNEL_LOOP(i, num) { compressed_offset[i] = offset[i * stride]; }
}

CompressOffset::CompressOffset(std::shared_ptr<CoreResourceManager> core, int num_compressed_offset,
                               core23::DataType type)
    : core_(core), num_compressed_offset_(num_compressed_offset) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  compressed_offset_ = core23::Tensor(params.shape({num_compressed_offset}).data_type(type));
}

void CompressOffset::compute(const core23::Tensor &offset, int batch_size,
                             core23::Tensor *compressed_offset) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  auto stream = core_->get_local_gpu()->get_stream();
  HCTR_CHECK(offset.data_type() == compressed_offset_.data_type());

  DISPATCH_INTEGRAL_FUNCTION_CORE23(offset.data_type().type(), offset_t, [&] {
    dim3 block_size(256);

    compress_offset_kernel<<<1, block_size, 0, stream>>>(offset.data<offset_t>(),
                                                         num_compressed_offset_, batch_size,
                                                         compressed_offset_.data<offset_t>());
  });

  *compressed_offset = compressed_offset_;
}

AverageCombiner::AverageCombiner(std::shared_ptr<CoreResourceManager> core, int num_gpus,
                                 int num_local_embedding, const std::vector<int> &ev_size_list,
                                 int universal_batch_size)
    : core_(core), num_gpus_(num_gpus), num_local_embedding_(num_local_embedding) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  int num_ev_size_sum = std::accumulate(ev_size_list.begin(), ev_size_list.end(), 0);

  // TODO: The float emb vec can be reduced to the scale of num of local embedding
  float_emb_vec_ = core23::Tensor(params.shape({universal_batch_size / num_gpus, num_ev_size_sum})
                                      .data_type(core23::ScalarType::Float));
}

void AverageCombiner::compute_feature_major(const core23::Tensor &dp_num_keys_per_bucket,
                                            const core23::Tensor &src_emb_vec,
                                            const core23::Tensor &d_local_embedding_list,
                                            const core23::Tensor &d_combiner_list,
                                            const core23::Tensor &d_ev_size_offset, int batch_size,
                                            int max_ev_size) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  int gpu_id = core_->get_global_gpu_id();
  auto stream = core_->get_local_gpu()->get_stream();
  int batch_size_per_gpu = batch_size / num_gpus_;

  DISPATCH_INTEGRAL_FUNCTION_CORE23(dp_num_keys_per_bucket.data_type().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(src_emb_vec.data_type().type(), emb_t, [&] {
      const offset_t *dp_num_keys_per_bucket_ptr = dp_num_keys_per_bucket.data<offset_t>();
      const int *local_embedding_ptr = d_local_embedding_list.data<int>();
      const int *d_ev_size_offset_ptr = d_ev_size_offset.data<int>();
      const emb_t *top_grad_ptr = src_emb_vec.data<emb_t>();
      const char *combiner_ptr = d_combiner_list.data<char>();
      float *float_emb_vec_ptr = float_emb_vec_.data<float>();
      int gpu_id = core_->get_global_gpu_id();

      auto multi_to_one_desc = make_MultiToOne<emb_t, float>(
          batch_size_per_gpu * num_local_embedding_, [=] __device__(int i) { return i; },
          [=] __device__(int i) {
            int bid = i % batch_size_per_gpu;
            int lookup_id = local_embedding_ptr[i / batch_size_per_gpu];

            if (combiner_ptr[lookup_id] == static_cast<char>(Combiner::Average)) {
              int idx = batch_size_per_gpu * lookup_id + bid;
              return static_cast<int>(dp_num_keys_per_bucket_ptr[idx]);
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

void AverageCombiner::compute_batch_major(const core23::Tensor &dp_num_keys_per_bucket,
                                          const core23::Tensor &src_emb_vec,
                                          const core23::Tensor &d_local_embedding_list,
                                          const core23::Tensor &d_combiner_list,
                                          const core23::Tensor &d_ev_size_offset, int batch_size,
                                          int max_ev_size, int num_lookup) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  int gpu_id = core_->get_global_gpu_id();
  auto stream = core_->get_local_gpu()->get_stream();
  int batch_size_per_gpu = batch_size / num_gpus_;

  DISPATCH_INTEGRAL_FUNCTION_CORE23(dp_num_keys_per_bucket.data_type().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(src_emb_vec.data_type().type(), emb_t, [&] {
      const offset_t *dp_num_keys_per_bucket_ptr = dp_num_keys_per_bucket.data<offset_t>();
      const int *local_embedding_ptr = d_local_embedding_list.data<int>();
      const int *d_ev_size_offset_ptr = d_ev_size_offset.data<int>();
      const emb_t *top_grad_ptr = src_emb_vec.data<emb_t>();
      const char *combiner_ptr = d_combiner_list.data<char>();
      float *float_emb_vec_ptr = float_emb_vec_.data<float>();
      int gpu_id = core_->get_global_gpu_id();

      auto multi_to_one_desc = make_MultiToOne<emb_t, float>(
          batch_size_per_gpu * num_lookup, [=] __device__(int i) { return i; },
          [=] __device__(int i) {
            int bid = i % batch_size_per_gpu;
            int lookup_id = local_embedding_ptr[i / batch_size_per_gpu];

            if (combiner_ptr[lookup_id] == static_cast<char>(Combiner::Average)) {
              int idx = batch_size_per_gpu * lookup_id + bid;
              return static_cast<int>(dp_num_keys_per_bucket_ptr[idx]);
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

            int batch_ev_offset = d_ev_size_offset_ptr[num_lookup] * bid;
            ;
            int ev_offset = d_ev_size_offset_ptr[lookup_id];
            return top_grad_ptr + batch_ev_offset + ev_offset;
          },
          [=] __device__(int i) {
            int bid = i % batch_size_per_gpu;
            int lookup_id = local_embedding_ptr[i / batch_size_per_gpu];

            int batch_ev_offset = d_ev_size_offset_ptr[num_lookup] * bid;
            int ev_offset = d_ev_size_offset_ptr[lookup_id];
            return float_emb_vec_ptr + batch_ev_offset + ev_offset;
          });
      copy_multi_to_one(multi_to_one_desc, max_ev_size, stream);
    });
  });
}

}  // namespace embedding
