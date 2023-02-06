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
#include "index_calculation.hpp"
#include "model_backward.hpp"
#include "model_forward.hpp"
#include "multi_to_one_reduce.cuh"
#include "multi_to_one_reduce_v2.cuh"

namespace embedding {

void LocalReduce::init(std::shared_ptr<CoreResourceManager> core,
                       const embedding::KernelParams& kernel_params, int max_ev_size,
                       size_t max_input_num) {
  HugeCTR::CudaDeviceContext ctx(core->get_device_id());

  this->core_ = core;
  this->kernel_params_ = kernel_params;

  int num_sms = kernel_params.num_sms;

  auto buffer_ptr = GetBuffer(core);
  this->partial_reduce_result_.partial_wgrad =
      buffer_ptr->reserve({num_sms * 4 * max_ev_size}, DeviceType::GPU, TensorScalarType::Float32);
  this->partial_reduce_result_.partial_keys =
      buffer_ptr->reserve({num_sms * 4}, DeviceType::GPU, TensorScalarType::UInt32);
  this->partial_reduce_result_.partial_ev_length =
      buffer_ptr->reserve({num_sms * 4}, DeviceType::GPU, TensorScalarType::Int32);
  this->partial_reduce_result_.partial_dst_offset_array =
      buffer_ptr->reserve({num_sms * 4}, DeviceType::GPU, TensorScalarType::UInt32);
  int max_partial_num = (max_input_num - 1) / EV_NUM + 1;
  this->partial_reduce_result_.partial_wgrad_new = buffer_ptr->reserve(
      {max_partial_num * max_ev_size}, DeviceType::GPU, TensorScalarType::Float32);
  this->partial_reduce_result_.partial_ev_length_new =
      buffer_ptr->reserve({max_partial_num}, DeviceType::GPU, TensorScalarType::Int32);
  this->partial_reduce_result_.partial_dst_id_array_new =
      buffer_ptr->reserve({max_partial_num}, DeviceType::GPU, TensorScalarType::UInt32);
  this->partial_reduce_result_.max_input_num = max_input_num;

  buffer_ptr->allocate();
}

void LocalReduce::local_reduce(const ReductionIndices& reduction_indices,
                               const ModelCommBuffer& src_buffer, Wgrad& wgrad, int batch_size) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  auto stream = core_->get_local_gpu()->get_stream();

  const auto& src_buffer_attr = src_buffer.attr;
  const auto& dst_attr = wgrad.attr;

  int batch_size_per_gpu = batch_size / src_buffer_attr.num_gpus;
  HCTR_CHECK_HINT(src_buffer_attr.layout == EmbeddingLayout::FeatureMajor,
                  "local reduce model comm buffer should be feature major");

  const int* src_id_to_ev_start_indices_ptr = src_buffer_attr.id_to_ev_start_indices.get<int>();
  const int* src_id_to_ev_size_ptr = reduction_indices.ev_sizes.get<int>();
  const uint32_t* src_ids_ptr = reduction_indices.src_ids.get<uint32_t>();

  const int* dst_table_id_to_ev_size_ptr = dst_attr.table_id_to_ev_size.get<int>();

  const int* dst_table_ids_ptr = wgrad.table_ids.get<int>();
  const uint32_t* dst_ev_start_indices_ptr = wgrad.ev_start_indices.get<uint32_t>();
  const uint32_t* dst_ids_ptr = reduction_indices.dst_ids.get<uint32_t>();
  float* dst_ptr = wgrad.data.get<float>();

  DISPATCH_FLOAT_AND_HALF_FUNCTION(src_buffer.attr.type.type(), emb_t, [&] {
    const emb_t** src_ptr = src_buffer.data.get<emb_t>();

    auto multi_to_one_desc_first_stage = make_MultiToOne_reduce_new<emb_t, float>(
        [=] __device__() { return reduction_indices.num_elements; },
        [=] __device__(int i) { return src_id_to_ev_size_ptr[i]; },
        [=] __device__(int i) { return dst_ids_ptr[i]; },
        [=] __device__(int i) {
          // model buffer bucket id layout:
          // gpu i:
          //   0, ..., batch_size_per_gpu | batch_size, ..., batch_size + batch_size_per_gpu | ... |
          //   i * batch_size, ..., i * batch_size + batch_size_per_gpu | ...
          uint32_t bucket_id = src_ids_ptr[i];
          int embedding_id = bucket_id / batch_size;
          int batch_id = bucket_id % batch_size;
          int gpu_id = batch_id / batch_size_per_gpu;
          int local_batch_id = batch_id % batch_size_per_gpu;
          int ev_size = src_id_to_ev_size_ptr[i];
          return src_ptr[gpu_id] +
                 batch_size_per_gpu * src_id_to_ev_start_indices_ptr[embedding_id] +
                 local_batch_id * ev_size;
        },
        [=] __device__(int i) {
          auto tmp_index = dst_ids_ptr[i];
          return dst_ptr + dst_ev_start_indices_ptr[tmp_index];
        });
    multi_to_one_reduce_v2(multi_to_one_desc_first_stage, reduction_indices, kernel_params_,
                           partial_reduce_result_, wgrad, src_buffer.attr.max_ev_size, stream);
  });
}

void dp_local_reduce_from_feature_major_top_grad(
    const KernelParams& kernel_params, const ReductionIndices& reduction_indices,
    const EmbeddingOutput& src_buffer, const Tensor& local_lookup_ids, int num_lookup, Wgrad& wgrad,
    PartialReduceResult& partial_reduce_result, int batch_size_per_gpu, int max_ev_size,
    cudaStream_t stream) {
  const auto& src_buffer_attr = src_buffer.attr;
  const auto& dst_attr = wgrad.attr;

  HCTR_CHECK_HINT(src_buffer_attr.layout == EmbeddingLayout::FeatureMajor,
                  "local reduce model comm buffer should be feature major");

  const int* local_lookup_ids_ptr = local_lookup_ids.get<int>();

  const int* src_id_to_ev_start_indices_ptr = src_buffer_attr.id_to_ev_start_indices.get<int>();
  const int* src_id_to_ev_size_ptr = reduction_indices.ev_sizes.get<int>();

  const uint32_t* src_ids_ptr = reduction_indices.src_ids.get<uint32_t>();

  const int* dst_table_id_to_ev_size_ptr = dst_attr.table_id_to_ev_size.get<int>();

  const int* dst_table_ids_ptr = wgrad.table_ids.get<int>();
  const uint32_t* dst_ev_start_indices_ptr = wgrad.ev_start_indices.get<uint32_t>();
  const uint32_t* dst_ids_ptr = reduction_indices.dst_ids.get<uint32_t>();
  float* dst_ptr = wgrad.data.get<float>();

  DISPATCH_FLOAT_AND_HALF_FUNCTION(src_buffer.attr.type.type(), emb_t, [&] {
    const emb_t* src_ptr = src_buffer.data.get<emb_t>();

    auto multi_to_one_desc_first_stage = make_MultiToOne_reduce<emb_t, float>(
        reduction_indices.num_elements, [=] __device__(int i) { return dst_ids_ptr[i]; },
        [=] __device__(int i) {
          return src_id_to_ev_size_ptr[i];
          ;
        },
        [=] __device__(int i) {
          return src_id_to_ev_size_ptr[i];
          ;
        },
        [=] __device__(int i) { return dst_ids_ptr[i]; },
        [=] __device__(int i) {
          // top grad buffer bucket id layout:
          // 0, ..., batch_size_per_gpu - 1 | batch_size_per_gpu, ..., 2 * batch_size_per_gpu - 1|
          // ...
          uint32_t bucket_id = src_ids_ptr[i];
          int local_lookup_id = bucket_id / batch_size_per_gpu;
          int lookup_id = local_lookup_ids_ptr[local_lookup_id];

          int batch_id = bucket_id % batch_size_per_gpu;
          int ev_size = src_id_to_ev_size_ptr[i];
          return src_ptr + batch_size_per_gpu * src_id_to_ev_start_indices_ptr[lookup_id] +
                 batch_id * ev_size;
        },
        [=] __device__(int i) {
          auto tmp_index = dst_ids_ptr[i];
          return dst_ptr + dst_ev_start_indices_ptr[tmp_index];
        });

    multi_to_one_reduce(multi_to_one_desc_first_stage, reduction_indices, kernel_params,
                        partial_reduce_result, wgrad, src_buffer.attr.max_ev_size, stream);
  });
}

void dp_local_reduce_from_batch_major_top_grad(
    const KernelParams& kernel_params, const ReductionIndices& reduction_indices,
    const EmbeddingOutput& src_buffer, const Tensor& local_lookup_ids, int num_lookup, Wgrad& wgrad,
    PartialReduceResult& partial_reduce_result, int batch_size_per_gpu, int max_ev_size,
    cudaStream_t stream) {
  const auto& src_buffer_attr = src_buffer.attr;
  const auto& dst_attr = wgrad.attr;

  HCTR_CHECK_HINT(src_buffer_attr.layout == EmbeddingLayout::BatchMajor,
                  "local reduce model comm buffer should be batch major");

  const int* local_lookup_ids_ptr = local_lookup_ids.get<int>();

  const int* src_id_to_ev_start_indices_ptr = src_buffer_attr.id_to_ev_start_indices.get<int>();
  const int* src_id_to_ev_size_ptr = reduction_indices.ev_sizes.get<int>();

  const uint32_t* src_ids_ptr = reduction_indices.src_ids.get<uint32_t>();

  const int* dst_table_id_to_ev_size_ptr = dst_attr.table_id_to_ev_size.get<int>();

  const int* dst_table_ids_ptr = wgrad.table_ids.get<int>();
  const uint32_t* dst_ev_start_indices_ptr = wgrad.ev_start_indices.get<uint32_t>();
  const uint32_t* dst_ids_ptr = reduction_indices.dst_ids.get<uint32_t>();
  float* dst_ptr = wgrad.data.get<float>();

  DISPATCH_FLOAT_AND_HALF_FUNCTION(src_buffer.attr.type.type(), emb_t, [&] {
    const emb_t* src_ptr = src_buffer.data.get<emb_t>();

    auto multi_to_one_desc_first_stage = make_MultiToOne_reduce<emb_t, float>(
        reduction_indices.num_elements, [=] __device__(int i) { return dst_ids_ptr[i]; },
        [=] __device__(int i) {
          return src_id_to_ev_size_ptr[i];
          ;
        },
        [=] __device__(int i) {
          return src_id_to_ev_size_ptr[i];
          ;
        },
        [=] __device__(int i) { return dst_ids_ptr[i]; },
        [=] __device__(int i) {
          // top grad buffer bucket id layout:
          // 0, ..., num_lookup - 1 | num_lookup, ... | ... batch_size_per_gpu * num_lookup - 1
          uint32_t bucket_id = src_ids_ptr[i];
          int batch_id = bucket_id % batch_size_per_gpu;
          int lookup_id = local_lookup_ids_ptr[bucket_id / batch_size_per_gpu];

          return src_ptr + batch_id * src_id_to_ev_start_indices_ptr[num_lookup] +
                 src_id_to_ev_start_indices_ptr[lookup_id];
        },
        [=] __device__(int i) {
          auto tmp_index = dst_ids_ptr[i];
          return dst_ptr + dst_ev_start_indices_ptr[tmp_index];
        });

    multi_to_one_reduce(multi_to_one_desc_first_stage, reduction_indices, kernel_params,
                        partial_reduce_result, wgrad, src_buffer.attr.max_ev_size, stream);
  });
}

void LocalReduce::local_reduce(const ReductionIndices& reduction_indices,
                               const EmbeddingOutput& src_buffer, Wgrad& wgrad,
                               const Tensor& local_lookup_ids, int num_lookup,
                               int num_global_lookup, int batch_size) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  auto stream = core_->get_local_gpu()->get_stream();
  int batch_size_per_gpu = batch_size / core_->get_global_gpu_count();

  cudaMemsetAsync(wgrad.data.get(), 0, wgrad.data.nbytes(), stream);

  if (src_buffer.attr.layout == EmbeddingLayout::FeatureMajor) {
    dp_local_reduce_from_feature_major_top_grad(
        kernel_params_, reduction_indices, src_buffer, local_lookup_ids, num_lookup, wgrad,
        partial_reduce_result_, batch_size_per_gpu, src_buffer.attr.max_ev_size, stream);
  } else {
    dp_local_reduce_from_batch_major_top_grad(
        kernel_params_, reduction_indices, src_buffer, local_lookup_ids, num_global_lookup, wgrad,
        partial_reduce_result_, batch_size_per_gpu, src_buffer.attr.max_ev_size, stream);
  }
}
}  // namespace embedding
