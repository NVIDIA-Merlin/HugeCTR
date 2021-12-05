/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cuda_runtime.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <iostream>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/data_simulator.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/frequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/update.cuh"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.cuh"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/shuffle/shuffle.cuh"
#include "HugeCTR/include/tensor2.hpp"
#include "HugeCTR/include/utils.cuh"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

namespace frequent_embedding_kernels {

template<typename dtype>
__global__ void reset_relevant_gradients(float* __restrict__ gradients,
                                         uint32_t embedding_vec_size,
                                         FrequentEmbeddingCompressionView<dtype>* indices,
                                         uint32_t num_instances) {
  const uint32_t num_network_cache_indices = indices->network_cache_indices_offsets[num_instances];
  for (uint32_t i = blockIdx.x; i < num_network_cache_indices; i += gridDim.x)
    gradients[indices->network_cache_indices[i] * embedding_vec_size + threadIdx.x] = 0.0f;
}

template <typename dtype, typename emtype>
__global__ void frequent_local_reduce(const emtype* __restrict__ gradients_in,
                                      float* __restrict__ gradients_out,
                                      size_t local_samples_offset,
                                      const dtype* __restrict__ category_frequent_index,
                                      uint32_t embedding_vec_size,
                                      FrequentEmbeddingCompressionView<dtype>* indices) {
  const uint32_t num_frequent_sample_indices = *indices->d_num_frequent_sample_indices;

  for (uint32_t i = blockIdx.x; i < num_frequent_sample_indices; i += gridDim.x) {
    uint32_t local_sample_index = indices->frequent_sample_indices[i];
    dtype category = indices->samples[local_samples_offset + local_sample_index];
    dtype frequent_index = category_frequent_index[category];

    atomicAdd(gradients_out + frequent_index * embedding_vec_size + threadIdx.x,
              TypeConvertFunc<float, emtype>::convert(
                  gradients_in[local_sample_index * embedding_vec_size + threadIdx.x]));
  }
}

template <typename emtype>
__forceinline__ __device__ void update_model_direct_common(
    const emtype* const* __restrict__ gradients_pointers, float* __restrict__ embedding_vectors,
    const uint32_t* __restrict__ model_cache_indices,
    const uint32_t* __restrict__ model_cache_indices_offsets, uint32_t num_instances,
    uint32_t model_id, uint32_t num_frequent_per_model, uint32_t embedding_vec_size, float lr) {}

template <typename dtype, typename emtype>
__global__ void update_model_direct(const emtype* const* __restrict__ gradients_pointers,
                                    float* __restrict__ embedding_vectors,
                                    FrequentEmbeddingCompressionView<dtype>* indices,
                                    uint32_t num_instances, uint32_t model_id,
                                    uint32_t num_frequent_per_model, uint32_t embedding_vec_size,
                                    const float* __restrict__ lr_ptr, const float scale) {
  float lr = __ldg(lr_ptr) / scale;
  const uint32_t offset = indices->model_cache_indices_offsets[model_id + 1];
  const uint32_t num_model_cache_indices = indices->model_cache_indices_offsets[num_instances];

  for (uint32_t i = blockIdx.x; i < num_model_cache_indices; i += gridDim.x) {
    int vid = (i + offset) % num_model_cache_indices;

    uint32_t frequent_index = indices->model_cache_indices[vid];
    uint32_t network_id;
    for (network_id = 0;
         network_id < num_instances && indices->model_cache_indices_offsets[network_id + 1] <= vid;
         network_id++)
      ;

    const emtype* gradients = gradients_pointers[network_id];

    uint32_t cache_location = frequent_index * embedding_vec_size + threadIdx.x;
    atomicAdd(embedding_vectors + cache_location,
              -lr * TypeConvertFunc<float, emtype>::convert(gradients[cache_location]));
  }
}

}  // namespace frequent_embedding_kernels

template <typename dtype>
FrequentEmbeddingBase<dtype>::FrequentEmbeddingBase() {
  CK_CUDA_THROW_(cudaMalloc(&indices_view_, sizeof(*indices_view_)));
}

template <typename dtype>
FrequentEmbeddingBase<dtype>::~FrequentEmbeddingBase() {
  cudaFree(indices_view_);
}

template <typename dtype>
void FrequentEmbeddingBase<dtype>::set_current_indices(
    FrequentEmbeddingCompression<dtype> *indices, cudaStream_t stream) {
  
  indices_ = indices;
  data_ = indices->get_data();
  CK_CUDA_THROW_(cudaMemcpyAsync(indices_view_, indices->get_device_view(),
      sizeof(*indices_view_), cudaMemcpyDeviceToDevice, stream));
}

template <typename dtype, typename emtype>
FrequentEmbedding<dtype, emtype>::FrequentEmbedding(
    const Model<dtype>& model,
    const GPUResource& gpu_resource, BuffPtr<emtype>& grouped_wgrad_buff,
    uint32_t embedding_vec_size, size_t max_num_frequent_categories)
    : model_(model),
      gpu_resource(gpu_resource),
      grouped_wgrad_buff_(grouped_wgrad_buff),
      embedding_vec_size_(embedding_vec_size),
      max_num_frequent_categories_(max_num_frequent_categories) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
  buf->reserve({max_num_frequent_categories, embedding_vec_size_}, &frequent_embedding_vectors_);
  if (sizeof(emtype) != sizeof(float)) {
    buf->reserve({max_num_frequent_categories, embedding_vec_size_}, &float_frequent_gradients_);
  }

  auto& gradients = get_gradients();
  if (grouped_wgrad_buff == NULL) {
    buf->reserve({max_num_frequent_categories, embedding_vec_size_}, &gradients);
  } else {
    grouped_wgrad_buff->reserve({max_num_frequent_categories, embedding_vec_size_}, &gradients);
  }

  if (model.communication_type == CommunicationType::NVLink_SingleNode) {
    buf->reserve({model.num_instances, 1}, &embedding_vectors_cache_pointers_);
    buf->reserve({model.num_instances, 1}, &partial_gradients_pointers_);
  }

  if (sizeof(emtype) != sizeof(float) &&
      model.communication_type == CommunicationType::NVLink_SingleNode) {
    buf->reserve({max_num_frequent_categories, embedding_vec_size_},
                 &frequent_embedding_vectors_cache_);
  }

  buf->allocate();
}

template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::initialize_embedding_vectors(
    const std::vector<size_t>& table_sizes, 
    size_t grouped_wgrad_offset_in_bytes) {
  CudaDeviceContext context(gpu_resource.get_device_id());

  const size_t num_tables = table_sizes.size();
  for (size_t model_id = 0; model_id < model_.num_instances; ++model_id) {
    for (size_t embedding = 0; embedding < num_tables; embedding++) {
      float up_bound = sqrt(1.f / table_sizes[embedding]);
      size_t offset =
          embedding_vec_size_ *
          model_.h_frequent_model_table_offsets[model_id * (num_tables + 1) + embedding];
      size_t num_elements =
          embedding_vec_size_ *
          (model_.h_frequent_model_table_offsets[model_id * (num_tables + 1) + embedding + 1] -
           model_.h_frequent_model_table_offsets[model_id * (num_tables + 1) + embedding]);
      UniformGenerator::fill(frequent_embedding_vectors_.get_ptr() + offset, num_elements,
                             -up_bound, up_bound, gpu_resource.get_sm_count(),
                             gpu_resource.get_replica_uniform_curand_generator(),
                             gpu_resource.get_stream());
    }
  }
  if (grouped_wgrad_buff_ != NULL) {
    // update wgrad tensors
    size_t grad_size = model_.num_frequent * embedding_vec_size_;
    if (sizeof(float) != sizeof(emtype)) {
      auto buf = std::make_shared<ExternalManagedBuffer>(
          (char*)grouped_wgrad_buff_->as_tensor().get_ptr() + grouped_wgrad_offset_in_bytes);
      frequent_gradients_ = Tensor2<emtype>({grad_size}, buf);
    } else {
      auto buf = std::make_shared<ExternalManagedBuffer>(
          (char*)grouped_wgrad_buff_->as_tensor().get_ptr() + grouped_wgrad_offset_in_bytes);
      float_frequent_gradients_ = Tensor2<float>({grad_size}, buf);
    }
  }
}

/* Single-node: refresh needed vectors in the cache of each network
 * Note: each network pulls from the models */
template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::forward_model(cudaStream_t stream) {
  const uint32_t num_instances = model_.num_instances;
  const uint32_t model_id = model_.global_instance_id;

  auto embedding_vectors_cache_pointers = embedding_vectors_cache_pointers_.get_ptr();
  auto frequent_embedding_vectors = frequent_embedding_vectors_.get_ptr();
  auto indices = this->indices_view_;
  auto embedding_vec_size = embedding_vec_size_;

  auto copy_desc = CopyDescriptors::make_OneToOne<float, emtype, 1>(
      embedding_vec_size,
      [=] __device__() { return indices->model_cache_indices_offsets[num_instances]; },
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<float, emtype, 1> {
        const uint32_t offset = indices->model_cache_indices_offsets[model_id + 1];
        const uint32_t num_model_cache_indices = indices->model_cache_indices_offsets[num_instances];
        int vid = (i + offset) % num_model_cache_indices;
        uint32_t frequent_index = indices->model_cache_indices[vid];

        uint32_t network_id;
        for (network_id = 0;
             network_id < num_instances && indices->model_cache_indices_offsets[network_id + 1] <= vid;
             network_id++)
          ;
        emtype* embedding_vectors_out = embedding_vectors_cache_pointers[network_id];

        const float* src_ptr = frequent_embedding_vectors + frequent_index * embedding_vec_size;
        emtype* dst_ptr = embedding_vectors_out + frequent_index * embedding_vec_size;

        return {
            src_ptr, {dst_ptr}, {static_cast<const void*>(src_ptr) != static_cast<void*>(dst_ptr)}};
      });

  PROFILE_RECORD("fre_forward_model.forward_model.start", stream, false);
  shuffle(copy_desc, stream, model_.num_frequent / 4);
  CK_CUDA_THROW_(cudaPeekAtLastError());
  PROFILE_RECORD("fre_forward_model.forward_model.stop", stream, false);
}

/* Single-node: refresh all vectors in the cache of each network */
template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::forward_model_eval(cudaStream_t stream) {
  const uint32_t num_instances = model_.num_instances;
  const uint32_t model_id = model_.global_instance_id;

  emtype** embedding_vectors_cache_pointers = embedding_vectors_cache_pointers_.get_ptr();
  const float* frequent_embedding_vectors = frequent_embedding_vectors_.get_ptr();
  size_t embedding_vec_size = embedding_vec_size_;
  const uint32_t num_frequent = model_.num_frequent;
  const uint32_t num_frequent_per_model = model_.num_frequent / num_instances;

  auto copy_desc = CopyDescriptors::make_OneToOne<float, emtype, 1>(
      embedding_vec_size,
      [=] __device__() { return num_frequent; },
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<float, emtype, 1> {
        // Shift pattern
        uint32_t shifted_i = (i + (model_id + 1) * num_frequent_per_model) % num_frequent;
        uint32_t network_id = shifted_i / num_frequent_per_model;
        uint32_t frequent_index =
            model_id * num_frequent_per_model + shifted_i % num_frequent_per_model;

        emtype* embedding_vectors_out = embedding_vectors_cache_pointers[network_id];

        const float* src_ptr = frequent_embedding_vectors + frequent_index * embedding_vec_size;
        emtype* dst_ptr = embedding_vectors_out + frequent_index * embedding_vec_size;

        return {
            src_ptr, {dst_ptr}, {static_cast<const void*>(src_ptr) != static_cast<void*>(dst_ptr)}};
      });

  PROFILE_RECORD("fre_forward_model.forward_model_eval.start", stream, false);
  shuffle(copy_desc, stream, model_.num_frequent);
  CK_CUDA_THROW_(cudaPeekAtLastError());
  PROFILE_RECORD("fre_forward_model.forward_model_eval.stop", stream, false);
}

template <typename dtype, typename emtype>
template <typename vectype>
void FrequentEmbedding<dtype, emtype>::forward_network_aux<vectype>(
    const vectype* embedding_vectors, emtype* interaction_layer_input, cudaStream_t stream) {
  uint32_t samples_per_instance = data_->samples.get_num_elements() / model_.num_instances;
  uint32_t global_sample_index_base = model_.global_instance_id * samples_per_instance;

  auto indices = this->indices_view_;
  auto category_frequent_index = model_.category_frequent_index.get_ptr();
  auto embedding_vec_size = embedding_vec_size_;

  auto copy_desc = CopyDescriptors::make_OneToOne<vectype, emtype, 1>(
      embedding_vec_size,
      [=] __device__() -> size_t { return *indices->d_num_frequent_sample_indices; },
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<vectype, emtype, 1> {
        auto index = indices->frequent_sample_indices[i];
        auto category = indices->samples[index + global_sample_index_base];
        auto frequent_index = category_frequent_index[category];

        return {embedding_vectors + frequent_index * embedding_vec_size,
                {interaction_layer_input + indices->frequent_sample_indices[i] * embedding_vec_size},
                {true}};
      });

  shuffle(copy_desc, stream, samples_per_instance);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

/* Concatenate the embedding vectors into the buffer for top-mlp input */
template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::forward_network(emtype* interaction_layer_input,
                                                       bool from_cache, cudaStream_t stream) {
  if (from_cache) {
    forward_network_aux(get_embedding_vectors_cache().get_ptr(), interaction_layer_input, stream);
  } else {
    forward_network_aux(frequent_embedding_vectors_.get_ptr(), interaction_layer_input, stream);
  }
}

/* Reduce gradients on each network */
template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::local_reduce(const emtype* gradients, cudaStream_t stream,
                                                    bool reset_all) {
  const uint32_t& num_instances = model_.num_instances;
  const uint32_t& network_id = model_.global_instance_id;
  size_t local_samples_size =
      ceildiv<size_t>(data_->batch_size, num_instances) * data_->table_sizes.size();

  int num_sm = gpu_resource.get_sm_count();
  int n_blocks = 16 * num_sm;  // TODO: better heuristics

  if (reset_all) { /* Set to zero all the gradients */
    if (model_.num_frequent > 0) {
      PROFILE_RECORD("fre_local_reduce.reset_all_gradients.start", stream, false);
      CK_CUDA_THROW_(cudaMemsetAsync(float_frequent_gradients_.get_ptr(), 0,
                                     model_.num_frequent * embedding_vec_size_ * sizeof(float),
                                     stream));
      PROFILE_RECORD("fre_local_reduce.reset_all_gradients.stop", stream, false);
    }
  } else { /* Set to zero the gradients of categories that appear in the batch */
    PROFILE_RECORD("fre_local_reduce.reset_relevant_gradients.start", stream, false);
    frequent_embedding_kernels::
        reset_relevant_gradients<<<n_blocks, embedding_vec_size_, 0, stream>>>(
            float_frequent_gradients_.get_ptr(), embedding_vec_size_,
            this->indices_view_, num_instances);
    CK_CUDA_THROW_(cudaPeekAtLastError());
    PROFILE_RECORD("fre_local_reduce.reset_relevant_gradients.stop", stream, false);
  }

  /* Local reduce */
  frequent_embedding_kernels::frequent_local_reduce<<<n_blocks, embedding_vec_size_, 0, stream>>>(
      gradients, float_frequent_gradients_.get_ptr(),
      network_id * local_samples_size,
      model_.category_frequent_index.get_ptr(), embedding_vec_size_,
      this->indices_view_);
  CK_CUDA_THROW_(cudaPeekAtLastError());

  if (sizeof(emtype) != sizeof(float)) {
    convert_array<<<1000, 128, 0, stream>>>(frequent_gradients_.get_ptr(),
                                            float_frequent_gradients_.get_ptr(),
                                            model_.num_frequent * embedding_vec_size_);
    CK_CUDA_THROW_(cudaPeekAtLastError());
  }
}

template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::update_model(float* dev_lr, float scale,
                                                    cudaStream_t stream) {
  sgd_global_update(get_gradients().get_ptr(), frequent_embedding_vectors_.get_ptr(),
                    model_.num_frequent, embedding_vec_size_, dev_lr, scale, stream);
}

/* Update model for single-node: direct write in category "owner"'s table, lr is a device variable
 */
template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::update_model_direct(float* dev_lr, float scale,
                                                           cudaStream_t stream) {
  const uint32_t& num_instances = model_.num_instances;
  const uint32_t& model_id = model_.global_instance_id;
  const uint32_t num_frequent_per_model = model_.num_frequent / num_instances;

  int num_sm = gpu_resource.get_sm_count();
  int n_blocks = 16 * num_sm;  // TODO: better heuristics

  /* Update models */
  PROFILE_RECORD("fre_update_model_direct.update_model_direct.start", stream, false);
  frequent_embedding_kernels::update_model_direct<<<n_blocks, embedding_vec_size_, 0, stream>>>(
      partial_gradients_pointers_.get_ptr(), frequent_embedding_vectors_.get_ptr(),
      this->indices_view_, num_instances,
      model_id, num_frequent_per_model, embedding_vec_size_, dev_lr, scale);
  CK_CUDA_THROW_(cudaPeekAtLastError());
  PROFILE_RECORD("fre_update_model_direct.update_model_direct.stop", stream, false);
}

template class FrequentEmbeddingBase<uint32_t>;
template class FrequentEmbeddingBase<long long>;

template class FrequentEmbedding<uint32_t, __half>;
template class FrequentEmbedding<uint32_t, float>;
template class FrequentEmbedding<long long, __half>;
template class FrequentEmbedding<long long, float>;

template void FrequentEmbedding<uint32_t, __half>::forward_network_aux<__half>(
    const __half* embedding_vectors, __half* interaction_layer_input, cudaStream_t stream);
template void FrequentEmbedding<uint32_t, __half>::forward_network_aux<float>(
    const float* embedding_vectors, __half* interaction_layer_input, cudaStream_t stream);
template void FrequentEmbedding<uint32_t, float>::forward_network_aux<float>(
    const float* embedding_vectors, float* interaction_layer_input, cudaStream_t stream);
template void FrequentEmbedding<long long, __half>::forward_network_aux<__half>(
    const __half* embedding_vectors, __half* interaction_layer_input, cudaStream_t stream);
template void FrequentEmbedding<long long, __half>::forward_network_aux<float>(
    const float* embedding_vectors, __half* interaction_layer_input, cudaStream_t stream);
template void FrequentEmbedding<long long, float>::forward_network_aux<float>(
    const float* embedding_vectors, float* interaction_layer_input, cudaStream_t stream);
}  // namespace hybrid_embedding

}  // namespace HugeCTR
