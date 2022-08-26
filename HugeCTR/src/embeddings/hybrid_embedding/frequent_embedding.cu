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

template <typename dtype>
__global__ void reset_relevant_gradients(float* __restrict__ gradients, uint32_t embedding_vec_size,
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
                                      const dtype* __restrict__ category_location,
                                      uint32_t embedding_vec_size,
                                      FrequentEmbeddingCompressionView<dtype>* indices) {
  const uint32_t num_frequent_sample_indices = *indices->d_num_frequent_sample_indices;

  for (uint32_t i = blockIdx.x; i < num_frequent_sample_indices; i += gridDim.x) {
    uint32_t local_sample_index = indices->frequent_sample_indices[i];
    dtype category = indices->samples[local_samples_offset + local_sample_index];
    dtype frequent_index = category_location[2 * category + 1];

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
FrequentEmbeddingBase<dtype>::FrequentEmbeddingBase() {}

template <typename dtype>
FrequentEmbeddingBase<dtype>::~FrequentEmbeddingBase() {}

template <typename dtype>
void FrequentEmbeddingBase<dtype>::set_current_indices(
    FrequentEmbeddingCompression<dtype>* indices) {
  indices_ = indices;
  data_ = indices->get_data();
  indices_view_ = indices->get_device_view();
}

template <typename dtype, typename emtype>
FrequentEmbeddingData<dtype, emtype>::FrequentEmbeddingData(const Model<dtype>& model,
                                                            const GPUResource& gpu_resource,
                                                            BuffPtr<emtype>& grouped_wgrad_buff,
                                                            uint32_t embedding_vec_size,
                                                            size_t max_num_frequent_categories)
    : model_(model),
      gpu_resource_(gpu_resource),
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

  buf->allocate();
}

template <typename dtype, typename emtype>
FrequentEmbeddingSingleNode<dtype, emtype>::FrequentEmbeddingSingleNode(
    const Model<dtype>& model, const GPUResource& gpu_resource, BuffPtr<emtype>& grouped_wgrad_buff,
    uint32_t embedding_vec_size, size_t max_num_frequent_categories)
    : frequent_data_(model, gpu_resource, grouped_wgrad_buff, embedding_vec_size,
                     max_num_frequent_categories) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();

  buf->reserve({model.num_instances, 1}, &embedding_vectors_cache_pointers_);
  buf->reserve({model.num_instances, 1}, &partial_gradients_pointers_);
  if (sizeof(emtype) != sizeof(float)) {
    buf->reserve({max_num_frequent_categories, embedding_vec_size},
                 &frequent_embedding_vectors_cache_);
  }
  buf->allocate();
}

template <typename dtype, typename emtype>
void FrequentEmbeddingMultiNode<dtype, emtype>::init_ar_comm(AllReduceInPlaceComm* ar_comm,
                                                             AllReduceInPlaceComm::Handle& handle,
                                                             int local_id) {
  auto& local_gpu = frequent_data_.gpu_resource_;
  CudaDeviceContext context(local_gpu.get_device_id());

  auto& gradients = frequent_data_.get_gradients();
  ar_comm->set_coll_buf(handle, gradients.get_ptr(), gradients.get_size_in_bytes(), local_id);
  ar_comm_ = std::make_unique<AllReduceComm<emtype>>(ar_comm, handle, &local_gpu);
}

template <typename dtype, typename emtype>
void FrequentEmbeddingData<dtype, emtype>::initialize_embedding_vectors(
    const std::vector<size_t>& table_sizes, size_t grouped_wgrad_offset_in_bytes) {
  CudaDeviceContext context(gpu_resource_.get_device_id());

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
                             -up_bound, up_bound, gpu_resource_.get_sm_count(),
                             gpu_resource_.get_replica_uniform_curand_generator(),
                             gpu_resource_.get_stream());
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
void FrequentEmbeddingSingleNode<dtype, emtype>::forward_model(cudaStream_t stream) {
  const uint32_t num_instances = frequent_data_.model_.num_instances;
  const uint32_t model_id = frequent_data_.model_.global_instance_id;

  auto embedding_vectors_cache_pointers = embedding_vectors_cache_pointers_.get_ptr();
  auto frequent_embedding_vectors = frequent_data_.frequent_embedding_vectors_.get_ptr();
  auto indices = this->indices_view_;
  auto embedding_vec_size = frequent_data_.embedding_vec_size_;

  auto copy_desc = CopyDescriptors::make_OneToOne<float, emtype, 1>(
      embedding_vec_size,
      [=] __device__() { return indices->model_cache_indices_offsets[num_instances]; },
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<float, emtype, 1> {
        const uint32_t offset = indices->model_cache_indices_offsets[model_id + 1];
        const uint32_t num_model_cache_indices =
            indices->model_cache_indices_offsets[num_instances];
        int vid = (i + offset) % num_model_cache_indices;
        uint32_t frequent_index = indices->model_cache_indices[vid];

        uint32_t network_id;
        for (network_id = 0; network_id < num_instances &&
                             indices->model_cache_indices_offsets[network_id + 1] <= vid;
             network_id++)
          ;
        emtype* embedding_vectors_out = embedding_vectors_cache_pointers[network_id];

        const float* src_ptr = frequent_embedding_vectors + frequent_index * embedding_vec_size;
        emtype* dst_ptr = embedding_vectors_out + frequent_index * embedding_vec_size;

        return {
            src_ptr, {dst_ptr}, {static_cast<const void*>(src_ptr) != static_cast<void*>(dst_ptr)}};
      });

  shuffle(copy_desc, stream, frequent_data_.model_.num_frequent / 4);
  HCTR_LIB_THROW(cudaPeekAtLastError());
}

/* Single-node: refresh all vectors in the cache of each network */
template <typename dtype, typename emtype>
void FrequentEmbeddingSingleNode<dtype, emtype>::forward_model_eval(cudaStream_t stream) {
  const uint32_t num_instances = frequent_data_.model_.num_instances;
  const uint32_t model_id = frequent_data_.model_.global_instance_id;

  emtype** embedding_vectors_cache_pointers = embedding_vectors_cache_pointers_.get_ptr();
  const float* frequent_embedding_vectors = frequent_data_.frequent_embedding_vectors_.get_ptr();
  size_t embedding_vec_size = frequent_data_.embedding_vec_size_;
  const uint32_t num_frequent = frequent_data_.model_.num_frequent;
  const uint32_t num_frequent_per_model = num_frequent / num_instances;

  auto copy_desc = CopyDescriptors::make_OneToOne<float, emtype, 1>(
      embedding_vec_size, [=] __device__() { return num_frequent; },
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

  shuffle(copy_desc, stream, num_frequent);
  HCTR_LIB_THROW(cudaPeekAtLastError());
}

template <typename dtype, typename emtype>
template <typename vectype>
void FrequentEmbeddingData<dtype, emtype>::forward_network<vectype>(
    const vectype* embedding_vectors, emtype* interaction_layer_input,
    FrequentEmbeddingBase<dtype>* base, cudaStream_t stream) {
  uint32_t samples_per_instance =
      base->data_->samples.get_num_elements() / this->model_.num_instances;
  uint32_t global_sample_index_base = model_.global_instance_id * samples_per_instance;

  auto indices = base->indices_view_;
  auto category_location = this->model_.category_location.get_ptr();
  auto embedding_vec_size = this->embedding_vec_size_;

  auto copy_desc = CopyDescriptors::make_OneToOne<vectype, emtype, 1>(
      embedding_vec_size,
      [=] __device__() -> size_t { return *indices->d_num_frequent_sample_indices; },
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<vectype, emtype, 1> {
        auto index = indices->frequent_sample_indices[i];
        auto category = indices->samples[index + global_sample_index_base];
        auto frequent_index = category_location[2 * category + 1];

        return {
            embedding_vectors + frequent_index * embedding_vec_size,
            {interaction_layer_input + indices->frequent_sample_indices[i] * embedding_vec_size},
            {true}};
      });

  shuffle(copy_desc, stream, samples_per_instance);
  HCTR_LIB_THROW(cudaPeekAtLastError());
}

/* Concatenate the embedding vectors into the buffer for top-mlp input */
template <typename dtype, typename emtype>
void FrequentEmbeddingSingleNode<dtype, emtype>::forward_network(emtype* interaction_layer_input,
                                                                 cudaStream_t stream) {
  frequent_data_.forward_network(get_embedding_vectors_cache().get_ptr(), interaction_layer_input,
                                 this, stream);
}

template <typename dtype, typename emtype>
void FrequentEmbeddingMultiNode<dtype, emtype>::forward_network(emtype* interaction_layer_input,
                                                                cudaStream_t stream) {
  frequent_data_.forward_network(frequent_data_.frequent_embedding_vectors_.get_ptr(),
                                 interaction_layer_input, this, stream);
}

/* Reduce gradients on each network */
template <typename dtype, typename emtype>
void FrequentEmbeddingData<dtype, emtype>::local_reduce(const emtype* gradients,
                                                        FrequentEmbeddingBase<dtype>* base,
                                                        cudaStream_t stream) {
  const auto num_instances = model_.num_instances;
  const auto network_id = model_.global_instance_id;
  size_t local_samples_size =
      ceildiv<size_t>(base->data_->batch_size, num_instances) * base->data_->table_sizes.size();

  int n_blocks = 16 * gpu_resource_.get_sm_count();
  auto embedding_vec_size = embedding_vec_size_;

  frequent_embedding_kernels::frequent_local_reduce<<<n_blocks, embedding_vec_size, 0, stream>>>(
      gradients, float_frequent_gradients_.get_ptr(), network_id * local_samples_size,
      model_.category_location.get_ptr(), embedding_vec_size, base->indices_view_);
  HCTR_LIB_THROW(cudaPeekAtLastError());

  if (sizeof(emtype) != sizeof(float)) {
    convert_array<<<1000, 128, 0, stream>>>(frequent_gradients_.get_ptr(),
                                            float_frequent_gradients_.get_ptr(),
                                            model_.num_frequent * embedding_vec_size);
    HCTR_LIB_THROW(cudaPeekAtLastError());
  }
}

template <typename dtype, typename emtype>
void FrequentEmbeddingSingleNode<dtype, emtype>::local_reduce(const emtype* gradients,
                                                              cudaStream_t stream) {
  auto num_instances = frequent_data_.model_.num_instances;
  int n_blocks = 16 * frequent_data_.gpu_resource_.get_sm_count();
  auto embedding_vec_size = frequent_data_.embedding_vec_size_;

  /* Set to zero the gradients of categories that appear in the batch */
  frequent_embedding_kernels::reset_relevant_gradients<<<n_blocks, embedding_vec_size, 0, stream>>>(
      frequent_data_.float_frequent_gradients_.get_ptr(), embedding_vec_size, this->indices_view_,
      num_instances);
  HCTR_LIB_THROW(cudaPeekAtLastError());

  frequent_data_.local_reduce(gradients, this, stream);
}

template <typename dtype, typename emtype>
void FrequentEmbeddingMultiNode<dtype, emtype>::local_reduce(const emtype* gradients,
                                                             cudaStream_t stream) {
  /* Set to zero all the gradients */
  if (frequent_data_.model_.num_frequent > 0) {
    HCTR_LIB_THROW(cudaMemsetAsync(
        frequent_data_.float_frequent_gradients_.get_ptr(), 0,
        frequent_data_.model_.num_frequent * frequent_data_.embedding_vec_size_ * sizeof(float),
        stream));
  }

  frequent_data_.local_reduce(gradients, this, stream);
}

template <typename dtype, typename emtype>
void FrequentEmbeddingMultiNode<dtype, emtype>::update_model(float* dev_lr, float scale,
                                                             cudaStream_t stream) {
  sgd_global_update(frequent_data_.get_gradients().get_ptr(),
                    frequent_data_.frequent_embedding_vectors_.get_ptr(),
                    frequent_data_.model_.num_frequent, frequent_data_.embedding_vec_size_, dev_lr,
                    scale, stream);
}

/* Update model for single-node: direct write in category "owner"'s table, lr is a device variable
 */
template <typename dtype, typename emtype>
void FrequentEmbeddingSingleNode<dtype, emtype>::update_model_direct(float* dev_lr, float scale,
                                                                     cudaStream_t stream) {
  const uint32_t& num_instances = frequent_data_.model_.num_instances;
  const uint32_t& model_id = frequent_data_.model_.global_instance_id;
  const uint32_t num_frequent_per_model = frequent_data_.model_.num_frequent / num_instances;

  int num_sm = frequent_data_.gpu_resource_.get_sm_count();
  int n_blocks = 8 * num_sm;  // TODO: better heuristics

  /* Update models */
  frequent_embedding_kernels::
      update_model_direct<<<n_blocks, frequent_data_.embedding_vec_size_, 0, stream>>>(
          partial_gradients_pointers_.get_ptr(),
          frequent_data_.frequent_embedding_vectors_.get_ptr(), this->indices_view_, num_instances,
          model_id, num_frequent_per_model, frequent_data_.embedding_vec_size_, dev_lr, scale);
  HCTR_LIB_THROW(cudaPeekAtLastError());
}

template <typename dtype, typename emtype>
void FrequentEmbeddingMultiNode<dtype, emtype>::communicate(cudaStream_t stream) {
  ar_comm_->communicate(stream);
}

template class FrequentEmbeddingBase<uint32_t>;
template class FrequentEmbeddingBase<long long>;

template class FrequentEmbeddingData<uint32_t, __half>;
template class FrequentEmbeddingData<uint32_t, float>;
template class FrequentEmbeddingData<long long, __half>;
template class FrequentEmbeddingData<long long, float>;

template class FrequentEmbeddingSingleNode<uint32_t, __half>;
template class FrequentEmbeddingSingleNode<uint32_t, float>;
template class FrequentEmbeddingSingleNode<long long, __half>;
template class FrequentEmbeddingSingleNode<long long, float>;

template class FrequentEmbeddingMultiNode<uint32_t, __half>;
template class FrequentEmbeddingMultiNode<uint32_t, float>;
template class FrequentEmbeddingMultiNode<long long, __half>;
template class FrequentEmbeddingMultiNode<long long, float>;

template void FrequentEmbeddingData<uint32_t, __half>::forward_network<__half>(
    const __half*, __half*, FrequentEmbeddingBase<uint32_t>*, cudaStream_t);
template void FrequentEmbeddingData<uint32_t, __half>::forward_network<float>(
    const float*, __half*, FrequentEmbeddingBase<uint32_t>*, cudaStream_t);
template void FrequentEmbeddingData<uint32_t, float>::forward_network<float>(
    const float*, float*, FrequentEmbeddingBase<uint32_t>*, cudaStream_t);
template void FrequentEmbeddingData<long long, __half>::forward_network<__half>(
    const __half*, __half*, FrequentEmbeddingBase<long long>*, cudaStream_t);
template void FrequentEmbeddingData<long long, __half>::forward_network<float>(
    const float*, __half*, FrequentEmbeddingBase<long long>*, cudaStream_t);
template void FrequentEmbeddingData<long long, float>::forward_network<float>(
    const float*, float*, FrequentEmbeddingBase<long long>*, cudaStream_t);
}  // namespace hybrid_embedding

}  // namespace HugeCTR
