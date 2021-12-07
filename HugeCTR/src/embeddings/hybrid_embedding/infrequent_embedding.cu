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
#include <utility>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/data_simulator.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/infrequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/update.cuh"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.cuh"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/shuffle/shuffle.cuh"
#include "HugeCTR/include/tensor2.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

namespace infrequent_embedding_kernels {

template <typename dtype, typename emtype>
__global__ void hier_update_model(InfrequentEmbeddingSelectionView<dtype>* indices,
                                  const dtype* __restrict__ category_location,
                                  const emtype* __restrict__ gradients,
                                  float* __restrict__ embedding_vectors,
                                  uint32_t embedding_vec_size, uint32_t num_instances,
                                  uint32_t local_samples_size, uint32_t local_comm_buff_size,
                                  const float* __restrict__ lr_ptr, const float scale) {
  float lr = __ldg(lr_ptr) / scale;
  const uint32_t num_indices = indices->model_indices_offsets[num_instances];

  // Load offset only when the network_id changes
  uint32_t previous_network_id = 0;
  uint32_t offset = 0;

  for (uint32_t i = blockIdx.x; i < num_indices; i += gridDim.x) {
    uint32_t index = indices->model_indices[i];
    dtype category = indices->samples[index];
    dtype location = category_location[2 * category + 1];
    uint32_t network_id = index / local_samples_size;
    if (network_id != previous_network_id) {
      offset = indices->model_indices_offsets[network_id];
      previous_network_id = network_id;
    }
    atomicAdd(
        embedding_vectors + location * embedding_vec_size + threadIdx.x,
        -lr * TypeConvertFunc<float, emtype>::convert(
                  gradients[embedding_vec_size * (network_id * local_comm_buff_size + i - offset) +
                            threadIdx.x]));
  }
}

template <typename dtype, typename emtype>
__global__ void infrequent_update_model_direct(
    const emtype* const* __restrict__ gradients_pointers, float* embedding_vectors,
    InfrequentEmbeddingSelectionView<dtype>* indices, const dtype* __restrict__ category_location,
    uint32_t num_instances, uint32_t model_id, uint32_t embedding_vec_size,
    uint32_t local_samples_size, const float* __restrict__ lr_ptr, const float scale) {
  float lr = __ldg(lr_ptr) / scale;
  // Shift pattern
  const uint32_t offset = indices->model_indices_offsets[model_id + 1];
  const uint32_t num_model_indices = indices->model_indices_offsets[num_instances];

  for (uint32_t i = blockIdx.x; i < num_model_indices; i += gridDim.x) {
    uint32_t vid = (i + offset) % num_model_indices;

    uint32_t index = indices->model_indices[vid];
    uint32_t network_id = index / local_samples_size;
    uint32_t local_index = index % local_samples_size;
    dtype category = indices->samples[index];
    uint32_t location = category_location[2 * category + 1];

    const emtype* gradients = gradients_pointers[network_id];

    atomicAdd(embedding_vectors + location * embedding_vec_size + threadIdx.x,
              -lr * TypeConvertFunc<float, emtype>::convert(
                        gradients[local_index * embedding_vec_size + threadIdx.x]));
  }
}

// template <typename dtype>
// __global__ void calculate_network_indices_mask(const dtype* __restrict__ local_samples,
//                                                const dtype* __restrict__ category_location,
//                                                bool* mask, uint32_t local_samples_size,
//                                                uint32_t num_instances) {
//   for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < local_samples_size;
//        i += gridDim.x * blockDim.x) {
//     dtype category = local_samples[i];
//     uint32_t model_id = static_cast<uint32_t>(category_location[2 * category]);
//     for (uint32_t section_id = 0; section_id < num_instances; section_id++) {
//       mask[local_samples_size * section_id + i] = (model_id == section_id);
//     }
//   }
// }

template <typename LambdaPtr>
static __global__ void offsets_to_sizes(size_t* sizes, LambdaPtr get_offsets_ptr,
                                        size_t element_size, uint32_t num_instances) {
  uint32_t* offsets = get_offsets_ptr();
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < num_instances;
       t += gridDim.x * blockDim.x) {
    sizes[t] = (offsets[t + 1] - offsets[t]) * element_size;
  }
}

}  // namespace infrequent_embedding_kernels

template <typename dtype>
InfrequentEmbeddingBase<dtype>::InfrequentEmbeddingBase() {
  CK_CUDA_THROW_(cudaMalloc(&indices_view_, sizeof(*indices_view_)));
}

template <typename dtype>
InfrequentEmbeddingBase<dtype>::~InfrequentEmbeddingBase() {
  cudaFree(indices_view_);
}

template <typename dtype>
void InfrequentEmbeddingBase<dtype>::set_current_indices(
    InfrequentEmbeddingSelection<dtype>* indices, cudaStream_t stream) {
  indices_ = indices;
  data_ = indices->get_data();
  CK_CUDA_THROW_(cudaMemcpyAsync(indices_view_, indices->get_device_view(), sizeof(*indices_view_),
                                 cudaMemcpyDeviceToDevice, stream));
}

template <typename dtype, typename emtype>
InfrequentEmbedding<dtype, emtype>::InfrequentEmbedding(const Model<dtype>& model,
                                                        const GPUResource& gpu_resource,
                                                        uint32_t embedding_vec_size)
    : model_(model), gpu_resource(gpu_resource), embedding_vec_size_(embedding_vec_size) {
  auto buf = GeneralBuffer2<CudaAllocator>::create();
  buf->reserve({ceildiv<size_t>(model.num_categories, model.num_instances), embedding_vec_size_},
               &infrequent_embedding_vectors_);
  buf->reserve({model_.num_instances}, &model_indices_sizes_);
  buf->reserve({model_.num_instances}, &model_indices_sizes_ptrs_);
  buf->reserve({model_.num_instances}, &network_indices_sizes_);
  buf->reserve({model_.num_instances}, &network_indices_sizes_ptrs_);

  buf->reserve({model.num_instances, 1}, &interaction_layer_input_pointers_train_);
  buf->reserve({model.num_instances, 1}, &interaction_layer_input_pointers_eval_);
  buf->reserve({model.num_instances, 1}, &gradients_pointers_);
  buf->allocate();

  auto managed_buf = GeneralBuffer2<CudaManagedAllocator>::create();
  managed_buf->reserve({model.num_instances + 1, 1}, &model_indices_offsets_);
  managed_buf->reserve({model.num_instances + 1, 1}, &network_indices_offsets_);
  managed_buf->allocate();
  int current_device;
  CK_CUDA_THROW_(cudaGetDevice(&current_device));
  CK_CUDA_THROW_(cudaMemAdvise(managed_buf->get_ptr(), managed_buf->get_size_in_bytes(),
                               cudaMemAdviseSetReadMostly, current_device));
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::initialize_embedding_vectors(
    const std::vector<size_t>& table_sizes) {
  CudaDeviceContext context(gpu_resource.get_device_id());

  const size_t num_tables = table_sizes.size();
  for (size_t i = 0; i < num_tables; i++) {
    float up_bound = sqrt(1.f / table_sizes[i]);

    const size_t offset = embedding_vec_size_ * model_.h_infrequent_model_table_offsets[i];
    const size_t number_of_vectors =
        model_.h_infrequent_model_table_offsets[i + 1] - model_.h_infrequent_model_table_offsets[i];
    UniformGenerator::fill(
        infrequent_embedding_vectors_.get_ptr() + offset, embedding_vec_size_ * number_of_vectors,
        -up_bound, up_bound, gpu_resource.get_sm_count(),
        gpu_resource.get_replica_variant_curand_generator(), gpu_resource.get_stream());
  }
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::forward_model(emtype* message_buffer,
                                                       cudaStream_t stream) {
  auto indices = this->indices_view_;
  auto category_location = model_.category_location.get_ptr();
  auto infrequent_embedding_vectors = infrequent_embedding_vectors_.get_ptr();
  auto embedding_vec_size = embedding_vec_size_;
  auto num_instances = model_.num_instances;

  auto copy_desc = CopyDescriptors::make_OneToOne<float, emtype, 1>(
      embedding_vec_size,
      [=] __device__() { return indices->model_indices_offsets[num_instances]; },
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<float, emtype, 1> {
        uint32_t index = indices->model_indices[i];
        dtype category = indices->samples[index];
        dtype location = category_location[2 * category + 1];

        return {infrequent_embedding_vectors + location * embedding_vec_size,
                {message_buffer + i * embedding_vec_size},
                {true}};
      });

  shuffle(copy_desc, stream, data_->samples.get_num_elements() / model_.num_instances / 8);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::fused_intra_forward_model(emtype** message_buffer,
                                                                   cudaStream_t stream) {
  auto indices = this->indices_view_;
  auto category_location = model_.category_location.get_ptr();
  auto infrequent_embedding_vectors = infrequent_embedding_vectors_.get_ptr();
  size_t embedding_vec_size = embedding_vec_size_;
  auto local_instance_id = model_.instance_id;
  auto num_instances = model_.num_instances;
  auto per_node_instances = num_instances / model_.h_num_instances_per_node.size();
  uint32_t local_samples_size =
      ceildiv<uint32_t>(data_->batch_size, num_instances) * data_->table_sizes.size();

  uint32_t local_comm_buff_size =
      ceildiv<uint32_t>(max_num_infrequent_per_batch_, model_.num_instances);

  auto copy_desc = CopyDescriptors::make_OneToOne<float, emtype, 1>(
      embedding_vec_size,
      [=] __device__() { return indices->model_indices_offsets[num_instances]; },
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<float, emtype, 1> {
        uint32_t num_selected = indices->model_indices_offsets[num_instances];
        uint32_t vid =
            (i + indices->model_indices_offsets[(local_instance_id + 1) % per_node_instances]) %
            num_selected;
        uint32_t index = indices->model_indices[vid];
        uint32_t network_id = (index / local_samples_size);
        dtype category = indices->samples[index];
        dtype location = category_location[2 * category + 1];
        uint32_t local_network_id = (network_id % per_node_instances);
        emtype* output_ptr =
            &message_buffer[local_network_id][(network_id - local_network_id + local_instance_id) *
                                              local_comm_buff_size * embedding_vec_size];

        return {
            infrequent_embedding_vectors + location * embedding_vec_size,
            {output_ptr + (vid - indices->model_indices_offsets[network_id]) * embedding_vec_size},
            {true}};
      });

  shuffle(copy_desc, stream, data_->samples.get_num_elements() / model_.num_instances / 8);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::forward_network(const emtype* message_buffer,
                                                         emtype* interaction_layer_input,
                                                         cudaStream_t stream) {
  auto indices = this->indices_view_;
  auto embedding_vec_size = embedding_vec_size_;
  auto num_instances = model_.num_instances;

  auto copy_desc = CopyDescriptors::make_OneToOne<emtype, emtype, 1>(
      embedding_vec_size,
      [=] __device__() { return indices->network_indices_offsets[num_instances]; },
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<emtype, emtype, 1> {
        uint32_t index = indices->network_indices[i];
        return {message_buffer + i * embedding_vec_size,
                {interaction_layer_input + index * embedding_vec_size},
                {true}};
      });

  shuffle(copy_desc, stream, data_->samples.get_num_elements() / model_.num_instances / 8);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::hier_forward_network(const emtype* message_buffer,
                                                              emtype* interaction_layer_input,
                                                              cudaStream_t stream) {
  auto indices = this->indices_view_;
  auto embedding_vec_size = embedding_vec_size_;
  auto num_instances = model_.num_instances;
  uint32_t local_samples_size =
      ceildiv<uint32_t>(data_->batch_size, model_.num_instances) * data_->table_sizes.size();
  uint32_t local_comm_buff_size =
      ceildiv<uint32_t>(max_num_infrequent_per_batch_, model_.num_instances);

  auto copy_desc = CopyDescriptors::make_OneToOne<emtype, emtype, 1>(
      embedding_vec_size,
      [=] __device__() { return indices->network_indices_offsets[num_instances]; },
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<emtype, emtype, 1> {
        uint32_t index = indices->network_indices[i];

        // Find model id and offset
        uint32_t model_id = 0;
        uint32_t offset = 0;
        uint32_t next_offset = indices->network_indices_offsets[1];
        while (next_offset <= i) {
          offset = next_offset;
          model_id++;
          next_offset = indices->network_indices_offsets[model_id + 1];
        }

        return {
            message_buffer + (model_id * local_comm_buff_size + i - offset) * embedding_vec_size,
            {interaction_layer_input + index * embedding_vec_size},
            {true}};
      });

  shuffle(copy_desc, stream, data_->samples.get_num_elements() / model_.num_instances / 8);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

/** Forward network for single GPU (no communications) */
template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::forward_network_direct(bool is_train,
                                                                cudaStream_t stream) {
  const uint32_t num_instances = model_.num_instances;
  const uint32_t model_id = model_.global_instance_id;
  uint32_t local_samples_size =
      ceildiv<uint32_t>(data_->batch_size, num_instances) * data_->table_sizes.size();

  auto interaction_layer_input_pointers = is_train
                                              ? interaction_layer_input_pointers_train_.get_ptr()
                                              : interaction_layer_input_pointers_eval_.get_ptr();
  auto indices = this->indices_view_;
  auto category_location = model_.category_location.get_ptr();
  auto model_table = infrequent_embedding_vectors_.get_ptr();
  auto embedding_vec_size = embedding_vec_size_;

  auto copy_desc = CopyDescriptors::make_OneToOne<float, emtype, 1>(
      embedding_vec_size,
      [=] __device__() { return indices->model_indices_offsets[num_instances]; },
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<float, emtype, 1> {
        const uint32_t offset = indices->model_indices_offsets[model_id + 1];
        const uint32_t num_model_indices = indices->model_indices_offsets[num_instances];
        const uint32_t vid = (i + offset) % num_model_indices;
        const uint32_t index = indices->model_indices[vid];

        const dtype category = indices->samples[index];
        const dtype location = category_location[2 * category + 1];

        const uint32_t network_id = index / local_samples_size;
        const uint32_t local_index = index % local_samples_size;

        emtype* interaction_layer_input = interaction_layer_input_pointers[network_id];

        return {model_table + location * embedding_vec_size,
                {interaction_layer_input + local_index * embedding_vec_size},
                {true}};
      });

  PROFILE_RECORD("inf_forward_network_direct.forward_network_direct.start", stream, false);
  shuffle(copy_desc, stream, local_samples_size / 10);
  CK_CUDA_THROW_(cudaPeekAtLastError());
  PROFILE_RECORD("inf_forward_network_direct.forward_network_direct.stop", stream, false);
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::update_network(const emtype* gradients,
                                                        emtype* message_buffer,
                                                        cudaStream_t stream) {
  auto indices = this->indices_view_;
  auto embedding_vec_size = embedding_vec_size_;
  auto num_instances = model_.num_instances;

  auto copy_desc = CopyDescriptors::make_OneToOne<emtype, emtype, 1>(
      embedding_vec_size,
      [=] __device__() { return indices->network_indices_offsets[num_instances]; },
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<emtype, emtype, 1> {
        uint32_t index = indices->network_indices[i];

        return {gradients + index * embedding_vec_size,
                {message_buffer + i * embedding_vec_size},
                {true}};
      });

  shuffle(copy_desc, stream, data_->samples.get_num_elements() / model_.num_instances / 8);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::fused_intra_update_network(const emtype* gradients,
                                                                    emtype** message_buffer,
                                                                    cudaStream_t stream) {
  auto indices = this->indices_view_;
  size_t embedding_vec_size = embedding_vec_size_;
  auto local_instance_id = model_.instance_id;
  auto num_instances = model_.num_instances;
  auto per_node_instances = num_instances / model_.h_num_instances_per_node.size();
  uint32_t local_comm_buff_size =
      ceildiv<uint32_t>(max_num_infrequent_per_train_batch_, model_.num_instances);

  auto copy_desc = CopyDescriptors::make_OneToOne<emtype, emtype, 1>(
      embedding_vec_size,
      [=] __device__() { return indices->network_indices_offsets[num_instances]; },
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<emtype, emtype, 1> {
        uint32_t num_selected = indices->network_indices_offsets[num_instances];
        uint32_t vid =
            (i + indices->network_indices_offsets[(local_instance_id + 1) % per_node_instances]) %
            num_selected;
        uint32_t index = indices->network_indices[vid];

        uint32_t model_id;
        for (model_id = 0;
             model_id < num_instances && indices->network_indices_offsets[model_id + 1] <= vid;
             model_id++)
          ;

        uint32_t local_model_id = (model_id % per_node_instances);
        emtype* output_ptr =
            &message_buffer[local_model_id][(model_id - local_model_id + local_instance_id) *
                                            local_comm_buff_size * embedding_vec_size];

        return {
            gradients + index * embedding_vec_size,
            {output_ptr + (vid - indices->network_indices_offsets[model_id]) * embedding_vec_size},
            {true}};
      });

  shuffle(copy_desc, stream, data_->samples.get_num_elements() / model_.num_instances / 8);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::update_model(const emtype* message_buffer, float* dev_lr,
                                                      float scale, cudaStream_t stream) {
  auto indices = this->indices_view_;
  const dtype* __restrict__ category_location = model_.category_location.get_ptr();
  auto num_instances = model_.num_instances;

  uint32_t n_blocks = gpu_resource.get_sm_count();

  sgd_atomic_update(
      message_buffer, infrequent_embedding_vectors_.get_ptr(),
      [indices, num_instances] __device__() {
        return indices->model_indices_offsets[num_instances];
      },
      [indices, category_location] __device__(uint32_t i) {
        uint32_t index = indices->model_indices[i];
        dtype category = indices->samples[index];
        return category_location[2 * category + 1];
      },
      n_blocks, embedding_vec_size_, dev_lr, scale, stream);
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::hier_update_model(const emtype* message_buffer,
                                                           float* dev_lr, float scale,
                                                           cudaStream_t stream) {
  const uint32_t& num_instances = model_.num_instances;
  uint32_t local_samples_size =
      ceildiv<uint32_t>(data_->batch_size, num_instances) * data_->table_sizes.size();
  uint32_t local_comm_buff_size =
      ceildiv<uint32_t>(max_num_infrequent_per_train_batch_, model_.num_instances);

  int num_sm = gpu_resource.get_sm_count();
  int n_blocks = 16 * num_sm;  // TODO: better heuristics

  infrequent_embedding_kernels::hier_update_model<<<n_blocks, embedding_vec_size_, 0, stream>>>(
      this->indices_view_, model_.category_location.get_ptr(), message_buffer,
      infrequent_embedding_vectors_.get_ptr(), embedding_vec_size_, model_.num_instances,
      local_samples_size, local_comm_buff_size, dev_lr, scale);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

/** Update model for single GPU (no communications), lr is a device variable */
template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::update_model_direct(float* dev_lr, float scale,
                                                             cudaStream_t stream) {
  const uint32_t& num_instances = model_.num_instances;
  uint32_t local_samples_size =
      ceildiv<uint32_t>(data_->batch_size, num_instances) * data_->table_sizes.size();

  int num_sm = gpu_resource.get_sm_count();
  int n_blocks = 16 * num_sm;  // TODO: better heuristics

  /* Each model reads from the gradients of each network */
  PROFILE_RECORD("inf_update_model_direct.infrequent_update_model_direct.start", stream, false);
  infrequent_embedding_kernels::
      infrequent_update_model_direct<<<n_blocks, embedding_vec_size_, 0, stream>>>(
          gradients_pointers_.get_ptr(), infrequent_embedding_vectors_.get_ptr(),
          this->indices_view_, model_.category_location.get_ptr(), model_.num_instances,
          model_.global_instance_id, embedding_vec_size_, local_samples_size, dev_lr, scale);
  CK_CUDA_THROW_(cudaPeekAtLastError());
  PROFILE_RECORD("inf_update_model_direct.infrequent_update_model_direct.stop", stream, false);
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::calculate_model_indices_sizes_from_offsets(
    cudaStream_t stream) {
  auto indices = this->indices_view_;
  constexpr size_t TPB = 256;
  const size_t n_blocks = ceildiv<size_t>(model_.num_instances, TPB);
  infrequent_embedding_kernels::offsets_to_sizes<<<n_blocks, TPB, 0, stream>>>(
      model_indices_sizes_.get_ptr(), [=] __device__() { return indices->model_indices_offsets; },
      embedding_vec_size_ * sizeof(emtype), model_.num_instances);
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::calculate_network_indices_sizes_from_offsets(
    cudaStream_t stream) {
  auto indices = this->indices_view_;
  constexpr size_t TPB = 256;
  const size_t n_blocks = ceildiv<size_t>(model_.num_instances, TPB);
  infrequent_embedding_kernels::offsets_to_sizes<<<n_blocks, TPB, 0, stream>>>(
      network_indices_sizes_.get_ptr(),
      [=] __device__() { return indices->network_indices_offsets; },
      embedding_vec_size_ * sizeof(emtype), model_.num_instances);
}

template class InfrequentEmbeddingBase<uint32_t>;
template class InfrequentEmbeddingBase<long long>;

template class InfrequentEmbedding<uint32_t, __half>;
template class InfrequentEmbedding<uint32_t, float>;
template class InfrequentEmbedding<long long, __half>;
template class InfrequentEmbedding<long long, float>;
}  // namespace hybrid_embedding

}  // namespace HugeCTR
