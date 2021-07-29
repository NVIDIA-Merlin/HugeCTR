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
__global__ void hier_update_model(
    const uint32_t* __restrict__ model_indices, const uint32_t* __restrict__ model_indices_offsets,
    const dtype* __restrict__ samples, const dtype* __restrict__ category_location,
    const emtype* __restrict__ gradients, float* __restrict__ embedding_vectors,
    uint32_t embedding_vec_size, uint32_t num_instances, uint32_t local_samples_size,
    uint32_t local_comm_buff_size, const float* __restrict__ lr_ptr, const float scale) {
  float lr = __ldg(lr_ptr) / scale;
  const uint32_t num_indices = model_indices_offsets[num_instances];

  // Load offset only when the network_id changes
  uint32_t previous_network_id = 0;
  uint32_t offset = 0;

  for (uint32_t i = blockIdx.x; i < num_indices; i += gridDim.x) {
    uint32_t index = model_indices[i];
    dtype category = samples[index];
    dtype location = category_location[2 * category + 1];
    uint32_t network_id = index / local_samples_size;
    if (network_id != previous_network_id) {
      offset = model_indices_offsets[network_id];
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
    const uint32_t* __restrict__ model_indices, const uint32_t* __restrict__ model_indices_offsets,
    const dtype* __restrict__ samples, const dtype* __restrict__ category_location,
    uint32_t num_instances, uint32_t model_id, uint32_t embedding_vec_size,
    uint32_t local_samples_size, const float* __restrict__ lr_ptr, const float scale) {
  float lr = __ldg(lr_ptr) / scale;
  // Shift pattern
  const uint32_t offset = __ldg(model_indices_offsets + model_id + 1);
  const uint32_t num_model_indices = __ldg(model_indices_offsets + num_instances);

  for (uint32_t i = blockIdx.x; i < num_model_indices; i += gridDim.x) {
    uint32_t vid = (i + offset) % num_model_indices;

    uint32_t index = model_indices[vid];
    uint32_t network_id = index / local_samples_size;
    uint32_t local_index = index % local_samples_size;
    dtype category = samples[index];
    uint32_t location = category_location[2 * category + 1];

    const emtype* gradients = gradients_pointers[network_id];

    atomicAdd(embedding_vectors + location * embedding_vec_size + threadIdx.x,
              -lr * TypeConvertFunc<float, emtype>::convert(
                        gradients[local_index * embedding_vec_size + threadIdx.x]));
  }
}

template <typename dtype>
__global__ void calculate_network_indices_mask(const dtype* __restrict__ local_samples,
                                               const dtype* __restrict__ category_location,
                                               bool* mask, uint32_t local_samples_size,
                                               uint32_t num_instances) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < local_samples_size;
       i += gridDim.x * blockDim.x) {
    dtype category = local_samples[i];
    uint32_t model_id = static_cast<uint32_t>(category_location[2 * category]);
    for (uint32_t section_id = 0; section_id < num_instances; section_id++) {
      mask[local_samples_size * section_id + i] = (model_id == section_id);
    }
  }
}

}  // namespace infrequent_embedding_kernels

template <typename dtype, typename emtype>
InfrequentEmbedding<dtype, emtype>::InfrequentEmbedding(const Data<dtype>& data_train,
                                                        const Data<dtype>& data_evaluate,
                                                        const Model<dtype>& model,
                                                        const GPUResource& gpu_resource,
                                                        uint32_t embedding_vec_size)
    : model_(model),
      data_train_(data_train),
      data_evaluate_(data_evaluate),
      data_(data_train),  // Temporary
      gpu_resource(gpu_resource),
      embedding_vec_size_(embedding_vec_size) {
  auto buf = GeneralBuffer2<CudaAllocator>::create();
  auto managed_buf = GeneralBuffer2<CudaManagedAllocator>::create();

  size_t universe_batch_size = std::max(data_train.batch_size, data_evaluate.batch_size);
  buf->reserve({ceildiv<size_t>(model.num_categories, model.num_instances), embedding_vec_size_},
               &infrequent_embedding_vectors_);
  buf->reserve({universe_batch_size, data_train.table_sizes.size()}, &model_indices_);
  managed_buf->reserve({model.num_instances + 1, 1}, &model_indices_offsets_);
  buf->reserve({model_.num_instances}, &model_indices_sizes_);
  buf->reserve({model_.num_instances},
               &model_indices_sizes_ptrs_);  // TODO: should be local instances
  buf->reserve(
      {ceildiv<size_t>(universe_batch_size, model.num_instances), data_train.table_sizes.size()},
      &network_indices_);
  managed_buf->reserve({model.num_instances + 1, 1}, &network_indices_offsets_);
  buf->reserve({model_.num_instances}, &network_indices_sizes_);
  buf->reserve({model_.num_instances}, &network_indices_sizes_ptrs_);

  // Temporary storage
  calculate_model_indices_temp_storage_bytes();
  calculate_network_indices_temp_storage_bytes();
  buf->reserve({model_indices_temp_storage_bytes, 1}, &model_indices_temp_storage_);
  buf->reserve({network_indices_temp_storage_bytes, 1}, &network_indices_temp_storage_);

  buf->reserve({model.num_instances, 1}, &interaction_layer_input_pointers_train_);
  buf->reserve({model.num_instances, 1}, &interaction_layer_input_pointers_eval_);
  buf->reserve({model.num_instances, 1}, &gradients_pointers_);
  buf->allocate();

  managed_buf->allocate();
  int current_device;
  CK_CUDA_THROW_(cudaGetDevice(&current_device));
  CK_CUDA_THROW_(cudaMemAdvise(managed_buf->get_ptr(), managed_buf->get_size_in_bytes(),
                               cudaMemAdviseSetReadMostly, current_device));
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::initialize_embedding_vectors() {
  CudaDeviceContext context(gpu_resource.get_device_id());

  const size_t num_tables = data_.table_sizes.size();
  for (size_t i = 0; i < num_tables; i++) {
    float up_bound = sqrt(1.f / data_.table_sizes[i]);

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
  auto model_indices = model_indices_.get_ptr();
  auto samples = data_.samples.get_ptr();
  auto category_location = model_.category_location.get_ptr();
  auto infrequent_embedding_vectors = infrequent_embedding_vectors_.get_ptr();
  auto embedding_vec_size = embedding_vec_size_;

  auto copy_desc = CopyDescriptors::make_OneToOne<float, emtype, 1>(
      model_indices_offsets_.get_ptr() + model_.num_instances, embedding_vec_size,
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<float, emtype, 1> {
        uint32_t index = model_indices[i];
        dtype category = samples[index];
        dtype location = category_location[2 * category + 1];

        return {infrequent_embedding_vectors + location * embedding_vec_size,
                {message_buffer + i * embedding_vec_size},
                {true}};
      });

  shuffle(copy_desc, stream, data_.samples.get_num_elements() / model_.num_instances / 8);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::fused_intra_forward_model(emtype** message_buffer,
                                                                   cudaStream_t stream) {
  auto model_indices = model_indices_.get_ptr();
  auto model_indices_offsets = model_indices_offsets_.get_ptr();
  auto samples = data_.samples.get_ptr();
  auto category_location = model_.category_location.get_ptr();
  auto infrequent_embedding_vectors = infrequent_embedding_vectors_.get_ptr();
  size_t embedding_vec_size = embedding_vec_size_;
  auto local_instance_id = model_.instance_id;
  auto num_instances = model_.num_instances;
  auto per_node_instances = num_instances / model_.h_num_instances_per_node.size();
  uint32_t local_samples_size =
      ceildiv<uint32_t>(data_.batch_size, num_instances) * data_.table_sizes.size();

  uint32_t local_comm_buff_size =
      ceildiv<uint32_t>(max_num_infrequent_per_batch_, model_.num_instances);

  auto copy_desc = CopyDescriptors::make_OneToOne<float, emtype, 1>(
      model_indices_offsets_.get_ptr() + num_instances, embedding_vec_size,
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<float, emtype, 1> {
        uint32_t num_selected = model_indices_offsets[num_instances];
        uint32_t vid = (i + model_indices_offsets[(local_instance_id + 1) % per_node_instances]) %
                       num_selected;
        uint32_t index = model_indices[vid];
        uint32_t network_id = (index / local_samples_size);
        dtype category = samples[index];
        dtype location = category_location[2 * category + 1];
        uint32_t local_network_id = (network_id % per_node_instances);
        emtype* output_ptr =
            &message_buffer[local_network_id][(network_id - local_network_id + local_instance_id) *
                                              local_comm_buff_size * embedding_vec_size];

        return {infrequent_embedding_vectors + location * embedding_vec_size,
                {output_ptr + (vid - model_indices_offsets[network_id]) * embedding_vec_size},
                {true}};
      });

  shuffle(copy_desc, stream, data_.samples.get_num_elements() / model_.num_instances / 8);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::forward_network(const emtype* message_buffer,
                                                         emtype* interaction_layer_input,
                                                         cudaStream_t stream) {
  auto network_indices = network_indices_.get_ptr();
  auto embedding_vec_size = embedding_vec_size_;

  auto copy_desc = CopyDescriptors::make_OneToOne<emtype, emtype, 1>(
      network_indices_offsets_.get_ptr() + model_.num_instances, embedding_vec_size,
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<emtype, emtype, 1> {
        uint32_t index = network_indices[i];
        return {message_buffer + i * embedding_vec_size,
                {interaction_layer_input + index * embedding_vec_size},
                {true}};
      });

  shuffle(copy_desc, stream, data_.samples.get_num_elements() / model_.num_instances / 8);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::hier_forward_network(const emtype* message_buffer,
                                                              emtype* interaction_layer_input,
                                                              cudaStream_t stream) {
  auto network_indices = network_indices_.get_ptr();
  auto network_indices_offsets = network_indices_offsets_.get_ptr();
  auto embedding_vec_size = embedding_vec_size_;
  uint32_t local_samples_size =
      ceildiv<uint32_t>(data_.batch_size, model_.num_instances) * data_.table_sizes.size();
  uint32_t local_comm_buff_size =
      ceildiv<uint32_t>(max_num_infrequent_per_batch_, model_.num_instances);

  auto copy_desc = CopyDescriptors::make_OneToOne<emtype, emtype, 1>(
      network_indices_offsets_.get_ptr() + model_.num_instances, embedding_vec_size,
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<emtype, emtype, 1> {
        uint32_t index = network_indices[i];

        // Find model id and offset
        uint32_t model_id = 0;
        uint32_t offset = 0;
        uint32_t next_offset = network_indices_offsets[1];
        while (next_offset <= i) {
          offset = next_offset;
          model_id++;
          next_offset = network_indices_offsets[model_id + 1];
        }

        return {
            message_buffer + (model_id * local_comm_buff_size + i - offset) * embedding_vec_size,
            {interaction_layer_input + index * embedding_vec_size},
            {true}};
      });

  shuffle(copy_desc, stream, data_.samples.get_num_elements() / model_.num_instances / 8);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

/** Forward network for single GPU (no communications) */
template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::forward_network_direct(bool is_train,
                                                                cudaStream_t stream) {
  const uint32_t num_instances = model_.num_instances;
  const uint32_t model_id = model_.global_instance_id;
  uint32_t local_samples_size =
      ceildiv<uint32_t>(data_.batch_size, num_instances) * data_.table_sizes.size();

  auto interaction_layer_input_pointers = is_train
                                              ? interaction_layer_input_pointers_train_.get_ptr()
                                              : interaction_layer_input_pointers_eval_.get_ptr();
  auto model_indices = model_indices_.get_ptr();
  auto model_indices_offsets = model_indices_offsets_.get_ptr();
  auto category_location = model_.category_location.get_ptr();
  auto samples = data_.samples.get_ptr();
  auto model_table = infrequent_embedding_vectors_.get_ptr();
  auto embedding_vec_size = embedding_vec_size_;

  auto copy_desc = CopyDescriptors::make_OneToOne<float, emtype, 1>(
      model_indices_offsets + num_instances, embedding_vec_size,
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<float, emtype, 1> {
        const uint32_t offset = model_indices_offsets[model_id + 1];
        const uint32_t num_model_indices = model_indices_offsets[num_instances];
        const uint32_t vid = (i + offset) % num_model_indices;
        const uint32_t index = model_indices[vid];

        const dtype category = samples[index];
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
  auto network_indices = network_indices_.get_ptr();
  auto embedding_vec_size = embedding_vec_size_;

  auto copy_desc = CopyDescriptors::make_OneToOne<emtype, emtype, 1>(
      network_indices_offsets_.get_ptr() + model_.num_instances, embedding_vec_size,
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<emtype, emtype, 1> {
        uint32_t index = network_indices[i];

        return {gradients + index * embedding_vec_size,
                {message_buffer + i * embedding_vec_size},
                {true}};
      });

  shuffle(copy_desc, stream, data_.samples.get_num_elements() / model_.num_instances / 8);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::fused_intra_update_network(const emtype* gradients,
                                                                    emtype** message_buffer,
                                                                    cudaStream_t stream) {
  auto network_indices = network_indices_.get_ptr();
  auto network_indices_offsets = network_indices_offsets_.get_ptr();
  size_t embedding_vec_size = embedding_vec_size_;
  auto local_instance_id = model_.instance_id;
  auto num_instances = model_.num_instances;
  auto per_node_instances = num_instances / model_.h_num_instances_per_node.size();
  uint32_t local_comm_buff_size =
      ceildiv<uint32_t>(max_num_infrequent_per_train_batch_, model_.num_instances);

  auto copy_desc = CopyDescriptors::make_OneToOne<emtype, emtype, 1>(
      network_indices_offsets_.get_ptr() + model_.num_instances, embedding_vec_size,
      [=] __device__(size_t i) -> CopyDescriptors::CopyDetails<emtype, emtype, 1> {
        uint32_t num_selected = network_indices_offsets[num_instances];
        uint32_t vid = (i + network_indices_offsets[(local_instance_id + 1) % per_node_instances]) %
                       num_selected;
        uint32_t index = network_indices[vid];

        uint32_t model_id;
        for (model_id = 0; model_id < num_instances && network_indices_offsets[model_id + 1] <= vid;
             model_id++)
          ;

        uint32_t local_model_id = (model_id % per_node_instances);
        emtype* output_ptr =
            &message_buffer[local_model_id][(model_id - local_model_id + local_instance_id) *
                                            local_comm_buff_size * embedding_vec_size];

        return {gradients + index * embedding_vec_size,
                {output_ptr + (vid - network_indices_offsets[model_id]) * embedding_vec_size},
                {true}};
      });

  shuffle(copy_desc, stream, data_.samples.get_num_elements() / model_.num_instances / 8);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::update_model(const emtype* message_buffer, float* dev_lr,
                                                      float scale, cudaStream_t stream) {
  const uint32_t* __restrict__ model_indices = model_indices_.get_ptr();
  const dtype* __restrict__ samples = data_.samples.get_ptr();
  const dtype* __restrict__ category_location = model_.category_location.get_ptr();

  uint32_t n_blocks = gpu_resource.get_sm_count();

  sgd_atomic_update(
      message_buffer, infrequent_embedding_vectors_.get_ptr(),
      model_indices_offsets_.get_ptr() + model_.num_instances,
      [model_indices, samples, category_location] __device__(uint32_t i) {
        uint32_t index = model_indices[i];
        dtype category = samples[index];
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
      ceildiv<uint32_t>(data_.batch_size, num_instances) * data_.table_sizes.size();
  uint32_t local_comm_buff_size =
      ceildiv<uint32_t>(max_num_infrequent_per_train_batch_, model_.num_instances);
  const uint32_t* __restrict__ model_indices = model_indices_.get_ptr();
  const dtype* __restrict__ samples = data_.samples.get_ptr();
  const dtype* __restrict__ category_location = model_.category_location.get_ptr();

  int num_sm = gpu_resource.get_sm_count();
  int n_blocks = 16 * num_sm;  // TODO: better heuristics

  infrequent_embedding_kernels::hier_update_model<<<n_blocks, embedding_vec_size_, 0, stream>>>(
      model_indices_.get_ptr(), model_indices_offsets_.get_ptr(), data_.samples.get_ptr(),
      model_.category_location.get_ptr(), message_buffer, infrequent_embedding_vectors_.get_ptr(),
      embedding_vec_size_, num_instances, local_samples_size, local_comm_buff_size, dev_lr, scale);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

/** Update model for single GPU (no communications), lr is a device variable */
template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::update_model_direct(float* dev_lr, float scale,
                                                             cudaStream_t stream) {
  const uint32_t& num_instances = model_.num_instances;
  uint32_t local_samples_size =
      ceildiv<uint32_t>(data_.batch_size, num_instances) * data_.table_sizes.size();

  int num_sm = gpu_resource.get_sm_count();
  int n_blocks = 16 * num_sm;  // TODO: better heuristics

  /* Each model reads from the gradients of each network */
  PROFILE_RECORD("inf_update_model_direct.infrequent_update_model_direct.start", stream, false);
  infrequent_embedding_kernels::
      infrequent_update_model_direct<<<n_blocks, embedding_vec_size_, 0, stream>>>(
          gradients_pointers_.get_ptr(), infrequent_embedding_vectors_.get_ptr(),
          model_indices_.get_ptr(), model_indices_offsets_.get_ptr(), data_.samples.get_ptr(),
          model_.category_location.get_ptr(), model_.num_instances, model_.global_instance_id,
          embedding_vec_size_, local_samples_size, dev_lr, scale);
  CK_CUDA_THROW_(cudaPeekAtLastError());
  PROFILE_RECORD("inf_update_model_direct.infrequent_update_model_direct.stop", stream, false);
}

template <typename dtype>
struct ModelIndicesSelectOp {
  const dtype* samples;
  const dtype* category_location;
  uint32_t my_model_id;
  __host__ __device__ __forceinline__ ModelIndicesSelectOp(const dtype* samples,
                                                           const dtype* category_location,
                                                           uint32_t my_model_id)
      : samples(samples), category_location(category_location), my_model_id(my_model_id) {}
  __device__ __forceinline__ bool operator()(const uint32_t& idx) const {
    dtype category = __ldg(samples + idx);
    dtype model_id = __ldg(category_location + 2 * category);
    return model_id == my_model_id;
  }
};

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::calculate_model_indices_temp_storage_bytes() {
  size_t max_batch_size = std::max(data_train_.batch_size, data_evaluate_.batch_size);

  cub::CountingInputIterator<uint32_t> counting(0);
  ModelIndicesSelectOp<dtype> select_op(nullptr, nullptr, 0);
  cub::DeviceSelect::If(nullptr, model_indices_temp_storage_bytes, counting, (uint32_t*)nullptr,
                        (uint32_t*)nullptr, max_batch_size * data_.table_sizes.size(), select_op,
                        0);
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::calculate_model_indices(cudaStream_t stream) {
  const uint32_t& num_instances = model_.num_instances;

  size_t local_batch_size = ceildiv<size_t>(data_.batch_size, num_instances);

  // Select indices of infrequent categories belonging to this model
  cub::CountingInputIterator<uint32_t> counting(0);
  ModelIndicesSelectOp<dtype> select_op(data_.samples.get_ptr(), model_.category_location.get_ptr(),
                                        model_.global_instance_id);
  PROFILE_RECORD("inf_calculate_model_indices.device_select_if.start", stream, false);
  cub::DeviceSelect::If(reinterpret_cast<void*>(model_indices_temp_storage_.get_ptr()),
                        model_indices_temp_storage_bytes, counting, model_indices_.get_ptr(),
                        model_indices_offsets_.get_ptr() + num_instances,
                        data_.batch_size * data_.table_sizes.size(), select_op, stream);
  PROFILE_RECORD("inf_calculate_model_indices.device_select_if.stop", stream, false);

  // Compute offsets
  constexpr size_t TPB = 256;
  const size_t n_blocks = ceildiv<size_t>(num_instances, TPB);
  PROFILE_RECORD("inf_calculate_model_indices.offsets_kernel.start", stream, false);
  offsets_kernel<<<n_blocks, TPB, 0, stream>>>(model_indices_.get_ptr(),
                                               model_indices_offsets_.get_ptr(), num_instances,
                                               local_batch_size * data_.table_sizes.size());
  PROFILE_RECORD("inf_calculate_model_indices.offsets_kernel.stop", stream, false);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

static __global__ void offsets_to_sizes(size_t* sizes, uint32_t* offsets, size_t element_size,
                                        uint32_t num_instances) {
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < num_instances;
       t += gridDim.x * blockDim.x) {
    sizes[t] = (offsets[t + 1] - offsets[t]) * element_size;
  }
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::calculate_model_indices_sizes_from_offsets(
    cudaStream_t stream) {
  constexpr size_t TPB = 256;
  const size_t n_blocks = ceildiv<size_t>(model_.num_instances, TPB);
  offsets_to_sizes<<<n_blocks, TPB, 0, stream>>>(
      model_indices_sizes_.get_ptr(), model_indices_offsets_.get_ptr(),
      embedding_vec_size_ * sizeof(emtype), model_.num_instances);
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::calculate_network_indices_temp_storage_bytes() {
  size_t max_batch_size = std::max(data_train_.batch_size, data_evaluate_.batch_size);
  const uint32_t num_instances = model_.num_instances;
  uint32_t samples_size = max_batch_size * data_.table_sizes.size();
  uint32_t local_samples_size = ceildiv<uint32_t>(samples_size, num_instances);

  // Calculate select bytes
  size_t select_bytes = 0;
  cub::CountingInputIterator<uint32_t> counting(0);
  cub::DeviceSelect::Flagged(nullptr, select_bytes, counting, (bool*)nullptr, (uint32_t*)nullptr,
                             (uint32_t*)nullptr, samples_size, 0);

  // Total size
  constexpr uint32_t align = 256;
  network_indices_temp_storage_bytes =
      alignTo<size_t>(sizeof(bool) * samples_size, align) + select_bytes;
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::calculate_network_indices(cudaStream_t stream) {
  const uint32_t num_instances = model_.num_instances;
  uint32_t samples_size = data_.batch_size * data_.table_sizes.size();
  uint32_t local_samples_size = ceildiv<uint32_t>(samples_size, num_instances);

  // Temporary storage
  constexpr uint32_t align = 256;
  char* scratch_ptr = network_indices_temp_storage_.get_ptr();
  size_t scratch_offset = 0;
  bool* d_mask = reinterpret_cast<bool*>(scratch_ptr + scratch_offset);
  scratch_offset += alignTo<size_t>(sizeof(bool) * samples_size, align);
  void* d_temp_storage = reinterpret_cast<void*>(scratch_ptr + scratch_offset);
  size_t temp_storage_bytes = network_indices_temp_storage_bytes - scratch_offset;

  // Compute mask (for each source GPU, whether each element in the batch is located there)
  constexpr uint32_t TPB_mask = 256;
  uint32_t n_blocks_mask = ceildiv<uint32_t>(local_samples_size, TPB_mask);
  PROFILE_RECORD("inf_calculate_network_indices.calculate_network_indices_mask.start", stream,
                 false);
  infrequent_embedding_kernels::
      calculate_network_indices_mask<<<n_blocks_mask, TPB_mask, 0, stream>>>(
          data_.samples.get_ptr() + model_.global_instance_id * local_samples_size,
          model_.category_location.get_ptr(), d_mask, local_samples_size, num_instances);
  CK_CUDA_THROW_(cudaPeekAtLastError());
  PROFILE_RECORD("inf_calculate_network_indices.calculate_network_indices_mask.stop", stream,
                 false);

  // Select indices according to the mask
  cub::CountingInputIterator<uint32_t> counting(0);
  PROFILE_RECORD("inf_calculate_network_indices.device_select_flagged.start", stream, false);
  cub::DeviceSelect::Flagged(
      d_temp_storage, temp_storage_bytes, counting, d_mask, network_indices_.get_ptr(),
      network_indices_offsets_.get_ptr() + num_instances, samples_size, stream);
  PROFILE_RECORD("inf_calculate_network_indices.device_select_flagged.stop", stream, false);

  // Compute offsets
  constexpr uint32_t TPB_offsets = 256;
  uint32_t n_blocks_offsets = ceildiv<uint32_t>(num_instances, TPB_offsets);
  PROFILE_RECORD("inf_calculate_network_indices.offsets_kernel.start", stream, false);
  offsets_kernel<<<n_blocks_offsets, TPB_offsets, 0, stream>>>(network_indices_.get_ptr(),
                                                               network_indices_offsets_.get_ptr(),
                                                               num_instances, local_samples_size);
  CK_CUDA_THROW_(cudaPeekAtLastError());
  PROFILE_RECORD("inf_calculate_network_indices.offsets_kernel.stop", stream, false);

  // Re-map indices between 0 and local_samples_size - 1
  uint32_t TPB_remap = 256;
  uint32_t n_blocks_remap = gpu_resource.get_sm_count();
  PROFILE_RECORD("inf_calculate_network_indices.modulo_kernel.start", stream, false);
  modulo_kernel<<<n_blocks_remap, TPB_remap, 0, stream>>>(
      network_indices_.get_ptr(), network_indices_offsets_.get_ptr() + num_instances,
      local_samples_size);
  CK_CUDA_THROW_(cudaPeekAtLastError());
  PROFILE_RECORD("inf_calculate_network_indices.modulo_kernel.stop", stream, false);
}

template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::calculate_network_indices_sizes_from_offsets(
    cudaStream_t stream) {
  constexpr size_t TPB = 256;
  const size_t n_blocks = ceildiv<size_t>(model_.num_instances, TPB);
  offsets_to_sizes<<<n_blocks, TPB, 0, stream>>>(
      network_indices_sizes_.get_ptr(), network_indices_offsets_.get_ptr(),
      embedding_vec_size_ * sizeof(emtype), model_.num_instances);
}

template class InfrequentEmbedding<uint32_t, __half>;
template class InfrequentEmbedding<uint32_t, float>;
template class InfrequentEmbedding<long long, __half>;
template class InfrequentEmbedding<long long, float>;
}  // namespace hybrid_embedding

}  // namespace HugeCTR
