/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"
#include "HugeCTR/include/utils.cuh"

namespace HugeCTR {

namespace {

// forward kernel function: this is an additional function for combiner=mean (only for Distributed
// Embedding)
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void forward_scale_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                     const TypeKey *row_offset,
                                     TypeEmbeddingComp *embedding_feature) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      size_t feature_row_index = bid * slot_num + i;
      int feature_num = row_offset[feature_row_index + 1] - row_offset[feature_row_index];
      size_t feature_index = feature_row_index * embedding_vec_size + tid;
      float feature =
          TypeConvertFunc<float, TypeEmbeddingComp>::convert(embedding_feature[feature_index]);
      float scaler = 1.0f;
      if (feature_num > 1) {
        scaler = 1.0f / (float)feature_num;
      }

      embedding_feature[feature_index] =
          TypeConvertFunc<TypeEmbeddingComp, float>::convert(feature * scaler);
    }
  }
}

template <typename TypeKey>
__global__ void forward_scale_align2_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                            const TypeKey *row_offset, __half *embedding_feature) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (bid < batch_size && tid < embedding_vec_size) {
    __half2 *embedding_feature2 = reinterpret_cast<__half2 *>(embedding_feature);

    for (int i = 0; i < slot_num; i++) {
      size_t feature_row_index = bid * slot_num + i;
      int feature_num = row_offset[feature_row_index + 1] - row_offset[feature_row_index];
      size_t feature_index = feature_row_index * embedding_vec_size + tid;
      __half2 feature2 = embedding_feature2[feature_index];

      float scaler = 1.0f;
      if (feature_num > 1) {
        scaler = 1.0f / feature_num;
      }
      __half2 scaler2 = __float2half2_rn(scaler);

      // store the embedding vector
      embedding_feature2[feature_row_index * embedding_vec_size + tid] = __hmul2(feature2, scaler2);
    }
  }
}

template <typename TypeKey, typename TypeEmbeddingComp>
void do_forward_scale(size_t batchsize_per_gpu, size_t slot_num, size_t embedding_vec_size,
                      const TypeKey *row_offset, TypeEmbeddingComp *embedding_feature,
                      cudaStream_t stream) {
  const size_t grid_size = batchsize_per_gpu;
  const size_t block_size = embedding_vec_size;
  forward_scale_kernel<<<grid_size, block_size, 0, stream>>>(
      batchsize_per_gpu, slot_num, embedding_vec_size, row_offset, embedding_feature);
};

template <typename TypeKey>
void do_forward_scale(size_t batchsize_per_gpu, size_t slot_num, size_t embedding_vec_size,
                      const TypeKey *row_offset, __half *embedding_feature, cudaStream_t stream) {
  const size_t grid_size = batchsize_per_gpu;
  if (embedding_vec_size % 2 == 0) {
    const size_t block_size = embedding_vec_size / 2;
    forward_scale_align2_kernel<<<grid_size, block_size, 0, stream>>>(
        batchsize_per_gpu, slot_num, embedding_vec_size / 2, row_offset, embedding_feature);
  } else {
    const size_t block_size = embedding_vec_size;
    forward_scale_kernel<<<grid_size, block_size, 0, stream>>>(
        batchsize_per_gpu, slot_num, embedding_vec_size, row_offset, embedding_feature);
  }
};

}  // namespace

/**
 * An additional function for the forward propagation when (combiner=mean).
 *  (only for DistributedSlotSparseEmbeddingHash)
 * @param batch_size batch size for the current mini-batch computation.
 * @param slot_num the number of slots
 * @param embedding_vec_size embedding vector size.
 * @param row_offset_allreduce_tensors row_offsets tensors after all_reduce of mulitple GPUs
 * @param output_tensors forward prop output tensors of multi GPUs
 * @param device_resources all gpus device resources.
 * @param context gpu device context, for switching device
 */
template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::forward_scale(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size,
    const TensorPtrs<TypeHashKey> &row_offset_allreduce_tensors,
    const TensorPtrs<TypeEmbeddingComp> &output_tensors, const GPUResourceGroup &device_resources) {
  CudaDeviceContext context;
  size_t local_gpu_count = device_resources.size();
  size_t total_gpu_count = device_resources.get_total_gpu_count();
  size_t batchsize_per_gpu = batch_size / total_gpu_count;

  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(device_resources[id].get_device_id());

    const TypeHashKey *row_offset =
        row_offset_allreduce_tensors[id]->get_ptr() + id * batchsize_per_gpu * slot_num;
    TypeEmbeddingComp *embedding_feature = output_tensors[id]->get_ptr();

    do_forward_scale(batchsize_per_gpu, slot_num, embedding_vec_size, row_offset, embedding_feature,
                     device_resources[id].get_stream());
  }

  return;
}

template void SparseEmbeddingFunctors::forward_scale<unsigned int, float>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size,
    const TensorPtrs<unsigned int> &row_offset_allreduce_tensors,
    const TensorPtrs<float> &output_tensors, const GPUResourceGroup &device_resources);

template void SparseEmbeddingFunctors::forward_scale<long long, float>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size,
    const TensorPtrs<long long> &row_offset_allreduce_tensors,
    const TensorPtrs<float> &output_tensors, const GPUResourceGroup &device_resources);

template void SparseEmbeddingFunctors::forward_scale<unsigned int, __half>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size,
    const TensorPtrs<unsigned int> &row_offset_allreduce_tensors,
    const TensorPtrs<__half> &output_tensors, const GPUResourceGroup &device_resources);

template void SparseEmbeddingFunctors::forward_scale<long long, __half>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size,
    const TensorPtrs<long long> &row_offset_allreduce_tensors,
    const TensorPtrs<__half> &output_tensors, const GPUResourceGroup &device_resources);

}  // namespace HugeCTR