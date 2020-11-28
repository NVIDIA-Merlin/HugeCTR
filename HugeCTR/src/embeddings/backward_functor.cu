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
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

namespace {
// backward kernel function: for combiner=sum
template <typename TypeEmbeddingComp>
__global__ void backward_sum_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                    const TypeEmbeddingComp *top_grad, TypeEmbeddingComp *wgrad) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      size_t feature_index = (size_t)(bid * slot_num + i) * embedding_vec_size + tid;
      wgrad[feature_index] = top_grad[feature_index];
    }
  }
}

__global__ void backward_sum_align2_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                           const __half *top_grad, __half *wgrad) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (bid < batch_size && tid < embedding_vec_size) {
    const __half2 *top_grad2 = reinterpret_cast<const __half2 *>(top_grad);
    __half2 *wgrad2 = reinterpret_cast<__half2 *>(wgrad);

    for (int i = 0; i < slot_num; i++) {
      size_t feature_index = (size_t)(bid * slot_num + i) * embedding_vec_size + tid;
      wgrad2[feature_index] = top_grad2[feature_index];
    }
  }
}

// backward kernel function: for combiner=mean
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void backward_mean_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                     const TypeKey *row_offset, const TypeEmbeddingComp *top_grad,
                                     TypeEmbeddingComp *wgrad) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      size_t feature_row_index = bid * slot_num + i;
      int value_num = row_offset[feature_row_index + 1] - row_offset[feature_row_index];
      float scaler = 1.0f;
      if (value_num > 1) {
        scaler = 1.0f / value_num;  // partial derivatice of MEAN
      }

      size_t feature_index = feature_row_index * embedding_vec_size + tid;
      float g = TypeConvertFunc<float, TypeEmbeddingComp>::convert(top_grad[feature_index]);
      g *= scaler;
      wgrad[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(g);
    }
  }
}

template <typename TypeKey>
__global__ void backward_mean_align2_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                            const TypeKey *row_offset, const __half *top_grad,
                                            __half *wgrad) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (bid < batch_size && tid < embedding_vec_size) {
    const __half2 *top_grad2 = reinterpret_cast<const __half2 *>(top_grad);
    __half2 *wgrad2 = reinterpret_cast<__half2 *>(wgrad);

    for (int i = 0; i < slot_num; i++) {
      size_t feature_row_index = bid * slot_num + i;
      int value_num = row_offset[feature_row_index + 1] - row_offset[feature_row_index];

      __half2 scaler = __float2half2_rn(1.0f);
      if (value_num > 1) {
        scaler = __float2half2_rn(1.0f / (float)value_num);  // partial derivatice of MEAN
      }

      size_t feature_index = feature_row_index * embedding_vec_size + tid;
      wgrad2[feature_index] = __hmul2(scaler, top_grad2[feature_index]);
    }
  }
}

template <typename TypeEmbeddingComp>
void backward_sum(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                  const TypeEmbeddingComp *top_grad, TypeEmbeddingComp *wgrad,
                  cudaStream_t stream) {
  const size_t grid_size = batch_size;  // each block corresponds to a sample
  const size_t block_size = embedding_vec_size;
  backward_sum_kernel<<<grid_size, block_size, 0, stream>>>(batch_size, slot_num,
                                                            embedding_vec_size, top_grad, wgrad);
}

template <>
void backward_sum<__half>(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                          const __half *top_grad, __half *wgrad, cudaStream_t stream) {
  const size_t grid_size = batch_size;  // each block corresponds to a sample
  if (embedding_vec_size % 2 == 0) {
    const size_t block_size =
        embedding_vec_size / 2;  // each thread corresponds to one element in an embedding vetor
    backward_sum_align2_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size, slot_num, embedding_vec_size / 2, top_grad, wgrad);
  } else {
    const size_t block_size = embedding_vec_size;
    backward_sum_kernel<<<grid_size, block_size, 0, stream>>>(batch_size, slot_num,
                                                              embedding_vec_size, top_grad, wgrad);
  }
}

template <typename TypeKey, typename TypeEmbeddingComp>
void backward_mean(size_t batch_size, size_t slot_size, size_t embedding_vec_size,
                   const TypeKey *row_offset, const TypeEmbeddingComp *top_grad,
                   TypeEmbeddingComp *wgrad, cudaStream_t stream) {
  const size_t grid_size = batch_size;  // each block corresponds to a sample
  const size_t block_size = embedding_vec_size;
  backward_mean_kernel<<<grid_size, block_size, 0, stream>>>(
      batch_size, slot_size, embedding_vec_size, row_offset, top_grad, wgrad);
}

template <typename TypeKey>
void backward_mean(size_t batch_size, size_t slot_size, size_t embedding_vec_size,
                   const TypeKey *row_offset, const __half *top_grad, __half *wgrad,
                   cudaStream_t stream) {
  const size_t grid_size = batch_size;  // each block corresponds to a sample
  if (embedding_vec_size % 2 == 0) {
    const size_t block_size = embedding_vec_size / 2;
    backward_mean_align2_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size, slot_size, embedding_vec_size / 2, row_offset, top_grad, wgrad);
  } else {
    const size_t block_size = embedding_vec_size;
    backward_mean_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size, slot_size, embedding_vec_size, row_offset, top_grad, wgrad);
  }
}

}  // namespace

/**
 * backward propagation for DistributedSlotSparseEmbeddingHash
 * The first step of backward propagation: computing the wgrad.
 * @param batch_size batch size for the current mini-batch computation.
 * @param slot_num the number of slots in hash table.
 * @param embedding_vec_size embedding vector size.
 * @param combiner combiner type: 0-sum, 1-mean
 * @param row_offset_allreduce_tensors row_offsets tensors after all_reduce of mulitple GPUs
 * @param embedding_feature_tensors embedding features tensors of multiplu GPUs, storing dgrad
 * from the top layer
 * @param wgrad_tensors wgrad tensors of multi GPUs, the output of this function.
 * @param device_resources all gpus device resources.
 * @param context gpu device context, for switching device
 */
template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::backward(size_t batch_size, size_t slot_num,
                                       size_t embedding_vec_size, int combiner,
                                       const Tensors2<TypeHashKey> &row_offset_allreduce_tensors,
                                       const Tensors2<TypeEmbeddingComp> &embedding_feature_tensors,
                                       Tensors2<TypeEmbeddingComp> &wgrad_tensors,
                                       const ResourceManager &resource_manager) {
  size_t local_gpu_count = resource_manager.get_local_gpu_count();

  CudaDeviceContext context;
  for (size_t id = 0; id < local_gpu_count; id++) {
    const auto &local_gpu = resource_manager.get_local_gpu(id);
    context.set_device(local_gpu->get_device_id());
    const TypeEmbeddingComp *top_grad = embedding_feature_tensors[id].get_ptr();
    const TypeHashKey *row_offset = row_offset_allreduce_tensors[id].get_ptr();
    TypeEmbeddingComp *wgrad = wgrad_tensors[id].get_ptr();

    if (combiner == 0)  // sum
    {
      backward_sum(batch_size, slot_num, embedding_vec_size, top_grad, wgrad,
                   local_gpu->get_stream());
    } else if (combiner == 1)  // mean
    {
      backward_mean(batch_size, slot_num, embedding_vec_size, row_offset, top_grad, wgrad,
                    local_gpu->get_stream());
    } else {
      CK_THROW_(Error_t::WrongInput, "Invalid combiner type ");
    }
  }

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::backward(size_t batch_size,
                                       const std::vector<size_t> &slot_num_per_gpu,
                                       size_t embedding_vec_size, int combiner,
                                       const Tensors2<TypeHashKey> &row_offset_allreduce_tensors,
                                       const Tensors2<TypeEmbeddingComp> &embedding_feature_tensors,
                                       Tensors2<TypeEmbeddingComp> &wgrad_tensors,
                                       const ResourceManager &resource_manager) {
  size_t local_gpu_count = resource_manager.get_local_gpu_count();

  CudaDeviceContext context;
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (slot_num_per_gpu[id] == 0) {
      continue;
    }

    const auto &local_gpu = resource_manager.get_local_gpu(id);
    context.set_device(local_gpu->get_device_id());
    const TypeEmbeddingComp *top_grad = embedding_feature_tensors[id].get_ptr();
    const TypeHashKey *row_offset = row_offset_allreduce_tensors[id].get_ptr();
    TypeEmbeddingComp *wgrad = wgrad_tensors[id].get_ptr();

    if (combiner == 0)  // sum
    {
      backward_sum(batch_size, slot_num_per_gpu[id], embedding_vec_size, top_grad, wgrad,
                   local_gpu->get_stream());
    } else if (combiner == 1)  // mean
    {
      backward_mean(batch_size, slot_num_per_gpu[id], embedding_vec_size, row_offset, top_grad,
                    wgrad, local_gpu->get_stream());
    } else {
      CK_THROW_(Error_t::WrongInput, "Invalid combiner type ");
    }
  }

  return;
}

template void SparseEmbeddingFunctors::backward<unsigned int, float>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner,
    const Tensors2<unsigned int> &row_offset_allreduce_tensors,
    const Tensors2<float> &embedding_feature_tensors, Tensors2<float> &wgrad_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::backward<long long, float>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner,
    const Tensors2<long long> &row_offset_allreduce_tensors,
    const Tensors2<float> &embedding_feature_tensors, Tensors2<float> &wgrad_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::backward<unsigned int, __half>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner,
    const Tensors2<unsigned int> &row_offset_allreduce_tensors,
    const Tensors2<__half> &embedding_feature_tensors, Tensors2<__half> &wgrad_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::backward<long long, __half>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner,
    const Tensors2<long long> &row_offset_allreduce_tensors,
    const Tensors2<__half> &embedding_feature_tensors, Tensors2<__half> &wgrad_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::backward<unsigned int, float>(
    size_t batch_size, const std::vector<size_t> &slot_num_per_gpu, size_t embedding_vec_size,
    int combiner, const Tensors2<unsigned int> &row_offset_allreduce_tensors,
    const Tensors2<float> &embedding_feature_tensors, Tensors2<float> &wgrad_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::backward<long long, float>(
    size_t batch_size, const std::vector<size_t> &slot_num_per_gpu, size_t embedding_vec_size,
    int combiner, const Tensors2<long long> &row_offset_allreduce_tensors,
    const Tensors2<float> &embedding_feature_tensors, Tensors2<float> &wgrad_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::backward<unsigned int, __half>(
    size_t batch_size, const std::vector<size_t> &slot_num_per_gpu, size_t embedding_vec_size,
    int combiner, const Tensors2<unsigned int> &row_offset_allreduce_tensors,
    const Tensors2<__half> &embedding_feature_tensors, Tensors2<__half> &wgrad_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::backward<long long, __half>(
    size_t batch_size, const std::vector<size_t> &slot_num_per_gpu, size_t embedding_vec_size,
    int combiner, const Tensors2<long long> &row_offset_allreduce_tensors,
    const Tensors2<__half> &embedding_feature_tensors, Tensors2<__half> &wgrad_tensors,
    const ResourceManager &resource_manager);

}  // namespace HugeCTR