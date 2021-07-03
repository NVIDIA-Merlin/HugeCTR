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

#include <inference/embedding_feature_combiner.hpp>
#include <utils.cuh>
#include <utils.hpp>
 
#include <algorithm>
#include <functional>
 
#ifndef NDEBUG
#include <iostream>
#endif
 
namespace HugeCTR {
 
namespace {

template <typename TypeEmbedding>
__global__ void embedding_feature_combine_kernel(const float* input, TypeEmbedding* output, const int* row_ptrs,
                                                int batch_size, int slot_num, int embedding_vec_size, 
                                                EmbeddingFeatureCombiner_t combiner_type) {
  const auto &block = cooperative_groups::this_thread_block();
  // each block partition corresponding to one sample
  const int bid = block.group_index().x;
  // each thread corresponding to one element in the embedding vector
  const int tid = block.thread_rank();
   
  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;
      int row_offset = row_ptrs[feature_row_index];                   // row offset within input
      int feature_num = row_ptrs[feature_row_index+1] - row_offset;  // num of feature vectors in one slot
       
      float tmp = 0.0f;
      // reduce in one slot
      for (int j = 0; j < feature_num; j++)
        tmp += input[(row_offset + j)*embedding_vec_size + tid];

      if (combiner_type == EmbeddingFeatureCombiner_t::Mean && feature_num > 1) {
        tmp /= feature_num;
      }
      output[feature_row_index*embedding_vec_size + tid] = tmp;
    } // end for
  } // end if
}

template <>
__global__ void embedding_feature_combine_kernel(const float* input, __half* output, const int* row_ptrs,
                                                int batch_size, int slot_num, int embedding_vec_size, 
                                                EmbeddingFeatureCombiner_t combiner_type) {
  const auto &block = cooperative_groups::this_thread_block();
  // each block partition corresponding to one sample
  const int bid = block.group_index().x;
  // each thread corresponding to one element in the embedding vector
  const int tid = block.thread_rank();
   
  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;
      int row_offset = row_ptrs[feature_row_index];                   // row offset within input
      int feature_num = row_ptrs[feature_row_index+1] - row_offset;  // num of feature vectors in one slot
       
      float tmp = 0.0f;
      // reduce in one slot
      for (int j = 0; j < feature_num; j++)
        tmp += input[(row_offset + j)*embedding_vec_size + tid];

      if (combiner_type == EmbeddingFeatureCombiner_t::Mean && feature_num > 1) {
        tmp /= feature_num;
      }
      output[feature_row_index*embedding_vec_size + tid] = __float2half(tmp);
    } // end for
  } // end if
}

template <typename TypeEmbedding, int TileSize>
__global__ void embedding_feature_combine_tiled_kernel(const float* input, TypeEmbedding* output, const int* row_ptrs,
                                                      int batch_size, int slot_num, int embedding_vec_size, 
                                                      EmbeddingFeatureCombiner_t combiner_type) {
  const auto &block = cooperative_groups::this_thread_block();
  const auto &tile = cooperative_groups::tiled_partition<TileSize>(block);
  // each block partition corresponding to one sample
  const int bid = block.group_index().x * tile.meta_group_size() + tile.meta_group_rank();
  // each thread corresponding to one element in the embedding vector
  const int tid = tile.thread_rank();

  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;
      int row_offset = row_ptrs[feature_row_index];                   // row offset within input
      int feature_num = row_ptrs[feature_row_index+1] - row_offset;  // num of feature vectors in one slot

      float tmp = 0.0f;
      // reduce in one slot
      for (int j = 0; j < feature_num; j++)
        tmp += input[(row_offset + j)*embedding_vec_size + tid];

      if (combiner_type == EmbeddingFeatureCombiner_t::Mean && feature_num > 1) {
        tmp /= feature_num;
      }
      output[feature_row_index*embedding_vec_size + tid] = tmp;
    } // end for
  } // end if
}

template <int TileSize>
__global__ void embedding_feature_combine_tiled_kernel(const float* input, __half* output, const int* row_ptrs,
                                                      int batch_size, int slot_num, int embedding_vec_size, 
                                                      EmbeddingFeatureCombiner_t combiner_type) {
  const auto &block = cooperative_groups::this_thread_block();
  const auto &tile = cooperative_groups::tiled_partition<TileSize>(block);
  // each block partition corresponding to one sample
  const int bid = block.group_index().x * tile.meta_group_size() + tile.meta_group_rank();
  // each thread corresponding to one element in the embedding vector
  const int tid = tile.thread_rank();

  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;
      int row_offset = row_ptrs[feature_row_index];                   // row offset within input
      int feature_num = row_ptrs[feature_row_index+1] - row_offset;  // num of feature vectors in one slot

      float tmp = 0.0f;
      // reduce in one slot
      for (int j = 0; j < feature_num; j++)
        tmp += input[(row_offset + j)*embedding_vec_size + tid];

      if (combiner_type == EmbeddingFeatureCombiner_t::Mean && feature_num > 1) {
        tmp /= feature_num;
      }
      output[feature_row_index*embedding_vec_size + tid] = __float2half(tmp);
    } // end for
  } // end if
}

template <typename TypeEmbedding>
void launch_embedding_feature_combine_kernel(const float* input, TypeEmbedding* output, const int* row_ptrs,
                                            int batch_size, int slot_num, int embedding_vec_size, 
                                            EmbeddingFeatureCombiner_t combiner_type, cudaStream_t stream) {
  if (embedding_vec_size <= 2) {
    embedding_feature_combine_tiled_kernel<TypeEmbedding, 2>
        <<< (batch_size - 1) / 32 + 1, 64, 0, stream>>>(input, output, row_ptrs, batch_size, 
                                                                        slot_num, embedding_vec_size, combiner_type);
  } else if (embedding_vec_size <= 4) {
    embedding_feature_combine_tiled_kernel<TypeEmbedding, 4>
        <<< (batch_size - 1) / 16 + 1, 64, 0, stream>>>(input, output, row_ptrs, batch_size, 
                                                                        slot_num, embedding_vec_size, combiner_type);
  } else if (embedding_vec_size <= 8) {
    embedding_feature_combine_tiled_kernel<TypeEmbedding, 8>
        <<< (batch_size - 1) / 8 + 1, 64, 0, stream>>>(input, output, row_ptrs, batch_size, 
                                                                        slot_num, embedding_vec_size, combiner_type);
  } else if (embedding_vec_size <= 16) {
    embedding_feature_combine_tiled_kernel<TypeEmbedding, 16>
        <<< (batch_size - 1) / 4 + 1, 64, 0, stream>>>(input, output, row_ptrs, batch_size, 
                                                                        slot_num, embedding_vec_size, combiner_type);
  } else if (embedding_vec_size <= 32) {
    embedding_feature_combine_tiled_kernel<TypeEmbedding, 32>
        <<< (batch_size - 1) / 2 + 1, 64, 0, stream>>>(input, output, row_ptrs, batch_size, 
                                                                        slot_num, embedding_vec_size, combiner_type);
  } else {
    // each thread corresponds to one element in an embedding vector
    embedding_feature_combine_kernel<<<batch_size, embedding_vec_size, 0, stream>>>(input, output, row_ptrs, batch_size, slot_num, embedding_vec_size, combiner_type);
  }
}

}  // end of namespace

template <typename TypeEmbedding>
EmbeddingFeatureCombiner<TypeEmbedding>::EmbeddingFeatureCombiner(const std::shared_ptr<Tensor2<float>>& in_tensor,
                                                  const std::shared_ptr<Tensor2<int>>& row_ptrs_tensor,
                                                  Tensor2<TypeEmbedding>& out_tensor, int batch_size, int slot_num, EmbeddingFeatureCombiner_t combiner_type, 
                                                  const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                                                  const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource), slot_num_(slot_num), batch_size_(batch_size), combiner_type_(combiner_type) {
  try {
    // error input checking
    const auto& in_dims = in_tensor->get_dimensions();
    const auto& row_ptrs_dims =row_ptrs_tensor->get_dimensions();
    if ((int)in_dims.size() != 2)
      CK_THROW_(Error_t::WrongInput, "The input tensor must be 2D");
    for (auto i : in_dims) {
      if (i == 0) {
        CK_THROW_(Error_t::WrongInput, "The input dims can not be 0");
      }
    }
 
    if ((int)row_ptrs_dims.size() != 1)
      CK_THROW_(Error_t::WrongInput, "The row pointers tensor must be 1D");
    if ((int)row_ptrs_dims[0] != batch_size * slot_num + 1)
      CK_THROW_(Error_t::WrongInput, "The dimension of row pointers tensor mismatch number of samples");
     
    embedding_vec_size_ = in_dims[1];
    std::vector<size_t> out_dims {static_cast<size_t>(batch_size_), static_cast<size_t>(slot_num_), static_cast<size_t>(embedding_vec_size_)};
    blobs_buff->reserve(out_dims, &out_tensor);
    out_tensors_.push_back(out_tensor);
    in_tensors_.push_back(in_tensor);
    row_ptrs_tensors_.push_back(row_ptrs_tensor);
    const Tensor2<float>* iptr1 = in_tensor.get();
    Tensor2<float>* iptr2 = in_tensors_[0].get();
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

template <typename TypeEmbedding>
void EmbeddingFeatureCombiner<TypeEmbedding>::fprop(bool is_train) {
  if (is_train)
    CK_THROW_(Error_t::IllegalCall, "The fprop() of EmbeddingFeatureCombiner should only be used for inference");
  
  CudaDeviceContext context(get_device_id());
  float* input = in_tensors_[0]->get_ptr();
  TypeEmbedding* output = out_tensors_[0].get_ptr();
  int* row_ptrs = row_ptrs_tensors_[0]->get_ptr();
 
  auto in_dims = in_tensors_[0]->get_dimensions();
  auto out_dims = out_tensors_[0].get_dimensions();
  launch_embedding_feature_combine_kernel(input, output, row_ptrs, batch_size_, slot_num_, embedding_vec_size_, combiner_type_, get_gpu().get_stream());

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template class EmbeddingFeatureCombiner<float>;
template class EmbeddingFeatureCombiner<__half>;
 
}  // namespace HugeCTR
 