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

#include <cpu/embedding_feature_combiner_cpu.hpp>
#include <utils.hpp>
 
#include <algorithm>
#include <functional>
 
#ifndef NDEBUG
#include <iostream>
#endif
 
namespace HugeCTR {
 
namespace {

template <typename TypeEmbedding>
void embedding_feature_combine_cpu(const float* input, TypeEmbedding* output, const int* row_ptrs,
                                  int batch_size, int slot_num, int embedding_vec_size, 
                                  EmbeddingFeatureCombiner_t combiner_type) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < slot_num; j++) {
      int feature_row_index = i * slot_num + j;
      int row_offset = row_ptrs[feature_row_index];                   // row offset within input
      int feature_num = row_ptrs[feature_row_index+1] - row_offset;  // num of feature vectors in one slot
      
      for (int k = 0; k < embedding_vec_size; k++) {
        float tmp = 0.0f;
        for (int l =0; l < feature_num; l++) {
          tmp += input[(row_offset + l)*embedding_vec_size + k];
        } // end for l
        if (combiner_type == EmbeddingFeatureCombiner_t::Mean)
          tmp /= feature_num;
        output[feature_row_index*embedding_vec_size + k] = tmp;
      } //end for k
    } // end for j
  } // end for i
}

template <>
void embedding_feature_combine_cpu(const float* input, __half* output, const int* row_ptrs,
                                  int batch_size, int slot_num, int embedding_vec_size, 
                                  EmbeddingFeatureCombiner_t combiner_type) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < slot_num; j++) {
      int feature_row_index = i * slot_num + j;
      int row_offset = row_ptrs[feature_row_index];                   // row offset within input
      int feature_num = row_ptrs[feature_row_index+1] - row_offset;  // num of feature vectors in one slot
      
      for (int k = 0; k < embedding_vec_size; k++) {
        float tmp = 0.0f;
        for (int l =0; l < feature_num; l++) {
          tmp += __half2float(input[(row_offset + l)*embedding_vec_size + k]);
        } // end for l
        if (combiner_type == EmbeddingFeatureCombiner_t::Mean && feature_num > 1) {         
          tmp /= feature_num;
        }
        output[feature_row_index*embedding_vec_size + k] = __float2half(tmp);
      } //end for k
    } // end for j
  } // end for i
}


}  // end of namespace

template <typename TypeEmbedding>
EmbeddingFeatureCombinerCPU<TypeEmbedding>::EmbeddingFeatureCombinerCPU(const std::shared_ptr<Tensor2<float>>& in_tensor,
                                                  const std::shared_ptr<Tensor2<int>>& row_ptrs_tensor,
                                                  Tensor2<TypeEmbedding>& out_tensor, int batch_size, int slot_num, EmbeddingFeatureCombiner_t combiner_type, 
                                                  const std::shared_ptr<GeneralBuffer2<HostAllocator>>& blobs_buff)
    : LayerCPU(), batch_size_(batch_size), slot_num_(slot_num), combiner_type_(combiner_type) {
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
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

template <typename TypeEmbedding>
void EmbeddingFeatureCombinerCPU<TypeEmbedding>::fprop(bool is_train) {
  if (is_train)
    CK_THROW_(Error_t::IllegalCall, "The fprop() of EmbeddingFeatureCombiner should only be used for inference");
  
  float* input = in_tensors_[0]->get_ptr();
  TypeEmbedding* output = out_tensors_[0].get_ptr();
  int* row_ptrs = row_ptrs_tensors_[0]->get_ptr();
 
  auto in_dims = in_tensors_[0]->get_dimensions();
  auto out_dims = out_tensors_[0].get_dimensions();
  embedding_feature_combine_cpu(input, output, row_ptrs, batch_size_, slot_num_,
                              embedding_vec_size_, combiner_type_);
}

template class EmbeddingFeatureCombinerCPU<float>;
template class EmbeddingFeatureCombinerCPU<__half>;
 
}  // namespace HugeCTR
 