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

#pragma once

#include <layer.hpp>
#include <cooperative_groups.h>

namespace HugeCTR {

enum class EmbeddingFeatureCombiner_t { Sum, Mean };

/**
 * Combine the embedding feature vectors by Sum or Mean
 * according to slot_num and row_ptrs
 */
template <typename T>
class EmbeddingFeatureCombiner: public Layer {
  /*
   * stores the references to the input tensors of this layer.
   */
  std::vector<std::shared_ptr<Tensor2<float>>> in_tensors_;
  /*
   * stores the references to the output tensors of this layer.
   */
  Tensors2<T> out_tensors_;
  /*
   * stores the references to the row pointers tensors of this layer.
   */
  std::vector<std::shared_ptr<Tensor2<int>>> row_ptrs_tensors_;

 public:
  /**
   * Ctor of EmbeddingFeatureCombiner.
   * @param in_tensor the embedding feature tensor, must be 2D
   * @param row_ptrs_tensor row pointers tensor, should be 1D (batch_size*slot_num+1,), which indicate which adjacent vectors belong to the same slot (i.e., feature field)
   * @param out_tensor the resulting output tensor, should be 3D (batch_size, slot_num, embedding_vec_size)
   * @param batch_size batch size
   * @param slot_num slot number
   * @param combiner_type combiner type for the features in the same slot, Sum or Mean
   * @param blobs_buff GeneralBuffer used to create the output tensor
   * @param gpu_resource available gpu resource
   */
  EmbeddingFeatureCombiner(const std::shared_ptr<Tensor2<float>>& in_tensor, const std::shared_ptr<Tensor2<int>>& row_ptrs_tensor, 
                 Tensor2<T>& out_tensor, int batch_size, int slot_num, EmbeddingFeatureCombiner_t combiner_type,
                 const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                 const std::shared_ptr<GPUResource>& gpu_resource);
  ~EmbeddingFeatureCombiner() {};

  /**
   * EmbeddingFeatureCombiner's combine operation
   */
  void fprop(bool is_train=false) override;

  void bprop() override { CK_THROW_(Error_t::IllegalCall, "The bprop() of EmbeddingFeatureCombiner is not implemented!"); }

private:
  int batch_size_;
  int slot_num_;
  int embedding_vec_size_;
  EmbeddingFeatureCombiner_t combiner_type_;
};

}  // namespace HugeCTR
