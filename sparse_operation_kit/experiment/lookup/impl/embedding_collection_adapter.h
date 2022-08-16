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

#ifndef EMBEDDING_COLLECTION_IMPL_H
#define EMBEDDING_COLLECTION_IMPL_H

// clang-format off
#include <cuda_fp16.h>

#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"

#include "HugeCTR/embedding/embedding_table.hpp"

#include "variable/kernels/dummy_var.h"
// clang-format on

namespace sok {

template <typename KeyType, typename DType>
class TFAdapter : public ::embedding::ILookup {
 public:
  TFAdapter();
  ~TFAdapter();

  void set(std::vector<tensorflow::core::RefCountPtr<tensorflow::Var>>& vars,
           std::vector<tensorflow::tf_shared_lock>& locks, std::vector<int>& dimensions,
           std::vector<int>& scale, cudaStream_t stream = 0);

  void lookup(const ::core::Tensor& keys, size_t num_keys, const ::core::Tensor& id_space_offset,
              size_t num_id_space_offset, const ::core::Tensor& id_space,
              ::core::TensorList& embedding_vec) override;

 private:
  int sm_count_;
  std::vector<float*> data_;
  std::vector<int> dimensions_;
  std::vector<int> id_space_to_local_index_;
  std::vector<int> scale_;
  float** d_data_;
  int* d_dimensions_;
  int* d_id_space_to_local_index_;
  int* d_scale_;
  cudaStream_t stream_;

  void free();
};

template <typename KeyType, typename DType>
class DummyVarAdapter : public ::embedding::ILookup {
 public:
  DummyVarAdapter();
  ~DummyVarAdapter() = default;

  void set(std::vector<tensorflow::core::RefCountPtr<tensorflow::DummyVar<KeyType, DType>>>& vars,
           std::vector<tensorflow::tf_shared_lock>& locks, std::vector<int>& dimensions,
           std::vector<int>& scale, cudaStream_t stream = 0);

  void lookup(const ::core::Tensor& keys, size_t num_keys, const ::core::Tensor& id_space_offset,
              size_t num_id_space_offset, const ::core::Tensor& id_space,
              ::core::TensorList& embedding_vec) override;

 private:
  int sm_count_;
  // std::vector<int> id_space_to_local_index_;
  std::vector<uint32_t> id_space_offset_;
  std::vector<int> id_space_;
  std::vector<std::shared_ptr<VariableBase<KeyType, DType>>> vars_;
  cudaStream_t stream_;
};

}  // namespace sok

#endif
