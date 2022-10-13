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
#pragma once

#include "HugeCTR/core/buffer.hpp"
#include "HugeCTR/core/registry.hpp"

namespace embedding {
using core::CoreResourceManager;
using core::DataType;
using core::Device;
using core::DeviceType;
using core::Shape;
using core::Tensor;
using core::TensorList;
using core::TensorScalarType;

class CompressOffset {
  std::shared_ptr<CoreResourceManager> core_;
  int num_compressed_offset_;
  Tensor compressed_offset_;

 public:
  CompressOffset() = default;

  CompressOffset(std::shared_ptr<CoreResourceManager> core, int num_compressed_offset);

  void compute(const Tensor &offset, int batch_size, Tensor *compressed_offset);
};

class AverageCombiner {
  std::shared_ptr<CoreResourceManager> core_;
  int num_gpus_;
  int num_local_embedding_;

 public:
  Tensor float_emb_vec_;

  AverageCombiner() = default;

  AverageCombiner(std::shared_ptr<CoreResourceManager> core, int num_gpus, int num_local_embedding,
                  const std::vector<int> &ev_size_list, int universal_batch_size);

  void compute(const Tensor &bucket_range, const Tensor &src_emb_vec,
               const Tensor &d_local_embedding_list, const Tensor &d_combiner_list,
               const Tensor &d_ev_size_offset, int batch_size, int max_ev_size);
};
}  // namespace embedding