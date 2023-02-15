/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <regularizer_factory.hpp>
#include <regularizers/l1_regularizer.hpp>
#include <regularizers/l2_regularizer.hpp>
#include <regularizers/no_regularizer.hpp>

namespace HugeCTR {

template <typename T>
std::shared_ptr<Regularizer<T>> create_regularizer(
    bool use_regularizer, Regularizer_t regularizer_type, float lambda,
    std::vector<core23::Tensor> weight_tensors, std::vector<core23::Tensor> wgrad_tensors,
    const int batch_size, const std::shared_ptr<GPUResource>& gpu_resource) {
  std::shared_ptr<Regularizer<T>> reg(
      new NoRegularizer<T>(weight_tensors, wgrad_tensors, batch_size, gpu_resource));
  if (use_regularizer) {
    switch (regularizer_type) {
      case Regularizer_t::L1: {
        reg.reset(
            new L1Regularizer<T>(weight_tensors, wgrad_tensors, batch_size, lambda, gpu_resource));
        break;
      }
      case Regularizer_t::L2: {
        reg.reset(
            new L2Regularizer<T>(weight_tensors, wgrad_tensors, batch_size, lambda, gpu_resource));
        break;
      }
      default: {
        assert(!"Error: no such regularizer!");
      }
    }
  }
  return reg;
}

template std::shared_ptr<Regularizer<float>> create_regularizer<float>(
    bool use_regularizer, Regularizer_t regularizer_type, float lambda,
    std::vector<core23::Tensor> weight_tensors, std::vector<core23::Tensor> wgrad_tensors,
    const int batch_size, const std::shared_ptr<GPUResource>& gpu_resource);

template std::shared_ptr<Regularizer<__half>> create_regularizer<__half>(
    bool use_regularizer, Regularizer_t regularizer_type, float lambda,
    std::vector<core23::Tensor> weight_tensors, std::vector<core23::Tensor> wgrad_tensors,
    const int batch_size, const std::shared_ptr<GPUResource>& gpu_resource);

}  // namespace HugeCTR
