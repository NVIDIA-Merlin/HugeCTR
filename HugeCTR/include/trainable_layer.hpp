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

#pragma once

#include "HugeCTR/include/layer.hpp"

namespace HugeCTR {
/**
 * @brief
 * Definition of a trainable layer class.
 */
template <typename T>
class TrainableLayer : public Layer {
 private:
  Tensors2<float> master_weights_;

 protected:
  /*
   * stores the weight tensors of this layer.
   */
  Tensors2<T> weights_;

  Tensors2<T> wgrads_;

  const std::shared_ptr<BufferBlock2<float>> master_weight_buff_;
  const std::shared_ptr<BufferBlock2<T>> weight_buff_;
  const std::shared_ptr<BufferBlock2<T>> wgrad_buff_;

  void set_weight(size_t idx, const std::vector<size_t>& dimensions);
  void set_wgrad(size_t idx, const std::vector<size_t>& dimensions);
  Tensor2<T>& get_weight(size_t idx);
  Tensor2<T>& get_wgrad(size_t idx);

 public:
  TrainableLayer(const std::shared_ptr<BufferBlock2<float>>& master_weight_buff,
                 const std::shared_ptr<BufferBlock2<T>>& weight_buff,
                 const std::shared_ptr<BufferBlock2<T>>& wgrad_buff,
                 const std::shared_ptr<GPUResource>& gpu_resource,
                 std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>())
      : Layer(gpu_resource, initializer_types),
        master_weight_buff_(master_weight_buff),
        weight_buff_(weight_buff),
        wgrad_buff_(wgrad_buff) {}
};
}  // namespace HugeCTR
