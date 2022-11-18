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

#include <cassert>
#include <type_traits>

#include "HugeCTR/include/layer.hpp"

namespace HugeCTR {
/**
 * @brief
 * Trainable layer is the common parent of all layers with weights
 * @tparams DType the data type of inputs, outputs, and weights
 * @tparams use_FP32_weight if specified, the weight data type is in FP32, not DType
 */
template <typename DType, bool use_FP32_weight = std::is_same<DType, float>::value>
class TrainableLayer : public Layer {
  // FP32 input/output but lower precision weight don't make much sense.
  static_assert(!(std::is_same<DType, float>::value && use_FP32_weight == false));

 protected:
  // Why WeightType is protected?
  // it is convenient for a child trainable to access the weight type,
  // especially if it wants to use FP32 weights but inputs/outputs the lower precision data.
  // A typical example is when DType is __half but use_FP32_weight is true.
  // Then, the child class should define the following alias to make thier code cleaner:
  // (1) using Base = TrainableLayer<DType, true>;
  // (2) using WeightType = typename Base::WeightType;
  // If  useFP32_weight is false, the aliases are not necessary.
  using WeightType = typename std::conditional<use_FP32_weight, float, DType>::type;

 private:
  Tensors2<float> master_weights_;
  Tensors2<WeightType> weights_;
  Tensors2<WeightType> wgrads_;
  const std::shared_ptr<BufferBlock2<float>> master_weight_buff_;
  const std::shared_ptr<BufferBlock2<WeightType>> weight_buff_;
  const std::shared_ptr<BufferBlock2<WeightType>> wgrad_buff_;

 protected:
  // @brief a modifier to reserve a weight tensor at idx with the specified dims.
  // @details
  // Usage: In a child class, this->set_weight(0, dims);
  void set_weight(size_t idx, const std::vector<size_t>& dimensions) {
    HCTR_CHECK_HINT(weights_.size() == idx, "Wrong index for setting weight tensors");

    Tensor2<WeightType> tensor;
    weight_buff_->reserve(dimensions, &tensor);
    weights_.push_back(tensor);

    // master weights are used only when compute weights have lower precision
    if constexpr (!use_FP32_weight) {
      HCTR_CHECK_HINT(master_weights_.size() == idx,
                      "Wrong index for setting master weight tensors");

      Tensor2<float> tensor;
      master_weight_buff_->reserve(dimensions, &tensor);
      master_weights_.push_back(tensor);
    }
  }
  // @brief a modifier to reserve a weight tensor at idx with the specified dims.
  // @details
  // Usage: In a child class, this->set_wgrad(0, dims);
  void set_wgrad(size_t idx, const std::vector<size_t>& dimensions) {
    HCTR_CHECK_HINT(wgrads_.size() == idx, "Wrong index for setting weight gradient tensors");

    Tensor2<WeightType> tensor;
    wgrad_buff_->reserve(dimensions, &tensor);
    wgrads_.push_back(tensor);
  }
  // @brief an accessor to get a weight tensor at idx
  // @details
  // Usage: In a child class, auto weight2 = this->get_weight(2);
  auto& get_weight(size_t idx) const {
    HCTR_CHECK_HINT(idx < weights_.size(), "Wrong index for getting weight tensors");
    return weights_[idx];
  }
  // @brief an accessor to get a wgrad tensor at idx
  // @details
  // Usage: In a child class, auto wgrad2 = this->get_wgrad(2);
  auto& get_wgrad(size_t idx) const {
    HCTR_CHECK_HINT(idx < wgrads_.size(), "Wrong index for getting weight gradient tensors");
    return wgrads_[idx];
  }

 public:
  /**
   * Ctor of TrainableLayer.
   * @param master_weight_buff the buffer to reserve master weight tensors, used only if WeightType
   * is not FP32.
   * @param weight_buff the buffer to reserve weight tensors
   * @param wgrad_buff the buffer to reserve weight gradient tensors
   * @param gpu_resource the abstraction of GPU where this dense layer resides
   * @param initializer_types the list of initializer types of all weight tensors
   */
  TrainableLayer(const std::shared_ptr<BufferBlock2<float>>& master_weight_buff,
                 const std::shared_ptr<BufferBlock2<WeightType>>& weight_buff,
                 const std::shared_ptr<BufferBlock2<WeightType>>& wgrad_buff,
                 const std::shared_ptr<GPUResource>& gpu_resource,
                 std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>())
      : Layer(gpu_resource, initializer_types),
        // if WeightType is float, master weights are not used at all
        master_weight_buff_(std::is_same<WeightType, float>::value ? nullptr : master_weight_buff),
        weight_buff_(weight_buff),
        wgrad_buff_(wgrad_buff) {}
};
}  // namespace HugeCTR
