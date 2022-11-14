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

#include <trainable_layer.hpp>

namespace HugeCTR {

template <typename T>
void TrainableLayer<T>::set_weight(size_t idx, const std::vector<size_t>& dimensions) {
  HCTR_CHECK_HINT(weights_.size() == idx, "Wrong index for setting weight tensors");
  {
    Tensor2<T> tensor;
    weight_buff_->reserve(dimensions, &tensor);
    weights_.push_back(tensor);
  }
  if (TensorScalarTypeFunc<T>::get_type() == TensorScalarType::Float16) {
    HCTR_CHECK_HINT(master_weights_.size() == idx, "Wrong index for setting master weight tensors");
    {
      Tensor2<float> tensor;
      master_weight_buff_->reserve(dimensions, &tensor);
      master_weights_.push_back(tensor);
    }
  }
}

template <typename T>
void TrainableLayer<T>::set_wgrad(size_t idx, const std::vector<size_t>& dimensions) {
  HCTR_CHECK_HINT(wgrads_.size() == idx, "Wrong index for setting weight gradient tensors");
  {
    Tensor2<T> tensor;
    wgrad_buff_->reserve(dimensions, &tensor);
    wgrads_.push_back(tensor);
  }
}

template <typename T>
Tensor2<T>& TrainableLayer<T>::get_weight(size_t idx) {
  HCTR_CHECK_HINT(idx < weights_.size(), "Wrong index for getting weight tensors");
  return weights_[idx];
}

template <typename T>
Tensor2<T>& TrainableLayer<T>::get_wgrad(size_t idx) {
  HCTR_CHECK_HINT(idx < wgrads_.size(), "Wrong index for getting weight gradient tensors");
  return wgrads_[idx];
}

template class TrainableLayer<float>;
template class TrainableLayer<__half>;

}  // namespace HugeCTR
