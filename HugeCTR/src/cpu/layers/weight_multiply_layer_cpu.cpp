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

#include <algorithm>
#include <cpu/layers/weight_multiply_layer_cpu.hpp>
#include <functional>
#include <utils.hpp>

namespace HugeCTR {

namespace {

template <typename T>
void weight_multiply_cpu(const T* input, const T* weight, T* output, int batch_size, int slot_num,
                         int embedding_vec_size) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < slot_num; j++) {
      for (int k = 0; k < embedding_vec_size; k++) {
        output[i * slot_num * embedding_vec_size + j * embedding_vec_size + k] =
            input[i * slot_num + j] * weight[j * embedding_vec_size + k];
      }
    }
  }
}

template <typename T>
void weight_multiply_wgrad_cpu(const T* top_grad, const T* input, T* wgrad, int batch_size,
                               int slot_num, int embedding_vec_size) {
  int len_w = slot_num * embedding_vec_size;
  for (int i = 0; i < len_w; i++) {
    double tmp = 0.0;
    for (int j = 0; j < batch_size; j++) {
      tmp += (double)input[j * slot_num + i / embedding_vec_size] * (double)top_grad[j * len_w + i];
    }
    wgrad[i] = (T)tmp;
  }
}

template <typename T>
void weight_multiply_dgrad_cpu(const T* top_grad, const T* weight, T* dgrad, int batch_size,
                               int slot_num, int embedding_vec_size) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < slot_num; j++) {
      T tmp = T(0.0);
      for (int k = 0; k < embedding_vec_size; k++) {
        tmp = tmp + T(top_grad[i * slot_num * embedding_vec_size + j * embedding_vec_size + k] *
                      weight[j * embedding_vec_size + k]);
      }
      dgrad[i * slot_num + j] = tmp;
    }
  }
}

}  // end of namespace

template <typename T>
WeightMultiplyLayerCPU<T>::WeightMultiplyLayerCPU(
    const std::shared_ptr<BufferBlock2<T>>& weight_buff,
    const std::shared_ptr<BufferBlock2<T>>& wgrad_buff,
    const std::shared_ptr<GeneralBuffer2<HostAllocator>>& blob_buff, const Tensor2<T>& in_tensor,
    Tensor2<T>& out_tensor, const std::vector<size_t>& weight_dims)
    : LayerCPU() {
  try {
    const auto& in_dims = in_tensor.get_dimensions();
    if (in_dims.size() != 2) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Only 2D tensors can be multiplied");
    }
    if (weight_dims.size() != 2) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Only 2D weights is allowed for weight_multiply layer");
    }
    if (weight_dims[0] != in_dims[1]) {
      HCTR_OWN_THROW(Error_t::WrongInput, "weight_dims[0] must be equal to in_dims[1]");
    }

    batch_size_ = in_dims[0];
    slot_num_ = weight_dims[0];
    embedding_vec_size_ = weight_dims[1];

    std::vector<size_t> out_dims{batch_size_, slot_num_ * embedding_vec_size_};
    blob_buff->reserve(out_dims, &out_tensor);
    in_tensors_.push_back(in_tensor);
    out_tensors_.push_back(out_tensor);

    {
      Tensor2<T> tensor;
      weight_buff->reserve(weight_dims, &tensor);
      weights_.push_back(tensor);
    }
    {
      Tensor2<T> tensor;
      wgrad_buff->reserve(weight_dims, &tensor);
      wgrad_.push_back(tensor);
    }

    blob_buff->reserve(out_dims, &wgrad_tmp_trans_);

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void WeightMultiplyLayerCPU<T>::fprop(bool is_train) {
  T* input = in_tensors_[0].get_ptr();
  T* weight = weights_[0].get_ptr();
  T* output = out_tensors_[0].get_ptr();
  weight_multiply_cpu(input, weight, output, batch_size_, slot_num_, embedding_vec_size_);
}

template <typename T>
void WeightMultiplyLayerCPU<T>::bprop() {}

template class WeightMultiplyLayerCPU<float>;
template class WeightMultiplyLayerCPU<__half>;

}  // namespace HugeCTR
