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

#include <cpu/layers/fm_order2_layer_cpu.hpp>
#include <utils.hpp>

namespace HugeCTR {

namespace {

inline float trunc_half(float a) { return __half2float(__float2half(a)); }

void fm_order2_fprop_cpu(const float* in, float* out, int batch_size, int slot_num,
                         int emb_vec_size) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < emb_vec_size; j++) {
      float sum = 0.0f;
      float square_sum = 0.0f;
      int offset = i * slot_num * emb_vec_size + j;
      for (int k = 0; k < slot_num; k++) {
        int index = offset + k * emb_vec_size;
        float input = in[index];
        sum += input;
        square_sum += input * input;
      }
      float sum_square = sum * sum;
      out[i * emb_vec_size + j] = 0.5f * (sum_square - square_sum);
    }
  }
}

void fm_order2_fprop_cpu(const __half* in, __half* out, int batch_size, int slot_num,
                         int emb_vec_size) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < emb_vec_size; j++) {
      float sum = 0.0f;
      float square_sum = 0.0f;
      int offset = i * slot_num * emb_vec_size + j;
      for (int k = 0; k < slot_num; k++) {
        int index = offset + k * emb_vec_size;
        float input = __half2float(in[index]);
        sum = trunc_half(sum + input);
        square_sum = trunc_half(square_sum + input * input);
      }
      float sum_square = trunc_half(sum * sum);
      out[i * emb_vec_size + j] = __float2half(0.5f * (sum_square - square_sum));
    }
  }
}

void fm_order2_bprop_cpu(const float* in, const float* top_grad, float* dgrad, int batch_size,
                         int slot_num, int emb_vec_size) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < emb_vec_size; j++) {
      float sum = 0.0f;
      int offset = i * slot_num * emb_vec_size + j;
      for (int k = 0; k < slot_num; k++) {
        int index = offset + k * emb_vec_size;
        sum += in[index];
      }
      for (int k = 0; k < slot_num; k++) {
        int index = offset + k * emb_vec_size;
        dgrad[index] = top_grad[i * emb_vec_size + j] * (sum - in[index]);
      }
    }
  }
}

void fm_order2_bprop_cpu(const __half* in, const __half* top_grad, __half* dgrad, int batch_size,
                         int slot_num, int emb_vec_size) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < emb_vec_size; j++) {
      float sum = 0.0f;
      int offset = i * slot_num * emb_vec_size + j;
      for (int k = 0; k < slot_num; k++) {
        int index = offset + k * emb_vec_size;
        sum = trunc_half(sum + __half2float(in[index]));
      }
      for (int k = 0; k < slot_num; k++) {
        int index = offset + k * emb_vec_size;
        dgrad[index] =
            __float2half(__half2float(top_grad[i * emb_vec_size + j]) * (sum - in[index]));
      }
    }
  }
}

}  // end of namespace

template <typename T>
FmOrder2LayerCPU<T>::FmOrder2LayerCPU(const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor)
    : LayerCPU() {
  try {
    const auto& in_dims = in_tensor.get_dimensions();
    if (in_dims.size() != 2) {
      HCTR_OWN_THROW(Error_t::WrongInput, "only 2D tensors can be used as input for FmOrder2Layer");
    }
    const auto& out_dims = out_tensor.get_dimensions();
    if (out_dims.size() != 2) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "only 2D tensors can be used as output for FmOrder2Layer");
    }
    if ((in_dims[1] % out_dims[1]) != 0) {
      HCTR_OWN_THROW(Error_t::WrongInput, "(in_dims[1] % out_dims[1]) != 0");
    }

    batch_size_ = in_dims[0];
    slot_num_ = in_dims[1] / out_dims[1];
    embedding_vec_size_ = out_dims[1];

    in_tensors_.push_back(in_tensor);
    out_tensors_.push_back(out_tensor);

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void FmOrder2LayerCPU<T>::fprop(bool is_train) {
  const T* in = in_tensors_[0].get_ptr();
  T* out = out_tensors_[0].get_ptr();
  fm_order2_fprop_cpu(in, out, batch_size_, slot_num_, embedding_vec_size_);
}

template <typename T>
void FmOrder2LayerCPU<T>::bprop() {
  T* in = in_tensors_[0].get_ptr();
  const T* out = out_tensors_[0].get_ptr();
  fm_order2_bprop_cpu(in, out, in, batch_size_, slot_num_, embedding_vec_size_);
}

template class FmOrder2LayerCPU<float>;
template class FmOrder2LayerCPU<__half>;

}  // end of namespace HugeCTR
