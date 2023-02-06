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

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <mma.h>

#include <common.hpp>
#include <cpu/layers/interaction_layer_cpu.hpp>
#include <type_traits>
#include <utils.hpp>

namespace HugeCTR {

namespace {

template <uint x>
struct Log2 {
  static constexpr uint value = 1 + Log2<x / 2>::value;
};
template <>
struct Log2<1> {
  static constexpr uint value = 0;
};

struct __align__(8) half4 {
  half2 vals[2];
};

template <typename T>
void concat_cpu(size_t height, size_t in_width, size_t out_width, size_t n_ins, size_t n_emb,
                bool fprop, T *h_concat, T *h_in_mlp, T *h_in_emb) {
  for (size_t ni = 0; ni < n_ins; ni++) {
    for (size_t h = 0; h < height; h++) {
      size_t in_idx_base = (ni == 0) ? h * in_width : h * in_width * n_emb;
      for (size_t w = 0; w < in_width; w++) {
        size_t in_idx = in_idx_base + w;
        size_t out_idx = h * out_width + ni * in_width + w;
        if (fprop) {
          h_concat[out_idx] = (ni == 0) ? h_in_mlp[in_idx] : h_in_emb[(ni - 1) * in_width + in_idx];
        } else {
          if (ni == 0) {
            h_in_mlp[in_idx] = h_in_mlp[in_idx] + h_concat[out_idx];
          } else {
            h_in_emb[in_idx + (ni - 1) * in_width] = h_concat[out_idx];
          }
        }
      }
    }
  }
}

template <typename T>
void matmul_cpu(size_t height, size_t in_width, size_t n_ins, T *h_concat, T *h_mat) {
  for (size_t p = 0; p < height; p++) {
    size_t concat_stride = n_ins * in_width * p;
    size_t mat_stride = n_ins * n_ins * p;
    for (size_t m = 0; m < n_ins; m++) {
      for (size_t n = 0; n < n_ins; n++) {
        float accum = 0.0f;
        for (size_t k = 0; k < in_width; k++) {
          accum += h_concat[concat_stride + m * in_width + k] *
                   h_concat[concat_stride + n * in_width + k];
        }
        h_mat[mat_stride + m * n_ins + n] = accum;
      }
    }
  }
}

template <typename T>
void gather_concat_cpu(size_t height, size_t in_width, size_t n_ins, T *h_in_mlp, T *h_mat,
                       T *h_ref) {
  size_t out_len = in_width + (n_ins * (n_ins + 1) / 2 - n_ins) + 1;
  for (size_t p = 0; p < height; p++) {
    size_t cur_idx = 0;
    size_t out_stride = p * out_len;
    size_t mat_stride = p * n_ins * n_ins;
    for (size_t i = 0; i < in_width; i++) {
      h_ref[out_stride + cur_idx++] = h_in_mlp[p * in_width + i];
    }
    for (size_t n = 0; n < n_ins; n++) {
      for (size_t m = 0; m < n_ins; m++) {
        if (n > m) {
          h_ref[out_stride + cur_idx++] = h_mat[mat_stride + m * n_ins + n];
        }
      }
    }
  }
}

}  // anonymous namespace

template <typename T>
InteractionLayerCPU<T>::InteractionLayerCPU(
    const Tensor2<T> &in_bottom_mlp_tensor, const Tensor2<T> &in_embeddings, Tensor2<T> &out_tensor,
    const std::shared_ptr<GeneralBuffer2<HostAllocator>> &blobs_buff, bool use_mixed_precision)
    : LayerCPU(), use_mixed_precision_(use_mixed_precision) {
  try {
    auto first_in_dims = in_bottom_mlp_tensor.get_dimensions();
    auto second_in_dims = in_embeddings.get_dimensions();

    if (first_in_dims.size() != 2) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Input Bottom MLP must be a 2D tensor");
    }

    if (second_in_dims.size() != 3) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Input Embeddings must be a 3D tensor");
    }

    if (first_in_dims[0] != second_in_dims[0]) {
      HCTR_OWN_THROW(Error_t::WrongInput, "the input tensors' batch sizes must be the same");
    }

    if (first_in_dims[1] != second_in_dims[2]) {
      HCTR_OWN_THROW(Error_t::WrongInput, "the input tensors' widths must be the same");
    }

    size_t n_ins = 1 + second_in_dims[1];
    if (std::is_same<T, __half>::value == false) {
      size_t concat_dims_width = first_in_dims[1] + second_in_dims[1] * second_in_dims[2];
      std::vector<size_t> concat_dims = {first_in_dims[0], concat_dims_width};

      {
        Tensor2<T> tensor;
        blobs_buff->reserve(concat_dims, &tensor);
        internal_tensors_.push_back(tensor);
      }
      {
        std::vector<size_t> mat_dims = {first_in_dims[0], n_ins * n_ins};
        Tensor2<T> tensor;
        blobs_buff->reserve(mat_dims, &tensor);
        internal_tensors_.push_back(tensor);
      }
      {
        Tensor2<T> tensor;
        blobs_buff->reserve(concat_dims, &tensor);
        internal_tensors_.push_back(tensor);
      }
    }

    int concat_len = n_ins * (n_ins + 1) / 2 - n_ins;
    std::vector<size_t> out_dims = {first_in_dims[0], first_in_dims[1] + concat_len + 1};
    blobs_buff->reserve(out_dims, &out_tensor);

    in_tensors_.push_back(in_bottom_mlp_tensor);
    in_tensors_.push_back(in_embeddings);
    out_tensors_.push_back(out_tensor);

  } catch (const std::runtime_error &rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
InteractionLayerCPU<T>::~InteractionLayerCPU(){};

template <typename T>
void InteractionLayerCPU<T>::fprop(bool is_train) {
  T *concat = internal_tensors_[0].get_ptr();
  T *in_mlp = get_in_tensors(is_train)[0].get_ptr();
  T *in_emb = get_in_tensors(is_train)[1].get_ptr();
  T *mat = internal_tensors_[1].get_ptr();
  T *gather = out_tensors_[0].get_ptr();
  size_t h = internal_tensors_[0].get_dimensions()[0];
  size_t out_w = internal_tensors_[0].get_dimensions()[1];
  size_t in_w = get_in_tensors(is_train)[0].get_dimensions()[1];
  size_t n_emb = get_in_tensors(is_train)[1].get_dimensions()[1];
  size_t n_ins = 1 + n_emb;

  concat_cpu(h, in_w, out_w, n_ins, n_emb, true, concat, in_mlp, in_emb);
  matmul_cpu(h, in_w, n_ins, concat, mat);
  gather_concat_cpu(h, in_w, n_ins, in_mlp, mat, gather);
}

template <typename T>
void InteractionLayerCPU<T>::bprop() {}

template <>
void InteractionLayerCPU<__half>::bprop() {}

template class InteractionLayerCPU<float>;
template class InteractionLayerCPU<__half>;

}  // namespace HugeCTR
