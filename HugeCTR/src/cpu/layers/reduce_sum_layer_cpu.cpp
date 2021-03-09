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

#include <algorithm>
#include <functional>
#include <utils.hpp>

#include <cpu/layers/reduce_sum_layer_cpu.hpp>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

template <typename T>
void reduce_sum_cpu(const T* input, T* output, std::vector<size_t> dims, int axis) {
  if (axis == 0) {
    if (dims.size() == 1) {
      for (size_t i = 0; i < dims[0]; i++) {
        output[0] = input[i];
      }
    } else if (dims.size() == 2) {
      for (size_t k = 0; k < dims[1]; k++) {
        output[k] = 0.0f;
        for (size_t i = 0; i < dims[0]; i++) {
          output[k] = output[k] + input[i * dims[1] + k];
        }
      }
    } else if (dims.size() == 3) {
      for (size_t j = 0; j < dims[1]; j++) {
        for (size_t k = 0; k < dims[2]; k++) {
          output[j * dims[2] + k] = 0.0f;
          for (size_t i = 0; i < dims[0]; i++) {
            output[j * dims[2] + k] =
                output[j * dims[2] + k] + input[i * dims[1] * dims[2] + j * dims[2] + k];
          }
        }
      }
    }
  } else if (axis == 1) {
    if (dims.size() == 2) {
      for (size_t i = 0; i < dims[0]; i++) {
        output[i] = 0.0f;
        for (size_t j = 0; j < dims[1]; j++) {
          output[i] = output[i] + input[i * dims[1] + j];
        }
      }
    } else if (dims.size() == 3) {
      for (size_t i = 0; i < dims[0]; i++) {
        for (size_t k = 0; k < dims[2]; k++) {
          output[i * dims[2] + k] = 0.0f;
          for (size_t j = 0; j < dims[1]; j++) {
            output[i * dims[2] + k] =
                output[i * dims[2] + k] + input[i * dims[1] * dims[2] + j * dims[2] + k];
          }
        }
      }
    }
  } else if (axis == 2) {
    for (size_t i = 0; i < dims[0]; i++) {
      for (size_t j = 0; j < dims[1]; j++) {
        output[i * dims[1] + j] = 0.0f;
        for (size_t k = 0; k < dims[2]; k++) {
          output[i * dims[1] + j] =
              output[i * dims[1] + j] + input[i * dims[1] * dims[2] + j * dims[2] + k];
        }
      }
    }
  }
}

template <typename T>
void reduce_sum_dgrad_cpu(const T* top_grad, T* dgrad, std::vector<size_t> dims, int axis) {
  if (axis == 0) {
    if (dims.size() == 2) {
      for (size_t j = 0; j < dims[1]; j++) {
        for (size_t i = 0; i < dims[0]; i++) {
          dgrad[i * dims[1] + j] = top_grad[j];
        }
      }
    } else if (dims.size() == 3) {
      for (size_t j = 0; j < dims[1]; j++) {
        for (size_t k = 0; k < dims[2]; k++) {
          for (size_t i = 0; i < dims[0]; i++) {
            dgrad[i * dims[1] * dims[2] + j * dims[2] + k] = top_grad[j * dims[2] + k];
          }
        }
      }
    }
  } else if (axis == 1) {
    if (dims.size() == 2) {
      for (size_t i = 0; i < dims[0]; i++) {
        for (size_t j = 0; j < dims[1]; j++) {
          dgrad[i * dims[1] + j] = top_grad[i];
        }
      }
    } else if (dims.size() == 3) {
      for (size_t i = 0; i < dims[0]; i++) {
        for (size_t k = 0; k < dims[2]; k++) {
          for (size_t j = 0; j < dims[1]; j++) {
            dgrad[i * dims[1] * dims[2] + j * dims[2] + k] = top_grad[i * dims[2] + k];
          }
        }
      }
    }
  } else if (axis == 2) {
    for (size_t i = 0; i < dims[0]; i++) {
      for (size_t j = 0; j < dims[1]; j++) {
        for (size_t k = 0; k < dims[2]; k++) {
          dgrad[i * dims[1] * dims[2] + j * dims[2] + k] = top_grad[i * dims[1] + j];
        }
      }
    }
  }
}

}  // end of namespace

template <typename T>
ReduceSumLayerCPU<T>::ReduceSumLayerCPU(const Tensor2<T>& in_tensor, Tensor2<T>& out_tensor,
                                  const std::shared_ptr<GeneralBuffer2<HostAllocator>>& blobs_buff,
                                  int axis)
    : LayerCPU(), axis_(axis) {
  try {
    // error input checking
    const auto& in_dims = in_tensor.get_dimensions();
    for (auto i : in_dims) {
      if (i == 0) {
        CK_THROW_(Error_t::WrongInput, "The input dims can not be 0");
      }
    }
    if (axis >= (int)(in_dims.size()) || axis < 0) {
      CK_THROW_(Error_t::WrongInput, "The axis is overflow");
    }

    std::vector<size_t> out_dims(in_dims.size());
    for (int i = 0; i < (int)(in_dims.size()); i++) {
      if (i == axis) {
        out_dims[i] = 1;
      } else {
        out_dims[i] = in_dims[i];
      }
    }

    blobs_buff->reserve(out_dims, &out_tensor);
    out_tensors_.push_back(out_tensor);
    in_tensors_.push_back(in_tensor);

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void ReduceSumLayerCPU<T>::fprop(bool is_train) {

  T* input = in_tensors_[0].get_ptr();
  T* output = out_tensors_[0].get_ptr();
  auto in_dims = in_tensors_[0].get_dimensions();
  auto out_dims = out_tensors_[0].get_dimensions();
  std::vector<size_t> dims;
  for (auto dim : in_dims) {
    dims.push_back(dim);
  }
  reduce_sum_cpu(input, output, dims, axis_);
}

template <typename T>
void ReduceSumLayerCPU<T>::bprop() {}

template class ReduceSumLayerCPU<float>;
template class ReduceSumLayerCPU<__half>;

}  // namespace HugeCTR
