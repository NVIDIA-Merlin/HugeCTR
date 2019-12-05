/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "HugeCTR/include/layers/reshape_layer.hpp"

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/tensor.hpp"

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

ReshapeLayer::ReshapeLayer(Tensor<float>& in_tensor, Tensor<float>** out_tensor, int leading_dim, int device_id)
    : Layer(device_id) {

  try {
    int o_device = -1;
    CK_CUDA_THROW_(get_set_device(get_device_id(), &o_device));

    std::vector<int> in_dims = in_tensor.get_dims();
    int im_idx = in_dims.size() - 1;
    if(leading_dim < in_dims[im_idx] || leading_dim % in_dims[im_idx] != 0) {
        CK_THROW_(Error_t::WrongInput,
            "leading_dim < in_dims[im_idx] or leading_dim % in_dims[2] != 0");
    }

    int n_in_elems = in_tensor.get_num_elements();
    if(leading_dim > n_in_elems) {
        CK_THROW_(Error_t::WrongInput,
            "leading_dim cannot be bigger than n_in_elems");
    }

    if(n_in_elems % leading_dim != 0) {
        CK_THROW_(Error_t::WrongInput,
            "n_in_elems % leading_dim != 0");
    }

    int trailing_dim = n_in_elems / leading_dim;
    std::vector<int> out_dims = {trailing_dim, leading_dim};
    *out_tensor = new Tensor<float>(out_dims, in_tensor, TensorFormat_t::HW);

    in_tensors_.push_back(std::ref(in_tensor));
    out_tensors_.push_back(std::ref(**out_tensor));

    CK_CUDA_THROW_(get_set_device(o_device));
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

}  // namespace HugeCTR
