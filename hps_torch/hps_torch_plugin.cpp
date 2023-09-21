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

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <hps/plugin/facade.hpp>

using namespace HierarchicalParameterServer;
using at::Tensor;

Tensor hps_embedding_lookup(const Tensor& input, const std::string& ps_config_file,
                            const std::string& model_name, const int64_t table_id,
                            const int64_t emb_vec_size) {
  at::DeviceGuard guard(input.device());
  Facade::instance()->init(ps_config_file.c_str(), pluginType_t::TENSORFLOW);

  AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
  AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");

  const int64_t device_id = input.device().index();
  const int64_t batch_size = input.size(0);
  const int64_t num_query = input.size(1);
  const int64_t num_elements = batch_size * num_query;

  auto output = at::zeros({batch_size, num_query, emb_vec_size}, input.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  bool i64_input_key = torch::kInt64 == input.dtype();

  if (i64_input_key) {
    Facade::instance()->forward(model_name.c_str(), table_id, device_id, num_elements, emb_vec_size,
                                input.data_ptr<int64_t>(), output.data_ptr<float>(), i64_input_key,
                                stream);
  } else {
    Facade::instance()->forward(model_name.c_str(), table_id, device_id, num_elements, emb_vec_size,
                                input.data_ptr<int32_t>(), output.data_ptr<float>(), i64_input_key,
                                stream);
  }
  return output;
}

TORCH_LIBRARY(hps_torch, m) { m.def("hps_embedding_lookup", &hps_embedding_lookup); }