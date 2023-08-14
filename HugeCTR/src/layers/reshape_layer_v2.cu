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

#include <common.hpp>
#include <layers/reshape_layer_v2.hpp>
#include <network_buffer_channels.hpp>
#include <utils.hpp>

namespace HugeCTR {

std::vector<int64_t> reshape_layer_utils::calc_output_shape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& out_shape_with_placeholder) {
  int64_t num_elements_wo_placeholder = 1;
  int count_placehoder = 0;
  for (auto n : out_shape_with_placeholder) {
    if (n < 0) {
      count_placehoder++;
      continue;
    }
    num_elements_wo_placeholder *= n;
  }
  HCTR_CHECK_HINT(num_elements_wo_placeholder > 0, "out dimension in Reshape Layer is illegal.");
  HCTR_CHECK_HINT(count_placehoder == 1, "The placeholder in reshape layer should equal to 1.");

  int64_t num_input_elements = 1;
  for (auto& n : input_shape) {
    num_input_elements *= n;
  }
  HCTR_CHECK_HINT(num_input_elements % num_elements_wo_placeholder == 0,
                  "Illegal Reshape out dimension.");

  int64_t placeholder_dimension = num_input_elements / num_elements_wo_placeholder;
  std::vector<int64_t> out_dimensions;
  for (auto n : out_shape_with_placeholder) {
    if (n < 0) {
      out_dimensions.push_back(placeholder_dimension);
    } else {
      out_dimensions.push_back(n);
    }
  }

  return out_dimensions;
}

template <typename T>
ReshapeLayerV2<T>::ReshapeLayerV2(const core23::Tensor& input_tensor, core23::Tensor& output_tensor,
                                  const std::vector<int64_t>& out_shape_with_placeholder,
                                  const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer({input_tensor}, {}, gpu_resource) {
  try {
    std::vector<int64_t> input_tensor_shape;
    for (int i = 0; i < input_tensor.dims(); ++i) {
      input_tensor_shape.push_back(input_tensor.shape()[i]);
    }
    std::vector<int64_t> out_dimensions =
        reshape_layer_utils::calc_output_shape(input_tensor_shape, out_shape_with_placeholder);

    core23::BufferParams buf{.channel = GetBlobsBufferChannel()};
    output_tensor =
        core23::Tensor(input_tensor.my_params().shape(out_dimensions).buffer_params(buf));

    output_tensors_.push_back(output_tensor);
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void ReshapeLayerV2<T>::fprop(bool is_train) {
  core23::Tensor& input_tensor = input_tensors_[0];
  core23::Tensor& output_tensor = output_tensors_[0];

  HCTR_LIB_THROW(cudaMemcpyAsync(output_tensor.data<T>(), input_tensor.data<T>(),
                                 input_tensor.num_bytes(), cudaMemcpyDeviceToDevice,
                                 get_gpu().get_stream()));
}

template <typename T>
void ReshapeLayerV2<T>::bprop() {
  core23::Tensor& input_tensor = input_tensors_[0];
  core23::Tensor& output_tensor = output_tensors_[0];

  HCTR_LIB_THROW(cudaMemcpyAsync(input_tensor.data<T>(), output_tensor.data<T>(),
                                 output_tensor.num_bytes(), cudaMemcpyDeviceToDevice,
                                 get_gpu().get_stream()));
}
template class ReshapeLayerV2<float>;
template class ReshapeLayerV2<__half>;

}  // namespace HugeCTR
