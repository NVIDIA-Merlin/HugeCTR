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

#include <HugeCTR/core23/tensor_operations.hpp>
#include <common.hpp>
#include <layers/select_layer.hpp>
#include <network_buffer_channels.hpp>
#include <utils.hpp>

namespace HugeCTR {

namespace {

template <typename T>
__global__ void select_kernel(T* input, T* output, int64_t num_before_selected_dim,
                              int64_t num_in_selected_dim, int64_t num_after_selected_dim,
                              const int64_t* const index, int64_t num_index, bool forward) {
  int base = blockIdx.x * blockDim.x + threadIdx.x;
  int input_vector_length = num_in_selected_dim * num_after_selected_dim;
  int output_vector_length = num_index * num_after_selected_dim;
  int output_n_elem = num_before_selected_dim * output_vector_length;
  for (int idx = base; idx < output_n_elem; idx += blockDim.x * gridDim.x) {
    int batch_id = idx / output_vector_length;
    int idx_in_batch = idx % output_vector_length;
    int output_slot_id = idx_in_batch / num_after_selected_dim;
    int idx_in_slot = idx_in_batch % num_after_selected_dim;
    int input_slot_id = __ldg(&index[output_slot_id]);
    int input_idx =
        batch_id * input_vector_length + input_slot_id * num_after_selected_dim + idx_in_slot;
    if (forward)
      output[idx] = input[input_idx];
    else
      input[input_idx] = output[idx];
  }
}

}  // anonymous namespace

template <typename T>
SelectLayer<T>::SelectLayer(const core23::Tensor& input_tensor, core23::Tensor& output_tensor,
                            int dim, const std::vector<int64_t>& index,
                            const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer({input_tensor}, {}, gpu_resource),
      num_before_selected_dim_(1),
      num_in_selected_dim_(1),
      num_after_selected_dim_(1),
      h_index_(index) {
  try {
    HCTR_CHECK_HINT(!index.empty(), "Select Layer does not support empty index");
    HCTR_CHECK_HINT(dim < input_tensor.dims(), "Dim is beyond range in Select Layer");
    HCTR_CHECK_HINT(dim != 0, "Dim 0 is not allowed to do select because it's batch dim.");

    int64_t max_index = *std::max_element(index.begin(), index.end());
    HCTR_CHECK_HINT(max_index < input_tensor.size(dim),
                    "Value in index is beyond range of specified dim in Select Layer.");

    num_in_selected_dim_ = input_tensor.shape()[dim];
    for (int i = 0; i < dim; ++i) {
      num_before_selected_dim_ *= input_tensor.shape()[i];
    }
    for (int i = dim + 1; i < input_tensor.dims(); ++i) {
      num_after_selected_dim_ *= input_tensor.shape()[i];
    }

    core23::Shape out_shape = input_tensor.shape();
    out_shape[dim] = index.size();
    core23::BufferParams buf_p{.channel = GetBlobsBufferChannel()};
    output_tensor = core23::Tensor(input_tensor.my_params().shape(out_shape).buffer_params(buf_p));

    d_index_ = core23::Tensor(input_tensor.my_params()
                                  .shape({static_cast<int64_t>(index.size())})
                                  .data_type(core23::ScalarType::Int64));
    output_tensors_.push_back(output_tensor);
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void SelectLayer<T>::initialize() {
  core23::copy_async(d_index_, h_index_, get_gpu().get_stream());
}

template <typename T>
void SelectLayer<T>::fprop(bool is_train) {
  prop_common(true, is_train, get_gpu().get_stream());
}

template <typename T>
void SelectLayer<T>::bprop() {
  prop_common(false, true, get_gpu().get_stream());
}

template <typename T>
void SelectLayer<T>::prop_common(bool forward, bool is_train, cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  core23::Tensor& input_tensor = input_tensors_[0];
  core23::Tensor& output_tensor = output_tensors_[0];

  int block_size = 128;
  int n_block = get_gpu().get_sm_count() * 16;
  T* in = input_tensor.data<T>();
  T* out = output_tensor.data<T>();
  select_kernel<<<n_block, block_size, 0, stream>>>(
      in, out, num_before_selected_dim_, num_in_selected_dim_, num_after_selected_dim_,
      d_index_.data<int64_t>(), h_index_.size(), forward);
}

template class SelectLayer<float>;
template class SelectLayer<__half>;

}  // namespace HugeCTR
