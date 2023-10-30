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
#include <core23/tensor_operations.hpp>
#include <layers/concat_3d_layer.hpp>
#include <network_buffer_channels.hpp>
#include <utils.hpp>

namespace HugeCTR {

__constant__ size_t const_vecs_size[1024];
namespace {
template <typename T>
__global__ void concat_3d_along_axis_1_kernel(bool forward, T** inputs, T* output, int batch_size,
                                              int slot_num, int output_width, int num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int threads_num = blockDim.x * gridDim.x;
  int out_size = batch_size * slot_num * output_width;

#pragma unroll
  for (int index = tid; index < out_size; index += threads_num) {
    int out_col = index % output_width;
    int sample_x_row = index / output_width;
    int out_sample = sample_x_row / slot_num;
    int out_row = sample_x_row % slot_num;

    int in_no = 0;
    int in_row = out_row;
    int accum_row = 0;

    for (int k = 0; k < num; k++) {
      if (out_row < accum_row + const_vecs_size[k]) {
        in_no = k;
        in_row -= accum_row;
        break;
      }
      accum_row += const_vecs_size[k];
    }
    T* in = inputs[in_no];
    int in_idx =
        out_sample * const_vecs_size[in_no] * output_width + in_row * output_width + out_col;
    if (forward) {
      output[index] = in[in_idx];
    } else {
      in[in_idx] = output[index];
    }
  }
}

template <typename T>
__global__ void concat_3d_along_axis_2_kernel(bool forward, T** inputs, T* output, int batch_size,
                                              int slot_num, int output_width, int num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int threads_num = blockDim.x * gridDim.x;
  int out_size = batch_size * slot_num * output_width;
#pragma unroll
  for (int index = tid; index < out_size; index += threads_num) {
    int row = index / output_width;
    int out_col = index % output_width;

    int in_no = 0;
    int in_col = out_col;
    int accum_width = 0;
    for (int k = 0; k < num; k++) {
      if (out_col < accum_width + const_vecs_size[k]) {
        in_no = k;
        in_col -= accum_width;
        break;
      }
      accum_width += const_vecs_size[k];
    }
    T* in = inputs[in_no];
    int in_idx = row * const_vecs_size[in_no] + in_col;
    if (forward) {
      output[index] = in[in_idx];
    } else {
      in[in_idx] = output[index];
    }
  }
}

}  // namespace

template <typename T>
Concat3DLayer<T>::Concat3DLayer(std::vector<core23::Tensor>& input_tensors,
                                core23::Tensor& output_tensor, int axis,
                                const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(input_tensors, {}, gpu_resource), axis_(axis), num_(input_tensors.size()) {
  try {
    if (input_tensors.empty()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Empty input tensors");
    }
    assert(axis < 3);

    for (size_t i = 0; i < num_; i++) {
      auto cur_tensor_shape = input_tensors[i].shape();
      if (i != 0) {
        auto first_tensor_shape = input_tensors[0].shape();
        if (cur_tensor_shape[0] != first_tensor_shape[0]) {
          HCTR_OWN_THROW(Error_t::WrongInput,
                         "All the input tensors must have the same shape in dimension 0");
        }
        if (axis == 1) {
          if (cur_tensor_shape[2] != first_tensor_shape[2]) {
            HCTR_OWN_THROW(
                Error_t::WrongInput,
                "When concatenating along axis 1, all the input tensors must have the same "
                "shape in dimension 2");
          }
        }
        if (axis == 2) {
          if (cur_tensor_shape[1] != first_tensor_shape[1]) {
            HCTR_OWN_THROW(
                Error_t::WrongInput,
                "When concatenating along axis 2, all the input tensors must have the same "
                "shape in dimension 1");
          }
        }
      }
      if (cur_tensor_shape.dims() != 3) {
        HCTR_OWN_THROW(Error_t::WrongInput, "Only 3D tensors can be concatenated");
      }
      if (i == 0) {
        batch_size_ = cur_tensor_shape[0];
        if (axis == 1) {
          new_width_ = cur_tensor_shape[2];
        }
        if (axis == 2) {
          new_slot_num_ = cur_tensor_shape[1];
        }
      }
      if (axis == 1) {
        new_slot_num_ += cur_tensor_shape[1];
        h_vecs_size_.push_back(cur_tensor_shape[1]);
      }
      if (axis == 2) {
        new_width_ += cur_tensor_shape[2];
        h_vecs_size_.push_back(cur_tensor_shape[2]);
      }
    }
    core23::Shape out_shape({static_cast<int64_t>(batch_size_), static_cast<int64_t>(new_slot_num_),
                             static_cast<int64_t>(new_width_)});
    core23::BufferParams buf_p{.channel = GetBlobsBufferChannel()};
    output_tensor =
        core23::Tensor(input_tensors[0].my_params().shape(out_shape).buffer_params(buf_p));

    output_tensors_.push_back(output_tensor);

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void Concat3DLayer<T>::initialize() {
  core23::Tensor h_ptrs(core23::TensorParams()
                            .device(core23::DeviceType::CPU)
                            .shape({static_cast<int64_t>(input_tensors_.size())})
                            .data_type(core23::ScalarType::Pointer));
  for (int64_t i = 0; i < h_ptrs.num_elements(); i++) {
    h_ptrs.data<T*>()[i] = input_tensors_[i].data<T>();
  }
  d_ptrs_ = core23::Tensor(h_ptrs.my_params().device(output_tensors_[0].device()));
  core23::copy_async(d_ptrs_, h_ptrs, get_gpu().get_stream());
  HCTR_LIB_THROW(cudaMemcpyToSymbolAsync(const_vecs_size, h_vecs_size_.data(),
                                         sizeof(size_t) * num_, 0, cudaMemcpyHostToDevice,
                                         get_gpu().get_stream()));
}

template <typename T>
void Concat3DLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  dim3 block_size(16, 16, 1);
  size_t n_sms = get_gpu().get_sm_count();
  dim3 grid_size(n_sms * 8, 1, 1);
  int axis = axis_;
  T** d_ptrs = d_ptrs_.data<T*>();
  T* output = output_tensors_[0].data<T>();
  if (axis == 1) {
    concat_3d_along_axis_1_kernel<<<grid_size, block_size, 0, get_gpu().get_stream()>>>(
        true, d_ptrs, output, batch_size_, new_slot_num_, new_width_, num_);
  }
  if (axis == 2) {
    concat_3d_along_axis_2_kernel<<<grid_size, block_size, 0, get_gpu().get_stream()>>>(
        true, d_ptrs, output, batch_size_, new_slot_num_, new_width_, num_);
  }
}

template <typename T>
void Concat3DLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());
  int axis = axis_;

  dim3 block_size(256, 1, 1);
  size_t n_sms = get_gpu().get_sm_count();
  dim3 grid_size(n_sms * 8, 1, 1);
  T** d_ptrs = d_ptrs_.data<T*>();
  T* output = output_tensors_[0].data<T>();
  if (axis == 1) {
    concat_3d_along_axis_1_kernel<<<grid_size, block_size, 0, get_gpu().get_stream()>>>(
        false, d_ptrs, output, batch_size_, new_slot_num_, new_width_, num_);
  }
  if (axis == 2) {
    concat_3d_along_axis_2_kernel<<<grid_size, block_size, 0, get_gpu().get_stream()>>>(
        false, d_ptrs, output, batch_size_, new_slot_num_, new_width_, num_);
  }
}

template class Concat3DLayer<float>;
template class Concat3DLayer<__half>;

}  // namespace HugeCTR
