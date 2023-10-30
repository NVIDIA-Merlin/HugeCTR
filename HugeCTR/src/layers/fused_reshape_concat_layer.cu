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
#include <functional>
#include <layers/fused_reshape_concat_layer.hpp>
#include <network_buffer_channels.hpp>
#include <utils.cuh>
#include <utils.hpp>
namespace HugeCTR {

namespace {

#define BLOCK_DIM_SIZE 32
template <typename T>
__global__ void fused_reshape_concat_kernel(bool forward, T** inputs, T* output_item, T* output_ad,
                                            int batch_size, int slot_num, size_t* vecs_size,
                                            int output_width, int num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int threads_num = blockDim.x * gridDim.x;
  int total_size = batch_size * slot_num * output_width;

  for (int index = tid; index < total_size; index += threads_num) {
    int row = index / output_width;
    int out_col = index % output_width;

    int in_no = 0;
    int in_col = out_col;
    int accum_width = 0;
    for (int k = 0; k < num; k++) {
      if (out_col < accum_width + vecs_size[k]) {
        in_no = k;
        in_col -= accum_width;
        break;
      }
      accum_width += vecs_size[k];
    }
    T* in = inputs[in_no];
    int in_idx = row * vecs_size[in_no] + in_col;
    int out_row = ((row + 1) % slot_num == 0) ? (row / slot_num) : (row - (row / slot_num));
    int out_idx = out_row * output_width + out_col;
    T* output = (row + 1) % slot_num == 0 ? output_ad : output_item;
    if (forward) {
      output[out_idx] = in[in_idx];
    } else {
      in[in_idx] = output[out_idx];
    }
  }
}

}  // end of namespace

template <typename T>
FusedReshapeConcatLayer<T>::FusedReshapeConcatLayer(
    const std::vector<core23::Tensor>& in_tensors, std::vector<core23::Tensor>& out_tensors,
    const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  try {
    if (in_tensors.empty()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Empty input tensors");
    }
    core23::BufferParams blobs_buffer_params = {};
    blobs_buffer_params.channel = GetBlobsBufferChannel();
    num_ = in_tensors.size();
    for (int64_t i = 0; i < num_; i++) {
      auto cur_in_dims = in_tensors[i].shape();
      if (i != 0) {
        auto first_in_dims = in_tensors[0].shape();
        if (cur_in_dims[0] != first_in_dims[0]) {
          HCTR_OWN_THROW(Error_t::WrongInput,
                         "All the input tensors must have the same batch_size");
        }
        if (cur_in_dims[1] != first_in_dims[1]) {
          HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must have the same slot_num");
        }
      }
      if (cur_in_dims.dims() != 3) {
        HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must be 3D");
      }
      if (i == 0) {
        batch_size_ = cur_in_dims[0];
        slot_num_ = cur_in_dims[1];
      }
      new_width_ += cur_in_dims[2];
      h_vecs_size_.push_back(static_cast<uint64_t>(cur_in_dims[2]));
    }

    {
      core23::Shape out_dims_item = {batch_size_ * (slot_num_ - 1), new_width_};

      core23::Tensor tensor_item(in_tensors[0]
                                     .my_params()
                                     .data_type(core23::ToScalarType<T>::value)
                                     .shape(out_dims_item)
                                     .buffer_params(blobs_buffer_params));
      out_tensors.push_back(tensor_item);

      core23::Shape out_dims_ad = {batch_size_, new_width_};
      core23::Tensor tensor_ad(in_tensors[0]
                                   .my_params()
                                   .data_type(core23::ToScalarType<T>::value)
                                   .shape(out_dims_ad)
                                   .buffer_params(blobs_buffer_params));
      out_tensors.push_back(tensor_ad);
    }

    for (const auto& in_tensor : in_tensors) {
      in_tensors_.push_back(in_tensor);
    }
    d_inputs_ = core23::Tensor(in_tensors[0]
                                   .my_params()
                                   .data_type(core23::ToScalarType<void*>::value)
                                   .shape({num_})
                                   .buffer_params(blobs_buffer_params));
    vecs_size_ = core23::Tensor(in_tensors[0]
                                    .my_params()
                                    .data_type(core23::ToScalarType<uint64_t>::value)
                                    .shape({num_})
                                    .buffer_params(blobs_buffer_params));

    for (auto& out_tensor : out_tensors) {
      out_tensors_.push_back(out_tensor);
    }

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void FusedReshapeConcatLayer<T>::initialize() {
  h_inputs_ = core23::Tensor(core23::TensorParams()
                                 .data_type(core23::ToScalarType<void*>::value)
                                 .shape({num_})
                                 .device(core23::Device(core23::DeviceType::CPU)));

  for (int64_t i = 0; i < num_; i++) {
    // data address
    uint64_t* to_write = h_inputs_.data<uint64_t>() + i;
    *to_write = reinterpret_cast<uint64_t>(in_tensors_[i].data());
  }
  HCTR_LIB_THROW(cudaMemcpyAsync((void*)vecs_size_.data(), (void*)h_vecs_size_.data(),
                                 num_ * sizeof(int64_t), cudaMemcpyHostToDevice,
                                 get_gpu().get_stream()));

  HCTR_LIB_THROW(cudaMemcpyAsync((void*)d_inputs_.data(), (void*)h_inputs_.data(),
                                 num_ * sizeof(T*), cudaMemcpyHostToDevice,
                                 get_gpu().get_stream()));
}

template <typename T>
void FusedReshapeConcatLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  std::vector<core23::Tensor>& out_tensors = out_tensors_;
  T* output_item = out_tensors[0].data<T>();
  T* output_ad = out_tensors[1].data<T>();
  dim3 block_size(256, 1, 1);
  size_t n_sms = get_gpu().get_sm_count();
  dim3 grid_size(n_sms * 8, 1, 1);
  fused_reshape_concat_kernel<<<grid_size, block_size, 0, get_gpu().get_stream()>>>(
      true, d_inputs_.data<T*>(), output_item, output_ad, batch_size_, slot_num_,
      vecs_size_.data<size_t>(), new_width_, num_);
}

template <typename T>
void FusedReshapeConcatLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());
  std::vector<core23::Tensor>& out_tensors = out_tensors_;
  T* output_item = out_tensors[0].data<T>();
  T* output_ad = out_tensors[1].data<T>();
  dim3 block_size(256, 1, 1);
  size_t n_sms = get_gpu().get_sm_count();
  dim3 grid_size(n_sms * 8, 1, 1);
  fused_reshape_concat_kernel<<<grid_size, block_size, 0, get_gpu().get_stream()>>>(
      false, d_inputs_.data<T*>(), output_item, output_ad, batch_size_, slot_num_,
      vecs_size_.data<size_t>(), new_width_, num_);
}

template class FusedReshapeConcatLayer<float>;

}  // namespace HugeCTR
