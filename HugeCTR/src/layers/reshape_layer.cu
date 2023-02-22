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
#include <layers/reshape_layer.hpp>
#include <utils.hpp>

namespace HugeCTR {

namespace {

template <typename T>
__global__ void reshape_kernel(T* input, T* output, int batch_size, int n_slot, int vector_length,
                               const int* const selected, int n_active_slot, bool forward) {
  int base = blockIdx.x * blockDim.x + threadIdx.x;
  int input_vector_length = n_slot * vector_length;
  int output_vector_length = n_active_slot * vector_length;
  int output_n_elem = batch_size * output_vector_length;
  for (int idx = base; idx < output_n_elem; idx += blockDim.x * gridDim.x) {
    int batch_id = idx / output_vector_length;
    int idx_in_batch = idx % output_vector_length;
    int output_slot_id = idx_in_batch / vector_length;
    int idx_in_slot = idx_in_batch % vector_length;
    int input_slot_id = __ldg(&selected[output_slot_id]);
    int input_idx = batch_id * input_vector_length + input_slot_id * vector_length + idx_in_slot;
    if (forward)
      output[idx] = input[input_idx];
    else
      input[input_idx] = output[idx];
  }
}

}  // anonymous namespace

template <typename T>
ReshapeLayer<T>::ReshapeLayer(const core23::Tensor& input_tensor, core23::Tensor& output_tensor,
                              const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer({input_tensor}, {output_tensor}, gpu_resource),
      in_place_(true),
      batch_size_(output_tensor.size(0)),
      n_slot_(output_tensor.dims() == 2 ? 0 : output_tensor.size(1)),
      vector_length_(output_tensor.size(output_tensor.dims() - 1)),
      n_active_slot_(0) {
  try {
    auto n_in_elems = input_tensor.num_elements();
    const auto out_shape = output_tensor.shape();
    switch (out_shape.dims()) {
      case 2:
        if (vector_length_ > n_in_elems) {
          HCTR_OWN_THROW(Error_t::WrongInput, "leading_dim cannot be bigger than n_in_elems");
        }
        if (n_in_elems % vector_length_ != 0) {
          HCTR_OWN_THROW(Error_t::WrongInput, "n_in_elems % leading_dim != 0");
        }
        break;
      case 3:
        if ((vector_length_ * n_slot_) > n_in_elems) {
          HCTR_OWN_THROW(Error_t::WrongInput,
                         "leading_dim and time_step cannot be bigger than n_in_elems");
        }
        if (n_in_elems % (vector_length_ * n_slot_) != 0) {
          HCTR_OWN_THROW(Error_t::WrongInput, "n_in_elems % (vector_length_ * n_slot_) != 0");
        }
        break;
      default:
        HCTR_OWN_THROW(Error_t::WrongInput, "Output tensor dimension is not supported.");
    }
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
ReshapeLayer<T>::ReshapeLayer(const core23::Tensor& input_tensor, core23::Tensor& output_tensor,
                              std::vector<int>& selected,
                              const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer({input_tensor}, {}, gpu_resource),
      in_place_(selected.empty()),
      batch_size_(0),
      n_slot_(0),
      vector_length_(0),
      n_active_slot_(selected.size()),
      selected_(selected) {
  try {
    const auto in_shape = input_tensor.shape();
    if (in_shape.size(1) < static_cast<int64_t>(n_active_slot_)) {
      HCTR_OWN_THROW(Error_t::WrongInput, "selected is invalid");
    }

    auto in_dims_1 = selected.empty() ? in_shape.size(1) : static_cast<int64_t>(n_active_slot_);
    const core23::Shape out_shape = {in_shape.size(0), in_dims_1 * in_shape.size(2)};
    output_tensor = core23::Tensor(input_tensor.my_params().shape(out_shape));

    if (!in_place_) {
      unsigned int i = 0;
      for (; i < in_shape.dims() - 2; i++) {
        batch_size_ += in_shape.size(i);
      }
      n_slot_ = in_shape.size(i++);
      vector_length_ = in_shape.size(i);

      selected_slots_tensor_ = core23::Tensor(input_tensor.my_params()
                                                  .shape({static_cast<int64_t>(n_active_slot_)})
                                                  .data_type(core23::ScalarType::Int32));
    }
    output_tensors_.push_back(output_tensor);
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
ReshapeLayer<T>::ReshapeLayer(const Tensor2<T>& in_tensor, Tensor2<T>& out_tensor,
                              const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                              const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource),
      in_place_(true),
      batch_size_(out_tensor.get_dimensions()[0]),
      n_slot_(0),
      vector_length_(0),
      n_active_slot_(0) {
  try {
    size_t n_in_elems = in_tensor.get_num_elements();
    const std::vector<size_t>& out_dims = out_tensor.get_dimensions();
    switch (out_dims.size()) {
      case 2:
        vector_length_ = out_tensor.get_dimensions()[1];
        if (size_t(vector_length_) > n_in_elems) {
          HCTR_OWN_THROW(Error_t::WrongInput, "leading_dim cannot be bigger than n_in_elems");
        }
        if (n_in_elems % vector_length_ != 0) {
          HCTR_OWN_THROW(Error_t::WrongInput, "n_in_elems % leading_dim != 0");
        }
        break;
      case 3:
        n_slot_ = out_tensor.get_dimensions()[1];
        vector_length_ = out_tensor.get_dimensions()[2];
        if (size_t(vector_length_ * n_slot_) > n_in_elems) {
          HCTR_OWN_THROW(Error_t::WrongInput,
                         "leading_dim and time_step cannot be bigger than n_in_elems");
        }
        if (n_in_elems % (vector_length_ * n_slot_) != 0) {
          HCTR_OWN_THROW(Error_t::WrongInput, "n_in_elems % (vector_length_ * n_slot_) != 0");
        }
        break;
      default:
        HCTR_OWN_THROW(Error_t::WrongInput, "Output tensor dimension is not supported.");
    }

    in_tensors_.push_back(in_tensor);
    out_tensors_.push_back(out_tensor);

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
ReshapeLayer<T>::ReshapeLayer(const Tensor2<T>& in_tensor, Tensor2<T>& out_tensor,
                              const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                              std::vector<int>& selected,
                              const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource),
      in_place_(selected.empty()),
      batch_size_(0),
      n_slot_(0),
      vector_length_(0),
      n_active_slot_(selected.size()),
      selected_(selected) {
  try {
    const std::vector<size_t>& in_dims = in_tensor.get_dimensions();
    if (in_dims[1] < n_active_slot_) {
      HCTR_OWN_THROW(Error_t::WrongInput, "selected is invalid");
    }

    size_t in_dims_1 = selected.empty() ? in_dims[1] : n_active_slot_;
    std::vector<size_t> out_dims = {in_dims[0], in_dims_1 * in_dims[2]};
    blobs_buff->reserve(out_dims, &out_tensor);

    if (!in_place_) {
      unsigned int i = 0;
      for (; i < in_dims.size() - 2; i++) batch_size_ += in_dims[i];
      n_slot_ = in_dims[i++];
      vector_length_ = in_dims[i];

      blobs_buff->reserve({n_active_slot_}, &selected_tensor_);
    }
    in_tensors_.push_back(in_tensor);
    out_tensors_.push_back(out_tensor);

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void ReshapeLayer<T>::initialize() {
  // TODO: this block will be removed later
  if (selected_slots_tensor_.empty()) {
    if (!in_place_) {
      HCTR_LIB_THROW(cudaMemcpyAsync(selected_tensor_.get_ptr(), &selected_.front(),
                                     n_active_slot_ * sizeof(int), cudaMemcpyHostToDevice,
                                     get_gpu().get_stream()));
    }
  } else {
    if (!in_place_) {
      HCTR_LIB_THROW(cudaMemcpyAsync(selected_slots_tensor_.data(), &selected_.front(),
                                     selected_slots_tensor_.num_bytes(), cudaMemcpyHostToDevice,
                                     get_gpu().get_stream()));
    }
  }
}

template <typename T>
void ReshapeLayer<T>::fprop(bool is_train) {
  prop_common(true, is_train, get_gpu().get_stream());
}

template <typename T>
void ReshapeLayer<T>::bprop() {
  prop_common(false, true, get_gpu().get_stream());
}

template <typename T>
void ReshapeLayer<T>::prop_common(bool forward, bool is_train, cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  // TODO: this block will be removed later
  if (input_tensors_.empty()) {
    Tensor2<T>& in_tensor = get_in_tensors(is_train)[0];
    Tensor2<T>& out_tensor = out_tensors_[0];

    if (in_place_) {
      if (forward) {
        HCTR_LIB_THROW(cudaMemcpyAsync(out_tensor.get_ptr(), in_tensor.get_ptr(),
                                       in_tensor.get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                                       stream));
      } else {
        HCTR_LIB_THROW(cudaMemcpyAsync(in_tensor.get_ptr(), out_tensor.get_ptr(),
                                       out_tensor.get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                                       stream));
      }
    } else {
      int block_size = 128;
      int n_block = get_gpu().get_sm_count() * 16;
      T* in = in_tensor.get_ptr();
      T* out = out_tensor.get_ptr();
      reshape_kernel<<<n_block, block_size>>>(in, out, batch_size_, n_slot_, vector_length_,
                                              selected_tensor_.get_ptr(), n_active_slot_, forward);
    }
  } else {
    core23::Tensor& input_tensor = input_tensors_[0];
    core23::Tensor& output_tensor = output_tensors_[0];

    if (in_place_) {
      if (forward) {
        HCTR_LIB_THROW(cudaMemcpyAsync(output_tensor.data<T>(), input_tensor.data<T>(),
                                       input_tensor.num_bytes(), cudaMemcpyDeviceToDevice, stream));
      } else {
        HCTR_LIB_THROW(cudaMemcpyAsync(input_tensor.data<T>(), output_tensor.data<T>(),
                                       output_tensor.num_bytes(), cudaMemcpyDeviceToDevice,
                                       stream));
      }
    } else {
      int block_size = 128;
      int n_block = get_gpu().get_sm_count() * 16;
      T* in = input_tensor.data<T>();
      T* out = output_tensor.data<T>();
      reshape_kernel<<<n_block, block_size>>>(in, out, batch_size_, n_slot_, vector_length_,
                                              selected_slots_tensor_.data<int32_t>(),
                                              n_active_slot_, forward);
    }
  }
#ifndef NDEBUG
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <typename T>
Tensors2<T>& ReshapeLayer<T>::get_in_tensors(bool is_train) {
  return in_tensors_;
}

template class ReshapeLayer<float>;
template class ReshapeLayer<__half>;

}  // namespace HugeCTR
