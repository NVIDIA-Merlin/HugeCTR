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

#include <layers/mlp_layer.hpp>
#include <type_traits>

namespace HugeCTR {

template class MLPLayer<float>;
template class MLPLayer<__half>;

template <typename T>
MLPLayer<T>::MLPLayer(const std::shared_ptr<BufferBlock2<float>>& master_weights_buff,
                      const std::shared_ptr<BufferBlock2<T>>& weights_buff,
                      const std::shared_ptr<BufferBlock2<T>>& weights_grad_buff,
                      const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                      const Tensors2<T>& bottom_tensors_, const Tensors2<T>& top_tensors,
                      const std::vector<size_t>& num_outputs,
                      const std::shared_ptr<GPUResource>& gpu_resource,
                      const std::vector<Activation_t>& acts, const std::vector<bool>& use_bias,
                      std::vector<Initializer_t> initializer_types, bool skip_head_dgrad,
                      bool async_wgrad, bool fuse_wb, bool enable_tf32_compute)
    : TrainableLayer<T>(master_weights_buff, weights_buff, weights_grad_buff, gpu_resource,
                        initializer_types),
      bottom_tensors_(bottom_tensors_),
      top_tensors_(top_tensors),
      num_outputs_(num_outputs),
      acts_(acts),
      use_bias_(use_bias),
      skip_head_dgrad_(skip_head_dgrad),
      async_wgrad_(async_wgrad),
      fuse_wb_(fuse_wb),
      enable_tf32_compute_(enable_tf32_compute) {
  int num_layers = num_outputs.size();
  train_tensors_.resize(num_layers);
  mask_tensors_.resize(num_layers);
  output_mask_.resize(num_layers);
  dact_tensors_.resize(num_layers);
  layer_desc_.resize(num_layers);
  layer_algo_.resize(num_layers);

  const auto sync_overlap_stream_env = std::getenv("HUGECTR_MLP_SYNC_OVERLAP_STREAM");
  if (nullptr != sync_overlap_stream_env && 1 == std::atoi(sync_overlap_stream_env)) {
    inner_sync_overlap_stream_ = true;
  } else {
    inner_sync_overlap_stream_ = false;
  }

  for (int i = 0; i < num_layers; i++) {
    const auto& bottom_tensor_dim =
        i == 0 ? bottom_tensors_[0].get_dimensions() : train_tensors_[i - 1].get_dimensions();
    size_t batch_size = bottom_tensor_dim[0];
    size_t input_size = bottom_tensor_dim[1];
    size_t output_size = num_outputs[i];

    std::vector<size_t> kernel_dim = {input_size, output_size};
    std::vector<size_t> bias_dim = {1, output_size};

    this->set_weight(i * 2, kernel_dim);
    kernels_.push_back(this->get_weight(i * 2));
    this->set_weight(i * 2 + 1, bias_dim);
    biases_.push_back(this->get_weight(i * 2 + 1));
    this->set_wgrad(i * 2, kernel_dim);
    kernels_grad_.push_back(this->get_wgrad(i * 2));
    this->set_wgrad(i * 2 + 1, bias_dim);
    db_tensors_.push_back(this->get_wgrad(i * 2 + 1));

    const auto& train_in_tensor = i == 0 ? bottom_tensors_[0] : train_tensors_[i - 1];
    size_t num_output = num_outputs[i];

    if (i != num_layers - 1) {
      blobs_buff->reserve({train_in_tensor.get_dimensions()[0], num_output}, &train_tensors_[i]);
      if (acts_[i] == Activation_t::Relu) {
        blobs_buff->reserve({train_in_tensor.get_dimensions()[0], num_output}, &mask_tensors_[i]);
        blobs_buff->reserve({train_in_tensor.get_dimensions()[0], num_output}, &dact_tensors_[i]);
      }
    } else {
      train_tensors_[i] = top_tensors[0];
      if (top_tensors.size() == 1) {
        if (acts_[i] == Activation_t::Relu) {
          blobs_buff->reserve({train_in_tensor.get_dimensions()[0], num_output}, &mask_tensors_[i]);
        }
        blobs_buff->reserve({train_in_tensor.get_dimensions()[0], num_output}, &dact_tensors_[i]);
      }
    }

    output_mask_[i] = (acts_[i] == Activation_t::Relu) && (i != num_layers - 1);
  }
}

template <typename T>
void MLPLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(this->get_device_id());
  int num_layers = num_outputs_.size();
  for (int i = 0; i < num_layers; i++) {
    const T* kernel = kernels_[i].get_ptr();
    const T* bottom = i == 0 ? bottom_tensors_[0].get_ptr() : train_tensors_[i - 1].get_ptr();
    T* top_fprop = train_tensors_[i].get_ptr();

    layer_functors_.fprop(kernel, bottom, top_fprop, layer_desc_[i], layer_algo_[i],
                          this->get_gpu().get_cublaslt_handle(), this->get_gpu().get_stream());
    if (i == num_layers - 1 && acts_[i] == Activation_t::Relu) {
      T* mask_out = mask_tensors_[i].get_ptr();
      size_t len = train_tensors_[i].get_num_elements();
      HCTR_LIB_THROW(cudaMemcpyAsync(mask_out, top_fprop, len * sizeof(T), cudaMemcpyDeviceToDevice,
                                     this->get_gpu().get_stream()));
    }
  }
}

template <typename T>
void MLPLayer<T>::bprop() {
  CudaDeviceContext context(this->get_device_id());

  int num_layers = num_outputs_.size();
  for (int i = num_layers - 1; i >= 0; i--) {
    const auto& bottom_tensor_dim =
        i == 0 ? bottom_tensors_[0].get_dimensions() : train_tensors_[i - 1].get_dimensions();
    size_t batch_size = bottom_tensor_dim[0];
    size_t top_size = num_outputs_[i];

    const T* kernel = kernels_[i].get_ptr();
    const T* train_top = train_tensors_[i].get_ptr();

    // Only the last layer needs the mask of itself to get the grad.
    const T* mask_top = (i == num_layers - 1 && acts_[i] == Activation_t::Relu)
                            ? mask_tensors_[i].get_ptr()
                            : nullptr;

    T* grad_top =
        acts_[i] == Activation_t::None ? train_tensors_[i].get_ptr() : dact_tensors_[i].get_ptr();
    T* kernel_grad = kernels_grad_[i].get_ptr();
    T* bottom = i == 0 ? bottom_tensors_[0].get_ptr() : train_tensors_[i - 1].get_ptr();

    T* bottom_bprop = nullptr;
    if (i != 0) {
      bottom_bprop = acts_[i - 1] == Activation_t::None ? train_tensors_[i - 1].get_ptr()
                                                        : dact_tensors_[i - 1].get_ptr();
    } else {
      if (bottom_tensors_.size() == 1) {
        // train_in_tensor
        bottom_bprop = bottom_tensors_[0].get_ptr();
      } else {
        bottom_bprop = bottom_tensors_[1].get_ptr();
      }
    }

    layer_functors_.bprop(kernel, bottom, train_top, mask_top, batch_size * top_size, grad_top,
                          bottom_bprop, kernel_grad, layer_desc_[i], layer_algo_[i],
                          this->get_gpu().get_cublaslt_handle(), this->get_gpu().get_stream(),
                          this->get_gpu().get_comp_overlap_stream(), event_overlap_, async_wgrad_,
                          i == 0 ? skip_head_dgrad_ : false);
    if (async_wgrad_ && i == 0) {
      if (inner_sync_overlap_stream_) {
        HCTR_LIB_THROW(cudaEventRecord(event_overlap_, this->get_gpu().get_comp_overlap_stream()));
        HCTR_LIB_THROW(cudaStreamWaitEvent(this->get_gpu().get_stream(), event_overlap_));
      } else {
        this->get_gpu().set_wgrad_event_sync(this->get_gpu().get_comp_overlap_stream());
      }
    }
  }
}

template <typename T>
void MLPLayer<T>::initialize() {
  CudaDeviceContext context(this->get_device_id());

  HCTR_LIB_THROW(cudaEventCreate(&event_overlap_));
  event_overlap_created_ = true;

  int num_layers = num_outputs_.size();
  for (int i = 0; i < num_layers; i++) {
    const auto& bottom_tensor_dim =
        i == 0 ? bottom_tensors_[0].get_dimensions() : train_tensors_[i - 1].get_dimensions();
    size_t batch_size = bottom_tensor_dim[0];
    size_t input_size = bottom_tensor_dim[1];
    size_t output_size = num_outputs_[i];

    const T* bias_ptr = nullptr;
    if (use_bias_[i]) {
      bias_ptr = biases_[i].get_ptr();
    }

    T* mask_out_ptr = nullptr;
    bool output_mask = output_mask_[i];
    if (output_mask) {
      mask_out_ptr = mask_tensors_[i].get_ptr();
    }
    layer_desc_[i].set_fprop_attr(bias_ptr, acts_[i], mask_out_ptr, batch_size, input_size,
                                  output_size, enable_tf32_compute_);

    T* mask_in_ptr = nullptr;
    if (i > 0) {
      if (acts_[i - 1] == Activation_t::Relu) {
        mask_in_ptr = mask_tensors_[i - 1].get_ptr();
      }
    }
    T* dbias_bottom_ptr = nullptr;
    T* dbias_top_ptr = nullptr;
    // If there is no ReLu, then dbias should be fused with wgrad.
    // Compute the bias gradient for this layer.
    if (fuse_wb_ || i == num_layers - 1 || acts_[i] == Activation_t::None) {
      if (use_bias_[i]) {
        dbias_top_ptr = db_tensors_[i].get_ptr();
      }
    }
    // Compute the bias gradient for bottom layer. For the last layer of MLP, it should compute both
    // gradients.
    if (!fuse_wb_ && i > 0 && use_bias_[i - 1] && acts_[i - 1] != Activation_t::None) {
      dbias_bottom_ptr = db_tensors_[i - 1].get_ptr();
    }
    layer_desc_[i].set_bprop_attr(dbias_bottom_ptr, dbias_top_ptr, mask_in_ptr, batch_size,
                                  input_size, output_size, enable_tf32_compute_);
    layer_algo_[i].set_fprop_algo(layer_desc_[i], this->get_gpu().get_cublaslt_handle());
    layer_algo_[i].set_bprop_algo(layer_desc_[i], this->get_gpu().get_cublaslt_handle());
  }
}

template <typename T>
void MLPLayer<T>::search_algorithm() {
  CudaDeviceContext context(this->get_device_id());
  int num_layers = num_outputs_.size();
  for (int i = 0; i < num_layers; i++) {
    T* kernel = kernels_[i].get_ptr();
    T* bottom = i == 0 ? bottom_tensors_[0].get_ptr() : train_tensors_[i - 1].get_ptr();
    T* top = train_tensors_[i].get_ptr();

    const auto& bottom_tensor_dim =
        i == 0 ? bottom_tensors_[0].get_dimensions() : train_tensors_[i - 1].get_dimensions();
    size_t batch_size = bottom_tensor_dim[0];
    size_t input_size = bottom_tensor_dim[1];
    size_t output_size = num_outputs_[i];

    layer_functors_.search_algorithm(
        bottom, top, kernel, batch_size, input_size, output_size, layer_desc_[i], layer_algo_[i],
        this->get_gpu().get_cublaslt_handle(), this->get_gpu().get_stream());
  }
}

template <typename T>
std::unique_ptr<DataSimulator> MLPLayer<T>::get_uniform_initializer(const int index) {
  int i = index / 2;
  size_t bottom_dim =
      i == 0 ? bottom_tensors_[0].get_dimensions()[1] : train_tensors_[i - 1].get_dimensions()[1];
  size_t top_dim = train_tensors_[i].get_dimensions()[1];

  float limit = 1.0f / ((0 == index % 2 ? bottom_dim : 0) + top_dim);
  return std::make_unique<UniformDataSimulator>(-1 * limit, limit);
}

template <typename T>
std::unique_ptr<DataSimulator> MLPLayer<T>::get_xavier_uniform_initializer(const int index) {
  int i = index / 2;
  size_t bottom_dim =
      i == 0 ? bottom_tensors_[0].get_dimensions()[1] : train_tensors_[i - 1].get_dimensions()[1];
  size_t top_dim = train_tensors_[i].get_dimensions()[1];

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Uniform,
                                                    0 == index % 2 ? bottom_dim : 0, top_dim);
}

template <typename T>
std::unique_ptr<DataSimulator> MLPLayer<T>::get_xavier_norm_initializer(const int index) {
  int i = index / 2;
  size_t bottom_dim =
      i == 0 ? bottom_tensors_[0].get_dimensions()[1] : train_tensors_[i - 1].get_dimensions()[1];
  size_t top_dim = train_tensors_[i].get_dimensions()[1];

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Norm,
                                                    0 == index % 2 ? bottom_dim : 0, top_dim);
}

template <typename T>
std::unique_ptr<DataSimulator> MLPLayer<T>::get_default_initializer(const int index) {
  int i = index / 2;
  size_t bottom_dim =
      i == 0 ? bottom_tensors_[0].get_dimensions()[1] : train_tensors_[i - 1].get_dimensions()[1];
  size_t top_dim = train_tensors_[i].get_dimensions()[1];

  std::unique_ptr<DataSimulator> simu(nullptr);
  if (0 == index % 2) {
    simu.reset(new VarianceScalingSimulator(1.f, data_simu::Mode_t::Fan_avg,
                                            data_simu::Distribution_t::Norm, bottom_dim, top_dim));
  } else {
    float stddev = sqrt(1.f / top_dim);
    simu.reset(new GaussianDataSimulator(0, stddev, -2 * stddev, 2 * stddev));
  }

  return simu;
}

}  // namespace HugeCTR
