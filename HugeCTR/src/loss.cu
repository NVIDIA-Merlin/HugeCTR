/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <loss.hpp>
#include <utils.cuh>
#include <vector>

namespace HugeCTR {

namespace {

template <typename T>
__forceinline__ __device__ void atomic_global_sum_div(T val, T *acc, float div) {
  val = warpReduceSum(val);
  if (threadIdx.x % warpSize == 0) {
    atomicAdd(acc, (T)(val / div));
  }
  return;
}

}  // namespace

template <typename T>
Loss<T>::Loss(const Tensor2<float> &label_tensor, const Tensor2<T> &input_tensor,
              const Tensor2<float> &loss_tensor, const std::shared_ptr<Regularizer<T>> &regularizer,
              const std::shared_ptr<GPUResource> &gpu_resource, int total_gpu_count, float scaler,
              bool gen_loss_summary)
    : regularizer_(regularizer),
      gpu_resource_(gpu_resource),
      total_gpu_count_(total_gpu_count),
      scaler_(scaler),
      gen_loss_summary_(gen_loss_summary) {
  label_tensors_.push_back(label_tensor);
  input_tensors_.push_back(input_tensor);
  loss_tensors_.push_back(loss_tensor);

  if (regularizer_ == nullptr) {
    HCTR_OWN_THROW(
        Error_t::WrongInput,
        "There is no regularizer specified. If you intend not to use any regularizer, pass a "
        "NoRegularizer object.");
  }
}

template <typename T>
void Loss<T>::compute_and_init(bool is_train) {
  Tensor2<T> &input_tensor = get_input_tensors(is_train)[0];
  const auto &input_dim = input_tensor.get_dimensions();
  int batch_size = input_dim[0];
  compute_and_init(is_train, batch_size);
}

template <typename T>
void Loss<T>::compute_and_init(bool is_train, long long current_batchsize) {
  CudaDeviceContext context(get_device_id());
  compute(is_train, current_batchsize, regularizer_compute_rterm());

  regularizer_initialize_wgrad(is_train);

#ifndef NDEBUG
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

// Note: current_batchsize here is the batchsize on this device
template <typename T>
void Loss<T>::compute(bool is_train, long long current_batchsize, float rterm) {
  Tensor2<T> &input_tensor = get_input_tensors(is_train)[0];
  const Tensor2<float> &label_tensor = get_label_tensors(is_train)[0];
  Tensor2<float> &loss_tensor = loss_tensors_[0];

  const auto &input_dim = input_tensor.get_dimensions();
  const auto &label_dim = label_tensor.get_dimensions();

  int batch_size = input_dim[0];
  int feature_dim = input_dim[1];

  T *input = input_tensor.get_ptr();
  const float *label = label_tensor.get_ptr();
  float *loss = loss_tensor.get_ptr();

  if (current_batchsize > batch_size && current_batchsize < 0) {
    HCTR_OWN_THROW(Error_t::WrongInput, "current_batchsize > batch_size && current_batchsize < 0");
  }

  do_compute(input, label, loss, current_batchsize, feature_dim, scaler_, rterm, label_weight,
             is_train, get_gpu().get_stream());
  if (is_train) {
    // once current_batchsize < batch_size in train we set the rest dgrad to 0
    if (current_batchsize < batch_size) {
      HCTR_LIB_THROW(cudaMemsetAsync(input + current_batchsize * feature_dim, 0,
                                     (batch_size - current_batchsize) * feature_dim * sizeof(T),
                                     get_gpu().get_stream()));
    }
  }
}

template <typename T>
float Loss<T>::regularizer_compute_rterm() {
  if (regularizer_ == nullptr) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "Null regularizer is not allowed in calling Loss::regularizer_compute_rterm().");
  }
  float rterm = 0.0f;
  regularizer_->compute_rterm();
  rterm = regularizer_->get_rterm();
  return rterm;
}

template <typename T>
void Loss<T>::regularizer_initialize_wgrad(bool is_train) {
  if (regularizer_ == nullptr) {
    HCTR_OWN_THROW(
        Error_t::WrongInput,
        "Null regularizer is not allowed in calling Loss::regularizer_initialize_wgrad().");
  }
  if (is_train) {
    regularizer_->initialize_wgrad();
  }
}

template <typename T>
CrossEntropyLoss<T>::CrossEntropyLoss(const Tensor2<float> &label_tensor,
                                      const Tensor2<T> &input_tensor,
                                      const Tensor2<float> &loss_tensor,
                                      const std::shared_ptr<Regularizer<T>> &regularizer,
                                      const std::shared_ptr<GPUResource> &gpu_resource,
                                      int total_gpu_count, float scaler, bool gen_loss_summary)
    : Loss<T>(label_tensor, input_tensor, loss_tensor, regularizer, gpu_resource, total_gpu_count,
              scaler, gen_loss_summary) {
  const auto &input_dim = input_tensor.get_dimensions();
  const auto &label_dim = label_tensor.get_dimensions();
  int feature_dim = input_dim[1];

  if (feature_dim != 2) {
    HCTR_OWN_THROW(Error_t::WrongInput, "The feature dimension of CE loss input should be 2");
  }
  if (input_dim[0] != label_dim[0]) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "The batch sizes of input tensor and label tensor are not same");
  }
}

// Suppose we use one thread to calculate one sample
template <typename T>
__global__ void CrossEntropy_Kernel(T *input, const float *label, float *cel_loss, int batch_size,
                                    int total_gpu_count, int feature_dim, float scaler, float rterm,
                                    float label_weight, bool is_train) {
  int tid = threadIdx.x;
  extern __shared__ float loss_s[];

  loss_s[tid] = 0.0f;

  float z0_exp, z1_exp, a0, a1;
  int id1, id2;

  for (int i = tid; i < batch_size; i += blockDim.x) {
    id1 = i * feature_dim;
    id2 = i * feature_dim + 1;
    z0_exp = exp((double)input[id1]);
    z1_exp = exp((double)input[id2]);

    a0 = z0_exp / (z0_exp + z1_exp);
    a1 = z1_exp / (z0_exp + z1_exp);

    bool no_click = label[i] < 0.5f;

    if (is_train) {
      // calculate the grad
      input[id1] = (a0 - (no_click ? 1.0f : 0.0f)) / batch_size * scaler / total_gpu_count;
      input[id2] = (a1 - (!no_click ? 1.0f : 0.0f)) / batch_size * scaler / total_gpu_count;
    } else {
      input[id1] = a0;
      input[id2] = a1;
    }

    loss_s[tid] += -1 * log(no_click ? a0 : a1);
  }
  __syncthreads();

  float loss_tmp = 0.0f;

  if (tid == 0) {
    for (int i = 0; i < blockDim.x; ++i) loss_tmp += loss_s[i];
    cel_loss[0] = (loss_tmp / batch_size + rterm) * label_weight;
  }
}

template <typename T>
void CrossEntropyLoss<T>::do_compute(T *input, const float *label, float *loss, int batch_size,
                                     int feature_dim, float scaler, float rterm, float label_weight,
                                     bool is_train, cudaStream_t stream) {
  int block_size = min(batch_size, 1024);
  size_t smem_size = block_size * sizeof(float);
  if (block_size > 0) {
    CrossEntropy_Kernel<<<1, block_size, smem_size, stream>>>(
        input, label, loss, batch_size, Loss<T>::get_total_gpu_count(), feature_dim, scaler, rterm,
        label_weight, is_train);
  }
}

template <typename T>
BinaryCrossEntropyLoss<T>::BinaryCrossEntropyLoss(
    const Tensor2<float> &label_tensor, const Tensor2<T> &input_tensor,
    const Tensor2<float> &loss_tensor, const std::shared_ptr<Regularizer<T>> &regularizer,
    const std::shared_ptr<GPUResource> &gpu_resource, int total_gpu_count, float scaler,
    bool gen_loss_summary)
    : Loss<T>(label_tensor, input_tensor, loss_tensor, regularizer, gpu_resource, total_gpu_count,
              scaler, gen_loss_summary) {
  const auto &input_dim = input_tensor.get_dimensions();
  int feature_dim = input_dim[1];
  if (feature_dim != 1) {
    HCTR_OWN_THROW(Error_t::WrongInput, "The feature dimension of BCE loss input should be 1");
  }
}

// Suppose we use one thread to calculate one sample
template <typename T>
__global__ void BinaryCrossEntropy_Kernel(T *input, const float *label, float *bce_loss,
                                          float scaler, int batch_size, int total_gpu_count,
                                          float rterm, float label_weight, bool is_train,
                                          bool gen_loss_summary) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float val = 0.0f;
  if (tid < batch_size) {
    const float x = input[tid];
    const float y = label[tid];
    if (x >= 0) {
      float exp_neg_x = exp(-x);
      input[tid] = is_train ? ((1 - y) - exp_neg_x / (1 + exp_neg_x)) * scaler / (float)batch_size /
                                  total_gpu_count
                            : 1 / (1 + exp_neg_x);
      val = x * (1 - y) + log(1 + exp_neg_x);
    } else {
      float exp_x = exp(x);
      input[tid] = is_train
                       ? (-y + exp_x / (1 + exp_x)) * scaler / (float)batch_size / total_gpu_count
                       : exp_x / (exp_x + 1);
      val = -x * y + log(1 + exp_x);
    }
  }
  if (false == gen_loss_summary) return;
  float ret = blockReduceSum(val) * label_weight;
  if (tid < batch_size) {
    if (threadIdx.x == 0) {
      ret = (blockIdx.x == 0) ? ret / batch_size + rterm : ret / batch_size;
      atomicAdd(bce_loss, ret);
    }
  }
}

template <typename T>
void BinaryCrossEntropyLoss<T>::do_compute(T *input, const float *label, float *loss,
                                           int batch_size, int feature_dim, float scaler,
                                           float rterm, float label_weight, bool is_train,
                                           cudaStream_t stream) {
  int block_size = 512;
  int grid_size = (batch_size + block_size - 1) / block_size;
  if (grid_size > 0) {
    if (true == Loss<T>::gen_loss_summary_) {
      HCTR_LIB_THROW(cudaMemsetAsync(loss, 0, sizeof(float), stream));
    }
    BinaryCrossEntropy_Kernel<<<grid_size, block_size, 0, stream>>>(
        input, label, loss, scaler, batch_size, Loss<T>::get_total_gpu_count(), rterm, label_weight,
        is_train, Loss<T>::gen_loss_summary_);
  }
}

__forceinline__ __device__ __host__ float cross_entropy_loss(float x, float y) {
  float loss = 0.f;
  if (x >= 0) {
    float exp_neg_x = exp(-x);
    loss = x * (1 - y) + log(1 + exp_neg_x);
  } else {
    float exp_x = exp(x);
    loss = -x * y + log(1 + exp_x);
  }
  return -loss;
}

__forceinline__ __device__ __host__ float cross_entropy_loss_backward(float x, float y) {
  float grad = 0.f;
  if (x >= 0) {
    float exp_neg_x = exp(-x);
    grad = ((1 - y) - exp_neg_x / (1 + exp_neg_x));
  } else {
    float exp_x = exp(x);
    grad = (-y + exp_x / (1 + exp_x));
  }
  return grad;
}

template <typename T>
__global__ void MultiCrossEntropy_Kernel(T *input, const float *label, const float *target_weight,
                                         float *bce_loss, int batchsize, int total_gpu_count,
                                         int labels_per_sample, float scaler, float rterm,
                                         bool is_train) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  float loss_s = 0.f;
  const int size = batchsize * labels_per_sample;
  for (int i = tid; i < size; i += num_threads) {
    int target_weight_idx = i % labels_per_sample;
    const float x = input[i];
    const float y = label[i];
    const float exp_x = exp(x);
    float loss =
        (label[i] < -0.5) ? 0.f : (target_weight[target_weight_idx] * cross_entropy_loss(x, y));
    loss_s += loss;
    if (is_train) {
      input[i] = (label[i] < -0.5)
                     ? 0.f
                     : (target_weight[target_weight_idx] * cross_entropy_loss_backward(x, y) /
                        size * scaler / total_gpu_count);
    } else {
      input[i] = exp_x / (exp_x + 1);
    }
  }

  atomic_global_sum_div(-loss_s, bce_loss, size);
  if (tid == 0) {
    atomicAdd(bce_loss, rterm);
  }
  return;
}

template <typename T>
void MultiCrossEntropyLoss<T>::do_compute(T *input, const float *label, float *loss, int batch_size,
                                          int feature_dim, float scaler, float rterm,
                                          float /*label_weight*/, bool is_train,
                                          cudaStream_t stream) {
  // label_weight not currently used, since multiCross is not used for multi-label models

  int labels_per_sample = feature_dim;
  HCTR_LIB_THROW(
      cudaMemsetAsync(loss, 0, Loss<T>::get_loss_tensors()[0].get_size_in_bytes(), stream));

  const int BLOCK_SIZE = 256;
  const int GRID_SIZE = min(40, (batch_size * labels_per_sample + BLOCK_SIZE - 1) / BLOCK_SIZE);
  float *target_weight = target_weight_.get_ptr();
  MultiCrossEntropy_Kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
      input, label, target_weight, loss, batch_size, Loss<T>::get_total_gpu_count(),
      labels_per_sample, scaler, rterm, is_train);
}

template <typename T>
MultiCrossEntropyLoss<T>::MultiCrossEntropyLoss(
    const Tensor2<float> &label_tensor, const Tensor2<T> &input_tensor,
    const Tensor2<float> &loss_tensor, const std::shared_ptr<Regularizer<T>> &regularizer,
    const std::vector<float> &target_weight, const std::shared_ptr<GPUResource> &gpu_resource,
    int total_gpu_count, float scaler, bool gen_loss_summary)
    : Loss<T>(label_tensor, input_tensor, loss_tensor, regularizer, gpu_resource, total_gpu_count,
              scaler, gen_loss_summary) {
  if (label_tensor.get_dimensions().size() != 2 || input_tensor.get_dimensions().size() != 2 ||
      label_tensor.get_dimensions()[0] != input_tensor.get_dimensions()[0] ||
      label_tensor.get_dimensions()[1] != input_tensor.get_dimensions()[1]) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Format of input tensor and label tensor don't match");
  }
  // verify the length of target_weight
  if (target_weight.size() != input_tensor.get_dimensions()[1]) {
    HCTR_OWN_THROW(Error_t::WrongInput, "target_weight.size() != input_tensor.get_dims()[0]");
  }

  // load target_weight to internal Tensor
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> internal_buff =
      GeneralBuffer2<CudaAllocator>::create();
  std::vector<size_t> twdim = {1, label_tensor.get_dimensions()[1]};
  internal_buff->reserve(twdim, &target_weight_);

  CudaDeviceContext context(Loss<T>::get_device_id());
  internal_buff->allocate();
  HCTR_LIB_THROW(cudaMemcpy(target_weight_.get_ptr(), target_weight.data(),
                            target_weight_.get_size_in_bytes(), cudaMemcpyHostToDevice));

  return;
}

template class Loss<__half>;
template class Loss<float>;
template class MultiCrossEntropyLoss<__half>;
template class MultiCrossEntropyLoss<float>;
template class CrossEntropyLoss<__half>;
template class CrossEntropyLoss<float>;
template class BinaryCrossEntropyLoss<__half>;
template class BinaryCrossEntropyLoss<float>;

}  // namespace HugeCTR
