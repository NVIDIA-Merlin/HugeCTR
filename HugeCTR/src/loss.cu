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

#include <vector>
#include "HugeCTR/include/loss.hpp"
#include "HugeCTR/include/utils.cuh"

namespace HugeCTR {

Loss::Loss(const std::shared_ptr<const Tensor<float>> &label_tensor,
           const std::shared_ptr<Tensor<float>> &input_tensor,
           const std::shared_ptr<Tensor<float>> &loss_tensor,
           const std::shared_ptr<Regularizer> regularizer, int device_id)
    : label_tensors_(1, label_tensor),
      input_tensors_(1, input_tensor),
      loss_tensors_(1, loss_tensor),
      regularizer_(regularizer),
      device_id_(device_id) {}

void Loss::fused_loss_computation(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  const auto &input_tensor = input_tensors_[0];
  const auto &label_tensor = label_tensors_[0];
  const auto &loss_tensor = loss_tensors_[0];

  const auto &input_dim = input_tensor->get_dims();
  const auto &label_dim = label_tensor->get_dims();

  bool row_major = (input_tensor->get_format() == TensorFormat_t::HW);

  int batch_size = row_major ? input_dim[0] : input_dim[1];
  int feature_dim = row_major ? input_dim[1] : input_dim[0];

  float *input = input_tensor->get_ptr();
  const float *label = label_tensor->get_ptr();
  float *loss = loss_tensor->get_ptr();

  float scaler = 1.f;
  float rterm = 0.0f;
  if (regularizer_) {
    regularizer_->compute_rterm(stream);
    rterm = regularizer_->get_rterm();
  }

  do_fused_loss_computation(input, label, loss, batch_size, feature_dim, scaler, rterm, stream);

  if (regularizer_) {
    regularizer_->initialize_wgrad(stream);
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

CrossEntropyLoss::CrossEntropyLoss(const std::shared_ptr<const Tensor<float>> &label_tensor,
                                   const std::shared_ptr<Tensor<float>> &input_tensor,
                                   const std::shared_ptr<Tensor<float>> &loss_tensor,
                                   const std::shared_ptr<Regularizer> regularizer, int device_id)
    : Loss(label_tensor, input_tensor, loss_tensor, regularizer, device_id) {
  if (input_tensor->get_format() != label_tensor->get_format())
    CK_THROW_(Error_t::WrongInput, "Format of input tensor and label tensor don't match");

  const auto &input_dim = input_tensor->get_dims();
  const auto &label_dim = label_tensor->get_dims();
  bool row_major = (input_tensor->get_format() == TensorFormat_t::HW);
  int feature_dim = row_major ? input_dim[1] : input_dim[0];

  if (feature_dim != 2)
    CK_THROW_(Error_t::WrongInput, "The feature dimension of CE loss input should be 2");
  if (row_major && input_dim[0] != label_dim[0])
    CK_THROW_(Error_t::WrongInput, "The batch sizes of input tensor and label tensor are not same");
  if (!row_major && input_dim[1] != label_dim[1])
    CK_THROW_(Error_t::WrongInput, "The batch sizes of input tensor and label tensor are not same");
}

// Suppose we use one thread to calculate one sample
__global__ void CrossEntropy_Kernel(float *input, const float *label, float *cel_loss,
                                    int batch_size, int feature_dim, bool row_major, float scaler,
                                    float rterm) {
  int tid = threadIdx.x;
  extern __shared__ float loss_s[];

  loss_s[tid] = 0.0f;

  float z0_exp, z1_exp, a0, a1;
  int id1, id2;

  for (int i = tid; i < batch_size; i += blockDim.x) {
    id1 = row_major ? i * feature_dim : i;
    id2 = row_major ? i * feature_dim + 1 : i + batch_size;
    z0_exp = exp((double)input[id1]);
    z1_exp = exp((double)input[id2]);

    a0 = z0_exp / (z0_exp + z1_exp);
    a1 = z1_exp / (z0_exp + z1_exp);

    bool no_click = label[i] < 0.5f;

    // calculate the grad
    input[id1] = (a0 - (no_click ? 1.0f : 0.0f)) / batch_size * scaler;
    input[id2] = (a1 - (!no_click ? 1.0f : 0.0f)) / batch_size * scaler;
    ;

    loss_s[tid] += -1 * log(no_click ? a0 : a1);
  }
  __syncthreads();

  float loss_tmp = 0.0f;

  if (tid == 0) {
    for (int i = 0; i < blockDim.x; ++i) loss_tmp += loss_s[i];
    cel_loss[0] = loss_tmp / batch_size + rterm;
  }
}

void CrossEntropyLoss::do_fused_loss_computation(float *input, const float *label, float *loss,
                                                 int batch_size, int feature_dim, float scaler,
                                                 float rterm, cudaStream_t stream) {
  bool row_major = (input_tensors_[0]->get_format() == TensorFormat_t::HW);
  int block_size = min(batch_size, 1024);
  CrossEntropy_Kernel<<<1, block_size, block_size * sizeof(float), stream>>>(
      input, label, loss, batch_size, feature_dim, row_major, scaler, rterm);
}

BinaryCrossEntropyLoss::BinaryCrossEntropyLoss(
    const std::shared_ptr<const Tensor<float>> &label_tensor,
    const std::shared_ptr<Tensor<float>> &input_tensor,
    const std::shared_ptr<Tensor<float>> &loss_tensor,
    const std::shared_ptr<Regularizer> regularizer, int device_id)
    : Loss(label_tensor, input_tensor, loss_tensor, regularizer, device_id) {
  if (input_tensor->get_format() != label_tensor->get_format())
    CK_THROW_(Error_t::WrongInput, "Format of input tensor and label tensor don't match");

  bool row_major = (input_tensor->get_format() == TensorFormat_t::HW);
  const auto &input_dim = input_tensor->get_dims();
  int feature_dim = row_major ? input_dim[1] : input_dim[0];
  if (feature_dim != 1)
    CK_THROW_(Error_t::WrongInput, "The feature dimension of BCE loss input should be 1");
}

// Suppose we use one thread to calculate one sample
__global__ void BinaryCrossEntropy_Kernel(float *input, const float *label, float *bce_loss,
                                          float scaler, int batch_size, float rterm) {
  int tid = threadIdx.x;
  extern __shared__ float loss_s[];
  loss_s[tid] = 0.0f;

  for (int i = tid; i < batch_size; i += blockDim.x) {
    const float x = input[i];

    const float y = label[i];
    if (x >= 0) {
      float exp_neg_x = exp(-x);
      loss_s[tid] += x * (1 - y) + log(1 + exp_neg_x);
      input[i] = ((1 - y) - exp_neg_x / (1 + exp_neg_x)) * scaler / (float)batch_size;
    } else {
      float exp_x = exp(x);
      loss_s[tid] += -x * y + log(1 + exp_x);
      input[i] = (-y + exp_x / (1 + exp_x)) * scaler / (float)batch_size;
    }
  }
  __syncthreads();

  float loss_tmp = 0.0f;
  if (tid == 0) {
    for (int i = 0; i < blockDim.x; ++i) loss_tmp += loss_s[i];
    bce_loss[0] = loss_tmp / batch_size + rterm;
  }
}

void BinaryCrossEntropyLoss::do_fused_loss_computation(float *input, const float *label,
                                                       float *loss, int batch_size, int feature_dim,
                                                       float scaler, float rterm,
                                                       cudaStream_t stream) {
  int block_size = min(batch_size, 1024);
  BinaryCrossEntropy_Kernel<<<1, block_size, block_size * sizeof(float), stream>>>(
      input, label, loss, scaler, batch_size, rterm);
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

__global__ void MultiCrossEntropy_Kernel(float *input, const float *label,
                                         const float *target_weight, float *bce_loss, int batchsize,
                                         int labels_per_sample, float scaler, float rterm) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  float loss_s = 0.f;
  const int size = batchsize * labels_per_sample;
  for (int i = tid; i < size; i += num_threads) {
    int target_weight_idx = i % labels_per_sample;
    const float x = input[i];
    const float y = label[i];
    float loss =
        (label[i] < -0.5) ? 0.f : (target_weight[target_weight_idx] * cross_entropy_loss(x, y));
    loss_s += loss;
    input[i] = (label[i] < -0.5) ? 0.f
                                 : (target_weight[target_weight_idx] *
                                    cross_entropy_loss_backward(x, y) / size * scaler);
  }

  atomic_global_sum_div(-loss_s, bce_loss, size);
  if (tid == 0) {
    atomicAdd(bce_loss, rterm);
  }
  return;
}

void MultiCrossEntropyLoss::do_fused_loss_computation(float *input, const float *label, float *loss,
                                                      int batch_size, int feature_dim, float scaler,
                                                      float rterm, cudaStream_t stream) {
  int labels_per_sample = feature_dim;
  cudaMemsetAsync(loss, 0, loss_tensors_[0]->get_size(), stream);

  const int BLOCK_SIZE = 256;
  const int GRID_SIZE = min(40, (batch_size * labels_per_sample - 1) / BLOCK_SIZE);
  float *target_weight = target_weight_->get_ptr();
  MultiCrossEntropy_Kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
      input, label, target_weight, loss, batch_size, labels_per_sample, scaler, rterm);
}

MultiCrossEntropyLoss::MultiCrossEntropyLoss(
    const std::shared_ptr<const Tensor<float>> &label_tensor,
    const std::shared_ptr<Tensor<float>> &input_tensor,
    const std::shared_ptr<Tensor<float>> &loss_tensor,
    const std::shared_ptr<Regularizer> regularizer, const std::vector<float> &target_weight,
    int device_id)
    : Loss(label_tensor, input_tensor, loss_tensor, regularizer, device_id) {
  if (label_tensor->get_dims().size() != 2 || label_tensor->get_format() != TensorFormat_t::HW ||
      input_tensor->get_dims().size() != 2 || input_tensor->get_format() != TensorFormat_t::HW ||
      label_tensor->get_dims()[0] != input_tensor->get_dims()[0] ||
      label_tensor->get_dims()[1] != input_tensor->get_dims()[1]) {
    CK_THROW_(Error_t::WrongInput, "Format of input tensor and label tensor don't match");
  }
  // verify the length of target_weight
  if (target_weight.size() != input_tensor->get_dims()[1]) {
    CK_THROW_(Error_t::WrongInput, "target_weight.size() != input_tensor.get_dims()[0]");
  }

  // load target_weight to internal Tensor
  internal_buff_.reset(new GeneralBuffer<float>());
  std::vector<size_t> twdim = {1, label_tensor->get_dims()[1]};
  target_weight_.reset(new Tensor<float>(twdim, internal_buff_, TensorFormat_t::HW));
  internal_buff_->init(device_id);
  CudaDeviceContext context(device_id);
  CK_CUDA_THROW_(cudaMemcpy(target_weight_->get_ptr(), target_weight.data(),
                            target_weight_->get_size(), cudaMemcpyHostToDevice));

  return;
}
}  // namespace HugeCTR
