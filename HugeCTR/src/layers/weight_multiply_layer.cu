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
#include <layers/weight_multiply_layer.hpp>
#include <linalg/coalesced_reduction.cuh>
#include <linalg/matrix_vector_op.cuh>
#include <linalg/reduce.cuh>
#include <utils.cuh>
#include <utils.hpp>
namespace HugeCTR {

namespace {

#define BLOCK_DIM_SIZE 32

template <typename T>
__global__ void weight_multiply_kernel(const T* input, const T* weight, T* output, int batch_size,
                                       int slot_num, int embedding_vec_size) {
  if ((blockIdx.x < batch_size) && (threadIdx.x < embedding_vec_size)) {
    for (int i = 0; i < slot_num; i++) {
      output[blockIdx.x * slot_num * embedding_vec_size + i * embedding_vec_size + threadIdx.x] =
          input[blockIdx.x * slot_num + i] * weight[i * embedding_vec_size + threadIdx.x];
    }
  }
}
template <typename T>
__global__ void weight_wgrad_reduce_kernel(const T* top_grad, T* wgrad, int batch_size,
                                           int slot_num, int embedding_vec_size) {
  T my_sum = T(0.f);
  for (int b = blockIdx.x; b < slot_num * embedding_vec_size; b += gridDim.x) {
    my_sum = T(0.f);
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
      my_sum += top_grad[b * batch_size + i];
    }
    my_sum = blockReduceSum(my_sum);
    if (!threadIdx.x) {
      wgrad[b] = my_sum;
    }
  }
}
template <typename T>
__global__ void weight_multiply_transpose_fuse_kernel(int batch_size, int slot_num,
                                                      int embedding_vec_size, const T* top_grad,
                                                      const T* input, T* wgrad_tmp_trans) {
  int row = batch_size;
  int col = slot_num * embedding_vec_size;
  __shared__ T sh_data[BLOCK_DIM_SIZE + 1][BLOCK_DIM_SIZE];

  int src_index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int src_index_y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((src_index_x < col) && (src_index_y < row)) {
    int index_in = src_index_y * col + src_index_x;
    sh_data[threadIdx.x][threadIdx.y] = top_grad[index_in] * input[index_in / embedding_vec_size];
  }

  __syncthreads();

  int dst_index_x = blockIdx.y * blockDim.y + threadIdx.x;
  int dst_index_y = blockIdx.x * blockDim.x + threadIdx.y;
  if ((dst_index_x < row) && (dst_index_y < col)) {
    int index_out = dst_index_y * row + dst_index_x;
    wgrad_tmp_trans[index_out] = sh_data[threadIdx.y][threadIdx.x];
  }
}

template <typename T>
__global__ void weight_multiply_dgrad_kernel(const T* top_grad, const T* weight, T* dgrad,
                                             int batch_size, int slot_num, int embedding_vec_size) {
  if ((blockIdx.x < batch_size) && (threadIdx.x < embedding_vec_size)) {
    for (int i = 0; i < slot_num; i++) {
      T local_sum = top_grad[blockIdx.x * slot_num * embedding_vec_size + i * embedding_vec_size +
                             threadIdx.x] *
                    weight[i * embedding_vec_size + threadIdx.x];

      local_sum = blockReduceSum(local_sum);
      if (threadIdx.x == 0) {
        dgrad[blockIdx.x * slot_num + i] = local_sum;
      }
    }
  }
}

template <typename T>
void weight_multiply_wgrad(const T* top_grad, const T* input, T* wgrad, T* wgrad_tmp_trans,
                           int batch_size, int slot_num, int embedding_vec_size,
                           cudaStream_t stream) {
  dim3 blockSize1(BLOCK_DIM_SIZE, BLOCK_DIM_SIZE, 1);
  dim3 gridSize1((slot_num * embedding_vec_size + blockSize1.x - 1) / blockSize1.x,
                 (batch_size + blockSize1.y - 1) / blockSize1.y, 1);
  weight_multiply_transpose_fuse_kernel<<<gridSize1, blockSize1, 0, stream>>>(
      batch_size, slot_num, embedding_vec_size, top_grad, input, wgrad_tmp_trans);
  // MLCommon::LinAlg::reduce sometimes produce wrong answer
  weight_wgrad_reduce_kernel<<<slot_num * embedding_vec_size, 512, 0, stream>>>(
      wgrad_tmp_trans, wgrad, batch_size, slot_num, embedding_vec_size);
}

template <>
void weight_multiply_wgrad<__half>(const __half* top_grad, const __half* input, __half* wgrad,
                                   __half* wgrad_tmp_trans, int batch_size, int slot_num,
                                   int embedding_vec_size, cudaStream_t stream) {
  dim3 blockSize1(BLOCK_DIM_SIZE, BLOCK_DIM_SIZE, 1);
  dim3 gridSize1((slot_num * embedding_vec_size + blockSize1.x - 1) / blockSize1.x,
                 (batch_size + blockSize1.y - 1) / blockSize1.y, 1);
  weight_multiply_transpose_fuse_kernel<<<gridSize1, blockSize1, 0, stream>>>(
      batch_size, slot_num, embedding_vec_size, top_grad, input, wgrad_tmp_trans);
  // coalescedReduction could lead to implicit shared memory misaligned address error reported by
  // compute-sanitizer, but sometimes gives right result...??
  // MLCommon::LinAlg::coalescedReduction(wgrad, wgrad_tmp_trans, batch_size,
  //                                      slot_num * embedding_vec_size, __float2half(0.0f),
  //                                      stream);
  weight_wgrad_reduce_kernel<<<slot_num * embedding_vec_size, 512, 0, stream>>>(
      wgrad_tmp_trans, wgrad, batch_size, slot_num, embedding_vec_size);
}

template <typename T>
void weight_multiply_dgrad(const T* top_grad, const T* weight, T* dgrad, int batch_size,
                           int slot_num, int embedding_vec_size, cudaStream_t stream) {
  dim3 blockSize(embedding_vec_size, 1, 1);  // note that embedding_vec_size should be < 1024
  dim3 gridSize(batch_size, 1, 1);
  weight_multiply_dgrad_kernel<<<gridSize, blockSize, 0, stream>>>(
      top_grad, weight, dgrad, batch_size, slot_num, embedding_vec_size);
}

}  // end of namespace

template <typename T>
WeightMultiplyLayer<T>::WeightMultiplyLayer(const core23::Tensor& in_tensor,
                                            core23::Tensor& out_tensor,
                                            const core23::Shape& weight_dims,
                                            const std::shared_ptr<GPUResource>& gpu_resource,
                                            std::vector<Initializer_t> initializer_types)
    : TrainableLayer<T>({in_tensor}, {}, gpu_resource, initializer_types) {
  try {
    const auto& in_dims = in_tensor.shape();
    if (in_dims.dims() != 2) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Only 2D tensors can be multiplied");
    }
    if (weight_dims.dims() != 2) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Only 2D weights is allowed for weight_multiply layer");
    }
    if (weight_dims.size(0) != in_dims.size(1)) {
      HCTR_OWN_THROW(Error_t::WrongInput, "weight_dims[0] must be equal to in_dims.size(1)");
    }

    batch_size_ = in_dims.size(0);
    slot_num_ = weight_dims.size(0);
    embedding_vec_size_ = weight_dims.size(1);

    core23::Shape out_dims{batch_size_, slot_num_ * embedding_vec_size_};

    core23::BufferParams blobs_buffer_params = {};
    blobs_buffer_params.channel = GetBlobsBufferChannel();
    core23::Device device(core23::DeviceType::GPU, gpu_resource->get_device_id());

    out_tensor = core23::Tensor(core23::TensorParams()
                                    .data_type(core23::ToScalarType<T>::value)
                                    .shape(out_dims)
                                    .device(device)
                                    .buffer_params(blobs_buffer_params));

    this->output_tensors_.push_back(out_tensor);

    this->set_weight(0, weight_dims);
    this->set_wgrad(0, weight_dims);

    wgrad_tmp_trans_ = core23::Tensor(core23::TensorParams()
                                          .data_type(core23::ToScalarType<T>::value)
                                          .shape(out_dims)
                                          .device(device)
                                          .buffer_params(blobs_buffer_params));
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
std::unique_ptr<DataSimulator> WeightMultiplyLayer<T>::get_uniform_initializer(const int index) {
  float bottom_dim = slot_num_;
  float top_dim = slot_num_ * embedding_vec_size_;

  float limit = 1.0f / ((0 == index ? bottom_dim : 0) + top_dim);
  return std::make_unique<UniformDataSimulator>(-1 * limit, limit);
}

template <typename T>
std::unique_ptr<DataSimulator> WeightMultiplyLayer<T>::get_xavier_uniform_initializer(
    const int index) {
  float bottom_dim = slot_num_;
  float top_dim = slot_num_ * embedding_vec_size_;

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Uniform,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

template <typename T>
std::unique_ptr<DataSimulator> WeightMultiplyLayer<T>::get_xavier_norm_initializer(
    const int index) {
  float bottom_dim = slot_num_;
  float top_dim = slot_num_ * embedding_vec_size_;

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Norm,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

template <typename T>
std::unique_ptr<DataSimulator> WeightMultiplyLayer<T>::get_default_initializer(const int index) {
  float bottom_dim = slot_num_;
  float top_dim = slot_num_ * embedding_vec_size_;

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Uniform,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

template <typename T>
void WeightMultiplyLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(this->get_device_id());

  T* input = this->input_tensors_[0].template data<T>();
  const T* weight = this->get_weight(0).template data<T>();
  T* output = this->output_tensors_[0].template data<T>();

  dim3 blockSize(embedding_vec_size_, 1, 1);
  dim3 gridSize(batch_size_, 1, 1);
  weight_multiply_kernel<<<gridSize, blockSize, 0, this->get_gpu().get_stream()>>>(
      input, weight, output, batch_size_, slot_num_, embedding_vec_size_);
}

template <typename T>
void WeightMultiplyLayer<T>::bprop() {
  CudaDeviceContext context(this->get_device_id());

  const T* weight = this->get_weight(0).template data<T>();
  T* wgrad = this->get_wgrad(0).template data<T>();
  T* wgrad_tmp_trans = wgrad_tmp_trans_.data<T>();
  T* input = this->input_tensors_[0].template data<T>();
  T* output = this->output_tensors_[0].template data<T>();
  weight_multiply_wgrad(output, input, wgrad, wgrad_tmp_trans, batch_size_, slot_num_,
                        embedding_vec_size_, this->get_gpu().get_stream());

  // CAUTION: dgrad computation will modify the "input", so it must be put after wgrad computation
  weight_multiply_dgrad(output, weight, input, batch_size_, slot_num_, embedding_vec_size_,
                        this->get_gpu().get_stream());
}

template class WeightMultiplyLayer<float>;
template class WeightMultiplyLayer<__half>;

}  // namespace HugeCTR
