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

#include <layers/fm_order2_layer.hpp>
#include <utils.hpp>

namespace HugeCTR {

namespace {
__global__ void fm_order2_kernel(const float* in, float* out, int batch_size, int slot_num,
                                 int emb_vec_size) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (tid < emb_vec_size && bid < batch_size) {
    float emb_sum = 0.0f;
    float emb_sum_square = 0.0f;
    float emb_square_sum = 0.0f;
    int offset = bid * slot_num * emb_vec_size + tid;

    for (int i = 0; i < slot_num; i++) {
      int index = offset + i * emb_vec_size;
      float temp = in[index];
      emb_sum += temp;
      emb_square_sum += temp * temp;
    }
    emb_sum_square = emb_sum * emb_sum;

    out[bid * emb_vec_size + tid] = 0.5f * (emb_sum_square - emb_square_sum);
  }
}

__global__ void fm_order2_dgrad_kernel(const float* in, const float* top_grad, float* dgrad,
                                       int batch_size, int slot_num, int emb_vec_size) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (tid < emb_vec_size && bid < batch_size) {
    float emb_sum = 0.0f;
    int offset = bid * slot_num * emb_vec_size + tid;

    for (int i = 0; i < slot_num; i++) {
      int index = offset + i * emb_vec_size;
      emb_sum += in[index];
    }
    float tgrad = top_grad[bid * emb_vec_size + tid];
    for (int i = 0; i < slot_num; i++) {
      int index = offset + i * emb_vec_size;
      dgrad[index] = tgrad * (emb_sum - in[index]);
    }
  }
}

}  // end of namespace

FmOrder2Layer::FmOrder2Layer(const Tensor2<float>& in_tensor, const Tensor2<float>& out_tensor,
                             const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  try {
    const auto& in_dims = in_tensor.get_dimensions();
    if (in_dims.size() != 2) {
      CK_THROW_(Error_t::WrongInput, "only 2D tensors can be used as input for FmOrder2Layer");
    }
    const auto& out_dims = out_tensor.get_dimensions();
    if (out_dims.size() != 2) {
      CK_THROW_(Error_t::WrongInput, "only 2D tensors can be used as output for FmOrder2Layer");
    }
    if ((in_dims[1] % out_dims[1]) != 0) {
      CK_THROW_(Error_t::WrongInput, "(in_dims[1] % out_dims[1]) != 0");
    }

    batch_size_ = in_dims[0];
    slot_num_ = in_dims[1] / out_dims[1];
    embedding_vec_size_ = out_dims[1];

    in_tensors_.push_back(in_tensor);
    out_tensors_.push_back(out_tensor);

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void FmOrder2Layer::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  const float* in = in_tensors_[0].get_ptr();
  float* out = out_tensors_[0].get_ptr();

  dim3 blockSize(embedding_vec_size_, 1, 1);
  dim3 grdiSize(batch_size_, 1, 1);
  fm_order2_kernel<<<grdiSize, blockSize, 0, get_gpu().get_stream()>>>(
      in, out, batch_size_, slot_num_, embedding_vec_size_);
}

void FmOrder2Layer::bprop() {
  CudaDeviceContext context(get_device_id());

  float* in = in_tensors_[0].get_ptr();
  const float* out = out_tensors_[0].get_ptr();

  dim3 blockSize(embedding_vec_size_, 1, 1);
  dim3 gridSize(batch_size_, 1, 1);
  fm_order2_dgrad_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(in,
                                                                             out,  // top_grad
                                                                             in,   // dgrad
                                                                             batch_size_, slot_num_,
                                                                             embedding_vec_size_);
}

}  // end of namespace HugeCTR
