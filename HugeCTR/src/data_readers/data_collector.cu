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

#include <common.hpp>
#include <data_readers/data_collector.hpp>

namespace HugeCTR {

template <typename TypeComp>
__global__ void split_kernel__(int batchsize, float* label_ptr, int label_dim, TypeComp* dense_ptr,
                               int dense_dim, const float* label_dense, int label_dense_dim) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < batchsize * label_dense_dim) {
    const int in_col = idx % label_dense_dim;
    const int in_row = idx / label_dense_dim;
    const int out_row = in_row;
    if (in_col < label_dim) {
      const int out_col = in_col;
      label_ptr[out_row * label_dim + out_col] = label_dense[idx];
    } else {
      const int out_col = in_col - label_dim;
      dense_ptr[out_row * dense_dim + out_col] = label_dense[idx];
    }
  }
  return;
}

template <typename TypeComp>
void split(Tensor2<float>& label_tensor, Tensor2<TypeComp>& dense_tensor,
           const Tensor2<float>& label_dense_buffer, const int label_dense_dim,
           cudaStream_t stream) {
  // check the input size
  assert(label_tensor.get_dimensions()[0] == dense_tensor.get_dimensions()[0]);
  assert(label_tensor.get_num_elements() + dense_tensor.get_num_elements() ==
         label_dense_buffer.get_num_elements());

  const int batchsize = label_tensor.get_dimensions()[0];
  const int label_dim = label_tensor.get_dimensions()[1];
  const int dense_dim = dense_tensor.get_dimensions()[1];
  // HCTR_LOG(INFO, WORLD, "%d, %d, %d\n", label_dim, dense_dim, label_dense_dim);

  const int BLOCK_DIM = 256;
  const int GRID_DIM = (label_dense_buffer.get_num_elements() - 1) / BLOCK_DIM + 1;
  assert(dense_dim >= 0 || "dense_dim should be >= 0");

  if (dense_dim > 0) {
    split_kernel__<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
        batchsize, label_tensor.get_ptr(), label_dim, dense_tensor.get_ptr(), dense_dim,
        label_dense_buffer.get_ptr(), label_dense_dim);
  } else if (dense_dim == 0) {
    split_kernel__<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
        batchsize, label_tensor.get_ptr(), label_dim, (TypeComp*)0, 0, label_dense_buffer.get_ptr(),
        label_dense_dim);

  } else {
    HCTR_OWN_THROW(Error_t::WrongInput, "dense_dim < 0");
  }

  return;
}

template <typename T>
void broadcast(const std::shared_ptr<ThreadBuffer>& thread_buffer,
               std::shared_ptr<BroadcastBuffer>& broadcast_buffer,
               std::vector<size_t>& last_batch_nnz_,
               const std::shared_ptr<ResourceManager>& resource_manager) {
  int param_num = thread_buffer->param_num;
  int dense_dim = thread_buffer->dense_dim;
  int label_dim = thread_buffer->label_dim;
  int batch_size = thread_buffer->batch_size;
  int batch_size_per_gpu = batch_size / resource_manager->get_global_gpu_count();
  int local_gpu_count = resource_manager->get_local_gpu_count();

#pragma omp parallel for num_threads(local_gpu_count)
  for (int i = 0; i < local_gpu_count; ++i) {
    auto local_gpu = resource_manager->get_local_gpu(i);

    CudaDeviceContext ctx(local_gpu->get_device_id());

    for (int param_id = 0; param_id < param_num; ++param_id) {
      auto src_sparse_tensor =
          SparseTensor<T>::stretch_from(thread_buffer->device_sparse_buffers[param_id]);
      auto dst_sparse_tensor =
          SparseTensor<T>::stretch_from(broadcast_buffer->sparse_buffers[i * param_num + param_id]);

      if (thread_buffer->is_fixed_length[param_id] &&
          last_batch_nnz_[i * param_num + param_id] == src_sparse_tensor.nnz()) {
        HCTR_LIB_THROW(cudaMemcpyAsync(dst_sparse_tensor.get_value_ptr(),
                                       src_sparse_tensor.get_value_ptr(),
                                       src_sparse_tensor.nnz() * sizeof(T),
                                       cudaMemcpyDeviceToDevice, local_gpu->get_p2p_stream()));
      } else {
        sparse_tensor_helper::cuda::copy_async(dst_sparse_tensor, src_sparse_tensor,
                                               cudaMemcpyDeviceToDevice,
                                               local_gpu->get_p2p_stream());
        last_batch_nnz_[i * param_num + param_id] = src_sparse_tensor.nnz();
      }
    }

    auto dst_dense_tensor = Tensor2<float>::stretch_from(broadcast_buffer->dense_tensors[i]);
    auto src_dense_tensor = Tensor2<float>::stretch_from(thread_buffer->device_dense_buffers);
    HCTR_LIB_THROW(cudaMemcpyAsync(
        dst_dense_tensor.get_ptr(),
        src_dense_tensor.get_ptr() + i * batch_size_per_gpu * (label_dim + dense_dim),
        batch_size_per_gpu * (label_dim + dense_dim) * sizeof(float), cudaMemcpyDeviceToDevice,
        local_gpu->get_p2p_stream()));
    HCTR_LIB_THROW(cudaStreamSynchronize(local_gpu->get_p2p_stream()));
  }
}

template void broadcast<unsigned int>(const std::shared_ptr<ThreadBuffer>& thread_buffer,
                                      std::shared_ptr<BroadcastBuffer>& broadcast_buffer,
                                      std::vector<size_t>& last_batch_nnz_,
                                      const std::shared_ptr<ResourceManager>& resource_manager);

template void broadcast<long long>(const std::shared_ptr<ThreadBuffer>& thread_buffer,
                                   std::shared_ptr<BroadcastBuffer>& broadcast_buffer,
                                   std::vector<size_t>& last_batch_nnz_,
                                   const std::shared_ptr<ResourceManager>& resource_manager);

template void split<float>(Tensor2<float>& label_tensor, Tensor2<float>& dense_tensor,
                           const Tensor2<float>& label_dense_buffer, const int label_dense_dim,
                           cudaStream_t stream);

template void split<__half>(Tensor2<float>& label_tensor, Tensor2<__half>& dense_tensor,
                            const Tensor2<float>& label_dense_buffer, const int label_dense_dim,
                            cudaStream_t stream);

}  // namespace HugeCTR
