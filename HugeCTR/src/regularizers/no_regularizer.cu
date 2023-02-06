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

#include <regularizers/no_regularizer.hpp>
#include <utility>
#include <utils.cuh>

namespace HugeCTR {
template <typename T>
NoRegularizer<T>::NoRegularizer(const Tensor2<float>& weight_buff, const Tensor2<T>& wgrad_buff,
                                const int batch_size,
                                const std::shared_ptr<GPUResource>& gpu_resource)
    : Regularizer<T>(weight_buff, wgrad_buff, batch_size, gpu_resource) {}

template <typename T>
void NoRegularizer<T>::do_compute_rterm(const float* weight, float* rterm, int num_elements) {
  *rterm = 0.0f;
}

template <typename T>
void NoRegularizer<T>::do_initialize_wgrad(const float* weight, T* wgrad, int num_elements,
                                           cudaStream_t stream) {
  // int n_blocks = Regularizer<T>::get_gpu().get_sm_count() * 4;
  // int block_size = 512;
  // initialize_array<<<n_blocks, block_size, 0, stream>>>(
  //     wgrad, num_elements, T(0.0f));
  HCTR_LIB_THROW(cudaMemsetAsync(wgrad, 0, num_elements * sizeof(T), stream));
}

template class NoRegularizer<__half>;
template class NoRegularizer<float>;

}  // namespace HugeCTR
