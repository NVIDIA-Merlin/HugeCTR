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

#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {
void SparseEmbeddingFunctors::sync_all_gpus(const ResourceManager& resource_manager) const {
  CudaDeviceContext context;

  size_t local_gpu_count = resource_manager.get_local_gpu_count();
  for (size_t id = 0; id < local_gpu_count; id++) {
    const auto& local_gpu = resource_manager.get_local_gpu(id);
    context.set_device(local_gpu->get_device_id());
    HCTR_LIB_THROW(cudaStreamSynchronize(local_gpu->get_stream()));
  }
}

}  // namespace HugeCTR