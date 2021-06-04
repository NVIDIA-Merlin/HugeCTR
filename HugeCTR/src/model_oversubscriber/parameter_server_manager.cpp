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

#include "HugeCTR/include/model_oversubscriber/parameter_server_manager.hpp"
#include <string>

namespace HugeCTR {

template <typename TypeKey>
ParameterServerManager<TypeKey>::ParameterServerManager(bool use_host_ps,
    const std::vector<std::string>& sparse_embedding_files,
    const std::vector<Embedding_t>& embedding_types,
    const std::vector<SparseEmbeddingHashParams>& embedding_params,
    size_t buffer_size, std::shared_ptr<ResourceManager> resource_manager) {
  try {
    if (sparse_embedding_files.size() == 0)
      CK_THROW_(Error_t::WrongInput, "must provide sparse_model_file. \
          if train from scratch, please specify a name to store the trained embedding model");

    if (embedding_params.size() != sparse_embedding_files.size())
      CK_THROW_(Error_t::WrongInput,
          std::string("embedding_params.size() != sparse_embedding_files.size()") + ": " +
          std::to_string(embedding_params.size()) + " != " + 
          std::to_string(sparse_embedding_files.size()));

    if (use_host_ps) {
      MESSAGE_("Host MEM-based Parameter Server is enabled");
    } else {
      MESSAGE_("SSD-based Parameter Server is enabled, performance may drop!!!");
    }

    size_t max_vec_size = 0;
    for (int i = 0; i < static_cast<int>(embedding_params.size()); i++) {
      size_t ith_vec_size = embedding_params[i].embedding_vec_size;
      max_vec_size = (ith_vec_size > max_vec_size) ? ith_vec_size : max_vec_size;

      MESSAGE_("construct sparse models for model oversubscriber: " + sparse_embedding_files[i]);
      ps_.push_back(std::make_shared<ParameterServer<TypeKey>>(use_host_ps,
          sparse_embedding_files[i], embedding_types[i], embedding_params[i].embedding_vec_size,
          resource_manager));
    }

    std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> blobs_buff =
      GeneralBuffer2<CudaHostAllocator>::create();

    Tensor2<TypeKey> tensor_keys;
    Tensor2<size_t> tensor_slot_id;
    blobs_buff->reserve({buffer_size}, &tensor_keys);
    blobs_buff->reserve({buffer_size}, &tensor_slot_id);

    blobs_buff->reserve({buffer_size, max_vec_size}, &(buf_bag_.embedding));
    blobs_buff->allocate();

    buf_bag_.keys = tensor_keys.shrink();
    buf_bag_.slot_id = tensor_slot_id.shrink();
    
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw rt_err;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  }
}

template class ParameterServerManager<long long>;
template class ParameterServerManager<unsigned>;

}  // namespace HugeCTR
