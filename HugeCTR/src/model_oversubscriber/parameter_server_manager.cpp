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

namespace HugeCTR {

template <typename TypeHashKey>
ParameterServerManager<TypeHashKey>::ParameterServerManager(
    const std::vector<SparseEmbeddingHashParams>& embedding_params,
    const Embedding_t embedding_type,
    const std::vector<std::string>& sparse_embedding_files,
    size_t buffer_size) {
  try {
    if (sparse_embedding_files.size() == 0)
      CK_THROW_(Error_t::WrongInput, "must provide sparse_model_file. \
          if train from scratch, please specify a name to store the trained embedding model");

    if (embedding_params.size() != sparse_embedding_files.size())
      CK_THROW_(Error_t::WrongInput, "num of embeddings and num of sparse_model_file don't equal");
      
    size_t max_vec_size = 0;
    for (int i = 0; i < static_cast<int>(embedding_params.size()); i++) {
      size_t ith_vec_size = embedding_params[i].embedding_vec_size;
      max_vec_size = (ith_vec_size > max_vec_size) ? ith_vec_size : max_vec_size;

      MESSAGE_("construct sparse models for model oversubscriber: " + sparse_embedding_files[i]);
      ps_.push_back(std::make_shared<ParameterServer<TypeHashKey>>
        (embedding_params[i], sparse_embedding_files[i], embedding_type));
    }

    std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> blobs_buff =
      GeneralBuffer2<CudaHostAllocator>::create();

    Tensor2<TypeHashKey> tensor_keys;
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
