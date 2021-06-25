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

    size_t max_vec_size = 0, max_voc_size_per_gpu = 0;
    for (int i = 0; i < static_cast<int>(embedding_params.size()); i++) {
      size_t ith_vec_size = embedding_params[i].embedding_vec_size;
      max_vec_size = (ith_vec_size > max_vec_size) ? ith_vec_size : max_vec_size;

      size_t tmp_voc_size = embedding_params[i].max_vocabulary_size_per_gpu;
      max_voc_size_per_gpu = (tmp_voc_size > max_voc_size_per_gpu) ?
                             tmp_voc_size : max_voc_size_per_gpu;

      MESSAGE_("construct sparse models for model oversubscriber: " + sparse_embedding_files[i]);
      ps_.push_back(std::make_shared<ParameterServer<TypeKey>>(use_host_ps,
          sparse_embedding_files[i], embedding_types[i], embedding_params[i].embedding_vec_size,
          resource_manager));
    }

    bool has_localized_embedding = false;
    for (auto type : embedding_types) {
      if (type != Embedding_t::DistributedSlotSparseEmbeddingHash) {
        has_localized_embedding = true;
        break;
      }
    }

    bool all_one_hot_embedding = true;
    for (auto type : embedding_types) {
      if (type != Embedding_t::LocalizedSlotSparseEmbeddingOneHot) {
        all_one_hot_embedding = false;
        break;
      }
    }

    auto host_blobs_buff = GeneralBuffer2<CudaHostAllocator>::create();

    Tensor2<TypeKey> tensor_keys;
    Tensor2<size_t> tensor_slot_id;
    host_blobs_buff->reserve({buffer_size}, &tensor_keys);
    host_blobs_buff->reserve({buffer_size}, &tensor_slot_id);
    host_blobs_buff->reserve({buffer_size, max_vec_size}, &(buf_bag_.embedding));

    buf_bag_.keys = tensor_keys.shrink();
    buf_bag_.slot_id = tensor_slot_id.shrink();

    const size_t local_gpu_count = resource_manager->get_local_gpu_count();

    for (size_t id = 0; id < local_gpu_count; id++) {
      Tensor2<float> tensor;
      host_blobs_buff->reserve({max_voc_size_per_gpu, max_vec_size}, &tensor);
      buf_bag_.h_value_tensors.push_back(tensor);

      if (has_localized_embedding && !all_one_hot_embedding) {
        Tensor2<size_t> tensor_slot_id;
        host_blobs_buff->reserve({max_voc_size_per_gpu}, &tensor_slot_id);
        buf_bag_.h_slot_id_tensors.push_back(tensor_slot_id);
      }
    }
    host_blobs_buff->allocate();

    CudaDeviceContext context;
    for (size_t id = 0; id < local_gpu_count; id++) {
      if (all_one_hot_embedding) break;
      context.set_device(resource_manager->get_local_gpu(id)->get_device_id());
      {
        auto uvm_blobs_buff = GeneralBuffer2<CudaManagedAllocator>::create();
        Tensor2<TypeKey> tensor;
        uvm_blobs_buff->reserve({max_voc_size_per_gpu}, &tensor);
        buf_bag_.uvm_key_tensor_bags.push_back(tensor.shrink());
        uvm_blobs_buff->allocate();
      }

      {
        auto hbm_blobs_buff = GeneralBuffer2<CudaAllocator>::create();
        Tensor2<size_t> tensor;
        hbm_blobs_buff->reserve({max_voc_size_per_gpu}, &tensor);
        buf_bag_.d_value_index_tensors.push_back(tensor);
        hbm_blobs_buff->allocate();
      }
    }

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
