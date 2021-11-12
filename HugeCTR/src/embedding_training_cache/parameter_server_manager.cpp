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

#include "embedding_training_cache/parameter_server_manager.hpp"

#include <algorithm>

namespace HugeCTR {

template <typename TypeKey>
ParameterServerManager<TypeKey>::ParameterServerManager(
    std::vector<TrainPSType_t>& ps_types, std::vector<std::string>& sparse_embedding_files,
    std::vector<Embedding_t> embedding_types,
    std::vector<SparseEmbeddingHashParams>& embedding_params, size_t buffer_size,
    std::shared_ptr<ResourceManager> resource_manager, std::vector<std::string>& local_paths,
    std::vector<HMemCacheConfig>& hmem_cache_configs) {
  try {
    if (sparse_embedding_files.size() == 0)
      CK_THROW_(Error_t::WrongInput,
                "must provide sparse_model_file. \
          if train from scratch, please specify a name to store the trained embedding model");

    if (embedding_params.size() != sparse_embedding_files.size())
      CK_THROW_(Error_t::WrongInput,
                std::string("embedding_params.size() != sparse_embedding_files.size()") + ": " +
                    std::to_string(embedding_params.size()) +
                    " != " + std::to_string(sparse_embedding_files.size()));

    if (embedding_params.size() != ps_types.size()) {
      CK_THROW_(Error_t::WrongInput, "Must specify the PS type for each embedding table");
    }

    {
      bool has_cached{std::any_of(ps_types.begin(), ps_types.end(),
                                  [](auto val) { return val == TrainPSType_t::Cached; })};
      int num_paths(local_paths.size());
      int num_procs(resource_manager->get_num_process());
      if (has_cached && (num_paths != num_procs)) {
        std::stringstream ss;
        ss << "Num of local_paths (" << num_paths << ") != Num of MPI ranks (" << num_procs << ")";
        CK_THROW_(Error_t::WrongInput, ss.str());
      }
    }

    for (size_t i{0}; i < ps_types.size(); i++) {
      switch (ps_types[i]) {
        case TrainPSType_t::Staged: {
          MESSAGE_("Enable HMEM-Based Parameter Server");
          ps_.push_back(std::make_shared<ParameterServer<TypeKey>>(
              ps_types[i], sparse_embedding_files[i], embedding_types[i],
              embedding_params[i].opt_params.optimizer, embedding_params[i].embedding_vec_size,
              resource_manager));
          break;
        }
        case TrainPSType_t::Cached: {
          MESSAGE_("Enable HMemCache-Based Parameter Server");
          if (ps_types.size() != hmem_cache_configs.size()) {
            CK_THROW_(Error_t::WrongInput, "ps_types.size() != hmem_cache_configs.size()");
          }
          for (auto& hmem_cache_config : hmem_cache_configs) {
            hmem_cache_config.block_capacity = buffer_size;
          }
          auto rank_id{resource_manager->get_process_id()};
          ps_.push_back(std::make_shared<ParameterServer<TypeKey>>(
              ps_types[i], sparse_embedding_files[i], embedding_types[i],
              embedding_params[i].opt_params.optimizer, embedding_params[i].embedding_vec_size,
              resource_manager, local_paths[rank_id], hmem_cache_configs[i]));
          break;
        }
        default: {
          CK_THROW_(Error_t::WrongInput, "Unsuppoted PS type");
        }
      }
    }

    auto it{std::max_element(
        embedding_params.begin(), embedding_params.end(), [](auto const& a, auto const& b) {
          return vec_per_line[a.opt_params.optimizer] < vec_per_line[b.opt_params.optimizer];
        })};
    size_t const num_vec_per_key{vec_per_line[it->opt_params.optimizer]};

    it = std::max_element(
        embedding_params.begin(), embedding_params.end(),
        [](auto const& a, auto const& b) { return a.embedding_vec_size < b.embedding_vec_size; });
    size_t const max_vec_size{it->embedding_vec_size};

    it = std::max_element(embedding_params.begin(), embedding_params.end(),
                          [](auto const& a, auto const& b) {
                            return a.max_vocabulary_size_per_gpu < b.max_vocabulary_size_per_gpu;
                          });
    size_t const max_voc_size_per_gpu{it->max_vocabulary_size_per_gpu};

    bool const has_localized_embedding{std::any_of(
        embedding_types.begin(), embedding_types.end(),
        [](auto type) { return type != Embedding_t::DistributedSlotSparseEmbeddingHash; })};

    bool const all_one_hot_embedding{std::all_of(
        embedding_types.begin(), embedding_types.end(),
        [](auto type) { return type == Embedding_t::LocalizedSlotSparseEmbeddingOneHot; })};

    auto host_blobs_buff{GeneralBuffer2<CudaHostAllocator>::create()};
    Tensor2<TypeKey> tensor_keys;
    Tensor2<size_t> tensor_slot_id;
    host_blobs_buff->reserve({buffer_size}, &tensor_keys);
    host_blobs_buff->reserve({buffer_size}, &tensor_slot_id);
    host_blobs_buff->reserve({buffer_size, max_vec_size}, &(buf_bag_.embedding));

    buf_bag_.opt_states.resize(num_vec_per_key - 1);
    for (auto& opt_state : buf_bag_.opt_states) {
      host_blobs_buff->reserve({buffer_size, max_vec_size}, &opt_state);
    }

    buf_bag_.keys = tensor_keys.shrink();
    buf_bag_.slot_id = tensor_slot_id.shrink();

    const size_t local_gpu_count{resource_manager->get_local_gpu_count()};
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
