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

#include "HugeCTR/include/model_oversubscriber/model_oversubscriber_impl.hpp"

namespace HugeCTR {

template <typename TypeHashKey, typename TypeEmbeddingComp>
ModelOversubscriberImpl<TypeHashKey, TypeEmbeddingComp>::ModelOversubscriberImpl(
    std::vector<std::shared_ptr<IEmbedding>>& embeddings,
    const std::vector<SparseEmbeddingHashParams<TypeEmbeddingComp>>& embedding_params,
    const SolverParser& solver_config, const std::string& temp_embedding_dir)
    : embeddings_(embeddings),
      ps_manager_(embedding_params, solver_config, temp_embedding_dir, get_max_embedding_size_()) {}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void ModelOversubscriberImpl<TypeHashKey, TypeEmbeddingComp>::load_(
    std::vector<std::string>& keyset_file_list) {
  try {
    if (keyset_file_list.size() != embeddings_.size()) {
      CK_THROW_(Error_t::WrongInput, "num of keyset_file and num of embeddings don't equal");
    }

    for (int i = 0; i < static_cast<int>(ps_manager_.get_size()); i++) {
      ps_manager_.get_parameter_server(i)->load_keyset_from_file(keyset_file_list[i]);
    }

    for (int i = 0; i < static_cast<int>(ps_manager_.get_size()); i++) {
      auto ptr_ps = ps_manager_.get_parameter_server(i);

      size_t hit_size = 0;
      ptr_ps->load_param_from_embedding_file(ps_manager_.get_embedding_ptr(),
                                             ps_manager_.get_keyset_ptr(), &hit_size);

      embeddings_[i]->load_parameters(ps_manager_.get_keyset_tensor(),
                                      ps_manager_.get_embedding_tensor(), hit_size);
    }
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw rt_err;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void ModelOversubscriberImpl<TypeHashKey, TypeEmbeddingComp>::store(
    std::vector<std::string> snapshot_file_list) {
  try {
    if (snapshot_file_list.size() && snapshot_file_list.size() != embeddings_.size()) {
      CK_THROW_(Error_t::WrongInput, "num of snapshot_file and num of embeddings don't equal");
    }

    for (int i = 0; i < static_cast<int>(embeddings_.size()); i++) {
      size_t dump_size = 0;
      embeddings_[i]->dump_parameters(ps_manager_.get_keyset_tensor(),
                                      ps_manager_.get_embedding_tensor(), &dump_size);

      auto ptr_ps = ps_manager_.get_parameter_server(i);
      ptr_ps->dump_param_to_embedding_file(ps_manager_.get_embedding_ptr(),
                                           ps_manager_.get_keyset_ptr(), dump_size);

      if (!snapshot_file_list.size()) {
        continue;
      }
      ptr_ps->dump_to_snapshot(snapshot_file_list[i]);
    }
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw rt_err;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void ModelOversubscriberImpl<TypeHashKey, TypeEmbeddingComp>::update(
    std::vector<std::string>& keyset_file_list) {
  try {
    store();
    for (auto& one_embedding : embeddings_) {
      one_embedding->reset();
    }
    load_(keyset_file_list);
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw rt_err;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void ModelOversubscriberImpl<TypeHashKey, TypeEmbeddingComp>::update(
  std::string& keyset_file) {
  try {
    std::vector<std::string> keyset_file_list(embeddings_.size(), keyset_file);
    update(keyset_file_list);
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw rt_err;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  }
}

template class ModelOversubscriberImpl<long long, __half>;
template class ModelOversubscriberImpl<long long, float>;
template class ModelOversubscriberImpl<unsigned, __half>;
template class ModelOversubscriberImpl<unsigned, float>;

}  // namespace HugeCTR
