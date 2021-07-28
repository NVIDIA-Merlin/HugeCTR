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

#include "HugeCTR/include/model_oversubscriber/model_oversubscriber_impl.hpp"

#include <sstream>
#include <string>

namespace HugeCTR {

namespace {

std::vector<Embedding_t> get_embedding_type(
    const std::vector<std::shared_ptr<IEmbedding>>& embeddings) {
  std::vector<Embedding_t> embedding_types;
  transform(embeddings.begin(), embeddings.end(), std::back_inserter(embedding_types),
            [](auto& emb) { return emb->get_embedding_type(); });
  return embedding_types;
}

}  // namespace

template <typename TypeKey>
ModelOversubscriberImpl<TypeKey>::ModelOversubscriberImpl(
    bool use_host_ps, std::vector<std::shared_ptr<IEmbedding>>& embeddings,
    const std::vector<SparseEmbeddingHashParams>& embedding_params,
    const std::vector<std::string>& sparse_embedding_files,
    std::shared_ptr<ResourceManager> resource_manager)
    : embeddings_(embeddings),
      ps_manager_(use_host_ps, sparse_embedding_files, get_embedding_type(embeddings),
                  embedding_params, get_max_embedding_size_(), resource_manager) {}

template <typename TypeKey>
void ModelOversubscriberImpl<TypeKey>::load_(std::vector<std::string>& keyset_file_list) {
  try {
    if (keyset_file_list.size() != embeddings_.size()) {
      CK_THROW_(Error_t::WrongInput, "num of keyset_file and num of embeddings don't equal");
    }

    for (size_t i = 0; i < ps_manager_.get_size(); i++) {
      auto ptr_ps = ps_manager_.get_parameter_server(i);
      ptr_ps->load_keyset_from_file(keyset_file_list[i]);

      size_t hit_size = 0;
      ptr_ps->pull(ps_manager_.get_buffer_bag(), hit_size);
      embeddings_[i]->load_parameters(ps_manager_.get_buffer_bag(), hit_size);
    }
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw rt_err;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  }
}

template <typename TypeKey>
void ModelOversubscriberImpl<TypeKey>::dump() {
  try {
    for (size_t i = 0; i < embeddings_.size(); i++) {
      auto ptr_ps = ps_manager_.get_parameter_server(i);

      size_t dump_size = 0;
      embeddings_[i]->dump_parameters(ps_manager_.get_buffer_bag(), &dump_size);
      ptr_ps->push(ps_manager_.get_buffer_bag(), dump_size);
    }
#ifdef ENABLE_MPI
    CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
#endif
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw rt_err;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  }
}

template <typename TypeKey>
void ModelOversubscriberImpl<TypeKey>::update(std::vector<std::string>& keyset_file_list) {
  try {
#ifndef KEY_HIT_RATIO
    MESSAGE_("Preparing embedding table for next pass", false, false);
#endif
    dump();
    for (auto& embedding : embeddings_) {
      embedding->reset();
      embedding->reset_optimizer();
    }
    load_(keyset_file_list);
#ifdef ENABLE_MPI
    CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
#endif
#ifndef KEY_HIT_RATIO
    MESSAGE_(" [DONE]", false, true, false);
#endif
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw rt_err;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  }
}

template <typename TypeKey>
void ModelOversubscriberImpl<TypeKey>::update(std::string& keyset_file) {
  std::vector<std::string> keyset_file_list(embeddings_.size(), keyset_file);
  update(keyset_file_list);
}

template <typename TypeKey>
std::vector<std::pair<std::vector<long long>, std::vector<float>>>
ModelOversubscriberImpl<TypeKey>::get_incremental_model(
    const std::vector<long long>& keys_to_load) {
  std::vector<std::pair<std::vector<long long>, std::vector<float>>> inc_model;
  size_t dump_size{0};

  for (size_t i = 0; i < embeddings_.size(); i++) {
    auto ptr_ps{ps_manager_.get_parameter_server(i)};
    auto key_vec_pair{ptr_ps->pull(keys_to_load)};

    dump_size += key_vec_pair.first.size();
    inc_model.push_back(std::move(key_vec_pair));
  }

#ifdef ENABLE_MPI
  CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
  CK_MPI_THROW_(MPI_Allreduce(&dump_size, &dump_size, 1, MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD));
#endif
  if (dump_size != keys_to_load.size()) {
    // CK_THROW_(Error_t::UnspecificError, "dump_size != key_to_load.size()");
    std::stringstream ss;
    ss << "WARNING: keyset file is insistent with dataset! Only " << dump_size << " out of "
       << keys_to_load.size() << " keys found in parameter server.";
    MESSAGE_(ss.str());
  }
  MESSAGE_("Get updated portion of embedding table [DONE}");

  return inc_model;
}

template class ModelOversubscriberImpl<long long>;
template class ModelOversubscriberImpl<unsigned>;

}  // namespace HugeCTR
