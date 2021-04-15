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

#include <HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp>
#include <HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp>
#include <HugeCTR/include/embeddings/localized_slot_sparse_embedding_one_hot.hpp>
#include <loss.hpp>
#include <optimizer.hpp>
#include <HugeCTR/pybind/model.hpp>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {

SparseEmbedding get_sparse_embedding_from_json(const nlohmann::json& j_sparse_embedding) {
  Embedding_t embedding_type;
  std::vector<size_t> slot_size_array;
  size_t max_vocabulary_size_per_gpu = 0;
  auto embedding_type_name = get_value_from_json<std::string>(j_sparse_embedding, "type");
  if (!find_item_in_map(embedding_type, embedding_type_name, EMBEDDING_TYPE_MAP)) {
    CK_THROW_(Error_t::WrongInput, "No such embedding type: " + embedding_type_name); 
  }
  auto j_hparam = get_json(j_sparse_embedding, "sparse_embedding_hparam");
  if (embedding_type == Embedding_t::DistributedSlotSparseEmbeddingHash) {
    max_vocabulary_size_per_gpu =
        get_value_from_json<size_t>(j_hparam, "max_vocabulary_size_per_gpu");
  } else {
    if (has_key_(j_hparam, "max_vocabulary_size_per_gpu")) {
      max_vocabulary_size_per_gpu =
          get_value_from_json<size_t>(j_hparam, "max_vocabulary_size_per_gpu");
    } else if (has_key_(j_hparam, "slot_size_array")) {
      auto slots = get_json(j_hparam, "slot_size_array");
      assert(slots.is_array());
      for (auto slot : slots) {
        slot_size_array.emplace_back(slot.get<size_t>());
      }
    } else {
      CK_THROW_(Error_t::WrongInput,
                "No max_vocabulary_size_per_gpu or slot_size_array in: " + embedding_type_name);
    }
  }
  auto bottom_name = get_value_from_json<std::string>(j_sparse_embedding, "bottom");
  auto top_name = get_value_from_json<std::string>(j_sparse_embedding, "top");
  auto embedding_vec_size = get_value_from_json<size_t>(j_hparam, "embedding_vec_size");
  auto combiner = get_value_from_json<int>(j_hparam, "combiner");
  std::shared_ptr<OptParamsPy> embedding_opt_params(new OptParamsPy());
  auto j_optimizer = get_json(j_sparse_embedding, "optimizer");
  auto optimizer_type_name = get_value_from_json<std::string>(j_optimizer, "type");
  auto update_type_name = get_value_from_json<std::string>(j_optimizer, "update_type");
  if (!find_item_in_map(embedding_opt_params->optimizer, optimizer_type_name, OPTIMIZER_TYPE_MAP)) {
    CK_THROW_(Error_t::WrongInput, "No such optimizer: " + optimizer_type_name);
  }
  if (!find_item_in_map(embedding_opt_params->update_type, update_type_name, UPDATE_TYPE_MAP)) {
    CK_THROW_(Error_t::WrongInput, "No such update type: " + update_type_name);
  }
  OptHyperParamsPy hyperparams;
  switch (embedding_opt_params->optimizer) {
    case Optimizer_t::Adam: {
      auto j_optimizer_hparam = get_json(j_optimizer, "adam_hparam");
      auto beta1 = get_value_from_json<float>(j_optimizer_hparam, "beta1");
      auto beta2 = get_value_from_json<float>(j_optimizer_hparam, "beta2");
      auto epsilon = get_value_from_json<float>(j_optimizer_hparam, "epsilon");
      hyperparams.adam.beta1 = beta1;
      hyperparams.adam.beta2 = beta2;
      hyperparams.adam.epsilon = epsilon;
      embedding_opt_params->hyperparams = hyperparams;
      break;
    }
    case Optimizer_t::MomentumSGD: {
      auto j_optimizer_hparam = get_json(j_optimizer, "momentum_sgd_hparam");
      auto factor = get_value_from_json<float>(j_optimizer_hparam, "momentum_factor");
      hyperparams.momentum.factor = factor;
      embedding_opt_params->hyperparams = hyperparams;
      break;
    }
    case Optimizer_t::Nesterov: {
      auto j_optimizer_hparam = get_json(j_optimizer, "nesterov_hparam");
      auto mu = get_value_from_json<float>(j_optimizer_hparam, "momentum_factor");
      hyperparams.nesterov.mu = mu;
      embedding_opt_params->hyperparams = hyperparams;
      break;
    }
    case Optimizer_t::SGD: {
      auto j_optimizer_hparam = get_json(j_optimizer, "sgd_hparam");
      auto atomic_update =  get_value_from_json<bool>(j_optimizer_hparam, "atomic_update");
      hyperparams.sgd.atomic_update = atomic_update;
      embedding_opt_params->hyperparams = hyperparams;
      break;
    }
    default: {
      assert(!"Error: no such optimizer && should never get here!");
    }
  }
  SparseEmbedding sparse_embedding = SparseEmbedding(embedding_type, max_vocabulary_size_per_gpu,
                                                    embedding_vec_size, combiner, top_name,
                                                    bottom_name, slot_size_array, embedding_opt_params);
  return sparse_embedding;
}

template <typename TypeKey, typename TypeFP>
void add_sparse_embedding(SparseEmbedding& sparse_embedding,
            std::map<std::string, SparseInput<TypeKey>>& sparse_input_map,
            std::vector<std::vector<TensorEntry>>& train_tensor_entries_list,
            std::vector<std::vector<TensorEntry>>& evaluate_tensor_entries_list,
            std::vector<std::shared_ptr<IEmbedding>>& embeddings,
            const std::shared_ptr<ResourceManager>& resource_manager,
            size_t batch_size, size_t batch_size_eval,
            OptParams<TypeFP>& embedding_opt_params) {
#ifdef ENABLE_MPI
  int num_procs = 1, pid = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif

  Embedding_t embedding_type = sparse_embedding.embedding_type;
  std::string bottom_name = sparse_embedding.bottom_name;
  std::string top_name = sparse_embedding.sparse_embedding_name;
  size_t max_vocabulary_size_per_gpu = sparse_embedding.max_vocabulary_size_per_gpu;
  size_t embedding_vec_size = sparse_embedding.embedding_vec_size;
  int combiner = sparse_embedding.combiner;

  SparseInput<TypeKey> sparse_input;
  if (!find_item_in_map(sparse_input, bottom_name, sparse_input_map)) {
    CK_THROW_(Error_t::WrongInput, "Cannot find bottom");
  }
  //embedding_opt_params.scaler = scaler;
  switch (embedding_type) {
    case Embedding_t::DistributedSlotSparseEmbeddingHash: {
      const SparseEmbeddingHashParams<TypeFP> embedding_params = {
          batch_size,
          batch_size_eval,
          max_vocabulary_size_per_gpu,
          {},
          embedding_vec_size,
          sparse_input.max_feature_num_per_sample,
          sparse_input.slot_num,
          combiner,  // combiner: 0-sum, 1-mean
          embedding_opt_params};
      embeddings.emplace_back(new DistributedSlotSparseEmbeddingHash<TypeKey, TypeFP>(
          sparse_input.train_row_offsets, sparse_input.train_values, sparse_input.train_nnz,
          sparse_input.evaluate_row_offsets, sparse_input.evaluate_values,
          sparse_input.evaluate_nnz, embedding_params, resource_manager));
      break;
    }
    case Embedding_t::LocalizedSlotSparseEmbeddingHash: {
      const SparseEmbeddingHashParams<TypeFP> embedding_params = {
          batch_size,
          batch_size_eval,
          max_vocabulary_size_per_gpu,
          sparse_embedding.slot_size_array,
          embedding_vec_size,
          sparse_input.max_feature_num_per_sample,
          sparse_input.slot_num,
          combiner,  // combiner: 0-sum, 1-mean
          embedding_opt_params};
      embeddings.emplace_back(new LocalizedSlotSparseEmbeddingHash<TypeKey, TypeFP>(
          sparse_input.train_row_offsets, sparse_input.train_values, sparse_input.train_nnz,
          sparse_input.evaluate_row_offsets, sparse_input.evaluate_values,
          sparse_input.evaluate_nnz, embedding_params, resource_manager));
      break;
    }
    case Embedding_t::LocalizedSlotSparseEmbeddingOneHot: {
      const SparseEmbeddingHashParams<TypeFP> embedding_params = {
          batch_size,
          batch_size_eval,
          0,
          sparse_embedding.slot_size_array,
          embedding_vec_size,
          sparse_input.max_feature_num_per_sample,
          sparse_input.slot_num,
          combiner,  // combiner: 0-sum, 1-mean
          embedding_opt_params};
      embeddings.emplace_back(new LocalizedSlotSparseEmbeddingOneHot<TypeKey, TypeFP>(
          sparse_input.train_row_offsets, sparse_input.train_values, sparse_input.train_nnz,
          sparse_input.evaluate_row_offsets, sparse_input.evaluate_values,
          sparse_input.evaluate_nnz, embedding_params, resource_manager));
      break;
    }
  }  // switch

  for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
    train_tensor_entries_list[i].push_back(
        {top_name, (embeddings.back()->get_train_output_tensors())[i]});
    evaluate_tensor_entries_list[i].push_back(
        {top_name, (embeddings.back()->get_evaluate_output_tensors())[i]});
  }            
}

template void add_sparse_embedding<long long, float>(SparseEmbedding&,
            std::map<std::string, SparseInput<long long>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::vector<std::shared_ptr<IEmbedding>>&,
            const std::shared_ptr<ResourceManager>&,
            size_t, size_t, OptParams<float>&);
template void add_sparse_embedding<long long, __half>(SparseEmbedding&,
            std::map<std::string, SparseInput<long long>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::vector<std::shared_ptr<IEmbedding>>&,
            const std::shared_ptr<ResourceManager>&,
            size_t, size_t, OptParams<__half>&);
template void add_sparse_embedding<unsigned int, float>(SparseEmbedding&,
            std::map<std::string, SparseInput<unsigned int>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::vector<std::shared_ptr<IEmbedding>>&,
            const std::shared_ptr<ResourceManager>&,
            size_t, size_t, OptParams<float>&);
template void add_sparse_embedding<unsigned int, __half>(SparseEmbedding&,
            std::map<std::string, SparseInput<unsigned int>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::vector<std::shared_ptr<IEmbedding>>&,
            const std::shared_ptr<ResourceManager>&,
            size_t, size_t, OptParams<__half>&);
} // namespace HugeCTR
