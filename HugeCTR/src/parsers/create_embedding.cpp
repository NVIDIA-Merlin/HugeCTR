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

#include <embeddings/distributed_slot_sparse_embedding_hash.hpp>
#include <embeddings/localized_slot_sparse_embedding_hash.hpp>
#include <embeddings/localized_slot_sparse_embedding_one_hot.hpp>
#include <loss.hpp>
#include <optimizer.hpp>
#include <parser.hpp>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {
template <typename TypeKey, typename TypeFP>
void create_embedding<TypeKey, TypeFP>::operator()(
    std::map<std::string, SparseInput<TypeKey>>& sparse_input_map,
    std::vector<TensorEntry>* train_tensor_entries_list,
    std::vector<TensorEntry>* evaluate_tensor_entries_list,
    std::vector<std::shared_ptr<IEmbedding>>& embeddings, Embedding_t embedding_type,
    const nlohmann::json& config, const std::shared_ptr<ResourceManager>& resource_manager,
    size_t batch_size, size_t batch_size_eval, bool use_mixed_precision, float scaler,
    const nlohmann::json& j_layers) {
#ifdef ENABLE_MPI
  int num_procs = 1, pid = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif

  auto j_optimizer = get_json(config, "optimizer");
  auto embedding_name = get_value_from_json<std::string>(j_layers, "type");

  auto bottom_name = get_value_from_json<std::string>(j_layers, "bottom");
  auto top_name = get_value_from_json<std::string>(j_layers, "top");

  auto j_hparam = get_json(j_layers, "sparse_embedding_hparam");
  size_t max_vocabulary_size_per_gpu = 0;
  if (embedding_type == Embedding_t::DistributedSlotSparseEmbeddingHash) {
    max_vocabulary_size_per_gpu =
        get_value_from_json<size_t>(j_hparam, "max_vocabulary_size_per_gpu");
  } else if (embedding_type == Embedding_t::LocalizedSlotSparseEmbeddingHash) {
    if (has_key_(j_hparam, "max_vocabulary_size_per_gpu")) {
      max_vocabulary_size_per_gpu =
          get_value_from_json<size_t>(j_hparam, "max_vocabulary_size_per_gpu");
    } else if (!has_key_(j_hparam, "slot_size_array")) {
      CK_THROW_(Error_t::WrongInput,
                "No max_vocabulary_size_per_gpu or slot_size_array in: " + embedding_name);
    }
  }
  auto embedding_vec_size = get_value_from_json<size_t>(j_hparam, "embedding_vec_size");
  auto combiner = get_value_from_json<int>(j_hparam, "combiner");

  SparseInput<TypeKey> sparse_input;
  if (!find_item_in_map(sparse_input, bottom_name, sparse_input_map)) {
    CK_THROW_(Error_t::WrongInput, "Cannot find bottom");
  }

  OptParams<TypeFP> embedding_opt_params;
  if (has_key_(j_layers, "optimizer")) {
    embedding_opt_params = get_optimizer_param<TypeFP>()(get_json(j_layers, "optimizer"));
  } else {
    embedding_opt_params = get_optimizer_param<TypeFP>()(j_optimizer);
  }
  embedding_opt_params.scaler = scaler;

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

      std::vector<size_t> slot_size_array;
      if (has_key_(j_hparam, "slot_size_array")) {
        auto slots = get_json(j_hparam, "slot_size_array");
        assert(slots.is_array());
        for (auto slot : slots) {
          slot_size_array.emplace_back(slot.get<size_t>());
        }
      }

      const SparseEmbeddingHashParams<TypeFP> embedding_params = {
          batch_size,
          batch_size_eval,
          max_vocabulary_size_per_gpu,
          slot_size_array,
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
      std::vector<size_t> slot_size_array;
      auto slots = get_json(j_hparam, "slot_size_array");
      assert(slots.is_array());
      for (auto slot : slots) {
        slot_size_array.emplace_back(slot.get<size_t>());
      }

      const SparseEmbeddingHashParams<TypeFP> embedding_params = {
          batch_size,
          batch_size_eval,
          0,
          slot_size_array,
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

template struct create_embedding<long long, float>;
template struct create_embedding<long long, __half>;
template struct create_embedding<unsigned int, float>;
template struct create_embedding<unsigned int, __half>;

}  // namespace HugeCTR
