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

#include <embeddings/distributed_slot_sparse_embedding_hash.hpp>
#include <embeddings/hybrid_sparse_embedding.hpp>
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
    size_t batch_size, size_t batch_size_eval, std::shared_ptr<ExchangeWgrad>& exchange_wgrad,
    bool use_mixed_precision, float scaler, const nlohmann::json& j_layers, bool use_cuda_graph,
    bool grouped_all_reduce) {
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
  if (!has_key_(j_hparam, "workspace_size_per_gpu_in_mb") &&
      !has_key_(j_hparam, "slot_size_array")) {
    CK_THROW_(Error_t::WrongInput, "need workspace_size_per_gpu_in_mb or slot_size_array");
  }
  size_t workspace_size_per_gpu_in_mb =
      get_value_from_json_soft<size_t>(j_hparam, "workspace_size_per_gpu_in_mb", 0);
  auto embedding_vec_size = get_value_from_json<size_t>(j_hparam, "embedding_vec_size");

  size_t max_vocabulary_size_per_gpu =
      (workspace_size_per_gpu_in_mb * 1024 * 1024) / (sizeof(float) * embedding_vec_size);

  auto combiner_str = get_value_from_json<std::string>(j_hparam, "combiner");

  int combiner;
  if (combiner_str == "sum") {
    combiner = 0;
  } else if (combiner_str == "mean") {
    combiner = 1;
  } else {
    CK_THROW_(Error_t::WrongInput, "No such combiner type: " + combiner_str);
  }

  std::vector<size_t> slot_size_array;
  if (has_key_(j_hparam, "slot_size_array")) {
    auto slots = get_json(j_hparam, "slot_size_array");
    assert(slots.is_array());
    for (auto slot : slots) {
      slot_size_array.emplace_back(slot.get<size_t>());
    }
  }

  SparseInput<TypeKey> sparse_input;
  if (!find_item_in_map(sparse_input, bottom_name, sparse_input_map)) {
    CK_THROW_(Error_t::WrongInput, "Cannot find bottom");
  }

  OptParams embedding_opt_params;
  if (has_key_(j_layers, "optimizer")) {
    embedding_opt_params = get_optimizer_param(get_json(j_layers, "optimizer"));
  } else {
    embedding_opt_params = get_optimizer_param(j_optimizer);
  }
  embedding_opt_params.scaler = scaler;

  switch (embedding_type) {
    case Embedding_t::DistributedSlotSparseEmbeddingHash: {
      const SparseEmbeddingHashParams embedding_params = {batch_size,
                                                          batch_size_eval,
                                                          max_vocabulary_size_per_gpu,
                                                          {},
                                                          embedding_vec_size,
                                                          sparse_input.max_feature_num_per_sample,
                                                          sparse_input.slot_num,
                                                          combiner,  // combiner: 0-sum, 1-mean
                                                          embedding_opt_params};

      embeddings.emplace_back(new DistributedSlotSparseEmbeddingHash<TypeKey, TypeFP>(
          sparse_input.train_sparse_tensors, sparse_input.evaluate_sparse_tensors, embedding_params,
          resource_manager));
      break;
    }
    case Embedding_t::LocalizedSlotSparseEmbeddingHash: {
      const SparseEmbeddingHashParams embedding_params = {batch_size,
                                                          batch_size_eval,
                                                          max_vocabulary_size_per_gpu,
                                                          slot_size_array,
                                                          embedding_vec_size,
                                                          sparse_input.max_feature_num_per_sample,
                                                          sparse_input.slot_num,
                                                          combiner,  // combiner: 0-sum, 1-mean
                                                          embedding_opt_params};

      embeddings.emplace_back(new LocalizedSlotSparseEmbeddingHash<TypeKey, TypeFP>(
          sparse_input.train_sparse_tensors, sparse_input.evaluate_sparse_tensors, embedding_params,
          resource_manager));

      break;
    }
    case Embedding_t::LocalizedSlotSparseEmbeddingOneHot: {
      const SparseEmbeddingHashParams embedding_params = {batch_size,
                                                          batch_size_eval,
                                                          0,
                                                          slot_size_array,
                                                          embedding_vec_size,
                                                          sparse_input.max_feature_num_per_sample,
                                                          sparse_input.slot_num,
                                                          combiner,  // combiner: 0-sum, 1-mean
                                                          embedding_opt_params};

      embeddings.emplace_back(new LocalizedSlotSparseEmbeddingOneHot<TypeKey, TypeFP>(
          sparse_input.train_sparse_tensors, sparse_input.evaluate_sparse_tensors, embedding_params,
          resource_manager));

      break;
    }
    case Embedding_t::HybridSparseEmbedding: {
      std::vector<size_t> slot_size_array;
      auto slots = get_json(j_hparam, "slot_size_array");
      assert(slots.is_array());
      for (auto slot : slots) {
        slot_size_array.emplace_back(slot.get<size_t>());
      }
      // FIXME need to access this variable in line 1394 for init_data_reader
      size_t num_iterations_statistics =
          get_value_from_json_soft<size_t>(j_hparam, "num_iterations_statistics", 20);
      auto max_num_frequent_categories =
          get_value_from_json_soft<size_t>(j_hparam, "max_num_frequent_categories", 1);
      auto max_num_infrequent_samples =
          get_value_from_json_soft<int64_t>(j_hparam, "max_num_infrequent_samples", -1);
      double p_dup_max = get_value_from_json_soft<double>(j_hparam, "p_dup_max", 1. / 100);
      double max_all_reduce_bandwidth =
          get_value_from_json_soft<double>(j_hparam, "max_all_reduce_bandwidth", 1.3e11);
      double max_all_to_all_bandwidth =
          get_value_from_json_soft<double>(j_hparam, "max_all_to_all_bandwidth", 1.9e11);
      double efficiency_bandwidth_ratio =
          get_value_from_json_soft<double>(j_hparam, "efficiency_bandwidth_ratio", 1.0);

      std::string communication_type_string;
      if (has_key_(j_hparam, "communication_type")) {
        communication_type_string =
            get_value_from_json<std::string>(j_hparam, "communication_type");
      } else {
        communication_type_string = "IB_NVLink";
      }
      hybrid_embedding::CommunicationType communication_type;
      if (!find_item_in_map(communication_type, communication_type_string,
                            COMMUNICATION_TYPE_MAP)) {
        CK_THROW_(Error_t::WrongInput, "No such communication type: " + communication_type_string);
      }

      std::string hybrid_embedding_type_string;
      if (has_key_(j_hparam, "hybrid_embedding_type")) {
        hybrid_embedding_type_string =
            get_value_from_json<std::string>(j_hparam, "hybrid_embedding_type");
      } else {
        hybrid_embedding_type_string = "Distributed";
      }
      hybrid_embedding::HybridEmbeddingType hybrid_embedding_type;
      if (!find_item_in_map(hybrid_embedding_type, hybrid_embedding_type_string,
                            HYBRID_EMBEDDING_TYPE_MAP)) {
        CK_THROW_(Error_t::WrongInput,
                  "No such hybrid embedding type: " + hybrid_embedding_type_string);
      }

      auto& embed_wgrad_buff =
          (grouped_all_reduce)
              ? std::dynamic_pointer_cast<GroupedExchangeWgrad<TypeFP>>(exchange_wgrad)
                    ->get_embed_wgrad_buffs()
              : std::dynamic_pointer_cast<NetworkExchangeWgrad<TypeFP>>(exchange_wgrad)
                    ->get_embed_wgrad_buffs();

      auto j_solver = get_json(config, "solver");
      bool graph_mode = get_value_from_json_soft<bool>(j_solver, "holistic_cuda_graph", false);

      const HybridSparseEmbeddingParams embedding_params = {
          batch_size,
          batch_size_eval,
          num_iterations_statistics,                                            // TBD
          max_num_frequent_categories * std::max(batch_size, batch_size_eval),  // TBD
          max_num_infrequent_samples,                                           // TBD
          p_dup_max,
          embedding_vec_size,
          sparse_input.slot_num,
          slot_size_array,
          communication_type,
          max_all_reduce_bandwidth,
          max_all_to_all_bandwidth,  // TBD
          efficiency_bandwidth_ratio,
          false,
          false,
          hybrid_embedding_type,
          embedding_opt_params};
      embeddings.emplace_back(new HybridSparseEmbedding<TypeKey, TypeFP>(
          sparse_input.train_sparse_tensors, sparse_input.evaluate_sparse_tensors, embedding_params,
          embed_wgrad_buff, get_gpu_learning_rate_schedulers(config, resource_manager), graph_mode,
          resource_manager, false, false));
      break;
    }
    default:
      CK_THROW_(Error_t::UnspecificError, "create embedding with no specified embedding type.");
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
