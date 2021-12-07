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

#include <HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp>
#include <HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp>
#include <HugeCTR/include/embeddings/localized_slot_sparse_embedding_one_hot.hpp>
#include <HugeCTR/pybind/model.hpp>
#include <embeddings/hybrid_sparse_embedding.hpp>
#include <loss.hpp>
#include <optimizer.hpp>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {

SparseEmbedding get_sparse_embedding_from_json(const nlohmann::json& j_sparse_embedding) {
  auto bottom_name = get_value_from_json<std::string>(j_sparse_embedding, "bottom");
  auto top_name = get_value_from_json<std::string>(j_sparse_embedding, "top");
  auto embedding_type_name = get_value_from_json<std::string>(j_sparse_embedding, "type");
  Embedding_t embedding_type;
  if (!find_item_in_map(embedding_type, embedding_type_name, EMBEDDING_TYPE_MAP)) {
    CK_THROW_(Error_t::WrongInput, "No such embedding type: " + embedding_type_name);
  }
  auto j_hparam = get_json(j_sparse_embedding, "sparse_embedding_hparam");

  if (!has_key_(j_hparam, "workspace_size_per_gpu_in_mb") &&
      !has_key_(j_hparam, "slot_size_array")) {
    CK_THROW_(Error_t::WrongInput, "need workspace_size_per_gpu_in_mb or slot_size_array");
  }
  size_t workspace_size_per_gpu_in_mb =
      get_value_from_json_soft<size_t>(j_hparam, "workspace_size_per_gpu_in_mb", 0);

  size_t embedding_vec_size = get_value_from_json<size_t>(j_hparam, "embedding_vec_size");

  auto combiner_str = get_value_from_json<std::string>(j_hparam, "combiner");

  std::vector<size_t> slot_size_array;
  if (has_key_(j_hparam, "slot_size_array")) {
    auto slots = get_json(j_hparam, "slot_size_array");
    assert(slots.is_array());
    for (auto slot : slots) {
      slot_size_array.emplace_back(slot.get<size_t>());
    }
  }

  std::shared_ptr<OptParamsPy> embedding_opt_params(new OptParamsPy());
  if (has_key_(j_sparse_embedding, "optimizer")) {
    auto j_optimizer = get_json(j_sparse_embedding, "optimizer");
    auto optimizer_type_name = get_value_from_json<std::string>(j_optimizer, "type");
    auto update_type_name = get_value_from_json<std::string>(j_optimizer, "update_type");
    embedding_opt_params->initialized = true;
    if (!find_item_in_map(embedding_opt_params->optimizer, optimizer_type_name,
                          OPTIMIZER_TYPE_MAP)) {
      CK_THROW_(Error_t::WrongInput, "No such optimizer: " + optimizer_type_name);
    }
    if (!find_item_in_map(embedding_opt_params->update_type, update_type_name, UPDATE_TYPE_MAP)) {
      CK_THROW_(Error_t::WrongInput, "No such update type: " + update_type_name);
    }
    OptHyperParams hyperparams;
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
      case Optimizer_t::AdaGrad: {
        auto j_optimizer_hparam = get_json(j_optimizer, "adagrad_hparam");
        auto initial_accu_value =
            get_value_from_json<float>(j_optimizer_hparam, "initial_accu_value");
        auto epsilon = get_value_from_json<float>(j_optimizer_hparam, "epsilon");
        hyperparams.adagrad.initial_accu_value = initial_accu_value;
        hyperparams.adagrad.epsilon = epsilon;
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
        auto atomic_update = get_value_from_json<bool>(j_optimizer_hparam, "atomic_update");
        hyperparams.sgd.atomic_update = atomic_update;
        embedding_opt_params->hyperparams = hyperparams;
        break;
      }
      default: {
        assert(!"Error: no such optimizer && should never get here!");
      }
    }
  }
  HybridEmbeddingParam hybrid_embedding_param;
  hybrid_embedding_param.max_num_frequent_categories =
      get_value_from_json_soft<size_t>(j_hparam, "max_num_frequent_categories", 1);
  hybrid_embedding_param.max_num_infrequent_samples =
      get_value_from_json_soft<int64_t>(j_hparam, "max_num_infrequent_samples", -1);
  hybrid_embedding_param.p_dup_max =
      get_value_from_json_soft<double>(j_hparam, "p_dup_max", 1. / 100);
  hybrid_embedding_param.max_all_reduce_bandwidth =
      get_value_from_json_soft<double>(j_hparam, "max_all_reduce_bandwidth", 1.3e11);
  hybrid_embedding_param.max_all_to_all_bandwidth =
      get_value_from_json_soft<double>(j_hparam, "max_all_to_all_bandwidth", 1.9e11);
  hybrid_embedding_param.efficiency_bandwidth_ratio =
      get_value_from_json_soft<double>(j_hparam, "efficiency_bandwidth_ratio", 1.0);
  hybrid_embedding_param.use_train_precompute_indices =
      get_value_from_json_soft<bool>(j_hparam, "use_train_precompute_indices", false);
  hybrid_embedding_param.use_eval_precompute_indices =
      get_value_from_json_soft<bool>(j_hparam, "use_eval_precompute_indices", false);
  std::string communication_type_string =
      get_value_from_json_soft<std::string>(j_hparam, "communication_type", "IB_NVLink");
  std::string hybrid_embedding_type_string =
      get_value_from_json_soft<std::string>(j_hparam, "hybrid_embedding_type", "Distributed");
  if (!find_item_in_map(hybrid_embedding_param.communication_type, communication_type_string,
                        COMMUNICATION_TYPE_MAP)) {
    CK_THROW_(Error_t::WrongInput, "No such communication type: " + communication_type_string);
  }
  if (!find_item_in_map(hybrid_embedding_param.hybrid_embedding_type, hybrid_embedding_type_string,
                        HYBRID_EMBEDDING_TYPE_MAP)) {
    CK_THROW_(Error_t::WrongInput,
              "No such hybrid embedding type: " + hybrid_embedding_type_string);
  }
  SparseEmbedding sparse_embedding = SparseEmbedding(
      embedding_type, workspace_size_per_gpu_in_mb, embedding_vec_size, combiner_str, top_name,
      bottom_name, slot_size_array, embedding_opt_params, hybrid_embedding_param);
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
                          OptParams& embedding_opt_params,
                          std::shared_ptr<ExchangeWgrad>& exchange_wgrad, bool use_cuda_graph,
                          bool grouped_all_reduce, bool use_holistic_cuda_graph,
                          size_t num_iterations_statistics, GpuLearningRateSchedulers& gpu_lr_sches,
                          bool overlap_ar_a2a) {
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
                                                          sparse_embedding.slot_size_array,
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
                                                          sparse_embedding.slot_size_array,
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
      auto& embed_wgrad_buff =
          (grouped_all_reduce)
              ? std::dynamic_pointer_cast<GroupedExchangeWgrad<TypeFP>>(exchange_wgrad)
                    ->get_embed_wgrad_buffs()
              : std::dynamic_pointer_cast<NetworkExchangeWgrad<TypeFP>>(exchange_wgrad)
                    ->get_embed_wgrad_buffs();

      const HybridSparseEmbeddingParams embedding_params = {
          batch_size,
          batch_size_eval,
          num_iterations_statistics,  // TBD
          sparse_embedding.hybrid_embedding_param.max_num_frequent_categories *
              std::max(batch_size, batch_size_eval),                           // TBD
          sparse_embedding.hybrid_embedding_param.max_num_infrequent_samples,  // TBD
          sparse_embedding.hybrid_embedding_param.p_dup_max,
          embedding_vec_size,
          sparse_input.slot_num,
          sparse_embedding.slot_size_array,
          sparse_embedding.hybrid_embedding_param.communication_type,
          sparse_embedding.hybrid_embedding_param.max_all_reduce_bandwidth,
          sparse_embedding.hybrid_embedding_param.max_all_to_all_bandwidth,  // TBD
          sparse_embedding.hybrid_embedding_param.efficiency_bandwidth_ratio,
          sparse_embedding.hybrid_embedding_param.use_train_precompute_indices,
          sparse_embedding.hybrid_embedding_param.use_eval_precompute_indices,
          sparse_embedding.hybrid_embedding_param.hybrid_embedding_type,
          embedding_opt_params};
      embeddings.emplace_back(new HybridSparseEmbedding<TypeKey, TypeFP>(
          sparse_input.train_sparse_tensors, sparse_input.evaluate_sparse_tensors, embedding_params,
          embed_wgrad_buff, gpu_lr_sches, use_holistic_cuda_graph, resource_manager,
          overlap_ar_a2a));
      break;
    }
    default:
      CK_THROW_(Error_t::UnspecificError, "add_sparse_embedding with no specified embedding type.");
  }  // switch

  for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
    train_tensor_entries_list[i].push_back(
        {top_name, (embeddings.back()->get_train_output_tensors())[i]});
    evaluate_tensor_entries_list[i].push_back(
        {top_name, (embeddings.back()->get_evaluate_output_tensors())[i]});
  }
}

template void add_sparse_embedding<long long, float>(
    SparseEmbedding&, std::map<std::string, SparseInput<long long>>&,
    std::vector<std::vector<TensorEntry>>&, std::vector<std::vector<TensorEntry>>&,
    std::vector<std::shared_ptr<IEmbedding>>&, const std::shared_ptr<ResourceManager>&, size_t,
    size_t, OptParams&, std::shared_ptr<ExchangeWgrad>&, bool, bool, bool, size_t,
    GpuLearningRateSchedulers&, bool);
template void add_sparse_embedding<long long, __half>(
    SparseEmbedding&, std::map<std::string, SparseInput<long long>>&,
    std::vector<std::vector<TensorEntry>>&, std::vector<std::vector<TensorEntry>>&,
    std::vector<std::shared_ptr<IEmbedding>>&, const std::shared_ptr<ResourceManager>&, size_t,
    size_t, OptParams&, std::shared_ptr<ExchangeWgrad>&, bool, bool, bool, size_t,
    GpuLearningRateSchedulers&, bool);
template void add_sparse_embedding<unsigned int, float>(
    SparseEmbedding&, std::map<std::string, SparseInput<unsigned int>>&,
    std::vector<std::vector<TensorEntry>>&, std::vector<std::vector<TensorEntry>>&,
    std::vector<std::shared_ptr<IEmbedding>>&, const std::shared_ptr<ResourceManager>&, size_t,
    size_t, OptParams&, std::shared_ptr<ExchangeWgrad>&, bool, bool, bool, size_t,
    GpuLearningRateSchedulers&, bool);
template void add_sparse_embedding<unsigned int, __half>(
    SparseEmbedding&, std::map<std::string, SparseInput<unsigned int>>&,
    std::vector<std::vector<TensorEntry>>&, std::vector<std::vector<TensorEntry>>&,
    std::vector<std::shared_ptr<IEmbedding>>&, const std::shared_ptr<ResourceManager>&, size_t,
    size_t, OptParams&, std::shared_ptr<ExchangeWgrad>&, bool, bool, bool, size_t,
    GpuLearningRateSchedulers&, bool);
}  // namespace HugeCTR
