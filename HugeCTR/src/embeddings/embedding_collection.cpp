/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <embeddings/embedding_collection.hpp>

#include "embedding/dense_model_parallel_embedding.hpp"
#include "embedding/hier_model_parallel_embedding.hpp"
#include "embedding/model_parallel_embedding.hpp"

namespace HugeCTR {

EmbeddingCollectionConfig split_column_wise_sharding_config(
    const EmbeddingCollectionConfig &user_ebc_config) {
  std::unordered_map<std::string, std::vector<std::string>> user_table_name_to_split_table_name;
  bool has_column_sharding_table = false;
  for (auto &shard : user_ebc_config.shard_strategy_) {
    auto table_placement_strategy = get_table_place_strategy(shard);
    auto table_names_with_column_wise_sharding_factor = get_table_group_strategy(shard);
    for (auto table_tuple : table_names_with_column_wise_sharding_factor) {
      std::string table_name = get_table_name(table_tuple);
      int column_wise_sharding_factor = get_column_wise_sharding_factor(table_tuple);

      if (table_placement_strategy == "dp") {
        HCTR_CHECK_HINT(column_wise_sharding_factor == 1,
                        "dp table does not support column-wise sharding.");
      }

      has_column_sharding_table |= (column_wise_sharding_factor > 1);

      for (int column_wise_sharding_id = 0; column_wise_sharding_id < column_wise_sharding_factor;
           ++column_wise_sharding_id) {
        user_table_name_to_split_table_name[table_name].push_back(
            table_name + "_" + std::to_string(column_wise_sharding_id));
      }
    }
  }

  if (!has_column_sharding_table) {
    return user_ebc_config;
  }

  auto split_ebc_config = user_ebc_config;

  split_ebc_config.emb_table_config_list_.clear();

  for (auto &user_emb_table_config : user_ebc_config.emb_table_config_list_) {
    int column_wise_sharding_factor =
        user_table_name_to_split_table_name[user_emb_table_config.name].size();
    for (size_t column_shard_id = 0; column_shard_id < column_wise_sharding_factor;
         ++column_shard_id) {
      auto split_emb_table_config = user_emb_table_config;
      split_emb_table_config.name =
          user_table_name_to_split_table_name[user_emb_table_config.name][column_shard_id];
      HCTR_CHECK_HINT(split_emb_table_config.table_param.ev_size % column_wise_sharding_factor == 0,
                      "ev_size can not be divided by column_wise_sharding_factor");
      split_emb_table_config.table_param.ev_size /= column_wise_sharding_factor;
      split_ebc_config.emb_table_config_list_.push_back(split_emb_table_config);
    }
  }

  split_ebc_config.dr_lookup_ids_.clear();
  split_ebc_config.lookup_configs_.clear();
  for (int dr_lookup_id = 0; dr_lookup_id < user_ebc_config.lookup_configs_.size();
       ++dr_lookup_id) {
    auto &user_lookup_config = user_ebc_config.lookup_configs_[dr_lookup_id];
    auto user_table_name = user_lookup_config.first;
    int column_wise_sharding_factor = user_table_name_to_split_table_name[user_table_name].size();
    for (size_t column_shard_id = 0; column_shard_id < column_wise_sharding_factor;
         ++column_shard_id) {
      auto split_name = user_table_name_to_split_table_name[user_table_name][column_shard_id];
      auto split_lookup_config = user_lookup_config;
      split_lookup_config.first = split_name;
      HCTR_CHECK_HINT(split_lookup_config.second.ev_size % column_wise_sharding_factor == 0,
                      "ev_size can not be divided by column_wise_sharding_factor");
      split_lookup_config.second.ev_size /= column_wise_sharding_factor;
      split_ebc_config.lookup_configs_.push_back(split_lookup_config);
      split_ebc_config.dr_lookup_ids_.push_back(dr_lookup_id);
    }
  }

  split_ebc_config.shard_strategy_.clear();
  for (auto &user_shard_strategy : user_ebc_config.shard_strategy_) {
    std::vector<TableVariant> split_group_strategy;
    auto user_group_strategy = get_table_group_strategy(user_shard_strategy);
    for (auto &table_tuple : user_group_strategy) {
      std::string user_table_name = get_table_name(table_tuple);
      auto &split_table_names = user_table_name_to_split_table_name[user_table_name];
      std::copy(split_table_names.begin(), split_table_names.end(),
                std::back_inserter(split_group_strategy));
    }
    auto tps = get_table_place_strategy(user_shard_strategy);
    if (tps == "dp") {
      split_ebc_config.shard_strategy_.push_back({"dp", split_group_strategy});
      continue;
    } else if (tps == "mp") {
      split_ebc_config.shard_strategy_.push_back({"mp", split_group_strategy});
    } else {
      HCTR_OWN_THROW(Error_t::IllegalCall, "unreachable.");
    }
  }

  split_ebc_config.shard_matrix_.clear();
  split_ebc_config.shard_matrix_.resize(user_ebc_config.shard_matrix_.size());
  for (auto &[user_table_name, split_table_names] : user_table_name_to_split_table_name) {
    std::vector<int> has_shard_gpu_ids;
    for (int gpu_id = 0; gpu_id < user_ebc_config.shard_matrix_.size(); ++gpu_id) {
      auto &current_gpu_shards = user_ebc_config.shard_matrix_[gpu_id];
      if (std::find(current_gpu_shards.begin(), current_gpu_shards.end(), user_table_name) !=
          current_gpu_shards.end()) {
        has_shard_gpu_ids.push_back(gpu_id);
      }
    }
    int num_shards = static_cast<int>(has_shard_gpu_ids.size());
    int column_wise_sharding_factor = static_cast<int>(split_table_names.size());
    HCTR_CHECK_HINT(num_shards % column_wise_sharding_factor == 0,
                    "column-wise sharding gpu can not be divided.");
    int row_wise_sharding_factor = num_shards / column_wise_sharding_factor;

    for (int column_wise_sharding_id = 0; column_wise_sharding_id < column_wise_sharding_factor;
         ++column_wise_sharding_id) {
      auto split_table_name = split_table_names[column_wise_sharding_id];
      for (int shard_id = 0; shard_id < static_cast<int>(has_shard_gpu_ids.size()); ++shard_id) {
        int gpu_id = has_shard_gpu_ids[shard_id];
        if (shard_id / row_wise_sharding_factor == column_wise_sharding_id) {
          split_ebc_config.shard_matrix_[gpu_id].push_back(split_table_name);
        }
      }
    }
  }

  split_ebc_config.compression_strategy_config_.clear();
  for (auto &[compression_strategy, user_table_names] :
       user_ebc_config.compression_strategy_config_) {
    for (auto &user_table_name : user_table_names) {
      auto &split_table_names = user_table_name_to_split_table_name[user_table_name];
      std::copy(
          split_table_names.begin(), split_table_names.end(),
          std::back_inserter(split_ebc_config.compression_strategy_config_[compression_strategy]));
    }
  }

  return split_ebc_config;
}

}  // namespace HugeCTR

namespace embedding {

EmbeddingCollection::EmbeddingCollection(
    std::shared_ptr<HugeCTR::ResourceManager> resource_manager,
    std::vector<std::shared_ptr<CoreResourceManager>> core,
    const EmbeddingCollectionParam &ebc_param, const EmbeddingCollectionParam &eval_ebc_param,
    const std::vector<EmbeddingTableParam> &emb_table_param_list,
    std::shared_ptr<HugeCTR::ExchangeWgrad> exchange_wgrad)
    : resource_manager_(resource_manager),
      ebc_param_(ebc_param),
      eval_ebc_param_(eval_ebc_param),
      emb_table_param_list_(emb_table_param_list) {
  for (size_t i = 0; i < emb_table_param_list.size(); ++i) {
    embedding_optimizers_.push_back(emb_table_param_list[i].opt_param);
  }
  int num_gpus = resource_manager->get_local_gpu_count();

  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    HugeCTR::CudaDeviceContext context(core[gpu_id]->get_device_id());
    // embedding storage
    embedding_tables_.push_back(create_grouped_embedding_tables(resource_manager, core[gpu_id],
                                                                ebc_param_, emb_table_param_list));
    // embedding ops
    embeddings_.push_back(create_grouped_embeddings(core[gpu_id], ebc_param_));
    eval_embeddings_.push_back(create_grouped_embeddings(core[gpu_id], eval_ebc_param_));
  }

  init_embedding_output_attrs(core);
  init_wgrad(core, exchange_wgrad);
  init_peer_buffer(core);
}

void EmbeddingCollection::init_embedding_output_attrs(
    std::vector<std::shared_ptr<CoreResourceManager>> core) {
  int num_gpus = resource_manager_->get_local_gpu_count();
  embedding_output_attrs_.resize(num_gpus);

  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    int num_grouped = static_cast<int>(ebc_param_.grouped_lookup_params.size());

    embedding_output_attrs_[gpu_id].resize(num_grouped);
    for (size_t grouped_id = 0; grouped_id < num_grouped; ++grouped_id) {
      embedding_output_attrs_[gpu_id][grouped_id].init(core[gpu_id], ebc_param_);
      embedding_output_attrs_[gpu_id][grouped_id].update_mutable_data(core[gpu_id], ebc_param_);
    }
  }
}

void EmbeddingCollection::init_wgrad(std::vector<std::shared_ptr<CoreResourceManager>> core,
                                     std::shared_ptr<HugeCTR::ExchangeWgrad> exchange_wgrad) {
  int num_gpus = resource_manager_->get_local_gpu_count();

  wgrad_list_.resize(num_gpus);
  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    HugeCTR::CudaDeviceContext context(core[gpu_id]->get_device_id());
    int num_grouped = static_cast<int>(ebc_param_.grouped_lookup_params.size());
    wgrad_list_[gpu_id].resize(num_grouped);
    for (size_t grouped_id = 0; grouped_id < num_grouped; ++grouped_id) {
      Wgrad &wgrad = wgrad_list_[gpu_id][grouped_id];
      auto embedding_group_type = ebc_param_.grouped_lookup_params[grouped_id].embedding_group_type;
      // 1. when mp
      if (embedding_group_type == EmbeddingGroupType::DenseModelParallel ||
          embedding_group_type == EmbeddingGroupType::SparseModelParallel) {
        WgradInitializer{core[gpu_id], ebc_param_, grouped_id,
                         embeddings_[gpu_id][grouped_id]->get_wgrad_attr()}
            .init(wgrad)
            .init_indices()
            .init_data();
        continue;
      }
      // 2. init table_id_to_vocabulary_size and check if there is dynamic table
      std::vector<int> table_id_to_vocabulary_size;
      std::transform(emb_table_param_list_.begin(), emb_table_param_list_.end(),
                     std::back_inserter(table_id_to_vocabulary_size),
                     [](const embedding::EmbeddingTableParam &table_param) {
                       return table_param.max_vocabulary_size;
                     });

      std::for_each(table_id_to_vocabulary_size.begin(), table_id_to_vocabulary_size.end(),
                    [](int vocabulary_size) {
                      HCTR_CHECK_HINT(vocabulary_size > 0, "vocabuary_size should > 0.");
                    });

      // 2. dense allreduce can be group or not grouped
      bool grouped = (ebc_param_.allreduce_strategy_ == AllreduceStrategy::GroupDense);
      if (ebc_param_.wgrad_type_.type() == core23::ScalarType::Float) {
        AllreduceWgradInitializer{core[gpu_id], ebc_param_, table_id_to_vocabulary_size, grouped_id,
                                  embeddings_[gpu_id][grouped_id]->get_wgrad_attr()}
            .init(wgrad)
            .init_indices()
            .init_data(grouped, HugeCTR::GetWgradBufferChannel());
      } else if (ebc_param_.wgrad_type_.type() == core23::ScalarType::Half) {
        AllreduceWgradInitializer{core[gpu_id], ebc_param_, table_id_to_vocabulary_size, grouped_id,
                                  embeddings_[gpu_id][grouped_id]->get_wgrad_attr()}
            .init(wgrad)
            .init_indices()
            .init_data(grouped, HugeCTR::GetWgradHalfBufferChannel());
      } else {
        HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput,
                       "Embedding wgrad type set wrong can't support!");
      }
    }
  }
}

void EmbeddingCollection::init_peer_buffer(std::vector<std::shared_ptr<CoreResourceManager>> core) {
  // collective init peer buffer
  if (ebc_param_.comm_strategy_ != CommunicationStrategy::Hierarchical) return;
  HCTR_CHECK(resource_manager_->all_p2p_enabled());
  int num_gpus = resource_manager_->get_local_gpu_count();

  gpu_barrier_ = std::make_unique<HugeCTR::GPUBarrier>(
      resource_manager_->get_local_gpu_count(), resource_manager_->get_local_gpu_device_id_list());

  auto init_hierarchical_embedding =
      [&](std::vector<std::vector<std::unique_ptr<IGroupedEmbeddingOp>>> &embeddings,
          size_t grouped_id) {
        std::vector<ModelCommBuffer *> model_comm_buffers;
        std::vector<IntraModelReductionBuffer *> intra_reduction_buffers;
        std::vector<IntraModelCommBuffer *> intra_model_comm_buffers;

        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
          HugeCTR::CudaDeviceContext context(core[gpu_id]->get_device_id());
          auto embedding =
              dynamic_cast<HierModelParallelEmbedding *>(embeddings[gpu_id][grouped_id].get());
          intra_reduction_buffers.push_back(embedding->get_intra_reduction_buffer());
          intra_model_comm_buffers.push_back(embedding->get_intra_model_comm_buffer());

          embedding->set_gpu_barrier(gpu_barrier_.get());
        }
        collective_init_peer_buffer(core, intra_reduction_buffers);
        collective_init_peer_buffer(core, intra_model_comm_buffers);
      };

  for (size_t grouped_id = 0; grouped_id < ebc_param_.grouped_lookup_params.size(); ++grouped_id) {
    if (ebc_param_.grouped_lookup_params[grouped_id].embedding_group_type ==
        EmbeddingGroupType::SparseModelParallel) {
      init_hierarchical_embedding(embeddings_, grouped_id);
      init_hierarchical_embedding(eval_embeddings_, grouped_id);
    }
  }
}

void EmbeddingCollection::cache_ddl_output(int gpu_id,
                                           const HugeCTR::DataDistributor::Result &input,
                                           HugeCTR::DataDistributor::Result &output,
                                           int batch_size) {
  HugeCTR::CudaDeviceContext context(resource_manager_->get_local_gpu(gpu_id)->get_device_id());
  auto stream = resource_manager_->get_local_gpu(gpu_id)->get_stream();
  HCTR_CHECK(output.size() == input.size());

  for (size_t grouped_id = 0; grouped_id < input.size(); ++grouped_id) {
    auto &dst_result = output[grouped_id];
    auto &src_result = input[grouped_id];

    HCTR_LIB_THROW(cudaMemcpyAsync(dst_result.keys.data(), src_result.keys.data(),
                                   src_result.keys.data_type().size() * src_result.h_num_keys,
                                   cudaMemcpyDeviceToDevice, stream));
    core23::copy_sync(dst_result.num_keys, src_result.num_keys);
    dst_result.h_num_keys = src_result.h_num_keys;

    auto &grouped_lookup_params = ebc_param_.grouped_lookup_params[grouped_id];
    core23::copy_async(dst_result.num_keys_per_bucket, src_result.num_keys_per_bucket, stream);

    if (grouped_lookup_params.embedding_group_type == EmbeddingGroupType::DataParallel ||
        grouped_lookup_params.embedding_group_type == EmbeddingGroupType::SparseModelParallel) {
      core23::copy_async(dst_result.bucket_range, src_result.bucket_range, stream);
    } else if (grouped_lookup_params.embedding_group_type ==
                   EmbeddingGroupType::DenseModelParallel ||
               grouped_lookup_params.embedding_group_type ==
                   EmbeddingGroupType::DenseModelParallelWithReduction) {
      core23::copy_async(dst_result.dense_compression_input.num_keys_per_table_offset,
                         src_result.dense_compression_input.num_keys_per_table_offset, stream);
      core23::copy_async(dst_result.dense_compression_input.table_ids,
                         src_result.dense_compression_input.table_ids, stream);

      auto &dst_compression_input =
          dst_result.dense_compression_input.model_parallel_compression_input;
      auto &src_compression_input =
          src_result.dense_compression_input.model_parallel_compression_input;
      core23::copy_sync(dst_compression_input.h_send_k_per_gpu,
                        src_compression_input.h_send_k_per_gpu);
      core23::copy_sync(dst_compression_input.h_recv_k_per_gpu,
                        src_compression_input.h_recv_k_per_gpu);
      core23::copy_async(dst_compression_input.model_reverse_idx,
                         src_compression_input.model_reverse_idx, stream);
      dst_compression_input.num_model_reverse_idx = src_compression_input.num_model_reverse_idx;
      core23::copy_async(dst_compression_input.network_reverse_idx,
                         src_compression_input.network_reverse_idx, stream);
      dst_compression_input.num_network_reverse_idx = src_compression_input.num_network_reverse_idx;
      core23::copy_async(dst_compression_input.network_dst_bucket_ids,
                         src_compression_input.network_dst_bucket_ids, stream);
    } else {
      HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "not supported embedding_type.");
    }
  }
}

void EmbeddingCollection::forward_per_gpu(Stage stage, bool is_train, int gpu_id,
                                          const HugeCTR::DataDistributor::Result &input,
                                          core23::Tensor &output_buffer, int batch_size) {
  // embedding ops
  auto &embeddings = is_train ? embeddings_[gpu_id] : eval_embeddings_[gpu_id];
  for (size_t grouped_id = 0; grouped_id < embeddings.size(); ++grouped_id) {
    if (!embeddings[grouped_id]->is_valid_stage(stage)) continue;

    ILookup *lookup = dynamic_cast<ILookup *>(get_table(gpu_id, grouped_id));
    EmbeddingOutput embedding_output{output_buffer, embedding_output_attrs_[gpu_id][grouped_id]};

    embeddings[grouped_id]->forward_per_gpu(stage, input[grouped_id], lookup, embedding_output,
                                            batch_size);
  }
}

void EmbeddingCollection::forward_per_gpu(bool is_train, int gpu_id,
                                          const HugeCTR::DataDistributor::Result &input,
                                          core23::Tensor &output_buffer, int batch_size) {
  std::vector<Stage> stages{Stage::DPForward, Stage::DenseMPModelForward,
                            Stage::DenseMPNetworkForward};

  if (ebc_param_.comm_strategy_ == CommunicationStrategy::Uniform) {
    stages.insert(stages.end(), {Stage::MPModelForward, Stage::MPNetworkdForward});
  } else if (ebc_param_.comm_strategy_ == CommunicationStrategy::Hierarchical) {
    stages.insert(stages.end(), {Stage::HierMPModelForward, Stage::HierMPNetworkForward});
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "comm strategy not supported in forward_per_gpu");
  }
  for (auto stage : stages) {
    forward_per_gpu(stage, is_train, gpu_id, input, output_buffer, batch_size);
  }
}

void EmbeddingCollection::backward_per_gpu(Stage stage, int gpu_id,
                                           const HugeCTR::DataDistributor::Result &input,
                                           const core23::Tensor &top_grad, int batch_size) {
  for (size_t grouped_id = 0; grouped_id < embeddings_[gpu_id].size(); ++grouped_id) {
    if (!embeddings_[gpu_id][grouped_id]->is_valid_stage(stage)) continue;

    EmbeddingOutput top_grad_buffer{top_grad, embedding_output_attrs_[gpu_id][grouped_id]};
    embeddings_[gpu_id][grouped_id]->backward_per_gpu(stage, input[grouped_id], top_grad_buffer,
                                                      wgrad_list_[gpu_id][grouped_id], batch_size);
  }
}

void EmbeddingCollection::backward_per_gpu(int gpu_id,
                                           const HugeCTR::DataDistributor::Result &input,
                                           const core23::Tensor &top_grad, int batch_size) {
  std::vector<Stage> stages{Stage::DPBackwardIndexCalculation,
                            Stage::DPLocalReduce,
                            Stage::DPAllreduce,
                            Stage::DenseMPBackwardIndexCalculation,
                            Stage::DenseMPNetworkBackward,
                            Stage::DenseMPLocalReduce};
  if (ebc_param_.comm_strategy_ == CommunicationStrategy::Uniform) {
    stages.insert(stages.end(), {Stage::MPBackwardIndexCalculation, Stage::MPNetworkBackward,
                                 Stage::MPLocalReduce});
  } else if (ebc_param_.comm_strategy_ == CommunicationStrategy::Hierarchical) {
    stages.insert(stages.end(), {Stage::HierMPBackwardIndexCalculation,
                                 Stage::HierMPNetworkBackward, Stage::HierMPLocalReduce});
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                   "comm strategy not supported in backward_per_gpu");
  }

  for (auto stage : stages) {
    backward_per_gpu(stage, gpu_id, input, top_grad, batch_size);
  }
}

void EmbeddingCollection::update_per_gpu(int gpu_id, EmbeddingGroupType embedding_group_type) {
  for (size_t grouped_id = 0; grouped_id < embeddings_[gpu_id].size(); ++grouped_id) {
    if (ebc_param_.grouped_lookup_params[grouped_id].embedding_group_type != embedding_group_type)
      continue;
    auto &wgrad = wgrad_list_[gpu_id][grouped_id];

    auto table = get_table(gpu_id, grouped_id);
    table->update(wgrad.unique_keys, wgrad.num_unique_keys, wgrad.table_ids, wgrad.ev_start_indices,
                  wgrad.data);
  }
}

void EmbeddingCollection::update_per_gpu(int gpu_id) {
  for (auto tps : {embedding::EmbeddingGroupType::DataParallel,
                   embedding::EmbeddingGroupType::SparseModelParallel,
                   embedding::EmbeddingGroupType::DenseModelParallel}) {
    update_per_gpu(gpu_id, tps);
  }
}

void EmbeddingCollection::set_learning_rate(float lr) {
  for (auto &table_list : embedding_tables_) {
    for (auto &t : table_list) {
      t->set_learning_rate(lr);
    }
  }
}

}  // namespace embedding
