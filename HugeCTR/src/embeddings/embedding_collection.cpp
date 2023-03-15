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

#include "embedding/hier_model_parallel_embedding.hpp"
#include "embedding/model_parallel_embedding.hpp"

namespace embedding {

EmbeddingCollection::EmbeddingCollection(
    std::shared_ptr<HugeCTR::ResourceManager> resource_manager,
    std::vector<std::shared_ptr<CoreResourceManager>> core,
    const EmbeddingCollectionParam &ebc_param, const EmbeddingCollectionParam &eval_ebc_param,
    const std::vector<EmbeddingTableParam> &emb_table_param_list)
    : ebc_param_(ebc_param), eval_ebc_param_(eval_ebc_param) {
  for (size_t i = 0; i < emb_table_param_list.size(); ++i) {
    embedding_optimizers_.push_back(emb_table_param_list[i].opt_param);
  }

  int num_gpus = resource_manager->get_local_gpu_count();
  embedding_output_attrs.resize(num_gpus);
  wgrad_list_.resize(num_gpus);

  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    HugeCTR::CudaDeviceContext context(core[gpu_id]->get_device_id());

    embedding_tables_.push_back(create_grouped_embedding_tables(resource_manager, core[gpu_id],
                                                                ebc_param_, emb_table_param_list));
    embeddings_.push_back(create_grouped_embeddings(core[gpu_id], ebc_param_));
    eval_embeddings_.push_back(create_grouped_embeddings(core[gpu_id], eval_ebc_param_));

    int num_grouped = static_cast<int>(ebc_param_.grouped_emb_params.size());
    embedding_output_attrs[gpu_id].resize(num_grouped);
    wgrad_list_[gpu_id].resize(num_grouped);
  }

  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    for (size_t grouped_id = 0; grouped_id < wgrad_list_[gpu_id].size(); ++grouped_id) {
      embedding_output_attrs[gpu_id][grouped_id].init(core[gpu_id], ebc_param);
      embedding_output_attrs[gpu_id][grouped_id].update_mutable_data(core[gpu_id], ebc_param);

      auto &wgrad = wgrad_list_[gpu_id][grouped_id];
      if (ebc_param.allreduce_strategy_ == AllreduceStrategy::Dense &&
          ebc_param.grouped_emb_params[grouped_id].table_placement_strategy ==
              TablePlacementStrategy::DataParallel &&
          !ebc_param.table_id_to_vocabulary_size.empty()) {
        AllreduceWgradInitializer{core[gpu_id], ebc_param, grouped_id,
                                  embeddings_[gpu_id][grouped_id]->get_wgrad_attr()}
            .init(wgrad)
            .init_indices()
            .init_data();
      } else {
        WgradInitializer{core[gpu_id], ebc_param, grouped_id,
                         embeddings_[gpu_id][grouped_id]->get_wgrad_attr()}
            .init(wgrad)
            .init_indices()
            .init_data();
      }
    }
  }

  // collective init peer buffer
  if (ebc_param.comm_strategy_ != CommunicationStrategy::Hierarchical) return;
  HCTR_CHECK(resource_manager->all_p2p_enabled());

  gpu_barrier_ = std::make_unique<HugeCTR::GPUBarrier>(
      resource_manager->get_local_gpu_count(), resource_manager->get_local_gpu_device_id_list());

  auto init_hierarchical_embedding =
      [&](std::vector<std::vector<std::unique_ptr<IGroupedEmbeddingOp>>> &embeddings,
          size_t grouped_id) {
        std::vector<ModelCommBuffer *> model_comm_buffers;
        std::vector<IntraModelCommBuffer *> intra_model_comm_buffers;

        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
          HugeCTR::CudaDeviceContext context(core[gpu_id]->get_device_id());
          auto embedding =
              dynamic_cast<HierModelParallelEmbedding *>(embeddings[gpu_id][grouped_id].get());
          model_comm_buffers.push_back(embedding->get_model_comm_buffer());
          intra_model_comm_buffers.push_back(embedding->get_intra_model_comm_buffer());

          embedding->set_gpu_barrier(gpu_barrier_.get());
        }
        collective_init_peer_buffer(core, model_comm_buffers);
        collective_init_peer_buffer(core, intra_model_comm_buffers);
      };

  for (size_t grouped_id = 0; grouped_id < ebc_param.grouped_emb_params.size(); ++grouped_id) {
    if (ebc_param.grouped_emb_params[grouped_id].table_placement_strategy ==
        TablePlacementStrategy::ModelParallel) {
      init_hierarchical_embedding(embeddings_, grouped_id);
      init_hierarchical_embedding(eval_embeddings_, grouped_id);
    }
  }
}

void EmbeddingCollection::forward_per_gpu(bool is_train, int gpu_id,
                                          const HugeCTR::DataDistributor::Result &input,
                                          core23::Tensor &output_buffer, int batch_size) {
  auto &embeddings = is_train ? embeddings_[gpu_id] : eval_embeddings_[gpu_id];

  for (size_t grouped_id = 0; grouped_id < embeddings.size(); ++grouped_id) {
    ILookup *lookup = dynamic_cast<ILookup *>(embedding_tables_[gpu_id][grouped_id].get());
    EmbeddingOutput embedding_output{output_buffer, embedding_output_attrs[gpu_id][grouped_id]};

    embeddings[grouped_id]->forward_per_gpu(input[grouped_id], lookup, embedding_output,
                                            batch_size);
  }
}

void EmbeddingCollection::backward_per_gpu(int gpu_id,
                                           const HugeCTR::DataDistributor::Result &input,
                                           const core23::Tensor &top_grad, int batch_size) {
  for (size_t grouped_id = 0; grouped_id < embeddings_[gpu_id].size(); ++grouped_id) {
    EmbeddingOutput top_grad_buffer{top_grad, embedding_output_attrs[gpu_id][grouped_id]};
    embeddings_[gpu_id][grouped_id]->backward_per_gpu(input[grouped_id], top_grad_buffer,
                                                      wgrad_list_[gpu_id][grouped_id], batch_size);
  }
}

void EmbeddingCollection::update_per_gpu(int gpu_id) {
  for (size_t grouped_id = 0; grouped_id < embedding_tables_[gpu_id].size(); ++grouped_id) {
    auto &wgrad = wgrad_list_[gpu_id][grouped_id];
    embedding_tables_[gpu_id][grouped_id]->update(wgrad.unique_keys, wgrad.num_unique_keys,
                                                  wgrad.table_ids, wgrad.ev_start_indices,
                                                  wgrad.data);
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
