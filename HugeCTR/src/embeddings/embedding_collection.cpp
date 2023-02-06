#include "embeddings/embedding_collection.hpp"

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

      auto &wgrad = wgrad_list_[gpu_id][grouped_id];
      if (ebc_param.indices_only_ &&
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
}

void EmbeddingCollection::forward_per_gpu(bool is_train, int gpu_id,
                                          const HugeCTR::DataDistributor::Result &input,
                                          Tensor &output_buffer, int batch_size) {
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
                                           const Tensor &top_grad, int batch_size) {
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
