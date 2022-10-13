#include "embeddings/embedding_collection.hpp"

namespace embedding {
EmbeddingCollection::EmbeddingCollection(
    std::shared_ptr<HugeCTR::ResourceManager> resource_manager,
    std::vector<std::shared_ptr<CoreResourceManager>> core,
    const EmbeddingCollectionParam &ebc_param, const EmbeddingCollectionParam &eval_ebc_param,
    const std::vector<EmbeddingTableParam> &emb_table_param_list)
    : ebc_param_(ebc_param), eval_ebc_param_(eval_ebc_param) {
  int num_gpus = resource_manager->get_local_gpu_count();

  unique_key_list_.resize(num_gpus);
  num_unique_key_list_.resize(num_gpus);
  num_unique_key_per_table_offset_list_.resize(num_gpus);
  num_table_offset_list_.resize(num_gpus);
  wgrad_list_.resize(num_gpus);
  wgrad_idx_offset_list_.resize(num_gpus);
  table_id_list_list_.resize(num_gpus);

  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    HugeCTR::CudaDeviceContext context(core[gpu_id]->get_device_id());

    preprocess_inputs_.push_back(std::make_unique<PreprocessInput>(core[gpu_id], ebc_param));
    embedding_tables_.push_back(create_grouped_embedding_tables(resource_manager, core[gpu_id],
                                                                ebc_param, emb_table_param_list));
    embeddings_.push_back(create_grouped_embeddings(core[gpu_id], ebc_param));
    eval_embeddings_.push_back(create_grouped_embeddings(core[gpu_id], eval_ebc_param));

    int num_grouped = static_cast<int>(ebc_param.grouped_emb_params.size());
    unique_key_list_[gpu_id].resize(num_grouped);
    num_unique_key_list_[gpu_id].resize(num_grouped);
    num_unique_key_per_table_offset_list_[gpu_id].resize(num_grouped);
    num_table_offset_list_[gpu_id].resize(num_grouped);
    wgrad_list_[gpu_id].resize(num_grouped);
    wgrad_idx_offset_list_[gpu_id].resize(num_grouped);
    table_id_list_list_[gpu_id].resize(num_grouped);
  }
}

void EmbeddingCollection::forward_per_gpu(bool is_train, int gpu_id, const Tensor &key,
                                          const Tensor &bucket_range, size_t num_keys,
                                          Tensor &output_buffer) {
  int batch_size = (bucket_range.get_num_elements() - 1) / ebc_param_.num_lookup;
  Tensor feature_major_key, feature_major_bucket_range;
  preprocess_inputs_[gpu_id]->compute(key, bucket_range, &feature_major_key,
                                      &feature_major_bucket_range, batch_size);
  if (is_train) {
    for (size_t emb_id = 0; emb_id < embeddings_[gpu_id].size(); ++emb_id) {
      ILookup *lookup = dynamic_cast<ILookup *>(embedding_tables_[gpu_id][emb_id].get());
      embeddings_[gpu_id][emb_id]->forward_per_gpu(feature_major_key, feature_major_bucket_range,
                                                   num_keys, lookup, output_buffer, batch_size);
    }
  } else {
    for (size_t emb_id = 0; emb_id < embeddings_[gpu_id].size(); ++emb_id) {
      ILookup *lookup = dynamic_cast<ILookup *>(embedding_tables_[gpu_id][emb_id].get());
      eval_embeddings_[gpu_id][emb_id]->forward_per_gpu(feature_major_key,
                                                        feature_major_bucket_range, num_keys,
                                                        lookup, output_buffer, batch_size);
    }
  }
}

void EmbeddingCollection::backward_per_gpu(int gpu_id, const Tensor &top_grad, bool allreduce) {
  for (size_t grouped_id = 0; grouped_id < embeddings_[gpu_id].size(); ++grouped_id) {
    embeddings_[gpu_id][grouped_id]->backward_per_gpu(
        top_grad, allreduce, &unique_key_list_[gpu_id][grouped_id],
        &num_unique_key_list_[gpu_id][grouped_id],
        &num_unique_key_per_table_offset_list_[gpu_id][grouped_id],
        &num_table_offset_list_[gpu_id][grouped_id], &table_id_list_list_[gpu_id][grouped_id],
        &wgrad_list_[gpu_id][grouped_id], &wgrad_idx_offset_list_[gpu_id][grouped_id]);
  }
}

void EmbeddingCollection::update_per_gpu(int gpu_id) {
  for (size_t grouped_id = 0; grouped_id < embedding_tables_[gpu_id].size(); ++grouped_id) {
    embedding_tables_[gpu_id][grouped_id]->update(
        unique_key_list_[gpu_id][grouped_id], num_unique_key_list_[gpu_id][grouped_id],
        num_unique_key_per_table_offset_list_[gpu_id][grouped_id],
        num_table_offset_list_[gpu_id][grouped_id], table_id_list_list_[gpu_id][grouped_id],
        wgrad_list_[gpu_id][grouped_id], wgrad_idx_offset_list_[gpu_id][grouped_id]);
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