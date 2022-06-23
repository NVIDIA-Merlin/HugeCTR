/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include "embedding_collection.hpp"

#include "data_parallel_embedding.hpp"
#include "localized_embedding.hpp"
namespace embedding {

EmbeddingCollectionForward::EmbeddingCollectionForward(
    std::shared_ptr<CoreResourceManager> core,
    const EmbeddingCollectionParam &embedding_collection_param,
    const std::vector<EmbeddingShardingParam> &embedding_sharding_params)
    : num_embedding_(embedding_collection_param.num_embedding),
      preprocess_input_(core, embedding_collection_param),
      process_output_(core, embedding_collection_param) {
  EmbeddingCollectionParam flatten_ebc_param = embedding_collection_param;
  std::vector<std::vector<EmbeddingShardingParam>> flatten_ebs_param{embedding_sharding_params};

  flatten_concat_embedding(&flatten_ebc_param, &flatten_ebs_param);
  global_embedding_data_ = GlobalEmbeddingData(core, flatten_ebc_param);

  for (auto &embedding_sharding_param : flatten_ebs_param[0]) {
    switch (embedding_sharding_param.table_placement_strategy) {
      case TablePlacementStrategy::DataParallel:
        embeddings_.push_back(std::make_unique<UniformDPEmbeddingForward>(
            core, flatten_ebc_param, global_embedding_data_, embedding_sharding_param));
        break;
      case TablePlacementStrategy::ModelParallel:
        embeddings_.push_back(std::make_unique<UniformLocalizedEmbeddingForward>(
            core, flatten_ebc_param, global_embedding_data_, embedding_sharding_param));
        break;
      default:
        HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError, "embedding forward create fail.");
    }
  }
}

void EmbeddingCollectionForward::forward_per_gpu(
    const Tensor &keys, const Tensor &bucket_range, size_t num_keys, const Tensor &sparse_weight,
    std::vector<ILookup *> &embedding_tables, Tensor &output_buffer,
    std::vector<ContextContainer *> *context_container_list) {
  context_container_list->clear();
  int batch_size = (bucket_range.get_num_elements() - 1) / num_embedding_;

  Tensor t_keys, t_bucket_range;
  preprocess_input_.compute(keys, bucket_range, num_keys, &t_keys, &t_bucket_range);

  for (size_t embedding_id = 0; embedding_id < embedding_tables.size(); ++embedding_id) {
    auto &embedding = embeddings_[embedding_id];

    ContextContainer *context_container = new ContextContainer();
    embedding->forward_per_gpu(t_keys, t_bucket_range, num_keys, sparse_weight,
                               embedding_tables[embedding_id], output_buffer, context_container);
    context_container_list->push_back(context_container);
  }
  process_output_.compute(global_embedding_data_.d_combiner_list_,
                          global_embedding_data_.d_ev_size_list_,
                          global_embedding_data_.d_ev_size_offset_, output_buffer, batch_size);
}

EmbeddingCollectionBackward::EmbeddingCollectionBackward(
    std::shared_ptr<CoreResourceManager> core,
    const EmbeddingCollectionParam &embedding_collection_param,
    const std::vector<EmbeddingShardingParam> &embedding_sharding_params)
    : process_output_(core, embedding_collection_param),
      is_utest_(embedding_collection_param.is_utest) {
  EmbeddingCollectionParam flatten_ebc_param = embedding_collection_param;
  std::vector<std::vector<EmbeddingShardingParam>> flatten_ebs_param{embedding_sharding_params};

  flatten_concat_embedding(&flatten_ebc_param, &flatten_ebs_param);
  global_embedding_data_ = GlobalEmbeddingData(core, flatten_ebc_param);

  for (auto &embedding_sharding_param : flatten_ebs_param[0]) {
    switch (embedding_sharding_param.table_placement_strategy) {
      case TablePlacementStrategy::DataParallel:
        embeddings_.push_back(std::make_unique<UniformDPEmbeddingBackward>(
            core, flatten_ebc_param, global_embedding_data_, embedding_sharding_param));
        break;
      case TablePlacementStrategy::ModelParallel:
        embeddings_.push_back(std::make_unique<UniformLocalizedEmbeddingBackward>(
            core, flatten_ebc_param, global_embedding_data_, embedding_sharding_param));
        break;
      default:
        HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError, "embedding backward create fail.");
    }
  }
}

void EmbeddingCollectionBackward::backward_per_gpu(
    std::vector<ContextContainer *> &context_container_list, Tensor &top_grad,
    std::vector<Tensor> *unique_key_list, std::vector<size_t> *num_unique_key_list,
    std::vector<Tensor> *unique_id_space_offset_list,
    std::vector<size_t> *num_unique_id_space_offset_list, std::vector<Tensor> *grad_ev_list,
    std::vector<Tensor> *unique_dst_idx_list, std::vector<Tensor> *unique_id_space_list_list,
    bool do_allreduce) {
  unique_key_list->clear();
  num_unique_key_list->clear();
  unique_id_space_offset_list->clear();
  num_unique_id_space_offset_list->clear();
  grad_ev_list->clear();
  unique_dst_idx_list->clear();
  unique_id_space_list_list->clear();

  int batch_size_per_gpu =
      top_grad.get_num_elements() / global_embedding_data_.h_ev_size_offset_.back();
  Tensor t_top_grad;
  process_output_.compute(
      global_embedding_data_.d_combiner_list_, global_embedding_data_.d_ev_size_list_,
      global_embedding_data_.d_ev_size_offset_, top_grad, batch_size_per_gpu, &t_top_grad);

  for (size_t embedding_id = 0; embedding_id < context_container_list.size(); ++embedding_id) {
    ContextContainer *context_container = context_container_list[embedding_id];

    Tensor unique_key;
    size_t num_unique_key;
    Tensor unique_id_space_offset;
    size_t num_unique_id_space_offset;
    Tensor grad_ev;
    Tensor unique_dst_idx;
    Tensor unique_id_space_list = context_container->unpack<core::Tensor>("unique_id_space_list");

    embeddings_[embedding_id]->backward_per_gpu(
        context_container, t_top_grad, do_allreduce, &unique_key, &num_unique_key,
        &unique_id_space_offset, &num_unique_id_space_offset, &grad_ev, &unique_dst_idx);

    unique_key_list->push_back(unique_key);
    num_unique_key_list->push_back(num_unique_key);
    unique_id_space_offset_list->push_back(unique_id_space_offset);
    num_unique_id_space_offset_list->push_back(num_unique_id_space_offset);
    grad_ev_list->push_back(grad_ev);
    unique_dst_idx_list->push_back(unique_dst_idx);
    unique_id_space_list_list->push_back(unique_id_space_list);
    if (!is_utest_) {
      delete context_container;
    }
  }
}
}  // namespace embedding
