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
#pragma once
#include "model_parallel_embedding.hpp"
#include "operators/weighted_model_backward.hpp"
#include "operators/weighted_model_forward.hpp"
#include "operators/weighted_mp_index_calculation.hpp"
#include "operators/weighted_network_backward.hpp"
#include "operators/weighted_network_forward.hpp"

namespace embedding {

namespace tf {
namespace swizzle_key {

void weighted_sparse_forward_per_gpu(std::shared_ptr<CoreResourceManager> core,
                                     const std::vector<Tensor> &keys,
                                     const std::vector<Tensor> &row_lengths,
                                     const std::vector<Tensor> &sp_weights,
                                     Tensor &key_all_gather_send_buffer,
                                     Tensor &row_lengths_all_gather_send_buffer,
                                     Tensor &sp_weight_all_gather_send_buffer);

void sparse_forward_per_gpu(std::shared_ptr<CoreResourceManager> core,
                            const std::vector<Tensor> &keys, const std::vector<Tensor> &row_lengths,
                            Tensor &key_all_gather_send_buffer,
                            Tensor &row_lengths_all_gather_send_buffer);
}  // namespace swizzle_key

namespace model_forward {

std::vector<size_t> get_model_comm_buffer_size(const UniformModelParallelEmbeddingMeta &meta,
                                               int num_gpus, int batch_size);

void weighted_sparse_forward_per_gpu(
    std::shared_ptr<CoreResourceManager> core, const UniformModelParallelEmbeddingMeta &meta,
    int global_gpu_id, const Tensor &key_all_gather_recv_buffer,
    const Tensor &row_lengths_all_gather_recv_buffer,
    const Tensor &sp_weights_all_gather_recv_buffer, ILookup *emb_storage,
    std::vector<Tensor> &emb_vec_model_buffer, int64_t *num_model_key, int64_t *num_model_offsets,
    Tensor *ret_model_key, Tensor *ret_model_offset, Tensor *ret_sp_weight);

void weighted_copy_model_keys_and_offsets(std::shared_ptr<CoreResourceManager> core,
                                          const Tensor &model_key, const Tensor &model_offset,
                                          const Tensor &model_sp_weight, Tensor &tf_model_key,
                                          Tensor &tf_model_offsets, Tensor &tf_sp_weight);

void sparse_forward_per_gpu(std::shared_ptr<CoreResourceManager> core,
                            const EmbeddingCollectionParam &ebc_param,
                            const UniformModelParallelEmbeddingMeta &meta,
                            const Tensor &key_all_gather_recv_buffer,
                            const Tensor &row_lengths_all_gather_recv_buffer, ILookup *emb_storage,
                            std::vector<Tensor> &emb_vec_model_buffer, int64_t *num_model_key,
                            int64_t *num_model_offsets, Tensor *ret_model_key,
                            Tensor *ret_model_offset);

void copy_model_keys_and_offsets(std::shared_ptr<CoreResourceManager> core, const Tensor &model_key,
                                 const Tensor &model_offset, Tensor &tf_model_key,
                                 Tensor &tf_model_offsets);
}  // namespace model_forward

namespace network_forward {

void weighted_sparse_forward_per_gpu(std::shared_ptr<CoreResourceManager> core,
                                     const UniformModelParallelEmbeddingMeta &meta,
                                     const std::vector<Tensor> &emb_vec_network_buffer,
                                     const std::vector<Tensor> &row_lengths, const Tensor &sp_sum,
                                     std::vector<Tensor> &forward_emb_vec);

void sparse_forward_per_gpu(std::shared_ptr<CoreResourceManager> core,
                            const UniformModelParallelEmbeddingMeta &meta,
                            const std::vector<Tensor> &emb_vec_network_buffer,
                            const std::vector<Tensor> &row_lengths,
                            std::vector<Tensor> &forward_emb_vec);
}  // namespace network_forward

namespace network_backward {

void weighted_backward_per_gpu(std::shared_ptr<CoreResourceManager> core,
                               const UniformModelParallelEmbeddingMeta &meta,
                               const std::vector<Tensor> &top_grad,
                               const std::vector<Tensor> &row_lengths,
                               std::vector<Tensor> &emb_vec_network_buffer, const Tensor &sp_sum);

void backward_per_gpu(std::shared_ptr<CoreResourceManager> core,
                      const UniformModelParallelEmbeddingMeta &meta,
                      const std::vector<Tensor> &top_grad, const std::vector<Tensor> &row_lengths,
                      std::vector<Tensor> &emb_vec_network_buffer);
}  // namespace network_backward

namespace model_backward {

void weighted_sparse_backward_per_gpu(
    std::shared_ptr<CoreResourceManager> core, const UniformModelParallelEmbeddingMeta &meta,
    const std::vector<Tensor> &emb_vec_model_buffer, const Tensor &model_key,
    const Tensor &model_offsets, const Tensor &model_sp_weight,
    std::vector<int> *num_unique_key_per_table, std::vector<int> *table_id_list,
    Tensor *ret_continous_unique_key, Tensor *ret_continous_emb_vec);

void sparse_backward_per_gpu(std::shared_ptr<CoreResourceManager> core,
                             const EmbeddingCollectionParam &ebc_param,
                             const UniformModelParallelEmbeddingMeta &meta,
                             const std::vector<Tensor> &emb_vec_model_buffer,
                             const Tensor &model_key, const Tensor &model_offsets,
                             std::vector<int> *num_unique_key_per_table,
                             std::vector<int> *table_id_list, Tensor *ret_continous_unique_key,
                             Tensor *ret_continous_emb_vec);

void copy_backward_key_and_emb_vec(std::shared_ptr<CoreResourceManager> core,
                                   const Tensor &continous_unique_key,
                                   const Tensor &continous_emb_vec, std::vector<Tensor> &unique_key,
                                   std::vector<Tensor> &emb_vec);
}  // namespace model_backward

}  // namespace tf
}  // namespace embedding
