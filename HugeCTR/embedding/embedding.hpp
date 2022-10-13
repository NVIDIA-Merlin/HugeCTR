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
#include <vector>

#include "embedding_table.hpp"

namespace embedding {

class IGroupedEmbeddingOp {
 public:
  virtual ~IGroupedEmbeddingOp() = default;

  virtual void forward_per_gpu(const Tensor &keys, const Tensor &bucket_range, size_t num_keys,
                               ILookup *embedding_table, Tensor &output_buffer, int batch_size) = 0;

  virtual void backward_per_gpu(const Tensor &top_grad, bool do_allreduce, Tensor *unique_key,
                                size_t *num_unique_key, Tensor *num_unique_key_per_table_offset,
                                size_t *num_table_offset, Tensor *table_id_list, Tensor *wgrad,
                                Tensor *wgrad_idx_offset) = 0;
};

std::vector<std::unique_ptr<IGroupedEmbeddingOp>> create_grouped_embeddings(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param);

namespace tf {

// convert table-major data parallel csr input into communication buffer for all-gather
class IAll2AllEmbeddingCollectionSwizzleKey {
 public:
  virtual ~IAll2AllEmbeddingCollectionSwizzleKey() = default;

  virtual void sparse_forward_per_gpu(
      const std::vector<Tensor> &keys,        /* num of lookup operation */
      const std::vector<Tensor> &row_lengths, /* num of lookup operation */
      Tensor &key_all_gather_send_buffer, Tensor &row_lengths_all_gather_send_buffer) = 0;
};

// read key from all gather result, do local lookup / combine, write result to all2all send
// buffer(emb_vec_model_buffer)
class IAll2AllEmbeddingCollectionModelForward {
 public:
  virtual ~IAll2AllEmbeddingCollectionModelForward() = default;

  virtual std::vector<size_t> get_model_comm_buffer_size(int batch_size) = 0;

  virtual void sparse_forward_per_gpu(const Tensor &key_all_gather_recv_buffer,
                                      const Tensor &row_lengths_all_gather_recv_buffer,
                                      ILookup *emb_storage,
                                      std::vector<Tensor> &emb_vec_model_buffer
                                      /*num of gpus */,
                                      int64_t *num_model_key, int64_t *num_model_offsets) = 0;

  virtual void copy_model_keys_and_offsets(Tensor &model_key, Tensor &model_offsets) = 0;
};

// all2all will happen to send data from emb_vec_model_buffer to emb_vec_network_buffer.
// read emb vec from emb_vec_network_buffer and write to forward_emb_vec.
class IAll2AllEmbeddingCollectionNetworkForward {
 public:
  virtual ~IAll2AllEmbeddingCollectionNetworkForward() = default;

  virtual void sparse_forward_per_gpu(
      const std::vector<Tensor> &emb_vec_network_buffer, /* num of gpus*/
      const std::vector<Tensor> &row_lengths,            /* num of lookup operations*/
      std::vector<Tensor> &forward_emb_vec /* num of lookup operations*/) = 0;
};

// the backward process of IAll2AllEmbeddingCollectionNetworkForward. It's the same for both spaarse
// and dense use case.
class IAll2AllEmbeddingCollectionNetworkBackward {
 public:
  virtual ~IAll2AllEmbeddingCollectionNetworkBackward() = default;

  virtual void backward_per_gpu(
      const std::vector<Tensor> &top_grad,
      const std::vector<Tensor> &row_lengths, /* num of lookup operations*/
      std::vector<Tensor> &emb_vec_network_buffer /* num of gpus*/) = 0;
};

// read emb vec from emb_vec_model_buffer and do grad accumulation to get grad.
class IAll2AllEmbeddingCollectionModelBackward {
 public:
  virtual ~IAll2AllEmbeddingCollectionModelBackward() = default;

  virtual void sparse_backward_per_gpu(const std::vector<Tensor> &emb_vec_model_buffer,
                                       const Tensor &model_key, const Tensor &model_offsets,
                                       std::vector<int> *num_unique_key_per_table,
                                       std::vector<int> *table_id_list) = 0;
  virtual void copy_backward_key_and_emb_vec(std::vector<Tensor> &unique_key,
                                             std::vector<Tensor> &emb_vec) = 0;
};

}  // namespace tf
}  // namespace embedding
