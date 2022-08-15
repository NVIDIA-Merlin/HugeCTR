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
#include "embedding_data.hpp"
#include "embedding_table.hpp"
#include "operators/compress_offset.hpp"
#include "operators/model_backward.hpp"
#include "operators/model_forward.hpp"
#include "operators/mp_index_calculation.hpp"
#include "operators/network_backward.hpp"
#include "operators/network_forward.hpp"
namespace embedding {

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
                                       std::vector<int> *unique_id_space_list) = 0;
  virtual void copy_backward_key_and_emb_vec(std::vector<Tensor> &unique_key,
                                             std::vector<Tensor> &emb_vec) = 0;
};

class All2AllEmbeddingCollectionSwizzleKey : public IAll2AllEmbeddingCollectionSwizzleKey {
  std::shared_ptr<CoreResourceManager> core_;

 public:
  All2AllEmbeddingCollectionSwizzleKey(std::shared_ptr<CoreResourceManager> core);

  void sparse_forward_per_gpu(const std::vector<Tensor> &keys,
                              const std::vector<Tensor> &row_lengths,
                              Tensor &key_all_gather_send_buffer,
                              Tensor &row_lengths_all_gather_send_buffer) override;
};

class All2AllEmbeddingCollectionModelForward : public IAll2AllEmbeddingCollectionModelForward {
  std::shared_ptr<CoreResourceManager> core_;
  GlobalEmbeddingData global_embedding_data_;
  LocalEmbeddingData local_embedding_data_;

  ModelIndexCalculation model_index_calculation_;
  CompressOffset compress_offset_;
  ModelForward model_forward_;

  Tensor model_key_, model_offsets_;

 public:
  All2AllEmbeddingCollectionModelForward(std::shared_ptr<CoreResourceManager> core,
                                         const EmbeddingCollectionParam &ebc_param,
                                         const EmbeddingShardParam &shard_param);

  std::vector<size_t> get_model_comm_buffer_size(int batch_size) override;

  void sparse_forward_per_gpu(const Tensor &key_all_gather_recv_buffer,
                              const Tensor &row_lengths_all_gather_recv_buffer,
                              ILookup *emb_storage, std::vector<Tensor> &emb_vec_model_buffer,
                              int64_t *num_model_key, int64_t *num_model_offsets) override;

  void copy_model_keys_and_offsets(Tensor &model_key, Tensor &model_offsets) override;
};

class All2AllEmbeddingCollectionNetworkForward : public IAll2AllEmbeddingCollectionNetworkForward {
  std::shared_ptr<CoreResourceManager> core_;
  GlobalEmbeddingData global_embedding_data_;
  LocalEmbeddingData local_embedding_data_;
  EmbeddingShardingParam sharding_param_;

  NetworkForward network_forward_;

 public:
  All2AllEmbeddingCollectionNetworkForward(std::shared_ptr<CoreResourceManager> core,
                                           const EmbeddingCollectionParam &ebc_param,
                                           const EmbeddingShardParam &shard_param);

  void sparse_forward_per_gpu(const std::vector<Tensor> &emb_vec_network_buffer,
                              const std::vector<Tensor> &row_lengths,
                              std::vector<Tensor> &forward_emb_vec) override;
};

class All2AllEmbeddingCollectionNetworkBackward
    : public IAll2AllEmbeddingCollectionNetworkBackward {
  std::shared_ptr<CoreResourceManager> core_;
  GlobalEmbeddingData global_embedding_data_;
  LocalEmbeddingData local_embedding_data_;
  EmbeddingShardingParam sharding_param_;

  NetworkBackward network_backward_;

 public:
  All2AllEmbeddingCollectionNetworkBackward(std::shared_ptr<CoreResourceManager> core,
                                            const EmbeddingCollectionParam &ebc_param,
                                            const EmbeddingShardParam &shard_param);

  void backward_per_gpu(const std::vector<Tensor> &top_grad, const std::vector<Tensor> &row_lengths,
                        std::vector<Tensor> &emb_vec_network_buffer) override;
};

class All2AllEmbeddingCollectionModelBackward : public IAll2AllEmbeddingCollectionModelBackward {
  std::shared_ptr<CoreResourceManager> core_;
  GlobalEmbeddingData global_embedding_data_;
  LocalEmbeddingData local_embedding_data_;
  ModelBackwardIndexCalculation model_backward_index_calculation_;
  ModelBackward model_backward_;

  Tensor continous_unique_key_, continous_emb_vec_;

 public:
  All2AllEmbeddingCollectionModelBackward(std::shared_ptr<CoreResourceManager> core,
                                          const EmbeddingCollectionParam &ebc_param,
                                          const EmbeddingShardParam &shard_param);

  void sparse_backward_per_gpu(const std::vector<Tensor> &emb_vec_model_buffer,
                               const Tensor &model_key, const Tensor &model_offsets,
                               std::vector<int> *num_unique_key_per_table,
                               std::vector<int> *unique_id_space_list) override;

  void copy_backward_key_and_emb_vec(std::vector<Tensor> &unique_key,
                                     std::vector<Tensor> &emb_vec) override;
};
}  // namespace tf
}  // namespace embedding
