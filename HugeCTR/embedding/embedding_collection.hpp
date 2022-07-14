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
#include "embedding.hpp"
#include "embedding_data.hpp"
namespace embedding {

class PreprocessInput {
 private:
  std::shared_ptr<CoreResourceManager> core_;

  int num_embedding_;
  int num_flatten_embedding_;

  Tensor t_key_;
  Tensor flatten_t_bucket_range_;
  Tensor d_embedding_offset_;
  Tensor d_combiner_list_;
  Tensor d_temp_scan_storage_;

  bool is_table_first_input_;

 public:
  PreprocessInput() = default;

  PreprocessInput(std::shared_ptr<CoreResourceManager> core,
                  const EmbeddingCollectionParam &ebc_param);

  void compute(const Tensor &key, const Tensor &bucket_range, size_t num_key,
               Tensor *preprocessed_key, Tensor *preprocessed_bucket_range);
};

class ProcessOutput {
 private:
  std::shared_ptr<CoreResourceManager> core_;

  int num_flatten_embedding_;

  Tensor original_embedding_id_list_;
  Tensor start_embedding_id_list_;
  Tensor hotness_list_;
  Tensor output_buffer_;

 public:
  ProcessOutput() = default;

  ProcessOutput(std::shared_ptr<CoreResourceManager> core,
                const EmbeddingCollectionParam &ebc_param);

  void compute(const Tensor &flatten_combiner_list, const Tensor &flatten_ev_size_list, const Tensor &flatten_ev_offset_list, Tensor &output_buffer, int batch_size);
};

class ProcessOutputBackward {
 private:
  std::shared_ptr<CoreResourceManager> core_;

  int num_flatten_embedding_;

  Tensor original_embedding_id_list_;
  Tensor start_embedding_id_list_;
  Tensor hotness_list_;
  Tensor output_buffer_;

 public:
  ProcessOutputBackward() = default;

  ProcessOutputBackward(std::shared_ptr<CoreResourceManager> core,
                const EmbeddingCollectionParam &ebc_param);

  void compute(const Tensor &flatten_combiner_list, const Tensor &flatten_ev_size_list, const Tensor &flatten_ev_offset_list, Tensor &output_buffer, int batch_size_per_gpu, Tensor *t_output_buffer);
};

class EmbeddingCollectionForward : public IEmbeddingCollectionForward {
  GlobalEmbeddingData global_embedding_data_;
  int num_embedding_;
  std::vector<std::unique_ptr<IEmbeddingForward>> embeddings_;
  PreprocessInput preprocess_input_;
  ProcessOutput process_output_;

 public:
  EmbeddingCollectionForward(std::shared_ptr<CoreResourceManager> core,
                             const EmbeddingCollectionParam &embedding_collection_param,
                             const std::vector<EmbeddingShardingParam> &embedding_sharding_params);

  void forward_per_gpu(const Tensor &key, const Tensor &bucket_range, size_t num_keys,
                       const Tensor &sparse_weight, std::vector<ILookup *> &embedding_tables,
                       Tensor &output_buffer,
                       std::vector<ContextContainer *> *context_container_list) override;
};

class EmbeddingCollectionBackward : public IEmbeddingCollectionBackward {
  GlobalEmbeddingData global_embedding_data_;
  std::vector<std::unique_ptr<IEmbeddingBackward>> embeddings_;
  ProcessOutputBackward process_output_;
  bool is_utest_;
  
 public:
  EmbeddingCollectionBackward(std::shared_ptr<CoreResourceManager> core,
                              const EmbeddingCollectionParam &embedding_collection_param,
                              const std::vector<EmbeddingShardingParam> &embedding_sharding_params);

  void backward_per_gpu(std::vector<ContextContainer *> &context_container_list,
                        Tensor &top_grad, std::vector<Tensor> *unique_key_list,
                        std::vector<size_t> *num_unique_key_list,
                        std::vector<Tensor> *unique_id_space_offset_list,
                        std::vector<size_t> *num_unique_id_space_offset_list,
                        std::vector<Tensor> *grad_ev_list, std::vector<Tensor> *unique_dst_idx_list,
                        std::vector<Tensor> *unique_id_space_list_list, bool do_allreduce) override;
};
}  // namespace embedding
