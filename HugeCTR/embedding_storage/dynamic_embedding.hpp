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

#include <core/registry.hpp>
#include <embedding_storage/embedding_table.hpp>
#include <map>
#include <mutex>

namespace embedding {

using HugeCTR::CudaDeviceContext;

class DynamicEmbeddingTable final : public IDynamicEmbeddingTable {
  std::shared_ptr<CoreResourceManager> core_;
  core::DataType key_type_;

  mutable std::mutex write_mutex_;

  void *table_;
  std::map<size_t, size_t> global_to_local_id_space_map_;
  std::vector<size_t> dim_per_class_;
  std::vector<int> h_table_ids_;

  HugeCTR::OptParams opt_param_;
  void *table_opt_states_;

  std::unique_ptr<TensorList> opt_state_view_;
  std::unique_ptr<TensorList> weight_view_;

 public:
  DynamicEmbeddingTable(const HugeCTR::GPUResource &gpu_resource,
                        std::shared_ptr<CoreResourceManager> core,
                        const std::vector<EmbeddingTableParam> &table_params,
                        const EmbeddingCollectionParam &ebc_param, size_t grouped_id,
                        const HugeCTR::OptParams &opt_param);

  std::vector<size_t> remap_id_space(const Tensor &id_space_list, cudaStream_t stream);

  void lookup(const Tensor &keys, size_t num_keys, const Tensor &id_space_offset,
              size_t num_id_space_offset, const Tensor &id_space,
              TensorList &embedding_vec) override;

  void update(const Tensor &keys, size_t num_keys, const Tensor &num_unique_key_per_table_offset,
              size_t num_table_offset, const Tensor &table_id_list, Tensor &wgrad,
              const Tensor &wgrad_idx_offset) override;

  void assign(const Tensor &unique_key, size_t num_unique_key,
              const Tensor &num_unique_key_per_table_offset, size_t num_table_offset,
              const Tensor &table_id_list, Tensor &embeding_vector,
              const Tensor &embedding_vector_offset) override;

  void load(Tensor &keys, Tensor &id_space_offset, Tensor &embedding_table, Tensor &ev_size_list,
            Tensor &id_space) override;

  void dump(Tensor *keys, Tensor *id_space_offset, Tensor *embedding_table, Tensor *ev_size_list,
            Tensor *id_space) override;

  void dump_by_id(Tensor *h_keys_tensor, Tensor *h_embedding_table, int table_id) override;

  void load_by_id(Tensor *h_keys_tensor, Tensor *h_embedding_table, int table_id) override;

  size_t size() const override;

  size_t capacity() const override;

  size_t key_num() const override;

  std::vector<size_t> size_per_table() const override;

  std::vector<size_t> capacity_per_table() const override;

  std::vector<size_t> key_num_per_table() const override;

  std::vector<int> table_ids() const override;

  std::vector<int> table_evsize() const override;

  void clear() override;

  void evict(const Tensor &keys, size_t num_keys, const Tensor &id_space_offset,
             size_t num_id_space_offset, const Tensor &id_space_list) override;

  void set_learning_rate(float lr) override { opt_param_.lr = lr; }
};

}  // namespace embedding
