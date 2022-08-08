#pragma once

#include <map>

#include "HugeCTR/core/registry.hpp"
#include "embedding_table.hpp"

namespace embedding {

using HugeCTR::CudaDeviceContext;

class DynamicEmbeddingTable final : public IDynamicEmbeddingTable {
  std::shared_ptr<CoreResourceManager> core_;
  core::DataType key_type_;
  void *table_;
  std::map<size_t, size_t> global_to_local_id_space_map_;

  HugeCTR::OptParams opt_param_;

 public:
  DynamicEmbeddingTable(std::shared_ptr<CoreResourceManager> core,
                        const std::vector<EmbeddingTableParam> &global_emb_table_param_list,
                        const EmbeddingCollectionParam &ebc_param,
                        const EmbeddingShardingParam &sharding_param);

  std::vector<size_t> remap_id_space(const Tensor &id_space_list, cudaStream_t stream);

  void lookup(const Tensor &keys, size_t num_keys, const Tensor &id_space_offset,
              size_t num_id_space_offset, const Tensor &id_space,
              TensorList &embedding_vec) override;

  void update(const Tensor &keys, size_t num_keys, const Tensor &id_space_offset,
              size_t num_id_space_offset, const Tensor &id_space_list, Tensor &grad_ev,
              const Tensor &grad_ev_offset) override;

  void load(Tensor &keys, Tensor &id_space_offset, Tensor &embedding_table, Tensor &ev_size_list,
            Tensor &id_space) override;

  void dump(Tensor *keys, Tensor *id_space_offset, Tensor *embedding_table, Tensor *ev_size_list,
            Tensor *id_space) override;

  size_t size() const override;

  size_t capacity() const override;

  void clear() override;

  void evict(const Tensor &keys, size_t num_keys, const Tensor &id_space_offset,
             size_t num_id_space_offset, const Tensor &id_space_list) override;

  void set_learning_rate(float lr) override { opt_param_.lr = lr; }
};
}  // namespace embedding
