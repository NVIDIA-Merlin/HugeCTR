#pragma once

#include <nccl.h>

#include <optional>
#include <unordered_map>
#include <vector>

#include "HugeCTR/embedding/operators/dp_index_calculation.hpp"
#include "HugeCTR/embedding/operators/mp_index_calculation.hpp"
#include "HugeCTR/embedding/operators/transpose_input.hpp"
#include "core/buffer.hpp"
#include "core/core.hpp"
#include "embedding/common.hpp"
#include "resource_manager.hpp"

namespace HugeCTR {

class DataDistributor {
 public:
  using Result = std::vector<embedding::EmbeddingInput>;

  DataDistributor(size_t batch_size, core::DataType scalar_type,
                  std::shared_ptr<ResourceManager> resource_manager,
                  std::vector<std::shared_ptr<core::CoreResourceManager>>& core_resource_managers,
                  const embedding::EmbeddingCollectionParam& ebc_param);

  ~DataDistributor();

  void distribute(int gpu_id, const std::vector<core::Tensor>& dp_keys,
                  const std::vector<core::Tensor>& dp_bucket_range, Result& output, int batch_size);

  void distribute(int gpu_id, const core::Tensor& fullbatch_keys,
                  const core::Tensor& fullbatch_bucket_range, Result& output, int batch_size);

  // TODO: remove when enable table filtering. This function is just to set the bucket ranges
  // because we return a global batch
  void init_fixed_bucket_ranges(core::Tensor& output_bucket_ranges) const;

 private:
  struct GpuCommData {
    int local_rank;
    // This is a performance optimization to prevent us from computing bucket ranges each iteration.
    // If the current_batch_size == last_batch_size then the bucket_ranges are the same.
    int last_batch_size;
    core::Tensor hotness_bucket_range;
    core::Tensor features;
    core::Tensor bucket_range;
  };

  size_t feature_id_to_group_id(size_t feature_id) const;

  void init_nccl_comms();

  void init_comm_data();

  void init_batch_major_fullbatch_input_preprocessor();

  void communicate_data(std::vector<core::Tensor> feature_shards, int gpu_id, cudaStream_t stream);

  std::shared_ptr<ResourceManager> resource_manager_;
  std::vector<std::shared_ptr<core::CoreResourceManager>> core_resource_managers_;
  std::vector<int> feature_pooling_factors_;
  std::vector<std::vector<int>> resident_feature_tables_;  // [gpu_id][feature_id]
  std::vector<ncclComm_t> comms_;
  std::vector<GpuCommData> gpu_comm_data_;

  size_t batch_size_;
  core::DataType scalar_type_;

  embedding::EmbeddingCollectionParam ebc_param_;
  std::unordered_map<size_t, size_t> feature_id_to_group_id_map_;
  std::unordered_map<size_t, size_t> feature_id_to_table_id_map_;

  size_t num_local_gpus_;
  size_t num_features_;
  int my_rank_;
  int n_ranks_;

  struct KeyFilterInitParams {
    int num_lookup;
    int global_gpu_id;
    int total_gpu_count;

    int num_local_lookup;
    int num_hotness;
    int num_local_hotness;

    core::Tensor d_local_lookup_ids;
    core::Tensor d_local_shard_ids;
    core::Tensor d_local_num_shards;

    KeyFilterInitParams(const std::shared_ptr<core::CoreResourceManager>& core_resource_manager,
                        const embedding::EmbeddingCollectionParam& ebc_param, size_t grouped_id);
  };
  std::vector<std::vector<KeyFilterInitParams>> key_filters_init_params_;

  struct KeyFilter {
    embedding::MPKeySelector mp_key_selector;
    embedding::ModelIndexCalculation mp_index_calculation;
    embedding::DPKeySelector dp_key_selector;
    embedding::DPIndexCalculation dp_index_calculation;
  };
  std::vector<std::vector<KeyFilter>> key_filters_;

  void init_key_filter();

  std::vector<std::unique_ptr<embedding::PreprocessInput>> preprocess_inputs_;
};

DataDistributor::Result allocate_output_for_data_distributor(
    std::shared_ptr<core::CoreResourceManager>& core_resource_manager,
    const embedding::EmbeddingCollectionParam& ebc_param);
}  // namespace HugeCTR