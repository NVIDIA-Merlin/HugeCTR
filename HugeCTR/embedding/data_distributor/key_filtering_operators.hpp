/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <core/core.hpp>
#include <embedding/common.hpp>
#include <memory>
#include <unordered_map>
#include <vector>

namespace HugeCTR {

// TODO: move to data reader
class ComputeDPBucketRangeOperator {
 public:
  ComputeDPBucketRangeOperator(std::shared_ptr<core::CoreResourceManager> core,
                               const embedding::EmbeddingCollectionParam &ebc_param);

  void operator()(std::vector<core23::Tensor> dp_bucket_ranges, core23::Tensor keys_per_bucket,
                  int current_batch_size, cudaStream_t stream);

 private:
  core23::Tensor h_ptrs_;
  core23::Tensor d_ptrs_;

  int global_gpu_id_ = 0;
  int batch_size_per_gpu_ = 0;
  core23::Tensor max_hotnesses_;
};

namespace mp {

class LabelAndCountKeysOperator {
 public:
  class Result {
   public:
    Result(std::shared_ptr<core::CoreResourceManager> resource_manager,
           const embedding::EmbeddingCollectionParam &ebc_param, size_t grouped_id);

    core23::Tensor local_labels;
    core23::Tensor keys_per_bucket;
    core23::Tensor keys_per_gpu;
    core23::Tensor flat_keys;
  };

  LabelAndCountKeysOperator(std::shared_ptr<core::CoreResourceManager> resource_manager,
                            const embedding::EmbeddingCollectionParam &ebc_param,
                            size_t grouped_id);

  void operator()(const DataDistributionInput &input, Result &output, cudaStream_t stream);

  std::vector<uint32_t> h_per_gpu_lookup_range;

 private:
  core23::Tensor lookup_gpu_ids;
  core23::Tensor lookup_num_gpus;
  core23::Tensor lookup_bucket_threads;  // used for better GPU work balancing
  core23::Tensor hotness_bucket_range;
  core23::Tensor gpu_lookup_range;
  core23::Tensor lookup_ids;

  int batch_size_ = 0;
  int batch_size_per_gpu_ = 0;
  int global_gpu_id_ = -1;
  int global_gpu_count_ = 0;

  std::shared_ptr<core::CoreResourceManager> core_;
};

class CountKeysOperator {
 public:
  CountKeysOperator(std::shared_ptr<core::CoreResourceManager> core,
                    const embedding::EmbeddingCollectionParam &ebc_param, size_t grouped_id);

  void operator()(core23::Tensor keys_per_bucket_gpu_major, core23::Tensor result_keys_per_gpu,
                  cudaStream_t stream);

 private:
  int batch_size_per_gpu_ = 0;
  int global_gpu_count_ = 0;
  int num_shards_ = 0;
};

/**
 * Convert buckets from GPU-major to feature-major
 */
class TransposeBucketsOperator {
 public:
  TransposeBucketsOperator(std::shared_ptr<core::CoreResourceManager> core,
                           const embedding::EmbeddingCollectionParam &ebc_param, size_t grouped_id);

  void operator()(core23::Tensor buckets_gpu_major, core23::Tensor buckets_feat_major,
                  cudaStream_t stream);

 private:
  int num_shards_ = 0;
  int batch_size_per_gpu_ = 0;
  int global_gpu_count_ = 0;
};

class SwizzleKeysOperator {
 public:
  SwizzleKeysOperator(std::shared_ptr<core::CoreResourceManager> resource_manager,
                      const embedding::EmbeddingCollectionParam &ebc_param, size_t grouped_id);

  void operator()(core23::Tensor src_bucket_range, core23::Tensor dst_bucket_range,
                  core23::Tensor keys, core23::Tensor result_keys, cudaStream_t stream);

 private:
  int num_shards_ = 0;
  int batch_size_per_gpu_ = 0;
  int global_gpu_count_ = 0;
  core23::Tensor shard_bucket_threads_;
};

}  // namespace mp

namespace dp {

class ConcatKeysAndBucketRangeOperator {
 public:
  ConcatKeysAndBucketRangeOperator(std::shared_ptr<core::CoreResourceManager> core,
                                   const embedding::EmbeddingCollectionParam &ebc_param,
                                   size_t grouped_id);

  void operator()(const DataDistributionInput &input, core23::Tensor &result_keys,
                  core23::Tensor &result_bucket_ranges, cudaStream_t stream);

 private:
  int batch_size_per_gpu_;
  std::vector<int> h_shard_ids_;
  core23::Tensor d_shard_ids_;
  core23::Tensor shard_bucket_threads_;
  core23::Tensor shard_ranges_;
};
}  // namespace dp

}  // namespace HugeCTR