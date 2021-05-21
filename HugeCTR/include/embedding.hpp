/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <optimizer.hpp>
#include <tensor2.hpp>
#include <gpu_learning_rate_scheduler.hpp>
#include <vector>

#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
namespace HugeCTR {

class IEmbedding {
 public:
  virtual ~IEmbedding() {}

  virtual TrainState train(bool is_train, int i, TrainState state) { 
    return TrainState();
  }
  virtual void forward(bool is_train, int eval_batch = -1) = 0;
  virtual void backward() = 0;
  virtual void update_params() = 0;
  virtual void init_params() = 0;
  virtual void load_parameters(std::istream& stream) = 0;
  virtual void dump_parameters(std::ostream& stream) const = 0;
  virtual void set_learning_rate(float lr) = 0;
  // TODO: a workaround to enable GPU LR for HE only; need a better way
  virtual GpuLearningRateSchedulers get_learning_rate_schedulers() const {
    return GpuLearningRateSchedulers();
  }
  virtual size_t get_params_num() const = 0;
  virtual size_t get_vocabulary_size() const = 0;
  virtual size_t get_max_vocabulary_size() const = 0;
  virtual void load_parameters(const TensorBag2& keys, const Tensor2<float>& embeddings,
                               size_t num) = 0;
  virtual void dump_parameters(TensorBag2 keys, Tensor2<float>& embeddings, size_t* num) const = 0;
  virtual void reset() = 0;

  virtual std::vector<TensorBag2> get_train_output_tensors() const = 0;
  virtual std::vector<TensorBag2> get_evaluate_output_tensors() const = 0;
  virtual void check_overflow() const = 0;
  virtual void get_forward_results_tf(const bool is_train, const bool on_gpu,
                                      void* const forward_result) = 0;
  virtual cudaError_t update_top_gradients(const bool on_gpu, const void* const top_gradients) = 0;
};

template <typename TypeEmbeddingComp>
struct SparseEmbeddingHashParams {
  size_t train_batch_size;  // batch size
  size_t evaluate_batch_size;
  size_t max_vocabulary_size_per_gpu;   // max row number of hash table for each gpu
  std::vector<size_t> slot_size_array;  // max row number for each slot
  size_t embedding_vec_size;            // col number of hash table value
  size_t max_feature_num;               // max feature number of all input samples of all slots
  size_t slot_num;                      // slot number
  int combiner;                         // 0-sum, 1-mean
  OptParams<TypeEmbeddingComp> opt_params;  // optimizer params
};

struct HybridSparseEmbeddingCategoryItem {
  size_t index;
  size_t global_gpu_id;
};

template <typename TypeEmbedding>
struct HybridSparseEmbeddingParams {
  size_t train_batch_size;
  size_t evaluate_batch_size;
  // std::vector<HybridSparseEmbeddingCategoryItem> categories;
  size_t num_iterations_statistics;
  size_t max_num_frequent_categories;  // max(train_batch_size, eval_batch_size) * # of batches for
                                       // frequent categories
  int64_t max_num_infrequent_samples;
  double p_dup_max;
  size_t embedding_vec_size;
  size_t slot_num;  // slot number
  std::vector<size_t> slot_size_array;
  hybrid_embedding::CommunicationType communication_type;
  double max_all_reduce_bandwidth;
  double max_all_to_all_bandwidth;
  double efficiency_bandwidth_ratio;
  hybrid_embedding::HybridEmbeddingType hybrid_embedding_type;
  OptParams<TypeEmbedding> opt_params;  // optimizer params
};

template <typename TypeEmbedding>
class IEmbeddingForPrefetcher {
 public:
  virtual void load_parameters(const TensorBag2& keys, const Tensor2<float>& embeddings,
                               size_t num) = 0;
  virtual void dump_parameters(TensorBag2 keys, Tensor2<float>& embeddings, size_t* num) const = 0;
  virtual void reset() = 0;
  virtual const SparseEmbeddingHashParams<TypeEmbedding>& get_embedding_params() const = 0;
};

}  // namespace HugeCTR
