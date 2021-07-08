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

#include <vector>

namespace HugeCTR {
struct BufferBag;
class IEmbedding {
 public:
  virtual ~IEmbedding() {}
  virtual void forward(bool is_train) = 0;
  virtual void backward() = 0;
  virtual void update_params() = 0;
  virtual void init_params() = 0;
  virtual void load_parameters(std::string sparse_model) = 0;
  virtual void dump_parameters(std::string sparse_model) const = 0;
  virtual void set_learning_rate(float lr) = 0;
  virtual size_t get_params_num() const = 0;
  virtual size_t get_vocabulary_size() const = 0;
  virtual size_t get_max_vocabulary_size() const = 0;

  virtual Embedding_t get_embedding_type() const = 0;
  virtual void load_parameters(BufferBag& buf_bag, size_t num) = 0;
  virtual void dump_parameters(BufferBag& buf_bag, size_t* num) const = 0;
  virtual void reset() = 0;

  virtual void dump_opt_states(std::ofstream& stream) = 0;
  virtual void load_opt_states(std::ifstream& stream) = 0;

  virtual std::vector<TensorBag2> get_train_output_tensors() const = 0;
  virtual std::vector<TensorBag2> get_evaluate_output_tensors() const = 0;
  virtual void check_overflow() const = 0;
  virtual void get_forward_results_tf(const bool is_train, const bool on_gpu, void* const forward_result) = 0;
  virtual cudaError_t update_top_gradients(const bool on_gpu, const void* const top_gradients) = 0;
};

struct SparseEmbeddingHashParams {
  size_t train_batch_size;  // batch size
  size_t evaluate_batch_size;
  size_t max_vocabulary_size_per_gpu;       // max row number of hash table for each gpu
  std::vector<size_t> slot_size_array;      // max row number for each slot
  size_t embedding_vec_size;                // col number of hash table value
  size_t max_feature_num;                   // max feature number of all input samples of all slots
  size_t slot_num;                          // slot number
  int combiner;                             // 0-sum, 1-mean
  OptParams opt_params;  // optimizer params

  size_t get_batch_size(bool is_train) const {
    if (is_train) {
      return train_batch_size;
    } else {
      return evaluate_batch_size;
    }
  }

  size_t get_universal_batch_size() const {
    return std::max(train_batch_size, evaluate_batch_size);
  }
  const Update_t& get_update_type() const { return opt_params.update_type; }
  const Optimizer_t& get_optimizer() const { return opt_params.optimizer; }
  OptParams& get_opt_params() { return opt_params; }
  size_t get_embedding_vec_size() const { return embedding_vec_size; }
  size_t get_max_feature_num() const { return max_feature_num; }
  size_t get_slot_num() const { return slot_num; }
  int get_combiner() const { return combiner; }
  size_t get_max_vocabulary_size_per_gpu() const {
    return max_vocabulary_size_per_gpu;
  }

};
struct BufferBag {
  TensorBag2 keys;
  TensorBag2 slot_id;
  Tensor2<float> embedding;

  Tensors2<float> h_value_tensors;
  Tensors2<size_t> h_slot_id_tensors;
  std::vector<TensorBag2> uvm_key_tensor_bags;
  Tensors2<size_t> d_value_index_tensors;
};
}  // namespace HugeCTR
