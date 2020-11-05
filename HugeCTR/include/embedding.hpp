/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

class IEmbedding {
 public:
  virtual ~IEmbedding() {}
  virtual void forward(bool is_train) = 0;
  virtual void backward() = 0;
  virtual void update_params() = 0;
  virtual void init_params() = 0;
  virtual void load_parameters(std::ifstream& stream) = 0;
  virtual void dump_parameters(std::ofstream& stream) const = 0;
  virtual void set_learning_rate(float lr) = 0;
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
};

template <typename TypeEmbeddingComp>
struct SparseEmbeddingHashParams {
  size_t train_batch_size;  // batch size
  size_t evaluate_batch_size;
  size_t max_vocabulary_size_per_gpu;       // max row number of hash table for each gpu
  std::vector<size_t> slot_size_array;      // max row number for each slot
  size_t embedding_vec_size;                // col number of hash table value
  size_t max_feature_num;                   // max feature number of all input samples of all slots
  size_t slot_num;                          // slot number
  int combiner;                             // 0-sum, 1-mean
  OptParams<TypeEmbeddingComp> opt_params;  // optimizer params
};

}  // namespace HugeCTR
