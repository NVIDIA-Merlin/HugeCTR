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

#ifndef HKV_VARIABLE_H
#define HKV_VARIABLE_H

#include <curand_kernel.h>

#include <string>

#include "merlin_hashtable.cuh"
#include "variable/impl/variable_base.h"

namespace sok {

template <typename KeyType, typename ValueType>
class HKVVariable : public VariableBase<KeyType, ValueType> {
 public:
  HKVVariable(int64_t dimension, int64_t initial_capacity, const std::string &initializer,
              size_t max_capacity = 0, size_t max_hbm_for_vectors = 0, size_t max_bucket_size = 128,
              float max_load_factor = 0.5f, int block_size = 128, int device_id = 0,
              bool io_by_cpu = false, const std::string &evict_strategy = "kLru",
              cudaStream_t stream = 0);

  ~HKVVariable() override;
  int64_t rows() override;
  int64_t cols() override;

  void eXport(KeyType *keys, ValueType *values, cudaStream_t stream = 0) override;
  void assign(const KeyType *keys, const ValueType *values, size_t num_keys,
              cudaStream_t stream = 0) override;

  void lookup(const KeyType *keys, ValueType *values, size_t num_keys,
              cudaStream_t stream = 0) override;
  void lookup(const KeyType *keys, ValueType **values, size_t num_keys,
              cudaStream_t stream = 0) override;
  void scatter_add(const KeyType *keys, const ValueType *values, size_t num_keys,
                   cudaStream_t stream = 0) override;
  void scatter_update(const KeyType *keys, const ValueType *values, size_t num_keys,
                      cudaStream_t stream = 0) override;

 private:
  using HKVTable = nv::merlin::HashTable<KeyType, ValueType, uint64_t>;
  std::unique_ptr<HKVTable> hkv_table_ = std::make_unique<HKVTable>();
  nv::merlin::HashTableOptions hkv_table_option_;

  size_t dimension_;
  size_t initial_capacity_;
  std::string initializer_;
  curandState *curand_states_;
  cudaStream_t stream_;
};

}  // namespace sok

#endif  // HKV_VARIABLE_H
