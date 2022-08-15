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

#ifndef DET_VARIABLE_H
#define DET_VARIABLE_H

#include <string>

#include "variable/impl/dynamic_embedding_table/cuCollections/include/cuco/dynamic_map.cuh"
#include "variable/impl/dynamic_embedding_table/cuCollections/include/cuco/initializer.cuh"
#include "variable/impl/variable_base.h"

namespace sok {

template <typename KeyType, typename ValueType>
class DETVariable : public VariableBase<KeyType, ValueType> {
 public:
  DETVariable(size_t dimension, size_t initial_capacity, const std::string &initializer,
              cudaStream_t stream = 0);

  ~DETVariable() override;
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
  std::unique_ptr<cuco::dynamic_map<KeyType, ValueType, cuco::initializer>> map_;

  size_t dimension_;
  size_t initial_capacity_;
  std::string initializer_;
  curandState *curand_states_;
};

}  // namespace sok

#endif  // DET_VARIABLE_H
