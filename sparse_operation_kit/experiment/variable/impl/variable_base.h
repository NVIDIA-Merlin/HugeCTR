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

#ifndef VARIABLE_BASE_H
#define VARIABLE_BASE_H

#include <cuda_runtime_api.h>

#include <memory>
#include <string>

namespace sok {

template <typename KeyType, typename ValueType>
class VariableBase {
 public:
  virtual ~VariableBase() = default;
  virtual int64_t rows() = 0;
  virtual int64_t cols() = 0;

  virtual void eXport(KeyType *keys, ValueType *values, cudaStream_t stream = 0) = 0;
  virtual void assign(const KeyType *keys, const ValueType *values, size_t num_keys,
                      cudaStream_t stream = 0) = 0;

  virtual void lookup(const KeyType *keys, ValueType *values, size_t num_keys,
                      cudaStream_t stream = 0) = 0;
  virtual void lookup(const KeyType *keys, ValueType **values, size_t num_keys,
                      cudaStream_t stream = 0) = 0;
  virtual void scatter_add(const KeyType *keys, const ValueType *values, size_t num_keys,
                           cudaStream_t stream = 0) = 0;
  virtual void scatter_update(const KeyType *keys, const ValueType *values, size_t num_keys,
                              cudaStream_t stream = 0) = 0;
};

class VariableFactory {
 public:
  template <typename KeyType, typename ValueType>
  static std::shared_ptr<VariableBase<KeyType, ValueType>> create(int64_t rows, int64_t cols,
                                                                  const std::string &type,
                                                                  const std::string &initializer,
                                                                  cudaStream_t stream = 0);
};

}  // namespace sok

#endif  // VARIABLE_BASE_H
