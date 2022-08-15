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

#ifndef DUMMY_VAR_H
#define DUMMY_VAR_H

#include <memory>
#include <string>

#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
// #include "tensorflow/stream_executor/stream.h"
// namespace stream_executor {
// class Stream;
// }  // namespace stream_executor

#include "variable/impl/variable_base.h"

namespace tensorflow {

// This class does not hold the actual data, it just holds a pointer
// to VariableBase(variable/impl/variable_base.h).
//
// It should be the only interface for tensorflow op to access the code
// under variable/impl/, to isolate the sok:: code from tensorflow:: code.
template <typename KeyType, typename ValueType>
class DummyVar : public ResourceBase {
 public:
  DummyVar(int64_t rows, int64_t cols, const std::string &type, const std::string &initializer,
           const std::string &container, const std::string &name, cudaStream_t stream);
  ~DummyVar() = default;

  std::string DebugString() const override;
  mutex *mu();

  int64_t rows();
  int64_t cols();

  void Export(void *keys, void *values, cudaStream_t stream);
  void Assign(const void *keys, const void *values, size_t num_keys, cudaStream_t stream);

  void SparseRead(const void *keys, void *values, size_t num_keys, cudaStream_t stream);
  void ScatterAdd(const void *keys, const void *values, size_t num_keys, cudaStream_t stream);
  void ScatterUpdate(const void *keys, const void *values, size_t num_keys, cudaStream_t stream);

  inline std::shared_ptr<sok::VariableBase<KeyType, ValueType>> get_var() { return var_; }

 private:
  std::shared_ptr<sok::VariableBase<KeyType, ValueType>> var_;
  std::string type_;
  std::string container_;
  std::string name_;
  mutex mu_;

  void check_var();
};

}  // namespace tensorflow

#endif  // DUMMY_VAR_H
