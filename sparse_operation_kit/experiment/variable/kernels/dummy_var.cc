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

#include "variable/kernels/dummy_var.h"

namespace tensorflow {

template <typename KeyType, typename ValueType>
DummyVar<KeyType, ValueType>::DummyVar(int64_t rows, int64_t cols, const std::string& type,
                                       const std::string& initializer, const std::string& container,
                                       const std::string& name, cudaStream_t stream)
    : var_(nullptr), type_(type), container_(container), name_(name) {
  var_ = sok::VariableFactory::create<KeyType, ValueType>(rows, cols, type, initializer, stream);
}

template <typename KeyType, typename ValueType>
std::string DummyVar<KeyType, ValueType>::DebugString() const {
  return "DummyVar: " + container_ + "/" + name_;
}

template <typename KeyType, typename ValueType>
mutex* DummyVar<KeyType, ValueType>::mu() {
  return &mu_;
}

template <typename KeyType, typename ValueType>
void DummyVar<KeyType, ValueType>::check_var() {
  if (var_ == nullptr) {
    throw std::runtime_error("var_ of DummyVar is nullptr!");
  }
}

template <typename KeyType, typename ValueType>
int64_t DummyVar<KeyType, ValueType>::rows() {
  check_var();
  return var_->rows();
}

template <typename KeyType, typename ValueType>
int64_t DummyVar<KeyType, ValueType>::cols() {
  check_var();
  return var_->cols();
}

template <typename KeyType, typename ValueType>
void DummyVar<KeyType, ValueType>::Export(void* keys, void* values, cudaStream_t stream) {
  check_var();
  var_->eXport(static_cast<KeyType*>(keys), static_cast<ValueType*>(values), stream);
}

template <typename KeyType, typename ValueType>
void DummyVar<KeyType, ValueType>::Assign(const void* keys, const void* values, size_t num_keys,
                                          cudaStream_t stream) {
  check_var();
  var_->assign(static_cast<const KeyType*>(keys), static_cast<const ValueType*>(values), num_keys,
               stream);
}

template <typename KeyType, typename ValueType>
void DummyVar<KeyType, ValueType>::SparseRead(const void* keys, void* values, size_t num_keys,
                                              cudaStream_t stream) {
  check_var();
  var_->lookup(static_cast<const KeyType*>(keys), static_cast<ValueType*>(values), num_keys,
               stream);
}

template <typename KeyType, typename ValueType>
void DummyVar<KeyType, ValueType>::ScatterAdd(const void* keys, const void* values, size_t num_keys,
                                              cudaStream_t stream) {
  check_var();
  var_->scatter_add(static_cast<const KeyType*>(keys), static_cast<const ValueType*>(values),
                    num_keys, stream);
}

template <typename KeyType, typename ValueType>
void DummyVar<KeyType, ValueType>::ScatterUpdate(const void* keys, const void* values,
                                                 size_t num_keys, cudaStream_t stream) {
  check_var();
  var_->scatter_update(static_cast<const KeyType*>(keys), static_cast<const ValueType*>(values),
                       num_keys, stream);
}

// explicit instance the template
template class DummyVar<int32_t, float>;
template class DummyVar<int64_t, float>;

}  // namespace tensorflow
