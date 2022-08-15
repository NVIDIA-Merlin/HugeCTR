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

#include "variable/impl/det_variable.h"
#include "variable/impl/variable_base.h"

namespace sok {

template <typename KeyType, typename ValueType>
std::shared_ptr<VariableBase<KeyType, ValueType>> VariableFactory::create(
    int64_t rows, int64_t cols, const std::string &type, const std::string &initializer,
    cudaStream_t stream) {
  // if (type == "xxx") {
  //   return nullptr;
  // }

  // default type
  return std::make_shared<DETVariable<KeyType, ValueType>>(cols, 2E4, initializer, stream);
}

template std::shared_ptr<VariableBase<int32_t, float>> VariableFactory::create<int32_t, float>(
    int64_t rows, int64_t cols, const std::string &type, const std::string &initializer,
    cudaStream_t stream);

template std::shared_ptr<VariableBase<int64_t, float>> VariableFactory::create<int64_t, float>(
    int64_t rows, int64_t cols, const std::string &type, const std::string &initializer,
    cudaStream_t stream);

}  // namespace sok
