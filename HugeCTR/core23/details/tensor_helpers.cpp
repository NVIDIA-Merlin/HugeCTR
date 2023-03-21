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

#include <core23/buffer_requirements.hpp>
#include <core23/details/tensor_helpers.hpp>
#include <core23/logger.hpp>
#include <core23/tensor_params.hpp>

namespace HugeCTR {

namespace core23 {

namespace {

int64_t GetValidAlignment(int64_t alignment, const DataType& data_type) {
  const size_t size = data_type.size();
  if (alignment == 0 || alignment < size) {
    HCTR_LOG_S(WARNING, ROOT) << "alignment(" << alignment << ") is too small. size(" << size
                              << ") is used instead." << std::endl;
    alignment = size;
  } else {
    auto rem = alignment % size;
    if (rem != 0) {
      HCTR_LOG_S(WARNING, ROOT) << "alignment(" << alignment << ") is invalid. ";
      alignment += size;
      alignment -= rem;
      HCTR_LOG_S(WARNING, ROOT) << "It is adjusted to " << alignment << std::endl;
    }
    return alignment;
  }
  return alignment;
}

}  // namespace

BufferRequirements ConvertToBufferRequirements(const TensorParams& tensor_params) {
  BufferRequirements requirements = {
      .num_bytes = tensor_params.shape().size() * tensor_params.data_type().size(),
      .alignment = GetValidAlignment(tensor_params.alignment(), tensor_params.data_type()),
      .stream = tensor_params.stream()};
  return requirements;
}

}  // namespace core23
}  // namespace HugeCTR