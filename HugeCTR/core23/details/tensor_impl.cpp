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

#include <core23/allocator_factory.hpp>
#include <core23/buffer_factory.hpp>
#include <core23/cuda_stream.hpp>
#include <core23/details/tensor_helpers.hpp>
#include <core23/details/tensor_impl.hpp>
#include <core23/device.hpp>
#include <core23/offsetted_buffer.hpp>
#include <memory>

namespace HugeCTR {

namespace core23 {

TensorImpl::TensorImpl(TensorParams params) : params_(params) {
  auto buffer = GetBuffer(params.buffer_params(), params.device(),
                          GetAllocator(params.allocator_params(), params.device()));
  auto buffer_requirements = ConvertToBufferRequirements(params);
  buffer->subscribe(this, buffer_requirements);
}

void* TensorImpl::data() const {
  if (offsetted_buffer()) {
    return offsetted_buffer()->data();
  }
  return bound_data_ ? bound_data_.value() : nullptr;
}

}  // namespace core23

}  // namespace HugeCTR