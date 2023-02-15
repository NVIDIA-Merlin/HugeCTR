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

#pragma once

#include <core23/allocator.hpp>
#include <core23/allocator_params.hpp>
#include <core23/device.hpp>
#include <functional>
#include <memory>
#include <variant>

namespace HugeCTR {

namespace core23 {

[[nodiscard]] std::unique_ptr<Allocator> GetAllocator(const AllocatorParams& allocator_params,
                                                      const Device& device);

}  // namespace core23

}  // namespace HugeCTR
