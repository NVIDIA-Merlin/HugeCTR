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
#include <gpu_resource.hpp>
#include <memory>

namespace HugeCTR {

/**
 * @brief Top-level ResourceManager interface
 *
 * The top level resource manager interface shared by various components
 */
class ResourceManagerBase {
 public:
  virtual void set_local_gpu(std::shared_ptr<GPUResource> gpu_resource, size_t local_gpu_id) = 0;
  virtual const std::shared_ptr<GPUResource>& get_local_gpu(size_t local_gpu_id) const = 0;
  virtual size_t get_local_gpu_count() const = 0;
  virtual size_t get_global_gpu_count() const = 0;
};

}  // namespace HugeCTR
