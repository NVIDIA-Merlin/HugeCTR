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
#pragma once

#include <cuda_runtime.h>

// For the tensor bags
#include <atomic>
#include <vector>

#include "HugeCTR/core23/tensor.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {

class RawPtrWrapper : public TensorBuffer2 {
 public:
  RawPtrWrapper(void* ptr) : ptr_(ptr) {}
  bool allocated() const override { return true; }
  void* get_ptr() override { return ptr_; }

 private:
  void* ptr_;
};
}  // namespace HugeCTR