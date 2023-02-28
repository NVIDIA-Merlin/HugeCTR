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

// clang-format off
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"

#include "lookup/impl/core_impl/core23_allocator.hpp"
// clang-format on

namespace tensorflow {

class SetDefaultAllocatorOp : public OpKernel {
 public:
  explicit SetDefaultAllocatorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {  
     tf_internal::set_default_alloctor();
     Tensor* status_tensor = nullptr;
     OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &status_tensor));
     status_tensor->flat<tstring>()(0) = "OK";

  }

};

  REGISTER_KERNEL_BUILDER(                                                           
      Name("SetDefaultAllocator").Device(DEVICE_GPU), 
      SetDefaultAllocatorOp)


}  // namespace tensorflow
