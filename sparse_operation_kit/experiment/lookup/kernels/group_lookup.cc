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
#include <vector>

#include <cuda_fp16.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"

#include "common/check.h"
#include "lookup/impl/group_lookup.h"
// clang-format on

namespace stream_executor {
namespace gpu {
cudaStream_t AsGpuStreamValue(Stream* stream);
}  // namespace gpu
}  // namespace stream_executor

namespace tensorflow {

template <typename KeyType, typename DType>
class GroupLookupOp : public OpKernel {
 public:
  explicit GroupLookupOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
    tasks_.resize(N_);
    launcher_.initialize(N_);
  }

  void Compute(OpKernelContext* ctx) override {
    std::vector<tf_shared_lock> locks;
    for (int i = 0; i < N_; ++i) {
      // step 1/5: Get tasks_[i].dimension
      auto handle = HandleFromInput(ctx, i);
      auto dtypes_and_shapes = handle.dtypes_and_shapes();
      auto shape = dtypes_and_shapes[0].shape;
      OP_REQUIRES(ctx, dtypes_and_shapes[0].dtype == DataType::DT_FLOAT,
                  errors::InvalidArgument("Type of variable must be float."));
      tasks_[i].dimension = shape.dim_size(1);

      // step 2/5: Get tasks_[i].input
      core::RefCountPtr<Var> var;
      OP_REQUIRES_OK(ctx, LookupResource(ctx, handle, &var));
      float* input = var->tensor()->flat<float>().data();
      bool is_unique = true;
      for (int j = 0; j < i; ++j) {
        if (input == tasks_[j].input) {
          is_unique = false;
          break;
        }
      }
      if (is_unique) {
        tf_shared_lock lock(*var->mu());
        locks.push_back(std::move(lock));
      }
      tasks_[i].input = input;

      // step 3/5: Get tasks_[i].key
      Tensor indices = ctx->input(N_ + i);
      tasks_[i].key = indices.data();

      // step 4/5: Get tasks_[i].num_keys
      int64_t num_indices = indices.NumElements();
      tasks_[i].num_keys = num_indices;

      // step 5/5: Get tasks_[i].output
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, {num_indices, shape.dim_size(1)}, &output));
      tasks_[i].output = output->data();
    }

    // Get compute stream of tensorflow
    auto device_ctx = ctx->op_device_context();
    OP_REQUIRES(ctx, device_ctx != nullptr, errors::Aborted("No valid device context."));
    auto stream = stream_executor::gpu::AsGpuStreamValue(device_ctx->stream());

    // Launch cuda kernel
    launcher_(tasks_, stream);
  }

 private:
  // The number of single lookup operations
  int N_;
  std::vector<sok::LookupTask<KeyType, DType>> tasks_;
  sok::LookupLauncher<KeyType, DType> launcher_;
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, dtype_tf, dtype)   \
  REGISTER_KERNEL_BUILDER(Name("GroupLookup")                          \
                              .Device(DEVICE_GPU)                      \
                              .HostMemory("handles")                   \
                              .TypeConstraint<key_type_tf>("Tindices") \
                              .TypeConstraint<dtype_tf>("dtype"),      \
                          GroupLookupOp<key_type, dtype>)

#if TF_VERSION_MAJOR == 1
REGISTER_GPU_KERNELS(int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, float, float);
REGISTER_GPU_KERNELS(int64, int64_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int32, int32_t, Eigen::half, __half);
#else
REGISTER_GPU_KERNELS(int64_t, int64_t, float, float);
REGISTER_GPU_KERNELS(int32_t, int32_t, float, float);
REGISTER_GPU_KERNELS(int64_t, int64_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int32_t, int32_t, Eigen::half, __half);
#endif

#undef REGISTER_GPU_KERNELS

}  // namespace tensorflow
