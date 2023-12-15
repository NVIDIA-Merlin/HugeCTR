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
#include <cuda_fp16.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"

#include "lookup/impl/reorder_kernel.h"
// clang-format on

namespace stream_executor {
namespace gpu {
cudaStream_t AsGpuStreamValue(Stream* stream);
}  // namespace gpu
}  // namespace stream_executor

namespace tensorflow {

template <typename DType>
class ReorderOp : public OpKernel {
 public:
  explicit ReorderOp(OpKernelConstruction* ctx) : OpKernel(ctx) { launcher_.initialize(); }

  void Compute(OpKernelContext* ctx) override {
    // input
    const Tensor* embedding = nullptr;
    const Tensor* order = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("embedding", &embedding));
    OP_REQUIRES_OK(ctx, ctx->input("order", &order));
    int64_t num_keys = embedding->dim_size(0);
    int64_t embedding_dimension = embedding->dim_size(1);
    int64_t num_orders = order->dim_size(0);
    OP_REQUIRES(ctx, num_keys == num_orders,
                errors::InvalidArgument("embedding.shape[0] != order.shape[0]."));

    // output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {num_keys, embedding_dimension}, &output));

    // stream
    auto device_ctx = ctx->op_device_context();
    OP_REQUIRES(ctx, device_ctx != nullptr, errors::Aborted("No valid device context."));
    cudaStream_t stream = stream_executor::gpu::AsGpuStreamValue(device_ctx->stream());

    // cuda kernel
    launcher_(embedding->data(), num_keys, embedding_dimension, order->data(), output->data(),
              stream);
  }

 private:
  sok::ReorderLauncher<DType> launcher_;
};

#define REGISTER_GPU_KERNELS(dtype_tf, dtype)                                                   \
  REGISTER_KERNEL_BUILDER(Name("Reorder").Device(DEVICE_GPU).TypeConstraint<dtype_tf>("dtype"), \
                          ReorderOp<dtype>)

REGISTER_GPU_KERNELS(float, float);
REGISTER_GPU_KERNELS(Eigen::half, __half);

#undef REGISTER_GPU_KERNELS

template <typename DType>
class GatherExOp : public OpKernel {
 public:
  explicit GatherExOp(OpKernelConstruction* ctx) : OpKernel(ctx) { launcher_.initialize(); }

  void Compute(OpKernelContext* ctx) override {
    // input
    const Tensor* grads = nullptr;
    const Tensor* indices = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("grads", &grads));
    OP_REQUIRES_OK(ctx, ctx->input("indices", &indices));
    int64_t num_keys = grads->dim_size(0);
    int64_t embedding_dimension = grads->dim_size(1);
    int64_t num_indices = indices->dim_size(0);
    OP_REQUIRES(ctx, num_keys == num_indices,
                errors::InvalidArgument("grads.shape[0] != indices.shape[0]."));

    // output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {num_keys, embedding_dimension}, &output));

    // stream
    auto device_ctx = ctx->op_device_context();
    OP_REQUIRES(ctx, device_ctx != nullptr, errors::Aborted("No valid device context."));
    cudaStream_t stream = stream_executor::gpu::AsGpuStreamValue(device_ctx->stream());

    // cuda kernel
    launcher_(grads->data(), num_keys, embedding_dimension, indices->data(), output->data(),
              stream);
  }

 private:
  sok::GatherExLauncher<DType> launcher_;
};

#define REGISTER_GPU_KERNELS(dtype_tf, dtype)                                                    \
  REGISTER_KERNEL_BUILDER(Name("GatherEx").Device(DEVICE_GPU).TypeConstraint<dtype_tf>("dtype"), \
                          GatherExOp<dtype>)

REGISTER_GPU_KERNELS(float, float);
REGISTER_GPU_KERNELS(Eigen::half, __half);

#undef REGISTER_GPU_KERNELS

}  // namespace tensorflow
