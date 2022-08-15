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

#include "lookup/impl/select_kernel.h"
// clang-format on

namespace stream_executor {
namespace gpu {
cudaStream_t AsGpuStreamValue(Stream* stream);
}  // namespace gpu
}  // namespace stream_executor

namespace tensorflow {

template <typename KeyType>
class DistSelectOp : public OpKernel {
 public:
  explicit DistSelectOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_splits", &num_splits_));
    launcher_.initialize(num_splits_);
  }

  void Compute(OpKernelContext* ctx) override {
    // input
    const Tensor* indices = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("indices", &indices));
    int64_t num_keys = indices->NumElements();

    // output
    Tensor* output = nullptr;
    Tensor* order = nullptr;
    Tensor* splits = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {num_keys}, &output));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {num_keys}, &order));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {num_splits_}, &splits));

    // temp buffer
    Tensor output_buffer;
    Tensor order_buffer;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(indices->dtype(), {num_keys * num_splits_}, &output_buffer));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, {num_keys * num_splits_}, &order_buffer));

    // stream
    auto device_ctx = ctx->op_device_context();
    OP_REQUIRES(ctx, device_ctx != nullptr, errors::Aborted("No valid device context."));
    cudaStream_t stream = stream_executor::gpu::AsGpuStreamValue(device_ctx->stream());

    // cuda kernel
    launcher_(indices->data(), num_keys, output->data(), output_buffer.data(), order->data(),
              order_buffer.data(), splits->data(), num_splits_, stream);
  }

 private:
  // num_splits_ means the number of GPUs
  int num_splits_;
  sok::SelectLauncher<KeyType> launcher_;
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type)                                  \
  REGISTER_KERNEL_BUILDER(                                                           \
      Name("DistSelect").Device(DEVICE_GPU).TypeConstraint<key_type_tf>("Tindices"), \
      DistSelectOp<key_type>)

#if TF_VERSION_MAJOR == 1
REGISTER_GPU_KERNELS(int64, int64_t);
REGISTER_GPU_KERNELS(int32, int32_t);
#else
REGISTER_GPU_KERNELS(int64_t, int64_t);
REGISTER_GPU_KERNELS(int32_t, int32_t);
#endif

#undef REGISTER_GPU_KERNELS

}  // namespace tensorflow
