/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "facade.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <exception>

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice; 

template <typename Device>
class CreateGlobalAdamOptimizerOp : public OpKernel {
public:
    explicit CreateGlobalAdamOptimizerOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("beta1", &beta1_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("beta2", &beta2_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    }
    void Compute(OpKernelContext* ctx) override {
        Tensor* optimizer_handle_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &optimizer_handle_tensor));

        try {
            SparseOperationKit::Facade::instance()->create_optimizer(/*optimizer_type=*/"Adam",
                                                                  /*optimizer_handle=*/optimizer_handle_tensor,
                                                                  /*hyper_params=*/{{"beta1", beta1_},
                                                                                    {"beta2", beta2_},
                                                                                    {"epsilon", epsilon_}});
        } catch (const std::exception& error) {
            ctx->SetStatus(errors::Aborted(error.what()));
            return;
        }
    }
private:
    float beta1_;
    float beta2_;
    float epsilon_;
};

REGISTER_KERNEL_BUILDER(Name("CreateGlobalAdamOptimizer")
                        .Device(DEVICE_GPU)
                        .HostMemory("optimizer_handle"),
                        CreateGlobalAdamOptimizerOp<GPUDevice>);


} // namespace tensorflow