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
#include "tensorflow/core/framework/resource_var.h"
#include <exception>

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice; 

template <typename Device>
class RestoreFromFileOp : public OpKernel {
public:
    explicit RestoreFromFileOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        const Tensor* var_handle_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("var_handle", &var_handle_tensor));
        const Tensor* filename_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("filename", &filename_tensor));

        Tensor* status_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &status_tensor));

        try {
            SparseOperationKit::Facade::instance()->restore_from_file(var_handle_tensor, 
                                            filename_tensor->flat<tstring>()(0));
        } catch (const std::exception& error) {
            ctx->SetStatus(errors::Aborted(error.what()));
            return;
        }
        status_tensor->flat<tstring>()(0) = "restored.";
    }
};

REGISTER_KERNEL_BUILDER(Name("RestoreFromFile")
                        .Device(DEVICE_CPU),
                        RestoreFromFileOp<CPUDevice>);

} // namespace tensorflow