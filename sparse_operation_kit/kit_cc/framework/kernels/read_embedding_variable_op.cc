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

#include "tensorflow/core/framework/op_kernel.h"
#include "embedding_variable.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice; 

class ReadEmbeddingVariableOp : public OpKernel {
public:
    explicit ReadEmbeddingVariableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
    }
    void Compute(OpKernelContext* ctx) override {
        // TODO: no need to read the resource handle??
        // core::RefCountPtr<EmbeddingVariable> variable;
        // const ResourceHandle& handle = HandleFromInput(ctx, 0);
        // auto status = LookupResource(ctx, handle, &variable);
        // OP_REQUIRES(ctx, status.ok(),
        //             errors::FailedPrecondition(
        //                 "Error while reading resource variable ", handle.name(),
        //                 " from Container: ", handle.container(),
        //                 ". This could mean that the variable was uninitialized. ",
        //                 status.ToString()));

        // // TODO: read TF Var handle
        // core::RefCountPtr<Var> tf_var;
        // const ResourceHandle& tf_handle = HandleFromInput(ctx, 1);
        // auto status = LookupResource(ctx, tf_handle, &tf_var);
        // OP_REQUIRES(ctx, status.ok(),
        //             errors::FailedPrecondition(
        //                 "Error while reading TF Var ", tf_handle.name(),
        //                 " from container: ", tf_handle.container(),
        //                 ". This could mean that you haven't create it. ",
        //                 status.ToString()));
        // Tensor* tf_tensor = tf_var->tensor();
        // float* host_buffer = nullptr;
        // std::cout << "[INFO]: tensor->ptr = " << tf_var->tensor()->data() << std::endl;
        // cudaMallocHost(&host_buffer, tf_tensor->NumElements() * sizeof(float), cudaHostAllocDefault);
        // cudaMemcpy(host_buffer, tf_tensor->data(), tf_tensor->NumElements() * sizeof(float),
        //             cudaMemcpyDefault);
        // // for (int i = 0; i < tf_tensor->NumElements(); i++) std::cout << host_buffer[i] << " ";
        // // std::cout << std::endl;
        // cudaFreeHost(host_buffer);

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &output_tensor));
    }
private:
    DataType dtype_;
};

REGISTER_KERNEL_BUILDER(Name("ReadEmbeddingVariableOp").Device(DEVICE_GPU)
                        .HostMemory("resource")
                        .HostMemory("tf_resource"),
                        ReadEmbeddingVariableOp);

} // namespace tensorflow