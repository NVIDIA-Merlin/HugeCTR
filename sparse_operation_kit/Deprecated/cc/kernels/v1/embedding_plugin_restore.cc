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

#include "wrapper_variables.h"
#include "embedding_utils.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "cuda_utils.h"
#include <memory>
#include <type_traits>

#include <iostream>

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice; 

/*this op is used to restore embedding table parameters from file.*/
template <typename Device>
class EmbeddingRestoreOp : public OpKernel {
public:
    explicit EmbeddingRestoreOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        /*get inputs*/
        const Tensor* embedding_name_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("embedding_name", &embedding_name_tensor));
        embedding_name_ = std::string(embedding_name_tensor->flat<tstring>()(0));
        const Tensor* file_name_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("file_name", &file_name_tensor));
        file_name_ = std::string(file_name_tensor->flat<tstring>()(0));

        /*restore embedding params*/
        OP_REQUIRES(ctx, wrapper, errors::Unavailable(__FILE__, ":", __FILE__, " ",
                            "There is no wrapper instance, you should call hugectr.init() first."));
        OP_REQUIRES_OK(ctx, wrapper->restore(embedding_name_, file_name_));
    }
private:
    std::string embedding_name_;
    std::string file_name_;
};

REGISTER_KERNEL_BUILDER(Name("HugectrEmbeddingRestore").Device(DEVICE_CPU), EmbeddingRestoreOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HugectrEmbeddingRestore").Device(DEVICE_GPU), EmbeddingRestoreOp<GPUDevice>);

} // namespace tensorflow