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


/*This op is used to save embedding table's parameters to file.*/
template <typename Device>
class EmbeddingSaveOp : public OpKernel {
public:
    explicit EmbeddingSaveOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        /*get inputs*/
        const Tensor* embedding_name_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("embedding_name", &embedding_name_tensor));
        embedding_name_ = std::string(embedding_name_tensor->flat<tstring>()(0));
        const Tensor* save_name_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("save_name", &save_name_tensor));
        save_name_ = std::string(save_name_tensor->flat<tstring>()(0));

        /*save embedding params*/
        OP_REQUIRES(ctx, wrapper, errors::Unavailable(__FILE__, ":", __FILE__, " ",
                            "There is no wrapper instance, you should call hugectr.init() first."));
        OP_REQUIRES_OK(ctx, wrapper->save(embedding_name_, save_name_));
    }
private:
    std::string embedding_name_;
    std::string save_name_;
};

REGISTER_KERNEL_BUILDER(Name("HugectrEmbeddingSave").Device(DEVICE_CPU), EmbeddingSaveOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HugectrEmbeddingSave").Device(DEVICE_GPU), EmbeddingSaveOp<GPUDevice>);


} // namespace tensorflow