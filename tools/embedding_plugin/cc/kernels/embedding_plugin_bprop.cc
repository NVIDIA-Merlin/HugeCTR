/*
* Copyright (c) 2020, NVIDIA CORPORATION.
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

/**
This op is back propagation of embedding plugin.
It take top gradients (gathered from each output top_gradient) as input,
and will compute the embedding gradient, then update those parameters.
*/
template <typename Device>
class EmbeddingBpropOp : public OpKernel {
public:
    explicit EmbeddingBpropOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor* top_gradients = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("top_gradients", &top_gradients));
        const Tensor* embedding_name = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("embedding_name", &embedding_name));
        std::string embedding_name_(embedding_name->flat<tstring>()(0));

        /*do bprop*/
        bool on_gpu = false;
        if (std::is_same<Device, CPUDevice>::value) { 
            on_gpu = false;
        } else if (std::is_same<Device, GPUDevice>::value) { 
            on_gpu = true;
        }
        OP_REQUIRES(ctx, wrapper, errors::Unavailable(__FILE__, ":", __FILE__, " ",
                            "There is no wrapper instance, you should call hugectr.init() first."));
        OP_REQUIRES_OK(ctx, wrapper->bprop(embedding_name_, top_gradients, on_gpu));
    }
};

REGISTER_KERNEL_BUILDER(Name("HugectrEmbeddingBprop").Device(DEVICE_CPU), 
                        EmbeddingBpropOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HugectrEmbeddingBprop").Device(DEVICE_GPU), 
                        EmbeddingBpropOp<GPUDevice>);

} // namespace tensorflow