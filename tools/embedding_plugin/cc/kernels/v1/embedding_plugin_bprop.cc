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
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
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
class EmbeddingBpropOp : public AsyncOpKernel {
public:
    explicit EmbeddingBpropOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    }

    void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
        /*get input*/ 
        const Tensor* top_gradients = nullptr;
        OP_REQUIRES_OK_ASYNC(ctx, ctx->input("top_gradients", &top_gradients), done);
        const Tensor* embedding_name = nullptr;
        OP_REQUIRES_OK_ASYNC(ctx, ctx->input("embedding_name", &embedding_name), done);
        std::string embedding_name_(embedding_name->flat<tstring>()(0));
        const Tensor* bp_trigger = nullptr;
        OP_REQUIRES_OK_ASYNC(ctx, ctx->input("bp_trigger", &bp_trigger), done);

        /*allocate output*/
        Tensor* bp_trigger_grad = nullptr;
        OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, bp_trigger->shape(), &bp_trigger_grad), done);

        /*do bprop*/
        bool on_gpu = false;
        if (std::is_same<Device, CPUDevice>::value) { 
            on_gpu = false;
        } else if (std::is_same<Device, GPUDevice>::value) { 
            on_gpu = true;
        }
        OP_REQUIRES_ASYNC(ctx, wrapper, errors::Unavailable(__FILE__, ":", __FILE__, " ",
                            "There is no wrapper instance, you should call hugectr.init() first."), done);

        auto func = [this, ctx, embedding_name_, top_gradients, on_gpu, done]() -> void {
            const auto& tf_stream = ctx->eigen_gpu_device().stream();
            OP_REQUIRES_OK_ASYNC(ctx, wrapper->bprop(embedding_name_, top_gradients, 
                                                    on_gpu, tf_stream), done);
            done();
        };

        DeviceContext* device_ctx = ctx->op_device_context();
        se::Stream* stream = device_ctx->stream();
        DeviceBase* device = ctx->device();
        const DeviceBase::GpuDeviceInfo* device_info = device->tensorflow_gpu_device_info();
        device_info->event_mgr->ThenExecute(stream, std::move(func));
    }

private:
    std::vector<cudaEvent_t> wrapper_events_;
};

// REGISTER_KERNEL_BUILDER(Name("HugectrEmbeddingBprop").Device(DEVICE_CPU), 
                        // EmbeddingBpropOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HugectrEmbeddingBprop").Device(DEVICE_GPU), 
                        EmbeddingBpropOp<GPUDevice>);

} // namespace tensorflow