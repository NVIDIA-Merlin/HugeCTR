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
#include "../v1/embedding_utils.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice; 

template <typename Device>
class BroadCastToAllDevicesOp : public AsyncOpKernel {
public:
    explicit BroadCastToAllDevicesOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training_));
    }
    void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
        /*get inputs*/
        const Tensor* embedding_name_tensor = nullptr;
        OP_REQUIRES_OK_ASYNC(ctx, ctx->input("embedding_name", &embedding_name_tensor), done);
        const Tensor* row_indices_tensor = nullptr;
        OP_REQUIRES_OK_ASYNC(ctx, ctx->input("row_indices", &row_indices_tensor), done);
        const Tensor* values_tensor = nullptr;
        OP_REQUIRES_OK_ASYNC(ctx, ctx->input("values", &values_tensor), done);

        /*allocate output*/
        OpOutputList output_list;
        OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("to_each_replica", &output_list), done);
        for (int i = 0; i < output_list.size(); ++i) {
            Tensor* temp = nullptr;
            OP_REQUIRES_OK_ASYNC(ctx, output_list.allocate(i, {1}, &temp), done);
        }

        /*do broadcasting*/
        auto broadcast_func = [this, ctx, done, embedding_name_tensor, 
                                row_indices_tensor, values_tensor]() -> void {
            OP_REQUIRES_ASYNC(ctx, HugeCTR::Version2::wrapper, errors::Aborted(__FILE__, ":", __LINE__, " ",
                            "wrapper does not exist, perhaps you should call init() first."), done);
            std::string embedding_name_string(embedding_name_tensor->flat<tstring>()(0));
            const auto& tf_stream = ctx->eigen_gpu_device().stream();
            OP_REQUIRES_OK_ASYNC(ctx, HugeCTR::Version2::wrapper->broadcast_then_convert_to_CSR(
                            row_indices_tensor, values_tensor, embedding_name_string, 
                            is_training_, tf_stream), done);
            done();
        };

        DeviceContext* device_ctx = ctx->op_device_context();
        se::Stream* stream = device_ctx->stream();
        DeviceBase* device = ctx->device();
        const DeviceBase::GpuDeviceInfo* device_info = device->tensorflow_gpu_device_info();
        device_info->event_mgr->ThenExecute(stream, std::move(broadcast_func));
    }
private:
    bool is_training_;
};

REGISTER_KERNEL_BUILDER(Name("V2HugectrBroadcastThenConvertToCSR").Device(DEVICE_GPU), 
                        BroadCastToAllDevicesOp<GPUDevice>);

} // namespace tensorflow