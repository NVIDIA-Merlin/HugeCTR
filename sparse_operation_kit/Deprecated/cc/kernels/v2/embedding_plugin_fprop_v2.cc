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
#include "tensorflow/core/framework/op_kernel.h"
#include <mutex>

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice; 

template <typename Device>
class EmbeddingFpropV2Op : public OpKernel {
public:
    explicit EmbeddingFpropV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training_));
    }
    void Compute(OpKernelContext* ctx) override {
        /*get inputs*/
        const Tensor* embedding_name_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("embedding_name", &embedding_name_tensor));
        const Tensor* replica_id_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("replica_id", &replica_id_tensor));
        const Tensor* to_each_replica_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("to_each_replica", &to_each_replica_tensor));
        const Tensor* bp_trigger_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("bp_trigger", &bp_trigger_tensor));
        if (bp_trigger_tensor->NumElements() > 1) {
            LOG(WARNING) << "bp_trigger is just used to invoke the backprop op for embedding plugin," << 
                            " too much elements will waste spaces."; 
        }
        const auto& tf_stream = ctx->eigen_gpu_device().stream();

        /*allocate output tensor*/
        Tensor* replica_forward_result_tensor = nullptr;
        TensorShape replica_forward_result_shape;
        OP_REQUIRES(ctx, HugeCTR::Version2::wrapper, errors::Aborted(__FILE__, ":", __LINE__, " ",
                    "There is no wrapper instance, hugectr.init() should be called first."));
        std::string embedding_name_string(embedding_name_tensor->flat<tstring>()(0));
        OP_REQUIRES_OK(ctx, HugeCTR::Version2::wrapper->get_replica_forward_result_shape(embedding_name_string, is_training_,
                    replica_forward_result_shape));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, replica_forward_result_shape, &replica_forward_result_tensor));

        /*do forward propagation*/
        OP_REQUIRES_OK(ctx, HugeCTR::Version2::wrapper->fprop_v2(embedding_name_string, is_training_,
                    replica_id_tensor, tf_stream, replica_forward_result_tensor));
    }
private:
    bool is_training_;
};

REGISTER_KERNEL_BUILDER(Name("V2HugectrEmbeddingFpropV2").Device(DEVICE_GPU), 
                        EmbeddingFpropV2Op<GPUDevice>);

} // namespace tensorflow