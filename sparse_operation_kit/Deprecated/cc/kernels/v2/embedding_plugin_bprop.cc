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
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice; 

template <typename Device>
class EmbeddingBpropOp : public OpKernel {
public:
    explicit EmbeddingBpropOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        /*get input tensor*/
        const Tensor* embedding_name_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("embedding_name", &embedding_name_tensor));
        std::string embedding_name(embedding_name_tensor->flat<tstring>()(0));
        const Tensor* replica_id_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("replica_id", &replica_id_tensor));
        const Tensor* replica_top_gradients_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("replica_top_gradients", &replica_top_gradients_tensor));
        const Tensor* bp_trigger_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("bp_trigger", &bp_trigger_tensor));
        
        /*set output*/
        Tensor* bp_trigger_grad_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({"bp_trigger"}, "bp_trigger_grad", 
                        bp_trigger_tensor->shape(), &bp_trigger_grad_tensor));

        /*do backward propagation*/
        const auto& tf_stream = ctx->eigen_gpu_device().stream();
        OP_REQUIRES(ctx, HugeCTR::Version2::wrapper, errors::Aborted(__FILE__, ":", __LINE__, " ",
                    "There is no wrapper instance, hugectr.init() should be called first."));
        OP_REQUIRES_OK(ctx, HugeCTR::Version2::wrapper->bprop(embedding_name, replica_id_tensor, 
                    replica_top_gradients_tensor, tf_stream));
    }   
};

REGISTER_KERNEL_BUILDER(Name("V2HugectrEmbeddingBprop").Device(DEVICE_GPU), 
                        EmbeddingBpropOp<GPUDevice>);

} // namespace tensorflow

