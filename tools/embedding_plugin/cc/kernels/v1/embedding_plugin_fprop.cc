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



/*This op will do (distribute_keys_on_CPU + embedding_fprop)
It take original keys (in SparseTensor format) as input, and will convert keys 
to CSR format when calling embedding fprop. And this op will distribute keys on 
CPU. 
Though it is very slow. But can be used to do the accuracy verification.
*/
template <typename Device>
class EmbeddingFpropOp : public OpKernel {
public:
    explicit EmbeddingFpropOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training_));
    }

    void Compute(OpKernelContext* ctx) override {
        /*get inputs*/
        const Tensor* sparse_indices = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("sparse_indices", &sparse_indices));
        const Tensor* values = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("values", &values));
        const Tensor* dense_shape = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("dense_shape", &dense_shape));
        const Tensor* embedding_name = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("embedding_name", &embedding_name));
        std::string embedding_name_(embedding_name->flat<tstring>()(0));

        /*bp_trigger input, just use to invoke computing grads*/
        const Tensor* bp_trigger = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("bp_trigger", &bp_trigger));
        auto bp_trigger_flat = bp_trigger->flat<float>();
        if (bp_trigger_flat.size() > 1) {
            LOG(WARNING) << "bp_trigger is just used to invoke the backprop op for embedding plugin," << 
                            " too much elements will waste spaces."; }

        /*allocate output spaces*/
        Tensor* forward_result = nullptr;
        TensorShape shape;
        OP_REQUIRES(ctx, wrapper, errors::Unavailable(__FILE__, ":", __FILE__, " ",
                            "There is no wrapper instance, you should call hugectr.init() first."));
        OP_REQUIRES_OK(ctx, wrapper->get_output_tensor_shape(embedding_name_, is_training_, shape));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &forward_result));

        /*forward propagation*/
        bool on_gpu = false;
        if (std::is_same<Device, CPUDevice>::value) {
            on_gpu = false;
        } else if (std::is_same<Device, GPUDevice>::value) {
            on_gpu = true;
        }
        OP_REQUIRES_OK(ctx, wrapper->fprop(sparse_indices, values, dense_shape, 
                                           embedding_name_, is_training_, forward_result, on_gpu));
    }

private:
    bool is_training_;
};

REGISTER_KERNEL_BUILDER(Name("HugectrEmbeddingFprop").Device(DEVICE_CPU), 
                        EmbeddingFpropOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HugectrEmbeddingFprop").Device(DEVICE_GPU), 
                        EmbeddingFpropOp<GPUDevice>);

} // namespace tensorflow