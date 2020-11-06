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
This op is used to do forward propagation.
It take a list of row_offsets, a list of value_tensors, nnz_array as inputs.
*/
template <typename Device>
class EmbeddingFpropV2Op : public OpKernel {
public:
    explicit EmbeddingFpropV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training_));
    }
    void Compute(OpKernelContext* ctx) override {
        OpInputList row_offsets_list;
        OP_REQUIRES_OK(ctx, ctx->input_list("row_offsets", &row_offsets_list));
        OpInputList value_tensors_list;
        OP_REQUIRES_OK(ctx, ctx->input_list("value_tensors", &value_tensors_list));
        const Tensor* nnz_array = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("nnz_array", &nnz_array));
        const Tensor* embedding_name = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("embedding_name", &embedding_name));
        std::string embedding_name_string(embedding_name->flat<tstring>()(0));

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
        OP_REQUIRES_OK(ctx, wrapper->get_output_tensor_shape(embedding_name_string, is_training_, shape));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &forward_result));

        /*forward propagation*/
        OP_REQUIRES_OK(ctx, wrapper->fprop_v2(row_offsets_list, value_tensors_list, nnz_array, 
                                              embedding_name_string, is_training_, forward_result));
    }
private:
    bool is_training_;
};

REGISTER_KERNEL_BUILDER(Name("HugectrEmbeddingFpropV2").Device(DEVICE_GPU), 
                        EmbeddingFpropV2Op<GPUDevice>);

} // namespace tensorflow