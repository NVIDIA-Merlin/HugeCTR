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


/* This op will convert input to CSR with forward propagation.
* Its inputs are values and row_indices from SparseTensor.
*/
template <typename Device>
class EmbeddingFpropV4Op : public OpKernel {
public:
    explicit EmbeddingFpropV4Op(OpKernelConstruction* ctx) : OpKernel(ctx){
        OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training_));
    }
    void Compute(OpKernelContext* ctx) override {
        /*get inputs*/
        const Tensor* row_indices_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("row_indices", &row_indices_tensor));
        const Tensor* values_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("values", &values_tensor));
        const Tensor* embedding_name_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("embedding_name", &embedding_name_tensor));
        std::string embedding_name_string(embedding_name_tensor->flat<tstring>()(0));
        OP_REQUIRES(ctx, row_indices_tensor->dtype() == DT_INT64, errors::InvalidArgument(__FILE__, ":", __LINE__, " ",
                                            "row_indices dtype should be int64."));
        OP_REQUIRES(ctx, row_indices_tensor->flat<long long>().size() == values_tensor->flat<long long>().size(), 
                        errors::InvalidArgument(__FILE__, ":", __LINE__, " ",
                                                "row_indices and values should have the same length."));
        
        /*bp_trigger input, just use to invoke computing grads*/
        const Tensor* bp_trigger = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("bp_trigger", &bp_trigger));
        auto bp_trigger_flat = bp_trigger->flat<float>();
        if (bp_trigger_flat.size() > 1) {
            LOG(WARNING) << "bp_trigger is just used to invoke the backprop op for embedding plugin," << 
                            " too much elements will waste spaces."; }

        /*allocate output*/
        Tensor* forward_result = nullptr;
        TensorShape shape;
        OP_REQUIRES(ctx, wrapper, errors::Unavailable(__FILE__, ":", __FILE__, " ",
                            "There is no wrapper instance, you should call hugectr.init() first."));
        OP_REQUIRES_OK(ctx, wrapper->get_output_tensor_shape(embedding_name_string, is_training_, shape));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &forward_result));

        /*do forward propagation*/
        OP_REQUIRES_OK(ctx, wrapper->fprop_v4(row_indices_tensor, values_tensor, embedding_name_string, 
                                              is_training_, forward_result));
    }

private:
    bool is_training_;

};


REGISTER_KERNEL_BUILDER(Name("HugectrEmbeddingFpropV4").Device(DEVICE_GPU), 
                        EmbeddingFpropV4Op<GPUDevice>);


} // namespace tensorflow