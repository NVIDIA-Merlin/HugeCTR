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

namespace {
    std::once_flag WARNING_ONCE_FLAG;
}

template <typename Device>
class EmbeddingFpropV1Op : public OpKernel {
public:
    explicit EmbeddingFpropV1Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("input_buffer_reset", &input_buffer_reset_));

        std::call_once(WARNING_ONCE_FLAG, &EmbeddingFpropV1Op<Device>::warning_func, this);
    }
    void Compute(OpKernelContext* ctx) override {
        /*get input tensor*/
        const Tensor* embedding_name_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("embedding_name", &embedding_name_tensor));
        std::string embedding_name(embedding_name_tensor->flat<tstring>()(0));
        const Tensor* replica_id_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("replica_id", &replica_id_tensor));
        const Tensor* row_offset_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("row_offset", &row_offset_tensor));
        const Tensor* values_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("values", &values_tensor));
        const Tensor* nnz_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("nnz", &nnz_tensor));
        const Tensor* bp_trigger_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("bp_trigger", &bp_trigger_tensor));
        auto bp_trigger_flat = bp_trigger_tensor->flat<float>();
        if (bp_trigger_flat.size() > 1) {
            LOG(WARNING) << "bp_trigger is just used to invoke the backprop op for embedding plugin," << 
                            " too much elements will waste spaces."; 
        }
        const auto& tf_stream = ctx->eigen_gpu_device().stream();

        /*allocate output tensor*/
        Tensor* replica_forward_result_tensor = nullptr;
        TensorShape replica_forward_result_shape;
        OP_REQUIRES(ctx, HugeCTR::Version2::wrapper, errors::Aborted(__FILE__, ":", __LINE__, " ",
                    "There is no wrapper instance, hugectr.init() should be called first."));
        OP_REQUIRES_OK(ctx, HugeCTR::Version2::wrapper->get_replica_forward_result_shape(embedding_name, is_training_,
                    replica_forward_result_shape));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, replica_forward_result_shape, &replica_forward_result_tensor));
        
        /*do forward propagation*/
        OP_REQUIRES_OK(ctx, HugeCTR::Version2::wrapper->fprop_v1(embedding_name, is_training_,
                    replica_id_tensor, row_offset_tensor, values_tensor, nnz_tensor, tf_stream,
                    input_buffer_reset_, replica_forward_result_tensor));
    }

private:
    bool is_training_;
    bool input_buffer_reset_;
    void warning_func() {
        if (input_buffer_reset_) {
            LOG(WARNING) << "Input buffer will be reset after each forward propagation.";
        } else {
            LOG(WARNING) << "Input buffer will not be reset after each forward propagation." << 
                            " Must make sure the lengeh of the inputs is >= to that of the input buffer." <<
                            " Otherwise, this op's result may not be correct.";
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("V2HugectrEmbeddingFpropV1").Device(DEVICE_GPU), 
                        EmbeddingFpropV1Op<GPUDevice>);

} // namespace tensorflow