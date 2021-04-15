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
class EmbeddingSaveOp : public OpKernel {
public:
    explicit EmbeddingSaveOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        /*get inputs*/
        const Tensor* embedding_name_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("embedding_name", &embedding_name_tensor));
        std::string embedding_name_string(embedding_name_tensor->flat<tstring>()(0));
        const Tensor* save_name_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("save_name", &save_name_tensor));
        std::string save_name_string(save_name_tensor->flat<tstring>()(0));

        /*do saving*/
        OP_REQUIRES(ctx, HugeCTR::Version2::wrapper, errors::Aborted(__FILE__, ":", __LINE__, " ",
                    "There is no wrapper instance, hugectr.init() should be called first."));
        OP_REQUIRES_OK(ctx, HugeCTR::Version2::wrapper->save(embedding_name_string, save_name_string));
    } 
};

REGISTER_KERNEL_BUILDER(Name("V2HugectrEmbeddingSave").Device(DEVICE_GPU), 
                        EmbeddingSaveOp<GPUDevice>);

} // namespace tensorflow