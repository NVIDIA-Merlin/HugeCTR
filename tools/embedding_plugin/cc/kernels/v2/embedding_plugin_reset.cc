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

/*
* This op is used to explicitly release resources created by hugectr_tf_ops.
*/
template <typename Device>
class ResetOp : public OpKernel {
public:
    explicit ResetOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        if (!HugeCTR::Version2::wrapper) {
            LOG(INFO) << "wrapper is not existed.";
        } else {
            HugeCTR::Version2::wrapper.reset(nullptr);
            LOG(INFO) << "embedding_plugin released resources.";
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("V2HugectrReset").Device(DEVICE_CPU), ResetOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("V2HugectrReset").Device(DEVICE_GPU), ResetOp<GPUDevice>);

} // namespace tensorflow