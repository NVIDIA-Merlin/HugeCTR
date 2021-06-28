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

#include "tensorflow/core/framework/op_kernel.h"
#include "threading_utils.h"
#include <iostream>

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice; 

struct Test {
    void work_func() {
        std::cout << __FILE__ << ":" << __LINE__ << " work function is called." << std::endl;
        // throw std::runtime_error("work func throw error.");
    }
};

Test t;

HugeCTR::Version2::BlockingCallOnce Call_once(8);

template <typename Device>
class ThreadingTestOp : public OpKernel {
public:
    explicit ThreadingTestOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) {
        const Tensor* input_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {1}, &output_tensor));
        try {
            Call_once(&Test::work_func, &t);
        } catch (const std::exception& error) {
            ctx->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                    error.what()));
            return;
        }
        
    }
};

REGISTER_KERNEL_BUILDER(Name("V2ThreadingTest").Device(DEVICE_GPU), ThreadingTestOp<GPUDevice>);

} // namespace tensorflow