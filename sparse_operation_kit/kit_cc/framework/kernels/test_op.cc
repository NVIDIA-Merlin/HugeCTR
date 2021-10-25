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

#include "tensorflow/core/framework/op_kernel.h"

// these headers are only needed in AsyncOpKernel
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"

#include <thread>
#include <mutex>
#include <chrono>
#include <type_traits>

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice; 

using ScopedActivateExecutorContext = stream_executor::cuda::ScopedActivateExecutorContext;

template <typename Device>
class TestOp : public AsyncOpKernel {
public:
    explicit TestOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx), mu_() {}
    void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
        const Tensor* x_tensor = nullptr;
        OP_REQUIRES_OK_ASYNC(ctx, ctx->input("x", &x_tensor), done);

        {
            std::cout << "\n[INFO]: x_tensor.NumElements = " << x_tensor->NumElements() << std::endl;
            std::cout << "\n[INFO]: this_thread_id = " << std::this_thread::get_id() << "\n" << std::endl;    
        }

        auto work_func = [this, ctx, done, x_tensor]() {
            if (std::is_same<Device, CPUDevice>::value) {
                // did no thing
            } else if (std::is_same<Device, GPUDevice>::value) {
                // Ensure that within the callback, the proper GPU settings are
                // configured.
                auto stream = ctx->op_device_context()->stream();
                ScopedActivateExecutorContext scoped_activation{stream->parent()};
            } else {
                ctx->SetStatus(errors::Aborted("Not supported Device Type"));
                done();
                return;
            }

            std::this_thread::sleep_for(std::chrono::seconds(3));

            // *y_tensor = *x_tensor;

            {
                std::cout << "\n[INFO]: work_func is called." << std::endl;
                std::cout << std::this_thread::get_id() << std::endl;
            }

            Tensor* y_tensor = nullptr;
            OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, x_tensor->shape(), &y_tensor), done);
            
            done();

            {
                std::cout << "\n[INFO]: after done() is called." << std::endl;
                std::cout << std::this_thread::get_id() << std::endl;
            }
        };

        if (std::is_same<Device, CPUDevice>::value) {
            ctx->device()->tensorflow_cpu_worker_threads()->workers->Schedule(std::move(work_func));
        } else if (std::is_same<Device, GPUDevice>::value) {
            auto stream = ctx->op_device_context()->stream();
            ctx->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(stream, std::move(work_func));
        } else {
            ctx->SetStatus(errors::Aborted("Not supported Device Type"));
            done();
            return;
        }

        {
            std::cout << "\n[INFO]: TestOp End" << "\n" << std::endl;
            std::cout << "\n[INFO]: this_thread_id = " << std::this_thread::get_id() << "\n" << std::endl;    
        }
    }
private:
    std::mutex mu_;
};

REGISTER_KERNEL_BUILDER(Name("Test").Device(DEVICE_GPU), 
                        TestOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("Test").Device(DEVICE_CPU),
                        TestOp<CPUDevice>);

} // tensorflow