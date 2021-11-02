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

#ifdef ASYNC_OP
    // these headers are only needed in AsyncOpKernel
    #include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
    #include "tensorflow/stream_executor/cuda/cuda_activation.h"
#endif

#include <mpi.h>

#include <thread>
#include <mutex>
#include <chrono>
#include <type_traits>

#define CK_MPI(ctx, cmd)                                           \
    do {                                                            \
        auto retval = (cmd);                                        \
        if (MPI_SUCCESS != retval) {                                \
            (ctx)->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, \
                ": MPI error code ", std::to_string(retval)));      \
            return;                                                 \
        }                                                           \
    } while (0) 


#define CK_MPI_ASYNC(ctx, cmd, done)    \
    do {                                \
        _CK_MPI(ctx, cmd);              \
        (done)();                       \
    } while (0) 

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice; 

#ifdef ASYNC_OP
using ScopedActivateExecutorContext = stream_executor::cuda::ScopedActivateExecutorContext;

template <typename Device>
class TestOp : public AsyncOpKernel {
public:
    explicit TestOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx), mu_() {}
    void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
        auto work_func = [this, ctx, done]() {
            const Tensor* x_tensor = nullptr;
            OP_REQUIRES_OK_ASYNC(ctx, ctx->input("x", &x_tensor), done);

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

            int init_flag = 0;
            CK_MPI_ASYNC(ctx, MPI_Initialized(&init_flag), done);
            if (1 == init_flag) {
                std::cout << "\n[INFO]: MPI has been Initialized." << std::endl;

                // FIXME: why sleep to long will make program seg fault.
                // std::this_thread::sleep_for(std::chrono::seconds(3)); 

                // int rank = 0;
                // CK_MPI_ASYNC(ctx, MPI_Comm_rank(MPI_COMM_WORLD, &rank), done);
                // std::cout << "\n[INFO] " << std::this_thread::get_id() 
                //           << " rank is: " << rank << std::endl;
                // if (0 == rank) {
                //     std::this_thread::sleep_for(std::chrono::seconds(3));
                // }

                // std::cout << "\n[INFO] " << std::this_thread::get_id() << " entered barrier." << std::endl;
                // // MPI_Request request;
                // // CK_MPI_ASYNC(ctx, MPI_Ibarrier(MPI_COMM_WORLD, &request), done);
                // CK_MPI_ASYNC(ctx, MPI_Barrier(MPI_COMM_WORLD), done);
                // std::cout << "\n[INFO] " << std::this_thread::get_id() << " exit barrier." << std::endl;

            } else {
                std::cout << "\n[INFO]: MPI has not been Initialized." << std::endl;
            }

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
#else
template <typename Device>
class TestOp : public OpKernel {
public:
    explicit TestOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        const Tensor* x_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));
        Tensor* y_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x_tensor->shape(), &y_tensor));

        int init_flag = 0;
        CK_MPI(ctx, MPI_Initialized(&init_flag));
        if (1 == init_flag) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            std::cout << "\n[INFO]: MPI has been Initialized." << std::endl;

            int rank = 0;
            CK_MPI(ctx, MPI_Comm_rank(MPI_COMM_WORLD, &rank));
            if (0 == rank) {
                std::this_thread::sleep_for(std::chrono::seconds(3));
            }

            std::cout << "\n[INFO]: " << rank << " enter mpi barrier." << std::endl;
            CK_MPI(ctx, MPI_Barrier(MPI_COMM_WORLD));
            std::cout << "\n[INFO]: " << rank << " exit mpi barrier." << std::endl;

        } else {
            std::cout << "\n[INFO]: MPI has not been Initialized." << std::endl;
        }
    }
};
#endif

REGISTER_KERNEL_BUILDER(Name("Test").Device(DEVICE_GPU), 
                        TestOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("Test").Device(DEVICE_CPU),
                        TestOp<CPUDevice>);

} // tensorflow