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
#include "embedding_utils.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "cuda_utils.h"
#include <memory>
#include <type_traits>

#include <iostream>

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice; 

/*
This op is used to create part of HugeCTR's solver/session.
*/
template <typename Device>
class InitOp : public OpKernel {
public:
    explicit InitOp(OpKernelConstruction* ctx) : OpKernel(ctx){
        OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("key_type", &key_type));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("value_type", &value_type));
        OP_REQUIRES(ctx, Utils::check_in_set(KEY_TYPE_SET, key_type),
                errors::InvalidArgument("key_type must be {uint32, int64}."));
        OP_REQUIRES(ctx, Utils::check_in_set(VALUE_TYPE_SET, value_type),
                errors::InvalidArgument("value_type must be {float, half}."));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
        OP_REQUIRES(ctx, batch_size_ > 0, errors::InvalidArgument(__FILE__, ":", __LINE__, " ",
                                                "batch_size should be > 0, but got ", batch_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size_eval", &batch_size_eval_));
        OP_REQUIRES(ctx, batch_size_eval_ > 0, errors::InvalidArgument(__FILE__, ":", __LINE__, " ",
                                                "batch_size_eval should be > 0, but got ", batch_size_eval_));

    }

    void Compute(OpKernelContext* ctx) override {
        /*vector<vector<int>> vvgpu*/
        std::vector<std::vector<int>> vecvecgpu;
        const Tensor* vvgpu = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("visiable_gpus", &vvgpu));
        /*check dims, should be {1, 2}*/
        int dims = vvgpu->dims();
        if (std::is_same<Device, CPUDevice>::value) {
            switch (dims) {
                case 1: { // vector
                    std::vector<int> vgpu;
                    auto vvgpu_flat = vvgpu->flat<int32>();
                    for (long int i = 0; i < vvgpu_flat.size(); ++i) {
                        int gpu = vvgpu_flat(i);
                        OP_REQUIRES(ctx, gpu >= 0, errors::InvalidArgument("Input tensor vvgpus's elements ",
                                        "should be >= 0, but get ", gpu));
                        vgpu.push_back(gpu);
                    }
                    vecvecgpu.push_back(vgpu);
                    break;
                }
                case 2: { // vector<vector>
                    auto vvgpu_flat = vvgpu->flat<int32>();
                    size_t dim_size_0 = vvgpu->dim_size(0);
                    size_t dim_size_1 = vvgpu->dim_size(1);
                    for (size_t dim_0 = 0; dim_0 < dim_size_0; ++dim_0) {
                        std::vector<int> vgpu;
                        vgpu.clear();
                        for (size_t dim_1 = 0; dim_1 < dim_size_1; ++dim_1) {
                            int gpu = vvgpu_flat(dim_0 * dim_size_1 + dim_1);
                            OP_REQUIRES(ctx, gpu >= 0, errors::InvalidArgument("Input tensor vvgpus's elements ",
                                        "should be >= 0, but get ", gpu));
                            vgpu.push_back(gpu);
                        }
                        vecvecgpu.push_back(vgpu);
                    }
                    break;
                }
                default: {
                    ctx->SetStatus(errors::InvalidArgument("The dims of input tensor 'vvgpu' should be {1, 2}, but get ", 
                                            dims));
                    return;
                }
            }
        } else if (std::is_same<Device, GPUDevice>::value) {
            switch (dims) {
                case 1: {
                    auto vvgpu_flat = vvgpu->flat<int32>();
                    std::vector<int> vgpu(vvgpu_flat.size(), 0);
                    PLUGIN_CUDA_CHECK(ctx, cudaMemcpy(vgpu.data(), vvgpu_flat.data(), 
                                            vvgpu_flat.size() * sizeof(int), cudaMemcpyDeviceToHost));
                    vecvecgpu.push_back(vgpu);
                    break;
                }
                case 2: {
                    auto vvgpu_flat = vvgpu->flat<int32>();
                    size_t dim_size_0 = vvgpu->dim_size(0);
                    size_t dim_size_1 = vvgpu->dim_size(1);
                    for (size_t dim_0 = 0; dim_0 < dim_size_0; ++dim_0) {
                        std::vector<int> vgpu(dim_size_1, 0);
                        PLUGIN_CUDA_CHECK(ctx, cudaMemcpy(vgpu.data(), vvgpu_flat.data() + dim_0 * dim_size_1,
                                                       dim_size_1 * sizeof(int), cudaMemcpyDeviceToHost));
                        vecvecgpu.push_back(vgpu);
                    }
                    break;
                }
                default: {
                    ctx->SetStatus(errors::InvalidArgument("The dims of input tensor 'vvgpu' should be {1, 2}, but get ", 
                                            dims));
                    return;
                }
            }
        } // if device

        /*check devices are unique*/
        std::string show_gpus = "";
        int before = -1;
        for (auto vec : vecvecgpu) {
            for (auto gpu : vec) {
                if (gpu > before) {
                    show_gpus += (std::to_string(gpu) + ",");
                    before = gpu;
                } else {
                    ctx->SetStatus(errors::InvalidArgument(__FILE__, ":", __LINE__, " ",
                                  "visiable_gpus for embedding plugin is not valid. They must be in ascending order",
                                  " and without repetition."));
                    return;
                }
            }
        }
        LOG(INFO) << "visiable_gpus for embedding plugin: " << show_gpus;

        /*create wrapper instance*/
        if (wrapper) {
            ctx->SetStatus(errors::AlreadyExists("An wrapper instance already exists. Maybe you called hugectr.init() ",
                            "more than once."));
            return;
        } else {
            if ("uint32" == key_type) {
                if ("float" == value_type) {
                    wrapper.reset(new HugeCTR::EmbeddingWrapper<unsigned int, float>(vecvecgpu, 
                                                static_cast<unsigned long long>(seed_),
                                                batch_size_, batch_size_eval_));
                } else if ("half" == value_type) {
                    wrapper.reset(new HugeCTR::EmbeddingWrapper<unsigned int, __half>(vecvecgpu, 
                                                static_cast<unsigned long long>(seed_),
                                                batch_size_, batch_size_eval_));
                }
            } else if ("int64" == key_type) {
                if ("float" == value_type) {
                    wrapper.reset(new HugeCTR::EmbeddingWrapper<long long, float>(vecvecgpu, 
                                                static_cast<unsigned long long>(seed_),
                                                batch_size_, batch_size_eval_));
                } else if ("half" == value_type) {
                    wrapper.reset(new HugeCTR::EmbeddingWrapper<long long, __half>(vecvecgpu, 
                                                static_cast<unsigned long long>(seed_),
                                                batch_size_, batch_size_eval_));
                }
            }
        }

    }

private:
    long long seed_;
    long long batch_size_;
    long long batch_size_eval_;
};

REGISTER_KERNEL_BUILDER(Name("HugectrInit").Device(DEVICE_CPU), InitOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HugectrInit").Device(DEVICE_GPU), InitOp<GPUDevice>);


} // namespace tensorflow