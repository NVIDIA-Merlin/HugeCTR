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

/*TODO: IMPORTANT!
* This op is only used to check whether distribute keys on GPU can work correctly.
* DO NOT USE IT for other purpose.
*/
template <typename Device>
class DistributeKeysGpu : public OpKernel {
public:
    explicit DistributeKeysGpu(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("slot_num", &slot_num_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("max_nnz", &max_nnz_));
        OP_REQUIRES(ctx, batch_size_ > 0 && slot_num_ > 0 && max_nnz_ > 0, errors::InvalidArgument(__FILE__, ":", __LINE__, " ",
                                            "Attrs batch_size <= 0 or slot_num <= 0 or max_nnz <= 0."));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("gpu_count", &gpu_count_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_type", &embedding_type_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor* embedding_name_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("embedding_name", &embedding_name_tensor));
        std::string embedding_name_string = std::string(embedding_name_tensor->flat<tstring>()(0));
        const Tensor* row_indices_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("row_indices", &row_indices_tensor));
        const Tensor* values_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("values", &values_tensor));


        /*calculate output shape*/
        Tensor* row_offsets_output = nullptr;
        Tensor* value_tensors_output = nullptr;
        Tensor* nnz_array_output = nullptr;

        if ("distributed" == embedding_type_) {
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {gpu_count_, batch_size_ * slot_num_ + 1}, &row_offsets_output));
            OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {gpu_count_, batch_size_ * slot_num_ * max_nnz_}, &value_tensors_output));
            OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {gpu_count_}, &nnz_array_output));
        } else if ("localized" == embedding_type_) {
            long long dev_slot_num = (slot_num_ / gpu_count_ + (slot_num_ % gpu_count_ == 0 ? 0 : 1));
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {gpu_count_, batch_size_ * dev_slot_num + 1}, &row_offsets_output));
            OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {gpu_count_, batch_size_ * dev_slot_num * max_nnz_}, &value_tensors_output));
            OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {gpu_count_}, &nnz_array_output));
        } else {
            ctx->SetStatus(errors::InvalidArgument(__FILE__, ":", __LINE__, " ",
                                                   "Unsupported embedding_type."));
            return;
        }

        /*reset output to zeros*/
        auto row_offsets_output_flat = row_offsets_output->flat<long long>();
        auto value_tensors_output_flat = value_tensors_output->flat<long long>();
        auto nnz_array_output_flat = nnz_array_output->flat<long long>();
        const cudaStream_t& tf_stream = ctx->eigen_gpu_device().stream();
        PLUGIN_CUDA_CHECK(ctx, cudaMemsetAsync(row_offsets_output_flat.data(), 0,
                                                row_offsets_output_flat.size() * sizeof(long long),
                                                tf_stream));
        PLUGIN_CUDA_CHECK(ctx, cudaMemsetAsync(value_tensors_output_flat.data(), 0,
                                                value_tensors_output_flat.size() * sizeof(long long),
                                                tf_stream));
        PLUGIN_CUDA_CHECK(ctx, cudaMemsetAsync(nnz_array_output_flat.data(), 0,
                                                nnz_array_output_flat.size() * sizeof(long long),
                                                tf_stream));
        PLUGIN_CUDA_CHECK(ctx, cudaStreamSynchronize(tf_stream));
        
        /*get output*/
        OP_REQUIRES(ctx, wrapper, errors::Unavailable(__FILE__, ":", __FILE__, " ",
                            "There is no wrapper instance, you should call hugectr.init() first."));
        OP_REQUIRES_OK(ctx, wrapper->distribute_keys_gpu(row_indices_tensor, values_tensor, embedding_name_string,
                                                         true, row_offsets_output, value_tensors_output, nnz_array_output));
    }

private:
    std::string embedding_name_;
    std::string embedding_type_;
    long long batch_size_;
    long long slot_num_;
    long long max_nnz_;
    long long gpu_count_;
};

REGISTER_KERNEL_BUILDER(Name("HugectrEmbeddingDistributeKeysGpu").Device(DEVICE_GPU), 
                        DistributeKeysGpu<GPUDevice>);

} // namespace tensorflow