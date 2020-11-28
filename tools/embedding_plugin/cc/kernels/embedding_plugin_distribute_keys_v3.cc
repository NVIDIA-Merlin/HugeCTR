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

/* This op is used to distribute keys to CSR format on GPU.
The memory objects are managed by wrapper.
TODO: But need more modification. Cause TF's datareader preprocessing is working on 
multi-thread, therefore each CPU-thread need independent hardware resources and memory
objects.
*/
template <typename Device>
class EmbeddingDistributeKeysV3Op : public OpKernel {
public:
    explicit EmbeddingDistributeKeysV3Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
        // FIXME: Therefore, this op should not be called.
        OP_REQUIRES(ctx, false, errors::Unavailable(__FILE__, ":", __LINE__, " Should not use this op."));
        
        OP_REQUIRES_OK(ctx, ctx->GetAttr("unique_name", &unique_name_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("gpu_count", &gpu_count_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("slot_num", &slot_num_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("max_nnz", &max_nnz_));
        std::string embedding_type_str;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_type", &embedding_type_str));
        OP_REQUIRES(ctx, HugeCTR::find_item_in_map(embedding_type_, embedding_type_str, EMBEDDING_TYPE_MAP),
                    errors::InvalidArgument("Attr embedding_type should be one of {distributed, localized}, but get ",
                                            embedding_type_str));

        /*try to allocate distributing spaces for this op*/
        OP_REQUIRES(ctx, wrapper, errors::Aborted(__FILE__, ":", __LINE__, " ",
                                    "hugectr.init() should be called first."));
        OP_REQUIRES_OK(ctx, wrapper->try_allocate_distributing_spaces(unique_name_, embedding_type_,
                                                                      batch_size_, slot_num_, max_nnz_));
    }
    void Compute(OpKernelContext* ctx) override {
        /*get input tensor*/
        const Tensor* all_keys_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("all_keys", &all_keys_tensor));
        OP_REQUIRES(ctx, all_keys_tensor->dims() == 3, errors::Aborted(__FILE__, ":", __LINE__, " ",
                                    "all_keys should have shape [batch_size, slot_num, max_nnz]."));
        OP_REQUIRES(ctx, all_keys_tensor->dim_size(0) == batch_size_, errors::Aborted(__FILE__, ":", __LINE__, " ",
                                    "all_keys batch_size is not equal to Attr: batch_size."));
        OP_REQUIRES(ctx, all_keys_tensor->dim_size(1) == slot_num_, errors::Aborted(__FILE__, ":", __LINE__, " ",
                                    "all_keys slot_num is not equal to Attr: slot_num."));
        OP_REQUIRES(ctx, all_keys_tensor->dim_size(2) == max_nnz_, errors::Aborted(__FILE__, ":", __LINE__, " ",
                                    "all_keys max_nnz is not equal to Attr: max_nnz."));

        /*allocate output_tensor*/
        std::vector<Tensor*> row_offsets_tensors(gpu_count_, nullptr);
        std::vector<Tensor*> value_tensors_tensors(gpu_count_, nullptr);
        Tensor* nnz_array_tensor = nullptr;
        for (int i = 0; i < gpu_count_; ++i) {
            OP_REQUIRES_OK(ctx, ctx->allocate_output(i, {batch_size_ * slot_num_ + 1}, &row_offsets_tensors[i]));
            OP_REQUIRES_OK(ctx, ctx->allocate_output(i + gpu_count_, {batch_size_ * slot_num_ * max_nnz_}, 
                                                     &value_tensors_tensors[i]));
        }
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2 * gpu_count_, {gpu_count_}, &nnz_array_tensor));

        /*do distributing keys*/
        OP_REQUIRES(ctx, wrapper, errors::Aborted(__FILE__, ":", __LINE__, " hugectr.init() should be called first."));
        OP_REQUIRES_OK(ctx, wrapper->do_distributing_keys(unique_name_,
                                                          all_keys_tensor,
                                                          row_offsets_tensors,
                                                          value_tensors_tensors,
                                                          nnz_array_tensor));

    }
private:
    int gpu_count_;
    std::string unique_name_;
    long long batch_size_;
    long long slot_num_;
    long long max_nnz_;
    HugeCTR::Embedding_t embedding_type_;
};

REGISTER_KERNEL_BUILDER(Name("HugectrEmbeddingDistributeKeysV3").Device(DEVICE_CPU), 
                        EmbeddingDistributeKeysV3Op<CPUDevice>);



} // namespace tensorflow