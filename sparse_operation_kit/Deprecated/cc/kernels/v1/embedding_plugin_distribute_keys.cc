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


/*This op will distribute keys to CSR format at CPU.
* outputs are list of row_offsets, list of value_tensors, nnz_array.
*/
template <typename Device>
class EmbeddingDistributeKeysOp : public OpKernel {
public:
    explicit EmbeddingDistributeKeysOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("gpu_count", &gpu_count_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_type", &embedding_type_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("max_feature_num", &max_feature_num_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("max_nnz", &max_nnz_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor* indices = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("indices", &indices));
        const Tensor* values = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("values", &values));
        const Tensor* dense_shape = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("dense_shape", &dense_shape));

        /*do converting*/
        // check dims
        OP_REQUIRES(ctx, indices->dims() == 2, errors::Aborted(__FILE__, ":", __LINE__, " ",
                                                "indices.dims should be 2, but got ", indices->dims()));
        OP_REQUIRES(ctx, values->dims() == 1, errors::Aborted(__FILE__, ":", __LINE__, " ",
                                                "values.dims should be 1, but got ", values->dims()));
        OP_REQUIRES(ctx, dense_shape->dims() == 1, errors::Aborted(__FILE__, ":", __LINE__, " ", 
                                                "dense_shape.dims should be 1, but got ", dense_shape->dims()));
        OP_REQUIRES(ctx, dense_shape->dim_size(0) == 3, errors::Aborted(__FILE__, ":", __LINE__, " ",
                                                "dense_shape.num_elements should be 3, but got ", 
                                                dense_shape->dim_size(0)));

        auto dense_shape_tensor = dense_shape->tensor<long long, 1>();
        long long samples_num = dense_shape_tensor(0), slot_num = dense_shape_tensor(1);
        OP_REQUIRES(ctx, samples_num % gpu_count_ == 0, errors::Aborted(__FILE__, ":", __LINE__, " ",
                                           "batch_size must be a multiple of GPU counts."));

        // allocate output tensors.
        std::vector<Tensor*> row_offsets_output(gpu_count_, nullptr);
        std::vector<Tensor*> value_tensors_output(gpu_count_, nullptr);
        Tensor* nnz_array_output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2 * gpu_count_, {gpu_count_}, &nnz_array_output));

        // create temp csr_chunk
        std::vector<std::unique_ptr<long long []>> row_offsets_(gpu_count_);
        std::vector<std::unique_ptr<long long []>> value_tensors_(gpu_count_);
        std::unique_ptr<long long []> nnz_array_(new long long[gpu_count_]());
        for (int dev_id = 0; dev_id < gpu_count_; ++dev_id) {
            // the two spaces are >= GPUs' CSR chunk buffer size
            long long num_row_offset_dim = samples_num * slot_num + 1;
            long long num_max_value_dim = (max_nnz_ * slot_num <= max_feature_num_)
                                            ? max_nnz_ * slot_num * samples_num 
                                            : max_feature_num_ * samples_num;

            row_offsets_[dev_id].reset(new long long[num_row_offset_dim]());
            value_tensors_[dev_id].reset(new long long[num_max_value_dim]());

            OP_REQUIRES_OK(ctx, ctx->allocate_output(dev_id, {num_row_offset_dim}, &row_offsets_output[dev_id]));
            OP_REQUIRES_OK(ctx, ctx->allocate_output(gpu_count_ + dev_id, {num_max_value_dim}, &value_tensors_output[dev_id]));
        }
        // apply to csr_chunk reset
        std::vector<int> size_of_row_offset_(gpu_count_, 0);
        std::vector<int> size_of_value_(gpu_count_, 0);

        auto indices_ptr = indices->flat<long long>().data();
        auto keys_ptr = values->flat<long long>().data();
        long long key_idx = 0;
        for (long long sample_id = 0; sample_id < samples_num; ++sample_id){
            for (long long slot_id = 0; slot_id < slot_num; ++slot_id) {
                if (embedding_type_ == "distributed") {
                    // new row
                    for (int dev_id = 0; dev_id < gpu_count_; ++dev_id) {
                        row_offsets_[dev_id][size_of_row_offset_[dev_id]++] = static_cast<long long>(size_of_value_[dev_id]);
                    }

                    // keys belong to same sample and same slot.
                    while (true) {
                        if (key_idx < values->dim_size(0) &&
                            sample_id == indices_ptr[key_idx * dense_shape->dim_size(0) + 0] &&
                            slot_id == indices_ptr[key_idx * dense_shape->dim_size(0) + 1]) {
                            int dev_id = keys_ptr[key_idx] % gpu_count_;
                            value_tensors_[dev_id][size_of_value_[dev_id]++] = 
                                static_cast<long long>(keys_ptr[key_idx]); // push back
                            ++key_idx;
                        } else {
                            break;
                        }
                    }
                } else if (embedding_type_ == "localized") {
                    // new row
                    unsigned int dev_id = slot_id % gpu_count_;
                    row_offsets_[dev_id][size_of_row_offset_[dev_id]++] = static_cast<long long>(size_of_value_[dev_id]);

                    // keys belong to same sample and same slot.
                    while (true) {
                        if (key_idx < values->dim_size(0) &&
                            sample_id == indices_ptr[key_idx * dense_shape->dim_size(0) + 0] && 
                            slot_id == indices_ptr[key_idx * dense_shape->dim_size(0) + 1]) {
                            value_tensors_[dev_id][size_of_value_[dev_id]++] = 
                                static_cast<long long>(keys_ptr[key_idx]); // push back
                            ++key_idx;
                        } else {
                            break;
                        }
                    }
                } else {
                    ctx->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                                                   "Unsupported embedding_type.", embedding_type_));
                    return;
                }
            } // for slot_id
        } // for sample_id

        // write the last index to row
        for (int dev_id = 0; dev_id < gpu_count_; ++dev_id) {
            row_offsets_[dev_id][size_of_row_offset_[dev_id]++] = static_cast<long long>(size_of_value_[dev_id]);
            nnz_array_[dev_id] = size_of_value_[dev_id]; // write nnz buffer
        }

        // copy data to output
        for (int dev_id = 0; dev_id < gpu_count_; ++dev_id){
            auto row_offset_flat = row_offsets_output[dev_id]->flat<long long>();
            memcpy(row_offset_flat.data(),
                   row_offsets_[dev_id].get(),
                   sizeof(long long) * row_offset_flat.size());
            auto value_tensor_flat = value_tensors_output[dev_id]->flat<long long>();
            memcpy(value_tensor_flat.data(),
                    value_tensors_[dev_id].get(),
                    sizeof(long long) * value_tensor_flat.size());
        }
        memcpy(nnz_array_output->flat<long long>().data(),
                nnz_array_.get(),
                sizeof(long long) * gpu_count_);

    }
private:
    int gpu_count_;
    std::string embedding_type_;
    int max_feature_num_;
    int max_nnz_;
};

REGISTER_KERNEL_BUILDER(Name("HugectrEmbeddingDistributeKeys").Device(DEVICE_CPU), 
                        EmbeddingDistributeKeysOp<CPUDevice>);

} // namespace tensorflow