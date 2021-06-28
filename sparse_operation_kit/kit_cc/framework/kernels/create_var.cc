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

#include "facade.h"
#include "tensor_buffer/embedding_buffer.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_var.h"
#include "embedding_variable.h"
#include <exception>
#include <vector>

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice; 

template <typename Device>
class CreateVarOp : public OpKernel {
public:
    explicit CreateVarOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("trainable", &trainable_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_hashtable", &use_hashtable_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &name_));
        if (2 != shape_.dims()) {
            ctx->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                "shape must be [vocabulary_size_per_gpu, embedding_vector_size]."));
            return;
        } 
        if (!shape_.IsFullyDefined()) {
            ctx->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                "shape must be fully defined."));
            return;
        }
    }
    void Compute(OpKernelContext* ctx) override {
        const Tensor* initial_value_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("initial_value", &initial_value_tensor)); 
        OP_REQUIRES(ctx, dtype_ == initial_value_tensor->dtype(), errors::Aborted(
                            __FILE__, ":", __LINE__, " The dtype is not consistent."));
        std::vector<int64_t> dims;
        auto helper = [this, &dims, &ctx](){
            dims.clear();
            for (auto iter = shape_.begin(); iter != shape_.end(); ++iter) {
                int64_t size_n = (*iter).size;
                if (size_n <= 0) {
                    ctx->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                        "the dim ", size_n, " should be > 0."));
                    return;
                }
                dims.push_back(size_n);
            } // for iter
        };
        helper();
        const Tensor* local_replica_id_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("local_replica_id", &local_replica_id_tensor));

        // This is the handle for EmbeddingVariable
        Tensor* var_handle_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &var_handle_tensor));
        core::RefCountPtr<EmbeddingVariable> emb_variable;
        ResourceHandle emb_handle = MakeResourceHandle<EmbeddingVariable>(ctx, 
                                        /*container=*/"EmbeddingVariableContainer",
                                        /*name=*/name_);
        OP_REQUIRES_OK(ctx, LookupOrCreateResource<EmbeddingVariable>(ctx, emb_handle, &emb_variable,
                                        /*creator=*/[var_handle_tensor, &emb_handle](EmbeddingVariable** ptr){
                                            *ptr = new EmbeddingVariable(var_handle_tensor);
                                            (*ptr)->SetHandle(emb_handle);
                                            return Status::OK();
                                        }));
        Tensor tensor;
        // OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, shape_, &emb_tensor));
        Tensor* emb_tensor = &tensor;

        try {
            switch (dtype_) {
                case DT_FLOAT: {
                    // it is numpy value, used as initial_value
                    SparseOperationKit::Facade::instance()->create_variables(local_replica_id_tensor->scalar<int32_t>()(),
                                                                          initial_value_tensor->flat<float>().data(),
                                                                          use_hashtable_, dims, name_, 
                                                                          trainable_, emb_variable,
                                                                          emb_tensor);
                    break;
                } 
                case DT_STRING: {
                    // it specified the initializer
                    SparseOperationKit::Facade::instance()->create_variables(local_replica_id_tensor->scalar<int32_t>()(),
                                                                          initial_value_tensor->flat<tstring>()(0),
                                                                          use_hashtable_, dims, name_, 
                                                                          trainable_, emb_variable,
                                                                          emb_tensor);
                    break;
                }
                case DT_RESOURCE: {
                    // it is the initial_value and specifed memory space.
                    core::RefCountPtr<Var> init_variable;
                    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &init_variable));
                    Tensor* var_tensor = init_variable->tensor();
                    SparseOperationKit::Facade::instance()->create_variables(local_replica_id_tensor->scalar<int32_t>()(),
                                                                          var_tensor->flat<float>().data(),
                                                                          use_hashtable_, dims, name_, 
                                                                          trainable_, emb_variable,
                                                                          emb_tensor);
                    break;
                }
                default: {
                    ctx->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                                    "Not supported dtype for initial_value."));
                    return;
                }
            } // switch
        } catch (const std::exception& error) {
            ctx->SetStatus(errors::Aborted(error.what()));
            return;
        }

        // This is the handle for TF Var
        Tensor* handle_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({}), &handle_tensor));
        core::RefCountPtr<Var> variable;
        ResourceHandle handle = MakeResourceHandle<Var>(ctx, 
                                        /*container=*/"VariableContainer",
                                        /*name=*/name_);
        OP_REQUIRES_OK(ctx, LookupOrCreateResource<Var>(ctx, handle, &variable,
                                        /*creator=*/[&handle, &emb_tensor, handle_tensor](Var** ptr) {
                                            *ptr = new Var(DT_FLOAT);
                                            *(*ptr)->tensor() = *emb_tensor;
                                            (*ptr)->is_initialized = true;
                                            handle_tensor->scalar<ResourceHandle>()() = handle;
                                            return Status::OK();
                                        }));
    }
private:
    bool trainable_;
    TensorShape shape_;
    DataType dtype_;
    bool use_hashtable_;
    std::string name_;
};

REGISTER_KERNEL_BUILDER(Name("CreateVar")
                        .Device(DEVICE_GPU)
                        .HostMemory("initial_value")
                        .HostMemory("local_replica_id")
                        .HostMemory("var_handle")
                        .HostMemory("handle"),
                        CreateVarOp<GPUDevice>);

} // namespace tensorflow