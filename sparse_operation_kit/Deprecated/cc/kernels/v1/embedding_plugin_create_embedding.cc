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
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "cuda_utils.h"
#include <memory>
#include <type_traits>

#include <iostream>

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice; 

/*
This op is used to create a hugectr's embedding layer instance.
*/
template <typename Device>
class CreateEmbeddingOp : public OpKernel {
public:
    explicit CreateEmbeddingOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("name_", &name_));
        std::string embedding_type;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_type", &embedding_type));
        OP_REQUIRES(ctx, HugeCTR::find_item_in_map(embedding_type_, embedding_type, EMBEDDING_TYPE_MAP),
                    errors::InvalidArgument("Attr embedding_type should be one of {distributed, localized}, but get ",
                                            embedding_type));
        std::string optimizer_type;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("optimizer_type", &optimizer_type));
        OP_REQUIRES(ctx, HugeCTR::find_item_in_map(optimizer_type_, optimizer_type, OPTIMIZER_TYPE_MAP),
                    errors::InvalidArgument("Attr optimizer should be one of {Adam, MomentumSGD, Nesterov, SGD}, but get ",
                                    optimizer_type));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("max_vocabulary_size_per_gpu", &max_vocabulary_size_per_gpu_));
        auto convert = [](const std::vector<long long>& vec_long, std::vector<size_t>& vec_size_t) {
            for (auto item : vec_long) {
                vec_size_t.push_back(static_cast<size_t>(item));
            }
        };
        std::vector<long long> vec_temp;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("slot_size_array", &vec_temp));
        convert(vec_temp, slot_size_array_);
        OP_REQUIRES_OK(ctx, ctx->GetAttr("opt_hparams", &opt_hparams_));
        std::string update_type_s;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("update_type", &update_type_s));
        OP_REQUIRES(ctx, HugeCTR::find_item_in_map(update_type_, update_type_s, UPDATE_TYPE_MAP),
                    errors::InvalidArgument("Attr update_type should be one of {Local, Global, LazyGlobal}, but got ",
                                            update_type_s));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("atomic_update", &atomic_update_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("scaler", &scaler_));
        OP_REQUIRES(ctx, Utils::check_in_set(SCALER_SET, scaler_), 
                    errors::InvalidArgument("scaler must be one of {1, 128, 256, 512, 1024}. But get ", scaler_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("slot_num", &slot_num_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("max_nnz", &max_nnz_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("max_feature_num", &max_feature_num_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_vec_size", &embedding_vec_size_));
        std::string combiner;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("combiner", &combiner));
        OP_REQUIRES(ctx, HugeCTR::find_item_in_map(combiner_, combiner, COMBINER_MAP),
                errors::InvalidArgument("Attr combiner must be {sum, mean}."));
    }

    void Compute(OpKernelContext* ctx) override {
        /*create embedding*/
        OP_REQUIRES(ctx, wrapper, errors::Unavailable(__FILE__, ":", __FILE__, " ",
                            "There is no wrapper instance, you should call hugectr.init() first."));
        OP_REQUIRES_OK(ctx, wrapper->create_embedding(name_, embedding_type_, optimizer_type_, 
                        max_vocabulary_size_per_gpu_, slot_size_array_, opt_hparams_, update_type_, 
                        atomic_update_, scaler_, static_cast<size_t>(slot_num_), static_cast<size_t>(max_nnz_), 
                        static_cast<size_t>(max_feature_num_), static_cast<size_t>(embedding_vec_size_), 
                        combiner_));

        /*initialization*/
        bool on_gpu = false;
        if (std::is_same<Device, CPUDevice>::value) {
            on_gpu = false;
        } else if (std::is_same<Device, GPUDevice>::value) {
            on_gpu = true;
        }

        /*set TF device*/
        auto stream = ctx->op_device_context()->stream();
        stream_executor::cuda::ScopedActivateExecutorContext scoped_activation{ stream->parent() };

        const Tensor* initial_value = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("init_value", &initial_value));
        OP_REQUIRES_OK(ctx, wrapper->init_embedding_params(name_, initial_value, on_gpu));

        /*output this instance name*/
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &output_tensor));
        auto output = output_tensor->flat<tstring>();
        output(0) = name_;
    }


private:
    std::string name_;
    HugeCTR::Embedding_t embedding_type_;
    HugeCTR::Optimizer_t optimizer_type_;
    long long max_vocabulary_size_per_gpu_;
    std::vector<size_t> slot_size_array_;
    std::vector<float> opt_hparams_;
    HugeCTR::Update_t update_type_;
    bool atomic_update_;
    float scaler_;
    long long slot_num_;
    long long max_nnz_;
    long long max_feature_num_;
    long long embedding_vec_size_;
    int combiner_;
};

// REGISTER_KERNEL_BUILDER(Name("HugectrCreateEmbedding").Device(DEVICE_CPU), 
//                         CreateEmbeddingOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("HugectrCreateEmbedding").Device(DEVICE_GPU), 
                        CreateEmbeddingOp<GPUDevice>);

} // namespace tensorflow