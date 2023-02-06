/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

// clang-format off
#include <vector>
#include <string>

#include <cuda_fp16.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"

#include "lookup/impl/embedding_collection.hpp"
#include "lookup/impl/embedding_collection_adapter.h"
#include "lookup/impl/hotness_calculate.h"
// clang-format on

namespace stream_executor {
namespace gpu {
cudaStream_t AsGpuStreamValue(Stream* stream);
}  // namespace gpu
}  // namespace stream_executor

namespace tensorflow {

template <typename KeyType, typename OffsetType, typename DType>
class EmbeddingCollectionBase : public OpKernel {
 protected:
  int num_lookups_;
  std::vector<std::string> combiners_;
  // std::vector<int> hotness_;
  std::vector<int> shard_;
  std::vector<int> dimensions_;

  int rank_;
  int num_ranks_;
  int id_in_local_rank_;
  int num_gpus_;

  int num_gpu_per_rank_;
  int global_gpu_id_;
  int num_local_lookups_;
  bool use_sp_weight_;

  std::unique_ptr<sok::EmbeddingCollectionParam> ebc_param_;
  std::unique_ptr<sok::UniformModelParallelEmbeddingMeta> meta_;

  std::shared_ptr<sok::CoreResourceManager> make_core_resource(OpKernelContext* ctx) {
    return std::make_shared<sok::TFCoreResourceManager>(ctx,
                                                        /*device_id*/ -1,
                                                        /*local_rank*/ rank_,
                                                        /*num_rank*/ num_ranks_,
                                                        /*id_in_local_rank*/ id_in_local_rank_,
                                                        /*num_gpu_per_rank*/ num_gpu_per_rank_);
  }

  void make_shard_matrix(std::vector<std::vector<int>>& shard_matrix) {
    shard_matrix.resize(num_gpus_);
    for (size_t i = 0; i < shard_matrix.size(); ++i) {
      for (size_t j = 0; j < shard_.size(); ++j) {
        if (shard_[j] < 0 || shard_[j] == static_cast<int>(i)) {
          // Distributed embedding
          // Localized embedding with embedding table on i_th GPU
          shard_matrix[i].push_back(1);
        } else {
          // Localized embedding with embedding table on other GPU
          shard_matrix[i].push_back(0);
        }
      }
    }
  }

  void update_meta(std::shared_ptr<sok::CoreResourceManager> tf_backend, int global_batch_size,
                   std::vector<int>& hotness) {
    if (!ebc_param_ || ebc_param_->universal_batch_size != global_batch_size) {
      std::vector<std::vector<int>> shard_matrix;
      this->make_shard_matrix(shard_matrix);
      this->ebc_param_ = sok::make_embedding_collection_param<KeyType, OffsetType, DType>(
          shard_matrix, this->num_lookups_, this->combiners_, hotness, this->dimensions_,
          global_batch_size, global_gpu_id_);
      this->meta_.reset(new sok::UniformModelParallelEmbeddingMeta(tf_backend, *ebc_param_, 0));
    } else {
      std::vector<std::vector<int>> shard_matrix;
      this->make_shard_matrix(shard_matrix);
      this->ebc_param_ = sok::make_embedding_collection_param<KeyType, OffsetType, DType>(
          shard_matrix, this->num_lookups_, this->combiners_, hotness, this->dimensions_,
          global_batch_size, global_gpu_id_);
      this->meta_->update_mutable_meta(tf_backend, *ebc_param_, 0);
    }
  }

 public:
  explicit EmbeddingCollectionBase(OpKernelConstruction* ctx)
      : OpKernel(ctx), ebc_param_(nullptr), meta_(nullptr) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_lookups", &num_lookups_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("combiners", &combiners_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shard", &shard_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dimensions", &dimensions_));

    OP_REQUIRES(ctx, combiners_.size() == static_cast<size_t>(num_lookups_),
                errors::InvalidArgument("len(combiners) != num_lookups."));
    OP_REQUIRES(ctx, shard_.size() == static_cast<size_t>(num_lookups_),
                errors::InvalidArgument("len(shard) != num_lookups."));
    OP_REQUIRES(ctx, dimensions_.size() == static_cast<size_t>(this->num_lookups_),
                errors::InvalidArgument("len(dimensions) != num_lookups."));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("rank", &rank_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_ranks", &num_ranks_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("id_in_local_rank", &id_in_local_rank_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_gpus", &num_gpus_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_sp_weight", &use_sp_weight_));

    // check rank/num_ranks/id_in_local_rank/num_gpus
    OP_REQUIRES(ctx, rank_ >= 0 && rank_ < num_ranks_, errors::InvalidArgument("Invalid rank."));
    OP_REQUIRES(ctx, (num_gpus_ % num_ranks_) == 0,
                errors::InvalidArgument("num_gpus % num_ranks must be 0."));
    OP_REQUIRES(ctx, id_in_local_rank_ >= 0 && id_in_local_rank_ < (num_gpus_ / num_ranks_),
                errors::InvalidArgument("Invalid id_in_local_rank."));

    for (int i = 0; i < num_lookups_; ++i) {
      OP_REQUIRES(ctx, shard_[i] < num_gpus_,
                  errors::InvalidArgument("Invalid target GPU of LocalizedEmbedding."));
    }

    num_gpu_per_rank_ = num_gpus_ / num_ranks_;
    global_gpu_id_ = rank_ * num_gpu_per_rank_ + id_in_local_rank_;

    num_local_lookups_ = 0;
    for (size_t i = 0; i < shard_.size(); ++i) {
      if (shard_[i] == -1 || shard_[i] == global_gpu_id_) {
        // shard_[i] == -1 means distributed embedding.
        // shard_[i] == global_gpu_id_ means localized embedding and embedding table is located in
        // this GPU.
        num_local_lookups_ += 1;
      }
    }
  }
};

// -----------------------------------------------------------------------------------------------
// HotnessCalculate
// -----------------------------------------------------------------------------------------------
template <typename DType>
class HotnessCalculateOp : public OpKernel {
 public:
  explicit HotnessCalculateOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    launcher_.initialize();
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_lookups", &num_lookups_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_gpus", &num_gpus_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* row_length_send_buffer = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("row_length_buffer", &row_length_send_buffer));
    int64_t input_len = row_length_send_buffer->dim_size(0);
    OP_REQUIRES(ctx, input_len % (num_lookups_ * num_gpus_) == 0,
                errors::InvalidArgument("input_len%(num_lookups_*num_gpus_) != 0"));
    size_t local_batchsize = input_len / num_lookups_ / num_gpus_;
    Tensor* hotness = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {num_lookups_}, &hotness));

    // temp buffer
    Tensor device_buffer;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, {num_lookups_}, &device_buffer));

    // stream
    auto device_ctx = ctx->op_device_context();
    OP_REQUIRES(ctx, device_ctx != nullptr, errors::Aborted("No valid device context."));
    cudaStream_t stream = stream_executor::gpu::AsGpuStreamValue(device_ctx->stream());

    // cuda kernel
    launcher_(row_length_send_buffer->data(), local_batchsize, num_lookups_, num_gpus_,
              device_buffer.data(), hotness->data(), stream);
  }

 private:
  sok::HotnessCalLauncher<DType> launcher_;
  int num_lookups_;
  int num_gpus_;
};

#define REGISTER_GPU_KERNELS(dtype_tf, dtype)                        \
  REGISTER_KERNEL_BUILDER(Name("HotnessCalculate")                   \
                              .Device(DEVICE_GPU)                    \
                              .HostMemory("hotness")                 \
                              .TypeConstraint<dtype_tf>("Tindices"), \
                          HotnessCalculateOp<dtype>)

#if TF_VERSION_MAJOR == 1
REGISTER_GPU_KERNELS(int64, int64_t);
REGISTER_GPU_KERNELS(int32, int32_t);
#else
REGISTER_GPU_KERNELS(int64_t, int64_t);
REGISTER_GPU_KERNELS(int32_t, int32_t);
#endif

#undef REGISTER_GPU_KERNELS

// -----------------------------------------------------------------------------------------------
// PreprocessingForward
// -----------------------------------------------------------------------------------------------
template <typename KeyType, typename OffsetType, typename DType>
class PreprocessingForwardOp : public EmbeddingCollectionBase<KeyType, OffsetType, DType> {
 public:
  explicit PreprocessingForwardOp(OpKernelConstruction* ctx)
      : EmbeddingCollectionBase<KeyType, OffsetType, DType>(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // Prepare inputs
    int64_t num_keys = 0;
    int64_t num_row_lengths = 0;
    std::vector<sok::Tensor> keys_sok;
    std::vector<sok::Tensor> row_lengths_sok;
    std::vector<sok::Tensor> sp_weights_sok;
    this->use_sp_weight_ = false;
    for (int i = 0; i < this->num_lookups_; ++i) {
      const Tensor& key_tf = ctx->input(i);
      keys_sok.push_back(sok::convert_tensor<KeyType>(&key_tf));
      num_keys += key_tf.NumElements();

      const Tensor& row_length_tf = ctx->input(this->num_lookups_ + i);
      row_lengths_sok.push_back(sok::convert_tensor<OffsetType>(&row_length_tf));
      num_row_lengths += row_length_tf.NumElements();
    }

    // Prepare outputs
    Tensor* key_send_buffer_tf = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {num_keys}, &key_send_buffer_tf));
    sok::Tensor key_send_buffer_sok(sok::convert_tensor<KeyType>(key_send_buffer_tf));

    Tensor* row_length_send_buffer_tf = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {num_row_lengths}, &row_length_send_buffer_tf));
    sok::Tensor row_length_send_buffer_sok(
        sok::convert_tensor<OffsetType>(row_length_send_buffer_tf));

    Tensor* sp_weight_send_buffer_tf = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {0}, &sp_weight_send_buffer_tf));

    sok::Tensor sp_weight_send_buffer_sok(sok::convert_tensor<DType>(sp_weight_send_buffer_tf));

    // Do forward
    ::embedding::tf::swizzle_key::sparse_forward_per_gpu(this->make_core_resource(ctx), keys_sok,
                                                         row_lengths_sok, key_send_buffer_sok,
                                                         row_length_send_buffer_sok);
  }
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, offset_type_tf, offset_type, dtype_tf, dtype) \
  REGISTER_KERNEL_BUILDER(Name("PreprocessingForward")                                            \
                              .Device(DEVICE_GPU)                                                 \
                              .TypeConstraint<key_type_tf>("Tindices")                            \
                              .TypeConstraint<offset_type_tf>("Toffsets")                         \
                              .TypeConstraint<dtype_tf>("dtype"),                                 \
                          PreprocessingForwardOp<key_type, offset_type, dtype>)
#if TF_VERSION_MAJOR == 1
REGISTER_GPU_KERNELS(int64, int64_t, int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int64, int64_t, int32, int32_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, int32, int32_t, float, float);
// REGISTER_GPU_KERNELS(int64, int64_t, int64, int64_t, Eigen::half, __half);
// REGISTER_GPU_KERNELS(int32, int32_t, int64, int64_t, Eigen::half, __half);
// REGISTER_GPU_KERNELS(int64, int64_t, int32, int32_t, Eigen::half, __half);
// REGISTER_GPU_KERNELS(int32, int32_t, int32, int32_t, Eigen::half, __half);
#else
REGISTER_GPU_KERNELS(int64_t, int64_t, int64_t, int64_t, float, float);
REGISTER_GPU_KERNELS(int32_t, int32_t, int64_t, int64_t, float, float);
REGISTER_GPU_KERNELS(int64_t, int64_t, int32_t, int32_t, float, float);
REGISTER_GPU_KERNELS(int32_t, int32_t, int32_t, int32_t, float, float);
// REGISTER_GPU_KERNELS(int64_t, int64_t, int64_t, int64_t, Eigen::half, __half);
// REGISTER_GPU_KERNELS(int32_t, int32_t, int64_t, int64_t, Eigen::half, __half);
// REGISTER_GPU_KERNELS(int64_t, int64_t, int32_t, int32_t, Eigen::half, __half);
// REGISTER_GPU_KERNELS(int32_t, int32_t, int32_t, int32_t, Eigen::half, __half);
#endif

#undef REGISTER_GPU_KERNELS

// -----------------------------------------------------------------------------------------------
// PreprocessingForwardWithWeight
// -----------------------------------------------------------------------------------------------
template <typename KeyType, typename OffsetType, typename DType>
class PreprocessingForwardWithWeightOp
    : public EmbeddingCollectionBase<KeyType, OffsetType, DType> {
 public:
  explicit PreprocessingForwardWithWeightOp(OpKernelConstruction* ctx)
      : EmbeddingCollectionBase<KeyType, OffsetType, DType>(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // Prepare inputs
    int64_t num_keys = 0;
    int64_t num_row_lengths = 0;
    int64_t num_sp_weights = 0;
    std::vector<sok::Tensor> keys_sok;
    std::vector<sok::Tensor> row_lengths_sok;
    std::vector<sok::Tensor> sp_weights_sok;
    for (int i = 0; i < this->num_lookups_; ++i) {
      const Tensor& key_tf = ctx->input(i);
      keys_sok.push_back(sok::convert_tensor<KeyType>(&key_tf));
      num_keys += key_tf.NumElements();

      const Tensor& row_length_tf = ctx->input(this->num_lookups_ + i);
      row_lengths_sok.push_back(sok::convert_tensor<OffsetType>(&row_length_tf));
      num_row_lengths += row_length_tf.NumElements();
      if (this->use_sp_weight_) {
        const Tensor& sp_weight_tf = ctx->input(this->num_lookups_ * 2 + i);
        sp_weights_sok.push_back(sok::convert_tensor<DType>(&sp_weight_tf));
        num_sp_weights += sp_weight_tf.NumElements();
      }
    }

    // Prepare outputs
    Tensor* key_send_buffer_tf = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {num_keys}, &key_send_buffer_tf));
    sok::Tensor key_send_buffer_sok(sok::convert_tensor<KeyType>(key_send_buffer_tf));

    Tensor* row_length_send_buffer_tf = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {num_row_lengths}, &row_length_send_buffer_tf));
    sok::Tensor row_length_send_buffer_sok(
        sok::convert_tensor<OffsetType>(row_length_send_buffer_tf));

    Tensor* sp_weight_send_buffer_tf = nullptr;
    if (this->use_sp_weight_) {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {num_sp_weights}, &sp_weight_send_buffer_tf));
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {0}, &sp_weight_send_buffer_tf));
    }
    sok::Tensor sp_weight_send_buffer_sok(sok::convert_tensor<DType>(sp_weight_send_buffer_tf));

    // Do forward
    ::embedding::tf::swizzle_key::weighted_sparse_forward_per_gpu(
        this->make_core_resource(ctx), keys_sok, row_lengths_sok, sp_weights_sok,
        key_send_buffer_sok, row_length_send_buffer_sok, sp_weight_send_buffer_sok);
  }
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, offset_type_tf, offset_type, dtype_tf, dtype) \
  REGISTER_KERNEL_BUILDER(Name("PreprocessingForwardWithWeight")                                  \
                              .Device(DEVICE_GPU)                                                 \
                              .TypeConstraint<key_type_tf>("Tindices")                            \
                              .TypeConstraint<offset_type_tf>("Toffsets")                         \
                              .TypeConstraint<dtype_tf>("dtype"),                                 \
                          PreprocessingForwardWithWeightOp<key_type, offset_type, dtype>)
#if TF_VERSION_MAJOR == 1
REGISTER_GPU_KERNELS(int64, int64_t, int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int64, int64_t, int32, int32_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, int32, int32_t, float, float);
// REGISTER_GPU_KERNELS(int64, int64_t, int64, int64_t, Eigen::half, __half);
// REGISTER_GPU_KERNELS(int32, int32_t, int64, int64_t, Eigen::half, __half);
// REGISTER_GPU_KERNELS(int64, int64_t, int32, int32_t, Eigen::half, __half);
// REGISTER_GPU_KERNELS(int32, int32_t, int32, int32_t, Eigen::half, __half);
#else
REGISTER_GPU_KERNELS(int64_t, int64_t, int64_t, int64_t, float, float);
REGISTER_GPU_KERNELS(int32_t, int32_t, int64_t, int64_t, float, float);
REGISTER_GPU_KERNELS(int64_t, int64_t, int32_t, int32_t, float, float);
REGISTER_GPU_KERNELS(int32_t, int32_t, int32_t, int32_t, float, float);
// REGISTER_GPU_KERNELS(int64_t, int64_t, int64_t, int64_t, Eigen::half, __half);
// REGISTER_GPU_KERNELS(int32_t, int32_t, int64_t, int64_t, Eigen::half, __half);
// REGISTER_GPU_KERNELS(int64_t, int64_t, int32_t, int32_t, Eigen::half, __half);
// REGISTER_GPU_KERNELS(int32_t, int32_t, int32_t, int32_t, Eigen::half, __half);
#endif

#undef REGISTER_GPU_KERNELS

// -----------------------------------------------------------------------------------------------
// LookupForward
// -----------------------------------------------------------------------------------------------
template <typename KeyType, typename OffsetType, typename DType, typename VarType, typename Adapter>
class LookupForwardOp : public EmbeddingCollectionBase<KeyType, OffsetType, DType> {
 private:
  Adapter adapter_;

 public:
  explicit LookupForwardOp(OpKernelConstruction* ctx)
      : EmbeddingCollectionBase<KeyType, OffsetType, DType>(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    std::vector<tf_shared_lock> locks;
    std::vector<core::RefCountPtr<VarType>> vars;
    std::vector<int> scale;
    for (int i = 0; i < this->num_lookups_; ++i) {
      auto handle = HandleFromInput(ctx, i);
      auto dtypes_and_shapes = handle.dtypes_and_shapes();
      auto shape = dtypes_and_shapes[0].shape;
      OP_REQUIRES(ctx, dtypes_and_shapes[0].dtype == DataType::DT_FLOAT,
                  errors::InvalidArgument("Type of variable must be float."));
      OP_REQUIRES(ctx, this->dimensions_[i] == shape.dim_size(1),
                  errors::InvalidArgument("Invalid dimension"));

      core::RefCountPtr<VarType> var;
      OP_REQUIRES_OK(ctx, LookupResource(ctx, handle, &var));
      vars.push_back(std::move(var));

      if (this->shard_[i] < 0) {
        scale.push_back(this->num_gpus_);
      } else {
        scale.push_back(1);
      }
    }

    // stream
    auto device_ctx = ctx->op_device_context();
    OP_REQUIRES(ctx, device_ctx != nullptr, errors::Aborted("No valid device context."));
    cudaStream_t stream = stream_executor::gpu::AsGpuStreamValue(device_ctx->stream());

    // Prepare inputs (except handles)
    const Tensor* key_recv_buffer = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("key_recv_buffer", &key_recv_buffer));
    sok::Tensor key_recv_buffer_tensor(sok::convert_tensor<KeyType>(key_recv_buffer));

    const Tensor* row_length_recv_buffer = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("row_length_recv_buffer", &row_length_recv_buffer));
    sok::Tensor row_length_recv_buffer_tensor(
        sok::convert_tensor<OffsetType>(row_length_recv_buffer));

    const Tensor* sp_weight = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_weight", &sp_weight));
    sok::Tensor sp_weight_recv_buffer_tensor(sok::convert_tensor<DType>(sp_weight));

    int global_batch_size = row_length_recv_buffer->NumElements() / this->num_lookups_;
    // Get hotness dynamic
    const Tensor* hotness = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("hotness", &hotness));
    std::vector<int> hotness_vector;
    int* t_hotness = (int*)hotness->data();
    int64_t hotness_num = hotness->NumElements();
    for (int64_t i = 0; i < hotness_num; ++i) {
      hotness_vector.push_back(t_hotness[i]);
    }

    // Instance 3g embedding
    auto tf_backend = this->make_core_resource(ctx);
    this->update_meta(tf_backend, global_batch_size, hotness_vector);

    // Prepare ILookup (i.e. embedding table)
    std::vector<int> ev_size_per_lookup;
    for (auto& p : this->ebc_param_->lookup_params) {
      ev_size_per_lookup.push_back(p.ev_size);
    }
    adapter_.set(vars, locks, this->dimensions_, scale, stream);

    // Prepare outputs
    auto buffer_size_list = ::embedding::tf::model_forward::get_model_comm_buffer_size(
        *this->meta_, tf_backend->get_global_gpu_count(), global_batch_size);
    std::vector<sok::Tensor> emb_vec_model_buffer;
    for (size_t i = 0; i < buffer_size_list.size(); ++i) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(i, {static_cast<int64_t>(buffer_size_list[i])}, &output));
      emb_vec_model_buffer.push_back(sok::convert_tensor<DType>(output));
    }

    // Do forward
    int64_t num_model_key, num_model_offsets;
    sok::Tensor ret_model_key, ret_model_offset, ret_sp_sum, ret_sp_weight;
    if (this->use_sp_weight_) {
      ::embedding::tf::model_forward::weighted_sparse_forward_per_gpu(
          tf_backend, *this->meta_, this->global_gpu_id_, key_recv_buffer_tensor,
          row_length_recv_buffer_tensor, sp_weight_recv_buffer_tensor, &adapter_,
          emb_vec_model_buffer, &num_model_key, &num_model_offsets, &ret_model_key,
          &ret_model_offset, &ret_sp_weight);
    } else {
      ::embedding::tf::model_forward::sparse_forward_per_gpu(
          tf_backend, *this->ebc_param_, *this->meta_, key_recv_buffer_tensor,
          row_length_recv_buffer_tensor, &adapter_, emb_vec_model_buffer, &num_model_key,
          &num_model_offsets, &ret_model_key, &ret_model_offset);
    }

    // Prepare model_key & model_offsets
    // Note the type of model_offsets is always uint32_t
    Tensor* model_key = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(this->num_gpus_, {num_model_key}, &model_key));
    sok::Tensor model_key_tensor(sok::convert_tensor<KeyType>(model_key));
    Tensor* model_offsets = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(this->num_gpus_ + 1, {num_model_offsets}, &model_offsets));
    sok::Tensor model_offsets_tensor(sok::convert_tensor<uint32_t>(model_offsets));

    Tensor* model_sp_weight = nullptr;
    if (this->use_sp_weight_) {
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(this->num_gpus_ + 2, {num_model_key}, &model_sp_weight));
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(this->num_gpus_ + 2, {0}, &model_sp_weight));
    }
    sok::Tensor sp_weight_tensor(sok::convert_tensor<uint32_t>(model_sp_weight));

    // Copy tensors that will be used in backward
    if (this->use_sp_weight_) {
      ::embedding::tf::model_forward::weighted_copy_model_keys_and_offsets(
          tf_backend, ret_model_key, ret_model_offset, ret_sp_weight, model_key_tensor,
          model_offsets_tensor, sp_weight_tensor);
    } else {
      ::embedding::tf::model_forward::copy_model_keys_and_offsets(
          tf_backend, ret_model_key, ret_model_offset, model_key_tensor, model_offsets_tensor);
    }
  }
};

// clang-format off
#define REGISTER_GPU_KERNELS(key_type_tf, key_type, offset_type_tf, offset_type, dtype_tf, dtype)  \
  REGISTER_KERNEL_BUILDER(Name("LookupForward")                                                    \
                              .Device(DEVICE_GPU)                                                  \
                              .HostMemory("handles")                                               \
                              .HostMemory("hotness")                                               \
                              .TypeConstraint<key_type_tf>("Tindices")                             \
                              .TypeConstraint<offset_type_tf>("Toffsets")                          \
                              .TypeConstraint<dtype_tf>("dtype"),                                  \
                          LookupForwardOp<key_type, offset_type, dtype, Var,                       \
                                          sok::TFAdapter<key_type, dtype>>)                        \
  REGISTER_KERNEL_BUILDER(Name("LookupForwardDynamic")                                             \
                              .Device(DEVICE_GPU)                                                  \
                              .HostMemory("handles")                                               \
                              .HostMemory("hotness")                                               \
                              .TypeConstraint<key_type_tf>("Tindices")                             \
                              .TypeConstraint<offset_type_tf>("Toffsets")                          \
                              .TypeConstraint<dtype_tf>("dtype"),                                  \
                          LookupForwardOp<key_type, offset_type, dtype, DummyVar<key_type, dtype>, \
                                          sok::DummyVarAdapter<key_type, dtype>>)
// clang-format on

#if TF_VERSION_MAJOR == 1
REGISTER_GPU_KERNELS(int64, int64_t, int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int64, int64_t, int32, int32_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, int32, int32_t, float, float);
// REGISTER_GPU_KERNELS(int64, int64_t, int64, int64_t, Eigen::half, __half);
// REGISTER_GPU_KERNELS(int32, int32_t, int64, int64_t, Eigen::half, __half);
// REGISTER_GPU_KERNELS(int64, int64_t, int32, int32_t, Eigen::half, __half);
// REGISTER_GPU_KERNELS(int32, int32_t, int32, int32_t, Eigen::half, __half);
#else
REGISTER_GPU_KERNELS(int64_t, int64_t, int64_t, int64_t, float, float);
REGISTER_GPU_KERNELS(int32_t, int32_t, int64_t, int64_t, float, float);
REGISTER_GPU_KERNELS(int64_t, int64_t, int32_t, int32_t, float, float);
REGISTER_GPU_KERNELS(int32_t, int32_t, int32_t, int32_t, float, float);
// REGISTER_GPU_KERNELS(int64_t, int64_t, int64_t, int64_t, Eigen::half, __half);
// REGISTER_GPU_KERNELS(int32_t, int32_t, int64_t, int64_t, Eigen::half, __half);
// REGISTER_GPU_KERNELS(int64_t, int64_t, int32_t, int32_t, Eigen::half, __half);
// REGISTER_GPU_KERNELS(int32_t, int32_t, int32_t, int32_t, Eigen::half, __half);
#endif

#undef REGISTER_GPU_KERNELS

// -----------------------------------------------------------------------------------------------
// LookupBackward
// -----------------------------------------------------------------------------------------------
template <typename KeyType, typename OffsetType, typename DType>
class LookupBackwardOp : public EmbeddingCollectionBase<KeyType, OffsetType, DType> {
 public:
  explicit LookupBackwardOp(OpKernelConstruction* ctx)
      : EmbeddingCollectionBase<KeyType, OffsetType, DType>(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    if (this->num_local_lookups_ == 0) {
      for (int i = 0; i < this->num_lookups_; ++i) {
        Tensor* unused = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(i, {0}, &unused));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(this->num_lookups_ + i, {0}, &unused));
      }
      return;
    }

    // Prepare input
    std::vector<sok::Tensor> emb_vec_buffer_grad;
    for (int i = 0; i < this->num_gpus_; ++i) {
      const Tensor& inp = ctx->input(i);
      emb_vec_buffer_grad.push_back(sok::convert_tensor<DType>(&inp));
    }
    const Tensor* model_key = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("model_key", &model_key));
    sok::Tensor model_key_tensor(sok::convert_tensor<KeyType>(model_key));
    const Tensor* model_offsets = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("model_offsets", &model_offsets));
    sok::Tensor model_offsets_tensor(sok::convert_tensor<uint32_t>(model_offsets));

    // Get global batch size
    int batch_size = (model_offsets->NumElements() - 1) / this->num_local_lookups_;

    const Tensor* hotness = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("hotness", &hotness));
    std::vector<int> hotness_vector;
    int* t_hotness = (int*)hotness->data();
    int64_t hotness_num = hotness->NumElements();
    for (int64_t i = 0; i < hotness_num; ++i) {
      hotness_vector.push_back(t_hotness[i]);
    }

    const Tensor* model_sp_weight = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("model_sp_weight", &model_sp_weight));
    sok::Tensor model_sp_weight_tensor(sok::convert_tensor<DType>(model_sp_weight));
    // Instance 3g embedding
    auto tf_backend = this->make_core_resource(ctx);
    this->update_meta(tf_backend, batch_size, hotness_vector);

    // Do backward
    std::vector<int> num_unique_key_per_table, unique_id_space_list;
    sok::Tensor ret_continous_unique_key, ret_continous_emb_vec;
    if (this->use_sp_weight_) {
      ::embedding::tf::model_backward::weighted_sparse_backward_per_gpu(
          tf_backend, *this->meta_, emb_vec_buffer_grad, model_key_tensor, model_offsets_tensor,
          model_sp_weight_tensor, &num_unique_key_per_table, &unique_id_space_list,
          &ret_continous_unique_key, &ret_continous_emb_vec);
    } else {
      ::embedding::tf::model_backward::sparse_backward_per_gpu(
          tf_backend, *this->ebc_param_, *this->meta_, emb_vec_buffer_grad, model_key_tensor,
          model_offsets_tensor, &num_unique_key_per_table, &unique_id_space_list,
          &ret_continous_unique_key, &ret_continous_emb_vec);
    }

    // Prepare output
    std::vector<sok::Tensor> unique_key, grad;
    for (int i = 0; i < this->num_lookups_; ++i) {
      int num_unique_key = 0;
      auto target_id_space_iter =
          std::find(unique_id_space_list.begin(), unique_id_space_list.end(), i);
      if (target_id_space_iter != unique_id_space_list.end()) {
        const auto idx = std::distance(unique_id_space_list.begin(), target_id_space_iter);
        num_unique_key = num_unique_key_per_table[idx];
      }

      Tensor* unique_key_tf = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, {num_unique_key}, &unique_key_tf));
      Tensor* grad_tf = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i + this->num_lookups_,
                                               {num_unique_key, this->dimensions_[i]}, &grad_tf));
      if (target_id_space_iter != unique_id_space_list.end()) {
        unique_key.push_back(sok::convert_tensor<KeyType>(unique_key_tf));
        grad.push_back(sok::convert_tensor<DType>(grad_tf));
      }
    }

    // Copy output
    ::embedding::tf::model_backward::copy_backward_key_and_emb_vec(
        tf_backend, ret_continous_unique_key, ret_continous_emb_vec, unique_key, grad);
  }
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, offset_type_tf, offset_type, dtype_tf, dtype) \
  REGISTER_KERNEL_BUILDER(Name("LookupBackward")                                                  \
                              .Device(DEVICE_GPU)                                                 \
                              .HostMemory("hotness")                                              \
                              .TypeConstraint<key_type_tf>("Tindices")                            \
                              .TypeConstraint<offset_type_tf>("Toffsets")                         \
                              .TypeConstraint<dtype_tf>("dtype"),                                 \
                          LookupBackwardOp<key_type, offset_type, dtype>)

#if TF_VERSION_MAJOR == 1
REGISTER_GPU_KERNELS(int64, int64_t, int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int64, int64_t, int32, int32_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, int32, int32_t, float, float);
REGISTER_GPU_KERNELS(int64, int64_t, int64, int64_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int32, int32_t, int64, int64_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int64, int64_t, int32, int32_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int32, int32_t, int32, int32_t, Eigen::half, __half);
#else
REGISTER_GPU_KERNELS(int64_t, int64_t, int64_t, int64_t, float, float);
REGISTER_GPU_KERNELS(int32_t, int32_t, int64_t, int64_t, float, float);
REGISTER_GPU_KERNELS(int64_t, int64_t, int32_t, int32_t, float, float);
REGISTER_GPU_KERNELS(int32_t, int32_t, int32_t, int32_t, float, float);
REGISTER_GPU_KERNELS(int64_t, int64_t, int64_t, int64_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int32_t, int32_t, int64_t, int64_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int64_t, int64_t, int32_t, int32_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int32_t, int32_t, int32_t, int32_t, Eigen::half, __half);
#endif

#undef REGISTER_GPU_KERNELS

// -----------------------------------------------------------------------------------------------
// PostprocessingForward
// -----------------------------------------------------------------------------------------------
template <typename KeyType, typename OffsetType, typename DType>
class PostprocessingForwardOp : public EmbeddingCollectionBase<KeyType, OffsetType, DType> {
 public:
  explicit PostprocessingForwardOp(OpKernelConstruction* ctx)
      : EmbeddingCollectionBase<KeyType, OffsetType, DType>(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // Prepare emb_vec_buffer_shape, this will be used in backward
    Tensor* emb_vec_buffer_shape = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(this->num_lookups_, {this->num_gpus_}, &emb_vec_buffer_shape));
#if TF_VERSION_MAJOR == 1
    int64* emb_vec_buffer_shape_ptr = emb_vec_buffer_shape->flat<int64>().data();
#else
    int64_t* emb_vec_buffer_shape_ptr = emb_vec_buffer_shape->flat<int64_t>().data();
#endif

    // Prepare input
    std::vector<sok::Tensor> emb_vec_buffer;
    for (int i = 0; i < this->num_gpus_; ++i) {
      const Tensor& emb_vec_buffer_tf = ctx->input(i);
      sok::Tensor emb_vec_buffer_tensor(sok::convert_tensor<DType>(&emb_vec_buffer_tf));
      emb_vec_buffer.push_back(emb_vec_buffer_tensor);
      emb_vec_buffer_shape_ptr[i] = emb_vec_buffer_tf.NumElements();
    }

    int batch_size = -1;
    std::vector<sok::Tensor> row_lengths;
    for (int i = 0; i < this->num_lookups_; ++i) {
      const Tensor& row_length = ctx->input(this->num_gpus_ + i);
      sok::Tensor row_length_tensor(sok::convert_tensor<OffsetType>(&row_length));
      row_lengths.push_back(row_length_tensor);
      if (batch_size == -1) {
        batch_size = row_length.NumElements();
      } else if (batch_size != row_length.NumElements()) {
        OP_REQUIRES(
            ctx, false,
            errors::InvalidArgument("shape[0] of each tensor in row_lengths are different."));
      }
    }

    const Tensor* sp_sum_buffer = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_sum", &sp_sum_buffer));
    sok::Tensor sp_sum_tensor(sok::convert_tensor<DType>(sp_sum_buffer));

    // Get global batch size
    int global_batch_size = batch_size * this->num_gpus_;

    const Tensor* hotness = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("hotness", &hotness));
    std::vector<int> hotness_vector;
    int* t_hotness = (int*)hotness->data();
    int64_t hotness_num = hotness->NumElements();
    for (int64_t i = 0; i < hotness_num; ++i) {
      hotness_vector.push_back(t_hotness[i]);
    }

    // Instance 3g embedding
    auto tf_backend = this->make_core_resource(ctx);
    this->update_meta(tf_backend, global_batch_size, hotness_vector);

    // Prepare output
    std::vector<sok::Tensor> emb_vec;
    for (int i = 0; i < this->num_lookups_; ++i) {
      Tensor* emb = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, {batch_size, this->dimensions_[i]}, &emb));
      emb_vec.push_back(sok::convert_tensor<DType>(emb));
    }

    // Do forward
    if (this->use_sp_weight_) {
      ::embedding::tf::network_forward::weighted_sparse_forward_per_gpu(
          tf_backend, *this->meta_, emb_vec_buffer, row_lengths, sp_sum_tensor, emb_vec);
    } else {
      ::embedding::tf::network_forward::sparse_forward_per_gpu(
          tf_backend, *this->meta_, emb_vec_buffer, row_lengths, emb_vec);
    }
  }
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, offset_type_tf, offset_type, dtype_tf, dtype) \
  REGISTER_KERNEL_BUILDER(Name("PostprocessingForward")                                           \
                              .Device(DEVICE_GPU)                                                 \
                              .HostMemory("emb_vec_buffer_shape")                                 \
                              .HostMemory("hotness")                                              \
                              .TypeConstraint<key_type_tf>("Tindices")                            \
                              .TypeConstraint<offset_type_tf>("Toffsets")                         \
                              .TypeConstraint<dtype_tf>("dtype"),                                 \
                          PostprocessingForwardOp<key_type, offset_type, dtype>)

#if TF_VERSION_MAJOR == 1
REGISTER_GPU_KERNELS(int64, int64_t, int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int64, int64_t, int32, int32_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, int32, int32_t, float, float);
REGISTER_GPU_KERNELS(int64, int64_t, int64, int64_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int32, int32_t, int64, int64_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int64, int64_t, int32, int32_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int32, int32_t, int32, int32_t, Eigen::half, __half);
#else
REGISTER_GPU_KERNELS(int64_t, int64_t, int64_t, int64_t, float, float);
REGISTER_GPU_KERNELS(int32_t, int32_t, int64_t, int64_t, float, float);
REGISTER_GPU_KERNELS(int64_t, int64_t, int32_t, int32_t, float, float);
REGISTER_GPU_KERNELS(int32_t, int32_t, int32_t, int32_t, float, float);
REGISTER_GPU_KERNELS(int64_t, int64_t, int64_t, int64_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int32_t, int32_t, int64_t, int64_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int64_t, int64_t, int32_t, int32_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int32_t, int32_t, int32_t, int32_t, Eigen::half, __half);
#endif

#undef REGISTER_GPU_KERNELS

// -----------------------------------------------------------------------------------------------
// PostprocessingBackward
// -----------------------------------------------------------------------------------------------
template <typename KeyType, typename OffsetType, typename DType>
class PostprocessingBackwardOp : public EmbeddingCollectionBase<KeyType, OffsetType, DType> {
 public:
  explicit PostprocessingBackwardOp(OpKernelConstruction* ctx)
      : EmbeddingCollectionBase<KeyType, OffsetType, DType>(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // Prepare input
    int batch_size = -1;
    std::vector<sok::Tensor> emb_vec_grad;
    for (int i = 0; i < this->num_lookups_; ++i) {
      const Tensor& emb_vec_grad_tf = ctx->input(i);
      sok::Tensor emb_vec_grad_tensor(sok::convert_tensor<DType>(&emb_vec_grad_tf));
      emb_vec_grad.push_back(emb_vec_grad_tensor);

      OP_REQUIRES(ctx, this->dimensions_[i] == emb_vec_grad_tf.dim_size(1),
                  errors::InvalidArgument("Invalid dimension"));
      if (batch_size == -1) {
        batch_size = emb_vec_grad_tf.dim_size(0);
      } else if (batch_size != emb_vec_grad_tf.dim_size(0)) {
        OP_REQUIRES(
            ctx, false,
            errors::InvalidArgument("shape[0] of each tensor in emb_vec_grad are different."));
      }
    }

    std::vector<sok::Tensor> row_lengths;
    for (int i = 0; i < this->num_lookups_; ++i) {
      const Tensor& row_length = ctx->input(this->num_lookups_ + 1 + i);
      sok::Tensor row_length_tensor(sok::convert_tensor<OffsetType>(&row_length));
      row_lengths.push_back(row_length_tensor);
    }

    // Get global batch size
    int global_batch_size = batch_size * this->num_gpus_;

    const Tensor* hotness = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("hotness", &hotness));
    std::vector<int> hotness_vector;
    int* t_hotness = (int*)hotness->data();
    int64_t hotness_num = hotness->NumElements();
    for (int64_t i = 0; i < hotness_num; ++i) {
      hotness_vector.push_back(t_hotness[i]);
    }

    const Tensor* sp_sum_buffer = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_sum", &sp_sum_buffer));
    sok::Tensor sp_sum_tensor(sok::convert_tensor<DType>(sp_sum_buffer));

    // instance 3g embedding
    auto tf_backend = this->make_core_resource(ctx);
    this->update_meta(tf_backend, global_batch_size, hotness_vector);

    // Prepare output
    const Tensor* emb_vec_buffer_shape = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("emb_vec_buffer_shape", &emb_vec_buffer_shape));
#if TF_VERSION_MAJOR == 1
    const int64* shape = emb_vec_buffer_shape->flat<int64>().data();
#else
    const int64_t* shape = emb_vec_buffer_shape->flat<int64_t>().data();
#endif
    std::vector<sok::Tensor> emb_vec_buffer_grad;
    for (int i = 0; i < this->num_gpus_; ++i) {
      Tensor* emb = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, {shape[i]}, &emb));
      emb_vec_buffer_grad.push_back(sok::convert_tensor<DType>(emb));
    }
    // Do backward
    if (this->use_sp_weight_) {
      ::embedding::tf::network_backward::weighted_backward_per_gpu(
          tf_backend, *this->meta_, emb_vec_grad, row_lengths, emb_vec_buffer_grad, sp_sum_tensor);
    } else {
      ::embedding::tf::network_backward::backward_per_gpu(tf_backend, *this->meta_, emb_vec_grad,
                                                          row_lengths, emb_vec_buffer_grad);
    }
  }
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, offset_type_tf, offset_type, dtype_tf, dtype) \
  REGISTER_KERNEL_BUILDER(Name("PostprocessingBackward")                                          \
                              .Device(DEVICE_GPU)                                                 \
                              .HostMemory("emb_vec_buffer_shape")                                 \
                              .HostMemory("hotness")                                              \
                              .TypeConstraint<key_type_tf>("Tindices")                            \
                              .TypeConstraint<offset_type_tf>("Toffsets")                         \
                              .TypeConstraint<dtype_tf>("dtype"),                                 \
                          PostprocessingBackwardOp<key_type, offset_type, dtype>)

#if TF_VERSION_MAJOR == 1
REGISTER_GPU_KERNELS(int64, int64_t, int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int64, int64_t, int32, int32_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, int32, int32_t, float, float);
REGISTER_GPU_KERNELS(int64, int64_t, int64, int64_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int32, int32_t, int64, int64_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int64, int64_t, int32, int32_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int32, int32_t, int32, int32_t, Eigen::half, __half);
#else
REGISTER_GPU_KERNELS(int64_t, int64_t, int64_t, int64_t, float, float);
REGISTER_GPU_KERNELS(int32_t, int32_t, int64_t, int64_t, float, float);
REGISTER_GPU_KERNELS(int64_t, int64_t, int32_t, int32_t, float, float);
REGISTER_GPU_KERNELS(int32_t, int32_t, int32_t, int32_t, float, float);
REGISTER_GPU_KERNELS(int64_t, int64_t, int64_t, int64_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int32_t, int32_t, int64_t, int64_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int64_t, int64_t, int32_t, int32_t, Eigen::half, __half);
REGISTER_GPU_KERNELS(int32_t, int32_t, int32_t, int32_t, Eigen::half, __half);
#endif

#undef REGISTER_GPU_KERNELS

}  // namespace tensorflow

#ifdef GOOGLE_CUDA
#ifdef TENSORFLOW_USE_GPU_EV
#include "lookup_adapter.hpp"

// clang-format off
namespace tensorflow {

template <typename KeyType, typename OffsetType, typename DType>
class LookupForwardEmbeddingVarGPUOp : public EmbeddingCollectionBase<KeyType, OffsetType, DType> {
 private:
  using VarType = EmbeddingVarGPU<KeyType, float>;
  EmbeddingVarGPUAdapter<KeyType, float> adapter_;

 public:
  explicit LookupForwardEmbeddingVarGPUOp(OpKernelConstruction* ctx) : EmbeddingCollectionBase<KeyType, OffsetType, DType>(ctx) {}
  
  void Compute(OpKernelContext* ctx) override {
    // std::vector<tf_shared_lock> locks;
    std::vector<core::RefCountPtr<VarType>> vars;
    std::vector<int> scale;
    std::vector<int> ev_size_per_lookup;
    for (int i = 0; i < this->num_lookups_; ++i) {
      auto handle = HandleFromInput(ctx, i);
      
      core::RefCountPtr<VarType> var;
      OP_REQUIRES_OK(ctx, LookupResource(ctx, handle, &var));
      ev_size_per_lookup.push_back(var->ValueLen());
      vars.push_back(std::move(var));

      if (this->shard_[i] < 0) {
        scale.push_back(this->num_gpus_);
      } else {
        scale.push_back(1);
      }
    }

    // stream
    auto device_ctx = ctx->op_device_context();
    OP_REQUIRES(ctx, device_ctx != nullptr, errors::Aborted("No valid device context."));
    cudaStream_t stream = stream_executor::gpu::AsGpuStreamValue(device_ctx->stream());
    
    const Tensor* key_recv_buffer = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("key_recv_buffer", &key_recv_buffer));
    sok::Tensor key_recv_buffer_tensor(sok::convert_tensor<KeyType>(key_recv_buffer));

    const Tensor* row_length_recv_buffer = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("row_length_recv_buffer", &row_length_recv_buffer));
    sok::Tensor row_length_recv_buffer_tensor(
        sok::convert_tensor<OffsetType>(row_length_recv_buffer));

    int global_batch_size = row_length_recv_buffer->NumElements() / this->num_lookups_;

    const Tensor* hotness = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("hotness", &hotness));
    std::vector<int> hotness_vector;
    int* t_hotness = (int*)hotness->data();
    int64_t hotness_num = hotness->NumElements();
    for (int64_t i =0;i<hotness_num;++i){
       hotness_vector.push_back(t_hotness[i]);
    }
    // Instance 3g embedding
    auto tf_backend = this->make_core_resource(ctx);
    this->update_meta(tf_backend, global_batch_size, hotness_vector);

    // Prepare outputs
    auto buffer_size_list = ::embedding::tf::model_forward::get_model_comm_buffer_size(*this->meta_, tf_backend->get_global_gpu_count(), global_batch_size);
    std::vector<sok::Tensor> emb_vec_model_buffer;
    for (size_t i = 0; i < buffer_size_list.size(); ++i) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(i, {static_cast<int64_t>(buffer_size_list[i])}, &output));
      emb_vec_model_buffer.push_back(sok::convert_tensor<DType>(output));
    }

    adapter_.set(ctx, vars, ev_size_per_lookup, stream);

    // Do forward
    int64_t num_model_key, num_model_offsets;
    sok::Tensor ret_model_key, ret_model_offset;
    ::embedding::tf::model_forward::sparse_forward_per_gpu(tf_backend, *this->meta_, key_recv_buffer_tensor, row_length_recv_buffer_tensor, &adapter_,
                                  emb_vec_model_buffer, &num_model_key, &num_model_offsets, &ret_model_key, &ret_model_offset);

    // Prepare model_key & model_offsets
    // Note the type of model_offsets is always uint32_t
    Tensor* model_key = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(this->num_gpus_, {num_model_key}, &model_key));
    sok::Tensor model_key_tensor(sok::convert_tensor<KeyType>(model_key));
    Tensor* model_offsets = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(this->num_gpus_ + 1, {num_model_offsets}, &model_offsets));
    sok::Tensor model_offsets_tensor(sok::convert_tensor<uint32_t>(model_offsets));

    // Copy tensors that will be used in backward
    ::embedding::tf::model_forward::copy_model_keys_and_offsets(tf_backend, ret_model_key, ret_model_offset, model_key_tensor, model_offsets_tensor);
    adapter_.clear_tmp_ev_list();
  }
};

#define REGISTER_GPU_KERNELS(key_type, offset_type, dtype)                   \
  REGISTER_KERNEL_BUILDER(                                                   \
    Name("LookupForwardEmbeddingVarGPU")                                     \
      .Device(DEVICE_GPU)                                                    \
      .HostMemory("handles")                                                 \
       .HostMemory("hotness")                                               \
      .TypeConstraint<key_type>("Tindices")                                  \
      .TypeConstraint<offset_type>("Toffsets")                               \
      .TypeConstraint<dtype>("dtype"),                                       \
    LookupForwardEmbeddingVarGPUOp<key_type, offset_type, dtype>)
REGISTER_GPU_KERNELS(int32, int32, float)
REGISTER_GPU_KERNELS(int32, int64, float)
REGISTER_GPU_KERNELS(int64, int32, float)
REGISTER_GPU_KERNELS(int64, int64, float)
#undef REGISTER_GPU_KERNELS
}  // namespace tensorflow
#endif
#endif
