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
  std::vector<int> hotness_;
  std::vector<int> shard_;
  std::vector<int> dimensions_;

  int rank_;
  int num_ranks_;
  int id_in_local_rank_;
  int num_gpus_;

  int num_gpu_per_rank_;
  int global_gpu_id_;
  int num_local_lookups_;

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
    for (int i = 0; i < shard_matrix.size(); ++i) {
      for (int j = 0; j < shard_.size(); ++j) {
        if (shard_[j] < 0 || shard_[j] == i) {
          // Distributed embedding
          // Localized embedding with embedding table on i_th GPU
          shard_matrix[i].push_back(1);
        } else {
          // Localized embedding with embedding table on other GPU
          shard_matrix[i].push_back(0);
        }
      }
    }

    // std::cout << "global_gpu_id: " << global_gpu_id_ << ", shard_matrix:" << std::endl;
    // for (int i = 0; i < shard_matrix.size(); ++i) {
    //   for (int j = 0; j < shard_matrix[i].size(); ++j) {
    //     std::cout << shard_matrix[i][j] << ", ";
    //   }
    //   std::cout << std::endl;
    // }
  }

 public:
  explicit EmbeddingCollectionBase(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_lookups", &num_lookups_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("combiners", &combiners_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hotness", &hotness_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shard", &shard_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dimensions", &dimensions_));

    OP_REQUIRES(ctx, combiners_.size() == num_lookups_,
                errors::InvalidArgument("len(combiners) != num_lookups."));
    OP_REQUIRES(ctx, hotness_.size() == num_lookups_,
                errors::InvalidArgument("len(hotness) != num_lookups."));
    OP_REQUIRES(ctx, shard_.size() == num_lookups_,
                errors::InvalidArgument("len(shard) != num_lookups."));
    OP_REQUIRES(ctx, dimensions_.size() == this->num_lookups_,
                errors::InvalidArgument("len(dimensions) != num_lookups."));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("rank", &rank_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_ranks", &num_ranks_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("id_in_local_rank", &id_in_local_rank_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_gpus", &num_gpus_));

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
    for (int i = 0; i < shard_.size(); ++i) {
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
// PreprocessingForward
// -----------------------------------------------------------------------------------------------
template <typename KeyType, typename OffsetType, typename DType>
class PreprocessingForwardOp : public EmbeddingCollectionBase<KeyType, OffsetType, DType> {
 public:
  explicit PreprocessingForwardOp(OpKernelConstruction* ctx)
      : EmbeddingCollectionBase<KeyType, OffsetType, DType>(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // instance 3g embedding
    std::unique_ptr<sok::ISwizzleKey> model =
        std::make_unique<sok::SwizzleKey>(this->make_core_resource(ctx));

    // Prepare inputs
    int64_t num_keys = 0;
    int64_t num_row_lengths = 0;
    std::vector<sok::Tensor> keys_sok;
    std::vector<sok::Tensor> row_lengths_sok;
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

    // Do forward
    model->sparse_forward_per_gpu(keys_sok, row_lengths_sok, key_send_buffer_sok,
                                  row_length_send_buffer_sok);
  }
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, offset_type_tf, offset_type) \
  REGISTER_KERNEL_BUILDER(Name("PreprocessingForward")                           \
                              .Device(DEVICE_GPU)                                \
                              .TypeConstraint<key_type_tf>("Tindices")           \
                              .TypeConstraint<offset_type_tf>("Toffsets"),       \
                          PreprocessingForwardOp<key_type, offset_type, float>)

#if TF_VERSION_MAJOR == 1
REGISTER_GPU_KERNELS(int64, int64_t, int64, int64_t);
REGISTER_GPU_KERNELS(int64, int64_t, int32, int32_t);
REGISTER_GPU_KERNELS(int32, int32_t, int64, int64_t);
REGISTER_GPU_KERNELS(int32, int32_t, int32, int32_t);
#else
REGISTER_GPU_KERNELS(int64_t, int64_t, int64_t, int64_t);
REGISTER_GPU_KERNELS(int64_t, int64_t, int32_t, int32_t);
REGISTER_GPU_KERNELS(int32_t, int32_t, int64_t, int64_t);
REGISTER_GPU_KERNELS(int32_t, int32_t, int32_t, int32_t);
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

    // Prepare ILookup (i.e. embedding table)
    adapter_.set(vars, locks, this->dimensions_, scale, stream);

    // Prepare inputs (except handles)
    const Tensor* key_recv_buffer = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("key_recv_buffer", &key_recv_buffer));
    sok::Tensor key_recv_buffer_tensor(sok::convert_tensor<KeyType>(key_recv_buffer));

    const Tensor* row_length_recv_buffer = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("row_length_recv_buffer", &row_length_recv_buffer));
    sok::Tensor row_length_recv_buffer_tensor(
        sok::convert_tensor<OffsetType>(row_length_recv_buffer));

    int global_batch_size = row_length_recv_buffer->NumElements() / this->num_lookups_;

    // Instance 3g embedding
    auto tf_backend = this->make_core_resource(ctx);
    std::vector<std::vector<int>> shard_matrix;
    this->make_shard_matrix(shard_matrix);

    sok::EmbeddingCollectionParam ebc_param = sok::make_embedding_collection_param<KeyType, OffsetType, DType>(shard_matrix, this->num_lookups_, this->combiners_, this->hotness_, this->dimensions_, global_batch_size);

    std::unique_ptr<sok::IModelForward> model =
        std::make_unique<sok::ModelForward>(tf_backend, ebc_param);

    // Prepare outputs
    auto buffer_size_list = model->get_model_comm_buffer_size(global_batch_size);
    std::vector<sok::Tensor> emb_vec_model_buffer;
    for (int i = 0; i < buffer_size_list.size(); ++i) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(i, {static_cast<int64_t>(buffer_size_list[i])}, &output));
      emb_vec_model_buffer.push_back(sok::convert_tensor<DType>(output));
    }

    // Do forward
    int64_t num_model_key, num_model_offsets;
    model->sparse_forward_per_gpu(key_recv_buffer_tensor, row_length_recv_buffer_tensor, &adapter_,
                                  emb_vec_model_buffer, &num_model_key, &num_model_offsets);

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
    model->copy_model_keys_and_offsets(model_key_tensor, model_offsets_tensor);
  }
};

// clang-format off
#define REGISTER_GPU_KERNELS(key_type_tf, key_type, offset_type_tf, offset_type, dtype_tf, dtype)  \
  REGISTER_KERNEL_BUILDER(Name("LookupForward")                                                    \
                              .Device(DEVICE_GPU)                                                  \
                              .HostMemory("handles")                                               \
                              .TypeConstraint<key_type_tf>("Tindices")                             \
                              .TypeConstraint<offset_type_tf>("Toffsets")                          \
                              .TypeConstraint<dtype_tf>("dtype"),                                  \
                          LookupForwardOp<key_type, offset_type, dtype, Var,                       \
                                          sok::TFAdapter<key_type, dtype>>)                        \
  REGISTER_KERNEL_BUILDER(Name("LookupForwardDynamic")                                             \
                              .Device(DEVICE_GPU)                                                  \
                              .HostMemory("handles")                                               \
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

    // Instance 3g embedding
    auto tf_backend = this->make_core_resource(ctx);
    std::vector<std::vector<int>> shard_matrix;
    this->make_shard_matrix(shard_matrix);
    sok::EmbeddingCollectionParam ebc_param = sok::make_embedding_collection_param<KeyType, OffsetType, DType>(shard_matrix, this->num_lookups_, this->combiners_, this->hotness_, this->dimensions_, batch_size);

    std::unique_ptr<sok::IModelBackward> model =
        std::make_unique<sok::ModelBackward>(tf_backend, ebc_param);

    // Do backward
    std::vector<int> num_unique_key_per_table, unique_id_space_list;
    model->sparse_backward_per_gpu(emb_vec_buffer_grad, model_key_tensor, model_offsets_tensor,
                                   &num_unique_key_per_table, &unique_id_space_list);

    // Prepare output
    std::vector<int> num_unique_key_per_handle;
    num_unique_key_per_handle.resize(this->num_lookups_, -1);
    for (int i = 0; i < num_unique_key_per_table.size(); ++i) {
      int id_space = unique_id_space_list[i];
      num_unique_key_per_handle[id_space] = num_unique_key_per_table[i];
    }
    for (int i = 0; i < this->num_lookups_; ++i) {
      if (num_unique_key_per_handle[i] == -1) {
        Tensor* unused = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(i, {0}, &unused));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(this->num_lookups_ + i, {0}, &unused));
      }
    }
    std::vector<sok::Tensor> unique_key, grad;
    for (int i = 0; i < num_unique_key_per_table.size(); ++i) {
      int id_space = unique_id_space_list[i];
      Tensor* unique_key_tf = nullptr;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(id_space, {num_unique_key_per_table[i]}, &unique_key_tf));
      Tensor* grad_tf = nullptr;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(id_space + this->num_lookups_,
                                    {num_unique_key_per_table[i], this->dimensions_[id_space]},
                                    &grad_tf));
      unique_key.push_back(sok::convert_tensor<KeyType>(unique_key_tf));
      grad.push_back(sok::convert_tensor<DType>(grad_tf));
    }

    // Copy output
    model->copy_backward_key_and_emb_vec(unique_key, grad);
  }
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, offset_type_tf, offset_type, dtype_tf, dtype) \
  REGISTER_KERNEL_BUILDER(Name("LookupBackward")                                                  \
                              .Device(DEVICE_GPU)                                                 \
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

    // Get global batch size
    int global_batch_size = batch_size * this->num_gpus_;

    // Instance 3g embedding
    auto tf_backend = this->make_core_resource(ctx);
    std::vector<std::vector<int>> shard_matrix;
    this->make_shard_matrix(shard_matrix);
    sok::EmbeddingCollectionParam ebc_param = sok::make_embedding_collection_param<KeyType, OffsetType, DType>(shard_matrix, this->num_lookups_, this->combiners_, this->hotness_, this->dimensions_, global_batch_size);

    std::unique_ptr<sok::INetworkForward> network_forward =
        std::make_unique<sok::NetworkForward>(tf_backend, ebc_param);

    // Prepare output
    std::vector<sok::Tensor> emb_vec;
    for (int i = 0; i < this->num_lookups_; ++i) {
      Tensor* emb = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, {batch_size, this->dimensions_[i]}, &emb));
      emb_vec.push_back(sok::convert_tensor<DType>(emb));
    }

    // Do forward
    network_forward->sparse_forward_per_gpu(emb_vec_buffer, row_lengths, emb_vec);
  }
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, offset_type_tf, offset_type, dtype_tf, dtype) \
  REGISTER_KERNEL_BUILDER(Name("PostprocessingForward")                                           \
                              .Device(DEVICE_GPU)                                                 \
                              .HostMemory("emb_vec_buffer_shape")                                 \
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

    // instance 3g embedding
    auto tf_backend = this->make_core_resource(ctx);
    std::vector<std::vector<int>> shard_matrix;
    this->make_shard_matrix(shard_matrix);
    sok::EmbeddingCollectionParam ebc_param = sok::make_embedding_collection_param<KeyType, OffsetType, DType>(shard_matrix, this->num_lookups_, this->combiners_, this->hotness_, this->dimensions_, global_batch_size);

    std::unique_ptr<sok::INetworkBackward> network_backward =
        std::make_unique<sok::NetworkBackward>(tf_backend, ebc_param);

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
    network_backward->backward_per_gpu(emb_vec_grad, row_lengths, emb_vec_buffer_grad);
  }
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, offset_type_tf, offset_type, dtype_tf, dtype) \
  REGISTER_KERNEL_BUILDER(Name("PostprocessingBackward")                                          \
                              .Device(DEVICE_GPU)                                                 \
                              .HostMemory("emb_vec_buffer_shape")                                 \
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
