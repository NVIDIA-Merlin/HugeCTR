/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/stream_executor/cuda/cuda_activation.h>
#include <tensorflow/stream_executor/gpu/gpu_stream.h>
#include <tensorflow/stream_executor/stream.h>
#include <tensorflow/stream_executor/stream_executor.h>

#include <hps/plugin/facade.hpp>

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;
using namespace HierarchicalParameterServer;
using namespace stream_executor::gpu;
using namespace stream_executor::cuda;

#ifdef HPS_ASYNC_OP
template <typename Device>
class Lookup : public AsyncOpKernel {
 public:
  explicit Lookup(OpKernelConstruction *ctx) : AsyncOpKernel(ctx), thread_pool_("", 1) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_name", &model_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_id", &table_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("emb_vec_size", &emb_vec_size_));
  }

  void ComputeAsync(OpKernelContext *ctx, DoneCallback done) override {
    auto work_func = [this, ctx, done]() {
      auto stream = ctx->op_device_context()->stream();
      ScopedActivateExecutorContext scoped_activation{stream->parent()};
      cudaStream_t gpu_stream = AsGpuStreamValue(stream);

      Tensor const *status_tensor = nullptr;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->input("init_status", &status_tensor), done);
      std::string init_status = status_tensor->flat<tstring>()(0);
      OP_REQUIRES_ASYNC(ctx, init_status == "OK",
                        errors::Aborted("hierarchical parameter server is not initialized."), done);

      Tensor const *values_tensor = nullptr;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->input("values", &values_tensor), done);

      Tensor const *global_replica_id_tensor = nullptr;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->input("global_replica_id", &global_replica_id_tensor), done);
      const int32_t global_replica_id_value = global_replica_id_tensor->scalar<int32_t>()();

      // allocate output
      Tensor *emb_vector_tensor = nullptr;
      TensorShape emb_vector_tensor_shape = values_tensor->shape();
      emb_vector_tensor_shape.AppendShape({emb_vec_size_});

      OP_REQUIRES_OK_ASYNC(
          ctx, ctx->allocate_output(0, emb_vector_tensor_shape, &emb_vector_tensor), done);

      // do forward propagation
      try {
        size_t num_keys = static_cast<size_t>(values_tensor->NumElements());
        size_t emb_vec_size = static_cast<size_t>(emb_vector_tensor->shape().dim_sizes().back());
        const void *values_ptr = values_tensor->data();
        void *emb_vector_ptr = emb_vector_tensor->data();
        bool i64_input_tensor = DT_INT64 == values_tensor->dtype();
        Facade::instance()->forward(model_name_.c_str(), table_id_, global_replica_id_value,
                                    num_keys, emb_vec_size, values_ptr, emb_vector_ptr,
                                    i64_input_tensor, gpu_stream);
      } catch (std::exception const &error) {
        ctx->SetStatus(errors::Aborted(error.what()));
        done();
        return;
      }
      done();
    };
    thread_pool_.submit(work_func);
  }

 private:
  std::string model_name_;
  tensorflow::int32 table_id_;
  tensorflow::int32 emb_vec_size_;
  HugeCTR::ThreadPool thread_pool_;
};

#else
template <typename Device>
class Lookup : public OpKernel {
 public:
  explicit Lookup(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_name", &model_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_id", &table_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("emb_vec_size", &emb_vec_size_));
  }

  void Compute(OpKernelContext *ctx) override {
    cudaStream_t gpu_stream = AsGpuStreamValue(ctx->op_device_context()->stream());

    Tensor const *status_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("init_status", &status_tensor));
    std::string init_status = status_tensor->flat<tstring>()(0);
    OP_REQUIRES(ctx, init_status == "OK",
                errors::Aborted("hierarchical parameter server is not initialized."));

    Tensor const *values_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("values", &values_tensor));

    Tensor const *global_replica_id_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("global_replica_id", &global_replica_id_tensor));
    const int32_t global_replica_id_value = global_replica_id_tensor->scalar<int32_t>()();

    // allocate output
    Tensor *emb_vector_tensor = nullptr;
    TensorShape emb_vector_tensor_shape = values_tensor->shape();
    emb_vector_tensor_shape.AppendShape({emb_vec_size_});

    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, emb_vector_tensor_shape, &emb_vector_tensor));

    // do forward propagation
    try {
      size_t num_keys = static_cast<size_t>(values_tensor->NumElements());
      size_t emb_vec_size = static_cast<size_t>(emb_vector_tensor->shape().dim_sizes().back());
      const void *values_ptr = values_tensor->data();
      void *emb_vector_ptr = emb_vector_tensor->data();
      bool i64_input_tensor = DT_INT64 == values_tensor->dtype();
      Facade::instance()->forward(model_name_.c_str(), table_id_, global_replica_id_value, num_keys,
                                  emb_vec_size, values_ptr, emb_vector_ptr, i64_input_tensor,
                                  gpu_stream);
    } catch (std::exception const &error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }
  }

  bool IsExpensive() override { return true; }

 private:
  std::string model_name_;
  tensorflow::int32 table_id_;
  tensorflow::int32 emb_vec_size_;
};
#endif

REGISTER_KERNEL_BUILDER(Name("Lookup").Device(DEVICE_GPU).HostMemory("global_replica_id"),
                        Lookup<GPUDevice>);

}  // namespace tensorflow