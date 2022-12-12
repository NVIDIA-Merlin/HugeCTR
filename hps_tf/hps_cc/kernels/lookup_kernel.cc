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

#include "config.h"
#include "hps/plugin/facade.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;
using namespace HierarchicalParameterServer;
using namespace stream_executor::gpu;

template <typename Device>
class Lookup : public OpKernel {
 public:
  explicit Lookup(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_name", &model_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_id", &table_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("emb_vec_size", &emb_vec_size_));
  }

  void Compute(OpKernelContext *ctx) override {
    // This stream synchronization is needed since HPS embedding lookup currently does not use the
    // CUDA stream in the TF context, in case that there are some ops/kernels processing the device
    // keys on this stream before HPS embedding lookup
    cudaStream_t gpu_stream = AsGpuStreamValue(ctx->op_device_context()->stream());
    HCTR_LIB_THROW(cudaStreamSynchronize(gpu_stream));

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
      Facade::instance()->forward(model_name_.c_str(), table_id_, global_replica_id_value, num_keys,
                                  emb_vec_size, values_ptr, emb_vector_ptr);
    } catch (std::exception const &error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }
  }

 private:
  std::string model_name_;
  tensorflow::int32 table_id_;
  tensorflow::int32 emb_vec_size_;
};

REGISTER_KERNEL_BUILDER(Name("Lookup").Device(DEVICE_GPU).HostMemory("global_replica_id"),
                        Lookup<GPUDevice>);

}  // namespace tensorflow