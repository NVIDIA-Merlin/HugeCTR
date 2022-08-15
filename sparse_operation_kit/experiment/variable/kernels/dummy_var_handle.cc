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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"

#include "variable/kernels/dummy_var.h"
// clang-format on

namespace stream_executor {
namespace gpu {
cudaStream_t AsGpuStreamValue(Stream* stream);
}  // namespace gpu
}  // namespace stream_executor

namespace tensorflow {

// -----------------------------------------------------------------------------------------------
// DummyVarHandle
// -----------------------------------------------------------------------------------------------
template <typename KeyType, typename ValueType>
class DummyVarHandleOp : public OpKernel {
 public:
  explicit DummyVarHandleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("container", &container_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key_type", &key_type_));
    DataType dtype;
    PartialTensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape));
    OP_REQUIRES(ctx, shape.dims() == 2, errors::Aborted("len(shape) must be 2"));
    OP_REQUIRES(ctx, shape.dim_size(0) <= 0, errors::Aborted("shape[0] must be None currently"));
    OP_REQUIRES(ctx, shape.dim_size(1) > 0, errors::Aborted("shape[1] must > 0"));
    dtypes_and_shapes_.push_back({dtype, shape});
    info_ = Info();
  }

  void Compute(OpKernelContext* ctx) override {
    if (name_ == ResourceHandle::ANONYMOUS_NAME) {
      AllocatorAttributes attr;
      attr.set_on_host(true);
      Tensor handle;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}), &handle, attr));
      handle.scalar<ResourceHandle>()() = MakeResourceHandle<DummyVar<KeyType, ValueType>>(
          ctx, container_, name_, dtypes_and_shapes_);
      std::cout << "[SOK INFO] Create anonymous " + info_ << std::endl;
      ctx->set_output(0, handle);
    } else {
      if (!initialized_.load()) {
        mutex_lock ml(mutex_);
        // Checking again to see if another thread has initialized the resource.
        if (!initialized_.load()) {
          // Create handle
          AllocatorAttributes attr;
          attr.set_on_host(true);
          OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}), &resource_, attr));
          resource_.scalar<ResourceHandle>()() = MakeResourceHandle<DummyVar<KeyType, ValueType>>(
              ctx, container_, name_, dtypes_and_shapes_);
          // std::cout << "[SOK INFO] Create " + info_ << std::endl;
          initialized_.store(true);
        }
      }
      ctx->set_output(0, resource_);
    }
  }

 private:
  std::string container_;
  std::string name_;
  std::string info_;
  DataType key_type_;
  std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes_;
  mutex mutex_;
  Tensor resource_;
  std::atomic<bool> initialized_{false};

  std::string Info() {
    std::string dtype = DataTypeString(dtypes_and_shapes_[0].dtype);
    std::string key_type = DataTypeString(key_type_);
    std::string dim_0 = std::to_string(dtypes_and_shapes_[0].shape.dim_size(0));
    std::string dim_1 = std::to_string(dtypes_and_shapes_[0].shape.dim_size(1));
    std::string shape = "[" + dim_0 + "," + dim_1 + "]";
    std::string info = "<DummyVar> handle: " + container_ + "/" + name_ + ", ";
    info += "key_type: " + key_type + ", dtype: " + dtype + ", shape: " + shape;
    return info;
  }
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, dtype_tf, dtype)   \
  REGISTER_KERNEL_BUILDER(Name("DummyVarHandle")                       \
                              .Device(DEVICE_GPU)                      \
                              .HostMemory("resource")                  \
                              .TypeConstraint<key_type_tf>("key_type") \
                              .TypeConstraint<dtype_tf>("dtype"),      \
                          DummyVarHandleOp<key_type, dtype>)
#if TF_VERSION_MAJOR == 1
REGISTER_GPU_KERNELS(int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, float, float);
#else
REGISTER_GPU_KERNELS(int64_t, int64_t, float, float);
REGISTER_GPU_KERNELS(int32_t, int32_t, float, float);
#endif
#undef REGISTER_GPU_KERNELS

// -----------------------------------------------------------------------------------------------
// DummyVarInitialize
// -----------------------------------------------------------------------------------------------
#define AlreadyInitializedError(ctx)                                                    \
  do {                                                                                  \
    (ctx)->SetStatus(errors::Aborted("DummyVar has already been initialized. ",         \
                                     "This might be caused by that sess.run(init_op) ", \
                                     "is called more than once."));                     \
    return;                                                                             \
  } while (0)

template <typename KeyType, typename ValueType>
class DummyVarInitializeOp : public OpKernel {
 public:
  explicit DummyVarInitializeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_type", &var_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key_type", &key_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
    if (initialized_.load(std::memory_order_acquire)) {
      AlreadyInitializedError(ctx);
    }
    mutex_lock ml(mu_);
    // check again to see if another thread has initialized it.
    if (initialized_.load(std::memory_order_acquire)) {
      AlreadyInitializedError(ctx);
    }

    // Get initializer
    std::string initializer;
    const Tensor* init_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("initializer", &init_tensor));
    if (DT_STRING == init_tensor->dtype()) {
      initializer = std::string(init_tensor->flat<tstring>()(0));
      if (initializer == "") {
        initializer = "random";
      }
    } else if (DT_FLOAT == init_tensor->dtype()) {
      float val = init_tensor->flat<float>()(0);
      initializer = std::to_string(val);
    } else {
      OP_REQUIRES(ctx, false, errors::InvalidArgument("Unsupported initializer"));
    }

    // Get shape from handle
    auto handle = HandleFromInput(ctx, 0);
    auto dtypes_and_shapes = handle.dtypes_and_shapes();
    PartialTensorShape shape = dtypes_and_shapes[0].shape;
    int64_t rows = shape.dim_size(0);
    int64_t cols = shape.dim_size(1);

    // Get cuda stream of tensorflow
    auto device_ctx = ctx->op_device_context();
    OP_REQUIRES(ctx, device_ctx != nullptr, errors::Aborted("No valid device context."));
    cudaStream_t stream = stream_executor::gpu::AsGpuStreamValue(device_ctx->stream());

    // Create DummyVar
    // TODO: maybe use LookupOrCreateResource is better?
    DummyVar<KeyType, ValueType>* var = new DummyVar<KeyType, ValueType>(
        rows, cols, var_type_, initializer, handle.container(), handle.name(), stream);
    OP_REQUIRES_OK(ctx, CreateResource<DummyVar<KeyType, ValueType>>(ctx, handle, var));

    // set the initialize flag
    initialized_.store(true, std::memory_order_release);

    std::string info =
        "[SOK INFO] " + handle.container() + "/" + handle.name() + " is initialized, ";
    info += "var_type: " + var_type_ + ", initializer: " + initializer;
    info += ", key_type: " + DataTypeString(key_type_);
    info += ", dtype: " + DataTypeString(dtype_);
    // std::cout << info << std::endl;
  }

 private:
  std::string var_type_;
  mutex mu_;
  std::atomic<bool> initialized_{false};
  DataType key_type_;
  DataType dtype_;
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, dtype_tf, dtype)   \
  REGISTER_KERNEL_BUILDER(Name("DummyVarInitialize")                   \
                              .Device(DEVICE_GPU)                      \
                              .HostMemory("resource")                  \
                              .HostMemory("initializer")               \
                              .TypeConstraint<key_type_tf>("key_type") \
                              .TypeConstraint<dtype_tf>("dtype"),      \
                          DummyVarInitializeOp<key_type, dtype>)
#if TF_VERSION_MAJOR == 1
REGISTER_GPU_KERNELS(int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, float, float);
#else
REGISTER_GPU_KERNELS(int64_t, int64_t, float, float);
REGISTER_GPU_KERNELS(int32_t, int32_t, float, float);
#endif
#undef REGISTER_GPU_KERNELS

// -----------------------------------------------------------------------------------------------
// DummyVarShape
// -----------------------------------------------------------------------------------------------
template <typename KeyType, typename ValueType, typename OutType>
class DummyVarShapeOp : public OpKernel {
 public:
  explicit DummyVarShapeOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<DummyVar<KeyType, ValueType>> var;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));
    var->mu()->lock_shared();
    int64_t rows = var->rows();
    int64_t cols = var->cols();
    var->mu()->unlock_shared();
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {2}, &output));
    *(static_cast<OutType*>(output->data())) = rows;
    *(static_cast<OutType*>(output->data()) + 1) = cols;
  }
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, dtype_tf, dtype, out_type_tf, out_type) \
  REGISTER_KERNEL_BUILDER(Name("DummyVarShape")                                             \
                              .Device(DEVICE_GPU)                                           \
                              .HostMemory("input")                                          \
                              .HostMemory("output")                                         \
                              .TypeConstraint<out_type_tf>("out_type")                      \
                              .TypeConstraint<key_type_tf>("key_type")                      \
                              .TypeConstraint<dtype_tf>("dtype"),                           \
                          DummyVarShapeOp<key_type, dtype, out_type>)
#if TF_VERSION_MAJOR == 1
REGISTER_GPU_KERNELS(int64, int64_t, float, float, int32, int32_t);
REGISTER_GPU_KERNELS(int32, int32_t, float, float, int32, int32_t);
REGISTER_GPU_KERNELS(int64, int64_t, float, float, int64, int64_t);
REGISTER_GPU_KERNELS(int32, int32_t, float, float, int64, int64_t);
#else
REGISTER_GPU_KERNELS(int64_t, int64_t, float, float, int32_t, int32_t);
REGISTER_GPU_KERNELS(int32_t, int32_t, float, float, int32_t, int32_t);
REGISTER_GPU_KERNELS(int64_t, int64_t, float, float, int64_t, int64_t);
REGISTER_GPU_KERNELS(int32_t, int32_t, float, float, int64_t, int64_t);
#endif
#undef REGISTER_GPU_KERNELS

}  // namespace tensorflow
