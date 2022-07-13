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
#include "tf_impl_ops_test.hpp"

#include "../buffer.hpp"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tf_backend.hpp"

using tensorflow::Allocator;
using tensorflow::DEVICE_CPU;
using tensorflow::DEVICE_GPU;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;

using core::Device;
using core::DeviceType;
using tf_internal::GPUResource;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace tf_internal {

template <typename Device>
class StorageImplTestOp : public OpKernel {
 public:
  explicit StorageImplTestOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("initial_size", &initial_size_));
    OP_REQUIRES_OK(context, context->GetAttr("extend_size", &extend_size_));
    OP_REQUIRES_OK(context, context->GetAttr("gpu_id", &gpu_id_));
    OP_REQUIRES_OK(context, context->GetAttr("cpu_id", &cpu_id_));
    OP_REQUIRES_OK(context, context->GetAttr("on_gpu", &on_gpu_));
  }

  void Compute(OpKernelContext* context) override {
    std::shared_ptr<core::CoreResourceManager> tf_backend =
        std::make_shared<TFCoreResourceManager>(context, 0, 1, 1);
    DeviceType dev_type = (on_gpu_ ? DeviceType::GPU : DeviceType::CPU);
    core::Device device(dev_type, gpu_id_);
    TFStorageImpl* tf_storge =
        dynamic_cast<tf_internal::TFStorageImpl*>(tf_backend->CreateStorage(device).get());
    const Tensor* tensor;

    tf_storge->extend((size_t)initial_size_);
    tf_storge->extend((size_t)extend_size_);
    tf_storge->allocate();
    tensor = tf_storge->get_tensor();
    int64_t tensor_size = tensor->NumElements();
    Allocator* allocator = tf_storge->allocator();

    LOG(WARNING) << " allocated pointer=" << static_cast<const void*>(tf_storge->get_ptr())
                 << ", total size=" << tensor_size << ", gpu_id=" << gpu_id_
                 << ", cpu_id=" << cpu_id_ << ", on_gpu=" << (on_gpu_ ? "True" : "False")
                 << ", allocator=" << allocator->Name();
  }

 private:
  int64_t initial_size_;
  int64_t extend_size_;
  int32_t gpu_id_;
  int32_t cpu_id_;
  bool on_gpu_;
};

template <typename Device>
class GpuResourceImplTestOp : public OpKernel {
 public:
  explicit GpuResourceImplTestOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    tf_internal::GPUResource gpu_resource(context);
    cudaStream_t stream = gpu_resource.get_stream();
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err == cudaSuccess) {
      LOG(WARNING) << "The default stream is got successfully!";
    }
  }
};

template <typename Device>
class TfBackendTestOpTest : public OpKernel {
 public:
  explicit TfBackendTestOpTest(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    std::shared_ptr<core::CoreResourceManager> tf_backend =
        std::make_shared<TFCoreResourceManager>(context, 0, 1, 1);
    auto buffer_ptr = core::GetBuffer(tf_backend);
    auto t = buffer_ptr->reserve({0}, DeviceType::CPU, core::TensorScalarType::Int32);
    buffer_ptr->allocate();
    t.get<int32_t>()[0] = 1;
    if (t.get<int32_t>()[0] != 1) {
      LOG(WARNING) << "TfBackendTestOpTest Fail!";
    }
  }
};

// Register OP
REGISTER_OP("StorageImplTest")
    .Attr("initial_size: int=0")
    .Attr("extend_size: int=0")
    .Attr("gpu_id: int=0")
    .Attr("cpu_id: int=0")
    .Attr("on_gpu: bool=true");

REGISTER_OP("GpuResourceImplTest");

REGISTER_OP("TfBackendTestOpTest")

// Register the CPU kernels.
#define REGISTER_CPU() \
  REGISTER_KERNEL_BUILDER(Name("StorageImplTest").Device(DEVICE_CPU), StorageImplTestOp<CPUDevice>);
REGISTER_CPU();
#undef REGISTER_CPU

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU() \
  REGISTER_KERNEL_BUILDER(Name("StorageImplTest").Device(DEVICE_GPU), StorageImplTestOp<GPUDevice>);
REGISTER_GPU();
#undef REGISTER_GPU
#endif  // GOOGLE_CUDA

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU()                                                    \
  REGISTER_KERNEL_BUILDER(Name("GpuResourceImplTest").Device(DEVICE_GPU), \
                          GpuResourceImplTestOp<GPUDevice>);
REGISTER_GPU();
#undef REGISTER_GPU
#endif  // GOOGLE_CUDA

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU()                                                    \
  REGISTER_KERNEL_BUILDER(Name("TfBackendTestOpTest").Device(DEVICE_GPU), \
                          GpuResourceImplTestOp<GPUDevice>);
REGISTER_GPU();
#undef REGISTER_GPU
#endif  // GOOGLE_CUDA

}  // namespace tf_internal
