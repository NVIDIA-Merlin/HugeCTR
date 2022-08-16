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
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "HugeCTR/embedding/all2all_embedding_collection.hpp"

#include "lookup/impl/core_impl/tf_impl_ops_test.hpp"
#include "lookup/impl/core_impl/tf_backend.hpp"
// clang-format on

using tensorflow::Allocator;
using tensorflow::DEVICE_CPU;
using tensorflow::DEVICE_GPU;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;

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
  }

  void Compute(OpKernelContext* context) override {
    std::shared_ptr<core::CoreResourceManager> tf_backend =
        std::make_shared<TFCoreResourceManager>(context, 0, 0, 1, 0, 1);
    core::Device gpu_device{core::DeviceType::GPU, gpu_id_};
    TFStorageImpl* tf_storge =
        dynamic_cast<tf_internal::TFStorageImpl*>(tf_backend->CreateStorage(gpu_device).get());
    const Tensor* tensor;

    tf_storge->extend((size_t)initial_size_);
    tf_storge->extend((size_t)extend_size_);
    tf_storge->allocate();
    tensor = tf_storge->get_tensor();
    int64_t tensor_size = tensor->NumElements();
    Allocator* allocator = tf_storge->allocator();

    LOG(WARNING) << " allocated pointer=" << static_cast<const void*>(tf_storge->get_ptr())
                 << ", total size=" << tensor_size << ", gpu_id=" << gpu_id_
                 << ", allocator=" << allocator->Name();

    core::Device cpu_device = core::DeviceType::CPU;
    TFStorageImpl* cpu_storge =
        dynamic_cast<tf_internal::TFStorageImpl*>(tf_backend->CreateStorage(cpu_device).get());
    cpu_storge->extend((size_t)initial_size_);
    cpu_storge->extend((size_t)extend_size_);
    cpu_storge->allocate();
  }

 private:
  int64_t initial_size_;
  int64_t extend_size_;
  int32_t gpu_id_;
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
class TfBackendTestTestOp : public OpKernel {
 public:
  explicit TfBackendTestTestOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32_t>();

    std::shared_ptr<core::CoreResourceManager> tf_backend =
        std::make_shared<TFCoreResourceManager>(context, 0, 0, 1, 0, 1);

    core::Storage storage =
        std::make_shared<TFStorageWrapper>(const_cast<int32_t*>(input.data()), input.size());
    std::vector<size_t> shape_vec{static_cast<size_t>(input.size())};
    auto t_impl = std::make_shared<core::TensorImpl>(storage, 0, shape_vec, core::DeviceType::GPU,
                                                     core::TensorScalarType::Int32);
    core::Tensor core_tensor{t_impl};

    std::unique_ptr<embedding::tf::IAll2AllEmbeddingCollectionSwizzleKey> swizzle_key =
        std::make_unique<embedding::tf::All2AllEmbeddingCollectionSwizzleKey>(tf_backend);
  }
};

// Register OP
REGISTER_OP("StorageImplTest")
    .Attr("initial_size: int=0")
    .Attr("extend_size: int=0")
    .Attr("gpu_id: int=0");

REGISTER_OP("GpuResourceImplTest");

REGISTER_OP("TfBackendTestTestOp").Input("input: int32");

// Register the GPU kernels.
REGISTER_KERNEL_BUILDER(Name("StorageImplTest").Device(DEVICE_GPU), StorageImplTestOp<GPUDevice>);

REGISTER_KERNEL_BUILDER(Name("GpuResourceImplTest").Device(DEVICE_GPU),
                        GpuResourceImplTestOp<GPUDevice>);

REGISTER_KERNEL_BUILDER(Name("TfBackendTestTestOp").Device(DEVICE_GPU),
                        TfBackendTestTestOp<GPUDevice>);

}  // namespace tf_internal
