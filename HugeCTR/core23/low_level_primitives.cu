
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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <algorithm>
#include <cmath>
#include <core23/allocator_factory.hpp>
#include <core23/cuda_primitives.cuh>
#include <core23/cuda_stream.hpp>
#include <core23/data_type_helpers.cuh>
#include <core23/details/host_launch_helpers.hpp>
#include <core23/details/pool_cuda_allocator.hpp>
#include <core23/device.hpp>
#include <core23/device_guard.hpp>
#include <core23/logger.hpp>
#include <core23/low_level_primitives.hpp>

namespace HugeCTR {

namespace core23 {

template <typename Type>
void fill_cpu(Type* data, int64_t num_elements, const Type val, const Device& device,
              std::optional<CUDAStream> stream_or) {
  if (stream_or) {
    FillParams<Type>* params = new FillParams<Type>(data, num_elements, val);
    HCTR_LIB_THROW(cudaLaunchHostFunc(static_cast<cudaStream_t>(stream_or.value()),
                                      static_cast<cudaHostFn_t>(fill_wrapper<Type>), params));
  } else {
    std::fill(data, data + num_elements, val);
  }
}

template <typename Type>
void fill_gpu(Type* data, int64_t num_elements, const Type val, const Device& device,
              std::optional<CUDAStream> stream_or) {
  CUDAStream stream;
  if (stream_or) {
    stream = *stream_or;
  }
  dim3 block(1024);
  dim3 grid((num_elements + block.x - 1) / block.x);
  fill_kernel<<<grid, block, 0, stream()>>>(data, num_elements, val);

  if (!stream_or) {
    cudaStreamSynchronize(stream());
  }
}

template <typename Type>
void fill_common(Type* data, int64_t num_elements, const Type val, const Device& device,
                 std::optional<CUDAStream> stream_or) {
  DeviceGuard device_guard(device);
  if (device.type() == DeviceType::CPU) {
    fill_cpu(data, num_elements, val, device, stream_or);
  } else {
    fill_gpu(data, num_elements, val, device, stream_or);
  }
}

template <typename Type>
void fill_sync(Type* data, int64_t num_elements, const Type val, const Device& device) {
  fill_common(data, num_elements, val, device, {});
}

template <typename Type>
void fill_async(Type* data, int64_t num_elements, const Type val, const Device& device,
                CUDAStream stream) {
  fill_common(data, num_elements, val, device, stream);
}

template <typename DstType, typename SrcType, typename Op>
void transform_async_common(DstType* dst, const SrcType* src, int64_t num_elements,
                            const Device& dst_device, const Device& src_device, CUDAStream stream,
                            Op op) {
  DeviceGuard device_guard(src_device.type() == DeviceType::CPU ? dst_device : src_device);
  if (dst_device == src_device) {
    if (src_device.type() == DeviceType::CPU) {
      TransformParams<DstType, SrcType, Op>* params =
          new TransformParams<DstType, SrcType, Op>(dst, src, num_elements, op);
      HCTR_LIB_THROW(cudaLaunchHostFunc(
          stream(), static_cast<cudaHostFn_t>(transform_wrapper<DstType, SrcType, Op>), params));
    } else {
      dim3 block(1024);
      dim3 grid((num_elements + block.x - 1) / block.x);
      transform_kernel<<<grid, block, 0, stream()>>>(dst, src, num_elements, op);
    }
  } else {
    if (src_device.type() == DeviceType::CPU) {
      AllocatorParams allocator_params;
      // TODO: change this line after introducing the ResourceManager
      allocator_params.custom_factory = [](const auto& params, const auto& device) {
        return std::unique_ptr<Allocator>(new PoolCUDAAllocator(device));
      };
      auto allocator = GetAllocator(allocator_params, dst_device);
      SrcType* tmp =
          static_cast<SrcType*>(allocator->allocate(num_elements * sizeof(SrcType), stream));
      copy_async(tmp, src, num_elements * sizeof(SrcType), dst_device, src_device, stream);
      transform_async_common(dst, tmp, num_elements, dst_device, dst_device, stream, op);
      allocator->deallocate(tmp, stream);
    } else {
      AllocatorParams allocator_params;
      // TODO: change this line after introducing the ResourceManager
      allocator_params.custom_factory = [](const auto& params, const auto& device) {
        return std::unique_ptr<Allocator>(new PoolCUDAAllocator(device));
      };
      auto allocator = GetAllocator(allocator_params, src_device);
      DstType* tmp =
          static_cast<DstType*>(allocator->allocate(num_elements * sizeof(DstType), stream));
      transform_async_common(tmp, src, num_elements, src_device, src_device, stream, op);
      copy_async(dst, tmp, num_elements * sizeof(DstType), dst_device, src_device, stream);
      allocator->deallocate(tmp, stream);
    }
  }
}

#define DEFINE_FILL(Type, _)                                                        \
  template void fill_sync<Type>(Type * data, int64_t num_elements, const Type val,  \
                                const Device& device);                              \
  template void fill_async<Type>(Type * data, int64_t num_elements, const Type val, \
                                 const Device& device, CUDAStream stream);

ALL_DATA_TYPES_SUPPORTED(DEFINE_FILL)

template <typename DstType, typename SrcType>
void convert_async(DstType* dst_data, const SrcType* src_data, int64_t num_elements,
                   const Device& dst_device, const Device& src_device, CUDAStream stream) {
  transform_async_common(dst_data, src_data, num_elements, dst_device, src_device, stream,
                         [] HCTR_HOST_DEVICE(SrcType in) -> DstType {
                           return TypeConverter<DstType, SrcType>::value(in);
                         });
}

#define DEFINE_CONVERT_ASYNC_COMMON(DstType, SrcType, _)                                          \
  template void convert_async<DstType, SrcType>(DstType*, const SrcType*, int64_t, const Device&, \
                                                const Device&, CUDAStream);
ALL_DATA_CONVERSIONS_SUPPORTED(DEFINE_CONVERT_ASYNC_COMMON)

template <typename Type>
void uniform_async(Type* data, int64_t num_elements, const Type a, const Type b,
                   const Device& device, CURANDGenerator generator, CUDAStream stream) {
  static_assert(std::is_floating_point<Type>::value);

  DeviceGuard device_guard(device);
  generator.set_stream(stream);
  if constexpr (ToScalarType<Type>::value == ScalarType::Float) {
    HCTR_LIB_THROW(curandGenerateUniform(generator(), data, num_elements));
  } else {
    static_assert(std::is_same<Type, double>::value);
    HCTR_LIB_THROW(curandGenerateUniformDouble(generator(), data, num_elements));
  }
  transform_async_common(data, data, num_elements, device, device, stream,
                         [a, b] HCTR_HOST_DEVICE(Type val) { return val * (b - a) + a; });
}

template <typename Type>
void normal_async(Type* data, int64_t num_elements, const Type mean, const Type stddev,
                  const Device& device, CURANDGenerator generator, CUDAStream stream) {
  static_assert(std::is_floating_point<Type>::value);

  DeviceGuard device_guard(device);
  generator.set_stream(stream);
  int64_t even_length = num_elements / 2 * 2;
  if constexpr (ToScalarType<Type>::value == ScalarType::Float) {
    HCTR_LIB_THROW(curandGenerateNormal(generator(), data, even_length, mean, stddev));
  } else {
    static_assert(std::is_same<Type, double>::value);
    HCTR_LIB_THROW(curandGenerateNormalDouble(generator(), data, even_length, mean, stddev));
  }
  if (auto odd_length = num_elements - even_length) {
    copy_async(data + even_length, data, odd_length, device, device, stream);
  }
}

template <typename Type>
void variance_scaling_async(Type* data, int64_t num_elements, Type scale,
                            const VarianceScalingMode mode,
                            const VarianceScalingDistribution distribution, const Type fan_in,
                            const Type fan_out, const Device& device, CURANDGenerator generator,
                            CUDAStream stream) {
  static_assert(std::is_floating_point<Type>::value);

  DeviceGuard device_guard(device);
  generator.set_stream(stream);
  const Type n = (mode == VarianceScalingMode::FanIn)
                     ? fan_in
                     : ((mode == VarianceScalingMode::FanOut) ? fan_out : (fan_in + fan_out) / 2.f);
  scale /= n;

  if (distribution == VarianceScalingDistribution::Uniform) {
    Type limit = std::sqrt(3.f * scale);
    uniform_async(data, num_elements, -limit, limit, device, generator, stream);
  } else {
    Type stddev = (distribution == VarianceScalingDistribution::TruncatedNormal)
                      ? std::sqrt(scale) / .87962566103423978
                      : std::sqrt(scale);
    normal_async(data, num_elements, TypeConverter<Type, float>::value(0.f), stddev, device,
                 generator, stream);
  }
}

template <typename Type>
void xavier_uniform_async(Type* data, int64_t num_elements, const Type fan_in, const Type fan_out,
                          const Device& device, CURANDGenerator generator, CUDAStream stream) {
  variance_scaling_async<Type>(data, num_elements, 1.f, VarianceScalingMode::FanAvg,
                               VarianceScalingDistribution::Uniform, fan_in, fan_out, device,
                               generator, stream);
}

template <typename Type>
void xavier_normal_async(Type* data, int64_t num_elements, const Type fan_in, const Type fan_out,
                         const Device& device, CURANDGenerator generator, CUDAStream stream) {
  variance_scaling_async<Type>(data, num_elements, 1.f, VarianceScalingMode::FanAvg,
                               VarianceScalingDistribution::TruncatedNormal, fan_in, fan_out,
                               device, generator, stream);
}

template <typename Type>
void he_uniform_async(Type* data, int64_t num_elements, const Type fan_in, const Device& device,
                      CURANDGenerator generator, CUDAStream stream) {
  variance_scaling_async<Type>(data, num_elements, 2.f, VarianceScalingMode::FanIn,
                               VarianceScalingDistribution::Uniform, fan_in, 0.f, device, generator,
                               stream);
}

template <typename Type>
void he_normal_async(Type* data, int64_t num_elements, const Type fan_in, const Device& device,
                     CURANDGenerator generator, CUDAStream stream) {
  variance_scaling_async<Type>(data, num_elements, 2.f, VarianceScalingMode::FanIn,
                               VarianceScalingDistribution::TruncatedNormal, fan_in, 0.f, device,
                               generator, stream);
}

#define ALL_RANDOM_SUPPORTED_TYPES(PH) \
  PH(float)                            \
  PH(double)

#define DEFINE_RANDOM_FUNCTION(Type)                                                               \
  template void uniform_async<Type>(Type * data, int64_t num_elements, const Type a, const Type b, \
                                    const Device& device, CURANDGenerator generator,               \
                                    CUDAStream stream);                                            \
  template void normal_async<Type>(Type * data, int64_t num_elements, const Type mean,             \
                                   const Type stddev, const Device& device,                        \
                                   CURANDGenerator generator, CUDAStream stream);                  \
  template void variance_scaling_async<Type>(                                                      \
      Type * data, int64_t num_elements, Type scale, const VarianceScalingMode mode,               \
      const VarianceScalingDistribution distribution, const Type fan_in, const Type fan_out,       \
      const Device& device, CURANDGenerator generator, CUDAStream stream);                         \
  template void xavier_uniform_async<Type>(Type * data, int64_t num_elements, const Type fan_in,   \
                                           const Type fan_out, const Device& device,               \
                                           CURANDGenerator generator, CUDAStream stream);          \
  template void xavier_normal_async<Type>(Type * data, int64_t num_elements, const Type fan_in,    \
                                          const Type fan_out, const Device& device,                \
                                          CURANDGenerator generator, CUDAStream stream);           \
  template void he_uniform_async<Type>(Type * data, int64_t num_elements, const Type fan_in,       \
                                       const Device& device, CURANDGenerator generator,            \
                                       CUDAStream stream);                                         \
  template void he_normal_async<Type>(Type * data, int64_t num_elements, const Type fan_in,        \
                                      const Device& device, CURANDGenerator generator,             \
                                      CUDAStream stream);

ALL_RANDOM_SUPPORTED_TYPES(DEFINE_RANDOM_FUNCTION)

}  // namespace core23
}  // namespace HugeCTR
