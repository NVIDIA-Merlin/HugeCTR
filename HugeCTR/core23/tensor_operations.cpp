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

#include <core23/data_type_helpers.cuh>
#include <core23/low_level_primitives.hpp>
#include <core23/tensor_operations.hpp>
#include <cstdint>

namespace HugeCTR {

namespace core23 {

#define DEFINE_FILL_ELSE_IF(Type, _)                                                         \
  else if (data.data_type() == ToScalarType<Type>::value) {                                  \
    fill_async<Type>(data.data<Type>(), data.num_elements(),                                 \
                     TypeConverter<Type, decltype(val)>::value(val), data.device(), stream); \
  }

#define DEFINE_CONVERT_ELSE_IF(DstType, SrcType, _)                                               \
  else if (dst.data_type() == ToScalarType<DstType>::value &&                                     \
           src.data_type() == ToScalarType<SrcType>::value) {                                     \
    convert_async<DstType, SrcType>(dst.data<DstType>(), src.data<SrcType>(), src.num_elements(), \
                                    dst.device(), src.device(), stream);                          \
  }

void zeros_sync(Tensor& data) {
  CUDAStream default_stream;
  zeros_async(data, default_stream);
  cudaStreamSynchronize(default_stream());
}
void zeros_async(Tensor& data, CUDAStream stream) {
  float val = 0.f;
  if (false) {
  }
  ALL_DATA_TYPES_SUPPORTED(DEFINE_FILL_ELSE_IF)
}

void copy_sync(Tensor& dst, const Tensor& src) {
  HCTR_THROW_IF(dst.num_bytes() != src.num_bytes(), HugeCTR::Error_t::IllegalCall,
                "Destination and Source Tensors have inconsitent sizes.");
  HCTR_THROW_IF(dst.data_type() != src.data_type(), HugeCTR::Error_t::IllegalCall,
                "Destination and Source Tensors have inconsitent data types.");
  copy_sync(dst.data(), src.data(), src.num_bytes(), dst.device(), src.device());
}
void copy_async(Tensor& dst, const Tensor& src, CUDAStream stream) {
  HCTR_THROW_IF(dst.num_bytes() != src.num_bytes(), HugeCTR::Error_t::IllegalCall,
                "Destination and Source Tensors have inconsitent sizes.");
  HCTR_THROW_IF(dst.data_type() != src.data_type(), HugeCTR::Error_t::IllegalCall,
                "Destination and Source Tensors have inconsitent data types.");
  copy_async(dst.data(), src.data(), src.num_bytes(), dst.device(), src.device(), stream);
}

template <typename T>
void copy_sync(std::vector<T>& dst, const Tensor& src) {
  HCTR_THROW_IF(dst.size() * sizeof(T) != src.num_bytes(), HugeCTR::Error_t::IllegalCall,
                "Destination and Source Tensors have inconsitent sizes.");
  HCTR_THROW_IF(ToScalarType<T>::value != src.data_type().type(), HugeCTR::Error_t::IllegalCall,
                "Destination and Source Tensors have inconsitent data types.");
  copy_sync(dst.data(), src.data(), src.num_bytes(), DeviceType::CPU, src.device());
}

template <typename T>
void copy_async(std::vector<T>& dst, const Tensor& src, CUDAStream stream) {
  HCTR_THROW_IF(dst.size() * sizeof(T) != src.num_bytes(), HugeCTR::Error_t::IllegalCall,
                "Destination and Source Tensors have inconsitent sizes.");
  HCTR_THROW_IF(ToScalarType<T>::value != src.data_type().type(), HugeCTR::Error_t::IllegalCall,
                "Destination and Source Tensors have inconsitent data types.");
  copy_async(dst.data(), src.data(), src.num_bytes(), DeviceType::CPU, src.device(), stream);
}

template <typename T>
void copy_sync(Tensor& dst, const std::vector<T>& src) {
  HCTR_THROW_IF(src.size() * sizeof(T) != dst.num_bytes(), HugeCTR::Error_t::IllegalCall,
                "Destination and Source Tensors have inconsitent sizes.");
  HCTR_THROW_IF(ToScalarType<T>::value != dst.data_type().type(), HugeCTR::Error_t::IllegalCall,
                "Destination and Source Tensors have inconsitent data types.");
  copy_sync(dst.data(), src.data(), src.size() * sizeof(T), dst.device(), DeviceType::CPU);
}

template <typename T>
void copy_async(Tensor& dst, const std::vector<T>& src, CUDAStream stream) {
  HCTR_THROW_IF(src.size() * sizeof(T) != dst.num_bytes(), HugeCTR::Error_t::IllegalCall,
                "Destination and Source Tensors have inconsitent sizes.");
  HCTR_THROW_IF(ToScalarType<T>::value != dst.data_type().type(), HugeCTR::Error_t::IllegalCall,
                "Destination and Source Tensors have inconsitent data types.");
  copy_async(dst.data(), src.data(), src.size() * sizeof(T), dst.device(), DeviceType::CPU, stream);
}

void convert_async(Tensor& dst, const Tensor& src, CUDAStream stream) {
  if (dst.data_type() == src.data_type()) {
    copy_async(dst, src, stream);
  }
  ALL_DATA_CONVERSIONS_SUPPORTED(DEFINE_CONVERT_ELSE_IF)
  else {
    HCTR_THROW_IF(false, HugeCTR::Error_t::IllegalCall,
                  "Casting from " + src.data_type().name() + " to " + dst.data_type().name() +
                      " is not implemented");
  }
}

void uniform_async(Tensor& data, const float a, const float b, CURANDGenerator generator,
                   CUDAStream stream) {
  if (data.data_type() == ToScalarType<float>::value) {
    uniform_async<float>(data.data<float>(), data.num_elements(), a, b, data.device(), generator,
                         stream);
  } else if (data.data_type() == ToScalarType<double>::value) {
    uniform_async<double>(data.data<double>(), data.num_elements(), a, b, data.device(), generator,
                          stream);
  } else {
    HCTR_THROW_IF(false, HugeCTR::Error_t::IllegalCall,
                  data.data_type().name() + " is not supported.");
  }
}
void normal_async(Tensor& data, const float mean, const float stddev, CURANDGenerator generator,
                  CUDAStream stream) {
  if (data.data_type() == ToScalarType<float>::value) {
    normal_async<float>(data.data<float>(), data.num_elements(), mean, stddev, data.device(),
                        generator, stream);
  } else if (data.data_type() == ToScalarType<double>::value) {
    normal_async<double>(data.data<double>(), data.num_elements(), mean, stddev, data.device(),
                         generator, stream);
  } else {
    HCTR_THROW_IF(false, HugeCTR::Error_t::IllegalCall,
                  data.data_type().name() + " is not supported.");
  }
}

#define DEFINE_TEMPLATE_FUNC_IN_TENSOR_OPERATRION(type)                                          \
  template void copy_sync<type>(std::vector<type> & dst, const Tensor& src);                     \
  template void copy_sync<type>(Tensor & dst, const std::vector<type>& src);                     \
  template void copy_async<type>(std::vector<type> & dst, const Tensor& src, CUDAStream stream); \
  template void copy_async<type>(Tensor & dst, const std::vector<type>& src, CUDAStream stream);

DEFINE_TEMPLATE_FUNC_IN_TENSOR_OPERATRION(__half)
DEFINE_TEMPLATE_FUNC_IN_TENSOR_OPERATRION(float)
DEFINE_TEMPLATE_FUNC_IN_TENSOR_OPERATRION(double)
DEFINE_TEMPLATE_FUNC_IN_TENSOR_OPERATRION(char)
DEFINE_TEMPLATE_FUNC_IN_TENSOR_OPERATRION(int8_t)
DEFINE_TEMPLATE_FUNC_IN_TENSOR_OPERATRION(uint8_t)
DEFINE_TEMPLATE_FUNC_IN_TENSOR_OPERATRION(int32_t)
DEFINE_TEMPLATE_FUNC_IN_TENSOR_OPERATRION(uint32_t)
DEFINE_TEMPLATE_FUNC_IN_TENSOR_OPERATRION(int64_t)
DEFINE_TEMPLATE_FUNC_IN_TENSOR_OPERATRION(uint64_t)
}  // namespace core23

}  // namespace HugeCTR
