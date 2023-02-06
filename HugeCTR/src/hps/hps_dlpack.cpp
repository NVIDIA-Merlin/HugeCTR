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

#include <hps/hps_dlpack.hpp>
#include <string>

namespace HugeCTR {

static DeviceType getATenDevice(const DLDevice& ctx) {
  switch (ctx.device_type) {
    case DLDeviceType::kDLCPU:
      return DeviceType::CPU;
    // if we are compiled under HIP, we cannot do cuda
    case DLDeviceType::kDLCUDA:
      return DeviceType::CUDA;
    case DLDeviceType::kDLCUDAHost:
      return DeviceType::CUDAHost;
    default:
      HCTR_THROW_IF(
          false, HugeCTR::Error_t::DataCheckError,
          "from_dlpack received an invalid device type: " + std::to_string(ctx.device_type));
  }
  return DeviceType::CPU;
}

DataType toScalarType(const DLDataType& dtype) {
  switch (dtype.code) {
    case DLDataTypeCode::kDLInt:
      switch (dtype.bits) {
        case 32:
          return DataType::Int;
        case 64:
          return DataType::Int64;
        default:
          HCTR_THROW_IF(true, HugeCTR::Error_t::DataCheckError,
                        "Unsupported kInt bits " + std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat:
      switch (dtype.bits) {
        case 32:
          return DataType::Float;
        default:
          HCTR_THROW_IF(true, HugeCTR::Error_t::DataCheckError,
                        "Unsupported kFloat bits " + std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLBfloat:
      switch (dtype.bits) {
        case 16:
          return DataType::Bfloat;
        default:
          HCTR_THROW_IF(true, HugeCTR::Error_t::DataCheckError,
                        "Unsupported kDLBfloat bits " + std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLComplex:
      switch (dtype.bits) {
        case 32:
          return DataType::Complex;
        case 64:
          return DataType::Complex;
        case 128:
          return DataType::Complex;
        default:
          HCTR_THROW_IF(true, HugeCTR::Error_t::DataCheckError,
                        "Unsupported kFloat bits " + std::to_string(dtype.bits));
      }
      break;
    default:
      HCTR_THROW_IF(true, HugeCTR::Error_t::DataCheckError,
                    "Unsupported code " + std::to_string(dtype.code));
  }
  HCTR_THROW_IF(true, HugeCTR::Error_t::DataCheckError,
                "Unsupported code " + std::to_string(dtype.code));
  return DataType::Complex;
}

HPSTensor fromDLPack(const DLManagedTensor* src) {
  HPSTensor* hpstensor(new HPSTensor);
  DeviceType device = getATenDevice(src->dl_tensor.device);
  DataType stype = toScalarType(src->dl_tensor.dtype);
  hpstensor->data = src->dl_tensor.data;
  hpstensor->device = device;
  hpstensor->type = stype;
  hpstensor->device_id = src->dl_tensor.device.device_id;
  hpstensor->strides = src->dl_tensor.strides;
  hpstensor->shape = src->dl_tensor.shape;
  hpstensor->ndim = src->dl_tensor.ndim;

  return *hpstensor;
}

}  // namespace HugeCTR