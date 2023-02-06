#pragma once

#include "utils.hpp"

namespace HugeCTR {

class DeviceTransfer {
 public:
  DeviceTransfer(size_t upload_gpu, uint8_t* upload_src, uint8_t* upload_dst, size_t size_bytes)
      : upload_gpu_(upload_gpu),
        upload_src_(const_cast<uint8_t*>(upload_src)),
        upload_dst_(const_cast<uint8_t*>(upload_dst)),
        size_bytes_(size_bytes) {}

  size_t get_device_id() const { return upload_gpu_; }

  void execute(const cudaStream_t& stream) {
    HCTR_LIB_THROW(
        cudaMemcpyAsync(upload_dst_, upload_src_, size_bytes_, cudaMemcpyHostToDevice, stream));
  }

 private:
  size_t upload_gpu_;
  uint8_t* upload_src_;
  uint8_t* upload_dst_;
  size_t size_bytes_;
};

}  // namespace HugeCTR