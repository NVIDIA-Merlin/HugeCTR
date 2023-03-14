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
#pragma once

#include <cuda_runtime.h>

// For the tensor bags
#include <atomic>
#include <core23/tensor.hpp>
#include <tensor2.hpp>
#include <vector>

struct iocb;

namespace HugeCTR {

enum class BufferStatus : int {
  IOReady = 0,
  IOInProcess = 1,
  UploadInProcess = 2,
  UploadSubmitted = 3,
  ReadReady = 4,
  PermanentlyResident = 5,
  Finished = 6
};

struct InternalBatchBuffer {
  int64_t id = -1;
  size_t size;
  int raw_device_id;

  std::vector<char*> dev_data;
  char* raw_host_ptr = nullptr;
  char* host_data;

  std::atomic<BufferStatus> status;
  std::vector<iocb*> io_reqs;
  int num_outstanding_reqs;
  std::atomic<cudaEvent_t*> ready_to_upload_event, safe_to_upload_event;
  int num_submitted_h2d_chunks;
  int num_submitted_broadcasts;
  bool preload_done;
  cudaEvent_t event;

  // Following the rule of 5 just in case
  // Only need the destructor here
  InternalBatchBuffer() { status.store(BufferStatus::IOReady); };
  InternalBatchBuffer(InternalBatchBuffer const& other) = delete;
  InternalBatchBuffer& operator=(InternalBatchBuffer const& other) = delete;

  InternalBatchBuffer(InternalBatchBuffer&& other) = default;
  InternalBatchBuffer& operator=(InternalBatchBuffer&& other) = default;

  ~InternalBatchBuffer() {
    for (auto ptr : dev_data) {
      HCTR_LIB_THROW(cudaFree(ptr));
    }
    HCTR_LIB_THROW(cudaHostUnregister(raw_host_ptr));
    free(raw_host_ptr);
  }
};

struct BatchDesc {
  size_t size_bytes;
  std::vector<char*> dev_data;
  bool cached;
  size_t id;
};

class RawPtrWrapper : public TensorBuffer2 {
 public:
  RawPtrWrapper(void* ptr) : ptr_(ptr) {}
  bool allocated() const override { return true; }
  void* get_ptr() override { return ptr_; }

 private:
  void* ptr_;
};

class RawPtrBuffer : public TensorBuffer2 {
 public:
  RawPtrBuffer(size_t size_bytes) { HCTR_LIB_THROW(cudaMalloc(&ptr_, size_bytes)); }
  bool allocated() const override { return true; }
  void* get_ptr() override { return ptr_; }
  ~RawPtrBuffer() override { cudaFree(ptr_); }

 private:
  void* ptr_;
};

}  // namespace HugeCTR