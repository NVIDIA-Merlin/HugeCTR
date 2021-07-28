#pragma once

#include <cuda_runtime.h>

#include <atomic>
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
      cudaFree(ptr);
    }
    cudaFreeHost(raw_host_ptr);
  }
};

struct BatchDesc {
  size_t size_bytes;
  std::vector<char*> dev_data;
};

}  // namespace HugeCTR