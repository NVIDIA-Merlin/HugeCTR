#pragma once

#include <cuda_runtime.h>
#include <libaio.h>

#include <atomic>
#include <string>
#include <vector>

namespace HugeCTR {

class InternalBatchBuffer;
class ResourceManager;
enum class WorkerStatus : int { OK, Terminate };

struct ThreadAsyncReaderParameters {
  size_t io_block_size;
  int io_alignment, io_depth;
  int num_h2d_chunks;
  bool wait_for_gpu_idle;
  bool loop;
};

class ThreadAsyncReader {
 public:
  ThreadAsyncReader(std::string fname, const ResourceManager* resource_manager,
                    size_t batch_size_bytes, int device_id, cudaStream_t stream,
                    std::vector<size_t> batch_ids, std::vector<InternalBatchBuffer*> dest_buffers,
                    ThreadAsyncReaderParameters params, size_t total_file_size);

  void load();
  void reset();

  ~ThreadAsyncReader();

 private:
  int fd_;
  size_t batch_size_bytes_;
  int device_id_;
  cudaStream_t stream_;
  int num_dest_buffers_;
  int max_num_blocks_per_batch_;
  size_t total_file_size_;
  io_context_t ioctx_;
  std::atomic<WorkerStatus> status_;

  std::vector<size_t> batch_ids_;
  std::vector<InternalBatchBuffer*> dest_buffers_;
  ThreadAsyncReaderParameters params_;
  int num_buffers_waiting_io_;

  void try_submit_io(size_t batch_id, int io_id);
  void wait_io();
  bool wait_for_gpu_idle(InternalBatchBuffer* buffer);
  void try_submit_upload(InternalBatchBuffer* buffer);
  void try_submit_p2p   (InternalBatchBuffer* buffer);
  bool check_completion (InternalBatchBuffer* buffer);
};

}  // namespace HugeCTR
