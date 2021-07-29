#pragma once

#include <data_readers/async_reader/async_reader_common.hpp>
#include <data_readers/async_reader/thread_async_reader.hpp>
#include <string>
#include <thread>
#include <vector>

namespace HugeCTR {

class ResourceManager;

class AsyncReaderImpl {
 public:
  AsyncReaderImpl(std::string fname, size_t batch_size_bytes,
                  const ResourceManager* resource_manager, int num_threads,
                  int num_batches_per_thread, size_t io_block_size, int io_depth, int io_alignment,
                  bool shuffle = false, bool wait_for_gpu_idle = false);

  bool is_currently_loading();
  void load_async();
  void reset();
  BatchDesc get_batch();
  void finalize_batch();
  void finalize_batch(cudaEvent_t* event);
  int get_last_batch_device();
  void wait_for_gpu_events(const std::vector<cudaEvent_t*> events);
  void wait_for_gpu_event(cudaEvent_t* event, int raw_device_id);
  ~AsyncReaderImpl();

 private:
  std::string fname_;
  size_t batch_size_bytes_;
  size_t num_batches_;
  const ResourceManager* resource_manager_;
  int num_devices_, num_threads_, num_batches_per_thread_;
  size_t io_block_size_;
  int io_depth_, io_alignment_;
  InternalBatchBuffer* last_buffer_ = nullptr;
  size_t total_file_size_;
  bool wait_for_gpu_idle_;
  int queue_id_;
  bool loop_ = true;
  cudaEvent_t event_success_;

  std::vector<size_t> batch_ids_;
  std::vector<std::unique_ptr<InternalBatchBuffer>> buffers_;
  std::vector<std::thread> threads_;
  std::vector<cudaStream_t> streams_;
  std::vector<std::vector<size_t>> thread_batch_ids_;
  std::vector<std::vector<size_t>> thread_buffer_ids_, gpu_thread_ids_;
  std::vector<std::unique_ptr<ThreadAsyncReader>> local_readers_;

  void create_workers();
};

}  // namespace HugeCTR
