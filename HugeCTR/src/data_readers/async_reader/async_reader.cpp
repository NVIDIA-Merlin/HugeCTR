#include "data_readers/async_reader/async_reader.hpp"

#include <cuda_runtime.h>
#include <numa.h>
#include <nvml.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>

#include "common.hpp"
#include "resource_manager.hpp"
#include "utils.hpp"

namespace HugeCTR {

AsyncReaderImpl::AsyncReaderImpl(std::string fname, size_t batch_size_bytes,
                                 const ResourceManager* resource_manager, int num_threads,
                                 int num_batches_per_thread, size_t io_block_size, int io_depth,
                                 int io_alignment, bool shuffle, bool wait_for_gpu_idle)
    :

      fname_(fname),
      batch_size_bytes_(batch_size_bytes),
      resource_manager_(resource_manager),
      num_devices_(resource_manager_->get_local_gpu_count()),
      num_threads_(num_threads),
      num_batches_per_thread_(num_batches_per_thread),
      io_block_size_(io_block_size),
      io_depth_(io_depth),
      io_alignment_(io_alignment),
      wait_for_gpu_idle_(wait_for_gpu_idle),
      queue_id_(0),
      thread_batch_ids_(num_threads_),
      thread_buffer_ids_(num_threads),
      gpu_thread_ids_(num_devices_),
      local_readers_(num_threads_) {
  total_file_size_ = std::filesystem::file_size(fname);
  num_batches_ = (total_file_size_ + batch_size_bytes_ - 1) / batch_size_bytes;
  batch_ids_.resize(num_batches_);
  std::iota(batch_ids_.begin(), batch_ids_.end(), 0);

  if (shuffle) {
    std::mt19937 gen(resource_manager_->get_local_cpu()->get_replica_uniform_seed());
    std::shuffle(batch_ids_.begin(), batch_ids_.end(), gen);
  }

  // Don't allocate more buffers that number of batches in the file
  buffers_.resize(std::min((size_t)num_threads_ * num_batches_per_thread, num_batches_));
  for (auto& buf : buffers_) {
    buf = std::make_unique<InternalBatchBuffer>();
    buf->dev_data.resize(num_devices_);
    for (int id = 0; id < num_devices_; id++) {
      auto device_id = resource_manager_->get_local_gpu(id)->get_device_id();
      CudaDeviceContext ctx(device_id);
      HCTR_LIB_THROW(cudaMalloc(&buf->dev_data[id], batch_size_bytes_));
    }
  }

  streams_.resize(num_devices_);
  for (int id = 0; id < num_devices_; id++) {
    auto device_id = resource_manager_->get_local_gpu(id)->get_device_id();
    CudaDeviceContext ctx(device_id);
    HCTR_LIB_THROW(cudaStreamCreateWithPriority(&streams_[id], cudaStreamNonBlocking, -100));
  }
  HCTR_LIB_THROW(cudaEventCreateWithFlags(&event_success_, cudaEventDisableTiming));

  // For correct perf benchmarking create the thread readers upfront
  create_workers();
}

void AsyncReaderImpl::create_workers() {
  // Use round-robin distribution
  for (size_t i = 0; i < num_batches_; i++) {
    int thid = i % num_threads_;
    thread_batch_ids_[thid].push_back(batch_ids_[i]);
    // HCTR_LOG(INFO, WORLD, "thread %d got buffer %lu\n", thid, i);
  }

  gpu_thread_ids_.clear();
  for (int thid = 0; thid < num_threads_; thid++) {
    threads_.emplace_back(std::thread([thid, this]() {
      int raw_id = thid % num_devices_;
      int device_id = resource_manager_->get_local_gpu(raw_id)->get_device_id();
      CudaCPUDeviceContext ctx(device_id);
      gpu_thread_ids_[raw_id].push_back(thid);

      std::vector<InternalBatchBuffer*> thread_buffer_ptrs;
      for (int i = 0; i < num_batches_per_thread_; i++) {
        size_t buf_id = i * num_threads_ + thid;
        if (buf_id < buffers_.size()) {
          buffers_[buf_id]->raw_device_id = raw_id;
          thread_buffer_ptrs.push_back(buffers_[buf_id].get());
          thread_buffer_ids_[thid].push_back(buf_id);
        }
      }

      local_readers_[thid] = std::make_unique<ThreadAsyncReader>(
          fname_, resource_manager_, batch_size_bytes_, raw_id, streams_[raw_id],
          thread_batch_ids_[thid], thread_buffer_ptrs,
          ThreadAsyncReaderParameters{io_block_size_, io_alignment_, io_depth_, num_devices_,
                                      wait_for_gpu_idle_, loop_},
          total_file_size_);
    }));
  }

  for (auto& thread : threads_) {
    thread.join();
  }
  threads_.clear();
}

bool AsyncReaderImpl::is_currently_loading() { return !threads_.empty(); }

size_t AsyncReaderImpl::get_num_buffers() { return buffers_.size(); }

void AsyncReaderImpl::load_async() {
  if (is_currently_loading()) {
    throw std::runtime_error("load_async() is called before the previous load_async finished!");
  }

  for (int thid = 0; thid < num_threads_; thid++) {
    threads_.emplace_back(std::thread([thid, this]() {
      int raw_id = thid % num_devices_;
      int device_id = resource_manager_->get_local_gpu(raw_id)->get_device_id();
      CudaCPUDeviceContext ctx(device_id);

      local_readers_[thid]->load();
    }));
  }
}

BatchDesc AsyncReaderImpl::get_batch() {
  if (!is_currently_loading()) {
    throw std::runtime_error(
        "Requested a batch from a file that is not being loaded. Please call load_async() first!");
  }

  for (size_t attempt = 0; attempt < buffers_.size(); attempt++) {
    last_buffer_ = buffers_[queue_id_].get();

    auto status = last_buffer_->status.load();
    while (status != BufferStatus::Finished) {
      if (status == BufferStatus::ReadReady || status == BufferStatus::PermanentlyResident) {
        return {last_buffer_->size, last_buffer_->dev_data,
                status == BufferStatus::PermanentlyResident};
      }
      if (wait_for_gpu_idle_) {
        last_buffer_->ready_to_upload_event.store(&event_success_);
      }

      status = last_buffer_->status.load();
    }

    queue_id_ = (queue_id_ + 1) % buffers_.size();
  }

  return {0, std::vector<char*>(0)};
}

void AsyncReaderImpl::wait_for_gpu_events(const std::vector<cudaEvent_t*> events) {
  if (!wait_for_gpu_idle_) {
    return;
  }
  assert(events.size() == (size_t)num_devices_);

  for (int thid = 0; thid < num_threads_; thid++) {
    int raw_id = thid % num_devices_;
    wait_for_gpu_event(events[raw_id], raw_id);
  }
}

void AsyncReaderImpl::wait_for_gpu_event(cudaEvent_t* event, int raw_device_id) {
  if (!wait_for_gpu_idle_) {
    return;
  }

  for (auto thid : gpu_thread_ids_[raw_device_id]) {
    for (auto bufid : thread_buffer_ids_[thid]) {
      if (buffers_[bufid]->status == BufferStatus::UploadInProcess) {
        buffers_[bufid]->ready_to_upload_event.store(event);
        // HCTR_LOG(INFO, WORLD, "storing %p to thread %d gpu %d\n", (void*)event, (int)thid,
        // (int)raw_id);
      }
    }
  }
}

void AsyncReaderImpl::finalize_batch() {
  // Don't update status of finished or resident buffers
  BufferStatus expected = BufferStatus::ReadReady;
  last_buffer_->status.compare_exchange_strong(expected, BufferStatus::IOReady);
  if (loop_ && last_buffer_->id == (int64_t)num_batches_ - 1) {
    queue_id_ = 0;
  } else {
    queue_id_ = (queue_id_ + 1) % buffers_.size();
  }
}

void AsyncReaderImpl::finalize_batch(cudaEvent_t* event) {
  last_buffer_->safe_to_upload_event.store(event);
  finalize_batch();
}

int AsyncReaderImpl::get_last_batch_device() {
  if (last_buffer_) {
    return last_buffer_->raw_device_id;
  } else {
    return buffers_[queue_id_]->raw_device_id;
  }
}

void AsyncReaderImpl::reset() {
  for (auto& reader : local_readers_) {
    reader->reset();
  }
  for (auto& thread : threads_) {
    thread.join();
  }
  threads_.clear();
  queue_id_ = 0;
}

AsyncReaderImpl::~AsyncReaderImpl() {
  reset();
  cudaEventDestroy(event_success_);
}

}  // namespace HugeCTR
