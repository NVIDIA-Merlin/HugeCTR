
#include "data_readers/async_reader/thread_async_reader.hpp"

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include <cassert>
#include <numeric>
#include <stdexcept>

#include "common.hpp"
#include "data_readers/async_reader/async_reader_common.hpp"
#include "data_readers/async_reader/broadcast.hpp"
#include "resource_manager.hpp"

namespace HugeCTR {

ThreadAsyncReader::ThreadAsyncReader(std::string fname, const ResourceManager* resource_mananager,
                                     size_t batch_size_bytes, int device_id, cudaStream_t stream,
                                     std::vector<size_t> batch_ids,
                                     std::vector<InternalBatchBuffer*> dest_buffers,
                                     ThreadAsyncReaderParameters params, 
                                     size_t total_file_size)
    : batch_size_bytes_(batch_size_bytes),
      device_id_(device_id),
      stream_(stream),
      total_file_size_(total_file_size),
      batch_ids_(batch_ids),
      dest_buffers_(dest_buffers), 
      params_(params),
      num_buffers_waiting_io_(0) {
#if (__cplusplus >= 201703L)
  static_assert(std::atomic<BufferStatus>::is_always_lock_free &&
                    std::atomic<WorkerStatus>::is_always_lock_free,
                "Compiler cannot use atomic enum class, need to change to int type");
#endif
  assert(params_.io_block_size % params_.io_alignment == 0);

  num_dest_buffers_ = dest_buffers_.size();

  fd_ = open(fname.c_str(), O_RDONLY | O_DIRECT);
  if (fd_ == -1) {
    throw std::runtime_error("No such file: " + fname);
  };

  max_num_blocks_per_batch_ = batch_size_bytes_ / params_.io_block_size + 2;
  for (auto buf : dest_buffers_) {
    assert((size_t)buf->raw_host_ptr % params_.io_alignment == 0);
    CK_CUDA_THROW_(cudaMallocHost(&buf->raw_host_ptr, max_num_blocks_per_batch_ * params_.io_block_size));
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&buf->event, cudaEventDisableTiming));

    buf->io_reqs.resize(max_num_blocks_per_batch_);
    for (auto& req : buf->io_reqs) {
      req = new iocb;
    }
  }

  for (auto buf : dest_buffers_) {
    buf->status.store(BufferStatus::IOReady);
  }
}

void ThreadAsyncReader::load() {
  size_t num_batches = batch_ids_.size();
  size_t processed = 0;
  std::vector<size_t> id_per_host_buffer(num_dest_buffers_);
  std::iota(id_per_host_buffer.begin(), id_per_host_buffer.end(), 0);

  status_.store(WorkerStatus::OK);
  for (auto buf : dest_buffers_) {
    buf->safe_to_upload_event.store(nullptr);
    buf->ready_to_upload_event.store(nullptr);
    buf->preload_done = false;
  }

  ioctx_ = 0;
  if (io_queue_init(params_.io_depth, &ioctx_) < 0) {
    throw std::runtime_error("io_setup failed");
  }

  while (status_.load() != WorkerStatus::Terminate) {
    // bool all_resident = true;
    // for (auto buf : dest_buffers_) {
    //   if (buf->status != BufferStatus::PermanentlyResident) {
    //     all_resident = false;
    //     break;
    //   }
    // }
    // if (all_resident){
    //   return;
    // }

    for (int i = 0; i < num_dest_buffers_; i++) {
      if (id_per_host_buffer[i] < num_batches) {
        try_submit_io(batch_ids_[id_per_host_buffer[i]], i);
      }
    }
    wait_io();
    for (int i = 0; i < num_dest_buffers_; i++) {
      if (id_per_host_buffer[i] < num_batches) {
        try_submit_p2p(dest_buffers_[i]);
      }
    }
    for (int i = 0; i < num_dest_buffers_; i++) {
      if (id_per_host_buffer[i] < num_batches) {
        try_submit_upload(dest_buffers_[i]);
      }
    }
    for (int i = 0; i < num_dest_buffers_; i++) {
      if (id_per_host_buffer[i] < num_batches) {
        if (check_completion(dest_buffers_[i])) {
          processed++;
          id_per_host_buffer[i] += num_dest_buffers_;
          if (params_.loop && id_per_host_buffer[i] >= num_batches) {
            id_per_host_buffer[i] = i;
          }
        }
      }
    }
    usleep(10);
    if (!params_.loop && processed >= num_batches) {
      break;
    }
  }

  if (io_destroy(ioctx_) < 0) {
    throw std::runtime_error("io_destroy failed");
  }

  CK_CUDA_THROW_(cudaStreamSynchronize(stream_));

  if (status_.load() != WorkerStatus::Terminate) {
    for (int i = 0; i < num_dest_buffers_; i++) {
      BufferStatus expected = BufferStatus::IOReady;
      while (!dest_buffers_[i]->status.compare_exchange_weak(expected, BufferStatus::Finished)) {
        expected = BufferStatus::IOReady;
      }
    }
  }
}

void ThreadAsyncReader::try_submit_io(size_t batch_id, int io_id) {
  auto& buffer = dest_buffers_[io_id];
  if (buffer->status.load() != BufferStatus::IOReady) {
    return;
  }

  // Maybe we have already loaded this batch before?!
  if (buffer->id == (int64_t)batch_id) {
    buffer->status.store(BufferStatus::PermanentlyResident);
    return;
  }

  buffer->status.store(BufferStatus::IOInProcess);

  size_t req_beg_offset = batch_id * batch_size_bytes_;
  size_t req_end_offset = std::min((batch_id + 1) * batch_size_bytes_, total_file_size_);
  size_t raw_beg_offset = (req_beg_offset / params_.io_block_size) * params_.io_block_size;
  size_t raw_end_offset = ((req_end_offset + params_.io_block_size - 1) / params_.io_block_size) * params_.io_block_size;
  size_t num_blocks = (raw_end_offset - raw_beg_offset) / params_.io_block_size;
  assert(num_blocks <= (size_t)max_num_blocks_per_batch_);

  buffer->id = batch_id;
  buffer->num_outstanding_reqs = num_blocks;
  buffer->num_submitted_h2d_chunks = 0;
  buffer->num_submitted_broadcasts = 0;
  buffer->size = req_end_offset - req_beg_offset;
  buffer->host_data = buffer->raw_host_ptr + (req_beg_offset - raw_beg_offset);
  assert(buffer->size % sizeof(float) == 0);

  for (size_t block = 0; block < num_blocks; block++) {
    auto req = buffer->io_reqs[block];

    int tmp = params_.io_block_size;
    //if (params_.wait_for_gpu_idle && buffer->id > 20000) tmp = 512;
    io_prep_pread(req, fd_, buffer->raw_host_ptr + params_.io_block_size * block, tmp, //params_.io_block_size,
                  raw_beg_offset + params_.io_block_size * block);
    req->data = (void*)buffer;
  }

  int ret = io_submit(ioctx_, num_blocks, buffer->io_reqs.data());
  num_buffers_waiting_io_ += 1;
  if (ret < 0) {
    throw std::runtime_error("io_submit failed");
  }
}

void ThreadAsyncReader::wait_io() {
  // if (num_buffers_waiting_io_ <= 0) {
  //   return;
  // }
  timespec timeout = {0, 10'000l};
  io_event events[max_num_blocks_per_batch_];
  int num_completed =
      io_getevents(ioctx_, max_num_blocks_per_batch_, max_num_blocks_per_batch_, events, &timeout);
  if (num_completed < 0) {
    throw std::runtime_error("io_getevents failed");
  }

  for (int b = 0; b < num_completed; b++) {
    auto req = events[b].obj;
    auto buffer = (InternalBatchBuffer*)req->data;
    buffer->num_outstanding_reqs--;
    assert(buffer->num_outstanding_reqs >= 0);
    if (buffer->num_outstanding_reqs == 0) {
      num_buffers_waiting_io_ -= 1;
      buffer->status.store(BufferStatus::UploadInProcess);
      if (params_.wait_for_gpu_idle) {
        buffer->ready_to_upload_event.store(nullptr);
      }
    }
  }
}

bool ThreadAsyncReader::wait_for_gpu_idle(InternalBatchBuffer* buffer) {
  if (params_.wait_for_gpu_idle && buffer->preload_done) {
    auto event_ptr = buffer->ready_to_upload_event.load();
    if (event_ptr == nullptr) {
      return false;
    }
    else {
      buffer->ready_to_upload_event.store(nullptr);
      CK_CUDA_THROW_(cudaStreamWaitEvent(stream_, *event_ptr));
    }
  }
  return true;
}

void ThreadAsyncReader::try_submit_upload(InternalBatchBuffer* buffer) {
  if (buffer->status.load() != BufferStatus::UploadInProcess || 
      buffer->num_submitted_h2d_chunks >= params_.num_h2d_chunks) {
    return;
  }
  if (!wait_for_gpu_idle(buffer)) {
    return;
  }

  // H2D upload
  // Wait until the buffers are consumed (one event after a barrier)
  if (buffer->num_submitted_h2d_chunks == 0 && buffer->safe_to_upload_event != nullptr) {
    CK_CUDA_THROW_(cudaStreamWaitEvent(stream_, *buffer->safe_to_upload_event));
  }

  size_t chunk_size = (buffer->size + params_.num_h2d_chunks - 1) / params_.num_h2d_chunks;
  size_t beg_offset = chunk_size * buffer->num_submitted_h2d_chunks;
  size_t end_offset = std::min(buffer->size, chunk_size * (buffer->num_submitted_h2d_chunks+1));

  //if (buffer->id < 10000)
  CK_CUDA_THROW_(cudaMemcpyAsync(buffer->dev_data[device_id_] + beg_offset,
                                 buffer->host_data + beg_offset,
                                 end_offset - beg_offset,
                                 cudaMemcpyHostToDevice, stream_));
  buffer->num_submitted_h2d_chunks++;
}

void ThreadAsyncReader::try_submit_p2p(InternalBatchBuffer* buffer) {
  if (buffer->status.load() != BufferStatus::UploadInProcess ||
      buffer->num_submitted_h2d_chunks < params_.num_h2d_chunks) {
    return;
  }
  if (!wait_for_gpu_idle(buffer)) {
    return;
  }

  // Broadcast to the other GPUs
  if (buffer->num_submitted_broadcasts != (int)buffer->dev_data.size()) {
    if (device_id_ != buffer->num_submitted_broadcasts) {
      //if (buffer->id < 5000 || (10000 < buffer->id && buffer->id < 15000))
      CK_CUDA_THROW_(cudaMemcpyAsync(buffer->dev_data[buffer->num_submitted_broadcasts],
                                     buffer->dev_data[device_id_],
                                     buffer->size,
                                     cudaMemcpyDeviceToDevice,
                                     stream_));
    }
    buffer->num_submitted_broadcasts++;
    return;
  }

  // Here we've submitted everything
  // There is no real need to make eventRecord atomic (wrt stream) with the rest,
  //  we only care that eventRecord is AFTER the H2D and the broadcast
  buffer->preload_done = true;
  buffer->num_submitted_h2d_chunks = 0;
  buffer->num_submitted_broadcasts = 0;
  CK_CUDA_THROW_(cudaEventRecord(buffer->event, stream_));
  buffer->status.store(BufferStatus::UploadSubmitted);
}

bool ThreadAsyncReader::check_completion(InternalBatchBuffer* buffer) {
  if (buffer->status.load() != BufferStatus::UploadSubmitted) {
    return false;
  }

  auto res = cudaEventQuery(buffer->event);
  if (res == cudaSuccess) {
    buffer->status.store(BufferStatus::ReadReady);
    return true;
  }
  if (res == cudaErrorNotReady) {
    return false;
  }
  CK_CUDA_THROW_(res);
  return false;
}

void ThreadAsyncReader::reset() {
  status_.store(WorkerStatus::Terminate);
  for (auto buf : dest_buffers_) {
    buf->status.store(BufferStatus::IOReady);
  }
}

ThreadAsyncReader::~ThreadAsyncReader() = default;

}  // namespace HugeCTR
