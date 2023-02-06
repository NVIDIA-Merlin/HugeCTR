#include "data_readers/multi_hot/detail/batch_file_reader.hpp"

#include <fcntl.h>
#include <numa.h>
#include <unistd.h>

#include "common.hpp"
#include "data_readers/multi_hot/detail/aio_context.hpp"

namespace HugeCTR {

#define round_up(x, y) ((((x) + ((y)-1)) / (y)) * (y))

BatchFileReader::BatchFileReader(const std::string& fname, size_t slot, size_t max_batches_inflight,
                                 std::unique_ptr<IBatchLocations> batch_locations)
    : slot_id_(slot)
      // having multiple IOs inflight to the same location will break data reader
      ,
      max_batches_inflight_(std::min(max_batches_inflight, batch_locations->count())),
      free_batches_(max_batches_inflight_),
      batch_locations_(std::move(batch_locations)),
      batch_locations_iterator_(batch_locations_->begin()),
      io_ctx_(new AIOContext(max_batches_inflight_)),
      buf_size_(round_up(
          batch_locations_->get_batch_size_bytes(),
          io_ctx_->get_alignment() * 2))  // double alignment because it can fall across 2 sectors
{
  tmp_completed_batches_.reserve(max_batches_inflight_);
  empty_batches_.reserve(max_batches_inflight_);

  for (size_t i = 0; i < max_batches_inflight_; ++i) {
    uint8_t* data = (uint8_t*)numa_alloc_local(
        buf_size_);  // aligned_alloc(io_ctx_->get_alignment(), buf_size);
    HCTR_LIB_THROW(cudaHostRegister(data, buf_size_, 0));

    batches_.emplace_back(this, data, batch_locations_->get_batch_size_bytes(), slot);
  }

  for (size_t i = 0; i < max_batches_inflight_; ++i) {
    free_batches_.push(batches_.data() + i);
  }

  fd_ = open(fname.c_str(), O_RDONLY | O_DIRECT);
  if (fd_ == -1) {
    throw std::runtime_error("No such file: " + fname);
  };
}

BatchFileReader::~BatchFileReader() {
  // Call destructor on IO context to wait for inflight IOs to complete first before we
  // free our buffers
  io_ctx_.reset();

  for (auto& batch : batches_) {
    cudaHostUnregister(batch.aligned_data);
    numa_free(batch.aligned_data, buf_size_);
    // free(batch.aligned_data);
  }
  close(fd_);
}

// SLOT:  0 1 2 3 0 1 2 3
// BFR A: 0 1 2 3 4 0 1 2 3
// BFR B: 0 1 2 3 0 1 2 0 1 2
// BFR B':0 1 2   0 1 2 // FIXME: edge case when
const std::vector<const BatchFileReader::Batch*>& BatchFileReader::read_batches(size_t timeout_us) {
  submit_reads();
  return collect(timeout_us);
}

void BatchFileReader::submit_reads() {
  Batch* batch = nullptr;
  while (true) {
    if (batch_locations_iterator_ == batch_locations_->end()) {
      if (num_inflight_ == 0) {
        batch_locations_iterator_ = batch_locations_->begin();
      } else {
        // Wait for batches from previous epoch to complete. Edge case where batch_i=0 from previous
        // epoch is inflight and return batch_i=0 from next epoch. Then we have two batches from
        // batch_i=0 inflight
        break;
      }
    }

    if (free_batches_.try_pop(batch)) {
      BatchFileLocation location = *batch_locations_iterator_;
      batch_locations_iterator_++;

      num_inflight_++;

      batch->size_bytes = location.size_bytes;
      batch->batch_id = location.id;
      batch->batch_i = location.i;
      batch->start_time = 0.f;
      batch->end_time = 0.f;

      // Why an empty batch? This is an edge case where we have batch_size/num_gpus and when the
      // batch is sharded, some GPUs may not have local batches to read.
      const bool empty_batch = location.size_bytes == 0;
      if (empty_batch) {
        batch->data = nullptr;  // no data to return
        empty_batches_.emplace_back(const_cast<const Batch*>(batch));
      } else {
        // Our data will start further into the buffer if the offset is not aligned
        size_t misalignment = location.offset % io_ctx_->get_alignment();
        batch->data = batch->aligned_data + misalignment;
        batch->start_time = time_double();

        IORequest io_req{fd_, batch->aligned_data, location.size_bytes, location.offset,
                         (void*)batch};
        io_ctx_->submit(io_req);
      }
    } else {
      break;  // queue depth full
    }
  }
}

const std::vector<const BatchFileReader::Batch*>& BatchFileReader::collect(size_t timeout_us) {
  std::vector<IOEvent> events = io_ctx_->collect(1, timeout_us);
  auto time = time_double();
  tmp_completed_batches_.clear();
  for (const auto& event : events) {
    auto batch = reinterpret_cast<Batch*>(event.user_data);
    batch->end_time = time;
    tmp_completed_batches_.emplace_back(const_cast<const Batch*>(batch));
  }
  for (const auto batch : empty_batches_) {
    tmp_completed_batches_.emplace_back(batch);
  }
  empty_batches_.clear();
  return tmp_completed_batches_;
}

void BatchFileReader::release_batch(const BatchFileReader::Batch* batch) {
  num_inflight_--;
  free_batches_.push(const_cast<BatchFileReader::Batch*>(batch));
}

size_t BatchFileReader::get_queue_depth() const { return max_batches_inflight_; }

}  // namespace HugeCTR