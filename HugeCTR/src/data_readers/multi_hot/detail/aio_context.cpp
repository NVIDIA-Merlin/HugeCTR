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

#include <alloca.h>

#include <cassert>
#include <data_readers/multi_hot/detail/aio_context.hpp>
#include <stdexcept>

namespace HugeCTR {

#define round_up(x, y) ((((x) + ((y)-1)) / (y)) * (y))

AIOContext::AIOContext(size_t io_depth) : io_depth_(io_depth), iocb_buffer_(io_depth) {
  tmp_events_.reserve(io_depth);

  for (size_t i = 0; i < io_depth; ++i) {
    free_cbs_.push(iocb_buffer_.data() + i);
  }

  ctx_ = 0;
  if (io_queue_init(io_depth, &ctx_) < 0) {
    throw std::runtime_error("io_queue_init failed");
  }
}

AIOContext::~AIOContext() {
  // app can't exit with AIO requests inflight
  (void)collect(num_inflight_, 1e6);  // wait 1s
  assert(num_inflight_ == 0);
}

void AIOContext::submit(const IORequest& request) {
  assert(!free_cbs_.empty());

  iocb* cb = free_cbs_.front();
  free_cbs_.pop();

  // For O_DIRECT, offsets and sizes need to be aligned
  size_t aligned_offset = (request.offset / get_alignment()) * get_alignment();
  size_t size = round_up(request.size + (request.offset - aligned_offset), get_alignment());

  io_prep_pread(cb, request.fd, request.data, size, aligned_offset);
  cb->data = request.user_data;

  iocb* cblist[] = {cb};
  int ret = io_submit(ctx_, 1, cblist);
  if (ret < 0) {
    throw std::runtime_error("io_submit failed");
  }
  num_inflight_++;
}

const std::vector<IOEvent>& AIOContext::collect(size_t min_reqs, size_t timeout_us) {
  timespec timeout = {0, (long)timeout_us * 1000};

  io_event* io_events = (io_event*)alloca(sizeof(io_event) * io_depth_);
  int num_completed = io_getevents(ctx_, min_reqs, io_depth_, io_events, &timeout);
  if (num_completed < 0) {
    throw std::runtime_error("io_getevents failed");
  }

  num_inflight_ -= num_completed;

  tmp_events_.clear();
  for (int i = 0; i < num_completed; ++i) {
    iocb* cb = io_events[i].obj;
    assert(cb);
    int ret = io_events[i].res;
    int ret2 = io_events[i].res2;

    if (ret < 0 || ret2 != 0) {
      throw std::runtime_error("io_getevents returned failed event: " +
                               std::string(strerror(-ret)));
    }

    IOEvent event;
    event.error = ret < 0 ? errno_to_enum(-ret) : IOError::IO_SUCCESS;
    event.user_data = (void*)cb->data;

    tmp_events_.emplace_back(event);

    free_cbs_.push(cb);
  }

  return tmp_events_;
}

IOError AIOContext::errno_to_enum(int err) {
  switch (err) {
    case 0:
      return IOError::IO_SUCCESS;
    case EAGAIN:
      return IOError::IO_EAGAIN;
    case EBADF:
      return IOError::IO_EBADF;
    case EFAULT:
      return IOError::IO_EFAULT;
    case EINVAL:
      return IOError::IO_EINVAL;
    case EINTR:
      return IOError::IO_EINTR;
    default:
      return IOError::IO_UNKNOWN;
  }
}

size_t AIOContext::get_alignment() const {
  return 4096;  // O_DIRECT requirement
}

}  // namespace HugeCTR