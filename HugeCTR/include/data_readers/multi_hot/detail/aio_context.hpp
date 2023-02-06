#pragma once

#include <libaio.h>

#include <queue>

#include "io_context.hpp"

namespace HugeCTR {

class AIOContext : public IOContext {
 public:
  AIOContext(size_t io_depth);
  ~AIOContext();

  void submit(const IORequest& request);
  const std::vector<IOEvent>& collect(size_t min_reqs, size_t timeout_us);
  size_t get_alignment() const;

 private:
  static IOError errno_to_enum(int err);

  size_t io_depth_ = 0;
  size_t num_inflight_ = 0;
  io_context_t ctx_ = 0;
  std::vector<IOEvent> tmp_events_;  // prevent dynamic memory allocation
  std::vector<iocb> iocb_buffer_;
  std::queue<iocb*> free_cbs_;
};

}  // namespace HugeCTR