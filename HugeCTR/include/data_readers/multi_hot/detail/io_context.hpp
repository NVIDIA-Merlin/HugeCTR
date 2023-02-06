#pragma once

#include <chrono>
#include <cstdint>
#include <vector>

namespace HugeCTR {

enum class IOError {
  IO_SUCCESS,
  IO_EAGAIN,  // context pipe full
  IO_EBADF,   // invalid file descriptor
  IO_EFAULT,  // invalid request arguments
  IO_EINVAL,  // invalid io context
  IO_EINTR,   // interrupted by signal handler
  IO_UNKNOWN  // unknown error
};

struct IORequest {
  int fd;
  uint8_t* data;
  size_t size;
  size_t offset;
  void* user_data;
};

// class IOReadRequest : public IORequest {
// public:
//  IOReadRequest(int fd, uint8_t* data, size_t length, size_t offset, void* user_data);
//};

struct IOEvent {
  IOError error;
  void* user_data;
};

class IOContext {
 public:
  virtual ~IOContext() = default;
  virtual void submit(const IORequest& request) = 0;
  virtual const std::vector<IOEvent>& collect(size_t min_reqs, size_t timeout_us) = 0;
  virtual size_t get_alignment() const = 0;
};

}  // namespace HugeCTR