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

#include <libaio.h>

#include <data_readers/multi_hot/detail/io_context.hpp>
#include <queue>

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