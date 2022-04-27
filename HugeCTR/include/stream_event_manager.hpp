/*
* Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "utils.hpp"

namespace HugeCTR {

class StreamEventManager {
public:
  StreamEventManager() {}

  ~StreamEventManager() {
    for (auto& s : stream_map_) {
      cudaStreamDestroy(s.second);
    }
    for (auto& e : event_map_) {
      cudaEventDestroy(e.second);
    }
  }

  cudaStream_t& get_stream(const std::string& key, unsigned int flags = 0, int priority = 0) {
    if (stream_map_.find(key) == stream_map_.end()) {
      cudaStream_t stream;
      HCTR_LIB_THROW(cudaStreamCreateWithPriority(&stream, flags, priority));
      stream_map_[key] = stream;
    }
    return stream_map_[key];
  }

  cudaEvent_t& get_event(const std::string& key, unsigned int flags = 0) {
    if (event_map_.find(key) == event_map_.end()) {
      cudaEvent_t event;
      HCTR_LIB_THROW(cudaEventCreateWithFlags(&event, flags));
      event_map_[key] = event;
    }
    return event_map_[key];
  }

private:
  std::unordered_map<std::string, cudaStream_t> stream_map_;
  std::unordered_map<std::string, cudaEvent_t> event_map_;
};

} // namespace HugeCTR