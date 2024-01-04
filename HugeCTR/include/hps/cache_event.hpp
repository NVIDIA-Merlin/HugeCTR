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

#include <embedding_cache_combined.h>

#include <iostream>
#include <map>
#include <shared_mutex>
#include <vector>

namespace ecache {
class ECEvent : public IECEvent {
  using mutex_type = std::shared_timed_mutex;
  using read_only_lock = std::shared_lock<mutex_type>;
  using updatable_lock = std::unique_lock<mutex_type>;

 public:
  ECEvent() {}
  ECError RecordStream(cudaStream_t stream) {
    updatable_lock lock(mtx_for_m);
    if (lookup_stream_event_map_.find(stream) == lookup_stream_event_map_.end()) {
      cudaEvent_t nEvent;
      CACHE_CUDA_ERR_CHK_AND_THROW(cudaEventCreate(&nEvent));
      lookup_stream_event_map_[stream] = nEvent;
    }
    return ECERROR_SUCCESS;
  }

  ~ECEvent() {
    for (auto& se : lookup_stream_event_map_) {
      cudaEventDestroy(se.second);
    }
  }

  ECError EventRecord() override {
    for (auto& se : lookup_stream_event_map_) {
      CACHE_CUDA_ERR_CHK_AND_THROW(cudaEventRecord(se.second, se.first));
    }
    return ECERROR_SUCCESS;
  }

  ECError EventWaitStream(cudaStream_t stream) override {
    read_only_lock lock(mtx_for_m);
    for (auto& se : lookup_stream_event_map_) {
      CACHE_CUDA_ERR_CHK_AND_THROW(cudaStreamWaitEvent(stream, se.second));
    }
    return ECERROR_SUCCESS;
  }

 private:
  std::map<cudaStream_t, cudaEvent_t> lookup_stream_event_map_;
  mutex_type mtx_for_m;
};

}  // namespace ecache
