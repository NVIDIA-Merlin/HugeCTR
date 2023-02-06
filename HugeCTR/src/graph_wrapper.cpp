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

#include <atomic>
#include <common.hpp>
#include <graph_wrapper.hpp>
#include <map>
#include <mutex>

namespace HugeCTR {

void GraphWrapper::capture(std::function<void(cudaStream_t)> workload, cudaStream_t stream) {
  if (initialized) {
    return;
  }

  HCTR_LIB_THROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  workload(stream);
  HCTR_LIB_THROW(cudaStreamEndCapture(stream, &graph));
  HCTR_LIB_THROW(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
  initialized = true;
}

void GraphWrapper::exec(cudaStream_t stream) {
  if (!initialized) {
    HCTR_OWN_THROW(Error_t::IllegalCall, "Trying to execute graph which was not captured");
  }
  HCTR_LIB_THROW(cudaGraphLaunch(graph_exec, stream));
}

}  // namespace HugeCTR