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

#include <cuda_runtime.h>

#include <functional>

namespace HugeCTR {

struct GraphWrapper {
  bool initialized = false;
  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;

  void capture(std::function<void(cudaStream_t)> workload, cudaStream_t stream);
  void exec(cudaStream_t stream);
};

}  // namespace HugeCTR