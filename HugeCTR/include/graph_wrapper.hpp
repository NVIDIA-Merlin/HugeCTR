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