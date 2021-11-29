#include "HugeCTR/include/graph_wrapper.hpp"
#include "HugeCTR/include/common.hpp"
#include <map>
#include <mutex>
#include <atomic>

namespace HugeCTR {

void GraphWrapper::capture(std::function<void(cudaStream_t)> workload, cudaStream_t stream) {
  if (initialized) {
    return;
  }

  CK_CUDA_THROW_(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  workload(stream);
  CK_CUDA_THROW_(cudaStreamEndCapture(stream, &graph));
  CK_CUDA_THROW_(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
  initialized = true;
}

void GraphWrapper::exec(cudaStream_t stream) {
  if (!initialized) {
    CK_THROW_(Error_t::IllegalCall, "Trying to execute graph which was not captured");
  }
  CK_CUDA_THROW_(cudaGraphLaunch(graph_exec, stream));
}

}