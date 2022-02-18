#include "HugeCTR/include/graph_wrapper.hpp"

#include <atomic>
#include <map>
#include <mutex>

#include "HugeCTR/include/common.hpp"

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