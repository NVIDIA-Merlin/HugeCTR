#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <unistd.h>

#include <profiler.hpp>

namespace HugeCTR {
namespace Profiler {

void DataProfiler::initialize(bool use_cuda_graph, bool exit_when_finished) {
  MESSAGE_("Profiler using PROFILING_MODE: data");
}

bool DataProfiler::record_data(const char* data_label_char, cudaStream_t stream,
                               const std::string& data, int device_id) {
  return false;
}

}  //  namespace Profiler

Profiler::DataProfiler global_data_profiler;

}  //  namespace HugeCTR