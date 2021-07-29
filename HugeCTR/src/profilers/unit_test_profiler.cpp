#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <unistd.h>

#include <profiler.hpp>

namespace HugeCTR {
namespace Profiler {

void UnitTestProfiler::initialize(bool use_cuda_graph, bool exit_when_finished) {
  MESSAGE_("Profiler using PROFILING_MODE: unit_test");
}

void UnitTestProfiler::record_event(const char* event_label_char, cudaStream_t stream,
                                   bool could_be_in_cuda_graph, int device_id,
                                   const std::string& extra_info) {}

bool UnitTestProfiler::iter_check() { return true; }

}  //  namespace Profiler

Profiler::UnitTestProfiler global_unit_test_profiler;

}  //  namespace HugeCTR