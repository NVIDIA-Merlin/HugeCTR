#include "data_readers/multi_hot/detail/system_latch.hpp"

namespace HugeCTR {

template <typename T>
__global__ void kernel_count_down(T* latch, T n) {
  atomicDec_system(latch, n);
}

void SystemLatch::device_count_down(cudaStream_t stream, value_type n, bool from_graph) {
  if (from_graph) {
    kernel_count_down<<<1, 1, 0, stream>>>(latch_, n);
  } else {
    cudaStreamAddCallback(stream, callback, (void*)latch_, 0);
  }
}

}  // namespace HugeCTR