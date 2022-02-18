#include "common.hpp"
#include "data_readers/async_reader/broadcast.hpp"
#include "utils.hpp"

namespace HugeCTR {

constexpr int copy_width = 4;

namespace {

inline __device__ float4 read4(const float* src, int n) {
  if (n == copy_width) {
    return *((float4*)src);
  } else {
    float4 res;
    if (n > 0) res.x = src[0];
    if (n > 1) res.y = src[1];
    if (n > 2) res.z = src[2];
    return res;
  }
}

inline __device__ void write4(float* dst, int n, float4 val) {
  if (n == copy_width) {
    *((float4*)dst) = val;
  } else {
    if (n > 0) dst[0] = val.x;
    if (n > 1) dst[1] = val.y;
    if (n > 2) dst[2] = val.z;
  }
}

__global__ void broadcast_kernel(float** addrs, const bool* p2p_accessible, int batch_size_floats,
                                 int num_dests, int src_id) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idx4 = idx * copy_width;
  int num_elems = min(batch_size_floats - idx4, copy_width);

  float4 src_val = read4(addrs[src_id] + idx4, num_elems);
  for (int i = 1; i < num_dests; i++) {
    int dst_id = (src_id + i) % num_dests;
    if (p2p_accessible[dst_id]) {
      write4(addrs[dst_id] + idx4, num_elems, src_val);
    }
  }
}

};  // namespace

void broadcast(float** dev_pointers, const bool* dev_p2p_accessible, int batch_size_floats,
               int num_dests, int src_id, cudaStream_t stream) {
  int block_size = 128;
  int grid_size = (batch_size_floats + copy_width * block_size - 1) / block_size;

  constexpr bool use_kernel = false;

  for (int i = 1; i < num_dests; i++) {
    int dst_id = (src_id + i) % num_dests;
    if (!dev_p2p_accessible[dst_id] || (!use_kernel)) {
      HCTR_LIB_THROW(cudaMemcpyAsync(dev_pointers[dst_id], dev_pointers[src_id],
                                     batch_size_floats * sizeof(float), cudaMemcpyDeviceToDevice,
                                     stream));
    }
  }

  if (use_kernel) {
    broadcast_kernel<<<grid_size, block_size, 0, stream>>>(dev_pointers, dev_p2p_accessible,
                                                           batch_size_floats, num_dests, src_id);
  }
}

}  // namespace HugeCTR
