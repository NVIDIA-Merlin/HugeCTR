#include "core/registry.hpp"
#include "gpu_kernels.hpp"
#include "include/utils.cuh"

namespace HugeCTR {

template <typename offset_t>
__global__ void compute_bucket_ranges_with_padding(offset_t* bucket_ranges,
                                                   const int* __restrict hotness_bucket_ranges,
                                                   int current_batch_size, int batch_size) {
  const int lookup = blockIdx.y;

  // e.g:
  // hotnesses:             [3, 5, 1, 2]
  // hotness_bucket_ranges: [0, 3, 8, 9, 11]
  // current_batch_size:    10
  // lookup_start_ranges:   [0,30,90,90,110]
  const int lookup_hotness_bucket_start = hotness_bucket_ranges[lookup];
  const int lookup_hotness_bucket_end = hotness_bucket_ranges[lookup + 1];
  const int lookup_hotness = lookup_hotness_bucket_end - lookup_hotness_bucket_start;
  const offset_t lookup_start_range = lookup_hotness_bucket_start * current_batch_size;
  const offset_t lookup_end_range = lookup_hotness_bucket_end * current_batch_size;

  // If we are on the last lookup, extend batch_size by 1 to account for end bucket
  int end_bucket = (lookup == gridDim.y - 1);

  CUDA_1D_KERNEL_LOOP(bucket_idx, batch_size + end_bucket) {
    const bool is_valid_bucket = bucket_idx < current_batch_size + end_bucket;
    bucket_ranges[lookup * batch_size + bucket_idx] =
        is_valid_bucket ? lookup_start_range + bucket_idx * lookup_hotness : lookup_end_range;
  }
}

void compute_fixed_bucket_ranges(core::Tensor hotness_bucket_range, int current_batch_size,
                                 int batch_size, core::Tensor bucket_range, cudaStream_t stream) {
  const size_t num_lookup = hotness_bucket_range.get_num_elements() - 1;

  DISPATCH_INTEGRAL_FUNCTION(bucket_range.dtype().type(), offset_t, [&] {
    compute_bucket_ranges_with_padding<<<dim3(144 * 8, num_lookup), 256, 0, stream>>>(
        bucket_range.get<offset_t>(), hotness_bucket_range.get<int>(), current_batch_size,
        batch_size);
  });
}

}  // namespace HugeCTR