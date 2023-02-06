#pragma once

#include "core/tensor.hpp"

namespace HugeCTR {

void compute_fixed_bucket_ranges(core::Tensor hotness_bucket_range, int current_batch_size,
                                 int batch_size, core::Tensor bucket_range, cudaStream_t stream);

}  // namespace HugeCTR