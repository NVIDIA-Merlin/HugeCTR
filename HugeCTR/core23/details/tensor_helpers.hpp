#pragma once

#include <cstdint>

namespace HugeCTR {

namespace core23 {

class TensorParams;
struct BufferRequirements;

BufferRequirements ConvertToBufferRequirements(const TensorParams& tensor_params);

template <typename Shape, int64_t Dims>
void compute_strides(const Shape& shape, int64_t (&strides)[Dims]) {
  int64_t stride = 1;
  for (int64_t dim = 0; dim < Dims; dim++) {
    strides[Dims - dim - 1] = stride;
    stride *= shape.size(Dims - dim - 1);
  }
}

template <int64_t Dims>
void compute_strides(const int64_t (&shape)[Dims], int64_t (&strides)[Dims]) {
  int64_t stride = 1;
  for (int64_t dim = 0; dim < Dims; dim++) {
    strides[Dims - dim - 1] = stride;
    stride *= shape[Dims - dim - 1];
  }
}

}  // namespace core23
}  // namespace HugeCTR