#include <base/debug/logger.hpp>
#include <core23/buffer_requirements.hpp>
#include <core23/details/tensor_helpers.hpp>
#include <core23/tensor_params.hpp>

namespace HugeCTR {

namespace core23 {

namespace {

int64_t GetValidAlignment(int64_t alignment, const DataType& data_type) {
  const size_t size = data_type.size();
  if (alignment == 0 || alignment < size) {
    HCTR_LOG_S(WARNING, ROOT) << "alignment(" << alignment << ") is too small. size(" << size
                              << ") is used instead." << std::endl;
    alignment = size;
  } else {
    auto rem = alignment % size;
    if (rem != 0) {
      HCTR_LOG_S(WARNING, ROOT) << "alignment(" << alignment << ") is invalid. ";
      alignment += size;
      alignment -= rem;
      HCTR_LOG_S(WARNING, ROOT) << "It is adjusted to " << alignment << std::endl;
    }
    return alignment;
  }
  return alignment;
}

}  // namespace

BufferRequirements ConvertToBufferRequirements(const TensorParams& tensor_params) {
  BufferRequirements requirements = {
      .num_bytes = tensor_params.shape().size() * tensor_params.data_type().size(),
      .alignment = GetValidAlignment(tensor_params.alignment(), tensor_params.data_type()),
      .stream = tensor_params.stream()};
  return requirements;
}

}  // namespace core23
}  // namespace HugeCTR