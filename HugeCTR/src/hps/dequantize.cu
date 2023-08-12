#include <hps/dequantize.hpp>

namespace HugeCTR {
template <typename InT, typename OutT>
Dequantize<InT, OutT>::Dequantize() {}

template <typename InT, typename OutT>
struct dequantize_func {
  __device__ __forceinline__ OutT operator()(float scale, InT x) const {
    if constexpr (is_fp8<InT>::value)
      return static_cast<float>(x) * scale;
    else
      return __fdividef(static_cast<float>(x), scale);
  }
};

template <typename InT, typename OutT>
__global__ void dequantize_output_kernel(const InT* input, const float* scales, OutT* output,
                                         size_t emb_vec_size) {
  // y = c / (scale)
  // if bias: y += expand_dims(bias, 0)
  // y = epilogue(y)
  const auto rescale_func = dequantize_func<InT, OutT>();
  size_t i = blockIdx.x;
  for (size_t j = threadIdx.x; j < emb_vec_size; j += blockDim.x) {
    size_t index = i * emb_vec_size + j;
    const float scale = scales[i];
    OutT v = rescale_func(scale, input[index]);
    output[index] = v;
  }
}

template <typename InT, typename OutT>
void Dequantize<InT, OutT>::dequantize(InT* input, OutT* output, OutT* scale, size_t batch_size,
                                       size_t emb_vec_size) const {
  const dim3 grid(std::min(int(batch_size), std::numeric_limits<int32_t>::max()));
  const dim3 block(std::min(int(emb_vec_size), 1024));
  dequantize_output_kernel<<<grid, block, 0>>>(input, scale, output, emb_vec_size);
  cudaDeviceSynchronize();
}

template class Dequantize<int16_t, float>;
template class Dequantize<int8_t, float>;
template class Dequantize<__nv_fp8_e4m3, float>;

}  // namespace HugeCTR