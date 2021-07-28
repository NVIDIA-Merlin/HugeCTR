/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <linalg/gemv.h>
#include <cublas_v2.h>
#include <math.h>

#include <layers/multi_cross_layer.hpp>
#include <linalg/binary_op.cuh>
// #include <linalg/gemm.cuh>
#include <linalg/matrix_vector_op.cuh>
#include <linalg/reduce.cuh>
#include <prims/linalg/matrix_multiplication.cuh>
#include <prims/cuda_utils.cuh>
#include <utils.cuh>
#include <utils.hpp>
#include <vector>

/** Overload of built-in atomicAdd for support on Pascal architectures */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 && __CUDA_ARCH__ < 700

__inline__ __device__ __half atomicAdd(__half* address, __half val) {
  size_t base_offset = ((size_t)address & 2);
  uint32_t* base_address = (uint32_t*)((char*)(address) - base_offset);

  uint32_t old = *base_address, assumed;
  do {
    assumed = old;
    {
      __half assumed_f16 = __ushort_as_half((uint16_t)(assumed >> (base_offset << 3)));
      uint32_t new_val = assumed;
      ((uint16_t*)(&new_val))[base_offset >> 1] = __half_as_ushort(__hadd(assumed_f16, val));
      old = atomicCAS(base_address, assumed, new_val);
    }
  } while (assumed != old);
  return __ushort_as_half((uint16_t)(old >> (base_offset << 3)));
}

#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 && __CUDA_ARCH__ < 700

namespace HugeCTR {

// kernels
namespace {

inline int calc_grid(int t, int b) { return (t - 1) / b + 1; }

template <typename T>
void matrix_vec_mul(Tensor2<T>& out, const Tensor2<T>& mat, const Tensor2<T>& vec,
                    cublasHandle_t cublas_handle, cudaStream_t stream);

template <>
void matrix_vec_mul(Tensor2<float>& out, const Tensor2<float>& mat, const Tensor2<float>& vec,
                    cublasHandle_t cublas_handle, cudaStream_t stream) {
  float* pout = out.get_ptr();
  const float* pmat = mat.get_ptr();
  const float* pvec = vec.get_ptr();

  const auto& dim = out.get_dimensions();
  const auto& idim = mat.get_dimensions();
  assert(dim.size() == 2 && idim.size() == 2 && idim[1] == vec.get_dimensions()[1] &&
         vec.get_dimensions()[0] == 1);
  assert(idim[0] == dim[0]);

  const int h = idim[0];
  const int w = idim[1];
  const float alpha = 1.0f;
  const float beta = 0.0f;

  CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, h, 1, w, &alpha, pmat, w,
                              pvec, w, &beta, pout, h));
}

template <>
void matrix_vec_mul(Tensor2<__half>& out, const Tensor2<__half>& mat, const Tensor2<__half>& vec,
                    cublasHandle_t cublas_handle, cudaStream_t stream) {
  __half* pout = out.get_ptr();
  const __half* pmat = mat.get_ptr();
  const __half* pvec = vec.get_ptr();

  const auto& dim = out.get_dimensions();
  const auto& idim = mat.get_dimensions();
  assert(dim.size() == 2 && idim.size() == 2 && idim[1] == vec.get_dimensions()[1] &&
         vec.get_dimensions()[0] == 1);
  assert(idim[0] == dim[0]);

  const int h = idim[0];
  const int w = idim[1];
  const __half alpha = 1.0f;
  const __half beta = 0.0f;

  CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  CUBLAS_CHECK(cublasHgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, h, 1, w, &alpha, pmat, w,
                              pvec, w, &beta, pout, h));
}

template <typename T>
void row_scaling(Tensor2<T>& o_mat, const Tensor2<T>& mat, const Tensor2<T>& vec,
                 cudaStream_t stream) {
  T* pout = o_mat.get_ptr();
  const T* pmat = mat.get_ptr();
  const T* pvec = vec.get_ptr();

  const auto& dim = o_mat.get_dimensions();
  const auto& idim = mat.get_dimensions();
  assert(dim.size() == 2 && idim.size() == 2 && dim[0] == vec.get_dimensions()[0] &&
         vec.get_dimensions()[1] == 1);
  assert(idim[0] == dim[0] && idim[1] == dim[1]);

  const int h = dim[0];
  const int w = dim[1];

  MLCommon::LinAlg::matrixVectorOp(pout, pmat, pvec, h, w, false, true,
                                   [] __device__(T a, T b) { return a * b; }, stream);
}

template <typename T>
void matrix_vec_add(Tensor2<T>& o_mat, const Tensor2<T>& mat, const Tensor2<T>& vec,
                    cudaStream_t stream) {
  T* pout = o_mat.get_ptr();
  const T* pmat = mat.get_ptr();
  const T* pvec = vec.get_ptr();

  const auto& dim = o_mat.get_dimensions();
  const auto& idim = mat.get_dimensions();
  assert(dim.size() == 2 && idim.size() == 2 && dim[1] == vec.get_dimensions()[1] &&
         vec.get_dimensions()[0] == 1);
  assert(idim[0] == dim[0] && idim[1] == dim[1]);

  const int h = dim[0];
  const int w = dim[1];

  MLCommon::LinAlg::matrixVectorOp(pout, pmat, pvec, h, w, false, false,
                                   [] __device__(T a, T b) { return a + b; }, stream);
}

template <typename T>
void matrix_add(Tensor2<T>& out_mat, const Tensor2<T>& mat_a, const Tensor2<T>& mat_b,
                cudaStream_t stream) {
  T* pout = out_mat.get_ptr();
  const T* pmat_a = mat_a.get_ptr();
  const T* pmat_b = mat_b.get_ptr();

  const auto& dim = out_mat.get_dimensions();
  const auto& idim1 = mat_a.get_dimensions();
  const auto& idim2 = mat_b.get_dimensions();
  assert(idim1[0] == dim[0] && idim1[1] == dim[1]);
  assert(idim2[0] == dim[0] && idim2[1] == dim[1]);

  const int h = dim[0];
  const int w = dim[1];

  MLCommon::LinAlg::binaryOp(pout, pmat_a, pmat_b, h * w,
                             [] __device__(T a, T b) { return a + b; }, stream);
}

/**
 * compute dot product for each pair of the rows in the two matrix,
 */
template <typename T>
__global__ void matrix_pair_mul_kernel(T* o_vec, const T* mat_a, int h, int w,
                                       const T* mat_b) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int wtid = tid % WARP_SIZE;  // thread id in warp
  const int wid = tid / WARP_SIZE;   // warp id
  const T* mat_a_with_offset = mat_a + wid * w;
  const T* mat_b_with_offset = mat_b + wid * w;
  if (wid < h) {
    T accum = 0.f;
    for (int i = wtid; i < w; i += WARP_SIZE) {
      accum += mat_a_with_offset[i] * mat_b_with_offset[i];
    }
    T val = warpReduceSum(accum);
    if (wtid == 0) {
      o_vec[wid] = val;
    }
  }
}

template <typename T>
void matrix_pair_mul(Tensor2<T>& o_vec, const Tensor2<T>& mat_a,
                     const Tensor2<T>& mat_b, cudaStream_t stream) {
  T* pout = o_vec.get_ptr();
  const T* pmat_a = mat_a.get_ptr();
  const T* pmat_b = mat_b.get_ptr();

  const auto& dim = mat_a.get_dimensions();

  const int h = dim[0];
  const int w = dim[1];
  assert(h == mat_b.get_dimensions()[0] && w == mat_a.get_dimensions()[1] &&
         h == o_vec.get_dimensions()[0] && 1 == o_vec.get_dimensions()[1]);

  const int BLOCK_DIM = 256;
  const int GRID_DIM = calc_grid(h * WARP_SIZE, BLOCK_DIM);
  matrix_pair_mul_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat_a, h, w, pmat_b);
}

template <typename T>
__global__ void mm_1d(T* out_mat, const T* vec_a, int h, const T* vec_b, int w) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < h * w) {
    const int col = tid % w;
    const int row = tid / w;
    out_mat[tid] = vec_a[row] * vec_b[col];
  }
}

template <typename T>
void out_product(Tensor2<T>& out_mat, const Tensor2<T>& vec_a, const Tensor2<T>& vec_b,
                 cudaStream_t stream) {
  T* pout = out_mat.get_ptr();
  const T* pvec_a = vec_a.get_ptr();
  const T* pvec_b = vec_b.get_ptr();
  const auto& dim = out_mat.get_dimensions();

  const int h = dim[0];
  const int w = dim[1];

  assert(h == vec_a.get_dimensions()[0] && w == vec_b.get_dimensions()[1] &&
         vec_a.get_dimensions()[1] == 1 && vec_b.get_dimensions()[0] == 1);

  const int BLOCK_DIM = 256;
  const int GRID_DIM = calc_grid(h * w, BLOCK_DIM);
  mm_1d<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pvec_a, h, pvec_b, w);
}

/**
 * Each row in `mat` scale with the coresponding element in vec. and accum across rows
 * The length of vec should be h.
 * @param o_mat: hxw
 * @param mat: hxw
 * @param vec: hx1
 */
template <typename T>
__global__ void row_scaling_sum_kernel(T* out, const T* mat, int h, int w,
                                       const T* vec) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int wtid = tid % WARP_SIZE;  // thread id in warp
  const int wid = tid / WARP_SIZE;   // warp id
  if (wid < w) {
    T accum = 0.f;
    for (int i = wtid; i < h; i += WARP_SIZE) {
      const int col = wid;
      const int idx = i * w + col;
      accum += mat[idx] * vec[i];
    }
    T val = warpReduceSum(accum);
    if (wtid == 0) {
      out[wid] += val;  // using += here to enable regularization
    }
  }
}

template <typename T>
void row_scaling_sum(Tensor2<T>& out, const Tensor2<T>& mat, const Tensor2<T>& vec,
                     cudaStream_t stream) {
  T* pout = out.get_ptr();
  const T* pmat = mat.get_ptr();
  const T* pvec = vec.get_ptr();

  const auto& dim = out.get_dimensions();
  const auto& idim = mat.get_dimensions();
  assert(dim.size() == 2 && idim.size() == 2 && idim[0] == vec.get_dimensions()[0] &&
         vec.get_dimensions()[1] == 1);
  assert(idim[1] == dim[1]);

  const int h = idim[0];
  const int w = idim[1];

  const int BLOCK_DIM = 256;
  const int GRID_DIM = calc_grid(w * WARP_SIZE, BLOCK_DIM);  // each col one warp

  row_scaling_sum_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat, h, w, pvec);
}

template <typename T>
void rows_sum(Tensor2<T>& out, const Tensor2<T>& mat, cudaStream_t stream) {
  T* pout = out.get_ptr();
  const T* pmat = mat.get_ptr();

  const auto& dim = out.get_dimensions();
  const auto& idim = mat.get_dimensions();
  assert(dim.size() == 2 && idim.size() == 2);
  assert(idim[1] == dim[1]);

  const int h = idim[0];
  const int w = idim[1];

  MLCommon::LinAlg::reduce(pout, pmat, h, w, (T)0, false, true, stream, false,
                           [] __device__(T in, int i) { return in; });
}

}  // namespace

/*
 * Equivalent TensorFlow Code:
 *
def forward(x, k, b, layers):
  y = []
  h = []
  for i in range(layers):
    v = tf.linalg.matvec(x if i == 0 else y[i - 1], k[i])
    v = tf.transpose(v)
    h.append(v)
    m = tf.multiply(x, v)
    m = tf.add(m, x if i == 0 else y[i - 1])
    m = tf.add(m, b[i])
    y.append(m)
  return y, h
 *
 */
template <typename T>
void MultiCrossForwardFunctor<T>::operator()(cudaStream_t stream, cublasHandle_t cublas_handle,
                                          const Tensor2<T>& input_tensor,
                                          const Tensors2<T>& kernel_tensors,
                                          const Tensors2<T>& bias_tensors,
                                          Tensors2<T>& layer_output_tensors,
                                          Tensors2<T>& layer_hidden_tensors,
                                          int num_layers) const {
  for (int i = 0; i < num_layers; i++) {
    matrix_vec_mul(layer_hidden_tensors[i], i == 0 ? input_tensor : layer_output_tensors[i - 1],
                   kernel_tensors[i], cublas_handle, stream);
    row_scaling(layer_output_tensors[i], input_tensor, layer_hidden_tensors[i], stream);
    matrix_add(layer_output_tensors[i], layer_output_tensors[i],
               i == 0 ? input_tensor : layer_output_tensors[i - 1], stream);
    matrix_vec_add(layer_output_tensors[i], layer_output_tensors[i], bias_tensors[i], stream);
  }
}

/*
 * Equivalent TensorFlow Code:
 *
def backward(x, k, y, h, dy, layers):
  dx = tf.zeros(x.shape)
  dk = []
  db = []
  for i in reversed(range(layers)):
    dx = tf.add(dx, tf.multiply(dy, h[i]))
    dv = tf.expand_dims(tf.reduce_sum(tf.multiply(dy, x), 1), 1)
    dk.insert(0, tf.linalg.matvec(x if i == 0 else y[i - 1], tf.transpose(dv), transpose_a=True))
    db.insert(0, tf.expand_dims(tf.reduce_sum(dy, 0), 0))
    dy = tf.add(dy, tf.matmul(dv, k[i]))
  dx = tf.add(dx, dy)
  return dx, dk, db
 *
 */
template <typename T>
void MultiCrossBackwardFunctor<T>::operator()(
    cudaStream_t stream, const Tensor2<T>& input_tensor, const Tensors2<T>& kernel_tensors,
    const Tensors2<T>& layer_output_tensors, const Tensors2<T>& layer_hidden_tensors,
    const Tensor2<T>& grad_tensor, Tensor2<T>& output_tensor,
    Tensors2<T>& kernel_output_tensors, Tensors2<T>& bias_output_tensors,
    Tensor2<T>& tmp_vec_tensor, Tensor2<T> tmp_mat_tensors[], int num_layers) const {
  cudaMemsetAsync(tmp_mat_tensors[2].get_ptr(), 0, tmp_mat_tensors[2].get_size_in_bytes(), stream);
  for (int i = num_layers - 1; i >= 0; i--) {
    row_scaling(tmp_mat_tensors[0], i == num_layers - 1 ? grad_tensor : tmp_mat_tensors[1],
                layer_hidden_tensors[i], stream);
    matrix_add(tmp_mat_tensors[2], tmp_mat_tensors[2], tmp_mat_tensors[0], stream);
    matrix_pair_mul(tmp_vec_tensor, i == num_layers - 1 ? grad_tensor : tmp_mat_tensors[1],
                    input_tensor, stream);
    row_scaling_sum(kernel_output_tensors[i], i == 0 ? input_tensor : layer_output_tensors[i - 1],
                    tmp_vec_tensor, stream);
    rows_sum(bias_output_tensors[i], i == num_layers - 1 ? grad_tensor : tmp_mat_tensors[1],
             stream);
    out_product(tmp_mat_tensors[0], tmp_vec_tensor, kernel_tensors[i], stream);
    matrix_add(tmp_mat_tensors[1], i == num_layers - 1 ? grad_tensor : tmp_mat_tensors[1],
               tmp_mat_tensors[0], stream);
  }
  matrix_add(output_tensor, tmp_mat_tensors[2], tmp_mat_tensors[1], stream);
}

template <typename T>
MultiCrossLayer<T>::MultiCrossLayer(const std::shared_ptr<BufferBlock2<float>>& master_weight_buff,
                                 const std::shared_ptr<BufferBlock2<T>>& weight_buff,
                                 const std::shared_ptr<BufferBlock2<T>>& wgrad_buff,
                                 const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                                 const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor,
                                 const std::shared_ptr<GPUResource>& gpu_resource, int num_layers,
                                 std::vector<Initializer_t> initializer_types)
    : Layer(gpu_resource, initializer_types), num_layers_(num_layers) {
  try {
    // check the in_tensor and out_tensor
    const auto& in_tensor_dim = in_tensor.get_dimensions();
    const auto& out_tensor_dim = out_tensor.get_dimensions();
    // 1. two dim?
    if (in_tensor_dim.size() != 2 || out_tensor_dim.size() != 2) {
      CK_THROW_(Error_t::WrongInput, "input or output tensor doesn't has two dimensions");
    }
    // 2. same dim?
    for (int i = 0; i < 2; i++) {
      if (in_tensor_dim[i] != out_tensor_dim[i]) {
        CK_THROW_(Error_t::WrongInput, "input and output tensor doesn't match");
      }
    }
    size_t vec_length = in_tensor_dim[1];
    size_t batchsize = in_tensor_dim[0];

    // check num_lyaers
    if (num_layers < 1) {
      CK_THROW_(Error_t::WrongInput, "num_layers < 1");
    }

    std::vector<size_t> weight_bias_dim = {1, vec_length};
    reserve_master_weight_tensor(master_weight_buff, weight_bias_dim);
    for (int i = 0; i < num_layers; i++) {
      // setup weights
      {
        Tensor2<T> tensor;
        weight_buff->reserve(weight_bias_dim, &tensor);
        weights_.push_back(tensor);
      }
      // setup bias
      {
        Tensor2<T> tensor;
        weight_buff->reserve(weight_bias_dim, &tensor);
        weights_.push_back(tensor);
      }
      // setup weight gradient
      {
        Tensor2<T> tensor;
        wgrad_buff->reserve(weight_bias_dim, &tensor);
        wgrad_.push_back(tensor);
      }
      // setup bias gradient
      {
        Tensor2<T> tensor;
        wgrad_buff->reserve(weight_bias_dim, &tensor);
        wgrad_.push_back(tensor);
      }
    }

    in_tensors_.push_back(in_tensor);
    out_tensors_.push_back(out_tensor);
    // setup blobs

    std::vector<size_t> blob_dim = {batchsize, vec_length};
    blob_tensors_.push_back(in_tensor);
    for (int i = 0; i < num_layers - 1; i++) {
      Tensor2<T> tensor;
      blobs_buff->reserve(blob_dim, &tensor);
      blob_tensors_.push_back(tensor);
    }
    blob_tensors_.push_back(out_tensor);

    for (int i = 0; i < 3; i++) {
      blobs_buff->reserve(blob_dim, &tmp_mat_tensors_[i]);
    }
    std::vector<size_t> tmp_vec_dim = {batchsize, 1};
    blobs_buff->reserve(tmp_vec_dim, &tmp_vec_tensor_);
    for (int i = 0; i < num_layers; i++) {
      Tensor2<T> tensor;
      blobs_buff->reserve(tmp_vec_dim, &tensor);
      vec_tensors_.push_back(tensor);
    }

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

template<>
void MultiCrossLayer<float>::reserve_master_weight_tensor(const std::shared_ptr<BufferBlock2<float>>& master_weight_buff,
                                                              const std::vector<size_t>& weight_bias_dim) {}

template<>
void MultiCrossLayer<__half>::reserve_master_weight_tensor(const std::shared_ptr<BufferBlock2<float>>& master_weight_buff,
                                                              const std::vector<size_t>& weight_bias_dim) {
  for (int i = 0; i < num_layers_; i++) {
    // setup weights
    {
      Tensor2<float> tensor;
      master_weight_buff->reserve(weight_bias_dim, &tensor);
      master_weights_.push_back(tensor);
    }
    // setup bias
    {
      Tensor2<float> tensor;
      master_weight_buff->reserve(weight_bias_dim, &tensor);
      master_weights_.push_back(tensor);
    }
  }
}

template <typename T>
void MultiCrossLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  Tensors2<T> kernel_tensors;
  Tensors2<T> bias_tensors;
  Tensors2<T> output_tensors;
  Tensors2<T> hidden_tensors;

  for (int i = 0; i < num_layers_; i++) {
    kernel_tensors.push_back(weights_[2 * i]);
    bias_tensors.push_back(weights_[2 * i + 1]);
  }

  for (int i = 0; i < num_layers_; i++) {
    output_tensors.push_back(blob_tensors_[i + 1]);
    hidden_tensors.push_back(vec_tensors_[i]);
  }

  MultiCrossForwardFunctor<T>()(get_gpu().get_stream(), get_gpu().get_cublas_handle(),
                             blob_tensors_[0], kernel_tensors, bias_tensors, output_tensors,
                             hidden_tensors, num_layers_);
}

template <typename T>
void MultiCrossLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());
  Tensors2<T> kernel_tensors;
  Tensors2<T> kernel_output_tensors;
  Tensors2<T> bias_output_tensors;
  Tensors2<T> forward_output_tensors;
  Tensors2<T> forward_hidden_tensors;

  for (int i = 0; i < num_layers_; i++) {
    kernel_tensors.push_back(weights_[2 * i]);
    kernel_output_tensors.push_back(wgrad_[2 * i]);
    bias_output_tensors.push_back(wgrad_[2 * i + 1]);
    forward_hidden_tensors.push_back(vec_tensors_[i]);
  }

  for (int i = 0; i < num_layers_ - 1; i++) {
    forward_output_tensors.push_back(blob_tensors_[i + 1]);
  }

  MultiCrossBackwardFunctor<T>()(get_gpu().get_stream(), blob_tensors_[0], kernel_tensors,
                              forward_output_tensors, forward_hidden_tensors,
                              blob_tensors_[num_layers_], blob_tensors_[0], kernel_output_tensors,
                              bias_output_tensors, tmp_vec_tensor_, tmp_mat_tensors_, num_layers_);
}

template <typename T>
std::unique_ptr<DataSimulator> MultiCrossLayer<T>::get_default_initializer(const int index) {
  const Tensor2<T>& in_tensor = in_tensors_[0];
  const Tensor2<T>& out_tensor = out_tensors_[0];
  float bottom_dim = in_tensor.get_dimensions()[1];
  float top_dim = out_tensor.get_dimensions()[1];

  std::unique_ptr<DataSimulator> simu(nullptr);
  if (0 == index) {
    simu.reset(new VarianceScalingSimulator(
        1.f, data_simu::Mode_t::Fan_avg, data_simu::Distribution_t::Uniform, bottom_dim, top_dim));
  } else if (1 == index) {
    simu.reset(new ConstantDataSimulator(0.0f));
  } else {
    CK_THROW_(Error_t::OutOfBound, "index != {0, 1}.");
  }

  return simu;
}

template class MultiCrossLayer<float>;
template class MultiCrossLayer<__half>;

}  // namespace HugeCTR
