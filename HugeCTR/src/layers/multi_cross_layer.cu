/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <math.h>
#include <vector>
#include "HugeCTR/include/layers/multi_cross_layer.hpp"
#include "HugeCTR/include/prims/linalg/matrix_multiplication.cuh"
#include "HugeCTR/include/utils.cuh"


#include <linalg/binary_op.cuh>
#include <linalg/gemm.cuh>
#include <linalg/gemv.h>
#include <linalg/matrix_vector_op.cuh>
#include <linalg/reduce.cuh>

namespace HugeCTR {

// kernels
namespace {

inline int calc_grid(int t, int b) { return (t - 1) / b + 1; }

/**
 * Each row in `mat`dot product with vec, length of vec should be w. Then adding bias for each of
 * the rows
 * @param out: hx1
 * @param mat: hxw
 * @param vec: 1xw
  
__global__ void matrix_vec_mul_kernel(float* out, const float* mat, int h, int w,
                                      const float* vec) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int wtid = tid % WARP_SIZE;  // thread id in warp
  const int wid = tid / WARP_SIZE;   // warp id
  const float* mat_with_offset = mat + wid * w;
  if (wid < h) {
    float accum = 0.;
    for (int i = wtid; i < w; i += WARP_SIZE) {
      accum += mat_with_offset[i] * vec[i];
    }
    float val = warpReduceSum(accum);
    if (wtid == 0) {
      out[wid] = val;
    }
  }
}
*/

void matrix_vec_mul(Tensor<float>& out, const Tensor<float>& mat, const Tensor<float>& vec,
                    cublasHandle_t cublas_handle, cudaStream_t stream) {
  float* pout = out.get_ptr();
  const float* pmat = mat.get_ptr();
  const float* pvec = vec.get_ptr();

  const auto& dim = out.get_dims();
  const auto& idim = mat.get_dims();
  assert(dim.size() == 2 && idim.size() == 2 && idim[1] == vec.get_dims()[1] &&
         vec.get_dims()[0] == 1);
  assert(idim[0] == dim[0]);

  const int h = idim[0];
  const int w = idim[1];
  
  //const int BLOCK_DIM = 256;
  //const int GRID_DIM = calc_grid(h * WARP_SIZE, BLOCK_DIM);
  //matrix_vec_mul_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat, h, w, pvec);
  
  // TODO: Get cublas handler out of the function 
  // cublasHandle_t handle;
  // CUBLAS_CHECK(cublasCreate(&handle));
  MLCommon::LinAlg::gemv<float>(pmat, w, h, pvec, pout, true, 1.f, 0.f, cublas_handle, stream);
  //CUBLAS_CHECK(cublasDestroy(handle));
}

/**
 * Each row in `mat` scale with the coresponding element in vec.
 * The length of vec should be h.
 * @param o_mat: hxw
 * @param mat: hxw
 * @param vec: hx1

__global__ void row_scaling_kenrel(float* o_mat, const float* mat, int h, int w, const float* vec) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < h * w) {
    const int row = tid / w;
    o_mat[tid] = mat[tid] * vec[row];
  }
}
*/

void row_scaling(Tensor<float>& o_mat, const Tensor<float>& mat, const Tensor<float>& vec,
                 cudaStream_t stream) {
  float* pout = o_mat.get_ptr();
  const float* pmat = mat.get_ptr();
  const float* pvec = vec.get_ptr();

  const auto& dim = o_mat.get_dims();
  const auto& idim = mat.get_dims();
  assert(dim.size() == 2 && idim.size() == 2 && dim[0] == vec.get_dims()[0] &&
         vec.get_dims()[1] == 1);
  assert(idim[0] == dim[0] && idim[1] == dim[1]);

  const int h = dim[0];
  const int w = dim[1];

  //const int BLOCK_DIM = 256;
  //const int GRID_DIM = calc_grid(h * w, BLOCK_DIM);

  //row_scaling_kenrel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat, h, w, pvec);

  MLCommon::LinAlg::matrixVectorOp(pout, pmat, pvec, h, w, false, true,
          [] __device__(float a, float b) { return a * b; }, stream);

}

/**
 * Each row in `mat` sum with  vec.
 * The length of vec should be w.
 * @param o_mat: hxw
 * @param mat: hxw
 * @param vec: 1xw

__global__ void matrix_vec_add_kenrel(float* o_mat, const float* mat, int h, int w,
                                      const float* vec) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < h * w) {
    const int col = tid % w;
    o_mat[tid] = mat[tid] + vec[col];
  }
}
*/

void matrix_vec_add(Tensor<float>& o_mat, const Tensor<float>& mat, const Tensor<float>& vec,
                    cudaStream_t stream) {
  float* pout = o_mat.get_ptr();
  const float* pmat = mat.get_ptr();
  const float* pvec = vec.get_ptr();

  const auto& dim = o_mat.get_dims();
  const auto& idim = mat.get_dims();
  assert(dim.size() == 2 && idim.size() == 2 && dim[1] == vec.get_dims()[1] &&
         vec.get_dims()[0] == 1);
  assert(idim[0] == dim[0] && idim[1] == dim[1]);

  const int h = dim[0];
  const int w = dim[1];

  //const int BLOCK_DIM = 256;
  //const int GRID_DIM = calc_grid(h * w, BLOCK_DIM);

  //matrix_vec_add_kenrel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat, h, w, pvec);

  MLCommon::LinAlg::matrixVectorOp(pout, pmat, pvec, h, w, false, false,
            [] __device__(float a, float b) { return a + b; }, stream);

}

/**
 * Pointwise adding

__global__ void matrix_add_kenrel(float* o_mat, const float* mat_a, int h, int w,
                                  const float* mat_b) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < h * w) {
    o_mat[tid] = mat_a[tid] + mat_b[tid];
  }
}
*/

void matrix_add(Tensor<float>& out_mat, const Tensor<float>& mat_a, const Tensor<float>& mat_b,
                cudaStream_t stream) {
  float* pout = out_mat.get_ptr();
  const float* pmat_a = mat_a.get_ptr();
  const float* pmat_b = mat_b.get_ptr();

  const auto& dim = out_mat.get_dims();
  const auto& idim1 = mat_a.get_dims();
  const auto& idim2 = mat_b.get_dims();
  assert(idim1[0] == dim[0] && idim1[1] == dim[1]);
  assert(idim2[0] == dim[0] && idim2[1] == dim[1]);

  const int h = dim[0];
  const int w = dim[1];

  //const int BLOCK_DIM = 256;
  //const int GRID_DIM = calc_grid(h * w, BLOCK_DIM);
  //matrix_add_kenrel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat_a, h, w, pmat_b);

  MLCommon::LinAlg::binaryOp(pout, pmat_a, pmat_b, h * w,
            [] __device__(float a, float b) { return a + b; }, stream);

}

/**
 * compute dot product for each pair of the rows in the two matrix,
 */
__global__ void matrix_pair_mul_kernel(float* o_vec, const float* mat_a, int h, int w,
                                       const float* mat_b) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int wtid = tid % WARP_SIZE;  // thread id in warp
  const int wid = tid / WARP_SIZE;   // warp id
  const float* mat_a_with_offset = mat_a + wid * w;
  const float* mat_b_with_offset = mat_b + wid * w;
  if (wid < h) {
    float accum = 0.f;
    for (int i = wtid; i < w; i += WARP_SIZE) {
      accum += mat_a_with_offset[i] * mat_b_with_offset[i];
    }
    float val = warpReduceSum(accum);
    if (wtid == 0) {
      o_vec[wid] = val;
    }
  }
}

void matrix_pair_mul(Tensor<float>& o_vec, const Tensor<float>& mat_a, const Tensor<float>& mat_b,
                     cudaStream_t stream) {
  float* pout = o_vec.get_ptr();
  const float* pmat_a = mat_a.get_ptr();
  const float* pmat_b = mat_b.get_ptr();

  const auto& dim = mat_a.get_dims();

  const int h = dim[0];
  const int w = dim[1];
  assert(h == mat_b.get_dims()[0] && w == mat_a.get_dims()[1] && h == o_vec.get_dims()[0] &&
         1 == o_vec.get_dims()[1]);

  const int BLOCK_DIM = 256;
  const int GRID_DIM = calc_grid(h * WARP_SIZE, BLOCK_DIM);
  matrix_pair_mul_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat_a, h, w, pmat_b);
}

void out_product(Tensor<float>& out_mat, const Tensor<float>& vec_a, const Tensor<float>& vec_b,
                 cudaStream_t stream) {
  float* pout = out_mat.get_ptr();
  const float* pvec_a = vec_a.get_ptr();
  const float* pvec_b = vec_b.get_ptr();
  const auto& dim = out_mat.get_dims();

  const int h = dim[0];
  const int w = dim[1];

  assert(h == vec_a.get_dims()[0] && w == vec_b.get_dims()[1] && vec_a.get_dims()[1] == 1 &&
         vec_b.get_dims()[0] == 1);
  
  const int BLOCK_DIM = 256;
  const int GRID_DIM = calc_grid(h * w, BLOCK_DIM);
  MLCommon::LinAlg::mm_1d<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pvec_a, h, pvec_b, w);
}

/**
 * Each row in `mat` scale with the coresponding element in vec. and accum across rows
 * The length of vec should be h.
 * @param o_mat: hxw
 * @param mat: hxw
 * @param vec: hx1
 */
__global__ void row_scaling_sum_kernel(float* out, const float* mat, int h, int w,
                                       const float* vec) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int wtid = tid % WARP_SIZE;  // thread id in warp
  const int wid = tid / WARP_SIZE;   // warp id
  if (wid < w) {
    float accum = 0.f;
    for (int i = wtid; i < h; i += WARP_SIZE) {
      const int col = wid;
      const int idx = i * w + col;
      accum += mat[idx] * vec[i];
    }
    float val = warpReduceSum(accum);
    if (wtid == 0) {
      out[wid] += val;  // using += here to enable regularization
    }
  }
}

void row_scaling_sum(Tensor<float>& out, const Tensor<float>& mat, const Tensor<float>& vec,
                     cudaStream_t stream) {
  float* pout = out.get_ptr();
  const float* pmat = mat.get_ptr();
  const float* pvec = vec.get_ptr();

  const auto& dim = out.get_dims();
  const auto& idim = mat.get_dims();
  assert(dim.size() == 2 && idim.size() == 2 && idim[0] == vec.get_dims()[0] &&
         vec.get_dims()[1] == 1);
  assert(idim[1] == dim[1]);

  const int h = idim[0];
  const int w = idim[1];

  const int BLOCK_DIM = 256;
  const int GRID_DIM = calc_grid(w * WARP_SIZE, BLOCK_DIM);  // each col one warp

  row_scaling_sum_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat, h, w, pvec);
}

/**
 * Accum across rows
 * @param o_mat: 1xw
 * @param mat: hxw

__global__ void row_sum_kernel(float* out, const float* mat, int h, int w) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int wtid = tid % WARP_SIZE;  // thread id in warp
  const int wid = tid / WARP_SIZE;   // warp id
  if (wid < w) {
    float accum = 0.f;
    for (int i = wtid; i < h; i += WARP_SIZE) {
      const int col = wid;
      const int idx = i * w + col;
      accum += mat[idx];
    }
    float val = warpReduceSum(accum);
    if (wtid == 0) {
      out[wid] += val;  // using += here to enable regularization
    }
  }
}
*/

void rows_sum(Tensor<float>& out, const Tensor<float>& mat, cudaStream_t stream) {
  float* pout = out.get_ptr();
  const float* pmat = mat.get_ptr();

  const auto& dim = out.get_dims();
  const auto& idim = mat.get_dims();
  assert(dim.size() == 2 && idim.size() == 2);
  assert(idim[1] == dim[1]);

  const int h = idim[0];
  const int w = idim[1];

  //const int BLOCK_DIM = 256;
  //const int GRID_DIM = calc_grid(w * WARP_SIZE, BLOCK_DIM);  // each col one warp

  //row_sum_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat, h, w);

  MLCommon::LinAlg::reduce(pout, pmat, h, w, (float)0, false, true, stream, false,
          [] __device__(float in, int i) { return in; });

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
void MultiCrossForwardFunctor::operator()(cudaStream_t stream, cublasHandle_t cublas_handle, 
                                          const Tensor<float>& input_tensor,
                                          const std::vector<const Tensor<float>*>& kernel_tensors,
                                          const std::vector<const Tensor<float>*>& bias_tensors,
                                          const std::vector<Tensor<float>*>& layer_output_tensors,
                                          const std::vector<Tensor<float>*>& layer_hidden_tensors,
                                          int num_layers) const {
  for (int i = 0; i < num_layers; i++) {
    matrix_vec_mul(*layer_hidden_tensors[i], i == 0 ? input_tensor : *layer_output_tensors[i - 1],
                   *kernel_tensors[i], cublas_handle, stream);
    row_scaling(*layer_output_tensors[i], input_tensor, *layer_hidden_tensors[i], stream);
    matrix_add(*layer_output_tensors[i], *layer_output_tensors[i],
               i == 0 ? input_tensor : *layer_output_tensors[i - 1], stream);
    matrix_vec_add(*layer_output_tensors[i], *layer_output_tensors[i], *bias_tensors[i], stream);
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
void MultiCrossBackwardFunctor::operator()(
    cudaStream_t stream, const Tensor<float>& input_tensor,
    const std::vector<const Tensor<float>*>& kernel_tensors,
    const std::vector<const Tensor<float>*>& layer_output_tensors,
    const std::vector<const Tensor<float>*>& layer_hidden_tensors, const Tensor<float>& grad_tensor,
    Tensor<float>& output_tensor, const std::vector<Tensor<float>*>& kernel_output_tensors,
    const std::vector<Tensor<float>*>& bias_output_tensors, Tensor<float>& tmp_vec_tensor,
    const std::vector<Tensor<float>*>& tmp_mat_tensors, int num_layers) const {
  cudaMemsetAsync(tmp_mat_tensors[2]->get_ptr(), 0, tmp_mat_tensors[2]->get_size(), stream);
  for (int i = num_layers - 1; i >= 0; i--) {
    row_scaling(*tmp_mat_tensors[0], i == num_layers - 1 ? grad_tensor : *tmp_mat_tensors[1],
                *layer_hidden_tensors[i], stream);
    matrix_add(*tmp_mat_tensors[2], *tmp_mat_tensors[2], *tmp_mat_tensors[0], stream);
    matrix_pair_mul(tmp_vec_tensor, i == num_layers - 1 ? grad_tensor : *tmp_mat_tensors[1],
                    input_tensor, stream);
    row_scaling_sum(*kernel_output_tensors[i], i == 0 ? input_tensor : *layer_output_tensors[i - 1],
                    tmp_vec_tensor, stream);
    rows_sum(*bias_output_tensors[i], i == num_layers - 1 ? grad_tensor : *tmp_mat_tensors[1],
             stream);
    out_product(*tmp_mat_tensors[0], tmp_vec_tensor, *kernel_tensors[i], stream);
    matrix_add(*tmp_mat_tensors[1], i == num_layers - 1 ? grad_tensor : *tmp_mat_tensors[1],
               *tmp_mat_tensors[0], stream);
  }
  matrix_add(output_tensor, *tmp_mat_tensors[2], *tmp_mat_tensors[1], stream);
}

MultiCrossLayer::MultiCrossLayer(const GeneralBufferPtr<float>& weight_buff,
                                 const GeneralBufferPtr<float>& wgrad_buff,
                                 const TensorPtr<float>& in_tensor,
                                 const TensorPtr<float>& out_tensor,  cublasHandle_t const& cublas_handle,
                                 int num_layers, int device_id,
                                 std::vector<Initializer_t> initializer_types)
    : cublas_handle_(cublas_handle), 
      Layer(device_id, initializer_types), 
      num_layers_(num_layers), 
      blobs_buff_(new GeneralBuffer<float>()) {
  try {
    // check the in_tensor and out_tensor
    const auto& in_tensor_dim = in_tensor->get_dims();
    const auto& out_tensor_dim = out_tensor->get_dims();
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
    for (int i = 0; i < num_layers; i++) {
      // setup weights
      weights_.emplace_back(new Tensor<float>(weight_bias_dim, weight_buff, TensorFormat_t::HW));
      // setup bias
      weights_.emplace_back(new Tensor<float>(weight_bias_dim, weight_buff, TensorFormat_t::HW));
      // setup weight gradient
      wgrad_.emplace_back(new Tensor<float>(weight_bias_dim, wgrad_buff, TensorFormat_t::HW));
      // setup bias gradient
      wgrad_.emplace_back(new Tensor<float>(weight_bias_dim, wgrad_buff, TensorFormat_t::HW));
    }

    in_tensors_.emplace_back(in_tensor);
    out_tensors_.emplace_back(out_tensor);
    // setup blobs
    std::vector<size_t> blob_dim = {batchsize, vec_length};
    blob_tensors_.emplace_back(in_tensor);
    for (int i = 0; i < num_layers - 1; i++) {
      blob_tensors_.emplace_back(new Tensor<float>(blob_dim, blobs_buff_, TensorFormat_t::HW));
    }
    blob_tensors_.emplace_back(out_tensor);

    for (int i = 0; i < 3; i++) {
      tmp_mat_tensors_[i].reset(new Tensor<float>(blob_dim, blobs_buff_, TensorFormat_t::HW));
    }
    std::vector<size_t> tmp_vec_dim = {batchsize, 1};
    tmp_vec_tensor_.reset(new Tensor<float>(tmp_vec_dim, blobs_buff_, TensorFormat_t::HW));
    for (int i = 0; i < num_layers; i++) {
      vec_tensors_.emplace_back(new Tensor<float>(tmp_vec_dim, blobs_buff_, TensorFormat_t::HW));
    }
    blobs_buff_->init(device_id);
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void MultiCrossLayer::fprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  std::vector<const Tensor<float>*> kernel_tensors;
  std::vector<const Tensor<float>*> bias_tensors;
  std::vector<Tensor<float>*> output_tensors;
  std::vector<Tensor<float>*> hidden_tensors;

  for (int i = 0; i < num_layers_; i++) {
    kernel_tensors.push_back(weights_[2 * i].get());
    bias_tensors.push_back(weights_[2 * i + 1].get());
  }

  for (int i = 0; i < num_layers_; i++) {
    output_tensors.push_back(blob_tensors_[i + 1].get());
    hidden_tensors.push_back(vec_tensors_[i].get());
  }

  MultiCrossForwardFunctor()(stream, cublas_handle_, *blob_tensors_[0], kernel_tensors, bias_tensors,
                             output_tensors, hidden_tensors, num_layers_);
}

void MultiCrossLayer::bprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  std::vector<const Tensor<float>*> kernel_tensors;
  std::vector<Tensor<float>*> kernel_output_tensors;
  std::vector<Tensor<float>*> bias_output_tensors;
  std::vector<const Tensor<float>*> forward_output_tensors;
  std::vector<const Tensor<float>*> forward_hidden_tensors;

  for (int i = 0; i < num_layers_; i++) {
    kernel_tensors.push_back(weights_[2 * i].get());
    kernel_output_tensors.push_back(wgrad_[2 * i].get());
    bias_output_tensors.push_back(wgrad_[2 * i + 1].get());
    forward_hidden_tensors.push_back(vec_tensors_[i].get());
  }

  for (int i = 0; i < num_layers_ - 1; i++) {
    forward_output_tensors.push_back(blob_tensors_[i + 1].get());
  }

  MultiCrossBackwardFunctor()(
      stream, *blob_tensors_[0], kernel_tensors, forward_output_tensors, forward_hidden_tensors,
      *blob_tensors_[num_layers_], *blob_tensors_[0], kernel_output_tensors, bias_output_tensors,
      *tmp_vec_tensor_,
      {tmp_mat_tensors_[0].get(), tmp_mat_tensors_[1].get(), tmp_mat_tensors_[2].get()},
      num_layers_);
}

std::unique_ptr<DataSimulator<float>> MultiCrossLayer::get_default_initializer(const int index) {
  const auto& in_tensor = in_tensors_[0];
  const auto& out_tensor = out_tensors_[0];
  float bottom_dim = in_tensor->get_dims()[1];
  float top_dim = out_tensor->get_dims()[1];

  std::unique_ptr<DataSimulator<float>> simu(nullptr);
  if (0 == index) {
    simu.reset(new VarianceScalingSimulator<float>(
        1.f, data_simu::Mode_t::Fan_avg, data_simu::Distribution_t::Uniform, bottom_dim, top_dim));
  } else if (1 == index) {
    auto zero_init = [] { return static_cast<float>(0); };
    simu.reset(new SingleDataSimulator<float>(zero_init));
  } else {
    CK_THROW_(Error_t::OutOfBound, "index != {0, 1}.");
  }

  return simu;
}

}  // namespace HugeCTR
