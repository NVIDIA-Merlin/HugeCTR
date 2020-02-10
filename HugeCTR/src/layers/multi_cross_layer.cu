/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "HugeCTR/include/layers/multi_cross_layer.hpp"
#include "HugeCTR/include/utils.cuh"
#include <math.h>
#include <vector>

namespace HugeCTR {
  MultiCrossLayer::MultiCrossLayer( const std::shared_ptr<GeneralBuffer<float>>& weight_buff,
				    const std::shared_ptr<GeneralBuffer<float>>& wgrad_buff,
				    std::shared_ptr<Tensor<float>>& in_tensor,
				    std::shared_ptr<Tensor<float>>& out_tensor,
				    int num_layers, int device_id): 
    num_layers_(num_layers), blobs_buff_(new GeneralBuffer<float>()), Layer(device_id) {
    try{
      // check the in_tensor and out_tensor
      const auto& in_tensor_dim = in_tensor->get_dims();
      const auto& out_tensor_dim = out_tensor->get_dims();
      // 1. two dim?
      if (in_tensor_dim.size() != 2 || out_tensor_dim.size() != 2) {
	CK_THROW_(Error_t::WrongInput, "input or output tensor doesn't has two dimensions");
      }
      // 2. same dim?
      for(int i = 0;i<2; i++){
	if(in_tensor_dim[i] != out_tensor_dim[i]){
	  CK_THROW_(Error_t::WrongInput, "input and output tensor doesn't match");
	}
      }
      int vec_length = in_tensor_dim[1]; 
      int batchsize = in_tensor_dim[0]; 

      // check num_lyaers
      if (num_layers < 1){
	  CK_THROW_(Error_t::WrongInput, "num_layers < 1");
      }

     std::vector<int> weight_bias_dim = {1, vec_length};
      for(int i = 0; i<num_layers; i++){
	//setup weights
	weights_.emplace_back(new Tensor<float>(weight_bias_dim, weight_buff, TensorFormat_t::HW));
	//setup bias
	weights_.emplace_back(new Tensor<float>(weight_bias_dim, weight_buff, TensorFormat_t::HW));
	//setup weight gradient
	wgrad_.emplace_back(new Tensor<float>(weight_bias_dim, wgrad_buff, TensorFormat_t::HW));
	//setup bias gradient
	wgrad_.emplace_back(new Tensor<float>(weight_bias_dim, wgrad_buff, TensorFormat_t::HW));
      }

      in_tensors_.emplace_back(in_tensor);
      out_tensors_.emplace_back(out_tensor);
      //setup blobs
      std::vector<int> blob_dim = {batchsize, vec_length};
      blob_tensors_.emplace_back(in_tensor);
      for(int i = 0; i<num_layers-1; i++){
	blob_tensors_.emplace_back(new Tensor<float>(blob_dim, blobs_buff_, TensorFormat_t::HW));
      }
      blob_tensors_.emplace_back(out_tensor);

      for(int i = 0; i<TMP_MATS; i++){
	tmp_mat_tensors_.emplace_back(new Tensor<float>(blob_dim, blobs_buff_, TensorFormat_t::HW));
      }
      std::vector<int> tmp_vec_dim = {batchsize, 1};
      for(int i = 0; i < TMP_VECS; i++){
	tmp_vec_tensors_.emplace_back(new Tensor<float>(tmp_vec_dim, blobs_buff_, TensorFormat_t::HW));
      }
      blobs_buff_->init(device_id);
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
  }
  //kernels
  namespace {

    /**
     * Each row in `mat`dot product with vec, length of vec should be w. Then adding bias for each of the rows
     * @param out: hx1
     * @param mat: hxw
     * @param vec: 1xw
     */
    __global__ void matrix_vec_mul_kernel(float* out, float* mat, int h, int w, float* vec, float bias){
      const int tid = blockDim.x*blockIdx.x+threadIdx.x;
      const int wtid = tid%WARP_SIZE; //thread id in warp
      const int wid = tid/WARP_SIZE; //warp id
      const float* mat_with_offset = mat + wid*w;
      if(wid < h){
	double accum = 0.;
	for(int i = wtid; i < w; i+=WARP_SIZE){
	  accum +=  mat_with_offset[i]*vec[i];
	}
	float val = warpReduceSum(accum);
	if(wtid == 0){
	  out[wid] = val + bias;
	}
      }
    }

    void matrix_vec_mul(std::shared_ptr<Tensor<float>> out, 
			std::shared_ptr<Tensor<float>> mat, 
			std::shared_ptr<Tensor<float>> vec, float bias,
			cudaStream_t stream){
      float* pout = out->get_ptr();
      float* pmat = mat->get_ptr();
      float* pvec = vec->get_ptr();

      const auto dim = out->get_dims();
      const auto idim = mat->get_dims();
      assert(dim.size() == 2 && idim.size() == 2 && idim[1] == vec->get_dims()[1] && vec->get_dims()[0] == 1);
      assert(idim[0] == dim[0]);

      const int h = idim[0];
      const int w = idim[1];

      const int BLOCK_DIM = 256;
      const int GRID_DIM = calc_grid(h*WARP_SIZE, BLOCK_DIM);

      matrix_vec_mul_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat, h, w, pvec, bias);
    }

    /**
     * Each row in `mat` scale with the coresponding element in vec. 
     * The length of vec should be h.
     * @param o_mat: hxw
     * @param mat: hxw
     * @param vec: hx1
     */
    __global__ void row_scaling_kenrel(float* o_mat, float* mat, int h, int w, float* vec){
      const int tid = blockDim.x*blockIdx.x+threadIdx.x;
      if(tid < h*w){
	const int row = tid/w;
	o_mat[tid] = mat[tid]*vec[row];
      }
    }
    
    void row_scaling(std::shared_ptr<Tensor<float>> o_mat,
		     std::shared_ptr<Tensor<float>> mat, 
		     std::shared_ptr<Tensor<float>>vec,
		     cudaStream_t stream){

      float* pout = o_mat->get_ptr();
      float* pmat = mat->get_ptr();
      float* pvec = vec->get_ptr();

      const auto dim = o_mat->get_dims();
      const auto idim = mat->get_dims();
      assert(dim.size() == 2 && idim.size() == 2 && dim[0] == vec->get_dims()[0] && vec->get_dims()[1] == 1);
      assert(idim[0] == dim[0] && idim[1] == dim [1]);

      const int h = dim[0];
      const int w = dim[1];

      const int BLOCK_DIM = 256;
      const int GRID_DIM = calc_grid(h*w, BLOCK_DIM);

      row_scaling_kenrel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat, h, w, pvec);
    }

    /**
     * Each row in `mat` sum with  vec. 
     * The length of vec should be w.
     * @param o_mat: hxw
     * @param mat: hxw
     * @param vec: 1xw
     */
    __global__ void matrix_vec_add_kenrel(float* o_mat, float* mat, int h, int w, float* vec){
      const int tid = blockDim.x*blockIdx.x+threadIdx.x;
      if(tid < h*w){
	const int col = tid%w;
	o_mat[tid] = mat[tid]+vec[col];
      }
    }

    void matrix_vec_add(std::shared_ptr<Tensor<float>> o_mat, 
			std::shared_ptr<Tensor<float>> mat, 
			std::shared_ptr<Tensor<float>> vec,
			cudaStream_t stream){

      float* pout = o_mat->get_ptr();
      float* pmat = mat->get_ptr();
      float* pvec = vec->get_ptr();

      const auto dim = o_mat->get_dims();
      const auto idim = mat->get_dims();
      assert(dim.size() == 2 && idim.size() == 2 && dim[1] == vec->get_dims()[1] && vec->get_dims()[0] == 1);
      assert(idim[0] == dim[0] && idim[1] == dim [1]);

      const int h = dim[0];
      const int w = dim[1];

      const int BLOCK_DIM = 256;
      const int GRID_DIM = calc_grid(h*w, BLOCK_DIM);
      
      matrix_vec_add_kenrel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat, h, w, pvec);
    }

    /**
     * Pointwise adding
     */

    __global__ void matrix_add_kenrel(float* o_mat, float* mat_a, int h, int w, float* mat_b){
      const int tid = blockDim.x*blockIdx.x+threadIdx.x;
      if(tid < h*w){
	o_mat[tid] = mat_a[tid]+mat_b[tid];
      }
    }

    void matrix_add(std::shared_ptr<Tensor<float>> out_mat, 
		    std::shared_ptr<Tensor<float>>  mat_a, 
		    std::shared_ptr<Tensor<float>> mat_b,
		    cudaStream_t stream){
      float* pout = out_mat->get_ptr();
      float* pmat_a = mat_a->get_ptr();
      float* pmat_b = mat_b->get_ptr();

      const auto dim = out_mat->get_dims();

      const int h = dim[0];
      const int w = dim[1];

      const int BLOCK_DIM = 256;
      const int GRID_DIM = calc_grid(h*w, BLOCK_DIM);
      matrix_add_kenrel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat_a, h, w, pmat_b);
    }
    /**
     * compute dot product for each pair of the rows in the two matrix, 
     */
    __global__ void matrix_pair_mul_kernel(float* o_mat, float* mat_a, int h, int w, float* mat_b){
      const int tid = blockDim.x*blockIdx.x+threadIdx.x;
      const int wtid = tid%WARP_SIZE; //thread id in warp
      const int wid = tid/WARP_SIZE; //warp id
      const float* mat_a_with_offset = mat_a + wid*w;
      const float* mat_b_with_offset = mat_b + wid*w;
      if(wid < h){
	double accum = 0.f;
	for(int i = wtid; i < w; i+=WARP_SIZE){
	  accum += mat_a_with_offset[i]*mat_b_with_offset[i];
	}
	float val = warpReduceSum(accum);
	if(wtid == 0){
	  o_mat[wid] = val;
	}
      }
    }
    void matrix_pair_mul(std::shared_ptr<Tensor<float>> out_mat, 
			 std::shared_ptr<Tensor<float>> mat_a, 
			 std::shared_ptr<Tensor<float>> mat_b,
			 cudaStream_t stream){
      float* pout = out_mat->get_ptr();
      float* pmat_a = mat_a->get_ptr();
      float* pmat_b = mat_b->get_ptr();

      const auto dim = mat_a->get_dims();

      const int h = dim[0];
      const int w = dim[1];
      assert(h == mat_b->get_dims()[0] && w == mat_a->get_dims()[1] && h == out_mat->get_dims()[0] && 1  == out_mat->get_dims()[1]);

      const int BLOCK_DIM = 256;
      const int GRID_DIM = calc_grid(h*WARP_SIZE, BLOCK_DIM);
      matrix_pair_mul_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat_a, h, w, pmat_b);
    }
  
  
    /**
     * out product of two vectors
     * @param out_mat: hxw
     * @param vec_a: hx1
     * @param vec_b: 1xw
     */
    __global__ void out_product_kernel(float* out_mat, float* vec_a, int h, float* vec_b, int w){
      const int tid = blockDim.x*blockIdx.x+threadIdx.x;
      if(tid < h*w){
	const int col = tid%w;
	const int row = tid/w;
	out_mat[tid] = vec_a[row]*vec_b[col];
      }
    }
    void out_product(std::shared_ptr<Tensor<float>> out_mat, 
			    std::shared_ptr<Tensor<float>> vec_a, 
			    std::shared_ptr<Tensor<float>> vec_b,
			    cudaStream_t stream){
      float* pout = out_mat->get_ptr();
      float* pvec_a = vec_a->get_ptr();
      float* pvec_b = vec_b->get_ptr();
      const auto dim = out_mat->get_dims();

      const int h = dim[0];
      const int w = dim[1];

      assert(h == vec_a->get_dims()[0] && w == vec_b->get_dims()[1] && 
	     vec_a->get_dims()[1] == 1 && vec_b->get_dims()[0] == 1);
      const int BLOCK_DIM = 256;
      const int GRID_DIM = calc_grid(h*w, BLOCK_DIM);
      out_product_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pvec_a, h, pvec_b, w);
    }

    /**
     * Each row in `mat` scale with the coresponding element in vec. and accum across rows
     * The length of vec should be h.
     * @param o_mat: hxw
     * @param mat: hxw
     * @param vec: hx1
     */
    __global__ void row_scaling_sum_kernel(float* out, float* mat, int h, int w, float* vec){
      const int tid = blockDim.x*blockIdx.x+threadIdx.x;
      const int wtid = tid%WARP_SIZE; //thread id in warp
      const int wid = tid/WARP_SIZE; //warp id
      if(wid < w){
	double accum = 0.f;
	for(int i = wtid; i < h; i+=WARP_SIZE){
	  const int col = wid;
	  const int idx = i*w+col;
	  accum += mat[idx]*vec[i];
	}
	float val = warpReduceSum(accum);
	if(wtid == 0){
	  out[wid] += val; //using += here to enable regularization
	}
      }
    }

    void row_scaling_sum(std::shared_ptr<Tensor<float>> out, 
			 std::shared_ptr<Tensor<float>> mat, 
			 std::shared_ptr<Tensor<float>> vec,
			 cudaStream_t stream){
      float* pout = out->get_ptr();
      float* pmat = mat->get_ptr();
      float* pvec = vec->get_ptr();

      const auto dim = out->get_dims();
      const auto idim = mat->get_dims();
      assert(dim.size() == 2 && idim.size() == 2 && idim[0] == vec->get_dims()[0] && vec->get_dims()[1] == 1);
      assert(idim[1] == dim[1]);

      const int h = idim[0];
      const int w = idim[1];

      const int BLOCK_DIM = 256;
      const int GRID_DIM = calc_grid(w*WARP_SIZE, BLOCK_DIM); //each col one warp

      row_scaling_sum_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat, h, w, pvec);
    }


    /**
     * Accum across rows
     * @param o_mat: 1xw
     * @param mat: hxw
     */
    __global__ void row_sum_kernel(float* out, float* mat, int h, int w){
      const int tid = blockDim.x*blockIdx.x+threadIdx.x;
      const int wtid = tid%WARP_SIZE; //thread id in warp
      const int wid = tid/WARP_SIZE; //warp id
      if(wid < w){
	double accum = 0.f;
	for(int i = wtid; i < h; i+=WARP_SIZE){
	  const int col = wid;
	  const int idx = i*w+col;
	  accum += mat[idx];
	}
	float val = warpReduceSum(accum);
	if(wtid == 0){
	  out[wid] += val; //using += here to enable regularization
	}
      }
    }

    void rows_sum(std::shared_ptr<Tensor<float>> out, 
			 std::shared_ptr<Tensor<float>> mat, 
			 cudaStream_t stream){
      float* pout = out->get_ptr();
      float* pmat = mat->get_ptr();

      const auto dim = out->get_dims();
      const auto idim = mat->get_dims();
      assert(dim.size() == 2 && idim.size() == 2);
      assert(idim[1] == dim[1]);

      const int h = idim[0];
      const int w = idim[1];

      const int BLOCK_DIM = 256;
      const int GRID_DIM = calc_grid(w*WARP_SIZE, BLOCK_DIM); //each col one warp

      row_sum_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream>>>(pout, pmat, h, w);
    }
  }


  void MultiCrossLayer::fprof_step_(std::shared_ptr<Tensor<float>> xL_next, //output
				    std::shared_ptr<Tensor<float>> x0, 
				    std::shared_ptr<Tensor<float>> xL,
				    std::shared_ptr<Tensor<float>> wL,
				    std::shared_ptr<Tensor<float>> bL,
				    cudaStream_t stream){
    //tmp_vec[0] = matrix_vec_mul(xL,wL)
    matrix_vec_mul(tmp_vec_tensors_[0], xL, wL, 0.f, stream);
    //tmp_mat[0] = row_scaling(x0,tmp_vec)
    row_scaling(tmp_mat_tensors_[0], x0, tmp_vec_tensors_[0], stream);
    //tmp_mat[1] = matrix_add(tmp_mat[0],xL)
    matrix_add(tmp_mat_tensors_[1], tmp_mat_tensors_[0], xL, stream);
    //xL_next = matrix_vec_add(tmp_mat[1],bL)
    matrix_vec_add(xL_next, tmp_mat_tensors_[1], bL, stream);

    return;
  }

  
  void MultiCrossLayer::fprop(cudaStream_t stream){
    auto x0 = blob_tensors_[0];
    for(unsigned int i=0; i<blob_tensors_.size()-1; i++){
      fprof_step_(blob_tensors_[i+1], x0, blob_tensors_[i], weights_[2*i], weights_[2*i+1], stream);
    }
    return;
  }

  void MultiCrossLayer::bprop_first_step_(std::shared_ptr<Tensor<float>> dxL_pre, //output
					  std::shared_ptr<Tensor<float>> dwL, //output
					  std::shared_ptr<Tensor<float>> dbL, //output
					  std::shared_ptr<Tensor<float>> x0, //Note: x0 and dxL_pre share the same buffer
					  std::shared_ptr<Tensor<float>> dxL,
					  std::shared_ptr<Tensor<float>> wL,
					  std::shared_ptr<Tensor<float>> bL,
					  cudaStream_t stream){
    //tmp_vec[0] = matrix_pair_mul(dxL, x0)
    matrix_pair_mul(tmp_vec_tensors_[0], dxL, x0, stream);
    //tmp_mat[0] = out_product(tmp_vec[0], wL)
    out_product(tmp_mat_tensors_[0], tmp_vec_tensors_[0], wL, stream);
    //tmp_vec[1] = matrix_vec_mul(x0, wL, 1.0)
    matrix_vec_mul(tmp_vec_tensors_[1], x0, wL, 1.0f, stream);
    //tmp_mat[1] = row_scaling(dxL, tmp_vec[1])
    row_scaling(tmp_mat_tensors_[1], dxL, tmp_vec_tensors_[1], stream);

    //dwL = row_scaling_sum(x0, tmp_vec[0])
    row_scaling_sum(dwL, x0, tmp_vec_tensors_[0], stream);

    //dbL = rows_sum(dxL)
    rows_sum(dbL, dxL, stream);

    //dxL_pre = matrix_add(tmp_mat[0], tmp_mat[1])
    matrix_add(dxL_pre, tmp_mat_tensors_[0], tmp_mat_tensors_[1], stream);
    
    return;
  }


  void MultiCrossLayer::bprop_step_(std::shared_ptr<Tensor<float>> dxL_pre, //output
				    std::shared_ptr<Tensor<float>> dwL, //output
				    std::shared_ptr<Tensor<float>> dbL, //output
				    std::shared_ptr<Tensor<float>> x0,
				    std::shared_ptr<Tensor<float>> xL, //Note: xL and dxL_pre share the same buffer
				    std::shared_ptr<Tensor<float>> dxL,
				    std::shared_ptr<Tensor<float>> wL,
				    std::shared_ptr<Tensor<float>> bL,
				    cudaStream_t stream){
    //tmp_vec[0] = matrix_pair_mul(dxL, x0)
    matrix_pair_mul(tmp_vec_tensors_[0], dxL, x0, stream);
    //tmp_mat[0] = out_product(tmp_vec[0], wL)
    out_product(tmp_mat_tensors_[0], tmp_vec_tensors_[0], wL, stream);
    
    //dwL = row_scaling_sum(xL, tmp_vec[0])
    row_scaling_sum(dwL, xL, tmp_vec_tensors_[0], stream);

    //dbL = rows_sum(dxL)
    rows_sum(dbL, dxL, stream);

    //dxL_pre = matrix_add(tmp_mat[0], dxL)
    matrix_add(dxL_pre, tmp_mat_tensors_[0], dxL, stream);

    return;
  }


  void MultiCrossLayer::bprop(cudaStream_t stream){
    auto& x0 = blob_tensors_[0];
    for(int i=blob_tensors_.size()-1; i>1; i--){
      bprop_step_(blob_tensors_[i-1], wgrad_[(i-1)*2], wgrad_[(i-1)*2+1], x0, blob_tensors_[i-1],
		  blob_tensors_[i], weights_[(i-1)*2], weights_[(i-1)*2+1], stream);
    }
    bprop_first_step_(x0, wgrad_[0], wgrad_[1], x0, blob_tensors_[1], weights_[0], weights_[1], stream);
    return;
  }

  std::vector<float> MultiCrossLayer::get_initializer() {
    std::vector<float> initializer;
    size_t weight_size = 0;
    for(const auto& w: weights_){
      weight_size += w->get_num_elements();
    }
    initializer.resize(weight_size);
    const auto& in_tensor = in_tensors_[0];
    float in_dim = in_tensor->get_dims()[1];
    float sigma = 1.f / sqrt(in_dim);
    HugeCTR::GaussianDataSimulator<float> fdata_sim(0.f, sigma, -2 * sigma, 2 * sigma);
    for (size_t i = 0; i < initializer.size(); i++) initializer[i] = fdata_sim.get_num();
    return initializer;
  }

} //namespace HugeCTR
