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

#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

#include <math.h>
#include <memory>
#include <vector>

using namespace HugeCTR;


class MultiCrossLayerTest{
private:
  const float eps = 1;
  const int batchsize_;
  const int w_;
  const int layers_;
  std::shared_ptr<GeneralBuffer<float>> blob_buf_;
  std::shared_ptr<GeneralBuffer<float>> weight_buf_;
  std::shared_ptr<GeneralBuffer<float>> wgrad_buf_;
  std::shared_ptr<Tensor<float>> input_;
  std::shared_ptr<Tensor<float>> output_;
  std::shared_ptr<MultiCrossLayer> layer_;
  GaussianDataSimulator<float> data_sim_;
  std::shared_ptr<std::vector<float>> h_in_;
  std::shared_ptr<std::vector<float>> h_out_;
  std::shared_ptr<std::vector<float>> h_weight_;
  void reset_(){
    for(auto& a: *h_in_){
      a = data_sim_.get_num();
    }
    for(auto& b: *h_out_){
      b = data_sim_.get_num();
    }
    for(auto& w: *h_weight_){
      w = data_sim_.get_num();
    }

    CK_CUDA_THROW_(cudaMemcpy(input_->get_ptr(), h_in_->data(), input_->get_size(), cudaMemcpyHostToDevice));
    CK_CUDA_THROW_(cudaMemcpy(output_->get_ptr(), h_out_->data(), output_->get_size(), cudaMemcpyHostToDevice));
    CK_CUDA_THROW_(cudaMemcpy(weight_buf_->get_ptr_with_offset(0), h_weight_->data(), weight_buf_->get_size(), cudaMemcpyHostToDevice));
    wgrad_buf_->reset_sync();

    return; 
  }


  void cpu_fprop_step_(std::shared_ptr<std::vector<float>> x0__, std::shared_ptr<std::vector<float>> xL__, 
		       std::shared_ptr<std::vector<float>> w__, std::shared_ptr<std::vector<float>> out__, 
		       std::shared_ptr<std::vector<float>> bias__){
    auto& x0 = *x0__;
    auto& xL = *xL__;
    auto& w = *w__;
    auto& out = *out__;
    auto& bias = *bias__;

    assert(x0.size()%w.size() == 0);
    const int batchsize = x0.size()/w.size();
    const int width = w.size();

    std::vector<double> tmp_v(batchsize, 0.f); //Batchsize
    std::vector<double> tmp_m(x0.size(), 0.f); //Batchsize*width
    //tmp_v = xL*w
    for(int i=0; i<batchsize; i++){
      for(int j=0; j< width; j++){
        tmp_v[i]+=xL[i*width + j]*w[j];
      }
    }
    //tmp_m = x0*tmp
    for(int i=0; i<batchsize; i++){
      for(int j=0; j< width; j++){
        tmp_m[i*width+j]+=x0[i*width + j]*tmp_v[i];
      }
    }
    //tmp_m+=xL
    //out = tmp_m+bias
    for(int i=0; i<batchsize; i++){
      for(int j=0; j< width; j++){
        out[i*width+j] = tmp_m[i*width+j]+xL[i*width+j]+bias[j];
      }
    }
    return;
  }

  void cpu_fprop_(){
    std::vector<std::shared_ptr<std::vector<float>>> blobs;
    blobs.emplace_back(h_in_);
    for(int i=0; i<layers_-1; i++){
      blobs.emplace_back(std::make_shared<std::vector<float>>(h_in_->size(), 0.f)); //Batchsize*width
    }
    blobs.emplace_back(h_out_);

    std::shared_ptr<std::vector<float>> weight_tmp = std::make_shared<std::vector<float>>(w_);
    std::shared_ptr<std::vector<float>> bias_tmp = std::make_shared<std::vector<float>>(w_);

    for(int i=0; i<layers_; i++){
      int offset = w_*2*i;
      std::copy(h_weight_->begin()+offset, h_weight_->begin()+offset+w_, weight_tmp->begin());
      std::copy(h_weight_->begin()+offset+w_, h_weight_->begin()+offset+2*w_, bias_tmp->begin());
      cpu_fprop_step_(blobs[0], blobs[i], weight_tmp, blobs[i+1], bias_tmp);
    }

    
    return;
  }

  void gpu_fprop_(){
    std::cout << "[HCDEBUG][CALL] " << __FUNCTION__ << " " << std::endl;
    layer_->fprop(cudaStreamDefault);
    return;
  }

  void cpu_bprop_step_(std::shared_ptr<std::vector<float>> x0__,std::shared_ptr<std::vector<float>> dxL__, 
		       std::shared_ptr<std::vector<float>> w__, std::shared_ptr<std::vector<float>> out__, 
		       std::shared_ptr<std::vector<float>> bias__){
    auto& x0 = *x0__;
    auto& dxL = *dxL__;
    auto& w = *w__;
    auto& out = *out__;
    auto& bias = *bias__;


    assert(x0.size()%w.size() == 0);
    const int batchsize = x0.size()/w.size();
    const int width = w.size();

    std::vector<double> tmp_v(batchsize, 0.f); //Batchsize
    std::vector<double> tmp_m(x0.size(), 0.f); //Batchsize*width

    //tmp_v = dxL*x0
    for(int i=0; i<batchsize; i++){
      for(int j=0; j<width; j++){
	tmp_v[i] += dxL[i*width+j]*x0[i*width+j];
      }
    }

    //tmp_m = out_product(tmp_vec, wd)
    for(int i=0; i<batchsize; i++){
      for(int j=0; j<width; j++){
	tmp_m[i*width+j] = tmp_v[i]*w[j];
      }
    }
    
    //tmp_m += dxL
    for(int i=0; i<batchsize; i++){
      for(int j=0; j<width; j++){
	out[i*width+j] =  dxL[i*width+j] + tmp_m[i*width+j];
      }
    }

    return;
  }

  void cpu_bprop_first_step_(std::shared_ptr<std::vector<float>> x0__, std::shared_ptr<std::vector<float>> dxL__, 
			     std::shared_ptr<std::vector<float>> w__, std::shared_ptr<std::vector<float>> out__, 
			     std::shared_ptr<std::vector<float>> bias__){
    auto& x0 = *x0__;
    auto& dxL = *dxL__;
    auto& w = *w__;
    auto& out = *out__;
    auto& bias = *bias__;

    assert(x0.size()%w.size() == 0);
    const int batchsize = x0.size()/w.size();
    const int width = w.size();

    
    std::vector<double> tmp_v(batchsize, 0.f); //Batchsize
    std::vector<double> tmp_v_b(batchsize, 0.f); //Batchsize
    std::vector<double> tmp_m(x0.size(), 0.f); //Batchsize*width
    std::vector<double> tmp_m_b(x0.size(), 0.f); //Batchsize*width

    //tmp_v = dxL*x0
    for(int i=0; i<batchsize; i++){
      for(int j=0; j<width; j++){
	tmp_v[i] += dxL[i*width+j]*x0[i*width+j];
      }
    }
    std::cout << "print first cpu:" << std::endl;
    std::cout << tmp_v[0] << std::endl;

    //tmp_m = out_product(tmp_vec, wd)
    for(int i=0; i<batchsize; i++){
      for(int j=0; j<width; j++){
	tmp_m[i*width+j] = tmp_v[i]*w[j];
      }
    }    

    //tmp_v_b = x0*w+1
    for(int i=0; i<batchsize; i++){
      for(int j=0; j<width; j++){
	tmp_v_b[i] += x0[i*width+j]*w[j];
      }
      tmp_v_b[i] += 1.f;
    }    

    //tmp_m_b = dxL*tmp_v_b
    for(int i=0; i<batchsize; i++){
      for(int j=0; j<width; j++){
	tmp_m_b[i*width+j] = dxL[i*width+j]*tmp_v_b[i];
      }
    }    

    //out = tmp_m + tmp_m_b
    for(int i=0; i<batchsize; i++){
      for(int j=0; j<width; j++){
	out[i*width+j] = tmp_m[i*width+j] + tmp_m_b[i*width+j];
      }
    }

    
    return;
  }
  

  void cpu_bprop_(){
    std::vector<std::shared_ptr<std::vector<float>>> blobs;
    blobs.emplace_back(h_in_);
    for(int i=0; i<layers_-1; i++){
      blobs.emplace_back(std::make_shared<std::vector<float>>(h_in_->size(), 0.f)); //Batchsize*width
    }
    blobs.emplace_back(h_out_);

    std::shared_ptr<std::vector<float>> weight_tmp = std::make_shared<std::vector<float>>(w_);
    std::shared_ptr<std::vector<float>> bias_tmp = std::make_shared<std::vector<float>>(w_);

    for(int i=layers_-1; i>=0; i--){
      int offset = w_*2*i;
      std::copy(h_weight_->begin()+offset, h_weight_->begin()+offset+w_, weight_tmp->begin());
      std::copy(h_weight_->begin()+offset+w_, h_weight_->begin()+offset+2*w_, bias_tmp->begin());
      if(i!=0)
	cpu_bprop_step_(blobs[0], blobs[i+1], weight_tmp, blobs[i], bias_tmp);
      else
	cpu_bprop_first_step_(blobs[0], blobs[i+1], weight_tmp, blobs[i], bias_tmp);
    }
    return;
  }

  void gpu_bprop_(){
    layer_->bprop(cudaStreamDefault);
    return;
  }

  void compare_(){
    std::vector<float> tmp_in(batchsize_* w_);
    std::vector<float> tmp_out(batchsize_* w_);

    CK_CUDA_THROW_(cudaMemcpy(tmp_in.data(), input_->get_ptr(), input_->get_size(), cudaMemcpyDeviceToHost));
    CK_CUDA_THROW_(cudaMemcpy(tmp_out.data(), output_->get_ptr(), output_->get_size(), cudaMemcpyDeviceToHost));

    //todo compare
    ASSERT_TRUE(test::compare_array_approx<float>(tmp_in.data(), h_in_->data(), h_in_->size(), eps));
    ASSERT_TRUE(test::compare_array_approx<float>(tmp_out.data(), h_out_->data(), h_out_->size(), eps));


    return;
  }
public:
  MultiCrossLayerTest(int batchsize, int w, int layers):
    batchsize_(batchsize), w_(w), layers_(layers),
    blob_buf_(new GeneralBuffer<float>()), weight_buf_(new GeneralBuffer<float>()),
    wgrad_buf_(new GeneralBuffer<float>()), data_sim_(0.0, 1.0, -10.0, 10.0)
  {
    std::vector<int> dim ={batchsize, w};

    //input
    input_.reset(new Tensor<float>(dim, blob_buf_, TensorFormat_t::HW));

    //output
    output_.reset(new Tensor<float>(dim, blob_buf_, TensorFormat_t::HW));

    //layer
    layer_.reset(new MultiCrossLayer(weight_buf_, wgrad_buf_, input_, output_, layers, 0));


    blob_buf_->init(0);
    weight_buf_->init(0);
    wgrad_buf_->init(0);


    h_in_ = std::make_shared<std::vector<float>>(batchsize*w, 0.f);
    h_out_ = std::make_shared<std::vector<float>>(batchsize*w, 0.f);
    h_weight_ = std::make_shared<std::vector<float>>(w*layers*2,0.f);
    return;
  }

  void fprop_test(){
    reset_();
    
    cpu_fprop_();

    gpu_fprop_();

    compare_();

    return; 
  }

  void bprop_test(){
    reset_();

    cpu_bprop_();

    gpu_bprop_();

    compare_();
    
    return; 
  }
};


TEST(multi_cross_layer, 1x1x1024_fprop){
  MultiCrossLayerTest test(1, 1024, 1);
  test.fprop_test();
}

TEST(multi_cross_layer, 1x1x1024_bprop){
  MultiCrossLayerTest test(1, 1024, 1);
  test.bprop_test();
}

TEST(multi_cross_layer, 2x1x1024_bprop){
  MultiCrossLayerTest test(1, 1024, 2);
  test.bprop_test();
}


TEST(multi_cross_layer, 3x1x1024_fprop){
  MultiCrossLayerTest test(1, 1024, 3);
  test.fprop_test();
}

TEST(multi_cross_layer, 3x1x1024_bprop){
  MultiCrossLayerTest test(1, 1024, 3);
  test.bprop_test();
}


TEST(multi_cross_layer, 3x4096x1024_fprop){
  MultiCrossLayerTest test(4096, 1024, 3);
  test.fprop_test();
}


TEST(multi_cross_layer, 3x4096x1024_bprop){
  MultiCrossLayerTest test(4096, 1024, 3);
  test.bprop_test();
}

TEST(multi_cross_layer, 3x40963x356_fprop){
  MultiCrossLayerTest test(40963, 356, 3);
  test.fprop_test();
}

TEST(multi_cross_layer, 3x40963x356_bprop){
  MultiCrossLayerTest test(40963, 356, 3);
  test.bprop_test();
}
