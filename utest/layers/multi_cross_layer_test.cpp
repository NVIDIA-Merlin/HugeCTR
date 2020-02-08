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
  std::vector<float> h_in_;
  std::vector<float> h_out_;

  void reset_(){
    for(auto& a: h_in_){
      a = data_sim_.get_num();
    }
    for(auto& b: h_out_){
      b = data_sim_.get_num();
    }
    CK_CUDA_THROW_(cudaMemcpy(input_->get_ptr(), h_in_.data(), input_->get_size(), cudaMemcpyHostToDevice));
    CK_CUDA_THROW_(cudaMemcpy(output_->get_ptr(), h_out_.data(), output_->get_size(), cudaMemcpyHostToDevice));
    weight_buf_->reset_sync();
    wgrad_buf_->reset_sync();

    return; 
  }

  void cpu_fprop_(){

    return;
  }

  void gpu_fprop_(){
    layer_->fprop(cudaStreamDefault);
    return;
  }

  void cpu_bprop_(){
    
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
    
    return;
  }
public:
  MultiCrossLayerTest(int batchsize, int w, int layers):
    batchsize_(batchsize), w_(w), layers_(layers),
    blob_buf_(new GeneralBuffer<float>()), weight_buf_(new GeneralBuffer<float>()),
    wgrad_buf_(new GeneralBuffer<float>()), data_sim_(0.0, 1.0, -10.0, 10.0),
    h_in_(batchsize*w, 0.f), h_out_(batchsize*w, 0.f)
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


TEST(multi_cross_layer, 3x4096x1024){
  MultiCrossLayerTest test(4096, 1024, 3);
  test.fprop_test();
  test.bprop_test();
}
