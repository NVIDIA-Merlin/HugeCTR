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

#pragma once

#include <functional>
#include <layer.hpp>
#include <vector>

namespace HugeCTR {

struct MultiCrossForwardFunctor {
  MultiCrossForwardFunctor() = default;
  MultiCrossForwardFunctor(const MultiCrossForwardFunctor&) = delete;
  MultiCrossForwardFunctor& operator=(const MultiCrossForwardFunctor&) = delete;

  void operator()(cudaStream_t stream, cublasHandle_t cublas_handle,
                  const Tensor2<float>& input_tensor, const Tensors2<float>& kernel_tensors,
                  const Tensors2<float>& bias_tensors, Tensors2<float>& layer_output_tensors,
                  Tensors2<float>& layer_hidden_tensors, int num_layers) const;
};

struct MultiCrossBackwardFunctor {
  MultiCrossBackwardFunctor() = default;
  MultiCrossBackwardFunctor(const MultiCrossBackwardFunctor&) = delete;
  MultiCrossBackwardFunctor& operator=(const MultiCrossBackwardFunctor&) = delete;

  void operator()(cudaStream_t stream, const Tensor2<float>& input_tensor,
                  const Tensors2<float>& kernel_tensors,
                  const Tensors2<float>& layer_output_tensors,
                  const Tensors2<float>& layer_hidden_tensors, const Tensor2<float>& grad_tensor,
                  Tensor2<float>& output_tensor, Tensors2<float>& kernel_output_tensors,
                  Tensors2<float>& bias_output_tensors, Tensor2<float>& tmp_vec_tensor,
                  Tensor2<float> tmp_mat_tensors[], int num_layers) const;
};

class MultiCrossLayer : public Layer {
 private:
  const int num_layers_;
  Tensors2<float> blob_tensors_; /**< vector of internal blobs' tensors */
  Tensors2<float> vec_tensors_;  //[h,1]

  Tensor2<float> tmp_mat_tensors_[3];  //[h,w]
  Tensor2<float> tmp_vec_tensor_;      //[h,1]

  /*
   * stores the weight tensors of this layer.
   */
  // Tensors<float> weights_; It is inherited from Layer, and named as weights_;
  /*
   * stores the weight gradient tensors of this layer.
   */
  Tensors2<float> wgrad_;
  /*
   * stores the references to the input tensors of this layer.
   */
  Tensors2<float> in_tensors_;
  /*
   * stores the references to the output tensors of this layer.
   */
  Tensors2<float> out_tensors_;

  const cublasHandle_t cublas_handle_;  // cublas handle
 public:
  /**
   * forward pass
   */
  void fprop(bool is_train, cudaStream_t stream) final;
  /**
   * backward pass
   */
  void bprop(cudaStream_t stream) final;

  MultiCrossLayer(const std::shared_ptr<BufferBlock2<float>>& weight_buff,
                  const std::shared_ptr<BufferBlock2<float>>& wgrad_buff,
                  const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                  const Tensor2<float>& in_tensor, const Tensor2<float>& out_tensor,
                  cublasHandle_t const& cublas_handle, int num_layers, int device_id,
                  std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>());
  MultiCrossLayer(const MultiCrossLayer&) = delete;
  MultiCrossLayer& operator=(const MultiCrossLayer&) = delete;

 private:
  std::unique_ptr<DataSimulator<float>> get_default_initializer(const int index) override;
};
}  // namespace HugeCTR
