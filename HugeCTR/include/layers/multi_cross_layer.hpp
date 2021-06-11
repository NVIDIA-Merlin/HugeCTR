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

template <typename T>
struct MultiCrossForwardFunctor {
  MultiCrossForwardFunctor() = default;
  MultiCrossForwardFunctor(const MultiCrossForwardFunctor&) = delete;
  MultiCrossForwardFunctor& operator=(const MultiCrossForwardFunctor&) = delete;

  void operator()(cudaStream_t stream, cublasHandle_t cublas_handle,
                  const Tensor2<T>& input_tensor, const Tensors2<T>& kernel_tensors,
                  const Tensors2<T>& bias_tensors, Tensors2<T>& layer_output_tensors,
                  Tensors2<T>& layer_hidden_tensors, int num_layers) const;
};

template <typename T>
struct MultiCrossBackwardFunctor {
  MultiCrossBackwardFunctor() = default;
  MultiCrossBackwardFunctor(const MultiCrossBackwardFunctor&) = delete;
  MultiCrossBackwardFunctor& operator=(const MultiCrossBackwardFunctor&) = delete;

  void operator()(cudaStream_t stream, const Tensor2<T>& input_tensor,
                  const Tensors2<T>& kernel_tensors,
                  const Tensors2<T>& layer_output_tensors,
                  const Tensors2<T>& layer_hidden_tensors, const Tensor2<T>& grad_tensor,
                  Tensor2<T>& output_tensor, Tensors2<T>& kernel_output_tensors,
                  Tensors2<T>& bias_output_tensors, Tensor2<T>& tmp_vec_tensor,
                  Tensor2<T> tmp_mat_tensors[], int num_layers) const;
};

template <typename T>
class MultiCrossLayer : public Layer {
 private:
  const int num_layers_;
  Tensors2<T> blob_tensors_; /**< vector of internal blobs' tensors */
  Tensors2<T> vec_tensors_;  //[h,1]

  Tensor2<T> tmp_mat_tensors_[3];  //[h,w]
  Tensor2<T> tmp_vec_tensor_;      //[h,1]

  /*
   * stores the weight tensors of this layer.
   */
  Tensors2<T> weights_;
  /*
   * stores the weight gradient tensors of this layer.
   */
  Tensors2<T> wgrad_;
  /*
   * stores the references to the input tensors of this layer.
   */
  Tensors2<T> in_tensors_;
  /*
   * stores the references to the output tensors of this layer.
   */
  Tensors2<T> out_tensors_;

 public:
  /**
   * forward pass
   */
  void fprop(bool is_train) final;
  /**
   * backward pass
   */
  void bprop() final;

  MultiCrossLayer(const std::shared_ptr<BufferBlock2<T>>& weight_buff,
                  const std::shared_ptr<BufferBlock2<T>>& wgrad_buff,
                  const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                  const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor,
                  const std::shared_ptr<GPUResource>& gpu_resource, int num_layers,
                  std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>());
  MultiCrossLayer(const MultiCrossLayer&) = delete;
  MultiCrossLayer& operator=(const MultiCrossLayer&) = delete;

 private:
  std::unique_ptr<DataSimulator> get_default_initializer(const int index) override;
};
}  // namespace HugeCTR
