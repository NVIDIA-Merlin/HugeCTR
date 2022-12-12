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

#pragma once

#include <functional>
#include <layers/functors/fused_fc_layer_functors.hpp>
#include <trainable_layer.hpp>
#include <vector>
namespace HugeCTR {

template <typename T>
struct MultiCrossForwardFunctor {
  MultiCrossForwardFunctor() = default;
  MultiCrossForwardFunctor(const MultiCrossForwardFunctor&) = delete;
  MultiCrossForwardFunctor& operator=(const MultiCrossForwardFunctor&) = delete;

  void operator()(cudaStream_t stream, cublasHandle_t cublas_handle, const Tensor2<T>& input_tensor,
                  const Tensors2<T>& kernel_tensors, const Tensors2<T>& bias_tensors,
                  Tensors2<T>& layer_output_tensors, Tensors2<T>& layer_hidden_tensors,
                  int num_layers) const;
};
template <typename T>
struct MultiCrossForwardFunctorv2 {
  GemmFunctor<T> gemm_functor_;
  MultiCrossForwardFunctorv2() = default;
  MultiCrossForwardFunctorv2(const MultiCrossForwardFunctorv2&) = delete;
  MultiCrossForwardFunctorv2& operator=(const MultiCrossForwardFunctorv2&) = delete;
  void search_algorithm(T* bottom, T* top, T* kernel, size_t batch_size, size_t input_size,
                        size_t output_size, const CublasFusedFCLayerDesc<T>& cublas_layer_desc,
                        cublasLtHandle_t cublaslt_handle, cudaStream_t stream);
  void operator()(cudaStream_t stream, const Tensor2<T>& input_tensor,
                  const Tensors2<T>& kernel_tensors, const Tensors2<T>& bias_tensors,
                  Tensors2<T>& XU_tensors, Tensors2<T>& layer_output_tensors,
                  Tensors2<T>& layer_hidden_tensors, int num_layers,
                  const std::vector<CublasDesc<T>>& xu_descr_,
                  const std::vector<CublasDesc<T>>& xuvb_descr_,
                  const std::vector<CublasAlgo<T>>& xu_fprop_algo_,
                  const std::vector<CublasAlgo<T>>& xuvb_fprop_algo_, cublasLtHandle_t = nullptr);
};

template <typename T>
struct MultiCrossBackwardFunctorv2 {
  GemmFunctor<T> gemm_functor_;

  MultiCrossBackwardFunctorv2() = default;
  MultiCrossBackwardFunctorv2(const MultiCrossBackwardFunctorv2&) = delete;
  MultiCrossBackwardFunctorv2& operator=(const MultiCrossBackwardFunctorv2&) = delete;
  void operator()(cudaStream_t stream, const Tensor2<T>& input_tensor,
                  const Tensors2<T>& kernel_tensors, const Tensors2<T>& layer_output_tensors,
                  const Tensors2<T>& layer_hidden_tensors, const Tensor2<T>& grad_tensor,
                  Tensor2<T>& output_tensor, Tensors2<T>& kernel_output_tensors,
                  Tensors2<T>& bias_output_tensors, Tensors2<T>& XU_tensors,
                  Tensor2<T> tmp_mat_tensors[], int num_layers,
                  const std::vector<CublasDesc<T>>& xu_descr_,
                  const std::vector<CublasDesc<T>>& xuvb_descr_,
                  const std::vector<CublasDesc<T>>& du_descrs_bprop_,
                  const std::vector<CublasDesc<T>>& dhidden_descrs_bprop_,
                  const std::vector<CublasAlgo<T>>& xu_bprop_algo_,
                  const std::vector<CublasAlgo<T>>& xuvb_bprop_algo_,
                  const std::vector<CublasAlgo<T>>& du_bprop_algos_,
                  const std::vector<CublasAlgo<T>>& dhidden_bprop_algos_,
                  cublasLtHandle_t = nullptr);
};

template <typename T>
struct MultiCrossBackwardFunctor {
  MultiCrossBackwardFunctor() = default;
  MultiCrossBackwardFunctor(const MultiCrossBackwardFunctor&) = delete;
  MultiCrossBackwardFunctor& operator=(const MultiCrossBackwardFunctor&) = delete;

  void operator()(cudaStream_t stream, const Tensor2<T>& input_tensor,
                  const Tensors2<T>& kernel_tensors, const Tensors2<T>& layer_output_tensors,
                  const Tensors2<T>& layer_hidden_tensors, const Tensor2<T>& grad_tensor,
                  Tensor2<T>& output_tensor, Tensors2<T>& kernel_output_tensors,
                  Tensors2<T>& bias_output_tensors, Tensor2<T>& tmp_vec_tensor,
                  Tensor2<T> tmp_mat_tensors[], int num_layers) const;
};

template <typename T>
class MultiCrossLayer : public TrainableLayer<T> {
 private:
  const int num_layers_;
  const size_t projection_dim_;
  Tensors2<T> blob_tensors_;    /**< vector of internal blobs' tensors, intermediate output of each
                                   interaction layer: T_4 */
  Tensors2<T> hidden_tensors_;  // DCNv1: x_i * w ; DCNv2: x * x_i * w + b; T_7
  Tensors2<T> XU_tensors_;      // DCNv2:

  Tensor2<T> tmp_mat_tensors_[4];  //[h,w]
  Tensor2<T> tmp_vec_tensor_;      //[h,1]

  /*
   * stores the references to the input tensors of this layer.
   */
  Tensors2<T> in_tensors_;
  /*
   * stores the references to the output tensors of this layer.
   */
  Tensors2<T> out_tensors_;

  std::vector<CublasDesc<T>> xu_descrs_fprop_;
  std::vector<CublasDesc<T>> xuvb_descrs_fprop_;
  std::vector<CublasDesc<T>> xu_descrs_bprop_;
  std::vector<CublasDesc<T>> xuvb_descrs_bprop_;
  std::vector<CublasDesc<T>> du_descrs_bprop_;
  std::vector<CublasDesc<T>> dhidden_descrs_bprop_;

  std::vector<CublasAlgo<T>> xu_fprop_algos_;
  std::vector<CublasAlgo<T>> xuvb_fprop_algos_;
  std::vector<CublasAlgo<T>> xu_bprop_algos_;
  std::vector<CublasAlgo<T>> xuvb_bprop_algos_;
  std::vector<CublasAlgo<T>> du_bprop_algos_;
  std::vector<CublasAlgo<T>> dhidden_bprop_algos_;

  MultiCrossForwardFunctorv2<T> dcnv2_forward_functor_;
  MultiCrossBackwardFunctorv2<T> dcnv2_backward_functor_;
  bool enable_tf32_compute_;

 public:
  /**
   * forward pass
   */
  void fprop(bool is_train) final;
  Tensors2<T>& get_hidden_tensors() { return hidden_tensors_; };
  Tensors2<T>& get_weight_tensor() { return XU_tensors_; };
  /**
   * backward pass
   */
  void search_algorithm() override;
  void bprop() final;
  void initialize() override;
  MultiCrossLayer(const std::shared_ptr<BufferBlock2<float>>& master_weight_buff,
                  const std::shared_ptr<BufferBlock2<T>>& weight_buff,
                  const std::shared_ptr<BufferBlock2<T>>& wgrad_buff,
                  const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                  const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor,
                  const std::shared_ptr<GPUResource>& gpu_resource, int num_layers,
                  size_t projection_dim = 0,
                  std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>(),
                  bool enable_tf32_compute = false);
  MultiCrossLayer(const MultiCrossLayer&) = delete;
  MultiCrossLayer& operator=(const MultiCrossLayer&) = delete;

 private:
  std::unique_ptr<DataSimulator> get_default_initializer(const int index) override;
};
}  // namespace HugeCTR
