/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudnn.h>

#include <trainable_layer.hpp>

namespace HugeCTR {

/**
 * GRU function (Interest Extractor Layer) as a derived class of Layer
 */
template <typename T>
class GRULayer : public TrainableLayer<T> {
  cublasGemmAlgo_t falgo_{CUBLAS_GEMM_DEFAULT};

  size_t workSpaceSize;
  size_t reserveSpaceSize;
  size_t inputTensorSize, outputTensorSize, hiddenTensorSize;

  std::vector<core23::Tensor> &get_in_tensors(bool is_train) { return this->input_tensors_; }

 public:
  /**
   * A method of implementing the forward pass of GRU
   * @param stream CUDA stream where the forward propagation is executed
   */
  void fprop(bool is_train) final;
  /**
   * A method of implementing the backward pass of GRU
   * @param stream CUDA stream where the backward propagation is executed
   */
  void bprop() final;

  /**
   * Ctor of GRULayer.
   * @param in_tensor the input tensor
   * @param out_tensor the output tensor which has the same dim with in_tensor
   * @param device_id the id of GPU where this layer belongs
   */
  GRULayer(const core23::Tensor &in_tensor, const core23::Tensor &out_tensor, int64_t hiddenSize,
           int64_t batch_size, int64_t SeqLength, int64_t embedding_vec_size,
           const std::shared_ptr<GPUResource> &gpu_resource,
           std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>());

 private:
  int *seqLengthArray = nullptr;
  int *devSeqLengthArray = nullptr;
  void *weightSpace = nullptr;
  void *dweightSpace = nullptr;
  void *workSpace = nullptr;
  void *reserveSpace = nullptr;
  void *hx = nullptr;

  cudnnHandle_t cudnnHandle;
  cudnnRNNDescriptor_t rnnDesc;
  cudnnRNNDataDescriptor_t in_Desc;
  cudnnRNNDataDescriptor_t out_Desc;
  cudnnTensorDescriptor_t cDesc;
  cudnnTensorDescriptor_t hDesc;
  cudnnDropoutDescriptor_t dropoutDesc;
  cudnnDataType_t data_type;

  int dimHidden[3];
  int strideHidden[3];
  unsigned long long seed;
  size_t stateSize;
  void *states;
  float dropout = 0;
  size_t weightSpaceSize;
  size_t seqLength_, miniBatch, embedding_vec_size_, m = 512;
  int hiddenSize_;  // = 512; //half of the seqLength
  int numLinearLayers;
};

}  // namespace HugeCTR
