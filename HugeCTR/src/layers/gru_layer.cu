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
#include <algorithm>
#include <common.hpp>
#include <functional>
#include <gpu_resource.hpp>
#include <include/utils.cuh>
#include <layers/element_wise_function.hpp>
#include <layers/gru_layer.hpp>
#include <linalg/matrix_vector_op.cuh>
#include <utils.cuh>
#include <utils.hpp>
#include <vector>

#include <linalg/binary_op.cuh>
#include <linalg/unary_op.cuh>
#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

template <typename T>
GRULayer<T>::GRULayer(const std::shared_ptr<BufferBlock2<T>>& weight_buff,
                      const std::shared_ptr<BufferBlock2<T>>& wgrad_buff,
                      const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor, size_t hiddenSize,
                      size_t batch_size, size_t SeqLength, size_t embedding_vec_size,
                      const std::shared_ptr<GPUResource>& gpu_resource,
                      std::vector<Initializer_t> initializer_types)
    : Layer(gpu_resource, initializer_types) {
  try {
    // check the in_tensor and out_tensor
    const auto& in_tensor_dim = in_tensor.get_dimensions();
    const auto& out_tensor_dim = out_tensor.get_dimensions();

    // 2. dim match?
    // seqLength = in_tensor_dim[1];
    // m = out_tensor_dim[1];
    // miniBatch = in_tensor_dim[0];
    // printf("m %lu n %lu k %lu \n ", m, n,k);
    hiddenSize_ = hiddenSize;
    miniBatch = batch_size;
    seqLength_ = SeqLength;
    embedding_vec_size_ = embedding_vec_size;

    inputTensorSize = miniBatch * seqLength_ * embedding_vec_size_;
    outputTensorSize = miniBatch * seqLength_ * hiddenSize_;
    hiddenTensorSize = miniBatch * hiddenSize_;

    // weightSpaceSize = m*k + m*m + 1*m; //include W, U weight matrixs and bias vector.

    // CK_CUDNN_THROW_(cudnnSetTensor4dDescriptorEx(hDesc, data_type, n, 1, 1, n,
    //  n, 1, 1, 1));

    // CK_CUDNN_THROW_(cudnnSetTensor4dDescriptorEx(cDesc, data_type, 1, n, m, n,
    //  n, 1, 1, 1));
    seqLengthArray = (int*)malloc(miniBatch * sizeof(int));

    for (size_t i = 0; i < miniBatch; i++) {
      seqLengthArray[i] = seqLength_;
    }

    // cudnnHandle= get_gpu().get_cudnn_handle();
    CK_CUDNN_THROW_(cudnnCreate(&cudnnHandle));
    data_type = CudnnDataType<T>::getType();
    CK_CUDNN_THROW_(cudnnCreateRNNDescriptor(&rnnDesc));
    CK_CUDNN_THROW_(cudnnCreateRNNDataDescriptor(&in_Desc));
    CK_CUDNN_THROW_(cudnnCreateRNNDataDescriptor(&out_Desc));
    CK_CUDNN_THROW_(cudnnCreateTensorDescriptor(&cDesc));
    CK_CUDNN_THROW_(cudnnCreateTensorDescriptor(&hDesc));
    CK_CUDNN_THROW_(cudnnCreateDropoutDescriptor(&dropoutDesc));

    CK_CUDNN_THROW_(cudnnSetRNNDataDescriptor(
        in_Desc,                                   // cudnnRNNDataDescriptor_t RNNDataDesc,
        data_type,                                 // cudnnDataType_t dataType,
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,  // CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                                                   // //cudnnRNNDataLayout_t layout,
        seqLength_,           // int maxSeqLength,
        miniBatch,            // int batchSize,
        embedding_vec_size_,  // int vectorSize,
        seqLengthArray,       // const int seqLengthArray[],
        NULL                  // void *paddingFill
        ));

    CK_CUDNN_THROW_(cudnnSetRNNDataDescriptor(
        out_Desc,                                  // cudnnRNNDataDescriptor_t RNNDataDesc,
        data_type,                                 // cudnnDataType_t dataType,
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,  // CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                                                   // //cudnnRNNDataLayout_t layout,
        seqLength_,      // int maxSeqLength,
        miniBatch,       // int batchSize,
        hiddenSize_,     // int vectorSize,
        seqLengthArray,  // const int seqLengthArray[],
        NULL             // void *paddingFill
        ));
    dimHidden[0] = 1 * 1;
    dimHidden[1] = miniBatch;
    dimHidden[2] = hiddenSize_;
    strideHidden[0] = dimHidden[1] * dimHidden[2];
    strideHidden[1] = dimHidden[2];
    strideHidden[2] = 1;
    CK_CUDNN_THROW_(cudnnSetTensorNdDescriptor(hDesc, data_type, 3, dimHidden, strideHidden));
    CK_CUDNN_THROW_(cudnnSetTensorNdDescriptor(cDesc, data_type, 3, dimHidden, strideHidden));

    CK_CUDNN_THROW_(cudnnDropoutGetStatesSize(cudnnHandle, &stateSize));
    cudaMalloc(&states, stateSize);
    seed = 0;  // 1337ull;
    CK_CUDNN_THROW_(
        cudnnSetDropoutDescriptor(dropoutDesc, cudnnHandle, dropout, states, stateSize, seed));

    CK_CUDNN_THROW_(cudnnSetRNNDescriptor_v8(
        rnnDesc,
        CUDNN_RNN_ALGO_STANDARD,    // cudnnRNNAlgo_t algo,
        CUDNN_GRU,                  // cudnnRNNMode_t cellMode,
        CUDNN_RNN_SINGLE_INP_BIAS,  // cudnnRNNBiasMode_t biasMode,
        CUDNN_UNIDIRECTIONAL,       // cudnnDirectionMode_t dirMode,
        CUDNN_LINEAR_INPUT,         // CUDNN_SKIP_INPUT, //CUDNN_LINEAR_INPUT, //cudnnRNNInputMode_t
                             // inputMode, CUDNN_SKIP_INPUT :without multiplying input by the weight
                             // matrix
        data_type,             // cudnnDataType_t dataType,
        data_type,             // cudnnDataType_t mathPrec,
        CUDNN_TENSOR_OP_MATH,  // CUDNN_DEFAULT_MATH , //cudnnMathType_t mathType,
        embedding_vec_size_,   // int32_t embedding_vec_size, When the inputMode=CUDNN_SKIP_INPUT,
                              // the embedding_vec_size should match the hiddenSize value
        hiddenSize_,                  // int32_t hiddenSize,
        hiddenSize_,                  // int32_t projSize,
        1,                            // int32_t numLayers, BIDIRECTIONAL=2
        dropoutDesc,                  // cudnnDropoutDescriptor_t dropoutDesc,
        CUDNN_RNN_PADDED_IO_DISABLED  // uint32_t auxFlags
        ));

    // const int seqLengthArray[in_tensor_dim[0]] = { [0...10] = int(in_tensor_dim[1]) };
    // const int seqLengthArray[m] ={n,n....n};
    // for(int i=0; i<in_tensor_dim[1]; i++)
    // = { [0 . . . 3 ] = 3 };

    CK_CUDNN_THROW_(cudnnGetRNNWeightSpaceSize(cudnnHandle, rnnDesc, &weightSpaceSize));
    CK_CUDNN_THROW_(cudnnGetRNNTempSpaceSizes(cudnnHandle, rnnDesc, CUDNN_FWD_MODE_TRAINING,
                                              in_Desc, &workSpaceSize, &reserveSpaceSize));
    // std::vector<size_t> weight_dim = {weightSpaceSize/sizeof(T), 1};
    // std::vector<size_t> dx_dim =  {inputTensorSize, 1};
    // std::vector<size_t> dy_dim =  {outputTensorSize, 1};
    // std::vector<size_t> dhx_dim = {hiddenTensorSize, 1};
    // std::vector<size_t> dhy_dim = {hiddenTensorSize, 1};
    // std::vector<size_t> dcx_dim = {hiddenTensorSize, 1};
    // std::vector<size_t> dcy_dim = {hiddenTensorSize, 1};

    std::vector<size_t> weight_dim = {1, weightSpaceSize / sizeof(T)};
    std::vector<size_t> hx_dim = {1, hiddenTensorSize};
    std::vector<size_t> dx_dim = {1, inputTensorSize};
    std::vector<size_t> dy_dim = {1, outputTensorSize};
    std::vector<size_t> dhx_dim = {1, hiddenTensorSize};
    std::vector<size_t> dhy_dim = {1, hiddenTensorSize};
    std::vector<size_t> dweigths_dim = {1, weightSpaceSize / sizeof(T)};
    // printf("weighsize %zu\n", weightSpaceSize/sizeof(T));

    {
      Tensor2<T> tensor;
      weight_buff->reserve(weight_dim, &tensor);
      weights_.push_back(tensor);
    }

    {
      Tensor2<T> tensor;
      weight_buff->reserve(hx_dim, &tensor);
      weights_.push_back(tensor);
    }

    {
      Tensor2<T> tensor;
      wgrad_buff->reserve(dx_dim, &tensor);
      wgrad_.push_back(tensor);
    }
    {
      Tensor2<T> tensor;
      wgrad_buff->reserve(dy_dim, &tensor);
      wgrad_.push_back(tensor);
    }
    {
      Tensor2<T> tensor;
      wgrad_buff->reserve(dhx_dim, &tensor);
      wgrad_.push_back(tensor);
    }
    {
      Tensor2<T> tensor;
      wgrad_buff->reserve(dhy_dim, &tensor);
      wgrad_.push_back(tensor);
    }
    {
      Tensor2<T> tensor;
      wgrad_buff->reserve(dweigths_dim, &tensor);
      wgrad_.push_back(tensor);
    }

    CK_CUDA_THROW_(cudaMalloc((void**)&devSeqLengthArray, miniBatch * sizeof(int)));
    CK_CUDA_THROW_(cudaMemcpy(devSeqLengthArray, seqLengthArray, miniBatch * sizeof(int),
                              cudaMemcpyHostToDevice));
    CK_CUDA_THROW_(cudaMalloc((void**)&weightSpace, weightSpaceSize));
    CK_CUDA_THROW_(cudaMalloc((void**)&workSpace, workSpaceSize));
    CK_CUDA_THROW_(cudaMalloc((void**)&reserveSpace, reserveSpaceSize));
    // CK_CUDA_THROW_(cudaMalloc((void **)&dweightSpace, weightSpaceSize));

    in_tensors_.push_back(in_tensor);
    out_tensors_.push_back(out_tensor);
    // Where should we create this cuBLAS handle?
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

//#define KERAS_CHECK
template <typename T>
void GRULayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  Tensor2<T>& in_tensor = get_in_tensors(is_train)[0];
  Tensor2<T>& out_tensor = out_tensors_[0];

  T* weight = weights_[0].get_ptr();
  T* hx = weights_[1].get_ptr();
  // T* Uweight = weights_[1].get_ptr();
  // T* bias = weights_[2].get_ptr();

  T* in = in_tensor.get_ptr();
  T* out = out_tensor.get_ptr();
// T* hx = weights_[0].get_ptr();
// printf("datatype %lu\n", sizeof(data_type));
// CK_CUDA_THROW_(cudaMalloc((void **)&in,  inputTensorSize * sizeof(T)));

// CK_CUBLAS_THROW_(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
//                            &alpha, weight, CUDA_R_32F, n, in, CUDA_R_32F, k, &beta, out,
//                            CUDA_R_32F, n, CUDA_R_32F, falgo_));
#ifdef KERAS_CHECK
  cudnnTensorDescriptor_t wDesc;
  cudnnTensorDescriptor_t bDesc;
  CK_CUDNN_THROW_(cudnnCreateTensorDescriptor(&wDesc));
  CK_CUDNN_THROW_(cudnnCreateTensorDescriptor(&bDesc));

  // Tensor2<T> linLayerMat;
  // Tensor2<T> linLayerBias;
  numLinearLayers = 6;  // cellMode == CUDNN_GRU
  for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
    T* linLayerMat = NULL;
    T* linLayerBias = NULL;
    int nbDims = 0;
    int dim[3] = {0, 0, 0}, stride[3];
    int layer = 0;
    // printf("weightSpaceSize %zu\n", weightSpaceSize);
    CK_CUDNN_THROW_(cudnnGetRNNWeightParams(cudnnHandle, rnnDesc, layer, weightSpaceSize,
                                            weights_[0].get_ptr(),  // weightSpace,
                                            linLayerID, wDesc,
                                            (void**)&linLayerMat,  //.get_ptr(),
                                            bDesc,
                                            (void**)&linLayerBias  //.get_ptr()
                                            ));

    if (linLayerMat) {
      CK_CUDNN_THROW_(cudnnGetTensorNdDescriptor(wDesc, 3, &data_type, &nbDims, dim, stride));
      size_t w = dim[0] * dim[1] * dim[2];
      T* h_weights = new T[w];
      cudaMemcpy(h_weights, linLayerMat, sizeof(T) * w, cudaMemcpyDeviceToHost);

      printf("W_%d %zu ", linLayerID, w);
      for (unsigned int i = 0; i < w; i++) {
        printf("%f ", h_weights[i]);
      }
      printf("\n");

      delete (h_weights);
    }

    if (linLayerBias) {
      CK_CUDNN_THROW_(cudnnGetTensorNdDescriptor(bDesc, 3, &data_type, &nbDims, dim, stride));
      size_t w = dim[0] * dim[1] * dim[2];
      T* h_weights = new T[w];
      cudaMemcpy(h_weights, linLayerBias, sizeof(T) * w, cudaMemcpyDeviceToHost);

      printf("B_%d %zu ", linLayerID, w);
      for (unsigned int i = 0; i < w; i++) {
        printf("%f ", h_weights[i]);
      }
      printf("\n");

      delete (h_weights);
    }
  }

  cudnnDestroyTensorDescriptor(wDesc);
  cudnnDestroyTensorDescriptor(bDesc);
#endif
  // CUDNN GRU
  // T tmp[hiddenTensorSize];
  // CK_CUDA_THROW_(cudaMemcpy(tmp, weight + weightSpaceSize/sizeof(T), sizeof(T) *
  // hiddenTensorSize, cudaMemcpyDeviceToHost)); for(size_t i=0;i<hiddenTensorSize;i++)
  //  if(tmp[i] != 0.0)
  //    printf("tmp[i] %f\n", tmp[i]);
  CK_CUDNN_THROW_(cudnnRNNForward(
      cudnnHandle, rnnDesc, CUDNN_FWD_MODE_TRAINING, devSeqLengthArray,
      in_Desc,   // xDesc,
      in,        // x, input data pointer
      out_Desc,  // yDesc,
      out,       // y, output data pointer
      hDesc,
      NULL,  // hx, Input. Pointer to the GPU buffer with the RNN initial hidden state, NULL:
             // initialized zero.
      NULL,  // hy,  Output. Pointer to the GPU buffer where the final RNN hidden state should be
             // stored. NULL: not saved.
      cDesc,  // cDesc, Input. A tensor descriptor, for LSTM networks only.
      NULL,   // cx,
      NULL,   // cy,
      weightSpaceSize,
      weight,  // weightSpace, The weight space buffer holds all RNN weight matrices and bias
               // vectors
      workSpaceSize,  // size_t workSpaceSize,
      workSpace,      // workSpace,
      reserveSpaceSize,
      reserveSpace  // reserveSpace
      ));

  // printf("forward end\n\n");
  // cudnnDestroy(cudnnHandle);
}

template <typename T>
void GRULayer<T>::bprop() {
  Tensor2<T>& in_tensor = get_in_tensors(true)[0];
  Tensor2<T>& out_tensor = out_tensors_[0];

  T* weight = weights_[0].get_ptr();
  T* in = in_tensor.get_ptr();
  T* out = out_tensor.get_ptr();
  T* dx = wgrad_[0].get_ptr();
  T* dy = wgrad_[1].get_ptr();
  T* dhx = wgrad_[2].get_ptr();
  T* dhy = wgrad_[3].get_ptr();
  T* dweightSpace = wgrad_[4].get_ptr();

  CK_CUDNN_THROW_(cudnnRNNBackwardData_v8(cudnnHandle,        // cudnnHandle_t handle,
                                          rnnDesc,            // cudnnRNNDescriptor_t rnnDesc,
                                          devSeqLengthArray,  // const int32_t devSeqLengths[],
                                          out_Desc,           // cudnnRNNDataDescriptor_t yDesc,
                                          out,                // const void *y, input
                                          dy,                 // const void *dy, input
                                          in_Desc,            // cudnnRNNDataDescriptor_t xDesc,
                                          dx,                 // void *dx, output
                                          hDesc,              // cudnnTensorDescriptor_t hDesc,
                                          NULL,               // hx, //const void *hx, input
                                          NULL,               // const void *dhy, input
                                          dhx,                // void *dhx, output
                                          cDesc,              // cudnnTensorDescriptor_t cDesc,
                                          NULL,  // cx, //const void *cx, for LSTM only, input
                                          NULL,  // const void *dcy, for LSTM only, input
                                          NULL,  // void *dcx, output
                                          weightSpaceSize,
                                          weight,  // weightSpace,
                                          workSpaceSize, workSpace, reserveSpaceSize,
                                          reserveSpace));

  // cudnnRNNBackwardWeights adds to the data in dw.
  CK_CUDA_THROW_(cudaMemset(dweightSpace, 0, weightSpaceSize));
  // T* h_hx=NULL;
  // cudaMemcpy(h_hx,hx,hiddenTensorSize*sizeof(T),cudaMemcpyDeviceToHost );
  // for(unsigned int i=0;i<hiddenTensorSize;i++)
  //    printf("hx %f \n",h_hx[i]);

  CK_CUDNN_THROW_(cudnnRNNBackwardWeights_v8(
      cudnnHandle, rnnDesc, CUDNN_WGRAD_MODE_ADD, devSeqLengthArray, in_Desc, in, hDesc,
      NULL,  // hx,
      out_Desc,
      out,  // output
      weightSpaceSize,
      dweightSpace,  // output
      workSpaceSize, workSpace, reserveSpaceSize, reserveSpace));

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif

  cudaFree(workSpace);
  cudaFree(reserveSpace);
  cudaFree(weightSpace);  // cudaFree(dweightSpace);
  // cudaFree(x); cudaFree(y); cudaFree(hx);
  cudaFree(states);
  cudnnDestroyRNNDataDescriptor(in_Desc);
  cudnnDestroyRNNDataDescriptor(out_Desc);

  cudnnDestroyTensorDescriptor(hDesc);
  cudnnDestroyTensorDescriptor(cDesc);

  cudnnDestroyDropoutDescriptor(dropoutDesc);
  cudnnDestroyRNNDescriptor(rnnDesc);
}

template class GRULayer<float>;
// template class GRULayer<__half>;

}  // namespace HugeCTR
