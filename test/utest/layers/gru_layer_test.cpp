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

#include "HugeCTR/include/layers/gru_layer.hpp"
//#include "rnn_example/cudnn_rnn_v6.hpp"
//#include "rnn_example/data.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <vector>

#include "HugeCTR/include/data_simulator.hpp"
#include "utest/test_utils.h"

using namespace HugeCTR;

namespace {

//#define KERAS_CHECK
template <typename T>
static bool check_cpu_gpu(T *cpu_p, T *gpu_p, size_t len) {
  T *cpu_tmp = (T *)malloc(sizeof(T) * len);
  HCTR_LIB_THROW(cudaMemcpy(cpu_tmp, gpu_p, sizeof(T) * len, cudaMemcpyDeviceToHost));
  T max_diff = fabs(cpu_p[0] - cpu_tmp[0]);
  bool flag = true;
  int start_pos = 0;
  for (unsigned int i = 0; i < len; ++i) {
    if (fabs(cpu_p[i] - cpu_tmp[i]) >= 1e-3 && fabs((cpu_p[i] - cpu_tmp[i]) / cpu_p[i]) >= 1e-3) {
      if (flag) start_pos = i;
      flag = false;
      // if(fabs(cpu_p[i] - cpu_tmp[i]) >= 0.03){
      //  HCTR_LOG(INFO, WORLD, "wrong at %d %.32f %.32f\n",i, cpu_p[i], cpu_tmp[i]);
      //  //break;
      //}
    }
    max_diff = max(max_diff, fabs(cpu_p[i] - cpu_tmp[i]));
  }
  // if (!flag) HCTR_LOG(INFO, WORLD, "max_diff %f, start at %d\n", max_diff, start_pos);
  free(cpu_tmp);
  return flag;
}

bool check_correctness(float *a, float *b, int len, float error) {
  float max_diff = fabs(a[0] - b[0]);
  bool flag = true;
  // int start_pos = 0;
  int count = 0;
  for (int i = 0; i < len; ++i) {
    if (fabs(a[i] - b[i]) >= error && fabs((a[i] - b[i]) / a[i]) >= error) {
      // if (flag) start_pos = i;
      flag = false;
      count++;
      // if(fabs(a[i] - b[i]) >= 0.1)
      // HCTR_LOG(INFO, WORLD, "i %d %f %f\n", i, a[i], b[i]);
    }
    max_diff = std::max(max_diff, fabs(a[i] - b[i]));
  }
  // HCTR_LOG(INFO, WORLD, "number %d\n", count);
  // if (!flag) HCTR_LOG(INFO, WORLD, "Fail matched max_diff %f, start at %d\n", max_diff,
  // start_pos);
  return flag;
}

template <typename T>
static void cpu_mm(T *a, T *b, T *c, size_t m, size_t k, size_t n) {
  for (unsigned int i = 0; i < m; ++i) {
    for (unsigned int j = 0; j < n; ++j) {
      c[i * n + j] = 0.0f;
      for (unsigned int kk = 0; kk < k; ++kk) c[i * n + j] += a[i * k + kk] * b[kk * n + j];
    }
  }
}

template <typename T>
static void cpu_add_bias(T *out, T *bias, size_t h) {
  for (unsigned int i = 0; i < h; ++i) {
    out[i] += bias[i];
  }
}

template <typename T>
static void cpu_dotm(T *a, T *b, size_t h) {
  for (unsigned int i = 0; i < h; ++i) {
    a[i] *= b[i];
  }
}

template <typename T>
void cpu_tanh(T *a, size_t h) {
  for (unsigned int i = 0; i < h; ++i) {
    a[i] = tanh(a[i]);
  }
}

template <typename T>
void cpu_fast_sigmoid(T *a, size_t h) {
  for (unsigned int i = 0; i < h; ++i) {
    a[i] = a[i] / (1 + abs(a[i]));
  }
}

template <typename T>
void cpu_sigmoid(T *a, size_t h) {
  for (unsigned int i = 0; i < h; ++i) {
    a[i] = 1 / (1 + exp((double)-a[i]));
  }
}

template <typename T>
void cpu_ht(T *It, T *h1_t, T *ht_1, T *ht, size_t h) {
  // ht = (1 - it) ◦ h't + it ◦ ht-1
  for (unsigned int i = 0; i < h; ++i) {
    ht[i] = (1 - It[i]) * h1_t[i] + It[i] * ht_1[i];
  }
}

template <typename T>
void cpu_gru(T *weight, T *in, T *hx, T *out, size_t h, size_t v, size_t b, size_t s) {
  T *It = new T[h * 2];
  T *Rt = new T[h * 2];
  T *h1_t = new T[h * 2];
  memset(It, 0, h * 2 * sizeof(T));
  memset(Rt, 0, h * 2 * sizeof(T));
  memset(h1_t, 0, h * 2 * sizeof(T));
  T *Wr = weight;
  T *Wi = weight + h * v;
  T *Wh = Wi + h * v;
  T *Rr = Wh + h * v;
  T *Ri = Rr + h * h;
  T *Rh = Ri + h * h;
  T *br = Rh + h * h;
  T *bi = br + h;
  T *bh = bi + h;
  for (unsigned int i = 0; i < s; i++) {    // time step
    for (unsigned int j = 0; j < b; j++) {  // batch
      // it = σ(Wixt + Riht-1 + bi)
      int in_index = j * v + i * v * b;
      int out_index = j * h + (i - 1) * h * b;
      T *t_hx = hx + j * h;
      cpu_mm(Wi, in + in_index, It, h, v, 1);  // Wi*Xt
      if (i != 0)
        cpu_mm(Ri, out + out_index, It + h, h, h, 1);  // Ri*ht-1
      else
        cpu_mm(Ri, t_hx, It + h, h, h, 1);  // Ri*ht-1
      cpu_add_bias(It, It + h, h);
      cpu_add_bias(It, bi, h);  // add bias
      cpu_sigmoid(It, h);

      // rt = σ(Wrxt + Rrht-1 + br)
      cpu_mm(Wr, in + in_index, Rt, h, v, 1);  // Wi*Xt
      if (i != 0)
        cpu_mm(Rr, out + out_index, Rt + h, h, h, 1);  // Ri*ht-1
      else
        cpu_mm(Rr, t_hx, Rt + h, h, h, 1);  // Ri*ht-1
      cpu_add_bias(Rt, Rt + h, h);
      cpu_add_bias(Rt, br, h);  // add bias
      cpu_sigmoid(Rt, h);

      // h1_t = tanh(Whxt + rt ◦ (Rhht-1) + bWh)
      cpu_mm(Wh, in + in_index, h1_t, h, v, 1);  // Wi*Xt
      if (i != 0)
        cpu_mm(Rh, out + out_index, h1_t + h, h, h, 1);  // Ri*ht-1
      else
        cpu_mm(Rh, t_hx, h1_t + h, h, h, 1);  // Ri*ht-1
      cpu_dotm(h1_t + h, Rt, h);              // rt ◦ (Rhht-1)

      cpu_add_bias(h1_t, h1_t + h, h);
      cpu_add_bias(h1_t, bh, h);  // add bias
      cpu_tanh(h1_t, h);

      // for(unsigned int a = 0; a < h; a++)
      //{
      //  if( It[a]<-0.99) //h1_t[a]>0.99 ||
      //    HCTR_LOG(INFO, WORLD, "satruate sigmoid timestep %d %f \n", i, h1_t[a]);
      //}
      // ht = (1 - it) ◦ h't + it ◦ ht-1
      T *ht_1 = i > 0 ? out + out_index : t_hx;
      cpu_ht(It, h1_t, ht_1, out + out_index + h * b, h);
      // cpu_ht(It, h1_t, out + out_index, out + out_index + h*b, h);
      // for(unsigned int a = 0; a < h; a++)
      //{
      //  if(ht_1[a]>0.99 || ht_1[a]<-0.99)
      //    HCTR_LOG(INFO, WORLD, "satruate sigmoid timestep %d %f \n", i, ht_1[a]);
      //}
    }
    // cpu_mm(Ri, out )
    // hidden + h*b + j * h + i * h * b
    // cpu_add_bias(out, )
  }
  delete[] It;
  delete[] Rt;
  delete[] h1_t;
}

template <typename T>
static void gru_layer_test(size_t batch_size, size_t hiddenSize, size_t embedding_vec_size,
                           size_t SeqLength) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> blobs_buff =
      GeneralBuffer2<CudaAllocator>::create();
  std::shared_ptr<BufferBlock2<T>> weight_buff = blobs_buff->create_block<T>();
  std::shared_ptr<BufferBlock2<T>> wgrad_buff = blobs_buff->create_block<T>();

  Tensor2<T> in_tensor;
  blobs_buff->reserve({1, batch_size * SeqLength * embedding_vec_size}, &in_tensor);
  Tensor2<T> out_tensor;
  blobs_buff->reserve({1, batch_size * SeqLength * hiddenSize}, &out_tensor);

  GRULayer<T> gru_layer(weight_buff, wgrad_buff, in_tensor, out_tensor, hiddenSize, batch_size,
                        SeqLength, embedding_vec_size, test::get_default_gpu());
  // Initialize tensors to 0 and choose cublas algorithms
  blobs_buff->allocate();
  gru_layer.initialize();
  // Reset tensors to 0 to ensure all the data are the same as original utest(clear the side effect
  // of optimize)
  Tensor2<T> weight = weight_buff->as_tensor();
  Tensor2<T> wgrad = wgrad_buff->as_tensor();
  size_t inputTensorSize = batch_size * SeqLength * embedding_vec_size;
  size_t outputTensorSize = batch_size * SeqLength * hiddenSize;
  size_t hiddenTensorSize = batch_size * hiddenSize;
  size_t weightSpaceSize =
      3 * hiddenSize * embedding_vec_size + 3 * hiddenSize * hiddenSize + 3 * hiddenSize;

  HCTR_LIB_THROW(cudaMemset(weight.get_ptr(), 0, weight.get_size_in_bytes()));
  HCTR_LIB_THROW(cudaMemset(wgrad.get_ptr(), 0, wgrad.get_size_in_bytes()));

  T *d_weight = weight.get_ptr();
  T *d_in = in_tensor.get_ptr();
  T *d_hx = d_weight + weightSpaceSize;
  T *d_out = out_tensor.get_ptr();
  T *d_dy = wgrad.get_ptr() + inputTensorSize;
  T *d_dhy = d_dy + outputTensorSize + hiddenTensorSize;

  std::unique_ptr<T[]> h_weight(new T[test::align_to_even(weightSpaceSize)]);
  // std::unique_ptr<T[]> h_bias_grad(new T[n]);
  std::unique_ptr<T[]> h_in(new T[test::align_to_even(inputTensorSize)]);
  std::unique_ptr<T[]> h_out(new T[test::align_to_even(outputTensorSize)]);
  std::unique_ptr<T[]> h_dx(new T[test::align_to_even(inputTensorSize)]);
  std::unique_ptr<T[]> h_dy(new T[test::align_to_even(outputTensorSize)]);
  std::unique_ptr<T[]> h_dhx(new T[test::align_to_even(hiddenTensorSize)]);
  std::unique_ptr<T[]> h_dhy(new T[test::align_to_even(hiddenTensorSize)]);
  std::unique_ptr<T[]> h_dweight(new T[test::align_to_even(weightSpaceSize)]);
// std::unique_ptr<T[]> h_bias(new T[test::align_to_even(m)]);
//#define RAND
//#define CUSTOMIZE
//#define RAND_WEIGHT
#define UNIFORM
//#define NORMAL
#ifdef RAND
  test::GaussianDataSimulator simulator(0.0f, 1.0f);
  simulator.fill(h_weight.get(), test::align_to_even(weightSpaceSize));
  simulator.fill(h_in.get(), test::align_to_even(inputTensorSize));
  // T* h_data= h_in.get();
  // HCTR_LOG(INFO, ROOT, "input %zu ", inputTensorSize);
  // for(unsigned int i=0;i<inputTensorSize;i++)
  //{
  //  HCTR_PRINT(INFO, "%f", h_data[i]);
  //}
  // HCTR_PRINT(INFO, "\n");
  // T* h_w = h_weight.get();
  // HCTR_LOG(INFO, ROOT, "out weight start:\n");
  // for(unsigned int i=0;i<weightSpaceSize/sizeof(T);i++)
  //{
  //  HCTR_PRINT(INFO, "%f ", h_w[i]);
  //}
  // HCTR_PRINT(INFO, "\nout weight end size %zu\n", weightSpaceSize/sizeof(T));
  // exit(1);
#elif defined CUSTOMIZE
  T *tptr = h_in.get();
  for (unsigned int i = 0; i < inputTensorSize; i++)
    tptr[i] = float(rand() % 30) / 10 - 1.5;  //[0.7, 1.3]
  tptr = h_weight.get();
  for (unsigned int i = 0; i < weightSpaceSize; i++) tptr[i] = float(rand() % 30) / 10 - 1.5;
#elif defined RAND_WEIGHT
  test::GaussianDataSimulator simulator(0.0f, 1.0f);
  simulator.fill(h_weight.get(), test::align_to_even(weightSpaceSize));

  T *tptr = h_in.get();
  for (unsigned int i = 0; i < inputTensorSize; i++) tptr[i] = 1;
#elif defined UNIFORM
  test::UniformDataSimulator simulator;
  simulator.fill(h_weight.get(), test::align_to_even(weightSpaceSize), -1, 1);  // 0.002, 0.2
  simulator.fill(h_in.get(), test::align_to_even(inputTensorSize), -1, 1);      // 0.7, 1.3
#elif defined NORMAL
  std::default_random_engine generator;
  std::normal_distribution<double> distribution_in(0, 1.0);        // 0.7, 1.3
  std::normal_distribution<double> distribution_weight(0.5, 0.5);  // 0.002, 0.2
  T *tptr = h_in.get();
  for (unsigned int i = 0; i < inputTensorSize; i++) tptr[i] = distribution_in(generator);  // 1.0
  tptr = h_weight.get();
  for (unsigned int i = 0; i < weightSpaceSize; i++) tptr[i] = distribution_weight(generator);
#else
  test::GaussianDataSimulator simulator(0.0f, 1.0f);
  simulator.fill(h_in.get(), test::align_to_even(inputTensorSize));
  T *tptr = h_weight.get();
  for (unsigned int i = 0; i < weightSpaceSize; i++) tptr[i] = 1.0;
#endif
// T *tptr = h_in.get();
// HCTR_LOG(INFO, ROOT, "inputX:\n");
// for(unsigned int i =0; i<inputTensorSize;i++){
//  HCTR_PRINT(INFO, "%.8f ", tptr[i]);
//  if((i+1)% embedding_vec_size==0)
//    HCTR_PRINT(INFO, "\n");
//}
// T *tptr = h_weight.get();
////HCTR_LOG(INFO, ROOT, "inputW: %zu %zu %zu\n", weightSpaceSize, hiddenSize*embedding_vec_size*3,
/// hiddenSize*hiddenSize*3 );
// for(unsigned int i =0; i<weightSpaceSize;i++){
//  HCTR_PRINT(INFO, "%.8f ", tptr[i]);
//  if(i<hiddenSize*embedding_vec_size*3 && (i+1)% embedding_vec_size==0)
//    HCTR_PRINT(INFO, "\n");
//  if(i>hiddenSize*embedding_vec_size*3 && i< hiddenSize*embedding_vec_size*3 +
//  hiddenSize*hiddenSize*3 && (i+1)% hiddenSize==0)
//    HCTR_PRINT(INFO, "\n");
//  if(i>=hiddenSize*embedding_vec_size*3 + hiddenSize*hiddenSize*3)
//    HCTR_PRINT(INFO, "\n");
//}
// T* tptr = h_in.get();
// T tmp[4] = {-0.747210, -1.379702, 0.995153, -1.136765};
// for(unsigned int i =0; i<inputTensorSize;i++)
//  tptr[i] = tmp[i];

// simulator.fill(h_bias.get(), test::align_to_even(m));
// cpu fprop
// Py_Initialize();
// PyRun_SimpleString("import sys; sys.path.append('../test/utest/layers/')");
// PyRun_SimpleString("print(sys.path)");
// PyRun_SimpleString("import keras_GRU;");
// PyRun_SimpleString("keras_GRU.GRU()");
// Py_Finalize();
#ifdef KERAS_CHECK
  T *d_dx = wgrad.get_ptr();
  T *d_dhx = wgrad.get_ptr() + inputTensorSize + outputTensorSize;
  T *d_dweight = wgrad.get_ptr() + inputTensorSize + outputTensorSize + 2 * hiddenTensorSize;

  // T* h_data= h_in.get();
  // HCTR_LOG(INFO, ROOT, "input %zu ", inputTensorSize);
  // for(unsigned int i=0;i<inputTensorSize;i++)
  //{
  //  HCTR_PRINT(INFO, "%f", h_data[i]);
  //}
  // HCTR_PRINT(INFO, "\n");

  HCTR_LOG(INFO, ROOT, "parameter 4 %zu %zu %zu %zu \n", batch_size, hiddenSize, embedding_vec_size,
           SeqLength);

  T *h_data = h_in.get();
  HCTR_LOG(INFO, ROOT, "input %zu ", inputTensorSize);
  for (unsigned int i = 0; i < inputTensorSize; i++) {
    HCTR_PRINT(INFO, "%f ", h_data[i]);
  }
  HCTR_PRINT(INFO, "\n");
#endif

  //#define CUDNNTEST

  T *C_in, *C_weight, *C_y;
  bool result;
#ifdef CUDNNTEST
  C_in = testX;
  C_weight = testW;
  C_y = refY;
#else
  C_in = h_in.get();
  C_weight = h_weight.get();
  C_y = NULL;
#endif
  // for (unsigned int i = 0; i < inputTensorSize; i++)
  //  if (C_in[i] > 1 || C_in[i] < -1) HCTR_LOG(INFO, ROOT, "C_in %f\n", C_in[i]);
  // for (unsigned int i = 0; i < weightSpaceSize; i++)
  //  if (C_weight[i] > 1 || C_weight[i] < -1) HCTR_LOG(INFO, ROOT, "C_weight %f\n", C_weight[i]);

  // for(unsigned int i=0;i<outputTensorSize; i++)
  //  if(C_y[i]>0.6 || C_y[i]<0.1)
  //    HCTR_LOG(INFO, ROOT, "C_y %f\n", C_y[i]);
  // memset(testHx, 0, sizeof(hiddenTensorSize)*sizeof(T));
  // HCTR_LIB_THROW(cudaMemcpy(d_weight, h_weight.get(), weightSpaceSize*sizeof(T),
  // cudaMemcpyHostToDevice)); HCTR_LIB_THROW(cudaMemcpy(d_in, h_in.get(), sizeof(T) *
  // inputTensorSize, cudaMemcpyHostToDevice));
  T *testHx = new T[hiddenTensorSize];
  memset(testHx, 0, hiddenTensorSize);
  HCTR_LIB_THROW(cudaMemcpy(d_hx, testHx, sizeof(T) * hiddenTensorSize, cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(
      cudaMemcpy(d_weight, C_weight, weightSpaceSize * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(d_in, C_in, sizeof(T) * inputTensorSize, cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  gru_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  HCTR_LIB_THROW(
      cudaMemcpy(h_out.get(), d_out, sizeof(T) * outputTensorSize, cudaMemcpyDeviceToHost));

  // HCTR_LOG(INFO, ROOT, "outputY:\n");
  // T* tptr = h_out.get();
  // for(unsigned int i =0; i<outputTensorSize;i++){
  //    HCTR_PRINT(INFO, "%.8f ", tptr[i]);
  //    if((i+1)% hiddenSize==0)
  //      HCTR_PRINT(INFO, "\n");
  //}

  // T* y = h_out.get();
  // for(unsigned int i=0;i<outputTensorSize; i++)
  //  if(y[i]>1 || y[i]< -1)
  //    HCTR_LOG(INFO, ROOT, "y %f\n", y[i]);
  // check refY
  if (C_y != NULL) {
    result = check_correctness(h_out.get(), C_y, outputTensorSize, 1e-3);
    if (result)
      HCTR_LOG(INFO, WORLD, "forward prop hugectr v8 matched with cudnnTest2\n");
    else
      HCTR_LOG(INFO, WORLD, "forward prop hugectr v8 failed match with cudnnTest2!!\n");
  }
  // rnn_v6(testX, testW, testHx, refY, h_dy.get(), h_dhy.get(), hiddenSize, batch_size,
  // SeqLength, embedding_vec_size,  h_dweight.get(), h_out.get());
  cpu_gru<T>(C_weight, C_in, testHx, h_out.get(), hiddenSize, embedding_vec_size, batch_size,
             SeqLength);

  // rnn_v6(C_in, C_weight, testHx, C_y, h_dy.get(), h_dhy.get(), hiddenSize, batch_size, SeqLength,
  //       embedding_vec_size, h_dweight.get(), h_out.get());

  T *cpu_y = new T[outputTensorSize];
  cpu_gru<T>(C_weight, C_in, testHx, cpu_y, hiddenSize, embedding_vec_size, batch_size, SeqLength);
  delete[] testHx;
  result = check_correctness(h_out.get(), cpu_y, outputTensorSize, 1e-3);
  if (result)
    HCTR_LOG(INFO, WORLD, "forward prop CPU matched with V8\n");
  else
    HCTR_LOG(INFO, WORLD, "forward prop CPU failed match with V8!!\n");

  if (C_y != NULL) {
    result = check_correctness(cpu_y, C_y, outputTensorSize, 1e-3);
    if (result)
      HCTR_LOG(INFO, WORLD, "forward prop CPU matched with cudnnTest\n");
    else
      HCTR_LOG(INFO, WORLD, "forward prop CPU failed match with cudnnTest!!\n");
  }
  // check refY end

#ifdef KERAS_CHECK
  HCTR_LIB_THROW(
      cudaMemcpy(h_weight.get(), d_weight, weightSpaceSize * sizeof(T), cudaMemcpyDeviceToHost));
  h_data = h_out.get();
  HCTR_LOG(INFO, ROOT, "output_gpu %zu ", outputTensorSize);
  for (unsigned int i = 0; i < outputTensorSize; i++) {
    HCTR_PRINT(INFO, "%f ", h_data[i]);
  }
  HCTR_PRINT(INFO, "\n");
#endif
// CPU reference
// cpu_gru<T>(h_weight.get(), h_in.get(), h_out.get(), hiddenSize, embedding_vec_size, batch_size,
// SeqLength);
#ifndef KERAS_CHECK
// T *d_out = out_tensor.get_ptr();
// ASSERT_EQ(true, check_cpu_gpu(h_out.get(), d_out, outputTensorSize)) << "fprop cross_check result
// fail"<< std::endl;
#endif
// CPU bprop TODO
// simulator.fill(h_dy.get(), test::align_to_even(outputTensorSize));
// simulator.fill(h_dhy.get(), test::align_to_even(hiddenTensorSize));
#ifdef KERAS_CHECK
  h_data = h_out.get();
  HCTR_LOG(INFO, ROOT, "output_cpu %zu ", outputTensorSize);
  for (unsigned int i = 0; i < outputTensorSize; i++) {
    HCTR_PRINT(INFO, "%.32f ", h_data[i]);
  }
  HCTR_PRINT(INFO, "\n");

  T *ptr = h_dy.get();
  for (unsigned int i = 0; i < outputTensorSize; i++) ptr[i] = 1.0;

  ptr = h_dhy.get();
  for (unsigned int i = 0; i < hiddenTensorSize; i++) ptr[i] = 0;

    // ptr = h_dy.get();
    // HCTR_LOG(INFO, ROOT, "dy %zu ", outputTensorSize);
    // for(unsigned int i=0;i<outputTensorSize;i++)
    //{
    //  HCTR_PRINT(INFO, "%f ", ptr[i]);
    //}
    // HCTR_PRINT(INFO, "\n");
    //
    // ptr = h_dhy.get();
    // HCTR_LOG(INFO, ROOT, "dhy %zu ", hiddenTensorSize);
    // for(unsigned int i=0;i<hiddenTensorSize;i++)
    //{
    //  HCTR_PRINT(INFO, "%f ", ptr[i]);
    //}
    // HCTR_PRINT(INFO, "\n");
#endif
  HCTR_LIB_THROW(
      cudaMemcpy(d_dy, h_dy.get(), sizeof(T) * outputTensorSize, cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(
      cudaMemcpy(d_dhy, h_dhy.get(), sizeof(T) * hiddenTensorSize, cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  gru_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

#ifdef KERAS_CHECK
  HCTR_LIB_THROW(cudaMemcpy(h_dx.get(), d_dx, sizeof(T) * inputTensorSize, cudaMemcpyDeviceToHost));
  HCTR_LIB_THROW(
      cudaMemcpy(h_dhx.get(), d_dhx, hiddenTensorSize * sizeof(T), cudaMemcpyDeviceToHost));
  HCTR_LIB_THROW(
      cudaMemcpy(h_dweight.get(), d_dweight, weightSpaceSize * sizeof(T), cudaMemcpyDeviceToHost));

  // ptr = h_dx.get();
  // HCTR_LOG(INFO, ROOT, "dx %zu ", inputTensorSize);
  // for(unsigned int i=0;i<inputTensorSize;i++)
  //{
  //  HCTR_PRINT(INFO, "%.32f ", ptr[i]);
  //}
  // HCTR_PRINT(INFO, "\n");
  //
  // ptr = h_dhx.get();
  // HCTR_LOG(INFO, ROOT, "dhx %zu ", hiddenTensorSize);
  // for(unsigned int i=0;i<hiddenTensorSize;i++)
  //{
  //  HCTR_PRINT(INFO, "%f ", ptr[i]);
  //}
  // HCTR_PRINT(INFO, "\n");

  ptr = h_dweight.get();
  HCTR_LOG(INFO, ROOT, "dweight %zu ", weightSpaceSize);
  for (unsigned int i = 0; i < weightSpaceSize; i++) {
    HCTR_PRINT(INFO, "%.32f ", ptr[i]);
  }
  HCTR_PRINT(INFO, "\n");

// std::fstream fs;
// fs.open("tmp.log", std::fstream::out | std::fstream::binary);
// fs.write((char*)&num_streams, 4);
// fs.write((const char*) &compressed[0], compressed.size()*sizeof(unsigned char));
// fs.close();
#endif
  // rnn_v6(h_in.get(), h_weight.get(), h_dy.get(), h_dhy.get(), hiddenSize, batch_size, SeqLength,
  // embedding_vec_size,  h_dweight.get(), h_out.get());
}

}  // namespace

// batch_size, size_t hiddenSize, size_t embedding_vec_size, size_t SeqLength
TEST(gru_layer, fp32_32x128x100x28) { gru_layer_test<float>(32, 128, 100, 28); }
TEST(gru_layer, fp32_32x128x100x128) { gru_layer_test<float>(32, 128, 100, 128); }
// TEST(gru_layer, fp32_1x2x3x4) { gru_layer_test<float>(1, 2, 3, 4); }
// TEST(gru_layer, fp32_10x20x20x10) { gru_layer_test<float>(10, 20, 20, 10); } //not work
TEST(gru_layer, fp32_8x16x32x12) { gru_layer_test<float>(8, 16, 32, 12); }
TEST(gru_layer, fp32_256x256x256x256) { gru_layer_test<float>(256, 256, 256, 256); }
TEST(gru_layer, fp32_32x32x100x32) { gru_layer_test<float>(32, 32, 100, 32); }
TEST(gru_layer, fp32_32x32x32x20) { gru_layer_test<float>(32, 32, 32, 20); }
TEST(gru_layer, fp32_32x64x100x32) { gru_layer_test<float>(32, 64, 100, 32); }
TEST(gru_layer, fp32_1x512x256x8) { gru_layer_test<float>(1, 512, 256, 8); }
TEST(gru_layer, fp32_256x128x256x256) { gru_layer_test<float>(256, 128, 256, 256); }
// TEST(gru_layer, fp32_128x256x256x32) { gru_layer_test<float>(128, 256, 256, 32); }
TEST(gru_layer, fp32_128x256x256x64) { gru_layer_test<float>(128, 256, 256, 64); }
TEST(gru_layer, fp32_128x256x256x128) { gru_layer_test<float>(128, 256, 256, 128); }
// TEST(gru_layer, fp32_1024x512x1024x128) { gru_layer_test<float>(1024, 512, 1024, 128); }
