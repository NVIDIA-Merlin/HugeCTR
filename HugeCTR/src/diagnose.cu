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

#include <diagnose.hpp>
#include <fstream>
#include <limits>
#include <utils.cuh>

namespace HugeCTR {

namespace diagnose {

__device__ float atomicMin(float* address, float val) {
  float old = val;
  do {
    val = old;
    old = atomicExch(address, val);
  } while (old < val);
  return old;
}

__device__ float atomicMax(float* address, float val) {
  float old = val;
  do {
    val = old;
    old = atomicExch(address, val);
  } while (old > val);
  return old;
}

template <typename T>
__global__ void histogram_kernel(const T* arr, size_t len, float* range) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
    float val = TypeConvertFunc<float, T>::convert(arr[i]);
    if (val <= 0) {
      atomicMin(range + 0, val);
      atomicMax(range + 1, val);
    }
    if (val >= 0) {
      atomicMin(range + 2, val);
      atomicMax(range + 3, val);
    }
  }
}

template <typename T>
__global__ void verify_kernel(const T* arr, size_t len, int* flag);

template <>
__global__ void verify_kernel<float>(const float* arr, size_t len, int* flag) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
    if (isnan(arr[i])) atomicAdd(flag, 1);
  }
}

template <>
__global__ void verify_kernel(const __half* arr, size_t len, int* flag) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
    if (__hisnan(arr[i])) {
      atomicAdd(flag, 1);
    }
  }
}

template <typename T>
__global__ void sample_kernel(const T* arr, int len, float* arr_sample, int stride,
                              int max_sample_len) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
    if (i % stride == 0) {
      int j = i / stride;
      if (j < max_sample_len) {
        arr_sample[j] = TypeConvertFunc<float, T>::convert(arr[i]);
      }
    }
  }
}

template <typename T>
void verify_and_histogram(const char* category, const Tensor2<T>& tensor,
                          const cudaStream_t& stream) {
  float h_array[4]{0.0f, -std::numeric_limits<float>::infinity(),
                   std::numeric_limits<float>::infinity(), 0.0f};
  int h_flag;
  float* d_array;
  int* d_flag;
  HCTR_LIB_THROW(cudaMalloc(&d_array, sizeof(h_array)));
  HCTR_LIB_THROW(cudaMalloc(&d_flag, sizeof(int)));
  HCTR_LIB_THROW(
      cudaMemcpyAsync(d_array, h_array, sizeof(h_array), cudaMemcpyHostToDevice, stream));
  HCTR_LIB_THROW(cudaMemsetAsync(d_flag, 0, sizeof(int), stream));
  histogram_kernel<<<160, 1024, 0, stream>>>(tensor.get_ptr(), tensor.get_num_elements(), d_array);
  verify_kernel<<<160, 1024, 0, stream>>>(tensor.get_ptr(), tensor.get_num_elements(), d_flag);
  HCTR_LIB_THROW(
      cudaMemcpyAsync(h_array, d_array, sizeof(h_array), cudaMemcpyDeviceToHost, stream));
  HCTR_LIB_THROW(cudaMemcpyAsync(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost, stream));
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  HCTR_LOG_S(INFO, ROOT) << "Diagnose for (" << category << "), Histogram [" << h_array[0] << ", "
                         << h_array[1] << "]"
                         << ", [" << h_array[2] << ", " << h_array[3] << "]" << std::endl;

  if (h_flag != 0) {
    std::ostringstream os;
    os << "Nan assert for " << category << " failed(" << h_flag << ").";
    HCTR_OWN_THROW(Error_t::DataCheckError, os.str());
  }
  HCTR_LIB_THROW(cudaFree(d_array));
  HCTR_LIB_THROW(cudaFree(d_flag));
}

template <typename T>
void sample_and_print(const char* category, const Tensor2<T>& tensor, size_t sample_count,
                      const cudaStream_t& stream) {
  if (sample_count == 0) return;

  std::unique_ptr<float[]> h_array(new float[sample_count]);

  float* d_array;
  HCTR_LIB_THROW(cudaMalloc(&d_array, sample_count * sizeof(float)));
  HCTR_LIB_THROW(cudaMemsetAsync(d_array, 0, sample_count * sizeof(float), stream));
  sample_kernel<<<160, 1024, 0, stream>>>(tensor.get_ptr(), tensor.get_num_elements(), d_array,
                                          tensor.get_num_elements() / sample_count, sample_count);
  HCTR_LIB_THROW(cudaMemcpyAsync(h_array.get(), d_array, sample_count * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  {
    auto log = HCTR_LOG_S(INFO, ROOT);
    log << "Diagnose for (" << category << "), Sampling [";
    for (size_t i = 0; i < min(sample_count, tensor.get_num_elements()); i++) {
      if (i != 0) log << ",";
      log << h_array[i];
    }
    log << "]" << std::endl;
  }

  HCTR_LIB_THROW(cudaFree(d_array));
}

template <typename T>
void sample_and_print(const char* category, const Tensor2<T>& tensor, int begin, int end,
                      const cudaStream_t& stream) {
  if (begin >= 0 && end <= static_cast<int>(tensor.get_num_elements()) && end > begin) {
  } else if (end < 0 && begin >= -static_cast<int>(tensor.get_num_elements()) && end > begin) {
    begin += tensor.get_num_elements();
    end += tensor.get_num_elements();
  } else {
    return;
  }

  std::unique_ptr<T[]> h_array(new T[end - begin]);
  HCTR_LIB_THROW(cudaMemcpyAsync(h_array.get(), tensor.get_ptr() + begin,
                                 (begin - end) * sizeof(float), cudaMemcpyDeviceToHost, stream));
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  {
    auto log = HCTR_LOG_S(INFO, ROOT);
    log << "Diagnose for (" << category << "), Sampling [";
    for (size_t i = 0; i < end - begin; i++) {
      if (i != 0) log << ",";
      log << h_array[i];
    }
    log << "]" << std::endl;
  }
}

template <typename T>
void dump(const char* filename, const Tensor2<T>& tensor, const cudaStream_t& stream) {
  std::unique_ptr<T[]> h_array(new T[tensor.get_num_elements()]);
  HCTR_LIB_THROW(cudaMemcpyAsync(h_array.get(), tensor.get_ptr(), tensor.get_size_in_bytes(),
                                 cudaMemcpyDeviceToHost, stream));
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  std::ofstream s(filename, std::ios::out | std::ios::binary);
  s.write(reinterpret_cast<const char*>(h_array.get()), tensor.get_size_in_bytes());
  s.close();
}

template void verify_and_histogram<float>(const char* category, const Tensor2<float>& tensor,
                                          const cudaStream_t& stream);

template void dump<unsigned int>(const char* filename, const Tensor2<unsigned int>& tensor,
                                 const cudaStream_t& stream);
template void dump<unsigned long>(const char* filename, const Tensor2<unsigned long>& tensor,
                                  const cudaStream_t& stream);
template void dump<long long>(const char* filename, const Tensor2<long long>& tensor,
                              const cudaStream_t& stream);
template void dump<float>(const char* filename, const Tensor2<float>& tensor,
                          const cudaStream_t& stream);
template void dump<__half>(const char* filename, const Tensor2<__half>& tensor,
                           const cudaStream_t& stream);

}  // namespace diagnose

}  // namespace HugeCTR
