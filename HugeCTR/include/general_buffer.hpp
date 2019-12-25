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

#pragma once

#include <vector>
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

/**
 * @brief General buffer on GPU memory.
 *
 * A class for GPU memory management. It allows tensors to share a continous memory
 * for high efficient memory transaction and unified parameter updating.
 * To allocate GPU memory, you call reserve() to register the memory size you want to
 * allocate (one or more times), and then call init() to allocate the memory in once.
 */
template <typename T>
class GeneralBuffer {
 private:
  T* ptr_{nullptr};          /**< pointer of memory allocated on GPU */
  size_t current_offset_{0}; /**< memory registered */
  int device_id_{-1};        /**< gpu id */
  bool initialized_{false};  /**< whether the gpu memory has been allocated */
 public:
  /**
   * Ctor
   */
  GeneralBuffer() {
    static_assert(std::is_same<T, float>::value || std::is_same<T, long long>::value ||
                      std::is_same<T, unsigned int>::value,
                  "Type not support");  // check the template parameters
  }
  GeneralBuffer(const GeneralBuffer& C) = delete;
  GeneralBuffer& operator=(const GeneralBuffer&) = delete;

  /**
   * Allocate memory on GPU according to memory registered by reserve().
   * It will allocate current_offset*sizeof(T) on target device and set ptr_.
   * @param the device_id target device id.
   */
  void init(int device_id) {
    if (initialized_ != false) CK_THROW_(Error_t::IllegalCall, "Initilized general buffer");
    device_id_ = device_id;
    CudaDeviceContext context(device_id);
    CK_CUDA_THROW_(cudaMalloc((void**)&ptr_, current_offset_ * sizeof(T)));
    CK_CUDA_THROW_(cudaMemset(ptr_, 0, current_offset_ * sizeof(T)));
    initialized_ = true;
  }

  /**
   * To set memory to 0 and synchronize.
   */
  void reset_sync() {
    if (initialized_ != true) CK_THROW_(Error_t::IllegalCall, "Not initialized");
    CudaDeviceContext context(device_id_);
    CK_CUDA_THROW_(cudaMemset(ptr_, 0, current_offset_ * sizeof(T)));
    CK_CUDA_THROW_(cudaDeviceSynchronize());
  }

  /**
   * Ctor.
   * Allocate specific size of buffer on device when constructing.
   * @param size number of elements * sizeof(T) memory will be allocated.
   * @param device_id target device id.
   */
  GeneralBuffer(size_t size, int device_id) {
    current_offset_ = size;
    init(device_id);
  }

  /**
   * Register num_elements*sizeof(T) memory.
   * Tensor will call this to aquire the current offset on elements also.
   * @param num_elements register num_elements * sizeof(T)
   * @return the offset before this register.
   */
  size_t reserve(size_t num_elements) {
    size_t tmp_offset_ = current_offset_;
    current_offset_ += num_elements;
    return tmp_offset_;
  }

  int get_device_id() const { return device_id_; }

  /**
   * Calculate the address of memory with offset.
   * Tensor can call this to aquire the real address of memory.
   * @param offset element offset on this buffer.
   * @return memory address on this offset.
   */
  T* get_ptr_with_offset(size_t offset) {
    try {
      if (initialized_ != true)
        CK_THROW_(Error_t::NotInitialized, "GeneralBuffer is not initialized");
      assert(ptr_ != nullptr);
      return ptr_ + offset;
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
    }
    return nullptr;
  }

  /**
   * Calculate the address of memory with offset.
   * Tensor can call this to aquire the real address of memory.
   * @param offset element offset on this buffer.
   * @return memory address on this offset.
   */
  const T* get_ptr_with_offset(size_t offset) const {
    try {
      if (initialized_ != true)
        CK_THROW_(Error_t::NotInitialized, "GeneralBuffer is not initialized");
      assert(ptr_ != nullptr);
      return ptr_ + offset;
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
    }
    return nullptr;
  }

  /**
   * Aquire the emory size of this buffer.
   */
  size_t get_size() const { return current_offset_ * sizeof(T); }

  /**
   * Get the number of elements can be stored in this buffer.
   */
  size_t get_num_elements() const { return current_offset_; }

  /**
   * Dtor
   */
  ~GeneralBuffer() {
    try {
      if (initialized_ == true) {
        CudaDeviceContext context(device_id_);
        CK_CUDA_THROW_(cudaFree(ptr_));
      }
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
    }
  }
};

/**
 * A helper function to print the buffer on screen.
 * When both begin and end are positive numbers: print the begin->end elements.
 * When both of them are negtive numbers: print the last elements:
 * @verbatim
    begin_ = buffer.get_num_elements() + begin;
    end_ = buffer.get_num_elements() + end;
 * @endverbatim
 */
template <typename T>
bool print_buffer(const GeneralBuffer<T>& buffer, int begin, int end) {
  int begin_;
  int end_;
  if (begin >= 0 && end <= static_cast<int>(buffer.get_num_elements()) && end > begin) {
    begin_ = begin;
    end_ = end;
  } else if (end < 0 && -begin <= static_cast<int>(buffer.get_num_elements()) && end > begin) {
    begin_ = buffer.get_num_elements() + begin;
    end_ = buffer.get_num_elements() + end;
  } else {
    return false;
  }
  CudaDeviceContext context(buffer.get_device_id());
  cudaDeviceSynchronize();
  T host_buff[end_ - begin_];
  cudaMemcpy(host_buff, buffer.get_ptr_with_offset(begin_), (end_ - begin_) * sizeof(T),
             cudaMemcpyDeviceToHost);
  std::cout << "Buffer: " << buffer.get_num_elements() << std::endl;
  std::cout << "begin: " << begin_ << " end: " << end_ << std::endl;
  for (int i = 0; i < end_ - begin_; i++) {
    std::cout << host_buff[i] << ",";
  }
  std::cout << std::endl;
  return true;
}

template <typename T>
using GeneralBuffers = std::vector<std::shared_ptr<GeneralBuffer<T>>>;

}  // namespace HugeCTR
