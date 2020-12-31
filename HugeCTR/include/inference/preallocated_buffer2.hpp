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
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {

template <typename T>
class PreallocatedBuffer2 : public TensorBuffer2 {

  void *ptr_;
  size_t size_in_bytes_;
  /**
   * @name: Ctor of PreallocatedBuffer2
   * @param ptr the pointer of the buffer that is pre-allocated
   * @param dimensions the dimensions of the tensor to be bound to this buffer
   */
  PreallocatedBuffer2(void* ptr, const std::vector<size_t> &dimensions) : ptr_(ptr) {
    size_in_bytes_ = get_num_elements_from_dimensions(dimensions) * TensorScalarSizeFunc<T>::get_element_size();
  }

public:
  static std::shared_ptr<PreallocatedBuffer2> create(void* ptr, const std::vector<size_t> &dimensions) {
    return std::shared_ptr<PreallocatedBuffer2>(new PreallocatedBuffer2(ptr, dimensions));
  }

  PreallocatedBuffer2(const PreallocatedBuffer2 &) = delete;
  PreallocatedBuffer2 &operator=(const PreallocatedBuffer2 &) = delete;

  template <typename TypeTensor>
  friend void bind_tensor_to_buffer(const std::shared_ptr<TensorBuffer2> &buffer, Tensor2<TypeTensor> *tensor);

  bool allocated() const override {
    if (ptr_ == nullptr) {
      CK_THROW_(Error_t::NotInitialized, "The buffer for Tensor2 should be pre allocated");
    }
    return true;
  }

  void *get_ptr() override { return ptr_; }   

}; // class PreallocatedBuffer2

template <typename TypeTensor>
void bind_tensor_to_buffer(const std::vector<size_t> &dimensions, const std::shared_ptr<TensorBuffer2> &buffer, std::shared_ptr<Tensor2<TypeTensor>>& tensor) {
  try {
    if (!buffer->allocated()) {
      CK_THROW_(Error_t::IllegalCall, "Cannot bind tensor to buffer that is not allocated");
    }
    tensor->set_buffer(buffer);
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

} // namespace HugeCTR