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
#include <memory>


namespace HugeCTR {

/**
 * @brief Pinned buffer on CPU memory.
 *
 * A class for CPU memory management. It alloc a piece of CPU memory,
 * make these memory can be copy to GPU without synchronization
 */
template <typename T>
class PinnedBuffer : public std::unique_ptr<T[]> {
private:
  const size_t size_;
 public:
  /**
   * Ctor
   */
  PinnedBuffer(size_t size) :  std::unique_ptr<T[]>(new T[size]), size_(size) {
    static_assert(std::is_pod<T>::value, "T must be a POD type.");
    // make sure these memory can be copy to GPU without synchronization
    CK_CUDA_THROW_(cudaHostRegister(this->get(), sizeof(T) * size, cudaHostRegisterDefault));
  }

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  PinnedBuffer(PinnedBuffer&&) = default;
  
  size_t get_num_elements() const {return size_;}
  ~PinnedBuffer() noexcept(false) {
    if (this->get()) {
      CK_CUDA_THROW_(cudaHostUnregister(this->get()));
    }
  }
};

}  // namespace HugeCTR
