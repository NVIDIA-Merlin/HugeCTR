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
#include <common.hpp>
#include <memory>
#include <vector>
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

enum class TensorScalarType { None, Void, Float32, Float16, Int64, UInt64, Int32, UInt32 };

namespace {

inline size_t get_num_elements_from_dimensions(const std::vector<size_t> &dimensions) {
  size_t elements = 1;
  for (size_t dim : dimensions) {
    elements *= dim;
  }
  return elements;
}

inline void *forward_void_pointer(void *ptr, size_t offset) {
  return reinterpret_cast<unsigned char *>(ptr) + offset;
}

template <typename T>
struct TensorScalarSizeFunc {
  static size_t get_element_size() { return sizeof(T); }
};

template <>
struct TensorScalarSizeFunc<void> {
  static size_t get_element_size() { return 1ul; }
};

template <typename T>
struct TensorScalarTypeFunc {};

template <>
struct TensorScalarTypeFunc<float> {
  static TensorScalarType get_type() { return TensorScalarType::Float32; }
};

template <>
struct TensorScalarTypeFunc<__half> {
  static TensorScalarType get_type() { return TensorScalarType::Float16; }
};

}  // namespace

class TensorBuffer2 {
 public:
  virtual ~TensorBuffer2() {}
  virtual bool allocated() const = 0;
  virtual void *get_ptr() = 0;
};

class TensorBag2 {
  template <typename T>
  friend class Tensor2;

  std::vector<size_t> dimensions_;
  std::shared_ptr<TensorBuffer2> buffer_;
  TensorScalarType scalar_type_;

  TensorBag2(const std::vector<size_t> dimensions, const std::shared_ptr<TensorBuffer2> &buffer,
             TensorScalarType scalar_type)
      : dimensions_(dimensions), buffer_(buffer), scalar_type_(scalar_type) {}

 public:
  TensorBag2() : scalar_type_(TensorScalarType::None) {}
};

template <typename T>
class Tensor2 {
  std::vector<size_t> dimensions_;
  size_t num_elements_;
  std::shared_ptr<TensorBuffer2> buffer_;

 public:
  static Tensor2 stretch_from(const TensorBag2 &bag) {
    if (bag.scalar_type_ != TensorScalarTypeFunc<T>::get_type()) {
      CK_THROW_(Error_t::WrongInput, "Inconsistent tensor type");
    }

    return Tensor2(bag.dimensions_, bag.buffer_);
  }

  Tensor2() : num_elements_(0) {}

  Tensor2(const std::vector<size_t> &dimensions, const std::shared_ptr<TensorBuffer2> &buffer)
      : dimensions_(dimensions),
        num_elements_(get_num_elements_from_dimensions(dimensions)),
        buffer_(buffer) {}

  TensorBag2 shrink() const {
    return TensorBag2(dimensions_, buffer_, TensorScalarTypeFunc<T>::get_type());
  }

  bool allocated() const { return buffer_ && buffer_->allocated(); }

  const std::vector<size_t> &get_dimensions() const { return dimensions_; }

  size_t get_num_elements() const { return num_elements_; }

  size_t get_size_in_bytes() const {
    return num_elements_ * TensorScalarSizeFunc<T>::get_element_size();
  }

  const T *get_ptr() const { return reinterpret_cast<const T *>(buffer_->get_ptr()); }

  T *get_ptr() { return reinterpret_cast<T *>(buffer_->get_ptr()); }
};

template <typename T>
using Tensors2 = std::vector<Tensor2<T>>;


/**
 * To print the tensor from begin to end.
 * When both begin and end are positive numbers: print the begin->end elements.
 * When both of them are negtive numbers: print the last elements:
 * @verbatim
 *  begin_ = get_size_from_dims(tensor.get_dims()) + begin;
 *  end_ = get_size_from_dims(tensor.get_dims()) + end;
 * @endverbatim
 */
template <typename T>
inline bool print_tensor(const Tensor2<T>& tensor, int begin, int end) {
  int begin_;
  int end_;
  //if (begin >= 0 && end <= (int)get_size_from_dims(tensor.get_dims()) && end > begin) {
  if (begin >= 0 && end <= (int)tensor.get_num_elements() && end > begin) {
    begin_ = begin;
    end_ = end;
  } else if (end < 0 && -begin <= (int)tensor.get_num_elements() && end > begin) {
    begin_ = get_size_from_dims(tensor.get_dimensions()) + begin;
    end_ = get_size_from_dims(tensor.get_dimensions()) + end;
  } else {
    return false;
  }
  //  CudaDeviceContext context(tensor.get_device_id());
  cudaDeviceSynchronize();
  assert(end_ > begin_ && begin_ >= 0 &&
         end_ < static_cast<int>(get_size_from_dims(tensor.get_dimensions())));
  T host_buff[end_ - begin_];
  cudaMemcpy(host_buff, tensor.get_ptr() + begin_, (end_ - begin_) * sizeof(T),
             cudaMemcpyDefault);
  std::cout << "Tensor: <";
  for (auto d : tensor.get_dimensions()) {
    std::cout << d << ",";
  }
#ifdef ENABLE_MPI
  int pid(-1), num_procs(-1);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  std::cout << "> in pid: " << pid << " of " << num_procs << " processes." << std::endl;
#else
  std::cout << ">" << std::endl;
#endif
  std::cout << "begin: " << begin_ << " end: " << end_ << std::endl;
  for (int i = 0; i < end_ - begin_; i++) {
    std::cout << host_buff[i] << ",";
  }
  std::cout << std::endl;
  return true;
}

template <>
inline bool print_tensor(const Tensor2<__half>& tensor, int begin, int end) {
  int begin_;
  int end_;
  if (begin >= 0 && end <= (int)get_size_from_dims(tensor.get_dimensions()) && end > begin) {
    begin_ = begin;
    end_ = end;
  } else if (end < 0 && -begin <= (int)get_size_from_dims(tensor.get_dimensions()) && end > begin) {
    begin_ = get_size_from_dims(tensor.get_dimensions()) + begin;
    end_ = get_size_from_dims(tensor.get_dimensions()) + end;
  } else {
    return false;
  }
  //  CudaDeviceContext context(tensor.get_device_id());
  cudaDeviceSynchronize();
  assert(end_ > begin_ && begin_ >= 0 &&
         end_ < static_cast<int>(get_size_from_dims(tensor.get_dimensions())));
  __half host_buff[end_ - begin_];
  cudaMemcpy(host_buff, tensor.get_ptr() + begin_, (end_ - begin_) * sizeof(__half),
             cudaMemcpyDefault);
  std::cout << "Tensor: <";
  for (auto d : tensor.get_dimensions()) {
    std::cout << d << ",";
  }
#ifdef ENABLE_MPI
  int pid(-1), num_procs(-1);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  std::cout << "> in pid: " << pid << " of " << num_procs << " processes." << std::endl;
#else
  std::cout << ">" << std::endl;
#endif
  std::cout << "begin: " << begin_ << " end: " << end_ << std::endl;
  for (int i = 0; i < end_ - begin_; i++) {
    std::cout << __half2float(host_buff[i]) << ",";
  }
  std::cout << std::endl;
  return true;
}

  
  
}  // namespace HugeCTR
