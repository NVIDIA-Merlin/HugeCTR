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

namespace HugeCTR {

enum class TensorScalarType { None, Void, Float32, Float16, Int64, UInt64, Int32, UInt32, Size_t };

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

template <>
struct TensorScalarTypeFunc<size_t> {
  static TensorScalarType get_type() { return TensorScalarType::Size_t; }
};

template <>
struct TensorScalarTypeFunc<long long> {
  static TensorScalarType get_type() { return TensorScalarType::Int64; }
};

template <>
struct TensorScalarTypeFunc<unsigned int> {
  static TensorScalarType get_type() { return TensorScalarType::UInt32; }
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
  const std::vector<size_t> &get_dimensions() const { return dimensions_; }
  void* get_ptr() { return buffer_->get_ptr(); }
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

  Tensor2(Tensor2 const &other) = default;
  Tensor2 &operator=(Tensor2 const &other) = default;

  Tensor2(Tensor2 &&other) = default;
  Tensor2 &operator=(Tensor2 &&other) = default;

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

  void set_buffer(const std::shared_ptr<TensorBuffer2>& buffer) { buffer_ = buffer; }

  const T *get_ptr() const { return reinterpret_cast<const T *>(buffer_->get_ptr()); }

  T *get_ptr() { return reinterpret_cast<T *>(buffer_->get_ptr()); }

  void reset_shape(const std::vector<size_t> &new_dimensions) {
    try {
      size_t new_num_elements = get_num_elements_from_dimensions(new_dimensions);
      if (new_num_elements > num_elements_) {
        CK_THROW_(Error_t::WrongInput, "new dimensions out of memory");
      }
      dimensions_ = new_dimensions;
      num_elements_ = new_num_elements;
    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
    }
  }
};

template <typename T>
using Tensors2 = std::vector<Tensor2<T>>;

template <typename T>
Tensors2<T> bags_to_tensors(const std::vector<TensorBag2> &bags) {
  Tensors2<T> tensors;
  for (const auto &bag : bags) {
    tensors.push_back(Tensor2<T>::stretch_from(bag));
  }
  return tensors;
}

template <typename T>
std::vector<TensorBag2> tensors_to_bags(const Tensors2<T> &tensors) {
  std::vector<TensorBag2> bags;
  for (const auto &tensor : tensors) {
    bags.push_back(tensor.shrink());
  }
  return bags;
}

}  // namespace HugeCTR
