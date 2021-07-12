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
#include <common.hpp>
#include <memory>
#include <numeric>
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

  void *get_ptr() { return buffer_->get_ptr(); }
};
using TensorBags2 = std::vector<TensorBag2>;

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

  void set_buffer(const std::shared_ptr<TensorBuffer2> &buffer) { buffer_ = buffer; }

  std::shared_ptr<TensorBuffer2> get_buffer() const { return buffer_; }

  const T *get_ptr() const { return reinterpret_cast<const T *>(buffer_->get_ptr()); }

  T *get_ptr() { return reinterpret_cast<T *>(buffer_->get_ptr()); }
};

template <typename T>
using Tensors2 = std::vector<Tensor2<T>>;

class SparseTensorBag {
  template <typename T>
  friend class SparseTensor;

  std::vector<size_t> dimensions_;
  std::shared_ptr<TensorBuffer2> value_buffer_;
  std::shared_ptr<TensorBuffer2> rowoffset_buffer_;
  std::shared_ptr<size_t> nnz_;
  size_t rowoffset_count_;
  TensorScalarType scalar_type_;

  SparseTensorBag(const std::vector<size_t> &dimensions,
                  const std::shared_ptr<TensorBuffer2> &value_buffer,
                  const std::shared_ptr<TensorBuffer2> &rowoffset_buffer,
                  const std::shared_ptr<size_t> &nnz, const size_t rowoffset_count,
                  TensorScalarType scalar_type)
      : dimensions_(dimensions),
        value_buffer_(value_buffer),
        rowoffset_buffer_(rowoffset_buffer),
        nnz_(nnz),
        rowoffset_count_(rowoffset_count),
        scalar_type_(scalar_type) {}

 public:
  SparseTensorBag() : scalar_type_(TensorScalarType::None) {}
  const std::vector<size_t> &get_dimensions() const { return dimensions_; }
};

template <typename T>
class SparseTensor {
  std::vector<size_t> dimensions_;
  std::shared_ptr<TensorBuffer2> value_buffer_;
  std::shared_ptr<TensorBuffer2> rowoffset_buffer_;
  std::shared_ptr<size_t> nnz_;  // maybe size_t for FixedLengthSparseTensor
  size_t rowoffset_count_;

 public:
  SparseTensor() {}
  SparseTensor(const std::vector<size_t> &dimensions,
               const std::shared_ptr<TensorBuffer2> &value_buffer,
               const std::shared_ptr<TensorBuffer2> &rowoffset_buffer,
               const std::shared_ptr<size_t> &nnz, const size_t rowoffset_count)
      : dimensions_(dimensions),
        value_buffer_(value_buffer),
        rowoffset_buffer_(rowoffset_buffer),
        nnz_(nnz),
        rowoffset_count_(rowoffset_count) {}

  SparseTensor(const Tensor2<T> &value_tensor, const Tensor2<T> &rowoffset_tensor,
               const std::shared_ptr<size_t> nnz)
      : dimensions_(value_tensor.get_dimensions()),
        value_buffer_(value_tensor.get_buffer()),
        rowoffset_buffer_(rowoffset_tensor.get_buffer()),
        nnz_(nnz),
        rowoffset_count_(rowoffset_tensor.get_num_elements()) {}

  static SparseTensor stretch_from(const SparseTensorBag &bag) {
    if (bag.scalar_type_ != TensorScalarTypeFunc<T>::get_type()) {
      CK_THROW_(Error_t::WrongInput, "Inconsistent sparse tensor type");
    }
    return SparseTensor(bag.dimensions_, bag.value_buffer_, bag.rowoffset_buffer_, bag.nnz_,
                        bag.rowoffset_count_);
  }

  SparseTensorBag shrink() const {
    return SparseTensorBag(dimensions_, value_buffer_, rowoffset_buffer_, nnz_, rowoffset_count_,
                           TensorScalarTypeFunc<T>::get_type());
  }

  T *get_value_ptr() { return reinterpret_cast<T *>(value_buffer_->get_ptr()); }

  const T *get_value_ptr() const { return reinterpret_cast<const T *>(value_buffer_->get_ptr()); }

  Tensor2<T> get_value_tensor() const { return Tensor2<T>({*nnz_}, value_buffer_); }

  T *get_rowoffset_ptr() { return reinterpret_cast<T *>(rowoffset_buffer_->get_ptr()); }

  const T *get_rowoffset_ptr() const {
    return reinterpret_cast<const T *>(rowoffset_buffer_->get_ptr());
  }

  Tensor2<T> get_rowoffset_tensor() const {
    return Tensor2<T>({rowoffset_count_}, rowoffset_buffer_);
  }

  const std::vector<size_t> &get_dimensions() const { return dimensions_; }

  size_t max_nnz() const { return get_num_elements_from_dimensions(dimensions_); }

  size_t nnz() const { return *nnz_; }

  std::shared_ptr<size_t> get_nnz_ptr() { return nnz_; }

  size_t rowoffset_count() const { return rowoffset_count_; }
};

template <typename T>
using SparseTensors = std::vector<SparseTensor<T>>;

template <typename T>
class CSR;
namespace sparse_tensor_helper {
namespace cuda {
template <typename T>
void copy_async(SparseTensor<T> &dst, const SparseTensor<T> &src, cudaMemcpyKind kind,
                cudaStream_t stream) {
  CK_CUDA_THROW_(cudaMemcpyAsync(dst.get_value_ptr(), src.get_value_ptr(), src.nnz() * sizeof(T),
                                 kind, stream));

  CK_CUDA_THROW_(cudaMemcpyAsync(dst.get_rowoffset_ptr(), src.get_rowoffset_ptr(),
                                 src.rowoffset_count() * sizeof(T), kind, stream));

  *dst.get_nnz_ptr() = src.nnz();
}

template <typename T>
void copy_async(SparseTensor<T> &dst, const CSR<T> &src, cudaStream_t stream) {
  CK_CUDA_THROW_(cudaMemcpyAsync(dst.get_value_ptr(), src.get_value_tensor().get_ptr(), src.get_num_values() * sizeof(T),
                                 cudaMemcpyHostToDevice, stream));

  CK_CUDA_THROW_(cudaMemcpyAsync(dst.get_rowoffset_ptr(), src.get_row_offset_tensor().get_ptr(),
                                 src.get_row_offset_tensor().get_size_in_bytes(), cudaMemcpyHostToDevice, stream));

  *dst.get_nnz_ptr() = src.get_num_values();
}
}  // namespace cuda
namespace cpu {
  template <typename T>
  void copy(SparseTensor<T> &dst, const SparseTensor<T> &src) {
    memcpy(dst.get_value_ptr(), src.get_value_ptr(), src.nnz() * sizeof(T));
    memcpy(dst.get_rowoffset_ptr(), src.get_rowoffset_ptr(), src.rowoffset_count() * sizeof(T));

    *dst.get_nnz_ptr() = src.nnz();
  }
}
}  // namespace sparse_tensor_helper
}  // namespace HugeCTR
