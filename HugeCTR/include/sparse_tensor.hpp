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

#include "core23/data_type.hpp"
#include "core23/tensor.hpp"
#include "core23/tensor_operations.hpp"
#include "data_readers/csr.hpp"
#include "data_readers/csr23.hpp"

namespace HugeCTR {

/*
  this is a semi-sparse tensor -- only the last dimension is sparse.
  For example: below is a 3d sparse tensor, sparsity only occurs along the inner-most (3rd)
  dimension

                                                  slot(2nd dim)

                                            key(3rd dim)

                                          +-----+-----+----+----+----+
                                          | 1,2 | 3   |    | 4  | 5  |
                                          +-----+-----+----+----+----+
                          batch(1st dim)  | 6   | 7,8 | 9  |    | 10 |
                                          +-----+-----+----+----+----+
                                          |     | 11  |    | 12 | 13 |
                                          +-----+-----+----+----+----+
                                          |     | 14  | 15 |    | 16 |
                                          +-----+-----+----+----+----+
*/
class SparseTensor23 {
 private:
  core23::Tensor value_;
  core23::Tensor offset_;
  // TODO Fix me
  std::shared_ptr<int64_t> nnz_;

 public:
  SparseTensor23(){};
  SparseTensor23(const SparseTensor23 &other) = default;
  SparseTensor23(SparseTensor23 &&other) = default;

  SparseTensor23 &operator=(const SparseTensor23 &other) = default;

  // TODO add device, shape, data type check
  SparseTensor23(const core23::Tensor &value, const core23::Tensor &offset,
                 const std::shared_ptr<int64_t> nnz)
      : value_(value), offset_(offset), nnz_(nnz){};
  // TODO add device, shape, data type check
  SparseTensor23(const core23::TensorParams &val_param, const core23::TensorParams &off_param,
                 const std::shared_ptr<int64_t> nnz)
      : value_(val_param), offset_(off_param), nnz_(nnz){};

  SparseTensor23(const core23::TensorParams &val_param, const int64_t slot_num)
      : value_(val_param), nnz_(std::make_shared<int64_t>()) {
    core23::Shape off_shape(val_param.shape());
    int64_t fact = slot_num;
    for (int64_t i = 0; i < off_shape.dims() - 1; i++) {
      fact *= off_shape[i];
    }
    fact++;

    core23::TensorParams off_param = val_param.shape({fact}).data_type(core23::ScalarType::Int32);
    offset_ = core23::Tensor(off_param);
  }
  SparseTensor23(const core23::Shape &val_shape, const core23::ScalarType &val_dtype,
                 const int64_t slot_num, core23::Device device = core23::Device(),
                 core23::TensorParams param = core23::TensorParams())
      : value_(val_shape, val_dtype, param.device(device)), nnz_(std::make_shared<int64_t>()) {
    core23::Shape off_shape(val_shape);

    int64_t fact = slot_num;
    for (int64_t i = 0; i < off_shape.dims() - 1; i++) {
      fact *= off_shape[i];
    }
    fact++;
    // the data type is defaulted to Int32
    offset_ = core23::Tensor(core23::Shape({fact}), core23::ScalarType::Int32,
                             core23::TensorParams().device(device));
  }
  SparseTensor23(const core23::Shape &val_shape, const core23::ScalarType &val_dtype,
                 const core23::ScalarType &off_dtype, const int64_t slot_num,
                 core23::Device device = core23::Device(),
                 core23::TensorParams param = core23::TensorParams())
      : value_(val_shape, val_dtype, param.device(device)), nnz_(std::make_shared<int64_t>()) {
    core23::Shape off_shape(val_shape);

    int64_t fact = slot_num;
    for (int64_t i = 0; i < off_shape.dims() - 1; i++) {
      fact *= off_shape[i];
    }
    fact++;
    offset_ =
        core23::Tensor(core23::Shape({fact}), off_dtype, core23::TensorParams().device(device));
  }
  void *get_value_ptr() { return value_.data(); }
  const void *get_value_ptr() const { return value_.data(); }
  core23::Tensor get_value_tensor() const { return value_; }
  core23::Tensor &get_value_tensor() { return value_; }

  void *get_rowoffset_ptr() { return offset_.data(); }
  const void *get_rowoffset_ptr() const { return offset_.data(); }
  core23::Tensor get_rowoffset_tensor() const { return offset_; }
  core23::Tensor &get_rowoffset_tensor() { return offset_; }

  int64_t max_nnz() const { return value_.num_elements(); }
  int64_t nnz() const { return *nnz_; }
  int64_t rowoffset_count() const { return offset_.num_elements(); }

  std::shared_ptr<int64_t> get_nnz_ptr() { return nnz_; }
};

using SparseTensors23 = std::vector<SparseTensor23>;

struct SparseTensorHelper {
  static void copy_async(SparseTensor23 &dst, const SparseTensor23 &src,
                         core23::CUDAStream stream = core23::CUDAStream()) {
    core23::copy_async(dst.get_value_tensor(), src.get_value_tensor(), stream);
    core23::copy_async(dst.get_rowoffset_tensor(), src.get_rowoffset_tensor(), stream);

    *dst.get_nnz_ptr() = src.nnz();
  }
  template <typename T>
  static void copy_async(SparseTensor23 &dst, const CSR23<T> &src,
                         core23::CUDAStream stream = core23::CUDAStream()) {
    core23::copy_async(dst.get_value_tensor(), src.get_value_tensor(), stream);
    core23::copy_async(dst.get_rowoffset_tensor(), src.get_row_offset_tensor(), stream);
    *dst.get_nnz_ptr() = src.get_num_values();
  }
};
}  // namespace HugeCTR
