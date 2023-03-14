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

#include <cuda_runtime_api.h>

#include <common.hpp>
#include <iostream>
#include <stdexcept>

#include "core23/tensor.hpp"
namespace HugeCTR {

/**
 * @brief A wrapper of simplified CSR23 buffer and related method.
 *
 * This class contains all the routines of data loading in CSR23 format, and
 * export the formated data (in CPU buffer) to users.
 * @verbatim
 * For example data:
 *   4,5,1,2
 *   3,5,1
 *   3,2
 * Will be convert to the form of:
 * row offset: 0,4,7,9
 * value: 4,5,1,2,3,5,1,3,2
 * @endverbatim
 */
template <typename T>
class CSR23 {
 private:
  const size_t num_rows_;       /**< num rows. */
  const size_t max_value_size_; /**< number of element of value the CSR23 matrix will have for
                                num_rows rows. */
  core23::Tensor row_offset_tensor_;
  core23::Tensor value_tensor_;
  T* row_offset_ptr_; /**< just offset on the buffer, note that the length of it is
                       * slot*batchsize+1.
                       */
  T* value_ptr_;      /**< pointer of value buffer. */

  size_t size_of_row_offset_; /**< num of rows in this CSR23 buffer */
  size_t size_of_value_;      /**< num of values in this CSR23 buffer */

  size_t check_point_row_;   /**< check point of size_of_row_offset_. */
  size_t check_point_value_; /**< check point of size_of_value__. */
 public:
  /**
   * Ctor
   * @param num_rows num of rows is expected
   * @param max_value_size max size of value buffer.
   */
  CSR23(size_t num_rows, size_t max_value_size)
      : num_rows_(num_rows),
        max_value_size_(max_value_size),
        size_of_row_offset_(0),
        size_of_value_(0) {
    static_assert(std::is_same<T, long long>::value || std::is_same<T, unsigned int>::value,
                  "type not support");
    core23::BufferParams buffer_params;
    core23::AllocatorParams allocator_params{.pinned = true};
    // buffer_params.channel = "HCTR_CSR_HOST_BUFFER";
    core23::TensorParams common_tensor_params =
        core23::TensorParams().buffer_params(buffer_params).device(core23::DeviceType::CPU);
    row_offset_tensor_ = core23::Tensor(core23::Shape({static_cast<int64_t>(num_rows + 1)}),
                                        core23::ToScalarType<T>::value, common_tensor_params);
    value_tensor_ = core23::Tensor(core23::Shape({static_cast<int64_t>(max_value_size)}),
                                   core23::ToScalarType<T>::value, common_tensor_params);
    // allocate eagerly
    row_offset_ptr_ = row_offset_tensor_.data<T>();
    value_ptr_ = value_tensor_.data<T>();
  }
  CSR23(const CSR23&) = delete;
  CSR23& operator=(const CSR23&) = delete;
  CSR23(CSR23&&) = default;

  inline void push_back_new_row(const T& value) {
    row_offset_ptr_[size_of_row_offset_] = static_cast<T>(size_of_value_);
    size_of_row_offset_++;
    value_ptr_[size_of_value_] = value;
    size_of_value_++;
  }

  /**
   * Push back a value to this object.
   * @param value the value to be pushed back.
   */
  inline void push_back(const T& value) {
    if (size_of_value_ >= max_value_size_) {
      std::ostringstream os;
      os << "CSR23 out of bound " << max_value_size_ << ", offset " << size_of_value_;
      HCTR_OWN_THROW(Error_t::OutOfBound, os.str());
    }
    value_ptr_[size_of_value_] = value;
    size_of_value_++;
  }

  /**
   * Insert a new row to CSR23
   * Whenever you want to add a new row, you need to call this.
   * When you have pushed back all the values, you need to call this method
   * again.
   */
  inline void new_row() {  // call before push_back values in this line
    if (size_of_row_offset_ > num_rows_) {
      HCTR_OWN_THROW(Error_t::OutOfBound, "CSR23 out of bound");
    }
    row_offset_ptr_[size_of_row_offset_] = static_cast<T>(size_of_value_);
    size_of_row_offset_++;
  }

  /**
   * Set check point.
   */
  void set_check_point() {
    check_point_row_ = size_of_row_offset_;
    check_point_value_ = size_of_value_;
  }
  /**
   * Give up current row.
   */
  void roll_back() {
    size_of_row_offset_ = check_point_row_;
    size_of_value_ = check_point_value_;
  }

  /**
   * To reset the CSR23 buffer.
   * You need to call reset when you want to overwrite the origial data in this CSR23.
   */
  void reset() {
    size_of_value_ = 0;
    size_of_row_offset_ = 0;
  }

  size_t get_num_values() const { return size_of_value_; }
  size_t get_num_rows() const { return num_rows_; }
  core23::Tensor& get_row_offset_tensor() { return row_offset_tensor_; };
  const core23::Tensor& get_row_offset_tensor() const { return row_offset_tensor_; };
  core23::Tensor& get_value_tensor() { return value_tensor_; };
  const core23::Tensor& get_value_tensor() const { return value_tensor_; };
  void update_value_size(size_t update_size) { size_of_value_ += update_size; };
  void update_row_offset(size_t update_size) { size_of_row_offset_ += update_size; };
};

}  // namespace HugeCTR