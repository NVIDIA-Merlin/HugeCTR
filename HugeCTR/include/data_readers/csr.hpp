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
#include <cuda_runtime_api.h>
#include <common.hpp>
#include <general_buffer2.hpp>
#include <iostream>
#include <stdexcept>

namespace HugeCTR {

/**
 * @brief A wrapper of simplified CSR buffer and related method.
 *
 * This class contains all the routines of data loading in CSR format, and
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
class CSR {
 private:
  const size_t num_rows_;       /**< num rows. */
  const size_t max_value_size_; /**< number of element of value the CSR matrix will have for num_rows
                                rows. */

  Tensor2<T> row_offset_tensor_;
  Tensor2<T> value_tensor_; /**< a unified buffer for row offset and value. */
  T* row_offset_ptr_; /**< just offset on the buffer, note that the length of it is slot*batchsize+1.
                   */
  T* value_ptr_;      /**< pointer of value buffer. */

  size_t size_of_row_offset_; /**< num of rows in this CSR buffer */
  size_t size_of_value_;      /**< num of values in this CSR buffer */

  size_t check_point_row_;   /**< check point of size_of_row_offset_. */
  size_t check_point_value_; /**< check point of size_of_value__. */
 public:
  /**
   * Ctor
   * @param num_rows num of rows is expected
   * @param max_value_size max size of value buffer.
   */
  CSR(size_t num_rows, size_t max_value_size) : num_rows_(num_rows), max_value_size_(max_value_size),
  size_of_row_offset_(0), size_of_value_(0) {
    static_assert(std::is_same<T, long long>::value || std::is_same<T, unsigned int>::value,
                  "type not support");
    if (max_value_size <= 0 && num_rows <= 0) {
      CK_THROW_(Error_t::WrongInput, "max_value_size <= 0 && num_rows <= 0");
    }

    std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> buff =
        GeneralBuffer2<CudaHostAllocator>::create();
    buff->reserve({num_rows + 1}, &row_offset_tensor_);
    buff->reserve({max_value_size}, &value_tensor_);
    buff->allocate();

    row_offset_ptr_ = row_offset_tensor_.get_ptr();
    value_ptr_ = value_tensor_.get_ptr();
  }
  CSR(const CSR&) = delete;
  CSR& operator=(const CSR&) = delete;
  CSR(CSR&&) = default;

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
    if (size_of_value_ >= max_value_size_)
      CK_THROW_(Error_t::OutOfBound, "CSR out of bound " + std::to_string(max_value_size_) +
                                         "offset" + std::to_string(size_of_value_));
    value_ptr_[size_of_value_] = value;
    size_of_value_++;
  }

  /**
   * Insert a new row to CSR
   * Whenever you want to add a new row, you need to call this.
   * When you have pushed back all the values, you need to call this method
   * again.
   */
  inline void new_row() {  // call before push_back values in this line
    if (size_of_row_offset_ > num_rows_) CK_THROW_(Error_t::OutOfBound, "CSR out of bound");
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
   * To reset the CSR buffer.
   * You need to call reset when you want to overwrite the origial data in this CSR.
   */
  void reset() {
    size_of_value_ = 0;
    size_of_row_offset_ = 0;
  }

  size_t get_num_values() const { return size_of_value_; }
  size_t get_num_rows() const { return num_rows_; }
  Tensor2<T>& get_row_offset_tensor() { return row_offset_tensor_; };
  const Tensor2<T>& get_row_offset_tensor() const { return row_offset_tensor_; };
  Tensor2<T>& get_value_tensor() { return value_tensor_; };
  const Tensor2<T>& get_value_tensor() const { return value_tensor_; };
  void update_value_size(size_t update_size) { size_of_value_ += update_size; };
  void update_row_offset(size_t update_size) { size_of_row_offset_ += update_size; };
};

}  // namespace HugeCTR
