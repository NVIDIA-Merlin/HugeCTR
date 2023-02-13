/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <common.hpp>
#include <condition_variable>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <data_readers/data_container_interface.hpp>
#include <data_readers/data_reader_common.hpp>
#include <data_readers/file_source_parquet.hpp>

#pragma once
namespace HugeCTR {
// TODO Replace raw pointer for tensor2 ?
template <typename T>
class DFContainer : public UnifiedContainer {
 public:
  int device_id_;
  float* raw_dense_;
  int32_t* raw_dense_offset_;
  T* raw_sparse_;
  int32_t* raw_sparse_offset_;

  std::vector<float*> dense_ptr_;
  std::vector<size_t> dense_size_bytes_;
  std::vector<size_t> dense_size_;
  std::vector<int32_t*> dense_offset_ptr_;
  size_t num_dense_;

  std::vector<T*> sparse_ptr_;
  std::vector<size_t> sparse_size_bytes_;
  std::vector<size_t> sparse_size_;
  std::vector<int32_t*> sparse_offset_ptr_;
  size_t num_sparse_;
  size_t num_rows_ = 0;
  size_t max_rows_ = 0;
  unsigned long current_row_ = 0;
  std::vector<size_t> max_dense_size_;
  std::vector<size_t> max_sparse_size_;
  bool allocated_ = false;
  // dense_dim_array_init_ works the way as copy on write of OS page memory management
  bool dense_dim_array_init_ = false;
  cudaStream_t copy_stream_;
  int32_t* h_copy_helper_buffer_;
  // deep copy and reset
  DFContainer& operator=(const DFContainer& rhs);
  DFContainer& operator+=(const DFContainer& rhs);
  cudaStream_t& get_copy_stream();
  void erase_front(size_t rows);
  void erase_back(size_t rows);

  long long get_available_rows() const;
  long unsigned int get_curr_row() const;
  long long get_num_rows() const;
  long long get_max_num_rows() const;
  void forward_row(size_t rows);
  DFContainer() = delete;
  DFContainer(int dev, size_t max_rows, std::vector<size_t> max_dense_size,
              std::vector<size_t> max_sparse_size, size_t dense_size_bytes_total);

  DFContainer(int dev, size_t max_rows, std::vector<size_t> max_sparse_size,
              size_t dense_size_bytes_total);

  void clear();
  bool is_allocated() { return allocated_; }
  void reset_ptrs();
  void init_dense_dim_array(std::vector<size_t> max_dense_size);
  ~DFContainer();
};

template <typename T>
void dump_table_data_to(cudf::table_view& table_view, std::map<int, int>& dense_idx_to_parquet_col_,
                        std::map<int, int>& categorical_idx_parquet_col_,
                        const std::vector<DataReaderSparseParam>& params,
                        std::shared_ptr<DFContainer<T>> df_ptrs_dst,
                        std::vector<size_t>& dense_dim_array, std::vector<int>& one_hot_slot_id,
                        std::vector<int>& sparse_nnz_array);

}  // namespace HugeCTR