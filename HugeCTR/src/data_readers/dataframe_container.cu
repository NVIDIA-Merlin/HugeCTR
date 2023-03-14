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

#include <data_readers/dataframe_container.hpp>
#include <fstream>
#include <utils.hpp>
namespace HugeCTR {

template <typename Lambda>
__global__ void vector_scalar_op(int32_t* output, const int32_t* input, size_t len, int32_t val,
                                 Lambda OP) {
  uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gtid < len) {
    output[gtid] = OP(input[gtid], val);
  }
}
template <typename Lambda>
__global__ void vector_scalar_op(int32_t* output, const int32_t* input, size_t len, int32_t* val,
                                 Lambda OP) {
  uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gtid < len) {
    output[gtid] = OP(input[gtid], *val);
  }
}

static void vector_dec(int32_t* dev_array, size_t len, int32_t decrements, cudaStream_t stream) {
  const uint32_t block_size = 256;
  dim3 block(block_size, 1, 1);
  dim3 grid((len + block.x - 1) / block.x, 1, 1);
  auto minus = [] __device__(int32_t a, int32_t b) { return a - b; };

  vector_scalar_op<<<grid, block, 0, stream>>>(dev_array, dev_array, len, decrements, minus);
}

// concate two prefix sum array.
// lhs [0,2,4,5], len_l = 3
// rhs [3,4], len_r = 1
// result : [0,2,4,5, 8,9], len = 4
static void concat_offset(int32_t* lhs_offset, size_t len_l, const int32_t* rhs_offset,
                          size_t len_r, cudaStream_t stream) {
  const uint32_t block_size = 256;
  dim3 block(block_size, 1, 1);
  dim3 grid((len_r + block.x - 1) / block.x, 1, 1);
  auto add = [] __device__(int32_t a, int32_t b) { return a + b; };
  vector_scalar_op<<<grid, block, 0, stream>>>(lhs_offset + len_l + 1, rhs_offset, len_r + 1,
                                               lhs_offset + len_l, add);
}
// max sparse bytes is induced from max_sparse_size
// max_dense_size should be read from row_group_reading_thread
template <typename T>
DFContainer<T>::DFContainer(int dev, size_t max_rows, std::vector<size_t> max_dense_size,
                            std::vector<size_t> max_sparse_size, size_t dense_size_bytes_total)
    : device_id_(dev),
      dense_ptr_({}),
      dense_size_bytes_({}),
      dense_size_({}),
      dense_offset_ptr_({}),
      num_dense_(0),
      sparse_ptr_({}),
      sparse_size_bytes_({}),
      sparse_size_({}),
      sparse_offset_ptr_({}),
      num_sparse_(0),
      num_rows_(0),
      max_rows_(max_rows),
      current_row_(0),
      max_dense_size_(max_dense_size),
      max_sparse_size_(max_sparse_size),
      allocated_(false) {
  if (allocated_) {
    HCTR_LOG(INFO, WORLD, "Allocate new DFContainer fails: already allocated\n");
    return;
  }
  CudaDeviceContext context(device_id_);
  auto num_dense = max_dense_size.size();
  auto num_sparse = max_sparse_size.size();
  this->num_dense_ = num_dense;
  this->num_sparse_ = num_sparse;
  auto sparse_size_bytes_total =
      std::accumulate(max_sparse_size.begin(), max_sparse_size.end(), 0) * max_rows * sizeof(T);
  dense_dim_array_init_ = false;
  allocated_ = true;
  HCTR_LIB_THROW(cudaMalloc(&raw_dense_, dense_size_bytes_total));
  HCTR_LIB_THROW(cudaMalloc(&raw_sparse_, sparse_size_bytes_total));
  HCTR_LIB_THROW(
      cudaHostAlloc(&h_copy_helper_buffer_, num_sparse_ * sizeof(int32_t), cudaHostAllocDefault));
  raw_dense_offset_ = nullptr;
  HCTR_LIB_THROW(cudaMalloc(&raw_sparse_offset_, sizeof(int32_t) * (max_rows + 1) * num_sparse));
  this->max_rows_ = max_rows;
  HCTR_LIB_THROW(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
};
template <typename T>
DFContainer<T>::~DFContainer() {
  this->clear();
  if (allocated_) {
    CudaDeviceContext context(device_id_);
    HCTR_LIB_THROW(cudaFree(raw_dense_));
    HCTR_LIB_THROW(cudaFree(raw_sparse_offset_));
    HCTR_LIB_THROW(cudaFree(raw_sparse_));
    HCTR_LIB_THROW(cudaFreeHost(h_copy_helper_buffer_));
    HCTR_LIB_THROW(cudaStreamDestroy(copy_stream_));
  }
}

// if expected_workers workers has succeeded consuming the data, reset the buffer state as
// ReadyForWrite

template <typename T>
DFContainer<T>& DFContainer<T>::operator=(const DFContainer& rhs) {
  if (!allocated_) {
    HCTR_OWN_THROW(Error_t::UnspecificError, "DFContainer copy fails: allocate first");
  }
  if (rhs.num_dense_ != this->num_dense_) {
    HCTR_OWN_THROW(Error_t::UnspecificError, "DFContainer copy fails: num_dense_ mismatches");
  }
  if (rhs.num_sparse_ != this->num_sparse_) {
    HCTR_OWN_THROW(Error_t::UnspecificError, "DFContainer copy fails: num_sparse_ mismatches");
  }
  CudaDeviceContext context(device_id_);
  this->reset_ptrs();
  for (size_t i = 0; i < num_dense_; i++) {
    float* data_from_ptr = rhs.dense_ptr_[i];
    float* data_to_ptr = this->dense_ptr_[i];
    size_t copy_bytes = rhs.dense_size_bytes_[i];
    HCTR_LIB_THROW(cudaMemcpyAsync(reinterpret_cast<void*>(data_to_ptr),
                                   reinterpret_cast<void*>(data_from_ptr), copy_bytes,
                                   cudaMemcpyDeviceToDevice, copy_stream_));
  }
  for (size_t i = 0; i < num_sparse_; i++) {
    T* data_from_ptr = rhs.sparse_ptr_[i];
    T* data_to_ptr = this->sparse_ptr_[i];
    size_t copy_bytes = rhs.sparse_size_bytes_[i];
    HCTR_LIB_THROW(cudaMemcpyAsync(reinterpret_cast<void*>(data_to_ptr),
                                   reinterpret_cast<void*>(data_from_ptr), copy_bytes,
                                   cudaMemcpyDeviceToDevice, copy_stream_));
  }
  for (size_t i = 0; i < num_sparse_; i++) {
    int32_t* data_from_ptr = rhs.sparse_offset_ptr_[i];
    int32_t* data_to_ptr = this->sparse_offset_ptr_[i];
    size_t copy_bytes = (rhs.num_rows_ + 1) * sizeof(int32_t);
    if (data_from_ptr) {
      HCTR_LIB_THROW(cudaMemcpyAsync(reinterpret_cast<void*>(data_to_ptr),
                                     reinterpret_cast<void*>(data_from_ptr), copy_bytes,
                                     cudaMemcpyDeviceToDevice, copy_stream_));
    } else {
      this->sparse_offset_ptr_[i] = nullptr;
    }
  }
  this->dense_size_ = rhs.dense_size_;
  this->dense_size_bytes_ = rhs.dense_size_bytes_;
  this->sparse_size_ = rhs.sparse_size_;
  this->sparse_size_bytes_ = rhs.sparse_size_bytes_;

  this->num_rows_ = rhs.num_rows_;
  this->current_row_ = 0;
  HCTR_LIB_THROW(cudaStreamSynchronize(copy_stream_));
  return *this;
}

//! caller should do boundary check
template <typename T>
void DFContainer<T>::forward_row(size_t rows) {
  if (this->current_row_ + rows > this->num_rows_) {
    HCTR_LOG(INFO, WORLD, "forward_row fails: caller should do boundary check\n");
    return;
  }
  this->current_row_ += rows;
}

template <typename T>
long unsigned int DFContainer<T>::get_curr_row() const {
  return current_row_;
}
template <typename T>
long long DFContainer<T>::get_available_rows() const {
  auto res = (long long)num_rows_ - (long long)current_row_;
  if (res < 0) {
    // TODO should throw
    HCTR_LOG(INFO, WORLD, "Out of Bound, current_row_ is %d num_rows_ is %d\n", current_row_,
             num_rows_);
  }
  return res;
}
template <typename T>
long long DFContainer<T>::get_num_rows() const {
  return (long long)num_rows_;
}
template <typename T>
long long DFContainer<T>::get_max_num_rows() const {
  return (long long)max_rows_;
}
// can copy across gpu
template <typename T>
DFContainer<T>& DFContainer<T>::operator+=(const DFContainer& rhs) {
  CudaDeviceContext context(device_id_);
  if (!this->allocated_) {
    HCTR_LOG(INFO, WORLD, "DFPointer concate fails: lhs not allocated yet\n");
    return *this;
  }
  if (rhs.current_row_ != 0) {
    HCTR_LOG(INFO, WORLD, "DFPointer concate fails: rhs must be clean\n");
    return *this;
  }
  size_t rows_after_concat = this->get_available_rows() + rhs.get_available_rows();
  // copy assign
  if (!this->num_rows_) {
    *this = rhs;
    return *this;
  }
  if (rows_after_concat > this->max_rows_) {
    HCTR_LOG(INFO, WORLD, "rows_after_concat %d but max_rows_ is %d\n", rows_after_concat,
             max_rows_);
    HCTR_OWN_THROW(Error_t::OutOfMemory, "DFPointer concate fails: no enough space");
  }
  this->erase_front(this->current_row_);
  for (size_t i = 0; i < num_dense_; i++) {
    float* data_from_ptr = rhs.dense_ptr_[i];
    float* data_to_ptr = this->dense_ptr_[i] + this->dense_size_[i];
    size_t copy_bytes = rhs.dense_size_bytes_[i];
    this->dense_size_[i] += rhs.dense_size_[i];
    this->dense_size_bytes_[i] += rhs.dense_size_bytes_[i];

    HCTR_LIB_THROW(cudaMemcpyAsync(reinterpret_cast<void*>(data_to_ptr),
                                   reinterpret_cast<void*>(data_from_ptr), copy_bytes,
                                   cudaMemcpyDeviceToDevice, copy_stream_));
  }
  for (size_t i = 0; i < num_sparse_; i++) {
    T* data_from_ptr = rhs.sparse_ptr_[i];
    T* data_to_ptr = this->sparse_ptr_[i] + this->sparse_size_[i];
    this->sparse_size_[i] += rhs.sparse_size_[i];
    this->sparse_size_bytes_[i] += rhs.sparse_size_bytes_[i];
    size_t copy_bytes = rhs.sparse_size_bytes_[i];
    HCTR_LIB_THROW(cudaMemcpyAsync(reinterpret_cast<void*>(data_to_ptr),
                                   reinterpret_cast<void*>(data_from_ptr), copy_bytes,
                                   cudaMemcpyDeviceToDevice, copy_stream_));
  }
  for (size_t i = 0; i < num_sparse_; i++) {
    if (!this->sparse_offset_ptr_[i]) {
      continue;
    }
    int32_t* rhs_offset = rhs.sparse_offset_ptr_[i];
    size_t len_r = rhs.num_rows_;
    int32_t* lhs_offset = this->sparse_offset_ptr_[i];
    size_t len_l = this->num_rows_;

    HCTR_LIB_THROW(cudaMemcpyAsync(
        reinterpret_cast<void*>(lhs_offset + len_l + 1), reinterpret_cast<void*>(rhs_offset + 1),
        (len_r) * sizeof(int32_t), cudaMemcpyDeviceToDevice, copy_stream_));
    concat_offset(lhs_offset, len_l, lhs_offset + len_l + 1, len_r - 1, copy_stream_);
  }
  HCTR_LIB_THROW(cudaStreamSynchronize(copy_stream_));

  this->current_row_ = 0;
  this->num_rows_ = rows_after_concat;

  return *this;
}

template <typename T>
void DFContainer<T>::clear() {
  dense_ptr_.clear();
  dense_size_bytes_.clear();
  dense_offset_ptr_.clear();
  sparse_ptr_.clear();
  sparse_offset_ptr_.clear();
  sparse_size_bytes_.clear();
  dense_size_.clear();
  sparse_size_.clear();
}
template <typename T>
void DFContainer<T>::init_dense_dim_array(std::vector<size_t> max_dense_size) {
  if (!dense_dim_array_init_) {
    this->max_dense_size_ = max_dense_size;
    dense_dim_array_init_ = true;
    num_dense_ = max_dense_size.size();
  }
}
template <typename T>
cudaStream_t& DFContainer<T>::get_copy_stream() {
  return copy_stream_;
}

template <typename T>
void DFContainer<T>::reset_ptrs() {
  if (!dense_dim_array_init_) {
    HCTR_OWN_THROW(Error_t::NotInitialized, " initialize max_dense_size_ first!\n");
  }
  this->clear();
  this->current_row_ = 0;
  this->num_rows_ = 0;
  this->dense_ptr_.resize(num_dense_, nullptr);
  this->dense_size_bytes_.resize(num_dense_, 0);
  this->dense_size_.resize(num_dense_, 0);
  this->dense_offset_ptr_.resize(num_dense_, nullptr);

  this->sparse_ptr_.resize(num_sparse_, nullptr);
  this->sparse_size_bytes_.resize(num_sparse_, 0);
  this->sparse_size_.resize(num_sparse_, 0);
  this->sparse_offset_ptr_.resize(num_sparse_, nullptr);

  size_t dev_offset = reinterpret_cast<size_t>(raw_dense_);
  for (size_t i = 0; i < num_dense_; i++) {
    this->dense_ptr_[i] = reinterpret_cast<float*>(dev_offset);
    dev_offset += sizeof(float) * max_rows_ * max_dense_size_[i];
  }
  dev_offset = reinterpret_cast<size_t>(raw_sparse_);
  for (size_t i = 0; i < num_sparse_; i++) {
    this->sparse_ptr_[i] = reinterpret_cast<T*>(dev_offset);
    dev_offset += sizeof(T) * max_rows_ * max_sparse_size_[i];
  }
  dev_offset = reinterpret_cast<size_t>(raw_sparse_offset_);
  // sparse offset
  for (size_t i = 0; i < num_sparse_; i++) {
    this->sparse_offset_ptr_[i] = reinterpret_cast<int32_t*>(dev_offset);
    dev_offset += sizeof(int32_t) * (max_rows_ + 1);
  }
}
// erase will impose data copy
// copy on write
template <typename T>
void DFContainer<T>::erase_front(size_t erase_rows) {
  CudaDeviceContext context(device_id_);
  if (erase_rows == 0) {
    return;
  }
  if (!this->allocated_) {
    HCTR_LOG(INFO, WORLD, "erase_front fails: trying to erase before allocation\n");
    return;
  }
  if (erase_rows > this->num_rows_) {
    HCTR_LOG(INFO, WORLD, "erase_front fails: erase out-of-bound\n");
    return;
  }
  long long rows_after_erase = (long long)num_rows_ - (long long)erase_rows;
  this->num_rows_ = rows_after_erase;
  this->current_row_ = 0;
  // copy dense_
  size_t dev_offset = reinterpret_cast<size_t>(raw_dense_);
  for (size_t i = 0; i < num_dense_; i++) {
    float* data_from_ptr = this->dense_ptr_[i] + this->max_dense_size_[i] * erase_rows;
    this->dense_size_[i] -= this->max_dense_size_[i] * erase_rows;
    this->dense_ptr_[i] = reinterpret_cast<float*>(dev_offset);
    dev_offset += sizeof(float) * max_rows_ * max_dense_size_[i];

    float* data_to_ptr = this->dense_ptr_[i];
    size_t copy_bytes =
        this->dense_size_bytes_[i] - this->max_dense_size_[i] * erase_rows * sizeof(float);
    this->dense_size_bytes_[i] = copy_bytes;
    HCTR_LIB_THROW(cudaMemcpyAsync(reinterpret_cast<void*>(data_to_ptr),
                                   reinterpret_cast<void*>(data_from_ptr), copy_bytes,
                                   cudaMemcpyDeviceToDevice, copy_stream_));
  }
  size_t num_variable_sparse = 0;
  for (size_t i = 0; i < num_sparse_; i++) {
    //! one hot
    if (!sparse_offset_ptr_[i]) {
      h_copy_helper_buffer_[i] = erase_rows;
      continue;
    }
    num_variable_sparse++;
    HCTR_LIB_THROW(
        cudaMemcpyAsync(reinterpret_cast<void*>(h_copy_helper_buffer_ + i),
                        reinterpret_cast<void*>(this->sparse_offset_ptr_[i] + erase_rows),
                        sizeof(int32_t), cudaMemcpyDeviceToHost, copy_stream_));
  }
  // no need to sync if all sparse fixed..
  if (num_variable_sparse > 0) HCTR_LIB_THROW(cudaStreamSynchronize(copy_stream_));

  dev_offset = reinterpret_cast<size_t>(raw_sparse_);
  for (size_t i = 0; i < num_sparse_; i++) {
    T* data_from_ptr = this->sparse_ptr_[i] + h_copy_helper_buffer_[i];
    this->sparse_ptr_[i] = reinterpret_cast<T*>(dev_offset);
    dev_offset += sizeof(T) * max_rows_ * max_sparse_size_[i];
    this->sparse_size_[i] -= h_copy_helper_buffer_[i];

    T* data_to_ptr = this->sparse_ptr_[i];
    size_t copy_bytes = this->sparse_size_bytes_[i] - h_copy_helper_buffer_[i] * sizeof(T);
    this->sparse_size_bytes_[i] = copy_bytes;
    HCTR_LIB_THROW(cudaMemcpyAsync(reinterpret_cast<void*>(data_to_ptr),
                                   reinterpret_cast<void*>(data_from_ptr), copy_bytes,
                                   cudaMemcpyDeviceToDevice, copy_stream_));
  }
  dev_offset = reinterpret_cast<size_t>(raw_sparse_offset_);
  // sparse offset
  for (size_t i = 0; i < num_sparse_; i++) {
    // skip because no offset
    if (!this->sparse_offset_ptr_[i]) {
      continue;
    }
    int32_t* data_from_ptr = this->sparse_offset_ptr_[i] + erase_rows;
    this->sparse_offset_ptr_[i] = reinterpret_cast<int32_t*>(dev_offset);
    dev_offset += sizeof(int32_t) * (max_rows_ + 1);
    int32_t* data_to_ptr = this->sparse_offset_ptr_[i];
    size_t copy_bytes = (this->num_rows_ + 1) * sizeof(int32_t);
    HCTR_LIB_THROW(cudaMemcpyAsync(reinterpret_cast<void*>(data_to_ptr),
                                   reinterpret_cast<void*>(data_from_ptr), copy_bytes,
                                   cudaMemcpyDeviceToDevice, copy_stream_));
    // TODO offset global update
    // will work on copy_stream_
    vector_dec(this->sparse_offset_ptr_[i], this->num_rows_ + 1, h_copy_helper_buffer_[i],
               copy_stream_);
  }
  HCTR_LIB_THROW(cudaStreamSynchronize(copy_stream_));
}

// copy on write
template <typename T>
void DFContainer<T>::erase_back(size_t erase_rows) {
  CudaDeviceContext context(device_id_);
  if (erase_rows == 0) {
    return;
  }
  if (!this->allocated_) {
    HCTR_LOG(INFO, WORLD, "erase_back fails: trying to erase before allocation\n");
    return;
  }
  if (erase_rows > this->num_rows_) {
    HCTR_LOG(INFO, WORLD, "erase_back fails: erase out-of-bound\n");
    return;
  }
  long long rows_after_erase = (long long)num_rows_ - (long long)erase_rows;
  this->num_rows_ = rows_after_erase;
  this->current_row_ = 0;
  // copy dense_
  size_t dev_offset = reinterpret_cast<size_t>(raw_dense_);
  for (size_t i = 0; i < num_dense_; i++) {
    float* data_from_ptr = this->dense_ptr_[i];
    this->dense_size_[i] = this->max_dense_size_[i] * rows_after_erase;
    this->dense_ptr_[i] = reinterpret_cast<float*>(dev_offset);
    dev_offset += sizeof(float) * max_rows_ * max_dense_size_[i];

    float* data_to_ptr = this->dense_ptr_[i];
    size_t copy_bytes = this->dense_size_[i] * sizeof(float);
    this->dense_size_bytes_[i] = copy_bytes;
    HCTR_LIB_THROW(cudaMemcpyAsync(reinterpret_cast<void*>(data_to_ptr),
                                   reinterpret_cast<void*>(data_from_ptr), copy_bytes,
                                   cudaMemcpyDeviceToDevice, copy_stream_));
  }

  size_t num_variable_sparse = 0;

  for (size_t i = 0; i < num_sparse_; i++) {
    // one hot!
    if (!sparse_offset_ptr_[i]) {
      h_copy_helper_buffer_[i] = rows_after_erase;
    } else {
      num_variable_sparse++;
      HCTR_LIB_THROW(
          cudaMemcpyAsync(reinterpret_cast<void*>(h_copy_helper_buffer_ + i),
                          reinterpret_cast<void*>(this->sparse_offset_ptr_[i] + rows_after_erase),
                          sizeof(int32_t), cudaMemcpyDeviceToHost, copy_stream_));
      HCTR_LIB_THROW(cudaStreamSynchronize(copy_stream_));
    }
  }
  // no need to sync if all sparse fixed..
  if (num_variable_sparse > 0) HCTR_LIB_THROW(cudaStreamSynchronize(copy_stream_));

  dev_offset = reinterpret_cast<size_t>(raw_sparse_);
  for (size_t i = 0; i < num_sparse_; i++) {
    T* data_from_ptr = this->sparse_ptr_[i];
    size_t copy_bytes = h_copy_helper_buffer_[i] * sizeof(T);

    this->sparse_size_[i] = h_copy_helper_buffer_[i];
    this->sparse_ptr_[i] = reinterpret_cast<T*>(dev_offset);
    dev_offset += sizeof(T) * max_rows_ * max_sparse_size_[i];
    T* data_to_ptr = this->sparse_ptr_[i];
    this->sparse_size_bytes_[i] = copy_bytes;
    HCTR_LIB_THROW(cudaMemcpyAsync(reinterpret_cast<void*>(data_to_ptr),
                                   reinterpret_cast<void*>(data_from_ptr), copy_bytes,
                                   cudaMemcpyDeviceToDevice, copy_stream_));
  }
  dev_offset = reinterpret_cast<size_t>(raw_sparse_offset_);
  // sparse offset
  for (size_t i = 0; i < num_sparse_; i++) {
    // skip because no offset
    if (!this->sparse_offset_ptr_[i]) {
      continue;
    }
    int32_t* data_from_ptr = this->sparse_offset_ptr_[i];
    this->sparse_offset_ptr_[i] = reinterpret_cast<int32_t*>(dev_offset);
    dev_offset += sizeof(int32_t) * (max_rows_ + 1);
    int32_t* data_to_ptr = this->sparse_offset_ptr_[i];
    size_t copy_bytes = (rows_after_erase + 1) * sizeof(int32_t);
    HCTR_LIB_THROW(cudaMemcpyAsync(reinterpret_cast<void*>(data_to_ptr),
                                   reinterpret_cast<void*>(data_from_ptr), copy_bytes,
                                   cudaMemcpyDeviceToDevice, copy_stream_));
  }

  HCTR_LIB_THROW(cudaStreamSynchronize(copy_stream_));
}

// shallow copy
template <typename T>
void dump_table_data_to(cudf::table_view& table_view, std::map<int, int>& dense_idx_to_parquet_col_,
                        std::map<int, int>& categorical_idx_parquet_col_,
                        const std::vector<DataReaderSparseParam>& params,
                        std::shared_ptr<DFContainer<T>> df_ptrs_dst,
                        std::vector<size_t>& dense_dim_array_, std::vector<int>& one_hot_slot_id,
                        std::vector<int>& sparse_nnz_array) {
  // HCTR_LOG(INFO,WORLD,"dump_table_data_to sizeof(T) %zd\n",sizeof(T));
  CudaDeviceContext context(df_ptrs_dst->device_id_);
  size_t num_label_dense_ = dense_idx_to_parquet_col_.size();
  std::vector<int> nums_slots_;
  std::vector<int> nnz_per_slot_from_param_;
  size_t total_num_slots_ = 0;
  for (auto& param : params) {
    nums_slots_.push_back(param.slot_num);
    total_num_slots_ += param.slot_num;
    nnz_per_slot_from_param_.insert(nnz_per_slot_from_param_.end(), param.nnz_per_slot.begin(),
                                    param.nnz_per_slot.end());
  }
  sparse_nnz_array = nnz_per_slot_from_param_;
  std::vector<size_t> dense_dim_array;
  // the data is shallow copied from cudf::column objects, so it does not actually owns the buffer
  df_ptrs_dst->clear();
  size_t num_rows = table_view.num_rows();
  if (df_ptrs_dst->dense_dim_array_init_ && df_ptrs_dst->num_dense_ != num_label_dense_) {
    HCTR_LOG(INFO, WORLD, "Dumping from cudf to ptrs fails: number of columns mismatches\n");
    HCTR_LOG(INFO, WORLD, "df_ptrs_dst->num_dense_ %d,num_label_dense_ %d\n",
             df_ptrs_dst->num_dense_, num_label_dense_);
    return;
  }
  if (df_ptrs_dst->num_sparse_ != total_num_slots_) {
    HCTR_LOG(INFO, WORLD, "Dumping from cudf to ptrs fails: number of columns mismatches\n");
    HCTR_LOG(INFO, WORLD, "df_ptrs_dst->num_sparse_ %d,total_num_slots_ %d \n",
             df_ptrs_dst->num_sparse_, total_num_slots_);
    return;
  }
  std::vector<cudf::column_view> dense_columns_view_ref;
  std::vector<T*> sparse_ptr_cudf(total_num_slots_);
  std::vector<int32_t*> sparse_offset_cudf(total_num_slots_);
  for (size_t k = 0; k < num_label_dense_; k++) {
    dense_columns_view_ref.emplace_back(table_view.column(dense_idx_to_parquet_col_.at(k)));
  }
  for (size_t col = 0; col < dense_columns_view_ref.size(); col++) {
    auto col_view = dense_columns_view_ref[col];
    cudf::type_id type_of_column = col_view.type().id();
    if (type_of_column == cudf::type_to_id<cudf::list_view>()) {
      cudf::column_view value_view = col_view.child(1);
      const size_t dense_scalar_num = value_view.size();
      /* on the premise that all elements are fixed-length.
       *  Thus dense_width = dense_scalar_num / size_df
       */
      size_t dense_width = dense_scalar_num / num_rows;
      if ((dense_scalar_num % num_rows)) {
        HCTR_OWN_THROW(Error_t::WrongInput, "dense dim is not fixed");
      }
      dense_dim_array.push_back(dense_width);
    } else {
      dense_dim_array.push_back(1);
    }
  }
  df_ptrs_dst->init_dense_dim_array(dense_dim_array);
  df_ptrs_dst->reset_ptrs();

  for (size_t col = 0; col < dense_columns_view_ref.size(); col++) {
    auto col_view = dense_columns_view_ref[col];
    cudf::type_id type_of_column = col_view.type().id();
    if (type_of_column == cudf::type_to_id<cudf::list_view>()) {
      cudf::column_view value_view = col_view.child(1);
      cudf::column_view row_offset_view = col_view.child(0);
      if ((static_cast<size_t>(row_offset_view.size()) != (num_rows + 1))) {
        HCTR_OWN_THROW(Error_t::WrongInput, "Parquet Rows not consistent");
      }

      if (value_view.type().id() != cudf::type_to_id<float>()) {
        HCTR_LOG_S(ERROR, WORLD) << "Parquet reader: Vector Dense KeyType Must be List[float]"
                                 << std::endl;
      }
      df_ptrs_dst->dense_size_[col] = (value_view.size());
      df_ptrs_dst->dense_size_bytes_[col] = (cudf::size_of(value_view.type()) * value_view.size());
      df_ptrs_dst->dense_offset_ptr_[col] = nullptr;
      df_ptrs_dst->dense_ptr_[col] = const_cast<float*>(value_view.data<float>());

    } else {
      if (col_view.type().id() != cudf::type_to_id<float>()) {
        HCTR_LOG_S(ERROR, WORLD) << "Parquet reader: Vector Dense KeyType Must be float"
                                 << std::endl;
      }
      df_ptrs_dst->dense_size_[col] = (col_view.size());
      df_ptrs_dst->dense_size_bytes_[col] = (cudf::size_of(col_view.type()) * col_view.size());
      df_ptrs_dst->dense_offset_ptr_[col] = nullptr;
      df_ptrs_dst->dense_ptr_[col] = const_cast<float*>(col_view.data<float>());
    }
  }
  for (size_t col = 0; col < total_num_slots_; col++) {
    auto const& col_view = table_view.column(categorical_idx_parquet_col_.at(col));
    cudf::type_id type_of_column = col_view.type().id();
    if (type_of_column == cudf::type_to_id<cudf::list_view>()) {
      cudf::column_view value_view = col_view.child(1);
      cudf::column_view row_offset_view = col_view.child(0);
      cudf::type_id val_id = value_view.type().id();
      if ((static_cast<size_t>(row_offset_view.size()) != (num_rows + 1))) {
        HCTR_OWN_THROW(Error_t::WrongInput, "Parquet Rows not consistent");
      }
      if (val_id != cudf::type_to_id<int32_t>() && val_id != cudf::type_to_id<int64_t>() &&
          val_id != cudf::type_to_id<uint32_t>() && val_id != cudf::type_to_id<uint64_t>()) {
        HCTR_LOG_S(ERROR, WORLD) << "Parquet worker : cat m-hot KeyType should "
                                    "be uint64/int64/int32/uint32"
                                 << std::endl;
        HCTR_OWN_THROW(Error_t::WrongInput, "Parquet key type error");
      }
      if (cudf::size_of(value_view.type()) != sizeof(T)) {
        HCTR_LOG(ERROR, WORLD, "Parquet col %d type is not consistent with solver.i64_input_key\n",
                 col);
        HCTR_OWN_THROW(Error_t::WrongInput, "Parquet key type error");
      }
      size_t copy_bytes = cudf::size_of(value_view.type()) * value_view.size();

      df_ptrs_dst->sparse_offset_ptr_[col] = const_cast<int32_t*>(row_offset_view.data<int32_t>());
      df_ptrs_dst->sparse_ptr_[col] = const_cast<T*>(value_view.data<T>());
      df_ptrs_dst->sparse_size_[col] = value_view.size();
      df_ptrs_dst->sparse_size_bytes_[col] = copy_bytes;
    } else {
      sparse_nnz_array[col] = 1;
      one_hot_slot_id.push_back((int)(col));
      cudf::type_id val_id = col_view.type().id();
      if (val_id != cudf::type_to_id<int32_t>() && val_id != cudf::type_to_id<int64_t>() &&
          val_id != cudf::type_to_id<uint32_t>() && val_id != cudf::type_to_id<uint64_t>()) {
        HCTR_LOG_S(ERROR, WORLD) << "Parquet worker : cat s-hot KeyType should "
                                    "be uint64/int64/int32/uint32"
                                 << std::endl;
        HCTR_OWN_THROW(Error_t::WrongInput, "Parquet key type error");
      }
      if (cudf::size_of(col_view.type()) != sizeof(T)) {
        HCTR_LOG(ERROR, WORLD, "Parquet col %d type is not consistent with solver.i64_input_key\n",
                 col);
        HCTR_OWN_THROW(Error_t::WrongInput, "Parquet key type error");
      }
      size_t copy_bytes = cudf::size_of(col_view.type()) * col_view.size();
      df_ptrs_dst->sparse_ptr_[col] = const_cast<T*>(col_view.data<T>());
      df_ptrs_dst->sparse_offset_ptr_[col] = nullptr;
      df_ptrs_dst->sparse_size_[col] = col_view.size();
      df_ptrs_dst->sparse_size_bytes_[col] = copy_bytes;
    }
  }
  if (dense_dim_array_.empty()) {
    dense_dim_array_.swap(dense_dim_array);
  } else {
    assert(dense_dim_array_.size() == dense_dim_array.size());
    for (size_t i = 0; i < dense_dim_array_.size(); i++) {
      if (dense_dim_array_[i] != dense_dim_array[i]) {
        HCTR_OWN_THROW(Error_t::WrongInput, "Parquet reader: Dense width not fixed\n");
      }
    }
  }
  df_ptrs_dst->num_rows_ = num_rows;
  HCTR_LIB_THROW(cudaStreamSynchronize(df_ptrs_dst->get_copy_stream()));
}
template void dump_table_data_to<unsigned int>(
    cudf::table_view& table_view, std::map<int, int>& dense_idx_to_parquet_col_,
    std::map<int, int>& categorical_idx_parquet_col_,
    const std::vector<DataReaderSparseParam>& params,
    std::shared_ptr<DFContainer<unsigned int>> df_ptrs_dst, std::vector<size_t>& dense_dim_array_,
    std::vector<int>& one_hot_slot_id, std::vector<int>& sparse_nnz_array);

template void dump_table_data_to<long long>(cudf::table_view& table_view,
                                            std::map<int, int>& dense_idx_to_parquet_col_,
                                            std::map<int, int>& categorical_idx_parquet_col_,
                                            const std::vector<DataReaderSparseParam>& params,
                                            std::shared_ptr<DFContainer<long long>> df_ptrs_dst,
                                            std::vector<size_t>& dense_dim_array_,
                                            std::vector<int>& one_hot_slot_id,
                                            std::vector<int>& sparse_nnz_array);

template struct DFContainer<unsigned int>;
template struct DFContainer<long long>;
}  // namespace HugeCTR