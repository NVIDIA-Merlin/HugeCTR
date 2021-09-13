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
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/data_readers/data_reader_worker_interface.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "data_readers/file_source_parquet.hpp"
#include "resource_managers/resource_manager_ext.hpp"
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#include "data_readers/file_list.hpp"
#include "data_readers/metadata.hpp"
#include "data_readers/parquet_data_converter.hpp"
#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {
template <class T>
class ParquetDataReaderWorker : public IDataReaderWorker {
 private:
  std::vector<DataReaderSparseParam> params_; /**< configuration of data reader sparse input */
  bool skip_read_{false}; /**< set to true when you want to stop the data reading */
  const int MAX_TRY = 10;
  long long records_in_file_;
  long long current_record_index_{0};
  int slots_{0};
  std::vector<long long> slot_offset_;
  std::vector<T> slot_offset_dtype_;
  std::shared_ptr<rmm::device_buffer> slot_offset_device_buf_; /**< GPU buffer w/ slot offset*/
  std::shared_ptr<rmm::mr::device_memory_resource>
      memory_resource_;                            /**< RMM device memory resource object */
  int device_id_{0};                               /**< GPU id for execution */
  cudaStream_t task_stream_;                       /**< Stream for Parquet to csr work  */
  cudaStream_t dense_stream_;                      /**< Stream for Parquet to dense work  */
  std::map<int, int> dense_idx_to_parquet_col_;    /**< dense feature to parquet column idx */
  std::map<int, int> categorical_idx_parquet_col_; /**< cat(slot) to parquet column idx */
  bool thread_resource_allocated_{false};  /**< Flag to set/allocate worker thread resources */
  std::unique_ptr<cudf::table> cached_df_; /**< Cached row_group from Parquet */
  Tensor2<int64_t> host_memory_pointer_staging_;
  /**< Pinned memory for async column dev ptr copy */
  Tensor2<T> host_pinned_csr_inc_; /**< Pinned memory to copy csr push_back values */
  long long row_group_size_;
  long long row_group_index_;
  long long row_group_carry_forward_;
  long long view_offset_;
  std::shared_ptr<ResourceManager> resource_manager_;

  ParquetFileSource* parquet_file_source() const {
    return static_cast<ParquetFileSource*>(source_.get());
  }

  void post_set_source() override {
    is_eof_ = false;
    buffer_->state.store(BufferState::ReadyForWrite);
  }

  void read_new_file() {
    std::set<int> tmp_col_index;
    auto source = parquet_file_source();
    for (int i = 0; i < MAX_TRY; i++) {
      Error_t err = source->next_source();
      if (err == Error_t::Success) {
        auto metadata = source->get_file_metadata();

        if (metadata.get_metadata_status()) {
          auto label_col_names = metadata.get_label_names();
          auto dense_col_names = metadata.get_cont_names();

          if (dense_idx_to_parquet_col_.size() !=
              (label_col_names.size() + dense_col_names.size())) {
            int i = 0;
            dense_idx_to_parquet_col_.clear();
            tmp_col_index.clear();
            for (auto& c : label_col_names) {
              tmp_col_index.insert(c.index);
            }
            for (auto it = tmp_col_index.begin(); it != tmp_col_index.end(); it++) {
              dense_idx_to_parquet_col_.insert(std::make_pair(i, *it));
              i++;
            }
            tmp_col_index.clear();
            for (auto& c : dense_col_names) {
              tmp_col_index.insert(c.index);
            }
            for (auto it = tmp_col_index.begin(); it != tmp_col_index.end(); it++) {
              dense_idx_to_parquet_col_.insert(std::make_pair(i, *it));
              i++;
            }
          }

          tmp_col_index.clear();

          tmp_col_index.clear();

          auto cat_col_names = metadata.get_cat_names();
          if (categorical_idx_parquet_col_.size() != cat_col_names.size()) {
            categorical_idx_parquet_col_.clear();
            int i = 0;
            for (auto& c : cat_col_names) {
              tmp_col_index.insert(c.index);
              categorical_idx_parquet_col_.insert(std::make_pair(i, c.index));
            }
            for (auto it = tmp_col_index.begin(); it != tmp_col_index.end(); it++) {
              categorical_idx_parquet_col_.insert(std::make_pair(i, *it));
              i++;
            }
          }
          records_in_file_ = source->get_num_rows();
          current_record_index_ = 0;
        } else {
          // raise exception
          CK_THROW_(Error_t::BrokenFile, "failed to read a file");
        }
        return;
      } else if (err == Error_t::EndOfFile) {
        // std::cout<<" catch EOF\n";
        throw internal_runtime_error(Error_t::EndOfFile, "EndOfFile");
      }
    }
    CK_THROW_(Error_t::BrokenFile, "failed to read a file");
  }

 public:
  /**
   * Ctor
   */
  ParquetDataReaderWorker(unsigned int worker_id, unsigned int worker_num,
                          // const std::shared_ptr<HeapEx<CSRChunk<T>>>& csr_heap,
                          const std::shared_ptr<GPUResource>& gpu_resource, int* loop_flag,
                          const std::shared_ptr<ThreadBuffer>& buffer, const std::string& file_list,
                          bool repeat, const std::vector<DataReaderSparseParam>& params,
                          const std::vector<long long>& slot_offset, int device_id,
                          const std::shared_ptr<ResourceManager>& resource_manager)
      : IDataReaderWorker(worker_id, worker_num, gpu_resource, !repeat, loop_flag, buffer),
        params_(params),
        slot_offset_(slot_offset),
        device_id_(device_id),
        row_group_carry_forward_(0),
        resource_manager_(resource_manager) {
    CudaCPUDeviceContext ctx(gpu_resource->get_device_id());
    
    std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> buff =
        GeneralBuffer2<CudaHostAllocator>::create();
    buff->reserve({1024}, &host_memory_pointer_staging_);
    buff->reserve({32}, &host_pinned_csr_inc_);
    buff->allocate();

    memory_resource_ = resource_manager_->get_device_rmm_device_memory_resource(device_id_);

    if (worker_id >= worker_num) {
      CK_THROW_(Error_t::BrokenFile, "ParquetDataReaderWorker: worker_id >= worker_num");
    }
    slots_ = 0;
    for (auto& p : params) {
      slots_ += p.slot_num;
    }
    source_ = std::make_shared<ParquetFileSource>(worker_id, worker_num, file_list, repeat);

    // assert((int)slot_offset_.size() == slots_);
    if ((int)slot_offset_.size() < slots_) {
      slot_offset_.resize(slots_, static_cast<long long int>(0));
    }
    for (auto& c : slot_offset_) {
      if ((c >= std::numeric_limits<T>::min()) && (c <= std::numeric_limits<T>::max()))
        slot_offset_dtype_.push_back((T)c);
      else
        CK_THROW_(Error_t::DataCheckError, "Slot offset value exceed the key type range");
    }
  }

  ~ParquetDataReaderWorker() {
    memory_resource_.reset();
    // dont have a good place to destroy resource - before worker threads exits
    if (thread_resource_allocated_) {
      CudaDeviceContext context(device_id_);
      cudaStreamSynchronize(task_stream_);
      cudaStreamSynchronize(dense_stream_);
      cached_df_.reset();
      slot_offset_device_buf_.reset();
      source_.reset();
      cudaStreamDestroy(task_stream_);
      cudaStreamDestroy(dense_stream_);
    }
  }

  /**
   * read a batch of data from data set to heap.
   */
  void read_a_batch();

  /**
   * skip data reading in read_a_batch()
   */
  void skip_read() { skip_read_ = true; }
};

//! Caution. For parquet worker in epoch mode, there's no filling empty samples logic
// for sparse data. The length of output row_offset is (current_batch_size + 1) but not
// batchsize + 1
template <class T>
void ParquetDataReaderWorker<T>::read_a_batch() {
  using dtype_dense = float;

  if (!thread_resource_allocated_) {
    // cant allocate and set resources in constructor
    CK_CUDA_THROW_(cudaSetDevice(device_id_));  // for multiple devices
    CK_CUDA_THROW_(cudaStreamCreateWithFlags(&task_stream_, cudaStreamNonBlocking));
    CK_CUDA_THROW_(cudaStreamCreateWithFlags(&dense_stream_, cudaStreamNonBlocking));
    size_t slot_offset_buf_size = sizeof(T) * slot_offset_dtype_.size();
    slot_offset_device_buf_ = std::make_shared<rmm::device_buffer>(
        slot_offset_buf_size, task_stream_, memory_resource_.get());

    CK_CUDA_THROW_(cudaMemcpyAsync(slot_offset_device_buf_->data(), slot_offset_dtype_.data(),
                                   slot_offset_buf_size, cudaMemcpyHostToDevice, task_stream_));
    thread_resource_allocated_ = true;
  }
  int current_batch_size = 0;

  try {
    auto source = parquet_file_source();
    if (!source->is_open()) {
      read_new_file();
    }
    // dense_buffers store only data for local gpus, clipped by batch_size_start_idx &
    // batch_size_end_idx
    const int dense_start = buffer_->batch_size_start_idx;  // dense buffer
    const int dense_end = buffer_->batch_size_end_idx;      // dense buffer
    int batch_size = buffer_->batch_size;
    const int label_dense_dim = buffer_->label_dim + buffer_->dense_dim;
    size_t param_num = buffer_->param_num;
    if (!skip_read_) {
      if (!wait_until_h2d_ready()) return;
      auto dst_dense_tensor = Tensor2<dtype_dense>::stretch_from(buffer_->device_dense_buffers);

      long long elements_to_read = batch_size;
      std::vector<cudf::table_view> table_view_for_concat;

      // have enough row inc row_group_index_ else slice and concat to next DF
      while (elements_to_read > 0) {
        if (!cached_df_ || (row_group_size_ == row_group_index_)) {
          auto tbl_w_metadata = source->read(-1, memory_resource_.get());
          if (row_group_carry_forward_ > 0 && table_view_for_concat.size() > 0) {
            std::vector<cudf::table_view> table_views_for_concat{table_view_for_concat[0],
                                                                 tbl_w_metadata.tbl->view()};
            // swap here will automatically release previous cached DF
            (cudf::concatenate(table_views_for_concat, memory_resource_.get())).swap(cached_df_);
            // roll back if concat happens between group
            current_record_index_ -= row_group_carry_forward_;
            row_group_size_ = cached_df_->num_rows();
            row_group_carry_forward_ = 0;
            table_view_for_concat.clear();
          } else {
            cached_df_.reset();
            tbl_w_metadata.tbl.swap(cached_df_);
            row_group_size_ = cached_df_->num_rows();
          }
          row_group_index_ = 0;
          view_offset_ = 0;
        }

        if ((row_group_size_ - row_group_index_) >= elements_to_read) {
          row_group_index_ += elements_to_read;
          current_record_index_ += elements_to_read;
          elements_to_read = 0;
          current_batch_size = 0;
        } else if (row_group_index_ < row_group_size_) {
          long long avail_rows = (row_group_size_ - row_group_index_);

          if (avail_rows < row_group_size_) {
            // slice and add to concat queue
            std::vector<cudf::size_type> slice_indices{
                (cudf::size_type)row_group_index_,
                (cudf::size_type)(row_group_index_ + avail_rows)};

            table_view_for_concat = cudf::slice(cached_df_->view(), slice_indices);
          } else {
            table_view_for_concat.emplace_back(std::move(cached_df_->view()));
          }
          row_group_index_ += avail_rows;
          current_record_index_ += avail_rows;
          row_group_carry_forward_ = avail_rows;
          current_batch_size = avail_rows;
        }

        // read_next_file if needed
        if (current_record_index_ >= records_in_file_) {
          try {
            read_new_file();  // set current_record_index_ to zero; can throw EOF

            // we merge last slice to next file, so need to move current_record_index_ forward
            current_record_index_ += row_group_carry_forward_;
            if (row_group_carry_forward_ > 0)
              records_in_file_ += row_group_carry_forward_;
            else if ((row_group_size_ - row_group_index_) > 0)
              records_in_file_ += (row_group_size_ - row_group_index_);
          } catch (const internal_runtime_error& rt_err) {
            Error_t err = rt_err.get_error();
            // last file
            if (err == Error_t::EndOfFile) {
              elements_to_read = 0;
            } else {
              std::cerr << rt_err.what() << std::endl;
              throw;
            }
          }
        }
      }

      cudf::table_view data_view = cached_df_->view();
      if (current_batch_size == 0) {
        current_batch_size = batch_size;
      }
      buffer_->current_batch_size = current_batch_size;
      // potential bugs here ??? it's hard to count how many pointers are used....
      int64_t num_of_pointer_staging =
          2 * (buffer_->label_dim + buffer_->label_dim + 1) + 2 * params_.size() + 2 * slots_;

      // PinnedBuffer extend on unique_ptr cant realloc properly and safely (cudaContext)
      if ((int64_t)host_memory_pointer_staging_.get_num_elements() < num_of_pointer_staging)
        CK_THROW_(Error_t::UnspecificError, "Parquet reader worker: not enough pinned storge");
      // calculate rows for each buffer
      size_t device_staging_dense_size = dense_end - dense_start;
      device_staging_dense_size *= ((size_t)label_dense_dim * sizeof(dtype_dense));
      std::vector<cudf::column_view> dense_columns_view_ref;
      for (int k = 0; k < label_dense_dim; k++) {
        dense_columns_view_ref.emplace_back(
            std::move(data_view.column(dense_idx_to_parquet_col_[k])));
        if (dense_columns_view_ref.back().type().id() != cudf::type_to_id<dtype_dense>()) {
          CK_THROW_(Error_t::WrongInput,
                    "Parquet reader: Dense KeyType and Parquet column type don't match");
        }
      }

      int offset_start = std::min(dense_start, current_batch_size);
      int offset_end = std::min(dense_end, current_batch_size);
      int samples_to_be_transposed = offset_end - offset_start;
      std::vector<dtype_dense*> dense_column_data_ptr;
      for (int k = 0; k < label_dense_dim; k++) {
        dtype_dense* column_ptr =
            const_cast<dtype_dense*>(dense_columns_view_ref[k].data<dtype_dense>());
        column_ptr =
            // only proceed dense for local gpu
            reinterpret_cast<dtype_dense*>((size_t)column_ptr + sizeof(dtype_dense) * view_offset_ +
                                           sizeof(dtype_dense) * offset_start);
        dense_column_data_ptr.push_back(column_ptr);
      }

      std::deque<rmm::device_buffer> rmm_resources;
      convert_parquet_dense_columns(dense_column_data_ptr, label_dense_dim,
                                    samples_to_be_transposed, dense_start, dense_end,
                                    reinterpret_cast<dtype_dense*>(dst_dense_tensor.get_ptr()),
                                    host_memory_pointer_staging_.get_ptr(), rmm_resources,
                                    memory_resource_.get(), dense_stream_);
      const int num_csr_buffers = param_num;
      auto dst_sparse_tensors = buffer_->device_sparse_buffers;
      // device output pointer
      std::vector<void*> device_csr_value_buffers(num_csr_buffers);
      std::vector<void*> device_csr_row_offset_buffers(num_csr_buffers);

      for (int k = 0; k < num_csr_buffers; k++) {
        auto dst_sparse_tensor = SparseTensor<T>::stretch_from(dst_sparse_tensors[k]);
        device_csr_value_buffers[k] = reinterpret_cast<void*>(dst_sparse_tensor.get_value_ptr());
        device_csr_row_offset_buffers[k] =
            reinterpret_cast<void*>(dst_sparse_tensor.get_rowoffset_ptr());
        size_t size_of_csr_roff_buffer = sizeof(T) * (params_[k].slot_num * batch_size + 1);
        CK_CUDA_THROW_(cudaMemsetAsync(device_csr_row_offset_buffers[k], 0, size_of_csr_roff_buffer,
                                       task_stream_));
      }

      int param_id = 0;
      int df_column_id = 0;

      int64_t* pinned_staging_buffer =
          reinterpret_cast<int64_t*>((size_t)host_memory_pointer_staging_.get_ptr() +
                                     sizeof(int64_t) * 2 * (label_dense_dim + 1));
      size_t pinned_buffer_offset_count = 0;

      for (auto& param : params_) {
        int slot_count = param.slot_num;
        std::vector<cudf::column_view> cat_columns_view_ref;
        for (int k = 0; k < slot_count; k++) {
          cat_columns_view_ref.emplace_back(
              std::move(data_view.column(categorical_idx_parquet_col_[df_column_id + k])));
        }

        /*
          slots input data: value, row_offset
          for m-hot slots , row_offset = column::child(0).data , value = column::child(1).data
          for s-hot row_offset = nullptr, value = column.data()
        */
        std::vector<T*> cat_column_data_ptr;
        std::vector<int32_t*> cat_column_row_offset_ptr;

        for (int k = 0; k < slot_count; k++) {
          cudf::type_id type_of_column = cat_columns_view_ref[k].type().id();

          // m-hot
          if (type_of_column == cudf::type_to_id<cudf::list_view>()) {
            cudf::column_view row_offset_view = cat_columns_view_ref[k].child(0);
            cudf::column_view value_view = cat_columns_view_ref[k].child(1);
            if (cudf::size_of(value_view.type()) != sizeof(T)) {
              CK_THROW_(Error_t::WrongInput,
                        "Parquet worker : cat m-hot KeyType does not match Parquet column type");
            }
            if (row_offset_view.type().id() != cudf::type_to_id<int32_t>()) {
              CK_THROW_(Error_t::WrongInput, "Parquet worker : row_offset type should be int32_t");
            }
            int32_t* row_ptr = const_cast<int32_t*>(row_offset_view.data<int32_t>());
            T* column_ptr = const_cast<T*>(value_view.data<T>());
            cat_column_row_offset_ptr.push_back(row_ptr);
            cat_column_data_ptr.push_back(column_ptr);
          }
          // s-hot
          else if (cudf::size_of(cat_columns_view_ref[k].type()) == sizeof(T)) {
            T* column_ptr = const_cast<T*>(cat_columns_view_ref[k].data<T>());
            column_ptr = reinterpret_cast<T*>((size_t)column_ptr);
            cat_column_data_ptr.push_back(column_ptr);
            cat_column_row_offset_ptr.push_back(nullptr);
          } else {
            CK_THROW_(Error_t::WrongInput,
                      "Parquet worker : cat m-hot KeyType does not match Parquet column type");
          }
        }
        T* dev_slot_offset_ptr = reinterpret_cast<T*>((size_t)slot_offset_device_buf_->data() +
                                                      (df_column_id * sizeof(T)));
        int64_t* pinned_staging_buffer_param = reinterpret_cast<int64_t*>(
            (size_t)pinned_staging_buffer + pinned_buffer_offset_count * sizeof(int64_t));
        {
          // optimize converter in the future when slots nnz for current param_id is fixed
          pinned_buffer_offset_count += convert_parquet_cat_columns(
              cat_column_data_ptr, cat_column_row_offset_ptr, view_offset_, param_num, param_id,
              param.max_nnz, slot_count, current_batch_size, resource_manager_->get_process_id(),
              resource_manager_, device_csr_value_buffers, device_csr_row_offset_buffers,
              pinned_staging_buffer_param, dev_slot_offset_ptr, rmm_resources,
              memory_resource_.get(), task_stream_);
        }
        df_column_id += param.slot_num;
        param_id++;
      }

      CK_CUDA_THROW_(cudaStreamSynchronize(task_stream_));

      // get nnz info
      for (size_t buffer_id = 0; buffer_id < param_num; buffer_id++) {
        //! caution ! not batch_size but current_batch_size
        int64_t last_row = current_batch_size * params_[buffer_id].slot_num;
        int64_t dev_row_pointer =
            reinterpret_cast<int64_t>(device_csr_row_offset_buffers[buffer_id]) +
            last_row * sizeof(T);

        auto dst_sparse_tensor = SparseTensor<T>::stretch_from(dst_sparse_tensors[buffer_id]);
        CK_CUDA_THROW_(cudaMemcpyAsync(reinterpret_cast<void*>(host_pinned_csr_inc_.get_ptr()),
                                       reinterpret_cast<void*>(dev_row_pointer), sizeof(T) * 1,
                                       cudaMemcpyDeviceToHost, task_stream_));
        CK_CUDA_THROW_(cudaStreamSynchronize(task_stream_));
        size_t nnz = static_cast<size_t>(host_pinned_csr_inc_.get_ptr()[0]);
        *dst_sparse_tensor.get_nnz_ptr() = nnz;
      }

      CK_CUDA_THROW_(cudaStreamSynchronize(task_stream_));
      CK_CUDA_THROW_(cudaStreamSynchronize(dense_stream_));

      view_offset_ = row_group_index_;
    }
    buffer_->state.store(BufferState::ReadyForRead);
  } catch (const internal_runtime_error& rt_err) {
    Error_t err = rt_err.get_error();
    if (err == Error_t::EndOfFile) {
      if (!wait_until_h2d_ready()) return;
      buffer_->current_batch_size = 0;
      assert(buffer_->state.load() == BufferState::Writing);
      is_eof_ = true;
      buffer_->state.store(BufferState::ReadyForRead);

      while (buffer_->state.load() != BufferState::ReadyForWrite) {
        usleep(2);
        if (*loop_flag_ == 0) return;  // in case main thread exit
      }
      return;  // need this return to run from begining
    } else {
      throw;
    }
  }
  return;
}

}  // namespace HugeCTR
