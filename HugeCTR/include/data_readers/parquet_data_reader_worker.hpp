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
  const unsigned int worker_id_{0};
  const unsigned int worker_num_{0};
  size_t buffer_length_;                          /**< buffer size for internal use */
  std::shared_ptr<HeapEx<CSRChunk<T>>> csr_heap_; /**< heap to cache the data set */
  std::vector<DataReaderSparseParam> params_;     /**< configuration of data reader sparse input */
  bool skip_read_{false}; /**< set to true when you want to stop the data reading */
  const int MAX_TRY = 10;
  long long records_in_file_;
  long long current_record_index_{0};
  int slots_{0};
  const std::vector<long long> slot_offset_;
  std::vector<T> slot_offset_dtype_;
  std::shared_ptr<rmm::mr::device_memory_resource>
      memory_resource_;                            /**< RMM device memory resource object */
  int device_id_{0};                               /**< GPU id for execution */
  cudaStream_t task_stream_;                       /**< Stream for Parquet to csr work  */
  cudaStream_t dense_stream_;                      /**< Stream for Parquet to dense work  */
  std::map<int, int> dense_idx_to_parquet_col_;    /**< dense feature to parquet column idx */
  std::map<int, int> categorical_idx_parquet_col_; /**< cat(slot) to parquet column idx */
  std::shared_ptr<rmm::device_buffer> slot_offset_device_buf_; /**< GPU buffer w/ slot offset*/
  bool thread_resource_allocated_{false};  /**< Flag to set/allocate worker thread resources */
  std::unique_ptr<cudf::table> cached_df_; /**< Cached row_group from Parquet */
  Tensor2<int64_t> host_memory_pointer_staging_;
  /**< Pinned memory for async column dev ptr copy */
  Tensor2<uint32_t> host_pinned_csr_inc_; /**< Pinned memory to copy csr push_back values */
  long long row_group_size_;
  long long row_group_index_;
  long long row_group_carry_forward_;
  long long view_offset_;
  std::shared_ptr<ResourceManager> resource_manager_;

  ParquetFileSource* parquet_file_source() const { 
    return static_cast<ParquetFileSource*>(source_.get());
  }

  void read_new_file() {
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
            dense_idx_to_parquet_col_.clear();
            int i = 0;
            for (auto& c : label_col_names) {
              dense_idx_to_parquet_col_.insert(std::make_pair(i, c.index));
              i++;
            }
            for (auto& c : dense_col_names) {
              dense_idx_to_parquet_col_.insert(std::make_pair(i, c.index));
              i++;
            }
          }

          auto cat_col_names = metadata.get_cat_names();
          if (categorical_idx_parquet_col_.size() != cat_col_names.size()) {
            categorical_idx_parquet_col_.clear();
            int i = 0;
            for (auto& c : cat_col_names) {
              categorical_idx_parquet_col_.insert(std::make_pair(i, c.index));
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
      }
    }
    CK_THROW_(Error_t::BrokenFile, "failed to read a file");
  }

 public:
  /**
   * Ctor
   */
  ParquetDataReaderWorker(unsigned int worker_id, unsigned int worker_num,
                          const std::shared_ptr<HeapEx<CSRChunk<T>>>& csr_heap,
                          const std::string& file_list, size_t buffer_length,
                          const std::vector<DataReaderSparseParam>& params,
                          const std::vector<long long>& slot_offset,
                          int device_id,
                          const std::shared_ptr<ResourceManager>& resource_manager)
      : worker_id_(worker_id),
        worker_num_(worker_num),
        buffer_length_(buffer_length),
        csr_heap_(csr_heap),
        params_(params),
        slot_offset_(slot_offset),
        device_id_(device_id),
        row_group_carry_forward_(0),
        resource_manager_(resource_manager) {
    std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> buff =
        GeneralBuffer2<CudaHostAllocator>::create();
    buff->reserve({1024}, &host_memory_pointer_staging_);
    buff->reserve({32}, &host_pinned_csr_inc_);
    buff->allocate();

    ResourceManagerExt* ext = dynamic_cast<ResourceManagerExt*>(resource_manager_.get());
    if(ext == nullptr) {
      CK_THROW_(Error_t::WrongInput, "Invalid ResourceManager");
    }
    memory_resource_ = ext->get_device_rmm_device_memory_resource(device_id_);

    if (worker_id >= worker_num) {
      CK_THROW_(Error_t::BrokenFile, "ParquetDataReaderWorker: worker_id >= worker_num");
    }
    slots_ = 0;
    for (auto& p : params) {
      slots_ += p.slot_num;
    }
    source_ = std::make_shared<ParquetFileSource>(worker_id, worker_num, file_list);

    assert((int)slot_offset_.size() == slots_);
    for (auto& c : slot_offset_) {
      if ((c >= std::numeric_limits<T>::min()) && (c <= std::numeric_limits<T>::max()))
        slot_offset_dtype_.push_back((T)c);
      else
        CK_THROW_(Error_t::DataCheckError, "Slot offset value exceed the key type range");
    }
  }

  ~ParquetDataReaderWorker() {
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

template <class T>
void ParquetDataReaderWorker<T>::read_a_batch() {
  // get csr chunk
  // staging on cpu csr_chunk heap in first prototype
  // will shift to gpu tensors in next iteration - need collector changes
  // and possible gpu location/memory issues??
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

  try {
    auto source = parquet_file_source();
    if (!source->is_open()) {
      read_new_file();
    }
    CSRChunk<T>* csr_chunk = csr_heap_->checkout_free_chunk(worker_id_);

    if (!skip_read_) {
      csr_chunk->set_current_batchsize(csr_chunk->get_batchsize());
      int batch_size = csr_chunk->get_batchsize();

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
          // elements_to_read -= avail_rows; //
          row_group_carry_forward_ = avail_rows;
        }

        // read_next_file if needed
        if (current_record_index_ >= records_in_file_) {
          read_new_file();  // set current_record_index_ to zero
          if (row_group_carry_forward_ > 0)
            records_in_file_ += row_group_carry_forward_;
          else if ((row_group_size_ - row_group_index_) > 0)
            records_in_file_ += (row_group_size_ - row_group_index_);
        }
      }

      cudf::table_view data_view = cached_df_->view();

      Tensors2<float>& label_dense_buffers = csr_chunk->get_label_buffers();  // get_num_elements
      const int label_dense_dim = csr_chunk->get_label_dense_dim();
      int num_dense_buffers = label_dense_buffers.size();

      int64_t num_of_pointer_staging =
          2 * (csr_chunk->get_label_dense_dim() + num_dense_buffers) +
          3 * (int)csr_chunk->get_csr_buffers().size() * params_.size() + slots_ +
          +csr_chunk->get_num_devices();

      // PinnedBuffer extend on unique_ptr cant realloc properly and safely (cudaContext)
      if ((int64_t)host_memory_pointer_staging_.get_num_elements() < num_of_pointer_staging)
        CK_THROW_(Error_t::UnspecificError, "Parquet reader worker: not enough pinned storge");

      // calculate rows for each buffer
      size_t device_staging_dense_size = ((batch_size - 1) / num_dense_buffers + 1);
      device_staging_dense_size *= ((size_t)label_dense_dim * sizeof(dtype_dense));

      std::vector<rmm::device_buffer> dense_data_buffers(num_dense_buffers);
      for (int k = 0; k < num_dense_buffers; k++) {
        dense_data_buffers[k] =
            rmm::device_buffer(device_staging_dense_size, dense_stream_, memory_resource_.get());
      }

      std::vector<cudf::column_view> dense_columns_view_ref;
      for (int k = 0; k < label_dense_dim; k++) {
        dense_columns_view_ref.emplace_back(
            std::move(data_view.column(dense_idx_to_parquet_col_[k])));
        if (dense_columns_view_ref.back().type().id() != cudf::type_to_id<dtype_dense>()) {
          CK_THROW_(Error_t::WrongInput,
                    "Parquet reader: Dense KeyType and Parquet column type don't match");
        }
      }

      std::vector<dtype_dense*> dense_column_data_ptr;
      for (int k = 0; k < label_dense_dim; k++) {
        dtype_dense* column_ptr =
            const_cast<dtype_dense*>(dense_columns_view_ref[k].data<dtype_dense>());
        column_ptr =
            reinterpret_cast<dtype_dense*>((size_t)column_ptr + sizeof(dtype_dense) * view_offset_);
        dense_column_data_ptr.push_back(column_ptr);
      }

      std::deque<rmm::device_buffer> rmm_resources;

      // kernel from dense_column_data_ptr to device_staging_dense_size with data transpose
      convert_parquet_dense_columns(dense_column_data_ptr, label_dense_dim, batch_size,
                                    num_dense_buffers, dense_data_buffers,
                                    host_memory_pointer_staging_.get_ptr(), rmm_resources,
                                    memory_resource_.get(), dense_stream_);

      for (int k = 0; k < num_dense_buffers; k++) {
        Tensor2<float>& buf = label_dense_buffers[k];
        size_t copy_size = buf.get_num_elements() * sizeof(dtype_dense);
        CK_CUDA_THROW_(cudaMemcpyAsync(buf.get_ptr(), dense_data_buffers[k].data(), copy_size,
                                       cudaMemcpyDeviceToHost, dense_stream_));
      }

      // device/buffer wise distribution is important
      // label device buffer - as many as csr buffers
      // transpose on gpu kernel
      // memcpy to csr label_dense_buffers

      csr_chunk->apply_to_csr_buffers(&CSR<T>::reset);
      csr_chunk->apply_to_csr_buffers(&CSR<T>::set_check_point);
      auto& csr_buffers = csr_chunk->get_csr_buffers();
      int num_csr_buffers = csr_buffers.size();
      std::vector<rmm::device_buffer> device_csr_value_buffers(num_csr_buffers);
      std::vector<rmm::device_buffer> device_csr_row_offset_buffers(num_csr_buffers);

      // caution for mhot
      // use nnz = 1, so device allocations needn't be variable length
      // @future - parquet multi-hot change these allocation size
      size_t size_of_csr_buffer = sizeof(T) * slots_ * batch_size;
      size_t size_of_csr_roff_buffer = sizeof(T) * (slots_ * batch_size + 1);
      for (int k = 0; k < num_csr_buffers; k++) {
        device_csr_value_buffers[k] =
            rmm::device_buffer(size_of_csr_buffer, task_stream_, memory_resource_.get());
        device_csr_row_offset_buffers[k] =
            rmm::device_buffer(size_of_csr_roff_buffer, task_stream_, memory_resource_.get());

        CK_CUDA_THROW_(cudaMemsetAsync(device_csr_row_offset_buffers[k].data(), 0,
                                       size_of_csr_roff_buffer, task_stream_));
      }

      size_t embed_param_offset_buf_size = num_csr_buffers * sizeof(uint32_t);
      rmm::device_buffer device_embed_param_start_offset(embed_param_offset_buf_size, task_stream_,
                                                         memory_resource_.get());

      CK_CUDA_THROW_(cudaMemsetAsync(device_embed_param_start_offset.data(), 0,
                                     embed_param_offset_buf_size, task_stream_));

      // categoricals
      // param_id unroll with param : params_ --> param.slot_num
      int csr_chunk_devices = csr_chunk->get_num_devices();
      int param_id = 0;
      int df_column_id = 0;
      int param_size = params_.size();

      int64_t* pinned_staging_buffer = reinterpret_cast<int64_t*>(
          (size_t)host_memory_pointer_staging_.get_ptr() +
          sizeof(int64_t) * 2 * (csr_chunk->get_label_dense_dim() + num_dense_buffers));
      size_t pinned_buffer_offset_count = 0;

      for (auto& param : params_) {
        bool distributed_slot = false;
        int slot_count = param.slot_num;
        std::vector<cudf::column_view> cat_columns_view_ref;
        for (int k = 0; k < slot_count; k++) {
          cat_columns_view_ref.emplace_back(
              std::move(data_view.column(categorical_idx_parquet_col_[df_column_id + k])));
          if (cudf::size_of(cat_columns_view_ref.back().type()) != sizeof(T)) {
            CK_THROW_(Error_t::WrongInput,
                      "Parquet reader: Slot KeyType and Parquet column type don't match");
          }
        }

        std::vector<T*> cat_column_data_ptr;
        for (int k = 0; k < slot_count; k++) {
          T* column_ptr = const_cast<T*>(cat_columns_view_ref[k].data<T>());
          column_ptr = reinterpret_cast<T*>((size_t)column_ptr + sizeof(T) * view_offset_);
          cat_column_data_ptr.push_back(column_ptr);
        }

        T* dev_slot_offset_ptr = reinterpret_cast<T*>((size_t)slot_offset_device_buf_->data() +
                                                      (df_column_id * sizeof(T)));

        int64_t* pinned_staging_buffer_param = reinterpret_cast<int64_t*>(
            (size_t)pinned_staging_buffer + pinned_buffer_offset_count * sizeof(int64_t));
        if (param.type == DataReaderSparse_t::Distributed) {
          distributed_slot = true;
          // added row to all device csr buffer based on param_id
          // distribute input based on input read feature % num_devices
          // csr_chunk->get_csr_buffer(param_id, dev_id).push_back(local_id);
          pinned_buffer_offset_count += convert_parquet_cat_columns(
              cat_column_data_ptr, param_size, param_id, slot_count, batch_size, num_csr_buffers,
              csr_chunk_devices, distributed_slot, resource_manager_->get_process_id(), resource_manager_,
              device_csr_value_buffers, device_csr_row_offset_buffers, pinned_staging_buffer_param,
              (uint32_t*)device_embed_param_start_offset.data(), dev_slot_offset_ptr, rmm_resources,
              memory_resource_.get(), task_stream_);

        } else if (param.type == DataReaderSparse_t::Localized) {
          // Add row to one buffer
          // buffer select based on k % num_gpus k -> loop on slot_num
          // csr_chunk->get_csr_buffer(param_id, dev_id).push_back(local_id);

          // for all slots
          // k is slot_id in current slot
          // int dev_id = k % csr_chunk->get_num_devices();
          // dev_id * num_params_ + param_id
          // call GPU kernel to rearrage the value in individual buffers
          // call GPU kernel for row_offset as well
          // row_increments go for respective buffers that got data

          pinned_buffer_offset_count += convert_parquet_cat_columns(
              cat_column_data_ptr, param_size, param_id, slot_count, batch_size, num_csr_buffers,
              csr_chunk_devices, distributed_slot, resource_manager_->get_process_id(), resource_manager_,
              device_csr_value_buffers, device_csr_row_offset_buffers, pinned_staging_buffer_param,
              (uint32_t*)device_embed_param_start_offset.data(), dev_slot_offset_ptr, rmm_resources,
              memory_resource_.get(), task_stream_);
        } else {
          CK_THROW_(Error_t::UnspecificError, "param.type is not defined");
        }
        df_column_id += param.slot_num;
        param_id++;
      }

      assert((int)host_pinned_csr_inc_.get_num_elements() >= num_csr_buffers);
      CK_CUDA_THROW_(
          cudaMemcpyAsync(host_pinned_csr_inc_.get_ptr(), device_embed_param_start_offset.data(),
                          embed_param_offset_buf_size, cudaMemcpyDeviceToHost, task_stream_));

      // need this sync here to adjust for memcpy size from D2H of csr bufs
      CK_CUDA_THROW_(cudaStreamSynchronize(task_stream_));

      // data copied to csr_chunk are based on valid-data
      for (int param_idx = 0; param_idx < param_size; param_idx++) {
        for (int device = 0; device < csr_chunk_devices; device++) {
          int buf_id = device * param_size + param_idx;

          // save on memcpy to host bottlenecks
          if (resource_manager_->get_process_id() == resource_manager_->get_process_id_from_gpu_global_id(device)) {
            size_t copy_size = host_pinned_csr_inc_.get_ptr()[buf_id] * sizeof(T);
            CK_CUDA_THROW_(cudaMemcpyAsync(csr_chunk->get_csr_buffer(buf_id).get_value_tensor().get_ptr(),
                                          device_csr_value_buffers[buf_id].data(), copy_size,
                                          cudaMemcpyDeviceToHost, task_stream_));

            if (params_[param_idx].type == DataReaderSparse_t::Distributed) {
              copy_size = sizeof(T) * (batch_size * params_[param_idx].slot_num + 1);
            } else {
              copy_size += sizeof(T);
            }

            CK_CUDA_THROW_(cudaMemcpyAsync(csr_chunk->get_csr_buffer(buf_id).get_row_offset_tensor().get_ptr(),
                                          device_csr_row_offset_buffers[buf_id].data(), copy_size,
                                          cudaMemcpyDeviceToHost, task_stream_));
          }
        }
      }

      for (int param_idx = 0; param_idx < param_size; param_idx++) {
        for (int device = 0; device < csr_chunk_devices; device++) {
          int buf_id = device * param_size + param_idx;
          csr_chunk->get_csr_buffer(buf_id).update_value_size(
              host_pinned_csr_inc_.get_ptr()[buf_id]);

          if (params_[param_idx].type == DataReaderSparse_t::Localized) {
            csr_chunk->get_csr_buffer(buf_id).update_row_offset(
                host_pinned_csr_inc_.get_ptr()[buf_id]);
          } else if (params_[param_idx].type == DataReaderSparse_t::Distributed) {
            csr_chunk->get_csr_buffer(buf_id).update_row_offset(batch_size *
                                                                params_[param_idx].slot_num);
          }
        }
      }

      CK_CUDA_THROW_(cudaStreamSynchronize(task_stream_));
      CK_CUDA_THROW_(cudaStreamSynchronize(dense_stream_));

      csr_chunk->apply_to_csr_buffers(&CSR<T>::new_row);

      // caching index increment
      view_offset_ = row_group_index_;
    }

    csr_heap_->commit_data_chunk(worker_id_, false);
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
  return;
}

}  // namespace HugeCTR
