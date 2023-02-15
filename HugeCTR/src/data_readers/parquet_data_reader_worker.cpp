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
#include <nvToolsExt.h>

#include <data_readers/metadata.hpp>
#include <data_readers/parquet_data_reader_worker.hpp>
namespace HugeCTR {
// producer
template <typename T>
void ParquetDataReaderWorker<T>::do_h2d() {
  if (!row_group_reader_) {
    HCTR_OWN_THROW(Error_t::NotInitialized, "please init parquet row group reader first\n");
  }
  while (!this->skip_read_ && loop_flag_->load()) {
    try {
      if (!row_group_reader_->source_available()) {
        row_group_reader_->read_new_file(strict_order_of_batches_ ? worker_id_ + 1 : 1);
      }
      if (row_group_reader_->get_local_row_group_id() >=
          row_group_reader_->get_current_num_row_groups()) {
        long long expected_next_num_group = strict_order_of_batches_ ? worker_num_ : 1;
        long long last_row_group_id =
            row_group_reader_->get_local_row_group_id() - expected_next_num_group;
        expected_next_num_group -=
            (row_group_reader_->get_current_num_row_groups() - 1) - last_row_group_id;
        try {
          row_group_reader_->read_new_file(expected_next_num_group);
        } catch (const internal_runtime_error& rt_err) {
          Error_t err = rt_err.get_error();
          if (err == Error_t::EndOfFile) {
            if (this->repeat_) {
              HCTR_OWN_THROW(Error_t::UnspecificError,
                             "Parquet reader worker:Should not reach EOF in repeat mode!\n");
            }
            throw;
          } else {
            HCTR_LOG(INFO, WORLD, " should Never reach here: \n");
            HCTR_LOG_S(INFO, WORLD) << rt_err.what() << std::endl;
            throw;
          }
        }
      }
    } catch (const internal_runtime_error& rt_err) {
      Error_t err = rt_err.get_error();
      if (err == Error_t::EndOfFile) {
        if (!row_group_reader_->wait_until_writeable()) {
          return;
        };
        if (row_group_reader_->get_accomplished_workers(worker_id_) != 0) {
          HCTR_OWN_THROW(Error_t::UnspecificError, "producer  Not zero\n");
        };
        // EOF will notify consumers to pause
        row_group_reader_->set_this_producer_status(BufferState::FileEOF);
        assert(row_group_reader_->get_accomplished_workers(worker_id_) == 0);
        // wait for set_source()
        if (!row_group_reader_->wait_until_writeable()) {
          return;
        };
        // awaken from set_source(), need to reset status as ReadyForWrite for new batch
        row_group_reader_->set_this_producer_status(BufferState::ReadyForWrite);
        row_group_reader_->reset_accomplished_worker();
        continue;
      } else {
        throw;
      }
    }
    // no eof, normal row_group
    if (!row_group_reader_->wait_until_writeable()) {
      return;
    }
    // will put data onto df_producer[worker_id_]
    auto err = row_group_reader_->get_one_read_group(params_, *this->dense_width_dim_,
                                                     this->one_hot_cols_, this->sparse_nnz_array_);
    if (err == Error_t::Success) {
      row_group_reader_->set_this_producer_status(BufferState::ReadyForRead);
      row_group_reader_->reset_read_flag();
    }
  }
}

// consumer
template <class T>
void ParquetDataReaderWorker<T>::read_a_batch() {
  CudaDeviceContext context(device_id_);
  using dtype_dense = float;
  int current_batch_size = -1;
  try {
    // dense_buffers store only data for local gpus, clipped by
    // batch_size_start_idx & batch_size_end_idx
    const int dense_start = buffer_->batch_size_start_idx;  // dense buffer
    const int dense_end = buffer_->batch_size_end_idx;      // dense buffer
    const int label_dense_dim = buffer_->label_dim + buffer_->dense_dim;
    int batch_size = buffer_->batch_size;
    size_t param_num = buffer_->param_num;
    if (!skip_read_) {
      auto dst_dense_tensor = Tensor2<dtype_dense>::stretch_from(buffer_->device_dense_buffers);
      long long elements_to_read = batch_size;
      long long elements_to_forward = batch_size;
      // if read file sequentially, read worker_num_ batches and discard
      // extraneous samples
      if (strict_order_of_batches_) {
        elements_to_read *= worker_num_;
        elements_to_forward *= worker_num_;
      }

      view_offset_ = 0;
      current_batch_size = batch_size;
      auto row_group_consumer = row_group_reader_->get_df_container_consumer();
      while (row_group_consumer->get_available_rows() < (elements_to_read)) {
        int creditor_id = this->worker_id_;
        if (strict_order_of_batches_) {
          creditor_id = global_row_group_id_ % this->worker_num_;
        }
        auto creditor_buffer = (row_group_reader_->get_df_container_producer(creditor_id));
        try {
          bool can_read = row_group_reader_->wait_until_readable(creditor_id);
          if (!can_read) {
            return;
          }
        } catch (const internal_runtime_error& rt_err) {
          Error_t err = rt_err.get_error();
          // both last batch and empty batch will catch eof
          if (err == Error_t::EndOfFile) {
            if (strict_order_of_batches_) {
              long long worker_start =
                  std::min((long long)(worker_id_)*batch_size,
                           (long long)row_group_consumer->get_available_rows());
              long long worker_end = std::min((long long)(worker_id_ + 1) * batch_size,
                                              (long long)row_group_consumer->get_available_rows());
              current_batch_size = worker_end - worker_start;
            } else {
              current_batch_size = row_group_consumer->get_available_rows();
            }
            elements_to_forward = row_group_consumer->get_available_rows();
            if (current_batch_size == 0) {
              if (!wait_until_h2d_ready()) {
                return;
              }
              is_eof_ = true;
              buffer_->current_batch_size = 0;
              assert(buffer_->state.load() == BufferState::Writing);
              // notify data collector the empty batch, it will switch state to
              // BufferState::ReadyForWrite
              buffer_->state.store(BufferState::ReadyForRead);
              while (buffer_->state.load() != BufferState::ReadyForWrite) {
                usleep(2);
                if (!loop_flag_->load()) {
                  return;
                }
              }
              std::unique_lock<std::mutex> lck(this->epoch_mtx_);
              this->epoch_cv_.wait(lck, [&]() { return *this->go_next_epoch_; });
              *this->go_next_epoch_ = 0;
              global_row_group_id_ = 0;
              return;
            } else {
              break;
            }

          } else {
            throw;
          }
        }
        // p2p copy
        if (!row_group_consumer->dense_dim_array_init_) {
          row_group_consumer->init_dense_dim_array(*this->dense_width_dim_);
        }
        // p2p
        nvtxRangePushA("p2p_row_group");
        *row_group_consumer += *creditor_buffer;
        nvtxRangePop();
        row_group_reader_->inc_accomplished_worker(creditor_id);
        global_row_group_id_++;
      }
      if (row_group_consumer->get_available_rows() <= elements_to_read) {
      }
      if (!wait_until_h2d_ready()) {
        return;
      }
      buffer_->current_batch_size = current_batch_size;

      view_offset_ = row_group_consumer->get_curr_row();
      row_group_consumer->forward_row(elements_to_forward);

      std::deque<rmm::device_buffer> rmm_buffers;

      if (!host_memory_pointer_staging_.allocated()) {
        HCTR_OWN_THROW(Error_t::UnspecificError,
                       "Parquet reader worker:Please allocate Pinned Buffer first");
      }
      int dense_dim_check =
          static_cast<int>(std::accumulate(dense_width_dim_->begin(), dense_width_dim_->end(), 0));

      if (dense_dim_check != label_dense_dim) {
        HCTR_LOG(INFO, WORLD, "worker %d dense_dim_check %d vs label_dense_dim %d \n", worker_id_,
                 dense_dim_check, label_dense_dim);
        HCTR_OWN_THROW(Error_t::WrongInput,
                       "Parquet reader: Dense dim of given file and dense dim "
                       "doesn't match ");
      }

      if (!host_memory_dense_dim_array_.allocated()) {
        HCTR_OWN_THROW(Error_t::UnspecificError,
                       "Parquet reader: Allocate pinned mem for "
                       "host_memory_dense_dim_array_ first");
      }
      //! TODO dense_idx_to_parquet_col_ is init by the first producer
      const int num_label_dense = dense_idx_to_parquet_col_.size();

      std::memcpy(reinterpret_cast<void*>(host_memory_dense_dim_array_.get_ptr()),
                  reinterpret_cast<void*>(dense_width_dim_->data()),
                  num_label_dense * sizeof(int64_t));
      HCTR_LIB_THROW(cudaMemcpyAsync(
          reinterpret_cast<void*>(device_memory_dense_dim_array_.get_ptr()),
          reinterpret_cast<void*>(host_memory_dense_dim_array_.get_ptr()),
          sizeof(int64_t) * dense_width_dim_->size(), cudaMemcpyHostToDevice, dense_stream_));
      int offset_start = std::min(dense_start, current_batch_size);
      int offset_end = std::min(dense_end, current_batch_size);
      int samples_to_be_transposed = offset_end - offset_start;
      std::vector<dtype_dense*> dense_column_data_ptr;
      long long view_offset_worker = view_offset_;
      // if strict_order_of_batches_==true, we need to discard extraneous batch
      if (strict_order_of_batches_) {
        view_offset_worker += worker_id_ * batch_size;
      }
      for (int k = 0; k < num_label_dense; k++) {
        dtype_dense* column_ptr = row_group_consumer->dense_ptr_[k];
        column_ptr =
            // only proceed dense for local gpu
            reinterpret_cast<dtype_dense*>(
                (size_t)column_ptr + sizeof(dtype_dense) * (offset_start + view_offset_worker) *
                                         dense_width_dim_->at(k));

        dense_column_data_ptr.push_back(column_ptr);
      }
      nvtxRangePushA("convert_parquet_dense_columns");
      convert_parquet_dense_columns(
          dense_column_data_ptr, num_label_dense,
          reinterpret_cast<int64_t*>(device_memory_dense_dim_array_.get_ptr()), label_dense_dim,
          samples_to_be_transposed, dense_start, dense_end,
          reinterpret_cast<dtype_dense*>(dst_dense_tensor.get_ptr()),
          host_memory_pointer_staging_.get_ptr(), rmm_buffers, memory_resource_.get(),
          dense_stream_);
      nvtxRangePop();

      {
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
          HCTR_LIB_THROW(cudaMemsetAsync(device_csr_row_offset_buffers[k], 0,
                                         size_of_csr_roff_buffer, task_stream_));
        }

        int param_id = 0;
        int df_column_id = 0;

        int64_t* pinned_staging_buffer =
            reinterpret_cast<int64_t*>((size_t)host_memory_pointer_staging_.get_ptr() +
                                       sizeof(int64_t) * 2 * (label_dense_dim + 1));
        size_t pinned_buffer_offset_count = 0;

        for (auto& param : params_) {
          int slot_count = param.slot_num;

          std::vector<T*> cat_column_data_ptr(
              row_group_consumer->sparse_ptr_.begin() + df_column_id,
              row_group_consumer->sparse_ptr_.begin() + df_column_id + slot_count);
          std::vector<int32_t*> cat_column_row_offset_ptr(
              row_group_consumer->sparse_offset_ptr_.begin() + df_column_id,
              row_group_consumer->sparse_offset_ptr_.begin() + df_column_id + slot_count);
          T* dev_slot_offset_ptr = reinterpret_cast<T*>((size_t)slot_offset_device_buf_->data() +
                                                        (df_column_id * sizeof(T)));
          int64_t* pinned_staging_buffer_param = reinterpret_cast<int64_t*>(
              (size_t)pinned_staging_buffer + pinned_buffer_offset_count * sizeof(int64_t));
          nvtxRangePushA("convert_parquet_cat_columns");
          {
            // optimize converter in the future when slots nnz for current
            // param_id is fixed
            pinned_buffer_offset_count += convert_parquet_cat_columns(
                cat_column_data_ptr, cat_column_row_offset_ptr, view_offset_worker, param_num,
                param_id, param.max_nnz, slot_count, current_batch_size,
                resource_manager_->get_process_id(), resource_manager_, device_csr_value_buffers,
                device_csr_row_offset_buffers, pinned_staging_buffer_param, dev_slot_offset_ptr,
                rmm_buffers, memory_resource_.get(), task_stream_);
          }
          nvtxRangePop();
          df_column_id += param.slot_num;
          param_id++;
        }

        HCTR_LIB_THROW(cudaStreamSynchronize(task_stream_));

        // get nnz info
        for (size_t buffer_id = 0; buffer_id < param_num; buffer_id++) {
          //! caution ! not batch_size but current_batch_size
          int64_t last_row = current_batch_size * params_[buffer_id].slot_num;
          int64_t dev_row_pointer =
              reinterpret_cast<int64_t>(device_csr_row_offset_buffers[buffer_id]) +
              last_row * sizeof(T);

          auto dst_sparse_tensor = SparseTensor<T>::stretch_from(dst_sparse_tensors[buffer_id]);
          HCTR_LIB_THROW(cudaMemcpyAsync(reinterpret_cast<void*>(host_pinned_csr_inc_.get_ptr()),
                                         reinterpret_cast<void*>(dev_row_pointer), sizeof(T) * 1,
                                         cudaMemcpyDeviceToHost, task_stream_));
          HCTR_LIB_THROW(cudaStreamSynchronize(task_stream_));
          size_t nnz = static_cast<size_t>(host_pinned_csr_inc_.get_ptr()[0]);
          *dst_sparse_tensor.get_nnz_ptr() = nnz;
        }
      }

      HCTR_LIB_THROW(cudaStreamSynchronize(task_stream_));
      HCTR_LIB_THROW(cudaStreamSynchronize(dense_stream_));
    }
    buffer_->state.store(BufferState::ReadyForRead);
  } catch (const internal_runtime_error& rt_err) {
    Error_t err = rt_err.get_error();
    if (err == Error_t::EndOfFile) {
      return;
    } else {
      throw;
    }
  }
  return;
}

// loop_flag is readonly
template <typename T>
ParquetDataReaderWorker<T>::ParquetDataReaderWorker(
    unsigned int worker_id, unsigned int worker_num,
    const std::shared_ptr<GPUResource>& gpu_resource,
    const std::shared_ptr<std::atomic<bool>>& loop_flag, volatile bool* end_flag,
    const std::shared_ptr<ThreadBuffer>& buffer, const std::string& file_list,
    bool strict_order_of_batches, bool repeat, const std::vector<DataReaderSparseParam>& params,
    const DataSourceParams& data_source_params, const std::vector<long long>& slot_offset,

    int device_id, std::shared_ptr<DFContainer<T>> df_container_consumer,
    std::vector<std::shared_ptr<DFContainer<T>>>& df_container_producer,
    std::vector<std::shared_ptr<std::atomic<BufferState>>>& producer_buffer_stats,
    std::vector<char>& workers_has_read,
    std::vector<std::shared_ptr<std::atomic<int>>>& accomplished_workers,
    const std::shared_ptr<ResourceManager>& resource_manager,
    std::shared_ptr<std::vector<size_t>> dense_width_dim, char* go_next_epoch,
    std::mutex& epoch_mtx, std::condition_variable& epoch_cv)

    : IDataReaderWorker(worker_id, worker_num, gpu_resource, !repeat, loop_flag, buffer),
      params_(params),
      strict_order_of_batches_(strict_order_of_batches),
      repeat_(repeat),
      slot_offset_(slot_offset),
      device_id_(device_id),
      thread_resource_allocated_(false),
      resource_manager_(resource_manager),
      dense_width_dim_(dense_width_dim),
      global_row_group_id_(0),
      go_next_epoch_(go_next_epoch),
      epoch_mtx_(epoch_mtx),
      epoch_cv_(epoch_cv) {
  CudaDeviceContext ctx(gpu_resource->get_device_id());

  std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> buff =
      GeneralBuffer2<CudaHostAllocator>::create();

  // For reading nnz
  buff->reserve({32}, &host_pinned_csr_inc_);

  memory_resource_ = resource_manager_->get_device_rmm_device_memory_resource(device_id_);

  if (worker_id >= worker_num) {
    HCTR_OWN_THROW(Error_t::BrokenFile, "ParquetDataReaderWorker: worker_id >= worker_num");
  }
  slots_ = 0;
  for (auto& p : params) {
    slots_ += p.slot_num;
  }
  size_t num_of_pointer_staging = (2 * (buffer_->label_dim + buffer_->dense_dim + 1) +
                                   2 * params_.size() * slots_ + 2 * slots_);
  // pinned buffer for dense feature converter
  buff->reserve({num_of_pointer_staging}, &host_memory_pointer_staging_);
  // global_batches_offset = worker_id * buffer->batch_size;
  // pinned dense dim , can't know dense_dim_array in advance
  // label_dim + dense_dim > label_num + dense_num
  buff->reserve({static_cast<size_t>(buffer_->label_dim + buffer_->dense_dim)},
                &host_memory_dense_dim_array_);
  buff->allocate();
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff_gpu = GeneralBuffer2<CudaAllocator>::create();
  // clone of dense_dim_array on gpu
  buff_gpu->reserve({static_cast<size_t>(buffer_->label_dim + buffer_->dense_dim)},
                    &device_memory_dense_dim_array_);
  buff_gpu->allocate();
  source_ = std::make_shared<ParquetFileSource>(
      worker_id, worker_num, file_list, strict_order_of_batches, repeat, data_source_params);

  if ((int)slot_offset_.size() < slots_) {
    slot_offset_.resize(slots_, static_cast<long long int>(0));
  }
  for (auto& c : slot_offset_) {
    if ((c >= std::numeric_limits<T>::min()) && (c <= std::numeric_limits<T>::max()))
      slot_offset_dtype_.push_back((T)c);
    else
      HCTR_OWN_THROW(Error_t::DataCheckError, "Slot offset value exceed the key type range");
  }
  if (!thread_resource_allocated_) {
    // cant allocate and set resources in constructor
    HCTR_LIB_THROW(cudaSetDevice(device_id_));  // for multiple devices
    HCTR_LIB_THROW(cudaStreamCreateWithFlags(&task_stream_, cudaStreamNonBlocking));
    HCTR_LIB_THROW(cudaStreamCreateWithFlags(&dense_stream_, cudaStreamNonBlocking));
    size_t slot_offset_buf_size = sizeof(T) * slot_offset_dtype_.size();
    slot_offset_device_buf_ = std::make_shared<rmm::device_buffer>(
        slot_offset_buf_size, task_stream_, memory_resource_.get());

    HCTR_LIB_THROW(cudaMemcpyAsync(slot_offset_device_buf_->data(), slot_offset_dtype_.data(),
                                   slot_offset_buf_size, cudaMemcpyHostToDevice, task_stream_));
    thread_resource_allocated_ = true;
  }
  row_group_reader_ = std::make_unique<RowGroupReadingThread<T>>(
      device_id, worker_id, worker_num, strict_order_of_batches ? worker_num : 1, end_flag,
      parquet_file_source(), memory_resource_.get(), strict_order_of_batches,
      dense_idx_to_parquet_col_, categorical_idx_parquet_col_, df_container_consumer,
      df_container_producer, producer_buffer_stats, workers_has_read, accomplished_workers);
}
template <typename T>
ParquetDataReaderWorker<T>::~ParquetDataReaderWorker() {
  CudaDeviceContext context(device_id_);
  memory_resource_.reset();  // this should trigger dtor
  if (thread_resource_allocated_) {
    this->skip_read();
    HCTR_LIB_THROW(cudaStreamSynchronize(task_stream_));
    HCTR_LIB_THROW(cudaStreamSynchronize(dense_stream_));
    slot_offset_device_buf_.reset();
    source_.reset();
    HCTR_LIB_THROW(cudaStreamDestroy(task_stream_));
    HCTR_LIB_THROW(cudaStreamDestroy(dense_stream_));
    thread_resource_allocated_ = false;
  }
}

template class ParquetDataReaderWorker<uint32_t>;
template class ParquetDataReaderWorker<long long>;
}  // namespace HugeCTR
