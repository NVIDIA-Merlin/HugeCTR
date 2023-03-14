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

#include <nvToolsExt.h>

#include <data_readers/data_reader_worker.hpp>
#include <fstream>

namespace HugeCTR {

namespace core23_reader {
template <typename T>
void DataReaderWorker<T>::read_new_file() {
  nvtxRangePushA("read_new_file");
  constexpr int MAX_TRY = 10;
  for (int i = 0; i < MAX_TRY; i++) {
    if (checker_->next_source(1) == Error_t::EndOfFile) {
      throw internal_runtime_error(Error_t::EndOfFile, "EndOfFile");
    }

    Error_t err = checker_->read(reinterpret_cast<char*>(&data_set_header_), sizeof(DataSetHeader));
    current_record_index_ = 0;
    if (!(data_set_header_.error_check == 0 && check_type_ == Check_t::None) &&
        !(data_set_header_.error_check == 1 && check_type_ == Check_t::Sum)) {
      HCTR_LOG_S(ERROR, WORLD) << "DataHeaderError " << HCTR_LOCATION() << std::endl;
      continue;
    }
    if (static_cast<size_t>(data_set_header_.slot_num) != total_slot_num_) {
      HCTR_LOG_S(ERROR, WORLD) << "DataHeaderError " << HCTR_LOCATION() << std::endl;
      continue;
    }
    if (err == Error_t::Success) {
      nvtxRangePop();
      return;
    }
  }
  HCTR_OWN_THROW(Error_t::BrokenFile, "failed to read a file");
  nvtxRangePop();
}

template <typename T>
DataReaderWorker<T>::DataReaderWorker(const int worker_id, const int worker_num,
                                      const std::shared_ptr<GPUResource>& gpu_resource,
                                      const std::shared_ptr<std::atomic<bool>>& loop_flag,
                                      const std::shared_ptr<ThreadBuffer23>& buffer,
                                      const std::string& file_list, size_t buffer_length,
                                      bool repeat, Check_t check_type,
                                      const std::vector<DataReaderSparseParam>& params)
    : IDataReaderWorker(worker_id, worker_num, gpu_resource, !repeat, loop_flag, buffer),
      buffer_length_(buffer_length),
      check_type_(check_type),
      params_(params),
      total_slot_num_(0),
      last_batch_nnz_(params.size(), 0) {
  if (worker_id >= worker_num) {
    HCTR_OWN_THROW(Error_t::BrokenFile, "DataReaderWorker: worker_id >= worker_num");
  }
  total_slot_num_ = 0;
  for (auto& p : params) {
    total_slot_num_ += p.slot_num;
  }
  source_ = std::make_shared<FileSource>(worker_id, worker_num, file_list, repeat);
  create_checker();

  int batch_size = buffer->batch_size;
  int batch_size_start_idx = buffer->batch_size_start_idx;
  int batch_size_end_idx = buffer->batch_size_end_idx;
  int label_dim = buffer->label_dim;
  int dense_dim = buffer->dense_dim;

  CudaCPUDeviceContext ctx(gpu_resource->get_device_id());
  core23::TensorParams common_tensor_params =
      core23::TensorParams().device(core23::DeviceType::CPU);
  host_dense_buffer_ =
      core23::Tensor({static_cast<int64_t>(batch_size_end_idx - batch_size_start_idx),
                      static_cast<int64_t>(label_dim + dense_dim)},
                     core23::ScalarType::Float, common_tensor_params);

  temp_host_dense_buffer_ = core23::Tensor({static_cast<int64_t>(label_dim + dense_dim)},
                                           core23::ScalarType::Float, common_tensor_params);

  for (auto& param : params) {
    host_sparse_buffer_.emplace_back(batch_size * param.slot_num,
                                     batch_size * param.max_feature_num);
  }
  // allocate eagerly
  host_dense_buffer_.data();
  temp_host_dense_buffer_.data();
}

template <typename T>
void DataReaderWorker<T>::read_a_batch() {
  long long current_batch_size = buffer23_->batch_size;
  int label_dim = buffer23_->label_dim;
  int dense_dim = buffer23_->dense_dim;
  int label_dense_dim = label_dim + dense_dim;
  int batch_size_start_idx = buffer23_->batch_size_start_idx;
  int batch_size_end_idx = buffer23_->batch_size_end_idx;
  nvtxRangePushA("read_a_batch_to_host");

  try {
    if (!checker_->is_open()) {
      read_new_file();
    }
  } catch (const internal_runtime_error& rt_err) {
    Error_t err = rt_err.get_error();
    // TODO: when in repeate mode and the dataset sample num can not devided by batchsize,
    // Norm/Raw have different behavior to last batch. Norm will fetch the data from the begining
    // of the datset, while Raw will output current_batchsize < batchsize. Comment by Alex Liu
    // (2021.7.4)
    if (err == Error_t::EndOfFile) {
      if (!wait_until_h2d_ready()) return;
      buffer23_->current_batch_size = 0;
      assert(buffer23_->state.load() == BufferState::Writing);
      is_eof_ = true;
      buffer23_->state.store(BufferState::ReadyForRead);

      while (buffer23_->state.load() != BufferState::ReadyForWrite) {
        usleep(2);
        if (!loop_flag_->load()) return;  // in case main thread exit
      }
      return;  // need this return to run from begining
    } else {
      throw;
    }
  }

  // if the EOF is faced, the current batch size can be changed later
  if (data_set_header_.label_dim + data_set_header_.dense_dim != label_dense_dim) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "data_set_header_.label_dim + data_set_header_.dense_dim != label_dense_dim");
  }

  for (auto& each_csr : host_sparse_buffer_) {
    each_csr.reset();
  }
  // batch loop
  for (int batch_idx = 0; batch_idx < buffer23_->batch_size; ++batch_idx) {
    if (batch_idx >= current_batch_size) {
      for (size_t param_id = 0; param_id < params_.size(); ++param_id) {
        auto& param = params_[param_id];
        auto& current_csr = host_sparse_buffer_[param_id];
        for (int k = 0; k < param.slot_num; k++) {
          current_csr.new_row();
        }
      }
      if (batch_idx >= batch_size_start_idx &&
          batch_idx < batch_size_end_idx) {  // only read local device dense data
        float* ptr =
            host_dense_buffer_.data<float>() + (batch_idx - batch_size_start_idx) * label_dense_dim;

        for (int j = 0; j < label_dense_dim; j++) {
          ptr[j] = 0.f;
        }
      }
      continue;
    }
    try {
      try {
        if (batch_idx >= batch_size_start_idx &&
            batch_idx < batch_size_end_idx) {  // only read local device dense data
          HCTR_OWN_THROW(checker_->read(reinterpret_cast<char*>(host_dense_buffer_.data<float>() +
                                                                (batch_idx - batch_size_start_idx) *
                                                                    label_dense_dim),
                                        sizeof(float) * label_dense_dim),
                         "failure in reading label_dense");
        } else {
          HCTR_OWN_THROW(checker_->read(reinterpret_cast<char*>(temp_host_dense_buffer_.data()),
                                        sizeof(float) * label_dense_dim),
                         "failure in reading label_dense");
        }

        for (size_t param_id = 0; param_id < params_.size(); ++param_id) {
          auto& current_csr = host_sparse_buffer_[param_id];
          current_csr.set_check_point();
        }
        for (size_t param_id = 0; param_id < params_.size(); ++param_id) {
          auto& param = params_[param_id];
          auto& current_csr = host_sparse_buffer_[param_id];
          for (int k = 0; k < param.slot_num; k++) {
            int nnz;
            HCTR_OWN_THROW(checker_->read(reinterpret_cast<char*>(&nnz), sizeof(int)),
                           "failure in reading nnz");
            if (nnz > (int)buffer_length_ || nnz < 0) {
              HCTR_LOG_S(ERROR, WORLD)
                  << "nnz > buffer_length_ | nnz < 0 nnz: " << nnz
                  << ". Please check if i64_input_key in config is compatible with dataset"
                  << HCTR_LOCATION() << std::endl;
            }
            current_csr.new_row();
            size_t num_value = current_csr.get_num_values();
            HCTR_OWN_THROW(
                checker_->read(
                    reinterpret_cast<char*>(static_cast<T*>(current_csr.get_value_tensor().data()) +
                                            num_value),
                    sizeof(T) * nnz),
                "failure in reading feature_ids_");
            current_csr.update_value_size(nnz);
          }
        }
      } catch (const internal_runtime_error& rt_err) {
        batch_idx--;  // restart i-th sample
        for (auto& each_csr : host_sparse_buffer_) {
          each_csr.roll_back();
        }
        Error_t err = rt_err.get_error();
        if (err == Error_t::DataCheckError) {
          HCTR_LOG_S(ERROR, WORLD) << "Error_t::DataCheckError " << HCTR_LOCATION() << std::endl;
        } else {            // Error_t::BrokenFile, Error_t::UnspecificEror, ...
          read_new_file();  // can throw Error_t::EOF
        }
      }

      current_record_index_++;

      // start a new file when finish one file read
      if (current_record_index_ >= data_set_header_.number_of_records) {
        read_new_file();  // can throw Error_t::EOF
      }
    } catch (const internal_runtime_error& rt_err) {
      Error_t err = rt_err.get_error();
      if (err == Error_t::EndOfFile) {
        current_batch_size = batch_idx + 1;
      } else {
        throw;
      }
    }
  }

  for (auto& each_csr : host_sparse_buffer_) {
    each_csr.new_row();
  }
  nvtxRangePop();

  // do h2d
  // wait buffer and schedule

  if (!wait_until_h2d_ready()) return;
  nvtxRangePushA("h2d");
  buffer23_->current_batch_size = current_batch_size;
  {
    CudaCPUDeviceContext context(gpu_resource_->get_device_id());
    auto dst_dense_tensor = buffer23_->device_dense_buffers;
    HCTR_LIB_THROW(cudaMemcpyAsync(dst_dense_tensor.data(), host_dense_buffer_.data(),
                                   host_dense_buffer_.num_bytes(), cudaMemcpyHostToDevice,
                                   gpu_resource_->get_memcpy_stream()));
    for (size_t param_id = 0; param_id < params_.size(); ++param_id) {
      auto dst_sparse_tensor = buffer23_->device_sparse_buffers[param_id];
      if (buffer23_->is_fixed_length[param_id] &&
          last_batch_nnz_[param_id] == host_sparse_buffer_[param_id].get_num_values()) {
        HCTR_LIB_THROW(cudaMemcpyAsync(dst_sparse_tensor.get_value_ptr(),
                                       host_sparse_buffer_[param_id].get_value_tensor().data(),
                                       host_sparse_buffer_[param_id].get_num_values() * sizeof(T),
                                       cudaMemcpyHostToDevice, gpu_resource_->get_memcpy_stream()));
      } else {
        SparseTensorHelper::copy_async(dst_sparse_tensor, host_sparse_buffer_[param_id],
                                       gpu_resource_->get_memcpy_stream());
        last_batch_nnz_[param_id] = host_sparse_buffer_[param_id].get_num_values();
      }
    }
    HCTR_LIB_THROW(cudaStreamSynchronize(gpu_resource_->get_memcpy_stream()));
  }
  assert(buffer23_->state.load() == BufferState::Writing);
  buffer23_->state.store(BufferState::ReadyForRead);
  nvtxRangePop();
}
}  // namespace core23_reader

template <typename T>
void DataReaderWorker<T>::read_new_file() {
  nvtxRangePushA("read_new_file");
  constexpr int MAX_TRY = 10;
  for (int i = 0; i < MAX_TRY; i++) {
    if (checker_->next_source(1) == Error_t::EndOfFile) {
      throw internal_runtime_error(Error_t::EndOfFile, "EndOfFile");
    }

    Error_t err = checker_->read(reinterpret_cast<char*>(&data_set_header_), sizeof(DataSetHeader));
    current_record_index_ = 0;
    if (!(data_set_header_.error_check == 0 && check_type_ == Check_t::None) &&
        !(data_set_header_.error_check == 1 && check_type_ == Check_t::Sum)) {
      HCTR_LOG_S(ERROR, WORLD) << "DataHeaderError " << HCTR_LOCATION() << std::endl;
      continue;
    }
    if (static_cast<size_t>(data_set_header_.slot_num) != total_slot_num_) {
      HCTR_LOG_S(ERROR, WORLD) << "DataHeaderError " << HCTR_LOCATION() << std::endl;
      continue;
    }
    if (err == Error_t::Success) {
      nvtxRangePop();
      return;
    }
  }
  HCTR_OWN_THROW(Error_t::BrokenFile, "failed to read a file");
  nvtxRangePop();
}

template <typename T>
DataReaderWorker<T>::DataReaderWorker(const int worker_id, const int worker_num,
                                      const std::shared_ptr<GPUResource>& gpu_resource,
                                      const std::shared_ptr<std::atomic<bool>>& loop_flag,
                                      const std::shared_ptr<ThreadBuffer>& buffer,
                                      const std::string& file_list, size_t buffer_length,
                                      bool repeat, Check_t check_type,
                                      const std::vector<DataReaderSparseParam>& params)
    : IDataReaderWorker(worker_id, worker_num, gpu_resource, !repeat, loop_flag, buffer),
      buffer_length_(buffer_length),
      check_type_(check_type),
      params_(params),
      total_slot_num_(0),
      last_batch_nnz_(params.size(), 0) {
  if (worker_id >= worker_num) {
    HCTR_OWN_THROW(Error_t::BrokenFile, "DataReaderWorker: worker_id >= worker_num");
  }
  total_slot_num_ = 0;
  for (auto& p : params) {
    total_slot_num_ += p.slot_num;
  }
  source_ = std::make_shared<FileSource>(worker_id, worker_num, file_list, repeat);
  create_checker();

  int batch_size = buffer->batch_size;
  int batch_size_start_idx = buffer->batch_size_start_idx;
  int batch_size_end_idx = buffer->batch_size_end_idx;
  int label_dim = buffer->label_dim;
  int dense_dim = buffer->dense_dim;

  CudaCPUDeviceContext ctx(gpu_resource->get_device_id());
  std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> buff =
      GeneralBuffer2<CudaHostAllocator>::create();

  buff->reserve({static_cast<size_t>(batch_size_end_idx - batch_size_start_idx),
                 static_cast<size_t>(label_dim + dense_dim)},
                &host_dense_buffer_);
  buff->reserve({static_cast<size_t>(label_dim + dense_dim)}, &temp_host_dense_buffer_);

  for (auto& param : params) {
    host_sparse_buffer_.emplace_back(batch_size * param.slot_num,
                                     batch_size * param.max_feature_num);
  }

  buff->allocate();
}

template <typename T>
void DataReaderWorker<T>::read_a_batch() {
  long long current_batch_size = buffer_->batch_size;
  int label_dim = buffer_->label_dim;
  int dense_dim = buffer_->dense_dim;
  int label_dense_dim = label_dim + dense_dim;
  int batch_size_start_idx = buffer_->batch_size_start_idx;
  int batch_size_end_idx = buffer_->batch_size_end_idx;
  nvtxRangePushA("read_a_batch_to_host");

  try {
    if (!checker_->is_open()) {
      read_new_file();
    }
  } catch (const internal_runtime_error& rt_err) {
    Error_t err = rt_err.get_error();
    // TODO: when in repeate mode and the dataset sample num can not devided by batchsize,
    // Norm/Raw have different behavior to last batch. Norm will fetch the data from the begining
    // of the datset, while Raw will output current_batchsize < batchsize. Comment by Alex Liu
    // (2021.7.4)
    if (err == Error_t::EndOfFile) {
      if (!wait_until_h2d_ready()) return;
      buffer_->current_batch_size = 0;
      assert(buffer_->state.load() == BufferState::Writing);
      is_eof_ = true;
      buffer_->state.store(BufferState::ReadyForRead);

      while (buffer_->state.load() != BufferState::ReadyForWrite) {
        usleep(2);
        if (!loop_flag_->load()) return;  // in case main thread exit
      }
      return;  // need this return to run from begining
    } else {
      throw;
    }
  }

  // if the EOF is faced, the current batch size can be changed later
  if (data_set_header_.label_dim + data_set_header_.dense_dim != label_dense_dim) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "data_set_header_.label_dim + data_set_header_.dense_dim != label_dense_dim");
  }

  for (auto& each_csr : host_sparse_buffer_) {
    each_csr.reset();
  }
  // batch loop
  for (int batch_idx = 0; batch_idx < buffer_->batch_size; ++batch_idx) {
    if (batch_idx >= current_batch_size) {
      for (size_t param_id = 0; param_id < params_.size(); ++param_id) {
        auto& param = params_[param_id];
        auto& current_csr = host_sparse_buffer_[param_id];
        for (int k = 0; k < param.slot_num; k++) {
          current_csr.new_row();
        }
      }
      if (batch_idx >= batch_size_start_idx &&
          batch_idx < batch_size_end_idx) {  // only read local device dense data
        float* ptr =
            host_dense_buffer_.get_ptr() + (batch_idx - batch_size_start_idx) * label_dense_dim;

        for (int j = 0; j < label_dense_dim; j++) {
          ptr[j] = 0.f;
        }
      }
      continue;
    }
    try {
      try {
        if (batch_idx >= batch_size_start_idx &&
            batch_idx < batch_size_end_idx) {  // only read local device dense data
          HCTR_OWN_THROW(checker_->read(reinterpret_cast<char*>(host_dense_buffer_.get_ptr() +
                                                                (batch_idx - batch_size_start_idx) *
                                                                    label_dense_dim),
                                        sizeof(float) * label_dense_dim),
                         "failure in reading label_dense");
        } else {
          HCTR_OWN_THROW(checker_->read(reinterpret_cast<char*>(temp_host_dense_buffer_.get_ptr()),
                                        sizeof(float) * label_dense_dim),
                         "failure in reading label_dense");
        }

        for (size_t param_id = 0; param_id < params_.size(); ++param_id) {
          auto& current_csr = host_sparse_buffer_[param_id];
          current_csr.set_check_point();
        }
        for (size_t param_id = 0; param_id < params_.size(); ++param_id) {
          auto& param = params_[param_id];
          auto& current_csr = host_sparse_buffer_[param_id];
          for (int k = 0; k < param.slot_num; k++) {
            int nnz;
            HCTR_OWN_THROW(checker_->read(reinterpret_cast<char*>(&nnz), sizeof(int)),
                           "failure in reading nnz");
            if (nnz > (int)buffer_length_ || nnz < 0) {
              HCTR_LOG_S(ERROR, WORLD)
                  << "nnz > buffer_length_ | nnz < 0 nnz: " << nnz
                  << ". Please check if i64_input_key in config is compatible with dataset"
                  << HCTR_LOCATION() << std::endl;
            }
            current_csr.new_row();
            size_t num_value = current_csr.get_num_values();

            HCTR_OWN_THROW(checker_->read(reinterpret_cast<char*>(
                                              current_csr.get_value_tensor().get_ptr() + num_value),
                                          sizeof(T) * nnz),
                           "failure in reading feature_ids_");
            current_csr.update_value_size(nnz);
          }
        }
      } catch (const internal_runtime_error& rt_err) {
        batch_idx--;  // restart i-th sample
        for (auto& each_csr : host_sparse_buffer_) {
          each_csr.roll_back();
        }
        Error_t err = rt_err.get_error();
        if (err == Error_t::DataCheckError) {
          HCTR_LOG_S(ERROR, WORLD) << "Error_t::DataCheckError " << HCTR_LOCATION() << std::endl;
        } else {            // Error_t::BrokenFile, Error_t::UnspecificEror, ...
          read_new_file();  // can throw Error_t::EOF
        }
      }

      current_record_index_++;

      // start a new file when finish one file read
      if (current_record_index_ >= data_set_header_.number_of_records) {
        read_new_file();  // can throw Error_t::EOF
      }
    } catch (const internal_runtime_error& rt_err) {
      Error_t err = rt_err.get_error();
      if (err == Error_t::EndOfFile) {
        current_batch_size = batch_idx + 1;
      } else {
        throw;
      }
    }
  }

  for (auto& each_csr : host_sparse_buffer_) {
    each_csr.new_row();
  }
  nvtxRangePop();
  // do h2d
  // wait buffer and schedule

  if (!wait_until_h2d_ready()) return;
  nvtxRangePushA("h2d");
  buffer_->current_batch_size = current_batch_size;
  {
    CudaCPUDeviceContext context(gpu_resource_->get_device_id());
    auto dst_dense_tensor = Tensor2<float>::stretch_from(buffer_->device_dense_buffers);
    HCTR_LIB_THROW(cudaMemcpyAsync(dst_dense_tensor.get_ptr(), host_dense_buffer_.get_ptr(),
                                   host_dense_buffer_.get_size_in_bytes(), cudaMemcpyHostToDevice,
                                   gpu_resource_->get_memcpy_stream()));

    for (size_t param_id = 0; param_id < params_.size(); ++param_id) {
      auto dst_sparse_tensor =
          SparseTensor<T>::stretch_from(buffer_->device_sparse_buffers[param_id]);
      if (buffer_->is_fixed_length[param_id] &&
          last_batch_nnz_[param_id] == host_sparse_buffer_[param_id].get_num_values()) {
        HCTR_LIB_THROW(cudaMemcpyAsync(dst_sparse_tensor.get_value_ptr(),
                                       host_sparse_buffer_[param_id].get_value_tensor().get_ptr(),
                                       host_sparse_buffer_[param_id].get_num_values() * sizeof(T),
                                       cudaMemcpyHostToDevice, gpu_resource_->get_memcpy_stream()));
      } else {
        sparse_tensor_helper::cuda::copy_async(dst_sparse_tensor, host_sparse_buffer_[param_id],
                                               gpu_resource_->get_memcpy_stream());
        last_batch_nnz_[param_id] = host_sparse_buffer_[param_id].get_num_values();
      }
    }
    HCTR_LIB_THROW(cudaStreamSynchronize(gpu_resource_->get_memcpy_stream()));
  }
  assert(buffer_->state.load() == BufferState::Writing);
  buffer_->state.store(BufferState::ReadyForRead);
  nvtxRangePop();
}

template class core23_reader::DataReaderWorker<uint32_t>;
template class core23_reader::DataReaderWorker<long long>;

template class DataReaderWorker<uint32_t>;
template class DataReaderWorker<long long>;
}  // namespace HugeCTR