/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <atomic>
#include <fstream>
#include <thread>
#include <vector>
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/csr.hpp"
#include "HugeCTR/include/csr_chunk.hpp"
#include "HugeCTR/include/data_collector.hpp"
#include "HugeCTR/include/data_reader_worker.hpp"
#include "HugeCTR/include/data_reader_worker_interface.hpp"
#include "HugeCTR/include/data_reader_worker_raw.hpp"
#include "HugeCTR/include/file_list.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "HugeCTR/include/gpu_resource.hpp"
#include "HugeCTR/include/heap.hpp"
#include "HugeCTR/include/tensor.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

/**
 * A helper function to read data from dataset to heap in a new thread.
 * @param data_reader a pointer of data_reader.
 * @param p_loop_flag a flag to control the loop,
          and break loop when DataReaderWorker is destroyed.
 */
template <typename TypeKey>
static void data_reader_thread_func_(const std::shared_ptr<IDataReaderWorker>& data_reader,
                                     int* p_loop_flag) {
  try {
    while ((*p_loop_flag) == 0) {
      usleep(2);
    }

    while (*p_loop_flag) {
      data_reader->read_a_batch();
    }

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

/**
 * A helper function to for reading data from
 * CSRChunk to data_reader (GPU) local buffer in a new thread.
 * @param data_reader a pointer of data_collector.
 * @param p_loop_flag a flag to control the loop and
                      break loop when DataReader is destroyed.
 */
template <typename TypeKey>
static void data_collector_thread_func_(
const std::shared_ptr<DataCollector<TypeKey>>& data_collector, int* p_loop_flag) {
  try {
    while ((*p_loop_flag) == 0) {
      usleep(2);
    }

    while (*p_loop_flag) {
      data_collector->collect();
    }
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

/**
 * @brief Data reading controller.
 *
 * Control the data reading from data set to embedding.
 * An instance of DataReader will maintain independent
 * threads for data reading (DataReaderWorker)
 * from dataset to heap. Meanwhile one independent
 * thread consumes the data (DataCollector),
 * and copy the data to GPU buffer.
 */
static int core_offset_ = 0;

template <typename TypeKey>
class DataReader {
 private:
  std::string file_list_;   /**< file list of data set */

  const int NumChunks{12};  /**< NumChunks will be used in HeapEx*/
  const int NumThreads{12}; /**< number of threads for data reading */

  std::shared_ptr<HeapEx<CSRChunk<TypeKey>>> csr_heap_; /**< heap to cache the data set */
  std::vector<std::shared_ptr<IDataReaderWorker>>
      data_readers_;                             /**< A vector of DataReaderWorker' pointer.*/
  std::vector<std::thread> data_reader_threads_; /**< A vector of the pointers of data reader .*/
  std::thread data_collector_thread_;            /**< A data_collector_thread. */

  GeneralBuffers<float> label_buffers_; /**< A gpu general buffer for label_buffer */
  GeneralBuffers<float> dense_buffers_fp32_; /**< A gpu general buffer for dense_buffer */
  GeneralBuffers<__half> dense_buffers_fp16_; /**< A gpu general buffer for dense_buffer */
  Tensors<float> label_tensors_;              /**< Label tensors for the usage of loss */
  ITensors dense_tensors_;              /**< Dense tensors for the usage of loss */
  Check_t check_type_;                        /**< check type */

  /* Each gpu will have several csr output for different embedding */
  GeneralBuffers<TypeKey>
      csr_buffers_; /**< csr_buffers contains row_offset_tensor and value_tensors */
  Tensors<TypeKey> row_offsets_tensors_; /**< row offset tensors*/
  Tensors<TypeKey> value_tensors_;       /**< value tensors */

  const std::vector<DataReaderSparseParam> params_;

  bool shared_output_flag_{false}; /**< whether this is a data reader for eval. It's only mark the
                                      output data, which is sharing output tensor with train. */

  std::shared_ptr<GPUResourceGroup> device_resources_; /**< gpu resource used in this data reader*/
  bool use_mixed_precision_{false};
  const size_t batchsize_;                             /**< batch size */

  const size_t label_dim_;      /**< dimention of label e.g. 1 for BinaryCrossEntropy */
  const size_t dense_dim_;      /**< dimention of dense */
  const DataReaderType_t type_; /**< type of data reader "Norm or Raw "*/
  const long long num_samples_; /**< only used in Raw*/
  const std::vector<long long> slot_offset_;
  int data_reader_loop_flag_{0}; /**< p_loop_flag a flag to control the loop */

  std::shared_ptr<DataCollector<TypeKey>> data_collector_; /**< pointer of DataCollector */
  std::shared_ptr<MmapOffsetList> file_offset_list_;

  bool data_shuffle_{false};

  void create_heap_workers_() {
    int max_feature_num_per_sample = 0;
    int total_gpu_count = device_resources_->get_total_gpu_count();

    for (auto& param : params_) {
      max_feature_num_per_sample += param.max_feature_num;

      if (param.max_feature_num <= 0 || param.slot_num <= 0) {
        CK_THROW_(Error_t::WrongInput, "param.max_feature_num <= 0 || param.slot_num <= 0");
      }
    }

    // init the heap
    csr_heap_.reset(new HeapEx<CSRChunk<TypeKey>>(NumThreads, total_gpu_count, batchsize_,
                                                  label_dim_ + dense_dim_, params_));

    assert(data_readers_.empty() && data_reader_threads_.empty());

    // create data reader

    switch (type_) {
      case DataReaderType_t::Norm: {
        for (int i = 0; i < NumThreads; i++) {
          std::shared_ptr<IDataReaderWorker> data_reader(
              new DataReaderWorker<TypeKey>(i, NumThreads, csr_heap_, file_list_,
                                            max_feature_num_per_sample, check_type_, params_));
          data_readers_.push_back(data_reader);
          data_reader_threads_.emplace_back(data_reader_thread_func_<TypeKey>, data_reader,
                                            &data_reader_loop_flag_);
        }
        break;
      }
      case DataReaderType_t::Raw: {
        {
          int slots = 0;
          for (auto& param : params_) {
            slots += param.slot_num;
          }
          file_offset_list_.reset(new MmapOffsetList(
              file_list_, num_samples_, (label_dim_ + dense_dim_ + slots) * sizeof(int), batchsize_,
              data_shuffle_, NumThreads));
        }

        for (int i = 0; i < NumThreads; i++) {
          std::shared_ptr<IDataReaderWorker> data_reader(
              new DataReaderWorkerRaw<TypeKey>(i, NumThreads, file_offset_list_, csr_heap_,
                                               file_list_, params_, slot_offset_, label_dim_));
          data_readers_.push_back(data_reader);
          data_reader_threads_.emplace_back(data_reader_thread_func_<TypeKey>, data_reader,
                                            &data_reader_loop_flag_);
          set_affinity(data_reader_threads_.back(), {}, true);
        }
        break;
      }
      default: { CK_THROW_(Error_t::WrongInput, "No such data reader type"); }
    }
  }

 public:
  /**
   * Reading a batch from cpu to gpu (embedding)
   */
  long long read_a_batch_to_device();  // read data from csr to tensors

  long long read_a_batch_to_device_delay_release();

  void ready_to_collect() { 
    data_collector_->set_ready_to_write(); 
  }

  void start() { data_reader_loop_flag_ = 1; }

  /**
   * Ctor
   */
  DataReader(const std::string& file_list_name, int batchsize, size_t label_dim, int dense_dim,
             Check_t check_type, std::vector<DataReaderSparseParam>& params,
             const std::shared_ptr<GPUResourceGroup>& gpu_resource_group,
             int num_chunk_threads = 31, bool use_mixed_precision = false
	     , DataReaderType_t type = DataReaderType_t::Norm,
             long long num_samples = 0, std::vector<long long> slot_size = std::vector<long long>(),
             bool cache_data = false, bool start_reading_from_beginning = true,
             bool data_shuffle = false);

  const Tensors<float>& get_label_tensors() const { return label_tensors_; }
  const ITensors& get_dense_tensors() const { return dense_tensors_; }
  const Tensors<TypeKey>& get_row_offsets_tensors() const { return row_offsets_tensors_; }
  const Tensors<TypeKey>& get_value_tensors() const { return value_tensors_; }
  const Tensors<TypeKey> get_row_offsets_tensors(int param_id) const {
    Tensors<TypeKey> tensors;
    for (unsigned int i = 0; i < device_resources_->size(); i++) {
      tensors.emplace_back(row_offsets_tensors_[i * params_.size() + param_id]);
    }
    return tensors;
  }
  const Tensors<TypeKey> get_value_tensors(int param_id) const {
    Tensors<TypeKey> tensors;
    for (unsigned int i = 0; i < device_resources_->size(); i++) {
      tensors.emplace_back(value_tensors_[i * params_.size() + param_id]);
    }
    return tensors;
  }

  ~DataReader();
};

template <typename TypeKey>
DataReader<TypeKey>::DataReader(const std::string& file_list_name, int batchsize, size_t label_dim,
                                int dense_dim, Check_t check_type,
                                std::vector<DataReaderSparseParam>& params,
                                const std::shared_ptr<GPUResourceGroup>& gpu_resource_group, 
                                int num_chunk_threads, bool use_mixed_precision,
				DataReaderType_t type, long long num_samples,
                                std::vector<long long> slot_offset, bool cache_data,
                                bool start_reading_from_beginning, bool data_shuffle)
    : file_list_(file_list_name),
      NumChunks(num_chunk_threads),
      NumThreads(num_chunk_threads),
      check_type_(check_type),
      params_(params),
      device_resources_(gpu_resource_group),
      use_mixed_precision_(use_mixed_precision),
      batchsize_(batchsize),
      label_dim_(label_dim),
      dense_dim_(dense_dim),
      type_(type),
      num_samples_(num_samples),
      slot_offset_(slot_offset),
      data_shuffle_(data_shuffle) {
  if (start_reading_from_beginning) {
    data_reader_loop_flag_ = 1;
  }

  int total_gpu_count = device_resources_->get_total_gpu_count();

  // input check
  if (total_gpu_count == 0 || batchsize <= 0 || label_dim <= 0 || dense_dim < 0 ||
      0 != batchsize_ % total_gpu_count) {
    CK_THROW_(Error_t::WrongInput,
              "total_gpu_count == 0 || batchsize <= 0 || label_dim <= 0 || dense_dim < 0 || 0 != "
              "batchsize_ % total_gpu_count");
  }

  create_heap_workers_();

  auto& device_list = device_resources_->get_device_list();

  // create label and dense tensor
  size_t batch_size_per_device = batchsize_ / total_gpu_count;
  for (auto device_id : device_list) {
    std::shared_ptr<GeneralBuffer<float>> tmp_label_buff(new GeneralBuffer<float>());
    label_tensors_.emplace_back(new Tensor<float>({batch_size_per_device, label_dim_},
                                                  tmp_label_buff, TensorFormat_t::HW));
    tmp_label_buff->init(device_id);
    label_buffers_.emplace_back(tmp_label_buff);
    if(use_mixed_precision_){
      std::shared_ptr<GeneralBuffer<__half>> tmp_dense_buff(new GeneralBuffer<__half>());
      dense_tensors_.emplace_back(new Tensor<__half>({batch_size_per_device, dense_dim_},
						     tmp_dense_buff, TensorFormat_t::HW));
      tmp_dense_buff->init(device_id);
      dense_buffers_fp16_.emplace_back(tmp_dense_buff);
    }
    else{
      std::shared_ptr<GeneralBuffer<float>> tmp_dense_buff(new GeneralBuffer<float>());
      dense_tensors_.emplace_back(new Tensor<float>({batch_size_per_device, dense_dim_},
						     tmp_dense_buff, TensorFormat_t::HW));
      tmp_dense_buff->init(device_id);
      dense_buffers_fp32_.emplace_back(tmp_dense_buff);
    }
  }

  // create value and row offset tensor
  for (auto device_id : device_list) {
    for (auto& param : params) {
      int slots = 0;
      if (param.type == DataReaderSparse_t::Distributed) {
        slots = param.slot_num;
      } else if (param.type == DataReaderSparse_t::Localized) {
        int mod_slots = param.slot_num % total_gpu_count;  // ceiling
        int global_id = gpu_resource_group->get_global_id(device_id);
        if (global_id < mod_slots) {
          slots = param.slot_num / total_gpu_count + 1;
        } else {
          slots = param.slot_num / total_gpu_count;
        }
      }
      std::shared_ptr<GeneralBuffer<TypeKey>> tmp_buffer(new GeneralBuffer<TypeKey>());
      std::vector<size_t> num_rows_dim = {1, batchsize_ * slots + 1};
      Tensor<TypeKey>* tmp_row_offset =
          new Tensor<TypeKey>(num_rows_dim, tmp_buffer, TensorFormat_t::HW);


      size_t num_max_value = (param.max_nnz * slots) <= param.max_feature_num
                                 ? (param.max_nnz * slots * batchsize_)
                                 : (param.max_feature_num * batchsize_);

      std::vector<size_t> num_max_value_dim = {1, num_max_value};

      Tensor<TypeKey>* tmp_value =
          new Tensor<TypeKey>(num_max_value_dim, tmp_buffer, TensorFormat_t::HW);
      tmp_buffer->init(device_id);

      row_offsets_tensors_.emplace_back(tmp_row_offset);
      value_tensors_.emplace_back(tmp_value);
      csr_buffers_.emplace_back(tmp_buffer);
    }
  }

  bool one_hot = type_ == DataReaderType_t::Raw ? true : false;


  if (cache_data) {
    data_collector_.reset(new DataCollector<TypeKey>(label_tensors_, dense_tensors_, csr_buffers_,
						     device_resources_, csr_heap_, use_mixed_precision_, one_hot,
								  (num_samples - 1) / batchsize_ + 1));
  } else {
    data_collector_.reset(new DataCollector<TypeKey>(label_tensors_, dense_tensors_, csr_buffers_,
						     device_resources_, csr_heap_, use_mixed_precision_, one_hot));
  }
  data_collector_thread_ =
    std::thread(data_collector_thread_func_<TypeKey>, data_collector_, &data_reader_loop_flag_);


  set_affinity(data_collector_thread_, {}, true);

  core_offset_ += 64;

  return;
}

template <typename TypeKey>
long long DataReader<TypeKey>::read_a_batch_to_device_delay_release() {
  long long current_batchsize;
  current_batchsize = data_collector_->read_a_batch_to_device();
  return current_batchsize;
}


template <typename TypeKey>
long long DataReader<TypeKey>::read_a_batch_to_device() {
  long long current_batchsize = read_a_batch_to_device_delay_release();
  data_collector_->set_ready_to_write_sync();
  return current_batchsize;
}

template <typename TypeKey>
DataReader<TypeKey>::~DataReader() {
  try {
    // stop all the loops
    data_reader_loop_flag_ = 0;
    for (auto& data_reader : data_readers_) {
      data_reader->skip_read();
    }
    data_collector_->stop();
    if (csr_heap_ != nullptr) {
      csr_heap_->break_and_return();
    }

    data_collector_thread_.join();

    for (auto& data_reader_thread : data_reader_threads_) {
      data_reader_thread.join();
    }

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

}  // namespace HugeCTR
