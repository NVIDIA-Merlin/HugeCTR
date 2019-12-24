/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
static void data_reader_thread_func_(DataReaderWorker<TypeKey>* data_reader,
                                     int* p_loop_flag) {
  try {
    while (*p_loop_flag) {
      data_reader->read_a_batch();
    }
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

/**
 * A helper function to for reading data from
 * CSRChunks to data_reader (GPU) local buffer in a new thread.
 * @param data_reader a pointer of data_collector.
 * @param p_loop_flag a flag to control the loop and
                      break loop when DataReader is destroyed.
 */
template <typename TypeKey>
static void data_collector_thread_func_(DataCollector<TypeKey>* data_collector, int* p_loop_flag) {
  try {
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
template <typename TypeKey>
class DataReader {
 private:
  FileList* file_list_{nullptr}; /**< file list of data set */
  const int NumChunks{31};       /**< NumChunks will be used in Heap*/
  const int NumThreads{20};      /**< number of threads for data reading */

  Heap<CSRChunk<TypeKey>>* csr_heap_{nullptr}; /**< heap to cache the data set */
  std::vector<DataReaderWorker<TypeKey>*>
      data_readers_; /**< A vector of DataReaderWorker' pointer.*/
  std::vector<std::thread*> data_reader_threads_; /**< A vector of the pointers of data reader .*/
  std::thread* data_collector_thread_{nullptr};   /**< A data_collector_thread. */
  GeneralBuffers<float> label_buffers_; /**< A gpu general buffer for label_buffer */
  Tensors<float> label_tensors_;        /**< Label tensors for the usage of loss */
  GeneralBuffers<TypeKey> csr_buffers_; /**< csr_buffers contains row_offset_tensor and value_tensors */
  Tensors<TypeKey> row_offsets_tensors_; /**< row offset tensors*/
  Tensors<TypeKey> value_tensors_;       /**< value tensors */
  bool shared_output_flag_{false}; /**< whether this is a data reader for eval. It's only mark the
                                      output data, which is sharing output tensor with train. */

  const GPUResourceGroup& device_resources_;
  const int batchsize_;
  const int label_dim_;                    /**< dimention of label e.g. 1 for BinaryCrossEntropy */
  const int slot_num_;                     /**< num of slots for reduce */
  const int max_feature_num_per_sample_;   /**< max possible nnz in a slot (to allocate buffer) */
  int data_reader_loop_flag_;              /**< p_loop_flag a flag to control the loop */
  DataCollector<TypeKey>* data_collector_; /**< pointer of DataCollector */

  /**
   * Ctor.
   * This ctor is only used when you already have a instant of DataReader and you want to
   * reuse the output tensor e.g. evaluation.
   * @params file_list_name file name of the data set file list.
   * @params prototype an instant of crated DataReader for output reuse.
   * @params num_chunks number of chunks in heap.
   * @params number of threads for data reading.
   */
  DataReader(const std::string& file_list_name, const DataReader& prototype, int num_chunks = 31,
             int num_threads = 20);

  /**
   * Ctor
   * This ctor is not a copy constructor, but only used in the slave process of evaluation,
   * which doesn't need file list and only receives data from other process.
   */
  DataReader(const DataReader& prototype);

 public:
  /**
   * Reading a batch from cpu to gpu (embedding)
   */
  void read_a_batch_to_device();  // read data from csr to tensors

  /**
   * Ctor
   */
  DataReader(const std::string& file_list_name, int batchsize, int label_dim, int slot_num,
             int max_feature_num_per_sample, const GPUResourceGroup& gpu_resource_group,
             int num_chunks = 31, int num_threads = 20);

  /**
   * Slave process of evaluation will call this to create a new object of DataReader
   */
  DataReader* clone_eval_with_shared_output() {
    DataReader* new_reader = new DataReader(*this);
    return new_reader;
  }

  /**
   * Slave process of evaluation will call this to create a new object of DataReader
   */
  DataReader* clone_eval_with_shared_output(const std::string& file_list_name) {
    DataReader* new_reader = new DataReader(file_list_name, *this);
    return new_reader;
  }

  const Tensors<float>& get_label_tensors() const { return label_tensors_; }
  const Tensors<TypeKey>& get_row_offsets_tensors() const {
    return row_offsets_tensors_;
  }
  const Tensors<TypeKey>& get_value_tensors() const { return value_tensors_; }
  ~DataReader();
};

template <typename TypeKey>
DataReader<TypeKey>::DataReader(const std::string& file_list_name,
                                const DataReader<TypeKey>& prototype, int num_chunks,
                                int num_threads)
    : file_list_(new FileList(file_list_name)),
      NumChunks(num_chunks),
      NumThreads(num_threads),
      label_buffers_(prototype.label_buffers_),
      label_tensors_(prototype.label_tensors_),
      csr_buffers_(prototype.csr_buffers_),
      row_offsets_tensors_(prototype.row_offsets_tensors_),
      value_tensors_(prototype.value_tensors_),
      device_resources_(prototype.device_resources_),
      batchsize_(prototype.batchsize_),
      label_dim_(prototype.label_dim_),
      slot_num_(prototype.slot_num_),
      max_feature_num_per_sample_(prototype.max_feature_num_per_sample_) {
  shared_output_flag_ = true;
  data_reader_loop_flag_ = 1;
  int total_gpu_count = device_resources_.get_total_gpu_count();
  if (total_gpu_count == 0 || batchsize_ <= 0 || label_dim_ <= 0 || slot_num_ <= 0 ||
      max_feature_num_per_sample_ <= 0 || 0 != batchsize_ % total_gpu_count) {
    CK_THROW_(Error_t::WrongInput,
              "total_gpu_count = 0 || batchsize <=0 || label_dim <= 0  || slot_num <= 0 || "
              "max_feature_num_per_sample <= 0|| batchsize_ % total_gpu_count != 0");
  }

  csr_heap_ = new Heap<CSRChunk<TypeKey>>(NumChunks, total_gpu_count, batchsize_, label_dim_,
                                          slot_num_, max_feature_num_per_sample_ * batchsize_);
  assert(data_readers_.empty() && data_reader_threads_.empty());
  for (int i = 0; i < NumThreads; i++) {
    DataReaderWorker<TypeKey>* data_reader =
        new DataReaderWorker<TypeKey>(*csr_heap_, *file_list_, max_feature_num_per_sample_);
    data_readers_.push_back(data_reader);
    data_reader_threads_.push_back(
        new std::thread(data_reader_thread_func_<TypeKey>, data_reader, &data_reader_loop_flag_));
  }

  data_collector_ =
      new DataCollector<TypeKey>(label_buffers_, csr_buffers_, device_resources_, csr_heap_);

  data_collector_thread_ = new std::thread(data_collector_thread_func_<TypeKey>, data_collector_,
                                           &data_reader_loop_flag_);
}

template <typename TypeKey>
DataReader<TypeKey>::DataReader(const DataReader<TypeKey>& prototype)
    : label_buffers_(prototype.label_buffers_),
      label_tensors_(prototype.label_tensors_),
      csr_buffers_(prototype.csr_buffers_),
      row_offsets_tensors_(prototype.row_offsets_tensors_),
      value_tensors_(prototype.value_tensors_),
      device_resources_(prototype.device_resources_),
      batchsize_(prototype.batchsize_),
      label_dim_(prototype.label_dim_),
      slot_num_(prototype.slot_num_),
      max_feature_num_per_sample_(prototype.max_feature_num_per_sample_) {
  shared_output_flag_ = true;
  data_reader_loop_flag_ = 1;
  int total_gpu_count = device_resources_.get_total_gpu_count();
  if (total_gpu_count == 0 || batchsize_ <= 0 || label_dim_ <= 0 || slot_num_ <= 0 ||
      max_feature_num_per_sample_ <= 0 || 0 != batchsize_ % total_gpu_count) {
    CK_THROW_(Error_t::WrongInput,
              "total_gpu_count = 0 || batchsize <=0 || label_dim <= 0  || slot_num <= 0 || "
              "max_feature_num_per_sample <= 0|| batchsize_ % total_gpu_count != 0");
  }

  data_collector_ = new DataCollector<TypeKey>(label_buffers_, csr_buffers_, device_resources_);

  data_collector_thread_ = new std::thread(data_collector_thread_func_<TypeKey>, data_collector_,
                                           &data_reader_loop_flag_);
}

template <typename TypeKey>
DataReader<TypeKey>::DataReader(const std::string& file_list_name, int batchsize, int label_dim,
                                int slot_num, int max_feature_num_per_sample,
                                const GPUResourceGroup& gpu_resource_group, int num_chunks,
                                int num_threads)
    : file_list_(new FileList(file_list_name)),
      NumChunks(num_chunks),
      NumThreads(num_threads),
      device_resources_(gpu_resource_group),
      batchsize_(batchsize),
      label_dim_(label_dim),
      slot_num_(slot_num),
      max_feature_num_per_sample_(max_feature_num_per_sample) {
  data_reader_loop_flag_ = 1;
  int total_gpu_count = device_resources_.get_total_gpu_count();
  if (total_gpu_count == 0 || batchsize <= 0 || label_dim <= 0 || slot_num <= 0 ||
      max_feature_num_per_sample <= 0 || 0 != batchsize_ % total_gpu_count) {
    CK_THROW_(Error_t::WrongInput,
              "total_gpu_count == 0 || batchsize <=0 || label_dim <= 0  || slot_num <= 0 || "
              "max_feature_num_per_sample <= 0|| batchsize_ % total_gpu_count != 0");
  }

  csr_heap_ = new Heap<CSRChunk<TypeKey>>(NumChunks, total_gpu_count, batchsize_, label_dim_,
                                          slot_num_, max_feature_num_per_sample_ * batchsize_);

  assert(data_readers_.empty() && data_reader_threads_.empty());
  for (int i = 0; i < NumThreads; i++) {
    DataReaderWorker<TypeKey>* data_reader =
        new DataReaderWorker<TypeKey>(*csr_heap_, *file_list_, max_feature_num_per_sample_);
    data_readers_.push_back(data_reader);
    data_reader_threads_.push_back(
        new std::thread(data_reader_thread_func_<TypeKey>, data_reader, &data_reader_loop_flag_));
  }

  auto& device_list = device_resources_.get_device_list();

  // create label tensor
  int batch_size_per_device = batchsize_ / total_gpu_count;
  std::vector<int> tmp_dim = {batch_size_per_device, label_dim_};
  assert(label_tensors_.empty() && label_buffers_.empty());
  for (auto device_id : device_list) {
    GeneralBuffer<float>* tmp_label_buff = new GeneralBuffer<float>();
    label_tensors_.push_back(new Tensor<float>(tmp_dim, *tmp_label_buff, TensorFormat_t::HW));
    tmp_label_buff->init(device_id);
    label_buffers_.push_back(tmp_label_buff);
  }
  // create value and row offset tensor
  std::vector<int> num_rows_dim = {1, batchsize_ * slot_num_ + 1};
  std::vector<int> num_max_value_dim = {1, max_feature_num_per_sample_ * batchsize_};
  for (auto device_id : device_list) {
    GeneralBuffer<TypeKey>* tmp_buffer = new GeneralBuffer<TypeKey>();
    Tensor<TypeKey>* tmp_row_offset =
        new Tensor<TypeKey>(num_rows_dim, *tmp_buffer, TensorFormat_t::HW);
    Tensor<TypeKey>* tmp_value =
        new Tensor<TypeKey>(num_max_value_dim, *tmp_buffer, TensorFormat_t::HW);
    row_offsets_tensors_.push_back(tmp_row_offset);
    value_tensors_.push_back(tmp_value);
    tmp_buffer->init(device_id);
    csr_buffers_.push_back(tmp_buffer);
  }

  data_collector_ =
      new DataCollector<TypeKey>(label_buffers_, csr_buffers_, device_resources_, csr_heap_, false);

  data_collector_thread_ = new std::thread(data_collector_thread_func_<TypeKey>, data_collector_,
                                           &data_reader_loop_flag_);
  return;
}

template <typename TypeKey>
void DataReader<TypeKey>::read_a_batch_to_device() {
  data_collector_->read_a_batch_to_device();
  return;
}

template <typename TypeKey>
DataReader<TypeKey>::~DataReader() {
  try {
    // stop all the loops
    for (auto data_reader : data_readers_) {
      data_reader->skip_read();
      // delete data_reader;
    }
    if (csr_heap_ != nullptr) {
      csr_heap_->break_and_return();
    }
    data_reader_loop_flag_ = 0;
    data_collector_->stop();
    // delete threads
    for (auto data_reader_thread : data_reader_threads_) {
      data_reader_thread->join();
      delete data_reader_thread;
    }

    data_collector_thread_->join();
    delete data_collector_thread_;
    delete data_collector_;

    if (shared_output_flag_ == false) {
      for (auto row_offsets_tensor : row_offsets_tensors_) {
        delete row_offsets_tensor;
      }
      for (auto value_tensor : value_tensors_) {
        delete value_tensor;
      }
      for (auto csr_buffer : csr_buffers_) {
        delete csr_buffer;
      }
      for (auto label_tensor : label_tensors_) {
        delete label_tensor;
      }
      for (auto label_buffer : label_buffers_) {
        delete label_buffer;
      }
    }
    // delete heap
    if (file_list_ != nullptr) {
      delete file_list_;
    }
    if (csr_heap_ != nullptr) {
      delete csr_heap_;
    }
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

}  // namespace HugeCTR
