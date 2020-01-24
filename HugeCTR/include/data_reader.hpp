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
static void data_reader_thread_func_(const std::shared_ptr<DataReaderWorker<TypeKey>>& data_reader,
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
 * CSRChunk to data_reader (GPU) local buffer in a new thread.
 * @param data_reader a pointer of data_collector.
 * @param p_loop_flag a flag to control the loop and
                      break loop when DataReader is destroyed.
 */
template <typename TypeKey>
static void data_collector_thread_func_(
    const std::shared_ptr<DataCollector<TypeKey>>& data_collector, int* p_loop_flag) {
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
  std::shared_ptr<FileList> file_list_; /**< file list of data set */
  const int NumChunks{31};              /**< NumChunks will be used in Heap*/
  const int NumThreads{20};             /**< number of threads for data reading */

  std::shared_ptr<Heap<CSRChunk<TypeKey>>> csr_heap_; /**< heap to cache the data set */
  std::vector<std::shared_ptr<DataReaderWorker<TypeKey>>>
      data_readers_;                             /**< A vector of DataReaderWorker' pointer.*/
  std::vector<std::thread> data_reader_threads_; /**< A vector of the pointers of data reader .*/
  std::thread data_collector_thread_;            /**< A data_collector_thread. */

  GeneralBuffers<float> label_dense_buffers_;          /**< A gpu general buffer for label_buffer */
  Tensors<float> label_tensors_;                 /**< Label tensors for the usage of loss */
  Tensors<float> dense_tensors_;                 /**< Dense tensors for the usage of loss */
  Check_t check_type_;                           /**< check type */

  /* Each gpu will have several csr output for different embedding */
  GeneralBuffers<TypeKey>
      csr_buffers_; /**< csr_buffers contains row_offset_tensor and value_tensors */
  Tensors<TypeKey> row_offsets_tensors_; /**< row offset tensors*/
  Tensors<TypeKey> value_tensors_;       /**< value tensors */
  
  const std::vector<DataReaderSparseParam> params_;

  bool shared_output_flag_{false}; /**< whether this is a data reader for eval. It's only mark the
                                      output data, which is sharing output tensor with train. */
  
  std::shared_ptr<GPUResourceGroup> device_resources_; /**< gpu resource used in this data reader*/
  const int batchsize_;                  /**< batch size */
  const int label_dim_;                  /**< dimention of label e.g. 1 for BinaryCrossEntropy */
  const int dense_dim_;                  /**< dimention of dense */
  int data_reader_loop_flag_;            /**< p_loop_flag a flag to control the loop */
  std::shared_ptr<DataCollector<TypeKey>> data_collector_; /**< pointer of DataCollector */

  void create_heap_workers_(){
    int max_feature_num_per_sample = 0;
    int slot_num = 0;
    int total_gpu_count = device_resources_->get_total_gpu_count();

    for(auto& param : params_){
      max_feature_num_per_sample += param.max_feature_num;

      /* Note that the slot_num here (csr row - 1) is larger than reqired for convinence. */
      /* For example: in a four gpu system, if param.type == Localized, and param.slot_num is 7*/
      /* You will expect slot_num == 2 because each of the CSR should share the same size */
      if(param.type == DataReaderSparse_t::Distributed){
	slot_num += param.slot_num;
      }
      else if(param.type == DataReaderSparse_t::Localized){
	slot_num += (param.slot_num - 1) / total_gpu_count + 1;
      }
      else{
	CK_THROW_(Error_t::WrongInput, "param.type is illegal");
      }
      if(param.max_feature_num <= 0 || param.slot_num <= 0){
	CK_THROW_(Error_t::WrongInput, "param.max_feature_num <= 0 || param.slot_num <= 0");
      }
    }      

    // init the heap
    csr_heap_.reset(new Heap<CSRChunk<TypeKey>>(NumChunks, total_gpu_count, batchsize_, label_dim_ + dense_dim_,
						  params_));

    assert(data_readers_.empty() && data_reader_threads_.empty());

    // create data reader
    for (int i = 0; i < NumThreads; i++) {
      std::shared_ptr<DataReaderWorker<TypeKey>> data_reader(
       new DataReaderWorker<TypeKey>(csr_heap_, *file_list_, max_feature_num_per_sample, check_type_, params_));
      data_readers_.push_back(data_reader);
      data_reader_threads_.emplace_back(data_reader_thread_func_<TypeKey>, data_reader,
					&data_reader_loop_flag_);
    }
  }

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
  DataReader(const std::string& file_list_name, int batchsize, int label_dim, int dense_dim,
	       Check_t check_type, std::vector<DataReaderSparseParam>& params,
	       const std::shared_ptr<GPUResourceGroup>& gpu_resource_group,
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
  const Tensors<float>& get_dense_tensors() const { return dense_tensors_; }
  const Tensors<TypeKey>& get_row_offsets_tensors() const { return row_offsets_tensors_; }
  const Tensors<TypeKey>& get_value_tensors() const { return value_tensors_; }
  const Tensors<TypeKey> get_row_offsets_tensors(int param_id) const {
    Tensors<TypeKey> tensors;
    for(unsigned int i=0; i< device_resources_->size(); i++){
      tensors.emplace_back(row_offsets_tensors_[i*params_.size() + param_id]);
    }
    return tensors;
  }
  const Tensors<TypeKey> get_value_tensors(int param_id) const {
    Tensors<TypeKey> tensors;
    for(unsigned int i=0; i< device_resources_->size(); i++){
      tensors.emplace_back(value_tensors_[i*params_.size() + param_id]);
    }
    return tensors;
  }


  ~DataReader();
};

template <typename TypeKey>
DataReader<TypeKey>::DataReader(const std::string& file_list_name,
                                const DataReader<TypeKey>& prototype, int num_chunks,
                                int num_threads)
    : file_list_(new FileList(file_list_name)),
      NumChunks(num_chunks),
      NumThreads(num_threads),
      label_dense_buffers_(prototype.label_dense_buffers_),
      label_tensors_(prototype.label_tensors_),
      dense_tensors_(prototype.dense_tensors_),
      check_type_(prototype.check_type_),
      csr_buffers_(prototype.csr_buffers_),
      row_offsets_tensors_(prototype.row_offsets_tensors_),
      value_tensors_(prototype.value_tensors_),
      params_(prototype.params_),
      device_resources_(prototype.device_resources_),
      batchsize_(prototype.batchsize_),
      label_dim_(prototype.label_dim_),
      dense_dim_(prototype.dense_dim_)
      {
  shared_output_flag_ = true;
  data_reader_loop_flag_ = 1;

  create_heap_workers_();

  data_collector_.reset(
     new DataCollector<TypeKey>(label_tensors_, dense_tensors_, csr_buffers_, device_resources_, csr_heap_));

  data_collector_thread_ =
      std::thread(data_collector_thread_func_<TypeKey>, data_collector_, &data_reader_loop_flag_);
}

template <typename TypeKey>
DataReader<TypeKey>::DataReader(const DataReader<TypeKey>& prototype)
    : label_dense_buffers_(prototype.label_dense_buffers_),
      label_tensors_(prototype.label_tensors_),
      dense_tensors_(prototype.dense_tensors_),
      check_type_(prototype.check_type_),
      csr_buffers_(prototype.csr_buffers_),
      row_offsets_tensors_(prototype.row_offsets_tensors_),
      value_tensors_(prototype.value_tensors_),
      params_(prototype.params_),
      device_resources_(prototype.device_resources_),
      batchsize_(prototype.batchsize_),
      label_dim_(prototype.label_dim_),
      dense_dim_(prototype.dense_dim_)
      {
  shared_output_flag_ = true;
  data_reader_loop_flag_ = 1;

  data_collector_.reset(
      new DataCollector<TypeKey>(label_tensors_, dense_tensors_, csr_buffers_, device_resources_));

  data_collector_thread_ =
      std::thread(data_collector_thread_func_<TypeKey>, data_collector_, &data_reader_loop_flag_);
}

template <typename TypeKey>
DataReader<TypeKey>::DataReader(const std::string& file_list_name, int batchsize, int label_dim, int dense_dim,
				Check_t check_type, std::vector<DataReaderSparseParam>& params,
                                const std::shared_ptr<GPUResourceGroup>& gpu_resource_group,
                                int num_chunks, int num_threads)
  : file_list_(new FileList(file_list_name)),
    NumChunks(num_chunks),
    NumThreads(num_threads),
    check_type_(check_type),
    params_(params),
    device_resources_(gpu_resource_group),
    batchsize_(batchsize),
    label_dim_(label_dim),
    dense_dim_(dense_dim)
{

  data_reader_loop_flag_ = 1;
  int total_gpu_count = device_resources_->get_total_gpu_count();

  // input check
  if (total_gpu_count == 0 || batchsize <= 0 || label_dim <= 0 || dense_dim < 0 ||
      0 != batchsize_ % total_gpu_count) {
    CK_THROW_(Error_t::WrongInput, "total_gpu_count == 0 || batchsize <= 0 || label_dim <= 0 || dense_dim < 0 || 0 != batchsize_ % total_gpu_count");
  }

  create_heap_workers_();

  auto& device_list = device_resources_->get_device_list();

  // create label and dense tensor
  int batch_size_per_device = batchsize_ / total_gpu_count;
  assert(label_tensors_.empty() && label_buffers_.empty() && dense_tensors_.empty());
  for (auto device_id : device_list) {
    std::shared_ptr<GeneralBuffer<float>> tmp_label_dense_buff(new GeneralBuffer<float>());
    label_tensors_.emplace_back(
      new Tensor<float>({batch_size_per_device, label_dim_}, tmp_label_dense_buff, TensorFormat_t::HW));
    dense_tensors_.emplace_back(
      new Tensor<float>({batch_size_per_device, dense_dim_}, tmp_label_dense_buff, TensorFormat_t::HW));

    tmp_label_dense_buff->init(device_id);
    label_dense_buffers_.emplace_back(tmp_label_dense_buff);
  }

  // create value and row offset tensor
  for (auto device_id : device_list) {
    for(auto& param : params){
      int slots = 0;
      if(param.type == DataReaderSparse_t::Distributed){
	slots = param.slot_num;
      }
      else if(param.type == DataReaderSparse_t::Localized){
	int mod_slots = param.slot_num % total_gpu_count; //ceiling
	int global_id = gpu_resource_group->get_global_id(device_id);
	if(global_id < mod_slots){
	  slots = param.slot_num / total_gpu_count + 1;
	}
	else{
	  slots = param.slot_num / total_gpu_count;
	}
      }
      std::shared_ptr<GeneralBuffer<TypeKey>> tmp_buffer(new GeneralBuffer<TypeKey>());
      std::vector<int> num_rows_dim = {1, batchsize_ * slots + 1};      
      Tensor<TypeKey>* tmp_row_offset =
	new Tensor<TypeKey>(num_rows_dim, tmp_buffer, TensorFormat_t::HW);
      std::vector<int> num_max_value_dim = {1, param.max_feature_num * batchsize_};
      Tensor<TypeKey>* tmp_value =
	new Tensor<TypeKey>(num_max_value_dim, tmp_buffer, TensorFormat_t::HW);
      tmp_buffer->init(device_id);

      row_offsets_tensors_.emplace_back(tmp_row_offset);
      value_tensors_.emplace_back(tmp_value);
      csr_buffers_.emplace_back(tmp_buffer);
    }
  }

  data_collector_.reset(new DataCollector<TypeKey>(label_tensors_, dense_tensors_, csr_buffers_, device_resources_,
						   csr_heap_, false));

  data_collector_thread_ =
    std::thread(data_collector_thread_func_<TypeKey>, data_collector_, &data_reader_loop_flag_);
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
    data_reader_loop_flag_ = 0;
    for (auto& data_reader : data_readers_) {
      data_reader->skip_read();
    }
    if (csr_heap_ != nullptr) {
      csr_heap_->break_and_return();
    }
    data_collector_->stop();

    data_collector_thread_.join();

    for (auto& data_reader_thread : data_reader_threads_) {
      data_reader_thread.join();
    }


  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

}  // namespace HugeCTR
