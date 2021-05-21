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

#include <atomic>
#include <common.hpp>
#include <data_reader.hpp>
#include <data_readers/csr.hpp>
#include <data_readers/csr_chunk.hpp>
#include <data_readers/data_collector.hpp>
#include <data_readers/data_reader_worker_group.hpp>
#include <data_readers/data_reader_worker_group_norm.hpp>

#ifndef DISABLE_CUDF
#include <data_readers/data_reader_worker_group_parquet.hpp>
#endif

#include <data_readers/data_reader_worker_group_raw.hpp>
#include <data_readers/file_list.hpp>
#include <fstream>
#include <gpu_resource.hpp>
#include <tensor2.hpp>
#include <utils.hpp>
#include <vector>

namespace HugeCTR {

/**
 * @brief Data reading controller.
 *
 * Control the data reading from data set to embedding.
 * An instance of DataReader will maintain independent
 * threads for data reading (IDataReaderWorker)
 * from dataset to heap. Meanwhile one independent
 * thread consumes the data (DataCollector),
 * and copy the data to GPU buffer.
 */
template <typename TypeKey>
class DataReader : public IDataReader {
 private:
  std::shared_ptr<HeapEx<CSRChunk<TypeKey>>> csr_heap_; /**< heap to cache the data set */
  Tensors2<float> label_tensors_;                       /**< Label tensors for the usage of loss */
  std::vector<TensorBag2> dense_tensors_;               /**< Dense tensors for the usage of loss */
  /* Each gpu will have several csr output for different embedding */
  Tensors2<TypeKey> csr_buffers_; /**< csr_buffers contains row_offset_tensor and value_tensors */
  Tensors2<TypeKey> row_offsets_tensors_; /**< row offset tensors*/
  Tensors2<TypeKey> value_tensors_;       /**< value tensors */
  std::vector<std::shared_ptr<size_t>> nnz_array_;
  const std::vector<DataReaderSparseParam> params_;
  std::shared_ptr<ResourceManager> resource_manager_; /**< gpu resource used in this data reader*/
  bool use_mixed_precision_{false};
  const size_t batchsize_; /**< batch size */
  const size_t label_dim_; /**< dimention of label e.g. 1 for BinaryCrossEntropy */
  const size_t dense_dim_; /**< dimention of dense */
  std::shared_ptr<DataCollector<TypeKey>> data_collector_; /**< pointer of DataCollector */
  std::shared_ptr<DataReaderWorkerGroup> worker_group_;
  long long current_batchsize_;

  bool repeat_;
  std::string file_list_path_;

 public:
  /**
   * Reading a batch from cpu to gpu (embedding)
   */
  TensorScalarType get_scalar_type() const override {
    return TensorScalarTypeFunc<TypeKey>::get_type();
  }

  long long read_a_batch_to_device() override;  // read data from csr to tensors

  long long read_a_batch_to_device_delay_release() override;

  long long get_current_batchsize_per_device(size_t local_id) override {
    if (batchsize_ % resource_manager_->get_global_gpu_count() != 0) {
      CK_THROW_(Error_t::UnspecificError,
                "batchsize_ % resource_manager_->get_global_gpu_count() != 0");
    }
    long long batchsize_per_device = batchsize_ / resource_manager_->get_global_gpu_count();
    size_t global_id = resource_manager_->get_gpu_global_id_from_local_id(local_id);
    long long remain_samples = current_batchsize_ - global_id * batchsize_per_device;
    if (remain_samples >= batchsize_per_device) {
      return batchsize_per_device;
    } else if (remain_samples > 0) {
      return remain_samples;
    } else {
      return 0;
    }
  }

  void ready_to_collect() override { data_collector_->set_ready_to_write_sync(); }

  bool is_started() const override { return worker_group_->is_started(); }
  void start() override { worker_group_->start(); }

  DataReader(int batchsize, size_t label_dim, int dense_dim,
             std::vector<DataReaderSparseParam>& params,
             const std::shared_ptr<ResourceManager>& resource_manager, bool repeat,
             int num_chunk_threads = 31, bool use_mixed_precision = false, int cache_num_iters = 0,
             Alignment_t aligned = Alignment_t::None);

  std::vector<TensorBag2> get_label_tensors() const override {
    return tensors_to_bags(label_tensors_);
  }
  std::vector<TensorBag2> get_dense_tensors() const override { return dense_tensors_; }
  std::vector<TensorBag2> get_row_offsets_tensors() const {
    return tensors_to_bags(row_offsets_tensors_);
  }
  std::vector<TensorBag2> get_value_tensors() const { return tensors_to_bags(value_tensors_); }
  const std::vector<std::shared_ptr<size_t>> get_nnz_array() const { return nnz_array_; }
  const Tensors2<TypeKey> get_row_offsets_tensors(int param_id) const {
    Tensors2<TypeKey> tensors;
    for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
      tensors.push_back(row_offsets_tensors_[i * params_.size() + param_id]);
    }
    return tensors;
  }
  const Tensors2<TypeKey> get_value_tensors(int param_id) const {
    Tensors2<TypeKey> tensors;
    for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
      tensors.push_back(value_tensors_[i * params_.size() + param_id]);
    }
    return tensors;
  }
  const std::vector<std::shared_ptr<size_t>> get_nnz_array(int param_id) const {
    std::vector<std::shared_ptr<size_t>> array;
    for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
      array.push_back(nnz_array_[i * params_.size() + param_id]);
    }
    return array;
  }

  void create_drwg_norm(std::string file_list, Check_t check_type,
                        bool start_reading_from_beginning = true) override {
    worker_group_.reset(new DataReaderWorkerGroupNorm<TypeKey>(
        csr_heap_, file_list, repeat_, check_type, params_, start_reading_from_beginning));
    file_list_path_ = file_list;
  }

  void create_drwg_raw(std::string file_name, long long num_samples,
                       const std::vector<long long> slot_offset, bool float_label_dense,
                       bool data_shuffle = false,
                       bool start_reading_from_beginning = true) override {
    worker_group_.reset(new DataReaderWorkerGroupRaw<TypeKey>(
        csr_heap_, file_name, num_samples, params_, slot_offset, label_dim_, dense_dim_, batchsize_,
        float_label_dense, data_shuffle, start_reading_from_beginning));
  }

#ifndef DISABLE_CUDF
  void create_drwg_parquet(std::string file_list, const std::vector<long long> slot_offset,
                           bool start_reading_from_beginning = true) override {
    // worker_group_.empty
    worker_group_.reset(new DataReaderWorkerGroupParquet<TypeKey>(csr_heap_, file_list, params_,
                                                                  slot_offset, resource_manager_,
                                                                  start_reading_from_beginning));
  }
#endif

  void set_file_list_source(std::string file_list = std::string()) override {
    // TODO: if the underlying workers are for Parquet, throw the exception
    try {
      if (worker_group_ != nullptr) {
        if (file_list.empty()) {
          if (file_list_path_.empty()) {
            throw internal_runtime_error(Error_t::NotInitialized, "invalid file_list path");
          } else {
            file_list = file_list_path_;
          }
        }
        bool repeat = repeat_;
        auto op = [file_list, repeat](int worker_id, int num_workers) {
          return std::shared_ptr<Source>(new FileSource(worker_id, num_workers, file_list, repeat));
        };
        csr_heap_->reset();
        worker_group_->set_source(op);
      } else {
        throw internal_runtime_error(Error_t::NotInitialized, "worker_group_ == nullptr");
      }
    } catch (const internal_runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
  }

  ~DataReader() override;
};

template <typename TypeKey>
DataReader<TypeKey>::DataReader(int batchsize, size_t label_dim, int dense_dim,
                                std::vector<DataReaderSparseParam>& params,
                                const std::shared_ptr<ResourceManager>& resource_manager,
                                bool repeat, int num_chunk_threads, bool use_mixed_precision,
                                int cache_num_iters, Alignment_t aligned)
    : params_(params),
      resource_manager_(resource_manager),
      use_mixed_precision_(use_mixed_precision),
      batchsize_(batchsize),
      label_dim_(label_dim),
      dense_dim_(dense_dim),
      repeat_(repeat)

{
  size_t local_gpu_count = resource_manager_->get_local_gpu_count();
  size_t total_gpu_count = resource_manager_->get_global_gpu_count();

  // input check
  if (total_gpu_count == 0 || batchsize <= 0 || label_dim <= 0 || dense_dim < 0 ||
      0 != batchsize_ % total_gpu_count) {
    CK_THROW_(Error_t::WrongInput,
              "total_gpu_count == 0 || batchsize <= 0 || label_dim <= 0 || dense_dim < 0 || 0 != "
              "batchsize_ % total_gpu_count");
  }

  // init the heap
  csr_heap_.reset(new HeapEx<CSRChunk<TypeKey>>(num_chunk_threads, total_gpu_count, batchsize_,
                                                label_dim_ + dense_dim_, params_));

  std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> buffs;
  for (size_t i = 0; i < local_gpu_count; i++) {
    buffs.push_back(GeneralBuffer2<CudaAllocator>::create());
  }

  size_t dense_dim_align8 = dense_dim_;
  if (aligned == Alignment_t::Auto) dense_dim_align8 = (dense_dim_ + 7) / 8 * 8;

  // create label and dense tensor
  size_t batch_size_per_device = batchsize_ / total_gpu_count;
  for (size_t i = 0; i < local_gpu_count; i++) {
    {
      Tensor2<float> tensor;
      buffs[i]->reserve({batch_size_per_device, label_dim_}, &tensor);
      label_tensors_.push_back(tensor);
    }
    if (use_mixed_precision_) {
      Tensor2<__half> tensor;
      buffs[i]->reserve({batch_size_per_device, dense_dim_align8}, &tensor);
      dense_tensors_.push_back(tensor.shrink());
    } else {
      Tensor2<float> tensor;
      buffs[i]->reserve({batch_size_per_device, dense_dim_align8}, &tensor);
      dense_tensors_.push_back(tensor.shrink());
    }
  }

  // create value and row offset tensor
  bool one_hot = true;
  for (size_t i = 0; i < local_gpu_count; i++) {
    for (auto& param : params) {
      int slots = 0;
      if (param.type == DataReaderSparse_t::Distributed) {
        slots = param.slot_num;
        one_hot = false;
      } else if (param.type == DataReaderSparse_t::Localized) {
        size_t mod_slots = static_cast<size_t>(param.slot_num) % total_gpu_count;  // ceiling
        size_t global_id = resource_manager_->get_local_gpu(i)->get_global_id();
        if (global_id < mod_slots) {
          slots = param.slot_num / total_gpu_count + 1;
        } else {
          slots = param.slot_num / total_gpu_count;
        }
      }

      std::shared_ptr<BufferBlock2<TypeKey>> blockbuff = buffs[i]->create_block<TypeKey>();

      std::vector<size_t> num_rows_dim = {1, batchsize_ * slots + 1};
      Tensor2<TypeKey> tmp_row_offset;
      blockbuff->reserve(num_rows_dim, &tmp_row_offset);

      size_t num_max_value = (param.max_nnz * slots) <= param.max_feature_num
                                 ? (param.max_nnz * slots * batchsize_)
                                 : (param.max_feature_num * batchsize_);
      if (param.max_nnz != 1) {
        one_hot = false;  // Note: here we have an assumption if max_nnz == 1 the input is one hot.
      }

      std::vector<size_t> num_max_value_dim = {1, num_max_value};

      Tensor2<TypeKey> tmp_value;
      blockbuff->reserve(num_max_value_dim, &tmp_value);

      row_offsets_tensors_.emplace_back(tmp_row_offset);
      value_tensors_.emplace_back(tmp_value);
      nnz_array_.emplace_back(new size_t);
      csr_buffers_.emplace_back(blockbuff->as_tensor());
    }
  }

  if (cache_num_iters) {
    data_collector_.reset(new DataCollector<TypeKey>(
        label_tensors_, dense_tensors_, csr_buffers_, nnz_array_, buffs, resource_manager_,
        csr_heap_, use_mixed_precision_, one_hot, cache_num_iters));
  } else {
    data_collector_.reset(new DataCollector<TypeKey>(label_tensors_, dense_tensors_, csr_buffers_,
                                                     nnz_array_, buffs, resource_manager_,
                                                     csr_heap_, use_mixed_precision_, one_hot));
  }
  data_collector_->start();

  for (size_t i = 0; i < local_gpu_count; i++) {
    CudaDeviceContext context(resource_manager_->get_local_gpu(i)->get_device_id());
    buffs[i]->allocate();
  }

  // zero-initialization
  for (size_t i = 0; i < local_gpu_count; i++) {
    const auto local_gpu = resource_manager_->get_local_gpu(i);
    if (use_mixed_precision_) {
      Tensor2<__half> tensor = Tensor2<__half>::stretch_from(dense_tensors_[i]);
      CK_CUDA_THROW_(cudaMemsetAsync(tensor.get_ptr(), 0,
                                     tensor.get_num_elements() * sizeof(__half),
                                     local_gpu->get_memcpy_stream()));
    } else {
      Tensor2<float> tensor = Tensor2<float>::stretch_from(dense_tensors_[i]);
      CK_CUDA_THROW_(cudaMemsetAsync(tensor.get_ptr(), 0, tensor.get_num_elements() * sizeof(float),
                                     local_gpu->get_memcpy_stream()));
    }
  }

  core_offset_ += 64;

  return;
}

template <typename TypeKey>
long long DataReader<TypeKey>::read_a_batch_to_device_delay_release() {
  long long current_batchsize;
  current_batchsize = data_collector_->read_a_batch_to_device();
  if (current_batchsize == 0) {
    data_collector_->set_ready_to_write_sync();
  }
  current_batchsize_ = current_batchsize;
  return current_batchsize;
}

template <typename TypeKey>
long long DataReader<TypeKey>::read_a_batch_to_device() {
  long long current_batchsize = read_a_batch_to_device_delay_release();
  data_collector_->set_ready_to_write_sync();
  current_batchsize_ = current_batchsize;
  return current_batchsize;
}

template <typename TypeKey>
DataReader<TypeKey>::~DataReader() {
  try {
    // stop all the loops
    worker_group_->end();

    data_collector_->stop();
    if (csr_heap_ != nullptr) {
      csr_heap_->break_and_return();
    }
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

}  // namespace HugeCTR
