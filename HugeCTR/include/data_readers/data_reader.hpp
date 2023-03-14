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
#pragma once

#include <atomic>
#include <common.hpp>
#include <data_reader.hpp>
#include <data_readers/csr.hpp>
#include <data_readers/data_collector.hpp>
#include <data_readers/data_reader_common.hpp>
#include <data_readers/data_reader_worker_group.hpp>
#include <data_readers/data_reader_worker_group_norm.hpp>

#ifndef DISABLE_CUDF
#include <data_readers/data_reader_worker_group_parquet.hpp>
#endif

#include <data_readers/data_reader_worker_group_raw.hpp>
#include <filesystem>
#include <fstream>
#include <gpu_resource.hpp>
#include <tensor2.hpp>
#include <utils.hpp>
#include <vector>
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

namespace HugeCTR {

namespace core23_reader {

template <typename TypeKey>
class DataReader : public IDataReader {
 private:
  std::vector<std::shared_ptr<ThreadBuffer23>> thread_buffers_;  // gpu_id -> thread_idx
  std::shared_ptr<BroadcastBuffer23> broadcast_buffer_;
  // TODO FIXME with core23
  std::shared_ptr<DataReaderOutput> output_;
  std::shared_ptr<DataReaderOutput23> output23_;

  std::shared_ptr<DataReaderWorkerGroup> worker_group_;
  std::shared_ptr<core23_reader::DataCollector<TypeKey>>
      data_collector_; /**< pointer of DataCollector */

  /* Each gpu will have several csr output for different embedding */
  const std::vector<DataReaderSparseParam> params_;
  std::shared_ptr<ResourceManager> resource_manager_; /**< gpu resource used in this data reader*/
  const long long batchsize_;                         /**< batch size */
  const size_t label_dim_; /**< dimention of label e.g. 1 for BinaryCrossEntropy */
  const size_t dense_dim_; /**< dimention of dense */
  long long current_batchsize_;

  bool repeat_;
  std::string file_name_;
  SourceType_t source_type_;
  const DataSourceParams data_source_params_;

 public:
  DataReader(int batchsize, size_t label_dim, int dense_dim,
             std::vector<DataReaderSparseParam> &params,
             const std::shared_ptr<ResourceManager> &resource_manager, bool repeat, int num_threads,
             bool use_mixed_precision,
             const DataSourceParams &data_source_params = DataSourceParams());
  ~DataReader();

  /**
   * Reading a batch from cpu to gpu (embedding)
   */
  TensorScalarType get_scalar_type() const override {
    return TensorScalarTypeFunc<TypeKey>::get_type();
  }

  long long read_a_batch_to_device() override;
  long long read_a_batch_to_device_delay_release() override;

  void ready_to_collect() override;

  long long get_current_batchsize_per_device(size_t local_id) override;

  long long get_full_batchsize() const override;

  bool current_batch_incomplete() const override;

  bool is_started() const override;

  void start() override;
  const std::vector<SparseTensorBag> &get_sparse_tensors(const std::string &name);
  const std::vector<TensorBag2> &get_label_tensors() const;
  const std::vector<TensorBag2> &get_dense_tensors() const;

  const std::vector<SparseTensor23> &get_sparse_core23_tensors(const std::string &name);
  const std::vector<core23::Tensor> &get_label_core23_tensors23() const;
  const std::vector<core23::Tensor> &get_dense_core23_tensors23() const;

  void create_drwg_norm(std::string file_name, Check_t check_type,
                        bool start_reading_from_beginning = true) override;

  void create_drwg_raw(std::string file_name, long long num_samples, bool float_label_dense,
                       bool data_shuffle = false,
                       bool start_reading_from_beginning = true) override;
#ifndef DISABLE_CUDF

  void create_drwg_parquet(std::string file_list, bool strict_order_of_batches,
                           const std::vector<long long> slot_offset,
                           bool start_reading_from_beginning = true,
                           long long max_samples_per_group = 0, int label_dense_num = 0,
                           int label_dense_dim = 0) override;
#endif
  void set_source(std::string file_name = std::string()) override;
};
};  // namespace core23_reader

template <typename TypeKey>
class DataReader : public IDataReader {
 private:
  std::vector<std::shared_ptr<ThreadBuffer>> thread_buffers_;  // gpu_id -> thread_idx
  std::shared_ptr<BroadcastBuffer> broadcast_buffer_;
  std::shared_ptr<DataReaderOutput> output_;

  std::shared_ptr<DataReaderWorkerGroup> worker_group_;
  std::shared_ptr<DataCollector<TypeKey>> data_collector_; /**< pointer of DataCollector */

  /* Each gpu will have several csr output for different embedding */
  const std::vector<DataReaderSparseParam> params_;
  std::shared_ptr<ResourceManager> resource_manager_; /**< gpu resource used in this data reader*/
  const long long batchsize_;                         /**< batch size */
  const size_t label_dim_; /**< dimention of label e.g. 1 for BinaryCrossEntropy */
  const size_t dense_dim_; /**< dimention of dense */
  long long current_batchsize_;

  bool repeat_;
  std::string file_name_;
  SourceType_t source_type_;
  const DataSourceParams data_source_params_;

 public:
  DataReader(int batchsize, size_t label_dim, int dense_dim,
             std::vector<DataReaderSparseParam> &params,
             const std::shared_ptr<ResourceManager> &resource_manager, bool repeat, int num_threads,
             bool use_mixed_precision,
             const DataSourceParams &data_source_params = DataSourceParams());
  ~DataReader() override;

  TensorScalarType get_scalar_type() const override {
    return TensorScalarTypeFunc<TypeKey>::get_type();
  }

  long long read_a_batch_to_device() override;
  long long read_a_batch_to_device_delay_release() override;
  void ready_to_collect() override;
  long long get_current_batchsize_per_device(size_t local_id) override;
  long long get_full_batchsize() const override;
  bool current_batch_incomplete() const override;
  bool is_started() const override;
  void start() override;
  const std::vector<SparseTensorBag> &get_sparse_tensors(const std::string &name);

  const std::vector<TensorBag2> &get_label_tensors() const;

  const std::vector<TensorBag2> &get_dense_tensors() const;

  void create_drwg_norm(std::string file_name, Check_t check_type,
                        bool start_reading_from_beginning = true) override;

  void create_drwg_raw(std::string file_name, long long num_samples, bool float_label_dense,
                       bool data_shuffle = false,
                       bool start_reading_from_beginning = true) override;

#ifndef DISABLE_CUDF

  void create_drwg_parquet(std::string file_list, bool strict_order_of_batches,
                           const std::vector<long long> slot_offset,
                           bool start_reading_from_beginning = true,
                           long long max_samples_per_group = 0, int label_dense_num = 0,
                           int label_dense_dim = 0) override;
#endif

  void set_source(std::string file_name = std::string()) override;
};
}  // namespace HugeCTR
