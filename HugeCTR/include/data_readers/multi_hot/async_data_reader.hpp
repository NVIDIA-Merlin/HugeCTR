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

#include <data_readers/multi_hot/detail/data_reader_impl.hpp>
#include <scheduleable.hpp>
#include <tensor2.hpp>

namespace HugeCTR {
namespace MultiHot {

template <typename SparseType>
class AsyncDataReader : public SchedulableDataReader {
  using LabelType = float;
  using InputType = int;

 public:
  AsyncDataReader(std::vector<FileSource> data_files,
                  const std::shared_ptr<ResourceManager>& resource_manager, size_t batch_size,
                  size_t num_threads_per_file, size_t num_batches_per_thread,
                  const std::vector<DataReaderSparseParam>& params, size_t label_dim,
                  size_t dense_dim, bool mixed_precision, bool shuffle,
                  bool schedule_uploads = false, bool is_dense_float = false);

  long long read_a_batch_to_device_delay_release() override;
  long long get_full_batchsize() const override;

  cudaStream_t get_split_3_way_stream(int raw_device_id) const {
    return s3w_streams_.at(raw_device_id);
  }
  cudaStream_t get_d2d_stream(int raw_device_id) const { return d2d_streams_.at(raw_device_id); }

  void set_schedule_streams(cudaStream_t s3w_stream, cudaStream_t d2d_stream,
                            int raw_device_id) override;

  void stream_wait_sparse_tensors(cudaStream_t stream, int raw_device_id, bool from_graph) override;
  void stream_wait_dense_tensors(cudaStream_t stream, int raw_device_id, bool from_graph) override;

  /**
   * @brief Once the batch is retrieved from the AsyncReaderImpl, the batch needs to be
   * split into its respective tensor buffers. This allows us to buffer the last N batches
   * with their respective tensors.
   */
  void set_tensor_buffering(size_t num_batches_to_buffer);

  bool current_batch_incomplete() const override;
  void ready_to_collect() override;
  long long read_a_batch_to_device() override;
  void schedule_split_3_way_here(cudaStream_t stream, int raw_device_id, bool from_graph) override;
  void schedule_d2d_here(cudaStream_t stream, int raw_device_id, bool from_graph) override;
  void schedule_here(cudaStream_t stream, int raw_device_id) override;
  void schedule_here_graph(cudaStream_t stream, int raw_device_id) override;
  void update_schedule_graph(int raw_device_id) override;

  size_t get_max_batches_inflight() const;
  bool is_mixed_precision();
  // TODO: need to get rid of this, pass the dims directly from Model to the HybridEmbedding
  void get_dimensions(size_t& label_dim, size_t& dense_dim, size_t& sparse_dim,
                      size_t& sample_size_items);

  long long get_current_batchsize_per_device(size_t local_id) override;
  TensorScalarType get_scalar_type() const override;
  bool is_started() const override;
  void start() override;

  std::vector<TensorBag2> get_label_tensors() const;
  std::vector<TensorBag2> get_dense_tensors() const;
  SparseTensors<SparseType> get_value_tensors() const;
  std::vector<std::vector<SparseTensor<SparseType>>> get_current_sparse_tensors() const;
  bool is_batch_cached() const { return current_batch_cached_; }
  size_t get_current_inflight_id() const { return inflight_id_; }  // TODO: remove?

  // FIXME: This is a temporary fix to get around the fact that HybridSpaseEmbedding
  // needs to be constructed with the SparseTensor buffers
  std::vector<std::vector<SparseTensor<SparseType>>> get_value_tensor_buffers() const;

  void create_drwg_norm(std::string file_list, Check_t check_type,
                        bool start_reading_from_beginning = true) override;
  void create_drwg_raw(std::string file_name, long long num_samples, bool float_label_dense,
                       bool data_shuffle, bool start_reading_from_beginning = true) override;
#ifndef DISABLE_CUDF
  void create_drwg_parquet(std::string file_list, bool strict_order_of_batches,
                           const std::vector<long long> slot_offset,
                           bool start_reading_from_beginning = true,
                           long long max_samples_per_group = 0, int label_dense_num = 0,
                           int label_dense_dim = 0) override;
#endif
  void set_source(std::string file_list = std::string()) override;

  ~AsyncDataReader();

 private:
  struct BatchTensors {
    size_t tag;
    std::vector<TensorBag2> label_tensors;
    std::vector<TensorBag2> dense_tensors;
    std::vector<Tensor2<SparseType*>> sparse_tensor_ptrs;
    std::vector<std::vector<SparseTensor<SparseType>>> sparse_tensors;
  };

  void assign_dense_and_label_tensors(TensorBag2 label_tensor, TensorBag2 dense_tensor,
                                      int raw_device_id, cudaStream_t stream);

  void init_batch_tensors(size_t num_inflight);

  const std::shared_ptr<ResourceManager> resource_manager_;
  std::unique_ptr<DataReaderImpl> reader_impl_;
  size_t sample_size_items_, current_batch_size_;
  bool mixed_precision_;
  size_t batch_size_, batch_size_per_dev_;
  size_t label_dim_, dense_dim_, sparse_dim_;
  std::vector<int> nnz_per_slot_;
  size_t total_nnz_;

  size_t inflight_id_ = 0;
  std::vector<BatchTensors> inflight_batch_tensors_;  // inflight batches

  std::vector<TensorBag2> label_tensors_;
  std::vector<TensorBag2> dense_tensors_;
  //  std::vector<SparseTensorBag> current_sparse_tensors_;
  std::vector<std::vector<SparseTensor<SparseType>>>
      current_sparse_tensors_;  // [gpu][categorical_feature]

  bool current_batch_cached_ = false;

  std::vector<cudaEvent_t> completion_events_;
  std::vector<cudaEvent_t> schedule_events_;
  std::vector<cudaEvent_t> split_schedule_events_;
  std::vector<cudaEvent_t> d2d_schedule_events_;

  std::vector<cudaStream_t> s3w_streams_;  // split_3_way streams
  std::vector<cudaStream_t> d2d_streams_;  // d2d copy streams

  bool cache_buffers_ = false;

  std::vector<Tensor2<int>> bucket_id_tensors_;
  std::vector<Tensor2<int>> bucket_position_tensors_;
  std::vector<Tensor2<int>> max_hotness_tensors_;
  bool is_dense_float_;
};

}  // namespace MultiHot
}  // namespace HugeCTR
