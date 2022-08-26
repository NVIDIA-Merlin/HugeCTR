#pragma once

#include <common.hpp>
#include <data_readers/async_reader/async_reader.hpp>
#include <data_readers/async_reader/async_reader_common.hpp>
#include <data_readers/async_reader/split_label_dense_sparse.hpp>
#include <tensor2.hpp>

#include "HugeCTR/include/data_source/hdfs_backend.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/indices_container.hpp"
#include "HugeCTR/include/graph_wrapper.hpp"
#include "scheduleable.hpp"

namespace HugeCTR {

template <typename SparseType>
class AsyncReader : public SchedulableDataReader {
  using LabelType = float;
  using InputType = int;

 public:
  // Default params: num_threads = num_local_gpus, io_block_size = 512000, io_depth = 2,
  // io_alignment = 512
  AsyncReader(std::string fname, size_t batch_size, size_t label_dim, size_t dense_dim,
              std::vector<DataReaderSparseParam>& params, bool mixed_precision,
              const std::shared_ptr<ResourceManager>& resource_manager, int num_threads,
              int num_batches_per_thread, size_t io_block_size, int io_depth, int io_alignment,
              bool shuffle = false, bool wait_for_gpu_idle = false,
              Alignment_t aligned = Alignment_t::None);

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
                           bool start_reading_from_beginning = true) override;
#endif
  void set_source(std::string file_list = std::string()) override;

  ~AsyncReader();

 private:
  struct BatchTensors {
    size_t tag;
    std::vector<TensorBag2> label_tensors;
    std::vector<TensorBag2> dense_tensors;
    std::vector<SparseTensorBag> sparse_tensors;
  };

  void assign_dense_and_label_tensors(TensorBag2 label_tensor, TensorBag2 dense_tensor,
                                      int raw_device_id, cudaStream_t stream);

  void init_batch_tensors(size_t num_inflight);

  const std::shared_ptr<ResourceManager> resource_manager_;
  std::unique_ptr<AsyncReaderImpl> reader_impl_;
  size_t sample_size_items_, current_batch_size_;
  bool mixed_precision_, wait_for_gpu_idle_;
  size_t batch_size_, batch_size_per_dev_;
  size_t label_dim_, dense_dim_, sparse_dim_;

  size_t inflight_id_ = 0;
  std::vector<BatchTensors> inflight_batch_tensors_;  // inflight batches

  std::vector<TensorBag2> label_tensors_;
  std::vector<TensorBag2> dense_tensors_;
  std::vector<SparseTensorBag> current_sparse_tensors_;

  bool current_batch_cached_ = false;

  std::vector<cudaEvent_t> completion_events_;
  std::vector<cudaEvent_t> schedule_events_;
  std::vector<cudaEvent_t> split_schedule_events_;
  std::vector<cudaEvent_t> d2d_schedule_events_;

  std::vector<cudaStream_t> s3w_streams_;  // split_3_way streams
  std::vector<cudaStream_t> d2d_streams_;  // d2d copy streams

  bool cache_buffers_ = false;
};

}  // namespace HugeCTR
