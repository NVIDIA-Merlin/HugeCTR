#include <common.hpp>
#include <data_reader.hpp>
#include <data_readers/async_reader/async_reader.hpp>
#include <data_readers/async_reader/async_reader_common.hpp>
#include <data_readers/async_reader/split_label_dense_sparse.hpp>
#include <tensor2.hpp>

namespace HugeCTR {

template <typename SparseType>
class AsyncReader : public IDataReaderWithScheduling {
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
  void ready_to_collect() override;
  long long read_a_batch_to_device() override;
  void schedule_here(cudaStream_t stream, int raw_device_id) override;
  void schedule_here_graph(cudaStream_t stream, int raw_device_id) override;
  void update_schedule_graph(int raw_device_id) override;

  long long get_current_batchsize_per_device(size_t local_id) override;
  TensorScalarType get_scalar_type() const override;
  bool is_started() const override;
  void start() override;

  std::vector<TensorBag2> get_label_tensors() const;
  std::vector<TensorBag2> get_dense_tensors() const;
  SparseTensors<SparseType> get_value_tensors() const;

  void create_drwg_norm(std::string file_list, Check_t check_type,
                        bool start_reading_from_beginning = true) override;
  void create_drwg_raw(std::string file_name, long long num_samples, bool float_label_dense,
                       bool data_shuffle, bool start_reading_from_beginning = true) override;
#ifndef DISABLE_CUDF
  void create_drwg_parquet(std::string file_list, const std::vector<long long> slot_offset,
                           bool start_reading_from_beginning = true) override;
#endif
  void set_source(std::string file_list = std::string()) override;

  ~AsyncReader();

 private:
  const std::shared_ptr<ResourceManager> resource_manager_;
  std::unique_ptr<AsyncReaderImpl> reader_impl_;
  size_t sample_size_items_, current_batch_size_;
  bool mixed_precision_, wait_for_gpu_idle_;
  size_t batch_size_, batch_size_per_dev_;

  std::vector<TensorBag2> label_tensors_;
  std::vector<TensorBag2> dense_tensors_;
  std::vector<TensorBag2> label_tensors_per_dev_;
  std::vector<TensorBag2> dense_tensors_per_dev_;
  // std::vector<TensorBag2> sparse_tensors_;
  std::vector<SparseTensor<SparseType>> sparse_tensors_;
  std::vector<cudaEvent_t> completion_events_, schedule_events_;
};

}  // namespace HugeCTR
