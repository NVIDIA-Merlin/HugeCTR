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

#include <unistd.h>
#include <common.hpp>
#include <csr.hpp>
#include <csr_chunk.hpp>
#include <heapex.hpp>
#include <resource_manager.hpp>
#include <utils.hpp>
#ifdef ENABLE_MPI
#include <mpi.h>
#endif
namespace HugeCTR {

#ifdef ENABLE_MPI
template <typename TypeKey>
struct ToMpiType;

template <>
struct ToMpiType<long long> {
  static MPI_Datatype T() { return MPI_LONG_LONG; }
};

template <>
struct ToMpiType<unsigned int> {
  static MPI_Datatype T() { return MPI_UNSIGNED; }
};

template <>
struct ToMpiType<float> {
  static MPI_Datatype T() { return MPI_FLOAT; }
};

#endif

template <typename TypeComp>
void split(Tensor2<float>& label_tensor, Tensor2<TypeComp>& dense_tensor,
           const Tensor2<float>& label_dense_buffer, cudaStream_t stream);

/**
 * @brief A helper class of data reader.
 *
 * This class implement asynchronized data collecting from heap
 * to output of data reader, thus data collection and training
 * can work in a pipeline.
 */
template <typename TypeKey>
class DataCollector {
 private:
  static int id_;

  enum STATUS { READY_TO_WRITE, READY_TO_READ, STOP };
  std::atomic<STATUS> stat_{READY_TO_WRITE};
  std::mutex stat_mtx_;
  std::condition_variable stat_cv_;
  std::shared_ptr<HeapEx<CSRChunk<TypeKey>>> csr_heap_;

  Tensors2<float> label_tensors_;
  std::vector<TensorBag2> dense_tensors_;
  Tensors2<TypeKey> csr_buffers_;
  std::vector<std::shared_ptr<size_t>> nnz_array_;
  std::shared_ptr<ResourceManager> resource_manager_;
  int num_params_;
  size_t counter_{0};
  int pid_{0}, num_procs_{1};
  std::vector<unsigned int> pre_nnz_;
  bool use_mixed_precision_;
  const bool one_hot_;
  const size_t cache_size_;

  Tensors2<float> label_dense_buffers_internal_;
  Tensors2<TypeKey> csr_buffers_internal_;

  struct InternalBuffer_ {
    Tensors2<float> label_dense_buffers_internal;
    Tensors2<TypeKey> csr_buffers_internal;
    std::vector<std::shared_ptr<size_t>> nnz_array_internal;
    long long current_batchsize{0};
  };

  std::vector<std::shared_ptr<InternalBuffer_>> internal_buffers_;

  bool reverse_;
  std::thread data_collector_thread_; /**< A data_collector_thread. */
  int data_reader_loop_flag_ = 1;

  void collect_blank_();
  void collect_();
  bool started_ = false;

 public:
  /**
   * Ctor.
   * @param label_tensors label tensors (GPU) of data reader.
   * @param dense_tensors dense tensors (GPU) of data reader.
   * @param csr_buffers csr buffers (GPU) of data reader.
   * @param device_resources gpu resources.
   * @param csr_heap heap of data reader.
   */

  DataCollector(const Tensors2<float>& label_tensors, const std::vector<TensorBag2>& dense_tensors,
                const Tensors2<TypeKey>& csr_buffers,
                const std::vector<std::shared_ptr<size_t>>& nnz_array,
                const std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>& buffs,
                const std::shared_ptr<ResourceManager>& resource_manager,
                const std::shared_ptr<HeapEx<CSRChunk<TypeKey>>>& csr_heap = nullptr,
                const bool use_mixed_precision = false, const bool one_hot = false,
                const size_t cache_size = 0);

  void set_ready_to_write();

  void set_ready_to_write_sync();
  /**
   * Collect data from heap to each GPU (node).
   */
  void collect();

  /**
   * Read a batch to device.
   */
  long long read_a_batch_to_device();

  /**
   * Break the collecting and stop. Only used in destruction.
   */
  void stop() {
    data_reader_loop_flag_ = 0;
#ifdef ENABLE_MPI
    CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
#endif
    stat_ = STOP;
    // stat_cv_.notify_all();
  }

  void start();

  /**
   * Dtor.
   */
  ~DataCollector() {
    if (stat_ != STOP) stop();
    data_collector_thread_.join();
  }
};

/**
 * A helper function to for reading data from
 * CSRChunk to data_reader (GPU) local buffer in a new thread.
 * @param data_reader a pointer of data_collector.
 * @param p_loop_flag a flag to control the loop and
                      break loop when DataReader is destroyed.
 */
template <typename TypeKey>
static void data_collector_thread_func_(DataCollector<TypeKey>* data_collector, int* p_loop_flag) {
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

template <typename TypeKey>
int DataCollector<TypeKey>::id_ = 0;

template <typename TypeKey>
DataCollector<TypeKey>::DataCollector(
    const Tensors2<float>& label_tensors, const std::vector<TensorBag2>& dense_tensors,
    const Tensors2<TypeKey>& csr_buffers, const std::vector<std::shared_ptr<size_t>>& nnz_array,
    const std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>& buffs,
    const std::shared_ptr<ResourceManager>& resource_manager,
    const std::shared_ptr<HeapEx<CSRChunk<TypeKey>>>& csr_heap, const bool use_mixed_precision,
    const bool one_hot, const size_t cache_size)
    : csr_heap_(csr_heap),
      label_tensors_(label_tensors),
      dense_tensors_(dense_tensors),
      csr_buffers_(csr_buffers),
      nnz_array_(nnz_array),
      resource_manager_(resource_manager),
      pre_nnz_(csr_buffers.size(), 0),
      use_mixed_precision_(use_mixed_precision),
      one_hot_(one_hot),
      cache_size_(cache_size),
      reverse_(false) {
  try {
    // input check
    if (stat_ != READY_TO_WRITE) {
      CK_THROW_(Error_t::WrongInput, "stat_ != READY_TO_WRITE");
    }
    if (label_tensors.size() != dense_tensors.size()) {
      CK_THROW_(Error_t::WrongInput, "label_tensors.size() != dense_tensors.size()");
    }

    // create internal buffers
    size_t local_gpu_count = resource_manager_->get_local_gpu_count();

    size_t num_internal_buffers = cache_size_ == 0 ? 1 : cache_size_;

    MESSAGE_("num_internal_buffers " + std::to_string(num_internal_buffers));

    for (size_t j = 0; j < num_internal_buffers; j++) {
      auto internal_buffer = std::make_shared<InternalBuffer_>();
      for (size_t i = 0; i < local_gpu_count; i++) {
        int buf_size = label_tensors_[i].get_num_elements();
        if (use_mixed_precision) {
          Tensor2<__half> dense_tensor = Tensor2<__half>::stretch_from(dense_tensors_[i]);
          buf_size += dense_tensor.get_num_elements();
        } else {
          Tensor2<float> dense_tensor = Tensor2<float>::stretch_from(dense_tensors_[i]);
          buf_size += dense_tensor.get_num_elements();
        }
        Tensor2<float> tensor;
        buffs[i]->reserve({static_cast<size_t>(buf_size)}, &tensor);
        internal_buffer->label_dense_buffers_internal.push_back(tensor);
      }
      size_t k = csr_buffers_.size() / local_gpu_count;
      for (size_t i = 0; i < csr_buffers_.size(); i++) {
        Tensor2<TypeKey> tensor;
        buffs[i / k]->reserve(csr_buffers_[i].get_dimensions(), &tensor);
        internal_buffer->csr_buffers_internal.push_back(tensor);
      }
      for (size_t i = 0; i < nnz_array_.size(); i++) {
        internal_buffer->nnz_array_internal.emplace_back(new size_t);
      }
      internal_buffers_.push_back(internal_buffer);
    }

    num_params_ = csr_buffers_.size() / local_gpu_count;

#ifdef ENABLE_MPI
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &pid_));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &num_procs_));
#endif

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }

  id_++;
}

template <typename TypeKey>
void DataCollector<TypeKey>::start() {
  if (started_ == false) {
    data_collector_thread_ =
        std::thread(data_collector_thread_func_<TypeKey>, this, &data_reader_loop_flag_);
    set_affinity(data_collector_thread_, {}, true);
    started_ = true;
  } else {
    CK_THROW_(Error_t::WrongInput, "Data collector has been started");
  }
}

template <typename TypeKey>
void DataCollector<TypeKey>::collect() {
  if (counter_ < cache_size_ || cache_size_ == 0) {
    collect_();
  } else {
    collect_blank_();
  }
}

template <typename TypeKey>
void DataCollector<TypeKey>::collect_blank_() {
  std::unique_lock<std::mutex> lock(stat_mtx_);

  while (stat_ != READY_TO_WRITE && stat_ != STOP) {
    // stat_cv_.wait(lock);
    usleep(2);
  }
  if (stat_ == STOP) {
    return;
  }

  stat_ = READY_TO_READ;
  lock.unlock();
  // stat_cv_.notify_all();
}

/**************************************
 * Each node will have one DataCollector.
 * Each iteration, one of the data collector will
 * send it's CSR buffers to remote node.
 ************************************/
template <typename TypeKey>
void DataCollector<TypeKey>::collect_() {
  std::unique_lock<std::mutex> lock(stat_mtx_);

  // my turn
  CSRChunk<TypeKey>* chunk_tmp = nullptr;

  int total_device_count = resource_manager_->get_global_gpu_count();
  csr_heap_->data_chunk_checkout(&chunk_tmp);

  while (stat_ != READY_TO_WRITE && stat_ != STOP) {
    usleep(2);
  }
  if (stat_ == STOP) {
    return;
  }

  const auto& csr_cpu_buffers = chunk_tmp->get_csr_buffers();
  const auto& label_dense_buffers = chunk_tmp->get_label_buffers();

  const int num_params =
      chunk_tmp->get_num_params();  // equal to the num of output of data reader in json
  if (num_params_ != num_params) {
    CK_THROW_(Error_t::WrongInput, "job_ is ???");
  }
  assert(static_cast<int>(label_dense_buffers.size()) == total_device_count);

  auto& internal_buffer = internal_buffers_[counter_ % internal_buffers_.size()];
  internal_buffer->current_batchsize = chunk_tmp->get_current_batchsize();

  for (int ix = 0; ix < total_device_count; ix++) {
    int i =
        ((id_ == 0 && !reverse_) || (id_ == 1 && reverse_)) ? ix : (total_device_count - 1 - ix);
    int pid = resource_manager_->get_pid_from_gpu_global_id(i);
    int label_copy_num = (label_dense_buffers[0]).get_num_elements();
    if (pid == pid_) {
      size_t local_id = resource_manager_->get_gpu_local_id_from_global_id(i);
      const auto& local_gpu = resource_manager_->get_local_gpu(local_id);

      CudaDeviceContext context(local_gpu->get_device_id());
      for (int j = 0; j < num_params; j++) {
        unsigned int nnz = csr_cpu_buffers[i * num_params + j]
                               .get_buffer()[csr_cpu_buffers[i * num_params + j].get_num_rows()];

        if (pre_nnz_[local_id * num_params + j] != nnz || cache_size_ != 0 || !one_hot_) {
          pre_nnz_[local_id * num_params + j] = nnz;
          int csr_copy_num = (csr_cpu_buffers[i * num_params + j].get_num_rows() +
                              csr_cpu_buffers[i * num_params + j].get_sizeof_value() + 1);
          CK_CUDA_THROW_(cudaMemcpyAsync(
              internal_buffer->csr_buffers_internal[local_id * num_params + j].get_ptr(),
              csr_cpu_buffers[i * num_params + j].get_buffer(), csr_copy_num * sizeof(TypeKey),
              cudaMemcpyHostToDevice, local_gpu->get_data_copy_stream()));
        } else {
          unsigned int offset = csr_cpu_buffers[i * num_params + j].get_num_rows() + 1;
          int csr_copy_num = csr_cpu_buffers[i * num_params + j].get_sizeof_value();
          CK_CUDA_THROW_(cudaMemcpyAsync(
              internal_buffer->csr_buffers_internal[local_id * num_params + j].get_ptr() + offset,
              csr_cpu_buffers[i * num_params + j].get_buffer() + offset,
              csr_copy_num * sizeof(TypeKey), cudaMemcpyHostToDevice,
              local_gpu->get_data_copy_stream()));
        }
        *(internal_buffer->nnz_array_internal[local_id * num_params + j]) = nnz;
      }
      CK_CUDA_THROW_(
          cudaMemcpyAsync(internal_buffer->label_dense_buffers_internal[local_id].get_ptr(),
                          label_dense_buffers[i].get_ptr(), label_copy_num * sizeof(float),
                          cudaMemcpyHostToDevice, local_gpu->get_data_copy_stream()));
    }
  }
  // sync
  for (int ix = 0; ix < total_device_count; ix++) {
    int i =
        ((id_ == 0 && !reverse_) || (id_ == 1 && reverse_)) ? ix : (total_device_count - 1 - ix);
    int pid = resource_manager_->get_pid_from_gpu_global_id(i);
    if (pid_ == pid) {
      size_t local_id = resource_manager_->get_gpu_local_id_from_global_id(i);
      const auto& local_gpu = resource_manager_->get_local_gpu(local_id);
      CudaDeviceContext context(local_gpu->get_device_id());
      CK_CUDA_THROW_(cudaStreamSynchronize(local_gpu->get_data_copy_stream()));
    }
  }

  reverse_ = !reverse_;

  csr_heap_->chunk_free_and_checkin();

  stat_ = READY_TO_READ;
}

template <typename TypeKey>
long long DataCollector<TypeKey>::read_a_batch_to_device() {
  auto& internal_buffer = internal_buffers_[counter_ % internal_buffers_.size()];
  while (stat_ != READY_TO_READ && stat_ != STOP) {
    usleep(2);
  }
  if (stat_ == STOP) {
    return internal_buffer->current_batchsize;
  }

  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    const auto& local_gpu = resource_manager_->get_local_gpu(i);
    CudaDeviceContext context(local_gpu->get_device_id());
    for (int j = 0; j < num_params_; j++) {
      int csr_id = i * num_params_ + j;
      CK_CUDA_THROW_(cudaMemcpyAsync(csr_buffers_[csr_id].get_ptr(),
                                     internal_buffer->csr_buffers_internal[csr_id].get_ptr(),
                                     csr_buffers_[csr_id].get_size_in_bytes(),
                                     cudaMemcpyDeviceToDevice, local_gpu->get_stream()));
      *(nnz_array_[csr_id]) = *(internal_buffer->nnz_array_internal[csr_id]);
    }
    if (use_mixed_precision_) {
      Tensor2<__half> tensor = Tensor2<__half>::stretch_from(dense_tensors_[i]);
      split(label_tensors_[i], tensor, internal_buffer->label_dense_buffers_internal[i],
            local_gpu->get_stream());
    } else {
      Tensor2<float> tensor = Tensor2<float>::stretch_from(dense_tensors_[i]);
      split(label_tensors_[i], tensor, internal_buffer->label_dense_buffers_internal[i],
            local_gpu->get_stream());
    }
  }
  counter_++;
  return internal_buffer->current_batchsize;
}

template <typename TypeKey>
void DataCollector<TypeKey>::set_ready_to_write_sync() {
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    const auto& local_gpu = resource_manager_->get_local_gpu(i);
    CudaDeviceContext context(local_gpu->get_device_id());
    cudaStreamSynchronize(local_gpu->get_stream());
  }
  set_ready_to_write();
}

template <typename TypeKey>
void DataCollector<TypeKey>::set_ready_to_write() {
  stat_ = READY_TO_WRITE;
}

}  // namespace HugeCTR
