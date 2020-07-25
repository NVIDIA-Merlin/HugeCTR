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
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/csr.hpp"
#include "HugeCTR/include/csr_chunk.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "HugeCTR/include/gpu_resource.hpp"
#include "HugeCTR/include/heapex.hpp"
#include "HugeCTR/include/tensor.hpp"
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
void split(std::shared_ptr<Tensor<float>> label_tensor, std::shared_ptr<Tensor<TypeComp>> dense_tensor, const std::shared_ptr<GeneralBuffer<float>> label_dense_buffer, cudaStream_t stream);

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

  Tensors<float> label_tensors_;
  ITensors dense_tensors_;
  GeneralBuffers<TypeKey> csr_buffers_;
  // GeneralBuffers<float> label_dense_buffers_internal_;
  // GeneralBuffers<TypeKey> csr_buffers_internal_;
  std::shared_ptr<GPUResourceGroup> device_resources_;
  int num_params_;
  size_t counter_{0};
  int pid_{0}, num_procs_{1};
  // long long current_batchsize_{0};
  // todo: risk:
  std::vector<unsigned int> pre_nnz_;
  bool use_mixed_precision_;
  const bool one_hot_;
  const size_t cache_size_;

  GeneralBuffers<float> label_dense_buffers_internal_;
  GeneralBuffers<TypeKey> csr_buffers_internal_;

  struct InternalBuffer_ {
    // Tensors<float> label_dense_buffers_internal;
    // Tensors<TypeKey> csr_buffers_internal;
    GeneralBuffers<float> label_dense_buffers_internal;
    GeneralBuffers<TypeKey> csr_buffers_internal;
    long long current_batchsize{0};
  };

  std::vector<std::shared_ptr<InternalBuffer_>> internal_buffers_;

  bool reverse_;

  void collect_blank_();
  void collect_();


 public:
  /**
   * Ctor.
   * @param label_tensors label tensors (GPU) of data reader.
   * @param dense_tensors dense tensors (GPU) of data reader.
   * @param csr_buffers csr buffers (GPU) of data reader.
   * @param device_resources gpu resources.
   * @param csr_heap heap of data reader.
   */

  DataCollector(const Tensors<float>& label_tensors, const ITensors& dense_tensors,
                const GeneralBuffers<TypeKey>& csr_buffers,
                const std::shared_ptr<GPUResourceGroup>& device_resources,
                const std::shared_ptr<HeapEx<CSRChunk<TypeKey>>>& csr_heap = nullptr,
		const bool use_mixed_precision = false,
                const bool one_hot = false, const size_t cache_size = 0);

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
#ifdef ENABLE_MPI
    CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
#endif
    stat_ = STOP;
    // stat_cv_.notify_all();
  }

  /**
   * Dtor.
   */
  ~DataCollector() {
    if (stat_ != STOP) stop();
  }
};

template <typename TypeKey>
int DataCollector<TypeKey>::id_ = 0;


template <typename TypeKey>
DataCollector<TypeKey>::DataCollector(const Tensors<float>& label_tensors,
                                      const ITensors& dense_tensors,
                                      const GeneralBuffers<TypeKey>& csr_buffers,
                                      const std::shared_ptr<GPUResourceGroup>& device_resources,
                                      const std::shared_ptr<HeapEx<CSRChunk<TypeKey>>>& csr_heap,
				      const bool use_mixed_precision,
                                      const bool one_hot, const size_t cache_size)
    : csr_heap_(csr_heap),
      label_tensors_(label_tensors),
      dense_tensors_(dense_tensors),
      csr_buffers_(csr_buffers),
      device_resources_(device_resources),
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
    auto& local_device_list = device_resources_->get_device_list();

    if (cache_size == 0) {
    }
    size_t num_internal_buffers = cache_size_ == 0 ? 1 : cache_size_;

    MESSAGE_("num_internal_buffers " + std::to_string(num_internal_buffers));

    for (size_t j = 0; j < num_internal_buffers; j++) {
      auto internal_buffer = std::make_shared<InternalBuffer_>();
      for (unsigned int i = 0; i < local_device_list.size(); i++) {
        assert(local_device_list[i] == label_tensors_[i]->get_device_id());
        assert(local_device_list[i] == dense_tensors_[i]->get_device_id());
        int buf_size =
            label_tensors_[i]->get_num_elements() + dense_tensors_[i]->get_num_elements();
        internal_buffer->label_dense_buffers_internal.emplace_back(
            std::make_shared<GeneralBuffer<float>>(buf_size, local_device_list[i]));
        // if(j == 0){
        //   MESSAGE_("buf_size " + std::to_string(buf_size) + " device " + std::to_string(i));
        // }
      }
      for (auto& cb : csr_buffers_) {
        internal_buffer->csr_buffers_internal.emplace_back(
            new GeneralBuffer<TypeKey>(cb->get_num_elements(), cb->get_device_id()));
        // if(j == 0){
        //   MESSAGE_("buf_size " + std::to_string(cb->get_num_elements()) + " device " +
        //   std::to_string(cb->get_device_id()));
        // }
      }
      internal_buffers_.push_back(internal_buffer);
    }

    num_params_ = csr_buffers_.size() / local_device_list.size();

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

  int total_device_count = device_resources_->get_total_gpu_count();
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
    int pid = device_resources_->get_pid(i);
    int label_copy_num = (label_dense_buffers[0]).get_num_elements();
    if (pid == pid_) {
      int local_id = device_resources_->get_local_id(i);
      CudaDeviceContext context(device_resources_->get_local_device_id(i));
      for (int j = 0; j < num_params; j++) {
        unsigned int nnz = csr_cpu_buffers[i * num_params + j]
                               .get_buffer()[csr_cpu_buffers[i * num_params + j].get_num_rows()];

        if (pre_nnz_[local_id * num_params + j] != nnz || cache_size_ != 0 || !one_hot_) {
          pre_nnz_[local_id * num_params + j] = nnz;
          int csr_copy_num = (csr_cpu_buffers[i * num_params + j].get_num_rows() +
                              csr_cpu_buffers[i * num_params + j].get_sizeof_value() + 1);
          CK_CUDA_THROW_(cudaMemcpyAsync(
              internal_buffer->csr_buffers_internal[local_id * num_params + j]->get_ptr_with_offset(
                  0),
              csr_cpu_buffers[i * num_params + j].get_buffer(), csr_copy_num * sizeof(TypeKey),
              cudaMemcpyHostToDevice, (*device_resources_)[local_id]->get_data_copy_stream(id_)));
        } else {
          unsigned int offset = csr_cpu_buffers[i * num_params + j].get_num_rows() + 1;
          int csr_copy_num = csr_cpu_buffers[i * num_params + j].get_sizeof_value();
          CK_CUDA_THROW_(cudaMemcpyAsync(
              internal_buffer->csr_buffers_internal[local_id * num_params + j]->get_ptr_with_offset(
                  0) +
                  offset,
              csr_cpu_buffers[i * num_params + j].get_buffer() + offset,
              csr_copy_num * sizeof(TypeKey), cudaMemcpyHostToDevice,
              (*device_resources_)[local_id]->get_data_copy_stream(id_)));
        }
      }
      CK_CUDA_THROW_(cudaMemcpyAsync(
          internal_buffer->label_dense_buffers_internal[local_id]->get_ptr_with_offset(0),
          label_dense_buffers[i].get(), label_copy_num * sizeof(float), cudaMemcpyHostToDevice,
          (*device_resources_)[local_id]->get_data_copy_stream(id_)));
    }
  }
  // sync
  for (int ix = 0; ix < total_device_count; ix++) {
    int i =
        ((id_ == 0 && !reverse_) || (id_ == 1 && reverse_)) ? ix : (total_device_count - 1 - ix);
    int pid = device_resources_->get_pid(i);
    if (pid_ == pid) {
      int local_id = device_resources_->get_local_id(i);
      CudaDeviceContext context(device_resources_->get_local_device_id(i));
      CK_CUDA_THROW_(
          cudaStreamSynchronize((*device_resources_)[local_id]->get_data_copy_stream(id_)));
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

  for (unsigned int i = 0; i < device_resources_->size(); i++) {
    CudaDeviceContext context((*device_resources_)[i]->get_device_id());
    for (int j = 0; j < num_params_; j++) {
      int csr_id = i * num_params_ + j;
      CK_CUDA_THROW_(
          cudaMemcpyAsync(csr_buffers_[csr_id]->get_ptr_with_offset(0),
                          internal_buffer->csr_buffers_internal[csr_id]->get_ptr_with_offset(0),
                          csr_buffers_[csr_id]->get_size(), cudaMemcpyDeviceToDevice,
                          (*device_resources_)[i]->get_stream()));
    }
    if(use_mixed_precision_){
      split(label_tensors_[i], dynamic_tensor_cast<__half>(dense_tensors_[i]), internal_buffer->label_dense_buffers_internal[i], (*device_resources_)[i]->get_stream());
    }
    else{
      split(label_tensors_[i], dynamic_tensor_cast<float>(dense_tensors_[i]), internal_buffer->label_dense_buffers_internal[i], (*device_resources_)[i]->get_stream());
    }
  }
  counter_++;
  return internal_buffer->current_batchsize;
}


template <typename TypeKey>
void DataCollector<TypeKey>::set_ready_to_write_sync() {
  for (unsigned int i = 0; i < device_resources_->size(); i++) {
    CudaDeviceContext context((*device_resources_)[i]->get_device_id());
    cudaStreamSynchronize((*device_resources_)[i]->get_stream());
  }
  set_ready_to_write();
}

template <typename TypeKey>
void DataCollector<TypeKey>::set_ready_to_write() {
  stat_ = READY_TO_WRITE;
}

}  // namespace HugeCTR
