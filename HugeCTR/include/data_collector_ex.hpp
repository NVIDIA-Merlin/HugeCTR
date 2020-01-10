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

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/csr.hpp"
#include "HugeCTR/include/csr_chunk_ex.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "HugeCTR/include/gpu_resource.hpp"
#include "HugeCTR/include/heap.hpp"
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

void split(std::shared_ptr<Tensor<float>> label_tensor, std::shared_ptr<Tensor<float>> dense_tensor, const std::shared_ptr<GeneralBuffer<float>> label_dense_buffer, cudaStream_t stream );

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
  enum STATUS { READY_TO_WRITE, READY_TO_READ, STOP };
  enum JOB { TRAIN, EVAL_MASTER, EVAL_SLAVE };
  STATUS stat_{READY_TO_WRITE};
  std::mutex stat_mtx_;
  std::condition_variable stat_cv_;
  JOB job_{TRAIN};
  std::shared_ptr<Heap<CSRChunkEx<TypeKey>>> csr_heap_;

  Tensors<float> label_tensors_;
  Tensors<float> dense_tensors_;
  GeneralBuffers<TypeKey> csr_buffers_;
  GeneralBuffers<float> label_dense_buffers_internal_;
  GeneralBuffers<TypeKey> csr_buffers_internal_;
  std::shared_ptr<GPUResourceGroup> device_resources_;
  int num_params_;
  long long counter_{0};
  int pid_{0}, num_procs_{1};

 public:
  /**
   * Ctor.
   * @param label_tensors label tensors (GPU) of data reader.
   * @param dense_tensors dense tensors (GPU) of data reader.
   * @param csr_buffers csr buffers (GPU) of data reader.
   * @param device_resources gpu resources.
   * @param csr_heap heap of data reader.
   * @param is_eval whether it's evaluation.
   */

  DataCollector(const Tensors<float>& label_tensors,
		const Tensors<float>& dense_tensors,
		const GeneralBuffers<TypeKey>& csr_buffers,
		const std::shared_ptr<GPUResourceGroup>& device_resources,
		const std::shared_ptr<Heap<CSRChunkEx<TypeKey>>>& csr_heap = nullptr,
		bool is_eval = true);



  /**
   * Collect data from heap to each GPU (node).
   */
  void collect();

  /**
   * Read a batch to device.
   */
  void read_a_batch_to_device();

  /**
   * Break the collecting and stop. Only used in destruction.
   */
  void stop() {
#ifdef ENABLE_MPI
    CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
#endif
    stat_ = STOP;
    stat_cv_.notify_all();
  }

  /**
   * Dtor.
   */
  ~DataCollector() {
    if (stat_ != STOP) stop();
  }
};

template <typename TypeKey>
DataCollector<TypeKey>::DataCollector(const Tensors<float>& label_tensors,
				      const Tensors<float>& dense_tensors,
                                      const GeneralBuffers<TypeKey>& csr_buffers,
                                      const std::shared_ptr<GPUResourceGroup>& device_resources,
                                      const std::shared_ptr<Heap<CSRChunkEx<TypeKey>>>& csr_heap,
                                      bool is_eval)
    : csr_heap_(csr_heap),
      label_tensors_(label_tensors),
      dense_tensors_(dense_tensors),
      csr_buffers_(csr_buffers),
      device_resources_(device_resources) {
  try {
    if (is_eval && csr_heap_ != nullptr) {
      job_ = EVAL_MASTER;
    } else if (is_eval && csr_heap_ == nullptr) {
      job_ = EVAL_SLAVE;
    } else if (is_eval == false && csr_heap_ != nullptr) {
      job_ = TRAIN;
    } else {
      CK_THROW_(Error_t::WrongInput, "job_ is ???");
    }
    // input check
    if (stat_ != READY_TO_WRITE) {
      CK_THROW_(Error_t::WrongInput, "stat_ != READY_TO_WRITE");
    }
    if (label_tensors.size() != dense_tensors.size()){
      CK_THROW_(Error_t::WrongInput, "label_tensors.size() != dense_tensors.size()");
    }
    
    // create internal buffers
    auto& local_device_list = device_resources_->get_device_list();
    for (unsigned int i=0; i < local_device_list.size(); i++){
      assert(local_device_list[i] == label_tensors_[i]->get_device_id());
      assert(local_device_list[i] == dense_tensors_[i]->get_device_id());
      int buf_size = label_tensors_[i]->get_num_elements()+dense_tensors_[i]->get_num_elements();
      label_dense_buffers_internal_.emplace_back(std::make_shared<GeneralBuffer<float>>(buf_size,local_device_list[i]));
    }
    for (auto& cb : csr_buffers_) {
      csr_buffers_internal_.emplace_back(
	  new GeneralBuffer<TypeKey>(cb->get_num_elements(), cb->get_device_id()));
    }
    num_params_ = csr_buffers_.size()/local_device_list.size();

#ifdef ENABLE_MPI
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &pid_));
    // for EVAL_*, num_procs_ must be 1, so that they have the unique data source (rank 0)
    if (job_ == TRAIN) CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &num_procs_));
#endif
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

/**************************************
 * Each node will have one DataCollector.
 * Each iteration, one of the data collector will
 * send it's CSR buffers to remote node.
 ************************************/
template <typename TypeKey>
void DataCollector<TypeKey>::collect() {
  const int LABEL_TAG_OFFSET = 100;

  {
    std::unique_lock<std::mutex> lock(stat_mtx_);
    while (stat_ != READY_TO_WRITE && stat_ != STOP) {
      stat_cv_.wait(lock);
    }
    if (stat_ == STOP) {
      return;
    }

    if ((job_ == TRAIN && pid_ == counter_ % num_procs_) || job_ == EVAL_MASTER) {
      // my turn
      CSRChunkEx<TypeKey>* chunk_tmp = nullptr;
      unsigned int key = 0;

      int total_device_count = device_resources_->get_total_gpu_count();

#ifdef ENABLE_MPI
      std::vector<MPI_Request> req;
      req.reserve((1+num_params_) * total_device_count);  // to prevent the reallocation
#endif
      csr_heap_->data_chunk_checkout(&chunk_tmp, &key);
      const auto& csr_cpu_buffers = chunk_tmp->get_csr_buffers();
      const auto& label_dense_buffers = chunk_tmp->get_label_buffers();
      const int num_params = chunk_tmp->get_num_params(); //equal to the num of output of data reader in json
      if(num_params_!= num_params){
	CK_THROW_(Error_t::WrongInput, "job_ is ???");
      }
      assert(static_cast<int>(label_dense_buffers.size()) == total_device_count);

      for (int i = 0; i < total_device_count; i++) {
        int pid = device_resources_->get_pid(i);
        int label_copy_num = (label_dense_buffers[0]).get_num_elements();
        if (pid_ == pid) {
          int local_id = device_resources_->get_local_id(i);
          CudaDeviceContext context(device_resources_->get_local_device_id(i));
	  for(int j = 0; j < num_params; j++){
	    int csr_copy_num = (csr_cpu_buffers[i*num_params + j].get_num_rows() +
				csr_cpu_buffers[i*num_params + j].get_sizeof_value() + 1);
	    CK_CUDA_THROW_(cudaMemcpyAsync(csr_buffers_internal_[local_id*num_params + j]->get_ptr_with_offset(0),
					   csr_cpu_buffers[i*num_params + j].get_buffer(),
					   csr_copy_num * sizeof(TypeKey), cudaMemcpyHostToDevice,
					   (*device_resources_)[local_id]->get_data_copy_stream()));
	  }
          CK_CUDA_THROW_(cudaMemcpyAsync(label_dense_buffers_internal_[local_id]->get_ptr_with_offset(0),
                                         label_dense_buffers[i].get(), label_copy_num * sizeof(float),
                                         cudaMemcpyHostToDevice,
                                         (*device_resources_)[local_id]->get_data_copy_stream()));
        } else {
#ifdef ENABLE_MPI
          int base_tag = (job_ == TRAIN) ? 1 : 3;
          int l_tag = (i + LABEL_TAG_OFFSET) << 2 | base_tag;
          req.resize(req.size() + 1 + num_params);
	  
	  for(int j = 0; j < num_params; j++){
	    int csr_tag = (i*num_params+j) << 2 | base_tag;
	    int csr_copy_num = (csr_cpu_buffers[i*num_params + j].get_num_rows() +
				csr_cpu_buffers[i*num_params + j].get_sizeof_value() + 1);
	    CK_MPI_THROW_(MPI_Isend(csr_cpu_buffers[i*num_params + j].get_buffer(), csr_copy_num,
				    ToMpiType<TypeKey>::T(), pid, csr_tag, MPI_COMM_WORLD,
				    (&req.back()) - 1 - j));
	  }
          CK_MPI_THROW_(MPI_Isend(label_dense_buffers[i].get(), label_copy_num, ToMpiType<float>::T(),
                                  pid, l_tag, MPI_COMM_WORLD, &req.back()));

#else
          assert(!"No MPI support");
#endif
        }
      }

      // sync
      for (int i = 0; i < total_device_count; i++) {
        int pid = device_resources_->get_pid(i);
        if (pid_ == pid) {
          int local_id = device_resources_->get_local_id(i);
          CudaDeviceContext context(device_resources_->get_local_device_id(i));
          CK_CUDA_THROW_(
              cudaStreamSynchronize((*device_resources_)[local_id]->get_data_copy_stream()));
        }
      }
#ifdef ENABLE_MPI
      CK_MPI_THROW_(MPI_Waitall(req.size(), &req.front(), MPI_STATUSES_IGNORE));
#endif
      csr_heap_->chunk_free_and_checkin(key);
    } else {
#ifdef ENABLE_MPI
      const auto& device_list = device_resources_->get_device_list();
      std::vector<MPI_Request> req;
      req.reserve((1+num_params_) * device_list.size());                     // to prevent the reallocation
      for (unsigned int i = 0; i < device_list.size(); i++) {  // local_id
        CudaDeviceContext context(device_list[i]);
        int base_tag = (job_ == TRAIN) ? 1 : 3;

        req.resize(req.size() + 1 + num_params_);
	for (int j = 0; j < num_params_; j++){
	  int csr_tag = ((device_resources_->get_global_id(device_list[i])*num_params_+j) << 2) | base_tag;
	  CK_MPI_THROW_(MPI_Irecv(csr_buffers_internal_[i*num_params_ + j]->get_ptr_with_offset(0),
				  csr_buffers_internal_[i*num_params_ + j]->get_num_elements(),
				  ToMpiType<TypeKey>::T(), counter_ % num_procs_, csr_tag,
				  MPI_COMM_WORLD, &(req.back()) - 1 - j));
	}
        int l_tag =
            (device_resources_->get_global_id(device_list[i]) + LABEL_TAG_OFFSET) << 2 | base_tag;
        CK_MPI_THROW_(MPI_Irecv(label_dense_buffers_internal_[i]->get_ptr_with_offset(0),
                                label_dense_buffers_internal_[i]->get_num_elements(),
                                ToMpiType<float>::T(), counter_ % num_procs_, l_tag, MPI_COMM_WORLD,
                                &req.back()));
      }

      CK_MPI_THROW_(MPI_Waitall(req.size(), &req.front(), MPI_STATUSES_IGNORE));

#else
      assert(!"No MPI support");
#endif
    }
    counter_++;
    stat_ = READY_TO_READ;
  }
  stat_cv_.notify_all();
}

template <typename TypeKey>
void DataCollector<TypeKey>::read_a_batch_to_device() {
  {
    std::unique_lock<std::mutex> lock(stat_mtx_);
    while (stat_ != READY_TO_READ && stat_ != STOP) {
      stat_cv_.wait(lock);
    }
    if (stat_ == STOP) {
      return;
    }

    for (unsigned int i = 0; i < device_resources_->size(); i++) {
      CudaDeviceContext context((*device_resources_)[i]->get_device_id());
      for(int j = 0; j < num_params_; j++){
	int csr_id = i*num_params_ + j;
	CK_CUDA_THROW_(cudaMemcpyAsync(csr_buffers_[csr_id]->get_ptr_with_offset(0),
				       csr_buffers_internal_[csr_id]->get_ptr_with_offset(0),
				       csr_buffers_[csr_id]->get_size(), cudaMemcpyDeviceToDevice,
				       (*device_resources_)[i]->get_stream()));
      }

      split(label_tensors_[i], dense_tensors_[i],
	    label_dense_buffers_internal_[i],
	    (*device_resources_)[i]->get_stream());

    }
    for (unsigned int i = 0; i < device_resources_->size(); i++) {
      CudaDeviceContext context((*device_resources_)[i]->get_device_id());
      CK_CUDA_THROW_(cudaStreamSynchronize((*device_resources_)[i]->get_stream()));
    }
    stat_ = READY_TO_WRITE;
  }
  stat_cv_.notify_all();
}

}  // namespace HugeCTR
