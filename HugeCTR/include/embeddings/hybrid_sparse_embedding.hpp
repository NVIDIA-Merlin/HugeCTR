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

#include <gpu_barrier.hpp>
#include <collectives/ib_comm.hpp>
#include <queue>
#include <random>
#include <resource_manager.hpp>
#include <utils.hpp>
#include <vector>

#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/calibration_data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/communication.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/frequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/infrequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/statistics.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/tensor2.hpp"
#include <collectives/ib_comm.hpp>
#include <collectives/all_reduce_comm.hpp>
#include <gpu_barrier.hpp>

using namespace HugeCTR::hybrid_embedding;

namespace HugeCTR {

template <typename TypeEmbedding>
struct HybridSparseEmbeddingParams {
  size_t train_batch_size;
  size_t evaluate_batch_size;
  size_t num_iterations_statistics;
  size_t max_num_frequent_categories;  // max(train_batch_size, eval_batch_size) * # of batches for
                                       // frequent categories
  int64_t max_num_infrequent_samples;
  double p_dup_max;
  size_t embedding_vec_size;
  size_t slot_num;  // slot number
  std::vector<size_t> slot_size_array;
  hybrid_embedding::CommunicationType communication_type;
  double max_all_reduce_bandwidth;
  double max_all_to_all_bandwidth;
  double efficiency_bandwidth_ratio;
  hybrid_embedding::HybridEmbeddingType hybrid_embedding_type;
  OptParams opt_params;  // optimizer params
};

///
/// Interface class for the hybrid embedding to HugeCTR. It is responsible for
/// persistent gpu memory allocation.
///
template <typename dtype, typename emtype>
class HybridSparseEmbedding : public IEmbedding {
 private:
  // Embedding models, one instance per frequent and the infrequent embedding
  // for each mlp-network in the train session.
  //

  // data-parallel embedding model
  std::vector<FrequentEmbedding<dtype, emtype>> frequent_embeddings_;
  // model-parallel embedding model
  std::vector<InfrequentEmbedding<dtype, emtype>> infrequent_embeddings_;
  // performs the communication scheme
  std::vector<std::unique_ptr<Communication>> frequent_comms_, infrequent_forward_comms_,
      infrequent_backward_comms_;
  std::vector<AllToAllStorage<emtype>> infrequent_forward_comm_buffers_,
      infrequent_backward_comm_buffers_;

  // Hier A2Av / custom AR impl
#ifdef ENABLE_MPI
  std::vector<cudaStream_t> comm_stream_;
  IbComm* ib_comm_;
  AllReduceInPlaceComm::Handle barrier_handle_;
#endif
  std::unique_ptr<GPUBarrier> gpu_barrier_;

  AllReduceInPlaceComm::Handle frequent_embedding_handle_;
  Tensors2<uint32_t> d_barrier_store_;

  // model_, data_, calibration_ and statistics_ are replications of the model
  // and input data on each gpu. The HybridSparseEmbedding class manages
  // it's scope / frees the memory.
  std::vector<hybrid_embedding::Model<dtype>> model_;
  std::vector<Data<dtype>> data_train_;
  std::vector<Data<dtype>> data_evaluate_;
  std::vector<Data<dtype>> data_statistics_;
  std::vector<CalibrationData> calibration_;
  std::vector<Statistics<dtype>> statistics_;

  // added by kefeng
  // std::vector<CudaPreAllocator> pre_alloc_bufs_;
  std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> bufs_;

  std::vector<std::unordered_map<std::string, cudaStream_t>> stream_map;
  std::vector<std::unordered_map<std::string, cudaEvent_t>> event_map;

  Tensors2<dtype> train_input_tensors_;
  Tensors2<dtype> evaluate_input_tensors_;
  HybridSparseEmbeddingParams<emtype> embedding_params_;
  std::shared_ptr<ResourceManager> resource_manager_;

  Tensors2<emtype> train_output_tensors_;    /**< The output tensors. */
  Tensors2<emtype> evaluate_output_tensors_; /**< The output tensors. */
  template <typename T> 
  using BuffPtr = std::shared_ptr<BufferBlock2<T>>;
  std::vector<BuffPtr<emtype>> grouped_wgrad_buff_;
  bool grouped_all_reduce_ = false;

  std::vector<OptParams> opt_params_; /**< Optimizer params. */

  GpuLearningRateSchedulers lr_scheds_;
  bool graph_mode_;
  
  void index_calculation(bool is_train, int eval_batch, int i, cudaStream_t stream);
  void forward          (bool is_train, int eval_batch, int i, cudaStream_t stream);
  void backward_pre_communication (int i, cudaStream_t stream);
  void backward_communications    (int i, cudaStream_t stream);
  void backward_post_communication(int i, cudaStream_t stream);

 protected:
  size_t get_batch_size(bool is_train) const {
    if (is_train) {
      return embedding_params_.train_batch_size;
    } else {
      return embedding_params_.evaluate_batch_size;
    }
  }
  size_t get_universal_batch_size() const {
    return std::max(embedding_params_.train_batch_size, embedding_params_.evaluate_batch_size);
  }
  size_t get_batch_size_per_gpu(bool is_train) const {
    return get_batch_size(is_train) / resource_manager_->get_global_gpu_count();
  }
  size_t get_embedding_vec_size() const { return embedding_params_.embedding_vec_size; }
  size_t get_slot_num() const { return embedding_params_.slot_num; }
  void get_num_instances_per_node(std::vector<uint32_t> &num_instances_per_node) {
    uint32_t total_gpu_count = resource_manager_->get_global_gpu_count();
    for (uint32_t gid = 0; gid < total_gpu_count; ++gid) {
      uint32_t nodeid = resource_manager_->get_process_id_from_gpu_global_id(gid);
      num_instances_per_node[nodeid] = num_instances_per_node[nodeid] + 1;
    }
    return;
  }

  const GPUResource &get_local_gpu(int i) const { return *resource_manager_->get_local_gpu(i); }

  size_t get_categories_num() {
    size_t num_categories = 0;
    for (size_t i = 0; i < embedding_params_.slot_size_array.size(); ++i) {
      num_categories += embedding_params_.slot_size_array[i];
    }
    return num_categories;
  }

  cudaStream_t& get_stream(uint32_t device_id, const std::string& key);
  cudaEvent_t&  get_event (uint32_t device_id, const std::string& key);
  void destroy_streams();
  void destroy_events();

 public:
  HybridSparseEmbedding(const Tensors2<dtype> &train_input_tensors,
                        const Tensors2<dtype> &evaluate_input_tensors,
                        const HybridSparseEmbeddingParams<emtype> &embedding_params,
                        const std::vector<BuffPtr<emtype>>& grouped_wgrad_buff,
                        const GpuLearningRateSchedulers lr_scheds,
                        bool graph_mode,
                        const std::shared_ptr<ResourceManager> &resource_manager);
  ~HybridSparseEmbedding();

  void init_model(const Tensors2<dtype> &data, size_t& wgrad_offset);

  TrainState train(bool is_train, int i, TrainState state) override;
  void forward(bool is_train, int eval_batch = -1) override;
  void backward() override;
  void update_params() override;
  void init_params() override;
  void load_parameters(std::string sparse_model) override;
  void dump_parameters(std::string sparse_model) const override;
  void set_learning_rate(float lr) override;
  // TODO(MLPERF): a workaround to enable GPU LR for HE only; need a better way
  GpuLearningRateSchedulers get_learning_rate_schedulers() const override;

  size_t get_params_num() const override;
  size_t get_vocabulary_size() const override;
  size_t get_max_vocabulary_size() const override;

  Embedding_t get_embedding_type() const override { return Embedding_t::HybridSparseEmbedding; }
  // TODO(MLPERF): implemented the empty virtual functions below
  void load_parameters(BufferBag &keys, size_t num) override {}
  void dump_parameters(BufferBag& keys, size_t *num) const override {}

  void reset() override {}

  void dump_opt_states(std::ofstream& stream) override {} 
  void load_opt_states(std::ifstream& stream) override {} 

  void check_overflow() const override {}
  void get_forward_results_tf(const bool is_train, const bool on_gpu,
                              void *const forward_result) override {}

  std::vector<TensorBag2> get_train_output_tensors() const override;
  std::vector<TensorBag2> get_evaluate_output_tensors() const override;

  cudaError_t update_top_gradients(const bool on_gpu, const void *const top_gradients) override { throw; }
};

}  // namespace HugeCTR
