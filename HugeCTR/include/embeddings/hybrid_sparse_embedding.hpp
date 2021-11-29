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

#include <collectives/all_reduce_comm.hpp>
#include <collectives/ib_comm.hpp>
#include <gpu_barrier.hpp>
#include <resource_manager.hpp>
#include <utils.hpp>
#include <queue>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>

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

#include "HugeCTR/include/data_readers/async_reader/async_reader_adapter.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/indices_container.hpp"

using namespace HugeCTR::hybrid_embedding;

namespace HugeCTR {

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
  bool use_train_precompute_indices, use_eval_precompute_indices;
  hybrid_embedding::HybridEmbeddingType hybrid_embedding_type;
  OptParams opt_params;  // optimizer params
};

///
/// Interface class for the hybrid embedding to HugeCTR. It is responsible for
/// persistent gpu memory allocation.
///
template <typename dtype, typename emtype>
class HybridSparseEmbedding : public IEmbedding {
 public:
  class StreamManager {
    std::vector<std::unordered_map<std::string, cudaStream_t>> stream_map;
    std::vector<std::unordered_map<std::string, cudaEvent_t>> event_map;

  public:
    StreamManager(int num_devices):
      stream_map(num_devices),
      event_map (num_devices) {
    }

    cudaStream_t& get_stream(uint32_t device_id, const std::string& key) {
      if (stream_map[device_id].find(key) == stream_map[device_id].end()) {
        cudaStream_t stream;
        CK_CUDA_THROW_(cudaStreamCreate(&stream));
        stream_map[device_id][key] = stream;
      }
      return stream_map[device_id][key];
    }

    cudaEvent_t& get_event(uint32_t device_id, const std::string& key) {
      if (event_map[device_id].find(key) == event_map[device_id].end()) {
        cudaEvent_t event;
        CK_CUDA_THROW_(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        event_map[device_id][key] = event;
      }
      return event_map[device_id][key];
    }

    ~StreamManager() {
      for (auto &sm : stream_map) {
        for (auto &s : sm) {
          cudaStreamDestroy(s.second);
        }
      }
      for (auto &em : event_map) {
        for (auto &e : em) {
          cudaEventDestroy(e.second);
        }
      }
    }
  };

 private:
  // Embedding models, one instance per frequent and the infrequent embedding
  // for each mlp-network in the train session.
  //

  // data-parallel embedding model
  std::vector<FrequentEmbedding<dtype, emtype>> frequent_embeddings_;
  std::vector<FrequentEmbeddingCompression<dtype>> frequent_embedding_train_indices_,
      frequent_embedding_evaluate_indices_;
  // model-parallel embedding model
  std::vector<InfrequentEmbedding<dtype, emtype>> infrequent_embeddings_;
  std::vector<InfrequentEmbeddingSelection<dtype>> infrequent_embedding_train_indices_,
      infrequent_embedding_evaluate_indices_;
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

  StreamManager stream_manager_;

  SparseTensors<dtype> train_input_tensors_;
  SparseTensors<dtype> evaluate_input_tensors_;
  HybridSparseEmbeddingParams embedding_params_;
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
  bool overlap_ar_a2a_;

  std::shared_ptr<IndexProcessor<dtype>> train_async_indices_, eval_async_indices_;

  // TODO: this parameter is not used by HE at all.
  // We should be in pursuit of merging SparseEmbeddingHashParams and HybridSparseEmbeddingParams
  SparseEmbeddingHashParams dummy_params_;

  void index_calculation(bool is_train, bool is_first_batch, int i, cudaStream_t stream);
  void forward(bool is_train, bool is_first_batch, int i, cudaStream_t stream, cudaEvent_t* evt_ptr);
  void backward_pre_communication(int i, cudaStream_t stream);
  void frequent_local_reduce(int i, cudaStream_t stream);
  void backward_communications(int i, cudaStream_t stream);
  void frequent_update(int i, cudaStream_t stream);
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
  void get_num_instances_per_node(std::vector<uint32_t>& num_instances_per_node) {
    uint32_t total_gpu_count = resource_manager_->get_global_gpu_count();
    for (uint32_t gid = 0; gid < total_gpu_count; ++gid) {
      uint32_t nodeid = resource_manager_->get_process_id_from_gpu_global_id(gid);
      num_instances_per_node[nodeid] = num_instances_per_node[nodeid] + 1;
    }
    return;
  }

  const GPUResource& get_local_gpu(int i) const { return *resource_manager_->get_local_gpu(i); }

  size_t get_categories_num() {
    size_t num_categories = 0;
    for (size_t i = 0; i < embedding_params_.slot_size_array.size(); ++i) {
      num_categories += embedding_params_.slot_size_array[i];
    }
    return num_categories;
  }

 public:
  HybridSparseEmbedding(const SparseTensors<dtype>& train_input_tensors,
                        const SparseTensors<dtype>& evaluate_input_tensors,
                        const HybridSparseEmbeddingParams& embedding_params,
                        const std::vector<BuffPtr<emtype>>& grouped_wgrad_buff,
                        const GpuLearningRateSchedulers lr_scheds, bool graph_mode,
                        const std::shared_ptr<ResourceManager>& resource_manager,
                        bool overlap_ar_a2a);
  ~HybridSparseEmbedding() = default;

  // TODO: consider to merge it with init_params
  void init_model(const SparseTensors<dtype>& data, size_t& wgrad_offset);
  void setup_async_mode(AsyncReader<dtype>* train_data_reader, AsyncReader<dtype>* eval_data_reader);

  TrainState train(bool is_train, int i, TrainState state) override;
  void forward(bool is_train, bool is_first_batch = true) override;
  void backward() override;
  void update_params() override;
  void init_params() override;
  void load_parameters(std::string sparse_model) override;
  void dump_parameters(std::string sparse_model) const override;
  void set_learning_rate(float lr) override;
  // TODO: a workaround to enable GPU LR for HE only; need a better way
  GpuLearningRateSchedulers get_learning_rate_schedulers() const override;

  size_t get_params_num() const override;
  size_t get_vocabulary_size() const override;
  size_t get_max_vocabulary_size() const override;

  Embedding_t get_embedding_type() const override { return Embedding_t::HybridSparseEmbedding; }
  // TODO: implemented the empty virtual functions below and in the corresponding CU file.
  void load_parameters(BufferBag& keys, size_t num) override {}
  void dump_parameters(BufferBag& keys, size_t* num) const override {}

  void dump_opt_states(std::ofstream& stream) override {}
  void load_opt_states(std::ifstream& stream) override {}
  void reset_optimizer() override {}
  void reset() override {}

  const SparseEmbeddingHashParams& get_embedding_params() const override { return dummy_params_; }
  void check_overflow() const override {}
  void get_forward_results_tf(const bool is_train, const bool on_gpu,
                              void* const forward_result) override {}

  std::vector<TensorBag2> get_train_output_tensors() const override;
  std::vector<TensorBag2> get_evaluate_output_tensors() const override;

  cudaError_t update_top_gradients(const bool on_gpu, const void* const top_gradients) override {
    throw;
  }

  void compute_indices(
      FrequentEmbeddingCompression<dtype>& compression,
      InfrequentEmbeddingSelection<dtype>& selection,
      CommunicationType communication_type,
      bool compute_network_cache_indices,
      cudaStream_t main_stream,
      StreamManager& manager,
      int raw_device_id,
      int sm_count);

};

}  // namespace HugeCTR
