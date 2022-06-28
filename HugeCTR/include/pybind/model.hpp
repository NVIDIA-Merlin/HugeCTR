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
#include <common.hpp>
#include <embedding.hpp>
#include <embedding_training_cache/embedding_training_cache.hpp>
#include <exchange_wgrad.hpp>
#include <graph_wrapper.hpp>
#include <hdfs_backend.hpp>
#include <hps/hier_parameter_server.hpp>
#include <hps/kafka_message.hpp>
#include <hps/message.hpp>
#include <loss.hpp>
#include <metrics.hpp>
#include <network.hpp>
#include <optimizer.hpp>
#include <parser.hpp>
#include <string>
#include <thread>
#include <utility>
#include <utils.hpp>

#include "HugeCTR/embedding/embedding.hpp"
#include "HugeCTR/embedding_storage/embedding_table.hpp"
#include "embedding_collection.hpp"

namespace HugeCTR {

namespace {

std::map<Layer_t, std::string> LAYER_TYPE_TO_STRING = {
    {Layer_t::BatchNorm, "BatchNorm"},
    {Layer_t::LayerNorm, "LayerNorm"},
    {Layer_t::BinaryCrossEntropyLoss, "BinaryCrossEntropyLoss"},
    {Layer_t::Cast, "Cast"},
    {Layer_t::Concat, "Concat"},
    {Layer_t::Dropout, "Dropout"},
    {Layer_t::ELU, "ELU"},
    {Layer_t::InnerProduct, "InnerProduct"},
    {Layer_t::Interaction, "Interaction"},
    {Layer_t::ReLU, "ReLU"},
    {Layer_t::Reshape, "Reshape"},
    {Layer_t::Sigmoid, "Sigmoid"},
    {Layer_t::Slice, "Slice"},
    {Layer_t::WeightMultiply, "WeightMultiply"},
    {Layer_t::FmOrder2, "FmOrder2"},
    {Layer_t::Add, "Add"},
    {Layer_t::ReduceSum, "ReduceSum"},
    {Layer_t::Softmax, "Softmax"},
    {Layer_t::Gather, "Gather"},
    {Layer_t::PReLU_Dice, "PReLU_Dice"},
    {Layer_t::GRU, "GRU"},
    {Layer_t::MatrixMultiply, "MatrixMultiply"},
    {Layer_t::MultiHeadAttention, "MultiHeadAttention"},
    {Layer_t::Scale, "Scale"},
    {Layer_t::FusedReshapeConcat, "FusedReshapeConcat"},
    {Layer_t::FusedReshapeConcatGeneral, "FusedReshapeConcatGeneral"},
    {Layer_t::Sub, "Sub"},
    {Layer_t::ReduceMean, "ReduceMean"},
    {Layer_t::CrossEntropyLoss, "CrossEntropyLoss"},
    {Layer_t::MultiCrossEntropyLoss, "MultiCrossEntropyLoss"},
    {Layer_t::ElementwiseMultiply, "ElementwiseMultiply"},
    {Layer_t::MultiCross, "MultiCross"}};

std::map<Layer_t, std::string> LAYER_TYPE_TO_STRING_MP = {
    {Layer_t::BatchNorm, "BatchNorm"},
    {Layer_t::LayerNorm, "LayerNorm"},
    {Layer_t::BinaryCrossEntropyLoss, "BinaryCrossEntropyLoss"},
    {Layer_t::Cast, "Cast"},
    {Layer_t::Concat, "Concat"},
    {Layer_t::Dropout, "Dropout"},
    {Layer_t::ELU, "ELU"},
    {Layer_t::InnerProduct, "InnerProduct"},
    {Layer_t::Interaction, "Interaction"},
    {Layer_t::ReLU, "ReLU"},
    {Layer_t::Reshape, "Reshape"},
    {Layer_t::Sigmoid, "Sigmoid"},
    {Layer_t::Slice, "Slice"},
    {Layer_t::WeightMultiply, "WeightMultiply"},
    {Layer_t::MultiHeadAttention, "MultiHeadAttention"},
    {Layer_t::FmOrder2, "FmOrder2"},
    {Layer_t::Add, "Add"},
    {Layer_t::ReduceSum, "ReduceSum"},
    {Layer_t::Softmax, "Softmax"},
    {Layer_t::ElementwiseMultiply, "ElementwiseMultiply"},
    {Layer_t::FusedInnerProduct, "FusedInnerProduct"},
    {Layer_t::MultiCross, "MultiCross"}};

std::map<Embedding_t, std::string> EMBEDDING_TYPE_TO_STRING = {
    {Embedding_t::DistributedSlotSparseEmbeddingHash, "DistributedSlotSparseEmbeddingHash"},
    {Embedding_t::LocalizedSlotSparseEmbeddingHash, "LocalizedSlotSparseEmbeddingHash"},
    {Embedding_t::LocalizedSlotSparseEmbeddingOneHot, "LocalizedSlotSparseEmbeddingOneHot"},
    {Embedding_t::HybridSparseEmbedding, "HybridSparseEmbedding"}};

std::map<DataReaderSparse_t, std::string> READER_SPARSE_TYPE_TO_STRING = {
    {DataReaderSparse_t::Distributed, "DistributedSlot"},
    {DataReaderSparse_t::Localized, "LocalizedSlot"}};

std::map<Initializer_t, std::string> INITIALIZER_TYPE_TO_STRING = {
    {Initializer_t::Uniform, "Uniform"},
    {Initializer_t::XavierNorm, "XavierNorm"},
    {Initializer_t::XavierUniform, "XavierUniform"},
    {Initializer_t::Zero, "Zero"}};

std::map<AllReduceAlgo, std::string> ALLREDUCE_ALGO_TO_STRING = {
    {AllReduceAlgo::ONESHOT, "OneShot"}, {AllReduceAlgo::NCCL, "NCCL"}};

std::map<hybrid_embedding::CommunicationType, std::string> HE_COMM_TYPE_TO_STRING = {
    {hybrid_embedding::CommunicationType::IB_NVLink_Hier, "IB_NVLink_Hierarchical"},
    {hybrid_embedding::CommunicationType::IB_NVLink, "IB_NVLink"},
    {hybrid_embedding::CommunicationType::NVLink_SingleNode, "NVLink_SingleNode"}};

std::map<hybrid_embedding::HybridEmbeddingType, std::string> HE_TYPE_TO_STRING = {
    {hybrid_embedding::HybridEmbeddingType::Distributed, "Distributed"}};

std::map<FcPosition_t, std::string> FC_POSITION_TO_STRING = {
    {FcPosition_t::Head, "Head"}, {FcPosition_t::Body, "Body"},
    {FcPosition_t::Tail, "Tail"}, {FcPosition_t::Isolated, "Isolated"},
    {FcPosition_t::None, "None"},
};

std::map<Activation_t, std::string> FC_ACTIVATION_TO_STRING = {{Activation_t::Relu, "Relu"},
                                                               {Activation_t::None, "None"}};

}  // end of namespace

struct DataReaderParams {
  DataReaderType_t data_reader_type;
  std::vector<std::string> source;
  std::vector<std::string> keyset;
  std::string eval_source;
  Check_t check_type;
  int cache_eval_data;
  long long num_samples;
  long long eval_num_samples;
  bool float_label_dense;
  bool read_file_sequentially;
  int num_workers;
  std::vector<long long int> slot_size_array;
  AsyncParam async_param;
  DataReaderParams(DataReaderType_t data_reader_type, std::string source, std::string keyset,
                   std::string eval_source, Check_t check_type, int cache_eval_data,
                   long long num_samples, long long eval_num_samples, bool float_label_dense,bool read_file_sequentially,
                   int num_workers, std::vector<long long>& slot_size_array,
                   const AsyncParam& async_param);
  DataReaderParams(DataReaderType_t data_reader_type, std::vector<std::string> source,
                   std::vector<std::string> keyset, std::string eval_source, Check_t check_type,
                   int cache_eval_data, long long num_samples, long long eval_num_samples,
                   bool float_label_dense,bool read_file_sequentially, int num_workers, std::vector<long long>& slot_size_array,
                   const AsyncParam& async_param);
};

struct Input {
  std::map<std::string, int> labels_;
  std::map<std::string, float> label_weights_;
  int dense_dim;
  std::string dense_name;
  std::vector<DataReaderSparseParam> data_reader_sparse_param_array;
  Input(int label_dim, std::string label_name, int dense_dim, std::string dense_name,
        std::vector<DataReaderSparseParam>& data_reader_sparse_param_array);
  Input(std::vector<int> label_dim, std::vector<std::string> label_name, int dense_dim,
        std::string dense_name, std::vector<DataReaderSparseParam>& data_reader_sparse_param_array);
  Input(std::vector<int> label_dim, std::vector<std::string> label_name,
        std::vector<float> label_weights, int dense_dim, std::string dense_name,
        std::vector<DataReaderSparseParam>& data_reader_sparse_param_array);
};

struct SparseEmbedding {
  Embedding_t embedding_type;
  size_t max_vocabulary_size_per_gpu;
  size_t workspace_size_per_gpu_in_mb;
  size_t max_vocabulary_size_global;
  size_t embedding_vec_size;
  int combiner;
  std::string sparse_embedding_name;
  std::string bottom_name;
  std::vector<size_t> slot_size_array;
  std::shared_ptr<OptParamsPy> embedding_opt_params;
  HybridEmbeddingParam hybrid_embedding_param;
  SparseEmbedding(Embedding_t embedding_type, size_t workspace_size_per_gpu_in_mb,
                  size_t embedding_vec_size, const std::string& combiner_str,
                  std::string sparse_embedding_name, std::string bottom_name,
                  std::vector<size_t>& slot_size_array,
                  std::shared_ptr<OptParamsPy>& embedding_opt_params,
                  const HybridEmbeddingParam& hybrid_embedding_param);

  void initialize_max_vocabulary_size_per_gpu();
};

struct EmbeddingTrainingCacheParams {
  bool use_embedding_training_cache;
  std::vector<TrainPSType_t> ps_types;
  std::vector<std::string> sparse_models;
  std::vector<std::string> local_paths;
  std::vector<HMemCacheConfig> hmem_cache_configs;
  std::vector<std::string> incremental_keyset_files;
  EmbeddingTrainingCacheParams(std::vector<TrainPSType_t>& _ps_types,
                               std::vector<std::string>& _sparse_models,
                               std::vector<std::string>& _local_paths,
                               std::vector<HMemCacheConfig>& _hmem_cache_configs);
  EmbeddingTrainingCacheParams();
};

struct DenseLayer {
  Layer_t layer_type;
  std::vector<std::string> bottom_names;
  std::vector<std::string> top_names;
  float factor;
  float eps;
  Initializer_t gamma_init_type;
  Initializer_t beta_init_type;
  float dropout_rate;
  float elu_alpha;
  size_t num_output;
  Initializer_t weight_init_type;
  Initializer_t bias_init_type;
  int num_layers;
  size_t leading_dim;
  size_t time_step;
  size_t batchsize;
  size_t SeqLength;
  size_t vector_size;
  bool selected;
  std::vector<int> selected_slots;
  std::vector<std::pair<int, int>> ranges;
  std::vector<int> indices;
  std::vector<size_t> weight_dims;
  size_t out_dim;
  int axis;
  std::vector<float> target_weight_vec;
  bool use_regularizer;
  Regularizer_t regularizer_type;
  float lambda;
  FcPosition_t pos_type;
  Activation_t act_type;
  DenseLayerSwitchs dense_layer_switches;
  DenseLayer(Layer_t layer_type, std::vector<std::string>& bottom_names,
             std::vector<std::string>& top_names, float factor = 1.0, float eps = 0.00001,
             Initializer_t gamma_init_type = Initializer_t::Default,
             Initializer_t beta_init_type = Initializer_t::Default, float dropout_rate = 0.5,
             float elu_alpha = 1.0, size_t num_output = 1,
             Initializer_t weight_init_type = Initializer_t::Default,
             Initializer_t bias_init_type = Initializer_t::Default, int num_layers = 0,
             size_t leading_dim = 1, size_t time_step = 0, size_t batchsize = 1,
             size_t SeqLength = 1, size_t vector_size = 1, bool selected = false,
             std::vector<int> selected_slots = std::vector<int>(),
             std::vector<std::pair<int, int>> ranges = std::vector<std::pair<int, int>>(),
             std::vector<int> indices = std::vector<int>(),
             std::vector<size_t> weight_dims = std::vector<size_t>(), size_t out_dim = 0,
             int axis = 1, std::vector<float> target_weight_vec = std::vector<float>(),
             bool use_regularizer = false, Regularizer_t regularizer_type = Regularizer_t::L1,
             float lambda = 0, FcPosition_t pos_type = FcPosition_t::None,
             Activation_t act_type = Activation_t::Relu,
             DenseLayerSwitchs dense_layer_switches = {false});
};

struct GroupDenseLayer {
  GroupLayer_t group_layer_type;
  std::vector<std::string> bottom_name_list;
  std::vector<std::string> top_name_list;
  std::vector<size_t> num_outputs;
  Activation_t last_act_type;
  GroupDenseLayer(GroupLayer_t group_layer_type, std::vector<std::string>& bottom_name_list,
                  std::vector<std::string>& top_name_list, std::vector<size_t>& num_outputs,
                  Activation_t last_act_type = Activation_t::Relu);
};

template <typename TypeKey>
void add_input(Input& input, DataReaderParams& reader_params,
               std::map<std::string, SparseInput<TypeKey>>& sparse_input_map,
               std::vector<std::vector<TensorEntry>>& train_tensor_entries_list,
               std::vector<std::vector<TensorEntry>>& evaluate_tensor_entries_list,
               std::shared_ptr<IDataReader>& train_data_reader,
               std::shared_ptr<IDataReader>& evaluate_data_reader,
               std::shared_ptr<IDataReader>& init_data_reader, size_t batch_size,
               size_t batch_size_eval, bool use_mixed_precision, bool repeat_dataset,
               bool enable_overlap, size_t num_iterations_statistics,
               const std::shared_ptr<ResourceManager> resource_manager);

template <typename TypeKey, typename TypeFP>
void add_sparse_embedding(SparseEmbedding& sparse_embedding,
                          std::map<std::string, SparseInput<TypeKey>>& sparse_input_map,
                          std::vector<std::vector<TensorEntry>>& train_tensor_entries_list,
                          std::vector<std::vector<TensorEntry>>& evaluate_tensor_entries_list,
                          std::vector<std::shared_ptr<IEmbedding>>& embeddings,
                          const std::shared_ptr<ResourceManager>& resource_manager,
                          size_t batch_size, size_t batch_size_eval,
                          OptParams& embedding_opt_params,
                          std::shared_ptr<ExchangeWgrad>& exchange_wgrad, bool use_cuda_graph,
                          bool grouped_all_reduce, bool use_holistic_cuda_graph,
                          size_t num_iterations_statistics, GpuLearningRateSchedulers& gpu_lr_sches,
                          bool overlap_ar_a2a, bool eval_overlap);

Input get_input_from_json(const nlohmann::json& j_input);

DenseLayer get_dense_layer_from_json(const nlohmann::json& j_dense_layer);

SparseEmbedding get_sparse_embedding_from_json(const nlohmann::json& j_sparse_embedding);

void save_graph_to_json(nlohmann::json& layer_config_array,
                        std::vector<DenseLayer>& dense_layer_params,
                        std::vector<SparseEmbedding>& sparse_embedding_params,
                        std::vector<Input>& input_params,
                        std::vector<std::shared_ptr<OptParamsPy>>& embedding_opt_params_list,
                        bool use_mixed_precision);

void calculate_tensor_dimensions(std::map<std::string, std::vector<int>>& tensor_shape_info_raw,
                                 DenseLayer& dense_layer);

void init_optimizer(OptParams& opt_params, const Solver& solver,
                    const std::shared_ptr<OptParamsPy>& opt_params_py);

void init_learning_rate_scheduler(std::shared_ptr<LearningRateScheduler>& lr_sch,
                                  const Solver& solver, GpuLearningRateSchedulers& gpu_lr_sches,
                                  const std::shared_ptr<ResourceManager>& resource_manager);
/**
 * @brief Main HugeCTR class
 *
 * This is a class supporting basic usages of hugectr, which includes
 * train; evaluation; get loss; load and download trained parameters.
 * To learn how to use those method, please refer to main.cpp.
 */
class Model {
 public:
  virtual ~Model();
  Model(const Solver& solver, const DataReaderParams& reader_params,
        std::shared_ptr<OptParamsPy>& opt_params,
        std::shared_ptr<EmbeddingTrainingCacheParams>& etc_params);
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  void graph_to_json(std::string graph_config_file);

  void construct_from_json(const std::string& graph_config_file, bool include_dense_network);

  virtual void add(Input& input);

  virtual void add(SparseEmbedding& sparse_embedding);

  virtual void add(DenseLayer& dense_layer);

  virtual void add(const EmbeddingCollectionPlaceHolder& embedding_collection);

  virtual void add_internal(DenseLayer& dense_layer);

  void add(GroupDenseLayer& group_dense_layer);

  void graph_analysis();

  virtual void compile();

  virtual void compile(std::vector<std::string>& label_names, std::vector<float>& label_weights);

  void update_label_weights(std::vector<std::string>& label_names,
                            std::vector<float>& label_weights);

  void summary();

  virtual void fit(int num_epochs, int max_iter, int display, int eval_interval, int snapshot,
                   std::string snapshot_prefix);

  void set_source(std::vector<std::string> source, std::vector<std::string> keyset,
                  std::string eval_source);

  void set_source(std::string source, std::string eval_source);

  virtual bool train(bool is_first_batch);

  virtual bool eval(bool is_first_batch);

  std::vector<std::pair<std::string, float>> get_eval_metrics();

  Error_t get_current_loss(float* loss);

  Error_t download_params_to_files(std::string prefix, int iter);

  Error_t export_predictions(const std::string& output_prediction_file_name,
                             const std::string& output_label_file_name);

  void check_overflow() const;

  void copy_weights_for_evaluation();

  void start_data_reading() {
    train_data_reader_->start();
    evaluate_data_reader_->start();
  }

  void reset_learning_rate_scheduler(float base_lr, size_t warmup_steps, size_t decay_start,
                                     size_t decay_steps, float decay_power, float end_lr) {
    if (!lr_sch_) {
      HCTR_OWN_THROW(Error_t::IllegalCall, "learning rate scheduler should be initialized first");
    }
    lr_sch_->reset(base_lr, warmup_steps, decay_start, decay_steps, decay_power, end_lr);
  }

  Error_t set_learning_rate(float lr) {
    float lr_embedding{0.f};
    float lr_dense = is_dense_trainable_ ? lr : 0.f;
    for (auto& embedding : embeddings_) {
      lr_embedding = embedding->is_trainable() ? lr : 0.f;
      embedding->set_learning_rate(lr_embedding);
    }
    for (auto& network : networks_) {
      network->set_learning_rate(lr_dense);
    }
    if (solver_.use_embedding_collection) {
      for (auto& table_list : table_major_ebc_table_list_) {
        for (auto& t : table_list) {
          t->set_learning_rate(lr);
        }
      }
    }
    return Error_t::Success;
  }

  long long get_params_num() {
    long long size = 0;
    for (auto& embedding : embeddings_) {
      size += embedding->get_params_num();
    }
    return static_cast<long long>(networks_[0]->get_params_num()) + size;
  }

  const std::shared_ptr<EmbeddingTrainingCache>& get_embedding_training_cache() const {
    if (!embedding_training_cache_) {
      HCTR_OWN_THROW(Error_t::IllegalCall, "embedding training cache should be initialized first");
    }
    return embedding_training_cache_;
  }

  const std::shared_ptr<IDataReader>& get_train_data_reader() const {
    if (!train_data_reader_) {
      HCTR_OWN_THROW(Error_t::IllegalCall, "train data reader should be initialized first");
    }
    return train_data_reader_;
  }
  const std::shared_ptr<IDataReader>& get_evaluate_data_reader() const {
    if (!evaluate_data_reader_) {
      HCTR_OWN_THROW(Error_t::IllegalCall, "evaluate data reader should be initialized first");
    }
    return evaluate_data_reader_;
  }
  const std::shared_ptr<LearningRateScheduler>& get_learning_rate_scheduler() const {
    if (!lr_sch_) {
      HCTR_OWN_THROW(Error_t::IllegalCall, "learning rate scheduler should be initialized first");
    }
    return lr_sch_;
  }

  bool use_gpu_learning_rate_scheduling() const {
    return !embeddings_[0]->get_learning_rate_schedulers().empty();
  }

  void load_dense_weights(const std::string& dense_model_file);
  void load_sparse_weights(const std::vector<std::string>& sparse_embedding_files);
  void load_sparse_weights(const std::map<std::string, std::string>& sparse_embedding_files_maps);
  void load_dense_optimizer_states(const std::string& dense_opt_states_file);
  void load_sparse_optimizer_states(const std::vector<std::string>& sparse_opt_states_files);
  void load_sparse_optimizer_states(
      const std::map<std::string, std::string>& sparse_opt_states_files_map);
  void freeze_embedding() {
    for (auto& one_embedding : embeddings_) {
      one_embedding->freeze();
    }
  };
  void freeze_embedding(const std::string& embedding_name) {
    if (embeddings_map_.find(embedding_name) == embeddings_map_.end()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "No such embedding name: " + embedding_name);
    }
    auto it = embeddings_map_.find(embedding_name);
    it->second->freeze();
  }
  void freeze_dense() { is_dense_trainable_ = false; };
  void unfreeze_embedding() {
    for (auto& one_embedding : embeddings_) {
      one_embedding->unfreeze();
    }
  };
  void unfreeze_embedding(const std::string& embedding_name) {
    if (embeddings_map_.find(embedding_name) == embeddings_map_.end()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "No such embedding name: " + embedding_name);
    }
    auto it = embeddings_map_.find(embedding_name);
    it->second->unfreeze();
  }
  void unfreeze_dense() { is_dense_trainable_ = true; };
  std::vector<std::pair<std::vector<long long>, std::vector<float>>>& get_incremental_model();
  void dump_incremental_model_2kafka();

 protected:
  Solver solver_;
  DataReaderParams reader_params_;
  OptParams opt_params_;
  std::shared_ptr<OptParamsPy> opt_params_py_;
  std::shared_ptr<EmbeddingTrainingCacheParams> etc_params_;
  std::vector<std::shared_ptr<OptParamsPy>> embedding_opt_params_list_;
  std::shared_ptr<MessageSink<long long>> message_sink_;
  std::shared_ptr<LearningRateScheduler> lr_sch_;
  GpuLearningRateSchedulers gpu_lr_sches_;

  std::map<std::string, SparseInput<long long>> sparse_input_map_64_;
  std::map<std::string, SparseInput<unsigned int>> sparse_input_map_32_;
  std::vector<std::vector<TensorEntry>> train_tensor_entries_list_;
  std::vector<std::vector<TensorEntry>> evaluate_tensor_entries_list_;
  std::map<std::string, bool> tensor_active_;

  std::map<std::string, float> label_weights_;

  bool data_reader_train_status_;
  bool data_reader_eval_status_;
  bool buff_allocated_;
  bool etc_created_;
  bool is_dense_trainable_;
  std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> blobs_buff_list_;
  std::vector<std::shared_ptr<BufferBlock2<float>>> train_weight_buff_list_;
  std::vector<std::shared_ptr<BufferBlock2<__half>>> train_weight_buff_half_list_;
  std::vector<std::shared_ptr<BufferBlock2<float>>> wgrad_buff_list_;
  std::vector<std::shared_ptr<BufferBlock2<__half>>> wgrad_buff_half_list_;
  std::vector<std::shared_ptr<BufferBlock2<float>>> evaluate_weight_buff_list_;
  std::vector<std::shared_ptr<BufferBlock2<__half>>> evaluate_weight_buff_half_list_;
  std::vector<std::shared_ptr<BufferBlock2<float>>> wgrad_buff_placeholder_list_;
  std::vector<std::shared_ptr<BufferBlock2<__half>>> wgrad_buff_half_placeholder_list_;

  std::vector<std::shared_ptr<BufferBlock2<float>>> opt_buff_list_;
  std::vector<std::shared_ptr<BufferBlock2<__half>>> opt_buff_half_list_;

  bool set_source_flag_{true};
  bool graph_finalized_{false};
  std::vector<std::pair<std::vector<long long>, std::vector<float>>> inc_sparse_model_;

  std::vector<DenseLayer> dense_layer_params_raw_;
  std::map<std::string, std::vector<int>> tensor_shape_info_raw_;
  std::vector<DenseLayer> dense_layer_params_;
  std::vector<SparseEmbedding> sparse_embedding_params_;
  std::vector<Input> input_params_;
  std::vector<std::string> data_input_info_; /**< data input name */
  std::map<std::string, std::vector<size_t>> tensor_shape_info_;
  std::vector<std::pair<std::string, std::string>>
      input_output_info_;                               /**< input output name of each layer. */
  std::vector<std::string> layer_info_;                 /**< type of each layer. */
  std::vector<std::shared_ptr<Network>> networks_;      /**< networks (dense) used in training. */
  std::vector<std::shared_ptr<IEmbedding>> embeddings_; /**< embedding */

  std::map<std::string, int> hotness_map_;
  std::vector<std::unique_ptr<embedding::IEmbeddingCollectionForward>> ebc_forward_list_;
  std::vector<std::unique_ptr<embedding::IEmbeddingCollectionForward>> eval_ebc_forward_list_;
  std::vector<std::unique_ptr<embedding::IEmbeddingCollectionBackward>> ebc_backward_list_;
  std::vector<std::vector<std::unique_ptr<embedding::IEmbeddingTable>>> table_major_ebc_table_list_;

  std::vector<std::vector<core::Tensor>> ebc_grad_key_list_;
  std::vector<std::vector<size_t>> ebc_num_grad_key_list_;
  std::vector<std::vector<core::Tensor>> ebc_grad_id_space_offset_list_;
  std::vector<std::vector<size_t>> ebc_num_grad_key_id_space_offset_list_;
  std::vector<std::vector<core::Tensor>> ebc_grad_ev_list_;
  std::vector<std::vector<core::Tensor>> ebc_grad_ev_offset_list_;
  std::vector<std::vector<core::Tensor>> ebc_grad_id_space_list_;

  std::vector<core::Tensor> train_ebc_key_list_;
  std::vector<core::Tensor> train_ebc_bucket_range_list_;
  std::vector<size_t*> train_ebc_num_keys_list_;
  std::vector<core::Tensor> train_ebc_sparse_weight_list_;
  std::vector<core::Tensor> evaluate_ebc_key_list_;
  std::vector<core::Tensor> evaluate_ebc_bucket_range_list_;
  std::vector<size_t*> evaluate_ebc_num_keys_list_;
  std::vector<core::Tensor> evaluate_ebc_sparse_weight_list_;

  std::vector<core::Tensor> train_ebc_outptut_;
  std::vector<core::Tensor> evaluate_ebc_outptut_;

  std::shared_ptr<EmbeddingTrainingCache>
      embedding_training_cache_; /**< embedding training cache for model oversubscribing. */

  std::shared_ptr<IDataReader>
      train_data_reader_; /**< data reader to reading data from data set to embedding. */
  std::shared_ptr<IDataReader> evaluate_data_reader_; /**< data reader for evaluation. */
  std::shared_ptr<ResourceManager>
      resource_manager_;     /**< GPU resources include handles and streams etc.*/
  metrics::Metrics metrics_; /**< evaluation metrics. */

  long long current_eval_batchsize_; /**< used for export prediction in epoch mode. */

  std::shared_ptr<IDataReader> init_data_reader_;
  std::shared_ptr<ExchangeWgrad> exchange_wgrad_;
  std::vector<GraphWrapper> train_graphs_;
  std::vector<cudaEvent_t> fork_events_;
  bool dlrm_bottom_mlp_;
  bool high_level_eval_;
  HugeCTR::Timer timer_log;
  std::map<std::string, std::shared_ptr<IEmbedding>> embeddings_map_;

  Error_t download_dense_params_to_files_(std::string weights_file,
                                          std::string dense_opt_states_file,
                                          const DataSourceParams& data_source_params);

  Error_t download_sparse_params_to_files_(const std::vector<std::string>& embedding_files,
                                           const std::vector<std::string>& sparse_opt_state_files,
                                           const DataSourceParams& data_source_params);

  template <typename TypeEmbeddingComp>
  std::shared_ptr<EmbeddingTrainingCache> create_embedding_training_cache_(
      const std::vector<TrainPSType_t>& ps_types,
      const std::vector<std::string>& sparse_embedding_files,
      const std::vector<std::string>& local_paths,
      const std::vector<HMemCacheConfig>& hmem_cache_configs);
  void init_params_for_dense_();
  void init_params_for_sparse_();
  void init_embedding_training_cache_(const std::vector<TrainPSType_t>& ps_types,
                                      const std::vector<std::string>& sparse_embedding_files,
                                      const std::vector<std::string>& local_paths,
                                      const std::vector<HMemCacheConfig>& hmem_cache_configs);
  Error_t load_params_for_dense_(const std::string& model_file,
                                 const DataSourceParams& data_source_params);
  Error_t load_params_for_sparse_(const std::vector<std::string>& embedding_file,
                                  const DataSourceParams& data_source_params);
  Error_t load_opt_states_for_dense_(const std::string& dense_opt_states_file,
                                     const DataSourceParams& data_source_params);
  Error_t load_opt_states_for_sparse_(const std::vector<std::string>& sparse_opt_states_files,
                                      const DataSourceParams& data_source_params);
  virtual void exchange_wgrad(size_t device_id);
  virtual void train_overlapped();
  virtual void add_dense_layer(DenseLayer& dense_layer);
  virtual void add_dense_layer_internal(
      DenseLayer& dense_layer, std::vector<TensorEntry>& tensor_entries,
      const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
      const std::shared_ptr<BufferBlock2<float>>& weight_buff,
      const std::shared_ptr<BufferBlock2<__half>>& weight_buff_half,
      const std::shared_ptr<BufferBlock2<float>>& wgrad_buff,
      const std::shared_ptr<BufferBlock2<__half>>& wgrad_buff_half,
      std::map<std::string, Tensor2<float>>& loss_tensor,
      std::vector<std::unique_ptr<Layer>>& layers,
      std::map<std::string, std::unique_ptr<ILoss>>& loss, bool enable_cuda_graph,
      bool async_mlp_wgrad, std::map<std::string, metrics::RawMetricMap>* raw_metrics,
      int num_networks_in_global, const std::shared_ptr<GPUResource>& gpu_resource,
      bool use_mixed_precision, bool enable_tf32_compute, float scaler, bool use_algorithm_search,
      std::vector<Layer*>* top_layers, std::vector<Layer*>* bottom_layers, bool dlrm_bottom_mlp);

  struct GraphScheduler {
   private:
    volatile size_t* executed_iter;
    size_t launched_iter;

   public:
    GraphScheduler(std::shared_ptr<ResourceManager> resource_manager) : launched_iter(0) {
      // set up trickling launch
      CudaCPUDeviceContext ctx(resource_manager->get_local_gpu(0)->get_device_id());
      HCTR_LIB_THROW(cudaMallocHost((void**)&executed_iter, sizeof(size_t)));
      *executed_iter = 0;
    }
    ~GraphScheduler() { cudaFreeHost(const_cast<size_t*>(executed_iter)); }
    void trickling() {
      // this function is called by the only thread, hence no need to specify the rank
      while (launched_iter > *(executed_iter) + 1) {
        usleep(10);
      }
      launched_iter++;
    }
    void record_execution(size_t local_rank, cudaStream_t stream) {
      // Only rank 0 needs to do the work
      if (local_rank == 0) inc_var(executed_iter, stream);
    }
  };
  std::unique_ptr<GraphScheduler> graph_scheduler_;
};

}  // namespace HugeCTR
