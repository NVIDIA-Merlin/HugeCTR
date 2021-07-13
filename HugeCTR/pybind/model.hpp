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
#include <parser.hpp>
#include <utils.hpp>
#include <common.hpp>
#include <embedding.hpp>
#include <optimizer.hpp>
#include <loss.hpp>
#include <metrics.hpp>
#include <network.hpp>
#include <string>
#include <thread>
#include <utility>
#include <HugeCTR/include/embedding.hpp>
#include <HugeCTR/include/model_oversubscriber/model_oversubscriber.hpp>

namespace HugeCTR {

namespace {

std::map<Layer_t, std::string> LAYER_TYPE_TO_STRING = {
  {Layer_t::BatchNorm, "BatchNorm"},
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
  {Layer_t::DotProduct, "DotProduct"},
  {Layer_t::CrossEntropyLoss, "CrossEntropyLoss"},
  {Layer_t::MultiCrossEntropyLoss, "MultiCrossEntropyLoss"},
  {Layer_t::ElementwiseMultiply, "ElementwiseMultiply"},
  {Layer_t::MultiCross, "MultiCross"}};

std::map<Layer_t, std::string> LAYER_TYPE_TO_STRING_MP = {
  {Layer_t::BatchNorm, "BatchNorm"},
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
  {Layer_t::DotProduct, "DotProduct"},
  {Layer_t::ElementwiseMultiply, "ElementwiseMultiply"},
  {Layer_t::FusedInnerProduct, "FusedInnerProduct"},
  {Layer_t::MultiCross, "MultiCross"}};

std::map<Embedding_t, std::string> EMBEDDING_TYPE_TO_STRING = {
    {Embedding_t::DistributedSlotSparseEmbeddingHash, "DistributedSlotSparseEmbeddingHash"},
    {Embedding_t::LocalizedSlotSparseEmbeddingHash, "LocalizedSlotSparseEmbeddingHash"},
    {Embedding_t::LocalizedSlotSparseEmbeddingOneHot, "LocalizedSlotSparseEmbeddingOneHot"}};

std::map<DataReaderSparse_t, std::string> READER_SPARSE_TYPE_TO_STRING = {
    {DataReaderSparse_t::Distributed, "DistributedSlot"},
    {DataReaderSparse_t::Localized, "LocalizedSlot"}};


std::map<Initializer_t, std::string> INITIALIZER_TYPE_TO_STRING = {
  {Initializer_t::Uniform, "Uniform"},
  {Initializer_t::XavierNorm, "XavierNorm"},
  {Initializer_t::XavierUniform, "XavierUniform"},
  {Initializer_t::Zero, "Zero"}};

} // end of namespace

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
  int num_workers;
  std::vector<long long int> slot_size_array;
  DataReaderParams(DataReaderType_t data_reader_type,
       std::vector<std::string> source,
       std::vector<std::string> keyset,
       std::string eval_source,
       Check_t check_type,
       int cache_eval_data,
       long long num_samples,
       long long eval_num_samples,
       bool float_label_dense,
       int num_workers,
       std::vector<long long int> slot_size_array = std::vector<long long int>());
};

struct Input {
  int label_dim;
  std::string label_name;
  int dense_dim;
  std::string dense_name;
  std::vector<DataReaderSparseParam> data_reader_sparse_param_array;
  Input(int label_dim,
       std::string label_name,
       int dense_dim,
       std::string dense_name,
       std::vector<DataReaderSparseParam>& data_reader_sparse_param_array);
};

struct SparseEmbedding {
  Embedding_t embedding_type;
  size_t max_vocabulary_size_per_gpu;
  size_t embedding_vec_size;
  int combiner;
  std::string sparse_embedding_name;
  std::string bottom_name;
  std::vector<size_t> slot_size_array; 
  std::shared_ptr<OptParamsPy> embedding_opt_params;
  
  SparseEmbedding(Embedding_t embedding_type,
                 size_t workspace_size_per_gpu_in_mb,
                 size_t embedding_vec_size,
                 const std::string &combiner_str,
                 std::string sparse_embedding_name,
                 std::string bottom_name,
                 std::vector<size_t>& slot_size_array,
                 std::shared_ptr<OptParamsPy>& embedding_opt_params);

};

struct ModelOversubscriberParams {
  bool use_model_oversubscriber;
  bool use_host_memory_ps;
  bool train_from_scratch;
  std::vector<std::string> trained_sparse_models;
  std::vector<std::string> dest_sparse_models;
  ModelOversubscriberParams(bool train_from_scratch, bool use_host_memory_ps,
                           std::vector<std::string>& trained_sparse_models,
                           std::vector<std::string>& dest_sparse_models);
  ModelOversubscriberParams();
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
  bool selected;
  std::vector<int> selected_slots;
  std::vector<std::pair<int, int>> ranges;
  std::vector<size_t> weight_dims;
  size_t out_dim;
  int axis;
  std::vector<float> target_weight_vec;
  bool use_regularizer;
  Regularizer_t regularizer_type;
  float lambda;
  DenseLayer(Layer_t layer_type,
            std::vector<std::string>& bottom_names,
            std::vector<std::string>& top_names,
            float factor = 1.0,
            float eps = 0.00001,
            Initializer_t gamma_init_type = Initializer_t::Default,
            Initializer_t beta_init_type = Initializer_t::Default,
            float dropout_rate = 0.5,
            float elu_alpha = 1.0,
            size_t num_output = 1,
            Initializer_t weight_init_type = Initializer_t::Default,
            Initializer_t bias_init_type = Initializer_t::Default,
            int num_layers = 0,
            size_t leading_dim = 1,
            bool selected = false,
            std::vector<int> selected_slots = std::vector<int>(),
            std::vector<std::pair<int, int>> ranges = std::vector<std::pair<int, int>>(),
            std::vector<size_t> weight_dims = std::vector<size_t>(),
            size_t out_dim = 0,
            int axis = 1,
            std::vector<float> target_weight_vec = std::vector<float>(),
            bool use_regularizer = false,
            Regularizer_t regularizer_type = Regularizer_t::L1,
            float lambda = 0);
};

template <typename TypeKey>
void add_input(Input& input, DataReaderParams& reader_params,
            std::map<std::string, SparseInput<TypeKey>>& sparse_input_map,
            std::vector<std::vector<TensorEntry>>& train_tensor_entries_list,
            std::vector<std::vector<TensorEntry>>& evaluate_tensor_entries_list,
            std::shared_ptr<IDataReader>& train_data_reader,
            std::shared_ptr<IDataReader>& evaluate_data_reader, size_t batch_size,
            size_t batch_size_eval, bool use_mixed_precision, bool repeat_dataset,
            const std::shared_ptr<ResourceManager> resource_manager);

template <typename TypeKey, typename TypeFP>
void add_sparse_embedding(SparseEmbedding& sparse_embedding,
            std::map<std::string, SparseInput<TypeKey>>& sparse_input_map,
            std::vector<std::vector<TensorEntry>>& train_tensor_entries_list,
            std::vector<std::vector<TensorEntry>>& evaluate_tensor_entries_list,
            std::vector<std::shared_ptr<IEmbedding>>& embeddings,
            const std::shared_ptr<ResourceManager>& resource_manager,
            size_t batch_size, size_t batch_size_eval,
            OptParams& embedding_opt_params);


Input get_input_from_json(const nlohmann::json& j_input);

DenseLayer get_dense_layer_from_json(const nlohmann::json& j_dense_layer);

SparseEmbedding get_sparse_embedding_from_json(
    const nlohmann::json& j_sparse_embedding);

void save_graph_to_json(nlohmann::json& layer_config_array,
                       std::vector<DenseLayer>& dense_layer_params,
                       std::vector<SparseEmbedding>& sparse_embedding_params,
                       std::vector<Input>& input_params,
                       std::vector<std::shared_ptr<OptParamsPy>>& embedding_opt_params_list);

void init_optimizer(OptParams& opt_params, const Solver& solver,
                    const std::shared_ptr<OptParamsPy>& opt_params_py);

void init_learning_rate_scheduler(
    std::shared_ptr<LearningRateScheduler>& lr_sch, const Solver& solver);
/**
 * @brief Main HugeCTR class
 *
 * This is a class supporting basic usages of hugectr, which includes
 * train; evaluation; get loss; load and download trained parameters.
 * To learn how to use those method, please refer to main.cpp.
 */
class Model {
 public:
  ~Model();
  Model(const Solver& solver,
      const DataReaderParams& reader_params, 
      std::shared_ptr<OptParamsPy>& opt_params,
      std::shared_ptr<ModelOversubscriberParams>& mos_params);
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  void graph_to_json(std::string graph_config_file);

  void construct_from_json(const std::string& graph_config_file,
                          bool include_dense_network);

  void add(Input& input);
  
  void add(SparseEmbedding& sparse_embedding);
  
  void add(DenseLayer& dense_layer);
  
  void compile();

  void summary();

  void fit(int num_epochs, int max_iter, int display, int eval_interval,
          int snapshot, std::string snapshot_prefix);

  void set_source(std::vector<std::string> source,
                  std::vector<std::string> keyset, std::string eval_source);

  void set_source(std::string source, std::string eval_source);

  bool train();

  bool eval();

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

  void reset_learning_rate_scheduler(float base_lr, size_t warmup_steps,
      size_t decay_start, size_t decay_steps, float decay_power, float end_lr) {
    if (!lr_sch_) {
      CK_THROW_(Error_t::IllegalCall,
          "learning rate scheduler should be initialized first");
    }
    lr_sch_->reset(base_lr, warmup_steps, decay_start, decay_steps, decay_power, end_lr);
  }

  Error_t set_learning_rate(float lr) {
    float lr_embedding = is_embedding_trainable_?lr:0.0;
    float lr_dense = is_dense_trainable_?lr:0.0;
    for (auto& embedding : embeddings_) {
      embedding->set_learning_rate(lr_embedding);
    }
    for (auto& network : networks_) {
      network->set_learning_rate(lr_dense);
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

  const std::shared_ptr<ModelOversubscriber>& get_model_oversubscriber() const {
    if (!model_oversubscriber_) {
      CK_THROW_(Error_t::IllegalCall, "model oversubscriber should be initialized first");
    }
    return model_oversubscriber_;
  }

  const std::shared_ptr<IDataReader>& get_train_data_reader() const {
    if (!train_data_reader_) {
      CK_THROW_(Error_t::IllegalCall, "train data reader should be initialized first");
    }
    return train_data_reader_; 
  }
  const std::shared_ptr<IDataReader>& get_evaluate_data_reader() const {
    if (!evaluate_data_reader_) {
      CK_THROW_(Error_t::IllegalCall, "evaluate data reader should be initialized first");
    }
    return evaluate_data_reader_; 
  }
  const std::shared_ptr<LearningRateScheduler>& get_learning_rate_scheduler() const {
    if (!lr_sch_) {
      CK_THROW_(Error_t::IllegalCall, "learning rate scheduler should be initialized first");
    }
    return lr_sch_; 
  }

  void load_dense_weights(const std::string& dense_model_file);
  void load_sparse_weights(const std::vector<std::string>& sparse_embedding_files);
  void load_dense_optimizer_states(const std::string& dense_opt_states_file);
  void load_sparse_optimizer_states(const std::vector<std::string>& sparse_opt_states_files);
  void freeze_embedding() { is_embedding_trainable_ = false; };
  void freeze_dense() { is_dense_trainable_ = false; };
  void unfreeze_embedding() { is_embedding_trainable_ = true; };
  void unfreeze_dense() { is_dense_trainable_ = true; };

 private:
  Solver solver_;
  DataReaderParams reader_params_;
  OptParams opt_params_;
  std::shared_ptr<OptParamsPy> opt_params_py_;
  std::shared_ptr<ModelOversubscriberParams> mos_params_;
  std::vector<std::shared_ptr<OptParamsPy>> embedding_opt_params_list_;
  std::shared_ptr<LearningRateScheduler> lr_sch_;
  std::map<std::string, SparseInput<long long>> sparse_input_map_64_;
  std::map<std::string, SparseInput<unsigned int>> sparse_input_map_32_;
  std::vector<std::vector<TensorEntry>> train_tensor_entries_list_;
  std::vector<std::vector<TensorEntry>> evaluate_tensor_entries_list_;
  std::map<std::string, bool> tensor_active_;
  
  bool data_reader_train_status_;
  bool data_reader_eval_status_;
  bool buff_allocated_;
  bool mos_created_;
  bool is_embedding_trainable_;
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

  std::vector<DenseLayer> dense_layer_params_;
  std::vector<SparseEmbedding> sparse_embedding_params_;
  std::vector<Input> input_params_;
  std::vector<std::string> data_input_info_; /**< data input name */
  std::map<std::string, std::vector<size_t>> tensor_shape_info_;
  std::vector<std::pair<std::string, std::string>> input_output_info_;   /**< input output name of each layer. */
  std::vector<std::string> layer_info_;   /**< type of each layer. */
  std::vector<std::shared_ptr<Network>> networks_;      /**< networks (dense) used in training. */
  std::vector<std::shared_ptr<IEmbedding>> embeddings_; /**< embedding */
  std::shared_ptr<ModelOversubscriber> model_oversubscriber_; /**< model oversubscriber for model oversubscribing. */

  std::shared_ptr<IDataReader> train_data_reader_; /**< data reader to reading data from data set to embedding. */
  std::shared_ptr<IDataReader> evaluate_data_reader_; /**< data reader for evaluation. */
  std::shared_ptr<ResourceManager> resource_manager_; /**< GPU resources include handles and streams etc.*/
  metrics::Metrics metrics_; /**< evaluation metrics. */
  
  Error_t download_dense_params_to_files_(std::string weights_file,
                                          std::string dense_opt_states_file);
                                          

  Error_t download_sparse_params_to_files_(const std::vector<std::string>& embedding_files,
                                          const std::vector<std::string>& sparse_opt_state_files);
  
  template <typename TypeEmbeddingComp>
  std::shared_ptr<ModelOversubscriber> create_model_oversubscriber_(
      bool use_host_memory_ps, const std::vector<std::string>& sparse_embedding_files);
  void init_params_for_dense_();
  void init_params_for_sparse_();
  void init_model_oversubscriber_(
      bool use_host_memory_ps, const std::vector<std::string>& sparse_embedding_files);
  Error_t load_params_for_dense_(const std::string& model_file);
  Error_t load_params_for_sparse_(const std::vector<std::string>& embedding_file);
  Error_t load_opt_states_for_dense_(const std::string& dense_opt_states_file);
  Error_t load_opt_states_for_sparse_(const std::vector<std::string>& sparse_opt_states_files);
};

} // namespace HugeCTR
