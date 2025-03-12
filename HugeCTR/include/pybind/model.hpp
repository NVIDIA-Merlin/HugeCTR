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

#include <common.hpp>
#include <core23_helper.hpp>
#include <core23_network.hpp>
#include <embedding.hpp>
#include <embedding/data_distributor/data_distributor.hpp>
#include <embedding_storage/weight_io/parameter_IO.hpp>
#include <embeddings/embedding_collection.hpp>
#include <exchange_wgrad.hpp>
#include <graph_wrapper.hpp>
#include <inference/preallocated_buffer2.hpp>
#include <io/filesystem.hpp>
#include <loss.hpp>
#include <metrics.hpp>
#include <optimizer.hpp>
#include <parser.hpp>
#include <pipeline.hpp>
#include <pybind/common_helpers.hpp>
#include <string>
#include <thread>
#include <utility>
#include <utils.hpp>

namespace HugeCTR {

class Network;

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
    {Layer_t::Select, "Select"},
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
    {Layer_t::MultiCross, "MultiCross"},
    {Layer_t::MLP, "MLP"},
    {Layer_t::SequenceMask, "SequenceMask"}};

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
    {Layer_t::Select, "Select"},
    {Layer_t::Sigmoid, "Sigmoid"},
    {Layer_t::Slice, "Slice"},
    {Layer_t::WeightMultiply, "WeightMultiply"},
    {Layer_t::MultiHeadAttention, "MultiHeadAttention"},
    {Layer_t::FmOrder2, "FmOrder2"},
    {Layer_t::Add, "Add"},
    {Layer_t::ReduceSum, "ReduceSum"},
    {Layer_t::Softmax, "Softmax"},
    {Layer_t::ElementwiseMultiply, "ElementwiseMultiply"},
    {Layer_t::MultiCross, "MultiCross"},
    {Layer_t::MLP, "MLP"},
    {Layer_t::SequenceMask, "SequenceMask"}};

std::set<Layer_t> TRAINABLE_LAYERS = {
    Layer_t::InnerProduct, Layer_t::MultiCross, Layer_t::WeightMultiply,     Layer_t::BatchNorm,
    Layer_t::LayerNorm,    Layer_t::GRU,        Layer_t::MultiHeadAttention, Layer_t::MLP};

std::map<Embedding_t, std::string> EMBEDDING_TYPE_TO_STRING = {
    {Embedding_t::DistributedSlotSparseEmbeddingHash, "DistributedSlotSparseEmbeddingHash"},
    {Embedding_t::LocalizedSlotSparseEmbeddingHash, "LocalizedSlotSparseEmbeddingHash"}};

std::map<Initializer_t, std::string> INITIALIZER_TYPE_TO_STRING = {
    {Initializer_t::Uniform, "Uniform"},
    {Initializer_t::XavierNorm, "XavierNorm"},
    {Initializer_t::XavierUniform, "XavierUniform"},
    {Initializer_t::Zero, "Zero"}};

std::map<AllReduceAlgo, std::string> ALLREDUCE_ALGO_TO_STRING = {
    {AllReduceAlgo::ONESHOT, "OneShot"}, {AllReduceAlgo::NCCL, "NCCL"}};

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
  DataSourceParams data_source_params;
  AsyncParam async_param;
  DataReaderParams(DataReaderType_t data_reader_type, std::string source, std::string keyset,
                   std::string eval_source, Check_t check_type, int cache_eval_data,
                   long long num_samples, long long eval_num_samples, bool float_label_dense,
                   bool read_file_sequentially, int num_workers,
                   std::vector<long long>& slot_size_array,
                   const DataSourceParams& data_source_params, const AsyncParam& async_param);
  DataReaderParams(DataReaderType_t data_reader_type, std::vector<std::string> source,
                   std::vector<std::string> keyset, std::string eval_source, Check_t check_type,
                   int cache_eval_data, long long num_samples, long long eval_num_samples,
                   bool float_label_dense, bool read_file_sequentially, int num_workers,
                   std::vector<long long>& slot_size_array,
                   const DataSourceParams& data_source_params, const AsyncParam& async_param);
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

  SparseEmbedding(Embedding_t embedding_type, size_t workspace_size_per_gpu_in_mb,
                  size_t embedding_vec_size, const std::string& combiner_str,
                  std::string sparse_embedding_name, std::string bottom_name,
                  std::vector<size_t>& slot_size_array,
                  std::shared_ptr<OptParamsPy>& embedding_opt_params);

  void initialize_max_vocabulary_size_per_gpu();
};

struct DenseLayerComputeConfig {
  bool async_wgrad;
  bool fuse_wb;
  DenseLayerComputeConfig();
  DenseLayerComputeConfig(bool async_wgrad, bool fuse_wb);
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
  size_t projection_dim;
  size_t out_dim;
  int axis;
  int max_sequence_len_from;
  int max_sequence_len_to;
  int num_attention_heads;
  bool transpose_b;
  std::vector<float> target_weight_vec;
  bool use_regularizer;
  Regularizer_t regularizer_type;
  float lambda;
  FcPosition_t pos_type;
  Activation_t act_type;
  std::vector<size_t> num_outputs;
  bool use_bias;
  std::vector<Activation_t> acts;
  std::vector<bool> biases;
  DenseLayerComputeConfig compute_config;

  // reshape layer param
  std::vector<int64_t> reshape_out_dimension;

  // select layer param
  int dim;
  std::vector<int64_t> index;

  DenseLayer(Layer_t layer_type, std::vector<std::string>& bottom_names,
             std::vector<std::string>& top_names, float factor = 1.0, float eps = 0.00001,
             Initializer_t gamma_init_type = Initializer_t::Default,
             Initializer_t beta_init_type = Initializer_t::Default, float dropout_rate = 0.5,
             float elu_alpha = 1.0, size_t num_output = 1,
             Initializer_t weight_init_type = Initializer_t::Default,
             Initializer_t bias_init_type = Initializer_t::Default, int num_layers = 0,
             size_t leading_dim = 0, size_t time_step = 0, size_t batchsize = 1,
             size_t SeqLength = 1, size_t vector_size = 1, bool selected = false,
             std::vector<int> selected_slots = std::vector<int>(),
             std::vector<std::pair<int, int>> ranges = std::vector<std::pair<int, int>>(),
             std::vector<int> indices = std::vector<int>(),
             std::vector<size_t> weight_dims = std::vector<size_t>(), size_t projection_dim = 0,
             size_t out_dim = 0, int axis = 1, int max_sequence_len_from = 1,
             int max_sequence_len_to = 1, int num_attention_heads = 1, bool transpose_b = false,
             std::vector<float> target_weight_vec = std::vector<float>(),
             bool use_regularizer = false, Regularizer_t regularizer_type = Regularizer_t::L1,
             float lambda = 0, FcPosition_t pos_type = FcPosition_t::None,
             Activation_t act_type = Activation_t::Relu,
             std::vector<size_t> num_outputs = std::vector<size_t>(), bool use_bias = true,
             std::vector<Activation_t> acts = std::vector<Activation_t>(),
             std::vector<bool> biases = std::vector<bool>(),
             DenseLayerComputeConfig compute_config = DenseLayerComputeConfig(),
             const std::vector<int64_t>& reshape_out_dimension = {}, int dim = 0,
             const std::vector<int64_t>& index = {});
};

class CopyOp {
 public:
  virtual core23::Tensor get_tensorbag() = 0;

  virtual void run() = 0;
};
class CopyOpImpl final : public CopyOp {
 private:
  std::shared_ptr<GPUResource> gpu_resource_;

  core23::Tensor in_tensor_;
  core23::Tensor out_tensor_;

 public:
  CopyOpImpl(const std::shared_ptr<GPUResource>& gpu_resource, const core23::Tensor& in_tensor)
      : gpu_resource_(gpu_resource), in_tensor_(in_tensor) {
    CudaDeviceContext context(gpu_resource->get_device_id());
    out_tensor_ = core23::Tensor(in_tensor.my_params());
    out_tensor_.data();
  }

  core23::Tensor get_tensorbag() override { return out_tensor_; }

  void run() override {
    CudaDeviceContext context(gpu_resource_->get_device_id());
    auto stream = gpu_resource_->get_stream();
    HCTR_CHECK(in_tensor_.num_bytes() == out_tensor_.num_bytes());
    HCTR_LIB_THROW(cudaMemcpyAsync(out_tensor_.data(), in_tensor_.data(), in_tensor_.num_bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
  }
};

template <typename TypeKey>
void add_input(Input& input, DataReaderParams& reader_params,
               std::map<std::string, SparseInput<TypeKey>>& sparse_input_map,
               std::vector<std::vector<TensorEntity>>& train_tensor_entities_list,
               std::vector<std::vector<TensorEntity>>& evaluate_tensor_entities_list,
               std::shared_ptr<IDataReader>& train_data_reader,
               std::shared_ptr<IDataReader>& evaluate_data_reader, size_t batch_size,
               size_t batch_size_eval, bool use_mixed_precision, bool repeat_dataset,
               bool train_intra_iteration_overlap, size_t num_iterations_statistics,
               const std::shared_ptr<ResourceManager>);

template <typename TypeKey, typename TypeFP>
void add_sparse_embedding(SparseEmbedding& sparse_embedding,
                          std::map<std::string, SparseInput<TypeKey>>& sparse_input_map,
                          std::vector<std::vector<TensorEntity>>& train_tensor_entities_list,
                          std::vector<std::vector<TensorEntity>>& evaluate_tensor_entities_list,
                          std::vector<std::shared_ptr<IEmbedding>>& embeddings,
                          const std::shared_ptr<ResourceManager>& resource_manager,
                          const std::shared_ptr<CollectiveManager>& collective_manager,
                          size_t batch_size, size_t batch_size_eval,
                          OptParams& embedding_opt_params,
                          std::shared_ptr<ExchangeWgrad>& exchange_wgrad, bool use_cuda_graph,
                          bool grouped_all_reduce, size_t num_iterations_statistics,
                          GpuLearningRateSchedulers& gpu_lr_sches);

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

void init_optimizer_params(OptParams& opt_params, const Solver& solver,
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
class Model final {
 public:
  ~Model();
  Model(const Solver& solver, const DataReaderParams& reader_params,
        std::shared_ptr<OptParamsPy>& opt_params);
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  void graph_to_json(std::string graph_config_file);

  void construct_from_json(const std::string& graph_config_file, bool include_dense_network);

  void add(Input& input);

  void add(SparseEmbedding& sparse_embedding);

  void add(DenseLayer& dense_layer);

  void add(const EmbeddingCollectionConfig& ebc_config);

  void graph_analysis();

  void compile();

  void compile(std::vector<std::string>& label_names, std::vector<float>& label_weights);

  void update_label_weights(std::vector<std::string>& label_names,
                            std::vector<float>& label_weights);

  void summary();

  void fit(int num_epochs, int max_iter, int display, int eval_interval, int snapshot,
           std::string snapshot_prefix);

  void set_source(std::string source, std::string eval_source);

  bool train();

  bool eval();

  std::vector<std::pair<std::string, float>> get_eval_metrics();

  Error_t get_current_loss(float* loss);

  Error_t download_params_to_files(std::string prefix, int iter);

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
    for (auto& ebc : ebc_list_) {
      ebc->set_learning_rate(lr);
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
    return !embeddings_.empty() && !embeddings_[0]->get_learning_rate_schedulers().empty();
  }

  void load_dense_weights(const std::string& dense_model_file);
  void load_sparse_weights(const std::vector<std::string>& sparse_embedding_files);
  void load_sparse_weights(const std::map<std::string, std::string>& sparse_embedding_files_maps);
  void load_dense_optimizer_states(const std::string& dense_opt_states_file);
  void load_sparse_optimizer_states(const std::vector<std::string>& sparse_opt_states_files);
  void embedding_load(const std::string& path, const std::vector<std::string>& table_names);
  void embedding_dump(const std::string& path, const std::vector<std::string>& table_names);
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

  std::tuple<size_t, size_t, std::vector<size_t>, int> get_tensor_info_by_name(
      const std::string& tensor_name, Tensor_t tensor_type);

  void check_out_tensor(Tensor_t tensor_type, int index, float* global_result);

 private:
  Solver solver_;
  DataReaderParams reader_params_;
  OptParams opt_params_;
  std::shared_ptr<OptParamsPy> opt_params_py_;
  std::vector<std::shared_ptr<OptParamsPy>> embedding_opt_params_list_;
  std::shared_ptr<LearningRateScheduler> lr_sch_;
  GpuLearningRateSchedulers gpu_lr_sches_;

  std::map<std::string, SparseInput<long long>> sparse_input_map_64_;
  std::map<std::string, SparseInput<unsigned int>> sparse_input_map_32_;

  std::vector<std::vector<TensorEntity>> train_tensor_entities_list_;
  std::vector<std::vector<TensorEntity>> evaluate_tensor_entities_list_;

  std::map<std::string, bool> tensor_active_;

  std::map<std::string, float> label_weights_;

  bool overflow_check_{true};
  bool data_reader_train_status_;
  bool data_reader_eval_status_;
  bool buff_allocated_;
  bool is_dense_trainable_;

  bool set_source_flag_{true};
  bool graph_finalized_{false};
  std::vector<std::pair<std::vector<long long>, std::vector<float>>> inc_sparse_model_;

  std::vector<DenseLayer> dense_layer_params_raw_;
  std::map<std::string, std::vector<int>> tensor_shape_info_raw_;
  std::vector<DenseLayer> dense_layer_params_;
  std::vector<SparseEmbedding> sparse_embedding_params_;
  std::vector<Input> input_params_;
  std::vector<std::string> data_input_info_; /**< data input name */
  std::map<std::string, core23::Shape> tensor_shape_info_;
  std::vector<std::pair<std::string, std::string>>
      input_output_info_;               /**< input output name of each layer. */
  std::vector<std::string> layer_info_; /**< type of each layer. */
  std::vector<std::shared_ptr<Network>> networks_;
  std::vector<std::shared_ptr<IEmbedding>> embeddings_; /**< embedding */

  using TableNameToGlobalIDDict = std::unordered_map<std::string, std::pair<int, int>>;
  TableNameToGlobalIDDict ebc_name_to_global_id_dict_;
  std::map<std::string, int> hotness_map_;
  std::vector<std::unique_ptr<embedding::EmbeddingCollection>> ebc_list_;

  std::vector<core23::Tensor> train_ebc_key_list_;
  std::vector<core23::Tensor> train_ebc_bucket_range_list_;
  std::vector<size_t*> train_ebc_num_keys_list_;
  std::vector<core23::Tensor> evaluate_ebc_key_list_;
  std::vector<core23::Tensor> evaluate_ebc_bucket_range_list_;
  std::vector<size_t*> evaluate_ebc_num_keys_list_;
  std::vector<core23::Tensor> evaluate_ebc_sparse_weight_list_;

  std::vector<DataDistributor::Result> train_ddl_output_;
  std::vector<DataDistributor::Result> cache_train_ddl_output_;
  std::vector<DataDistributor::Result> evaluate_ddl_output_;
  std::vector<DataDistributor::Result> cache_evaluate_ddl_output_;

  std::vector<core23::Tensor> train_ebc_outptut_;
  std::vector<core23::Tensor> evaluate_ebc_outptut_;

  std::shared_ptr<IDataReader>
      train_data_reader_; /**< data reader to reading data from data set to embedding. */
  std::shared_ptr<IDataReader> evaluate_data_reader_; /**< data reader for evaluation. */
  std::shared_ptr<ResourceManager>
      resource_manager_; /**< GPU resources include handles and streams etc.*/
  std::shared_ptr<CollectiveManager> collective_manager_;
  std::shared_ptr<embedding::EmbeddingParameterIO> embedding_para_io_;
  metrics::Metrics metrics_; /**< evaluation metrics. */

  std::shared_ptr<IDataReader> init_data_reader_;
  std::shared_ptr<ExchangeWgrad> exchange_wgrad_;
  bool embedding_dependent_;
  bool high_level_eval_;
  HugeCTR::Timer timer_log;
  std::map<std::string, std::shared_ptr<IEmbedding>> embeddings_map_;
  std::set<std::string> embedding_dependent_tensors_;

  std::shared_ptr<DataDistributor> train_data_distributor_, eval_data_distributor_;

  std::vector<std::shared_ptr<TrainingCallback>> training_callbacks_;

  Error_t download_dense_params_to_files_(std::string weights_file,
                                          std::string dense_opt_states_file);

  Error_t download_sparse_params_to_files_(const std::vector<std::string>& embedding_files,
                                           const std::vector<std::string>& sparse_opt_state_files);

  void init_params_for_dense_();
  void init_params_for_sparse_();
  Error_t load_params_for_dense_(const std::string& model_file);
  Error_t load_params_for_sparse_(const std::vector<std::string>& embedding_file);
  Error_t load_opt_states_for_dense_(const std::string& dense_opt_states_file);
  Error_t load_opt_states_for_sparse_(const std::vector<std::string>& sparse_opt_states_files);
  void exchange_wgrad(size_t device_id);
  void pre_add_dense_layer(DenseLayer& dense_layer);
  void add_dense_layers(std::vector<DenseLayer>& dense_layers);

  void create_networks();
  void build_networks();
  void initialize();
  void create_metrics();
  void create_pipelines();
  std::vector<core23::Tensor> wgrad_tensor_successor_;

  size_t number_of_networks() const;

  struct Graph {
    // train and eval can be called directly by user
    bool is_first_train_batch_ = true;
    bool is_last_train_batch_ = true;
    bool is_first_eval_batch_ = true;
    bool is_last_eval_batch_ = true;
    std::vector<Pipeline> train_pipeline_;
    std::vector<Pipeline> evaluate_pipeline_;

    // cache network input in prefetch
    std::vector<std::shared_ptr<CopyOp>> train_copy_ops_;
    std::vector<std::shared_ptr<CopyOp>> evaluate_copy_ops_;
  };

  Graph graph_;

  void create_copy_ops_for_network_input(const std::string& dense_name,
                                         const std::string& label_name, bool is_train);
  bool is_scheduled_datareader() {
    return (reader_params_.data_reader_type == DataReaderType_t::RawAsync);
  }
  void create_train_network_pipeline(std::vector<std::shared_ptr<Network>>& networks);
  void create_eval_network_pipeline(std::vector<std::shared_ptr<Network>>& networks);
  void create_train_pipeline_with_ebc(std::vector<std::shared_ptr<Network>>& networks);
  void create_evaluate_pipeline_with_ebc(std::vector<std::shared_ptr<Network>>& networks);

  bool skip_prefetch_in_last_batch(bool is_train);
  long long read_a_batch(bool is_train);
  void train_pipeline(size_t current_batch_size);
  void evaluate_pipeline(size_t current_batch_size);
  void train_pipeline_with_ebc();
  void evaluate_pipeline_with_ebc();
};

}  // namespace HugeCTR
