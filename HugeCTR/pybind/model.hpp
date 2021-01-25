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
#include <parser.hpp>
#include <common.hpp>
#include <embedding.hpp>
#include <optimizer.hpp>
#include <loss.hpp>
#include <metrics.hpp>
#include <network.hpp>
#include <string>
#include <thread>
#include <utility>
#include <HugeCTR/include/embeddings/embedding.hpp>
#include <HugeCTR/include/model_oversubscriber/model_oversubscriber.hpp>
#include <HugeCTR/pybind/optimizer.hpp>

namespace HugeCTR {

struct Input {
  DataReaderType_t data_reader_type;
  std::string source;
  std::string eval_source;
  Check_t check_type;
  int cache_eval_data;
  int label_dim;
  std::string label_name;
  int dense_dim;
  std::string dense_name;
  long long num_samples;
  long eval_num_samples;
  bool float_label_dense;
  int num_workers;
  std::vector<long long> slot_size_array;
  std::vector<DataReaderSparseParam> data_reader_sparse_param_array;
  std::vector<std::string> sparse_names;
  Input(DataReaderType_t data_reader_type,
       std::string source,
       std::string eval_source,
       Check_t check_type,
       int cache_eval_data,
       int label_dim,
       std::string label_name,
       int dense_dim,
       std::string dense_name,
       long long num_samples,
       long long eval_num_samples,
       bool float_label_dense,
       int num_workers,
       std::vector<long long>& slot_size_array,
       std::vector<DataReaderSparseParam>& data_reader_sparse_param_array,
       std::vector<std::string>& sparse_names);
};


struct SparseEmbedding {
  Embedding_t embedding_type;
  size_t max_vocabulary_size_per_gpu;
  size_t embedding_vec_size;
  int combiner;
  std::string sparse_embedding_name;
  std::string bottom_name;
  std::vector<size_t> slot_size_array;
  SparseEmbedding(Embedding_t embedding_type,
                 size_t max_vocabulary_size_per_gpu,
                 size_t embedding_vec_size,
                 int combiner,
                 std::string sparse_embedding_name,
                 std::string bottom_name,
                 std::vector<size_t>& slot_size_array);
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
            float factor,
            float eps,
            Initializer_t gamma_init_type,
            Initializer_t beta_init_type,
            float dropout_rate,
            float elu_alpha,
            size_t num_output,
            Initializer_t weight_init_type,
            Initializer_t bias_init_type,
            int num_layers,
            size_t leading_dim,
            bool selected,
            std::vector<int>& selected_slots,
            std::vector<std::pair<int, int>>& ranges,
            std::vector<size_t>& weight_dims,
            size_t out_dim,
            int axis,
            std::vector<float>& target_weight_vec,
            bool use_regularizer,
            Regularizer_t regularizer_type,
            float lambda);
};

template <typename TypeKey>
void add_input(Input& input,
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
            OptParams<TypeFP>& embedding_opt_params);


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
  Model(const SolverParser& solver_parser, std::shared_ptr<OptParamsBase>& opt_params);
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  void add(Input& input);
  
  void add(SparseEmbedding& sparse_embedding);
  
  void add(DenseLayer& dense_layer);
  
  void compile();

  void summary();

  void fit();

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

  Error_t set_learning_rate(float lr) {
    for (auto& embedding : embeddings_) {
      embedding->set_learning_rate(lr);
    }
    for (auto& network : networks_) {
      network->set_learning_rate(lr);
    }
    return Error_t::Success;
  }

  const std::shared_ptr<ModelOversubscriber>& get_model_oversubscriber() const {
    return model_oversubscriber_;
  }

  long long get_params_num() {
    long long size = 0;
    for (auto& embedding : embeddings_) {
      size += embedding->get_params_num();
    }
    return static_cast<long long>(networks_[0]->get_params_num()) + size;
  }

  const std::shared_ptr<IDataReader>& get_train_data_reader() const { return train_data_reader_; }
  const std::shared_ptr<IDataReader>& get_evaluate_data_reader() const { return evaluate_data_reader_; }

 private:
  SolverParser solver_;
  OptParams<float> opt_params_32_;
  OptParams<__half> opt_params_16_;
  std::shared_ptr<LearningRateScheduler> lr_sch_;
  std::map<std::string, SparseInput<long long>> sparse_input_map_64_;
  std::map<std::string, SparseInput<unsigned int>> sparse_input_map_32_;
  std::vector<std::vector<TensorEntry>> train_tensor_entries_list_;
  std::vector<std::vector<TensorEntry>> evaluate_tensor_entries_list_;
  
  std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> blobs_buff_list_;
  std::vector<std::shared_ptr<BufferBlock2<float>>> train_weight_buff_list_;
  std::vector<std::shared_ptr<BufferBlock2<__half>>> train_weight_buff_half_list_;
  std::vector<std::shared_ptr<BufferBlock2<float>>> wgrad_buff_list_;
  std::vector<std::shared_ptr<BufferBlock2<__half>>> wgrad_buff_half_list_;
  std::vector<std::shared_ptr<BufferBlock2<float>>> evaluate_weight_buff_list_;
  std::vector<std::shared_ptr<BufferBlock2<__half>>> evaluate_weight_buff_half_list_;
  std::vector<std::shared_ptr<BufferBlock2<float>>> wgrad_buff_placeholder_list_;
  std::vector<std::shared_ptr<BufferBlock2<__half>>> wgrad_buff_half_placeholder_list_;

  std::vector<std::pair<std::string, std::string>> input_output_info_;   /**< input output name of each layer. */
  std::vector<std::string> layer_info_;   /**< type of each layer. */
  std::vector<std::shared_ptr<Network>> networks_;      /**< networks (dense) used in training. */
  std::vector<std::shared_ptr<IEmbedding>> embeddings_; /**< embedding */
  std::shared_ptr<ModelOversubscriber> model_oversubscriber_; /**< model oversubscriber for model oversubscribing. */

  std::shared_ptr<IDataReader> train_data_reader_; /**< data reader to reading data from data set to embedding. */
  std::shared_ptr<IDataReader> evaluate_data_reader_; /**< data reader for evaluation. */
  std::shared_ptr<ResourceManager> resource_manager_; /**< GPU resources include handles and streams etc.*/
  metrics::Metrics metrics_; /**< evaluation metrics. */
  
  Error_t download_params_to_files_(std::string weights_file,
                                    const std::vector<std::string>& embedding_files);

  template <typename TypeEmbeddingComp>
  std::shared_ptr<ModelOversubscriber> create_model_oversubscriber_();

  Error_t init_or_load_params_for_dense_(const std::string& model_file);

  Error_t init_or_load_params_for_sparse_(const std::vector<std::string>& embedding_file);
};

} // namespace HugeCTR