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
#include <common.hpp>
#include <embedding.hpp>
#include <metrics.hpp>
#include <network.hpp>
#include <parser.hpp>
#include <string>
#include <thread>
#include <utility>

#include "HugeCTR/include/embeddings/embedding.hpp"
#include <exchange_wgrad.hpp>

namespace HugeCTR {

/**
 * @brief Main HugeCTR class
 *
 * This is a class supporting basic usages of hugectr, which includes
 * train; evaluation; get loss; load and download trained parameters.
 * To learn how to use those method, please refer to main.cpp.
 */
class Session {
 public:
  /**
   * Dtor of SessionImpl.
   */
  ~Session();
  Session(const SolverParser& solver_config, const std::string& config_file);
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  Error_t initialize();
  /**
   * The all in one training method.
   * This method processes one iteration of a training, including one forward, one backward and
   * parameter update
   */
  bool train();
  /**
   * The all in one evaluation method.
   * This method processes one forward of evaluation.
   */
  bool eval(int eval_batch = -1);

  std::vector<std::pair<std::string, float>> get_eval_metrics();

  void start_data_reading() {
    train_data_reader_->start();
    evaluate_data_reader_->start();
  }

  /**
   * Get current loss from the loss tensor.
   * @return loss in float
   */
  Error_t get_current_loss(float* loss);
  /**
   * Download trained parameters to file.
   * @param weights_file file name of output dense model
   * @param embedding_file file name of output sparse model
   */
  Error_t download_params_to_files(std::string prefix, int iter);

  /**
   * export predictions to file
   * @param output_file_name file name of destination filename
   */
  Error_t export_predictions(const std::string& output_prediction_file_name, const std::string& output_label_file_name);

  /**
   * Set learning rate while training
   * @param lr learning rate.
   */
  Error_t set_learning_rate(float lr) {
    for (auto& embedding : embeddings_) {
      embedding->set_learning_rate(lr);
    }
    for (auto& network : networks_) {
      network->set_learning_rate(lr);
    }
    return Error_t::Success;
  }

  /**
   * generate a dense model and initilize with small random values.
   * @param model_file dense model initilized
   */
  Error_t init_params(std::string model_file) { return Error_t::Success; };
  /**
   * get the number of parameters (reserved for debug)
   */
  long long get_params_num() {
    long long size = 0;
    for (auto& embedding : embeddings_) {
      size += embedding->get_params_num();
    }
    return static_cast<long long>(networks_[0]->get_params_num()) + size;
  }

  void check_overflow() const;

  const std::shared_ptr<IDataReader>& get_train_data_reader() const { return train_data_reader_; }
  const std::shared_ptr<IDataReader>& get_evaluate_data_reader() const {
    return evaluate_data_reader_;
  }

  void copy_weights_for_evaluation();

  bool use_gpu_learning_rate_scheduling() const {
    return !embeddings_[0]->get_learning_rate_schedulers().empty();
  }

 private:
  std::vector<std::shared_ptr<Network>> networks_;      /**< networks (dense) used in training. */
  std::vector<std::shared_ptr<IEmbedding>> embeddings_; /**< embedding */
  std::shared_ptr<IDataReader> init_data_reader_;
  std::shared_ptr<IDataReader>
      train_data_reader_; /**< data reader to reading data from data set to embedding. */
  std::shared_ptr<IDataReader> evaluate_data_reader_; /**< data reader for evaluation. */
  std::shared_ptr<ResourceManager>
      resource_manager_; /**< GPU resources include handles and streams etc.*/
  std::shared_ptr<Parser> parser_;
  std::shared_ptr<ExchangeWgrad> exchange_wgrad_;
  Error_t download_params_to_files_(std::string weights_file,
                                    std::string dense_opt_states_file,
                                    const std::vector<std::string>& embedding_files,
                                    const std::vector<std::string>& sparse_opt_state_files);

  metrics::Metrics metrics_;
  SolverParser solver_config_;

  struct HolisticCudaGraph {
    std::vector<bool> initialized;
    std::vector<cudaGraphExec_t> instance;
    std::vector<cudaEvent_t> fork_event;
  } train_graph_;


  // TODO(MLPERF): these two variables for export_predictions.
  // There may be a better place for them.
  bool use_mixed_precision_;
  size_t batchsize_eval_;

  /**
   * A method load trained parameters for dense model.
   * @param model_file dense model generated by training
   */
  Error_t init_or_load_params_for_dense_(const std::string& model_file);

  /**
   * A method load optimizer states for dense model.
   * @param dense_opt_states_file optimizer states generated in training 
   */
  Error_t load_opt_states_for_dense_(const std::string& dense_opt_states_file);

  /**
   * A method initialize or load trained parameters for sparse model.
   * @param embedding_model_file sparse model generated by training
   */
  Error_t init_or_load_params_for_sparse_(const std::vector<std::string>& embedding_file);

  Error_t load_opt_states_for_sparse_(const std::vector<std::string>& sparse_opt_states_files);
  void exchange_wgrad(size_t device_id);
  void train_overlapped();
};

}  // namespace HugeCTR
