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
#include <fstream>
#include <functional>
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/data_reader.hpp"
#include "HugeCTR/include/device_map.hpp"
#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/gpu_resource.hpp"
#include "HugeCTR/include/metrics.hpp"
#include "HugeCTR/include/network.hpp"
#include "nlohmann/json.hpp"

namespace HugeCTR {

/**
 * @brief The parser of configure file (in json format).
 *
 * The builder of each layer / optimizer in HugeCTR.
 * Please see User Guide to learn how to write a configure file.
 * @verbatim
 * Some Restrictions:
 *  1. Embedding should be the first element of layers.
 *  2. layers should be listed from bottom to top.
 * @endverbatim
 */
class Parser {
 private:

  nlohmann::json config_;  /**< configure file. */
  size_t batch_size_;      /**< batch size. */
  size_t batch_size_eval_; /**< batch size. */
  const bool use_mixed_precision_{false};
  const float scaler_{1.f};

 public:
  /**
   * Ctor.
   * Ctor only verify the configure file, doesn't create pipeline.
   */

  Parser(const std::string& configure_file, size_t batch_size, size_t batch_size_eval,
         bool use_mixed_precision = false, float scaler = 1.0f)
      : batch_size_(batch_size),
        batch_size_eval_(batch_size_eval),
        use_mixed_precision_(use_mixed_precision),
        scaler_(scaler) {
    try {
      std::ifstream file(configure_file);
      if (!file.is_open()) {
        CK_THROW_(Error_t::FileCannotOpen, "file.is_open() failed: " + configure_file);
      }
      file >> config_;
      file.close();
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
    return;
  }
  typedef long long TYPE_1;
  typedef unsigned int TYPE_2;

  /**
   * Create the pipeline, which includes data reader, embedding.
   */
  void create_pipeline(std::unique_ptr<DataReader<TYPE_1>>& data_reader,
                       std::unique_ptr<DataReader<TYPE_1>>& data_reader_eval,
                       std::vector<std::unique_ptr<IEmbedding>>& embedding,
                       std::vector<std::unique_ptr<IEmbedding>>& embedding_eval,
                       std::vector<std::unique_ptr<Network>>& network,
                       std::vector<std::unique_ptr<Network>>& network_eval,
                       const std::shared_ptr<GPUResourceGroup>& gpu_resource_group);

  /**
   * Create the pipeline, which includes data reader, embedding.
   */
  void create_pipeline(std::unique_ptr<DataReader<TYPE_2>>& data_reader,
                       std::unique_ptr<DataReader<TYPE_2>>& data_reader_eval,
                       std::vector<std::unique_ptr<IEmbedding>>& embedding,
                       std::vector<std::unique_ptr<IEmbedding>>& embedding_eval,
                       std::vector<std::unique_ptr<Network>>& network,
                       std::vector<std::unique_ptr<Network>>& network_eval,
                       const std::shared_ptr<GPUResourceGroup>& gpu_resource_group);
};

std::unique_ptr<LearningRateScheduler> get_learning_rate_scheduler(
    const std::string configure_file);

/**
 * Solver Parser.
 * This class is designed to parse the solver clause of the configure file.
 */
struct SolverParser {
  std::string configure_file;
  unsigned int seed;                           /**< seed of data simulator */
  LrPolicy_t lr_policy;                        /**< the only fixed lr is supported now. */
  int display;                                 /**< the interval of loss display. */
  int max_iter;                                /**< the number of iterations for training */
  int snapshot;                                /**< the number of iterations for a snapshot */
  std::string snapshot_prefix;                 /**< naming prefix of snapshot file */
  int eval_interval;                           /**< the interval of evaluations */
  int eval_batches;                            /**< the number of batches for evaluations */
  int batchsize_eval;                          /**< batchsize for eval */
  int batchsize;                               /**< batchsize */
  std::string model_file;                      /**< name of model file */
  std::vector<std::string> embedding_files;    /**< name of embedding file */
  std::vector<int> device_list;                /**< device_list */
  std::shared_ptr<const DeviceMap> device_map; /**< device map */
  bool use_mixed_precision;
  float scaler;
  std::map<metrics::Type, float> metrics_spec;
  bool i64_input_key;
  SolverParser(const std::string& file);
};

template <typename T>
struct SparseInput {
  Tensors<T> row;
  Tensors<T> value;
  Tensors<T> row_eval;
  Tensors<T> value_eval;
  size_t slot_num;
  size_t max_feature_num_per_sample;
  SparseInput(int slot_num_in, int max_feature_num_per_sample_in)
      : slot_num(slot_num_in), max_feature_num_per_sample(max_feature_num_per_sample_in) {}
  SparseInput() {}
};


#define HAS_KEY_(j_in, key_in)                                          \
  do {                                                                  \
    const nlohmann::json& j__ = (j_in);                                 \
    const std::string& key__ = (key_in);                                \
    if (j__.find(key__) == j__.end())                                   \
      CK_THROW_(Error_t::WrongInput, "[Parser] No Such Key: " + key__); \
  } while (0)

#define CK_SIZE_(j_in, j_size)                                                                  \
  do {                                                                                          \
    const nlohmann::json& j__ = (j_in);                                                         \
    if (j__.size() != (j_size)) CK_THROW_(Error_t::WrongInput, "[Parser] Array size is wrong"); \
  } while (0)

#define FIND_AND_ASSIGN_INT_KEY(out, json)      \
  do {                                          \
    out = 0;                                    \
    if (json.find(#out) != json.end()) {        \
      out = json.find(#out).value().get<int>(); \
    }                                           \
  } while (0)

#define FIND_AND_ASSIGN_STRING_KEY(out, json)           \
  do {                                                  \
    out.clear();                                        \
    if (json.find(#out) != json.end()) {                \
      out = json.find(#out).value().get<std::string>(); \
    }                                                   \
  } while (0)

static const std::map<std::string, Optimizer_t> OPTIMIZER_TYPE_MAP = {
    {"Adam", Optimizer_t::Adam},
    {"MomentumSGD", Optimizer_t::MomentumSGD},
    {"Nesterov", Optimizer_t::Nesterov},
    {"SGD", Optimizer_t::SGD}};

static const std::map<std::string, Regularizer_t> REGULARIZER_TYPE_MAP = {
    {"L1", Regularizer_t::L1},
    {"L2", Regularizer_t::L2},
};

inline bool has_key_(const nlohmann::json& j_in, const std::string& key_in) {
  if (j_in.find(key_in) == j_in.end()) {
    return false;
  } else {
    return true;
  }
}

inline const nlohmann::json& get_json(const nlohmann::json& json, const std::string key) {
  HAS_KEY_(json, key);
  return json.find(key).value();
}

template <typename T>
inline T get_value_from_json(const nlohmann::json& json, const std::string key) {
  HAS_KEY_(json, key);
  auto value = json.find(key).value();
  CK_SIZE_(value, 1);
  return value.get<T>();
}

template <typename T>
inline T get_value_from_json_soft(const nlohmann::json& json, const std::string key, T B) {
  if (has_key_(json, key)) {
    auto value = json.find(key).value();
    CK_SIZE_(value, 1);
    return value.get<T>();
  } else {
    MESSAGE_(key + " is not specified using default: " + std::to_string(B));
    return B;
  }
}


void parse_data_layer_helper(const nlohmann::json& j, int& label_dim, int& dense_dim,
                             Check_t& check_type, std::string& source_data,
                             std::vector<DataReaderSparseParam>& data_reader_sparse_param_array,
                             std::string& eval_source, std::string& top_strs_label,
                             std::string& top_strs_dense, std::vector<std::string>& sparse_names,
                             std::map<std::string, SparseInput<long long>>& sparse_input_map);


}  // namespace HugeCTR
