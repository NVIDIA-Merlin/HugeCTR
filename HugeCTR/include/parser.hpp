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
#include <core23/allocator.hpp>
#include <core23/allocator_factory.hpp>
#include <core23/allocator_params.hpp>
#include <core23/buffer.hpp>
#include <core23/buffer_channel_helpers.hpp>
#include <core23/buffer_client.hpp>
#include <core23/buffer_factory.hpp>
#include <core23/buffer_params.hpp>
#include <core23/data_type.hpp>
#include <core23/details/pool_cuda_allocator.hpp>
#include <core23/device.hpp>
#include <core23/device_type.hpp>
#include <core23/offsetted_buffer.hpp>
#include <core23/tensor.hpp>
#include <core23/tensor_params.hpp>
#include <data_readers/data_reader.hpp>
#include <device_map.hpp>
#include <embedding.hpp>
#include <exchange_wgrad.hpp>
#include <fstream>
#include <functional>
#include <gpu_learning_rate_scheduler.hpp>
#include <gpu_resource.hpp>
#include <io/hadoop_filesystem.hpp>
#include <learning_rate_scheduler.hpp>
#include <metrics.hpp>
#include <nlohmann/json.hpp>
#include <training_callback.hpp>

namespace HugeCTR {

// inline to avoid build error: multiple definition
inline nlohmann::json read_json_file(const std::string& filename) {
  nlohmann::json config;
  std::ifstream file_stream(filename);
  if (!file_stream.is_open()) {
    HCTR_OWN_THROW(Error_t::FileCannotOpen, "file_stream.is_open() failed: " + filename);
  }
  file_stream >> config;
  file_stream.close();
  return config;
}

struct Solver {
  std::string model_name;
  unsigned long long seed; /**< seed of data simulator */
  LrPolicy_t lr_policy;    /**< the only fixed lr is supported now. */
  float lr;
  size_t warmup_steps;
  size_t decay_start;
  size_t decay_steps;
  float decay_power;
  float end_lr;
  int max_eval_batches;                /**< the number of batches for evaluations */
  int batchsize_eval;                  /**< batchsize for eval */
  int batchsize;                       /**< batchsize */
  std::vector<std::vector<int>> vvgpu; /**< device map */
  bool repeat_dataset;
  DeviceMap::Layout device_layout;
  bool use_mixed_precision;
  bool enable_tf32_compute;
  float scaler;
  std::map<metrics::Type, float> metrics_spec;
  bool i64_input_key;
  bool use_algorithm_search;
  bool use_cuda_graph;
  bool gen_loss_summary;
  bool train_intra_iteration_overlap;
  bool train_inter_iteration_overlap;
  bool eval_intra_iteration_overlap;
  bool eval_inter_iteration_overlap;
  bool use_embedding_collection;
  AllReduceAlgo all_reduce_algo;
  bool grouped_all_reduce;
  size_t num_iterations_statistics;
  bool perf_logging;
  bool drop_incomplete_batch;
  std::string kafka_brokers;
  DataSourceParams data_source_params;
  std::vector<std::shared_ptr<TrainingCallback>> training_callbacks;
  Solver() {}
};

std::unique_ptr<LearningRateScheduler> get_learning_rate_scheduler(
    const std::string configure_file);

GpuLearningRateSchedulers get_gpu_learning_rate_schedulers(
    const nlohmann::json& config, const std::shared_ptr<ResourceManager>& resource_manager);

#define HAS_KEY_(j_in, key_in)                                               \
  do {                                                                       \
    const nlohmann::json& j__ = (j_in);                                      \
    const std::string& key__ = (key_in);                                     \
    if (j__.find(key__) == j__.end())                                        \
      HCTR_OWN_THROW(Error_t::WrongInput, "[Parser] No Such Key: " + key__); \
  } while (0)

#define CK_SIZE_(j_in, j_size)                                             \
  do {                                                                     \
    const nlohmann::json& j__ = (j_in);                                    \
    if (j__.size() != (j_size))                                            \
      HCTR_OWN_THROW(Error_t::WrongInput, "[Parser] Array size is wrong"); \
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

const std::map<std::string, Layer_t> LAYER_TYPE_MAP = {
    {"Add", Layer_t::Add},
    {"BatchNorm", Layer_t::BatchNorm},
    {"LayerNorm", Layer_t::LayerNorm},
    {"BinaryCrossEntropyLoss", Layer_t::BinaryCrossEntropyLoss},
    {"Cast", Layer_t::Cast},
    {"Concat", Layer_t::Concat},
    {"CrossEntropyLoss", Layer_t::CrossEntropyLoss},
    {"Dropout", Layer_t::Dropout},
    {"ElementwiseMultiply", Layer_t::ElementwiseMultiply},
    {"ELU", Layer_t::ELU},
    {"FmOrder2", Layer_t::FmOrder2},
    {"InnerProduct", Layer_t::InnerProduct},
    {"MLP", Layer_t::MLP},
    {"Interaction", Layer_t::Interaction},
    {"MultiCross", Layer_t::MultiCross},
    {"MultiCrossEntropyLoss", Layer_t::MultiCrossEntropyLoss},
    {"WeightMultiply", Layer_t::WeightMultiply},
    {"ReduceSum", Layer_t::ReduceSum},
    {"Softmax", Layer_t::Softmax},
    {"Gather", Layer_t::Gather},
    {"PReLU_Dice", Layer_t::PReLU_Dice},
    {"GRU", Layer_t::GRU},
    {"MatrixMultiply", Layer_t::MatrixMultiply},
    {"MultiHeadAttention", Layer_t::MultiHeadAttention},
    {"Scale", Layer_t::Scale},
    {"FusedReshapeConcat", Layer_t::FusedReshapeConcat},
    {"FusedReshapeConcatGeneral", Layer_t::FusedReshapeConcatGeneral},
    {"Sub", Layer_t::Sub},
    {"ReduceMean", Layer_t::ReduceMean},
    {"ReLU", Layer_t::ReLU},
    {"Reshape", Layer_t::Reshape},
    {"Sigmoid", Layer_t::Sigmoid},
    {"Slice", Layer_t::Slice},
    {"SequenceMask", Layer_t::SequenceMask}};
const std::map<std::string, Layer_t> LAYER_TYPE_MAP_MP = {
    {"Add", Layer_t::Add},
    {"BatchNorm", Layer_t::BatchNorm},
    {"LayerNorm", Layer_t::LayerNorm},
    {"BinaryCrossEntropyLoss", Layer_t::BinaryCrossEntropyLoss},
    {"Cast", Layer_t::Cast},
    {"Concat", Layer_t::Concat},
    {"CrossEntropyLoss", Layer_t::CrossEntropyLoss},
    {"Dropout", Layer_t::Dropout},
    {"ElementwiseMultiply", Layer_t::ElementwiseMultiply},
    {"ELU", Layer_t::ELU},
    {"FmOrder2", Layer_t::FmOrder2},
    {"MLP", Layer_t::MLP},
    {"InnerProduct", Layer_t::InnerProduct},
    {"Interaction", Layer_t::Interaction},
    {"MultiCross", Layer_t::MultiCross},
    {"MultiCrossEntropyLoss", Layer_t::MultiCrossEntropyLoss},
    {"WeightMultiply", Layer_t::WeightMultiply},
    {"ReduceSum", Layer_t::ReduceSum},
    {"Softmax", Layer_t::Softmax},
    {"ReLU", Layer_t::ReLU},
    {"Reshape", Layer_t::Reshape},
    {"Sigmoid", Layer_t::Sigmoid},
    {"Slice", Layer_t::Slice},
    {"SequenceMask", Layer_t::SequenceMask}};
const std::map<std::string, Embedding_t> EMBEDDING_TYPE_MAP = {
    {"DistributedSlotSparseEmbeddingHash", Embedding_t::DistributedSlotSparseEmbeddingHash},
    {"LocalizedSlotSparseEmbeddingHash", Embedding_t::LocalizedSlotSparseEmbeddingHash}};
const std::map<std::string, Initializer_t> INITIALIZER_TYPE_MAP = {
    {"Uniform", Initializer_t::Uniform},
    {"XavierNorm", Initializer_t::XavierNorm},
    {"XavierUniform", Initializer_t::XavierUniform},
    {"Zero", Initializer_t::Zero}};
static const std::map<std::string, AllReduceAlgo> ALLREDUCE_ALGO_MAP = {
    {"Oneshot", AllReduceAlgo::ONESHOT}, {"NCCL", AllReduceAlgo::NCCL}};

static const std::map<std::string, Optimizer_t> OPTIMIZER_TYPE_MAP = {
    {"Ftrl", Optimizer_t::Ftrl},
    {"Adam", Optimizer_t::Adam},
    {"RMSProp", Optimizer_t::RMSProp},
    {"AdaGrad", Optimizer_t::AdaGrad},
    {"MomentumSGD", Optimizer_t::MomentumSGD},
    {"Nesterov", Optimizer_t::Nesterov},
    {"SGD", Optimizer_t::SGD}};

static const std::map<std::string, Update_t> UPDATE_TYPE_MAP = {
    {"Local", Update_t::Local}, {"Global", Update_t::Global}, {"LazyGlobal", Update_t::LazyGlobal}};

static const std::map<std::string, Regularizer_t> REGULARIZER_TYPE_MAP = {
    {"L1", Regularizer_t::L1},
    {"L2", Regularizer_t::L2},
};

static const std::map<std::string, FcPosition_t> FCPOSITION_TYPE_MAP = {
    {"Head", FcPosition_t::Head},
    {"Body", FcPosition_t::Body},
    {"Tail", FcPosition_t::Tail},
    {"Isolated", FcPosition_t::Isolated},
    {"None", FcPosition_t::None}};

static const std::map<std::string, Activation_t> ACTIVATION_TYPE_MAP = {
    {"Relu", Activation_t::Relu},
    {"None", Activation_t::None},
    {"Unspecified", Activation_t::Unspecified}};

static const std::map<std::string, Alignment_t> ALIGNED_TYPE_MAP = {
    {"Auto", Alignment_t::Auto},
    {"None", Alignment_t::None},
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
inline T get_value_from_json_soft(const nlohmann::json& json, const std::string key,
                                  T default_value) {
  if (has_key_(json, key)) {
    auto value = json.find(key).value();
    CK_SIZE_(value, 1);
    return value.get<T>();
  } else {
    HCTR_LOG_S(INFO, ROOT) << key << " is not specified using default: " << default_value
                           << std::endl;
    return default_value;
  }
}

template <>
inline std::string get_value_from_json_soft(const nlohmann::json& json, const std::string key,
                                            const std::string default_value) {
  if (has_key_(json, key)) {
    auto value = json.find(key).value();
    CK_SIZE_(value, 1);
    return value.get<std::string>();
  } else {
    HCTR_LOG_S(INFO, ROOT) << key << " is not specified using default: " << default_value
                           << std::endl;
    return default_value;
  }
}

OptParams get_optimizer_param(const nlohmann::json& j_optimizer);

inline void analyze_tensor(std::map<std::string, unsigned int>& tensor_usage,
                           std::string bottom_name) {
  if (tensor_usage.find(bottom_name) == tensor_usage.end()) {
    tensor_usage.insert(std::pair<std::string, unsigned int>(bottom_name, 0));
  }
  tensor_usage[bottom_name] += 1;
}

inline void activate_tensor(std::map<std::string, bool>& tensor_active, std::string top_name) {
  if (tensor_active.find(top_name) != tensor_active.end()) {
    HCTR_OWN_THROW(Error_t::WrongInput, top_name + ", top tensor name already exists");
  }
  tensor_active.insert(std::pair<std::string, bool>(top_name, true));
}

inline void deactivate_tensor(std::map<std::string, bool>& tensor_active, std::string bottom_name) {
  if (tensor_active.find(bottom_name) == tensor_active.end()) {
    HCTR_OWN_THROW(Error_t::WrongInput, bottom_name + ", bottom tensor name does not exists");
  }
  if (tensor_active[bottom_name] == false) {
    HCTR_OWN_THROW(Error_t::WrongInput, bottom_name + ", bottom tensor already consumed");
  }
  tensor_active[bottom_name] = false;
}

inline std::vector<std::string> get_layer_names(const nlohmann::json& json) {
  std::vector<std::string> layer_names;
  if (json.is_array()) {
    for (auto j : json) {
      layer_names.push_back(j.get<std::string>());
    }
  } else {
    layer_names.push_back(json.get<std::string>());
  }
  return layer_names;
}

inline void check_graph(std::map<std::string, bool>& tensor_active,
                        const nlohmann::json& j_layers) {
  // activate label, dense and sparse input tensors
  const nlohmann::json& j_data = j_layers[0];
  if (has_key_(get_json(j_data, "label"), "top")) {
    auto label_name_arr = get_json(get_json(j_data, "label"), "top");
    if (label_name_arr.is_array()) {
      for (int i = 0; i < label_name_arr.size(); ++i) {
        auto label_name = label_name_arr[i].get<std::string>();
        activate_tensor(tensor_active, label_name);
      }
    } else {
      auto label_name = get_value_from_json<std::string>(get_json(j_data, "label"), "top");
      activate_tensor(tensor_active, label_name);
    }
  }
  auto dense_name = get_value_from_json<std::string>(get_json(j_data, "dense"), "top");
  activate_tensor(tensor_active, dense_name);
  auto j_sparse = get_json(j_data, "sparse");
  for (unsigned int k = 0; k < j_sparse.size(); k++) {
    const auto sparse_name = get_value_from_json<std::string>(j_sparse[k], "top");
    activate_tensor(tensor_active, sparse_name);
  }
  // deactivate bottom tensors and activate top tensors
  for (unsigned int i = 1; i < j_layers.size(); i++) {
    auto bottom = get_json(j_layers[i], "bottom");
    auto top = get_json(j_layers[i], "top");
    std::vector<std::string> bottom_names = get_layer_names(bottom);
    std::vector<std::string> top_names = get_layer_names(top);
    for (auto& bottom_name : bottom_names) {
      deactivate_tensor(tensor_active, bottom_name);
    }
    for (auto& top_name : top_names) {
      activate_tensor(tensor_active, top_name);
    }
  }
}

inline int get_max_feature_num_per_sample_from_nnz_per_slot(const nlohmann::json& j) {
  int max_feature_num_per_sample = 0;
  auto slot_num = get_value_from_json<int>(j, "slot_num");
  auto nnz_per_slot = get_json(j, "nnz_per_slot");
  if (nnz_per_slot.is_array()) {
    if (nnz_per_slot.size() != static_cast<size_t>(slot_num)) {
      HCTR_OWN_THROW(Error_t::WrongInput, "nnz_per_slot.size() != slot_num");
    }
    for (int slot_id = 0; slot_id < slot_num; ++slot_id) {
      max_feature_num_per_sample += nnz_per_slot[slot_id].get<int>();
    }
  } else {
    int max_nnz = nnz_per_slot.get<int>();
    max_feature_num_per_sample += max_nnz * slot_num;
  }
  return max_feature_num_per_sample;
}

inline int get_max_nnz_from_nnz_per_slot(const nlohmann::json& j) {
  int max_nnz = 0;
  auto slot_num = get_value_from_json<int>(j, "slot_num");
  auto nnz_per_slot = get_json(j, "nnz_per_slot");
  if (nnz_per_slot.is_array()) {
    if (nnz_per_slot.size() != static_cast<size_t>(slot_num)) {
      HCTR_OWN_THROW(Error_t::WrongInput, "nnz_per_slot.size() != slot_num");
    }
    max_nnz = *std::max_element(nnz_per_slot.begin(), nnz_per_slot.end());
  } else {
    max_nnz = nnz_per_slot.get<int>();
  }
  return max_nnz;
}

}  // namespace HugeCTR
