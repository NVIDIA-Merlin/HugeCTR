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


#include "HugeCTR/include/parser.hpp"
#include "HugeCTR/include/device_map.hpp"
#include "HugeCTR/include/layer.hpp"
#include "HugeCTR/include/layers/batch_norm_layer.hpp"
#include "HugeCTR/include/layers/concat_layer.hpp"
#include "HugeCTR/include/layers/elu_layer.hpp"
#include "HugeCTR/include/layers/fully_connected_layer.hpp"
#include "HugeCTR/include/layers/relu_layer.hpp"
#include "HugeCTR/include/loss.hpp"
#include "HugeCTR/include/optimizers/adam_optimizer.hpp"
#include "HugeCTR/include/optimizers/momentum_sgd.hpp"
#include "HugeCTR/include/optimizers/nesterov_optimizer.hpp"

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {

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
    {"Nesterov", Optimizer_t::Nesterov}};

bool has_key_(const nlohmann::json& j_in, const std::string& key_in) {
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

void assign_first_tensor(std::map<std::string, Tensor<float>*>& tensor_list,
                         const nlohmann::json& j_array, Tensor<float>& in_tensor) {
  // get the top of embedding layer
  auto tensor_name = get_value_from_json<std::string>(j_array[0], "top");
  auto p = tensor_list.insert(std::pair<std::string, Tensor<float>*>(tensor_name, &in_tensor));
  if (p.second == false) {
    CK_THROW_(Error_t::WrongInput, "Tensor insert failed");
  }
}

struct InputOutputInfo {
  Tensor<float>* input;
  std::string output;
};

InputOutputInfo get_input_tensor_and_output_name(
    const nlohmann::json& json, std::map<std::string, Tensor<float>*> tensor_list) {
  auto bottom_str = get_value_from_json<std::string>(json, "bottom");
  auto top_str = get_value_from_json<std::string>(json, "top");

  if (top_str == bottom_str) {
    CK_THROW_(Error_t::WrongInput, "top.get<std::string>() == bottom.get<std::string>()");
  }

  Tensor<float>* tensor_ptr;
  if (!find_item_in_map(&tensor_ptr, bottom_str, tensor_list)) {
    CK_THROW_(Error_t::WrongInput, "No such bottom: " + bottom_str);
  }

  InputOutputInfo input_output_info = {.input = tensor_ptr, .output = top_str};

  return input_output_info;
}

struct TensorPair {
  Tensor<float>* tensor;
  std::string name;
};

void add_tensor_to_network(TensorPair output_tensor_pair,
                           std::map<std::string, Tensor<float>*>& tensor_list,
                           std::vector<Tensor<float>*>& tensors) {
  auto p = tensor_list.insert(
      std::pair<std::string, Tensor<float>*>(output_tensor_pair.name, output_tensor_pair.tensor));

  if (p.second == false) {
    CK_THROW_(Error_t::WrongInput, "Tensor insert failed");
  }
  tensors.push_back(output_tensor_pair.tensor);
}

OptParams get_optimizer_param(const nlohmann::json& j_optimizer) {
  // create optimizer
  auto optimizer_name = get_value_from_json<std::string>(j_optimizer, "type");
  Optimizer_t optimizer_type;
  if (!find_item_in_map(&optimizer_type, optimizer_name, OPTIMIZER_TYPE_MAP)) {
    CK_THROW_(Error_t::WrongInput, "No such optimizer: " + optimizer_name);
  }

  OptHyperParams opt_hyper_params;
  memset(&opt_hyper_params, 0, sizeof(opt_hyper_params));
  OptParams opt_params;

  switch (optimizer_type) {
    case Optimizer_t::Adam: {
      auto j_hparam = get_json(j_optimizer, "adam_hparam");
      auto alpha = get_value_from_json<float>(j_hparam, "alpha");
      auto beta1 = get_value_from_json<float>(j_hparam, "beta1");
      auto beta2 = get_value_from_json<float>(j_hparam, "beta2");
      auto epsilon = get_value_from_json<float>(j_hparam, "epsilon");
      opt_hyper_params.adam.beta1 = beta1;
      opt_hyper_params.adam.beta2 = beta2;
      opt_hyper_params.adam.epsilon = epsilon;
      opt_params = {0, alpha, opt_hyper_params};
      break;
    }
    case Optimizer_t::MomentumSGD: {
      auto j_hparam = get_json(j_optimizer, "momentum_sgd_hparam");
      auto learning_rate = get_value_from_json<float>(j_hparam, "learning_rate");
      auto momentum_factor = get_value_from_json<float>(j_hparam, "momentum_factor");
      opt_hyper_params.momentum.factor = momentum_factor;
      opt_params = {1, learning_rate, opt_hyper_params};
      break;
    }
    case Optimizer_t::Nesterov: {
      auto j_hparam = get_json(j_optimizer, "nesterov_hparam");
      auto learning_rate = get_value_from_json<float>(j_hparam, "learning_rate");
      auto momentum_factor = get_value_from_json<float>(j_hparam, "momentum_factor");
      opt_hyper_params.nesterov.mu = momentum_factor;
      opt_params = {2, learning_rate, opt_hyper_params};
      break;
    }
    default:
      assert(!"Error: no such optimizer && should never get here!");
  }
  return opt_params;
}
/*
 * Create single network
 *
 */
Network* create_network(const nlohmann::json& j_array, const nlohmann::json& j_optimizer,
                        Tensor<float>& in_tensor, const Tensor<float>& label_tensor, int batch_size,
                        int device_id, const GPUResource* gpu_resource) {
  const std::map<std::string, Layer_t> LAYER_TYPE_MAP = {
      {"BatchNorm", Layer_t::BatchNorm},
      {"BinaryCrossEntropyLoss", Layer_t::BinaryCrossEntropyLoss},
      {"Concat", Layer_t::Concat},
      {"CrossEntropyLoss", Layer_t::CrossEntropyLoss},
      {"ELU", Layer_t::ELU},
      {"InnerProduct", Layer_t::InnerProduct},
      {"MultiCrossEntropyLoss", Layer_t::MultiCrossEntropyLoss},
      {"ReLU", Layer_t::ReLU},
  };

  Network* network =
      new Network(in_tensor, label_tensor, batch_size, device_id, gpu_resource, false);
  std::map<std::string, Tensor<float>*> tensor_list;
  tensor_list.clear();

  assign_first_tensor(tensor_list, j_array, in_tensor);

  std::vector<Tensor<float>*>& tensors = network->tensors_;
  Tensor<float>*& forward_temp_tensors = network->forward_temp_tensors_;
  Tensor<float>*& backward_temp_tensors = network->backward_temp_tensors_;
  int& is_speedup = network->is_speedup_;
  std::vector<Layer*>& layers = network->layers_;
  GeneralBuffer<float>& blobs_buff = network->blobs_buff_;
  GeneralBuffer<float>& weight_buff = network->weight_buff_;
  GeneralBuffer<float>& wgrad_buff = network->wgrad_buff_;
  Tensor<float>*& loss_tensor = network->loss_tensor_;
  Loss*& loss = network->loss_;

  assert(tensors.empty());
  assert(layers.empty());

  for (unsigned int i = 1; i < j_array.size(); i++) {
    const nlohmann::json& j = j_array[i];
    const auto layer_type_name = get_value_from_json<std::string>(j, "type");

    Layer_t layer_type;
    if (!find_item_in_map(&layer_type, layer_type_name, LAYER_TYPE_MAP)) {
      CK_THROW_(Error_t::WrongInput, "No such layer: " + layer_type_name);
    }
    auto input_output_info = get_input_tensor_and_output_name(j, tensor_list);
    TensorPair output_tensor_pair;
    output_tensor_pair.name = input_output_info.output;

    switch (layer_type) {
      case Layer_t::BatchNorm: {
        auto bn_in_tensor = input_output_info.input;
        // establish out tensor
        std::vector<int> tmp_dim;
        Tensor<float>* bn_out_tensor = new Tensor<float>(
            tmp_dim = {batch_size, (bn_in_tensor->get_dims())[1]}, blobs_buff, TensorFormat_t::HW);
        output_tensor_pair.tensor = bn_out_tensor;

        // get BN params
        auto j_bn_hparam = get_json(j, "bn_param");
        auto is_training = get_value_from_json<bool>(j_bn_hparam, "is_training");
        auto factor = get_value_from_json<float>(j_bn_hparam, "factor");
        auto eps = get_value_from_json<float>(j_bn_hparam, "eps");

        BatchNormLayer::Params params = {is_training, factor, eps};
        layers.push_back(new BatchNormLayer(weight_buff, wgrad_buff, *bn_in_tensor, *bn_out_tensor,
                                            params, *(gpu_resource->get_cudnn_handle_ptr()),
                                            device_id));
        break;
      }
      case Layer_t::BinaryCrossEntropyLoss: {
        auto binary_cross_entropy_loss_in_tensor = input_output_info.input;
        std::vector<int> tmp_dim;
        loss_tensor = new Tensor<float>(tmp_dim = {1, 1}, blobs_buff, TensorFormat_t::HW);
        loss = new BinaryCrossEntropyLoss(const_cast<Tensor<float>&>(label_tensor),
                                          *binary_cross_entropy_loss_in_tensor, *loss_tensor,
                                          device_id);
        break;
      }
      case Layer_t::Concat: {
        auto in_tensor = input_output_info.input;
        std::vector<int> slot_mask;
        auto selected_it = j.find("selected");
        if (selected_it != j.end()) {
          nlohmann::json selected = (selected_it.value());
          for (auto slot_obj : selected) {
            int slot_id = slot_obj.get<int>();
            if (slot_id < 0) CK_THROW_(Error_t::WrongInput, "slot_id < 0");
            slot_mask.push_back(slot_id);
          }
        }

        // establish out tensor
        std::vector<int> in_dims = in_tensor->get_dims();
        int n_batch = in_dims[0];
        int n_slot = in_dims[1];
        int vector_length = in_dims[2];
        int n_active_slot = slot_mask.empty() ? n_slot : int(slot_mask.size());
        std::vector<int> out_dims = {n_batch, n_active_slot * vector_length};
        TensorFormat_t out_format = TensorFormat_t::HW;
        Tensor<float>* out_tensor = slot_mask.empty()
                                        ? new Tensor<float>(out_dims, *in_tensor, out_format)
                                        : new Tensor<float>(out_dims, blobs_buff, out_format);
        output_tensor_pair.tensor = out_tensor;
        layers.push_back(new ConcatLayer(*in_tensor, *out_tensor, slot_mask, device_id));

        break;
      }
      case Layer_t::CrossEntropyLoss: {
        auto cross_entropy_loss_in_tensor = input_output_info.input;
        std::vector<int> tmp_dim;
        loss_tensor = new Tensor<float>(tmp_dim = {1, 1}, blobs_buff, TensorFormat_t::HW);
        loss = new CrossEntropyLoss(const_cast<Tensor<float>&>(label_tensor),
                                    *cross_entropy_loss_in_tensor, *loss_tensor, device_id);
        break;
      }
      case Layer_t::ELU: {
        auto elu_in_tensor = input_output_info.input;

        // establish out tensor
        std::vector<int> tmp_dim;
        Tensor<float>* elu_out_tensor = new Tensor<float>(
            tmp_dim = {batch_size, (elu_in_tensor->get_dims())[1]}, blobs_buff, TensorFormat_t::HW);
        output_tensor_pair.tensor = elu_out_tensor;
        // get ELU params
        auto j_elu_hparam = get_json(j, "elu_param");
        auto alpha = get_value_from_json<float>(j_elu_hparam, "alpha");
        layers.push_back(new EluLayer(*elu_in_tensor, *elu_out_tensor, alpha, device_id));

        break;
      }
      case Layer_t::InnerProduct: {
        auto fc_in_tensor = input_output_info.input;
        // establish out tensor
        auto j_fc_param = get_json(j, "fc_param");
        auto output = get_value_from_json<int>(j_fc_param, "num_output");
        std::vector<int> tmp_dim;
        int speedup = 0;
        if (has_key_(j, "speedup")) {
          speedup = get_value_from_json<int>(j, "speedup");
        }
        Tensor<float>* out_tensor;
        if (speedup == 1) {
          is_speedup = 1;
          std::cout << "is_speedup: " << is_speedup << std::endl;
          std::vector<int> in_dims = fc_in_tensor->get_dims();
          int bs = in_dims[0]; // batch_size = BS
          out_tensor = new Tensor<float>(
              tmp_dim = {bs, output}, blobs_buff, TensorFormat_t::HW);
          backward_temp_tensors = out_tensor;
        } else {
          out_tensor = new Tensor<float>(
              tmp_dim = {batch_size, output}, blobs_buff, TensorFormat_t::HW);  // batch_size = BS/N
        }
        output_tensor_pair.tensor = out_tensor;
        // establish layer
        Layer* fc_layer = new FullyConnectedLayer(
            weight_buff, wgrad_buff, *fc_in_tensor, *out_tensor, TensorFormat_t::HW,
            *(gpu_resource->get_cublas_handle_ptr()), device_id);
        layers.push_back(fc_layer);
        break;
      }
      case Layer_t::MultiCrossEntropyLoss: {
        auto multi_cross_entropy_loss_in_tensor = input_output_info.input;
        std::vector<int> tmp_dim;
        loss_tensor = new Tensor<float>(tmp_dim = {1, 1}, blobs_buff, TensorFormat_t::HW);

        auto tweight = get_json(j, "target_weight");
        std::vector<float> target_weight_vec;
        for (auto tweight_tmp : tweight) {
          float tweight_val = tweight_tmp.get<float>();
          target_weight_vec.push_back(tweight_val);
        }
        loss = new MultiCrossEntropyLoss(const_cast<Tensor<float>&>(label_tensor),
                                         *multi_cross_entropy_loss_in_tensor, *loss_tensor,
                                         target_weight_vec, device_id);
        break;
      }
      case Layer_t::ReLU: {
        Tensor<float>* relu_in_tensor;
        std::vector<int> tmp_dim;
        int speedup = 0;
        if (has_key_(j, "speedup")) {
          speedup = get_value_from_json<int>(j, "speedup");
        }
        if (speedup == 1) {
            auto relu_in_tensor_old = input_output_info.input;
            relu_in_tensor = new Tensor<float>(
                tmp_dim = {batch_size, (relu_in_tensor_old->get_dims())[1]}, blobs_buff, TensorFormat_t::HW);
            forward_temp_tensors = relu_in_tensor;
        } else {
          relu_in_tensor = input_output_info.input;
        }
        // establish out tensor
        Tensor<float>* relu_out_tensor =
            new Tensor<float>(tmp_dim = {batch_size, (relu_in_tensor->get_dims())[1]}, blobs_buff,
                              TensorFormat_t::HW);
        output_tensor_pair.tensor = relu_out_tensor;
        layers.push_back(new ReluLayer(*relu_in_tensor, *relu_out_tensor, device_id));

        break;
      }
      default:
        assert(!"Error: no such layer && should never get here!");
    }  // end of switch

    if (!(layer_type == Layer_t::CrossEntropyLoss ||
          layer_type == Layer_t::BinaryCrossEntropyLoss ||
          layer_type == Layer_t::MultiCrossEntropyLoss)) {
      add_tensor_to_network(output_tensor_pair, tensor_list, tensors);
    }
  }

  // create optimizer
  auto opt_param = get_optimizer_param(j_optimizer);

  switch (static_cast<Optimizer_t>(opt_param.optimizer)) {
    case Optimizer_t::Adam: {
      auto alpha = opt_param.lr;
      auto beta1 = opt_param.hyperparams.adam.beta1;
      auto beta2 = opt_param.hyperparams.adam.beta2;
      auto epsilon = opt_param.hyperparams.adam.epsilon;
      network->optimizer_ =
          new AdamOptimizer(weight_buff, wgrad_buff, device_id, alpha, beta1, beta2, epsilon);
      break;
    }
    case Optimizer_t::MomentumSGD: {
      auto learning_rate = opt_param.lr;
      auto momentum_factor = opt_param.hyperparams.momentum.factor;
      network->optimizer_ =
          new MomentumSGD(weight_buff, wgrad_buff, device_id, learning_rate, momentum_factor);
      break;
    }
    case Optimizer_t::Nesterov: {
      auto learning_rate = opt_param.lr;
      auto momentum_factor = opt_param.hyperparams.nesterov.mu;
      network->optimizer_ =
          new NesterovOptimizer(weight_buff, wgrad_buff, device_id, learning_rate, momentum_factor);
      break;
    }
    default:
      assert(!"Error: no such optimizer && should never get here!");
  }
  weight_buff.init(device_id);
  wgrad_buff.init(device_id);
  blobs_buff.init(device_id);

  return network;
}

template <typename TypeKey>
static void create_pipeline_internal(DataReader<TypeKey>** data_reader,
                                     Embedding<TypeKey>** embedding, std::vector<Network*>* network,
                                     GPUResourceGroup& gpu_resource_group, nlohmann::json config,
                                     int batch_size) {
  try {
    int num_procs = 1, pid = 0;
#ifdef ENABLE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif

    const std::map<std::string, Embedding_t> EMBEDDING_TYPE_MAP = {
        {"SparseEmbeddingHash", Embedding_t::SparseEmbeddingHash}};
    int max_feature_num_per_sample;
    {
      // Create Data Reader
      auto j = get_json(config, "data");
      auto j_source = get_json(j, "source");
      std::string source_data;
      if (j_source.is_array()) {
        int num_nodes = j_source.size();
        if (num_nodes != num_procs) {
          CK_THROW_(Error_t::WrongInput, "num_nodes != num_procs");
        }
        source_data = j_source[pid].get<std::string>();
      } else {
        if (num_procs > 1) {
          CK_THROW_(Error_t::WrongInput, "num_procs > 1");
        }
        source_data = get_value_from_json<std::string>(j, "source");
      }
      max_feature_num_per_sample = get_value_from_json<int>(j, "max_feature_num_per_sample");
      auto label_dim = get_value_from_json<int>(j, "label_dim");
      auto slot_num = get_value_from_json<int>(j, "slot_num");
      data_reader[0] = new DataReader<TypeKey>(source_data, batch_size, label_dim, slot_num,
                                               max_feature_num_per_sample, gpu_resource_group);
      data_reader[1] = nullptr;
      std::string eval_source;
      FIND_AND_ASSIGN_STRING_KEY(eval_source, j);
      if (eval_source.empty() == false) {
        if (pid == 0) {  // master process
          data_reader[1] = data_reader[0]->clone_eval_with_shared_output(eval_source);
        } else {  // slave process
          data_reader[1] = data_reader[0]->clone_eval_with_shared_output();
        }
      }
    }
    /* Create Embedding */
    {
      // optimizer configuration
      auto j_optimizer = get_json(config, "optimizer");
      auto opt_params = get_optimizer_param(j_optimizer);

      auto j = get_json(config, "layers");
      // embedding should be the first layer in json
      auto embedding_name = get_value_from_json<std::string>(j[0], "type");

      Embedding_t embedding_type;
      if (!find_item_in_map(&embedding_type, embedding_name, EMBEDDING_TYPE_MAP)) {
        *embedding = nullptr;
        CK_THROW_(Error_t::WrongInput, "Not supported embedding type: " + embedding_name);
      }

      auto j_hparam = get_json(j[0], "sparse_embedding_hparam");
      auto vocabulary_size = get_value_from_json<int>(j_hparam, "vocabulary_size");
      auto embedding_vec_size = get_value_from_json<int>(j_hparam, "embedding_vec_size");
      auto combiner = get_value_from_json<int>(j_hparam, "combiner");
      auto slot_num = get_value_from_json<int>(j_hparam, "slot_num");
      int speedup = 0;
      if (has_key_(j_hparam, "speedup")) {
        speedup = get_value_from_json<int>(j_hparam, "speedup");
      }
      switch (embedding_type) {
        case Embedding_t::SparseEmbeddingHash: {
          auto load_factor = get_value_from_json<float>(j_hparam, "load_factor");
          const SparseEmbeddingHashParams embedding_params = {
              batch_size,
              vocabulary_size,
              load_factor,
              embedding_vec_size,
              max_feature_num_per_sample,
              slot_num,
              combiner,  // combiner: 0-sum, 1-mean, 2-sqrtn
              opt_params,
              speedup};
          *embedding = EmbeddingCreator::create_sparse_embedding_hash(
              (*data_reader)->get_row_offsets_tensors(), (*data_reader)->get_value_tensors(),
              embedding_params, gpu_resource_group);
          break;
        }
        default: { assert(!"Error: no such option && should never get here!"); }
      }
    }
    /* Create Network */
    {
      if (!network->empty()) {
        CK_THROW_(Error_t::WrongInput, "vector network is not empty");
      }

      auto j_layers_array = get_json(config, "layers");
      auto j_optimizer = get_json(config, "optimizer");

      std::vector<Tensor<float>*>& embedding_tensors = (*embedding)->get_output_tensors();
      const std::vector<Tensor<float>*>& label_tensors = (*data_reader)->get_label_tensors();

      int i = 0;
      int total_gpu_count = gpu_resource_group.get_total_gpu_count();
      if (0 != batch_size % total_gpu_count) {
        CK_THROW_(Error_t::WrongInput, "0 != batch_size\%total_gpu_count");
      }
      std::vector<int> device_list = gpu_resource_group.get_device_list();
      for (auto device_id : device_list) {
        network->push_back(create_network(j_layers_array, j_optimizer, *(embedding_tensors[i]),
                                          *(label_tensors[i]), batch_size / total_gpu_count,
                                          device_id, gpu_resource_group[i]));
        i++;
      }
    }

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void Parser::create_pipeline(DataReader<TYPE_1>** data_reader, Embedding<TYPE_1>** embedding,
                             std::vector<Network*>* network, GPUResourceGroup& gpu_resource_group) {
  create_pipeline_internal<TYPE_1>(data_reader, embedding, network, gpu_resource_group, config_,
                                   batch_size_);
}

void Parser::create_pipeline(DataReader<TYPE_2>** data_reader, Embedding<TYPE_2>** embedding,
                             std::vector<Network*>* network, GPUResourceGroup& gpu_resource_group) {
  create_pipeline_internal<TYPE_2>(data_reader, embedding, network, gpu_resource_group, config_,
                                   batch_size_);
}

SolverParser::SolverParser(std::string configure_file) {
  try {
    int num_procs = 1, pid = 0;
#ifdef ENABLE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif

    /* file read to json */
    nlohmann::json config;
    std::ifstream file_stream(configure_file);
    if (!file_stream.is_open()) {
      CK_THROW_(Error_t::FileCannotOpen, "file_stream.is_open() failed: " + configure_file);
    }
    file_stream >> config;
    file_stream.close();

    const std::map<std::string, LrPolicy_t> LR_POLICY = {{"fixed", LrPolicy_t::fixed}};

    /* parse the solver */
    auto j = get_json(config, "solver");

    auto lr_policy_string = get_value_from_json<std::string>(j, "lr_policy");
    if (!find_item_in_map(&lr_policy, lr_policy_string, LR_POLICY)) {
      CK_THROW_(Error_t::WrongInput, "No such poliicy: " + lr_policy_string);
    }

    display = get_value_from_json<int>(j, "display");
    max_iter = get_value_from_json<int>(j, "max_iter");
    snapshot = get_value_from_json<int>(j, "snapshot");
    batchsize = get_value_from_json<int>(j, "batchsize");
    snapshot_prefix = get_value_from_json<std::string>(j, "snapshot_prefix");
    model_file = get_value_from_json<std::string>(j, "model_file");

    FIND_AND_ASSIGN_INT_KEY(eval_interval, j);
    FIND_AND_ASSIGN_INT_KEY(eval_batches, j);
    FIND_AND_ASSIGN_STRING_KEY(embedding_file, j);

    auto gpu_array = get_json(j, "gpu");
    assert(device_list.empty());
    std::vector<std::vector<int>> vvgpu;
    // todo: output the device map
    if (gpu_array[0].is_array()) {
      int num_nodes = gpu_array.size();
      if (num_nodes != num_procs) {
        CK_THROW_(Error_t::WrongInput, "num_nodes != num_procs");
      } else {
        for (auto gpu : gpu_array) {
          std::vector<int> vgpu;
          assert(vgpu.empty());
          for (auto gpu_tmp : gpu) {
            int gpu_id = gpu_tmp.get<int>();
            vgpu.push_back(gpu_id);
            if (gpu_id < 0) {
              CK_THROW_(Error_t::WrongInput, "gpu_id < 0");
            }
          }
          vvgpu.push_back(vgpu);
        }
      }
    } else {
      if (num_procs > 1) {
        CK_THROW_(Error_t::WrongInput, "num_procs > 1");
      }
      std::vector<int> vgpu;
      for (auto gpu_tmp : gpu_array) {
        int gpu_id = gpu_tmp.get<int>();
        vgpu.push_back(gpu_id);
        if (gpu_id < 0) {
          CK_THROW_(Error_t::WrongInput, "gpu_id < 0");
        }
      }
      vvgpu.push_back(vgpu);
    }

    device_map = new DeviceMap(vvgpu, pid);
    device_list = device_map->get_device_list();
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

}  // namespace HugeCTR
