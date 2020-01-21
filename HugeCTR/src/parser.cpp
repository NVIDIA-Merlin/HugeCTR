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
#include "HugeCTR/include/layers/dropout_layer.hpp"
#include "HugeCTR/include/layers/elu_layer.hpp"
#include "HugeCTR/include/layers/fully_connected_layer.hpp"
#include "HugeCTR/include/layers/relu_layer.hpp"
#include "HugeCTR/include/layers/reshape_layer.hpp"
#include "HugeCTR/include/layers/slice_layer.hpp"
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

void assign_first_tensor(std::map<std::string, std::shared_ptr<Tensor<float>>>& tensor_list,
                         const nlohmann::json& j_array,
                         const std::shared_ptr<Tensor<float>>& in_tensor) {
  // get the top of embedding layer
  auto tensor_name = get_value_from_json<std::string>(j_array[0], "top");
  auto p = tensor_list.emplace(tensor_name, in_tensor);
  if (p.second == false) {
    CK_THROW_(Error_t::WrongInput, "Tensor insert failed");
  }
}

struct InputOutputInfo {
  Tensors<float> input;
  std::vector<std::string> output;
};

std::vector<std::string> get_layer_names(const nlohmann::json& json) {
  std::vector<std::string> layer_names;
  if(json.is_array()) {
    for(auto j : json) {
      layer_names.push_back(j.get<std::string>());
    }
  }
  else {
    layer_names.push_back(json.get<std::string>());
  }

  return layer_names;
}

InputOutputInfo get_input_tensor_and_output_name(
    const nlohmann::json& json, std::map<std::string, std::shared_ptr<Tensor<float>>> tensor_list) {
  auto bottom = get_json(json, "bottom");
  std::vector<std::string> bottom_strs = get_layer_names(bottom);

  auto top = get_json(json, "top");
  std::vector<std::string> top_strs = get_layer_names(top);

  Tensors<float> bottom_tensors;
  for(auto& bstr : bottom_strs) {
    for(auto& tstr : top_strs) {
      if (bstr == tstr) {
        CK_THROW_(Error_t::WrongInput, "bottom and top include a same layer name");
      }
    }
    std::shared_ptr<Tensor<float>> tensor;
    if (!find_item_in_map(tensor, bstr, tensor_list)) {
      CK_THROW_(Error_t::WrongInput, "No such bottom: " + bstr);
    }
    bottom_tensors.push_back(tensor);
  }
  return {bottom_tensors, top_strs};
}

struct TensorPair {
  std::shared_ptr<Tensor<float>> tensor;
  std::string name;
};

void add_tensor_to_network(TensorPair& output_tensor_pair,
                           std::map<std::string, std::shared_ptr<Tensor<float>>>& tensor_list,
                           Tensors<float>& tensors) {
  auto p = tensor_list.emplace(output_tensor_pair.name, output_tensor_pair.tensor);

  if (p.second == false) {
    CK_THROW_(Error_t::WrongInput, "Tensor insert failed");
  }
  tensors.push_back(output_tensor_pair.tensor);
}

OptParams get_optimizer_param(const nlohmann::json& j_optimizer) {
  // create optimizer
  auto optimizer_name = get_value_from_json<std::string>(j_optimizer, "type");
  Optimizer_t optimizer_type;
  if (!find_item_in_map(optimizer_type, optimizer_name, OPTIMIZER_TYPE_MAP)) {
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
			const std::map<std::string, std::shared_ptr<Tensor<float>>>& tensor_list_in,
			int batch_size,
                        int device_id, const std::shared_ptr<const GPUResource>& gpu_resource) {
  const std::map<std::string, Layer_t> LAYER_TYPE_MAP = {
      {"BatchNorm", Layer_t::BatchNorm},
      {"BinaryCrossEntropyLoss", Layer_t::BinaryCrossEntropyLoss},
      {"Concat", Layer_t::Concat},
      {"CrossEntropyLoss", Layer_t::CrossEntropyLoss},
      {"Dropout", Layer_t::Dropout},
      {"ELU", Layer_t::ELU},
      {"InnerProduct", Layer_t::InnerProduct},
      {"MultiCrossEntropyLoss", Layer_t::MultiCrossEntropyLoss},
      {"ReLU", Layer_t::ReLU},
      {"Reshape", Layer_t::Reshape},
      {"Slice", Layer_t::Slice},
  };

  std::unique_ptr<Network> network(
      new Network(batch_size, device_id, gpu_resource, false));
  std::map<std::string, std::shared_ptr<Tensor<float>>> tensor_list(tensor_list_in);

  auto& tensors = network->tensors_;
  auto& layers = network->layers_;
  const auto& blobs_buff = network->blobs_buff_;
  const auto& weight_buff = network->weight_buff_;
  const auto& wgrad_buff = network->wgrad_buff_;
  auto& loss_tensor = network->loss_tensor_;
  auto& loss = network->loss_;

  assert(tensors.empty());
  assert(layers.empty());

  for (unsigned int i = 1; i < j_array.size(); i++) {
    const nlohmann::json& j = j_array[i];
    const auto layer_type_name = get_value_from_json<std::string>(j, "type");
    Layer_t layer_type;
    if (!find_item_in_map(layer_type, layer_type_name, LAYER_TYPE_MAP)) {
      //CK_THROW_(Error_t::WrongInput, "No such layer: " + layer_type_name);
      continue;
    }
    auto input_output_info = get_input_tensor_and_output_name(j, tensor_list);
    std::vector<TensorPair> output_tensor_pairs;

    switch (layer_type) {
      case Layer_t::BatchNorm: {
        const auto& bn_in_tensor = input_output_info.input[0];
        // establish out tensor
        std::shared_ptr<Tensor<float>> bn_out_tensor(new Tensor<float>(
            {batch_size, (bn_in_tensor->get_dims())[1]}, blobs_buff, TensorFormat_t::HW));
        output_tensor_pairs.push_back({bn_out_tensor, input_output_info.output[0]});

        // get BN params
        auto j_bn_hparam = get_json(j, "bn_param");
        auto is_training = get_value_from_json<bool>(j_bn_hparam, "is_training");
        auto factor = get_value_from_json<float>(j_bn_hparam, "factor");
        auto eps = get_value_from_json<float>(j_bn_hparam, "eps");

        BatchNormLayer::Params params = {is_training, factor, eps};
        layers.emplace_back(new BatchNormLayer(weight_buff, wgrad_buff, bn_in_tensor, bn_out_tensor,
                                               params, gpu_resource->get_cudnn_handle(),
                                               device_id));
        break;
      }
      case Layer_t::BinaryCrossEntropyLoss: {
	if(input_output_info.input.size() != 2){
	  CK_THROW_(Error_t::WrongInput, "bottom of BinaryCrossEntropyLoss must be two dim");
	}
        const auto& binary_cross_entropy_loss_in_tensor = input_output_info.input[0];
	const auto& label_tensor = input_output_info.input[1];
        loss_tensor.reset(new Tensor<float>({1, 1}, blobs_buff, TensorFormat_t::HW));
        loss.reset(new BinaryCrossEntropyLoss(label_tensor, binary_cross_entropy_loss_in_tensor,
                                              loss_tensor, device_id));
        break;
      }
      case Layer_t::Concat: {
        auto& in_tensors = input_output_info.input;
        std::shared_ptr<Tensor<float>> out_tensor;
        layers.emplace_back(new ConcatLayer(in_tensors, out_tensor, blobs_buff, device_id));
        output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});
        break;
      }
      case Layer_t::CrossEntropyLoss: {
	if(input_output_info.input.size() != 2){
	  CK_THROW_(Error_t::WrongInput, "bottom of CrossEntropyLoss must be two dim");
	}
        const auto& cross_entropy_loss_in_tensor = input_output_info.input[0];
	const auto& label_tensor = input_output_info.input[1];
        loss_tensor.reset(new Tensor<float>({1, 1}, blobs_buff, TensorFormat_t::HW));
        loss.reset(new CrossEntropyLoss(label_tensor, cross_entropy_loss_in_tensor, loss_tensor,
                                        device_id));
        break;
      }
      case Layer_t::Dropout: {
        const auto& do_in_tensor = input_output_info.input[0];

        // establish out tensor
        std::shared_ptr<Tensor<float>> do_out_tensor(new Tensor<float>(
            {batch_size, (do_in_tensor->get_dims())[1]}, blobs_buff, TensorFormat_t::HW));
        output_tensor_pairs.push_back({do_out_tensor, input_output_info.output[0]});
        // get ELU params
        auto rate_it = j.find("rate");
        auto rate = (rate_it != j.end())? rate_it->get<float>() : 0.5f;
        layers.emplace_back(new DropoutLayer(do_in_tensor,
                                             do_out_tensor,
                                             rate,
                                             gpu_resource->get_curand_generator(),
                                             device_id));

        break;
      }
      case Layer_t::ELU: {
        const auto& elu_in_tensor = input_output_info.input[0];

        // establish out tensor
        std::shared_ptr<Tensor<float>> elu_out_tensor(new Tensor<float>(
            {batch_size, (elu_in_tensor->get_dims())[1]}, blobs_buff, TensorFormat_t::HW));
        output_tensor_pairs.push_back({elu_out_tensor, input_output_info.output[0]});
        // get ELU params
        auto j_elu_hparam = get_json(j, "elu_param");
        auto alpha = get_value_from_json<float>(j_elu_hparam, "alpha");
        layers.emplace_back(new EluLayer(elu_in_tensor, elu_out_tensor, alpha, device_id));

        break;
      }
      case Layer_t::InnerProduct: {
        const auto& fc_in_tensor = input_output_info.input[0];
        // establish out tensor
        auto j_fc_param = get_json(j, "fc_param");
        auto output = get_value_from_json<int>(j_fc_param, "num_output");
        std::shared_ptr<Tensor<float>> out_tensor(
            new Tensor<float>({batch_size, output}, blobs_buff, TensorFormat_t::HW));
        output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});
        // establish layer
        Layer* fc_layer = new FullyConnectedLayer(weight_buff, wgrad_buff, fc_in_tensor, out_tensor,
                                                  TensorFormat_t::HW,
                                                  gpu_resource->get_cublas_handle(), device_id);
        layers.emplace_back(fc_layer);
        break;
      }
      case Layer_t::MultiCrossEntropyLoss: {
	if(input_output_info.input.size() != 2){
	  CK_THROW_(Error_t::WrongInput, "bottom of MultiCrossEntropyLoss must be two dim");
	}
        const auto& multi_cross_entropy_loss_in_tensor = input_output_info.input[0];
	const auto& label_tensor = input_output_info.input[1];
        loss_tensor.reset(new Tensor<float>({1, 1}, blobs_buff, TensorFormat_t::HW));

        auto tweight = get_json(j, "target_weight");
        std::vector<float> target_weight_vec;
        for (auto tweight_tmp : tweight) {
          float tweight_val = tweight_tmp.get<float>();
          target_weight_vec.push_back(tweight_val);
        }
        loss.reset(new MultiCrossEntropyLoss(label_tensor, multi_cross_entropy_loss_in_tensor,
                                             loss_tensor, target_weight_vec, device_id));
        break;
      }
      case Layer_t::ReLU: {
        const auto& relu_in_tensor = input_output_info.input[0];

        // establish out tensor
        std::shared_ptr<Tensor<float>> relu_out_tensor(new Tensor<float>(
            {batch_size, (relu_in_tensor->get_dims())[1]}, blobs_buff, TensorFormat_t::HW));
        output_tensor_pairs.push_back({relu_out_tensor, input_output_info.output[0]});
        layers.emplace_back(new ReluLayer(relu_in_tensor, relu_out_tensor, device_id));
        break;
      }
      case Layer_t::Reshape: {
        const auto& in_tensor = input_output_info.input[0];
        std::shared_ptr<Tensor<float>> out_tensor;

        auto selected_it = j.find("selected");
        // selective reshape
        if(selected_it != j.end()) { 
          std::vector<int> selected;
          nlohmann::json j_selected = (selected_it.value());
          for (auto slot_obj : j_selected) {
            int slot_id = slot_obj.get<int>();
            if (slot_id < 0) CK_THROW_(Error_t::WrongInput, "slot_id < 0");
            selected.push_back(slot_id);
          }
          layers.emplace_back(new ReshapeLayer(in_tensor, out_tensor, blobs_buff, selected, device_id));
        }
        // general purpose reshape
        else {
          auto leading_dim_it = j.find("leading_dim");
          auto in_dims = in_tensor->get_dims();
          // if leading_dim is not specified, default leading_dim = n_slots * vector_length
          int leading_dim = (leading_dim_it != j.end())?
            (*leading_dim_it).get<int>() : in_tensor->get_num_elements() / in_dims[0];
          layers.emplace_back(new ReshapeLayer(in_tensor, out_tensor, leading_dim, device_id));
        }

        output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});

        break;
      }
      case Layer_t::Slice: {
        const auto& in_tensor = input_output_info.input[0];

        std::set<std::pair<int,int>> ranges;
        auto j_ranges = get_json(j, "ranges");
        assert(j_ranges.is_array());
        for(auto j_range : j_ranges) {
          assert(j_range.is_array());
          ranges.insert({j_range[0].get<int>(), j_range[1].get<int>()});
        }

        Tensors<float> out_tensors;
        layers.emplace_back(new SliceLayer(in_tensor, out_tensors, blobs_buff, ranges, device_id));
        for(size_t i = 0; i < out_tensors.size(); i++) {
          output_tensor_pairs.push_back({out_tensors[i], input_output_info.output[i]});
        }
        break;
      }
      default:
        assert(!"Error: no such layer && should never get here!");
    }  // end of switch

    if (!(layer_type == Layer_t::CrossEntropyLoss ||
          layer_type == Layer_t::BinaryCrossEntropyLoss ||
          layer_type == Layer_t::MultiCrossEntropyLoss)) {
      for(auto& output_tensor_pair : output_tensor_pairs) {
        add_tensor_to_network(output_tensor_pair, tensor_list, tensors);
      }
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
      network->optimizer_.reset(
          new AdamOptimizer(weight_buff, wgrad_buff, device_id, alpha, beta1, beta2, epsilon));
      break;
    }
    case Optimizer_t::MomentumSGD: {
      auto learning_rate = opt_param.lr;
      auto momentum_factor = opt_param.hyperparams.momentum.factor;
      network->optimizer_.reset(
          new MomentumSGD(weight_buff, wgrad_buff, device_id, learning_rate, momentum_factor));
      break;
    }
    case Optimizer_t::Nesterov: {
      auto learning_rate = opt_param.lr;
      auto momentum_factor = opt_param.hyperparams.nesterov.mu;
      network->optimizer_.reset(new NesterovOptimizer(weight_buff, wgrad_buff, device_id,
                                                      learning_rate, momentum_factor));
      break;
    }
    default:
      assert(!"Error: no such optimizer && should never get here!");
  }
  weight_buff->init(device_id);
  wgrad_buff->init(device_id);
  blobs_buff->init(device_id);

  return network.release();
}

template <typename TypeKey>
static void create_pipeline_internal(std::unique_ptr<DataReader<TypeKey>>& data_reader,
                                     std::unique_ptr<DataReader<TypeKey>>& data_reader_eval,
                                     std::vector<std::unique_ptr<Embedding<TypeKey>>>& embedding,
                                     std::vector<std::unique_ptr<Network>>& network,
                                     const std::shared_ptr<GPUResourceGroup>& gpu_resource_group,
                                     nlohmann::json config, int batch_size) {
  try {
    int num_procs = 1, pid = 0;
#ifdef ENABLE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif

    std::map<std::string, SparseInput<TypeKey>> sparse_input_map;
    std::map<std::string, std::shared_ptr<Tensor<float>>> tensor_maps[gpu_resource_group->size()];
    {
      if (!network.empty()) {
        CK_THROW_(Error_t::WrongInput, "vector network is not empty");
      }

      auto j_layers_array = get_json(config, "layers");
      auto j_optimizer = get_json(config, "optimizer");

      
      {
	// Create Data Reader
	const nlohmann::json& j = j_layers_array[0];
	const auto layer_type_name = get_value_from_json<std::string>(j, "type");
	if(layer_type_name.compare("Data")!=0){
	  CK_THROW_(Error_t::WrongInput, "the first layer is not Data layer:" + layer_type_name);
	}
	
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
	

	auto j_label = get_json(j, "label");
	auto top_strs_label = get_value_from_json<std::string>(j_label, "top");
	auto label_dim = get_value_from_json<int>(j_label, "label_dim");

	auto j_dense = get_json(j, "dense");
	auto top_strs_dense = get_value_from_json<std::string>(j_dense, "top");
	auto dense_dim = get_value_from_json<int>(j_dense, "dense_dim");

	const std::map<std::string, Check_t> CHECK_TYPE_MAP = {
	  {"Sum", Check_t::Sum},
	  {"None", Check_t::None}
	};

	Check_t check_type;
	const auto check_str = get_value_from_json<std::string>(j, "check");
	if (!find_item_in_map(check_type, check_str, CHECK_TYPE_MAP)){
	  CK_THROW_(Error_t::WrongInput, "Not supported check type: " + check_str);
	}
	
	std::vector<DataReaderSparseParam> data_reader_sparse_param_array;

	const std::map<std::string, DataReaderSparse_t> DATA_TYPE_MAP = {
	  {"DistributedSlot", DataReaderSparse_t::Distributed},
	  {"Localized", DataReaderSparse_t::Localized},
	};

	auto j_sparse = get_json(j, "sparse");
	std::vector<std::string> sparse_names;

	for(unsigned int i = 0; i < j_sparse.size(); i++){
	  DataReaderSparseParam param;
	  
	  const nlohmann::json& js = j_sparse[i];
	  const auto sparse_name = get_value_from_json<std::string>(js, "top");
	  const auto data_type_name = get_value_from_json<std::string>(js, "type");
	  if (!find_item_in_map(param.type, data_type_name, DATA_TYPE_MAP)){
	    CK_THROW_(Error_t::WrongInput, "Not supported data type: " + data_type_name);
	  }
	  param.max_feature_num = get_value_from_json<int>(js, "max_feature_num_per_sample");
	  param.slot_num = get_value_from_json<int>(js, "slot_num");
	  data_reader_sparse_param_array.push_back(param);
	  SparseInput<TypeKey> sparse_input(param.slot_num, param.max_feature_num);
	  sparse_input_map.emplace(sparse_name, sparse_input);
	  sparse_names.push_back(sparse_name);
	}

	data_reader.reset(new DataReader<TypeKey>(source_data, batch_size, label_dim, dense_dim, check_type,
						  data_reader_sparse_param_array, gpu_resource_group));

	for(unsigned int i = 0; i < gpu_resource_group->size(); i++){
	  tensor_maps[i].emplace(top_strs_label, (data_reader->get_label_tensors())[i]);
	  tensor_maps[i].emplace(top_strs_dense, (data_reader->get_dense_tensors())[i]);
	}

	for(unsigned int i = 0; i < j_sparse.size(); i++){
	  const auto& sparse_input = sparse_input_map.find(sparse_names[i]);
	  sparse_input->second.row = data_reader->get_row_offsets_tensors(i);
	  sparse_input->second.value = data_reader->get_value_tensors(i);
	}
	data_reader_eval = nullptr;
	std::string eval_source;
	FIND_AND_ASSIGN_STRING_KEY(eval_source, j);
	if (eval_source.empty() == false) {
	  if (pid == 0) {  // master process
	    data_reader_eval.reset(data_reader->clone_eval_with_shared_output(eval_source));
	  } else {  // slave process
	    data_reader_eval.reset(data_reader->clone_eval_with_shared_output());
	  }
	}
      }

      /* Create Embedding */
      {

	auto opt_params = get_optimizer_param(j_optimizer);

	
	const std::map<std::string, Embedding_t> EMBEDDING_TYPE_MAP = {
	  {"SparseEmbedding", Embedding_t::SparseEmbeddingHash},
	  {"LocalizedSlotSparseEmbedding", Embedding_t::LocalizedSlotSparseEmbedding}
	};
	for (unsigned int i = 1; i < j_layers_array.size(); i++) {
	  //if not embedding then break
	  const nlohmann::json& j = j_layers_array[i];
	  auto embedding_name = get_value_from_json<std::string>(j, "type");
	  Embedding_t embedding_type;
	  if (!find_item_in_map(embedding_type, embedding_name, EMBEDDING_TYPE_MAP)) {
	    break;
	  }
	  auto bottom_name = get_value_from_json<std::string>(j, "bottom");
	  auto top_name = get_value_from_json<std::string>(j, "top");

	  auto j_hparam = get_json(j, "sparse_embedding_hparam");
	  auto vocabulary_size = get_value_from_json<int>(j_hparam, "vocabulary_size");
	  auto embedding_vec_size = get_value_from_json<int>(j_hparam, "embedding_vec_size");
	  auto combiner = get_value_from_json<int>(j_hparam, "combiner");
	  
	  SparseInput<TypeKey> sparse_input;

	  if (!find_item_in_map(sparse_input, bottom_name, sparse_input_map)) {
	    CK_THROW_(Error_t::WrongInput, "Cannot find bottom");
	  }

	  switch (embedding_type) {
	  case Embedding_t::SparseEmbeddingHash: {
	    auto load_factor = get_value_from_json<float>(j_hparam, "load_factor");
	    const SparseEmbeddingHashParams embedding_params = {
	      batch_size,
	      vocabulary_size,
	      load_factor,
	      embedding_vec_size,
	      sparse_input.max_feature_num_per_sample,
	      sparse_input.slot_num,
	      combiner,  // combiner: 0-sum, 1-mean, 2-sqrtn
	      opt_params};
	    embedding.emplace_back(EmbeddingCreator::create_sparse_embedding_hash(
          	   sparse_input.row, sparse_input.value,
		   embedding_params, gpu_resource_group));
	    for(unsigned int i = 0; i < gpu_resource_group->size(); i++){
	      tensor_maps[i].emplace(top_name, (embedding.back()->get_output_tensors())[i]);
	    }
	    break;
	  }
	  case Embedding_t::LocalizedSlotSparseEmbedding: {
	    //TODO fill with LocalizedSlotSparseEmbedding
	  }
	  default: { assert(!"Error: no such option && should never get here!"); }
	  }
	}
      }

      int i = 0;
      int total_gpu_count = gpu_resource_group->get_total_gpu_count();
      if (0 != batch_size % total_gpu_count) {
        CK_THROW_(Error_t::WrongInput, "0 != batch_size\%total_gpu_count");
      }
      const auto& device_list = gpu_resource_group->get_device_list();
      for (auto device_id : device_list) {
        network.emplace_back(create_network(j_layers_array, j_optimizer, tensor_maps[i],
                                            batch_size / total_gpu_count,
                                            device_id, (*gpu_resource_group)[i]));
        i++;
      }
    }

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void Parser::create_pipeline(std::unique_ptr<DataReader<TYPE_1>>& data_reader,
                             std::unique_ptr<DataReader<TYPE_1>>& data_reader_eval,
                             std::vector<std::unique_ptr<Embedding<TYPE_1>>>& embedding,
                             std::vector<std::unique_ptr<Network>>& network,
                             const std::shared_ptr<GPUResourceGroup>& gpu_resource_group) {
  create_pipeline_internal<TYPE_1>(data_reader, data_reader_eval, embedding, network,
                                   gpu_resource_group, config_, batch_size_);
}

void Parser::create_pipeline(std::unique_ptr<DataReader<TYPE_2>>& data_reader,
                             std::unique_ptr<DataReader<TYPE_2>>& data_reader_eval,
			     std::vector<std::unique_ptr<Embedding<TYPE_2>>>& embedding,
                             std::vector<std::unique_ptr<Network>>& network,
                             const std::shared_ptr<GPUResourceGroup>& gpu_resource_group) {
  create_pipeline_internal<TYPE_2>(data_reader, data_reader_eval, embedding, network,
                                   gpu_resource_group, config_, batch_size_);
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
    if (!find_item_in_map(lr_policy, lr_policy_string, LR_POLICY)) {
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
    //FIND_AND_ASSIGN_STRING_KEY(embedding_file, j);
    if(has_key_(j, "embedding_files")){
      auto j_embedding_files = get_json(j, "embedding_files");
      if(j_embedding_files.is_array()) {
	for(unsigned int i = 0; i < j_embedding_files; i++){
	  embedding_files.push_back(j_embedding_files[i].get<std::string>());
	}
      } else {
	embedding_files.push_back(get_value_from_json<std::string>(j, "embedding_file"));
      }
    }
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

    device_map.reset(new DeviceMap(vvgpu, pid));
    device_list = device_map->get_device_list();
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

}  // namespace HugeCTR
