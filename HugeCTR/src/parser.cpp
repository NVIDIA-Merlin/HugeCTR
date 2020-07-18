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

#include "HugeCTR/include/parser.hpp"
#include "HugeCTR/include/device_map.hpp"
#include "HugeCTR/include/layer.hpp"
#include "HugeCTR/include/layers/add_layer.hpp"
#include "HugeCTR/include/layers/batch_norm_layer.hpp"
#include "HugeCTR/include/layers/cast_layer.hpp"
#include "HugeCTR/include/layers/concat_layer.hpp"
#include "HugeCTR/include/layers/dropout_layer.hpp"
#include "HugeCTR/include/layers/elu_layer.hpp"
#include "HugeCTR/include/layers/fm_order2_layer.hpp"
#include "HugeCTR/include/layers/fully_connected_layer.hpp"
#include "HugeCTR/include/layers/fully_connected_layer_half.hpp"
#include "HugeCTR/include/layers/fused_fully_connected_layer.hpp"
#include "HugeCTR/include/layers/interaction_layer.hpp"
#include "HugeCTR/include/layers/multi_cross_layer.hpp"
#include "HugeCTR/include/layers/multiply_layer.hpp"
#include "HugeCTR/include/layers/reduce_sum_layer.hpp"
#include "HugeCTR/include/layers/relu_layer.hpp"
#include "HugeCTR/include/layers/reshape_layer.hpp"
#include "HugeCTR/include/layers/slice_layer.hpp"
#include "HugeCTR/include/loss.hpp"
#include "HugeCTR/include/metrics.hpp"
#include "HugeCTR/include/optimizers/adam_optimizer.hpp"
#include "HugeCTR/include/optimizers/momentum_sgd.hpp"
#include "HugeCTR/include/optimizers/nesterov_optimizer.hpp"
#include "HugeCTR/include/optimizers/sgd_optimizer.hpp"
#include "HugeCTR/include/regularizers/l1_regularizer.hpp"
#include "HugeCTR/include/regularizers/l2_regularizer.hpp"
#include "HugeCTR/include/regularizers/no_regularizer.hpp"

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {

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
  ITensors input;
  std::vector<std::string> output;
};

std::vector<std::string> get_layer_names(const nlohmann::json& json) {
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

InputOutputInfo get_input_tensor_and_output_name(
    const nlohmann::json& json, std::map<std::string, std::shared_ptr<ITensor>> tensor_list) {
  auto bottom = get_json(json, "bottom");
  std::vector<std::string> bottom_strs = get_layer_names(bottom);

  auto top = get_json(json, "top");
  std::vector<std::string> top_strs = get_layer_names(top);

  ITensors bottom_tensors;
  for (auto& bstr : bottom_strs) {
    for (auto& tstr : top_strs) {
      if (bstr == tstr) {
        CK_THROW_(Error_t::WrongInput, "bottom and top include a same layer name");
      }
    }
    std::shared_ptr<ITensor> tensor;
    if (!find_item_in_map(tensor, bstr, tensor_list)) {
      CK_THROW_(Error_t::WrongInput, "No such bottom: " + bstr);
    }
    bottom_tensors.push_back(tensor);
  }
  return {bottom_tensors, top_strs};
}

struct TensorPair {
  std::shared_ptr<ITensor> tensor;
  std::string name;
};

void add_tensor_to_network(TensorPair& output_tensor_pair,
                           std::map<std::string, std::shared_ptr<ITensor>>& tensor_list) {
  auto p = tensor_list.emplace(output_tensor_pair.name, output_tensor_pair.tensor);

  if (p.second == false) {
    std::cout << "tensor name:" << output_tensor_pair.name << std::endl;
    CK_THROW_(Error_t::WrongInput, "Tensor insert failed");
  }
}

template <typename Type>
OptParams<Type> get_optimizer_param(const nlohmann::json& j_optimizer) {
  // create optimizer
  auto optimizer_name = get_value_from_json<std::string>(j_optimizer, "type");
  Optimizer_t optimizer_type;
  if (!find_item_in_map(optimizer_type, optimizer_name, OPTIMIZER_TYPE_MAP)) {
    CK_THROW_(Error_t::WrongInput, "No such optimizer: " + optimizer_name);
  }

  OptHyperParams opt_hyper_params;
  memset(&opt_hyper_params, 0, sizeof(opt_hyper_params));
  OptParams<Type> opt_params;

  bool global_update = false;
  global_update = get_value_from_json<bool>(j_optimizer, "global_update");

  switch (optimizer_type) {
    case Optimizer_t::Adam: {
      auto j_hparam = get_json(j_optimizer, "adam_hparam");
      auto alpha = (Type)get_value_from_json<float>(j_hparam, "alpha");
      auto beta1 = (Type)get_value_from_json<float>(j_hparam, "beta1");
      auto beta2 = (Type)get_value_from_json<float>(j_hparam, "beta2");
      auto epsilon = (Type)get_value_from_json<float>(j_hparam, "epsilon");
      opt_hyper_params.adam.beta1 = beta1;
      opt_hyper_params.adam.beta2 = beta2;
      opt_hyper_params.adam.epsilon = epsilon;
      opt_params = {Optimizer_t::Adam, alpha, opt_hyper_params, global_update};
      break;
    }
    case Optimizer_t::MomentumSGD: {
      auto j_hparam = get_json(j_optimizer, "momentum_sgd_hparam");
      auto learning_rate = (Type)get_value_from_json<float>(j_hparam, "learning_rate");
      auto momentum_factor = (Type)get_value_from_json<float>(j_hparam, "momentum_factor");
      opt_hyper_params.momentum.factor = momentum_factor;
      opt_params = {Optimizer_t::MomentumSGD, learning_rate, opt_hyper_params, global_update};
      break;
    }
    case Optimizer_t::Nesterov: {
      auto j_hparam = get_json(j_optimizer, "nesterov_hparam");
      auto learning_rate = (Type)get_value_from_json<float>(j_hparam, "learning_rate");
      auto momentum_factor = (Type)get_value_from_json<float>(j_hparam, "momentum_factor");
      opt_hyper_params.nesterov.mu = momentum_factor;
      opt_params = {Optimizer_t::Nesterov, learning_rate, opt_hyper_params, global_update};
      break;
    }
    case Optimizer_t::SGD: {
      auto j_hparam = get_json(j_optimizer, "sgd_hparam");
      auto learning_rate = (Type)get_value_from_json<float>(j_hparam, "learning_rate");
      opt_params = {Optimizer_t::SGD, learning_rate, opt_hyper_params, global_update};
      break;
    }
    default:
      assert(!"Error: no such optimizer && should never get here!");
  }
  return opt_params;
}

template <typename T>
std::shared_ptr<Regularizer<T>> create_regularizer(
    const nlohmann::json& j, const std::shared_ptr<GeneralBuffer<float>>& weight_buff,
    const std::shared_ptr<GeneralBuffer<T>>& wgrad_buff, const int batch_size,
    cublasHandle_t cublas_handle, const int device_id) {
  std::shared_ptr<Regularizer<T>> reg(
      new NoRegularizer<T>(weight_buff, wgrad_buff, batch_size, device_id));
  auto reg_it = j.find("regularizer");
  if (reg_it != j.end()) {
    Regularizer_t reg_type;
    auto reg_name = reg_it->get<std::string>();
    if (!find_item_in_map(reg_type, reg_name, REGULARIZER_TYPE_MAP)) {
      CK_THROW_(Error_t::WrongInput, "No such regularizer: " + reg_name);
    }
    switch (reg_type) {
      case Regularizer_t::L1: {
        const auto lambda = get_value_from_json<float>(j, "lambda");
        reg.reset(new L1Regularizer<T>(weight_buff, wgrad_buff, batch_size, lambda, cublas_handle,
                                       device_id));
        break;
      }
      case Regularizer_t::L2: {
        const auto lambda = get_value_from_json<float>(j, "lambda");
        reg.reset(new L2Regularizer<T>(weight_buff, wgrad_buff, batch_size, lambda, cublas_handle,
                                       device_id));
        break;
      }
      default: { assert(!"Error: no such regularizer!"); }
    }
  }
  return reg;
}

const std::map<std::string, Layer_t> LAYER_TYPE_MAP = {
    {"BatchNorm", Layer_t::BatchNorm},
    {"BinaryCrossEntropyLoss", Layer_t::BinaryCrossEntropyLoss},
    {"Concat", Layer_t::Concat},
    {"CrossEntropyLoss", Layer_t::CrossEntropyLoss},
    {"Dropout", Layer_t::Dropout},
    {"ELU", Layer_t::ELU},
    {"InnerProduct", Layer_t::InnerProduct},
    {"Interaction", Layer_t::Interaction},
    {"MultiCrossEntropyLoss", Layer_t::MultiCrossEntropyLoss},
    {"ReLU", Layer_t::ReLU},
    {"Reshape", Layer_t::Reshape},
    {"Slice", Layer_t::Slice},
    {"Multiply", Layer_t::Multiply},
    {"FmOrder2", Layer_t::FmOrder2},
    {"Add", Layer_t::Add},
    {"ReduceSum", Layer_t::ReduceSum},
    {"MultiCross", Layer_t::MultiCross}};
const std::map<std::string, Layer_t> LAYER_TYPE_MAP_MP = {
    {"BinaryCrossEntropyLoss", Layer_t::BinaryCrossEntropyLoss},
    {"Concat", Layer_t::Concat},
    {"Cast", Layer_t::Cast},
    {"InnerProduct", Layer_t::InnerProduct},
    {"FusedInnerProduct", Layer_t::FusedInnerProduct},
    {"Interaction", Layer_t::Interaction},
    {"Reshape", Layer_t::Reshape},
    {"Slice", Layer_t::Slice},
    {"ReLU", Layer_t::ReLU}};
const std::map<std::string, Embedding_t> EMBEDDING_TYPE_MAP = {
    {"DistributedSlotSparseEmbeddingHash", Embedding_t::DistributedSlotSparseEmbeddingHash},
    {"LocalizedSlotSparseEmbeddingHash", Embedding_t::LocalizedSlotSparseEmbeddingHash},
    {"LocalizedSlotSparseEmbeddingOneHot", Embedding_t::LocalizedSlotSparseEmbeddingOneHot}};
const std::map<std::string, Initializer_t> INITIALIZER_TYPE_MAP = {
    {"Uniform", Initializer_t::Uniform},
    {"XavierNorm", Initializer_t::XavierNorm},
    {"XavierUniform", Initializer_t::XavierUniform},
    {"Zero", Initializer_t::Zero}};

/*
 * Create single network
 *
 */
Network* create_network(const nlohmann::json& j_array, const nlohmann::json& j_optimizer,
                        const std::map<std::string, std::shared_ptr<ITensor>>& tensor_list_in,
                        int device_id, int num_networks_in_global,
                        const std::shared_ptr<const GPUResource>& gpu_resource,
                        bool use_mixed_precision, float scaler) {
  std::unique_ptr<Network> network(new Network(device_id, gpu_resource, use_mixed_precision));
  std::map<std::string, std::shared_ptr<ITensor>> tensor_list(tensor_list_in);

  auto& layers = network->layers_;
  const auto& blobs_buff = network->blobs_buff_;
  const auto& blobs_buff_half = network->blobs_buff_half_;
  const auto& weight_buff = network->weight_buff_;
  const auto& weight_buff_half = network->weight_buff_half_;
  const auto& wgrad_buff = network->wgrad_buff_;
  const auto& wgrad_buff_half = network->wgrad_buff_half_;
  auto& loss_tensor = network->loss_tensor_;
  auto& loss = network->loss_;

  assert(layers.empty());

  for (unsigned int i = 1; i < j_array.size(); i++) {
    const nlohmann::json& j = j_array[i];
    const auto layer_type_name = get_value_from_json<std::string>(j, "type");
    Layer_t layer_type;

    const auto& layer_map = use_mixed_precision ? LAYER_TYPE_MAP_MP : LAYER_TYPE_MAP;

    if (!find_item_in_map(layer_type, layer_type_name, layer_map)) {
      Embedding_t embedding_type;
      if (!find_item_in_map(embedding_type, layer_type_name, EMBEDDING_TYPE_MAP)) {
        CK_THROW_(Error_t::WrongInput, "No such layer: " + layer_type_name);
      }
      continue;
    }

    std::vector<TensorPair> output_tensor_pairs;
    auto input_output_info = get_input_tensor_and_output_name(j, tensor_list);
    switch (layer_type) {
      case Layer_t::BatchNorm: {
        const auto& bn_in_tensor = input_output_info.input[0];
        // establish out tensor
        TensorPtr<float> bn_out_tensor(
            new Tensor<float>(bn_in_tensor->get_dims(), blobs_buff, TensorFormat_t::HW));
        output_tensor_pairs.push_back({bn_out_tensor, input_output_info.output[0]});

        // get BN params
        auto j_bn_hparam = get_json(j, "bn_param");
        auto factor = get_value_from_json<float>(j_bn_hparam, "factor");
        auto eps = get_value_from_json<float>(j_bn_hparam, "eps");
        // establish initializer
        std::vector<Initializer_t> initializer_types(2, Initializer_t::Default);
        if (has_key_(j_bn_hparam, "gamma_init")){
          const auto gamma_init_name = get_value_from_json<std::string>(j_bn_hparam, "gamma_init");
          Initializer_t gamma_init_type;
          if (!find_item_in_map(gamma_init_type, gamma_init_name, INITIALIZER_TYPE_MAP)){
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + gamma_init_name);
          } else {
            initializer_types[0] = gamma_init_type;
          }
        }
        if (has_key_(j_bn_hparam, "beta_init")){
          const auto beta_init_name = get_value_from_json<std::string>(j_bn_hparam, "beta_init");
          Initializer_t beta_init_type;
          if (!find_item_in_map(beta_init_type, beta_init_name, INITIALIZER_TYPE_MAP)){
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + beta_init_name);
          } else {
            initializer_types[1] = beta_init_type;
          }
        }

        BatchNormLayer::Params params = {factor, eps};
        layers.emplace_back(new BatchNormLayer(
            weight_buff, wgrad_buff, dynamic_tensor_cast<float>(bn_in_tensor),
            bn_out_tensor, params, gpu_resource->get_cudnn_handle(), device_id, initializer_types));
        break;
      }
      case Layer_t::BinaryCrossEntropyLoss: {
        if (input_output_info.input.size() != 2) {
          CK_THROW_(Error_t::WrongInput, "bottom of BinaryCrossEntropyLoss must be two dim");
        }
        const auto& binary_cross_entropy_loss_in_tensor = input_output_info.input[0];
        const auto& label_tensor = input_output_info.input[1];
        loss_tensor.reset(new Tensor<float>({1, 1}, blobs_buff, TensorFormat_t::HW));
        if (use_mixed_precision) {
          loss.reset(new BinaryCrossEntropyLoss<__half>(
	      dynamic_tensor_cast<float>(label_tensor),
              dynamic_tensor_cast<__half>(binary_cross_entropy_loss_in_tensor),
              loss_tensor,
              create_regularizer(j, weight_buff, wgrad_buff_half,
                                 (binary_cross_entropy_loss_in_tensor->get_dims())[0],
                                 gpu_resource->get_cublas_handle(), device_id),
              device_id, num_networks_in_global, scaler));
        } else {
          loss.reset(new BinaryCrossEntropyLoss<float>(
              dynamic_tensor_cast<float>(label_tensor),
              dynamic_tensor_cast<float>(binary_cross_entropy_loss_in_tensor),
              loss_tensor,
              create_regularizer(j, weight_buff, wgrad_buff,
                                 (binary_cross_entropy_loss_in_tensor->get_dims())[0],
                                 gpu_resource->get_cublas_handle(), device_id),
              device_id, num_networks_in_global, scaler));
        }
        break;
      }
      case Layer_t::Concat: {
        auto& in_tensors = input_output_info.input;
        if (use_mixed_precision) {
          std::shared_ptr<Tensor<__half>> out_tensor;
          layers.emplace_back(new ConcatLayer<__half>(tensor_vec_dynamic_cast<__half>(in_tensors),
                                              out_tensor, blobs_buff_half, device_id));
          output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});
        } else {
          std::shared_ptr<Tensor<float>> out_tensor;
          layers.emplace_back(new ConcatLayer<float>(tensor_vec_dynamic_cast<float>(in_tensors),
                                              out_tensor, blobs_buff, device_id));
          output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});
        }
        break;
      }
      case Layer_t::CrossEntropyLoss: {
        if (input_output_info.input.size() != 2) {
          CK_THROW_(Error_t::WrongInput, "bottom of CrossEntropyLoss must be two dim");
        }
        const auto& cross_entropy_loss_in_tensor = input_output_info.input[0];
        const auto& label_tensor = input_output_info.input[1];
        loss_tensor.reset(new Tensor<float>({1, 1}, blobs_buff, TensorFormat_t::HW));
        if (use_mixed_precision) {
          loss.reset(new CrossEntropyLoss<__half>(
	      dynamic_tensor_cast<float>(label_tensor),
              dynamic_tensor_cast<__half>(cross_entropy_loss_in_tensor), loss_tensor,
              create_regularizer(j, weight_buff, wgrad_buff_half,
                                 (cross_entropy_loss_in_tensor->get_dims())[0],
                                 gpu_resource->get_cublas_handle(), device_id),
              device_id, num_networks_in_global, scaler));
        } else {
          loss.reset(new CrossEntropyLoss<float>(
              dynamic_tensor_cast<float>(label_tensor),
              dynamic_tensor_cast<float>(cross_entropy_loss_in_tensor), loss_tensor,
              create_regularizer(j, weight_buff, wgrad_buff,
                                 (cross_entropy_loss_in_tensor->get_dims())[0],
                                 gpu_resource->get_cublas_handle(), device_id),
              device_id, num_networks_in_global, scaler));
        }
        break;
      }
      case Layer_t::Dropout: {
        const auto& do_in_tensor = input_output_info.input[0];

        // establish out tensor
        std::shared_ptr<Tensor<float>> do_out_tensor(
            new Tensor<float>(do_in_tensor->get_dims(), blobs_buff, TensorFormat_t::HW));
        output_tensor_pairs.push_back({do_out_tensor, input_output_info.output[0]});
        // get ELU params
        auto rate_it = j.find("rate");
        auto rate = (rate_it != j.end()) ? rate_it->get<float>() : 0.5f;
        layers.emplace_back(new DropoutLayer(dynamic_tensor_cast<float>(do_in_tensor),
                                             do_out_tensor, rate,
                                             gpu_resource->get_curand_generator(), device_id));
        network->enable_cuda_graph_ = false;

        break;
      }
      case Layer_t::ELU: {
        const auto& elu_in_tensor = input_output_info.input[0];

        // establish out tensor
        std::shared_ptr<Tensor<float>> elu_out_tensor(
            new Tensor<float>(elu_in_tensor->get_dims(), blobs_buff, TensorFormat_t::HW));
        output_tensor_pairs.push_back({elu_out_tensor, input_output_info.output[0]});
        // get ELU params
        auto j_elu_hparam = get_json(j, "elu_param");
        auto alpha = get_value_from_json<float>(j_elu_hparam, "alpha");
        layers.emplace_back(new EluLayer(dynamic_tensor_cast<float>(elu_in_tensor),
                                         elu_out_tensor, alpha, device_id));

        break;
      }

      case Layer_t::FusedInnerProduct: {
        const auto& fc_in_tensor = input_output_info.input[0];

        auto j_fc_param = get_json(j, "fc_param");
        // establish initializer
        std::vector<Initializer_t> initializer_types(2, Initializer_t::Default);
        if (has_key_(j_fc_param, "weight_init")){
          const auto weight_init_name = get_value_from_json<std::string>(j_fc_param, "weight_init");
          Initializer_t weight_init_type;
          if (!find_item_in_map(weight_init_type, weight_init_name, INITIALIZER_TYPE_MAP)){
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + weight_init_name);
          } else {
            initializer_types[0] = weight_init_type;
          }
        }
        if (has_key_(j_fc_param, "bias_init")){
          const auto bias_init_name = get_value_from_json<std::string>(j_fc_param, "bias_init");
          Initializer_t bias_init_type;
          if (!find_item_in_map(bias_init_type, bias_init_name, INITIALIZER_TYPE_MAP)){
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + bias_init_name);
          } else {
            initializer_types[1] = bias_init_type;
          }
        }
        // establish out tensor
        auto output = get_value_from_json<size_t>(j_fc_param, "num_output");
        if (use_mixed_precision) {
          std::shared_ptr<Tensor<__half>> out_tensor(new Tensor<__half>(
              {(fc_in_tensor->get_dims())[0], output}, blobs_buff_half, TensorFormat_t::HW));
          output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});

          // establish layer
          Layer* fc_layer = new FusedFullyConnectedLayer(
              weight_buff, weight_buff_half, wgrad_buff_half, blobs_buff, blobs_buff_half,
              dynamic_tensor_cast<__half>(fc_in_tensor), out_tensor,
              TensorFormat_t::HW, gpu_resource->get_cublas_handle(), device_id, initializer_types);
          layers.emplace_back(fc_layer);
        } else {
          CK_THROW_(Error_t::WrongInput, "FusedInnerProduct support half only");
        }
        break;
      }

      case Layer_t::Cast: {
        const auto& in_tensor = input_output_info.input[0];
        if (use_mixed_precision) {
          std::shared_ptr<Tensor<__half>> out_tensor(
              new Tensor<__half>(in_tensor->get_dims(), blobs_buff_half, TensorFormat_t::HW));
          output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});
          layers.emplace_back(new CastLayer(dynamic_tensor_cast<float>(in_tensor),
                                            out_tensor, device_id));
        } else {
          CK_THROW_(Error_t::WrongInput, "Cast supports half only");
        }
        break;
      }

      case Layer_t::InnerProduct: {
        const auto& fc_in_tensor = input_output_info.input[0];

        auto j_fc_param = get_json(j, "fc_param");
        // establish initializer
        std::vector<Initializer_t> initializer_types(2, Initializer_t::Default);
        if (has_key_(j_fc_param, "weight_init")){
          const auto weight_init_name = get_value_from_json<std::string>(j_fc_param, "weight_init");
          Initializer_t weight_init_type;
          if (!find_item_in_map(weight_init_type, weight_init_name, INITIALIZER_TYPE_MAP)){
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + weight_init_name);
          } else {
            initializer_types[0] = weight_init_type;
          }
        }
        if (has_key_(j_fc_param, "bias_init")){
          const auto bias_init_name = get_value_from_json<std::string>(j_fc_param, "bias_init");
          Initializer_t bias_init_type;
          if (!find_item_in_map(bias_init_type, bias_init_name, INITIALIZER_TYPE_MAP)){
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + bias_init_name);
          } else {
            initializer_types[1] = bias_init_type;
          }
        }

        // establish out tensor
        auto output = get_value_from_json<size_t>(j_fc_param, "num_output");

        if (use_mixed_precision) {
          std::shared_ptr<Tensor<__half>> out_tensor(new Tensor<__half>(
              {(fc_in_tensor->get_dims())[0], output}, blobs_buff_half, TensorFormat_t::HW));
          // establish layer
          Layer* fc_layer = new FullyConnectedLayerHalf(
              weight_buff, weight_buff_half, wgrad_buff_half, blobs_buff_half,
              dynamic_tensor_cast<__half>(fc_in_tensor), out_tensor,
              TensorFormat_t::HW, gpu_resource->get_cublas_handle(), device_id, initializer_types);
          layers.emplace_back(fc_layer);
          output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});
        } else {
          std::shared_ptr<Tensor<float>> out_tensor(new Tensor<float>(
              {(fc_in_tensor->get_dims())[0], output}, blobs_buff, TensorFormat_t::HW));
          // establish layer
          Layer* fc_layer = new FullyConnectedLayer(
              weight_buff, wgrad_buff, dynamic_tensor_cast<float>(fc_in_tensor),
              out_tensor, TensorFormat_t::HW, gpu_resource->get_cublas_handle(), device_id,
              use_mixed_precision, initializer_types);
          layers.emplace_back(fc_layer);
          output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});
        }
        break;
      }

      case Layer_t::Interaction: {
        // lambda template could be a better solution here, but there's not support in c++11
        auto input_output_info = get_input_tensor_and_output_name(j, tensor_list);
        auto& in_mlp_tensor = input_output_info.input[0];
        auto& in_emb_tensor = input_output_info.input[1];
        if (use_mixed_precision) {
          std::shared_ptr<Tensor<__half>> out_tensor;
          layers.emplace_back(new InteractionLayer<__half>(
	      dynamic_tensor_cast<__half>(in_mlp_tensor),
              dynamic_tensor_cast<__half>(in_emb_tensor), out_tensor,
              blobs_buff_half,  // todo cannot use this blobs_buff here need half
              gpu_resource->get_cublas_handle(), use_mixed_precision, device_id));
          output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});

        } else {
          std::shared_ptr<Tensor<float>> out_tensor;
          layers.emplace_back(new InteractionLayer<float>(
              dynamic_tensor_cast<float>(in_mlp_tensor),
              dynamic_tensor_cast<float>(in_emb_tensor), out_tensor, blobs_buff,
              gpu_resource->get_cublas_handle(), use_mixed_precision, device_id));
          output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});
        }

        break;
      }
      case Layer_t::MultiCross: {
        auto& mc_in_tensor = input_output_info.input[0];

        auto j_mc_param = get_json(j, "mc_param");
        // establish initializer
        std::vector<Initializer_t> initializer_types(2, Initializer_t::Default);
        if (has_key_(j_mc_param, "weight_init")){
          const auto weight_init_name = get_value_from_json<std::string>(j_mc_param, "weight_init");
          Initializer_t weight_init_type;
          if (!find_item_in_map(weight_init_type, weight_init_name, INITIALIZER_TYPE_MAP)){
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + weight_init_name);
          } else {
            initializer_types[0] = weight_init_type;
          }
        }
        if (has_key_(j_mc_param, "bias_init")){
          const auto bias_init_name = get_value_from_json<std::string>(j_mc_param, "bias_init");
          Initializer_t bias_init_type;
          if (!find_item_in_map(bias_init_type, bias_init_name, INITIALIZER_TYPE_MAP)){
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + bias_init_name);
          } else {
            initializer_types[1] = bias_init_type;
          }
        }

        // establish out tensor
        auto num_layers = get_value_from_json<int>(j_mc_param, "num_layers");
        std::shared_ptr<Tensor<float>> out_tensor(
            new Tensor<float>(mc_in_tensor->get_dims(), blobs_buff, TensorFormat_t::HW));
        output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});
        // establish layer
        Layer* mc_layer = new MultiCrossLayer(
            weight_buff, wgrad_buff, dynamic_tensor_cast<float>(mc_in_tensor),
            out_tensor, num_layers, device_id, initializer_types);
        layers.emplace_back(mc_layer);
        break;
      }

      case Layer_t::MultiCrossEntropyLoss: {
        if (input_output_info.input.size() != 2) {
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

        if (use_mixed_precision) {
          loss.reset(new MultiCrossEntropyLoss<__half>(
              dynamic_tensor_cast<float>(label_tensor),
              dynamic_tensor_cast<__half>(multi_cross_entropy_loss_in_tensor),
              loss_tensor,
              create_regularizer(j, weight_buff, wgrad_buff_half,
                                 (multi_cross_entropy_loss_in_tensor->get_dims())[0],
                                 gpu_resource->get_cublas_handle(), device_id),
              target_weight_vec, device_id, num_networks_in_global, scaler));
        } else {
          loss.reset(new MultiCrossEntropyLoss<float>(
              dynamic_tensor_cast<float>(label_tensor),
              dynamic_tensor_cast<float>(multi_cross_entropy_loss_in_tensor),
              loss_tensor,
              create_regularizer(j, weight_buff, wgrad_buff,
                                 (multi_cross_entropy_loss_in_tensor->get_dims())[0],
                                 gpu_resource->get_cublas_handle(), device_id),
              target_weight_vec, device_id, num_networks_in_global, scaler));
        }
        break;
      }
      case Layer_t::ReLU: {
        const auto& relu_in_tensor = input_output_info.input[0];

        if (use_mixed_precision) {
          std::shared_ptr<Tensor<__half>> relu_out_tensor(
              new Tensor<__half>(relu_in_tensor->get_dims(), blobs_buff_half, TensorFormat_t::HW));
          layers.emplace_back(
              new ReluLayer<__half>(std::dynamic_pointer_cast<Tensor<__half>>(relu_in_tensor),
                                relu_out_tensor, device_id));
          output_tensor_pairs.push_back({relu_out_tensor, input_output_info.output[0]});
        } else {
          // establish out tensor
          std::shared_ptr<Tensor<float>> relu_out_tensor(
              new Tensor<float>(relu_in_tensor->get_dims(), blobs_buff, TensorFormat_t::HW));
          layers.emplace_back(
              new ReluLayer<float>(std::dynamic_pointer_cast<Tensor<float>>(relu_in_tensor),
                            relu_out_tensor, device_id));
          output_tensor_pairs.push_back({relu_out_tensor, input_output_info.output[0]});
        }

        break;
      }
      case Layer_t::Reshape: {
        const auto& in_tensor = input_output_info.input[0];

        auto selected_it = j.find("selected");
        // selective reshape
        if (selected_it != j.end()) {
          std::vector<int> selected;
          nlohmann::json j_selected = (selected_it.value());
          for (auto slot_obj : j_selected) {
            int slot_id = slot_obj.get<int>();
            if (slot_id < 0) CK_THROW_(Error_t::WrongInput, "slot_id < 0");
            selected.push_back(slot_id);
          }

          if (use_mixed_precision) {
            std::shared_ptr<Tensor<__half>> out_tensor;
            layers.emplace_back(new ReshapeLayer<__half>(dynamic_tensor_cast<__half>(in_tensor),
                                                 out_tensor, blobs_buff_half, selected, device_id));
            output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});
          } else {
            std::shared_ptr<Tensor<float>> out_tensor;
            layers.emplace_back(new ReshapeLayer<float>(dynamic_tensor_cast<float>(in_tensor),
                                                 out_tensor, blobs_buff, selected, device_id));
            output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});
          }
        }
        // general purpose reshape
        else {
          auto leading_dim_it = j.find("leading_dim");
          auto in_dims = in_tensor->get_dims();
          // if leading_dim is not specified, default leading_dim = n_slots * vector_length
          int leading_dim = (leading_dim_it != j.end())
                                ? (*leading_dim_it).get<int>()
                                : in_tensor->get_num_elements() / in_dims[0];
          if (use_mixed_precision) {
            std::shared_ptr<Tensor<__half>> out_tensor;
            layers.emplace_back(new ReshapeLayer<__half>(dynamic_tensor_cast<__half>(in_tensor),
                                                 out_tensor, leading_dim, device_id));
            output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});
          } else {
            std::shared_ptr<Tensor<float>> out_tensor;
            layers.emplace_back(new ReshapeLayer<float>(dynamic_tensor_cast<float>(in_tensor),
                                                 out_tensor, leading_dim, device_id));
            output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});
          }
        }
        break;
      }
      case Layer_t::Slice: {
        const auto& in_tensor = input_output_info.input[0];

        std::vector<std::pair<int, int>> ranges;
        auto j_ranges = get_json(j, "ranges");
        assert(j_ranges.is_array());
        for (auto j_range : j_ranges) {
          assert(j_range.is_array());
          ranges.emplace_back(std::make_pair(j_range[0].get<int>(), j_range[1].get<int>()));
        }

        if (use_mixed_precision) {
          Tensors<__half> out_tensors;
          layers.emplace_back(new SliceLayer<__half>(dynamic_tensor_cast<__half>(in_tensor),
                                             out_tensors, blobs_buff_half, ranges, device_id));
          for (size_t i = 0; i < out_tensors.size(); i++) {
            output_tensor_pairs.push_back({out_tensors[i], input_output_info.output[i]});
          }
        } else {
          Tensors<float> out_tensors;
          layers.emplace_back(new SliceLayer<float>(dynamic_tensor_cast<float>(in_tensor),
                                             out_tensors, blobs_buff, ranges, device_id));
          for (size_t i = 0; i < out_tensors.size(); i++) {
            output_tensor_pairs.push_back({out_tensors[i], input_output_info.output[i]});
          }

        }
        break;
      }
      case Layer_t::Multiply: {
        const auto& in_tensor = input_output_info.input[0];

        std::vector<size_t> weight_dims;
        auto dims = get_json(j, "weight_dims");
        assert(dims.is_array());
        for (auto dim : dims) {
          weight_dims.emplace_back(dim.get<size_t>());
        }

        // establish initializer
        std::vector<Initializer_t> initializer_types(1, Initializer_t::Default);
        if (has_key_(j, "weight_init")){
          const auto weight_init_name = get_value_from_json<std::string>(j, "weight_init");
          Initializer_t weight_init_type;
          if (!find_item_in_map(weight_init_type, weight_init_name, INITIALIZER_TYPE_MAP)){
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + weight_init_name);
          } else {
            initializer_types[0] = weight_init_type;
          }
        }

        std::shared_ptr<Tensor<float>> out_tensor;
        Layer* mul_layer = new MultiplyLayer(weight_buff, wgrad_buff, blobs_buff,
                                              dynamic_tensor_cast<float>(in_tensor),
                                              out_tensor, weight_dims, device_id, initializer_types);
        layers.emplace_back(mul_layer);
        output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});
        break;
      }
      case Layer_t::FmOrder2: {
        const auto& in_tensor = input_output_info.input[0];
        auto out_dim = get_json(j, "out_dim").get<size_t>();

        // std::shared_ptr<Tensor<float>> out_tensor(
        //     new Tensor<float>({batch_size, out_dim}, blobs_buff, TensorFormat_t::HW));
        std::shared_ptr<Tensor<float>> out_tensor(new Tensor<float>(
            {(in_tensor->get_dims())[0], out_dim}, blobs_buff, TensorFormat_t::HW));

        layers.emplace_back(new FmOrder2Layer(dynamic_tensor_cast<float>(in_tensor),
                                              out_tensor, device_id));
        output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});
        break;
      }
      case Layer_t::Add: {
        auto& in_tensors = input_output_info.input;
        std::shared_ptr<Tensor<float>> out_tensor(
            new Tensor<float>(in_tensors[0]->get_dims(), blobs_buff, in_tensors[0]->get_format()));
        layers.emplace_back(new AddLayer(tensor_vec_dynamic_cast<float>(in_tensors),
                                         out_tensor, device_id));
        output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});
        break;
      }
      case Layer_t::ReduceSum: {
        auto& in_tensor = input_output_info.input[0];
        std::shared_ptr<Tensor<float>> out_tensor;
        int axis = get_json(j, "axis").get<int>();
        layers.emplace_back(new ReduceSumLayer(dynamic_tensor_cast<float>(in_tensor),
                                               out_tensor, blobs_buff, axis, device_id));
        output_tensor_pairs.push_back({out_tensor, input_output_info.output[0]});
        break;
      }
      default:
        assert(!"Error: no such layer && should never get here!");
    }  // end of switch

    if (!(layer_type == Layer_t::CrossEntropyLoss ||
          layer_type == Layer_t::BinaryCrossEntropyLoss ||
          layer_type == Layer_t::MultiCrossEntropyLoss)) {
      for (auto& output_tensor_pair : output_tensor_pairs) {
        add_tensor_to_network(output_tensor_pair, tensor_list);
      }
    } else {
      network->raw_metrics_[metrics::RawType::Loss] = loss_tensor;
      network->raw_metrics_[metrics::RawType::Pred] = input_output_info.input[0];
      network->raw_metrics_[metrics::RawType::Label] = input_output_info.input[1];
    }
  }  // for layers

  // create optimizer
  auto opt_param = get_optimizer_param<float>(j_optimizer);

  switch (static_cast<Optimizer_t>(opt_param.optimizer)) {
    case Optimizer_t::Adam: {
      auto alpha = opt_param.lr;
      auto beta1 = opt_param.hyperparams.adam.beta1;
      auto beta2 = opt_param.hyperparams.adam.beta2;
      auto epsilon = opt_param.hyperparams.adam.epsilon;
      network->optimizer_.reset(new AdamOptimizer(weight_buff, wgrad_buff, device_id, alpha, beta1,
                                                  beta2, epsilon, scaler));
      break;
    }
    case Optimizer_t::MomentumSGD: {
      auto learning_rate = opt_param.lr;
      auto momentum_factor = opt_param.hyperparams.momentum.factor;
      network->optimizer_.reset(new MomentumSGD(weight_buff, wgrad_buff, device_id, learning_rate,
                                                momentum_factor, scaler));
      break;
    }
    case Optimizer_t::Nesterov: {
      auto learning_rate = opt_param.lr;
      auto momentum_factor = opt_param.hyperparams.nesterov.mu;
      network->optimizer_.reset(new NesterovOptimizer(weight_buff, wgrad_buff, device_id,
                                                      learning_rate, momentum_factor, scaler));
      break;
    }
    case Optimizer_t::SGD: {
      auto learning_rate = opt_param.lr;
      if (use_mixed_precision) {
        network->optimizer_.reset(new SgdOptimizer<__half>(
            weight_buff, wgrad_buff_half, weight_buff_half, device_id, learning_rate, scaler));
      } else {
        network->optimizer_.reset(
            new SgdOptimizer<float>(weight_buff, wgrad_buff, nullptr, device_id, learning_rate, scaler));
      }
      break;
    }
    default:
      assert(!"Error: no such optimizer && should never get here!");
  }
  weight_buff->init(device_id);
  wgrad_buff->init(device_id);
  blobs_buff->init(device_id);
  if (use_mixed_precision) {
    weight_buff_half->init(device_id);
    wgrad_buff_half->init(device_id);
    blobs_buff_half->init(device_id);
  }

#ifndef DATA_READING_TEST
  network->optimize();
#endif

  return network.release();
}

template <typename TypeKey>
static void create_pipeline_internal(std::unique_ptr<DataReader<TypeKey>>& data_reader,
                                     std::unique_ptr<DataReader<TypeKey>>& data_reader_eval,
                                     std::vector<std::unique_ptr<IEmbedding>>& embedding,
                                     std::vector<std::unique_ptr<IEmbedding>>& embedding_eval,
                                     std::vector<std::unique_ptr<Network>>& network,
                                     std::vector<std::unique_ptr<Network>>& network_eval,
                                     const std::shared_ptr<GPUResourceGroup>& gpu_resource_group,
                                     nlohmann::json config, size_t batch_size,
                                     size_t batch_size_eval, bool use_mixed_precision,
                                     float scaler) {
  try {
#ifdef ENABLE_MPI
    int num_procs = 1, pid = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif

    std::map<std::string, SparseInput<TypeKey>> sparse_input_map;
    std::map<std::string, std::shared_ptr<ITensor>> tensor_maps[gpu_resource_group->size()];
    std::map<std::string, std::shared_ptr<ITensor>> tensor_maps_eval[gpu_resource_group->size()];
    {
      if (!network.empty()) {
        CK_THROW_(Error_t::WrongInput, "vector network is not empty");
      }

      auto j_layers_array = get_json(config, "layers");
      auto j_optimizer = get_json(config, "optimizer");

      // Create Data Reader
      {
        const nlohmann::json& j = j_layers_array[0];
        const auto layer_type_name = get_value_from_json<std::string>(j, "type");
        if (layer_type_name.compare("Data") != 0) {
          CK_THROW_(Error_t::WrongInput, "the first layer is not Data layer:" + layer_type_name);
        }


	
        const std::map<std::string, DataReaderType_t> DATA_READER_MAP = {
            {"Norm", DataReaderType_t::Norm}, {"Raw", DataReaderType_t::Raw}};

        DataReaderType_t format = DataReaderType_t::Norm;
        if (has_key_(j, "format")) {
          const auto data_format_name = get_value_from_json<std::string>(j, "format");
          if (!find_item_in_map(format, data_format_name, DATA_READER_MAP)) {
            CK_THROW_(Error_t::WrongInput, "No such data format: " + data_format_name);
          }
        }

        auto cache_eval_data = get_value_from_json_soft<bool>(j, "cache_eval_data", false);

        std::string source_data = get_value_from_json<std::string>(j, "source");

        auto j_label = get_json(j, "label");
        auto top_strs_label = get_value_from_json<std::string>(j_label, "top");
        auto label_dim = get_value_from_json<int>(j_label, "label_dim");

        auto j_dense = get_json(j, "dense");
        auto top_strs_dense = get_value_from_json<std::string>(j_dense, "top");
        auto dense_dim = get_value_from_json<int>(j_dense, "dense_dim");

        const std::map<std::string, Check_t> CHECK_TYPE_MAP = {{"Sum", Check_t::Sum},
                                                               {"None", Check_t::None}};

        Check_t check_type;
        const auto check_str = get_value_from_json<std::string>(j, "check");
        if (!find_item_in_map(check_type, check_str, CHECK_TYPE_MAP)) {
          CK_THROW_(Error_t::WrongInput, "Not supported check type: " + check_str);
        }

        std::vector<DataReaderSparseParam> data_reader_sparse_param_array;

        const std::map<std::string, DataReaderSparse_t> DATA_TYPE_MAP = {
            {"DistributedSlot", DataReaderSparse_t::Distributed},
            {"LocalizedSlot", DataReaderSparse_t::Localized},
        };

        auto j_sparse = get_json(j, "sparse");
        std::vector<std::string> sparse_names;

        for (unsigned int i = 0; i < j_sparse.size(); i++) {
          DataReaderSparseParam param;

          const nlohmann::json& js = j_sparse[i];
          const auto sparse_name = get_value_from_json<std::string>(js, "top");
          const auto data_type_name = get_value_from_json<std::string>(js, "type");
          if (!find_item_in_map(param.type, data_type_name, DATA_TYPE_MAP)) {
            CK_THROW_(Error_t::WrongInput, "Not supported data type: " + data_type_name);
          }
          param.max_feature_num = get_value_from_json<int>(js, "max_feature_num_per_sample");
          param.max_nnz = get_value_from_json_soft<int>(js, "max_nnz", param.max_feature_num);
          param.slot_num = get_value_from_json<int>(js, "slot_num");
          data_reader_sparse_param_array.push_back(param);
          SparseInput<TypeKey> sparse_input(param.slot_num, param.max_feature_num);
          sparse_input_map.emplace(sparse_name, sparse_input);
          sparse_names.push_back(sparse_name);
        }

        data_reader_eval = nullptr;
        std::string eval_source;
        FIND_AND_ASSIGN_STRING_KEY(eval_source, j);

        switch (format) {
          case DataReaderType_t::Norm: {
#ifdef VAL
            data_reader.reset(new DataReader<TypeKey>(source_data, batch_size, label_dim, dense_dim,
                                                      check_type, data_reader_sparse_param_array,
                                                      gpu_resource_group, 1));
#else
            data_reader.reset(new DataReader<TypeKey>(source_data, batch_size, label_dim, dense_dim,
                                                      check_type, data_reader_sparse_param_array,
                                                      gpu_resource_group));

#endif

#ifdef VAL
            data_reader_eval.reset(new DataReader<TypeKey>(
                eval_source, batch_size_eval, label_dim, dense_dim, check_type,
                data_reader_sparse_param_array, gpu_resource_group, 1));
#else
            data_reader_eval.reset(new DataReader<TypeKey>(
                eval_source, batch_size_eval, label_dim, dense_dim, check_type,
                data_reader_sparse_param_array, gpu_resource_group));

#endif

            //            }

            break;
          }
          case DataReaderType_t::Raw: {
            const auto num_samples = get_value_from_json<long long>(j, "num_samples");
            const auto eval_num_samples = get_value_from_json<long long>(j, "eval_num_samples");
            std::vector<long long> slot_offset;
            if (has_key_(j, "slot_size_array")) {
              auto slot_size_array = get_json(j, "slot_size_array");
              if (!slot_size_array.is_array()) {
                CK_THROW_(Error_t::WrongInput, "!slot_size_array.is_array()");
              }
              long long slot_sum = 0;
              for (auto j_slot_size : slot_size_array) {
                slot_offset.push_back(slot_sum);
                long long slot_size = j_slot_size.get<long long>();
                slot_sum += slot_size;
              }
              MESSAGE_("Vocabulary size: " + std::to_string(slot_sum));
            }

#ifdef VAL
            data_reader.reset(new DataReader<TypeKey>(source_data, batch_size, label_dim, dense_dim,
                                                      check_type, data_reader_sparse_param_array,
                                                      gpu_resource_group, 1, format, num_samples,
                                                      slot_offset, false, false, true));
#else
            data_reader.reset(new DataReader<TypeKey>(source_data, batch_size, label_dim, dense_dim,
                                                      check_type, data_reader_sparse_param_array,
                                                      gpu_resource_group, 12, format, num_samples,
                                                      slot_offset, false, false, true));

#endif

            //            if (eval_source.empty() == false) {
            // data_reader_eval.reset(
            //     data_reader->clone_eval_with_shared_output(eval_source, eval_num_samples));
#ifdef VAL
            data_reader_eval.reset(new DataReader<TypeKey>(
                eval_source, batch_size_eval, label_dim, dense_dim, check_type,
                data_reader_sparse_param_array, gpu_resource_group, 1, format, eval_num_samples,
                slot_offset, cache_eval_data, false, false));
#else
            data_reader_eval.reset(new DataReader<TypeKey>(
                eval_source, batch_size_eval, label_dim, dense_dim, check_type,
                data_reader_sparse_param_array, gpu_resource_group, 12, format, eval_num_samples,
                slot_offset, cache_eval_data, false, false));

#endif

            //            }

            break;
          }
          default: { assert(!"Error: no such option && should never get here!"); }
        }

        for (unsigned int i = 0; i < gpu_resource_group->size(); i++) {
          tensor_maps[i].emplace(top_strs_label, (data_reader->get_label_tensors())[i]);
          tensor_maps[i].emplace(top_strs_dense, (data_reader->get_dense_tensors())[i]);

          // todo should be replaced to data_reader_eval then
          tensor_maps_eval[i].emplace(top_strs_label, (data_reader_eval->get_label_tensors())[i]);
          tensor_maps_eval[i].emplace(top_strs_dense, (data_reader_eval->get_dense_tensors())[i]);
        }

        for (unsigned int i = 0; i < j_sparse.size(); i++) {
          const auto& sparse_input = sparse_input_map.find(sparse_names[i]);
          sparse_input->second.row = data_reader->get_row_offsets_tensors(i);
          sparse_input->second.value = data_reader->get_value_tensors(i);
          sparse_input->second.row_eval = data_reader_eval->get_row_offsets_tensors(i);
          sparse_input->second.value_eval = data_reader_eval->get_value_tensors(i);
        }
      }

      // Create Embedding
      {
        for (unsigned int i = 1; i < j_layers_array.size(); i++) {
          // if not embedding then break
          const nlohmann::json& j = j_layers_array[i];
          auto embedding_name = get_value_from_json<std::string>(j, "type");
          Embedding_t embedding_type;
          if (!find_item_in_map(embedding_type, embedding_name, EMBEDDING_TYPE_MAP)) {
            Layer_t layer_type;
            if (!find_item_in_map(layer_type, embedding_name, LAYER_TYPE_MAP) &&
                !find_item_in_map(layer_type, embedding_name, LAYER_TYPE_MAP_MP)) {
              CK_THROW_(Error_t::WrongInput, "No such layer: " + embedding_name);
            }
            break;
          }

          auto bottom_name = get_value_from_json<std::string>(j, "bottom");
          auto top_name = get_value_from_json<std::string>(j, "top");

          auto j_hparam = get_json(j, "sparse_embedding_hparam");
          size_t max_vocabulary_size = 0;
          size_t max_vocabulary_size_per_gpu = 0;
          if(embedding_type == Embedding_t::DistributedSlotSparseEmbeddingHash) {
            max_vocabulary_size = get_value_from_json<size_t>(j_hparam, "max_vocabulary_size");
          } else if(embedding_type == Embedding_t::LocalizedSlotSparseEmbeddingHash) {
	    if (has_key_(j_hparam, "max_vocabulary_size_per_gpu")) {
	      max_vocabulary_size_per_gpu = get_value_from_json<size_t>(j_hparam, "max_vocabulary_size_per_gpu");
	    }
	    else if(!has_key_(j_hparam, "slot_size_array")){
	      CK_THROW_(Error_t::WrongInput, "No max_vocabulary_size_per_gpu or slot_size_array in: " + embedding_name);
	    }
          }
          auto embedding_vec_size = get_value_from_json<size_t>(j_hparam, "embedding_vec_size");
          auto combiner = get_value_from_json<int>(j_hparam, "combiner");

          SparseInput<TypeKey> sparse_input;
          if (!find_item_in_map(sparse_input, bottom_name, sparse_input_map)) {
            CK_THROW_(Error_t::WrongInput, "Cannot find bottom");
          }

          if (use_mixed_precision) {
            OptParams<__half> embedding_opt_params;
            if (has_key_(j, "optimizer")) {
              embedding_opt_params = get_optimizer_param<__half>(get_json(j, "optimizer"));
            } else {
              embedding_opt_params = get_optimizer_param<__half>(j_optimizer);
            }
            embedding_opt_params.scaler = scaler;

            switch (embedding_type) {
              case Embedding_t::DistributedSlotSparseEmbeddingHash: {
                const SparseEmbeddingHashParams<__half> embedding_params = {
                    batch_size,
                    max_vocabulary_size,
                    0,
                    {},
                    embedding_vec_size,
                    sparse_input.max_feature_num_per_sample,
                    sparse_input.slot_num,
                    combiner,  // combiner: 0-sum, 1-mean
                    embedding_opt_params};
                auto emb_ptr = EmbeddingCreator::create_distributed_sparse_embedding_hash(
                    sparse_input.row, sparse_input.value, embedding_params, gpu_resource_group);
                embedding.emplace_back(emb_ptr);
                // todo:
                embedding_eval.emplace_back(
                    EmbeddingCreator::clone_eval(sparse_input.row_eval, sparse_input.value_eval,
                                                 batch_size_eval, gpu_resource_group, emb_ptr));

                break;
              }
              case Embedding_t::LocalizedSlotSparseEmbeddingHash: {
#ifndef NCCL_A2A
                auto j_plan = get_json(j, "plan_file");
                std::string plan_file;
                if (j_plan.is_array()) {
                  int num_nodes = j_plan.size();
                  if (num_nodes != num_procs) {
                    CK_THROW_(Error_t::WrongInput, "num_nodes != num_procs");
                  }
                  plan_file = j_plan[pid].get<std::string>();
                } else {
                  if (num_procs > 1) {
                    CK_THROW_(Error_t::WrongInput, "num_procs > 1");
                  }
                  plan_file = get_value_from_json<std::string>(j, "plan_file");
                }

                std::ifstream ifs(plan_file);
                if (!ifs) {
                  CK_THROW_(Error_t::WrongInput, "plan file " + plan_file + " can bot be open");
                }
#else
                std::string plan_file = "";
#endif
                std::vector<size_t> slot_size_array;
                if (has_key_(j_hparam, "slot_size_array")) {
                  auto slots = get_json(j_hparam, "slot_size_array");
                  assert(slots.is_array());
                  for (auto slot : slots) {
                    slot_size_array.emplace_back(slot.get<size_t>());
                  }
                }

                const SparseEmbeddingHashParams<__half> embedding_params = {
                    batch_size,
                    0,
                    max_vocabulary_size_per_gpu,
                    slot_size_array,
                    embedding_vec_size,
                    sparse_input.max_feature_num_per_sample,
                    sparse_input.slot_num,
                    combiner,  // combiner: 0-sum, 1-mean
                    embedding_opt_params};
                auto emb_ptr = EmbeddingCreator::create_localized_sparse_embedding_hash(
                    sparse_input.row, sparse_input.value, embedding_params, plan_file,
                    gpu_resource_group);
                embedding.emplace_back(emb_ptr);
                embedding_eval.emplace_back(
                    EmbeddingCreator::clone_eval(sparse_input.row_eval, sparse_input.value_eval,
                                                 batch_size_eval, gpu_resource_group, emb_ptr));

                break;
              }
              case Embedding_t::LocalizedSlotSparseEmbeddingOneHot: {
                std::string plan_file = "";
                std::vector<size_t> slot_size_array;
                auto slots = get_json(j_hparam, "slot_size_array");
                assert(slots.is_array());
                for (auto slot : slots) {
                  slot_size_array.emplace_back(slot.get<size_t>());
                }

                const SparseEmbeddingHashParams<__half> embedding_params = {
                    batch_size,
                    0,
                    0,
                    slot_size_array,
                    embedding_vec_size,
                    sparse_input.max_feature_num_per_sample,
                    sparse_input.slot_num,
                    combiner,  // combiner: 0-sum, 1-mean
                    embedding_opt_params};
                auto emb_ptr = EmbeddingCreator::create_localized_sparse_embedding_one_hot(
                    sparse_input.row, sparse_input.value, embedding_params, plan_file,
                    gpu_resource_group);
                embedding.emplace_back(emb_ptr);
                embedding_eval.emplace_back(
                    EmbeddingCreator::clone_eval(sparse_input.row_eval, sparse_input.value_eval,
                                                 batch_size_eval, gpu_resource_group, emb_ptr));

                break;
              }
            }  // switch
            for (unsigned int i = 0; i < gpu_resource_group->size(); i++) {
              tensor_maps[i].emplace(top_name, (embedding.back()->get_output_tensors())[i]);
              tensor_maps_eval[i].emplace(top_name,
                                          (embedding_eval.back()->get_output_tensors())[i]);
            }

          }  //	  if(use_mixed_precision)
          else {
            OptParams<float> embedding_opt_params;
            if (has_key_(j, "optimizer")) {
              embedding_opt_params = get_optimizer_param<float>(get_json(j, "optimizer"));
            } else {
              embedding_opt_params = get_optimizer_param<float>(j_optimizer);
            }
            embedding_opt_params.scaler = scaler;

            switch (embedding_type) {
              case Embedding_t::DistributedSlotSparseEmbeddingHash: {
                const SparseEmbeddingHashParams<float> embedding_params = {
                    batch_size,
                    max_vocabulary_size,
                    0,
                    {},
                    embedding_vec_size,
                    sparse_input.max_feature_num_per_sample,
                    sparse_input.slot_num,
                    combiner,  // combiner: 0-sum, 1-mean
                    embedding_opt_params};
                auto emb_ptr = EmbeddingCreator::create_distributed_sparse_embedding_hash(
                    sparse_input.row, sparse_input.value, embedding_params, gpu_resource_group);
                embedding.emplace_back(emb_ptr);
                embedding_eval.emplace_back(
                    EmbeddingCreator::clone_eval(sparse_input.row_eval, sparse_input.value_eval,
                                                 batch_size_eval, gpu_resource_group, emb_ptr));

                break;
              }
              case Embedding_t::LocalizedSlotSparseEmbeddingHash: {
#ifndef NCCL_A2A
                auto j_plan = get_json(j, "plan_file");
                std::string plan_file;
                if (j_plan.is_array()) {
                  int num_nodes = j_plan.size();
                  if (num_nodes != num_procs) {
                    CK_THROW_(Error_t::WrongInput, "num_nodes != num_procs");
                  }
                  plan_file = j_plan[pid].get<std::string>();
                } else {
                  if (num_procs > 1) {
                    CK_THROW_(Error_t::WrongInput, "num_procs > 1");
                  }
                  plan_file = get_value_from_json<std::string>(j, "plan_file");
                }

                std::ifstream ifs(plan_file);
                if (!ifs) {
                  CK_THROW_(Error_t::WrongInput, "plan file " + plan_file + " can bot be open");
                }
#else
                std::string plan_file = "";
#endif
                std::vector<size_t> slot_size_array;
                if (has_key_(j_hparam, "slot_size_array")) {
                  auto slots = get_json(j_hparam, "slot_size_array");
                  assert(slots.is_array());
                  for (auto slot : slots) {
                    slot_size_array.emplace_back(slot.get<size_t>());
                  }
                }

                const SparseEmbeddingHashParams<float> embedding_params = {
                    batch_size,
                    0,
                    max_vocabulary_size_per_gpu,
                    slot_size_array,
                    embedding_vec_size,
                    sparse_input.max_feature_num_per_sample,
                    sparse_input.slot_num,
                    combiner,  // combiner: 0-sum, 1-mean
                    embedding_opt_params};
                auto emb_ptr = EmbeddingCreator::create_localized_sparse_embedding_hash(
                    sparse_input.row, sparse_input.value, embedding_params, plan_file,
                    gpu_resource_group);
                embedding.emplace_back(emb_ptr);
                embedding_eval.emplace_back(
                    EmbeddingCreator::clone_eval(sparse_input.row_eval, sparse_input.value_eval,
                                                 batch_size_eval, gpu_resource_group, emb_ptr));

                break;
              }
              case Embedding_t::LocalizedSlotSparseEmbeddingOneHot: {
                std::string plan_file = "";
                std::vector<size_t> slot_size_array;
                auto slots = get_json(j_hparam, "slot_size_array");
                assert(slots.is_array());
                for (auto slot : slots) {
                  slot_size_array.emplace_back(slot.get<size_t>());
                }

                const SparseEmbeddingHashParams<float> embedding_params = {
                    batch_size,
                    0,
                    0,
                    slot_size_array,
                    embedding_vec_size,
                    sparse_input.max_feature_num_per_sample,
                    sparse_input.slot_num,
                    combiner,  // combiner: 0-sum, 1-mean
                    embedding_opt_params};
                auto emb_ptr = EmbeddingCreator::create_localized_sparse_embedding_one_hot(
                    sparse_input.row, sparse_input.value, embedding_params, plan_file,
                    gpu_resource_group);
                embedding.emplace_back(emb_ptr);
                embedding_eval.emplace_back(
                    EmbeddingCreator::clone_eval(sparse_input.row_eval, sparse_input.value_eval,
                                                 batch_size_eval, gpu_resource_group, emb_ptr));

                break;
              }
            }  // switch
            for (unsigned int i = 0; i < gpu_resource_group->size(); i++) {
              tensor_maps[i].emplace(top_name, (embedding.back()->get_output_tensors())[i]);

              // should be replaced with embedding_eval then
              // tensor_maps_eval[i].emplace(top_name, (embedding.back()->get_output_tensors())[i]);
              tensor_maps_eval[i].emplace(top_name,
                                          (embedding_eval.back()->get_output_tensors())[i]);
            }
          }  //    if(!use_mixed_precision)
        }    // for ()
      }      // Create Embedding

      // create network
      int i = 0;
      int total_gpu_count = gpu_resource_group->get_total_gpu_count();
      if (0 != batch_size % total_gpu_count) {
        CK_THROW_(Error_t::WrongInput, "0 != batch_size\%total_gpu_count");
      }
      const auto& device_list = gpu_resource_group->get_device_list();
      for (auto device_id : device_list) {
        network.emplace_back(create_network(j_layers_array, j_optimizer, tensor_maps[i], device_id,
                                            total_gpu_count, (*gpu_resource_group)[i],
                                            use_mixed_precision, scaler));
        network_eval.emplace_back(
            create_network(j_layers_array, j_optimizer, tensor_maps_eval[i], device_id,
                           total_gpu_count, (*gpu_resource_group)[i], use_mixed_precision, scaler));
        if (use_mixed_precision) {
          network_eval.back()->set_weight_half(network.back()->get_weight_half());
        }
        network_eval.back()->set_weight(
            network.back()->get_weight());  // to do: should be unnecessary
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
                             std::vector<std::unique_ptr<IEmbedding>>& embedding,
                             std::vector<std::unique_ptr<IEmbedding>>& embedding_eval,
                             std::vector<std::unique_ptr<Network>>& network,
                             std::vector<std::unique_ptr<Network>>& network_eval,
                             const std::shared_ptr<GPUResourceGroup>& gpu_resource_group) {
  create_pipeline_internal<TYPE_1>(data_reader, data_reader_eval, embedding, embedding_eval,
                                   network, network_eval, gpu_resource_group, config_, batch_size_,
                                   batch_size_eval_, use_mixed_precision_, scaler_);
}

void Parser::create_pipeline(std::unique_ptr<DataReader<TYPE_2>>& data_reader,
                             std::unique_ptr<DataReader<TYPE_2>>& data_reader_eval,
                             std::vector<std::unique_ptr<IEmbedding>>& embedding,
                             std::vector<std::unique_ptr<IEmbedding>>& embedding_eval,
                             std::vector<std::unique_ptr<Network>>& network,
                             std::vector<std::unique_ptr<Network>>& network_eval,
                             const std::shared_ptr<GPUResourceGroup>& gpu_resource_group) {
  create_pipeline_internal<TYPE_2>(data_reader, data_reader_eval, embedding, embedding_eval,
                                   network, network_eval, gpu_resource_group, config_, batch_size_,
                                   batch_size_eval_, use_mixed_precision_, scaler_);
}

}  // namespace HugeCTR
