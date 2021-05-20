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

#include <data_readers/async_reader/async_reader_adapter.hpp>
#include <data_readers/data_reader.hpp>
#include <embeddings/distributed_slot_sparse_embedding_hash.hpp>
#include <embeddings/hybrid_sparse_embedding.hpp>
#include <embeddings/localized_slot_sparse_embedding_hash.hpp>
#include <embeddings/localized_slot_sparse_embedding_one_hot.hpp>
#include <layer.hpp>
#include <layers/add_layer.hpp>
#include <layers/batch_norm_layer.hpp>
#include <layers/cast_layer.hpp>
#include <layers/concat_layer.hpp>
#include <layers/dot_product_layer.hpp>
#include <layers/dropout_cudnn_layer.hpp>
#include <layers/dropout_layer.hpp>
#include <layers/elu_layer.hpp>
#include <layers/fm_order2_layer.hpp>
#include <layers/fully_connected_layer.hpp>
#include <layers/fully_connected_layer_half.hpp>
#include <layers/fused_fully_connected_layer.hpp>
#include <layers/fused_relu_bias_fully_connected_layer.hpp>
#include <layers/interaction_layer.hpp>
#include <layers/multi_cross_layer.hpp>
#include <layers/multiply_layer.hpp>
#include <layers/reduce_sum_layer.hpp>
#include <layers/relu_layer.hpp>
#include <layers/reshape_layer.hpp>
#include <layers/slice_layer.hpp>
#include <loss.hpp>
#include <metrics.hpp>
#include <optimizer.hpp>
#include <parser.hpp>
#include <regularizers/l1_regularizer.hpp>
#include <regularizers/l2_regularizer.hpp>
#include <regularizers/no_regularizer.hpp>
#include <exchange_wgrad.hpp>

#include "common.hpp"

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {

struct InputOutputInfo {
  std::vector<TensorBag2> inputs;
  std::vector<std::string> output_names;
};

static bool get_tensor_from_entries(const std::vector<TensorEntry> tensor_entries,
                                    const std::string& name, TensorBag2* bag) {
  for (const TensorEntry& entry : tensor_entries) {
    if (entry.name == name) {
      *bag = entry.bag;
      return true;
    }
  }
  return false;
}

static std::vector<std::string> get_layer_names(const nlohmann::json& json) {
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

static InputOutputInfo get_input_tensor_and_output_name(
    const nlohmann::json& json, const std::vector<TensorEntry>& tensor_entries) {
  auto bottom = get_json(json, "bottom");
  auto top = get_json(json, "top");

  std::vector<std::string> bottom_names = get_layer_names(bottom);
  std::vector<std::string> top_names = get_layer_names(top);

  std::vector<TensorBag2> bottom_bags;

  for (auto& bottom_name : bottom_names) {
    for (auto& top_name : top_names) {
      if (bottom_name == top_name) {
        CK_THROW_(Error_t::WrongInput, "bottom and top include a same layer name");
      }
    }
    TensorBag2 bag;
    if (!get_tensor_from_entries(tensor_entries, bottom_name, &bag)) {
      CK_THROW_(Error_t::WrongInput, "No such bottom: " + bottom_name);
    }
    bottom_bags.push_back(bag);
  }
  return {bottom_bags, top_names};
}

template <typename Type>
static OptParams<Type> get_optimizer_param(const nlohmann::json& j_optimizer) {
  // create optimizer
  auto optimizer_name = get_value_from_json<std::string>(j_optimizer, "type");
  Optimizer_t optimizer_type;
  if (!find_item_in_map(optimizer_type, optimizer_name, OPTIMIZER_TYPE_MAP)) {
    CK_THROW_(Error_t::WrongInput, "No such optimizer: " + optimizer_name);
  }

  OptHyperParams<Type> opt_hyper_params;
  memset((void*)&opt_hyper_params, 0, sizeof(opt_hyper_params));
  OptParams<Type> opt_params;

  Update_t update_type = Update_t::Local;
  if (has_key_(j_optimizer, "update_type")) {
    std::string update_name = get_value_from_json<std::string>(j_optimizer, "update_type");
    if (!find_item_in_map(update_type, update_name, UPDATE_TYPE_MAP)) {
      CK_THROW_(Error_t::WrongInput, "No such update type: " + update_name);
    }
  } else if (has_key_(j_optimizer, "global_update")) {
    bool global_update = get_value_from_json<bool>(j_optimizer, "global_update");
    if (global_update) update_type = Update_t::Global;
  } else {
    MESSAGE_("update_type is not specified, using default: local");
  }

  switch (optimizer_type) {
    case Optimizer_t::Adam: {
      auto j_hparam = get_json(j_optimizer, "adam_hparam");
      float learning_rate = get_value_from_json<float>(j_hparam, "learning_rate");
      float beta1 = get_value_from_json<float>(j_hparam, "beta1");
      float beta2 = get_value_from_json<float>(j_hparam, "beta2");
      float epsilon = get_value_from_json<float>(j_hparam, "epsilon");
      opt_hyper_params.adam.beta1 = beta1;
      opt_hyper_params.adam.beta2 = beta2;
      opt_hyper_params.adam.epsilon = epsilon;
      opt_params = {Optimizer_t::Adam, learning_rate, opt_hyper_params, update_type};
      break;
    }
    case Optimizer_t::MomentumSGD: {
      auto j_hparam = get_json(j_optimizer, "momentum_sgd_hparam");
      float learning_rate = get_value_from_json<float>(j_hparam, "learning_rate");
      float momentum_factor = get_value_from_json<float>(j_hparam, "momentum_factor");
      opt_hyper_params.momentum.factor = momentum_factor;
      opt_params = {Optimizer_t::MomentumSGD, learning_rate, opt_hyper_params, update_type};
      break;
    }
    case Optimizer_t::Nesterov: {
      auto j_hparam = get_json(j_optimizer, "nesterov_hparam");
      float learning_rate = get_value_from_json<float>(j_hparam, "learning_rate");
      float momentum_factor = get_value_from_json<float>(j_hparam, "momentum_factor");
      opt_hyper_params.nesterov.mu = momentum_factor;
      opt_params = {Optimizer_t::Nesterov, learning_rate, opt_hyper_params, update_type};
      break;
    }
    case Optimizer_t::SGD: {
      auto j_hparam = get_json(j_optimizer, "sgd_hparam");
      auto learning_rate = get_value_from_json<float>(j_hparam, "learning_rate");
      if (has_key_(j_hparam, "atomic_update")) {
        opt_hyper_params.sgd.atomic_update = get_value_from_json<bool>(j_hparam, "atomic_update");
      }
      opt_params = {Optimizer_t::SGD, learning_rate, opt_hyper_params, update_type};
      break;
    }
    default:
      assert(!"Error: no such optimizer && should never get here!");
  }
  return opt_params;
}

template <typename T>
static std::shared_ptr<Regularizer<T>> create_regularizer(
    const nlohmann::json& j, const Tensor2<float>& weight_buff, const Tensor2<T>& wgrad_buff,
    const int batch_size, const std::shared_ptr<GPUResource>& gpu_resource) {
  std::shared_ptr<Regularizer<T>> reg(
      new NoRegularizer<T>(weight_buff, wgrad_buff, batch_size, gpu_resource));
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
        reg.reset(new L1Regularizer<T>(weight_buff, wgrad_buff, batch_size, lambda, gpu_resource));
        break;
      }
      case Regularizer_t::L2: {
        const auto lambda = get_value_from_json<float>(j, "lambda");
        reg.reset(new L2Regularizer<T>(weight_buff, wgrad_buff, batch_size, lambda, gpu_resource));
        break;
      }
      default: {
        assert(!"Error: no such regularizer!");
      }
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
    {"MultiCross", Layer_t::MultiCross},
    {"DotProduct", Layer_t::DotProduct}};
const std::map<std::string, Layer_t> LAYER_TYPE_MAP_MP = {
    {"BinaryCrossEntropyLoss", Layer_t::BinaryCrossEntropyLoss},
    {"Concat", Layer_t::Concat},
    {"Cast", Layer_t::Cast},
    {"InnerProduct", Layer_t::InnerProduct},
    {"FusedInnerProduct", Layer_t::FusedInnerProduct},
    {"Interaction", Layer_t::Interaction},
    {"Reshape", Layer_t::Reshape},
    {"Slice", Layer_t::Slice},
    {"ReLU", Layer_t::ReLU},
    {"Dropout", Layer_t::Dropout},
    {"Add", Layer_t::Add}};
const std::map<std::string, Embedding_t> EMBEDDING_TYPE_MAP = {
    {"DistributedSlotSparseEmbeddingHash", Embedding_t::DistributedSlotSparseEmbeddingHash},
    {"LocalizedSlotSparseEmbeddingHash", Embedding_t::LocalizedSlotSparseEmbeddingHash},
    {"LocalizedSlotSparseEmbeddingOneHot", Embedding_t::LocalizedSlotSparseEmbeddingOneHot},
    {"HybridSparseEmbedding", Embedding_t::HybridSparseEmbedding}};
const std::map<std::string, Initializer_t> INITIALIZER_TYPE_MAP = {
    {"Uniform", Initializer_t::Uniform},
    {"XavierNorm", Initializer_t::XavierNorm},
    {"XavierUniform", Initializer_t::XavierUniform},
    {"Zero", Initializer_t::Zero}};

void create_layers(const nlohmann::json& j_array, std::vector<TensorEntry>& tensor_entries,
                   const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                   const std::shared_ptr<BufferBlock2<float>>& weight_buff,
                   const std::shared_ptr<BufferBlock2<__half>>& weight_buff_half,
                   const std::shared_ptr<BufferBlock2<float>>& wgrad_buff,
                   const std::shared_ptr<BufferBlock2<__half>>& wgrad_buff_half,
                   Tensor2<float>& loss_tensor, const std::shared_ptr<GPUResource>& gpu_resource,
                   bool use_mixed_precision, bool enable_tf32_compute, 
                   int num_networks_in_global,
                   float scaler, bool& enable_cuda_graph,
                   std::vector<std::unique_ptr<Layer>>& layers, std::unique_ptr<ILoss>& loss,
                   metrics::RawMetricMap* raw_metrics,
                   std::vector<Layer*>* top_layers = nullptr,
                   std::vector<Layer*>* bottom_layers = nullptr) {
  bool skip_dgrad = true;
  bool is_bottom_mlp = true;

  auto emplaceback_layer = [&is_bottom_mlp, &layers, &bottom_layers, &top_layers](Layer* layer) {
    if(is_bottom_mlp) {
      if (bottom_layers) {
        bottom_layers->emplace_back(layer);
      }
    }else {
      if (top_layers) {
        top_layers->emplace_back(layer);
      }
    }
    layers.emplace_back(layer);
  };

  for (size_t i = 1; i < j_array.size(); i++) {
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

    // check if we should use bottom mlp or top mlp
    auto bottom = get_json(j, "bottom");
    std::vector<std::string> bottom_strs = get_layer_names(bottom);
    for(const std::string& str: bottom_strs) {
      if(str.find("embedding") != std::string::npos) {
        is_bottom_mlp = false;
      }
    }

    std::vector<TensorEntry> output_tensor_entries;
    auto input_output_info = get_input_tensor_and_output_name(j, tensor_entries);
    switch (layer_type) {
      case Layer_t::BatchNorm: {
        Tensor2<float> bn_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        // establish out tensor
        Tensor2<float> bn_out_tensor;
        blobs_buff->reserve(bn_in_tensor.get_dimensions(), &bn_out_tensor);
        output_tensor_entries.push_back(
            {input_output_info.output_names[0], bn_out_tensor.shrink()});

        // get BN params
        auto j_bn_hparam = get_json(j, "bn_param");
        auto factor = get_value_from_json<float>(j_bn_hparam, "factor");
        auto eps = get_value_from_json<float>(j_bn_hparam, "eps");
        // establish initializer
        std::vector<Initializer_t> initializer_types(2, Initializer_t::Default);
        if (has_key_(j_bn_hparam, "gamma_init")) {
          const auto gamma_init_name = get_value_from_json<std::string>(j_bn_hparam, "gamma_init");
          Initializer_t gamma_init_type;
          if (!find_item_in_map(gamma_init_type, gamma_init_name, INITIALIZER_TYPE_MAP)) {
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + gamma_init_name);
          } else {
            initializer_types[0] = gamma_init_type;
          }
        }
        if (has_key_(j_bn_hparam, "beta_init")) {
          const auto beta_init_name = get_value_from_json<std::string>(j_bn_hparam, "beta_init");
          Initializer_t beta_init_type;
          if (!find_item_in_map(beta_init_type, beta_init_name, INITIALIZER_TYPE_MAP)) {
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + beta_init_name);
          } else {
            initializer_types[1] = beta_init_type;
          }
        }

        BatchNormLayer::Params params = {factor, eps};
        emplaceback_layer(new BatchNormLayer(weight_buff, wgrad_buff, blobs_buff, bn_in_tensor,
                                               bn_out_tensor, params, gpu_resource,
                                               initializer_types));
        break;
      }
      case Layer_t::BinaryCrossEntropyLoss: {
        if (input_output_info.inputs.size() != 2) {
          CK_THROW_(Error_t::WrongInput, "bottom of BinaryCrossEntropyLoss must be two dim");
        }
        Tensor2<float> label_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[1]);
        blobs_buff->reserve({1, 1}, &loss_tensor);
        if (use_mixed_precision) {
          Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);

          loss.reset(new BinaryCrossEntropyLoss<__half>(
              label_tensor, in_tensor, loss_tensor,
              create_regularizer(j, weight_buff->as_tensor(), wgrad_buff_half->as_tensor(),
                                 in_tensor.get_dimensions()[0], gpu_resource),
              gpu_resource, num_networks_in_global, scaler));
        } else {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);

          loss.reset(new BinaryCrossEntropyLoss<float>(
              label_tensor, in_tensor, loss_tensor,
              create_regularizer(j, weight_buff->as_tensor(), wgrad_buff->as_tensor(),
                                 in_tensor.get_dimensions()[0], gpu_resource),
              gpu_resource, num_networks_in_global, scaler));
        }
        break;
      }
      case Layer_t::Concat: {
        if (use_mixed_precision) {
          Tensors2<__half> in_tensors;
          for (const TensorBag2& bag : input_output_info.inputs) {
            in_tensors.push_back(Tensor2<__half>::stretch_from(bag));
          }
          Tensor2<__half> out_tensor;
          emplaceback_layer(
              new ConcatLayer<__half>(in_tensors, out_tensor, blobs_buff, gpu_resource));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensors2<float> in_tensors;
          for (const TensorBag2& bag : input_output_info.inputs) {
            in_tensors.push_back(Tensor2<float>::stretch_from(bag));
          }
          Tensor2<float> out_tensor;
          emplaceback_layer(
              new ConcatLayer<float>(in_tensors, out_tensor, blobs_buff, gpu_resource));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        }
        break;
      }
      case Layer_t::CrossEntropyLoss: {
        if (input_output_info.inputs.size() != 2) {
          CK_THROW_(Error_t::WrongInput, "bottom of CrossEntropyLoss must be two dim");
        }
        Tensor2<float> label_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[1]);
        blobs_buff->reserve({1, 1}, &loss_tensor);
        if (use_mixed_precision) {
          Tensor2<__half> cross_entropy_loss_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.inputs[0]);

          loss.reset(new CrossEntropyLoss<__half>(
              label_tensor, cross_entropy_loss_in_tensor, loss_tensor,
              create_regularizer(j, weight_buff->as_tensor(), wgrad_buff_half->as_tensor(),
                                 cross_entropy_loss_in_tensor.get_dimensions()[0], gpu_resource),
              gpu_resource, num_networks_in_global, scaler));
        } else {
          Tensor2<float> cross_entropy_loss_in_tensor =
              Tensor2<float>::stretch_from(input_output_info.inputs[0]);

          loss.reset(new CrossEntropyLoss<float>(
              label_tensor, cross_entropy_loss_in_tensor, loss_tensor,
              create_regularizer(j, weight_buff->as_tensor(), wgrad_buff->as_tensor(),
                                 cross_entropy_loss_in_tensor.get_dimensions()[0], gpu_resource),
              gpu_resource, num_networks_in_global, scaler));
        }
        break;
      }
      case Layer_t::Dropout: {
        if (use_mixed_precision) {
          Tensor2<__half> do_in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          // establish out tensor
          Tensor2<__half> do_out_tensor;
          blobs_buff->reserve(do_in_tensor.get_dimensions(), &do_out_tensor);
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], do_out_tensor.shrink()});
          // get ELU params
          auto rate_it = j.find("rate");
          auto rate = (rate_it != j.end()) ? rate_it->get<float>() : 0.5f;
#ifndef PREFER_CUDNN
          emplaceback_layer(new DropoutLayer<__half>(do_in_tensor, do_out_tensor, blobs_buff,
                                                       rate, gpu_resource));
#else
          emplaceback_layer(new DropoutCudnnLayer<__half>(do_in_tensor, do_out_tensor, blobs_buff,
                                                            rate, gpu_resource));
#endif
        } else {
          // establish out tensor
          Tensor2<float> do_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> do_out_tensor;
          blobs_buff->reserve(do_in_tensor.get_dimensions(), &do_out_tensor);
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], do_out_tensor.shrink()});
          // get ELU params
          auto rate_it = j.find("rate");
          auto rate = (rate_it != j.end()) ? rate_it->get<float>() : 0.5f;
#ifndef PREFER_CUDNN
          emplaceback_layer(
              new DropoutLayer<float>(do_in_tensor, do_out_tensor, blobs_buff, rate, gpu_resource));
#else
          emplaceback_layer(new DropoutCudnnLayer<float>(do_in_tensor, do_out_tensor, blobs_buff,
                                                           rate, gpu_resource));
#endif
        }
        enable_cuda_graph = false;

        break;
      }
      case Layer_t::ELU: {
        Tensor2<float> elu_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);

        // establish out tensor
        Tensor2<float> elu_out_tensor;
        blobs_buff->reserve(elu_in_tensor.get_dimensions(), &elu_out_tensor);
        output_tensor_entries.push_back(
            {input_output_info.output_names[0], elu_out_tensor.shrink()});
        // get ELU params
        auto j_elu_hparam = get_json(j, "elu_param");
        auto alpha = get_value_from_json<float>(j_elu_hparam, "alpha");
        emplaceback_layer(new EluLayer(elu_in_tensor, elu_out_tensor, alpha, gpu_resource));

        break;
      }

      case Layer_t::FusedInnerProduct: {
        auto j_fc_param = get_json(j, "fc_param");
        // establish initializer
        std::vector<Initializer_t> initializer_types(2, Initializer_t::Default);
        if (has_key_(j_fc_param, "weight_init")) {
          const auto weight_init_name = get_value_from_json<std::string>(j_fc_param, "weight_init");
          Initializer_t weight_init_type;
          if (!find_item_in_map(weight_init_type, weight_init_name, INITIALIZER_TYPE_MAP)) {
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + weight_init_name);
          } else {
            initializer_types[0] = weight_init_type;
          }
        }
        if (has_key_(j_fc_param, "bias_init")) {
          const auto bias_init_name = get_value_from_json<std::string>(j_fc_param, "bias_init");
          Initializer_t bias_init_type;
          if (!find_item_in_map(bias_init_type, bias_init_name, INITIALIZER_TYPE_MAP)) {
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + bias_init_name);
          } else {
            initializer_types[1] = bias_init_type;
          }
        }
        // check the position of this layer
        FcPosition_t pos_type = FcPosition_t::None;
        int input_size = input_output_info.inputs.size();
        int output_size = input_output_info.output_names.size();
        if (has_key_(j, "position")) {
          auto pos_str = get_value_from_json<std::string>(j, "position");
          if (!find_item_in_map(pos_type, pos_str, FCPOSITION_TYPE_MAP)) {
            CK_THROW_(Error_t::WrongInput, "No such position: " + pos_str);
          } else if (pos_type == FcPosition_t::Head && input_size == 1 && output_size == 4) {
          } else if (pos_type == FcPosition_t::Body && input_size == 4 && output_size == 4) {
          } else if (pos_type == FcPosition_t::Tail && input_size == 4 && output_size == 1) {
          } else if (pos_type == FcPosition_t::Isolated && input_size == 1 && output_size == 1) {
          } else
            CK_THROW_(Error_t::WrongInput,
                      "The position and dimension of bottom and top layer aren't compatible: " +
                          layer_type_name);
        }
        // check the activation functino of this layer
        Activation_t act_type = Activation_t::Relu;
        if (has_key_(j, "activation")) {
          auto act_name = get_value_from_json<std::string>(j, "activation");
          if (!find_item_in_map(act_type, act_name, ACTIVATION_TYPE_MAP)) {
            CK_THROW_(Error_t::WrongInput, "No such activation: " + act_name);
          }
          if (act_type == Activation_t::None && pos_type != FcPosition_t::Tail)
            CK_THROW_(Error_t::WrongInput,
                      "The layer without activation function must be the last layer in MLP.");
        }
        // establish out tensor
        auto output = get_value_from_json<size_t>(j_fc_param, "num_output");
        if (use_mixed_precision) {
          Tensor2<__half> train_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> mask_in_tensor, dRelu_in_tensor, db_in_tensor;
          if (pos_type == FcPosition_t::Body || pos_type == FcPosition_t::Tail) {
            mask_in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[1]);
            dRelu_in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[2]);
            db_in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[3]);
          }
          Tensor2<__half> train_out_tensor, mask_out_tensor, dRelu_out_tensor, db_out_tensor;
          blobs_buff->reserve({(train_in_tensor.get_dimensions())[0], output}, &train_out_tensor);
          blobs_buff->reserve({(train_in_tensor.get_dimensions())[0], output}, &mask_out_tensor);
          blobs_buff->reserve({(train_in_tensor.get_dimensions())[0], output}, &dRelu_out_tensor);
          // blobs_buff->reserve({(train_in_tensor.get_dimensions())[0], output}, &db_out_tensor);

          // establish layer
          if (pos_type == FcPosition_t::None) {
            emplaceback_layer(new FusedFullyConnectedLayer(
                weight_buff, weight_buff_half, wgrad_buff_half, blobs_buff, train_in_tensor,
                train_out_tensor, gpu_resource, initializer_types));
          } else {
            emplaceback_layer(new FusedReluBiasFullyConnectedLayer(
                weight_buff, weight_buff_half, wgrad_buff_half, blobs_buff, train_in_tensor,
                mask_in_tensor, dRelu_in_tensor, db_in_tensor, train_out_tensor, mask_out_tensor,
                dRelu_out_tensor, db_out_tensor, gpu_resource, pos_type, act_type, skip_dgrad,
                initializer_types));
          }

          if (pos_type == FcPosition_t::Tail || pos_type == FcPosition_t::Isolated ||
              pos_type == FcPosition_t::None)
            output_tensor_entries.push_back(
                {input_output_info.output_names[0], train_out_tensor.shrink()});
          else {
            output_tensor_entries.push_back(
                {input_output_info.output_names[0], train_out_tensor.shrink()});
            output_tensor_entries.push_back(
                {input_output_info.output_names[1], mask_out_tensor.shrink()});
            output_tensor_entries.push_back(
                {input_output_info.output_names[2], dRelu_out_tensor.shrink()});
            output_tensor_entries.push_back(
                {input_output_info.output_names[3], db_out_tensor.shrink()});
          }
        } else {
          CK_THROW_(Error_t::WrongInput, "FusedInnerProduct support half only");
        }
        break;
      }

      case Layer_t::Cast: {
        if (use_mixed_precision) {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> out_tensor;
          blobs_buff->reserve(in_tensor.get_dimensions(), &out_tensor);
          emplaceback_layer(new CastLayer(in_tensor, out_tensor, gpu_resource));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          CK_THROW_(Error_t::WrongInput, "Cast supports half only");
        }
        break;
      }

      case Layer_t::InnerProduct: {
        auto j_fc_param = get_json(j, "fc_param");
        // establish initializer
        std::vector<Initializer_t> initializer_types(2, Initializer_t::Default);
        if (has_key_(j_fc_param, "weight_init")) {
          const auto weight_init_name = get_value_from_json<std::string>(j_fc_param, "weight_init");
          Initializer_t weight_init_type;
          if (!find_item_in_map(weight_init_type, weight_init_name, INITIALIZER_TYPE_MAP)) {
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + weight_init_name);
          } else {
            initializer_types[0] = weight_init_type;
          }
        }
        if (has_key_(j_fc_param, "bias_init")) {
          const auto bias_init_name = get_value_from_json<std::string>(j_fc_param, "bias_init");
          Initializer_t bias_init_type;
          if (!find_item_in_map(bias_init_type, bias_init_name, INITIALIZER_TYPE_MAP)) {
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + bias_init_name);
          } else {
            initializer_types[1] = bias_init_type;
          }
        }

        // establish out tensor
        auto output = get_value_from_json<size_t>(j_fc_param, "num_output");

        if (use_mixed_precision) {
          Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> fc_out_tensor;
          blobs_buff->reserve({in_tensor.get_dimensions()[0], output}, &fc_out_tensor);

          // establish layer
          emplaceback_layer(new FullyConnectedLayerHalf(
              weight_buff, weight_buff_half, wgrad_buff_half, blobs_buff, in_tensor, fc_out_tensor,
              gpu_resource, initializer_types));
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], fc_out_tensor.shrink()});
        } else {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> fc_out_tensor;
          blobs_buff->reserve({in_tensor.get_dimensions()[0], output}, &fc_out_tensor);
          // establish layer
          emplaceback_layer(new FullyConnectedLayer(
              weight_buff, wgrad_buff, in_tensor, fc_out_tensor, gpu_resource, use_mixed_precision,
              enable_tf32_compute, initializer_types));
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], fc_out_tensor.shrink()});
        }
        break;
      }

      case Layer_t::Interaction: {
        // lambda template could be a better solution here, but there's not support in c++11
        if (use_mixed_precision) {
          if (gpu_resource->get_cc_major() < 7) {
            CK_THROW_(Error_t::WrongInput, "InteractionLayer<__half> is not supported in SM " +
                                               std::to_string(gpu_resource->get_cc_major()) + "." +
                                               std::to_string(gpu_resource->get_cc_minor()));
          }

          Tensor2<__half> in_mlp_tensor =
              Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> in_emb_tensor =
              Tensor2<__half>::stretch_from(input_output_info.inputs[1]);
          Tensor2<__half> out_tensor;

          emplaceback_layer(new InteractionLayer<__half>(
              in_mlp_tensor, in_emb_tensor, out_tensor,
              blobs_buff,  // todo cannot use this blobs_buff here need half
              gpu_resource, use_mixed_precision, enable_tf32_compute));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});

        } else {
          Tensor2<float> in_mlp_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> in_emb_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[1]);
          Tensor2<float> out_tensor;
          emplaceback_layer(
              new InteractionLayer<float>(in_mlp_tensor, in_emb_tensor, out_tensor, blobs_buff,
                                          gpu_resource, use_mixed_precision, enable_tf32_compute));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        }

        break;
      }
      case Layer_t::MultiCross: {
        auto j_mc_param = get_json(j, "mc_param");
        // establish initializer
        std::vector<Initializer_t> initializer_types(2, Initializer_t::Default);
        if (has_key_(j_mc_param, "weight_init")) {
          const auto weight_init_name = get_value_from_json<std::string>(j_mc_param, "weight_init");
          Initializer_t weight_init_type;
          if (!find_item_in_map(weight_init_type, weight_init_name, INITIALIZER_TYPE_MAP)) {
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + weight_init_name);
          } else {
            initializer_types[0] = weight_init_type;
          }
        }
        if (has_key_(j_mc_param, "bias_init")) {
          const auto bias_init_name = get_value_from_json<std::string>(j_mc_param, "bias_init");
          Initializer_t bias_init_type;
          if (!find_item_in_map(bias_init_type, bias_init_name, INITIALIZER_TYPE_MAP)) {
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + bias_init_name);
          } else {
            initializer_types[1] = bias_init_type;
          }
        }

        // establish out tensor
        auto num_layers = get_value_from_json<int>(j_mc_param, "num_layers");
        Tensor2<float> mc_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> out_tensor;
        blobs_buff->reserve(mc_in_tensor.get_dimensions(), &out_tensor);
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        // establish layer
        emplaceback_layer(new MultiCrossLayer(weight_buff, wgrad_buff, blobs_buff, mc_in_tensor,
                                                out_tensor, gpu_resource, num_layers,
                                                initializer_types));
        break;
      }

      case Layer_t::MultiCrossEntropyLoss: {
        if (input_output_info.inputs.size() != 2) {
          CK_THROW_(Error_t::WrongInput, "bottom of MultiCrossEntropyLoss must be two dim");
        }

        auto tweight = get_json(j, "target_weight");
        std::vector<float> target_weight_vec;
        for (auto tweight_tmp : tweight) {
          float tweight_val = tweight_tmp.get<float>();
          target_weight_vec.push_back(tweight_val);
        }

        Tensor2<float> label_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[1]);
        blobs_buff->reserve({1, 1}, &loss_tensor);

        if (use_mixed_precision) {
          Tensor2<__half> multi_cross_entropy_loss_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          loss.reset(new MultiCrossEntropyLoss<__half>(
              label_tensor, multi_cross_entropy_loss_in_tensor, loss_tensor,
              create_regularizer(j, weight_buff->as_tensor(), wgrad_buff_half->as_tensor(),
                                 multi_cross_entropy_loss_in_tensor.get_dimensions()[0],
                                 gpu_resource),
              target_weight_vec, gpu_resource, num_networks_in_global, scaler));
        } else {
          Tensor2<float> multi_cross_entropy_loss_in_tensor =
              Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          loss.reset(new MultiCrossEntropyLoss<float>(
              label_tensor, multi_cross_entropy_loss_in_tensor, loss_tensor,
              create_regularizer(j, weight_buff->as_tensor(), wgrad_buff->as_tensor(),
                                 multi_cross_entropy_loss_in_tensor.get_dimensions()[0],
                                 gpu_resource),
              target_weight_vec, gpu_resource, num_networks_in_global, scaler));
        }
        break;
      }
      case Layer_t::ReLU: {
        if (use_mixed_precision) {
          Tensor2<__half> relu_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> relu_out_tensor;
          blobs_buff->reserve(relu_in_tensor.get_dimensions(), &relu_out_tensor);
          emplaceback_layer(new ReluLayer<__half>(relu_in_tensor, relu_out_tensor, gpu_resource));
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], relu_out_tensor.shrink()});
        } else {
          // establish out tensor
          Tensor2<float> relu_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> relu_out_tensor;
          blobs_buff->reserve(relu_in_tensor.get_dimensions(), &relu_out_tensor);
          emplaceback_layer(new ReluLayer<float>(relu_in_tensor, relu_out_tensor, gpu_resource));
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], relu_out_tensor.shrink()});
        }

        break;
      }
      case Layer_t::Reshape: {
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
            Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
            Tensor2<__half> out_tensor;
            emplaceback_layer(new ReshapeLayer<__half>(in_tensor, out_tensor, blobs_buff,
                                                         selected, gpu_resource));
            output_tensor_entries.push_back(
                {input_output_info.output_names[0], out_tensor.shrink()});
          } else {
            Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
            Tensor2<float> out_tensor;
            emplaceback_layer(
                new ReshapeLayer<float>(in_tensor, out_tensor, blobs_buff, selected, gpu_resource));
            output_tensor_entries.push_back(
                {input_output_info.output_names[0], out_tensor.shrink()});
          }
        }
        // general purpose reshape
        else {
          auto leading_dim_it = j.find("leading_dim");

          // if leading_dim is not specified, default leading_dim = n_slots * vector_length

          if (use_mixed_precision) {
            Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
            Tensor2<__half> out_tensor;
            const auto& in_dims = in_tensor.get_dimensions();
            size_t leading_dim = (leading_dim_it != j.end())
                                     ? (*leading_dim_it).get<int>()
                                     : in_tensor.get_num_elements() / in_dims[0];
            emplaceback_layer(new ReshapeLayer<__half>(in_tensor, out_tensor, blobs_buff,
                                                       leading_dim, gpu_resource));
            output_tensor_entries.push_back(
                {input_output_info.output_names[0], out_tensor.shrink()});
          } else {
            Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
            Tensor2<float> out_tensor;
            const auto& in_dims = in_tensor.get_dimensions();
            size_t leading_dim = (leading_dim_it != j.end())
                                     ? (*leading_dim_it).get<int>()
                                     : in_tensor.get_num_elements() / in_dims[0];
            emplaceback_layer(new ReshapeLayer<float>(in_tensor, out_tensor, blobs_buff,
                                                      leading_dim, gpu_resource));
            output_tensor_entries.push_back(
                {input_output_info.output_names[0], out_tensor.shrink()});
          }
        }
        break;
      }
      case Layer_t::Slice: {
        std::vector<std::pair<int, int>> ranges;
        auto j_ranges = get_json(j, "ranges");
        assert(j_ranges.is_array());
        for (auto j_range : j_ranges) {
          assert(j_range.is_array());
          ranges.emplace_back(std::make_pair(j_range[0].get<int>(), j_range[1].get<int>()));
        }

        if (use_mixed_precision) {
          Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensors2<__half> out_tensors;
          emplaceback_layer(
              new SliceLayer<__half>(in_tensor, out_tensors, blobs_buff, ranges, gpu_resource));
          for (size_t i = 0; i < out_tensors.size(); i++) {
            output_tensor_entries.push_back(
                {input_output_info.output_names[i], out_tensors[i].shrink()});
          }
        } else {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensors2<float> out_tensors;
          emplaceback_layer(
              new SliceLayer<float>(in_tensor, out_tensors, blobs_buff, ranges, gpu_resource));
          for (size_t i = 0; i < out_tensors.size(); i++) {
            output_tensor_entries.push_back(
                {input_output_info.output_names[i], out_tensors[i].shrink()});
          }
        }
        break;
      }
      case Layer_t::Multiply: {
        std::vector<size_t> weight_dims;
        auto dims = get_json(j, "weight_dims");
        assert(dims.is_array());
        for (auto dim : dims) {
          weight_dims.emplace_back(dim.get<size_t>());
        }

        // establish initializer
        std::vector<Initializer_t> initializer_types(1, Initializer_t::Default);
        if (has_key_(j, "weight_init")) {
          const auto weight_init_name = get_value_from_json<std::string>(j, "weight_init");
          Initializer_t weight_init_type;
          if (!find_item_in_map(weight_init_type, weight_init_name, INITIALIZER_TYPE_MAP)) {
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + weight_init_name);
          } else {
            initializer_types[0] = weight_init_type;
          }
        }

        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> out_tensor;
        emplaceback_layer(new MultiplyLayer(weight_buff, wgrad_buff, blobs_buff, in_tensor,
                                              out_tensor, weight_dims, gpu_resource,
                                              initializer_types));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        break;
      }
      case Layer_t::FmOrder2: {
        auto out_dim = get_json(j, "out_dim").get<size_t>();

        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> out_tensor;
        blobs_buff->reserve({in_tensor.get_dimensions()[0], out_dim}, &out_tensor);

        emplaceback_layer(new FmOrder2Layer(in_tensor, out_tensor, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        break;
      }
      case Layer_t::Add: {
        if (use_mixed_precision) {
          Tensors2<__half> in_tensors;
          for (const auto& bag : input_output_info.inputs) {
            in_tensors.push_back(Tensor2<__half>::stretch_from(bag));
          }
          Tensor2<__half> out_tensor;
          blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
          emplaceback_layer(
              new AddLayer<__half>(in_tensors, out_tensor, blobs_buff, gpu_resource));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensors2<float> in_tensors;
          for (const auto& bag : input_output_info.inputs) {
            in_tensors.push_back(Tensor2<float>::stretch_from(bag));
          }
          Tensor2<float> out_tensor;
          blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
          emplaceback_layer(
              new AddLayer<float>(in_tensors, out_tensor, blobs_buff, gpu_resource));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        }
        break;
      }
      case Layer_t::ReduceSum: {
        int axis = get_json(j, "axis").get<int>();

        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> out_tensor;
        emplaceback_layer(
            new ReduceSumLayer(in_tensor, out_tensor, blobs_buff, axis, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        break;
      }
      case Layer_t::DotProduct: {
        Tensors2<float> in_tensors;
        for (const auto& bag : input_output_info.inputs) {
          in_tensors.push_back(Tensor2<float>::stretch_from(bag));
        }
        Tensor2<float> out_tensor;
        blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
        emplaceback_layer(new DotProductLayer(in_tensors, out_tensor, blobs_buff, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        break;
      }
      default:
        assert(!"Error: no such layer && should never get here!");
    }  // end of switch

    if (!(layer_type == Layer_t::CrossEntropyLoss ||
          layer_type == Layer_t::BinaryCrossEntropyLoss ||
          layer_type == Layer_t::MultiCrossEntropyLoss)) {
      for (auto& output_tensor_entry : output_tensor_entries) {
        tensor_entries.push_back(output_tensor_entry);
      }
    } else if (raw_metrics) {
      (*raw_metrics)[metrics::RawType::Loss] = loss_tensor.shrink();
      (*raw_metrics)[metrics::RawType::Pred] = input_output_info.inputs[0];
      (*raw_metrics)[metrics::RawType::Label] = input_output_info.inputs[1];
    }

    skip_dgrad = false;
  }  // for layers
}

/*
 * Create single network
 *
 */
template <typename T> 
using BuffPtr = std::shared_ptr<BufferBlock2<T>>;
Network* create_network(const nlohmann::json& j_array, const nlohmann::json& j_optimizer,
                        std::vector<TensorEntry>& train_tensor_entries,
                        std::vector<TensorEntry>& evaluate_tensor_entries,
                        int num_networks_in_global, int num_local_gpu_count,
                        std::shared_ptr<ExchangeWgrad>& exchange_wgrad,
                        const std::shared_ptr<CPUResource>& cpu_resource,
                        const std::shared_ptr<GPUResource>& gpu_resource, bool use_mixed_precision,
                        bool enable_tf32_compute, float scaler, bool use_algorithm_search,
                        bool use_cuda_graph, bool grouped_all_reduce) {
  Network* network = new Network(cpu_resource, gpu_resource, use_mixed_precision, use_cuda_graph);

  auto& train_layers = network->train_layers_;
  auto* bottom_layers = &network->bottom_layers_;
  auto* top_layers = &network->top_layers_;
  auto& evaluate_layers = network->evaluate_layers_;
  auto& train_loss_tensor = network->train_loss_tensor_;
  auto& evaluate_loss_tensor = network->evaluate_loss_tensor_;
  auto& train_loss = network->train_loss_;
  auto& evaluate_loss = network->evaluate_loss_;
  auto& enable_cuda_graph = network->enable_cuda_graph_;
  auto& raw_metrics = network->raw_metrics_;

  std::shared_ptr<GeneralBuffer2<CudaAllocator>> blobs_buff =
      GeneralBuffer2<CudaAllocator>::create();

  std::shared_ptr<BufferBlock2<float>> train_weight_buff = blobs_buff->create_block<float>();
  std::shared_ptr<BufferBlock2<__half>> train_weight_buff_half = blobs_buff->create_block<__half>();
  std::shared_ptr<BufferBlock2<float>> wgrad_buff = NULL;
  std::shared_ptr<BufferBlock2<__half>> wgrad_buff_half = NULL;
  if (use_mixed_precision) {
    auto id = gpu_resource->get_local_id();
    wgrad_buff_half = (grouped_all_reduce) ? 
      std::dynamic_pointer_cast<GroupedExchangeWgrad<__half>>(exchange_wgrad)->get_network_wgrad_buffs()[id] :
      std::dynamic_pointer_cast<NetworkExchangeWgrad<__half>>(exchange_wgrad)->get_network_wgrad_buffs()[id];
    wgrad_buff = blobs_buff->create_block<float>(); // placeholder
  }
  else {
    auto id = gpu_resource->get_local_id();
    wgrad_buff = (grouped_all_reduce) ? 
      std::dynamic_pointer_cast<GroupedExchangeWgrad<float>>(exchange_wgrad)->get_network_wgrad_buffs()[id] :
      std::dynamic_pointer_cast<NetworkExchangeWgrad<float>>(exchange_wgrad)->get_network_wgrad_buffs()[id];
    wgrad_buff_half = blobs_buff->create_block<__half>(); // placeholder
  }
  std::shared_ptr<BufferBlock2<float>> evaluate_weight_buff = blobs_buff->create_block<float>();
  std::shared_ptr<BufferBlock2<__half>> evaluate_weight_buff_half =
      blobs_buff->create_block<__half>();
  std::shared_ptr<BufferBlock2<float>> wgrad_buff_placeholder = blobs_buff->create_block<float>();
  std::shared_ptr<BufferBlock2<__half>> wgrad_buff_half_placeholder =
      blobs_buff->create_block<__half>();

  // create train layers
  create_layers(j_array, train_tensor_entries, blobs_buff, train_weight_buff,
                train_weight_buff_half, wgrad_buff, wgrad_buff_half, train_loss_tensor,
                gpu_resource, use_mixed_precision, enable_tf32_compute, num_networks_in_global,
                scaler, enable_cuda_graph, train_layers, train_loss, nullptr, top_layers,
                bottom_layers);

  // create evaluate layers
  create_layers(j_array, evaluate_tensor_entries, blobs_buff, evaluate_weight_buff,
                evaluate_weight_buff_half, wgrad_buff_placeholder, wgrad_buff_half_placeholder,
                evaluate_loss_tensor, gpu_resource, use_mixed_precision, enable_tf32_compute,
                num_networks_in_global, scaler, enable_cuda_graph, evaluate_layers, evaluate_loss,
                &raw_metrics);

  // create optimizer
  auto opt_param = get_optimizer_param<float>(j_optimizer);

  network->optimizer_ = std::move(Optimizer::Create(
      opt_param, train_weight_buff->as_tensor(), wgrad_buff->as_tensor(),
      wgrad_buff_half->as_tensor(), use_mixed_precision, scaler, blobs_buff, gpu_resource));

  network->train_weight_tensor_ = train_weight_buff->as_tensor();
  network->train_weight_tensor_half_ = train_weight_buff_half->as_tensor();
  network->evaluate_weight_tensor_ = evaluate_weight_buff->as_tensor();
  network->evaluate_weight_tensor_half_ = evaluate_weight_buff_half->as_tensor();
  
  CudaDeviceContext context(gpu_resource->get_device_id());
  blobs_buff->allocate();
  return network;
}

template <typename TypeKey>
static void parse_data_layer(const nlohmann::json& j, int& label_dim, int& dense_dim,
                             Check_t& check_type, std::string& source_data,
                             std::vector<DataReaderSparseParam>& data_reader_sparse_param_array,
                             std::string& eval_source, std::string& top_strs_label,
                             std::string& top_strs_dense, std::vector<std::string>& sparse_names,
                             std::map<std::string, SparseInput<TypeKey>>& sparse_input_map) {
  source_data = get_value_from_json<std::string>(j, "source");

  auto j_label = get_json(j, "label");
  top_strs_label = get_value_from_json<std::string>(j_label, "top");
  label_dim = get_value_from_json<int>(j_label, "label_dim");

  auto j_dense = get_json(j, "dense");
  top_strs_dense = get_value_from_json<std::string>(j_dense, "top");
  dense_dim = get_value_from_json<int>(j_dense, "dense_dim");

  const std::map<std::string, Check_t> CHECK_TYPE_MAP = {{"Sum", Check_t::Sum},
                                                         {"None", Check_t::None}};

  const auto check_str = get_value_from_json<std::string>(j, "check");
  if (!find_item_in_map(check_type, check_str, CHECK_TYPE_MAP)) {
    CK_THROW_(Error_t::WrongInput, "Not supported check type: " + check_str);
  }

  const std::map<std::string, DataReaderSparse_t> DATA_TYPE_MAP = {
      {"DistributedSlot", DataReaderSparse_t::Distributed},
      {"LocalizedSlot", DataReaderSparse_t::Localized},
  };

  auto j_sparse = get_json(j, "sparse");
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
  FIND_AND_ASSIGN_STRING_KEY(eval_source, j);
}

void parse_data_layer_helper(const nlohmann::json& j, int& label_dim, int& dense_dim,
                             Check_t& check_type, std::string& source_data,
                             std::vector<DataReaderSparseParam>& data_reader_sparse_param_array,
                             std::string& eval_source, std::string& top_strs_label,
                             std::string& top_strs_dense, std::vector<std::string>& sparse_names,
                             std::map<std::string, SparseInput<long long>>& sparse_input_map) {
  parse_data_layer(j, label_dim, dense_dim, check_type, source_data, data_reader_sparse_param_array,
                   eval_source, top_strs_label, top_strs_dense, sparse_names, sparse_input_map);
}

template <typename TypeKey, typename TypeFP>
static void create_embeddings(std::map<std::string, SparseInput<TypeKey>>& sparse_input_map,
                              std::vector<TensorEntry>* train_tensor_entries_list,
                              std::vector<TensorEntry>* evaluate_tensor_entries_list,
                              std::vector<std::shared_ptr<IEmbedding>>& embeddings,
                              Embedding_t embedding_type, const nlohmann::json& config,
                              const std::shared_ptr<ResourceManager>& resource_manager,
                              size_t batch_size, size_t batch_size_eval, 
                              std::shared_ptr<ExchangeWgrad>& exchange_wgrad,
                              bool use_mixed_precision,
                              float scaler, const nlohmann::json& j_layers,
                              bool use_cuda_graph = false,
                              bool grouped_all_reduce = false) {
  auto j_optimizer = get_json(config, "optimizer");
  auto embedding_name = get_value_from_json<std::string>(j_layers, "type");

  auto bottom_name = get_value_from_json<std::string>(j_layers, "bottom");
  auto top_name = get_value_from_json<std::string>(j_layers, "top");

  auto& embed_wgrad_buff = (grouped_all_reduce) ? 
    std::dynamic_pointer_cast<GroupedExchangeWgrad<TypeFP>>(exchange_wgrad)->get_embed_wgrad_buffs() :
    std::dynamic_pointer_cast<NetworkExchangeWgrad<TypeFP>>(exchange_wgrad)->get_embed_wgrad_buffs();

  auto j_hparam = get_json(j_layers, "sparse_embedding_hparam");
  size_t max_vocabulary_size_per_gpu = 0;
  if (embedding_type == Embedding_t::DistributedSlotSparseEmbeddingHash) {
    max_vocabulary_size_per_gpu =
        get_value_from_json<size_t>(j_hparam, "max_vocabulary_size_per_gpu");
  } else if (embedding_type == Embedding_t::LocalizedSlotSparseEmbeddingHash) {
    if (has_key_(j_hparam, "max_vocabulary_size_per_gpu")) {
      max_vocabulary_size_per_gpu =
          get_value_from_json<size_t>(j_hparam, "max_vocabulary_size_per_gpu");
    } else if (!has_key_(j_hparam, "slot_size_array")) {
      CK_THROW_(Error_t::WrongInput,
                "No max_vocabulary_size_per_gpu or slot_size_array in: " + embedding_name);
    }
  }
  auto embedding_vec_size = get_value_from_json<size_t>(j_hparam, "embedding_vec_size");
  auto combiner = get_value_from_json<int>(j_hparam, "combiner");

  SparseInput<TypeKey> sparse_input;
  if (!find_item_in_map(sparse_input, bottom_name, sparse_input_map)) {
    CK_THROW_(Error_t::WrongInput, "Cannot find bottom");
  }

  OptParams<TypeFP> embedding_opt_params;
  if (has_key_(j_layers, "optimizer")) {
    embedding_opt_params = get_optimizer_param<TypeFP>(get_json(j_layers, "optimizer"));
  } else {
    embedding_opt_params = get_optimizer_param<TypeFP>(j_optimizer);
  }
  embedding_opt_params.scaler = scaler;

  switch (embedding_type) {
    case Embedding_t::DistributedSlotSparseEmbeddingHash: {
      const SparseEmbeddingHashParams<TypeFP> embedding_params = {
          batch_size,
          batch_size_eval,
          max_vocabulary_size_per_gpu,
          {},
          embedding_vec_size,
          sparse_input.max_feature_num_per_sample,
          sparse_input.slot_num,
          combiner,  // combiner: 0-sum, 1-mean
          embedding_opt_params};

      embeddings.emplace_back(new DistributedSlotSparseEmbeddingHash<TypeKey, TypeFP>(
          sparse_input.train_row_offsets, sparse_input.train_values, sparse_input.train_nnz,
          sparse_input.evaluate_row_offsets, sparse_input.evaluate_values,
          sparse_input.evaluate_nnz, embedding_params, resource_manager));
      break;
    }
    case Embedding_t::LocalizedSlotSparseEmbeddingHash: {
#ifndef NCCL_A2A

      auto j_plan = get_json(j_layers, "plan_file");
      std::string plan_file;
      if (j_plan.is_array()) {
        int num_nodes = j_plan.size();
        if (num_nodes != resource_manager->get_num_process()) {
          CK_THROW_(Error_t::WrongInput, "num_nodes != num_procs");
        }
        plan_file = j_plan[resource_manager->get_process_id()].get<std::string>();
      } else {
        if (resource_manager->get_num_process() > 1) {
          CK_THROW_(Error_t::WrongInput, "num_procs > 1");
        }
        plan_file = get_value_from_json<std::string>(j_layers, "plan_file");
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

      const SparseEmbeddingHashParams<TypeFP> embedding_params = {
          batch_size,
          batch_size_eval,
          max_vocabulary_size_per_gpu,
          slot_size_array,
          embedding_vec_size,
          sparse_input.max_feature_num_per_sample,
          sparse_input.slot_num,
          combiner,  // combiner: 0-sum, 1-mean
          embedding_opt_params};

      embeddings.emplace_back(new LocalizedSlotSparseEmbeddingHash<TypeKey, TypeFP>(
          sparse_input.train_row_offsets, sparse_input.train_values, sparse_input.train_nnz,
          sparse_input.evaluate_row_offsets, sparse_input.evaluate_values,
          sparse_input.evaluate_nnz, embedding_params, plan_file, resource_manager));

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

      const SparseEmbeddingHashParams<TypeFP> embedding_params = {
          batch_size,
          batch_size_eval,
          0,
          slot_size_array,
          embedding_vec_size,
          sparse_input.max_feature_num_per_sample,
          sparse_input.slot_num,
          combiner,  // combiner: 0-sum, 1-mean
          embedding_opt_params};

      embeddings.emplace_back(new LocalizedSlotSparseEmbeddingOneHot<TypeKey, TypeFP>(
          sparse_input.train_row_offsets, sparse_input.train_values, sparse_input.train_nnz,
          sparse_input.evaluate_row_offsets, sparse_input.evaluate_values,
          sparse_input.evaluate_nnz, embedding_params, plan_file, resource_manager,
          use_cuda_graph));

      break;
    }

    case Embedding_t::HybridSparseEmbedding: {
      std::vector<size_t> slot_size_array;
      auto slots = get_json(j_hparam, "slot_size_array");
      assert(slots.is_array());
      for (auto slot : slots) {
        slot_size_array.emplace_back(slot.get<size_t>());
      }
      // FIXME need to access this variable in line 1394 for init_data_reader
      size_t num_iterations_statistics =
          get_value_from_json_soft<size_t>(j_hparam, "num_iterations_statistics", 20);
      auto max_num_frequent_categories =
          get_value_from_json_soft<size_t>(j_hparam, "max_num_frequent_categories", 1);
      auto max_num_infrequent_samples =
          get_value_from_json_soft<int64_t>(j_hparam, "max_num_infrequent_samples", -1);
      double p_dup_max = get_value_from_json_soft<double>(j_hparam, "p_dup_max", 1. / 100);
      double max_all_reduce_bandwidth =
          get_value_from_json_soft<double>(j_hparam, "max_all_reduce_bandwidth", 1.3e11);
      double max_all_to_all_bandwidth =
          get_value_from_json_soft<double>(j_hparam, "max_all_to_all_bandwidth", 1.9e11);
      double efficiency_bandwidth_ratio =
          get_value_from_json_soft<double>(j_hparam, "efficiency_bandwidth_ratio", 1.0);

      const std::map<std::string, hybrid_embedding::CommunicationType> COMMUNICATION_TYPE_MAP = {
          {"IB_NVLink_Hierarchical", hybrid_embedding::CommunicationType::IB_NVLink_Hier},
          {"IB_NVLink", hybrid_embedding::CommunicationType::IB_NVLink},
          {"NVLink_SingleNode", hybrid_embedding::CommunicationType::NVLink_SingleNode}};
      std::string communication_type_string;
      if (has_key_(j_hparam, "communication_type")) {
        communication_type_string =
            get_value_from_json<std::string>(j_hparam, "communication_type");
      } else {
        communication_type_string = "IB_NVLink";
      }
      hybrid_embedding::CommunicationType communication_type;
      if (!find_item_in_map(communication_type, communication_type_string,
                            COMMUNICATION_TYPE_MAP)) {
        CK_THROW_(Error_t::WrongInput, "No such communication type: " + communication_type_string);
      }

      const std::map<std::string, hybrid_embedding::HybridEmbeddingType> HYBRID_EMBEDDING_TYPE_MAP =
          {{"Distributed", hybrid_embedding::HybridEmbeddingType::Distributed}};
      std::string hybrid_embedding_type_string;
      if (has_key_(j_hparam, "hybrid_embedding_type")) {
        hybrid_embedding_type_string =
            get_value_from_json<std::string>(j_hparam, "hybrid_embedding_type");
      } else {
        hybrid_embedding_type_string = "Distributed";
      }
      hybrid_embedding::HybridEmbeddingType hybrid_embedding_type;
      if (!find_item_in_map(hybrid_embedding_type, hybrid_embedding_type_string,
                            HYBRID_EMBEDDING_TYPE_MAP)) {
        CK_THROW_(Error_t::WrongInput,
                  "No such hybrid embedding type: " + hybrid_embedding_type_string);
      }

      auto j_solver = get_json(config, "solver");
      bool graph_mode = get_value_from_json_soft<bool>(j_solver, "holistic_cuda_graph", false);

      const HybridSparseEmbeddingParams<TypeFP> embedding_params = {
          batch_size,
          batch_size_eval,
          num_iterations_statistics,                                            // TBD
          max_num_frequent_categories * std::max(batch_size, batch_size_eval),  // TBD
          max_num_infrequent_samples,  // TBD
          p_dup_max,
          embedding_vec_size,
          sparse_input.slot_num,
          slot_size_array,
          communication_type,
          max_all_reduce_bandwidth,
          max_all_to_all_bandwidth,  // TBD
          efficiency_bandwidth_ratio,
          hybrid_embedding_type,
          embedding_opt_params};
      embeddings.emplace_back(new HybridSparseEmbedding<TypeKey, TypeFP>(
          sparse_input.train_values, sparse_input.evaluate_values, embedding_params,
          embed_wgrad_buff,
          get_gpu_learning_rate_schedulers(config, resource_manager),
          graph_mode,
          resource_manager));
      break;
    }
  }  // switch
  for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
    train_tensor_entries_list[i].push_back(
        {top_name, (embeddings.back()->get_train_output_tensors())[i]});
    evaluate_tensor_entries_list[i].push_back(
        {top_name, (embeddings.back()->get_evaluate_output_tensors())[i]});
  }
}

void create_allreduce_comm(
    const std::shared_ptr<ResourceManager>& resource_manager,
    std::shared_ptr<ExchangeWgrad>& exchange_wgrad,
    Parser& parser) {
  auto config = parser.config_;
  bool use_mixed_precision = parser.use_mixed_precision_;

  auto ar_algo = AllReduceAlgo::NCCL;
  bool grouped_all_reduce = false;
  if (has_key_(config, "all_reduce")) {
    auto j_all_reduce = get_json(config, "all_reduce");
    std::string ar_algo_name = "Oneshot";
    if (has_key_(j_all_reduce, "algo")) {
      ar_algo_name = get_value_from_json<std::string>(j_all_reduce, "algo");
    }
    if (has_key_(j_all_reduce, "grouped")) {
      grouped_all_reduce = get_value_from_json<bool>(j_all_reduce, "grouped");
    }
    MESSAGE_("Using All-reduce algorithm " + ar_algo_name);
    if (!find_item_in_map(ar_algo, ar_algo_name, ALLREDUCE_ALGO_MAP)) {
      CK_THROW_(Error_t::WrongInput, "All reduce algo unknown: " + ar_algo_name);
    }
  }

  resource_manager->set_ar_comm(ar_algo, use_mixed_precision);

  parser.grouped_all_reduce_ = grouped_all_reduce;
  if (grouped_all_reduce) {
    if (use_mixed_precision) {
      exchange_wgrad =  std::make_shared<GroupedExchangeWgrad<__half>>(resource_manager);
    } else {
      exchange_wgrad =  std::make_shared<GroupedExchangeWgrad<float>>(resource_manager);
    }
  }
  else {
    if (use_mixed_precision) {
      exchange_wgrad =  std::make_shared<NetworkExchangeWgrad<__half>>(resource_manager);
    } else {
      exchange_wgrad =  std::make_shared<NetworkExchangeWgrad<float>>(resource_manager);
    }
  }

}

template <typename TypeKey>
static void create_pipeline_internal(std::shared_ptr<IDataReader>& init_data_reader,std::shared_ptr<IDataReader>& train_data_reader,
                                     std::shared_ptr<IDataReader>& evaluate_data_reader,
                                     std::vector<std::shared_ptr<IEmbedding>>& embeddings,
                                     std::vector<std::shared_ptr<Network>>& networks,
                                     const std::shared_ptr<ResourceManager>& resource_manager,
                                     std::shared_ptr<ExchangeWgrad>& exchange_wgrad,
                                     Parser& parser) {
  try {
    nlohmann::json config = parser.config_;
    size_t batch_size = parser.batch_size_;
    size_t batch_size_eval = parser.batch_size_eval_;
    bool use_mixed_precision = parser.use_mixed_precision_;
    float scaler = parser.scaler_;
    bool enable_tf32_compute = parser.enable_tf32_compute_;
    bool use_algorithm_search = parser.use_algorithm_search_;
    bool use_cuda_graph = parser.use_cuda_graph_;

    create_allreduce_comm(resource_manager, exchange_wgrad, parser);
    bool grouped_all_reduce = parser.grouped_all_reduce_;

    std::map<std::string, SparseInput<TypeKey>> sparse_input_map;
    std::vector<TensorEntry> train_tensor_entries_list[resource_manager->get_local_gpu_count()];
    std::vector<TensorEntry> evaluate_tensor_entries_list[resource_manager->get_local_gpu_count()];
    {
      if (!networks.empty()) {
        CK_THROW_(Error_t::WrongInput, "vector network is not empty");
      }

      auto j_layers_array = get_json(config, "layers");
      auto j_optimizer = get_json(config, "optimizer");

      // Create Data Reader

      // This is a hack for now
      // std::unique_ptr<AsyncReader<TypeKey>> init_data_reader;
      {
        const nlohmann::json& j = j_layers_array[0];
        const auto layer_type_name = get_value_from_json<std::string>(j, "type");
        if (layer_type_name.compare("Data") != 0) {
          CK_THROW_(Error_t::WrongInput, "the first layer is not Data layer:" + layer_type_name);
        }

        const std::map<std::string, DataReaderType_t> DATA_READER_MAP = {
            {"Norm", DataReaderType_t::Norm},
            {"Raw", DataReaderType_t::Raw},
            {"Parquet", DataReaderType_t::Parquet},
            {"RawAsync", DataReaderType_t::RawAsync}};

        DataReaderType_t format = DataReaderType_t::Norm;
        if (has_key_(j, "format")) {
          const auto data_format_name = get_value_from_json<std::string>(j, "format");
          if (!find_item_in_map(format, data_format_name, DATA_READER_MAP)) {
            CK_THROW_(Error_t::WrongInput, "No such data format: " + data_format_name);
          }
        }

        auto cache_eval_data = get_value_from_json_soft<int>(j, "cache_eval_data", 0);

        std::string source_data = get_value_from_json<std::string>(j, "source");

        auto j_label = get_json(j, "label");
        auto top_strs_label = get_value_from_json<std::string>(j_label, "top");
        auto label_dim = get_value_from_json<int>(j_label, "label_dim");

        auto j_dense = get_json(j, "dense");
        auto top_strs_dense = get_value_from_json<std::string>(j_dense, "top");
        auto dense_dim = get_value_from_json<int>(j_dense, "dense_dim");
        Alignment_t aligned_type = Alignment_t::None;
        if (has_key_(j_dense, "aligned")) {
          auto aligned_str = get_value_from_json<std::string>(j_dense, "aligned");
          if (!find_item_in_map(aligned_type, aligned_str, ALIGNED_TYPE_MAP)) {
            CK_THROW_(Error_t::WrongInput, "Not supported aligned type: " + aligned_str);
          }
        }

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

        evaluate_data_reader = nullptr;
        std::string eval_source;
        FIND_AND_ASSIGN_STRING_KEY(eval_source, j);

#ifdef VAL
        const int NUM_THREADS = 1;
#else
        const int NUM_THREADS = (format == DataReaderType_t::Parquet)
                                    ? resource_manager->get_local_gpu_count()
                                    : ((format == DataReaderType_t::Raw) ? 32 : 12);
#endif
        // TODO: merge conflict is inevitable because parser was split into
        // multiple files in master branch.
        // To minimize the sheer agony in resolving it,the original data reader code
        // is preserved inside `else` block.
        if (format == DataReaderType_t::RawAsync) {
          auto ja = get_json(j, "async_param");
          auto num_threads = get_value_from_json<int>(ja, "num_threads");
          auto num_batches_per_thread = get_value_from_json<int>(ja, "num_batches_per_thread");
          auto io_block_size = get_value_from_json<int>(ja, "io_block_size");
          auto io_depth = get_value_from_json<int>(ja, "io_depth");
          auto io_alignment = get_value_from_json<int>(ja, "io_alignment");
          auto shuffle = get_value_from_json_soft<bool>(ja, "shuffle", false);
          size_t num_iterations_statistics =
              get_value_from_json_soft<size_t>(j, "num_iterations_statistics", 20);
          MESSAGE_("AsyncReader: num_threads = " + std::to_string(num_threads));
          MESSAGE_("AsyncReader: num_batches_per_thread = " +
                   std::to_string(num_batches_per_thread));
          MESSAGE_("AsyncReader: io_block_size = " + std::to_string(io_block_size));
          MESSAGE_("AsyncReader: io_depth = " + std::to_string(io_depth));
          MESSAGE_("AsyncReader: io_alignment = " + std::to_string(io_alignment));
          MESSAGE_("AsyncReader: num_iterations_statistics = " +
                   std::to_string(num_iterations_statistics));
          MESSAGE_("AsyncReader: shuffle = " + std::string(shuffle ? "ON" : "OFF"));

          // If the overlap is disabled, scheduling the data reader should be off
          auto j_solver = get_json(config, "solver");
          auto overlap = get_value_from_json_soft<bool>(j_solver, "enable_overlap", false);

          train_data_reader.reset(new AsyncReader<TypeKey>(
              source_data, batch_size, label_dim, dense_dim, data_reader_sparse_param_array,
              use_mixed_precision, resource_manager, num_threads, num_batches_per_thread,
              io_block_size, io_depth, io_alignment, shuffle, overlap, aligned_type));

          // If we want to cache eval, make sure we have enough buffers
          auto eval_num_batches_per_thread = num_batches_per_thread;
          if (cache_eval_data > num_threads * num_batches_per_thread) {
            eval_num_batches_per_thread = (cache_eval_data + num_threads - 1) / num_threads;
            MESSAGE_("AsyncReader: eval reader increased batches per thread to " +
                std::to_string(eval_num_batches_per_thread) + " to accommodate for the caching");
          }
          // Small IO block may lead to too many AIO requests which hang, 
          // so use a larger one for eval and init which are typically larger than train
          evaluate_data_reader.reset(new AsyncReader<TypeKey>(
              eval_source, batch_size_eval, label_dim, dense_dim, data_reader_sparse_param_array,
              use_mixed_precision, resource_manager, num_threads, eval_num_batches_per_thread,
              io_block_size*8, io_depth, io_alignment, false, false, aligned_type));

          init_data_reader.reset(new AsyncReader<TypeKey>(
              source_data, num_iterations_statistics * batch_size, label_dim, dense_dim,
              data_reader_sparse_param_array, use_mixed_precision, resource_manager, 1, 1,
              io_block_size*8, 4, io_alignment, false, false, aligned_type));

          for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
            train_tensor_entries_list[i].push_back(
                {top_strs_label, train_data_reader->get_label_tensors()[i]});
            evaluate_tensor_entries_list[i].push_back(
                {top_strs_label, evaluate_data_reader->get_label_tensors()[i]});

            if (use_mixed_precision) {
              train_tensor_entries_list[i].push_back(
                  {top_strs_dense, train_data_reader->get_dense_tensors()[i]});
              evaluate_tensor_entries_list[i].push_back(
                  {top_strs_dense, evaluate_data_reader->get_dense_tensors()[i]});
            } else {
              train_tensor_entries_list[i].push_back(
                  {top_strs_dense, train_data_reader->get_dense_tensors()[i]});
              evaluate_tensor_entries_list[i].push_back(
                  {top_strs_dense, evaluate_data_reader->get_dense_tensors()[i]});
            }
          }

          if (j_sparse.size() > 1) {
            CK_THROW_(Error_t::WrongInput, "Only one sparse input is supported.");
          }

          const auto& sparse_input = sparse_input_map.find(sparse_names[0]);
          sparse_input->second.train_values =
              bags_to_tensors<TypeKey>(train_data_reader->get_value_tensors());
          sparse_input->second.evaluate_values =
              bags_to_tensors<TypeKey>(evaluate_data_reader->get_value_tensors());
        } else {
          DataReader<TypeKey>* data_reader_tk = new DataReader<TypeKey>(
              batch_size, label_dim, dense_dim, data_reader_sparse_param_array, resource_manager,
              parser.repeat_dataset_, NUM_THREADS, use_mixed_precision, false, aligned_type);
          train_data_reader.reset(data_reader_tk);
          DataReader<TypeKey>* data_reader_eval_tk = new DataReader<TypeKey>(
              batch_size_eval, label_dim, dense_dim, data_reader_sparse_param_array,
              resource_manager, parser.repeat_dataset_, NUM_THREADS, use_mixed_precision,
              cache_eval_data, aligned_type);
          evaluate_data_reader.reset(data_reader_eval_tk);

          auto f = [&j]() -> std::vector<long long> {
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
            return slot_offset;
          };

          switch (format) {
            case DataReaderType_t::Norm: {
              bool start_right_now = parser.repeat_dataset_;
              train_data_reader->create_drwg_norm(source_data, check_type, start_right_now);
              evaluate_data_reader->create_drwg_norm(eval_source, check_type, start_right_now);
              break;
            }
            case DataReaderType_t::Raw: {
              const auto num_samples = get_value_from_json<long long>(j, "num_samples");
              const auto eval_num_samples = get_value_from_json<long long>(j, "eval_num_samples");
              std::vector<long long> slot_offset = f();
              bool float_label_dense =
                  get_value_from_json_soft<bool>(j, "float_label_dense", false);
              train_data_reader->create_drwg_raw(source_data, num_samples, slot_offset, float_label_dense,
                                           true, false);
              evaluate_data_reader->create_drwg_raw(eval_source, eval_num_samples, slot_offset,
                                                float_label_dense, false, false);

              break;
            }
            case DataReaderType_t::Parquet: {
#ifdef DISABLE_CUDF
              CK_THROW_(Error_t::WrongInput, "Parquet is not supported under DISABLE_CUDF");
#else
              // @Future: Should be slot_offset here and data_reader ctor should
              // be TypeKey not long long
              std::vector<long long> slot_offset = f();
              train_data_reader->create_drwg_parquet(source_data, slot_offset, true);
              evluate_data_reader->create_drwg_parquet(eval_source, slot_offset, true);
#endif
              break;
            }
            default: {
              assert(!"Error: no such option && should never get here!");
            }
          }

          for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
            train_tensor_entries_list[i].push_back(
                {top_strs_label, data_reader_tk->get_label_tensors()[i]});
            evaluate_tensor_entries_list[i].push_back(
                {top_strs_label, data_reader_eval_tk->get_label_tensors()[i]});

            if (use_mixed_precision) {
              train_tensor_entries_list[i].push_back(
                  {top_strs_dense, data_reader_tk->get_dense_tensors()[i]});
              evaluate_tensor_entries_list[i].push_back({top_strs_dense, data_reader_eval_tk->get_dense_tensors()[i]});
            } else {
              train_tensor_entries_list[i].push_back(
                  {top_strs_dense, data_reader_tk->get_dense_tensors()[i]});
              evaluate_tensor_entries_list[i].push_back({top_strs_dense, data_reader_eval_tk->get_dense_tensors()[i]});
            }
          }

          for (unsigned int i = 0; i < j_sparse.size(); i++) {
            const auto& sparse_input = sparse_input_map.find(sparse_names[i]);
            sparse_input->second.train_row_offsets = data_reader_tk->get_row_offsets_tensors(i);
            sparse_input->second.train_values = data_reader_tk->get_value_tensors(i);
            sparse_input->second.train_nnz = data_reader_tk->get_nnz_array(i);
            sparse_input->second.evaluate_row_offsets =
                data_reader_eval_tk->get_row_offsets_tensors(i);
            sparse_input->second.evaluate_values = data_reader_eval_tk->get_value_tensors(i);
            sparse_input->second.evaluate_nnz = data_reader_eval_tk->get_nnz_array(i);
          }
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

          if (use_mixed_precision) {
            create_embeddings<TypeKey, __half>(
                sparse_input_map, train_tensor_entries_list, evaluate_tensor_entries_list,
                embeddings, embedding_type, config, resource_manager, batch_size, batch_size_eval,
                exchange_wgrad,
                use_mixed_precision, scaler, j, use_cuda_graph, grouped_all_reduce);
          } else {
            create_embeddings<TypeKey, float>(
                sparse_input_map, train_tensor_entries_list, evaluate_tensor_entries_list,
                embeddings, embedding_type, config, resource_manager, batch_size, batch_size_eval,
                exchange_wgrad,
                use_mixed_precision, scaler, j, use_cuda_graph, grouped_all_reduce);
          }
        }  // for ()
      }    // Create Embedding

      // create network
      int total_gpu_count = resource_manager->get_global_gpu_count();
      int local_gpu_count = resource_manager->get_local_gpu_count();
      if (0 != batch_size % total_gpu_count) {
        CK_THROW_(Error_t::WrongInput, "0 != batch_size\%total_gpu_count");
      }
      for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
        networks.emplace_back(create_network(
            j_layers_array, j_optimizer, train_tensor_entries_list[i],
            evaluate_tensor_entries_list[i], total_gpu_count, local_gpu_count,
            exchange_wgrad,
            resource_manager->get_local_cpu(),
            resource_manager->get_local_gpu(i), use_mixed_precision, enable_tf32_compute, scaler,
            use_algorithm_search, use_cuda_graph, grouped_all_reduce));
      }
    }
    exchange_wgrad->allocate();

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void Parser::create_pipeline(std::shared_ptr<IDataReader>& init_data_reader,std::shared_ptr<IDataReader>& train_data_reader,
                             std::shared_ptr<IDataReader>& evaluate_data_reader,
                             std::vector<std::shared_ptr<IEmbedding>>& embeddings,
                             std::vector<std::shared_ptr<Network>>& networks,
                             const std::shared_ptr<ResourceManager>& resource_manager,
                             std::shared_ptr<ExchangeWgrad>& exchange_wgrad) {
  if (i64_input_key_) {
    create_pipeline_internal<long long>(init_data_reader, train_data_reader, evaluate_data_reader, embeddings,
                                        networks, resource_manager, exchange_wgrad, *this);
  } else {
    create_pipeline_internal<unsigned int>(init_data_reader, train_data_reader, evaluate_data_reader,
                                           embeddings, networks, resource_manager, exchange_wgrad, *this);
  }
}
template <typename TypeKey>
static void initialize_pipeline_internal(std::shared_ptr<IDataReader>& init_data_reader,
                                         std::vector<std::shared_ptr<IEmbedding>>& embedding,
                                         const std::shared_ptr<ResourceManager>& resource_manager,
                                         std::shared_ptr<ExchangeWgrad>& exchange_wgrad,
                                         Parser& parser) {
  try {
    nlohmann::json config = parser.config_;
    auto j_layers_array = get_json(config, "layers");
    bool use_mixed_precision = parser.use_mixed_precision_;
    size_t embed_wgrad_size = 0;
    for (unsigned int i = 1; i < j_layers_array.size(); i++) {
      const nlohmann::json& j = j_layers_array[i];
      auto embedding_name = get_value_from_json<std::string>(j, "type");
      Embedding_t embedding_type = Embedding_t::LocalizedSlotSparseEmbeddingOneHot;
      (void)find_item_in_map(embedding_type, embedding_name, EMBEDDING_TYPE_MAP);
      if (embedding_type == Embedding_t::HybridSparseEmbedding) {
        if (use_mixed_precision) {
          //#warning "we should find a better way than dynamic_pointer_cast"
          std::shared_ptr<HybridSparseEmbedding<TypeKey, __half>> hybrid_embedding =
              std::dynamic_pointer_cast<HybridSparseEmbedding<TypeKey, __half>>(embedding[i - 1]);

          init_data_reader->start();
          init_data_reader->read_a_batch_to_device();
          hybrid_embedding->init_model(
              bags_to_tensors<TypeKey>(init_data_reader->get_value_tensors()), embed_wgrad_size);
        } else {
          std::shared_ptr<HybridSparseEmbedding<TypeKey, float>> hybrid_embedding =
              std::dynamic_pointer_cast<HybridSparseEmbedding<TypeKey, float>>(embedding[i - 1]);
          init_data_reader->start();
          init_data_reader->read_a_batch_to_device();
          hybrid_embedding->init_model(
              bags_to_tensors<TypeKey>(init_data_reader->get_value_tensors()), embed_wgrad_size);
        }
      }
    }
    if (parser.grouped_all_reduce_) {
      exchange_wgrad->update_embed_wgrad_size(embed_wgrad_size);
    }
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}
void Parser::initialize_pipeline(std::shared_ptr<IDataReader>& init_data_reader,
                                 std::vector<std::shared_ptr<IEmbedding>>& embedding,
                                 const std::shared_ptr<ResourceManager>& resource_manager,
                                 std::shared_ptr<ExchangeWgrad>& exchange_wgrad) {
  if (i64_input_key_) {
    initialize_pipeline_internal<long long>(init_data_reader, embedding, resource_manager, exchange_wgrad, *this);
  } else {
    initialize_pipeline_internal<unsigned int>(init_data_reader, embedding, resource_manager, exchange_wgrad,
                                               *this);
  }
}
}  // namespace HugeCTR
