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

#include <data_readers/data_reader.hpp>
#include <embeddings/distributed_slot_sparse_embedding_hash.hpp>
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
#include <layers/interaction_layer.hpp>
#include <layers/multi_cross_layer.hpp>
#include <layers/multiply_layer.hpp>
#include <layers/reduce_sum_layer.hpp>
#include <layers/relu_layer.hpp>
#include <layers/reshape_layer.hpp>
#include <layers/sigmoid_layer.hpp>
#include <layers/slice_layer.hpp>
#include <loss.hpp>
#include <metrics.hpp>
#include <optimizer.hpp>
#include <parser.hpp>
#include <regularizers/l1_regularizer.hpp>
#include <regularizers/l2_regularizer.hpp>
#include <regularizers/no_regularizer.hpp>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {

struct InputOutputInfo {
  std::vector<TensorBag2> train_input;
  std::vector<TensorBag2> evaluate_input;
  std::vector<std::string> output;
};

static bool get_tensor_from_entries(const std::vector<TensorEntry> tensor_entries,
                                    const std::string& name, TensorUse use, TensorBag2* bag) {
  if (use == TensorUse::General) {
    CK_THROW_(Error_t::WrongInput, "Type should not be general");
  }
  for (const TensorEntry& entry : tensor_entries) {
    if (entry.name == name && (entry.use == TensorUse::General || entry.use == use)) {
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
  std::vector<std::string> bottom_strs = get_layer_names(bottom);

  auto top = get_json(json, "top");
  std::vector<std::string> top_strs = get_layer_names(top);

  std::vector<TensorBag2> bottom_train_tensors;
  std::vector<TensorBag2> bottom_evaluate_tensors;

  for (auto& bstr : bottom_strs) {
    for (auto& tstr : top_strs) {
      if (bstr == tstr) {
        CK_THROW_(Error_t::WrongInput, "bottom and top include a same layer name");
      }
    }
    TensorBag2 tensor;
    if (!get_tensor_from_entries(tensor_entries, bstr, TensorUse::Train, &tensor)) {
      CK_THROW_(Error_t::WrongInput, "No such bottom: " + bstr);
    }
    bottom_train_tensors.push_back(tensor);
    if (!get_tensor_from_entries(tensor_entries, bstr, TensorUse::Evaluate, &tensor)) {
      CK_THROW_(Error_t::WrongInput, "No such bottom: " + bstr);
    }

    bottom_evaluate_tensors.push_back(tensor);
  }
  return {bottom_train_tensors, bottom_evaluate_tensors, top_strs};
}

struct TensorPair {
  TensorBag2 tensor;
  std::string name;
};

static void add_tensor_to_network(TensorPair& output_tensor_pair,
                                  std::vector<TensorEntry>& tensor_entries) {
  tensor_entries.push_back(
      {output_tensor_pair.name, TensorUse::General, output_tensor_pair.tensor});
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
  memset(&opt_hyper_params, 0, sizeof(opt_hyper_params));
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
    {"Sigmoid", Layer_t::Sigmoid},
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
    {"Sigmoid", Layer_t::Sigmoid},
    {"Slice", Layer_t::Slice},
    {"ReLU", Layer_t::ReLU},
    {"Dropout", Layer_t::Dropout},
    {"Add", Layer_t::Add}};
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
std::unique_ptr<Network> Network::create_network(
    const nlohmann::json& j_array, const nlohmann::json& j_optimizer,
    std::vector<TensorEntry>& tensor_entries, int num_networks_in_global,
    const std::shared_ptr<CPUResource>& cpu_resource,
    const std::shared_ptr<GPUResource>& gpu_resource, bool use_mixed_precision, float scaler,
    bool use_algorithm_search, bool use_cuda_graph, bool inference_flag) {
  std::unique_ptr<Network> network(
      new Network(cpu_resource, gpu_resource, use_mixed_precision, use_cuda_graph));

  auto& layers = network->layers_;
  auto& loss_tensor = network->loss_tensor_;
  auto& loss = network->loss_;

  std::shared_ptr<GeneralBuffer2<CudaAllocator>> blobs_buff =
      GeneralBuffer2<CudaAllocator>::create();

  std::shared_ptr<BufferBlock2<float>> weight_buff = blobs_buff->create_block<float>();
  std::shared_ptr<BufferBlock2<__half>> weight_buff_half = blobs_buff->create_block<__half>();
  std::shared_ptr<BufferBlock2<float>> wgrad_buff = blobs_buff->create_block<float>();
  std::shared_ptr<BufferBlock2<__half>> wgrad_buff_half = blobs_buff->create_block<__half>();

  assert(layers.empty());

  for (unsigned int i = 1; i < j_array.size(); i++) {
    const nlohmann::json& j = j_array[i];
    const auto layer_type_name = get_value_from_json<std::string>(j, "type");
    const std::string layer_name = get_value_from_json<std::string>(j, "name");
    MESSAGE_("layer name: " + layer_name);
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
    auto input_output_info =
        get_input_tensor_and_output_name(j, tensor_entries);
    switch (layer_type) {
      case Layer_t::BatchNorm: {
        Tensor2<float> bn_in_tensor =
            Tensor2<float>::stretch_from(input_output_info.train_input[0]);
        // establish out tensor
        Tensor2<float> bn_out_tensor;
        blobs_buff->reserve(bn_in_tensor.get_dimensions(), &bn_out_tensor);
        output_tensor_pairs.push_back({bn_out_tensor.shrink(), input_output_info.output[0]});

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
        layers.emplace_back(new BatchNormLayer(weight_buff, wgrad_buff, blobs_buff, bn_in_tensor,
                                               bn_out_tensor, params, gpu_resource,
                                               initializer_types));
        break;
      }
      case Layer_t::BinaryCrossEntropyLoss: {
        if (input_output_info.train_input.size() != 2 ||
            input_output_info.evaluate_input.size() != 2) {
          CK_THROW_(Error_t::WrongInput, "bottom of BinaryCrossEntropyLoss must be two dim");
        }
        if (inference_flag) {
          CK_THROW_(Error_t::WrongInput, "inference network must NOT have BinaryCrossEntropyLoss");
        }
        Tensor2<float> train_label_tensor =
            Tensor2<float>::stretch_from(input_output_info.train_input[1]);
        Tensor2<float> evaluate_label_tensor =
            Tensor2<float>::stretch_from(input_output_info.evaluate_input[1]);
        blobs_buff->reserve({1, 1}, &loss_tensor);
        if (use_mixed_precision) {
          Tensor2<__half> train_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.train_input[0]);
          Tensor2<__half> evaluate_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.evaluate_input[0]);

          loss.reset(new BinaryCrossEntropyLoss<__half>(
              train_label_tensor, train_in_tensor, evaluate_label_tensor, evaluate_in_tensor,
              loss_tensor,
              create_regularizer(j, weight_buff->as_tensor(), wgrad_buff_half->as_tensor(),
                                 train_in_tensor.get_dimensions()[0], gpu_resource),
              gpu_resource, num_networks_in_global, scaler));
        } else {
          Tensor2<float> train_in_tensor =
              Tensor2<float>::stretch_from(input_output_info.train_input[0]);
          Tensor2<float> evaluate_in_tensor =
              Tensor2<float>::stretch_from(input_output_info.evaluate_input[0]);

          loss.reset(new BinaryCrossEntropyLoss<float>(
              train_label_tensor, train_in_tensor, evaluate_label_tensor, evaluate_in_tensor,
              loss_tensor,
              create_regularizer(j, weight_buff->as_tensor(), wgrad_buff->as_tensor(),
                                 train_in_tensor.get_dimensions()[0], gpu_resource),
              gpu_resource, num_networks_in_global, scaler));
        }
        break;
      }
      case Layer_t::Concat: {
        if (use_mixed_precision) {
          Tensors2<__half> train_in_tensors;
          for (const TensorBag2& t : input_output_info.train_input) {
            train_in_tensors.push_back(Tensor2<__half>::stretch_from(t));
          }
          Tensors2<__half> evaluate_in_tensors;
          for (const TensorBag2& t : input_output_info.evaluate_input) {
            evaluate_in_tensors.push_back(Tensor2<__half>::stretch_from(t));
          }
          Tensor2<__half> out_tensor;
          layers.emplace_back(new ConcatLayer<__half>(train_in_tensors, evaluate_in_tensors,
                                                      out_tensor, blobs_buff, gpu_resource));
          output_tensor_pairs.push_back({out_tensor.shrink(), input_output_info.output[0]});
        } else {
          Tensors2<float> train_in_tensors;
          for (const TensorBag2& t : input_output_info.train_input) {
            train_in_tensors.push_back(Tensor2<float>::stretch_from(t));
          }
          Tensors2<float> evaluate_in_tensors;
          for (const TensorBag2& t : input_output_info.evaluate_input) {
            evaluate_in_tensors.push_back(Tensor2<float>::stretch_from(t));
          }
          Tensor2<float> out_tensor;
          layers.emplace_back(new ConcatLayer<float>(train_in_tensors, evaluate_in_tensors,
                                                     out_tensor, blobs_buff, gpu_resource));
          output_tensor_pairs.push_back({out_tensor.shrink(), input_output_info.output[0]});
        }
        break;
      }
      case Layer_t::CrossEntropyLoss: {
        if (input_output_info.train_input.size() != 2) {
          CK_THROW_(Error_t::WrongInput, "bottom of CrossEntropyLoss must be two dim");
        }
        if (inference_flag) {
          CK_THROW_(Error_t::WrongInput, "inference network must NOT have CrossEntropyLoss");
        }
        Tensor2<float> label_tensor =
            Tensor2<float>::stretch_from(input_output_info.train_input[1]);
        blobs_buff->reserve({1, 1}, &loss_tensor);
        if (use_mixed_precision) {
          Tensor2<__half> cross_entropy_loss_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.train_input[0]);

          loss.reset(new CrossEntropyLoss<__half>(
              label_tensor, cross_entropy_loss_in_tensor, loss_tensor,
              create_regularizer(j, weight_buff->as_tensor(), wgrad_buff_half->as_tensor(),
                                 cross_entropy_loss_in_tensor.get_dimensions()[0], gpu_resource),
              gpu_resource, num_networks_in_global, scaler));
        } else {
          Tensor2<float> cross_entropy_loss_in_tensor =
              Tensor2<float>::stretch_from(input_output_info.train_input[0]);

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
          Tensor2<__half> do_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.train_input[0]);
          // establish out tensor
          Tensor2<__half> do_out_tensor;
          blobs_buff->reserve(do_in_tensor.get_dimensions(), &do_out_tensor);
          output_tensor_pairs.push_back({do_out_tensor.shrink(), input_output_info.output[0]});
          // get ELU params
          auto rate_it = j.find("rate");
          auto rate = (rate_it != j.end()) ? rate_it->get<float>() : 0.5f;
#ifndef PREFER_CUDNN
          layers.emplace_back(new DropoutLayer<__half>(do_in_tensor, do_out_tensor, blobs_buff,
                                                       rate, gpu_resource));
#else
          layers.emplace_back(new DropoutCudnnLayer<__half>(do_in_tensor, do_out_tensor, blobs_buff,
                                                            rate, gpu_resource));
#endif
        } else {
          // establish out tensor
          Tensor2<float> do_in_tensor =
              Tensor2<float>::stretch_from(input_output_info.train_input[0]);
          Tensor2<float> do_out_tensor;
          blobs_buff->reserve(do_in_tensor.get_dimensions(), &do_out_tensor);
          output_tensor_pairs.push_back({do_out_tensor.shrink(), input_output_info.output[0]});
          // get ELU params
          auto rate_it = j.find("rate");
          auto rate = (rate_it != j.end()) ? rate_it->get<float>() : 0.5f;
#ifndef PREFER_CUDNN
          layers.emplace_back(
              new DropoutLayer<float>(do_in_tensor, do_out_tensor, blobs_buff, rate, gpu_resource));
#else
          layers.emplace_back(new DropoutCudnnLayer<float>(do_in_tensor, do_out_tensor, blobs_buff,
                                                           rate, gpu_resource));
#endif
        }
        network->enable_cuda_graph_ = false;

        break;
      }
      case Layer_t::ELU: {
        Tensor2<float> elu_in_tensor =
            Tensor2<float>::stretch_from(input_output_info.train_input[0]);

        // establish out tensor
        Tensor2<float> elu_out_tensor;
        blobs_buff->reserve(elu_in_tensor.get_dimensions(), &elu_out_tensor);
        output_tensor_pairs.push_back({elu_out_tensor.shrink(), input_output_info.output[0]});
        // get ELU params
        auto j_elu_hparam = get_json(j, "elu_param");
        auto alpha = get_value_from_json<float>(j_elu_hparam, "alpha");
        layers.emplace_back(new EluLayer(elu_in_tensor, elu_out_tensor, alpha, gpu_resource));

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
        // establish out tensor
        auto output = get_value_from_json<size_t>(j_fc_param, "num_output");
        if (use_mixed_precision) {
          Tensor2<__half> train_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.train_input[0]);
          Tensor2<__half> evaluate_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.evaluate_input[0]);
          Tensor2<__half> fc_out_tensor;
          blobs_buff->reserve({(train_in_tensor.get_dimensions())[0], output}, &fc_out_tensor);
          output_tensor_pairs.push_back({fc_out_tensor.shrink(), input_output_info.output[0]});

          // establish layer
          layers.emplace_back(new FusedFullyConnectedLayer(
              weight_buff, weight_buff_half, wgrad_buff_half, blobs_buff, train_in_tensor,
              evaluate_in_tensor, fc_out_tensor, gpu_resource, initializer_types));
        } else {
          CK_THROW_(Error_t::WrongInput, "FusedInnerProduct support half only");
        }
        break;
      }

      case Layer_t::Cast: {
        if (use_mixed_precision) {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.train_input[0]);
          Tensor2<__half> out_tensor;
          blobs_buff->reserve(in_tensor.get_dimensions(), &out_tensor);
          output_tensor_pairs.push_back({out_tensor.shrink(), input_output_info.output[0]});
          layers.emplace_back(new CastLayer(in_tensor, out_tensor, gpu_resource));
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
          Tensor2<__half> train_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.train_input[0]);
          Tensor2<__half> evaluate_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.evaluate_input[0]);
          Tensor2<__half> fc_out_tensor;
          blobs_buff->reserve({train_in_tensor.get_dimensions()[0], output}, &fc_out_tensor);

          // establish layer
          layers.emplace_back(new FullyConnectedLayerHalf(
              weight_buff, weight_buff_half, wgrad_buff_half, blobs_buff, train_in_tensor,
              evaluate_in_tensor, fc_out_tensor, gpu_resource, initializer_types));
          output_tensor_pairs.push_back({fc_out_tensor.shrink(), input_output_info.output[0]});
        } else {
          Tensor2<float> train_in_tensor =
              Tensor2<float>::stretch_from(input_output_info.train_input[0]);
          Tensor2<float> evaluate_in_tensor =
              Tensor2<float>::stretch_from(input_output_info.evaluate_input[0]);
          Tensor2<float> fc_out_tensor;
          blobs_buff->reserve({train_in_tensor.get_dimensions()[0], output}, &fc_out_tensor);
          // establish layer
          layers.emplace_back(new FullyConnectedLayer(
              weight_buff, wgrad_buff, train_in_tensor, evaluate_in_tensor, fc_out_tensor,
              gpu_resource, use_mixed_precision, initializer_types));
          output_tensor_pairs.push_back({fc_out_tensor.shrink(), input_output_info.output[0]});
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

          Tensor2<__half> train_in_mlp_tensor =
              Tensor2<__half>::stretch_from(input_output_info.train_input[0]);
          Tensor2<__half> evaluate_in_mlp_tensor =
              Tensor2<__half>::stretch_from(input_output_info.evaluate_input[0]);
          Tensor2<__half> train_in_emb_tensor =
              Tensor2<__half>::stretch_from(input_output_info.train_input[1]);
          Tensor2<__half> evaluate_in_emb_tensor =
              Tensor2<__half>::stretch_from(input_output_info.evaluate_input[1]);
          Tensor2<__half> out_tensor;

          layers.emplace_back(new InteractionLayer<__half>(
              train_in_mlp_tensor, evaluate_in_mlp_tensor, train_in_emb_tensor,
              evaluate_in_emb_tensor, out_tensor,
              blobs_buff,  // todo cannot use this blobs_buff here need half
              gpu_resource, use_mixed_precision));
          output_tensor_pairs.push_back({out_tensor.shrink(), input_output_info.output[0]});

        } else {
          Tensor2<float> train_in_mlp_tensor =
              Tensor2<float>::stretch_from(input_output_info.train_input[0]);
          Tensor2<float> evaluate_in_mlp_tensor =
              Tensor2<float>::stretch_from(input_output_info.evaluate_input[0]);
          Tensor2<float> train_emb_tensor =
              Tensor2<float>::stretch_from(input_output_info.train_input[1]);
          Tensor2<float> evaluate_emb_tensor =
              Tensor2<float>::stretch_from(input_output_info.evaluate_input[1]);
          Tensor2<float> out_tensor;
          layers.emplace_back(new InteractionLayer<float>(
              train_in_mlp_tensor, evaluate_in_mlp_tensor, train_emb_tensor, evaluate_emb_tensor,
              out_tensor, blobs_buff, gpu_resource, use_mixed_precision));
          output_tensor_pairs.push_back({out_tensor.shrink(), input_output_info.output[0]});
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
        Tensor2<float> mc_in_tensor =
            Tensor2<float>::stretch_from(input_output_info.train_input[0]);
        Tensor2<float> out_tensor;
        blobs_buff->reserve(mc_in_tensor.get_dimensions(), &out_tensor);
        output_tensor_pairs.push_back({out_tensor.shrink(), input_output_info.output[0]});
        // establish layer
        layers.emplace_back(new MultiCrossLayer(weight_buff, wgrad_buff, blobs_buff, mc_in_tensor,
                                                out_tensor, gpu_resource, num_layers,
                                                initializer_types));
        break;
      }

      case Layer_t::MultiCrossEntropyLoss: {
        if (input_output_info.train_input.size() != 2) {
          CK_THROW_(Error_t::WrongInput, "bottom of MultiCrossEntropyLoss must be two dim");
        }
        if (inference_flag) {
          CK_THROW_(Error_t::WrongInput, "inference network must NOT have CrossEntropyLoss");
        }
        auto tweight = get_json(j, "target_weight");
        std::vector<float> target_weight_vec;
        for (auto tweight_tmp : tweight) {
          float tweight_val = tweight_tmp.get<float>();
          target_weight_vec.push_back(tweight_val);
        }

        Tensor2<float> label_tensor =
            Tensor2<float>::stretch_from(input_output_info.train_input[1]);
        blobs_buff->reserve({1, 1}, &loss_tensor);

        if (use_mixed_precision) {
          Tensor2<__half> multi_cross_entropy_loss_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.train_input[0]);
          loss.reset(new MultiCrossEntropyLoss<__half>(
              label_tensor, multi_cross_entropy_loss_in_tensor, loss_tensor,
              create_regularizer(j, weight_buff->as_tensor(), wgrad_buff_half->as_tensor(),
                                 multi_cross_entropy_loss_in_tensor.get_dimensions()[0],
                                 gpu_resource),
              target_weight_vec, gpu_resource, num_networks_in_global, scaler));
        } else {
          Tensor2<float> multi_cross_entropy_loss_in_tensor =
              Tensor2<float>::stretch_from(input_output_info.train_input[0]);
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
              Tensor2<__half>::stretch_from(input_output_info.train_input[0]);
          Tensor2<__half> relu_out_tensor;
          blobs_buff->reserve(relu_in_tensor.get_dimensions(), &relu_out_tensor);
          layers.emplace_back(new ReluLayer<__half>(relu_in_tensor, relu_out_tensor, gpu_resource));
          output_tensor_pairs.push_back({relu_out_tensor.shrink(), input_output_info.output[0]});
        } else {
          // establish out tensor
          Tensor2<float> relu_in_tensor =
              Tensor2<float>::stretch_from(input_output_info.train_input[0]);
          Tensor2<float> relu_out_tensor;
          blobs_buff->reserve(relu_in_tensor.get_dimensions(), &relu_out_tensor);
          layers.emplace_back(new ReluLayer<float>(relu_in_tensor, relu_out_tensor, gpu_resource));
          output_tensor_pairs.push_back({relu_out_tensor.shrink(), input_output_info.output[0]});
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
            Tensor2<__half> in_tensor =
                Tensor2<__half>::stretch_from(input_output_info.train_input[0]);
            Tensor2<__half> out_tensor;
            layers.emplace_back(new ReshapeLayer<__half>(in_tensor, out_tensor, blobs_buff,
                                                         selected, gpu_resource));
            output_tensor_pairs.push_back({out_tensor.shrink(), input_output_info.output[0]});
          } else {
            Tensor2<float> in_tensor =
                Tensor2<float>::stretch_from(input_output_info.train_input[0]);
            Tensor2<float> out_tensor;
            layers.emplace_back(
                new ReshapeLayer<float>(in_tensor, out_tensor, blobs_buff, selected, gpu_resource));
            output_tensor_pairs.push_back({out_tensor.shrink(), input_output_info.output[0]});
          }
        }
        // general purpose reshape
        else {
          auto leading_dim_it = j.find("leading_dim");

          // if leading_dim is not specified, default leading_dim = n_slots * vector_length

          if (use_mixed_precision) {
            Tensor2<__half> train_in_tensor =
                Tensor2<__half>::stretch_from(input_output_info.train_input[0]);
            Tensor2<__half> evaluate_in_tensor =
                Tensor2<__half>::stretch_from(input_output_info.evaluate_input[0]);
            Tensor2<__half> out_tensor;
            const auto& in_dims = train_in_tensor.get_dimensions();
            size_t leading_dim = (leading_dim_it != j.end())
                                     ? (*leading_dim_it).get<int>()
                                     : train_in_tensor.get_num_elements() / in_dims[0];
            layers.emplace_back(new ReshapeLayer<__half>(train_in_tensor, evaluate_in_tensor,
                                                         out_tensor, blobs_buff, leading_dim,
                                                         gpu_resource));
            output_tensor_pairs.push_back({out_tensor.shrink(), input_output_info.output[0]});
          } else {
            Tensor2<float> train_in_tensor =
                Tensor2<float>::stretch_from(input_output_info.train_input[0]);
            Tensor2<float> evaluate_in_tensor =
                Tensor2<float>::stretch_from(input_output_info.evaluate_input[0]);
            Tensor2<float> out_tensor;
            const auto& in_dims = train_in_tensor.get_dimensions();
            size_t leading_dim = (leading_dim_it != j.end())
                                     ? (*leading_dim_it).get<int>()
                                     : train_in_tensor.get_num_elements() / in_dims[0];
            layers.emplace_back(new ReshapeLayer<float>(train_in_tensor, evaluate_in_tensor,
                                                        out_tensor, blobs_buff, leading_dim,
                                                        gpu_resource));
            output_tensor_pairs.push_back({out_tensor.shrink(), input_output_info.output[0]});
          }
        }
        break;
      }
      case Layer_t::Sigmoid: {
        if (use_mixed_precision) {
          Tensor2<__half> sigmoid_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.train_input[0]);
          Tensor2<__half> sigmoid_out_tensor;
          blobs_buff->reserve(sigmoid_in_tensor.get_dimensions(), &sigmoid_out_tensor);
          layers.emplace_back(new SigmoidLayer<__half>(sigmoid_in_tensor, sigmoid_out_tensor, gpu_resource));
          output_tensor_pairs.push_back({sigmoid_out_tensor.shrink(), input_output_info.output[0]});
        } else {
          // establish out tensor
          Tensor2<float> sigmoid_in_tensor =
              Tensor2<float>::stretch_from(input_output_info.train_input[0]);
          Tensor2<float> sigmoid_out_tensor;
          blobs_buff->reserve(sigmoid_in_tensor.get_dimensions(), &sigmoid_out_tensor);
          layers.emplace_back(new SigmoidLayer<float>(sigmoid_in_tensor, sigmoid_out_tensor, gpu_resource));
          output_tensor_pairs.push_back({sigmoid_out_tensor.shrink(), input_output_info.output[0]});
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
          Tensor2<__half> train_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.train_input[0]);
          Tensor2<__half> evaluate_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.evaluate_input[0]);
          Tensors2<__half> out_tensors;
          layers.emplace_back(new SliceLayer<__half>(
              train_in_tensor, evaluate_in_tensor, out_tensors, blobs_buff, ranges, gpu_resource));
          for (size_t i = 0; i < out_tensors.size(); i++) {
            output_tensor_pairs.push_back({out_tensors[i].shrink(), input_output_info.output[i]});
          }
        } else {
          Tensor2<float> train_in_tensor =
              Tensor2<float>::stretch_from(input_output_info.train_input[0]);
          Tensor2<float> evaluate_in_tensor =
              Tensor2<float>::stretch_from(input_output_info.evaluate_input[0]);
          Tensors2<float> out_tensors;
          layers.emplace_back(new SliceLayer<float>(train_in_tensor, evaluate_in_tensor,
                                                    out_tensors, blobs_buff, ranges, gpu_resource));
          for (size_t i = 0; i < out_tensors.size(); i++) {
            output_tensor_pairs.push_back({out_tensors[i].shrink(), input_output_info.output[i]});
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

        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.train_input[0]);
        Tensor2<float> out_tensor;
        layers.emplace_back(new MultiplyLayer(weight_buff, wgrad_buff, blobs_buff, in_tensor,
                                              out_tensor, weight_dims, gpu_resource,
                                              initializer_types));
        output_tensor_pairs.push_back({out_tensor.shrink(), input_output_info.output[0]});
        break;
      }
      case Layer_t::FmOrder2: {
        auto out_dim = get_json(j, "out_dim").get<size_t>();

        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.train_input[0]);
        Tensor2<float> out_tensor;
        blobs_buff->reserve({in_tensor.get_dimensions()[0], out_dim}, &out_tensor);

        layers.emplace_back(new FmOrder2Layer(in_tensor, out_tensor, gpu_resource));
        output_tensor_pairs.push_back({out_tensor.shrink(), input_output_info.output[0]});
        break;
      }
      case Layer_t::Add: {
        if (use_mixed_precision) {
          Tensors2<__half> in_tensors;
          for (const auto& t : input_output_info.train_input) {
            in_tensors.push_back(Tensor2<__half>::stretch_from(t));
          }
          Tensor2<__half> out_tensor;
          blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
          layers.emplace_back(
              new AddLayer<__half>(in_tensors, out_tensor, blobs_buff, gpu_resource));
          output_tensor_pairs.push_back({out_tensor.shrink(), input_output_info.output[0]});
        } else {
          Tensors2<float> in_tensors;
          for (const auto& t : input_output_info.train_input) {
            in_tensors.push_back(Tensor2<float>::stretch_from(t));
          }
          Tensor2<float> out_tensor;
          blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
          layers.emplace_back(
              new AddLayer<float>(in_tensors, out_tensor, blobs_buff, gpu_resource));
          output_tensor_pairs.push_back({out_tensor.shrink(), input_output_info.output[0]});
        }
        break;
      }
      case Layer_t::ReduceSum: {
        int axis = get_json(j, "axis").get<int>();

        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.train_input[0]);
        Tensor2<float> out_tensor;
        layers.emplace_back(
            new ReduceSumLayer(in_tensor, out_tensor, blobs_buff, axis, gpu_resource));
        output_tensor_pairs.push_back({out_tensor.shrink(), input_output_info.output[0]});
        break;
      }
      case Layer_t::DotProduct: {
        Tensors2<float> in_tensors;
        for (const auto& t : input_output_info.train_input) {
          in_tensors.push_back(Tensor2<float>::stretch_from(t));
        }
        Tensor2<float> out_tensor;
        blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
        layers.emplace_back(new DotProductLayer(in_tensors, out_tensor, blobs_buff, gpu_resource));
        output_tensor_pairs.push_back({out_tensor.shrink(), input_output_info.output[0]});
        break;
      }
      default:
        assert(!"Error: no such layer && should never get here!");
    }  // end of switch

    if (!(layer_type == Layer_t::CrossEntropyLoss ||
          layer_type == Layer_t::BinaryCrossEntropyLoss ||
          layer_type == Layer_t::MultiCrossEntropyLoss)) {
      for (auto& output_tensor_pair : output_tensor_pairs) {
        add_tensor_to_network(output_tensor_pair, tensor_entries);
      }
    } else {
      if (!inference_flag) {
        network->raw_metrics_[metrics::RawType::Loss] = loss_tensor.shrink();
        network->raw_metrics_[metrics::RawType::Pred] = input_output_info.evaluate_input[0];
        network->raw_metrics_[metrics::RawType::Label] = input_output_info.evaluate_input[1];
      }
    }
  }  // for layers

  // create optimizer
  if (!inference_flag) {
    auto opt_param = get_optimizer_param<float>(j_optimizer);

    network->optimizer_ = std::move(Optimizer::Create(
        opt_param, weight_buff->as_tensor(), wgrad_buff->as_tensor(), wgrad_buff_half->as_tensor(),
        use_mixed_precision, scaler, blobs_buff, gpu_resource));
  }

  network->weight_tensor_ = weight_buff->as_tensor();
  network->wgrad_tensor_ = wgrad_buff->as_tensor();
  network->weight_tensor_half_ = weight_buff_half->as_tensor();
  network->wgrad_tensor_half_ = wgrad_buff_half->as_tensor();

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
static void create_embedding(std::map<std::string, SparseInput<TypeKey>>& sparse_input_map,
                             std::vector<TensorEntry>* tensor_entries_list,
                             std::vector<std::shared_ptr<IEmbedding>>& embedding,
                             Embedding_t embedding_type, const nlohmann::json& config,
                             const std::shared_ptr<ResourceManager>& resource_manager,
                             size_t batch_size, size_t batch_size_eval, bool use_mixed_precision,
                             float scaler, const nlohmann::json& j_layers) {
  auto j_optimizer = get_json(config, "optimizer");
  auto embedding_name = get_value_from_json<std::string>(j_layers, "type");

  auto bottom_name = get_value_from_json<std::string>(j_layers, "bottom");
  auto top_name = get_value_from_json<std::string>(j_layers, "top");

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

      embedding.emplace_back(new DistributedSlotSparseEmbeddingHash<TypeKey, TypeFP>(
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

      embedding.emplace_back(new LocalizedSlotSparseEmbeddingHash<TypeKey, TypeFP>(
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

      embedding.emplace_back(new LocalizedSlotSparseEmbeddingOneHot<TypeKey, TypeFP>(
          sparse_input.train_row_offsets, sparse_input.train_values, sparse_input.train_nnz,
          sparse_input.evaluate_row_offsets, sparse_input.evaluate_values,
          sparse_input.evaluate_nnz, embedding_params, plan_file, resource_manager));

      break;
    }
  }  // switch
  for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
    tensor_entries_list[i].push_back(
        {top_name, TensorUse::Train, (embedding.back()->get_train_output_tensors())[i]});
    tensor_entries_list[i].push_back(
        {top_name, TensorUse::Evaluate, (embedding.back()->get_evaluate_output_tensors())[i]});
  }
}

template <typename TypeKey>
static void create_pipeline_internal(std::shared_ptr<IDataReader>& data_reader,
                                     std::shared_ptr<IDataReader>& data_reader_eval,
                                     std::vector<std::shared_ptr<IEmbedding>>& embedding,
                                     std::vector<std::unique_ptr<Network>>& network,
                                     const std::shared_ptr<ResourceManager>& resource_manager,
                                     Parser& parser) {
  try {
    nlohmann::json config = parser.config_;
    size_t batch_size = parser.batch_size_;
    size_t batch_size_eval = parser.batch_size_eval_;
    bool use_mixed_precision = parser.use_mixed_precision_;
    float scaler = parser.scaler_;
    bool use_algorithm_search = parser.use_algorithm_search_;
    bool use_cuda_graph = parser.use_cuda_graph_;

    std::map<std::string, SparseInput<TypeKey>> sparse_input_map;
    std::vector<TensorEntry> tensor_entries_list[resource_manager->get_local_gpu_count()];
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
            {"Norm", DataReaderType_t::Norm},
            {"Raw", DataReaderType_t::Raw},
            {"Parquet", DataReaderType_t::Parquet}};

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

#ifdef VAL
        const int NUM_THREADS = 1;
#else
        const int NUM_THREADS =
            format == DataReaderType_t::Parquet ? resource_manager->get_local_gpu_count() : 12;
#endif

        DataReader<TypeKey>* data_reader_tk = new DataReader<TypeKey>(
            batch_size, label_dim, dense_dim, data_reader_sparse_param_array, resource_manager,
            parser.repeat_dataset_, NUM_THREADS, use_mixed_precision, false);
        data_reader.reset(data_reader_tk);
        DataReader<TypeKey>* data_reader_eval_tk = new DataReader<TypeKey>(
            batch_size_eval, label_dim, dense_dim, data_reader_sparse_param_array, resource_manager,
            parser.repeat_dataset_, NUM_THREADS, use_mixed_precision, cache_eval_data);
        data_reader_eval.reset(data_reader_eval_tk);

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
            data_reader->create_drwg_norm(source_data, check_type, start_right_now);
            data_reader_eval->create_drwg_norm(eval_source, check_type, start_right_now);
            break;
          }
          case DataReaderType_t::Raw: {
            const auto num_samples = get_value_from_json<long long>(j, "num_samples");
            const auto eval_num_samples = get_value_from_json<long long>(j, "eval_num_samples");
            std::vector<long long> slot_offset = f();
            bool float_label_dense = get_value_from_json_soft<bool>(j, "float_label_dense", false);
            data_reader->create_drwg_raw(source_data, num_samples, slot_offset, float_label_dense,
                                         true, false);
            data_reader_eval->create_drwg_raw(eval_source, eval_num_samples, slot_offset,
                                              float_label_dense, false, false);

            break;
          }
          case DataReaderType_t::Parquet: {
            // @Future: Should be slot_offset here and data_reader ctor should
            // be TypeKey not long long
            std::vector<long long> slot_offset = f();
            data_reader->create_drwg_parquet(source_data, slot_offset, true);
            data_reader_eval->create_drwg_parquet(eval_source, slot_offset, true);
            break;
          }
          default: {
            assert(!"Error: no such option && should never get here!");
          }
        }

        for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
          tensor_entries_list[i].push_back(
              {top_strs_label, TensorUse::Train, data_reader_tk->get_label_tensors()[i].shrink()});
          tensor_entries_list[i].push_back({top_strs_label, TensorUse::Evaluate,
                                            data_reader_eval_tk->get_label_tensors()[i].shrink()});

          if (use_mixed_precision) {
            tensor_entries_list[i].push_back(
                {top_strs_dense, TensorUse::Train, data_reader_tk->get_dense_tensors()[i]});
            tensor_entries_list[i].push_back(
                {top_strs_dense, TensorUse::Evaluate, data_reader_eval_tk->get_dense_tensors()[i]});
          } else {
            tensor_entries_list[i].push_back(
                {top_strs_dense, TensorUse::Train, data_reader_tk->get_dense_tensors()[i]});
            tensor_entries_list[i].push_back(
                {top_strs_dense, TensorUse::Evaluate, data_reader_eval_tk->get_dense_tensors()[i]});
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
            create_embedding<TypeKey, __half>(sparse_input_map, tensor_entries_list, embedding,
                                              embedding_type, config, resource_manager, batch_size,
                                              batch_size_eval, use_mixed_precision, scaler, j);
          } else {
            create_embedding<TypeKey, float>(sparse_input_map, tensor_entries_list, embedding,
                                             embedding_type, config, resource_manager, batch_size,
                                             batch_size_eval, use_mixed_precision, scaler, j);
          }
        }  // for ()
      }    // Create Embedding

      // create network
      int total_gpu_count = resource_manager->get_global_gpu_count();
      if (0 != batch_size % total_gpu_count) {
        CK_THROW_(Error_t::WrongInput, "0 != batch_size\%total_gpu_count");
      }
      for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
        network.emplace_back(Network::create_network(
            j_layers_array, j_optimizer, tensor_entries_list[i], total_gpu_count,
            resource_manager->get_local_cpu(), resource_manager->get_local_gpu(i),
            use_mixed_precision, scaler, use_algorithm_search, use_cuda_graph, false));
      }
    }

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void Parser::create_pipeline(std::shared_ptr<IDataReader>& data_reader,
                             std::shared_ptr<IDataReader>& data_reader_eval,
                             std::vector<std::shared_ptr<IEmbedding>>& embedding,
                             std::vector<std::unique_ptr<Network>>& network,
                             const std::shared_ptr<ResourceManager>& resource_manager) {
  if (i64_input_key_) {
    create_pipeline_internal<long long>(data_reader, data_reader_eval, embedding, network,
                                        resource_manager, *this);
  } else {
    create_pipeline_internal<unsigned int>(data_reader, data_reader_eval, embedding, network,
                                           resource_manager, *this);
  }
}

}  // namespace HugeCTR
