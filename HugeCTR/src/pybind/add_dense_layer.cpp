/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <layer.hpp>
#include <layers/add_layer.hpp>
#include <layers/batch_norm_layer.hpp>
#include <layers/cast_layer.hpp>
#include <layers/concat_layer.hpp>
#include <layers/dot_product_layer.hpp>
#include <layers/dropout_layer.hpp>
#include <layers/elementwise_multiply_layer.hpp>
#include <layers/elu_layer.hpp>
#include <layers/fm_order2_layer.hpp>
#include <layers/fully_connected_layer.hpp>
#include <layers/fully_connected_layer_half.hpp>
#include <layers/fused_fully_connected_layer.hpp>
#include <layers/fused_relu_bias_fully_connected_layer.hpp>
#include <layers/fused_reshape_concat_general_layer.hpp>
#include <layers/fused_reshape_concat_layer.hpp>
#include <layers/gather_layer.hpp>
#include <layers/gru_layer.hpp>
#include <layers/interaction_layer.hpp>
#include <layers/matrix_multiply_layer.hpp>
#include <layers/multi_cross_layer.hpp>
#include <layers/prelu_dice_layer.hpp>
#include <layers/reduce_mean_layer.hpp>
#include <layers/reduce_sum_layer.hpp>
#include <layers/relu_layer.hpp>
#include <layers/reshape_layer.hpp>
#include <layers/scale_layer.hpp>
#include <layers/sigmoid_layer.hpp>
#include <layers/slice_layer.hpp>
#include <layers/softmax_layer.hpp>
#include <layers/sub_layer.hpp>
#include <layers/weight_multiply_layer.hpp>
#include <pybind/model.hpp>
#include <regularizers/l1_regularizer.hpp>
#include <regularizers/l2_regularizer.hpp>
#include <regularizers/no_regularizer.hpp>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {

void save_graph_to_json(nlohmann::json& layer_config_array,
                        std::vector<DenseLayer>& dense_layer_params,
                        std::vector<SparseEmbedding>& sparse_embedding_params,
                        std::vector<Input>& input_params,
                        std::vector<std::shared_ptr<OptParamsPy>>& embedding_opt_params_list,
                        bool use_mixed_precision) {
  nlohmann::json input_config;
  input_config["type"] = "Data";
  nlohmann::json input_label_config;
  nlohmann::json input_dense_config;
  nlohmann::json input_sparse_config_array = nlohmann::json::array();
  assert(input_params.size() == 1);
  Input input_param = input_params[0];
  input_label_config["top"] = input_param.label_name;
  input_label_config["label_dim"] = input_param.label_dim;
  input_dense_config["top"] = input_param.dense_name;
  input_dense_config["dense_dim"] = input_param.dense_dim;
  for (size_t i = 0; i < input_param.data_reader_sparse_param_array.size(); ++i) {
    nlohmann::json input_sparse_config;
    input_sparse_config["top"] = input_param.data_reader_sparse_param_array[i].top_name;
    input_sparse_config["type"] =
        READER_SPARSE_TYPE_TO_STRING[input_param.data_reader_sparse_param_array[i].type];
    input_sparse_config["nnz_per_slot"] =
        input_param.data_reader_sparse_param_array[i].nnz_per_slot;
    input_sparse_config["is_fixed_length"] =
        input_param.data_reader_sparse_param_array[i].is_fixed_length;
    input_sparse_config["slot_num"] = input_param.data_reader_sparse_param_array[i].slot_num;
    input_sparse_config_array.push_back(input_sparse_config);
  }
  input_config["label"] = input_label_config;
  input_config["dense"] = input_dense_config;
  input_config["sparse"] = input_sparse_config_array;
  layer_config_array.push_back(input_config);
  for (size_t i = 0; i < sparse_embedding_params.size(); ++i) {
    nlohmann::json sparse_config;
    sparse_config["type"] = EMBEDDING_TYPE_TO_STRING[sparse_embedding_params[i].embedding_type];
    sparse_config["bottom"] = sparse_embedding_params[i].bottom_name;
    sparse_config["top"] = sparse_embedding_params[i].sparse_embedding_name;
    nlohmann::json sparse_hparam_config;
    sparse_hparam_config["workspace_size_per_gpu_in_mb"] =
        sparse_embedding_params[i].workspace_size_per_gpu_in_mb;
    sparse_hparam_config["max_vocabulary_size_global"] =
        sparse_embedding_params[i].max_vocabulary_size_global;
    sparse_hparam_config["embedding_vec_size"] = sparse_embedding_params[i].embedding_vec_size;
    if (sparse_embedding_params[i].combiner == 0) {
      sparse_hparam_config["combiner"] = "sum";
    } else if (sparse_embedding_params[i].combiner == 1) {
      sparse_hparam_config["combiner"] = "mean";
    } else {
      CK_THROW_(Error_t::WrongInput, "combiner error");
    }
    if (sparse_embedding_params[i].slot_size_array.size() > 0) {
      sparse_hparam_config["slot_size_array"] = sparse_embedding_params[i].slot_size_array;
    }
    if (sparse_embedding_params[i].embedding_type == Embedding_t::HybridSparseEmbedding) {
      sparse_hparam_config["max_num_frequent_categories"] =
          sparse_embedding_params[i].hybrid_embedding_param.max_num_frequent_categories;
      sparse_hparam_config["max_num_infrequent_samples"] =
          sparse_embedding_params[i].hybrid_embedding_param.max_num_infrequent_samples;
      sparse_hparam_config["p_dup_max"] =
          sparse_embedding_params[i].hybrid_embedding_param.p_dup_max;
      sparse_hparam_config["max_all_reduce_bandwidth"] =
          sparse_embedding_params[i].hybrid_embedding_param.max_all_reduce_bandwidth;
      sparse_hparam_config["max_all_to_all_bandwidth"] =
          sparse_embedding_params[i].hybrid_embedding_param.max_all_to_all_bandwidth;
      sparse_hparam_config["efficiency_bandwidth_ratio"] =
          sparse_embedding_params[i].hybrid_embedding_param.efficiency_bandwidth_ratio;
      sparse_hparam_config["communication_type"] =
          HE_COMM_TYPE_TO_STRING[sparse_embedding_params[i]
                                     .hybrid_embedding_param.communication_type];
      sparse_hparam_config["hybrid_embedding_type"] =
          HE_TYPE_TO_STRING[sparse_embedding_params[i]
                                .hybrid_embedding_param.hybrid_embedding_type];
    }
    sparse_config["sparse_embedding_hparam"] = sparse_hparam_config;
    nlohmann::json optimizer_config;
    nlohmann::json optimizer_hparam_config;
    optimizer_config["update_type"] =
        embedding_opt_params_list[i]->update_type == Update_t::Global
            ? "Global"
            : (embedding_opt_params_list[i]->update_type == Update_t::Local ? "Local"
                                                                            : "LazyGlobal");
    switch (embedding_opt_params_list[i]->optimizer) {
      case Optimizer_t::Adam: {
        optimizer_config["type"] = "Adam";
        optimizer_hparam_config["beta1"] = embedding_opt_params_list[i]->hyperparams.adam.beta1;
        optimizer_hparam_config["beta2"] = embedding_opt_params_list[i]->hyperparams.adam.beta2;
        optimizer_hparam_config["epsilon"] = embedding_opt_params_list[i]->hyperparams.adam.epsilon;
        optimizer_config["adam_hparam"] = optimizer_hparam_config;
        break;
      }
      case Optimizer_t::AdaGrad: {
        optimizer_config["type"] = "AdaGrad";
        optimizer_hparam_config["initial_accu_value"] =
            embedding_opt_params_list[i]->hyperparams.adagrad.initial_accu_value;
        optimizer_hparam_config["epsilon"] =
            embedding_opt_params_list[i]->hyperparams.adagrad.epsilon;
        optimizer_config["adagrad_hparam"] = optimizer_hparam_config;
        break;
      }
      case Optimizer_t::MomentumSGD: {
        optimizer_config["type"] = "MomentumSGD";
        optimizer_hparam_config["momentum_factor"] =
            embedding_opt_params_list[i]->hyperparams.momentum.factor;
        optimizer_config["momentum_sgd_hparam"] = optimizer_hparam_config;
        break;
      }
      case Optimizer_t::Nesterov: {
        optimizer_config["type"] = "Nesterov";
        optimizer_hparam_config["momentum_factor"] =
            embedding_opt_params_list[i]->hyperparams.nesterov.mu;
        optimizer_config["nesterov_hparam"] = optimizer_hparam_config;
        break;
      }
      case Optimizer_t::SGD: {
        optimizer_config["type"] = "SGD";
        optimizer_hparam_config["atomic_update"] =
            embedding_opt_params_list[i]->hyperparams.sgd.atomic_update;
        optimizer_config["sgd_hparam"] = optimizer_hparam_config;
        break;
      }
      default: {
        assert(!"Error: no such optimizer && should never get here!");
      }
    }
    sparse_config["optimizer"] = optimizer_config;
    layer_config_array.push_back(sparse_config);
  }
  auto& layer_type_to_string = use_mixed_precision ? LAYER_TYPE_TO_STRING_MP : LAYER_TYPE_TO_STRING;
  for (size_t i = 0; i < dense_layer_params.size(); ++i) {
    nlohmann::json layer_config;
    layer_config["type"] = layer_type_to_string[dense_layer_params[i].layer_type];
    if (dense_layer_params[i].bottom_names.size() == 1) {
      layer_config["bottom"] = dense_layer_params[i].bottom_names[0];
    } else {
      layer_config["bottom"] = dense_layer_params[i].bottom_names;
    }
    if (dense_layer_params[i].top_names.size() == 1) {
      layer_config["top"] = dense_layer_params[i].top_names[0];
    } else {
      layer_config["top"] = dense_layer_params[i].top_names;
    }
    switch (dense_layer_params[i].layer_type) {
      case Layer_t::BatchNorm: {
        nlohmann::json bn_param_config;
        bn_param_config["factor"] = dense_layer_params[i].factor;
        bn_param_config["eps"] = dense_layer_params[i].eps;
        if (dense_layer_params[i].gamma_init_type != Initializer_t::Default) {
          bn_param_config["gamma_init"] =
              INITIALIZER_TYPE_TO_STRING[dense_layer_params[i].gamma_init_type];
        }
        if (dense_layer_params[i].beta_init_type != Initializer_t::Default) {
          bn_param_config["beta_init"] =
              INITIALIZER_TYPE_TO_STRING[dense_layer_params[i].beta_init_type];
        }
        layer_config["bn_param"] = bn_param_config;
        break;
      }
      case Layer_t::Dropout: {
        layer_config["rate"] = dense_layer_params[i].dropout_rate;
        break;
      }
      case Layer_t::ELU: {
        nlohmann::json elu_param_config;
        elu_param_config["alpha"] = dense_layer_params[i].elu_alpha;
        layer_config["elu_param"] = elu_param_config;
        break;
      }
      case Layer_t::FusedInnerProduct: {
        nlohmann::json fc_param_config;
        fc_param_config["num_output"] = dense_layer_params[i].num_output;
        if (dense_layer_params[i].weight_init_type != Initializer_t::Default) {
          fc_param_config["weight_init"] =
              INITIALIZER_TYPE_TO_STRING[dense_layer_params[i].weight_init_type];
        }
        if (dense_layer_params[i].bias_init_type != Initializer_t::Default) {
          fc_param_config["bias_init"] =
              INITIALIZER_TYPE_TO_STRING[dense_layer_params[i].bias_init_type];
        }
        layer_config["position"] = FC_POSITION_TO_STRING[dense_layer_params[i].pos_type];
        layer_config["activation"] = FC_ACTIVATION_TO_STRING[dense_layer_params[i].act_type];
        layer_config["fc_param"] = fc_param_config;
        break;
      }
      case Layer_t::InnerProduct: {
        nlohmann::json fc_param_config;
        fc_param_config["num_output"] = dense_layer_params[i].num_output;
        if (dense_layer_params[i].weight_init_type != Initializer_t::Default) {
          fc_param_config["weight_init"] =
              INITIALIZER_TYPE_TO_STRING[dense_layer_params[i].weight_init_type];
        }
        if (dense_layer_params[i].bias_init_type != Initializer_t::Default) {
          fc_param_config["bias_init"] =
              INITIALIZER_TYPE_TO_STRING[dense_layer_params[i].bias_init_type];
        }
        layer_config["fc_param"] = fc_param_config;
        break;
      }
      case Layer_t::MultiCross: {
        nlohmann::json mc_param_config;
        mc_param_config["num_layers"] = dense_layer_params[i].num_layers;
        if (dense_layer_params[i].weight_init_type != Initializer_t::Default) {
          mc_param_config["weight_init"] =
              INITIALIZER_TYPE_TO_STRING[dense_layer_params[i].weight_init_type];
        }
        if (dense_layer_params[i].bias_init_type != Initializer_t::Default) {
          mc_param_config["bias_init"] =
              INITIALIZER_TYPE_TO_STRING[dense_layer_params[i].bias_init_type];
        }
        layer_config["mc_param"] = mc_param_config;
        break;
      }
      case Layer_t::Reshape: {
        if (dense_layer_params[i].selected) {
          layer_config["selected"] = dense_layer_params[i].selected_slots;
        } else {
          layer_config["leading_dim"] = dense_layer_params[i].leading_dim;
          layer_config["time_step"] = dense_layer_params[i].time_step;
        }
        break;
      }
      case Layer_t::Slice: {
        layer_config["ranges"] = dense_layer_params[i].ranges;
        break;
      }
      case Layer_t::WeightMultiply: {
        layer_config["weight_dims"] = dense_layer_params[i].weight_dims;
        if (dense_layer_params[i].weight_init_type != Initializer_t::Default) {
          layer_config["weight_init"] =
              INITIALIZER_TYPE_TO_STRING[dense_layer_params[i].weight_init_type];
        }
        break;
      }
      case Layer_t::FmOrder2: {
        layer_config["out_dim"] = dense_layer_params[i].out_dim;
        break;
      }
      case Layer_t::ReduceSum: {
        layer_config["axis"] = dense_layer_params[i].axis;
        break;
      }
      case Layer_t::ReduceMean: {
        layer_config["axis"] = dense_layer_params[i].axis;
        break;
      }
      case Layer_t::Gather: {
        layer_config["indices"] = dense_layer_params[i].indices;
        break;
      }
      case Layer_t::GRU: {
        nlohmann::json gru_param_config;
        gru_param_config["num_output"] = dense_layer_params[i].num_output;
        gru_param_config["batchsize"] = dense_layer_params[i].batchsize;
        gru_param_config["SeqLength"] = dense_layer_params[i].SeqLength;
        gru_param_config["vector_size"] = dense_layer_params[i].vector_size;
        if (dense_layer_params[i].weight_init_type != Initializer_t::Default) {
          gru_param_config["weight_init"] =
              INITIALIZER_TYPE_TO_STRING[dense_layer_params[i].weight_init_type];
        }
        if (dense_layer_params[i].bias_init_type != Initializer_t::Default) {
          gru_param_config["bias_init"] =
              INITIALIZER_TYPE_TO_STRING[dense_layer_params[i].bias_init_type];
        }
        layer_config["gru_param"] = gru_param_config;
        break;
      }
      case Layer_t::PReLU_Dice: {
        nlohmann::json prelu_dice_param_config;
        prelu_dice_param_config["alpha"] = dense_layer_params[i].elu_alpha;
        prelu_dice_param_config["eps"] = dense_layer_params[i].eps;
        layer_config["prelu_dice_param"] = prelu_dice_param_config;
        break;
      }
      case Layer_t::Scale: {
        nlohmann::json scale_param_config;
        scale_param_config["axis"] = dense_layer_params[i].axis;
        scale_param_config["factor"] = dense_layer_params[i].factor;
        layer_config["scale_param"] = scale_param_config;
        break;
      }
      case Layer_t::MultiCrossEntropyLoss: {
        if (dense_layer_params[i].target_weight_vec.size() > 0) {
          layer_config["target_weight"] = dense_layer_params[i].target_weight_vec;
        }
        if (dense_layer_params[i].use_regularizer) {
          layer_config["regularizer"] =
              dense_layer_params[i].regularizer_type == Regularizer_t::L1 ? "L1" : "L2";
          layer_config["lambda"] = dense_layer_params[i].lambda;
        }
        break;
      }
      case Layer_t::BinaryCrossEntropyLoss: {
        if (dense_layer_params[i].use_regularizer) {
          layer_config["regularizer"] =
              dense_layer_params[i].regularizer_type == Regularizer_t::L1 ? "L1" : "L2";
          layer_config["lambda"] = dense_layer_params[i].lambda;
        }
        break;
      }
      case Layer_t::CrossEntropyLoss: {
        if (dense_layer_params[i].use_regularizer) {
          layer_config["regularizer"] =
              dense_layer_params[i].regularizer_type == Regularizer_t::L1 ? "L1" : "L2";
          layer_config["lambda"] = dense_layer_params[i].lambda;
        }
        break;
      }
      default: {
        break;
      }
    }
    layer_config_array.push_back(layer_config);
  }
}

DenseLayer get_dense_layer_from_json(const nlohmann::json& j_dense_layer) {
  Layer_t layer_type;
  auto layer_type_name = get_value_from_json<std::string>(j_dense_layer, "type");
  if (!find_item_in_map(layer_type, layer_type_name, LAYER_TYPE_MAP) &&
      !find_item_in_map(layer_type, layer_type_name, LAYER_TYPE_MAP_MP)) {
    CK_THROW_(Error_t::WrongInput, "No such layer: " + layer_type_name);
  }
  auto bottom = get_json(j_dense_layer, "bottom");
  auto top = get_json(j_dense_layer, "top");
  std::vector<std::string> bottom_names = get_layer_names(bottom);
  std::vector<std::string> top_names = get_layer_names(top);
  DenseLayer dense_layer = DenseLayer(layer_type, bottom_names, top_names);
  switch (layer_type) {
    case Layer_t::BatchNorm: {
      auto j_bn_hparam = get_json(j_dense_layer, "bn_param");
      auto factor = get_value_from_json<float>(j_bn_hparam, "factor");
      auto eps = get_value_from_json<float>(j_bn_hparam, "eps");
      dense_layer.factor = factor;
      dense_layer.eps = eps;
      if (has_key_(j_bn_hparam, "gamma_init")) {
        const auto gamma_init_name = get_value_from_json<std::string>(j_bn_hparam, "gamma_init");
        Initializer_t gamma_init_type;
        if (find_item_in_map(gamma_init_type, gamma_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.gamma_init_type = gamma_init_type;
        } else {
          CK_THROW_(Error_t::WrongInput, "No such initializer: " + gamma_init_name);
        }
      }
      if (has_key_(j_bn_hparam, "beta_init")) {
        const auto beta_init_name = get_value_from_json<std::string>(j_bn_hparam, "beta_init");
        Initializer_t beta_init_type;
        if (find_item_in_map(beta_init_type, beta_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.beta_init_type = beta_init_type;
        } else {
          CK_THROW_(Error_t::WrongInput, "No such initializer: " + beta_init_name);
        }
      }
      break;
    }
    case Layer_t::Dropout: {
      auto rate_it = j_dense_layer.find("rate");
      if (rate_it != j_dense_layer.end()) {
        dense_layer.dropout_rate = rate_it->get<float>();
      }
      break;
    }
    case Layer_t::ELU: {
      auto j_elu_hparam = get_json(j_dense_layer, "elu_param");
      auto alpha = get_value_from_json<float>(j_elu_hparam, "alpha");
      dense_layer.elu_alpha = alpha;
      break;
    }
    case Layer_t::FusedInnerProduct: {
      auto j_fc_param = get_json(j_dense_layer, "fc_param");
      if (has_key_(j_fc_param, "weight_init")) {
        const auto weight_init_name = get_value_from_json<std::string>(j_fc_param, "weight_init");
        Initializer_t weight_init_type;
        if (find_item_in_map(weight_init_type, weight_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.weight_init_type = weight_init_type;
        } else {
          CK_THROW_(Error_t::WrongInput, "No such initializer: " + weight_init_name);
        }
      }
      if (has_key_(j_fc_param, "bias_init")) {
        const auto bias_init_name = get_value_from_json<std::string>(j_fc_param, "bias_init");
        Initializer_t bias_init_type;
        if (find_item_in_map(bias_init_type, bias_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.bias_init_type = bias_init_type;
        } else {
          CK_THROW_(Error_t::WrongInput, "No such initializer: " + bias_init_name);
        }
      }
      if (has_key_(j_dense_layer, "position")) {
        const auto position_name = get_value_from_json<std::string>(j_dense_layer, "position");
        FcPosition_t pos_type;
        if (find_item_in_map(pos_type, position_name, FCPOSITION_TYPE_MAP)) {
          dense_layer.pos_type = pos_type;
        } else {
          CK_THROW_(Error_t::WrongInput, "No such position: " + position_name);
        }
      }
      if (has_key_(j_dense_layer, "activation")) {
        const auto act_name = get_value_from_json<std::string>(j_dense_layer, "activation");
        Activation_t act_type;
        if (find_item_in_map(act_type, act_name, ACTIVATION_TYPE_MAP)) {
          dense_layer.act_type = act_type;
        } else {
          CK_THROW_(Error_t::WrongInput, "No such activation: " + act_name);
        }
      }
      // establish out tensor
      auto output = get_value_from_json<size_t>(j_fc_param, "num_output");
      dense_layer.num_output = output;
      break;
    }
    case Layer_t::InnerProduct: {
      auto j_fc_param = get_json(j_dense_layer, "fc_param");
      if (has_key_(j_fc_param, "weight_init")) {
        const auto weight_init_name = get_value_from_json<std::string>(j_fc_param, "weight_init");
        Initializer_t weight_init_type;
        if (find_item_in_map(weight_init_type, weight_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.weight_init_type = weight_init_type;
        } else {
          CK_THROW_(Error_t::WrongInput, "No such initializer: " + weight_init_name);
        }
      }
      if (has_key_(j_fc_param, "bias_init")) {
        const auto bias_init_name = get_value_from_json<std::string>(j_fc_param, "bias_init");
        Initializer_t bias_init_type;
        if (find_item_in_map(bias_init_type, bias_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.bias_init_type = bias_init_type;
        } else {
          CK_THROW_(Error_t::WrongInput, "No such initializer: " + bias_init_name);
        }
      }
      // establish out tensor
      auto output = get_value_from_json<size_t>(j_fc_param, "num_output");
      dense_layer.num_output = output;
      break;
    }
    case Layer_t::MultiCross: {
      auto j_mc_param = get_json(j_dense_layer, "mc_param");
      if (has_key_(j_mc_param, "weight_init")) {
        const auto weight_init_name = get_value_from_json<std::string>(j_mc_param, "weight_init");
        Initializer_t weight_init_type;
        if (find_item_in_map(weight_init_type, weight_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.weight_init_type = weight_init_type;
        } else {
          CK_THROW_(Error_t::WrongInput, "No such initializer: " + weight_init_name);
        }
      }
      if (has_key_(j_mc_param, "bias_init")) {
        const auto bias_init_name = get_value_from_json<std::string>(j_mc_param, "bias_init");
        Initializer_t bias_init_type;
        if (find_item_in_map(bias_init_type, bias_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.bias_init_type = bias_init_type;
        } else {
          CK_THROW_(Error_t::WrongInput, "No such initializer: " + bias_init_name);
        }
      }
      auto num_layers = get_value_from_json<int>(j_mc_param, "num_layers");
      dense_layer.num_layers = num_layers;
      break;
    }
    case Layer_t::Reshape: {
      auto selected_it = j_dense_layer.find("selected");
      if (selected_it != j_dense_layer.end()) {
        std::vector<int> selected;
        nlohmann::json j_selected = (selected_it.value());
        for (auto slot_obj : j_selected) {
          int slot_id = slot_obj.get<int>();
          if (slot_id < 0) CK_THROW_(Error_t::WrongInput, "slot_id < 0");
          selected.push_back(slot_id);
        }
        dense_layer.selected = true;
        dense_layer.selected_slots = selected;
      } else {
        auto leading_dim = get_value_from_json<size_t>(j_dense_layer, "leading_dim");
        dense_layer.selected = false;
        dense_layer.leading_dim = leading_dim;
        auto time_step = j_dense_layer.find("time_step");
        if (time_step != j_dense_layer.end()) {
          dense_layer.time_step = time_step.value();
        } else {
          dense_layer.time_step = 0;
        }
      }
      break;
    }
    case Layer_t::Slice: {
      std::vector<std::pair<int, int>> ranges;
      auto j_ranges = get_json(j_dense_layer, "ranges");
      assert(j_ranges.is_array());
      for (auto j_range : j_ranges) {
        assert(j_range.is_array());
        ranges.emplace_back(std::make_pair(j_range[0].get<int>(), j_range[1].get<int>()));
      }
      dense_layer.ranges = ranges;
      break;
    }
    case Layer_t::ReduceMean: {
      int axis = get_json(j_dense_layer, "axis").get<int>();
      dense_layer.axis = axis;
      break;
    }
    case Layer_t::Gather: {
      std::vector<int> indices;
      auto j_indices = get_json(j_dense_layer, "indices");
      assert(j_indices.is_array());
      for (auto j_indice : j_indices) {
        indices.emplace_back(j_indice.get<int>());
      }
      dense_layer.indices = indices;
      break;
    }
    case Layer_t::GRU: {
      auto j_gru_param = get_json(j_dense_layer, "gru_param");
      if (has_key_(j_gru_param, "weight_init")) {
        const auto weight_init_name = get_value_from_json<std::string>(j_gru_param, "weight_init");
        Initializer_t weight_init_type;
        if (find_item_in_map(weight_init_type, weight_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.weight_init_type = weight_init_type;
        } else {
          CK_THROW_(Error_t::WrongInput, "No such initializer: " + weight_init_name);
        }
      }
      if (has_key_(j_gru_param, "bias_init")) {
        const auto bias_init_name = get_value_from_json<std::string>(j_gru_param, "bias_init");
        Initializer_t bias_init_type;
        if (find_item_in_map(bias_init_type, bias_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.bias_init_type = bias_init_type;
        } else {
          CK_THROW_(Error_t::WrongInput, "No such initializer: " + bias_init_name);
        }
      }
      auto num_output = get_value_from_json<int>(j_gru_param, "num_output");
      auto batchsize = get_value_from_json<int>(j_gru_param, "batchsize");
      auto SeqLength = get_value_from_json<int>(j_gru_param, "SeqLength");
      auto vector_size = get_value_from_json<int>(j_gru_param, "vector_size");
      dense_layer.num_output = num_output;
      dense_layer.batchsize = batchsize;
      dense_layer.SeqLength = SeqLength;
      dense_layer.vector_size = vector_size;
      break;
    }
    case Layer_t::PReLU_Dice: {
      auto j_prelu_dice_hparam = get_json(j_dense_layer, "prelu_dice_param");
      auto alpha = get_value_from_json<float>(j_prelu_dice_hparam, "elu_alpha");
      auto epsilon = get_value_from_json<float>(j_prelu_dice_hparam, "eps");
      dense_layer.elu_alpha = alpha;
      dense_layer.eps = epsilon;
      break;
    }
    case Layer_t::Scale: {
      auto j_scale_hparam = get_json(j_dense_layer, "scale_param");
      int axis = get_value_from_json<float>(j_scale_hparam, "axis");
      auto factor = get_value_from_json<float>(j_scale_hparam, "factor");
      dense_layer.axis = axis;
      dense_layer.factor = factor;
      break;
    }
    case Layer_t::WeightMultiply: {
      std::vector<size_t> weight_dims;
      auto dims = get_json(j_dense_layer, "weight_dims");
      assert(dims.is_array());
      for (auto dim : dims) {
        weight_dims.emplace_back(dim.get<size_t>());
      }
      dense_layer.weight_dims = weight_dims;
      if (has_key_(j_dense_layer, "weight_init")) {
        const auto weight_init_name =
            get_value_from_json<std::string>(j_dense_layer, "weight_init");
        Initializer_t weight_init_type;
        if (find_item_in_map(weight_init_type, weight_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.weight_init_type = weight_init_type;
        } else {
          CK_THROW_(Error_t::WrongInput, "No such initializer: " + weight_init_name);
        }
      }
      break;
    }
    case Layer_t::FmOrder2: {
      auto out_dim = get_json(j_dense_layer, "out_dim").get<size_t>();
      dense_layer.out_dim = out_dim;
      break;
    }
    case Layer_t::ReduceSum: {
      int axis = get_json(j_dense_layer, "axis").get<int>();
      dense_layer.axis = axis;
      break;
    }
    case Layer_t::MultiCrossEntropyLoss: {
      auto tweight = get_json(j_dense_layer, "target_weight");
      std::vector<float> target_weight_vec;
      for (auto tweight_tmp : tweight) {
        float tweight_val = tweight_tmp.get<float>();
        target_weight_vec.push_back(tweight_val);
      }
      dense_layer.target_weight_vec = target_weight_vec;
      auto reg_it = j_dense_layer.find("regularizer");
      if (reg_it != j_dense_layer.end()) {
        Regularizer_t reg_type;
        auto reg_name = reg_it->get<std::string>();
        if (find_item_in_map(reg_type, reg_name, REGULARIZER_TYPE_MAP)) {
          const auto lambda = get_value_from_json<float>(j_dense_layer, "lambda");
          dense_layer.use_regularizer = true;
          dense_layer.regularizer_type = reg_type;
          dense_layer.lambda = lambda;
        } else {
          CK_THROW_(Error_t::WrongInput, "No such regularizer: " + reg_name);
        }
      } else {
        dense_layer.use_regularizer = false;
      }
      break;
    }
    case Layer_t::BinaryCrossEntropyLoss: {
      auto reg_it = j_dense_layer.find("regularizer");
      if (reg_it != j_dense_layer.end()) {
        Regularizer_t reg_type;
        auto reg_name = reg_it->get<std::string>();
        if (find_item_in_map(reg_type, reg_name, REGULARIZER_TYPE_MAP)) {
          const auto lambda = get_value_from_json<float>(j_dense_layer, "lambda");
          dense_layer.use_regularizer = true;
          dense_layer.regularizer_type = reg_type;
          dense_layer.lambda = lambda;
        } else {
          CK_THROW_(Error_t::WrongInput, "No such regularizer: " + reg_name);
        }
      } else {
        dense_layer.use_regularizer = false;
      }
      break;
    }
    case Layer_t::CrossEntropyLoss: {
      auto reg_it = j_dense_layer.find("regularizer");
      if (reg_it != j_dense_layer.end()) {
        Regularizer_t reg_type;
        auto reg_name = reg_it->get<std::string>();
        if (find_item_in_map(reg_type, reg_name, REGULARIZER_TYPE_MAP)) {
          const auto lambda = get_value_from_json<float>(j_dense_layer, "lambda");
          dense_layer.use_regularizer = true;
          dense_layer.regularizer_type = reg_type;
          dense_layer.lambda = lambda;
        } else {
          CK_THROW_(Error_t::WrongInput, "No such regularizer: " + reg_name);
        }
      } else {
        dense_layer.use_regularizer = false;
      }
      break;
    }
    default: {
      break;
    }
  }
  return dense_layer;
}

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

static InputOutputInfo get_input_tensor_and_output_name(
    std::vector<std::string>& bottom_names, std::vector<std::string>& top_names,
    const std::vector<TensorEntry>& tensor_entries) {
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

template <typename T>
static std::shared_ptr<Regularizer<T>> create_regularizer(
    bool use_regularizer, Regularizer_t regularizer_type, float lambda,
    const Tensor2<float>& weight_buff, const Tensor2<T>& wgrad_buff, const int batch_size,
    const std::shared_ptr<GPUResource>& gpu_resource) {
  std::shared_ptr<Regularizer<T>> reg(
      new NoRegularizer<T>(weight_buff, wgrad_buff, batch_size, gpu_resource));
  if (use_regularizer) {
    switch (regularizer_type) {
      case Regularizer_t::L1: {
        reg.reset(new L1Regularizer<T>(weight_buff, wgrad_buff, batch_size, lambda, gpu_resource));
        break;
      }
      case Regularizer_t::L2: {
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

void Model::add_dense_layer_internal(
    DenseLayer& dense_layer, std::vector<TensorEntry>& tensor_entries,
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
    const std::shared_ptr<BufferBlock2<float>>& weight_buff,
    const std::shared_ptr<BufferBlock2<__half>>& weight_buff_half,
    const std::shared_ptr<BufferBlock2<float>>& wgrad_buff,
    const std::shared_ptr<BufferBlock2<__half>>& wgrad_buff_half, Tensor2<float>& loss_tensor,
    std::vector<std::unique_ptr<Layer>>& layers, std::unique_ptr<ILoss>& loss,
    bool enable_cuda_graph, bool async_mlp_wgrad, metrics::RawMetricMap* raw_metrics,
    int num_networks_in_global, const std::shared_ptr<GPUResource>& gpu_resource,
    bool use_mixed_precision, bool enable_tf32_compute, float scaler, bool use_algorithm_search,
    std::vector<Layer*>* top_layers, std::vector<Layer*>* bottom_layers, bool dlrm_bottom_mlp) {
  bool skip_dgrad = layers.size() == 0;
  Layer_t layer_type = dense_layer.layer_type;
  const auto& layer_type_to_string =
      use_mixed_precision ? LAYER_TYPE_TO_STRING_MP : LAYER_TYPE_TO_STRING;
  if (layer_type_to_string.find(layer_type) == layer_type_to_string.end()) {
    if (use_mixed_precision) {
      auto layer_type_name = LAYER_TYPE_TO_STRING[layer_type];
      CK_THROW_(Error_t::WrongInput, "Mixed precision not supported for: " + layer_type_name);
    } else {
      auto layer_type_name = LAYER_TYPE_TO_STRING_MP[layer_type];
      CK_THROW_(Error_t::WrongInput, "Single precision not supported for: " + layer_type_name);
    }
  }
  std::vector<TensorEntry> output_tensor_entries;
  auto input_output_info = get_input_tensor_and_output_name(dense_layer.bottom_names,
                                                            dense_layer.top_names, tensor_entries);
  switch (layer_type) {
    case Layer_t::BatchNorm: {
      if (use_mixed_precision) {
        Tensor2<__half> bn_in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        // establish out tensor
        Tensor2<__half> bn_out_tensor;
        blobs_buff->reserve(bn_in_tensor.get_dimensions(), &bn_out_tensor);
        output_tensor_entries.push_back(
            {input_output_info.output_names[0], bn_out_tensor.shrink()});
        std::vector<Initializer_t> initializer_types{dense_layer.gamma_init_type,
                                                     dense_layer.beta_init_type};
        BatchNormLayer<__half>::Params params = {dense_layer.factor, dense_layer.eps};
        layers.emplace_back(new BatchNormLayer<__half>(weight_buff, wgrad_buff, blobs_buff,
                                                       bn_in_tensor, bn_out_tensor, params,
                                                       gpu_resource, initializer_types));
      } else {
        Tensor2<float> bn_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        // establish out tensor
        Tensor2<float> bn_out_tensor;
        blobs_buff->reserve(bn_in_tensor.get_dimensions(), &bn_out_tensor);
        output_tensor_entries.push_back(
            {input_output_info.output_names[0], bn_out_tensor.shrink()});
        std::vector<Initializer_t> initializer_types{dense_layer.gamma_init_type,
                                                     dense_layer.beta_init_type};
        BatchNormLayer<float>::Params params = {dense_layer.factor, dense_layer.eps};
        layers.emplace_back(new BatchNormLayer<float>(weight_buff, wgrad_buff, blobs_buff,
                                                      bn_in_tensor, bn_out_tensor, params,
                                                      gpu_resource, initializer_types));
      }
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
            create_regularizer(dense_layer.use_regularizer, dense_layer.regularizer_type,
                               dense_layer.lambda, weight_buff->as_tensor(),
                               wgrad_buff_half->as_tensor(), in_tensor.get_dimensions()[0],
                               gpu_resource),
            gpu_resource, num_networks_in_global, scaler));
      } else {
        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        loss.reset(new BinaryCrossEntropyLoss<float>(
            label_tensor, in_tensor, loss_tensor,
            create_regularizer(dense_layer.use_regularizer, dense_layer.regularizer_type,
                               dense_layer.lambda, weight_buff->as_tensor(),
                               wgrad_buff->as_tensor(), in_tensor.get_dimensions()[0],
                               gpu_resource),
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
        layers.emplace_back(
            new ConcatLayer<__half>(in_tensors, out_tensor, blobs_buff, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      } else {
        Tensors2<float> in_tensors;
        for (const TensorBag2& bag : input_output_info.inputs) {
          in_tensors.push_back(Tensor2<float>::stretch_from(bag));
        }
        Tensor2<float> out_tensor;
        layers.emplace_back(
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
            create_regularizer(dense_layer.use_regularizer, dense_layer.regularizer_type,
                               dense_layer.lambda, weight_buff->as_tensor(),
                               wgrad_buff_half->as_tensor(),
                               cross_entropy_loss_in_tensor.get_dimensions()[0], gpu_resource),
            gpu_resource, num_networks_in_global, scaler));
      } else {
        Tensor2<float> cross_entropy_loss_in_tensor =
            Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        loss.reset(new CrossEntropyLoss<float>(
            label_tensor, cross_entropy_loss_in_tensor, loss_tensor,
            create_regularizer(dense_layer.use_regularizer, dense_layer.regularizer_type,
                               dense_layer.lambda, weight_buff->as_tensor(),
                               wgrad_buff->as_tensor(),
                               cross_entropy_loss_in_tensor.get_dimensions()[0], gpu_resource),
            gpu_resource, num_networks_in_global, scaler));
      }
      break;
    }
    case Layer_t::Dropout: {
      if (use_mixed_precision) {
        Tensor2<__half> do_in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        Tensor2<__half> do_out_tensor;
        blobs_buff->reserve(do_in_tensor.get_dimensions(), &do_out_tensor);
        output_tensor_entries.push_back(
            {input_output_info.output_names[0], do_out_tensor.shrink()});
        float rate = dense_layer.dropout_rate;
        layers.emplace_back(
            new DropoutLayer<__half>(do_in_tensor, do_out_tensor, blobs_buff, rate, gpu_resource));
      } else {
        Tensor2<float> do_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> do_out_tensor;
        blobs_buff->reserve(do_in_tensor.get_dimensions(), &do_out_tensor);
        output_tensor_entries.push_back(
            {input_output_info.output_names[0], do_out_tensor.shrink()});
        float rate = dense_layer.dropout_rate;
        layers.emplace_back(
            new DropoutLayer<float>(do_in_tensor, do_out_tensor, blobs_buff, rate, gpu_resource));
      }
      // to be fixed
      break;
    }
    case Layer_t::ELU: {
      if (use_mixed_precision) {
        Tensor2<__half> elu_in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        Tensor2<__half> elu_out_tensor;
        blobs_buff->reserve(elu_in_tensor.get_dimensions(), &elu_out_tensor);
        output_tensor_entries.push_back(
            {input_output_info.output_names[0], elu_out_tensor.shrink()});
        float alpha = dense_layer.elu_alpha;
        layers.emplace_back(
            new EluLayer<__half>(elu_in_tensor, elu_out_tensor, alpha, gpu_resource));
      } else {
        Tensor2<float> elu_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> elu_out_tensor;
        blobs_buff->reserve(elu_in_tensor.get_dimensions(), &elu_out_tensor);
        output_tensor_entries.push_back(
            {input_output_info.output_names[0], elu_out_tensor.shrink()});
        float alpha = dense_layer.elu_alpha;
        layers.emplace_back(
            new EluLayer<float>(elu_in_tensor, elu_out_tensor, alpha, gpu_resource));
      }
      break;
    }
    case Layer_t::FusedInnerProduct: {
      std::vector<Initializer_t> initializer_types{dense_layer.weight_init_type,
                                                   dense_layer.bias_init_type};
      // check the position of this layer
      int input_size = input_output_info.inputs.size();
      int output_size = input_output_info.output_names.size();
      size_t output = dense_layer.num_output;
      auto pos_type = dense_layer.pos_type;
      auto act_type = dense_layer.act_type;
      bool head_mask_in = pos_type == FcPosition_t::Head && input_size == 2;
      if (skip_dgrad && pos_type == FcPosition_t::Head && input_size == 2) {
        CK_THROW_(Error_t::WrongInput,
                  "FusedInnerProduct Head Layer should have only one input tensors when it is the "
                  "first dense layer");
      }
      if (async_mlp_wgrad && !skip_dgrad && pos_type == FcPosition_t::Head && input_size == 1) {
        CK_THROW_(Error_t::WrongInput,
                  "FusedInnerProduct Head Layer should have two input tensors when turning on "
                  "async wgrad knob");
      }
      if (pos_type == FcPosition_t::Head && input_size == 1 && output_size == 4) {
      } else if (pos_type == FcPosition_t::Head && input_size == 2 && output_size == 4) {
      } else if (pos_type == FcPosition_t::Body && input_size == 4 && output_size == 4) {
      } else if (pos_type == FcPosition_t::Tail && input_size == 4 && output_size == 1) {
      } else if (pos_type == FcPosition_t::Isolated && input_size == 1 && output_size == 1) {
      } else if (pos_type == FcPosition_t::None && input_size == 1 && output_size == 1) {
      } else {
        CK_THROW_(Error_t::WrongInput,
                  "The position and dimension of bottom and top layer aren't compatible: " +
                      LAYER_TYPE_TO_STRING_MP[layer_type]);
      }
      if (act_type == Activation_t::None && pos_type != FcPosition_t::Tail) {
        CK_THROW_(Error_t::WrongInput,
                  "The layer without activation function must be the last layer in MLP.");
      }
      if (use_mixed_precision) {
        Tensor2<__half> train_in_tensor =
            Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        Tensor2<__half> mask_in_tensor, dRelu_in_tensor, db_in_tensor;
        if (pos_type == FcPosition_t::Body || pos_type == FcPosition_t::Tail) {
          mask_in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[1]);
          dRelu_in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[2]);
          db_in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[3]);
        } else if (pos_type == FcPosition_t::Head && input_size == 2) {
          mask_in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[1]);
        }
        Tensor2<__half> train_out_tensor, mask_out_tensor, dRelu_out_tensor, db_out_tensor;
        blobs_buff->reserve({(train_in_tensor.get_dimensions())[0], output}, &train_out_tensor);
        blobs_buff->reserve({(train_in_tensor.get_dimensions())[0], output}, &mask_out_tensor);
        blobs_buff->reserve({(train_in_tensor.get_dimensions())[0], output}, &dRelu_out_tensor);

        // establish layer
        if (pos_type == FcPosition_t::None) {
          layers.emplace_back(new FusedFullyConnectedLayer(
              weight_buff, weight_buff_half, wgrad_buff_half, blobs_buff, train_in_tensor,
              train_out_tensor, gpu_resource, initializer_types));
        } else {
          layers.emplace_back(new FusedReluBiasFullyConnectedLayer(
              weight_buff, weight_buff_half, wgrad_buff_half, blobs_buff, train_in_tensor,
              mask_in_tensor, dRelu_in_tensor, db_in_tensor, train_out_tensor, mask_out_tensor,
              dRelu_out_tensor, db_out_tensor, gpu_resource, pos_type, act_type, skip_dgrad,
              initializer_types, async_mlp_wgrad, head_mask_in));
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
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        layers.emplace_back(new CastLayer<float, __half>(in_tensor, out_tensor, gpu_resource));
      } else {
        Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> out_tensor;
        blobs_buff->reserve(in_tensor.get_dimensions(), &out_tensor);
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        layers.emplace_back(new CastLayer<__half, float>(in_tensor, out_tensor, gpu_resource));
      }
      break;
    }
    case Layer_t::InnerProduct: {
      std::vector<Initializer_t> initializer_types{dense_layer.weight_init_type,
                                                   dense_layer.bias_init_type};
      size_t output = dense_layer.num_output;
      if (use_mixed_precision) {
        Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        Tensor2<__half> fc_out_tensor;
        blobs_buff->reserve({in_tensor.get_dimensions()[0], output}, &fc_out_tensor);
        layers.emplace_back(new FullyConnectedLayer<__half>(
            weight_buff, weight_buff_half, wgrad_buff_half, blobs_buff, in_tensor, fc_out_tensor,
            gpu_resource, initializer_types));
        output_tensor_entries.push_back(
            {input_output_info.output_names[0], fc_out_tensor.shrink()});
      } else {
        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> fc_out_tensor;
        blobs_buff->reserve({in_tensor.get_dimensions()[0], output}, &fc_out_tensor);
        layers.emplace_back(new FullyConnectedLayer<float>(
            weight_buff, wgrad_buff, in_tensor, fc_out_tensor, gpu_resource, use_mixed_precision,
            enable_tf32_compute, initializer_types));
        output_tensor_entries.push_back(
            {input_output_info.output_names[0], fc_out_tensor.shrink()});
      }
      break;
    }
    case Layer_t::Interaction: {
      if (input_output_info.inputs.size() != 2) {
        CK_THROW_(Error_t::WrongInput, "InteractionLayer needs two input tensors ");
      }
      if (input_output_info.output_names.size() != 2 &&
          input_output_info.output_names.size() != 1) {
        CK_THROW_(Error_t::WrongInput, "InteractionLayer should have one or two output tensors");
      }
      if (input_output_info.output_names.size() == 1 && async_mlp_wgrad == true) {
        CK_THROW_(
            Error_t::WrongInput,
            "InteractionLayer should have two output tensors when turning on async wgrad knob");
      }
      if (input_output_info.output_names.size() == 2 && !use_mixed_precision) {
        CK_THROW_(Error_t::WrongInput,
                  "InteractionLayer<float> should have only one output tensor");
      }
      if (use_mixed_precision) {
        if (gpu_resource->get_cc_major() < 7) {
          CK_THROW_(Error_t::WrongInput, "InteractionLayer<__half> is not supported in SM " +
                                             std::to_string(gpu_resource->get_cc_major()) + "." +
                                             std::to_string(gpu_resource->get_cc_minor()));
        }
        Tensor2<__half> in_mlp_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        Tensor2<__half> in_emb_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[1]);
        Tensor2<__half> out_tensor, grad_tensor;
        if (input_output_info.output_names.size() == 2) {
          layers.emplace_back(new InteractionLayer<__half>(
              in_mlp_tensor, in_emb_tensor, out_tensor, grad_tensor, blobs_buff, gpu_resource,
              use_mixed_precision, enable_tf32_compute));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
          output_tensor_entries.push_back(
              {input_output_info.output_names[1], grad_tensor.shrink()});
        } else {
          layers.emplace_back(
              new InteractionLayer<__half>(in_mlp_tensor, in_emb_tensor, out_tensor, blobs_buff,
                                           gpu_resource, use_mixed_precision, enable_tf32_compute));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        }
      } else {
        Tensor2<float> in_mlp_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> in_emb_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[1]);
        Tensor2<float> out_tensor;
        layers.emplace_back(new InteractionLayer<float>(in_mlp_tensor, in_emb_tensor, out_tensor,
                                                        blobs_buff, gpu_resource,
                                                        use_mixed_precision, enable_tf32_compute));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      }
      break;
    }
    case Layer_t::MultiCross: {
      std::vector<Initializer_t> initializer_types{dense_layer.weight_init_type,
                                                   dense_layer.bias_init_type};
      int num_layers = dense_layer.num_layers;
      if (use_mixed_precision) {
        Tensor2<__half> mc_in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        Tensor2<__half> out_tensor;
        blobs_buff->reserve(mc_in_tensor.get_dimensions(), &out_tensor);
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        layers.emplace_back(new MultiCrossLayer<__half>(
            weight_buff, weight_buff_half, wgrad_buff_half, blobs_buff, mc_in_tensor, out_tensor,
            gpu_resource, num_layers, initializer_types));
      } else {
        Tensor2<float> mc_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> out_tensor;
        blobs_buff->reserve(mc_in_tensor.get_dimensions(), &out_tensor);
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        layers.emplace_back(new MultiCrossLayer<float>(
            weight_buff, weight_buff, wgrad_buff, blobs_buff, mc_in_tensor, out_tensor,
            gpu_resource, num_layers, initializer_types));
      }
      break;
    }
    case Layer_t::MultiCrossEntropyLoss: {
      if (input_output_info.inputs.size() != 2) {
        CK_THROW_(Error_t::WrongInput, "bottom of MultiCrossEntropyLoss must be two dim");
      }
      Tensor2<float> label_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[1]);
      blobs_buff->reserve({1, 1}, &loss_tensor);
      if (use_mixed_precision) {
        Tensor2<__half> multi_cross_entropy_loss_in_tensor =
            Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        loss.reset(new MultiCrossEntropyLoss<__half>(
            label_tensor, multi_cross_entropy_loss_in_tensor, loss_tensor,
            create_regularizer(
                dense_layer.use_regularizer, dense_layer.regularizer_type, dense_layer.lambda,
                weight_buff->as_tensor(), wgrad_buff_half->as_tensor(),
                multi_cross_entropy_loss_in_tensor.get_dimensions()[0], gpu_resource),
            dense_layer.target_weight_vec, gpu_resource, num_networks_in_global, scaler));
      } else {
        Tensor2<float> multi_cross_entropy_loss_in_tensor =
            Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        loss.reset(new MultiCrossEntropyLoss<float>(
            label_tensor, multi_cross_entropy_loss_in_tensor, loss_tensor,
            create_regularizer(
                dense_layer.use_regularizer, dense_layer.regularizer_type, dense_layer.lambda,
                weight_buff->as_tensor(), wgrad_buff->as_tensor(),
                multi_cross_entropy_loss_in_tensor.get_dimensions()[0], gpu_resource),
            dense_layer.target_weight_vec, gpu_resource, num_networks_in_global, scaler));
      }
      break;
    }
    case Layer_t::ReLU: {
      if (use_mixed_precision) {
        Tensor2<__half> relu_in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        Tensor2<__half> relu_out_tensor;
        blobs_buff->reserve(relu_in_tensor.get_dimensions(), &relu_out_tensor);
        layers.emplace_back(new ReluLayer<__half>(relu_in_tensor, relu_out_tensor, gpu_resource));
        output_tensor_entries.push_back(
            {input_output_info.output_names[0], relu_out_tensor.shrink()});
      } else {
        Tensor2<float> relu_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> relu_out_tensor;
        blobs_buff->reserve(relu_in_tensor.get_dimensions(), &relu_out_tensor);
        layers.emplace_back(new ReluLayer<float>(relu_in_tensor, relu_out_tensor, gpu_resource));
        output_tensor_entries.push_back(
            {input_output_info.output_names[0], relu_out_tensor.shrink()});
      }
      break;
    }
    case Layer_t::Reshape: {
      bool selected = dense_layer.selected;
      if (selected) {
        if (use_mixed_precision) {
          Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> out_tensor;
          layers.emplace_back(new ReshapeLayer<__half>(in_tensor, out_tensor, blobs_buff,
                                                       dense_layer.selected_slots, gpu_resource));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> out_tensor;
          layers.emplace_back(new ReshapeLayer<float>(in_tensor, out_tensor, blobs_buff,
                                                      dense_layer.selected_slots, gpu_resource));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        }
      } else {
        size_t leading_dim = dense_layer.leading_dim;
        size_t time_step = dense_layer.time_step;
        if (use_mixed_precision) {
          Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> out_tensor;
          if (time_step == 0) {  // 2D output
            blobs_buff->reserve({in_tensor.get_num_elements() / leading_dim, leading_dim},
                                &out_tensor);
          } else {  // 3D output
            size_t batch_size = in_tensor.get_num_elements() / leading_dim / time_step;
            blobs_buff->reserve({batch_size, time_step, leading_dim}, &out_tensor);
          }
          layers.emplace_back(
              new ReshapeLayer<__half>(in_tensor, out_tensor, blobs_buff, gpu_resource));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> out_tensor;
          if (time_step == 0) {  // 2D output
            blobs_buff->reserve({in_tensor.get_num_elements() / leading_dim, leading_dim},
                                &out_tensor);
          } else {  // 3D output
            size_t batch_size = in_tensor.get_num_elements() / leading_dim / time_step;
            blobs_buff->reserve({batch_size, time_step, leading_dim}, &out_tensor);
          }
          layers.emplace_back(
              new ReshapeLayer<float>(in_tensor, out_tensor, blobs_buff, gpu_resource));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        }
      }
      break;
    }
    case Layer_t::Sigmoid: {
      if (use_mixed_precision) {
        Tensor2<__half> sigmoid_in_tensor =
            Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        Tensor2<__half> sigmoid_out_tensor;
        blobs_buff->reserve(sigmoid_in_tensor.get_dimensions(), &sigmoid_out_tensor);
        layers.emplace_back(
            new SigmoidLayer<__half>(sigmoid_in_tensor, sigmoid_out_tensor, gpu_resource));
        output_tensor_entries.push_back(
            {input_output_info.output_names[0], sigmoid_out_tensor.shrink()});
      } else {
        Tensor2<float> sigmoid_in_tensor =
            Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> sigmoid_out_tensor;
        blobs_buff->reserve(sigmoid_in_tensor.get_dimensions(), &sigmoid_out_tensor);
        layers.emplace_back(
            new SigmoidLayer<float>(sigmoid_in_tensor, sigmoid_out_tensor, gpu_resource));
        output_tensor_entries.push_back(
            {input_output_info.output_names[0], sigmoid_out_tensor.shrink()});
      }
      break;
    }
    case Layer_t::Slice: {
      if (use_mixed_precision) {
        Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        Tensors2<__half> out_tensors;
        layers.emplace_back(new SliceLayer<__half>(in_tensor, out_tensors, blobs_buff,
                                                   dense_layer.ranges, gpu_resource));
        for (size_t i = 0; i < out_tensors.size(); i++) {
          output_tensor_entries.push_back(
              {input_output_info.output_names[i], out_tensors[i].shrink()});
        }
      } else {
        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensors2<float> out_tensors;
        layers.emplace_back(new SliceLayer<float>(in_tensor, out_tensors, blobs_buff,
                                                  dense_layer.ranges, gpu_resource));
        for (size_t i = 0; i < out_tensors.size(); i++) {
          output_tensor_entries.push_back(
              {input_output_info.output_names[i], out_tensors[i].shrink()});
        }
      }
      break;
    }
    case Layer_t::WeightMultiply: {
      if (use_mixed_precision) {
        std::vector<Initializer_t> initializer_types{dense_layer.weight_init_type};
        Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        Tensor2<__half> out_tensor;
        layers.emplace_back(new WeightMultiplyLayer<__half>(
            weight_buff, weight_buff_half, wgrad_buff_half, blobs_buff, in_tensor, out_tensor,
            dense_layer.weight_dims, gpu_resource, initializer_types));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      } else {
        std::vector<Initializer_t> initializer_types{dense_layer.weight_init_type};
        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> out_tensor;
        layers.emplace_back(new WeightMultiplyLayer<float>(
            weight_buff, weight_buff, wgrad_buff, blobs_buff, in_tensor, out_tensor,
            dense_layer.weight_dims, gpu_resource, initializer_types));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      }
      break;
    }
    case Layer_t::FmOrder2: {
      if (use_mixed_precision) {
        size_t out_dim = dense_layer.out_dim;
        Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        Tensor2<__half> out_tensor;
        blobs_buff->reserve({in_tensor.get_dimensions()[0], out_dim}, &out_tensor);
        layers.emplace_back(new FmOrder2Layer<__half>(in_tensor, out_tensor, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      } else {
        size_t out_dim = dense_layer.out_dim;
        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> out_tensor;
        blobs_buff->reserve({in_tensor.get_dimensions()[0], out_dim}, &out_tensor);
        layers.emplace_back(new FmOrder2Layer<float>(in_tensor, out_tensor, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      }
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
        layers.emplace_back(new AddLayer<__half>(in_tensors, out_tensor, blobs_buff, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      } else {
        Tensors2<float> in_tensors;
        for (const auto& bag : input_output_info.inputs) {
          in_tensors.push_back(Tensor2<float>::stretch_from(bag));
        }
        Tensor2<float> out_tensor;
        blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
        layers.emplace_back(new AddLayer<float>(in_tensors, out_tensor, blobs_buff, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      }
      break;
    }
    case Layer_t::ReduceSum: {
      int axis = dense_layer.axis;
      if (use_mixed_precision) {
        Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        Tensor2<__half> out_tensor;
        layers.emplace_back(
            new ReduceSumLayer<__half>(in_tensor, out_tensor, blobs_buff, axis, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      } else {
        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> out_tensor;
        layers.emplace_back(
            new ReduceSumLayer<float>(in_tensor, out_tensor, blobs_buff, axis, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      }
      break;
    }
    case Layer_t::ReduceMean: {
      Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
      Tensor2<float> out_tensor;
      layers.emplace_back(new ReduceMeanLayer<float>(in_tensor, out_tensor, blobs_buff,
                                                     dense_layer.axis, gpu_resource));
      output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      break;
    }
    case Layer_t::Sub: {
      Tensors2<float> in_tensors;
      for (const auto& bag : input_output_info.inputs) {
        in_tensors.push_back(Tensor2<float>::stretch_from(bag));
      }
      Tensor2<float> out_tensor;
      blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
      layers.emplace_back(new SubLayer<float>(in_tensors, out_tensor, blobs_buff, gpu_resource));
      output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      break;
    }
    case Layer_t::Gather: {
      Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
      Tensor2<float> out_tensor;
      layers.emplace_back(new GatherLayer<float>(in_tensor, out_tensor, blobs_buff,
                                                 dense_layer.indices, gpu_resource));
      output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      break;
    }
    case Layer_t::GRU: {
      std::vector<Initializer_t> initializer_types{dense_layer.weight_init_type,
                                                   dense_layer.bias_init_type};
      Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
      Tensor2<float> gru_out_tensor;
      blobs_buff->reserve({in_tensor.get_dimensions()[0], dense_layer.num_output}, &gru_out_tensor);
      layers.emplace_back(new GRULayer<float>(weight_buff, wgrad_buff, in_tensor, gru_out_tensor,
                                              dense_layer.num_output, dense_layer.batchsize,
                                              dense_layer.SeqLength, dense_layer.vector_size,
                                              gpu_resource, initializer_types));
      output_tensor_entries.push_back({input_output_info.output_names[0], gru_out_tensor.shrink()});

      break;
    }
    case Layer_t::MatrixMultiply: {
      Tensors2<float> in_tensors;
      for (const auto& bag : input_output_info.inputs) {
        in_tensors.push_back(Tensor2<float>::stretch_from(bag));
      }
      Tensor2<float> out_tensor;
      layers.emplace_back(
          new MatrixMultiplyLayer<float>(in_tensors, out_tensor, blobs_buff, gpu_resource));
      output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      break;
    }
    case Layer_t::Softmax: {
      Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
      Tensor2<float> out_tensor;
      blobs_buff->reserve(in_tensor.get_dimensions(), &out_tensor);
      output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      layers.emplace_back(new SoftmaxLayer<float>(in_tensor, out_tensor, blobs_buff, gpu_resource));
      break;
    }
    case Layer_t::PReLU_Dice: {
      Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
      Tensor2<float> out_tensor;
      blobs_buff->reserve(in_tensor.get_dimensions(), &out_tensor);
      output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      layers.emplace_back(new PRelu_Dice_Layer<float>(
          in_tensor, out_tensor, blobs_buff, dense_layer.elu_alpha, dense_layer.eps, gpu_resource));
      break;
    }
    case Layer_t::Scale: {
      Tensor2<float> scale_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
      Tensor2<float> scale_out_tensor;
      layers.emplace_back(new ScaleLayer<float>(scale_in_tensor, scale_out_tensor, blobs_buff,
                                                dense_layer.axis, dense_layer.factor,
                                                gpu_resource));
      output_tensor_entries.push_back(
          {input_output_info.output_names[0], scale_out_tensor.shrink()});
      break;
    }
    case Layer_t::FusedReshapeConcat: {
      Tensors2<float> in_tensors;
      for (const auto& bag : input_output_info.inputs) {
        in_tensors.push_back(Tensor2<float>::stretch_from(bag));
      }
      Tensors2<float> out_tensors;
      layers.emplace_back(
          new FusedReshapeConcatLayer<float>(in_tensors, out_tensors, blobs_buff, gpu_resource));
      for (size_t i = 0; i < out_tensors.size(); i++) {
        output_tensor_entries.push_back(
            {input_output_info.output_names[i], out_tensors[i].shrink()});
      }
      break;
    }
    case Layer_t::FusedReshapeConcatGeneral: {
      Tensors2<float> in_tensors;
      for (const auto& bag : input_output_info.inputs) {
        in_tensors.push_back(Tensor2<float>::stretch_from(bag));
      }
      Tensor2<float> out_tensor;
      layers.emplace_back(new FusedReshapeConcatGeneralLayer<float>(in_tensors, out_tensor,
                                                                    blobs_buff, gpu_resource));
      output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      break;
    }
    case Layer_t::DotProduct: {
      if (use_mixed_precision) {
        Tensors2<__half> in_tensors;
        for (const auto& bag : input_output_info.inputs) {
          in_tensors.push_back(Tensor2<__half>::stretch_from(bag));
        }
        Tensor2<__half> out_tensor;
        blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
        layers.emplace_back(
            new DotProductLayer<__half>(in_tensors, out_tensor, blobs_buff, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      } else {
        Tensors2<float> in_tensors;
        for (const auto& bag : input_output_info.inputs) {
          in_tensors.push_back(Tensor2<float>::stretch_from(bag));
        }
        Tensor2<float> out_tensor;
        blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
        layers.emplace_back(
            new DotProductLayer<float>(in_tensors, out_tensor, blobs_buff, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      }
      break;
    }
    case Layer_t::ElementwiseMultiply: {
      if (use_mixed_precision) {
        Tensors2<__half> in_tensors;
        for (const auto& bag : input_output_info.inputs) {
          in_tensors.push_back(Tensor2<__half>::stretch_from(bag));
        }
        Tensor2<__half> out_tensor;
        blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
        layers.emplace_back(
            new ElementwiseMultiplyLayer<__half>(in_tensors, out_tensor, blobs_buff, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      } else {
        Tensors2<float> in_tensors;
        for (const auto& bag : input_output_info.inputs) {
          in_tensors.push_back(Tensor2<float>::stretch_from(bag));
        }
        Tensor2<float> out_tensor;
        blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
        layers.emplace_back(
            new ElementwiseMultiplyLayer<float>(in_tensors, out_tensor, blobs_buff, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      }
      break;
    }
    default: {
      assert(!"Error: no such layer && should never get here!");
    }
  }  // end of switch
  if (!(layer_type == Layer_t::CrossEntropyLoss || layer_type == Layer_t::BinaryCrossEntropyLoss ||
        layer_type == Layer_t::MultiCrossEntropyLoss)) {
    for (auto& output_tensor_entry : output_tensor_entries) {
      tensor_entries.push_back(output_tensor_entry);
    }
    if (dlrm_bottom_mlp) {
      if (bottom_layers) {
        bottom_layers->emplace_back(layers.back().get());
      }
    } else {
      if (top_layers) {
        top_layers->emplace_back(layers.back().get());
      }
    }
  } else if (raw_metrics) {
    (*raw_metrics)[metrics::RawType::Loss] = loss_tensor.shrink();
    (*raw_metrics)[metrics::RawType::Pred] = input_output_info.inputs[0];
    (*raw_metrics)[metrics::RawType::Label] = input_output_info.inputs[1];
  }
}

void Model::add_dense_layer(DenseLayer& dense_layer) {
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    // add dense layer for train
    add_dense_layer_internal(
        dense_layer, train_tensor_entries_list_[i], blobs_buff_list_[i], train_weight_buff_list_[i],
        train_weight_buff_half_list_[i], wgrad_buff_list_[i], wgrad_buff_half_list_[i],
        networks_[i]->train_loss_tensor_, networks_[i]->train_layers_, networks_[i]->train_loss_,
        networks_[i]->enable_cuda_graph_, solver_.async_mlp_wgrad, nullptr,
        resource_manager_->get_global_gpu_count(), resource_manager_->get_local_gpu(i),
        solver_.use_mixed_precision, solver_.enable_tf32_compute, solver_.scaler,
        solver_.use_algorithm_search, &networks_[i]->top_layers_, &networks_[i]->bottom_layers_,
        dlrm_bottom_mlp_);
    // add dense layer for evaluation
    add_dense_layer_internal(dense_layer, evaluate_tensor_entries_list_[i], blobs_buff_list_[i],
                             evaluate_weight_buff_list_[i], evaluate_weight_buff_half_list_[i],
                             wgrad_buff_placeholder_list_[i], wgrad_buff_half_placeholder_list_[i],
                             networks_[i]->evaluate_loss_tensor_, networks_[i]->evaluate_layers_,
                             networks_[i]->evaluate_loss_, networks_[i]->enable_cuda_graph_,
                             solver_.async_mlp_wgrad, &(networks_[i]->raw_metrics_),
                             resource_manager_->get_global_gpu_count(),
                             resource_manager_->get_local_gpu(i), solver_.use_mixed_precision,
                             solver_.enable_tf32_compute, solver_.scaler,
                             solver_.use_algorithm_search, nullptr, nullptr, false);
  }
}

void calculate_tensor_dimensions(std::map<std::string, std::vector<int>>& tensor_shape_info_raw,
                                 DenseLayer& dense_layer) {
  switch (dense_layer.layer_type) {
    case Layer_t::BatchNorm: {
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], tensor_shape_info_raw[dense_layer.bottom_names[0]]));
      break;
    }
    case Layer_t::BinaryCrossEntropyLoss: {
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], tensor_shape_info_raw[dense_layer.bottom_names[0]]));
      break;
    }
    case Layer_t::Concat: {
      int batch_size = tensor_shape_info_raw[dense_layer.bottom_names[0]][0];
      int out_dim{0};
      for (auto& bottom_name : dense_layer.bottom_names) {
        out_dim += tensor_shape_info_raw[bottom_name][1];
      }
      tensor_shape_info_raw.insert(
          std::make_pair(dense_layer.top_names[0], std::vector<int>{batch_size, out_dim}));
      break;
    }
    case Layer_t::CrossEntropyLoss: {
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], tensor_shape_info_raw[dense_layer.bottom_names[0]]));
      break;
    }
    case Layer_t::Dropout: {
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], tensor_shape_info_raw[dense_layer.bottom_names[0]]));
      break;
    }
    case Layer_t::ELU: {
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], tensor_shape_info_raw[dense_layer.bottom_names[0]]));
      break;
    }
    case Layer_t::FusedInnerProduct: {
      int batch_size = tensor_shape_info_raw[dense_layer.bottom_names[0]][0];
      int num_output = dense_layer.num_output;
      auto pos_type = dense_layer.pos_type;
      if (pos_type == FcPosition_t::None || pos_type == FcPosition_t::Isolated ||
          pos_type == FcPosition_t::Tail) {
        tensor_shape_info_raw.insert(
            std::make_pair(dense_layer.top_names[0], std::vector<int>{batch_size, num_output}));
      } else if (pos_type == FcPosition_t::Head || pos_type == FcPosition_t::Body) {
        tensor_shape_info_raw.insert(
            std::make_pair(dense_layer.top_names[0], std::vector<int>{batch_size, num_output}));
        tensor_shape_info_raw.insert(
            std::make_pair(dense_layer.top_names[1], std::vector<int>{batch_size, num_output}));
        tensor_shape_info_raw.insert(
            std::make_pair(dense_layer.top_names[2], std::vector<int>{batch_size, num_output}));
        tensor_shape_info_raw.insert(
            std::make_pair(dense_layer.top_names[3], std::vector<int>{1, num_output}));
      }
      break;
    }
    case Layer_t::Cast: {
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], tensor_shape_info_raw[dense_layer.bottom_names[0]]));
      break;
    }
    case Layer_t::InnerProduct: {
      int batch_size = tensor_shape_info_raw[dense_layer.bottom_names[0]][0];
      int num_output = dense_layer.num_output;
      tensor_shape_info_raw.insert(
          std::make_pair(dense_layer.top_names[0], std::vector<int>{batch_size, num_output}));
      break;
    }
    case Layer_t::Interaction: {
      int batch_size = tensor_shape_info_raw[dense_layer.bottom_names[1]][0];
      int slot_num = tensor_shape_info_raw[dense_layer.bottom_names[1]][1];
      int vec_size = tensor_shape_info_raw[dense_layer.bottom_names[1]][2];
      int out_dim = vec_size + (slot_num + 1) * (slot_num + 2) / 2 - (slot_num + 1) + 1;
      for (const auto& top_name : dense_layer.top_names) {
        tensor_shape_info_raw.insert(
            std::make_pair(top_name, std::vector<int>{batch_size, out_dim}));
      }
      break;
    }
    case Layer_t::MultiCross: {
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], tensor_shape_info_raw[dense_layer.bottom_names[0]]));
      break;
    }
    case Layer_t::MultiCrossEntropyLoss: {
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], tensor_shape_info_raw[dense_layer.bottom_names[0]]));
      break;
    }
    case Layer_t::ReLU: {
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], tensor_shape_info_raw[dense_layer.bottom_names[0]]));
      break;
    }
    case Layer_t::Reshape: {
      int batch_size = tensor_shape_info_raw[dense_layer.bottom_names[0]][0];
      int reshape_time_step = dense_layer.leading_dim;
      int leading_dim = dense_layer.leading_dim;
      if (reshape_time_step == 0) {
        tensor_shape_info_raw.insert(
            std::make_pair(dense_layer.top_names[0], std::vector<int>{batch_size, leading_dim}));
      } else {
        tensor_shape_info_raw.insert(
            std::make_pair(dense_layer.top_names[0],
                           std::vector<int>{batch_size, reshape_time_step, leading_dim}));
      }
      break;
    }
    case Layer_t::Sigmoid: {
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], tensor_shape_info_raw[dense_layer.bottom_names[0]]));
      break;
    }
    case Layer_t::Slice: {
      int batch_size = tensor_shape_info_raw[dense_layer.bottom_names[0]][0];
      for (unsigned int i = 0; i < dense_layer.top_names.size(); i++) {
        tensor_shape_info_raw.insert(std::make_pair(
            dense_layer.top_names[i],
            std::vector<int>{batch_size,
                             dense_layer.ranges[i].second - dense_layer.ranges[i].first}));
      }
      break;
    }
    case Layer_t::WeightMultiply: {
      int batch_size = tensor_shape_info_raw[dense_layer.bottom_names[0]][0];
      int out_dim = dense_layer.weight_dims[0] * dense_layer.weight_dims[1];
      tensor_shape_info_raw.insert(
          std::make_pair(dense_layer.top_names[0], std::vector<int>{batch_size, out_dim}));
      break;
    }
    case Layer_t::FmOrder2: {
      int batch_size = tensor_shape_info_raw[dense_layer.bottom_names[0]][0];
      int out_dim = dense_layer.out_dim;
      tensor_shape_info_raw.insert(
          std::make_pair(dense_layer.top_names[0], std::vector<int>{batch_size, out_dim}));
      break;
    }
    case Layer_t::Add: {
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], tensor_shape_info_raw[dense_layer.bottom_names[0]]));
      break;
    }
    case Layer_t::ReduceSum: {
      std::vector<int> out_dims = tensor_shape_info_raw[dense_layer.bottom_names[0]];
      out_dims[dense_layer.axis] = 1;
      tensor_shape_info_raw.insert(std::make_pair(dense_layer.top_names[0], out_dims));
      break;
    }
    case Layer_t::ReduceMean: {
      std::vector<int> out_dims = tensor_shape_info_raw[dense_layer.bottom_names[0]];
      out_dims[dense_layer.axis] = 1;
      tensor_shape_info_raw.insert(std::make_pair(dense_layer.top_names[0], out_dims));
      break;
    }
    case Layer_t::Sub: {
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], tensor_shape_info_raw[dense_layer.bottom_names[0]]));
      break;
    }
    case Layer_t::Gather: {
      int out_batch_size = dense_layer.indices.size();
      int out_dim = tensor_shape_info_raw[dense_layer.bottom_names[0]][1];
      tensor_shape_info_raw.insert(
          std::make_pair(dense_layer.top_names[0], std::vector<int>{out_batch_size, out_dim}));
      break;
    }
    case Layer_t::GRU: {
      int batch_size = tensor_shape_info_raw[dense_layer.bottom_names[0]][0];
      int num_output = dense_layer.num_output;
      tensor_shape_info_raw.insert(
          std::make_pair(dense_layer.top_names[0], std::vector<int>{batch_size, num_output}));
      break;
    }
    case Layer_t::MatrixMultiply: {
      auto& dim1 = tensor_shape_info_raw[dense_layer.bottom_names[0]];
      auto& dim2 = tensor_shape_info_raw[dense_layer.bottom_names[1]];
      if (dim1.size() == 3) {
        tensor_shape_info_raw.insert(
            std::make_pair(dense_layer.top_names[0], std::vector<int>{dim1[0], dim1[1], dim2[2]}));
      } else {
        tensor_shape_info_raw.insert(
            std::make_pair(dense_layer.top_names[0], std::vector<int>{dim1[0], dim2[1]}));
      }
      break;
    }
    case Layer_t::Softmax: {
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], tensor_shape_info_raw[dense_layer.bottom_names[0]]));
      break;
    }
    case Layer_t::PReLU_Dice: {
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], tensor_shape_info_raw[dense_layer.bottom_names[0]]));
      break;
    }
    case Layer_t::Scale: {
      std::vector<int> out_dims = tensor_shape_info_raw[dense_layer.bottom_names[0]];
      if (dense_layer.axis == 0) {
        out_dims[1] *= dense_layer.factor;
      } else {
        out_dims[0] *= dense_layer.factor;
      }
      tensor_shape_info_raw.insert(std::make_pair(dense_layer.top_names[0], out_dims));
      break;
    }
    case Layer_t::FusedReshapeConcat: {
      int batch_size = tensor_shape_info_raw[dense_layer.bottom_names[0]][0];
      int slot_num = tensor_shape_info_raw[dense_layer.bottom_names[0]][1];
      int out_width{0};
      for (auto& bottom_name : dense_layer.bottom_names) {
        out_width += tensor_shape_info_raw[bottom_name][2];
      }
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], std::vector<int>{batch_size * (slot_num - 1), out_width}));
      tensor_shape_info_raw.insert(
          std::make_pair(dense_layer.top_names[1], std::vector<int>{batch_size, out_width}));
      break;
    }
    case Layer_t::FusedReshapeConcatGeneral: {
      int batch_size = tensor_shape_info_raw[dense_layer.bottom_names[0]][0];
      int slot_num = tensor_shape_info_raw[dense_layer.bottom_names[0]][1];
      int out_width{0};
      for (auto& bottom_name : dense_layer.bottom_names) {
        out_width += tensor_shape_info_raw[bottom_name][2];
      }
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], std::vector<int>{batch_size * slot_num, out_width}));
      break;
    }
    case Layer_t::DotProduct: {
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], tensor_shape_info_raw[dense_layer.bottom_names[0]]));
      break;
    }
    case Layer_t::ElementwiseMultiply: {
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], tensor_shape_info_raw[dense_layer.bottom_names[0]]));
      break;
    }
    default: {
      assert(!"Error: no such layer && should never get here!");
    }
  }  // end of switch
}

}  // namespace HugeCTR
