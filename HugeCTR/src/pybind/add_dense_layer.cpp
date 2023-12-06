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
#include <core23_network.hpp>
#include <layer.hpp>
#include <layers/add_layer.hpp>
#include <layers/batch_norm_layer.hpp>
#include <layers/cast_layer.hpp>
#include <layers/concat_3d_layer.hpp>
#include <layers/concat_layer.hpp>
#include <layers/dropout_layer.hpp>
#include <layers/elementwise_multiply_layer.hpp>
#include <layers/elu_layer.hpp>
#include <layers/fm_order2_layer.hpp>
#include <layers/fully_connected_layer.hpp>
#include <layers/fully_connected_layer_half.hpp>
#include <layers/fused_reshape_concat_general_layer.hpp>
#include <layers/fused_reshape_concat_layer.hpp>
#include <layers/gather_layer.hpp>
#include <layers/gru_layer.hpp>
#include <layers/interaction_layer.hpp>
#include <layers/layer_norm_layer.hpp>
#include <layers/masked_softmax_layer.hpp>
#include <layers/matrix_multiply_layer.hpp>
#include <layers/mlp_layer.hpp>
#include <layers/multi_cross_layer.hpp>
#include <layers/multi_head_attention_layer.hpp>
#include <layers/prelu_dice_layer.hpp>
#include <layers/reduce_mean_layer.hpp>
#include <layers/reduce_sum_layer.hpp>
#include <layers/relu_layer.hpp>
#include <layers/reshape_layer.hpp>
#include <layers/reshape_layer_v2.hpp>
#include <layers/scale_layer.hpp>
#include <layers/sequence_mask_layer.hpp>
#include <layers/sigmoid_layer.hpp>
#include <layers/slice_layer.hpp>
#include <layers/softmax_layer.hpp>
#include <layers/sub_layer.hpp>
#include <layers/weight_multiply_layer.hpp>
#include <network_buffer_channels.hpp>
#include <network_helpers.hpp>
#include <pybind/add_dense_layer_helpers.hpp>
#include <pybind/model.hpp>
#include <regularizers/l1_regularizer.hpp>
#include <regularizers/l2_regularizer.hpp>
#include <regularizers/no_regularizer.hpp>
#include <unordered_map>
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

  std::vector<std::string> label_names;
  std::vector<int> label_dims;
  for (auto& label : input_param.labels_) {
    label_names.emplace_back(label.first);
    label_dims.emplace_back(label.second);
  }

  if (input_param.labels_.size() > 1) {
    input_label_config["top"] = label_names;
    input_label_config["label_dim"] = label_dims;
  } else {
    input_label_config["top"] = label_names[0];
    input_label_config["label_dim"] = label_dims[0];
  }
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
      HCTR_OWN_THROW(Error_t::WrongInput, "combiner error");
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
      case Optimizer_t::Ftrl:
        optimizer_config["type"] = "Ftrl";
        optimizer_hparam_config["beta"] = embedding_opt_params_list[i]->hyperparams.ftrl.beta;
        optimizer_hparam_config["lambda1"] = embedding_opt_params_list[i]->hyperparams.ftrl.lambda1;
        optimizer_hparam_config["lambda2"] = embedding_opt_params_list[i]->hyperparams.ftrl.lambda2;
        optimizer_config["ftrl_hparam"] = optimizer_hparam_config;
        break;

      case Optimizer_t::Adam:
        optimizer_config["type"] = "Adam";
        optimizer_hparam_config["beta1"] = embedding_opt_params_list[i]->hyperparams.adam.beta1;
        optimizer_hparam_config["beta2"] = embedding_opt_params_list[i]->hyperparams.adam.beta2;
        optimizer_hparam_config["epsilon"] = embedding_opt_params_list[i]->hyperparams.adam.epsilon;
        optimizer_config["adam_hparam"] = optimizer_hparam_config;
        break;

      case Optimizer_t::AdaGrad:
        optimizer_config["type"] = "AdaGrad";
        optimizer_hparam_config["initial_accu_value"] =
            embedding_opt_params_list[i]->hyperparams.adagrad.initial_accu_value;
        optimizer_hparam_config["epsilon"] =
            embedding_opt_params_list[i]->hyperparams.adagrad.epsilon;
        optimizer_config["adagrad_hparam"] = optimizer_hparam_config;
        break;

      case Optimizer_t::MomentumSGD:
        optimizer_config["type"] = "MomentumSGD";
        optimizer_hparam_config["momentum_factor"] =
            embedding_opt_params_list[i]->hyperparams.momentum.factor;
        optimizer_config["momentum_sgd_hparam"] = optimizer_hparam_config;
        break;

      case Optimizer_t::Nesterov:
        optimizer_config["type"] = "Nesterov";
        optimizer_hparam_config["momentum_factor"] =
            embedding_opt_params_list[i]->hyperparams.nesterov.mu;
        optimizer_config["nesterov_hparam"] = optimizer_hparam_config;
        break;

      case Optimizer_t::SGD:
        optimizer_config["type"] = "SGD";
        optimizer_hparam_config["atomic_update"] =
            embedding_opt_params_list[i]->hyperparams.sgd.atomic_update;
        optimizer_config["sgd_hparam"] = optimizer_hparam_config;
        break;

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

    // Don't insert slice layer, it will be auto-added with Input
    if (layer_config["bottom"] == "combined_multi_label") {
      continue;
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
      case Layer_t::LayerNorm: {
        nlohmann::json ln_param_config;
        ln_param_config["eps"] = dense_layer_params[i].eps;
        if (dense_layer_params[i].gamma_init_type != Initializer_t::Default) {
          ln_param_config["gamma_init"] =
              INITIALIZER_TYPE_TO_STRING[dense_layer_params[i].gamma_init_type];
        }
        if (dense_layer_params[i].beta_init_type != Initializer_t::Default) {
          ln_param_config["beta_init"] =
              INITIALIZER_TYPE_TO_STRING[dense_layer_params[i].beta_init_type];
        }
        layer_config["ln_param"] = ln_param_config;
        break;
      }
      case Layer_t::Dropout: {
        layer_config["rate"] = dense_layer_params[i].dropout_rate;
        break;
      }
      case Layer_t::SequenceMask: {
        layer_config["max_sequence_len_from"] = dense_layer_params[i].max_sequence_len_from;
        layer_config["max_sequence_len_to"] = dense_layer_params[i].max_sequence_len_to;
        break;
      }
      case Layer_t::ELU: {
        nlohmann::json elu_param_config;
        elu_param_config["alpha"] = dense_layer_params[i].elu_alpha;
        layer_config["elu_param"] = elu_param_config;
        break;
      }
      case Layer_t::MultiHeadAttention: {
        layer_config["num_attention_heads"] = dense_layer_params[i].num_attention_heads;
        layer_config["transpose_b"] = dense_layer_params[i].transpose_b;
        break;
      }
      case Layer_t::MLP: {
        nlohmann::json mlp_param_config;
        mlp_param_config["num_output"] = dense_layer_params[i].num_output;
        mlp_param_config["num_outputs"] = dense_layer_params[i].num_outputs;
        if (dense_layer_params[i].biases.empty()) {
          mlp_param_config["use_bias"] = dense_layer_params[i].use_bias;
        } else {
          mlp_param_config["biases"] = dense_layer_params[i].biases;
        }
        if (dense_layer_params[i].acts.empty()) {
          mlp_param_config["activation"] = FC_ACTIVATION_TO_STRING[dense_layer_params[i].act_type];
        } else {
          std::vector<std::string> s_acts;
          for (auto act : dense_layer_params[i].acts) {
            s_acts.push_back(FC_ACTIVATION_TO_STRING[act]);
          }
          mlp_param_config["activations"] = s_acts;
        }
        if (dense_layer_params[i].weight_init_type != Initializer_t::Default) {
          mlp_param_config["weight_init"] =
              INITIALIZER_TYPE_TO_STRING[dense_layer_params[i].weight_init_type];
        }
        if (dense_layer_params[i].bias_init_type != Initializer_t::Default) {
          mlp_param_config["bias_init"] =
              INITIALIZER_TYPE_TO_STRING[dense_layer_params[i].bias_init_type];
        }
        layer_config["mlp_param"] = mlp_param_config;
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
      case Layer_t::Concat: {
        layer_config["axis"] = dense_layer_params[i].axis;
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
      case Layer_t::Softmax: {
        layer_config["factor"] = dense_layer_params[i].factor;
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
  Layer_t layer_type = Layer_t::Unknown;
  auto layer_type_name = get_value_from_json<std::string>(j_dense_layer, "type");
  if (!find_item_in_map(layer_type, layer_type_name, LAYER_TYPE_MAP) &&
      !find_item_in_map(layer_type, layer_type_name, LAYER_TYPE_MAP_MP)) {
    HCTR_OWN_THROW(Error_t::WrongInput, "No such layer: " + layer_type_name);
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
          HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + gamma_init_name);
        }
      }
      if (has_key_(j_bn_hparam, "beta_init")) {
        const auto beta_init_name = get_value_from_json<std::string>(j_bn_hparam, "beta_init");
        Initializer_t beta_init_type;
        if (find_item_in_map(beta_init_type, beta_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.beta_init_type = beta_init_type;
        } else {
          HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + beta_init_name);
        }
      }
      break;
    }
    case Layer_t::LayerNorm: {
      auto j_ln_hparam = get_json(j_dense_layer, "ln_param");
      auto eps = get_value_from_json<float>(j_ln_hparam, "eps");
      dense_layer.eps = eps;
      if (has_key_(j_ln_hparam, "gamma_init")) {
        const auto gamma_init_name = get_value_from_json<std::string>(j_ln_hparam, "gamma_init");
        Initializer_t gamma_init_type;
        if (find_item_in_map(gamma_init_type, gamma_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.gamma_init_type = gamma_init_type;
        } else {
          HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + gamma_init_name);
        }
      }
      if (has_key_(j_ln_hparam, "beta_init")) {
        const auto beta_init_name = get_value_from_json<std::string>(j_ln_hparam, "beta_init");
        Initializer_t beta_init_type;
        if (find_item_in_map(beta_init_type, beta_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.beta_init_type = beta_init_type;
        } else {
          HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + beta_init_name);
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
    case Layer_t::SequenceMask: {
      auto max_sequence_len_from = get_json(j_dense_layer, "max_sequence_len_from");
      auto max_sequence_len_to = get_json(j_dense_layer, "max_sequence_len_to");
      dense_layer.max_sequence_len_from = max_sequence_len_from;
      dense_layer.max_sequence_len_to = max_sequence_len_to;
      break;
    }
    case Layer_t::MLP: {
      auto j_mlp_param = get_json(j_dense_layer, "mlp_param");
      if (has_key_(j_mlp_param, "weight_init")) {
        const auto weight_init_name = get_value_from_json<std::string>(j_mlp_param, "weight_init");
        Initializer_t weight_init_type;
        if (find_item_in_map(weight_init_type, weight_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.weight_init_type = weight_init_type;
        } else {
          HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + weight_init_name);
        }
      }
      if (has_key_(j_mlp_param, "bias_init")) {
        const auto bias_init_name = get_value_from_json<std::string>(j_mlp_param, "bias_init");
        Initializer_t bias_init_type;
        if (find_item_in_map(bias_init_type, bias_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.bias_init_type = bias_init_type;
        } else {
          HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + bias_init_name);
        }
      }
      if (has_key_(j_mlp_param, "num_outputs")) {
        std::vector<size_t> num_outputs;
        auto nums = get_json(j_mlp_param, "num_outputs");
        assert(nums.is_array());
        for (auto num : nums) {
          num_outputs.emplace_back(num.get<size_t>());
        }
        dense_layer.num_outputs = num_outputs;
      }
      if (has_key_(j_mlp_param, "use_bias")) {
        auto use_bias = get_value_from_json<bool>(j_mlp_param, "use_bias");
        dense_layer.use_bias = use_bias;
      }
      if (has_key_(j_mlp_param, "biases")) {
        std::vector<bool> biases;
        auto j_biases = get_json(j_mlp_param, "biases");
        assert(j_biases.is_array());
        for (auto bias : j_biases) {
          biases.emplace_back(bias.get<bool>());
        }
        dense_layer.biases = biases;
      }
      if (has_key_(j_mlp_param, "activation")) {
        const auto act_name = get_value_from_json<std::string>(j_mlp_param, "activation");
        Activation_t act_type;
        if (find_item_in_map(act_type, act_name, ACTIVATION_TYPE_MAP)) {
          dense_layer.act_type = act_type;
        } else {
          HCTR_OWN_THROW(Error_t::WrongInput, "No such activation: " + act_name);
        }
      }
      if (has_key_(j_mlp_param, "activations")) {
        std::vector<Activation_t> acts;
        auto j_acts = get_json(j_mlp_param, "activations");
        assert(j_acts.is_array());
        for (const auto& j_act : j_acts) {
          auto act_name = j_act.get<std::string>();
          Activation_t act_type;
          if (find_item_in_map(act_type, act_name, ACTIVATION_TYPE_MAP)) {
            acts.emplace_back(act_type);
          } else {
            HCTR_OWN_THROW(Error_t::WrongInput, "No such activation: " + act_name);
          }
        }
        dense_layer.acts = acts;
      }
      // establish out tensor
      auto output = get_value_from_json<size_t>(j_mlp_param, "num_output");
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
          HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + weight_init_name);
        }
      }
      if (has_key_(j_fc_param, "bias_init")) {
        const auto bias_init_name = get_value_from_json<std::string>(j_fc_param, "bias_init");
        Initializer_t bias_init_type;
        if (find_item_in_map(bias_init_type, bias_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.bias_init_type = bias_init_type;
        } else {
          HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + bias_init_name);
        }
      }
      // establish out tensor
      auto output = get_value_from_json<size_t>(j_fc_param, "num_output");
      dense_layer.num_output = output;
      break;
    }
    case Layer_t::Concat: {
      auto axis_it = j_dense_layer.find("axis");
      if (axis_it != j_dense_layer.end()) {
        dense_layer.axis = axis_it->get<int>();
      } else {
        dense_layer.axis = 1;
      }
      break;
    }
    case Layer_t::MultiHeadAttention: {
      auto num_attention_heads_it = j_dense_layer.find("num_attention_heads");
      auto transpose_b_it = j_dense_layer.find("transpose_b");
      if (num_attention_heads_it != j_dense_layer.end()) {
        dense_layer.num_attention_heads = num_attention_heads_it->get<int>();
      } else {
        dense_layer.num_attention_heads = 1;
      }
      if (transpose_b_it != j_dense_layer.end()) {
        dense_layer.transpose_b = transpose_b_it->get<bool>();
      } else {
        dense_layer.transpose_b = false;
      }
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
          HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + weight_init_name);
        }
      }
      if (has_key_(j_mc_param, "bias_init")) {
        const auto bias_init_name = get_value_from_json<std::string>(j_mc_param, "bias_init");
        Initializer_t bias_init_type;
        if (find_item_in_map(bias_init_type, bias_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.bias_init_type = bias_init_type;
        } else {
          HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + bias_init_name);
        }
      }
      if (has_key_(j_mc_param, "projection_dim")) {
        const auto projection_dim = get_value_from_json<int>(j_mc_param, "projection_dim");
        dense_layer.projection_dim = projection_dim;
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
          if (slot_id < 0) {
            HCTR_OWN_THROW(Error_t::WrongInput, "slot_id < 0");
          }
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
          HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + weight_init_name);
        }
      }
      if (has_key_(j_gru_param, "bias_init")) {
        const auto bias_init_name = get_value_from_json<std::string>(j_gru_param, "bias_init");
        Initializer_t bias_init_type;
        if (find_item_in_map(bias_init_type, bias_init_name, INITIALIZER_TYPE_MAP)) {
          dense_layer.bias_init_type = bias_init_type;
        } else {
          HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + bias_init_name);
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
          HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + weight_init_name);
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
    case Layer_t::Softmax: {
      auto factor_it = j_dense_layer.find("factor");
      if (factor_it != j_dense_layer.end()) {
        dense_layer.factor = factor_it->get<float>();
      } else {
        dense_layer.axis = 1.0f;
      }
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
          HCTR_OWN_THROW(Error_t::WrongInput, "No such regularizer: " + reg_name);
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
          HCTR_OWN_THROW(Error_t::WrongInput, "No such regularizer: " + reg_name);
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
          HCTR_OWN_THROW(Error_t::WrongInput, "No such regularizer: " + reg_name);
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

void Model::add_dense_layers(std::vector<DenseLayer>& dense_layers) {
  auto add_dense_layers_op = [&dense_layers, this](bool is_train) {
    size_t num_local_gpus = resource_manager_->get_local_gpu_count();
    std::vector<std::vector<std::unique_ptr<Layer>>> layers(num_local_gpus);
    std::vector<std::map<std::string, std::unique_ptr<ILoss>>> losses(num_local_gpus);
    std::vector<metrics::Core23MultiLossMetricMap> raw_metrics(num_local_gpus);
    std::vector<std::vector<Layer*>> top_layers(num_local_gpus);
    std::vector<std::vector<Layer*>> bottom_layers(num_local_gpus);
    for (auto& dense_layer : dense_layers) {
      if (is_train) {
        pre_add_dense_layer(dense_layer);
      }
      for (size_t i = 0; i < num_local_gpus; i++) {
        add_dense_layer_impl(
            dense_layer,
            is_train ? train_tensor_entities_list_[i] : evaluate_tensor_entities_list_[i],
            layers[i], losses[i], is_train ? nullptr : &raw_metrics[i],
            resource_manager_->get_global_gpu_count(), resource_manager_->get_local_gpu(i),
            solver_.use_mixed_precision, solver_.enable_tf32_compute, solver_.scaler,
            solver_.use_algorithm_search, is_train ? &top_layers[i] : nullptr,
            is_train ? &bottom_layers[i] : nullptr, is_train ? embedding_dependent_ : false,
            solver_);
      }
    }
    for (size_t i = 0; i < num_local_gpus; i++) {
      if (is_train) {
        networks_[i]->set_train_layers(std::move(layers[i]));
        networks_[i]->set_train_losses(std::move(losses[i]), label_weights_);
        networks_[i]->set_top_and_bottom_layers(std::move(top_layers[i]),
                                                std::move(bottom_layers[i]));
      } else {
        networks_[i]->set_evaluate_layers(std::move(layers[i]));
        networks_[i]->set_evaluate_losses(std::move(losses[i]), label_weights_);
        networks_[i]->set_raw_metrics(std::move(raw_metrics[i]));
      }
    }
  };

  add_dense_layers_op(true);

  std::unordered_map<NetworkBufferChannelType, std::string> original_channel;
  const std::unordered_map<NetworkBufferChannelType, std::string> new_channel = {
      {NetworkBufferChannelType::Blobs, "EVAL_BLOBS"},
      {NetworkBufferChannelType::WeightHalf, "EVAL_WEIGHT_HALF"},
      {NetworkBufferChannelType::Weight, "EVAL_WEIGHT"},
      {NetworkBufferChannelType::Wgrad, "EVAL_WGRAD"},
      {NetworkBufferChannelType::WgradHalf, "EVAL_WGRAD_HALF"},
  };

  //! Embeddings and Train layers should use default channels;
  //! set new buffer channel for eval layers
  for (auto it = new_channel.begin(); it != new_channel.end(); it++) {
    auto original = SetNetworkBufferChannel(it->first, it->second);
    original_channel.emplace(std::make_pair(it->first, original));
  }
  add_dense_layers_op(false);

  //! Restore the channel
  for (auto it = original_channel.begin(); it != original_channel.end(); it++) {
    SetNetworkBufferChannel(it->first, it->second);
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
    case Layer_t::LayerNorm: {
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], tensor_shape_info_raw[dense_layer.bottom_names[0]]));
      break;
    }
    case Layer_t::SequenceMask: {
      int batch_size = tensor_shape_info_raw[dense_layer.bottom_names[0]][0];
      int max_sequence_len_from = dense_layer.max_sequence_len_from;
      int max_sequence_len_to = dense_layer.max_sequence_len_to;
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0],
          std::vector<int>{batch_size, 1, max_sequence_len_from, max_sequence_len_to}));
      break;
    }
    case Layer_t::BinaryCrossEntropyLoss: {
      tensor_shape_info_raw.insert(std::make_pair(
          dense_layer.top_names[0], tensor_shape_info_raw[dense_layer.bottom_names[0]]));
      break;
    }
    case Layer_t::Concat: {
      int batch_size = tensor_shape_info_raw[dense_layer.bottom_names[0]][0];
      if (tensor_shape_info_raw[dense_layer.bottom_names[0]].size() == 2) {
        int out_dim{0};
        for (auto& bottom_name : dense_layer.bottom_names) {
          out_dim += tensor_shape_info_raw[bottom_name][1];
        }
        tensor_shape_info_raw.insert(
            std::make_pair(dense_layer.top_names[0], std::vector<int>{batch_size, out_dim}));
      }
      if (tensor_shape_info_raw[dense_layer.bottom_names[0]].size() == 3) {
        int slot_num{0};
        int out_width{0};
        if (dense_layer.axis == 1) {
          for (auto& bottom_name : dense_layer.bottom_names) {
            slot_num += tensor_shape_info_raw[bottom_name][1];
          }
          out_width = tensor_shape_info_raw[dense_layer.bottom_names[0]][2];
        }
        if (dense_layer.axis == 2) {
          for (auto& bottom_name : dense_layer.bottom_names) {
            out_width += tensor_shape_info_raw[bottom_name][2];
          }
          slot_num = tensor_shape_info_raw[dense_layer.bottom_names[0]][1];
        }
        tensor_shape_info_raw.insert(std::make_pair(
            dense_layer.top_names[0], std::vector<int>{batch_size, slot_num, out_width}));
      }
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
    case Layer_t::MLP: {
      int batch_size = tensor_shape_info_raw[dense_layer.bottom_names[0]][0];
      int num_output = dense_layer.num_output;
      if (dense_layer.top_names.size() == 1) {
        tensor_shape_info_raw.insert(
            std::make_pair(dense_layer.top_names[0], std::vector<int>{batch_size, num_output}));
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
      auto& dim1 = tensor_shape_info_raw[dense_layer.bottom_names[0]];
      int num_output = dense_layer.num_output;
      if (dim1.size() == 3) {
        tensor_shape_info_raw.insert(std::make_pair(
            dense_layer.top_names[0],
            std::vector<int>{batch_size, tensor_shape_info_raw[dense_layer.bottom_names[0]][1],
                             num_output}));
      } else if (dim1.size() == 2) {
        tensor_shape_info_raw.insert(
            std::make_pair(dense_layer.top_names[0], std::vector<int>{batch_size, num_output}));
      } else {
        HCTR_OWN_THROW(Error_t::WrongInput, "InnerProductLayer needs 2D or 3D input tensor");
      }
      break;
    }
    case Layer_t::MultiHeadAttention: {
      // [batchsize, ]
      auto& dim1 = tensor_shape_info_raw[dense_layer.bottom_names[0]];
      // auto& dim2 = tensor_shape_info_raw[dense_layer.bottom_names[1]];
      tensor_shape_info_raw.insert(std::make_pair(dense_layer.top_names[0], dim1));

      // if (dim1.size() == 4) {
      //   if (dense_layer.transpose_b) {
      //     tensor_shape_info_raw.insert(std::make_pair(
      //         dense_layer.top_names[0], std::vector<int>{dim1[0], dim1[1], dim1[2], dim2[2]}));
      //   } else {
      //     tensor_shape_info_raw.insert(std::make_pair(
      //         dense_layer.top_names[0], std::vector<int>{dim1[0], dim1[2], dim2[3] * dim1[1]}));
      //   }
      // } else if (dim1.size() == 3) {
      //   tensor_shape_info_raw.insert(std::make_pair(
      //       dense_layer.top_names[0],
      //       std::vector<int>{dim1[0], dense_layer.num_attention_heads, dim1[1], dim2[1]}));
      //   tensor_shape_info_raw.insert(
      //       std::make_pair(dense_layer.top_names[1],
      //                      std::vector<int>{dim1[0], dense_layer.num_attention_heads, dim1[1],
      //                                       dim1[2] / dense_layer.num_attention_heads}));
      // } else {
      //   HCTR_OWN_THROW(Error_t::WrongInput,
      //                  "MultiHeadAttentionLayer needs two 4D or 3D input tensors ");
      // }
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
      int leading_dim = dense_layer.leading_dim;
      if (leading_dim > 0) {
        int batch_size = tensor_shape_info_raw[dense_layer.bottom_names[0]][0];
        int reshape_time_step = dense_layer.time_step;
        if (reshape_time_step == 0) {
          tensor_shape_info_raw.insert(
              std::make_pair(dense_layer.top_names[0], std::vector<int>{batch_size, leading_dim}));
        } else {
          tensor_shape_info_raw.insert(
              std::make_pair(dense_layer.top_names[0],
                             std::vector<int>{batch_size, reshape_time_step, leading_dim}));
        }
      } else {
        auto& shape = dense_layer.reshape_out_dimension;

        auto& input_shape = tensor_shape_info_raw[dense_layer.bottom_names[0]];
        std::vector<int64_t> int64_input_shape(input_shape.size());
        std::transform(input_shape.begin(), input_shape.end(), int64_input_shape.begin(),
                       [](const int& i) { return static_cast<int64_t>(i); });
        auto int64_out_shape = reshape_layer_utils::calc_output_shape(int64_input_shape, shape);
        std::vector<int> out_shape(int64_out_shape.size());
        std::transform(int64_out_shape.begin(), int64_out_shape.end(), out_shape.begin(),
                       [](const int64_t& i) { return static_cast<int>(i); });
        tensor_shape_info_raw.insert(std::make_pair(dense_layer.top_names[0], out_shape));
      }
      break;
    }
    case Layer_t::Select: {
      auto& dim = dense_layer.dim;
      auto& index = dense_layer.index;

      auto& in_shape = tensor_shape_info_raw[dense_layer.bottom_names[0]];
      HCTR_CHECK(dim < in_shape.size());
      HCTR_CHECK(index.size() <= in_shape[dim]);
      auto out_shape = in_shape;
      out_shape[dim] = index.size();
      tensor_shape_info_raw.insert(std::make_pair(dense_layer.top_names[0], out_shape));
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
      if (dim1.size() == 4) {
        tensor_shape_info_raw.insert(std::make_pair(
            dense_layer.top_names[0], std::vector<int>{dim1[0], dim1[1], dim1[2], dim2[3]}));
      } else if (dim1.size() == 3) {
        tensor_shape_info_raw.insert(
            std::make_pair(dense_layer.top_names[0], std::vector<int>{dim1[0], dim1[1], dim2[2]}));
      } else if (dim1.size() == 2) {
        tensor_shape_info_raw.insert(
            std::make_pair(dense_layer.top_names[0], std::vector<int>{dim1[0], dim2[1]}));
      } else {
        HCTR_OWN_THROW(Error_t::WrongInput,
                       "MatrixMultiplyLayer needs two 2D, 3D or 4D input tensors ");
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
