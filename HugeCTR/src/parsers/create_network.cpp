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
#include <layers/concat_3d_layer.hpp>
#include <layers/concat_layer.hpp>
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
#include <layers/layer_norm_layer.hpp>
#include <layers/matrix_multiply_layer.hpp>
#include <layers/multi_cross_layer.hpp>
#include <layers/multi_head_attention_layer.hpp>
#include <layers/prelu_dice_layer.hpp>
#include <layers/reduce_mean_layer.hpp>
#include <layers/reduce_sum_layer.hpp>
#include <layers/relu_layer.hpp>
#include <layers/reshape_layer.hpp>
#include <layers/scale_layer.hpp>
#include <layers/sequence_mask_layer.hpp>
#include <layers/sigmoid_layer.hpp>
#include <layers/slice_layer.hpp>
#include <layers/softmax_layer.hpp>
#include <layers/sub_layer.hpp>
#include <layers/weight_multiply_layer.hpp>
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
    const nlohmann::json& json, const std::vector<TensorEntry>& tensor_entries) {
  auto bottom = get_json(json, "bottom");
  auto top = get_json(json, "top");

  std::vector<std::string> bottom_names = get_layer_names(bottom);
  std::vector<std::string> top_names = get_layer_names(top);

  std::vector<TensorBag2> bottom_bags;

  for (auto& bottom_name : bottom_names) {
    for (auto& top_name : top_names) {
      if (bottom_name == top_name) {
        HCTR_OWN_THROW(Error_t::WrongInput, "bottom and top include a same layer name");
      }
    }
    TensorBag2 bag;
    if (!get_tensor_from_entries(tensor_entries, bottom_name, &bag)) {
      HCTR_OWN_THROW(Error_t::WrongInput, "No such bottom: " + bottom_name);
    }
    bottom_bags.push_back(bag);
  }
  return {bottom_bags, top_names};
}

template <typename T>
static std::shared_ptr<Regularizer<T>> create_regularizer(
    const nlohmann::json& j, const Tensor2<float>& weight_buff, const Tensor2<T>& wgrad_buff,
    const int batch_size, const std::shared_ptr<GPUResource>& gpu_resource) {
  std::shared_ptr<Regularizer<T>> reg(
      new NoRegularizer<T>(weight_buff, wgrad_buff, batch_size, gpu_resource));
  auto reg_it = j.find("regularizer");
  if (reg_it != j.end()) {
    Regularizer_t reg_type = Regularizer_t::None;
    auto reg_name = reg_it->get<std::string>();
    if (!find_item_in_map(reg_type, reg_name, REGULARIZER_TYPE_MAP)) {
      HCTR_OWN_THROW(Error_t::WrongInput, "No such regularizer: " + reg_name);
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

void create_layers(const nlohmann::json& j_array, std::vector<TensorEntry>& tensor_entries,
                   const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                   const std::shared_ptr<BufferBlock2<float>>& weight_buff,
                   const std::shared_ptr<BufferBlock2<__half>>& weight_buff_half,
                   const std::shared_ptr<BufferBlock2<float>>& wgrad_buff,
                   const std::shared_ptr<BufferBlock2<__half>>& wgrad_buff_half,
                   std::map<std::string, Tensor2<float>>& loss_tensors,
                   const std::shared_ptr<GPUResource>& gpu_resource, bool use_mixed_precision,
                   bool enable_tf32_compute, int num_networks_in_global, float scaler,
                   bool& enable_cuda_graph, bool inference_flag,
                   std::vector<std::unique_ptr<Layer>>& layers,
                   std::map<std::string, std::unique_ptr<ILoss>>& losses,
                   metrics::MultiLossMetricMap* raw_metrics,
                   std::vector<Layer*>* top_layers = nullptr,
                   std::vector<Layer*>* bottom_layers = nullptr) {
  bool skip_dgrad = true;
  bool is_bottom_mlp = true;

  auto emplaceback_layer = [&is_bottom_mlp, &layers, &bottom_layers, &top_layers](Layer* layer) {
    if (is_bottom_mlp) {
      if (bottom_layers) {
        bottom_layers->emplace_back(layer);
      }
    } else {
      if (top_layers) {
        top_layers->emplace_back(layer);
      }
    }
    layers.emplace_back(layer);
  };

  for (unsigned int i = 1; i < j_array.size(); i++) {
    const nlohmann::json& j = j_array[i];
    const auto layer_type_name = get_value_from_json<std::string>(j, "type");
    Layer_t layer_type;

    const auto& layer_map = use_mixed_precision ? LAYER_TYPE_MAP_MP : LAYER_TYPE_MAP;

    if (!find_item_in_map(layer_type, layer_type_name, layer_map)) {
      Embedding_t embedding_type;
      if (!find_item_in_map(embedding_type, layer_type_name, EMBEDDING_TYPE_MAP)) {
        HCTR_OWN_THROW(Error_t::WrongInput, "No such layer: " + layer_type_name);
      }
      continue;
    }

    // TODO: to make it generalized, we should not assume that the bottom name
    // includes "embedding". We need a better way to analyze such dependencies.
    auto bottom = get_json(j, "bottom");
    std::vector<std::string> bottom_strs = get_layer_names(bottom);
    for (const std::string& str : bottom_strs) {
      if (str.find("embedding") != std::string::npos) {
        is_bottom_mlp = false;
      }
    }

    std::vector<TensorEntry> output_tensor_entries;
    auto input_output_info = get_input_tensor_and_output_name(j, tensor_entries);
    switch (layer_type) {
      case Layer_t::BatchNorm: {
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
            HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + gamma_init_name);
          } else {
            initializer_types[0] = gamma_init_type;
          }
        }
        if (has_key_(j_bn_hparam, "beta_init")) {
          const auto beta_init_name = get_value_from_json<std::string>(j_bn_hparam, "beta_init");
          Initializer_t beta_init_type;
          if (!find_item_in_map(beta_init_type, beta_init_name, INITIALIZER_TYPE_MAP)) {
            HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + beta_init_name);
          } else {
            initializer_types[1] = beta_init_type;
          }
        }

        if (use_mixed_precision) {
          Tensor2<__half> bn_in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          // establish out tensor
          Tensor2<__half> bn_out_tensor;
          blobs_buff->reserve(bn_in_tensor.get_dimensions(), &bn_out_tensor);
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], bn_out_tensor.shrink()});

          BatchNormLayer<__half>::Params params = {factor, eps};
          emplaceback_layer(new BatchNormLayer<__half>(weight_buff, wgrad_buff, blobs_buff,
                                                       bn_in_tensor, bn_out_tensor, params,
                                                       gpu_resource, initializer_types));
        } else {
          Tensor2<float> bn_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          // establish out tensor
          Tensor2<float> bn_out_tensor;
          blobs_buff->reserve(bn_in_tensor.get_dimensions(), &bn_out_tensor);
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], bn_out_tensor.shrink()});

          BatchNormLayer<float>::Params params = {factor, eps};
          emplaceback_layer(new BatchNormLayer<float>(weight_buff, wgrad_buff, blobs_buff,
                                                      bn_in_tensor, bn_out_tensor, params,
                                                      gpu_resource, initializer_types));
        }

        break;
      }
      case Layer_t::LayerNorm: {
        // get LN params
        auto j_ln_hparam = get_json(j, "ln_param");
        auto eps = get_value_from_json<float>(j_ln_hparam, "eps");
        // establish initializer
        std::vector<Initializer_t> initializer_types(2, Initializer_t::Default);
        if (has_key_(j_ln_hparam, "gamma_init")) {
          const auto gamma_init_name = get_value_from_json<std::string>(j_ln_hparam, "gamma_init");
          Initializer_t gamma_init_type;
          if (!find_item_in_map(gamma_init_type, gamma_init_name, INITIALIZER_TYPE_MAP)) {
            HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + gamma_init_name);
          } else {
            initializer_types[0] = gamma_init_type;
          }
        }
        if (has_key_(j_ln_hparam, "beta_init")) {
          const auto beta_init_name = get_value_from_json<std::string>(j_ln_hparam, "beta_init");
          Initializer_t beta_init_type;
          if (!find_item_in_map(beta_init_type, beta_init_name, INITIALIZER_TYPE_MAP)) {
            HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + beta_init_name);
          } else {
            initializer_types[1] = beta_init_type;
          }
        }

        if (use_mixed_precision) {
          Tensor2<__half> ln_in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          // establish out tensor
          Tensor2<__half> ln_out_tensor;
          blobs_buff->reserve(ln_in_tensor.get_dimensions(), &ln_out_tensor);
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], ln_out_tensor.shrink()});

          LayerNormLayer<__half>::Params params = {eps};
          emplaceback_layer(new LayerNormLayer<__half>(weight_buff_half, wgrad_buff_half,
                                                       blobs_buff, ln_in_tensor, ln_out_tensor,
                                                       params, gpu_resource, initializer_types));
        } else {
          Tensor2<float> ln_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          // establish out tensor
          Tensor2<float> ln_out_tensor;
          blobs_buff->reserve(ln_in_tensor.get_dimensions(), &ln_out_tensor);
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], ln_out_tensor.shrink()});

          LayerNormLayer<float>::Params params = {eps};
          emplaceback_layer(new LayerNormLayer<float>(weight_buff, wgrad_buff, blobs_buff,
                                                      ln_in_tensor, ln_out_tensor, params,
                                                      gpu_resource, initializer_types));
        }

        break;
      }
      case Layer_t::BinaryCrossEntropyLoss: {
        if (input_output_info.inputs.size() != 2) {
          HCTR_OWN_THROW(Error_t::WrongInput, "bottom of BinaryCrossEntropyLoss must be two dim");
        }
        if (inference_flag) {
          HCTR_LOG(
              INFO, ROOT,
              "Inference stage skip BinaryCrossEntropyLoss layer, replaced by Sigmoid layer\n");
          if (use_mixed_precision) {
            Tensor2<__half> sigmoid_in_tensor =
                Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
            Tensor2<__half> sigmoid_out_tensor;
            blobs_buff->reserve(sigmoid_in_tensor.get_dimensions(), &sigmoid_out_tensor);
            emplaceback_layer(
                new SigmoidLayer<__half>(sigmoid_in_tensor, sigmoid_out_tensor, gpu_resource));
            output_tensor_entries.push_back({"sigmoid", sigmoid_out_tensor.shrink()});
          } else {
            // establish out tensor
            Tensor2<float> sigmoid_in_tensor =
                Tensor2<float>::stretch_from(input_output_info.inputs[0]);
            Tensor2<float> sigmoid_out_tensor;
            blobs_buff->reserve(sigmoid_in_tensor.get_dimensions(), &sigmoid_out_tensor);
            emplaceback_layer(
                new SigmoidLayer<float>(sigmoid_in_tensor, sigmoid_out_tensor, gpu_resource));
            output_tensor_entries.push_back({"sigmoid", sigmoid_out_tensor.shrink()});
          }
          break;
        }
        Tensor2<float> label_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[1]);

        // create new loss tensor
        auto name = input_output_info.output_names[0];
        Tensor2<float> new_loss_tensor;
        blobs_buff->reserve({1, 1}, &new_loss_tensor);

        // create new loss item
        std::unique_ptr<ILoss> new_loss;

        if (use_mixed_precision) {
          Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);

          new_loss.reset(new BinaryCrossEntropyLoss<__half>(
              label_tensor, in_tensor, new_loss_tensor,
              create_regularizer(j, weight_buff->as_tensor(), wgrad_buff_half->as_tensor(),
                                 in_tensor.get_dimensions()[0], gpu_resource),
              gpu_resource, num_networks_in_global, scaler));
        } else {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);

          new_loss.reset(new BinaryCrossEntropyLoss<float>(
              label_tensor, in_tensor, new_loss_tensor,
              create_regularizer(j, weight_buff->as_tensor(), wgrad_buff->as_tensor(),
                                 in_tensor.get_dimensions()[0], gpu_resource),
              gpu_resource, num_networks_in_global, scaler));
        }
        loss_tensors.insert(std::pair(name, new_loss_tensor));
        losses.insert(std::pair(name, std::move(new_loss)));
        break;
      }
      case Layer_t::Concat: {
        auto axis_it = j.find("axis");
        auto axis = (axis_it != j.end()) ? axis_it->get<int>() : 1;
        if (use_mixed_precision) {
          Tensors2<__half> in_tensors;
          for (const TensorBag2& bag : input_output_info.inputs) {
            in_tensors.push_back(Tensor2<__half>::stretch_from(bag));
          }
          Tensor2<__half> out_tensor;
          if (in_tensors[0].get_dimensions().size() == 2) {
            emplaceback_layer(
                new ConcatLayer<__half>(in_tensors, out_tensor, blobs_buff, gpu_resource));
          }
          if (in_tensors[0].get_dimensions().size() == 3) {
            emplaceback_layer(
                new Concat3DLayer<__half>(in_tensors, out_tensor, blobs_buff, axis, gpu_resource));
          }
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensors2<float> in_tensors;
          for (const TensorBag2& bag : input_output_info.inputs) {
            in_tensors.push_back(Tensor2<float>::stretch_from(bag));
          }
          Tensor2<float> out_tensor;
          if (in_tensors[0].get_dimensions().size() == 2) {
            emplaceback_layer(
                new ConcatLayer<float>(in_tensors, out_tensor, blobs_buff, gpu_resource));
          }
          if (in_tensors[0].get_dimensions().size() == 3) {
            emplaceback_layer(
                new Concat3DLayer<float>(in_tensors, out_tensor, blobs_buff, axis, gpu_resource));
          }
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        }
        break;
      }
      case Layer_t::CrossEntropyLoss: {
        if (input_output_info.inputs.size() != 2) {
          HCTR_OWN_THROW(Error_t::WrongInput, "bottom of CrossEntropyLoss must be two dim");
        }
        if (inference_flag) {
          HCTR_LOG(INFO, ROOT,
                   "Inference stage skip CrossEntropyLoss layer, replaced by Softmax layer\n");
          if (use_mixed_precision) {
            Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
            Tensor2<__half> out_tensor;
            blobs_buff->reserve(in_tensor.get_dimensions(), &out_tensor);
            output_tensor_entries.push_back(
                {input_output_info.output_names[0], out_tensor.shrink()});
            emplaceback_layer(
                new SoftmaxLayer<__half>(in_tensor, out_tensor, blobs_buff, gpu_resource));
          } else {
            Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
            Tensor2<float> out_tensor;
            blobs_buff->reserve(in_tensor.get_dimensions(), &out_tensor);
            output_tensor_entries.push_back(
                {input_output_info.output_names[0], out_tensor.shrink()});
            emplaceback_layer(
                new SoftmaxLayer<float>(in_tensor, out_tensor, blobs_buff, gpu_resource));
          }
          break;
        }
        Tensor2<float> label_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[1]);
        // create new loss tensor
        auto name = input_output_info.output_names[0];
        Tensor2<float> new_loss_tensor;
        blobs_buff->reserve({1, 1}, &new_loss_tensor);

        // create new loss item
        std::unique_ptr<ILoss> new_loss;

        if (use_mixed_precision) {
          Tensor2<__half> cross_entropy_loss_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.inputs[0]);

          new_loss.reset(new CrossEntropyLoss<__half>(
              label_tensor, cross_entropy_loss_in_tensor, new_loss_tensor,
              create_regularizer(j, weight_buff->as_tensor(), wgrad_buff_half->as_tensor(),
                                 cross_entropy_loss_in_tensor.get_dimensions()[0], gpu_resource),
              gpu_resource, num_networks_in_global, scaler));
        } else {
          Tensor2<float> cross_entropy_loss_in_tensor =
              Tensor2<float>::stretch_from(input_output_info.inputs[0]);

          new_loss.reset(new CrossEntropyLoss<float>(
              label_tensor, cross_entropy_loss_in_tensor, new_loss_tensor,
              create_regularizer(j, weight_buff->as_tensor(), wgrad_buff->as_tensor(),
                                 cross_entropy_loss_in_tensor.get_dimensions()[0], gpu_resource),
              gpu_resource, num_networks_in_global, scaler));
        }
        loss_tensors.insert(std::pair(name, new_loss_tensor));
        losses.insert(std::pair(name, std::move(new_loss)));
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
          emplaceback_layer(new DropoutLayer<__half>(do_in_tensor, do_out_tensor, blobs_buff, rate,
                                                     gpu_resource));
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
          emplaceback_layer(
              new DropoutLayer<float>(do_in_tensor, do_out_tensor, blobs_buff, rate, gpu_resource));
        }

        break;
      }
      case Layer_t::SequenceMask: {
        if (use_mixed_precision) {
          Tensor2<__half> smask_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> smask_out_tensor;
          auto max_sequence_len = get_json(j, "max_sequence_len");
          blobs_buff->reserve({smask_in_tensor.get_dimensions()[0], 1, 1, max_sequence_len},
                              &smask_out_tensor);
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], smask_out_tensor.shrink()});
          emplaceback_layer(new SequenceMaskLayer<__half>(
              smask_in_tensor, smask_out_tensor, max_sequence_len, blobs_buff, gpu_resource));
        } else {
          Tensor2<float> smask_in_tensor =
              Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> smask_out_tensor;
          auto max_sequence_len = get_json(j, "max_sequence_len");
          blobs_buff->reserve({smask_in_tensor.get_dimensions()[0], 1, 1, max_sequence_len},
                              &smask_out_tensor);
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], smask_out_tensor.shrink()});
          emplaceback_layer(new SequenceMaskLayer<float>(
              smask_in_tensor, smask_out_tensor, max_sequence_len, blobs_buff, gpu_resource));
        }
        break;
      }
      case Layer_t::ELU: {
        if (use_mixed_precision) {
          Tensor2<__half> elu_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.inputs[0]);

          // establish out tensor
          Tensor2<__half> elu_out_tensor;
          blobs_buff->reserve(elu_in_tensor.get_dimensions(), &elu_out_tensor);
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], elu_out_tensor.shrink()});
          // get ELU params
          auto j_elu_hparam = get_json(j, "elu_param");
          auto alpha = get_value_from_json<float>(j_elu_hparam, "alpha");
          emplaceback_layer(
              new EluLayer<__half>(elu_in_tensor, elu_out_tensor, alpha, gpu_resource));

        } else {
          Tensor2<float> elu_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);

          // establish out tensor
          Tensor2<float> elu_out_tensor;
          blobs_buff->reserve(elu_in_tensor.get_dimensions(), &elu_out_tensor);
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], elu_out_tensor.shrink()});
          // get ELU params
          auto j_elu_hparam = get_json(j, "elu_param");
          auto alpha = get_value_from_json<float>(j_elu_hparam, "alpha");
          emplaceback_layer(
              new EluLayer<float>(elu_in_tensor, elu_out_tensor, alpha, gpu_resource));
        }
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
            HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + weight_init_name);
          } else {
            initializer_types[0] = weight_init_type;
          }
        }
        if (has_key_(j_fc_param, "bias_init")) {
          const auto bias_init_name = get_value_from_json<std::string>(j_fc_param, "bias_init");
          Initializer_t bias_init_type;
          if (!find_item_in_map(bias_init_type, bias_init_name, INITIALIZER_TYPE_MAP)) {
            HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + bias_init_name);
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
            HCTR_OWN_THROW(Error_t::WrongInput, "No such position: " + pos_str);
          } else if (pos_type == FcPosition_t::Head && input_size == 1 && output_size == 4) {
          } else if (pos_type == FcPosition_t::Body && input_size == 4 && output_size == 4) {
          } else if (pos_type == FcPosition_t::Tail && input_size == 4 && output_size == 1) {
          } else if (pos_type == FcPosition_t::Isolated && input_size == 1 && output_size == 1) {
          } else {
            HCTR_OWN_THROW(
                Error_t::WrongInput,
                "The position and dimension of bottom and top layer aren't compatible: " +
                    layer_type_name);
          }
        }

        // check the activation functino of this layer
        Activation_t act_type = Activation_t::Relu;
        if (has_key_(j, "activation")) {
          auto act_name = get_value_from_json<std::string>(j, "activation");
          if (!find_item_in_map(act_type, act_name, ACTIVATION_TYPE_MAP)) {
            HCTR_OWN_THROW(Error_t::WrongInput, "No such activation: " + act_name);
          }
          if (act_type == Activation_t::None && pos_type != FcPosition_t::Tail)
            HCTR_OWN_THROW(Error_t::WrongInput,
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
          HCTR_OWN_THROW(Error_t::WrongInput, "FusedInnerProduct support half only");
        }
        break;
      }

      case Layer_t::Cast: {
        if (use_mixed_precision) {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> out_tensor;
          blobs_buff->reserve(in_tensor.get_dimensions(), &out_tensor);
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
          emplaceback_layer(new CastLayer<float, __half>(in_tensor, out_tensor, gpu_resource));
        } else {
          Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> out_tensor;
          blobs_buff->reserve(in_tensor.get_dimensions(), &out_tensor);
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
          emplaceback_layer(new CastLayer<__half, float>(in_tensor, out_tensor, gpu_resource));
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
            HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + weight_init_name);
          } else {
            initializer_types[0] = weight_init_type;
          }
        }
        if (has_key_(j_fc_param, "bias_init")) {
          const auto bias_init_name = get_value_from_json<std::string>(j_fc_param, "bias_init");
          Initializer_t bias_init_type;
          if (!find_item_in_map(bias_init_type, bias_init_name, INITIALIZER_TYPE_MAP)) {
            HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + bias_init_name);
          } else {
            initializer_types[1] = bias_init_type;
          }
        }

        // establish out tensor
        auto output = get_value_from_json<size_t>(j_fc_param, "num_output");

        if (use_mixed_precision) {
          Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> fc_out_tensor;
          if (in_tensor.get_dimensions().size() == 2) {
            blobs_buff->reserve({in_tensor.get_dimensions()[0], output}, &fc_out_tensor);
          } else if (in_tensor.get_dimensions().size() == 3) {
            blobs_buff->reserve(
                {in_tensor.get_dimensions()[0], in_tensor.get_dimensions()[1], output},
                &fc_out_tensor);
          }

          // establish layer
          emplaceback_layer(new FullyConnectedLayer<__half>(
              weight_buff, weight_buff_half, wgrad_buff_half, blobs_buff, in_tensor, fc_out_tensor,
              gpu_resource, initializer_types));
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], fc_out_tensor.shrink()});
        } else {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> fc_out_tensor;

          if (in_tensor.get_dimensions().size() == 2) {
            blobs_buff->reserve({in_tensor.get_dimensions()[0], output}, &fc_out_tensor);
          } else if (in_tensor.get_dimensions().size() == 3) {
            blobs_buff->reserve(
                {in_tensor.get_dimensions()[0], in_tensor.get_dimensions()[1], output},
                &fc_out_tensor);
          }
          // establish layer
          emplaceback_layer(new FullyConnectedLayer<float>(
              weight_buff, wgrad_buff, in_tensor, fc_out_tensor, gpu_resource, use_mixed_precision,
              enable_tf32_compute, initializer_types));
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], fc_out_tensor.shrink()});
        }
        break;
      }
      case Layer_t::MultiHeadAttention: {
        if (input_output_info.inputs.size() != 2) {
          HCTR_OWN_THROW(Error_t::WrongInput, "MultiHeadAttentionLayer needs two input tensors ");
        }
        if (use_mixed_precision) {
          Tensors2<__half> in_tensors;
          for (const TensorBag2& bag : input_output_info.inputs) {
            in_tensors.push_back(Tensor2<__half>::stretch_from(bag));
          }
          Tensor2<__half> out_tensor;
          layers.emplace_back(
              new MultiHeadAttentionLayer<__half>(in_tensors, out_tensor, blobs_buff, gpu_resource,
                                                  use_mixed_precision, enable_tf32_compute));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensors2<float> in_tensors;
          for (const auto& bag : input_output_info.inputs) {
            in_tensors.push_back(Tensor2<float>::stretch_from(bag));
          }
          Tensor2<float> out_tensor;
          layers.emplace_back(new MultiHeadAttentionLayer<float>(in_tensors, out_tensor, blobs_buff,
                                                                 gpu_resource, use_mixed_precision,
                                                                 enable_tf32_compute));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        }
        break;
      }
      case Layer_t::Interaction: {
        // TODO: lambda template could be a better solution here, but there's not support in c++11
        if (use_mixed_precision) {
          if (gpu_resource->get_cc_major() < 7) {
            std::ostringstream os;
            os << "InteractionLayer<__half> is not supported in SM " << gpu_resource->get_cc_major()
               << '.' << gpu_resource->get_cc_minor();
            HCTR_OWN_THROW(Error_t::WrongInput, os.str());
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
          emplaceback_layer(new InteractionLayer<float>(in_mlp_tensor, in_emb_tensor, out_tensor,
                                                        blobs_buff, gpu_resource,
                                                        use_mixed_precision, enable_tf32_compute));
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
            HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + weight_init_name);
          } else {
            initializer_types[0] = weight_init_type;
          }
        }
        if (has_key_(j_mc_param, "bias_init")) {
          const auto bias_init_name = get_value_from_json<std::string>(j_mc_param, "bias_init");
          Initializer_t bias_init_type;
          if (!find_item_in_map(bias_init_type, bias_init_name, INITIALIZER_TYPE_MAP)) {
            HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + bias_init_name);
          } else {
            initializer_types[1] = bias_init_type;
          }
        }

        // establish out tensor
        auto num_layers = get_value_from_json<int>(j_mc_param, "num_layers");
        if (use_mixed_precision) {
          Tensor2<__half> mc_in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> out_tensor;
          blobs_buff->reserve(mc_in_tensor.get_dimensions(), &out_tensor);
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
          // establish layer
          emplaceback_layer(new MultiCrossLayer<__half>(
              weight_buff, weight_buff_half, wgrad_buff_half, blobs_buff, mc_in_tensor, out_tensor,
              gpu_resource, num_layers, initializer_types));
        } else {
          Tensor2<float> mc_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> out_tensor;
          blobs_buff->reserve(mc_in_tensor.get_dimensions(), &out_tensor);
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
          // establish layer
          emplaceback_layer(new MultiCrossLayer<float>(
              weight_buff, weight_buff, wgrad_buff, blobs_buff, mc_in_tensor, out_tensor,
              gpu_resource, num_layers, initializer_types));
        }
        break;
      }

      case Layer_t::MultiCrossEntropyLoss: {
        if (input_output_info.inputs.size() != 2) {
          HCTR_OWN_THROW(Error_t::WrongInput, "bottom of MultiCrossEntropyLoss must be two dim");
        }
        if (inference_flag) {
          HCTR_LOG(INFO, ROOT,
                   "Inference stage skip MultiCrossEntropyLoss layer, replaced by Sigmoid layer\n");
          if (use_mixed_precision) {
            Tensor2<__half> sigmoid_in_tensor =
                Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
            Tensor2<__half> sigmoid_out_tensor;
            blobs_buff->reserve(sigmoid_in_tensor.get_dimensions(), &sigmoid_out_tensor);
            emplaceback_layer(
                new SigmoidLayer<__half>(sigmoid_in_tensor, sigmoid_out_tensor, gpu_resource));
            output_tensor_entries.push_back({"sigmoid", sigmoid_out_tensor.shrink()});
          } else {
            // establish out tensor
            Tensor2<float> sigmoid_in_tensor =
                Tensor2<float>::stretch_from(input_output_info.inputs[0]);
            Tensor2<float> sigmoid_out_tensor;
            blobs_buff->reserve(sigmoid_in_tensor.get_dimensions(), &sigmoid_out_tensor);
            emplaceback_layer(
                new SigmoidLayer<float>(sigmoid_in_tensor, sigmoid_out_tensor, gpu_resource));
            output_tensor_entries.push_back({"sigmoid", sigmoid_out_tensor.shrink()});
          }
          break;
        }
        auto tweight = get_json(j, "target_weight");
        std::vector<float> target_weight_vec;
        for (auto tweight_tmp : tweight) {
          float tweight_val = tweight_tmp.get<float>();
          target_weight_vec.push_back(tweight_val);
        }

        Tensor2<float> label_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[1]);
        // create new loss tensor
        auto name = input_output_info.output_names[0];
        Tensor2<float> new_loss_tensor;
        blobs_buff->reserve({1, 1}, &new_loss_tensor);

        // create new loss item
        std::unique_ptr<ILoss> new_loss;

        if (use_mixed_precision) {
          Tensor2<__half> multi_cross_entropy_loss_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          new_loss.reset(new MultiCrossEntropyLoss<__half>(
              label_tensor, multi_cross_entropy_loss_in_tensor, new_loss_tensor,
              create_regularizer(j, weight_buff->as_tensor(), wgrad_buff_half->as_tensor(),
                                 multi_cross_entropy_loss_in_tensor.get_dimensions()[0],
                                 gpu_resource),
              target_weight_vec, gpu_resource, num_networks_in_global, scaler));
        } else {
          Tensor2<float> multi_cross_entropy_loss_in_tensor =
              Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          new_loss.reset(new MultiCrossEntropyLoss<float>(
              label_tensor, multi_cross_entropy_loss_in_tensor, new_loss_tensor,
              create_regularizer(j, weight_buff->as_tensor(), wgrad_buff->as_tensor(),
                                 multi_cross_entropy_loss_in_tensor.get_dimensions()[0],
                                 gpu_resource),
              target_weight_vec, gpu_resource, num_networks_in_global, scaler));
        }
        loss_tensors.insert(std::pair(name, new_loss_tensor));
        losses.insert(std::pair(name, std::move(new_loss)));
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
      case Layer_t::ReduceMean: {
        int axis = get_json(j, "axis").get<int>();
        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> out_tensor;
        emplaceback_layer(
            new ReduceMeanLayer<float>(in_tensor, out_tensor, blobs_buff, axis, gpu_resource));
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
        emplaceback_layer(new SubLayer<float>(in_tensors, out_tensor, blobs_buff, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        break;
      }
      case Layer_t::Gather: {
        std::vector<int> indices;
        auto j_indices = get_json(j, "indices");
        assert(j_indices.is_array());
        for (auto j_index : j_indices) {
          indices.emplace_back(int(j_index));
        }
        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> out_tensor;
        emplaceback_layer(
            new GatherLayer<float>(in_tensor, out_tensor, blobs_buff, indices, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});

        break;
      }
      case Layer_t::GRU: {
        auto j_gru_param = get_json(j, "gru_param");
        // establish initializer
        std::vector<Initializer_t> initializer_types(2, Initializer_t::Default);
        if (has_key_(j_gru_param, "weight_init")) {
          const auto weight_init_name =
              get_value_from_json<std::string>(j_gru_param, "weight_init");
          Initializer_t weight_init_type;
          if (!find_item_in_map(weight_init_type, weight_init_name, INITIALIZER_TYPE_MAP)) {
            HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + weight_init_name);
          } else {
            initializer_types[0] = weight_init_type;
          }
        }
        if (has_key_(j_gru_param, "bias_init")) {
          const auto bias_init_name = get_value_from_json<std::string>(j_gru_param, "bias_init");
          Initializer_t bias_init_type;
          if (!find_item_in_map(bias_init_type, bias_init_name, INITIALIZER_TYPE_MAP)) {
            HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + bias_init_name);
          } else {
            initializer_types[1] = bias_init_type;
          }
        }

        // establish out tensor
        auto output = get_value_from_json<size_t>(j_gru_param, "num_output");
        auto batchsize = get_value_from_json<size_t>(j_gru_param, "batchsize");
        auto SeqLength = get_value_from_json<size_t>(j_gru_param, "SeqLength");
        auto embedding_vec_size = get_value_from_json<size_t>(j_gru_param, "vector_size");

        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> gru_out_tensor;
        blobs_buff->reserve({in_tensor.get_dimensions()[0], output}, &gru_out_tensor);
        // establish layer
        emplaceback_layer(new GRULayer<float>(weight_buff, wgrad_buff, in_tensor, gru_out_tensor,
                                              output, batchsize, SeqLength, embedding_vec_size,
                                              gpu_resource, initializer_types));
        output_tensor_entries.push_back(
            {input_output_info.output_names[0], gru_out_tensor.shrink()});

        break;
      }
      case Layer_t::MatrixMultiply: {
        Tensors2<float> in_tensors;
        for (const auto& bag : input_output_info.inputs) {
          in_tensors.push_back(Tensor2<float>::stretch_from(bag));
        }
        Tensor2<float> out_tensor;
        blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
        layers.emplace_back(
            new MatrixMultiplyLayer<float>(in_tensors, out_tensor, blobs_buff, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        break;
      }
      case Layer_t::Softmax: {
        if (use_mixed_precision) {
          HCTR_OWN_THROW(Error_t::WrongInput, "Softmax layer does not support fp16");
        } else {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> out_tensor;
          blobs_buff->reserve(in_tensor.get_dimensions(), &out_tensor);
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
          emplaceback_layer(
              new SoftmaxLayer<float>(in_tensor, out_tensor, blobs_buff, gpu_resource));
        }
        break;
      }
      case Layer_t::PReLU_Dice: {
        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> out_tensor;
        blobs_buff->reserve(in_tensor.get_dimensions(), &out_tensor);
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        // get PReLU_Dice params
        auto j_prelu_dice_param = get_json(j, "prelu_dice_param");
        auto alpha = get_value_from_json<float>(j_prelu_dice_param, "alpha");
        auto epsilon = get_value_from_json<float>(j_prelu_dice_param, "eps");
        emplaceback_layer(new PRelu_Dice_Layer<float>(in_tensor, out_tensor, blobs_buff, alpha,
                                                      epsilon, gpu_resource));
        break;
      }
      case Layer_t::Scale: {
        Tensor2<float> scale_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> scale_out_tensor;
        // get Scale params
        auto j_scale_param = get_json(j, "scale_param");
        auto axis = get_value_from_json<float>(j_scale_param, "axis");
        auto factor = get_value_from_json<float>(j_scale_param, "factor");
        emplaceback_layer(new ScaleLayer<float>(scale_in_tensor, scale_out_tensor, blobs_buff, axis,
                                                factor, gpu_resource));
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
        emplaceback_layer(
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
        emplaceback_layer(new FusedReshapeConcatGeneralLayer<float>(in_tensors, out_tensor,
                                                                    blobs_buff, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
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
            if (slot_id < 0) {
              HCTR_OWN_THROW(Error_t::WrongInput, "slot_id < 0");
            }
            selected.push_back(slot_id);
          }

          if (use_mixed_precision) {
            Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
            Tensor2<__half> out_tensor;
            emplaceback_layer(new ReshapeLayer<__half>(in_tensor, out_tensor, blobs_buff, selected,
                                                       gpu_resource));
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
          auto j_time_step = j.find("time_step");
          // if leading_dim is not specified, default leading_dim = n_slots * vector_length
          if (use_mixed_precision) {
            Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
            Tensor2<__half> out_tensor;
            const auto& in_dims = in_tensor.get_dimensions();
            size_t leading_dim = (leading_dim_it != j.end())
                                     ? (*leading_dim_it).get<int>()
                                     : in_tensor.get_num_elements() / in_dims[0];
            size_t time_step = (j_time_step != j.end()) ? (*j_time_step).get<int>() : 0;
            if (time_step == 0) {  // 2D output
              blobs_buff->reserve({in_tensor.get_num_elements() / leading_dim, leading_dim},
                                  &out_tensor);
            } else {  // 3D output
              size_t batch_size = in_tensor.get_num_elements() / leading_dim / time_step;
              blobs_buff->reserve({batch_size, time_step, leading_dim}, &out_tensor);
            }
            emplaceback_layer(
                new ReshapeLayer<__half>(in_tensor, out_tensor, blobs_buff, gpu_resource));
            output_tensor_entries.push_back(
                {input_output_info.output_names[0], out_tensor.shrink()});
          } else {
            Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
            Tensor2<float> out_tensor;
            const auto& in_dims = in_tensor.get_dimensions();
            size_t leading_dim = (leading_dim_it != j.end())
                                     ? (*leading_dim_it).get<int>()
                                     : in_tensor.get_num_elements() / in_dims[0];
            size_t time_step = (j_time_step != j.end()) ? (*j_time_step).get<int>() : 0;
            if (time_step == 0) {  // 2D output
              blobs_buff->reserve({in_tensor.get_num_elements() / leading_dim, leading_dim},
                                  &out_tensor);
            } else {  // 3D output
              size_t batch_size = in_tensor.get_num_elements() / leading_dim / time_step;
              blobs_buff->reserve({batch_size, time_step, leading_dim}, &out_tensor);
            }
            emplaceback_layer(
                new ReshapeLayer<float>(in_tensor, out_tensor, blobs_buff, gpu_resource));
            output_tensor_entries.push_back(
                {input_output_info.output_names[0], out_tensor.shrink()});
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
          emplaceback_layer(
              new SigmoidLayer<__half>(sigmoid_in_tensor, sigmoid_out_tensor, gpu_resource));
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], sigmoid_out_tensor.shrink()});
        } else {
          // establish out tensor
          Tensor2<float> sigmoid_in_tensor =
              Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> sigmoid_out_tensor;
          blobs_buff->reserve(sigmoid_in_tensor.get_dimensions(), &sigmoid_out_tensor);
          emplaceback_layer(
              new SigmoidLayer<float>(sigmoid_in_tensor, sigmoid_out_tensor, gpu_resource));
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], sigmoid_out_tensor.shrink()});
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
      case Layer_t::WeightMultiply: {
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
            HCTR_OWN_THROW(Error_t::WrongInput, "No such initializer: " + weight_init_name);
          } else {
            initializer_types[0] = weight_init_type;
          }
        }

        if (use_mixed_precision) {
          Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> out_tensor;
          emplaceback_layer(new WeightMultiplyLayer<__half>(
              weight_buff, weight_buff_half, wgrad_buff_half, blobs_buff, in_tensor, out_tensor,
              weight_dims, gpu_resource, initializer_types));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> out_tensor;
          emplaceback_layer(new WeightMultiplyLayer<float>(
              weight_buff, weight_buff, wgrad_buff, blobs_buff, in_tensor, out_tensor, weight_dims,
              gpu_resource, initializer_types));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        }
        break;
      }
      case Layer_t::FmOrder2: {
        auto out_dim = get_json(j, "out_dim").get<size_t>();

        if (use_mixed_precision) {
          Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> out_tensor;
          blobs_buff->reserve({in_tensor.get_dimensions()[0], out_dim}, &out_tensor);

          emplaceback_layer(new FmOrder2Layer<__half>(in_tensor, out_tensor, gpu_resource));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> out_tensor;
          blobs_buff->reserve({in_tensor.get_dimensions()[0], out_dim}, &out_tensor);

          emplaceback_layer(new FmOrder2Layer<float>(in_tensor, out_tensor, gpu_resource));
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
          emplaceback_layer(new AddLayer<__half>(in_tensors, out_tensor, blobs_buff, gpu_resource));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensors2<float> in_tensors;
          for (const auto& bag : input_output_info.inputs) {
            in_tensors.push_back(Tensor2<float>::stretch_from(bag));
          }
          Tensor2<float> out_tensor;
          blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
          emplaceback_layer(new AddLayer<float>(in_tensors, out_tensor, blobs_buff, gpu_resource));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        }
        break;
      }
      case Layer_t::ReduceSum: {
        int axis = get_json(j, "axis").get<int>();

        if (use_mixed_precision) {
          Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> out_tensor;
          emplaceback_layer(
              new ReduceSumLayer<__half>(in_tensor, out_tensor, blobs_buff, axis, gpu_resource));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> out_tensor;
          emplaceback_layer(
              new ReduceSumLayer<float>(in_tensor, out_tensor, blobs_buff, axis, gpu_resource));
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
          emplaceback_layer(new ElementwiseMultiplyLayer<__half>(in_tensors, out_tensor, blobs_buff,
                                                                 gpu_resource));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensors2<float> in_tensors;
          for (const auto& bag : input_output_info.inputs) {
            in_tensors.push_back(Tensor2<float>::stretch_from(bag));
          }
          Tensor2<float> out_tensor;
          blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
          emplaceback_layer(new ElementwiseMultiplyLayer<float>(in_tensors, out_tensor, blobs_buff,
                                                                gpu_resource));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        }
        break;
      }
      default:
        assert(!"Error: no such layer && should never get here!");
    }  // end of switch
    if (!inference_flag &&
        (layer_type == Layer_t::CrossEntropyLoss || layer_type == Layer_t::BinaryCrossEntropyLoss ||
         layer_type == Layer_t::MultiCrossEntropyLoss)) {
      if (raw_metrics) {
        std::string name = get_layer_names(bottom)[1];
        Tensor2<float> lookup_loss_tensor = loss_tensors.find(name)->second;

        metrics::RawMetricMap new_map;
        new_map.insert(std::make_pair(metrics::RawType::Loss, lookup_loss_tensor.shrink()));
        new_map.insert(std::make_pair(metrics::RawType::Pred, input_output_info.inputs[0]));
        new_map.insert(std::make_pair(metrics::RawType::Label, input_output_info.inputs[1]));
      }
    } else {
      for (auto& output_tensor_entry : output_tensor_entries) {
        tensor_entries.push_back(output_tensor_entry);
      }
    }

    skip_dgrad = false;
  }  // for layers
}

/*
 * Create single network
 *
 */
Network* Network::create_network(const nlohmann::json& j_array, const nlohmann::json& j_optimizer,
                                 std::vector<TensorEntry>& train_tensor_entries,
                                 std::vector<TensorEntry>& evaluate_tensor_entries,
                                 int num_networks_in_global,
                                 std::shared_ptr<ExchangeWgrad>& exchange_wgrad,
                                 const std::shared_ptr<CPUResource>& cpu_resource,
                                 const std::shared_ptr<GPUResource>& gpu_resource,
                                 bool use_mixed_precision, bool enable_tf32_compute, float scaler,
                                 bool use_algorithm_search, bool use_cuda_graph,
                                 bool inference_flag, bool grouped_all_reduce) {
  Network* network = new Network(cpu_resource, gpu_resource, use_mixed_precision, use_cuda_graph);

  auto& train_layers = network->train_layers_;
  auto* bottom_layers = &network->bottom_layers_;
  auto* top_layers = &network->top_layers_;
  auto& evaluate_layers = network->evaluate_layers_;
  auto& train_loss_tensors = network->train_loss_tensors_;
  auto& evaluate_loss_tensors = network->evaluate_loss_tensors_;
  auto& train_losses = network->train_losses_;
  auto& evaluate_losses = network->evaluate_losses_;
  auto& enable_cuda_graph = network->enable_cuda_graph_;
  auto& raw_metrics = network->raw_metrics_;

  std::shared_ptr<GeneralBuffer2<CudaAllocator>> blobs_buff =
      GeneralBuffer2<CudaAllocator>::create();

  std::shared_ptr<BufferBlock2<float>> train_weight_buff = blobs_buff->create_block<float>();
  std::shared_ptr<BufferBlock2<__half>> train_weight_buff_half = blobs_buff->create_block<__half>();
  std::shared_ptr<BufferBlock2<float>> wgrad_buff = nullptr;
  std::shared_ptr<BufferBlock2<__half>> wgrad_buff_half = nullptr;

  if (!inference_flag) {
    if (use_mixed_precision) {
      auto id = gpu_resource->get_local_id();
      wgrad_buff_half =
          (grouped_all_reduce)
              ? std::dynamic_pointer_cast<GroupedExchangeWgrad<__half>>(exchange_wgrad)
                    ->get_network_wgrad_buffs()[id]
              : std::dynamic_pointer_cast<NetworkExchangeWgrad<__half>>(exchange_wgrad)
                    ->get_network_wgrad_buffs()[id];
      wgrad_buff = blobs_buff->create_block<float>();  // placeholder
    } else {
      auto id = gpu_resource->get_local_id();
      wgrad_buff = (grouped_all_reduce)
                       ? std::dynamic_pointer_cast<GroupedExchangeWgrad<float>>(exchange_wgrad)
                             ->get_network_wgrad_buffs()[id]
                       : std::dynamic_pointer_cast<NetworkExchangeWgrad<float>>(exchange_wgrad)
                             ->get_network_wgrad_buffs()[id];
      wgrad_buff_half = blobs_buff->create_block<__half>();  // placeholder
    }
  } else {
    wgrad_buff = blobs_buff->create_block<float>();
    wgrad_buff_half = blobs_buff->create_block<__half>();
  }

  std::shared_ptr<BufferBlock2<float>> evaluate_weight_buff = blobs_buff->create_block<float>();
  std::shared_ptr<BufferBlock2<__half>> evaluate_weight_buff_half =
      blobs_buff->create_block<__half>();
  std::shared_ptr<BufferBlock2<float>> wgrad_buff_placeholder = blobs_buff->create_block<float>();
  std::shared_ptr<BufferBlock2<__half>> wgrad_buff_half_placeholder =
      blobs_buff->create_block<__half>();

  std::shared_ptr<BufferBlock2<float>> opt_buff = blobs_buff->create_block<float>();
  std::shared_ptr<BufferBlock2<__half>> opt_buff_half = blobs_buff->create_block<__half>();

  // TODO: implement multiple loss support in create_layers
  if (!inference_flag) {
    // create train layers
    create_layers(j_array, train_tensor_entries, blobs_buff, train_weight_buff,
                  train_weight_buff_half, wgrad_buff, wgrad_buff_half, train_loss_tensors,
                  gpu_resource, use_mixed_precision, enable_tf32_compute, num_networks_in_global,
                  scaler, enable_cuda_graph, inference_flag, train_layers, train_losses, nullptr,
                  top_layers, bottom_layers);
  }

  // create evaluate layers
  create_layers(j_array, evaluate_tensor_entries, blobs_buff, evaluate_weight_buff,
                evaluate_weight_buff_half, wgrad_buff_placeholder, wgrad_buff_half_placeholder,
                evaluate_loss_tensors, gpu_resource, use_mixed_precision, enable_tf32_compute,
                num_networks_in_global, scaler, enable_cuda_graph, inference_flag, evaluate_layers,
                evaluate_losses, &raw_metrics);

  // create optimizer
  if (!inference_flag) {
    if (use_mixed_precision) {
      auto opt_param = get_optimizer_param(j_optimizer);

      network->optimizer_ = std::move(Optimizer::Create(
          opt_param, train_weight_buff->as_tensor(), train_weight_buff_half->as_tensor(),
          wgrad_buff_half->as_tensor(), scaler, opt_buff_half, gpu_resource, use_mixed_precision));
    } else {
      auto opt_param = get_optimizer_param(j_optimizer);

      network->optimizer_ = std::move(Optimizer::Create(
          opt_param, train_weight_buff->as_tensor(), train_weight_buff_half->as_tensor(),
          wgrad_buff->as_tensor(), scaler, opt_buff, gpu_resource, use_mixed_precision));
    }
  } else {
    try {
      TensorEntry pred_tensor_entry = evaluate_tensor_entries.back();
      if (use_mixed_precision) {
        network->pred_tensor_half_ = Tensor2<__half>::stretch_from(pred_tensor_entry.bag);
      } else {
        network->pred_tensor_ = Tensor2<float>::stretch_from(pred_tensor_entry.bag);
      }
    } catch (const std::runtime_error& rt_err) {
      HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
      throw;
    }
  }

  network->train_weight_tensor_ = train_weight_buff->as_tensor();
  network->train_weight_tensor_half_ = train_weight_buff_half->as_tensor();
  network->wgrad_tensor_ = wgrad_buff->as_tensor();
  network->wgrad_tensor_half_ = wgrad_buff_half->as_tensor();
  network->evaluate_weight_tensor_ = evaluate_weight_buff->as_tensor();
  network->evaluate_weight_tensor_half_ = evaluate_weight_buff_half->as_tensor();
  network->opt_tensor_ = opt_buff->as_tensor();
  network->opt_tensor_half_ = opt_buff_half->as_tensor();

  CudaDeviceContext context(gpu_resource->get_device_id());
  blobs_buff->allocate();

  return network;
}

}  // namespace HugeCTR
