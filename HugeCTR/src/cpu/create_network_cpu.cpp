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

#include <cpu/network_cpu.hpp>
#include <cpu/layer_cpu.hpp>
#include <cpu/layers//add_layer_cpu.hpp>
#include <cpu/layers/batch_norm_layer_cpu.hpp>
#include <cpu/layers/cast_layer_cpu.hpp>
#include <cpu/layers/concat_layer_cpu.hpp>
#include <cpu/layers/dot_product_layer_cpu.hpp>
#include <cpu/layers/dropout_layer_cpu.hpp>
#include <cpu/layers/elu_layer_cpu.hpp>
#include <cpu/layers/fm_order2_layer_cpu.hpp>
#include <cpu/layers/fully_connected_layer_cpu.hpp>
#include <cpu/layers/fully_connected_layer_half_cpu.hpp>
#include <cpu/layers/fused_fully_connected_layer_cpu.hpp>
#include <cpu/layers/interaction_layer_cpu.hpp>
#include <cpu/layers/multi_cross_layer_cpu.hpp>
#include <cpu/layers/reduce_sum_layer_cpu.hpp>
#include <cpu/layers/relu_layer_cpu.hpp>
#include <cpu/layers/reshape_layer_cpu.hpp>
#include <cpu/layers/sigmoid_layer_cpu.hpp>
#include <cpu/layers/slice_layer_cpu.hpp>
#include <cpu/layers/weight_multiply_layer_cpu.hpp>

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

void create_layers(const nlohmann::json& j_array, std::vector<TensorEntry>& tensor_entries,
                   const std::shared_ptr<GeneralBuffer2<HostAllocator>>& blobs_buff,
                   const std::shared_ptr<BufferBlock2<float>>& weight_buff,
                   const std::shared_ptr<BufferBlock2<__half>>& weight_buff_half,
                   const std::shared_ptr<BufferBlock2<float>>& wgrad_buff,
                   const std::shared_ptr<BufferBlock2<__half>>& wgrad_buff_half,
                   bool use_mixed_precision,
                   std::vector<std::unique_ptr<LayerCPU>>& layers) {
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

    std::vector<TensorEntry> output_tensor_entries;
    auto input_output_info = get_input_tensor_and_output_name(j, tensor_entries);
    if (layer_type == Layer_t::CrossEntropyLoss ||
        layer_type == Layer_t::BinaryCrossEntropyLoss ||
        layer_type == Layer_t::MultiCrossEntropyLoss) {
      CK_THROW_(Error_t::WrongInput, "Loss layer is not supported for NetworkCPU");
    }
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

        if (use_mixed_precision) {
          Tensor2<__half> bn_in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          // establish out tensor
          Tensor2<__half> bn_out_tensor;
          blobs_buff->reserve(bn_in_tensor.get_dimensions(), &bn_out_tensor);
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], bn_out_tensor.shrink()});

          BatchNormLayerCPU<__half>::Params params = {factor, eps};
          layers.emplace_back(new BatchNormLayerCPU<__half>(weight_buff, wgrad_buff, blobs_buff,
                                                         bn_in_tensor, bn_out_tensor, params));
        } else {
          Tensor2<float> bn_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          // establish out tensor
          Tensor2<float> bn_out_tensor;
          blobs_buff->reserve(bn_in_tensor.get_dimensions(), &bn_out_tensor);
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], bn_out_tensor.shrink()});

          BatchNormLayerCPU<float>::Params params = {factor, eps};
          layers.emplace_back(new BatchNormLayerCPU<float>(weight_buff, wgrad_buff, blobs_buff,
                                                        bn_in_tensor, bn_out_tensor, params));
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
              new ConcatLayerCPU<__half>(in_tensors, out_tensor, blobs_buff));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensors2<float> in_tensors;
          for (const TensorBag2& bag : input_output_info.inputs) {
            in_tensors.push_back(Tensor2<float>::stretch_from(bag));
          }
          Tensor2<float> out_tensor;
          layers.emplace_back(
              new ConcatLayerCPU<float>(in_tensors, out_tensor, blobs_buff));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
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
          layers.emplace_back(new DropoutLayerCPU<__half>(do_in_tensor, do_out_tensor, blobs_buff,
                                                       rate));
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
          layers.emplace_back(
              new DropoutLayerCPU<float>(do_in_tensor, do_out_tensor, blobs_buff, rate));
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
          layers.emplace_back(
              new EluLayerCPU<__half>(elu_in_tensor, elu_out_tensor, alpha));

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
          layers.emplace_back(
              new EluLayerCPU<float>(elu_in_tensor, elu_out_tensor, alpha));
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
          blobs_buff->reserve({(in_tensor.get_dimensions())[0], output}, &fc_out_tensor);
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], fc_out_tensor.shrink()});

          // establish layer
          layers.emplace_back(new FusedFullyConnectedLayerCPU(
              weight_buff, weight_buff_half, wgrad_buff_half, blobs_buff, in_tensor, fc_out_tensor));
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
          layers.emplace_back(new CastLayerCPU<float, __half>(in_tensor, out_tensor));
        } else {
          Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> out_tensor;
          blobs_buff->reserve(in_tensor.get_dimensions(), &out_tensor);
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
          layers.emplace_back(new CastLayerCPU<__half, float>(in_tensor, out_tensor));
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
          layers.emplace_back(new FullyConnectedLayerCPU<__half>(
              weight_buff, weight_buff_half, wgrad_buff_half, blobs_buff, in_tensor, fc_out_tensor));
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], fc_out_tensor.shrink()});
        } else {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> fc_out_tensor;
          blobs_buff->reserve({in_tensor.get_dimensions()[0], output}, &fc_out_tensor);
          // establish layer
          layers.emplace_back(new FullyConnectedLayerCPU<float>(
              weight_buff, wgrad_buff, in_tensor, fc_out_tensor, use_mixed_precision));
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], fc_out_tensor.shrink()});
        }
        break;
      }

      case Layer_t::Interaction: {
        // lambda template could be a better solution here, but there's not support in c++11
        if (use_mixed_precision) {
          Tensor2<__half> in_mlp_tensor =
              Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> in_emb_tensor =
              Tensor2<__half>::stretch_from(input_output_info.inputs[1]);
          Tensor2<__half> out_tensor;

          layers.emplace_back(new InteractionLayerCPU<__half>(
              in_mlp_tensor, in_emb_tensor, out_tensor,
              blobs_buff,  // todo cannot use this blobs_buff here need half
              use_mixed_precision));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});

        } else {
          Tensor2<float> in_mlp_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> in_emb_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[1]);
          Tensor2<float> out_tensor;
          layers.emplace_back(
              new InteractionLayerCPU<float>(in_mlp_tensor, in_emb_tensor, out_tensor, blobs_buff,
                                          use_mixed_precision));
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
        layers.emplace_back(new MultiCrossLayerCPU(weight_buff, wgrad_buff, blobs_buff, mc_in_tensor,
                                                out_tensor, num_layers));
        break;
      }
      case Layer_t::ReLU: {
        if (use_mixed_precision) {
          Tensor2<__half> relu_in_tensor =
              Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> relu_out_tensor;
          blobs_buff->reserve(relu_in_tensor.get_dimensions(), &relu_out_tensor);
          layers.emplace_back(new ReluLayerCPU<__half>(relu_in_tensor, relu_out_tensor));
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], relu_out_tensor.shrink()});
        } else {
          // establish out tensor
          Tensor2<float> relu_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> relu_out_tensor;
          blobs_buff->reserve(relu_in_tensor.get_dimensions(), &relu_out_tensor);
          layers.emplace_back(new ReluLayerCPU<float>(relu_in_tensor, relu_out_tensor));
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
            layers.emplace_back(new ReshapeLayerCPU<__half>(in_tensor, out_tensor, blobs_buff,
                                                         selected));
            output_tensor_entries.push_back(
                {input_output_info.output_names[0], out_tensor.shrink()});
          } else {
            Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
            Tensor2<float> out_tensor;
            layers.emplace_back(
                new ReshapeLayerCPU<float>(in_tensor, out_tensor, blobs_buff, selected));
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
            layers.emplace_back(new ReshapeLayerCPU<__half>(in_tensor, out_tensor, blobs_buff,
                                                         leading_dim));
            output_tensor_entries.push_back(
                {input_output_info.output_names[0], out_tensor.shrink()});
          } else {
            Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
            Tensor2<float> out_tensor;
            const auto& in_dims = in_tensor.get_dimensions();
            size_t leading_dim = (leading_dim_it != j.end())
                                     ? (*leading_dim_it).get<int>()
                                     : in_tensor.get_num_elements() / in_dims[0];
            layers.emplace_back(new ReshapeLayerCPU<float>(in_tensor, out_tensor, blobs_buff,
                                                        leading_dim));
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
          layers.emplace_back(
              new SigmoidLayerCPU<__half>(sigmoid_in_tensor, sigmoid_out_tensor));
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], sigmoid_out_tensor.shrink()});
        } else {
          // establish out tensor
          Tensor2<float> sigmoid_in_tensor =
              Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> sigmoid_out_tensor;
          blobs_buff->reserve(sigmoid_in_tensor.get_dimensions(), &sigmoid_out_tensor);
          layers.emplace_back(
              new SigmoidLayerCPU<float>(sigmoid_in_tensor, sigmoid_out_tensor));
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
          layers.emplace_back(
              new SliceLayerCPU<__half>(in_tensor, out_tensors, blobs_buff, ranges));
          for (size_t i = 0; i < out_tensors.size(); i++) {
            output_tensor_entries.push_back(
                {input_output_info.output_names[i], out_tensors[i].shrink()});
          }
        } else {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensors2<float> out_tensors;
          layers.emplace_back(
              new SliceLayerCPU<float>(in_tensor, out_tensors, blobs_buff, ranges));
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
            CK_THROW_(Error_t::WrongInput, "No such initializer: " + weight_init_name);
          } else {
            initializer_types[0] = weight_init_type;
          }
        }

        if (use_mixed_precision) {
          Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> out_tensor;
          layers.emplace_back(
              new WeightMultiplyLayerCPU<__half>(weight_buff_half, wgrad_buff_half, blobs_buff, in_tensor,
                                              out_tensor, weight_dims));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> out_tensor;
          layers.emplace_back(new WeightMultiplyLayerCPU<float>(weight_buff, wgrad_buff, blobs_buff,
                                                             in_tensor, out_tensor, weight_dims));
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

          layers.emplace_back(new FmOrder2LayerCPU<__half>(in_tensor, out_tensor));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> out_tensor;
          blobs_buff->reserve({in_tensor.get_dimensions()[0], out_dim}, &out_tensor);

          layers.emplace_back(new FmOrder2LayerCPU<float>(in_tensor, out_tensor));
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
          layers.emplace_back(
              new AddLayerCPU<__half>(in_tensors, out_tensor, blobs_buff));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensors2<float> in_tensors;
          for (const auto& bag : input_output_info.inputs) {
            in_tensors.push_back(Tensor2<float>::stretch_from(bag));
          }
          Tensor2<float> out_tensor;
          blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
          layers.emplace_back(
              new AddLayerCPU<float>(in_tensors, out_tensor, blobs_buff));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        }
        break;
      }
      case Layer_t::ReduceSum: {
        int axis = get_json(j, "axis").get<int>();

        if (use_mixed_precision) {
          Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> out_tensor;
          layers.emplace_back(
              new ReduceSumLayerCPU<__half>(in_tensor, out_tensor, blobs_buff, axis));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> out_tensor;
          layers.emplace_back(
              new ReduceSumLayerCPU<float>(in_tensor, out_tensor, blobs_buff, axis));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        }
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
              new DotProductLayerCPU<__half>(in_tensors, out_tensor, blobs_buff));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensors2<float> in_tensors;
          for (const auto& bag : input_output_info.inputs) {
            in_tensors.push_back(Tensor2<float>::stretch_from(bag));
          }
          Tensor2<float> out_tensor;
          blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
          layers.emplace_back(
              new DotProductLayerCPU<float>(in_tensors, out_tensor, blobs_buff));
          output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        }
        break;
      }
      default:
        assert(!"Error: no such layer && should never get here!");
    }  // end of switch

    for (auto& output_tensor_entry : output_tensor_entries) {
      tensor_entries.push_back(output_tensor_entry);
    }
  }  // for layers
  for (auto entry:tensor_entries) {
    std::cout << "[HUGECTR][INFO] layer: "<< entry.name << std::endl;
  }
}
              
/*
 * Create single network
 *
 */
NetworkCPU* NetworkCPU::create_network(const nlohmann::json& j_array,
                                 std::vector<TensorEntry>& tensor_entries,
                                 const std::shared_ptr<CPUResource>& cpu_resource,
                                 bool use_mixed_precision) {
  NetworkCPU* network = new NetworkCPU(cpu_resource, use_mixed_precision);

  auto& layers = network->layers_;

  std::shared_ptr<GeneralBuffer2<HostAllocator>> blobs_buff =
      GeneralBuffer2<HostAllocator>::create();

  std::shared_ptr<BufferBlock2<float>> weight_buff = blobs_buff->create_block<float>();
  std::shared_ptr<BufferBlock2<__half>> weight_buff_half = blobs_buff->create_block<__half>();
  std::shared_ptr<BufferBlock2<float>> wgrad_buff = blobs_buff->create_block<float>();
  std::shared_ptr<BufferBlock2<__half>> wgrad_buff_half = blobs_buff->create_block<__half>();

  // create layers
  create_layers(j_array, tensor_entries, blobs_buff, weight_buff,
                weight_buff_half, wgrad_buff, wgrad_buff_half,
                use_mixed_precision, layers);

  TensorEntry pred_tensor_entry = tensor_entries.back();
  network->pred_tensor_ = Tensor2<float>::stretch_from(pred_tensor_entry.bag);
  network->weight_tensor_ = weight_buff->as_tensor();
  network->weight_tensor_half_ = weight_buff_half->as_tensor();
  network->wgrad_tensor_ = wgrad_buff->as_tensor();
  network->wgrad_tensor_half_ = wgrad_buff_half->as_tensor();
  blobs_buff->allocate();

  return network;
}

}  // namespace HugeCTR
