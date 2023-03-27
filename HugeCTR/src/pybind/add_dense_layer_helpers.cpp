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

#include <core23/data_type.hpp>
#include <core23/tensor.hpp>
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
#include <layers/scale_layer.hpp>
#include <layers/sequence_mask_layer.hpp>
#include <layers/sigmoid_layer.hpp>
#include <layers/slice_layer.hpp>
#include <layers/softmax_layer.hpp>
#include <layers/sub_layer.hpp>
#include <layers/weight_multiply_layer.hpp>
#include <network_buffer_channels.hpp>
#include <parser.hpp>
#include <pybind/add_dense_layer_helpers.hpp>
#include <pybind/model.hpp>
#include <regularizer_factory.hpp>

namespace HugeCTR {

namespace {

template <typename DType, template <class T> class LossType>
std::unique_ptr<ILoss> create_loss(core23::Tensor& label_tensor, core23::Tensor& input_tensor,
                                   core23::Tensor& loss_tensor, const DenseLayer& dense_layer,
                                   const std::vector<std::unique_ptr<Layer>>& layers,
                                   int gpu_count_in_total,
                                   const std::shared_ptr<GPUResource>& gpu_resource, float scaler,
                                   bool gen_loss_summary) {
  auto weight_tensors = std::is_same_v<DType, __half>
                            ? get_master_weight_tensor_vector<float>(layers)
                            : get_weight_tensor_vector<float>(layers);
  auto wgrad_tensors = get_wgrad_tensor_vector<DType>(layers);
  const int batch_size = input_tensor.size(0);
  auto regularizer = create_regularizer<DType>(
      dense_layer.use_regularizer, dense_layer.regularizer_type, dense_layer.lambda, weight_tensors,
      wgrad_tensors, batch_size, gpu_resource);
  if constexpr (std::is_same_v<LossType<DType>, MultiCrossEntropyLoss<DType>>) {
    return std::make_unique<LossType<DType>>(label_tensor, input_tensor, loss_tensor, regularizer,
                                             dense_layer.target_weight_vec, gpu_resource,
                                             gpu_count_in_total, scaler, gen_loss_summary);
  } else {
    return std::make_unique<LossType<DType>>(label_tensor, input_tensor, loss_tensor, regularizer,
                                             gpu_resource, gpu_count_in_total, scaler,
                                             gen_loss_summary);
  }
}

}  // namespace

std::optional<core23::Tensor> get_tensor_from_entities(
    const std::vector<TensorEntity> tensor_entities, const std::string& name) {
  for (const TensorEntity& entity : tensor_entities) {
    if (entity.name == name) {
      return entity.tensor;
    }
  }
  return {};
}

InputTensorsAndOutputNames get_input_tensors_and_output_names(
    const std::vector<std::string>& bottom_names, const std::vector<std::string>& top_names,
    const std::vector<TensorEntity>& tensor_entities) {
  std::vector<core23::Tensor> bottom_tensors;
  for (auto& bottom_name : bottom_names) {
    for (auto& top_name : top_names) {
      if (bottom_name == top_name) {
        HCTR_OWN_THROW(Error_t::WrongInput, "bottom and top include a same layer name");
      }
    }
    if (auto tensor = get_tensor_from_entities(tensor_entities, bottom_name)) {
      bottom_tensors.push_back(*tensor);
    } else {
      HCTR_OWN_THROW(Error_t::WrongInput, "No such bottom: " + bottom_name);
    }
  }
  return {bottom_tensors, top_names};
}

void add_dense_layer_impl(DenseLayer& dense_layer, std::vector<TensorEntity>& tensor_entities,
                          std::vector<std::unique_ptr<Layer>>& layers,
                          std::map<std::string, std::unique_ptr<ILoss>>& losses,
                          metrics::Core23MultiLossMetricMap* raw_metrics, int gpu_count_in_total,
                          const std::shared_ptr<GPUResource>& gpu_resource,
                          bool use_mixed_precision, bool enable_tf32_compute, float scaler,
                          bool use_algorithm_search,
                          std::vector<Layer*>* embedding_dependent_layers,
                          std::vector<Layer*>* embedding_independent_layers,
                          bool embedding_dependent, const Solver& solver) {
  bool skip_dgrad = layers.size() == 0;

  Layer_t layer_type = dense_layer.layer_type;
  const auto& layer_type_to_string =
      use_mixed_precision ? LAYER_TYPE_TO_STRING_MP : LAYER_TYPE_TO_STRING;
  if (layer_type_to_string.find(layer_type) == layer_type_to_string.end()) {
    auto layer_type_name = layer_type_to_string.at(layer_type);
    std::string prefix = use_mixed_precision ? "Mixed" : "Single";
    HCTR_OWN_THROW(Error_t::WrongInput,
                   prefix + "Mixed precision not supported for: " + layer_type_name);
  }

  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(core23::Device(core23::DeviceType::GPU, gpu_resource->get_device_id()))
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(GetBlobsBufferChannel());

  std::vector<TensorEntity> output_tensor_entities;
  auto input_output_info = get_input_tensors_and_output_names(
      dense_layer.bottom_names, dense_layer.top_names, tensor_entities);
  switch (layer_type) {
    case Layer_t::BatchNorm: {
      auto& bn_in_tensor = input_output_info.input_tensors[0];
      core23::Tensor bn_out_tensor(tensor_params.shape(bn_in_tensor.shape()));
      output_tensor_entities.push_back({input_output_info.output_names[0], bn_out_tensor});
      std::vector<Initializer_t> initializer_types{dense_layer.gamma_init_type,
                                                   dense_layer.beta_init_type};
      if (use_mixed_precision) {
        Core23TempBatchNormLayer<__half>::Params params = {dense_layer.factor, dense_layer.eps};
        layers.emplace_back(new Core23TempBatchNormLayer<__half>(
            bn_in_tensor, bn_out_tensor, params, gpu_resource, initializer_types));
      } else {
        Core23TempBatchNormLayer<float>::Params params = {dense_layer.factor, dense_layer.eps};
        layers.emplace_back(new Core23TempBatchNormLayer<float>(bn_in_tensor, bn_out_tensor, params,
                                                                gpu_resource, initializer_types));
      }
      break;
    }
    case Layer_t::LayerNorm: {
      auto& ln_in_tensor = input_output_info.input_tensors[0];
      core23::Tensor ln_out_tensor(tensor_params.shape(ln_in_tensor.shape()));
      output_tensor_entities.push_back({input_output_info.output_names[0], ln_out_tensor});
      std::vector<Initializer_t> initializer_types{dense_layer.gamma_init_type,
                                                   dense_layer.beta_init_type};

      if (use_mixed_precision) {
        Core23TempLayerNormLayer<__half>::Params params = {dense_layer.eps};
        layers.emplace_back(new Core23TempLayerNormLayer<__half>(
            ln_in_tensor, ln_out_tensor, params, gpu_resource, initializer_types));
      } else {
        Core23TempLayerNormLayer<float>::Params params = {dense_layer.eps};
        layers.emplace_back(new Core23TempLayerNormLayer<float>(ln_in_tensor, ln_out_tensor, params,
                                                                gpu_resource, initializer_types));
      }
      break;
    }
    case Layer_t::BinaryCrossEntropyLoss: {
      if (input_output_info.input_tensors.size() != 2) {
        HCTR_OWN_THROW(Error_t::WrongInput, "BinaryCrossEntropyLoss must have two inputs");
      }

      if (input_output_info.input_tensors[0].shape() !=
          input_output_info.input_tensors[1].shape()) {
        std::string err_msg =
            "predition tensor and label tensor should have the same shape, got: " +
            input_output_info.input_tensors[0].shape().str() + " and " +
            input_output_info.input_tensors[1].shape().str();
        HCTR_OWN_THROW(Error_t::WrongInput, err_msg.c_str());
      }

      auto& input_tensor = input_output_info.input_tensors[0];
      auto& label_tensor = input_output_info.input_tensors[1];
      core23::Tensor loss_tensor(tensor_params.shape({1, 1}).data_type(core23::ScalarType::Float));

      std::unique_ptr<ILoss> new_loss;
      if (use_mixed_precision) {
        new_loss = create_loss<__half, BinaryCrossEntropyLoss>(
            label_tensor, input_tensor, loss_tensor, dense_layer, layers, gpu_count_in_total,
            gpu_resource, scaler, solver.gen_loss_summary);
      } else {
        new_loss = create_loss<float, BinaryCrossEntropyLoss>(
            label_tensor, input_tensor, loss_tensor, dense_layer, layers, gpu_count_in_total,
            gpu_resource, scaler, solver.gen_loss_summary);
      }

      std::string name = dense_layer.bottom_names[1];
      losses.insert(std::make_pair(name, std::move(new_loss)));
      break;
    }
    case Layer_t::Concat: {
      [[maybe_unused]] auto axis = dense_layer.axis;
      auto& in_tensors = input_output_info.input_tensors;
      core23::Tensor out_tensor;  // out_tensor.empty() == true
      std::unique_ptr<Layer> layer;
      // TODO: remove if-else once the refactoring is done
      if (in_tensors[0].dims() == 2) {
        if (use_mixed_precision) {
          layer.reset(new ConcatLayer<__half>(in_tensors, out_tensor, gpu_resource));
        } else {
          layer.reset(new ConcatLayer<float>(in_tensors, out_tensor, gpu_resource));
        }
      } else if (in_tensors[0].dims() == 3) {
        if (use_mixed_precision) {
          layer.reset(new Concat3DLayer<__half>(in_tensors, out_tensor, axis, gpu_resource));
        } else {
          layer.reset(new Concat3DLayer<float>(in_tensors, out_tensor, axis, gpu_resource));
        }
      } else {
        HCTR_DIE("Concatenation of %lld-dimensional Tensors is not supported!\n",
                 in_tensors[0].dims());
      }
      layers.emplace_back(std::move(layer));
      output_tensor_entities.push_back({input_output_info.output_names[0], out_tensor});
      break;
    }
    case Layer_t::CrossEntropyLoss: {
      if (input_output_info.input_tensors.size() != 2) {
        HCTR_OWN_THROW(Error_t::WrongInput, "CrossEntropyLoss must have two inputs");
      }

      if (input_output_info.input_tensors[0].shape() !=
          input_output_info.input_tensors[1].shape()) {
        std::string err_msg =
            "predition tensor and label tensor should have the same shape, got: " +
            input_output_info.input_tensors[0].shape().str() + " and " +
            input_output_info.input_tensors[1].shape().str();
        HCTR_OWN_THROW(Error_t::WrongInput, err_msg.c_str());
      }

      auto& input_tensor = input_output_info.input_tensors[0];
      auto& label_tensor = input_output_info.input_tensors[1];
      core23::Tensor loss_tensor(tensor_params.shape({1, 1}).data_type(core23::ScalarType::Float));

      std::unique_ptr<ILoss> new_loss;
      if (use_mixed_precision) {
        new_loss = create_loss<__half, CrossEntropyLoss>(
            label_tensor, input_tensor, loss_tensor, dense_layer, layers, gpu_count_in_total,
            gpu_resource, scaler, solver.gen_loss_summary);
      } else {
        new_loss = create_loss<float, CrossEntropyLoss>(
            label_tensor, input_tensor, loss_tensor, dense_layer, layers, gpu_count_in_total,
            gpu_resource, scaler, solver.gen_loss_summary);
      }

      std::string name = dense_layer.bottom_names[1];
      losses.insert(std::make_pair(name, std::move(new_loss)));
      break;
    }
    case Layer_t::Dropout: {
      auto& in_tensor = input_output_info.input_tensors[0];
      core23::Tensor out_tensor(tensor_params.shape(in_tensor.shape()));
      [[maybe_unused]] float rate = dense_layer.dropout_rate;
      std::unique_ptr<Layer> layer;
      // TODO: remove if-else once the refactoring is done
      if (use_mixed_precision) {
        layer.reset(new DropoutLayer<__half>(in_tensor, out_tensor, rate, gpu_resource));
      } else {
        layer.reset(new DropoutLayer<float>(in_tensor, out_tensor, rate, gpu_resource));
      }
      layers.emplace_back(std::move(layer));
      output_tensor_entities.push_back({input_output_info.output_names[0], out_tensor});

      break;
    }
    case Layer_t::ELU: {
      auto& in_tensor = input_output_info.input_tensors[0];
      core23::Tensor out_tensor(tensor_params.shape(in_tensor.shape()));
      float alpha = dense_layer.elu_alpha;
      std::unique_ptr<Layer> layer;
      // TODO: remove if-else once the refactoring is done
      if (use_mixed_precision) {
        layer.reset(new EluLayer<__half>(in_tensor, out_tensor, alpha, gpu_resource));
      } else {
        layer.reset(new EluLayer<float>(in_tensor, out_tensor, alpha, gpu_resource));
      }
      layers.emplace_back(std::move(layer));
      output_tensor_entities.push_back({input_output_info.output_names[0], out_tensor});
      break;
    }
    case Layer_t::SequenceMask: {
      auto& input_tensor = input_output_info.input_tensors[0];
      auto max_sequence_len = dense_layer.max_sequence_len;
      core23::Tensor output_tensor(
          tensor_params.shape({input_tensor.size(0), 1, 1, max_sequence_len}));
      std::unique_ptr<Layer> layer;
      // TODO: remove if-else once the refactoring is done
      if (use_mixed_precision) {
        layer.reset(new SequenceMaskLayer<__half>(input_tensor, output_tensor, max_sequence_len,
                                                  gpu_resource));
      } else {
        layer.reset(new SequenceMaskLayer<float>(input_tensor, output_tensor, max_sequence_len,
                                                 gpu_resource));
      }
      layers.emplace_back(std::move(layer));
      output_tensor_entities.push_back({input_output_info.output_names[0], output_tensor});
      break;
    }
    case Layer_t::MLP: {
      std::vector<Initializer_t> initializer_types{dense_layer.weight_init_type,
                                                   dense_layer.bias_init_type};
      int input_size = input_output_info.input_tensors.size();
      int output_size = input_output_info.output_names.size();
      std::vector<core23::Tensor> in_tensors;
      auto& train_in_tensor = input_output_info.input_tensors[0];
      in_tensors.push_back(train_in_tensor);
      if (input_size == 2) {
        auto& mask_in_tensor = input_output_info.input_tensors[1];
        in_tensors.push_back(mask_in_tensor);
      }

      // TODO: remove when dense_layer.num_outputs is std::vector<int64_t>;
      std::vector<int64_t> num_outputs(dense_layer.num_outputs.begin(),
                                       dense_layer.num_outputs.end());

      std::vector<core23::Tensor> train_out_tensors;
      int64_t batch_size = train_in_tensor.shape().size(0);
      int64_t output_dim = *num_outputs.rbegin();
      if (output_size == 1) {
        train_out_tensors.emplace_back(tensor_params.shape({batch_size, output_dim}));
      } else {
        HCTR_OWN_THROW(Error_t::WrongInput, "MLP layer can only have one output.");
      }

      std::vector<Activation_t> acts(num_outputs.size(), dense_layer.act_type);
      if (!dense_layer.acts.empty()) {
        if (acts.size() != dense_layer.acts.size()) {
          HCTR_OWN_THROW(Error_t::WrongInput,
                         "The number of activations should be equal to the number of layers.");
        }
        acts = dense_layer.acts;
      }

      std::vector<bool> biases(num_outputs.size(), dense_layer.use_bias);
      if (!dense_layer.biases.empty()) {
        if (biases.size() != dense_layer.biases.size()) {
          HCTR_OWN_THROW(Error_t::WrongInput,
                         "The number of biases should be equal to the number of layers.");
        }
        biases = dense_layer.biases;
      }

      if (use_mixed_precision) {
        layers.emplace_back(new Core23TempMLPLayer<__half>(
            in_tensors, train_out_tensors, num_outputs, gpu_resource, acts, biases,
            initializer_types, skip_dgrad, dense_layer.compute_config.async_wgrad,
            dense_layer.compute_config.fuse_wb, enable_tf32_compute));
      } else {
        layers.emplace_back(new Core23TempMLPLayer<float>(
            in_tensors, train_out_tensors, num_outputs, gpu_resource, acts, biases,
            initializer_types, skip_dgrad, dense_layer.compute_config.async_wgrad,
            dense_layer.compute_config.fuse_wb, enable_tf32_compute));
      }

      if (output_size == 1) {
        output_tensor_entities.push_back({input_output_info.output_names[0], train_out_tensors[0]});
      }
      break;
    }
    case Layer_t::FusedInnerProduct: {
      std::vector<Initializer_t> initializer_types{dense_layer.weight_init_type,
                                                   dense_layer.bias_init_type};
      // check the position of this layer
      int input_size = input_output_info.input_tensors.size();
      int output_size = input_output_info.output_names.size();
      int64_t output = dense_layer.num_output;
      auto pos_type = dense_layer.pos_type;
      auto act_type = dense_layer.act_type;
      bool head_mask_in = pos_type == FcPosition_t::Head && input_size == 2;
      if (skip_dgrad && pos_type == FcPosition_t::Head && input_size == 2) {
        HCTR_OWN_THROW(
            Error_t::WrongInput,
            "FusedInnerProduct Head Layer should have only one input tensors when it is the "
            "first dense layer");
      }
      if (dense_layer.compute_config.async_wgrad && !skip_dgrad && pos_type == FcPosition_t::Head &&
          input_size == 1) {
        HCTR_OWN_THROW(Error_t::WrongInput,
                       "FusedInnerProduct Head Layer should have two input tensors when turning on "
                       "async wgrad knob");
      }
      if (pos_type == FcPosition_t::Head && skip_dgrad && input_size == 1 && output_size == 4) {
      } else if (pos_type == FcPosition_t::Head && !skip_dgrad && input_size == 2 &&
                 output_size == 4) {
      } else if (!dense_layer.compute_config.async_wgrad && pos_type == FcPosition_t::Head &&
                 !skip_dgrad && input_size == 1 && output_size == 4) {
      } else if (pos_type == FcPosition_t::Body && input_size == 4 && output_size == 4) {
      } else if (pos_type == FcPosition_t::Tail && input_size == 4 && output_size == 1) {
      } else if (pos_type == FcPosition_t::Isolated && input_size == 1 && output_size == 1) {
      } else if (pos_type == FcPosition_t::None && input_size == 1 && output_size == 1) {
      } else {
        HCTR_OWN_THROW(Error_t::WrongInput,
                       "The position and dimension of bottom and top layer aren't compatible: " +
                           LAYER_TYPE_TO_STRING_MP[layer_type]);
      }
      if (act_type == Activation_t::None && pos_type != FcPosition_t::Tail) {
        HCTR_OWN_THROW(Error_t::WrongInput,
                       "The layer without activation function must be the last layer in MLP.");
      }
      if (use_mixed_precision) {
        auto& train_in_tensor = input_output_info.input_tensors[0];
        core23::Tensor mask_in_tensor, dRelu_in_tensor, db_in_tensor;
        if (pos_type == FcPosition_t::Body || pos_type == FcPosition_t::Tail) {
          mask_in_tensor = input_output_info.input_tensors[1];
          dRelu_in_tensor = input_output_info.input_tensors[2];
          db_in_tensor = input_output_info.input_tensors[3];
        } else if (pos_type == FcPosition_t::Head && input_size == 2) {
          mask_in_tensor = input_output_info.input_tensors[1];
        }
        core23::Tensor train_out_tensor(
            tensor_params.shape({train_in_tensor.shape().size(0), output}));
        core23::Tensor mask_out_tensor(
            tensor_params.shape({train_in_tensor.shape().size(0), output}));
        core23::Tensor dRelu_out_tensor(
            tensor_params.shape({train_in_tensor.shape().size(0), output}));
        core23::Tensor db_out_tensor;

        if (pos_type == FcPosition_t::None) {
          layers.emplace_back(new Core23TempFusedFullyConnectedLayer(
              train_in_tensor, train_out_tensor, gpu_resource, initializer_types));
        } else {
          layers.emplace_back(new Core23TempFusedReluBiasFullyConnectedLayer(
              train_in_tensor, mask_in_tensor, dRelu_in_tensor, db_in_tensor, train_out_tensor,
              mask_out_tensor, dRelu_out_tensor, db_out_tensor, gpu_resource, pos_type, act_type,
              skip_dgrad, initializer_types, dense_layer.compute_config.async_wgrad, head_mask_in,
              dense_layer.compute_config.fuse_wb));
        }

        if (pos_type == FcPosition_t::Tail || pos_type == FcPosition_t::Isolated ||
            pos_type == FcPosition_t::None)
          output_tensor_entities.push_back({input_output_info.output_names[0], train_out_tensor});
        else {
          output_tensor_entities.push_back({input_output_info.output_names[0], train_out_tensor});
          output_tensor_entities.push_back({input_output_info.output_names[1], mask_out_tensor});
          output_tensor_entities.push_back({input_output_info.output_names[2], dRelu_out_tensor});
          output_tensor_entities.push_back({input_output_info.output_names[3], db_out_tensor});
        }
      } else {
        HCTR_OWN_THROW(Error_t::WrongInput, "FusedInnerProduct support half only");
      }
      break;
    }
    case Layer_t::Cast: {
      auto& input_tensor = input_output_info.input_tensors[0];
      core23::Tensor output_tensor(tensor_params.shape(input_tensor.shape()));
      std::unique_ptr<Layer> layer;
      if (use_mixed_precision) {
        layer.reset(new CastLayer<float, __half>(input_tensor, output_tensor, gpu_resource));
      } else {
        layer.reset(new CastLayer<__half, float>(input_tensor, output_tensor, gpu_resource));
      }
      layers.emplace_back(std::move(layer));
      output_tensor_entities.push_back({input_output_info.output_names[0], output_tensor});
      break;
    }
    case Layer_t::InnerProduct: {
      std::vector<Initializer_t> initializer_types{dense_layer.weight_init_type,
                                                   dense_layer.bias_init_type};
      int64_t output = dense_layer.num_output;
      auto& in_tensor = input_output_info.input_tensors[0];
      core23::Tensor fc_out_tensor;
      if (in_tensor.shape().dims() == 2) {
        fc_out_tensor = core23::Tensor(tensor_params.shape({in_tensor.shape().size(0), output}));
      } else if (in_tensor.shape().dims() == 3) {
        fc_out_tensor = core23::Tensor(
            tensor_params.shape({in_tensor.shape().size(0), in_tensor.shape().size(1), output}));
      }

      if (use_mixed_precision) {
        layers.emplace_back(new Core23TempFullyConnectedLayer<__half>(
            in_tensor, fc_out_tensor, gpu_resource, initializer_types));
      } else {
        layers.emplace_back(new Core23TempFullyConnectedLayer<float>(
            in_tensor, fc_out_tensor, gpu_resource, use_mixed_precision, enable_tf32_compute,
            initializer_types));
      }
      output_tensor_entities.push_back({input_output_info.output_names[0], fc_out_tensor});
      break;
    }
    case Layer_t::MultiHeadAttention: {
      if (input_output_info.input_tensors.size() < 2) {
        HCTR_OWN_THROW(Error_t::WrongInput,
                       "MultiHeadAttentionLayer needs at lease two input tensors ");
      }
      [[maybe_unused]] auto num_attention_heads = dense_layer.num_attention_heads;
      [[maybe_unused]] auto transpose_b = dense_layer.transpose_b;

      auto& in_tensors = input_output_info.input_tensors;
      if (in_tensors[0].shape().dims() != 4 && in_tensors[1].shape().dims() != 3) {
        HCTR_OWN_THROW(Error_t::WrongInput,
                       "MultiHeadAttentionLayer needs 3D or 4D input tensors ");
      }
      std::vector<core23::Tensor> out_tensors;
      if (use_mixed_precision) {
        layers.emplace_back(new MultiHeadAttentionLayer<__half>(
            in_tensors, out_tensors, num_attention_heads, transpose_b, gpu_resource,
            use_mixed_precision, enable_tf32_compute));
      } else {
        layers.emplace_back(new MultiHeadAttentionLayer<float>(
            in_tensors, out_tensors, num_attention_heads, transpose_b, gpu_resource,
            use_mixed_precision, enable_tf32_compute));
      }

      for (size_t i = 0; i < out_tensors.size(); i++) {
        output_tensor_entities.push_back({input_output_info.output_names[i], out_tensors[i]});
      }
      break;
    }
    case Layer_t::Interaction: {
      if (input_output_info.input_tensors.size() != 2) {
        HCTR_OWN_THROW(Error_t::WrongInput, "InteractionLayer needs two input tensors ");
      }
      if (input_output_info.output_names.size() != 2 &&
          input_output_info.output_names.size() != 1) {
        HCTR_OWN_THROW(Error_t::WrongInput,
                       "InteractionLayer should have one or two output tensors");
      }
      if (input_output_info.output_names.size() == 1 &&
          dense_layer.compute_config.async_wgrad == true) {
        HCTR_OWN_THROW(
            Error_t::WrongInput,
            "InteractionLayer should have two output tensors when turning on async wgrad knob");
      }
      if (input_output_info.output_names.size() == 2 && !use_mixed_precision) {
        HCTR_OWN_THROW(Error_t::WrongInput,
                       "InteractionLayer<float> should have only one output tensor");
      }

      if (use_mixed_precision && gpu_resource->get_cc_major() < 7) {
        std::ostringstream os;
        os << "InteractionLayer<__half> is not supported in SM " << gpu_resource->get_cc_major()
           << '.' << gpu_resource->get_cc_minor();
        HCTR_OWN_THROW(Error_t::WrongInput, os.str());
      }
      auto& input_mlp_tensor = input_output_info.input_tensors[0];
      auto& input_emb_tensor = input_output_info.input_tensors[1];
      core23::Tensor output_tensor, grad_tensor;
      if (input_output_info.output_names.size() == 2) {
        layers.emplace_back(new InteractionLayer<__half>(input_mlp_tensor, input_emb_tensor,
                                                         output_tensor, grad_tensor, gpu_resource,
                                                         use_mixed_precision, enable_tf32_compute));
        output_tensor_entities.push_back({input_output_info.output_names[0], output_tensor});
        output_tensor_entities.push_back({input_output_info.output_names[1], grad_tensor});
      } else {
        if (use_mixed_precision) {
          layers.emplace_back(
              new InteractionLayer<__half>(input_mlp_tensor, input_emb_tensor, output_tensor,
                                           gpu_resource, use_mixed_precision, enable_tf32_compute));
        } else {
          layers.emplace_back(
              new InteractionLayer<float>(input_mlp_tensor, input_emb_tensor, output_tensor,
                                          gpu_resource, use_mixed_precision, enable_tf32_compute));
        }
        output_tensor_entities.push_back({input_output_info.output_names[0], output_tensor});
      }
      break;
    }
    case Layer_t::MultiCross: {
      // currently, only default init type is supported
      std::vector<Initializer_t> initializer_types{dense_layer.weight_init_type,
                                                   dense_layer.bias_init_type};
      [[maybe_unused]] int num_layers = dense_layer.num_layers;
      [[maybe_unused]] int projection_dim = dense_layer.projection_dim;

      auto& mc_in_tensor = input_output_info.input_tensors[0];
      core23::Tensor out_tensor(tensor_params.shape(mc_in_tensor.shape()));
      output_tensor_entities.push_back({input_output_info.output_names[0], out_tensor});

      if (use_mixed_precision) {
        layers.emplace_back(new Core23TempMultiCrossLayer<__half>(
            mc_in_tensor, out_tensor, gpu_resource, num_layers, projection_dim, initializer_types,
            enable_tf32_compute));
      } else {
        layers.emplace_back(new Core23TempMultiCrossLayer<float>(
            mc_in_tensor, out_tensor, gpu_resource, num_layers, projection_dim, initializer_types,
            enable_tf32_compute));
      }
      break;
    }
    case Layer_t::MultiCrossEntropyLoss: {
      if (input_output_info.input_tensors.size() != 2) {
        HCTR_OWN_THROW(Error_t::WrongInput, "MultiCrossEntropyLoss must have two inputs");
      }

      if (input_output_info.input_tensors[0].shape() !=
          input_output_info.input_tensors[1].shape()) {
        std::string err_msg =
            "predition tensor and label tensor should have the same shape, got: " +
            input_output_info.input_tensors[0].shape().str() + " and " +
            input_output_info.input_tensors[1].shape().str();
        HCTR_OWN_THROW(Error_t::WrongInput, err_msg.c_str());
      }

      auto& input_tensor = input_output_info.input_tensors[0];
      auto& label_tensor = input_output_info.input_tensors[1];
      core23::Tensor loss_tensor(tensor_params.shape({1, 1}).data_type(core23::ScalarType::Float));

      std::unique_ptr<ILoss> new_loss;
      if (use_mixed_precision) {
        new_loss = create_loss<__half, MultiCrossEntropyLoss>(
            label_tensor, input_tensor, loss_tensor, dense_layer, layers, gpu_count_in_total,
            gpu_resource, scaler, solver.gen_loss_summary);
      } else {
        new_loss = create_loss<float, MultiCrossEntropyLoss>(
            label_tensor, input_tensor, loss_tensor, dense_layer, layers, gpu_count_in_total,
            gpu_resource, scaler, solver.gen_loss_summary);
      }

      std::string name = dense_layer.bottom_names[1];
      losses.insert(std::make_pair(name, std::move(new_loss)));
      break;
    }
    case Layer_t::ReLU: {
      auto& in_tensor = input_output_info.input_tensors[0];
      core23::Tensor out_tensor(tensor_params.shape(in_tensor.shape()));
      std::unique_ptr<Layer> layer;
      // TODO: remove if-else once the refactoring is done
      if (use_mixed_precision) {
        layer.reset(new ReluLayer<__half>(in_tensor, out_tensor, gpu_resource));
      } else {
        layer.reset(new ReluLayer<float>(in_tensor, out_tensor, gpu_resource));
      }
      layers.emplace_back(std::move(layer));
      output_tensor_entities.push_back({input_output_info.output_names[0], out_tensor});
      break;
    }
    case Layer_t::Reshape: {
      bool selected = dense_layer.selected;
      auto& input_tensor = input_output_info.input_tensors[0];
      core23::Tensor output_tensor;
      if (selected) {
        if (use_mixed_precision) {
          layers.emplace_back(new ReshapeLayer<__half>(input_tensor, output_tensor,
                                                       dense_layer.selected_slots, gpu_resource));
        } else {
          layers.emplace_back(new ReshapeLayer<float>(input_tensor, output_tensor,
                                                      dense_layer.selected_slots, gpu_resource));
        }
      } else {
        int64_t leading_dim = dense_layer.leading_dim;
        int64_t time_step = dense_layer.time_step;
        if (time_step == 0) {  // 2D output
          output_tensor = core23::Tensor(
              tensor_params.shape({input_tensor.num_elements() / leading_dim, leading_dim}));
        } else {  // 3D output
          int64_t batch_size = input_tensor.num_elements() / leading_dim / time_step;
          output_tensor = core23::Tensor(tensor_params.shape({batch_size, time_step, leading_dim}));
        }
        if (use_mixed_precision) {
          layers.emplace_back(new ReshapeLayer<__half>(input_tensor, output_tensor, gpu_resource));
        } else {
          layers.emplace_back(new ReshapeLayer<float>(input_tensor, output_tensor, gpu_resource));
        }
      }
      output_tensor_entities.push_back({input_output_info.output_names[0], output_tensor});
      break;
    }
    case Layer_t::Sigmoid: {
      auto& input_tensor = input_output_info.input_tensors[0];
      core23::Tensor output_tensor(tensor_params.shape(input_tensor.shape()));
      if (use_mixed_precision) {
        layers.emplace_back(new SigmoidLayer<__half>(input_tensor, output_tensor, gpu_resource));
      } else {
        layers.emplace_back(new SigmoidLayer<float>(input_tensor, output_tensor, gpu_resource));
      }
      output_tensor_entities.push_back({input_output_info.output_names[0], output_tensor});
      break;
    }
    case Layer_t::Slice: {
      [[maybe_unused]] auto& input_tensor = input_output_info.input_tensors[0];
      std::vector<core23::Tensor> output_tensors;
      if (use_mixed_precision) {
        layers.emplace_back(
            new SliceLayer<__half>(input_tensor, output_tensors, dense_layer.ranges, gpu_resource));
      } else {
        layers.emplace_back(
            new SliceLayer<float>(input_tensor, output_tensors, dense_layer.ranges, gpu_resource));
      }
      for (size_t i = 0; i < output_tensors.size(); i++) {
        output_tensor_entities.push_back({input_output_info.output_names[i], output_tensors[i]});
      }
      break;
    }
    case Layer_t::WeightMultiply: {
      auto& in_tensor = input_output_info.input_tensors[0];
      core23::Tensor out_tensor;
      std::vector<Initializer_t> initializer_types{dense_layer.weight_init_type};
      if (dense_layer.weight_dims.size() != 2) {
        HCTR_OWN_THROW(Error_t::WrongInput, "Only 2D weights is allowed for weight_multiply layer");
      }
      core23::Shape weight_dims{static_cast<int64_t>(dense_layer.weight_dims[0]),
                                static_cast<int64_t>(dense_layer.weight_dims[1])};
      if (use_mixed_precision) {
        layers.emplace_back(new Core23TempWeightMultiplyLayer<__half>(
            in_tensor, out_tensor, weight_dims, gpu_resource, initializer_types));
      } else {
        layers.emplace_back(new Core23TempWeightMultiplyLayer<float>(
            in_tensor, out_tensor, weight_dims, gpu_resource, initializer_types));
      }
      output_tensor_entities.push_back({input_output_info.output_names[0], in_tensor});
      break;
    }
    case Layer_t::FmOrder2: {
      int64_t out_dim = dense_layer.out_dim;
      auto& input_tensor = input_output_info.input_tensors[0];
      core23::Tensor output_tensor(tensor_params.shape({input_tensor.shape().size(0), out_dim}));
      if (use_mixed_precision) {
        layers.emplace_back(new FmOrder2Layer<__half>(input_tensor, output_tensor, gpu_resource));
      } else {
        layers.emplace_back(new FmOrder2Layer<float>(input_tensor, output_tensor, gpu_resource));
      }
      output_tensor_entities.push_back({input_output_info.output_names[0], output_tensor});
      break;
    }
    case Layer_t::Add: {
      auto& input_tensors = input_output_info.input_tensors;
      core23::Tensor output_tensor(tensor_params.shape(input_tensors[0].shape()));
      // TODO: fill the details including layers.emplace_back
      if (use_mixed_precision) {
        layers.emplace_back(new AddLayer<__half>(input_tensors, output_tensor, gpu_resource));
      } else {
        layers.emplace_back(new AddLayer<float>(input_tensors, output_tensor, gpu_resource));
      }
      output_tensor_entities.push_back({input_output_info.output_names[0], output_tensor});
      break;
    }
    case Layer_t::ReduceSum: {
      [[maybe_unused]] int axis = dense_layer.axis;
      [[maybe_unused]] auto& in_tensor = input_output_info.input_tensors[0];
      core23::Tensor out_tensor;
      if (use_mixed_precision) {
        layers.emplace_back(new ReduceSumLayer<__half>(in_tensor, out_tensor, axis, gpu_resource));
      } else {
        layers.emplace_back(new ReduceSumLayer<float>(in_tensor, out_tensor, axis, gpu_resource));
      }
      output_tensor_entities.push_back({input_output_info.output_names[0], out_tensor});
      break;
    }
    case Layer_t::ReduceMean: {
      [[maybe_unused]] int axis = dense_layer.axis;
      [[maybe_unused]] auto& in_tensor = input_output_info.input_tensors[0];
      core23::Tensor out_tensor;
      // TODO: fill the details including layers.emplace_back
      output_tensor_entities.push_back({input_output_info.output_names[0], out_tensor});
      break;
    }
    case Layer_t::Sub: {
      auto& in_tensors = input_output_info.input_tensors;
      core23::Tensor out_tensor(tensor_params.shape(in_tensors[0].shape()));
      // TODO: fill the details including layers.emplace_back
      output_tensor_entities.push_back({input_output_info.output_names[0], out_tensor});
      break;
    }
    case Layer_t::Gather: {
      [[maybe_unused]] auto& in_tensor = input_output_info.input_tensors[0];
      core23::Tensor out_tensor;
      // TODO: fill the details including layers.emplace_back
      output_tensor_entities.push_back({input_output_info.output_names[0], out_tensor});
      break;
    }
    case Layer_t::GRU: {
      std::vector<Initializer_t> initializer_types{dense_layer.weight_init_type,
                                                   dense_layer.bias_init_type};
      auto& in_tensor = input_output_info.input_tensors[0];
      int64_t num_output = dense_layer.num_output;
      core23::Tensor gru_out_tensor(tensor_params.shape({in_tensor.shape().size(0), num_output}));
      layers.emplace_back(new Core23TempGRULayer<float>(
          in_tensor, gru_out_tensor, dense_layer.num_output, dense_layer.batchsize,
          dense_layer.SeqLength, dense_layer.vector_size, gpu_resource, initializer_types));
      output_tensor_entities.push_back({input_output_info.output_names[0], gru_out_tensor});
      break;
    }
    case Layer_t::MatrixMultiply: {
      [[maybe_unused]] auto& in_tensors = input_output_info.input_tensors;
      core23::Tensor out_tensor;
      // TODO: fill the details including layers.emplace_back
      output_tensor_entities.push_back({input_output_info.output_names[0], out_tensor});
      break;
    }
    case Layer_t::Softmax: {
      if (input_output_info.input_tensors.size() != 2) {
        auto& in_tensor = input_output_info.input_tensors[0];
        core23::Tensor out_tensor(tensor_params.shape(in_tensor.shape()));
        // TODO: fill the details including layers.emplace_back
        output_tensor_entities.push_back({input_output_info.output_names[0], out_tensor});
      } else {
        HCTR_THROW_IF(use_mixed_precision, Error_t::IllegalCall,
                      "MaskedSoftMaxLayer is not available for the mixed precision mode");
        [[maybe_unused]] auto scale_factor = dense_layer.factor;
        auto& in_tensor = input_output_info.input_tensors[0];
        auto& mask_tensor = input_output_info.input_tensors[1];
        std::vector<core23::Tensor> in_tensors{in_tensor, mask_tensor};
        core23::Tensor out_tensor(tensor_params.shape(in_tensor.shape()));
        // TODO: fill the details including layers.emplace_back
        output_tensor_entities.push_back({input_output_info.output_names[0], out_tensor});
      }
      break;
    }
    case Layer_t::PReLU_Dice: {
      auto& in_tensor = input_output_info.input_tensors[0];
      core23::Tensor out_tensor(tensor_params.shape(in_tensor.shape()));
      output_tensor_entities.push_back({input_output_info.output_names[0], out_tensor});
      // TODO: fill the details including layers.emplace_back
      break;
    }
    case Layer_t::Scale: {
      [[maybe_unused]] auto& scale_in_tensor = input_output_info.input_tensors[0];
      core23::Tensor scale_out_tensor;
      // TODO: fill the details including layers.emplace_back
      output_tensor_entities.push_back({input_output_info.output_names[0], scale_out_tensor});
      break;
    }
    case Layer_t::FusedReshapeConcat: {
      [[maybe_unused]] auto& in_tensors = input_output_info.input_tensors;
      std::vector<core23::Tensor> out_tensors;
      // TODO: fill the details including layers.emplace_back
      for (size_t i = 0; i < out_tensors.size(); i++) {
        output_tensor_entities.push_back({input_output_info.output_names[i], out_tensors[i]});
      }
      break;
    }
    case Layer_t::FusedReshapeConcatGeneral: {
      [[maybe_unused]] auto& in_tensors = input_output_info.input_tensors;
      core23::Tensor out_tensor;
      // TODO: fill the details including layers.emplace_back
      output_tensor_entities.push_back({input_output_info.output_names[0], out_tensor});
      break;
    }
    case Layer_t::ElementwiseMultiply: {
      auto& in_tensors = input_output_info.input_tensors;
      core23::Tensor out_tensor(tensor_params.shape(in_tensors[0].shape()));
      // TODO: fill the details including layers.emplace_back
      output_tensor_entities.push_back({input_output_info.output_names[0], out_tensor});
      break;
    }
    default: {
      assert(!"Error: no such layer && should never get here!");
    }
  }  // end of switch
  if (!(layer_type == Layer_t::CrossEntropyLoss || layer_type == Layer_t::BinaryCrossEntropyLoss ||
        layer_type == Layer_t::MultiCrossEntropyLoss)) {
    for (auto& output_tensor_entity : output_tensor_entities) {
      tensor_entities.push_back(output_tensor_entity);
    }
    if (!embedding_dependent) {
      if (embedding_independent_layers) {
        // embedding_independent_layers->emplace_back(layers.back().get());
      }
    } else {
      if (embedding_dependent_layers) {
        // embedding_dependent_layers->emplace_back(layers.back().get());
      }
    }
  } else if (raw_metrics) {
    // Create new set of metrics and add to raw metrics map
    std::string name = dense_layer.bottom_names[1];

    const core23::Tensor& lookup_loss_tensor = losses.find(name)->second->get_loss_tensors()[0];

    metrics::Core23RawMetricMap new_map;
    new_map.insert(std::make_pair(metrics::RawType::Loss, lookup_loss_tensor));
    new_map.insert(std::make_pair(metrics::RawType::Pred, input_output_info.input_tensors[0]));
    new_map.insert(std::make_pair(metrics::RawType::Label, input_output_info.input_tensors[1]));
    raw_metrics->insert(std::make_pair(name, new_map));
  }
}

}  // namespace HugeCTR
