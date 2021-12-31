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
#include <pybind/model_perf_ext.hpp>
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

void ModelPerfExt::add_dense_layer_internal(
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
        auto regularizer = create_regularizer(
            dense_layer.use_regularizer, dense_layer.regularizer_type, dense_layer.lambda,
            weight_buff->as_tensor(), wgrad_buff_half->as_tensor(), in_tensor.get_dimensions()[0],
            gpu_resource);
        if (true == solver_.overlap_init_wgrad) {
          regularizer->set_overlapped();
        }
        loss.reset(new BinaryCrossEntropyLoss<__half>(
            label_tensor, in_tensor, loss_tensor, regularizer, gpu_resource, num_networks_in_global,
            scaler, solver_.gen_loss_summary));
      } else {
        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        auto regularizer = create_regularizer(dense_layer.use_regularizer,
                                              dense_layer.regularizer_type, dense_layer.lambda,
                                              weight_buff->as_tensor(), wgrad_buff->as_tensor(),
                                              in_tensor.get_dimensions()[0], gpu_resource);
        if (true == solver_.overlap_init_wgrad) {
          regularizer->set_overlapped();
        }
        loss.reset(new BinaryCrossEntropyLoss<float>(
            label_tensor, in_tensor, loss_tensor, regularizer, gpu_resource, num_networks_in_global,
            scaler, solver_.gen_loss_summary));
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
      if (pos_type == FcPosition_t::Head && skip_dgrad && input_size == 1 && output_size == 4) {
      } else if (pos_type == FcPosition_t::Head && !skip_dgrad && input_size == 2 &&
                 output_size == 4) {
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
        } else if (pos_type == FcPosition_t::Head && !skip_dgrad) {
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
      if (use_mixed_precision) {
        if (gpu_resource->get_cc_major() < 7) {
          CK_THROW_(Error_t::WrongInput, "InteractionLayer<__half> is not supported in SM " +
                                             std::to_string(gpu_resource->get_cc_major()) + "." +
                                             std::to_string(gpu_resource->get_cc_minor()));
        }
        if (input_output_info.inputs.size() != 2) {
          CK_THROW_(Error_t::WrongInput, "InteractionLayer<__half> needs two output tensors ");
        }
        Tensor2<__half> in_mlp_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        Tensor2<__half> in_emb_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[1]);
        Tensor2<__half> out_tensor, grad_tensor;
        layers.emplace_back(new InteractionLayer<__half>(in_mlp_tensor, in_emb_tensor, out_tensor,
                                                         grad_tensor, blobs_buff, gpu_resource,
                                                         use_mixed_precision, enable_tf32_compute));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
        output_tensor_entries.push_back({input_output_info.output_names[1], grad_tensor.shrink()});
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

void ModelPerfExt::add_dense_layer(DenseLayer& dense_layer) {
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

}  // namespace HugeCTR
