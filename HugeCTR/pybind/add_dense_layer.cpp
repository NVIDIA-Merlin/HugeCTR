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
#include <layers/weight_multiply_layer.hpp>
#include <layers/reduce_sum_layer.hpp>
#include <layers/relu_layer.hpp>
#include <layers/reshape_layer.hpp>
#include <layers/sigmoid_layer.hpp>
#include <layers/slice_layer.hpp>
#include <regularizers/l1_regularizer.hpp>
#include <regularizers/l2_regularizer.hpp>
#include <regularizers/no_regularizer.hpp>
#include <HugeCTR/pybind/model.hpp>

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
  std::vector<std::string>& bottom_names,
  std::vector<std::string>& top_names,
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
    bool use_regularizer, Regularizer_t regularizer_type, float lambda, const Tensor2<float>& weight_buff, const Tensor2<T>& wgrad_buff,
    const int batch_size, const std::shared_ptr<GPUResource>& gpu_resource) {
  std::shared_ptr<Regularizer<T>> reg(new NoRegularizer<T>(weight_buff, wgrad_buff, batch_size, gpu_resource));
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


void add_dense_layer_internal(DenseLayer& dense_layer,
                std::vector<TensorEntry>& tensor_entries,
                const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                const std::shared_ptr<BufferBlock2<float>>& weight_buff,
                const std::shared_ptr<BufferBlock2<__half>>& weight_buff_half,
                const std::shared_ptr<BufferBlock2<float>>& wgrad_buff,
                const std::shared_ptr<BufferBlock2<__half>>& wgrad_buff_half,
                Tensor2<float>& loss_tensor,
                std::vector<std::unique_ptr<Layer>>& layers,
                std::unique_ptr<ILoss>& loss,
                bool enable_cuda_graph,
                metrics::RawMetricMap* raw_metrics,
                int num_networks_in_global,
                const std::shared_ptr<GPUResource>& gpu_resource,
                bool use_mixed_precision,
                bool enable_tf32_compute,
                float scaler,
                bool use_algorithm_search) {
  Layer_t layer_type = dense_layer.layer_type;
  std::vector<TensorEntry> output_tensor_entries;
  auto input_output_info = get_input_tensor_and_output_name(dense_layer.bottom_names,
                                                            dense_layer.top_names,
                                                            tensor_entries);
  switch (layer_type) {
    case Layer_t::BatchNorm: {
      if (use_mixed_precision) {
        Tensor2<__half> bn_in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        // establish out tensor
        Tensor2<__half> bn_out_tensor;
        blobs_buff->reserve(bn_in_tensor.get_dimensions(), &bn_out_tensor);
        output_tensor_entries.push_back(
              {input_output_info.output_names[0], bn_out_tensor.shrink()});
        std::vector<Initializer_t> initializer_types{dense_layer.gamma_init_type, dense_layer.beta_init_type};
        BatchNormLayer<__half>::Params params = {dense_layer.factor, dense_layer.eps};
        layers.emplace_back(new BatchNormLayer<__half>(weight_buff, wgrad_buff, blobs_buff, bn_in_tensor,
                                                bn_out_tensor, params, gpu_resource,
                                                initializer_types));
      } else {
        Tensor2<float> bn_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        // establish out tensor
        Tensor2<float> bn_out_tensor;
        blobs_buff->reserve(bn_in_tensor.get_dimensions(), &bn_out_tensor);
        output_tensor_entries.push_back(
              {input_output_info.output_names[0], bn_out_tensor.shrink()});
        std::vector<Initializer_t> initializer_types{dense_layer.gamma_init_type, dense_layer.beta_init_type};
        BatchNormLayer<float>::Params params = {dense_layer.factor, dense_layer.eps};
        layers.emplace_back(new BatchNormLayer<float>(weight_buff, wgrad_buff, blobs_buff, bn_in_tensor,
                                                bn_out_tensor, params, gpu_resource,
                                                initializer_types));
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
            create_regularizer(dense_layer.use_regularizer,
                            dense_layer.regularizer_type, dense_layer.lambda,
                            weight_buff->as_tensor(), wgrad_buff_half->as_tensor(),
                            in_tensor.get_dimensions()[0], gpu_resource),
            gpu_resource, num_networks_in_global, scaler));
      } else {
        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        loss.reset(new BinaryCrossEntropyLoss<float>(
            label_tensor, in_tensor, loss_tensor,
            create_regularizer(dense_layer.use_regularizer,
                            dense_layer.regularizer_type, dense_layer.lambda,
                            weight_buff->as_tensor(), wgrad_buff->as_tensor(),
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
            create_regularizer(dense_layer.use_regularizer,
                              dense_layer.regularizer_type, dense_layer.lambda,
                              weight_buff->as_tensor(), wgrad_buff_half->as_tensor(),
                              cross_entropy_loss_in_tensor.get_dimensions()[0], gpu_resource),
            gpu_resource, num_networks_in_global, scaler));
      } else {
        Tensor2<float> cross_entropy_loss_in_tensor =
            Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        loss.reset(new CrossEntropyLoss<float>(
            label_tensor, cross_entropy_loss_in_tensor, loss_tensor,
            create_regularizer(dense_layer.use_regularizer,
                              dense_layer.regularizer_type, dense_layer.lambda,
                              weight_buff->as_tensor(), wgrad_buff->as_tensor(),
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
#ifndef PREFER_CUDNN
        layers.emplace_back(new DropoutLayer<__half>(do_in_tensor, do_out_tensor, blobs_buff,
                                                      rate, gpu_resource));
#else
        layers.emplace_back(new DropoutCudnnLayer<__half>(do_in_tensor, do_out_tensor, blobs_buff,
                                                          rate, gpu_resource));
#endif
      } else {
        Tensor2<float> do_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> do_out_tensor;
        blobs_buff->reserve(do_in_tensor.get_dimensions(), &do_out_tensor);
        output_tensor_entries.push_back(
            {input_output_info.output_names[0], do_out_tensor.shrink()});
        float rate = dense_layer.dropout_rate;
#ifndef PREFER_CUDNN
        layers.emplace_back(
            new DropoutLayer<float>(do_in_tensor, do_out_tensor, blobs_buff, rate, gpu_resource));
#else
        layers.emplace_back(new DropoutCudnnLayer<float>(do_in_tensor, do_out_tensor, blobs_buff,
                                                          rate, gpu_resource));
#endif
      }
      // to be fixed
      enable_cuda_graph = false;
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
        layers.emplace_back(new EluLayer<__half>(elu_in_tensor, elu_out_tensor, alpha, gpu_resource));
      } else {
        Tensor2<float> elu_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> elu_out_tensor;
        blobs_buff->reserve(elu_in_tensor.get_dimensions(), &elu_out_tensor);
        output_tensor_entries.push_back(
            {input_output_info.output_names[0], elu_out_tensor.shrink()});
        float alpha = dense_layer.elu_alpha;
        layers.emplace_back(new EluLayer<float>(elu_in_tensor, elu_out_tensor, alpha, gpu_resource));
      }
      break;
    }
    case Layer_t::FusedInnerProduct: {
      std::vector<Initializer_t> initializer_types{dense_layer.weight_init_type, dense_layer.bias_init_type};
      size_t output = dense_layer.num_output;
      if (use_mixed_precision) {
        Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        Tensor2<__half> fc_out_tensor;
        blobs_buff->reserve({(in_tensor.get_dimensions())[0], output}, &fc_out_tensor);
        output_tensor_entries.push_back(
            {input_output_info.output_names[0], fc_out_tensor.shrink()});
        layers.emplace_back(new FusedFullyConnectedLayer(
            weight_buff, weight_buff_half, wgrad_buff_half, blobs_buff, in_tensor, fc_out_tensor,
            gpu_resource, initializer_types));
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
      std::vector<Initializer_t> initializer_types{dense_layer.weight_init_type, dense_layer.bias_init_type};
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
        Tensor2<__half> in_mlp_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
        Tensor2<__half> in_emb_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[1]);
        Tensor2<__half> out_tensor;
        layers.emplace_back(
            new InteractionLayer<__half>(in_mlp_tensor, in_emb_tensor, out_tensor, blobs_buff,
                                        gpu_resource, use_mixed_precision, enable_tf32_compute));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      } else {
        Tensor2<float> in_mlp_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> in_emb_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[1]);
        Tensor2<float> out_tensor;
        layers.emplace_back(
            new InteractionLayer<float>(in_mlp_tensor, in_emb_tensor, out_tensor, blobs_buff,
                                        gpu_resource, use_mixed_precision, enable_tf32_compute));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      }
      break;
    }
    case Layer_t::MultiCross: {
      std::vector<Initializer_t> initializer_types{dense_layer.weight_init_type, dense_layer.bias_init_type};
      int num_layers = dense_layer.num_layers;
      Tensor2<float> mc_in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
      Tensor2<float> out_tensor;
      blobs_buff->reserve(mc_in_tensor.get_dimensions(), &out_tensor);
      output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      layers.emplace_back(new MultiCrossLayer(weight_buff, wgrad_buff, blobs_buff, mc_in_tensor,
                                              out_tensor, gpu_resource, num_layers, initializer_types));
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
            create_regularizer(dense_layer.use_regularizer,
                              dense_layer.regularizer_type, dense_layer.lambda,
                              weight_buff->as_tensor(), wgrad_buff_half->as_tensor(),
                              multi_cross_entropy_loss_in_tensor.get_dimensions()[0],
                              gpu_resource),
            dense_layer.target_weight_vec, gpu_resource, num_networks_in_global, scaler));
      } else {
        Tensor2<float> multi_cross_entropy_loss_in_tensor =
            Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        loss.reset(new MultiCrossEntropyLoss<float>(
            label_tensor, multi_cross_entropy_loss_in_tensor, loss_tensor,
            create_regularizer(dense_layer.use_regularizer,
                              dense_layer.regularizer_type, dense_layer.lambda,
                              weight_buff->as_tensor(), wgrad_buff->as_tensor(),
                              multi_cross_entropy_loss_in_tensor.get_dimensions()[0],
                              gpu_resource),
            dense_layer.target_weight_vec, gpu_resource, num_networks_in_global, scaler));
      }
      break;
    }
    case Layer_t::ReLU: {
      if (use_mixed_precision) {
        Tensor2<__half> relu_in_tensor =
            Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
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
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> out_tensor;
          layers.emplace_back(new ReshapeLayer<float>(in_tensor, out_tensor, blobs_buff,
                                                    dense_layer.selected_slots, gpu_resource));
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], out_tensor.shrink()});
        }
      }
      else {
        size_t leading_dim = dense_layer.leading_dim;
        if (use_mixed_precision) {
          Tensor2<__half> in_tensor = Tensor2<__half>::stretch_from(input_output_info.inputs[0]);
          Tensor2<__half> out_tensor;
          layers.emplace_back(new ReshapeLayer<__half>(in_tensor, out_tensor, blobs_buff,
                                                        leading_dim, gpu_resource));
          output_tensor_entries.push_back(
              {input_output_info.output_names[0], out_tensor.shrink()});
        } else {
          Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
          Tensor2<float> out_tensor;
          layers.emplace_back(new ReshapeLayer<float>(in_tensor, out_tensor, blobs_buff,
                                                      leading_dim, gpu_resource));
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
        layers.emplace_back(
            new SliceLayer<__half>(in_tensor, out_tensors, blobs_buff, dense_layer.ranges, gpu_resource));
        for (size_t i = 0; i < out_tensors.size(); i++) {
          output_tensor_entries.push_back(
              {input_output_info.output_names[i], out_tensors[i].shrink()});
        }
      } else {
        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensors2<float> out_tensors;
        layers.emplace_back(
            new SliceLayer<float>(in_tensor, out_tensors, blobs_buff, dense_layer.ranges, gpu_resource));
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
        layers.emplace_back(new WeightMultiplyLayer<__half>(weight_buff_half, wgrad_buff_half, blobs_buff, in_tensor,
                                              out_tensor, dense_layer.weight_dims, gpu_resource,
                                              initializer_types));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      } else {
        std::vector<Initializer_t> initializer_types{dense_layer.weight_init_type};
        Tensor2<float> in_tensor = Tensor2<float>::stretch_from(input_output_info.inputs[0]);
        Tensor2<float> out_tensor;
        layers.emplace_back(new WeightMultiplyLayer<float>(weight_buff, wgrad_buff, blobs_buff, in_tensor,
                                              out_tensor, dense_layer.weight_dims, gpu_resource,
                                              initializer_types));
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
        layers.emplace_back(
            new AddLayer<__half>(in_tensors, out_tensor, blobs_buff, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      } else {
        Tensors2<float> in_tensors;
        for (const auto& bag : input_output_info.inputs) {
          in_tensors.push_back(Tensor2<float>::stretch_from(bag));
        }
        Tensor2<float> out_tensor;
        blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
        layers.emplace_back(
            new AddLayer<float>(in_tensors, out_tensor, blobs_buff, gpu_resource));
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
    case Layer_t::DotProduct: {
      if (use_mixed_precision) {
        Tensors2<__half> in_tensors;
        for (const auto& bag : input_output_info.inputs) {
          in_tensors.push_back(Tensor2<__half>::stretch_from(bag));
        }
        Tensor2<__half> out_tensor;
        blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
        layers.emplace_back(new DotProductLayer<__half>(in_tensors, out_tensor, blobs_buff, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      } else {
        Tensors2<float> in_tensors;
        for (const auto& bag : input_output_info.inputs) {
          in_tensors.push_back(Tensor2<float>::stretch_from(bag));
        }
        Tensor2<float> out_tensor;
        blobs_buff->reserve(in_tensors[0].get_dimensions(), &out_tensor);
        layers.emplace_back(new DotProductLayer<float>(in_tensors, out_tensor, blobs_buff, gpu_resource));
        output_tensor_entries.push_back({input_output_info.output_names[0], out_tensor.shrink()});
      }
      break;
    }
    default: {
      assert(!"Error: no such layer && should never get here!");
    }
  } // end of switch
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
}

void add_dense_layer(DenseLayer& dense_layer,
                std::vector<std::vector<TensorEntry>>& train_tensor_entries_list,
                std::vector<std::vector<TensorEntry>>& evaluate_tensor_entries_list,
                const std::shared_ptr<ResourceManager>& resource_manager,
                bool use_mixed_precision,
                bool enable_tf32_compute,
                float scaler,
                bool use_algorithm_search,
                bool use_cuda_graph,
                std::vector<std::shared_ptr<Network>>& networks,
                std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>& blobs_buff_list,
                std::vector<std::shared_ptr<BufferBlock2<float>>>& train_weight_buff_list,
                std::vector<std::shared_ptr<BufferBlock2<__half>>>& train_weight_buff_half_list,
                std::vector<std::shared_ptr<BufferBlock2<float>>>& wgrad_buff_list,
                std::vector<std::shared_ptr<BufferBlock2<__half>>>& wgrad_buff_half_list, 
                std::vector<std::shared_ptr<BufferBlock2<float>>>& evaluate_weight_buff_list,
                std::vector<std::shared_ptr<BufferBlock2<__half>>>& evaluate_weight_buff_half_list,
                std::vector<std::shared_ptr<BufferBlock2<float>>>& wgrad_buff_placeholder_list,
                std::vector<std::shared_ptr<BufferBlock2<__half>>>& wgrad_buff_half_placeholder_list) {
  for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
    // add dense layer for train
    add_dense_layer_internal(dense_layer,
                train_tensor_entries_list[i],
                blobs_buff_list[i],
                train_weight_buff_list[i],
                train_weight_buff_half_list[i],
                wgrad_buff_list[i],
                wgrad_buff_half_list[i],
                networks[i]->train_loss_tensor_,
                networks[i]->train_layers_,
                networks[i]->train_loss_,
                networks[i]->enable_cuda_graph_,
                nullptr,
                resource_manager->get_global_gpu_count(),
                resource_manager->get_local_gpu(i),
                use_mixed_precision,
                enable_tf32_compute,
                scaler,
                use_algorithm_search);
    // add dense layer for evaluation
    add_dense_layer_internal(dense_layer,
                evaluate_tensor_entries_list[i],
                blobs_buff_list[i],
                evaluate_weight_buff_list[i],
                evaluate_weight_buff_half_list[i],
                wgrad_buff_placeholder_list[i],
                wgrad_buff_half_placeholder_list[i],
                networks[i]->evaluate_loss_tensor_,
                networks[i]->evaluate_layers_,
                networks[i]->evaluate_loss_,
                networks[i]->enable_cuda_graph_,
                &(networks[i]->raw_metrics_),
                resource_manager->get_global_gpu_count(),
                resource_manager->get_local_gpu(i),
                use_mixed_precision,
                enable_tf32_compute,
                scaler,
                use_algorithm_search);
  }
}

} // namespace HugeCTR