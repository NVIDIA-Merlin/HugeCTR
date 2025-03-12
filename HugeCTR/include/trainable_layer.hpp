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
#pragma once

#include <cassert>
#include <data_simulator.hpp>
#include <layer.hpp>
#include <network_buffer_channels.hpp>
#include <type_traits>

namespace HugeCTR {

/**
 * @brief
 * Trainable layer is the common parent of all layers with weights
 * @tparams DType the data type of inputs, outputs, and weights
 * @tparams use_FP32_weight if specified, the weight data type is in FP32, not DType
 */
template <typename DType, bool use_FP32_weight = std::is_same<DType, float>::value>
class TrainableLayer : public Layer {
  // FP32 input/output but lower precision weight don't make much sense.
  static_assert(!(std::is_same<DType, float>::value && use_FP32_weight == false));

 protected:
  // Why WeightType is protected?
  // it is convenient for a child trainable to access the weight type,
  // especially if it wants to use FP32 weights but inputs/outputs the lower precision data.
  // A typical example is when DType is __half but use_FP32_weight is true.
  // Then, the child class should define the following alias to make their code cleaner:
  // (1) using Base = TrainableLayer<DType, true>;
  // (2) using WeightType = typename Base::WeightType;
  // If  useFP32_weight is false, the aliases are not necessary.
  using WeightType = typename std::conditional<use_FP32_weight, float, DType>::type;

 private:
  std::vector<core23::Tensor> master_weights_;
  core23::TensorParams master_weights_params_;
  std::vector<core23::Tensor> weights_;
  core23::TensorParams weights_params_;
  std::vector<core23::Tensor> wgrads_;
  core23::TensorParams wgrads_params_;
  core23::Device device_;
  // Layer initializers.
  // if the layer need a specific weight initialization, override each function accordingly.
  virtual std::unique_ptr<DataSimulator> get_zero_initializer(const int index) {
    return std::make_unique<ConstantDataSimulator>(0.0f);
  }
  virtual std::unique_ptr<DataSimulator> get_uniform_initializer(const int index) {
    return get_default_initializer(index);
  };
  virtual std::unique_ptr<DataSimulator> get_xavier_uniform_initializer(const int index) {
    return get_default_initializer(index);
  };
  virtual std::unique_ptr<DataSimulator> get_xavier_norm_initializer(const int index) {
    return get_default_initializer(index);
  };
  virtual std::unique_ptr<DataSimulator> get_default_initializer(const int index) {
    return get_zero_initializer(index);
  };

 protected:
  // @brief a modifier to reserve a weight tensor at idx with the specified dims.
  // @details
  // Usage: In a child class, this->set_weight(0, dims);
  void set_weight(size_t idx, const core23::Shape& dimensions) {
    HCTR_CHECK_HINT(weights_.size() == idx, "Wrong index for setting weight tensors");

    weights_.emplace_back(weights_params_.data_type(core23::ToScalarType<WeightType>::value)
                              .device(device_)
                              .shape(dimensions));

    // master weights are used only when compute weights have lower precision
    if constexpr (!use_FP32_weight) {
      HCTR_CHECK_HINT(master_weights_.size() == idx,
                      "Wrong index for setting master weight tensors");

      master_weights_.emplace_back(master_weights_params_.data_type(core23::ScalarType::Float)
                                       .device(device_)
                                       .shape(dimensions));
    }
  }
  // @brief a modifier to reserve a weight tensor at idx with the specified dims.
  // @details
  // Usage: In a child class, this->set_wgrad(0, dims);
  void set_wgrad(size_t idx, const core23::Shape& dimensions) {
    HCTR_CHECK_HINT(wgrads_.size() == idx, "Wrong index for setting weight gradient tensors");

    wgrads_.emplace_back(wgrads_params_.data_type(core23::ToScalarType<WeightType>::value)
                             .device(device_)
                             .shape(dimensions));
  }
  // @brief an accessor to get a weight tensor at idx
  // @details
  // Usage: In a child class, auto weight2 = this->get_weight(2);
  auto& get_weight(size_t idx) {
    HCTR_CHECK_HINT(idx < weights_.size(), "Wrong index for getting weight tensors");
    return weights_[idx];
  }
  // @brief an accessor to get a wgrad tensor at idx
  // @details
  // Usage: In a child class, auto wgrad2 = this->get_wgrad(2);
  auto& get_wgrad(size_t idx) {
    HCTR_CHECK_HINT(idx < wgrads_.size(), "Wrong index for getting weight gradient tensors");
    return wgrads_[idx];
  }
  using Layer::input_tensors_;
  using Layer::output_tensors_;

 public:
  // @brief a parameter initialization function
  // @details
  // init_params calls the specific initializers to initialize parameters. The types of initializers
  // are specified by initializer_types_.
  void init_params(const curandGenerator_t& generator) override;

  /**
   * Ctor of TrainableLayer.
   * @param gpu_resource the abstraction of GPU where this dense layer resides
   * @param initializer_types the list of initializer types of all weight tensors
   */
  TrainableLayer(const std::vector<core23::Tensor>& input_tensors,
                 const std::vector<core23::Tensor>& output_tensors,
                 const std::shared_ptr<GPUResource>& gpu_resource,
                 std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>())
      : Layer(input_tensors, output_tensors, gpu_resource, initializer_types),
        master_weights_params_(core23::TensorParams()
                                   .alignment(sizeof(float))
                                   .buffer_channel(GetWeightBufferChannel())),
        weights_params_(core23::TensorParams()
                            .alignment(sizeof(WeightType))
                            .buffer_channel(std::is_same<WeightType, float>::value
                                                ? GetWeightBufferChannel()
                                                : GetWeightHalfBufferChannel())),
        wgrads_params_(core23::TensorParams()
                           .alignment(sizeof(WeightType))
                           .buffer_channel(std::is_same<WeightType, float>::value
                                               ? GetWgradBufferChannel()
                                               : GetWgradHalfBufferChannel())),
        device_(core23::DeviceType::GPU, gpu_resource->get_device_id()) {
    static_assert(std::is_same_v<WeightType, float> || std::is_same_v<WeightType, __half>);
  }

  std::vector<core23::Tensor> get_master_weights() { return master_weights_; };
  std::vector<core23::Tensor> get_weights() { return weights_; };
  std::vector<core23::Tensor> get_wgrads() { return wgrads_; };
};

template <typename DType, bool use_FP32_weight>
void TrainableLayer<DType, use_FP32_weight>::init_params(const curandGenerator_t& generator) {
  std::vector<core23::Tensor> weights = master_weights_;
  if constexpr (std::is_same<DType, float>::value && use_FP32_weight) {
    weights = weights_;
  }

  std::vector<core23::Tensor> weight_cpu_tensors;
  core23::AllocatorParams alloc_params{.pinned = true};
  for (const core23::Tensor& weight : weights) {
    core23::TensorParams params =
        weight.my_params().device(core23::DeviceType::CPU).allocator_params(alloc_params);
    weight_cpu_tensors.emplace_back(params);
  }

  std::vector<std::unique_ptr<DataSimulator>> simulators;
  // each weight has its own initializer
  for (int index = 0; index < static_cast<int>(weights.size()); ++index) {
    switch (initializer_types_[index % initializer_types_.size()]) {
      case Initializer_t::Uniform: {
        simulators.push_back(get_uniform_initializer(index));
        break;
      }
      case Initializer_t::XavierNorm: {
        simulators.push_back(get_xavier_norm_initializer(index));
        break;
      }
      case Initializer_t::XavierUniform: {
        simulators.push_back(get_xavier_uniform_initializer(index));
        break;
      }
      case Initializer_t::Zero: {
        simulators.push_back(get_zero_initializer(index));
        break;
      }
      case Initializer_t::Default: {
        simulators.push_back(get_default_initializer(index));
        break;
      }
      default: {
        HCTR_OWN_THROW(Error_t::OutOfBound, "Not supported initializer.");
        break;
      }
    }
  }

  for (size_t i = 0; i < weights.size(); ++i) {
    simulators[i]->fill(weight_cpu_tensors[i], generator);
    HCTR_LIB_THROW(cudaMemcpyAsync(weights[i].data(), weight_cpu_tensors[i].data(),
                                   weights[i].num_bytes(), cudaMemcpyHostToDevice,
                                   get_gpu().get_stream()));
  }
}

}  // namespace HugeCTR
