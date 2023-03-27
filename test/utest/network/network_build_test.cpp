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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <omp.h>

#include <core23/logger.hpp>
#include <core23/low_level_primitives.hpp>
#include <core23/tensor.hpp>
#include <core23_network.hpp>
#include <layers/fully_connected_layer.hpp>
#include <layers/fully_connected_layer_half.hpp>
#include <layers/relu_layer.hpp>
#include <loss.hpp>
#include <network_helpers.hpp>
#include <optimizer.hpp>
#include <random>
#include <regularizer_factory.hpp>
#include <resource_managers/resource_manager_ext.hpp>
#include <tuple>
#include <utest/test_utils.hpp>

namespace {

using namespace HugeCTR;

template <typename T>
std::tuple<std::vector<std::unique_ptr<Layer>>, std::map<std::string, std::unique_ptr<ILoss>>>
add_layers(std::shared_ptr<GPUResource> gpu, core23::Tensor& label_tensor, core23::Tensor& tensor0,
           int64_t batch_size, int64_t width, float scaler) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;
  bool enable_tf32_compute = false;
  int total_gpu_count = 1;
  bool gen_loss_summary = true;

  std::vector<core23::Tensor> tensors(1, tensor0);
  std::vector<std::unique_ptr<Layer>> train_layers;
  for (int64_t i = 1; i <= 5; i++) {
    auto bottom_tensor = tensors.back();
    if (i % 2 == 1) {
      tensors.emplace_back(
          bottom_tensor.my_params().shape({bottom_tensor.size(0), bottom_tensor.size(1) / 8}));
    } else {
      tensors.emplace_back(bottom_tensor.my_params());
    }
  }
  core23::Tensor loss_tensor(tensors[5].my_params().shape({1, 1}));

  for (int64_t i = 1; i <= 5; i++) {
    if (i % 2 == 1) {
      train_layers.emplace_back(new Core23TempFullyConnectedLayer<float>(
          tensors[i - 1], tensors[i], gpu, use_mixed_precision, enable_tf32_compute,
          {Initializer_t::Default, Initializer_t::Default}));
    } else {
      train_layers.emplace_back(new ReluLayer<float>(tensors[i - 1], tensors[i], gpu));
    }
  }

  auto weight_tensors = get_trainable_tensor_vector<float>(
      train_layers, [](auto& layer) -> auto { return layer->get_weights(); });
  auto wgrad_tensors = get_trainable_tensor_vector<float>(
      train_layers, [](auto& layer) -> auto { return layer->get_wgrads(); });

  auto regularizer = create_regularizer<float>(false, Regularizer_t::None, 0.01f, weight_tensors,
                                               wgrad_tensors, batch_size, gpu);

  std::map<std::string, std::unique_ptr<ILoss>> losses;
  losses.insert({"loss0", std::make_unique<BinaryCrossEntropyLoss<float>>(
                              label_tensor, tensors.back(), loss_tensor, regularizer, gpu,
                              total_gpu_count, scaler, gen_loss_summary)});

  return {std::move(train_layers), std::move(losses)};
}

void initialize_label_and_input(core23::Tensor& label_tensor, core23::Tensor& input_tensor) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, 1);
  std::vector<int> h_label(label_tensor.num_elements());
  for (size_t i = 0; i < h_label.size(); i++) {
    h_label[i] = dist(gen);
  }

  CudaDeviceContext context(label_tensor.device().index());
  core23::CUDAStream stream;
  core23::convert_async<float, int>(label_tensor.data<float>(), h_label.data(),
                                    label_tensor.num_elements(), label_tensor.device(),
                                    core23::DeviceType::CPU, stream);
  core23::CURANDGenerator generator(input_tensor.device());
  core23::normal_async<float>(input_tensor.data<float>(), input_tensor.num_elements(), 0.f, 1.f,
                              input_tensor.device(), generator, stream);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream()));
}

template <typename T>
void network_build_test() {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;
  float scaler = 1024.f;

  int64_t batch_size = 1024;
  int64_t width = 512;

  std::vector<int> device_vec(core23::Device::count());
  std::generate(device_vec.begin(), device_vec.end(), [dev = 0]() mutable { return dev++; });
  std::vector<std::vector<int>> vvgpu(1, device_vec);
  const auto& resource_manager = ResourceManagerExt::create(vvgpu, 0);

  std::vector<std::shared_ptr<Core23TempNetwork>> networks;
  std::vector<std::pair<core23::Tensor, core23::Tensor>> train_label_and_first_tensors;
  std::vector<std::pair<core23::Tensor, core23::Tensor>> evaluate_label_and_first_tensors;
  for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
    auto& gpu = resource_manager->get_local_gpu(i);
    networks.emplace_back(
        new Core23TempNetwork(resource_manager->get_local_cpu(), gpu, use_mixed_precision));

    auto network = networks.back();

    std::random_device rd;

    core23::TensorParams tensor_params =
        core23::TensorParams()
            .device(core23::Device(core23::DeviceType::GPU, gpu->get_device_id()))
            .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float);

    core23::Tensor train_label_tensor(tensor_params.shape({batch_size, 1}));
    core23::Tensor train_tensor0(train_label_tensor.my_params().shape({batch_size, width}));
    train_label_and_first_tensors.push_back({train_label_tensor, train_tensor0});
    auto [train_layers, train_losses] =
        add_layers<T>(gpu, train_label_tensor, train_tensor0, batch_size, width, scaler);

    core23::Tensor evaluate_label_tensor(tensor_params.shape({batch_size, 1}));
    core23::Tensor evaluate_tensor0(evaluate_label_tensor.my_params().shape({batch_size, width}));
    evaluate_label_and_first_tensors.push_back({evaluate_label_tensor, evaluate_tensor0});
    auto [evaluate_layers, evaluate_losses] =
        add_layers<T>(gpu, evaluate_label_tensor, evaluate_tensor0, batch_size, width, scaler);

    auto weight_tensors = get_trainable_tensor_vector<float>(
        train_layers, [](auto& layer) -> auto { return layer->get_weights(); });
    auto weight_half_tensors = std::vector<core23::Tensor>();
    auto wgrad_tensors = get_trainable_tensor_vector<float>(
        train_layers, [](auto& layer) -> auto { return layer->get_wgrads(); });

    std::map<std::string, float> label_weights;
    for (auto& pair : train_losses) {
      label_weights.insert({pair.first, 1.f});
    }

    network->set_train_layers(std::move(train_layers));
    network->set_train_losses(std::move(train_losses), label_weights);

    network->set_evaluate_layers(std::move(evaluate_layers));
    network->set_evaluate_losses(std::move(evaluate_losses), label_weights);

    OptParams opt_params;
    opt_params.lr = 0.001;
    opt_params.optimizer = Optimizer_t::Adam;
    opt_params.update_type = Update_t::Global;
    opt_params.scaler = scaler;
    opt_params.hyperparams.adam.beta1 = 0.9;
    opt_params.hyperparams.adam.beta2 = 0.999;
    opt_params.hyperparams.adam.epsilon = 0.0000001;
    auto optimizer = Optimizer::Create<float>(opt_params, weight_tensors, weight_half_tensors,
                                              wgrad_tensors, scaler, gpu, use_mixed_precision);
    network->set_optimizer(std::move(optimizer));
  }

#pragma omp parallel num_threads(networks.size())
  {
    int64_t id = omp_get_thread_num();
    auto& network = networks[id];
    CudaDeviceContext(network->get_device_id());
    auto [train_label_tensor, train_tensor0] = train_label_and_first_tensors[id];
    initialize_label_and_input(train_label_tensor, train_tensor0);
    auto [evaluate_label_tensor, evaluate_tensor0] = evaluate_label_and_first_tensors[id];
    initialize_label_and_input(evaluate_label_tensor, evaluate_tensor0);

    network->initialize(true);
    network->init_params(id);
    cudaStreamSynchronize(resource_manager->get_local_gpu(id)->get_stream());
    id++;
  }

  for (int64_t train_iter = 0; train_iter < 8192; train_iter++) {
#pragma omp parallel num_threads(networks.size())
    {
      int64_t id = omp_get_thread_num();
      auto& network = networks[id];
      CudaDeviceContext context(network->get_device_id());
      network->train(batch_size);
    }
#pragma omp parallel num_threads(networks.size())
    {
      int64_t id = omp_get_thread_num();
      auto& network = networks[id];
      CudaDeviceContext context(network->get_device_id());
      network->exchange_wgrad();
      network->update_params();
    }
    if (train_iter % 128 == 127) {
      float train_loss = 0.f;
      for (auto& network : networks) {
        train_loss += network->get_loss();
      }
      train_loss /= resource_manager->get_local_gpu_count();
      HCTR_LOG_S(INFO, ROOT) << "training loss: " << train_loss << " at " << train_iter
                             << std::endl;

      for (auto& network : networks) {
        CudaDeviceContext context(network->get_device_id());
        network->copy_weights_from_train_layers_to_evaluate_layers();
        network->copy_non_trainable_params_from_train_layers_to_evaluate_layers();
        network->eval(batch_size);
      }
    }
  }
}

}  // namespace

TEST(test_network, network_build_fp32) { network_build_test<float>(); }
