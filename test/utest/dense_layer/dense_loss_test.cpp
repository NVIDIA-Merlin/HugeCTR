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

#include <loss.hpp>
#include <utest/dense_layer/dense_layer_test_common.hpp>

namespace {

using namespace HugeCTR;

template <typename T, template <class M> class LossType>
void dense_loss_layer_test(int64_t width) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;
  int64_t batch_size = 1024;

  core23::TensorParams tensor_params =
      core23::TensorParams()
          .shape({batch_size, width})
          .device(core23::Device(core23::DeviceType::GPU, test::get_default_gpu()->get_device_id()))
          .data_type(core23::ScalarType::Float);

  core23::Tensor bottom_tensor(tensor_params);
  core23::Tensor label_tensor(tensor_params);
  TensorEntity bottom_tensor_entity = {"fc1", bottom_tensor};
  TensorEntity label_tensor_entity = {"label1", label_tensor};

  std::vector<TensorEntity> tensor_entities = {bottom_tensor_entity, label_tensor_entity};
  std::vector<std::string> bottom_names{bottom_tensor_entity.name, label_tensor_entity.name};
  std::vector<std::string> top_names = {"loss1"};

  Layer_t layer_type = std::is_same_v<LossType<T>, BinaryCrossEntropyLoss<T>>
                           ? Layer_t::BinaryCrossEntropyLoss
                           : std::is_same_v<LossType<T>, CrossEntropyLoss<T>>
                                 ? Layer_t::CrossEntropyLoss
                                 : Layer_t::MultiCrossEntropyLoss;

  DenseLayer dense_layer(layer_type, bottom_names, top_names);
  if (layer_type == Layer_t::MultiCrossEntropyLoss) {
    dense_layer.target_weight_vec = {0.5, 0.5};
  }
  test::dense_layer_test_common<LossType<T>>(tensor_entities, dense_layer, use_mixed_precision,
                                             false, false, 1024.f, false, false);
}

}  // namespace

TEST(test_dense_layer, bce_loss_layer_fp32) {
  dense_loss_layer_test<float, BinaryCrossEntropyLoss>(1);
}
TEST(test_dense_layer, bce_loss_layer_fp16) {
  dense_loss_layer_test<__half, BinaryCrossEntropyLoss>(1);
}
TEST(test_dense_layer, ce_loss_layer_fp32) { dense_loss_layer_test<float, CrossEntropyLoss>(2); }
TEST(test_dense_layer, mce_loss_layer_fp32) {
  dense_loss_layer_test<float, MultiCrossEntropyLoss>(2);
}