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

#include <layers/interaction_layer.hpp>
#include <utest/dense_layer/dense_layer_test_common.hpp>

namespace {

using namespace HugeCTR;

template <typename T>
void dense_interaction_layer_test(bool grad) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;
  int64_t batch_size = 1024;
  int64_t width = 128;
  int64_t n_emb = 26;

  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(core23::Device(core23::DeviceType::GPU, test::get_default_gpu()->get_device_id()))
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float);

  std::vector<TensorEntity> tensor_entities;
  std::vector<std::string> bottom_names;
  tensor_entities.push_back({"mlp", tensor_params.shape({batch_size, width})});
  bottom_names.push_back(tensor_entities.back().name);
  tensor_entities.push_back({"emb", tensor_params.shape({batch_size, n_emb, width})});
  bottom_names.push_back(tensor_entities.back().name);

  Layer_t layer_type = Layer_t::Interaction;
  std::vector<std::string> top_names = {"interaction1"};
  if (grad) {
    top_names.push_back("interaction1_grad");
  }

  DenseLayer dense_layer(layer_type, bottom_names, top_names);
  test::dense_layer_test_common<InteractionLayer<T>>(
      tensor_entities, dense_layer, use_mixed_precision, false, false, 1024.f, false, false);
}

}  // namespace

TEST(test_dense_layer, interaction_layer_fp32) { dense_interaction_layer_test<float>(false); }
TEST(test_dense_layer, interaction_layer_fp16) { dense_interaction_layer_test<__half>(false); }
TEST(test_dense_layer, interaction_layer_fp16_grad) { dense_interaction_layer_test<__half>(true); }