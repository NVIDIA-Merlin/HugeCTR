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

#include <gtest/gtest.h>

#include <core23/tensor_container.hpp>
#include <trainable_layer.hpp>
#include <utest/test_utils.hpp>

using namespace HugeCTR;

namespace {

/**
 * @brief
 * This is a dummy trainable layer class to demonstrate how to reserve weights and weight
 * gradients for trainable layers. The fprop and bprop methods are not actually implemented.
 */
template <typename DType, bool use_FP32_weight>
class DummyTrainableLayer : public Core23TempTrainableLayer<DType, use_FP32_weight> {
  using Base = Core23TempTrainableLayer<DType, use_FP32_weight>;
  using WeightType = typename Base::WeightType;

  std::vector<core23::Tensor> in_tensors_;
  std::vector<core23::Tensor> out_tensors_;

 public:
  DummyTrainableLayer(const core23::Tensor& in_tensor, const core23::Tensor& out_tensor,
                      const std::shared_ptr<GPUResource>& gpu_resource,
                      std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>())
      : Base(gpu_resource, initializer_types) {
    const auto& in_tensor_dim = in_tensor.shape();
    const auto& out_tensor_dim = out_tensor.shape();

    if (in_tensor_dim.dims() != 2 || in_tensor_dim.dims() != 2 ||
        in_tensor_dim.size(0) != out_tensor_dim.size(0)) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Wrong dimensions for input and output tensors");
    }

    int64_t input_size = in_tensor_dim.size(in_tensor_dim.dims() - 1);
    int64_t output_size = out_tensor_dim.size(out_tensor_dim.dims() - 1);

    core23::Shape dim0 = {input_size, output_size};
    core23::Shape dim1 = {1, output_size};

    this->set_weight(0, dim0);
    this->set_weight(1, dim1);
    this->set_wgrad(0, dim0);
    this->set_wgrad(1, dim1);

    in_tensors_.push_back(in_tensor);
    out_tensors_.push_back(out_tensor);
  }

  void fprop(bool is_train) override {
    auto weight_0 = this->get_weight(0);
    auto weight_1 = this->get_weight(1);
  };

  void bprop() override {
    auto wgrad_0 = this->get_wgrad(0);
    auto wgrad_1 = this->get_wgrad(1);
  };
};

template <typename DType, bool use_FP32_weight>
void trainable_layer_test(int64_t batch_size, int64_t in_dim, int64_t out_dim) {
  core23::BufferParams blobs_buffer_params = {};
  blobs_buffer_params.channel = GetBlobsBufferChannel();

  core23::Tensor input_tensor = core23::Tensor(core23::TensorParams()
                                                   .data_type(core23::ToScalarType<DType>::value)
                                                   .shape({batch_size, in_dim})
                                                   .buffer_params(blobs_buffer_params));

  core23::Tensor output_tensor = core23::Tensor(core23::TensorParams()
                                                    .data_type(core23::ToScalarType<DType>::value)
                                                    .shape({batch_size, out_dim})
                                                    .buffer_params(blobs_buffer_params));

  if constexpr (std::is_same<DType, float>::value || use_FP32_weight) {
    DummyTrainableLayer<DType, use_FP32_weight> dummy_trainable_layer(input_tensor, output_tensor,
                                                                      test::get_default_gpu());
    dummy_trainable_layer.initialize();
    dummy_trainable_layer.fprop(true);
    dummy_trainable_layer.bprop();

    auto weights = dummy_trainable_layer.get_weights();
    auto wgrads = dummy_trainable_layer.get_wgrads();

    core23::TensorContainer<float, 1, 1> weights_container(std::move(weights),
                                                           {static_cast<int64_t>(weights.size())});
    core23::TensorContainer<float, 1, 1> wgrads_container(std::move(wgrads),
                                                          {static_cast<int64_t>(wgrads.size())});

    ASSERT_TRUE(weights_container.flatten().size(0) == wgrads_container.flatten().size(0));
  } else {
    DummyTrainableLayer<DType, use_FP32_weight> dummy_trainable_layer(input_tensor, output_tensor,
                                                                      test::get_default_gpu());
    dummy_trainable_layer.initialize();
    dummy_trainable_layer.fprop(true);
    dummy_trainable_layer.bprop();

    auto master_weights = dummy_trainable_layer.get_master_weights();
    auto weights = dummy_trainable_layer.get_weights();
    auto wgrads = dummy_trainable_layer.get_wgrads();

    core23::TensorContainer<float, 1, 1> master_weights_container(
        std::move(master_weights), {static_cast<int64_t>(master_weights.size())});
    core23::TensorContainer<__half, 1, 1> weights_container(std::move(weights),
                                                            {static_cast<int64_t>(weights.size())});
    core23::TensorContainer<__half, 1, 1> wgrads_container(std::move(wgrads),
                                                           {static_cast<int64_t>(wgrads.size())});
    ASSERT_TRUE(master_weights_container.flatten().size(0) == weights_container.flatten().size(0));
    ASSERT_TRUE(master_weights_container.flatten().size(0) == wgrads_container.flatten().size(0));
  }
}

}  // namespace

TEST(trainable_layer, fp32_32_4096x1024x10) { trainable_layer_test<float, true>(4096, 1024, 10); }
// TEST(trainable_layer, fp32_16_4096x1024x10) { trainable_layer_test<float, false>(4096, 1024, 10);
// }  // error
TEST(trainable_layer, fp16_16_4096x1024x10) { trainable_layer_test<__half, false>(4096, 1024, 10); }
TEST(trainable_layer, fp16_32_4096x1024x10) { trainable_layer_test<__half, true>(4096, 1024, 10); }
