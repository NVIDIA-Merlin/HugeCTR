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
class DummyTrainableLayer : public TrainableLayer<DType, use_FP32_weight> {
  using Base = TrainableLayer<DType, use_FP32_weight>;
  using WeightType = typename Base::WeightType;

  Tensors2<DType> in_tensors_;

  Tensors2<DType> out_tensors_;

 public:
  DummyTrainableLayer(const std::shared_ptr<BufferBlock2<float>>& master_weight_buff,
                      const std::shared_ptr<BufferBlock2<WeightType>>& weight_buff,
                      const std::shared_ptr<BufferBlock2<WeightType>>& wgrad_buff,
                      const Tensor2<DType>& in_tensor, const Tensor2<DType>& out_tensor,
                      const std::shared_ptr<GPUResource>& gpu_resource,
                      std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>())
      : Base(master_weight_buff, weight_buff, wgrad_buff, gpu_resource, initializer_types) {
    const auto& in_tensor_dim = in_tensor.get_dimensions();
    const auto& out_tensor_dim = out_tensor.get_dimensions();

    if (in_tensor_dim.size() != 2 || in_tensor_dim.size() != 2 ||
        in_tensor_dim[0] != out_tensor_dim[0]) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Wrong dimensions for input and output tensors");
    }

    size_t input_size = in_tensor_dim[in_tensor_dim.size() - 1];
    size_t output_size = out_tensor_dim[out_tensor_dim.size() - 1];

    std::vector<size_t> dim0 = {input_size, output_size};
    std::vector<size_t> dim1 = {1, output_size};

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
void trainable_layer_test(size_t batch_size, size_t in_dim, size_t out_dim) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();

  using WeightType = typename std::conditional<use_FP32_weight, float, DType>::type;

  std::shared_ptr<BufferBlock2<WeightType>> weight_buff = buf->create_block<WeightType>();
  std::shared_ptr<BufferBlock2<WeightType>> wgrad_buff = buf->create_block<WeightType>();

  Tensor2<DType> input_tensor;
  buf->reserve({batch_size, in_dim}, &input_tensor);
  Tensor2<DType> output_tensor;
  buf->reserve({batch_size, out_dim}, &output_tensor);

  auto test_impl = [&buf, &weight_buff, &wgrad_buff, &input_tensor,
                    &output_tensor](auto master_weight_buff) {
    DummyTrainableLayer<DType, use_FP32_weight> dummy_trainable_layer(
        master_weight_buff, weight_buff, wgrad_buff, input_tensor, output_tensor,
        test::get_default_gpu());
    buf->allocate();
    dummy_trainable_layer.initialize();
    dummy_trainable_layer.fprop(true);
    dummy_trainable_layer.bprop();
  };

  if constexpr (std::is_same<DType, float>::value || use_FP32_weight) {
    test_impl(weight_buff);

    ASSERT_TRUE(weight_buff->as_tensor().get_num_elements() ==
                wgrad_buff->as_tensor().get_num_elements());
  } else {
    std::shared_ptr<BufferBlock2<float>> master_weight_buff = buf->create_block<float>();
    test_impl(master_weight_buff);

    ASSERT_TRUE(master_weight_buff->as_tensor().get_num_elements() ==
                weight_buff->as_tensor().get_num_elements());
    ASSERT_TRUE(master_weight_buff->as_tensor().get_num_elements() ==
                wgrad_buff->as_tensor().get_num_elements());
  }
}

}  // namespace

TEST(trainable_layer, fp32_32_4096x1024x10) { trainable_layer_test<float, true>(4096, 1024, 10); }
// TEST(trainable_layer, fp32_16_4096x1024x10) { trainable_layer_test<float, false>(4096, 1024, 10);
// }  // error
TEST(trainable_layer, fp16_16_4096x1024x10) { trainable_layer_test<__half, false>(4096, 1024, 10); }
TEST(trainable_layer, fp16_32_4096x1024x10) { trainable_layer_test<__half, true>(4096, 1024, 10); }
