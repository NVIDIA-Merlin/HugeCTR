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

#include <core23/data_type_helpers.cuh>
#include <core23/low_level_primitives.hpp>
#include <core23/shape.hpp>
#include <core23/tensor.hpp>
#include <layers/concat_3d_layer.hpp>
#include <memory>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

template <typename T>
struct Eps {
  static T value();
};

template <>
struct Eps<float> {
  static constexpr float value() { return 1e-6f; }
};

template <typename T>
void python_concat(T *output, T **inputs, int64_t batch_size, const std::vector<int64_t> &slot_num,
                   const std::vector<int64_t> &vec_size, int axis) {
  std::cout << "===========Python concat==========" << std::endl;
  auto num = vec_size.size();
  std::string temp_name = "tmpdata.bin";
  std::ofstream py_input(temp_name.c_str(), std::ios::binary | std::ios::out);
  char *input_ptr = nullptr;
  std::vector<T> result;

  for (size_t i = 0; i < vec_size.size(); i++) {
    std::cout << batch_size << " " << slot_num[i] << " " << vec_size[i] << std::endl;
    py_input.write(reinterpret_cast<const char *>(&batch_size), sizeof(int));
    py_input.write(reinterpret_cast<const char *>(&slot_num[i]), sizeof(int));
    py_input.write(reinterpret_cast<const char *>(&vec_size[i]), sizeof(int));
    input_ptr = (char *)inputs[i];
    py_input.write(input_ptr, batch_size * slot_num[i] * vec_size[i] * sizeof(float));
  }
  py_input.close();

  std::stringstream command;

  command << "python3 python_concat.py " << num << " " << 3 << " " << axis << " "
          << temp_name;  // test 3-d inputs
  auto py_output = popen(command.str().c_str(), "r");
  // int dummy = fscanf(py_output, "%f", &result);
  T dummy = 0;
  size_t idx = 0;
  while (fscanf(py_output, "%*c%f", &dummy) != EOF) {
    output[idx] = dummy;
    idx++;
  }

  std::cout << result.size() << std::endl;

  pclose(py_output);
}

template <typename T>
void concat_3d_layer_test(int64_t batch_size, std::vector<int64_t> slot_num,
                          std::vector<int64_t> vec_size, int axis) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);
  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(core23::GetRandomBufferChannel());

  std::vector<core23::Tensor> bottom_tensors;
  auto blob_buffer_channel = core23::GetRandomBufferChannel();

  int64_t n_ins = vec_size.size();
  std::unique_ptr<T *[]> h_ins(new T *[n_ins]);

  int64_t out_slot_num = 0;
  int64_t out_vector_size = 0;
  std::cout << "Num of Input: " << n_ins << std::endl;

  for (int64_t i = 0; i < n_ins; i++) {
    std::cout << "Input: " << i << std::endl;
    int64_t embedding_vec_size = vec_size[i];
    int64_t seq_len = slot_num[i];

    std::cout << "Input Shape: " << batch_size << " " << seq_len << " " << embedding_vec_size
              << std::endl;

    core23::Shape in_shape = {batch_size, seq_len, embedding_vec_size};
    bottom_tensors.emplace_back(tensor_params.shape(in_shape));

    if (axis == 1) {
      out_slot_num += seq_len;
      if (i == 0) {
        out_vector_size = vec_size[0];
      }
    }
    if (axis == 2) {
      out_vector_size += embedding_vec_size;
      if (i == 0) {
        out_slot_num = slot_num[0];
      }
    }
  }
  core23::Tensor top_tensor;
  int64_t out_size = batch_size * out_slot_num * out_vector_size;
  Concat3DLayer<T> concat_3d_layer(bottom_tensors, top_tensor, axis, test::get_default_gpu());

  std::cout << "Input Size" << std::endl;
  for (int i = 0; i < n_ins; i++) {
    std::cout << batch_size << " " << slot_num[i] << " " << vec_size[i] << std::endl;
    h_ins[i] = new T[batch_size * slot_num[i] * vec_size[i]];
    test::normal_sync_cpu(h_ins[i], batch_size * slot_num[i] * vec_size[i], 0.f, 1.f, generator);

    core23::copy_sync(bottom_tensors[i].data(), h_ins[i], bottom_tensors[i].num_bytes(),
                      bottom_tensors[i].device(), core23::DeviceType::CPU);
  }

  concat_3d_layer.initialize();

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  concat_3d_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  std::unique_ptr<T[]> h_out(new T[out_size]);
  std::unique_ptr<T[]> h_python_out(new T[out_size]);

  core23::copy_sync(h_out.get(), top_tensor.data(), top_tensor.num_bytes(), core23::DeviceType::CPU,
                    top_tensor.device());

  python_concat(h_python_out.get(), h_ins.get(), batch_size, slot_num, vec_size, axis);

  ASSERT_TRUE(
      test::compare_array_approx<T>(h_out.get(), h_python_out.get(), out_size, Eps<T>::value()));

  // bprop*/
}

}  // namespace

TEST(concat_3d_layer, fp32_4x10x10) {
  std::vector<int64_t> items;
  std::vector<int64_t> slot_num;
  int64_t batch_size = 4;
  int64_t goodID_size = 3, shopID_size = 3, cateID_size = 4;
  items.push_back(goodID_size);
  slot_num.push_back(10);
  items.push_back(shopID_size);
  slot_num.push_back(10);
  items.push_back(cateID_size);
  slot_num.push_back(10);
  concat_3d_layer_test<float>(batch_size, slot_num, items, 2);
}
TEST(concat_3d_layer, fp32_4x10x13) {
  std::vector<int64_t> items;
  std::vector<int64_t> slot_num;
  int64_t batch_size = 4;
  int64_t goodID_size = 3, shopID_size = 6, cateID_size = 4;
  items.push_back(goodID_size);
  slot_num.push_back(10);
  items.push_back(shopID_size);
  slot_num.push_back(10);
  items.push_back(cateID_size);
  slot_num.push_back(10);
  concat_3d_layer_test<float>(batch_size, slot_num, items, 2);
}
TEST(concat_3d_layer, fp32_4x10x11) {
  std::vector<int64_t> items;
  std::vector<int64_t> slot_num;
  int64_t batch_size = 4;
  int64_t goodID_size = 4, shopID_size = 3, cateID_size = 4;
  items.push_back(goodID_size);
  slot_num.push_back(10);
  items.push_back(shopID_size);
  slot_num.push_back(10);
  items.push_back(cateID_size);
  slot_num.push_back(10);
  concat_3d_layer_test<float>(batch_size, slot_num, items, 2);
}

TEST(concat_3d_layer, fp32_2048x150x768) {
  std::vector<int64_t> items;
  std::vector<int64_t> slot_num;
  int64_t batch_size = 2048;
  int64_t slot_shot_his = 64, slot_long_his = 128, slot_target = 1;
  items.push_back(768);
  slot_num.push_back(slot_shot_his);
  items.push_back(768);
  slot_num.push_back(slot_long_his);
  items.push_back(768);
  slot_num.push_back(slot_target);
  concat_3d_layer_test<float>(batch_size, slot_num, items, 1);
}
