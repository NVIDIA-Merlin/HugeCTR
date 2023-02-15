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

#include <layers/concat_3d_layer.hpp>
#include <utest/test_utils.hpp>
#include <utils.hpp>
#include <vector>

using namespace std;
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
void python_concat(T *output, T **inputs, int batch_size, const std::vector<int> &slot_num,
                   const std::vector<int> &vec_size, int axis) {
  cout << "===========Python concat==========" << endl;
  auto num = vec_size.size();
  std::string temp_name = "tmpdata.bin";
  std::ofstream py_input(temp_name.c_str(), std::ios::binary | std::ios::out);
  char *input_ptr = nullptr;
  std::vector<T> result;

  for (size_t i = 0; i < vec_size.size(); i++) {
    cout << batch_size << " " << slot_num[i] << " " << vec_size[i] << endl;
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

  cout << result.size() << endl;

  pclose(py_output);
}
template <typename T>
void concat_3d_layer_general_test(size_t batch_size, std::vector<int> slot_num,
                                  std::vector<int> vec_size, int axis) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
  int num = vec_size.size();
  // assert(item.size() == slot_num.size());
  Tensors2<T> in_tensors;
  size_t out_slot_num = 0;
  size_t out_vector_size = 0;

  for (int i = 0; i < num; i++) {
    size_t embedding_vec_size = vec_size[i];
    size_t seq_len = slot_num[i];
    vector<size_t> dims_in = {batch_size, seq_len, embedding_vec_size};
    Tensor2<T> in_tensor;
    buff->reserve(dims_in, &in_tensor);
    in_tensors.push_back(in_tensor);
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
  cout << "Input Size" << endl;
  for (int i = 0; i < num; i++) {
    cout << batch_size << " " << slot_num[i] << " " << vec_size[i] << endl;
  }

  Tensor2<T> out_tensor;
  vector<size_t> dims_out = {batch_size, out_slot_num, out_vector_size};
  size_t out_size = batch_size * out_slot_num * out_vector_size;
  buff->reserve(dims_out, &out_tensor);

  cout << "Output Size" << endl;
  cout << batch_size << " " << out_slot_num << " " << out_vector_size << endl;

  Concat3DLayer<T> concat_3d_layer(in_tensors, out_tensor, buff, axis, test::get_default_gpu());
  buff->allocate();
  concat_3d_layer.initialize();

  std::unique_ptr<T *[]> h_d_ins(new T *[num]);
  for (int i = 0; i < num; i++) {
    h_d_ins[i] = in_tensors[i].get_ptr();
  }
  T *d_out = out_tensor.get_ptr();

  std::unique_ptr<T *[]> h_ins(new T *[num]);
  for (int i = 0; i < num; i++) {
    h_ins[i] = new T[batch_size * slot_num[i] * vec_size[i]];
  }

  std::unique_ptr<T *[]> h_ins_b(new T *[num]);
  for (int i = 0; i < num; i++) {
    h_ins_b[i] = new T[batch_size * slot_num[i] * vec_size[i]];
  }
  std::unique_ptr<T[]> h_out(new T[out_size]);
  std::unique_ptr<T[]> h_python_out(new T[out_size]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  // fprop
  for (int i = 0; i < num; i++) {
    size_t size = batch_size * slot_num[i] * vec_size[i];
    simulator.fill(h_ins[i], size);
    HCTR_LIB_THROW(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice));
  }
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  cout << "Before fprop" << endl;

  /*for (int i = 0; i < num; i++) {
    cout << "Input " << i << endl;
    for (size_t j = 0; j < batch_size * slot_num[i] * vec_size[i]; j++) {
      cout << h_ins[i][j] << " ";
      if (((j + 1) % vec_size[i]) == 0) {
        cout << endl;
        if (((j + 1) % (slot_num[i] * vec_size[i])) == 0) {
          cout << endl;
        }
      }
    }
  }*/

  concat_3d_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  HCTR_LIB_THROW(cudaMemcpy(h_out.get(), d_out, out_size * sizeof(T), cudaMemcpyDeviceToHost));

  python_concat(h_python_out.get(), h_ins.get(), batch_size, slot_num, vec_size, axis);

  /*cout << "After fprop" << endl;
  cout << "Output " << endl;
  for (size_t j = 0; j < batch_size * out_slot_num * out_vector_size; j++) {
    cout << h_out.get()[j] << " ";
    if (((j + 1) % out_vector_size) == 0) {
      cout << endl;
      if (((j + 1) % (out_slot_num * out_vector_size)) == 0) {
        cout << endl;
      }
    }
  }

  cout << "Python Output " << endl;
  for (size_t j = 0; j < batch_size * out_slot_num * out_vector_size; j++) {
    cout << h_python_out.get()[j] << " ";
    if (((j + 1) % out_vector_size) == 0) {
      cout << endl;
      if (((j + 1) % (out_slot_num * out_vector_size)) == 0) {
        cout << endl;
      }
    }
  }*/

  ASSERT_TRUE(
      test::compare_array_approx<T>(h_out.get(), h_python_out.get(), out_size, Eps<T>::value()));

  // bprop
}
}  // namespace

TEST(concat_3d_layer, fp32_4x10x10) {
  std::vector<int> items;
  std::vector<int> slot_num;
  int batch_size = 4;
  int goodID_size = 3, shopID_size = 3, cateID_size = 4;
  items.push_back(goodID_size);
  slot_num.push_back(10);
  items.push_back(shopID_size);
  slot_num.push_back(10);
  items.push_back(cateID_size);
  slot_num.push_back(10);
  concat_3d_layer_general_test<float>(batch_size, slot_num, items, 2);
}
TEST(concat_3d_layer, fp32_4x10x13) {
  std::vector<int> items;
  std::vector<int> slot_num;
  int batch_size = 4;
  int goodID_size = 3, shopID_size = 6, cateID_size = 4;
  items.push_back(goodID_size);
  slot_num.push_back(10);
  items.push_back(shopID_size);
  slot_num.push_back(10);
  items.push_back(cateID_size);
  slot_num.push_back(10);
  concat_3d_layer_general_test<float>(batch_size, slot_num, items, 2);
}
TEST(concat_3d_layer, fp32_4x10x11) {
  std::vector<int> items;
  std::vector<int> slot_num;
  int batch_size = 4;
  int goodID_size = 4, shopID_size = 3, cateID_size = 4;
  items.push_back(goodID_size);
  slot_num.push_back(10);
  items.push_back(shopID_size);
  slot_num.push_back(10);
  items.push_back(cateID_size);
  slot_num.push_back(10);
  concat_3d_layer_general_test<float>(batch_size, slot_num, items, 2);
}

TEST(concat_3d_layer, fp32_2048x150x768) {
  std::vector<int> items;
  std::vector<int> slot_num;
  int batch_size = 2048;
  int slot_shot_his = 64, slot_long_his = 128, slot_target = 1;
  items.push_back(768);
  slot_num.push_back(slot_shot_his);
  items.push_back(768);
  slot_num.push_back(slot_long_his);
  items.push_back(768);
  slot_num.push_back(slot_target);
  concat_3d_layer_general_test<float>(batch_size, slot_num, items, 1);
}

TEST(concat_3d_layer, fp32_1024x300x512) {
  std::vector<int> items;
  std::vector<int> slot_num;
  int batch_size = 1024;
  int goodID_size = 128, shopID_size = 128, cateID_size = 128;
  items.push_back(goodID_size);
  slot_num.push_back(100);
  items.push_back(shopID_size);
  slot_num.push_back(100);
  items.push_back(cateID_size);
  slot_num.push_back(100);
  concat_3d_layer_general_test<float>(batch_size, slot_num, items, 2);
}

TEST(concat_3d_layer, fp32_4x16x8) {
  std::vector<int> items;
  std::vector<int> slot_num;
  int batch_size = 4;
  int slot_shot_his = 3, slot_long_his = 4, slot_target = 1;
  items.push_back(8);
  slot_num.push_back(slot_shot_his);
  items.push_back(8);
  slot_num.push_back(slot_long_his);
  items.push_back(8);
  slot_num.push_back(slot_target);
  concat_3d_layer_general_test<float>(batch_size, slot_num, items, 1);
}

TEST(concat_3d_layer, fp32_4x16x7) {
  std::vector<int> items;
  std::vector<int> slot_num;
  int batch_size = 4;
  int slot_shot_his = 3, slot_long_his = 4, slot_target = 1;
  items.push_back(7);
  slot_num.push_back(slot_shot_his);
  items.push_back(7);
  slot_num.push_back(slot_long_his);
  items.push_back(7);
  slot_num.push_back(slot_target);
  concat_3d_layer_general_test<float>(batch_size, slot_num, items, 1);
}
