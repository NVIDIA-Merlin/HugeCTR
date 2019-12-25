/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "HugeCTR/include/network.hpp"
#include "HugeCTR/include/device_map.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace HugeCTR;

TEST(network_test, basic_network) {
  test::mpi_init();

  std::shared_ptr<GeneralBuffer<float>> buff(new GeneralBuffer<float>());
  const int batchsize = 128;
  std::shared_ptr<Tensor<float>> in_tensor(
      new Tensor<float>({batchsize, 128}, buff, TensorFormat_t::HW));
  std::shared_ptr<Tensor<float>> label_tensor(
      new Tensor<float>({batchsize, 1}, buff, TensorFormat_t::HW));
  buff->init(0);
  std::shared_ptr<DeviceMap> device_map(new DeviceMap({{0}}, 0));
  GPUResourceGroup gpu_resource_group(device_map);
  Network basic_network(in_tensor, label_tensor, batchsize, 0, gpu_resource_group[0]);
  // load parameters from CPU to GPU
  std::unique_ptr<float[]> params(new float[basic_network.get_params_num()]);
  basic_network.upload_params_to_device(params.get());

  // train
  basic_network.train();

  // download parameters
  std::unique_ptr<float[]> updated_params(new float[basic_network.get_params_num()]);
  basic_network.download_params_to_host(updated_params.get());
  // verification
}
