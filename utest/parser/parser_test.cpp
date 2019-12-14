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

#include "HugeCTR/include/parser.hpp"
#include <cuda_profiler_api.h>
#include <vector>
#include "gtest/gtest.h"
#include "utest/test_utils.h"

/**
 * Note This test share the same input data with session, so we can only test after session_test.
 */

using namespace HugeCTR;
typedef long long TypeKey;

template <typename TypeKey>
void test_parser(std::string& json_name) {
  std::vector<int> device_list = {0};
  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back(device_list);
  DeviceMap device_map(vvgpu, 0);
  int batch_size = 4096;
  Parser p(json_name, batch_size);
  std::unique_ptr<DataReader<TypeKey>> data_reader;
  std::unique_ptr<DataReader<TypeKey>> data_reader_eval;
  std::unique_ptr<Embedding<TypeKey>> embedding;
  std::vector<std::unique_ptr<Network>> networks;
  std::shared_ptr<GPUResourceGroup> gpu_resource_group(new GPUResourceGroup(device_map));

  p.create_pipeline(data_reader, data_reader_eval, embedding, networks, gpu_resource_group);
  return;
}

TEST(parser_test, simple_sparse_embedding) {
  test::mpi_init();
  std::string json_name = PROJECT_HOME_ + "utest/parser/simple_sparse_embedding.json";
  test_parser<long long>(json_name);
}

TEST(parser_test, basic_parser2) { std::string json_name("basic_parser2.json"); }

TEST(parser_test, basic_parser3) { std::string json_name("basic_parser3.json"); }
