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
#include<iostream>
#include<fstream>
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
  std::vector<int> device_list{0,1};
  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back(device_list);
  std::shared_ptr<DeviceMap> device_map(new DeviceMap(vvgpu, 0));
  int batch_size = 4096;
  Parser p(json_name, batch_size);
  std::unique_ptr<DataReader<TypeKey>> data_reader;
  std::unique_ptr<DataReader<TypeKey>> data_reader_eval;
  std::vector<std::unique_ptr<Embedding<TypeKey>>> embedding;
  std::vector<std::unique_ptr<Network>> networks;
  std::shared_ptr<GPUResourceGroup> gpu_resource_group(new GPUResourceGroup(device_map));

  p.create_pipeline(data_reader, data_reader_eval, embedding, networks, gpu_resource_group);
  return;
}



const long long label_dim = 1;
const long long dense_dim = 64;
const int max_nnz = 10;
typedef long long T;
const int vocabulary_size = 1603616;
const std::string prefix("./simple_sparse_embedding/simple_sparse_embedding");
const std::string file_list_name = prefix + "_file_list.txt";
const int num_files = 3;
const long long num_records = 4096 * 3;
const long long slot_num = 20;
const Check_t CHK = Check_t::Sum;


TEST(parser_test, simple_sparse_embedding) {
  test::mpi_init();
  HugeCTR::data_generation<T, CHK>(file_list_name, prefix, num_files, num_records, slot_num,
				   vocabulary_size, label_dim, dense_dim, max_nnz);

  std::string json_name = PROJECT_HOME_ + "utest/simple_sparse_embedding.json";
  std::string plan_name = PROJECT_HOME_ + "utest/all2all_plan_dgx_{0,1}.json";
  std::ifstream src;
  std::ofstream dst;

  src.open(plan_name, std::ios::in);
  dst.open("./all2all_plan.json", std::ofstream::out);
  std::filebuf* inbuf  = src.rdbuf();
  dst << inbuf;
  src.close();
  dst.close();
  test_parser<long long>(json_name);
}

TEST(parser_test, basic_parser2) { std::string json_name("basic_parser2.json"); }

TEST(parser_test, basic_parser3) { std::string json_name("basic_parser3.json"); }
