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


#include "HugeCTR/include/data_parser.hpp"
#include "gtest/gtest.h"

using namespace HugeCTR;

TEST(data_parser, InputParser) {
  // setup DataSimulator and InputParser
  UnifiedDataSimulator<long long> uSimulator(4211111111, 4222222222);
  InputParser iParser("temp.data", uSimulator);
  const int N(16);
  // write data into file
  iParser.write(N);
  iParser.write(N);
  // read data from file
  long long* index = new long long int[N];
  iParser.read(N, index);
  // print
  for (int i = 0; i < N; i++) {
    std::cout << index[i] << ",";
  }
  iParser.read(N, index);
  // print
  for (int i = 0; i < N; i++) {
    std::cout << index[i] << ",";
  }
}

TEST(data_parser, ParameterParser) {
  GaussianDataSimulator<float> gSimulator(0, 20, -100., 100.);
  ParameterParser pParser("gtemp.data", gSimulator);
  const int N(16);
  // write data into file
  pParser.write(N);
  pParser.write(N);
  // read data from file
  float* params = new float[N];
  pParser.read(N, params);
  // print
  for (int i = 0; i < N; i++) {
    std::cout << params[i] << ",";
  }
  pParser.read(N, params);
  // print
  for (int i = 0; i < N; i++) {
    std::cout << params[i] << ",";
  }
}

TEST(data_parser, InputParserll) {
  // setup DataSimulator and InputParser
  GaussianDataSimulator<long long> gSimulator(0, 20, 0, 100);
  InputParser iParser("gltemp.data", gSimulator);
  const int N(16);
  // write data into file
  iParser.write(N);
  iParser.write(N);
  // read data from file
  long long* index = new long long int[N];
  iParser.read(N, index);
  // print
  for (int i = 0; i < N; i++) {
    std::cout << index[i] << ",";
  }
  iParser.read(N, index);
  // print
  for (int i = 0; i < N; i++) {
    std::cout << index[i] << ",";
  }
}
