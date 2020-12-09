/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "HugeCTR/include/inference/session_inference.hpp"

#include <cuda_profiler_api.h>

#include <memory>

#include "HugeCTR/include/data_generator.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace HugeCTR;
namespace {
void test_inference_session(const std::string& config_file_path) {
  InferenceSession inference_session(config_file_path, 0);
}
}  // namespace

TEST(inference_session_test, inference_parser_test) {
  std::string json_name = PROJECT_HOME_ + "utest/simple_inference_config.json";
  test_inference_session(json_name);
}