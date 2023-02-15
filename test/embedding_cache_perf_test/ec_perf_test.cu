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

#include <base/debug/logger.hpp>
#include <embedding_cache_perf_test/ec_test_helper.cuh>
#include <gpu_cache/include/nv_gpu_cache.hpp>
#include <iostream>
#include <nlohmann/json.hpp>

namespace {
constexpr size_t ONE_GiB = 1024 * 1024 * 1024;
constexpr size_t num_key_candidates = 10000000;
constexpr size_t key_range = num_key_candidates;
constexpr size_t embedding_vec_size = 128;
constexpr size_t num_sets = 15625;
constexpr size_t num_keys_to_fill = 1000000;
constexpr float alpha = 1.05;
constexpr int repeat_times = 10;

std::string get_json_key_name(size_t batch_size, size_t num_hot) {
  return std::to_string(batch_size) + "x" + std::to_string(num_hot);
}

inline nlohmann::json read_json_file(const std::string& filename) {
  nlohmann::json config;
  std::ifstream file_stream(filename);
  if (!file_stream.is_open()) {
    HCTR_OWN_THROW(HugeCTR::Error_t::FileCannotOpen, "file_stream.is_open() failed: " + filename);
  }
  file_stream >> config;
  file_stream.close();
  return config;
}

void ec_perf_test(std::string json_file, size_t num_hot) {
  size_t max_batch_size = 65536;
  size_t expected_throughput = 10;

  auto throughput_json = read_json_file(json_file);

  EcTestHelper test_helper(max_batch_size, num_hot, embedding_vec_size, num_sets, alpha,
                           num_key_candidates, key_range);
  test_helper.fill_cache_linear(num_key_candidates);
  for (int i = 0; i < repeat_times; i++) {
    test_helper.fill_cache(num_keys_to_fill);
  }
  std::cout << "batch_size," << '\t' << "num_keys_in_query," << '\t' << "throughput" << std::endl;
  size_t batch_size = 65536;
  while (batch_size >= 1024) {
    for (int i = 0; i < repeat_times; i++) {
      test_helper.test_query(batch_size);
    }
    size_t expected_throughput = throughput_json[get_json_key_name(batch_size, num_hot)].get<int>();
    auto time_list = test_helper.get_time_list();
    std::vector<double> throughput_list;
    for (auto time : time_list) {
      throughput_list.push_back(test_helper.get_memory_read_in_bytes(batch_size) * 2 /
                                (time / 1000.0f) / ONE_GiB);
    }
    auto max_throughput = *std::max_element(throughput_list.begin(), throughput_list.end());
    test_helper.clear_results();
    ASSERT_GE(max_throughput, (double)expected_throughput);
    std::cout << batch_size << '\t' << batch_size * num_hot << '\t' << max_throughput << std::endl;
    batch_size /= 2;
  }
}

}  // namespace

TEST(ec_perf_test, zipf_distribution) {
  ec_perf_test("/workdir/test/embedding_cache_perf_test/expected_throughput.json", 64);
};
