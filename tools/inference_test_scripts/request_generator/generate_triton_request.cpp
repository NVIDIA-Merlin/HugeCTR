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

#include <omp.h>
#include <time.h>

#include <algorithm>
#include <argparse/argparse.hpp>
#include <chrono>
#include <ctime>
#include <inference_key_generator.hpp>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <vector>

using namespace HugeCTR;

void generate_triton_request(const std::string output_path, const int dense_dim,
                             const std::vector<int> sparse_dims,
                             const std::vector<long long> number_unique_keys, bool int64_key,
                             const std::string dense_name, const std::string sparse_name,
                             const std::string rowoffset_name, const int num_batch,
                             const int batch_size, bool powerlaw, const float alpha,
                             const float hot_key_percentage, const float hot_key_coverage) {
  nlohmann::json request_json;
  nlohmann::json data_array = nlohmann::json::array();

  // TODO: make this parallel
  for (int i = 0; i < num_batch; i++) {
    nlohmann::json batch_json;
    nlohmann::json dense_json;
    nlohmann::json sparse_json;
    nlohmann::json rowoffset_json;

    // Generate dense values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> dense_array;
    for (int i = 0; i < dense_dim * batch_size; i++) {
      dense_array.push_back(dist(gen));
    }
    dense_json["content"] = dense_array;
    std::vector<int> dense_shape_array = {dense_dim * batch_size};
    dense_json["shape"] = dense_shape_array;

    // Generate sparse keys
    int sparse_dim_sum = std::accumulate(sparse_dims.begin(), sparse_dims.end(), 0);
    if (int64_key) {
      std::vector<long long> sparse_array;
      for (int i = 0; i < sparse_dims.size(); i++) {
        if (powerlaw) {
          key_vector_generator_by_powerlaw(sparse_array, batch_size * sparse_dims[i],
                                           number_unique_keys[i], alpha);
        } else {
          key_vector_generator_by_hotkey(sparse_array, batch_size * sparse_dims[i],
                                         number_unique_keys[i], hot_key_percentage,
                                         hot_key_coverage);
        }
      }
      sparse_json["content"] = sparse_array;
    } else {
      std::vector<unsigned int> sparse_array;
      for (int i = 0; i < sparse_dims.size(); i++) {
        if (powerlaw) {
          key_vector_generator_by_powerlaw(sparse_array, batch_size * sparse_dims[i],
                                           number_unique_keys[i], alpha);
        } else {
          key_vector_generator_by_hotkey(sparse_array, batch_size * sparse_dims[i],
                                         number_unique_keys[i], hot_key_percentage,
                                         hot_key_coverage);
        }
      }
      sparse_json["content"] = sparse_array;
    }
    std::vector<int> sparse_shape_array = {sparse_dim_sum * batch_size};
    sparse_json["shape"] = sparse_shape_array;

    // Generate row offsets
    std::vector<unsigned int> rowoffset_array;
    for (unsigned int i = 0; i < sparse_dims.size(); i++) {
      for (unsigned int j = 0; j < sparse_dims[i] * batch_size + 1; j++) {
        rowoffset_array.push_back(j);
      }
    }
    rowoffset_json["content"] = rowoffset_array;
    std::vector<int> rowoffset_shape_array = {sparse_dim_sum * batch_size +
                                              (int)sparse_dims.size()};
    rowoffset_json["shape"] = rowoffset_shape_array;

    batch_json[dense_name] = dense_json;
    batch_json[sparse_name] = sparse_json;
    batch_json[rowoffset_name] = rowoffset_json;
    data_array.push_back(batch_json);
  }

  request_json["data"] = data_array;
  std::ofstream file_stream(output_path);
  file_stream << std::setw(2) << request_json;
  file_stream.close();
  HCTR_LOG(INFO, ROOT, "Save the request JSON to %s successfully\n", output_path.c_str());
  return;
}

int main(int argc, char* argv[]) {
  argparse::ArgumentParser args("Triton Performance Analyzer Request Generator");

  args.add_argument("--output_path")
      .help("The path of the output JSON file")
      .required()
      .action([](const std::string& value) { return value; });

  args.add_argument("--dense_dim")
      .help("The number of batches to do benchmark")
      .required()
      .action([](const std::string& value) { return std::stoi(value); });

  args.add_argument("--sparse_dims")
      .help("The sparse dim of each embedding table, e.g. 2,26")
      .required()
      .action([](const std::string& value) { return value; });

  args.add_argument("--num_unique_keys")
      .help("The number of unique_keys in each embedding table")
      .required()
      .action([](const std::string& value) { return value; });

  args.add_argument("--int64_key")
      .help("Use int64 key type or not.")
      .default_value(true)
      .implicit_value(true);

  args.add_argument("--dense_name")
      .help("The input name of dense tensors, should be consistent to your Triton config.")
      .default_value<std::string>("DES")
      .action([](const std::string& value) { return value; });

  args.add_argument("--sparse_name")
      .help("The input name of sparse tensors, should be consistent to your Triton config.")
      .default_value<std::string>("CATCOLUMN")
      .action([](const std::string& value) { return value; });

  args.add_argument("--rowoffset_name")
      .help("The input name of row offset, should be consistent to your Triton config.")
      .default_value<std::string>("ROWINDEX")
      .action([](const std::string& value) { return value; });

  args.add_argument("--num_batch")
      .help("The number of batches to do benchmark")
      .default_value(10)
      .action([](const std::string& value) { return std::stoi(value); });

  args.add_argument("--batch_size")
      .help("The number of samples in each batch")
      .default_value(64)
      .action([](const std::string& value) { return std::stoi(value); });

  args.add_argument("--powerlaw")
      .help("Generate the queried key based on the power distribution")
      .default_value(false)
      .implicit_value(true);

  args.add_argument("--alpha")
      .help("Alpha of power distribution")
      .default_value<float>(1.2)
      .action([](const std::string& value) { return std::stof(value); });

  args.add_argument("--hot_key_percentage")
      .help("Percentage of hot keys in embedding tables")
      .default_value<float>(0.2)
      .action([](const std::string& value) { return std::stof(value); });

  args.add_argument("--hot_key_coverage")
      .help("The probability of the hot key in each iteration")
      .default_value<float>(0.8)
      .action([](const std::string& value) { return std::stof(value); });

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    exit(1);
  }

  std::vector<int> sparse_dims;
  std::string sparse_dims_string = args.get<std::string>("--sparse_dims");
  while (sparse_dims_string.size()) {
    auto comma = sparse_dims_string.find(",");
    if (comma != std::string::npos) {
      sparse_dims.push_back(std::stoi(sparse_dims_string.substr(0, comma)));
      sparse_dims_string.erase(0, comma + 1);
    } else {
      sparse_dims.push_back(std::stoi(sparse_dims_string));
      break;
    }
  }

  std::vector<long long> num_unique_keys;
  std::string num_unique_keys_string = args.get<std::string>("--num_unique_keys");
  while (num_unique_keys_string.size()) {
    auto comma = num_unique_keys_string.find(",");
    if (comma != std::string::npos) {
      num_unique_keys.push_back(std::stoll(num_unique_keys_string.substr(0, comma)));
      num_unique_keys_string.erase(0, comma + 1);
    } else {
      num_unique_keys.push_back(std::stoll(num_unique_keys_string));
      break;
    }
  }

  generate_triton_request(
      args.get<std::string>("--output_path"), args.get<int>("--dense_dim"), sparse_dims,
      num_unique_keys, args.get<bool>("--int64_key"), args.get<std::string>("--dense_name"),
      args.get<std::string>("--sparse_name"), args.get<std::string>("--rowoffset_name"),
      args.get<int>("--num_batch"), args.get<int>("--batch_size"), args.get<bool>("--powerlaw"),
      args.get<float>("--alpha"), args.get<float>("--hot_key_percentage"),
      args.get<float>("--hot_key_coverage"));
}