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

#include "HugeCTR/include/data_generator.hpp"
#include <sys/stat.h>
#include <fstream>
#include <ios>
#include <iostream>
#include <sstream>
#include <vector>
#include "HugeCTR/include/parser.hpp"
#include "nlohmann/json.hpp"

using namespace HugeCTR;

static std::string usage_str_raw = "usage: ./data_generator your_config.json [option:--long-tail <long|medium|short>]";
static std::string usage_str =
    "usage: ./data_generator your_config.json data_folder vocabulary_size max_nnz [option:#files] "
    "[option:#samples per file] [option:--long-tail <long|medium|short>]";
static int NUM_FILES = 128;
static int NUM_SAMPLES_PER_FILE = 40960;
static bool long_tail = false;
static float alpha = 0.0;

int main(int argc, char* argv[]) {
  if ((argc != 2 && argc != 4 && argc != 5 && argc != 6 && argc != 7 && argc != 9) ||
      (argc == 4 && strcmp(argv[2], "--long-tail") != 0 ) ||
      (argc == 4 && strcmp(argv[2], "--long-tail") == 0 && strcmp(argv[3], "long") != 0 && strcmp(argv[3], "medium") != 0 && strcmp(argv[3], "short") !=0 )||
      (argc == 9 && strcmp(argv[7], "--long-tail") != 0 ) ||
      (argc == 9 && strcmp(argv[7], "--long-tail") == 0 && strcmp(argv[8], "long") != 0 && strcmp(argv[8], "medium") != 0 && strcmp(argv[8], "short") !=0 )) {
    std::cout << "To generate raw format: " << usage_str_raw << std::endl;
    std::cout << "To generate norm format: " << usage_str << std::endl;
    exit(-1);
  }

  // Parsing the configure file:
  std::string config_file = argv[1];
  std::ifstream file(config_file);
  if (!file.is_open()) {
    std::cerr << "file.is_open() failed: " + config_file << std::endl;
  }
  nlohmann::json config;
  file >> config;
  file.close();
  auto j_layers_array = config.find("layers").value();

  const std::map<std::string, DataReaderType_t> Data_READER_TYPE = {
      {"Norm", DataReaderType_t::Norm}, {"Raw", DataReaderType_t::Raw}};

  DataReaderType_t format = DataReaderType_t::Norm;
  if (has_key_(j_layers_array[0], "format")) {
    const auto data_format_name = get_value_from_json<std::string>(j_layers_array[0], "format");
    if (!find_item_in_map(format, data_format_name, Data_READER_TYPE)) {
      CK_THROW_(Error_t::WrongInput, "No such data format: " + data_format_name);
    }
  }

  switch (format) {
    case DataReaderType_t::Norm: {
      std::string data_folder = argv[2];
      size_t vocabulary_size = std::stoul(argv[3]);
      int max_nnz = atoi(argv[4]);
      if (argc == 6 || argc == 7 || argc == 9) {
        NUM_FILES = atoi(argv[5]);
      }
      if (argc == 7 || argc == 9) {
        NUM_SAMPLES_PER_FILE = atoi(argv[6]);
      }

      if (argc == 9) {
        long_tail = true;
        if (strcmp(argv[8], "long") == 0)
          alpha = 1.0;
        else if (strcmp(argv[8], "medium") == 0)
          alpha = 3.0;
        else
          alpha = 5.0;
      }

      std::cout << "Configure File: " << config_file << ", Data Folder: " << data_folder
                << ", Vocabulary Size: " << vocabulary_size << ", Max NNZ:" << max_nnz
                << ", #files: " << NUM_FILES << ", #samples per file: " << NUM_SAMPLES_PER_FILE
                << ", Use power law distribution: " << long_tail << ", alpha of power law: " << alpha << std::endl;

      int label_dim = 0, dense_dim = 0;
      Check_t check_type;
      std::string source_data;
      std::vector<DataReaderSparseParam> data_reader_sparse_param_array;
      std::string eval_source;
      std::string top_strs_label, top_strs_dense;
      std::vector<std::string> sparse_names;
      std::map<std::string, SparseInput<long long>> sparse_input_map;
      parse_data_layer_helper(j_layers_array[0], label_dim, dense_dim, check_type, source_data,
                              data_reader_sparse_param_array, eval_source, top_strs_label,
                              top_strs_dense, sparse_names, sparse_input_map);

      int num_slot = 0;
      for (auto& param : data_reader_sparse_param_array) {
        if (param.slot_num * max_nnz > param.max_feature_num) {
          std::cerr
              << "Error: max_nnz * slot_num cannot be larger than max_feature_num in config file!"
              << std::endl;
          exit(-1);
        }
        num_slot += param.slot_num;
      }

      check_make_dir(data_folder);

      /*parse the solver*/
      bool i64_input_key(false);
      auto j_solver = get_json(config, "solver");
      if (has_key_(j_solver, "input_key_type")) {
        auto str = get_value_from_json<std::string>(j_solver, "input_key_type");
        if (0 == str.compare("I64")) {
          i64_input_key = true;
          MESSAGE_("input_key_type is I64.");
        } else if (0 == str.compare("I32")) {
          i64_input_key = false;
          MESSAGE_("input_key_type is I32.");
        } else {
          CK_THROW_(Error_t::WrongInput, "input_key_type must be {I64 or I32}");
        }
      } else {
        i64_input_key = false;
        MESSAGE_("Default input_key_type is I32.");
      }

      if (check_type == Check_t::Sum) {
        if (i64_input_key) {  // I64 = long long
          data_generation_for_test<long long, Check_t::Sum>(
              source_data, data_folder + "/train/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot,
              vocabulary_size, label_dim, dense_dim, max_nnz, long_tail, alpha);
          data_generation_for_test<long long, Check_t::Sum>(
              eval_source, data_folder + "/val/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot,
              vocabulary_size, label_dim, dense_dim, max_nnz, long_tail, alpha);
        } else {  // I32 = unsigned int
          data_generation_for_test<unsigned int, Check_t::Sum>(
              source_data, data_folder + "/train/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot,
              vocabulary_size, label_dim, dense_dim, max_nnz, long_tail, alpha);
          data_generation_for_test<unsigned int, Check_t::Sum>(
              eval_source, data_folder + "/val/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot,
              vocabulary_size, label_dim, dense_dim, max_nnz, long_tail, alpha);
        }
      } else {
        if (i64_input_key) {  // I64 = long long
          data_generation_for_test<long long, Check_t::None>(
              source_data, data_folder + "/train/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot,
              vocabulary_size, label_dim, dense_dim, max_nnz, long_tail, alpha);
          data_generation_for_test<long long, Check_t::None>(
              eval_source, data_folder + "/val/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot,
              vocabulary_size, label_dim, dense_dim, max_nnz, long_tail, alpha);
        } else {  // I32 = unsigned int
          data_generation_for_test<unsigned int, Check_t::None>(
              source_data, data_folder + "/train/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot,
              vocabulary_size, label_dim, dense_dim, max_nnz, long_tail, alpha);
          data_generation_for_test<unsigned int, Check_t::None>(
              eval_source, data_folder + "/val/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot,
              vocabulary_size, label_dim, dense_dim, max_nnz, long_tail, alpha);
        }
      }
      break;
    }
    case DataReaderType_t::Raw: {
      std::cout << "Configure File: " << config_file << std::endl;

      const auto num_samples = get_value_from_json<long long>(j_layers_array[0], "num_samples");
      const auto eval_num_samples =
          get_value_from_json<long long>(j_layers_array[0], "eval_num_samples");

      std::vector<long long> slot_size_array;
      if (has_key_(j_layers_array[0], "slot_size_array")) {
        auto temp_array = get_json(j_layers_array[0], "slot_size_array");
        if (!temp_array.is_array()) {
          CK_THROW_(Error_t::WrongInput, "!slot_size_array.is_array()");
        }
        long long slot_num = 0;
        for (auto j_slot_size : temp_array) {
          long long slot_size = j_slot_size.get<long long>();
          slot_size_array.push_back(slot_size);
          slot_num += slot_size;
        }
        MESSAGE_("vocabulary size: " + std::to_string(slot_num));
      } else {
        CK_THROW_(Error_t::WrongInput, "No such key in json file: slot_size_array.");
      }

      const auto j_label = get_json(j_layers_array[0], "label");
      const auto label_dim = get_value_from_json<int>(j_label, "label_dim");

      const auto j_dense = get_json(j_layers_array[0], "dense");
      const auto dense_dim = get_value_from_json<int>(j_dense, "dense_dim");

      const auto source_data = get_value_from_json<std::string>(j_layers_array[0], "source");
      const auto eval_source = get_value_from_json<std::string>(j_layers_array[0], "eval_source");
      if (file_exist(source_data)) {
        CK_THROW_(Error_t::WrongInput, source_data + " already exist!");
        exit(-1);
      }
      if (file_exist(eval_source)) {
        CK_THROW_(Error_t::WrongInput, eval_source + " already exist!");
        exit(-1);
      }

      const auto float_label_dense =
          get_value_from_json_soft<bool>(j_layers_array[0], "float_label_dense", false);

      std::string source_dir;
      const size_t last_slash_idx = source_data.rfind('/');
      if (std::string::npos != last_slash_idx) {
        source_dir = source_data.substr(0, last_slash_idx);
      }
      check_make_dir(source_dir);

      std::string eval_dir;
      const size_t last_slash_idx_eval = eval_source.rfind('/');
      if (std::string::npos != last_slash_idx_eval) {
        eval_dir = eval_source.substr(0, last_slash_idx_eval);
      }
      check_make_dir(eval_dir);

      // train data
      data_generation_for_raw(source_data, num_samples, label_dim, dense_dim,
                              slot_size_array.size(), float_label_dense, slot_size_array, 
                              long_tail, alpha);
      // eval data
      data_generation_for_raw(eval_source, eval_num_samples, label_dim, dense_dim,
                              slot_size_array.size(), float_label_dense, slot_size_array,
                              long_tail, alpha);
      break;
    }
    default: {
      assert(!"Error: no such option && should never get here!");
      break;
    }
  }

  MESSAGE_("Data generation done.");
  return 0;
}
