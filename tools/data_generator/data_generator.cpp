/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <unordered_set>
#include <vector>

#include "HugeCTR/include/parser.hpp"
#include "HugeCTR/include/utils.hpp"
#ifdef ENABLE_MPI
#include <mpi.h>
#endif
using namespace HugeCTR;

static std::string usage_str_raw =
    "usage: ./data_generator --config-file your_config.json --distribution <powerlaw | unified> "
    "[option: --nnz-array <nnz array in csv: one hot>] [option: --alpha xxx or --longtail <long | "
    "medium | short>]";
static std::string usage_str =
    "usage: ./data_generator --config-file your_config.json --voc-size-array <vocabulary size "
    "array in csv> --distribution <powerlaw | unified> [option: --nnz-array <nnz array in csv: one "
    "hot>] [option: --alpha xxx or --longtail <long | medium | short>] [option:--data-folder "
    "<folder_path: ./>] [option:--files <number of files: 128>] [option:--samples <samples per "
    "file: 40960>]";
static int NUM_FILES = 128;
static int NUM_SAMPLES_PER_FILE = 40960;
static std::unordered_set<std::string> TAIL_TYPE{"long", "medium", "short"};
static std::string TAIL{"medium"};
static bool use_long_tail = false;
static float alpha = 0.0;

void parse_common_arguments(const nlohmann::json& j, DataReaderType_t& format, int& label_dim,
                            int& dense_dim, int& num_slot, std::string& source_data,
                            std::string& eval_source, bool& i64_input_key) {
  source_data = get_value_from_json<std::string>(j, "source");
  FIND_AND_ASSIGN_STRING_KEY(eval_source, j);
  label_dim = get_value_from_json<int>(j, "label_dim");
  dense_dim = get_value_from_json<int>(j, "dense_dim");

  const std::map<std::string, DataReaderType_t> Data_READER_TYPE = {
      {"Norm", DataReaderType_t::Norm}, {"Raw", DataReaderType_t::Raw}};
  format = DataReaderType_t::Norm;
  if (has_key_(j, "format")) {
    const auto data_format_name = get_value_from_json<std::string>(j, "format");
    if (!find_item_in_map(format, data_format_name, Data_READER_TYPE)) {
      CK_THROW_(Error_t::WrongInput, "No such data format: " + data_format_name);
    }
  }
  num_slot = get_value_from_json<int>(j, "num_slot");
  i64_input_key = false;
  if (has_key_(j, "input_key_type")) {
    auto str = get_value_from_json<std::string>(j, "input_key_type");
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
}

static void parse_norm_arguments(const nlohmann::json& j, Check_t& check_type) {
  const std::map<std::string, Check_t> CHECK_TYPE_MAP = {{"Sum", Check_t::Sum},
                                                         {"None", Check_t::None}};

  const auto check_str = get_value_from_json<std::string>(j, "check");
  if (!find_item_in_map(check_type, check_str, CHECK_TYPE_MAP)) {
    CK_THROW_(Error_t::WrongInput, "Not supported check type: " + check_str);
  }
}

void parse_raw_arguments(const nlohmann::json& j, long long& num_samples,
                         long long& eval_num_samples, std::vector<size_t>& slot_size_array,
                         bool& float_label_dense) {
  if (has_key_(j, "slot_size_array")) {
    auto temp_array = get_json(j, "slot_size_array");
    if (!temp_array.is_array()) {
      CK_THROW_(Error_t::WrongInput, "!slot_size_array.is_array()");
    }
    long long slot_num = 0;
    for (auto j_slot_size : temp_array) {
      size_t slot_size = j_slot_size.get<size_t>();
      slot_size_array.push_back(slot_size);
      slot_num += slot_size;
    }
    MESSAGE_("vocabulary size: " + std::to_string(slot_num));

  } else {
    CK_THROW_(Error_t::WrongInput, "No such key in json file: slot_size_array.");
  }

  float_label_dense = get_value_from_json_soft<bool>(j, "float_label_dense", false);

  num_samples = get_value_from_json<long long>(j, "num_samples");
  eval_num_samples = get_value_from_json<long long>(j, "eval_num_samples");
}

int main(int argc, char* argv[]) {
  if (ArgParser::has_arg("help", argc, argv)) {
    std::cout << "To generate raw format: " << usage_str_raw << std::endl;
    std::cout << "To generate norm format: " << usage_str << std::endl;
    exit(-1);
  }

#ifdef ENABLE_MPI
  int provided;
  CK_MPI_THROW_(MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided));
#endif
  // Parsing the configure file:
  auto config_file = ArgParser::get_arg<std::string>("config-file", argc, argv);
  std::ifstream file(config_file);
  if (!file.is_open()) {
    std::cerr << "file.is_open() failed: " + config_file << std::endl;
  }
  nlohmann::json config;
  file >> config;
  file.close();

  DataReaderType_t format;
  int label_dim, dense_dim, num_slot;
  std::string source_data;
  std::string eval_source;
  bool i64_input_key;
  parse_common_arguments(config, format, label_dim, dense_dim, num_slot, source_data, eval_source,
                         i64_input_key);

  auto get_dist = [&]() {
    auto distri = ArgParser::get_arg<std::string>("distribution", argc, argv, "powerlow");
    // todo: not exactly, 'p/u' cannot represent "powerlow/unified".
    switch (distri[0]) {
      case 'p':
        use_long_tail = true;
        if (!ArgParser::has_arg("alpha", argc, argv)) {
          TAIL = ArgParser::get_arg<std::string>("long-tail", argc, argv, "medium");
          if (use_long_tail && TAIL_TYPE.find(TAIL) != std::end(TAIL_TYPE)) {
            if (TAIL == "long")
              alpha = 1.0;
            else if (TAIL == "medium")
              alpha = 3.0;
            else
              alpha = 5.0;
          }
        } else {
          alpha = ArgParser::get_arg<float>("alpha", argc, argv, 1.3);
        }
        break;
      case 'u':
        use_long_tail = false;
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "No such distribution: " + distri);
    }
  };

  std::vector<int> nnz_array =
      ArgParser::get_arg<std::vector<int>>("nnz-array", argc, argv, std::vector<int>());

  switch (format) {
    case DataReaderType_t::Norm: {
      std::string data_folder =
          ArgParser::get_arg<std::string>("data-folder", argc, argv, std::string("./"));
      // to do: add default value support
      std::vector<size_t> voc_size_array =
          ArgParser::get_arg<std::vector<size_t>>("voc-size-array", argc, argv);

      if (nnz_array.empty()) {
        for (size_t i = 0; i < voc_size_array.size(); i++) {
          nnz_array.push_back(1);
        }
      }

      NUM_FILES = ArgParser::get_arg<int>("files", argc, argv, 128);
      NUM_SAMPLES_PER_FILE = ArgParser::get_arg<int>("samples", argc, argv, 40960);

      get_dist();

      std::cout << "Configure File: " << config_file << ", Data Folder: " << data_folder
                << ", voc_size_array: " << vec_to_string(voc_size_array)
                << ", nnz array: " << vec_to_string(nnz_array) << ", #files: " << NUM_FILES
                << ", #samples per file: " << NUM_SAMPLES_PER_FILE
                << ", Use power law distribution: " << use_long_tail
                << ", alpha of power law: " << alpha << std::endl;

      Check_t check_type;
      parse_norm_arguments(config, check_type);

      check_make_dir(data_folder);

      if (check_type == Check_t::Sum) {
        if (i64_input_key) {  // I64 = long long
          data_generation_for_test2<long long, Check_t::Sum>(
              source_data, data_folder + "/train/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot,
              voc_size_array, label_dim, dense_dim, nnz_array, use_long_tail, alpha);
          data_generation_for_test2<long long, Check_t::Sum>(
              eval_source, data_folder + "/val/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot,
              voc_size_array, label_dim, dense_dim, nnz_array, use_long_tail, alpha);
        } else {  // I32 = unsigned int
          data_generation_for_test2<unsigned int, Check_t::Sum>(
              source_data, data_folder + "/train/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot,
              voc_size_array, label_dim, dense_dim, nnz_array, use_long_tail, alpha);
          data_generation_for_test2<unsigned int, Check_t::Sum>(
              eval_source, data_folder + "/val/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot,
              voc_size_array, label_dim, dense_dim, nnz_array, use_long_tail, alpha);
        }
      } else {
        if (i64_input_key) {  // I64 = long long
          data_generation_for_test2<long long, Check_t::None>(
              source_data, data_folder + "/train/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot,
              voc_size_array, label_dim, dense_dim, nnz_array, use_long_tail, alpha);
          data_generation_for_test2<long long, Check_t::None>(
              eval_source, data_folder + "/val/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot,
              voc_size_array, label_dim, dense_dim, nnz_array, use_long_tail, alpha);
        } else {  // I32 = unsigned int
          data_generation_for_test2<unsigned int, Check_t::None>(
              source_data, data_folder + "/train/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot,
              voc_size_array, label_dim, dense_dim, nnz_array, use_long_tail, alpha);
          data_generation_for_test2<unsigned int, Check_t::None>(
              eval_source, data_folder + "/val/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot,
              voc_size_array, label_dim, dense_dim, nnz_array, use_long_tail, alpha);
        }
      }
      break;
    }
    case DataReaderType_t::Raw: {
      get_dist();
      std::vector<size_t> slot_size_array;
      long long num_samples, eval_num_samples;
      bool float_label_dense;
      parse_raw_arguments(config, num_samples, eval_num_samples, slot_size_array,
                          float_label_dense);

      std::cout << "Configure File: " << config_file << ", Number of train samples: " << num_samples
                << ", Number of eval samples: " << eval_num_samples
                << ", Use power law distribution: " << use_long_tail
                << ", alpha of power law: " << alpha << std::endl;

      if (nnz_array.empty()) {
        for (size_t i = 0; i < slot_size_array.size(); i++) {
          nnz_array.push_back(1);
        }
      }

      if (slot_size_array.size() != nnz_array.size()) {
        CK_THROW_(Error_t::WrongInput,
                  "The length of slot size array  != nnz array in command line option");
      }

      if (file_exist(source_data)) {
        CK_THROW_(Error_t::WrongInput, source_data + " already exist!");
        exit(-1);
      }
      if (file_exist(eval_source)) {
        CK_THROW_(Error_t::WrongInput, eval_source + " already exist!");
        exit(-1);
      }

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

      if (i64_input_key) {  // I64 = long long
        // train data
        data_generation_for_raw<long long>(source_data, num_samples, label_dim, dense_dim,
                                           float_label_dense, slot_size_array, nnz_array,
                                           use_long_tail, alpha, nullptr);
        // eval data
        data_generation_for_raw<long long>(eval_source, eval_num_samples, label_dim, dense_dim,
                                           float_label_dense, slot_size_array, nnz_array,
                                           use_long_tail, alpha, nullptr);
      } else {
        // train data
        data_generation_for_raw<unsigned int>(source_data, num_samples, label_dim, dense_dim,
                                              float_label_dense, slot_size_array, nnz_array,
                                              use_long_tail, alpha, nullptr);
        // eval data
        data_generation_for_raw<unsigned int>(eval_source, eval_num_samples, label_dim, dense_dim,
                                              float_label_dense, slot_size_array, nnz_array,
                                              use_long_tail, alpha, nullptr);
      }

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
