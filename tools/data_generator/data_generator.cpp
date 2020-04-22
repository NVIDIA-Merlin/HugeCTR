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

#include "HugeCTR/include/utils.hpp"
#include "HugeCTR/include/parser.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <ios>
#include <sstream>
#include <sys/stat.h>
#include <vector>

using namespace HugeCTR;

static std::string usage_str = "usage: ./data_generator your_config.json data_folder vocabulary_size max_nnz [option:#files] [option:#samples per file]"; 
static int NUM_FILES = 128;
static int NUM_SAMPLES_PER_FILE = 40960;

int main(int argc, char* argv[]){
  if (argc != 5 && argc != 6 && argc != 7){
    std::cout << usage_str << std::endl;
    exit(-1);
  }

  std::string config_file = argv[1];
  std::string data_folder = argv[2];
  size_t vocabulary_size = std::stoul(argv[3]);
  int max_nnz = atoi(argv[4]);
  if (argc == 6 || argc == 7){
    NUM_FILES = atoi(argv[5]);
  }
  if (argc == 7){
    NUM_SAMPLES_PER_FILE = atoi(argv[6]);
  }
  
  std::cout << "Configure File: " << config_file << ", Data Folder: " << data_folder << ", Vocabulary Size: " << vocabulary_size << ", Max NNZ:" << max_nnz << ", #files: " << NUM_FILES << ", #samples per file: " << NUM_SAMPLES_PER_FILE << std::endl;

  //Parsing the configure file:
  std::ifstream file(config_file);
  if (!file.is_open()) {
    std::cerr << "file.is_open() failed: " + config_file << std::endl;
  }
  nlohmann::json config;
  file >> config;
  file.close();

  auto j_layers_array = config.find("layers").value();

  int label_dim = 0, dense_dim = 0;
  Check_t check_type;
  std::string source_data;
  std::vector<DataReaderSparseParam> data_reader_sparse_param_array;
  std::string eval_source;
  std::string top_strs_label, top_strs_dense;
  std::vector<std::string> sparse_names;
  std::map<std::string, SparseInput<long long>> sparse_input_map;
  parse_data_layer_helper(j_layers_array[0], label_dim, dense_dim, check_type, source_data, data_reader_sparse_param_array,
		   eval_source, top_strs_label, top_strs_dense, sparse_names, sparse_input_map);

  int num_slot = 0;
  for(auto& param: data_reader_sparse_param_array){
    if(param.slot_num*max_nnz > param.max_feature_num){
      std::cerr << "Error: max_nnz * slot_num cannot be larger than max_feature_num in config file!" << std::endl;
      exit(-1);
    }
    num_slot+=param.slot_num;
  }
  if(check_type == Check_t::Sum){
    data_generation<long long, Check_t::Sum>(source_data, data_folder + "/train/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot, vocabulary_size, label_dim, dense_dim, max_nnz);
    data_generation<long long, Check_t::Sum>(eval_source, data_folder + "/val/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot, vocabulary_size, label_dim, dense_dim, max_nnz);
  }
  else{
    data_generation<long long, Check_t::None>(source_data, data_folder + "/train/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot, vocabulary_size, label_dim, dense_dim, max_nnz);
    data_generation<long long, Check_t::None>(eval_source, data_folder + "/val/gen_", NUM_FILES, NUM_SAMPLES_PER_FILE, num_slot, vocabulary_size, label_dim, dense_dim, max_nnz);    
  }

  return 0;
}
