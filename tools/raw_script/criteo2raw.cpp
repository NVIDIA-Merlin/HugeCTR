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

#include "HugeCTR/include/utils.hpp"
#include <fstream>
#include <iostream>
#include <ios>
#include <sstream>
#include <sys/stat.h>
#include <vector>
using namespace HugeCTR;

static std::string usage_str = "usage: ./criteo2raw in.txt out.bin"; 

static const int dense_dim = 13;
static const int label_dim = 1;
static const int SLOT_NUM = 26;

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

int main(int argc, char* argv[]){
  if (argc != 3){
    std::cout << usage_str << std::endl;
    exit(-1);
  }

  //open txt file
  std::ifstream txt_file(argv[1], std::ifstream::binary);
  if(!txt_file.is_open()){
    std::cerr << "Cannot open argv[1]" << std::endl;
  }

  std::ofstream out_file(argv[2], std::ofstream::out);
  int num_samples = 0;
  do{
    std::string line;
    std::getline(txt_file, line);
    if(txt_file.eof()){
      txt_file.close();
      out_file.close();
      std::cout << "#samples: " << num_samples << std::endl;
      break;
    }
    std::vector<std::string> vec_string;
    split(line, ' ', vec_string);
    if(vec_string.size() != dense_dim+label_dim+SLOT_NUM){
      std::cout << "Error: vec_string.size() != dense_dim+label_dim+SLOT_NUM" << std::endl;
      exit(-1);
    }
    for(int j = 0; j < dense_dim+label_dim; j++){
      float label_dense = std::stod(vec_string[j]);
      out_file.write(reinterpret_cast<char*>(&label_dense), sizeof(float));
    }
    for(int j = dense_dim+label_dim; j < dense_dim+label_dim+SLOT_NUM; j++){
      int sparse = std::stod(vec_string[j]);
      out_file.write(reinterpret_cast<char*>(&sparse), sizeof(int));
    }
    num_samples++;
  }while(1);
  return 0;
}
