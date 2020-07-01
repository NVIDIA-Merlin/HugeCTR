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
#include <fstream>
#include <iostream>
#include <ios>
#include <sstream>
#include <sys/stat.h>
#include <vector>

using namespace HugeCTR;

static std::string usage_str = "usage: ./criteo2hugectr in.txt dir/prefix file_list.txt [option:#keys for wide model]"; 
static const int N = 40960; //number of samples per data file
static int KEYS_WIDE_MODEL = 0;
static const int KEYS_DENSE_MODEL = 26;
static const int dense_dim = 13;
typedef unsigned int T;
static const long long label_dim = 1;
static int SLOT_NUM = 26;
const int RANGE[] = {0,1460,2018,337396,549106,549411,549431,561567,562200,562203,613501,618803,951403,954582,954609,966800,1268011,1268021,1272862,1274948,1274952,1599225,1599242,1599257,1678991,1679087,1737709};

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

int main(int argc, char* argv[]){
  

  const std::string tmp_file_list_name("file_list.tmp");

  if (argc != 4 && argc != 5){
    std::cout << usage_str << std::endl;
    exit(-1);
  }

  if (argc == 5){
    KEYS_WIDE_MODEL = atoi(argv[4]);
    SLOT_NUM += 1;
  }
  else{
    std::cout << "Checking the range of each key..." << std::endl;
  }

  //open txt file
  std::ifstream txt_file(argv[1], std::ifstream::binary);
  if(!txt_file.is_open()){
    std::cerr << "Cannot open argv[1]" << std::endl;
  }
  //create a data file under prefix
  std::string data_prefix(argv[2]);
  std::string directory;
  const size_t last_slash_idx = data_prefix.rfind('/');
  if (std::string::npos != last_slash_idx){
      directory = data_prefix.substr(0, last_slash_idx);
  }
  check_make_dir(directory);
  int file_counter = 0;
  std::ofstream file_list_tmp(tmp_file_list_name, std::ofstream::out);
  if(!file_list_tmp.is_open()){
    std::cerr << "Cannot open file_list.tmp" << std::endl;
  }

  do{
    std::string data_file_name(data_prefix + std::to_string(file_counter) + ".data");
    std::ofstream data_file(data_file_name, std::ofstream::binary);
    file_list_tmp << (data_file_name + "\n");
    std::cout << data_file_name << std::endl;
    if(!data_file.is_open()){
      std::cerr << "Cannot open data_file" << std::endl;
    }

    DataWriter<Check_t::Sum> data_writer(data_file);
    DataSetHeader header = {1, static_cast<long long>(N), label_dim, dense_dim, static_cast<long long>(SLOT_NUM), 0, 0, 0
                            };
    data_writer.append(reinterpret_cast<char*>(&header), sizeof(DataSetHeader));
    data_writer.write();
    //read N lines
    int i = 0;
    for(; i < N; i++){
      //read a line
      std::string line;
      std::getline(txt_file, line);
      //if end
      if(txt_file.eof()){
        txt_file.close();
        data_file.seekp(std::ios_base::beg);
        DataSetHeader last_header = {1, static_cast<long long>(i), label_dim, dense_dim,
                                     static_cast<long long>(SLOT_NUM), 0, 0, 0};
	data_writer.append(reinterpret_cast<char*>(&last_header), sizeof(DataSetHeader));
	data_writer.write();
        data_file.close();
        file_list_tmp.close();
        //redirect
        {
          std::ifstream tmp(tmp_file_list_name);
          if(!tmp.is_open()){
            std::cerr << "Cannot open " << tmp_file_list_name << std::endl;
          }

          std::cout << "Opening " << argv[3] << std::endl;
          std::ofstream file_list(argv[3], std::ofstream::out);
          if(!file_list.is_open()){
            std::cerr << "Cannot open " << argv[3] << std::endl;
          }

          
          file_list << (std::to_string(file_counter+1) + "\n");
          file_list << tmp.rdbuf();

          tmp.close();
          file_list.close();
        }

        return 0;
      }
      std::vector<std::string> vec_string;
      split(line, ' ', vec_string);
      if(vec_string.size() != (unsigned int)(KEYS_WIDE_MODEL+KEYS_DENSE_MODEL+dense_dim+label_dim)) //first one is label
        {
          std::cerr << "vec_string.size() != KEYS_WIDE_MODEL+KEYS_DENSE_MODEL+dense_dim+label_dim" << std::endl;
          std::cerr << line << std::endl;
          exit(-1);
        }
#ifndef NDEBUG
      std::cout << std::endl;
#endif
      for(int j = 0; j < dense_dim+label_dim; j++){
	float label_dense = std::stod(vec_string[j]);
	data_writer.append(reinterpret_cast<char*>(&label_dense), sizeof(float));
#ifndef NDEBUG
      std::cout << label_dense << ' ';
#endif
      }
      
      if(KEYS_WIDE_MODEL != 0){
	data_writer.append(reinterpret_cast<char*>(&KEYS_WIDE_MODEL), sizeof(int));
	for(int j = dense_dim+label_dim; j < KEYS_WIDE_MODEL+dense_dim+label_dim; j++){
	  T key = static_cast<T>(std::stoll(vec_string[j]));
	  data_writer.append(reinterpret_cast<char*>(&key), sizeof(T));
#ifndef NDEBUG
	  std::cout << key << ',';
#endif
	}
      }
      int nnz = 1;
      for(int j = KEYS_WIDE_MODEL+dense_dim+label_dim; j <  KEYS_WIDE_MODEL+KEYS_DENSE_MODEL+dense_dim+label_dim; j++){
        T key = static_cast<T>(std::stoll(vec_string[j]));
	data_writer.append(reinterpret_cast<char*>(&nnz), sizeof(int));
	data_writer.append(reinterpret_cast<char*>(&key), sizeof(T));
	if(KEYS_WIDE_MODEL == 0){
	  if(key < RANGE[j-dense_dim-label_dim] || key > RANGE[j+1-dense_dim-label_dim]){
	    std::cout << key << " in feature:" << j << " our of range:" << RANGE[j] << "," << RANGE[j+1] << std::endl;
	  }
	}
#ifndef NDEBUG
	std::cout << key << ',';
#endif
      }
      data_writer.write();
    }
    data_file.close();
    file_counter ++;
#ifndef NDEBUG
    std::cout << std::endl;
    if(file_counter > 2){
      break;
    }
#endif
  }while(1);

  return 0;
}
