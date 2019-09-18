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

/**
 * DEBUG: g++ -g -o criteo2hugectr -std=c++11 criteo2hugectr.cpp 
 * RELEASE: g++ -DNDEBUG -o criteo2hugectr -std=c++11 criteo2hugectr.cpp 
 */

#include <fstream>
#include <iostream>
#include <ios>
#include <sstream>
#include <sys/stat.h>
#include <vector>

static std::string usage_str = "usage: ./criteo2hugectr in.txt dir/prefix file_list.txt"; 

static const int N = 40960; //number of samples per data file
static const int KEYS_PER_SAMPLE = 39;
static const int SLOT_NUM = 1;
typedef long long T;
static const long long label_dim = 1;
static const int voc_size = 1603616;

inline void check_make_dir(std::string finalpath){
  if (mkdir(finalpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
    {
      if( errno == EEXIST ) {
	std::cout << (finalpath + " exist") << std::endl;
      } else {
	std::cerr << ("cannot create" + finalpath + ": unexpected error") << std::endl;
      }
    }
}


std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

typedef struct DataSetHeader_{
  long long number_of_records; //the number of samples in this data file
  long long label_dim; //dimension of label
  long long slot_num;
  long long reserved; //reserved for future use
} DataSetHeader;


int main(int argc, char* argv[]){
  const std::string tmp_file_list_name("file_list.tmp");

  if (argc != 4){
    std::cout << usage_str << std::endl;
    exit(-1);
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

    DataSetHeader header = {static_cast<long long>(N), label_dim, static_cast<long long>(SLOT_NUM), 0};
    data_file.write(reinterpret_cast<char*>(&header), sizeof(DataSetHeader));
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
	DataSetHeader last_header = {static_cast<long long>(i), label_dim, static_cast<long long>(SLOT_NUM), 0};
	data_file.write(reinterpret_cast<char*>(&last_header), sizeof(DataSetHeader));
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

	  
	  file_list << (std::to_string(file_counter) + "\n");
	  file_list << tmp.rdbuf();

	  tmp.close();
	  file_list.close();
	}

	return 0;
      }
      std::vector<std::string> vec_string;
      split(line, ' ', vec_string);
      if(vec_string.size() != KEYS_PER_SAMPLE+1) //first one is label
	{
	  std::cerr << "vec_string.size() != KEYS_PER_SAMPLE+1" << std::endl;
	  std::cerr << line << std::endl;
	  exit(-1);
	}
      int label = std::stoi(vec_string[0]);
#ifndef NDEBUG
      std::cout << std::endl;
      std::cout << label << ' ';
#endif
      data_file.write(reinterpret_cast<char*>(&label), sizeof(int));
      std::vector<T> slots[SLOT_NUM];
      for(int j = 0; j < KEYS_PER_SAMPLE; j++){
	T key = static_cast<T>(std::stoi(vec_string[j+1]));
	int slot_id = key%SLOT_NUM;
	slots[slot_id].push_back(key);
      }
      for(int j = 0; j < SLOT_NUM; j++){
	int nnz = slots[j].size();
	data_file.write(reinterpret_cast<char*>(&nnz), sizeof(int));
	for(T key : slots[j]){
	  data_file.write(reinterpret_cast<char*>(&key), sizeof(T));
#ifndef NDEBUG
	  std::cout << j << ',' << key << ' ';
#endif
	}
      }
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
