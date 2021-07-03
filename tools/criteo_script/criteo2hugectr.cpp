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

#include <sys/stat.h>

#include <fstream>
#include <ios>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "HugeCTR/include/data_generator.hpp"

using namespace HugeCTR;

static std::string usage_str =
    "usage: ./criteo2hugectr in.txt dir/prefix file_list.txt [option:#keys for wide model,default "
    "is 0] [option: Number of files in each file_list.txt,default is 0(all in one file)]";
static const int N = 40960;  // number of samples per data file
static int KEYS_WIDE_MODEL = 0;
static const int KEYS_DENSE_MODEL = 26;
static const int dense_dim = 13;
typedef unsigned int T;
static const long long label_dim = 1;
static int SLOT_NUM = 26;
static int FILELIST_LENGTH = 0;  // number of files in each file_list.txt
std::unordered_set<unsigned int> keyset;

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

int main(int argc, char *argv[]) {
  const std::string tmp_file_list_name("file_list.tmp");

  if (argc != 4 && argc != 5 && argc != 6) {
    std::cout << usage_str << std::endl;
    exit(-1);
  }

  if (argc == 5) {
    if (atoi(argv[4]) != 0) {
      KEYS_WIDE_MODEL = atoi(argv[4]);
      SLOT_NUM += 1;
    } else {
      std::cout << "Checking the range of each key..." << std::endl;
    }
  }

  if (argc == 6) {
    if (atoi(argv[4]) != 0) {
      KEYS_WIDE_MODEL = atoi(argv[4]);
      SLOT_NUM += 1;
      std::cout << "slot_num for w&D is:" << SLOT_NUM << std::endl;
    }
    if (atoi(argv[5]) > 0) {
      FILELIST_LENGTH = atoi(argv[5]);
      std::cout << FILELIST_LENGTH << std::endl;
    } else {
      std::cerr << "The number of files in file_list should greater than 0 (default is 0)..." << std::endl;
    }
  }

  // open txt file
  std::ifstream txt_file(argv[1], std::ifstream::binary);
  if (!txt_file.is_open()) {
    std::cerr << "Cannot open argv[1]" << std::endl;
  }
  // create a data file under prefix
  std::string data_prefix(argv[2]);
  std::string directory;
  const size_t last_slash_idx = data_prefix.rfind('/');
  if (std::string::npos != last_slash_idx) {
    directory = data_prefix.substr(0, last_slash_idx);
  }
  check_make_dir(directory);
  // check file_list.txt prefix
  std::string file_name(argv[3]);
  if (file_name.find(".") == std::string::npos) {
    std::cerr << "Please provide aviable file_list with extension(.txt) " << std::endl;
    exit(-1);
  }
  const size_t last_point_idx = file_name.rfind('.');
  std::string file_name_prefix = file_name.substr(0, last_point_idx);
  std::string file_name_postfix = file_name.substr(last_point_idx);

  int file_counter = 0;
  std::ofstream file_list_tmp(tmp_file_list_name, std::ofstream::out);

  if (!file_list_tmp.is_open()) {
    std::cerr << "Cannot open file_list.tmp" << std::endl;
  }

  std::string keyset_name(file_name_prefix + ".keyset");
  if (FILELIST_LENGTH > 0) {
    keyset_name = file_name_prefix + "." + std::to_string(file_counter) + ".keyset";
  }
  std::ofstream keyset_file(keyset_name, std::ofstream::binary);
  DataWriter<Check_t::None> keyset_writer(keyset_file);

  do {
    std::string data_file_name(data_prefix + std::to_string(file_counter) + ".data");
    std::ofstream data_file(data_file_name, std::ofstream::binary);
    file_list_tmp << (data_file_name + "\n");
    std::cout << data_file_name << std::endl;
    if (!data_file.is_open()) {
      std::cerr << "Cannot open data_file" << std::endl;
    }

    DataWriter<Check_t::Sum> data_writer(data_file);
    DataSetHeader header = {
        1, static_cast<long long>(N), label_dim, dense_dim, static_cast<long long>(SLOT_NUM), 0, 0,
        0};
    data_writer.append(reinterpret_cast<char *>(&header), sizeof(DataSetHeader));
    data_writer.write();
    // read N lines
    int i = 0;
    for (; i < N; i++) {
      // read a line
      std::string line;
      std::getline(txt_file, line);
      // if end
      if (txt_file.eof()) {
        txt_file.close();
        if (i == 0 && keyset.size() == 0) {
          return 0;
        }
        data_file.seekp(std::ios_base::beg);
        DataSetHeader last_header = {1,
                                     static_cast<long long>(i),
                                     label_dim,
                                     dense_dim,
                                     static_cast<long long>(SLOT_NUM),
                                     0,
                                     0,
                                     0};
        data_writer.append(reinterpret_cast<char *>(&last_header), sizeof(DataSetHeader));
        data_writer.write();
        data_file.close();
        std::cout << "last keyset size is:" << keyset.size() << std::endl;
        keyset_writer.write();
        keyset_file.close();

        file_list_tmp.close();
        // redirect
        {
          std::ifstream tmp(tmp_file_list_name);
          if (!tmp.is_open()) {
            std::cerr << "Cannot open " << tmp_file_list_name << std::endl;
          }

          std::cout << "Opening " << argv[3] << std::endl;
          // std::ofstream file_list(argv[3], std::ofstream::out);

          std::string file_list_name = file_name;

          if (FILELIST_LENGTH > 0) {
            file_list_name = file_name_prefix + "." +
                             std::to_string((file_counter / FILELIST_LENGTH)) + file_name_postfix;
          }

          std::ofstream file_list(file_list_name, std::ofstream::out);

          if (!file_list.is_open()) {
            std::cerr << "Cannot open " << argv[3] << std::endl;
          }

          // file_list << (std::to_string(file_counter + 1) + "\n");
          if (i != 0) {
            file_counter++;
          }
          if (FILELIST_LENGTH == 0) {
            file_list << (std::to_string(file_counter) + "\n");
          } else if (file_counter % FILELIST_LENGTH != 0 && FILELIST_LENGTH != 1) {
            file_list << (std::to_string(file_counter % FILELIST_LENGTH) + "\n");
          } else {
            file_list << (std::to_string(FILELIST_LENGTH) + "\n");
          }

          file_list << tmp.rdbuf();

          tmp.close();
          file_list.close();
        }

        return 0;
      }
      std::vector<std::string> vec_string;
      split(line, ' ', vec_string);
      if (vec_string.size() != (unsigned int)(KEYS_WIDE_MODEL + KEYS_DENSE_MODEL + dense_dim +
                                              label_dim))  // first one is label
      {
        std::cerr << "vec_string.size() != KEYS_WIDE_MODEL+KEYS_DENSE_MODEL+dense_dim+label_dim"
                  << std::endl;
        std::cerr << line << std::endl;
        exit(-1);
      }
#ifndef NDEBUG
      std::cout << std::endl;
#endif
      for (int j = 0; j < dense_dim + label_dim; j++) {
        float label_dense = std::stod(vec_string[j]);
        data_writer.append(reinterpret_cast<char *>(&label_dense), sizeof(float));
#ifndef NDEBUG
        std::cout << label_dense << ' ';
#endif
      }

      if (KEYS_WIDE_MODEL != 0) {
        data_writer.append(reinterpret_cast<char *>(&KEYS_WIDE_MODEL), sizeof(int));
        for (int j = dense_dim + label_dim; j < KEYS_WIDE_MODEL + dense_dim + label_dim; j++) {
          T key = static_cast<T>(std::stoll(vec_string[j]));
          data_writer.append(reinterpret_cast<char *>(&key), sizeof(T));
          if (keyset.find(key) == keyset.end()) {
            keyset.insert(key);
            keyset_writer.append(reinterpret_cast<char *>(&key), sizeof(T));
          }

#ifndef NDEBUG
          std::cout << key << ',';
#endif
        }
      }
      int nnz = 1;
      for (int j = KEYS_WIDE_MODEL + dense_dim + label_dim;
           j < KEYS_WIDE_MODEL + KEYS_DENSE_MODEL + dense_dim + label_dim; j++) {
        T key = static_cast<T>(std::stoll(vec_string[j]));
        data_writer.append(reinterpret_cast<char *>(&nnz), sizeof(int));
        data_writer.append(reinterpret_cast<char *>(&key), sizeof(T));
        if (keyset.find(key) == keyset.end()) {
          keyset.insert(key);
          keyset_writer.append(reinterpret_cast<char *>(&key), sizeof(T));
        }
#ifndef NDEBUG
        std::cout << key << ',';
#endif
      }
      data_writer.write();
    }
    data_file.close();

    if (FILELIST_LENGTH > 0 && (file_counter + 1) % FILELIST_LENGTH == 0) {
      int NOFile = (file_counter / FILELIST_LENGTH);
      file_list_tmp.close();
      std::ifstream tmp(tmp_file_list_name);
      if (!tmp.is_open()) {
        std::cerr << "Cannot open " << tmp_file_list_name << std::endl;
      }

      std::cout << "Opening " << argv[3] << std::endl;

      std::string file_list_name =
          file_name_prefix + "." + std::to_string(NOFile) + file_name_postfix;

      std::ofstream file_list(file_list_name, std::ofstream::out);
      if (!file_list.is_open()) {
        std::cerr << "Cannot open " << argv[3] << std::endl;
      }

      file_list << (std::to_string(FILELIST_LENGTH) + "\n");
      file_list << tmp.rdbuf();

      tmp.close();
      file_list.close();

      std::cout << std::to_string(NOFile) << " keyset size is: " << keyset.size() << std::endl;
      keyset.clear();
      keyset_writer.write();
      keyset_file.close();
      keyset_name = file_name_prefix + "." + std::to_string(NOFile + 1) + ".keyset";
      keyset_file.open(keyset_name, std::ofstream::binary);
      DataWriter<Check_t::None> keyset_writer(keyset_file);

      std::cout << "reopen file_list_tmp" << std::endl;
      file_list_tmp.open(tmp_file_list_name);
      if (!file_list_tmp.is_open()) {
        std::cerr << "Cannot open file_list.tmp during iteration" << std::endl;
      }
    }
    file_counter++;

#ifndef NDEBUG
    std::cout << std::endl;
    if (file_counter > 2) {
      break;
    }
#endif
  } while (1);

  return 0;
}

