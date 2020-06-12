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

#pragma once

#include <fstream>
#include <memory>
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/data_simulator.hpp"

namespace HugeCTR {

enum class IOmode { read, write };

template <class T>
class DataParser {
 public:
  DataParser(std::string file_name, DataSimulator<T>* data_sim)
      : data_sim_(data_sim),
        file_stream_(file_name, std::fstream::trunc | std::fstream::in | std::fstream::out |
                                    std::fstream::binary),
        file_name_(file_name){};  // open by both read and write
  void switch_io_mode(IOmode io_mode) {
    if (io_mode_ != io_mode) {
      // fflush
      // if(io_mode == IOmode::read){
      // 	std::cout << "w->r" <<std::endl;
      // }
      file_stream_.flush();
      // seek to the begining
      file_stream_.seekg(0, std::ios::beg);
      io_mode_ = io_mode;
    }
  }

 protected:
  std::unique_ptr<DataSimulator<T>> data_sim_;
  std::fstream file_stream_;

 private:
  std::string file_name_;
  IOmode io_mode_{IOmode::read};
};

class InputParser : public DataParser<long long> {
 public:
  InputParser() = delete;
  using DataParser<long long>::DataParser;
  void write(long long num_index);
  void read(int num_index, long long* index);
};

class ParameterParser : public DataParser<float> {
 public:
  ParameterParser() = delete;
  using DataParser<float>::DataParser;
  void write(long long num_params);
  void read(int num_params, float* params);
  void fake_read(int num_params, float* params);  // only for test
};

}  // namespace HugeCTR
