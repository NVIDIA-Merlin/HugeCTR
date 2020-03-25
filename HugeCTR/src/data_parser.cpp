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

#include "HugeCTR/include/data_parser.hpp"

namespace HugeCTR {
void InputParser::write(long long num_index) {
  switch_io_mode(IOmode::write);
  for (long long i = 0; i < num_index; i++) {
    long long index = data_sim_->get_num();
    file_stream_.write(reinterpret_cast<char*>(&index), sizeof(long long));
  }
  return;
}

void InputParser::read(int num_index, long long* index) {
  switch_io_mode(IOmode::read);
  file_stream_.exceptions(std::ios::eofbit | std::ios::failbit | std::ios::badbit);
  try {
    file_stream_.read(reinterpret_cast<char*>(index), sizeof(long long) * num_index);
  } catch (std::ios_base::iostate e) {
    std::string err_string;
    if (e == std::ios::eofbit) err_string = "eofbit";
    if (e == std::ios::failbit) err_string = "failbit";
    if (e == std::ios::badbit) err_string = "badbit";
    std::cerr << "[HCDEBUG][Error] IOstream Error: file read failure (InputParser) " << err_string
              << std::endl;
    exit(-1);
  }
  return;
}

void ParameterParser::write(long long num_params) {
  switch_io_mode(IOmode::write);
  for (int i = 0; i < num_params; i++) {
    float param = data_sim_->get_num();
    file_stream_.write(reinterpret_cast<char*>(&param), sizeof(float));
  }
  return;
}

void ParameterParser::read(int num_params, float* params) {
  switch_io_mode(IOmode::read);
  try {
    file_stream_.read(reinterpret_cast<char*>(params), sizeof(float) * num_params);
    file_stream_.seekg(0, std::ios::beg);  // only for test
  } catch (std::fstream::failure e) {
    std::cerr << "[HCDEBUG][Error] IOstream Error: file read failure (ParameterParser)"
              << std::endl;
    exit(-1);
  }
  return;
}

void ParameterParser::fake_read(int num_params, float* params) {
  switch_io_mode(IOmode::read);
  // file_stream_.read(reinterpret_cast<char*>(params), sizeof(float)*num_params);
  // file_stream_.seekg(0, std::ios::beg);       //only for test
  for (int i = 0; i < num_params; i++) {
    params[i] = 0.f;
  }
  return;
}

}  // namespace HugeCTR
