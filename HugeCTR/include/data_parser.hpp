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
#include <iostream>
#include <memory>
#include <random>
#include "HugeCTR/include/common.hpp"

namespace HugeCTR {

enum class DistributeType { Unified, Gaussian };

enum class IOmode { read, write };

template <class T>
class DataSimulator {
 public:
  DataSimulator(DistributeType type) : type_(type){};
  virtual T get_num() = 0;
  DistributeType get_distribute_type() const { return type_; };

 private:
  DistributeType type_;
};

template <class T>
class UnifiedDataSimulator : public DataSimulator<T> {
 public:
  UnifiedDataSimulator(T min, T max)
      : DataSimulator<T>(DistributeType::Unified), rd_(), gen_(rd_()), dis_(min, max) {}
  T get_num() final { return static_cast<T>(dis_(gen_)); }

 private:
  std::random_device rd_;
  std::mt19937 gen_;
  std::uniform_real_distribution<> dis_;
};

template <>
class UnifiedDataSimulator<long long> : public DataSimulator<long long> {
 public:
  UnifiedDataSimulator(long long min, long long max)
      : DataSimulator<long long>(DistributeType::Unified), rd_(), gen_(rd_()), dis_(min, max) {}
  long long get_num() final { return dis_(gen_); }

 private:
  std::random_device rd_;
  std::mt19937 gen_;
  std::uniform_int_distribution<long long> dis_;
};

template <>
class UnifiedDataSimulator<int> : public DataSimulator<int> {
 public:
  UnifiedDataSimulator(int min, int max) //both included
      : DataSimulator<int>(DistributeType::Unified), rd_(), gen_(rd_()), dis_(min, max) {}
  int get_num() final { return dis_(gen_); }

 private:
  std::random_device rd_;
  std::mt19937 gen_;
  std::uniform_int_distribution<int> dis_;
};

template <class T>
class GaussianDataSimulator : public DataSimulator<T> {
 public:
  GaussianDataSimulator(float mu, float sigma, T min, T max)
      : DataSimulator<T>(DistributeType::Gaussian),
        rd_(),
        gen_(rd_()),
        dis_(mu, sigma),
        min_(min),
        max_(max) {
    if (min_ > max_) {
      ERROR_MESSAGE_("min_ > max_");
    }
  }
  T get_num() final {
    while (1) {
      T tmp = static_cast<T>(dis_(gen_));
      if (tmp <= max_ && tmp >= min_) return tmp;
      if (tmp < min_) continue;
      if (tmp > max_) continue;
      ERROR_MESSAGE_("wrong path");
    }
    return 0;
  }

 private:
  std::random_device rd_;
  std::mt19937 gen_;
  std::normal_distribution<> dis_;
  T min_;
  T max_;
};

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
