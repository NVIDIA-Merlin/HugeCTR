/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <base/debug/logger.hpp>
#include <common.hpp>
#include <fstream>
#include <memory>
#include <random>

namespace HugeCTR {

template <typename T>
class IKeySimulator {
 public:
  virtual ~IKeySimulator() {}
  virtual T get_num() = 0;
};

template <typename T>
class FloatStandardUniformKeySimulator {
 public:
  FloatStandardUniformKeySimulator(T min, T max) : gen_(std::random_device()()), dis_(min, max) {}

  T get_num() { return dis_(gen_); }

 private:
  std::mt19937 gen_;
  std::uniform_real_distribution<T> dis_;
};

template <typename T>
class IntUniformKeySimulator : public IKeySimulator<T> {
 public:
  IntUniformKeySimulator(T min, T max) : gen_(std::random_device()()), dis_(min, max) {}

  T get_num() override { return dis_(gen_); }

 private:
  std::mt19937 gen_;
  std::uniform_int_distribution<T> dis_;
};

template <typename T>
class IntPowerLawKeySimulator : public IKeySimulator<T> {
 public:
  IntPowerLawKeySimulator(T min, T max, float alpha)
      : gen_(std::random_device()()), dis_(0, 1), alpha_(alpha) {
    min_ = 1.0;
    max_ = max - min + 1.0;
    offset_ = min - 1.0;  // to handle the case min_ <= 0
  }

  T get_num() override {
    double x = dis_(gen_);
    double y = (pow((pow(max_, 1 - alpha_) - pow(min_, 1 - alpha_)) * x + pow(min_, 1 - alpha_),
                    1.0 / (1.0 - alpha_)));
    return static_cast<T>(round(y) + offset_);
  }

 private:
  std::mt19937 gen_;
  std::uniform_real_distribution<float> dis_;
  float alpha_;  // requiring alpha_ > 0 and alpha_ != 1.0
  double min_, max_, offset_;
};

template <typename T>
void batch_key_generator_by_powerlaw(T* data, size_t batch_size, size_t embedding_size,
                                     float alpha = 0.0) {
  HCTR_CHECK_HINT(embedding_size > 0, "Invalid Embedding size: less or equal to 0");
  IntPowerLawKeySimulator<T> powerlaw_simulator(0, embedding_size, alpha);
  for (size_t idx = 0; idx < batch_size; idx++) {
    data[idx] = powerlaw_simulator.get_num();
  }
}

template <typename T>
void batch_key_generator_by_hotkey(T* data, size_t batch_size, size_t embedding_size,
                                   float hot_key_percentage = 0.2, float hot_key_coverage = 0.8) {
  HCTR_CHECK_HINT(embedding_size > 0, "Invalid Embedding size: less or equal to 0");
  HCTR_CHECK_HINT(hot_key_percentage >= 0.0 && hot_key_percentage <= 1.0,
                  "Hot key percentage should be in range [0.0, 1.0]");
  HCTR_CHECK_HINT(hot_key_coverage >= 0.0 && hot_key_coverage <= 1.0,
                  "Hot key coverage should be in range [0.0, 1.0]");
  IntUniformKeySimulator<T> hotkey_simulator(0, embedding_size * hot_key_percentage);
  IntUniformKeySimulator<T> nonhotkey_simulator(embedding_size * hot_key_percentage,
                                                embedding_size);
  FloatStandardUniformKeySimulator<float> standard_simulator(0, 1);
  for (size_t idx = 0; idx < batch_size; idx++) {
    if (standard_simulator.get_num() < hot_key_coverage) {
      data[idx] = hotkey_simulator.get_num();
    } else {
      data[idx] = nonhotkey_simulator.get_num();
    }
  }
}

template <typename T>
void key_vector_generator_by_powerlaw(std::vector<T>& data, size_t batch_size,
                                      size_t embedding_size, float alpha = 0.0) {
  HCTR_CHECK_HINT(embedding_size > 0, "Invalid Embedding size: less or equal to 0");
  IntPowerLawKeySimulator<T> powerlaw_simulator(0, embedding_size, alpha);
  for (size_t idx = 0; idx < batch_size; idx++) {
    data.push_back(powerlaw_simulator.get_num());
  }
}

template <typename T>
void key_vector_generator_by_hotkey(std::vector<T>& data, size_t batch_size, size_t embedding_size,
                                    float hot_key_percentage = 0.2, float hot_key_coverage = 0.8) {
  HCTR_CHECK_HINT(embedding_size > 0, "Invalid Embedding size: less or equal to 0");
  HCTR_CHECK_HINT(hot_key_percentage >= 0.0 && hot_key_percentage <= 1.0,
                  "Hot key percentage should be in range [0.0, 1.0]");
  HCTR_CHECK_HINT(hot_key_coverage >= 0.0 && hot_key_coverage <= 1.0,
                  "Hot key coverage should be in range [0.0, 1.0]");
  IntUniformKeySimulator<T> hotkey_simulator(0, embedding_size * hot_key_percentage);
  IntUniformKeySimulator<T> nonhotkey_simulator(embedding_size * hot_key_percentage,
                                                embedding_size);
  FloatStandardUniformKeySimulator<float> standard_simulator(0, 1);
  for (size_t idx = 0; idx < batch_size; idx++) {
    if (standard_simulator.get_num() < hot_key_coverage) {
      data.push_back(hotkey_simulator.get_num());
    } else {
      data.push_back(nonhotkey_simulator.get_num());
    }
  }
}
}  // namespace HugeCTR