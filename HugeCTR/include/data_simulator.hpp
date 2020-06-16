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
#include <random>
#include <algorithm>

namespace HugeCTR {

class RandomEngine {
  std::random_device rd_;
  unsigned int seed_;
  std::mt19937 gen_;

 public:
  RandomEngine() : rd_(), seed_(rd_()), gen_(seed_) {
    MESSAGE_("Initial seed is " + std::to_string(seed_));
  }

  template <typename T>
  void set_seed(const T& t) {
    gen_.seed(t);
  }

  template <typename T>
  typename T::result_type get_num(T& t) {
    return t(gen_);
  }

  template <typename It>
  void shuffle(It begin, It end) {
    std::shuffle(begin, end, gen_);
  }

  static RandomEngine& get() {
    static RandomEngine engine;
    return engine;
  }
};

template <typename T>
class DataSimulator {
 public:
  virtual T get_num() = 0;
};

template <typename T>
class UnifiedDataSimulator : public DataSimulator<T> {
 public:
  UnifiedDataSimulator(T min, T max) : dis_(min, max) {}

  T get_num() final { return RandomEngine::get().get_num(dis_); }

 private:
  std::uniform_real_distribution<T> dis_;
};

template <>
class UnifiedDataSimulator<long long> : public DataSimulator<long long> {
 public:
  UnifiedDataSimulator(long long min, long long max) : dis_(min, max) {}

  long long get_num() final { return RandomEngine::get().get_num(dis_); }

 private:
  std::uniform_int_distribution<long long> dis_;
};

template <>
class UnifiedDataSimulator<int> : public DataSimulator<int> {
 public:
  UnifiedDataSimulator(int min, int max) : dis_(min, max) {}

  int get_num() final { return RandomEngine::get().get_num(dis_); }

 private:
  std::uniform_int_distribution<int> dis_;
};

template <>
class UnifiedDataSimulator<unsigned int> : public DataSimulator<unsigned int> {
 public:
  UnifiedDataSimulator(unsigned int min, unsigned int max) : dis_(min, max) {}

  unsigned int get_num() final { return RandomEngine::get().get_num(dis_); }

 private:
  std::uniform_int_distribution<unsigned int> dis_;
};

template <typename T>
class GaussianDataSimulator : public DataSimulator<T> {
 public:
  GaussianDataSimulator(float mu, float sigma, T min, T max)
      : dis_(mu, sigma), min_(min), max_(max) {
    if (min_ > max_) {
      ERROR_MESSAGE_("min_ > max_");
    }
  }

  T get_num() final {
    while (1) {
      T tmp = RandomEngine::get().get_num(dis_);
      if (tmp <= max_ && tmp >= min_) return tmp;
      if (tmp < min_) continue;
      if (tmp > max_) continue;
      ERROR_MESSAGE_("wrong path");
    }
    return 0;
  }

 private:
  std::normal_distribution<T> dis_;
  T min_;
  T max_;
};

}  // namespace HugeCTR