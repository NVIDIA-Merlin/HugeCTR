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
#include <math.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <random>

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

/*
 * Wrap of Zeros and Ones initializer.
 */
template <typename T>
class SingleDataSimulator : public DataSimulator<T> {
 public:
  SingleDataSimulator(std::function<T()> func) : func_(func) {}

  T get_num() final { return func_(); }

 private:
  std::function<T()> func_;
};

namespace data_simu {
/*
 * Variance Scaling Initializer
 * Which can be used to simulate Xavier(glorot) initialization.
 */
enum class Mode_t {
  Fan_in,   // number of input units in the weight tensor
  Fan_out,  // number of output units in the weight tensor
  Fan_avg   // average of the numbers of input and output units
};

enum class Distribution_t { Uniform, Norm };

}  // namespace data_simu

template <typename T>
class VarianceScalingSimulator : public DataSimulator<T> {
 public:
  VarianceScalingSimulator(float scale, data_simu::Mode_t mode,
                           data_simu::Distribution_t distribution, float in_dim, float out_dim,
                           bool truncated = true)
      : simulator_(nullptr), scale_(scale), truncated_(truncated) {
    switch (mode) {
      case data_simu::Mode_t::Fan_in: {
        scale_ /= std::max(1.0f, in_dim);
        break;
      }
      case data_simu::Mode_t::Fan_out: {
        scale_ /= std::max(1.0f, out_dim);
        break;
      }
      case data_simu::Mode_t::Fan_avg: {
        scale_ /= std::max(1.0f, (in_dim + out_dim) / 2.f);
        break;
      }
      default: {
        ERROR_MESSAGE_("mode should be one of {Fan_in, Fan_out, Fan_avg}.");
        break;
      }
    }

    switch (distribution) {
      case data_simu::Distribution_t::Uniform: {
        float limit = sqrt(3.f * scale_);
        simulator_.reset(new UnifiedDataSimulator<T>(-1 * limit, limit));
        break;
      }
      case data_simu::Distribution_t::Norm: {
        if (truncated_) {
          float stddev = sqrt(scale_) / .87962566103423978;
          simulator_.reset(new GaussianDataSimulator<T>(0, stddev, -2 * stddev, 2 * stddev));
        } else {
          float stddev = sqrt(scale_);
          simulator_.reset(new GaussianDataSimulator<T>(0, stddev, -10 * stddev, 10 * stddev));
        }
        break;
      }
      default: {
        ERROR_MESSAGE_("distribution should be one of {Uniform, Norm}.");
        break;
      }
    }
  }

  T get_num() final { return simulator_->get_num(); }

 private:
  std::unique_ptr<DataSimulator<T>> simulator_;
  float scale_;
  bool truncated_;
};

}  // namespace HugeCTR