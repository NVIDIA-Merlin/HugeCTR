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
#include <gpu_resource.hpp>
#include <tensor2.hpp>

namespace HugeCTR {

class UniformGenerator {
 public:
  template <typename T>
  static void fill(T* ptr, size_t num_elements, T a, T b, size_t sm_count,
                   const curandGenerator_t& generator, const cudaStream_t& stream);
};

class HostUniformGenerator {
 public:
  template <typename T>
  static void fill(Tensor2<T>& tensor, T a, T b, const curandGenerator_t& gen);
};

class NormalGenerator {
 public:
  template <typename T>
  static void fill(Tensor2<T>& tensor, float mean, float stddev, size_t sm_count,
                   const curandGenerator_t& generator, const cudaStream_t& stream);
};

class HostNormalGenerator {
 public:
  template <typename T>
  static void fill(Tensor2<T>& tensor, float mean, float stddev, const curandGenerator_t& gen);
};

class DataSimulator {
 public:
  virtual ~DataSimulator() {}
  virtual void fill(Tensor2<float>& tensor, const curandGenerator_t& gen) = 0;
};

/*
 * Wrap of Zeros and Ones initializer.
 */
class ConstantDataSimulator : public DataSimulator {
 public:
  ConstantDataSimulator(float value) : value_(value) {}

  void fill(Tensor2<float>& tensor, const curandGenerator_t& gen) override;

 private:
  float value_;
};

class UniformDataSimulator : public DataSimulator {
 public:
  UniformDataSimulator(float min, float max) : min_(min), max_(max) {}

  void fill(Tensor2<float>& tensor, const curandGenerator_t& gen) override;

 private:
  float min_;
  float max_;
};

class GaussianDataSimulator : public DataSimulator {
 public:
  GaussianDataSimulator(float mu, float sigma, float min, float max) : mu_(mu), sigma_(sigma) {}

  void fill(Tensor2<float>& tensor, const curandGenerator_t& gen) override;

 private:
  float mu_;
  float sigma_;
};

namespace data_simu {
enum class Mode_t {
  Fan_in,   // number of input units in the weight tensor
  Fan_out,  // number of output units in the weight tensor
  Fan_avg   // average of the numbers of input and output units
};

enum class Distribution_t { Uniform, Norm };
}  // namespace data_simu

class VarianceScalingSimulator : public DataSimulator {
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
        simulator_.reset(new UniformDataSimulator(-1 * limit, limit));
        break;
      }
      case data_simu::Distribution_t::Norm: {
        if (truncated_) {
          float stddev = sqrt(scale_) / .87962566103423978;
          simulator_.reset(new GaussianDataSimulator(0, stddev, -2 * stddev, 2 * stddev));
        } else {
          float stddev = sqrt(scale_);
          simulator_.reset(new GaussianDataSimulator(0, stddev, -10 * stddev, 10 * stddev));
        }
        break;
      }
      default: {
        ERROR_MESSAGE_("distribution should be one of {Uniform, Norm}.");
        break;
      }
    }
  }

  void fill(Tensor2<float>& tensor, const curandGenerator_t& gen) override {
    simulator_->fill(tensor, gen);
  }

 private:
  std::unique_ptr<DataSimulator> simulator_;
  float scale_;
  bool truncated_;
};
}  // namespace HugeCTR
