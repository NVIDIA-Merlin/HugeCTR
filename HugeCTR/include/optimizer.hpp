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

#include <common.hpp>
#include <core23/buffer_channel_helpers.hpp>
#include <core23/data_type.hpp>
#include <core23/tensor_container.hpp>
#include <general_buffer2.hpp>
#include <gpu_learning_rate_scheduler.hpp>
#include <gpu_resource.hpp>
#include <network_buffer_channels.hpp>
#include <network_helpers.hpp>
#include <optional>

namespace HugeCTR {

using WeightTensors = core23::TensorContainer<float, 1, 1>;
using WeightHalfTensors = core23::TensorContainer<__half, 1, 1>;
template <typename T>
using WgradTensors = core23::TensorContainer<T, 1, 1>;

struct FtrlOptHyperParams {
  static constexpr size_t num_parameters_per_weight = 2;

  float beta = 0.0f;
  float lambda1 = 0.0f;
  float lambda2 = 0.0f;

  bool operator==(const FtrlOptHyperParams& other) const {
    return (beta == other.beta) && (lambda1 == other.lambda1) && (lambda2 == other.lambda2);
  }

  bool operator!=(const FtrlOptHyperParams& other) const { return !(*this == other); }
};

struct AdamOptHyperParams {
  static constexpr size_t num_parameters_per_weight = 2;

  uint64_t times = 0;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1e-7f;

  inline float bias() const {
    return std::sqrt(1 - std::pow(beta2, times)) / (1 - std::pow(beta1, times));
  }

  bool operator==(const AdamOptHyperParams& other) const {
    return (times == other.times) && (beta1 == other.beta1) && (beta2 == other.beta2) &&
           (epsilon == other.epsilon);
  }

  bool operator!=(const AdamOptHyperParams& other) const { return !(*this == other); }
};

struct RMSPropOptHyperParams {
  static constexpr size_t num_parameters_per_weight = 1;

  float beta = 0.9f;
  float epsilon = 1e-7f;

  bool operator==(const RMSPropOptHyperParams& other) const {
    return (beta == other.beta) && (epsilon == other.epsilon);
  }

  bool operator!=(const RMSPropOptHyperParams& other) const { return !(*this == other); }
};

struct AdaGradOptHyperParams {
  static constexpr size_t num_parameters_per_weight = 1;

  float initial_accu_value = 0.f;
  float epsilon = 1e-7f;

  bool operator==(const AdaGradOptHyperParams& other) const {
    return (initial_accu_value == other.initial_accu_value) && (epsilon == other.epsilon);
  }

  bool operator!=(const AdaGradOptHyperParams& other) const { return !(*this == other); }
};

struct MomentumSGDOptHyperParams {
  static constexpr size_t num_parameters_per_weight = 1;

  float factor = 0.1f;

  bool operator==(const MomentumSGDOptHyperParams& other) const { return (factor == other.factor); }

  bool operator!=(const MomentumSGDOptHyperParams& other) const { return !(*this == other); }
};

struct NesterovOptHyperParams {
  static constexpr size_t num_parameters_per_weight = 1;

  float mu = 0.9f;

  bool operator==(const NesterovOptHyperParams& other) const { return (mu == other.mu); }

  bool operator!=(const NesterovOptHyperParams& other) const { return !(*this == other); }
};

struct SGDOptHyperParams {
  static constexpr size_t num_parameters_per_weight = 0;

  bool atomic_update = false;

  bool operator==(const SGDOptHyperParams& other) const {
    return (atomic_update == other.atomic_update);
  }

  bool operator!=(const SGDOptHyperParams& other) const { return !(*this == other); }
};

// TODO: use union type should be better ???
struct OptHyperParams {
  FtrlOptHyperParams ftrl;
  AdamOptHyperParams adam;
  RMSPropOptHyperParams rmsprop;
  AdaGradOptHyperParams adagrad;
  MomentumSGDOptHyperParams momentum;
  NesterovOptHyperParams nesterov;
  SGDOptHyperParams sgd;

  bool operator==(const OptHyperParams& other) const {
    return (ftrl == other.ftrl) && (adam == other.adam) && (rmsprop == other.rmsprop) &&
           (adagrad == other.adagrad) && (momentum == other.momentum) &&
           (nesterov == other.nesterov) && (sgd == other.sgd);
  }

  bool operator!=(const OptHyperParams& other) const { return !(*this == other); }
};

// Comment: Maybe it's better to seperate this class as std::variant<SGDParams, AdamParams, ...> and
// use function overload to deal with different params for different update algorithm
struct OptParams {
  Optimizer_t optimizer;
  float lr;
  OptHyperParams hyperparams;
  Update_t update_type;
  float scaler;

  inline static size_t num_parameters_per_weight(Optimizer_t opt_type) {
    switch (opt_type) {
      case Optimizer_t::Ftrl:
        return FtrlOptHyperParams::num_parameters_per_weight;
      case Optimizer_t::Adam:
        return AdamOptHyperParams::num_parameters_per_weight;
      case Optimizer_t::RMSProp:
        return RMSPropOptHyperParams::num_parameters_per_weight;
      case Optimizer_t::AdaGrad:
        return AdaGradOptHyperParams::num_parameters_per_weight;
      case Optimizer_t::MomentumSGD:
        return MomentumSGDOptHyperParams::num_parameters_per_weight;
      case Optimizer_t::Nesterov:
        return NesterovOptHyperParams::num_parameters_per_weight;
      case Optimizer_t::SGD:
        return SGDOptHyperParams::num_parameters_per_weight;
      default:
        HCTR_OWN_THROW(Error_t::NotInitialized, "OptParams not correctly initialized.");
        return 0;
    }
  }

  inline size_t num_parameters_per_weight() const { return num_parameters_per_weight(optimizer); }

  bool operator==(const OptParams& other) const {
    return (optimizer == other.optimizer) && (lr == other.lr) &&
           (hyperparams == other.hyperparams) && (update_type == other.update_type) &&
           (scaler == other.scaler);
  }

  bool operator!=(const OptParams& other) const { return !(*this == other); }
};

class OptParamsPy {
 public:
  Optimizer_t optimizer;
  Update_t update_type;
  OptHyperParams hyperparams;
  bool initialized;

  OptParamsPy();
  OptParamsPy(Optimizer_t optimizer_type, Update_t update_t, OptHyperParams opt_hyper_params);
};

/**
 * @brief Base class for all optimizers
 */
class Optimizer {
 public:
  /**
   * Helper to create a speicifed Optimizer object
   */
  template <typename T>
  static std::unique_ptr<Optimizer> Create(const OptParams& params,
                                           const Tensor2<float>& weight_main,
                                           const Tensor2<__half>& weight_main_half,
                                           const Tensor2<T>& wgrad, const float scaler,
                                           const std::shared_ptr<BufferBlock2<float>>& opt_buff,
                                           const std::shared_ptr<GPUResource>& gpu_resource,
                                           bool use_mixed_precision);
  /**
   * Helper to create a speicifed Optimizer object
   */
  template <typename T>
  static std::unique_ptr<Optimizer> Create(const OptParams& params,
                                           std::vector<core23::Tensor> weight_tensors,
                                           std::vector<core23::Tensor> weight_half_tensors,
                                           std::vector<core23::Tensor> wgrade_tensors,
                                           const float scaler,
                                           const std::shared_ptr<GPUResource>& gpu_resource,
                                           bool use_mixed_precision);

  /**
   * Constructor of Optimizer.
   * @param weight_main weights to be updated
   * @param wgrad gradient for weights
   * @param device_id the id of GPU where update kernel is launched
   * @param learning_rate learning rate
   */
  Optimizer(const Tensor2<float>& weight_main, const std::shared_ptr<GPUResource>& gpu_resource,
            float learning_rate, float scaler)
      : weight_main_(weight_main),

        weight_tensors_({}),
        gpu_resource_(gpu_resource),
        lr_(learning_rate),
        scaler_(scaler),
        optimizer_type_(Optimizer_t::DEFAULT) {
    if (lr_ < 0.) {
      HCTR_OWN_THROW(Error_t::WrongInput, "lr < 0");
    }
  }
  /*
   * Constructor of Optimizer with new Tensor
   * @param weight_tensors TensorContainer of all the layers' weights
   * @param gpu_resource GPU where update kernel is launched
   * @param learning_rate learning rate
   * @param scaler scaler for the gradients
   */
  Optimizer(std::optional<WeightTensors> weight_tensors,
            const std::shared_ptr<GPUResource>& gpu_resource, float learning_rate, float scaler)
      : weight_tensors_(weight_tensors),
        gpu_resource_(gpu_resource),
        lr_(learning_rate),
        scaler_(scaler),
        optimizer_type_(Optimizer_t::DEFAULT) {
    if (lr_ < 0.) {
      HCTR_OWN_THROW(Error_t::WrongInput, "lr < 0");
    }
  }

  virtual ~Optimizer() = default;

  virtual void initialize() {}

  /**
   * update the weights using gradient
   * @param stream cuda stream used by update kernel
   */
  virtual void update() = 0;

  virtual std::vector<core23::Tensor> get_opt_state_tensors() { return {}; }

  /**
   * update the learning rate
   * @param lr the learning rate
   */
  void set_learning_rate(float lr) {
    if (gpu_learning_rate_scheduler_ != nullptr) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "set_learning_rate cannot be used together with set_learing_rate_scheduler");
    }

    if (lr < 0) {
      HCTR_OWN_THROW(Error_t::WrongInput, "lr < 0");
    }
    lr_ = lr;
  }

  void set_learning_rate_scheduler(std::shared_ptr<GpuLearningRateScheduler>& sched) {
    gpu_learning_rate_scheduler_ = sched;
  }

  const Optimizer_t& get_optimizer_type() { return optimizer_type_; }

 protected:
  Tensor2<float> weight_main_;
  std::optional<WeightTensors> weight_tensors_;
  std::shared_ptr<GPUResource> gpu_resource_;
  float lr_;  // learning rate
  const float scaler_;
  Optimizer_t optimizer_type_;

  std::shared_ptr<GpuLearningRateScheduler> gpu_learning_rate_scheduler_;

  int get_device_id() const { return gpu_resource_->get_device_id(); }
};

struct SparseEmbeddingHashParams;
template <typename TypeEmbeddingComp>
struct OptimizerTensor {
  Tensor2<TypeEmbeddingComp> opt_z_tensors_;  // FTRL z variable.
  Tensor2<TypeEmbeddingComp> opt_n_tensors_;  // FTRL n variable.
  Tensor2<TypeEmbeddingComp>
      opt_m_tensors_; /**< The mi variable storage for adam optimizer in the update_params(). */
  Tensor2<TypeEmbeddingComp>
      opt_v_tensors_; /**< The vi variable storage for adam optimizer in the update_params(). */
  Tensor2<uint64_t> opt_prev_time_tensors_; /**< The previous update time storage for lazy adam
                                                  in update_params(). */
  Tensor2<TypeEmbeddingComp> opt_momentum_tensors_; /**< The momentum variable storage
                                           for the momentum optimizer in the update_params(). */
  Tensor2<TypeEmbeddingComp> opt_accm_tensors_;     /**< The accm variable storage for the
                                                         nesterov optimizer in the update_params(). */
};

template <typename TypeHashKey, typename TypeEmbeddingComp>
class EmbeddingOptimizer {
  Tensor2<void> temp_storage_encode_tensors_;

  Tensor2<void> temp_storage_sort_tensors_; /**< The temp memory for the CUB lib sorting
                                                      API in update_params(). */

  Tensor2<void> temp_storage_scan_tensors_; /**< The temp memory for the CUB lib scaning API
                                                      in update_params(). */

  Tensor2<TypeHashKey> sample_id_tensors_; /**< The temp memory to store the sample ids of hash
                                              table value in      update_params(). */

  Tensor2<TypeHashKey> sample_id_sort_tensors_;   /**< The temp memory to store the sorted sample
                                                     ids of hash table value in update_params(). */
  Tensor2<size_t> hash_value_index_sort_tensors_; /**< The temp memory to store the sorted hash
                                                        table value indexes in update_params(). */

  Tensor2<size_t> hash_value_index_sort_unique_tensors_;

  Tensor2<uint32_t> hash_value_index_count_tensors_;
  Tensor2<uint32_t> new_hash_value_flag_tensors_;
  Tensor2<uint32_t> hash_value_flag_sumed_tensors_;
  Tensor2<uint32_t>
      hash_value_index_count_offset_tensors_; /**< The temp memory to store the offset of each count
                                                 of hash table value indexes in update_params(). */

  Tensor2<uint32_t> hash_value_index_count_counter_tensors_; /**< The temp memory to store the
                                                                counter of the count of hash table
                                                                value indexes in update_params(). */
  SparseEmbeddingHashParams& param;

 public:
  OptimizerTensor<TypeEmbeddingComp> opt_tensors_;

  EmbeddingOptimizer(size_t max_vocabulary_size_per_gpu_, SparseEmbeddingHashParams& param,
                     const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& buf);

  void initialize(const GPUResource& local_gpu);

  void reset(GPUResource const& local_gpu) { initialize(local_gpu); }

  void update(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
              size_t max_vocabulary_size_per_gpu, size_t nnz,
              const Tensor2<TypeHashKey>& row_offset, Tensor2<size_t>& hash_value_index,
              const Tensor2<TypeEmbeddingComp>& wgrad, Tensor2<float>& hash_table_value,
              size_t sm_count, cudaStream_t stream);
};

}  // namespace HugeCTR
