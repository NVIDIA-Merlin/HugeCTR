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

#pragma once
#include <common.hpp>
#include <general_buffer2.hpp>
#include <gpu_resource.hpp>

namespace HugeCTR {

struct AdamOptHyperParams {
  uint64_t times = 0;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1e-7f;
};

struct AdaGradParams {
  float initial_accu_value = 0.f;
  float epsilon = 1e-7f;
};

struct MomentumSGDOptHyperParams {
  float factor = 0.1f;
};

struct NesterovOptHyperParams {
  float mu = 0.9f;
};

struct SGDOptHyperParams {
  bool atomic_update = false;
};

// TODO: use union type should be better ???
struct OptHyperParams {
  AdamOptHyperParams adam;
  MomentumSGDOptHyperParams momentum;
  NesterovOptHyperParams nesterov;
  SGDOptHyperParams sgd;
  AdaGradParams adagrad;
};

struct OptParams {
  Optimizer_t optimizer;
  float lr;
  OptHyperParams hyperparams;
  Update_t update_type;
  float scaler;
};

// 
class OptParamsPy {
public:
  Optimizer_t optimizer;
  Update_t update_type;
  OptHyperParams hyperparams;
  bool initialized;
  OptParamsPy(Optimizer_t optimizer_type, Update_t update_t, OptHyperParams opt_hyper_params);
  OptParamsPy();
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
  static std::unique_ptr<Optimizer> Create(
      const OptParams& params, const Tensor2<float>& weight_main, const Tensor2<T>& wgrad,
      const float scaler,
      const std::shared_ptr<BufferBlock2<T>>& opt_buff,
      const std::shared_ptr<GPUResource>& gpu_resource);

  /**
   * Constructor of Optimizer.
   * @param weight_main weights to be updated
   * @param wgrad gradient for weights
   * @param device_id the id of GPU where update kernel is launched
   * @param learning_rate learning rate
   */
  Optimizer(const Tensor2<float>& weight_main, 
            const std::shared_ptr<GPUResource>& gpu_resource, float learning_rate, float scaler)
      : weight_main_(weight_main),
        gpu_resource_(gpu_resource),
        lr_(learning_rate),
        scaler_(scaler) {
    if (lr_ < 0.) {
      CK_THROW_(Error_t::WrongInput, "lr < 0");
    }
  }

  virtual ~Optimizer() = default;

  virtual void initialize() {}

  /**
   * update the weights using gradient
   * @param stream cuda stream used by update kernel
   */
  virtual void update() = 0;

  /**
   * update the learning rate
   * @param lr the learning rate
   */
  void set_learning_rate(float lr) {
    if (lr < 0) {
      CK_THROW_(Error_t::WrongInput, "lr < 0");
    }
    lr_ = lr;
  }

 protected:
  Tensor2<float> weight_main_;
  std::shared_ptr<GPUResource> gpu_resource_;
  float lr_;  // learning rate
  const float scaler_;

  int get_device_id() const { return gpu_resource_->get_device_id(); }
};

struct SparseEmbeddingHashParams;
template <typename TypeEmbeddingComp>
struct OptimizerTensor {
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
class EmbeddingOptimizer{
  
  Tensor2<void> temp_storage_encode_tensors_;

  Tensor2<void> temp_storage_sort_tensors_;     /**< The temp memory for the CUB lib sorting
                                                          API in update_params(). */

  Tensor2<void> temp_storage_scan_tensors_; /**< The temp memory for the CUB lib scaning API
                                                      in update_params(). */
  
  Tensor2<TypeHashKey> sample_id_tensors_; /**< The temp memory to store the sample ids of hash
                                              table value in      update_params(). */
  
  Tensor2<TypeHashKey> sample_id_sort_tensors_; /**< The temp memory to store the sorted sample
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
  SparseEmbeddingHashParams &param;
public:
  OptimizerTensor<TypeEmbeddingComp> opt_tensors_;

  EmbeddingOptimizer(size_t max_vocabulary_size_per_gpu_, SparseEmbeddingHashParams &param, const std::shared_ptr<GeneralBuffer2<CudaAllocator>> &buf);
  
  void initialize(const GPUResource &local_gpu);
  
  void update(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size,
    size_t max_vocabulary_size_per_gpu,  size_t nnz,
    const Tensor2<TypeHashKey> &row_offset, Tensor2<size_t> &hash_value_index,
    const Tensor2<TypeEmbeddingComp> &wgrad,
    Tensor2<float> &hash_table_value, size_t sm_count, cudaStream_t stream);

};

}  // namespace HugeCTR
