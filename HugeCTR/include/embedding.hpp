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
#include <functional>
#include <vector>
#include "HugeCTR/include/data_reader.hpp"
#include "HugeCTR/include/gpu_resource.hpp"
#include "HugeCTR/include/tensor.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

class IEmbedding {
 public:
  virtual ~IEmbedding() {}
  virtual void forward() = 0;
  virtual void backward() = 0;
  virtual void update_params() = 0;
  virtual void init_params() = 0;
  virtual void upload_params_to_device(std::ifstream& weight_stream) = 0;
  virtual void download_params_to_host(std::ofstream& weight_stream) = 0;
  virtual void set_learning_rate(float lr) = 0;
  virtual size_t get_params_num() = 0;
  virtual const ITensors get_output_tensors() const = 0;
  virtual void check_overflow() const = 0;
};

/**
 * @brief The base class of embedding layers.
 *
 * This class has responsibility to get the tensors
 * from the former layer and to allocate buffers for the output tensors of the embedding
 * layer. As a base class, it declares all the virtual functions that need to be implemented
 * in embedding layers' training process, including forward propagation and backward
 * propagation. The forward propagation is corresponding to the API forward(). The backward
 * propagation is divided into 2-stage APIs: backward() and update_params(). This class also
 * provides the operations for uploading/downloading embedding models (which is also known as
 * embedding tables) to/from GPUs from/to host file stream, which are named as
 * upload_params_to_device() and download_params_to_host().
 */
template <typename TypeKey, typename TypeEmbeddingComp>
class Embedding : public IEmbedding {
 protected:
  GeneralBuffers<TypeEmbeddingComp> output_buffers_; /**< The buffer for storing output tensors. */
  Tensors<TypeEmbeddingComp> output_tensors_;        /**< The output tensors. */
  const Tensors<TypeKey> row_offsets_tensors_; /**< The row_offsets tensors of the input data. */
  const Tensors<TypeKey> value_tensors_;       /**< The value tensors of the input data. */
  std::shared_ptr<GPUResourceGroup> device_resources_; /**< The GPU device resources. */
  const int batchsize_; /**< The batch size of the input data for the current training process. */
  const TypeEmbeddingComp scaler_;

 public:
  /**
   * The constructor of Embedding class.
   * @param row_offsets_tensors the row_offsets tensors of the input data(refer to row offset vector
   * in sparse matrix CSR format).
   * @param value_tensors the value tensors of the input data(refer to value vector in sparse matrix
   * CSR format).
   * @param batchsize the batch size of the input data
   * @param slot_num the number of slots of the hash table
   * @param embedding_vec_size the dim size of the embedding feature vector.
   * @param gpu_resource_group the GPU device resource group
   * @param scaler scaler factor for mixed precision
   */
  Embedding(const Tensors<TypeKey>& row_offsets_tensors, const Tensors<TypeKey>& value_tensors,
            size_t batchsize, size_t slot_num, size_t embedding_vec_size,
            const std::shared_ptr<GPUResourceGroup>& gpu_resource_group, TypeEmbeddingComp scaler);

  virtual Embedding* clone_eval(const Tensors<TypeKey>& row_offsets_tensors,
                                const Tensors<TypeKey>& value_tensors, size_t batchsize,
                                const std::shared_ptr<GPUResourceGroup>& gpu_resource_group) {
    return nullptr;
  }

  /**
   * The declaration for indicating that there is no default copy construtor in this class.
   */
  Embedding(const Embedding& C) = delete;
  /**
   * The destructor of Embedding class.
   */
  virtual ~Embedding() {}
  /**
   * The forward propagation of embedding layer.
   */
  virtual void forward() = 0;
  /**
   * The first stage of backward propagation of embedding layer,
   * which only computes the wgrad by the dgrad from the top layer.
   */
  virtual void backward() = 0;
  /**
   * The second stage of backward propagation of embedding layer, which
   * updates the embedding table weights by wgrad(from backward()) and
   * optimizer.
   */
  virtual void update_params() = 0;
  /**
   * Initialize the embedding table
   */
  virtual void init_params() = 0;
  /**
   * Read the embedding table from the weight_stream on the host, and
   * upload it onto multi-GPUs global memory.
   * @param weight_stream the host file stream for reading data from.
   */
  virtual void upload_params_to_device(std::ifstream& weight_stream) = 0;
  /**
   * Download the embedding table from multi-GPUs global memroy to CPU memory
   * and write it to the weight_stream on the host.
   * @param weight_stream the host file stream for writing data to.
   */
  virtual void download_params_to_host(
      std::ofstream& weight_stream) = 0;  // please refer to file format definition of HugeCTR
  /**
   * Get the total size of embedding tables on all GPUs.
   */
  virtual size_t get_params_num() = 0;
  /**
   * Return the output tensors.
   */
  const ITensors get_output_tensors() const override {
    ITensors ts;
    for (auto& t : output_tensors_) {
      ts.emplace_back(t);
    }
    return ts;
  }

  // only used for results check
  /**
   * Get the forward() results from GPUs and copy them to the host pointer
   * embedding_feature. This function is only used for unit test.
   * @param embedding_feature the host pointer for storing the forward()
   * results.
   */
  virtual void get_forward_results(TypeEmbeddingComp* embedding_feature) = 0;
  /**
   * Get the backward() results from GPUs and copy them to the host pointer
   * wgrad. The wgrad on each GPU should be the same. This function is only
   * used for unit test.
   * @param wgrad the host pointer for stroing the backward() results.
   * @param devIndex the GPU device id.
   */
  virtual void get_backward_results(TypeEmbeddingComp* wgrad, int devIndex) = 0;
  /**
   * Get the update_params() results(the hash table, including hash_table_keys
   * and hash_table_values) from GPUs and copy them to the host pointers.
   * This function is only used for unit test.
   * @param hash_table_key the host pointer for stroing the hash table keys.
   * @param hash_table_value the host pointer for stroing the hash table values.
   */
  virtual void get_update_params_results(TypeKey* hash_table_key, float* hash_table_value) = 0;

  virtual void set_learning_rate(float lr) = 0;
};


template <typename TypeKey, typename TypeEmbeddingComp>
Embedding<TypeKey, TypeEmbeddingComp>::Embedding(
    const Tensors<TypeKey>& row_offsets_tensors, const Tensors<TypeKey>& value_tensors,
    size_t batchsize, size_t slot_num, size_t embedding_vec_size,
    const std::shared_ptr<GPUResourceGroup>& gpu_resource_group, TypeEmbeddingComp scaler)
    : row_offsets_tensors_(row_offsets_tensors),
      value_tensors_(value_tensors),
      device_resources_(gpu_resource_group),
      batchsize_(batchsize),
      scaler_(scaler) {
  try {
    // Error check
    if (batchsize < 1 || slot_num < 1 || embedding_vec_size < 1) {
      CK_THROW_(Error_t::WrongInput, "batchsize < 1 || slot_num < 1 || embedding_vec_size < 1");
    }

    if (embedding_vec_size > 1024) {
      CK_THROW_(Error_t::WrongInput,
                "the embedding_vec_size can not be more than 1024 in embedding layer");
    }

    const auto& device_list = device_resources_->get_device_list();
    size_t gpu_count = device_list.size();
    if (row_offsets_tensors.size() != gpu_count || value_tensors.size() != gpu_count) {
      CK_THROW_(Error_t::WrongInput,
                "either row_offsets_tensors.size() or value_tensors.size() isn't gpu_count");
    }

    assert(output_buffers_.empty());
    for (size_t i = 0; i < gpu_count; i++) {
      std::shared_ptr<GeneralBuffer<TypeEmbeddingComp>> buff(
          new GeneralBuffer<TypeEmbeddingComp>());
      const size_t batchsize_per_device = batchsize / device_resources_->get_total_gpu_count();
      std::vector<size_t> output_dims = {batchsize_per_device, slot_num, embedding_vec_size};
      output_tensors_.emplace_back(
          new Tensor<TypeEmbeddingComp>(output_dims, buff, TensorFormat_t::HSW));
      buff->init(device_list[i]);
      output_buffers_.emplace_back(buff);
    }

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
  return;
}

template<typename T>
struct AdamOptHyperParams {  // TODO: move to optimizer
  uint64_t times = 0;
  float alpha_t = 0.0f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1e-6f;
  T* m_ptr = nullptr;
  T* v_ptr = nullptr;
};

template<typename T>
struct MomentumSGDOptHyperParams {
  float factor = 0.1f;
  T* momentum_ptr = nullptr;
};

template<typename T>
struct NesterovOptHyperParams {
  float mu = 0.9f;
  T* accm_ptr = nullptr;
};

// TODO: use union type should be better ???
template <typename TypeEmbeddingComp>
struct OptHyperParams {
  AdamOptHyperParams<TypeEmbeddingComp> adam;
  MomentumSGDOptHyperParams<TypeEmbeddingComp> momentum;
  NesterovOptHyperParams<TypeEmbeddingComp> nesterov;
};

template <typename TypeEmbeddingComp>
struct OptParams {
  Optimizer_t optimizer;
  float lr;
  OptHyperParams<TypeEmbeddingComp> hyperparams;
  bool global_update;
  float scaler;
};


template <typename TypeEmbeddingComp>
struct SparseEmbeddingHashParams {
  size_t batch_size;       // batch size
  size_t max_vocabulary_size_per_gpu; // max row number of hash table for each gpu
  std::vector<size_t> slot_size_array; // max row number for each slot
  size_t embedding_vec_size;                // col number of hash table value
  size_t max_feature_num;                   // max feature number of all input samples of all slots
  size_t slot_num;                          // slot number
  int combiner;                             // 0-sum, 1-mean
  OptParams<TypeEmbeddingComp> opt_params;  // optimizer params
};

// Embedding should be register here
struct EmbeddingCreator {
  typedef long long TYPE_1;
  typedef unsigned int TYPE_2;

  static Embedding<TYPE_1, float>* create_distributed_sparse_embedding_hash(
      const Tensors<TYPE_1>& row_offsets_tensors, const Tensors<TYPE_1>& value_tensors,
      SparseEmbeddingHashParams<float> embedding_params,
      const std::shared_ptr<GPUResourceGroup>& gpu_resource_group);

  static Embedding<TYPE_2, float>* create_distributed_sparse_embedding_hash(
      const Tensors<TYPE_2>& row_offsets_tensors, const Tensors<TYPE_2>& value_tensors,
      SparseEmbeddingHashParams<float> embedding_params,
      const std::shared_ptr<GPUResourceGroup>& gpu_resource_group);

  static Embedding<TYPE_1, __half>* create_distributed_sparse_embedding_hash(
      const Tensors<TYPE_1>& row_offsets_tensors, const Tensors<TYPE_1>& value_tensors,
      SparseEmbeddingHashParams<__half> embedding_params,
      const std::shared_ptr<GPUResourceGroup>& gpu_resource_group);

  static Embedding<TYPE_2, __half>* create_distributed_sparse_embedding_hash(
      const Tensors<TYPE_2>& row_offsets_tensors, const Tensors<TYPE_2>& value_tensors,
      SparseEmbeddingHashParams<__half> embedding_params,
      const std::shared_ptr<GPUResourceGroup>& gpu_resource_group);

  static Embedding<TYPE_1, float>* create_localized_sparse_embedding_hash(
      const Tensors<TYPE_1>& row_offsets_tensors, const Tensors<TYPE_1>& value_tensors,
      SparseEmbeddingHashParams<float> embedding_params, const std::string plan_file,
      const std::shared_ptr<GPUResourceGroup>& gpu_resource_group);

  static Embedding<TYPE_2, float>* create_localized_sparse_embedding_hash(
      const Tensors<TYPE_2>& row_offsets_tensors, const Tensors<TYPE_2>& value_tensors,
      SparseEmbeddingHashParams<float> embedding_params, const std::string plan_file,
      const std::shared_ptr<GPUResourceGroup>& gpu_resource_group);

  static Embedding<TYPE_1, __half>* create_localized_sparse_embedding_hash(
      const Tensors<TYPE_1>& row_offsets_tensors, const Tensors<TYPE_1>& value_tensors,
      SparseEmbeddingHashParams<__half> embedding_params, const std::string plan_file,
      const std::shared_ptr<GPUResourceGroup>& gpu_resource_group);

  static Embedding<TYPE_2, __half>* create_localized_sparse_embedding_hash(
      const Tensors<TYPE_2>& row_offsets_tensors, const Tensors<TYPE_2>& value_tensors,
      SparseEmbeddingHashParams<__half> embedding_params, const std::string plan_file,
      const std::shared_ptr<GPUResourceGroup>& gpu_resource_group);

  static Embedding<TYPE_1, float>* create_localized_sparse_embedding_one_hot(
      const Tensors<TYPE_1>& row_offsets_tensors, const Tensors<TYPE_1>& value_tensors,
      SparseEmbeddingHashParams<float> embedding_params, const std::string plan_file,
      const std::shared_ptr<GPUResourceGroup>& gpu_resource_group);

  static Embedding<TYPE_2, float>* create_localized_sparse_embedding_one_hot(
      const Tensors<TYPE_2>& row_offsets_tensors, const Tensors<TYPE_2>& value_tensors,
      SparseEmbeddingHashParams<float> embedding_params, const std::string plan_file,
      const std::shared_ptr<GPUResourceGroup>& gpu_resource_group);

  static Embedding<TYPE_1, __half>* create_localized_sparse_embedding_one_hot(
      const Tensors<TYPE_1>& row_offsets_tensors, const Tensors<TYPE_1>& value_tensors,
      SparseEmbeddingHashParams<__half> embedding_params, const std::string plan_file,
      const std::shared_ptr<GPUResourceGroup>& gpu_resource_group);

  static Embedding<TYPE_2, __half>* create_localized_sparse_embedding_one_hot(
      const Tensors<TYPE_2>& row_offsets_tensors, const Tensors<TYPE_2>& value_tensors,
      SparseEmbeddingHashParams<__half> embedding_params, const std::string plan_file,
      const std::shared_ptr<GPUResourceGroup>& gpu_resource_group);

  static Embedding<TYPE_1, __half>* clone_eval(
      const Tensors<TYPE_1>& row_offsets_tensors, const Tensors<TYPE_1>& value_tensors,
      size_t batchsize, const std::shared_ptr<GPUResourceGroup>& gpu_resource_group,
      Embedding<TYPE_1, __half>* embedding);

  static Embedding<TYPE_1, float>* clone_eval(
      const Tensors<TYPE_1>& row_offsets_tensors, const Tensors<TYPE_1>& value_tensors,
      size_t batchsize, const std::shared_ptr<GPUResourceGroup>& gpu_resource_group,
      Embedding<TYPE_1, float>* embedding);

  static Embedding<TYPE_2, __half>* clone_eval(
      const Tensors<TYPE_2>& row_offsets_tensors, const Tensors<TYPE_2>& value_tensors,
      size_t batchsize, const std::shared_ptr<GPUResourceGroup>& gpu_resource_group,
      Embedding<TYPE_2, __half>* embedding);

  static Embedding<TYPE_2, float>* clone_eval(
      const Tensors<TYPE_2>& row_offsets_tensors, const Tensors<TYPE_2>& value_tensors,
      size_t batchsize, const std::shared_ptr<GPUResourceGroup>& gpu_resource_group,
      Embedding<TYPE_2, float>* embedding);
};

}  // namespace HugeCTR
