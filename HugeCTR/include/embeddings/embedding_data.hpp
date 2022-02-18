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
#include <embedding.hpp>
#include <general_buffer2.hpp>
#include <resource_manager.hpp>
#include <unordered_map>

#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"
#include "HugeCTR/include/utils.hpp"
namespace HugeCTR {

template <typename TypeKey, typename TypeEmbeddingComp>
class EmbeddingData {
 public:
  const Embedding_t embedding_type_;
  SparseEmbeddingHashParams embedding_params_; /**< Sparse embedding hash params. */

  std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>
      bufs_;                                         /**< The buffer for storing output tensors. */
  Tensors2<TypeEmbeddingComp> train_output_tensors_; /**< The output tensors. */
  Tensors2<TypeEmbeddingComp> evaluate_output_tensors_; /**< The output tensors. */
  Tensors2<TypeKey> train_row_offsets_tensors_; /**< The row_offsets tensors of the input data. */
  Tensors2<TypeKey> train_value_tensors_;       /**< The value tensors of the input data. */
  std::vector<std::shared_ptr<size_t>> train_nnz_array_;
  Tensors2<TypeKey>
      evaluate_row_offsets_tensors_;         /**< The row_offsets tensors of the input data. */
  Tensors2<TypeKey> evaluate_value_tensors_; /**< The value tensors of the input data. */
  std::vector<std::shared_ptr<size_t>> evaluate_nnz_array_;

  std::shared_ptr<ResourceManager> resource_manager_; /**< The GPU device resources. */

  SparseTensors<TypeKey> train_keys_;
  SparseTensors<TypeKey> evaluate_keys_;
  Tensors2<TypeKey> embedding_offsets_;

  bool is_trainable_{true};

  size_t get_batch_size_per_gpu(bool is_train) const {
    return embedding_params_.get_batch_size(is_train) / resource_manager_->get_global_gpu_count();
  }

  size_t get_universal_batch_size_per_gpu() const {
    return embedding_params_.get_universal_batch_size() / resource_manager_->get_global_gpu_count();
  }

  ResourceManager& get_resource_manager() const { return *resource_manager_; }

  const GPUResource& get_local_gpu(int i) const { return *resource_manager_->get_local_gpu(i); }

  const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& get_buffer(size_t i) const {
    return bufs_[i];
  }

  Tensors2<TypeEmbeddingComp>& get_output_tensors(bool is_train) {
    if (is_train) {
      return train_output_tensors_;
    } else {
      return evaluate_output_tensors_;
    }
  }

  SparseTensors<TypeKey>& get_input_keys(bool is_train) {
    return is_train ? train_keys_ : evaluate_keys_;
  }

  Tensors2<TypeKey>& get_row_offsets_tensors(bool is_train) {
    if (is_train) {
      return train_row_offsets_tensors_;
    } else {
      return evaluate_row_offsets_tensors_;
    }
  }

  Tensors2<TypeKey>& get_value_tensors(bool is_train) {
    if (is_train) {
      return train_value_tensors_;
    } else {
      return evaluate_value_tensors_;
    }
  }

  std::vector<std::shared_ptr<size_t>>& get_nnz_array(bool is_train) {
    if (is_train) {
      return train_nnz_array_;
    } else {
      return evaluate_nnz_array_;
    }
  }

  /**
   * The constructor of Embedding class.
   * @param row_offsets_tensors the row_offsets tensors of the input data(refer to row offset vector
   * in sparse matrix CSR format).
   * @param value_tensors the value tensors of the input data(refer to value vector in sparse matrix
   * CSR format).
   * @param batchsize the batch size of the input data
   * @param slot_num the number of slots of the hash table
   * @param embedding_vec_size the dim size of the embedding feature vector.
   * @param resource_manager the GPU device resource group
   * @param scaler scaler factor for mixed precision
   */
  EmbeddingData(const Tensors2<TypeKey>& train_row_offsets_tensors,
                const Tensors2<TypeKey>& train_value_tensors,
                const std::vector<std::shared_ptr<size_t>>& train_nnz_array,
                const Tensors2<TypeKey>& evaluate_row_offsets_tensors,
                const Tensors2<TypeKey>& evaluate_value_tensors,
                const std::vector<std::shared_ptr<size_t>>& evaluate_nnz_array,
                const Embedding_t embedding_type, const SparseEmbeddingHashParams& embedding_params,
                const std::shared_ptr<ResourceManager>& resource_manager)
      : embedding_type_(embedding_type),
        embedding_params_(embedding_params),
        train_row_offsets_tensors_(train_row_offsets_tensors),
        train_value_tensors_(train_value_tensors),
        train_nnz_array_(train_nnz_array),
        evaluate_row_offsets_tensors_(evaluate_row_offsets_tensors),
        evaluate_value_tensors_(evaluate_value_tensors),
        evaluate_nnz_array_(evaluate_nnz_array),
        resource_manager_(resource_manager) {
    try {
      // Error check
      if (embedding_params.train_batch_size < 1 || embedding_params.evaluate_batch_size < 1 ||
          embedding_params.slot_num < 1 || embedding_params.embedding_vec_size < 1) {
        HCTR_OWN_THROW(Error_t::WrongInput,
                       "batchsize < 1 || slot_num < 1 || embedding_vec_size < 1");
      }

      if (embedding_params.embedding_vec_size > 1024) {
        HCTR_OWN_THROW(Error_t::WrongInput,
                       "the embedding_vec_size can not be more than 1024 in embedding layer");
      }

      size_t total_gpu_count = resource_manager_->get_global_gpu_count();
      size_t local_gpu_count = resource_manager_->get_local_gpu_count();

      if (train_row_offsets_tensors.size() != local_gpu_count ||
          train_value_tensors.size() != local_gpu_count ||
          evaluate_row_offsets_tensors.size() != local_gpu_count ||
          evaluate_value_tensors.size() != local_gpu_count) {
        HCTR_OWN_THROW(
            Error_t::WrongInput,
            "either row_offsets_tensors.size() or value_tensors.size() isn't local_gpu_count_");
      }

      assert(bufs_.empty());
      for (size_t i = 0; i < local_gpu_count; i++) {
        std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf =
            GeneralBuffer2<CudaAllocator>::create();
        bufs_.push_back(buf);

        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({get_batch_size_per_gpu(true), embedding_params_.slot_num,
                      embedding_params_.embedding_vec_size},
                     &tensor);
        train_output_tensors_.push_back(tensor);
        buf->reserve({get_batch_size_per_gpu(false), embedding_params_.slot_num,
                      embedding_params_.embedding_vec_size},
                     &tensor);
        evaluate_output_tensors_.push_back(tensor);
      }

      for (size_t i = 0; i < local_gpu_count; i++) {
        train_keys_.emplace_back(train_value_tensors_[i], train_row_offsets_tensors_[i],
                                 train_nnz_array_[i]);
        evaluate_keys_.emplace_back(evaluate_value_tensors_[i], evaluate_row_offsets_tensors_[i],
                                    evaluate_nnz_array_[i]);
      }
    } catch (const std::runtime_error& rt_err) {
      HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
      throw;
    }
    return;
  }

  /**
   * The constructor of Embedding class.
   * @param row_offsets_tensors the row_offsets tensors of the input data(refer to row offset vector
   * in sparse matrix CSR format).
   * @param value_tensors the value tensors of the input data(refer to value vector in sparse matrix
   * CSR format).
   * @param batchsize the batch size of the input data
   * @param slot_num the number of slots of the hash table
   * @param embedding_vec_size the dim size of the embedding feature vector.
   * @param resource_manager the GPU device resource group
   * @param scaler scaler factor for mixed precision
   */
  EmbeddingData(const Embedding_t embedding_type, const SparseTensors<TypeKey>& train_keys,
                const SparseTensors<TypeKey>& evaluate_keys,
                const SparseEmbeddingHashParams& embedding_params,
                const std::shared_ptr<ResourceManager>& resource_manager)
      : embedding_type_(embedding_type),
        embedding_params_(embedding_params),
        train_keys_(train_keys),
        evaluate_keys_(evaluate_keys),
        resource_manager_(resource_manager) {
    try {
      // Error check
      if (embedding_params.train_batch_size < 1 || embedding_params.evaluate_batch_size < 1 ||
          embedding_params.slot_num < 1 || embedding_params.embedding_vec_size < 1) {
        HCTR_OWN_THROW(Error_t::WrongInput,
                       "batchsize < 1 || slot_num < 1 || embedding_vec_size < 1");
      }

      if (embedding_params.embedding_vec_size > 1024) {
        HCTR_OWN_THROW(Error_t::WrongInput,
                       "the embedding_vec_size can not be more than 1024 in embedding layer");
      }

      size_t total_gpu_count = resource_manager_->get_global_gpu_count();
      size_t local_gpu_count = resource_manager_->get_local_gpu_count();

      assert(bufs_.empty());
      for (size_t i = 0; i < local_gpu_count; i++) {
        CudaDeviceContext context(get_local_gpu(i).get_device_id());
        auto buf = GeneralBuffer2<CudaAllocator>::create();
        bufs_.push_back(buf);

        {
          Tensor2<TypeEmbeddingComp> tensor;
          buf->reserve({get_batch_size_per_gpu(true), embedding_params_.slot_num,
                        embedding_params_.embedding_vec_size},
                       &tensor);
          train_output_tensors_.push_back(tensor);
        }
        {
          Tensor2<TypeEmbeddingComp> tensor;
          buf->reserve({get_batch_size_per_gpu(false), embedding_params_.slot_num,
                        embedding_params_.embedding_vec_size},
                       &tensor);
          evaluate_output_tensors_.push_back(tensor);
        }
        {
          Tensor2<TypeKey> tensor;
          buf->reserve({embedding_params.slot_size_array.size()}, &tensor);
          embedding_offsets_.push_back(tensor);
        }
      }

    } catch (const std::runtime_error& rt_err) {
      HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
      throw;
    }
    return;
  }
  /**
   * The declaration for indicating that there is no default copy construtor in this class.
   */
  DISALLOW_COPY_AND_MOVE(EmbeddingData)
};

#define USE_EMBEDDING_DATA_FUNCTION(embedding_data)                                          \
  Embedding_t get_embedding_type() const override { return embedding_data.embedding_type_; } \
  std::vector<TensorBag2> get_train_output_tensors() const override {                        \
    std::vector<TensorBag2> bags;                                                            \
    for (const auto& t : embedding_data.train_output_tensors_) {                             \
      bags.push_back(t.shrink());                                                            \
    }                                                                                        \
    return bags;                                                                             \
  }                                                                                          \
  std::vector<TensorBag2> get_evaluate_output_tensors() const override {                     \
    std::vector<TensorBag2> bags;                                                            \
    for (const auto& t : embedding_data.evaluate_output_tensors_) {                          \
      bags.push_back(t.shrink());                                                            \
    }                                                                                        \
    return bags;                                                                             \
  }                                                                                          \
  void set_learning_rate(float lr) override {                                                \
    embedding_data.embedding_params_.opt_params.lr = lr;                                     \
  }                                                                                          \
  const SparseEmbeddingHashParams& get_embedding_params() const override {                   \
    return embedding_data.embedding_params_;                                                 \
  }
}  // namespace HugeCTR
