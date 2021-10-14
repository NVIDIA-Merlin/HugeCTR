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
#include <omp.h>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embeddings/embedding_data.hpp"
#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {
/**
 * The LocalizedSlotSparseEmbeddingOneHot class inherits from Embedding class, which is the base
 * class for implementing all embedding layers. In this class, the slots in the embedding table
 * are assigned to a single GPU separately, which are called localized slots. For example, slot-0 on
 * GPU-0, slot-1 on GPU-1, slot-2 on GPU-0, slot-3 on GPU-1, etc. This class is very simple to the
 * LocalizedSlotSparseEmbeddingHash, but optimized for performance according to the "one-hot"
 * feature. So, there are several assumptions in this class: 1) The mapping method from keys to
 * embedding row_indices is linear, so there is no hashtable in this class; 2) all the features are
 * one-hot, while multi-hot is not supported in this class; 3) Implement P2P access in forward prop,
 * fused forward_sum+all2all+reorder, so there is no all2all in forward and backward prop, and can
 * only support single node. 4) only support SGD optimizer by now.
 */

template <typename TypeHashKey, typename TypeEmbeddingComp>
class LocalizedSlotSparseEmbeddingOneHot : public IEmbedding {
 private:
  // define tensors
  EmbeddingData<TypeHashKey, TypeEmbeddingComp> embedding_data_;
  Tensors2<float> hash_table_value_tensors_; /**< Hash table value. */
  std::vector<Tensors2<float>> value_table_tensors_;

  Tensors2<size_t> hash_table_slot_id_tensors_; /**< the tensors for storing slot ids */
  Tensors2<size_t> hash_value_index_tensors_;   /**< Hash value index. The index is corresponding to
                                                   the line   number of the value. */
  Tensors2<TypeEmbeddingComp>
      embedding_feature_tensors_; /**< the output tensor of the forward(). */
  Tensor2<TypeEmbeddingComp *> train_embedding_features_;
  Tensor2<TypeEmbeddingComp *> evaluate_embedding_features_;
  Tensors2<TypeEmbeddingComp> wgrad_tensors_; /**< the input tensor of the backward(). */

  Tensors2<size_t> top_categories_;
  std::vector<size_t> size_top_categories_;

  size_t max_vocabulary_size_;
  size_t max_vocabulary_size_per_gpu_;   /**< Max vocabulary size for each GPU. */
  std::vector<size_t> slot_num_per_gpu_; /* slot_num per GPU */
  std::vector<size_t> slot_size_array_;

  SparseEmbeddingFunctors functors_;

  Tensors2<TypeEmbeddingComp> all2all_tensors_; /**< the temple buffer to store all2all results */
  Tensors2<TypeEmbeddingComp> utest_all2all_tensors_;
  Tensors2<TypeEmbeddingComp> utest_reorder_tensors_;
  Tensors2<TypeEmbeddingComp> utest_backward_temp_tensors_;
  Tensors2<TypeEmbeddingComp> utest_forward_temp_tensors_;

  Tensors2<uint32_t> mapping_offsets_per_gpu_tensors_;

  Tensor2<TypeEmbeddingComp *> &get_embedding_features(bool is_train) {
    if (is_train) {
      return train_embedding_features_;
    } else {
      return evaluate_embedding_features_;
    }
  }

  /**
   * Calculate the max vocabulary size per GPU.
   * @param total_gpu_count total GPU count.
   * @param local_gpu_count local GPU count.
   * @param slot_sizes an array which stores the size of the slots to be initialized.
   * @param device_resources GPU device resources.
   */
  static size_t cal_max_voc_size_per_gpu(const std::vector<size_t> slot_sizes,
                                         const ResourceManager &resource_manager) {
    size_t local_gpu_count = resource_manager.get_local_gpu_count();
    size_t total_gpu_count = resource_manager.get_global_gpu_count();

    size_t max_voc_size = 0;
    for (size_t id = 0; id < local_gpu_count; id++) {
      size_t global_id = resource_manager.get_local_gpu(id)->get_global_id();

      size_t total_size = 0;
      for (size_t i = 0; i < slot_sizes.size(); i++) {
        if ((i % total_gpu_count) == global_id) {
          total_size += slot_sizes[i];
        }
      }

      if (total_size > max_voc_size) {
        max_voc_size = total_size;
      }
    }

    return max_voc_size;
  }

  /**
   * Initialize the hash table and embedding table on local GPUs. This function is only used
   * by LocalizedSparseEmbeddingHash.
   * @param slot_sizes an array which stores the size of the slots to be initialized.
   * @param embedding_vec_size embedding vector size.
   * @param hash_table_value_tensors embedding table tensors.
   * @param hash_table_slot_id_tensors slot ids tensors.
   */
  void init_embedding(const std::vector<size_t> slot_sizes, size_t embedding_vec_size,
                      std::vector<Tensors2<float>> &hash_table_value_tensors,
                      Tensors2<size_t> &hash_table_slot_id_tensors);

  /**
   * load_parameters() for LocalizedSlotSparseEmbeddingOnehot
   * @param keys the memory buffer storing keys.
   * @param slot_id the memory buffer storing slot_id.
   * @param embeddings the memory buffer storing embedding vectors.
   * @param num the number of unique keys (embedding vectors) in keys (embeddings).
   * @param embedding_vec_size embedding vector size.
   * @param hash_table_value_tensors the hash table value on multi GPUs.
   * @param slot_sizes the size for each slot
   * @param mapping_offsets_per_gpu_tensors the mapping offset of each slot on every GPU
   */
  void load_parameters(const Tensor2<TypeHashKey> &keys, const Tensor2<size_t> &slot_id,
                       const Tensor2<float> &embeddings, size_t num, size_t embedding_vec_size,
                       Tensors2<float> &hash_table_value_tensors,
                       const std::vector<size_t> &slot_sizes,
                       const Tensors2<uint32_t> &mapping_offsets_per_gpu_tensors);

  /**
   * dump_parameters for LocalizedSlotSparseEmbeddingOnehot.
   * @param sparse_model the folder name of sparse model.
   * @param embedding_vec_size embedding vector size.
   * @param hash_table_value_tensors the hash table value on multi-GPU.
   * @param slot_sizes the size for each slot
   */
  void dump_parameters(const std::string &sparse_model, size_t embedding_vec_size,
                       const Tensors2<float> &hash_table_value_tensors,
                       const std::vector<size_t> &slot_sizes) const;

  /**
   * dump_parameters for LocalizedSlotSparseEmbeddingOnehot.
   * @param keys the memory buffer to store keys.
   * @param slot_id the memory buffer to store slot_id.
   * @param embeddings the memory buffer to store embedding vectors.
   * @param num pointer to store the number of unique keys (embedding vectors).
   * @param embedding_vec_size embedding vector size.
   * @param hash_table_value_tensors the hash table value on multi-GPU.
   * @param slot_sizes the size for each slot
   */
  void dump_parameters(Tensor2<TypeHashKey> &keys, Tensor2<size_t> &slot_id,
                       Tensor2<float> &embeddings, size_t *num, size_t embedding_vec_size,
                       const Tensors2<float> &hash_table_value_tensors,
                       const std::vector<size_t> &slot_sizes) const;

 public:
  /**
   * The constructor of LocalizedSlotSparseEmbeddingOneHot.
   * @param row_offsets_tensors row offsets of the input tensor(refer to row offset vector in sparse
   * matrix CSR format).
   * @param hash_key_tensors hash keys of the input tensor(refer to value vector in sparse matrix
   * CSR format).
   * @param embedding_params embedding params for initialization.
   * @param resource_manager the GPU resource group
   */
  LocalizedSlotSparseEmbeddingOneHot(const Tensors2<TypeHashKey> &train_row_offsets_tensors,
                                     const Tensors2<TypeHashKey> &train_value_tensors,
                                     const std::vector<std::shared_ptr<size_t>> &train_nnz_array,
                                     const Tensors2<TypeHashKey> &evaluate_row_offsets_tensors,
                                     const Tensors2<TypeHashKey> &evaluate_value_tensors,
                                     const std::vector<std::shared_ptr<size_t>> &evaluate_nnz_array,
                                     const SparseEmbeddingHashParams &embedding_params,
                                     const std::shared_ptr<ResourceManager> &resource_manager);

  LocalizedSlotSparseEmbeddingOneHot(const SparseTensors<TypeHashKey> &train_keys,
                                     const SparseTensors<TypeHashKey> &evaluate_keys,
                                     const SparseEmbeddingHashParams &embedding_params,
                                     const std::shared_ptr<ResourceManager> &resource_manager);

  void filter_keys_per_gpu(bool is_train, size_t id, size_t global_id, size_t global_num);

  void data_to_unique_categories_per_gpu(bool is_train, size_t id);
  /**
   * The forward propagation of embedding layer.
   */
  void forward(bool is_train, int eval_batch = -1) override {
    CudaDeviceContext context;

#pragma omp parallel for num_threads(embedding_data_.get_resource_manager().get_local_gpu_count())
    for (size_t i = 0; i < embedding_data_.get_resource_manager().get_local_gpu_count(); i++) {
      context.set_device(
          embedding_data_.get_local_gpu(i).get_device_id());  // set device
                                                              // for forward_fuse method
      if (embedding_data_.embedding_params_.do_unique_key_flag) {
        data_to_unique_categories_per_gpu(is_train, i);
      }
      if (embedding_data_.embedding_params_.is_data_parallel) {
        filter_keys_per_gpu(is_train, i, embedding_data_.get_local_gpu(i).get_global_id(),
                            embedding_data_.get_resource_manager().get_global_gpu_count());
      }
      functors_.forward_mapping_per_gpu(
          embedding_data_.embedding_params_.get_batch_size(is_train), slot_num_per_gpu_[i],
          embedding_data_.get_value_tensors(is_train)[i],
          *embedding_data_.get_nnz_array(is_train)[i], mapping_offsets_per_gpu_tensors_[i],
          hash_value_index_tensors_[i], embedding_data_.get_local_gpu(i).get_stream());

      // fuse forward+all2all+reorder into one kernel
      functors_.forward_fuse_per_gpu(
          i, embedding_data_.get_resource_manager().get_local_gpu_count(),
          embedding_data_.embedding_params_.get_batch_size(is_train),
          embedding_data_.get_batch_size_per_gpu(is_train),
          embedding_data_.embedding_params_.slot_num, slot_num_per_gpu_[i],
          embedding_data_.embedding_params_.embedding_vec_size,
          embedding_data_.embedding_params_.combiner,
          embedding_data_.get_row_offsets_tensors(is_train)[i], hash_value_index_tensors_[i],
          hash_table_value_tensors_[i], get_embedding_features(is_train),
          embedding_data_.get_local_gpu(i).get_sm_count(),
          embedding_data_.get_local_gpu(i).get_stream());
    }

    functors_.sync_all_gpus(embedding_data_.get_resource_manager());

    return;
  }

  /**
   * The first stage of backward propagation of embedding layer,
   * which computes the wgrad by the dgrad from the top layer.
   */
  void backward() override {
    functors_.sync_all_gpus(embedding_data_.get_resource_manager());

    CudaDeviceContext context;
    for (size_t i = 0; i < embedding_data_.get_resource_manager().get_local_gpu_count(); i++) {
      context.set_device(embedding_data_.get_local_gpu(i).get_device_id());

      functors_.backward_fuse_per_gpu(
          i, embedding_data_.get_resource_manager().get_local_gpu_count(),
          embedding_data_.embedding_params_.get_batch_size(true),
          embedding_data_.get_batch_size_per_gpu(true), embedding_data_.embedding_params_.slot_num,
          slot_num_per_gpu_[i], embedding_data_.embedding_params_.embedding_vec_size,
          embedding_data_.embedding_params_.combiner, get_embedding_features(true),
          wgrad_tensors_[i], embedding_data_.get_local_gpu(i).get_sm_count(),
          embedding_data_.get_local_gpu(i).get_stream());
    }

    return;
  }

  /**
   * The second stage of backward propagation of embedding layer, which
   * updates the hash table by wgrad(from backward()) and optimizer.
   */
  void update_params() override {
    // accumulate times for adam optimizer
    embedding_data_.embedding_params_.opt_params.hyperparams.adam.times++;
#pragma omp parallel num_threads(embedding_data_.get_resource_manager().get_local_gpu_count())
    {
      size_t id = omp_get_thread_num();
      CudaDeviceContext context(embedding_data_.get_local_gpu(id).get_device_id());

      // do update params operation: only support SGD
      functors_.update_params(
          embedding_data_.embedding_params_.embedding_vec_size,
          embedding_data_.embedding_params_.opt_params, *embedding_data_.get_nnz_array(true)[id],
          hash_value_index_tensors_[id], wgrad_tensors_[id], hash_table_value_tensors_[id],
          top_categories_[id], size_top_categories_[id],
          embedding_data_.get_local_gpu(id).get_sm_count(),
          embedding_data_.get_local_gpu(id).get_stream());
    }

    return;
  }

  /**
   * Initialize the embedding table
   */
  void init_params() override {
    // do hash table value initialization
    if (slot_size_array_.size() == embedding_data_.embedding_params_.slot_num) {
      init_embedding(slot_size_array_, embedding_data_.embedding_params_.embedding_vec_size,
                     value_table_tensors_, hash_table_slot_id_tensors_);
    } else {
      throw std::runtime_error(
          std::string("[HCDEBUG][ERROR] Runtime error: the size of slot_sizes != slot_num\n"));
    }
  }

  /**
   * Read the hash table from the weight_stream on the host, and
   * upload it onto multi-GPUs global memory.
   * @param sparse_model the folder name of sparse model.
   */
  void load_parameters(std::string sparse_model) override;
  void load_parameters(BufferBag &buf_bag, size_t num) override;
  /**
   * Download the hash table from multi-GPUs global memory to CPU memory
   * and write it to the weight_stream on the host.
   * @param sparse_model the folder name of sparse model.
   */
  void dump_parameters(std::string sparse_model) const override;
  void dump_parameters(BufferBag &buf_bag, size_t *num) const override;

  void dump_opt_states(std::ofstream &stream) override {}
  void load_opt_states(std::ifstream &stream) override {}
  void reset_optimizer() override {}

  /**
   * Reset the embedding
   */
  void reset() override;

  /**
   * Get the total size of hash tables on all GPUs.
   */
  size_t get_params_num() const override {
    return (max_vocabulary_size_ * embedding_data_.embedding_params_.embedding_vec_size);
  }

  size_t get_vocabulary_size() const override { return max_vocabulary_size_; }

  size_t get_max_vocabulary_size() const override { return max_vocabulary_size_; }

  // only used for results check
  /**
   * Get the forward() results from GPUs and copy them to the host pointer
   * embedding_feature. This function is only used for unit test.
   * @param embedding_feature the host pointer for storing the forward()
   * results.
   */
  void get_forward_results(bool is_train, Tensor2<TypeEmbeddingComp> &embedding_feature) {
    size_t memcpy_size = embedding_data_.get_batch_size_per_gpu(is_train) *
                         embedding_data_.embedding_params_.slot_num *
                         embedding_data_.embedding_params_.embedding_vec_size;

    functors_.get_forward_results(memcpy_size, embedding_data_.get_output_tensors(is_train),
                                  embedding_feature, utest_forward_temp_tensors_,
                                  embedding_data_.get_resource_manager());

    return;
  }

  /**
   * Get the forward() results from GPUs and copy them to tensorflow's tensor.
   */
  void get_forward_results_tf(const bool is_train, const bool on_gpu,
                              void *const forward_result) override {
    size_t memcpy_size = embedding_data_.get_batch_size_per_gpu(is_train) *
                         embedding_data_.embedding_params_.slot_num *
                         embedding_data_.embedding_params_.embedding_vec_size;
    functors_.get_forward_results(memcpy_size, embedding_data_.get_output_tensors(is_train),
                                  forward_result, utest_forward_temp_tensors_,
                                  embedding_data_.get_resource_manager(), on_gpu);
    return;
  }

  /**
   * Get the backward() results from GPUs and copy them to the host pointer
   * wgrad. The wgrad on each GPU should be the same. This function is only
   * used for unit test.
   * @param wgrad the host pointer for storing the backward() results.
   * @param devIndex the GPU device id.
   */
  void get_backward_results(Tensor2<TypeEmbeddingComp> &wgrad, int devIndex) {
    CudaDeviceContext context(embedding_data_.get_local_gpu(0).get_device_id());

#ifndef ENABLE_MPI
    if (embedding_data_.get_resource_manager().get_global_gpu_count() > 1) {
      functors_.all2all_forward(embedding_data_.get_batch_size_per_gpu(true), slot_num_per_gpu_,
                                embedding_data_.embedding_params_.embedding_vec_size,
                                wgrad_tensors_, utest_all2all_tensors_,
                                embedding_data_.get_resource_manager());
    } else {
      CK_CUDA_THROW_(cudaMemcpyAsync(
          utest_all2all_tensors_[0].get_ptr(), wgrad_tensors_[0].get_ptr(),
          embedding_data_.get_batch_size_per_gpu(true) * slot_num_per_gpu_[0] *
              embedding_data_.embedding_params_.embedding_vec_size * sizeof(TypeEmbeddingComp),
          cudaMemcpyDeviceToDevice, embedding_data_.get_local_gpu(0).get_stream()));
    }
#else
    if (embedding_data_.get_resource_manager().get_global_gpu_count() > 1) {
      functors_.all2all_forward(
          embedding_data_.get_batch_size_per_gpu(true), embedding_data_.embedding_params_.slot_num,
          embedding_data_.embedding_params_.embedding_vec_size, wgrad_tensors_,
          utest_all2all_tensors_, embedding_data_.get_resource_manager());
    } else {
      CK_CUDA_THROW_(cudaMemcpyAsync(
          utest_all2all_tensors_[0].get_ptr(), wgrad_tensors_[0].get_ptr(),
          embedding_data_.get_batch_size_per_gpu(true) * slot_num_per_gpu_[0] *
              embedding_data_.embedding_params_.embedding_vec_size * sizeof(TypeEmbeddingComp),
          cudaMemcpyDeviceToDevice, embedding_data_.get_local_gpu(0).get_stream()));
    }
#endif

    // reorder
    functors_.forward_reorder(
        embedding_data_.get_batch_size_per_gpu(true), embedding_data_.embedding_params_.slot_num,
        embedding_data_.embedding_params_.embedding_vec_size, utest_all2all_tensors_,
        utest_reorder_tensors_, embedding_data_.get_resource_manager());

    // there are batch_size_per_gpu samples' wgard on each GPU
    size_t memcpy_size = embedding_data_.get_batch_size_per_gpu(true) *
                         embedding_data_.embedding_params_.slot_num *
                         embedding_data_.embedding_params_.embedding_vec_size;

    // nccl gather
    functors_.all_gather(memcpy_size,
                         utest_reorder_tensors_,        // send
                         utest_backward_temp_tensors_,  // recv
                         embedding_data_.get_resource_manager());

    // memcpy H2D
    functors_.get_backward_results(
        devIndex, embedding_data_.get_resource_manager().get_global_gpu_count() * memcpy_size,
        utest_backward_temp_tensors_, wgrad, embedding_data_.get_resource_manager());

    return;
  }

  /**
   * Get the update_params() results(the hash table, including hash_table_keys
   * and hash_table_values) from GPUs and copy them to the host pointers.
   * This function is only used for unit test.
   * @param hash_table_key the host pointer for storing the hash table keys.
   * @param hash_table_value the host pointer for storing the hash table values.
   */
  void get_update_params_results(Tensor2<TypeHashKey> &hash_table_key,
                                 Tensor2<float> &hash_table_value) {}

  void check_overflow() const override {}

  /** only used in tf embedding plugin to distribute top_gradients to each GPUs' output tensor.
   */
  cudaError_t update_top_gradients(const bool on_gpu, const void *const top_gradients) override {
    auto output_tensors = embedding_data_.get_output_tensors(true);
    CudaDeviceContext context;

    const auto top_gradients_internel = reinterpret_cast<const TypeEmbeddingComp *>(top_gradients);
    cudaMemcpyKind direction = (on_gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice);

    cudaError_t error = cudaError_t::cudaSuccess;
    for (size_t dev_id = 0; dev_id < embedding_data_.get_resource_manager().get_local_gpu_count();
         ++dev_id) {
      context.set_device(embedding_data_.get_local_gpu(dev_id).get_device_id());

      error = cudaMemcpyAsync(
          output_tensors[dev_id].get_ptr(),
          top_gradients_internel + dev_id * output_tensors[dev_id].get_num_elements(),
          output_tensors[dev_id].get_size_in_bytes(), direction,
          embedding_data_.get_local_gpu(dev_id).get_stream());
      if (error != cudaError_t::cudaSuccess) return error;
    }

    for (size_t dev_id = 0; dev_id < embedding_data_.get_resource_manager().get_local_gpu_count();
         ++dev_id) {
      context.set_device(embedding_data_.get_local_gpu(dev_id).get_device_id());
      error = cudaStreamSynchronize(embedding_data_.get_local_gpu(dev_id).get_stream());
      if (error != cudaError_t::cudaSuccess) return error;
    }

    return cudaError_t::cudaSuccess;
  }

  USE_EMBEDDING_DATA_FUNCTION(embedding_data_)
};  // end of class LocalizedSlotSparseEmbeddingOneHot

}  // namespace HugeCTR
