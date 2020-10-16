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
#include <omp.h>
#include <vector>
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embeddings/embedding.hpp"
#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {
/**
 * The DistributedSlotSparseEmbeddingHash class inherits from Embedding class, which is the
 * base class for implementing all embedding layers. In this class, some of the slots in the
 * embedding table are assigned to multiple GPUs, which are called distributed slots. For
 * example, slot-0 on GPU-0/GPU-1, slot-1 on GPU-0/GPU-1, etc. The embedding table is encapsulated
 * in a hash table. The key in the hash table is called as hash_table_key, and the value in
 * the hash table is called as hash_table_value_index that means it indicates the embedding
 * feature's row number in the embedding table, and the embedding feature is called as
 * hash_table_value. This class implements all the operations needed by the training process of
 * embedding layer, including forward propagation and backward propagation. The forward propagation
 * is corresponding to the API forward(). The backward propagation is divided into 2-stage APIs:
 * backward() and update_params(). The class also provides the operations for uploading hash tables
 * (including hash_table_key, hash_table_value_index and hash_table_value) from a host file to
 * GPUs(which named upload_params_to_device()), and for downloading hash tables from GPUs to
 * a host file(which named download_params_to_host()).
 */
template <typename TypeHashKey, typename TypeEmbeddingComp>
class DistributedSlotSparseEmbeddingHash : public Embedding<TypeHashKey, TypeEmbeddingComp> {
  using Base = Embedding<TypeHashKey, TypeEmbeddingComp>;

  using NvHashTable = HashTable<TypeHashKey, size_t>;

 private:
  std::vector<std::shared_ptr<NvHashTable>> hash_tables_; /**< Hash table.  */

  // define tensors
  Tensors2<float> hash_table_value_tensors_;  /**< Hash table value. */
  Tensors2<size_t> hash_value_index_tensors_; /**< Hash table value index. The index is
                                                   corresponding to the line number of the value. */
  Tensors2<TypeEmbeddingComp>
      embedding_feature_tensors_;             /**< the output tensor of the forward(). */
  Tensors2<TypeEmbeddingComp> wgrad_tensors_; /**< the input tensor of the backward(). */
  Tensors2<TypeEmbeddingComp>
      opt_m_tensors_; /**< The mi variable storage for adam optimizer in the update_params(). */
  Tensors2<TypeEmbeddingComp>
      opt_v_tensors_; /**< The vi variable storage for adam optimizer in the update_params(). */
  Tensors2<TypeEmbeddingComp> opt_momentum_tensors_; /**< The momentum variable storage
                                           for the momentum optimizer in the update_params(). */
  Tensors2<TypeEmbeddingComp> opt_accm_tensors_;     /**< The accm variable storage for the
                                                         nesterov optimizer in the update_params(). */
  Tensors2<TypeHashKey>
      row_offset_allreduce_tensors_; /**< The temp memory to store the row_offset after all_reduce
                                        operation among multi-gpu in forward(). */
  Tensors2<size_t> hash_value_index_sort_tensors_; /**< The temp memory to store the sorted hash
                                                        table value indexes in update_params(). */

  Tensors2<uint32_t>
      hash_value_index_count_offset_tensors_; /**< The temp memory to store the offset of each count
                                                 of hash table value indexes in update_params(). */
  Tensors2<uint32_t> hash_value_index_count_counter_tensors_; /**< The temp memory to store the
                                                                counter of the count of hash table
                                                                value indexes in update_params(). */

  Tensors2<uint32_t> new_hash_value_flag_tensors_;
  Tensors2<uint32_t> hash_value_flag_sumed_tensors_;

  Tensors2<TypeHashKey> sample_id_tensors_; /**< The temp memory to store the sample ids of hash
                                              table value in      update_params(). */
  Tensors2<TypeHashKey> sample_id_sort_tensors_; /**< The temp memory to store the sorted sample
                                                   ids of hash table value in update_params(). */
  Tensors2<void> temp_storage_sort_tensors_;     /**< The temp memory for the CUB lib sorting
                                                          API in update_params(). */

  Tensors2<void> temp_storage_scan_tensors_; /**< The temp memory for the CUB lib scaning API
                                                      in update_params(). */

  Tensors2<size_t> deltaw_hash_value_index_tensors_; /**< The temp memory to store the hash table
                                                          indexes of deltaw in update_params(). */
  Tensors2<float> deltaw_tensors_; /**< The temp memory to store the deltaw in update_params(). */

  Tensors2<TypeEmbeddingComp> utest_forward_temp_tensors_;

  size_t max_vocabulary_size_;         /**< Max vocabulary size for each GPU. */
  size_t max_vocabulary_size_per_gpu_; /**< Max vocabulary size for each GPU. */

  SparseEmbeddingFunctors functors_;

  /**
   * Initialize the embedding table on local GPUs.
   * @param max_vocabulary_size_per_gpu max vocabulary size per GPU.
   * @param embedding_vec_size embedding vector size.
   * @param hash_table_value_tensors embedding table tensors.
   * @param resource_manager GPU device resources.
   */
  void init_embedding(size_t max_vocabulary_size_per_gpu, size_t embedding_vec_size,
                      Tensors2<float> &hash_table_value_tensors);

  /**
   * upload_params_to_device for DistributedSlotSparseEmbeddingHash.
   * @param weight_stream weight file stream to read.
   * @param vocabulary_size the total row number of hash table.
   * @param embedding_vec_size embedding vector size.
   * @param max_vocabulary_size_per_gpu max vocabulary size for each GPU
   * @param hash_table_value_tensors the tensors of hash table value on multi GPUs.
   * @param hash_tables the hash tables on multi GPUs
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  void upload_params_to_device(
      std::ifstream &weight_stream, size_t vocabulary_size, size_t embedding_vec_size,
      size_t max_vocabulary_size_per_gpu, Tensors2<float> &hash_table_value_tensors,
      std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables);

  /**
   * download_params_to_host for DistributedSlotSparseEmbeddingHash
   * download hash_table from GPUs to CPU.
   * @param weight_stream weight file stream to write.
   * @param vocabulary_size the total row number of hash table.
   * @param embedding_vec_size embedding vector size.
   * @param hash_table_value_tensors the tensors of hash table value on multi GPUs.
   * @param hash_tables the hash tables on multi GPUs
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  void download_params_to_host(
      std::ofstream &weight_stream, size_t vocabulary_size, size_t embedding_vec_size,
      const Tensors2<float> &hash_table_value_tensors,
      const std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables) const;

 public:
  /**
   * The constructor of DistributedSlotSparseEmbeddingHash.
   * @param row_offsets_tensors row offsets of the input tensor(refer to row offset vector in sparse
   * matrix CSR format).
   * @param hash_key_tensors hash keys of the input tensor(refer to value vector in sparse matrix
   * CSR format).
   * @param embedding_params embedding params for initialization.
   * @param resource_manager the GPU resource group
   */
  DistributedSlotSparseEmbeddingHash(
      const Tensors2<TypeHashKey> &train_row_offsets_tensors,
      const Tensors2<TypeHashKey> &train_value_tensors,
      const std::vector<std::shared_ptr<size_t>> &train_nnz_array,
      const Tensors2<TypeHashKey> &evaluate_row_offsets_tensors,
      const Tensors2<TypeHashKey> &evaluate_value_tensors,
      const std::vector<std::shared_ptr<size_t>> &evaluate_nnz_array,
      const SparseEmbeddingHashParams<TypeEmbeddingComp> &embedding_params,
      const std::shared_ptr<ResourceManager> &resource_manager);

  /**
   * The forward propagation of embedding layer.
   */
  void forward(bool is_train) override {
    // Read data from input_buffers_ -> look up -> write to output_tensors

    CudaDeviceContext context;

    for (size_t i = 0; i < Base::get_resource_manager().get_local_gpu_count(); i++) {
      context.set_device(Base::get_local_gpu(i).get_device_id());
      functors_.forward_per_gpu(
          Base::get_batch_size(is_train), Base::get_slot_num(), Base::get_embedding_vec_size(), 0,
          is_train, Base::get_row_offsets_tensors(is_train)[i],
          Base::get_value_tensors(is_train)[i], *Base::get_nnz_array(is_train)[i], *hash_tables_[i],
          hash_table_value_tensors_[i], hash_value_index_tensors_[i], embedding_feature_tensors_[i],
          Base::get_local_gpu(i).get_stream());
    }

    // do reduce scatter
    size_t recv_count = Base::get_batch_size_per_gpu(is_train) * Base::get_slot_num() *
                        Base::get_embedding_vec_size();
    functors_.reduce_scatter(recv_count, embedding_feature_tensors_,
                             Base::get_output_tensors(is_train), Base::get_resource_manager());

    // scale for combiner=mean after reduction
    if (Base::get_combiner() == 1) {
      size_t send_count = Base::get_batch_size(is_train) * Base::get_slot_num() + 1;
      functors_.all_reduce(send_count, Base::get_row_offsets_tensors(is_train),
                           row_offset_allreduce_tensors_, Base::get_resource_manager());

      // do average
      functors_.forward_scale(Base::get_batch_size(is_train), Base::get_slot_num(),
                              Base::get_embedding_vec_size(), row_offset_allreduce_tensors_,
                              Base::get_output_tensors(is_train), Base::get_resource_manager());
    }

    return;
  }

  /**
   * The first stage of backward propagation of embedding layer,
   * which only computes the wgrad by the dgrad from the top layer.
   */
  void backward() override {
    // Read dgrad from output_tensors -> compute wgrad

    // do all-gather to collect the top_grad
    size_t send_count =
        Base::get_batch_size_per_gpu(true) * Base::get_slot_num() * Base::get_embedding_vec_size();
    functors_.all_gather(send_count, Base::get_output_tensors(true), embedding_feature_tensors_,
                         Base::get_resource_manager());

    // do backward
    functors_.backward(Base::get_batch_size(true), Base::get_slot_num(),
                       Base::get_embedding_vec_size(), Base::get_combiner(),
                       row_offset_allreduce_tensors_, embedding_feature_tensors_, wgrad_tensors_,
                       Base::get_resource_manager());

    return;
  }

  /**
   * The second stage of backward propagation of embedding layer, which
   * updates the hash table by wgrad(from backward()) and optimizer.
   */
  void update_params() override {
#pragma omp parallel num_threads(Base::get_resource_manager().get_local_gpu_count())
    {
      size_t id = omp_get_thread_num();
      CudaDeviceContext context(Base::get_local_gpu(id).get_device_id());

      // accumulate times for adam optimizer
      Base::get_opt_params(id).hyperparams.adam.times++;

      // do update params operation
      functors_.update_params(
          Base::get_batch_size(true), Base::get_slot_num(), Base::get_embedding_vec_size(),
          max_vocabulary_size_per_gpu_, Base::get_opt_params(id), *Base::get_nnz_array(true)[id],
          Base::get_row_offsets_tensors(true)[id], hash_value_index_tensors_[id],
          sample_id_tensors_[id], sample_id_sort_tensors_[id], hash_value_index_sort_tensors_[id],
          hash_value_index_count_offset_tensors_[id], new_hash_value_flag_tensors_[id],
          hash_value_flag_sumed_tensors_[id], hash_value_index_count_counter_tensors_[id],
          temp_storage_sort_tensors_[id], temp_storage_scan_tensors_[id], wgrad_tensors_[id],
          deltaw_hash_value_index_tensors_[id], deltaw_tensors_[id], hash_table_value_tensors_[id],
          Base::get_local_gpu(id).get_sm_count(), Base::get_local_gpu(id).get_stream());
    }

    return;
  }

  /**
   * Initialize the embedding table
   */
  void init_params() override {
    init_embedding(max_vocabulary_size_per_gpu_, Base::get_embedding_vec_size(),
                   hash_table_value_tensors_);
  }

  /**
   * Read the hash table from the weight_stream on the host, and
   * upload it onto multi-GPUs global memory.
   * @param weight_stream the host file stream for reading data from.
   */
  void upload_params_to_device(std::ifstream &weight_stream) override {
    // check if file is opened successfully
    if (!weight_stream.is_open()) {
      CK_THROW_(Error_t::WrongInput, "Error: file not open for reading");
    }

    upload_params_to_device(weight_stream, max_vocabulary_size_, Base::get_embedding_vec_size(),
                            max_vocabulary_size_per_gpu_, hash_table_value_tensors_, hash_tables_);

    return;
  }

  /**
   * Download the hash table from multi-GPUs global memroy to CPU memory
   * and write it to the weight_stream on the host.
   * @param weight_stream the host file stream for writing data to.
   */
  void download_params_to_host(std::ofstream &weight_stream) const override {
    // check if the file is opened successfully
    if (!weight_stream.is_open()) {
      CK_THROW_(Error_t::WrongInput, "Error: file not open for writing");
      return;
    }

    download_params_to_host(weight_stream, max_vocabulary_size_, Base::get_embedding_vec_size(),
                            hash_table_value_tensors_, hash_tables_);

    return;
  }

  /**
   * Get the total size of hash tables on all GPUs.
   */
  size_t get_params_num() const override {
    // Read data from input_buffers_ -> look up -> write to output_tensors

    size_t total_size = 0;

    CudaDeviceContext context;

    // need to collect the <key, value> pair count from all GPUs and do sum reduction
    for (size_t id = 0; id < Base::get_resource_manager().get_local_gpu_count(); id++) {
      context.set_device(Base::get_local_gpu(id).get_device_id());
      total_size += hash_tables_[id]->get_size(Base::get_local_gpu(id).get_stream());
      CK_CUDA_THROW_(cudaStreamSynchronize(Base::get_local_gpu(id).get_stream()));
    }

    total_size *= Base::get_embedding_vec_size();

    return total_size;
  }

  // only used for results check
  /**
   * Get the forward() results from GPUs and copy them to the host pointer
   * embedding_feature. This function is only used for unit test.
   * @param embedding_feature the host pointer for storing the forward()
   * results.
   */
  void get_forward_results(bool is_train, Tensor2<TypeEmbeddingComp> &embedding_feature) override {
    size_t memcpy_size = Base::get_batch_size_per_gpu(is_train) * Base::get_slot_num() *
                         Base::get_embedding_vec_size();

    functors_.get_forward_results(memcpy_size, Base::get_output_tensors(is_train),
                                  embedding_feature, utest_forward_temp_tensors_,
                                  Base::get_resource_manager());

    return;
  }

  /**
   * Get the backward() results from GPUs and copy them to the host pointer
   * wgrad. The wgrad on each GPU should be the same. This function is only
   * used for unit test.
   * @param wgrad the host pointer for stroing the backward() results.
   * @param devIndex the GPU device id.
   */
  void get_backward_results(Tensor2<TypeEmbeddingComp> &wgrad, int devIndex) override {
    // wgard shuld be the same on multi-gpus after backward()
    size_t memcpy_size =
        Base::get_batch_size(true) * Base::get_slot_num() * Base::get_embedding_vec_size();

    functors_.get_backward_results(devIndex, memcpy_size, wgrad_tensors_, wgrad,
                                   Base::get_resource_manager());

    return;
  }

  /**
   * Get the update_params() results(the hash table, including hash_table_keys
   * and hash_table_values) from GPUs and copy them to the host pointers.
   * This function is only used for unit test.
   * @param hash_table_key the host pointer for stroing the hash table keys.
   * @param hash_table_value the host pointer for stroing the hash table values.
   */
  void get_update_params_results(Tensor2<TypeHashKey> &hash_table_key,
                                 Tensor2<float> &hash_table_value) override {
    functors_.get_update_params_results(Base::get_embedding_vec_size(), max_vocabulary_size_,
                                        hash_table_value_tensors_, hash_tables_, hash_table_key,
                                        hash_table_value, Base::get_resource_manager());

    return;
  }

  /**
   * Check overflow
   * @param lr
   */
  void check_overflow() const override {
    CudaDeviceContext context;

    for (size_t id = 0; id < Base::get_resource_manager().get_local_gpu_count(); id++) {
      context.set_device(Base::get_local_gpu(id).get_device_id());
      size_t count = hash_tables_[id]->get_size(Base::get_local_gpu(id).get_stream());
      if (count > max_vocabulary_size_per_gpu_) {
        CK_THROW_(Error_t::OutOfBound, "Runtime vocabulary size (" + std::to_string(count) +
                                           ") exceeds max_vocabulary_size_per_gpu (" +
                                           std::to_string(max_vocabulary_size_per_gpu_) +
                                           ") on GPU " +
                                           std::to_string(Base::get_local_gpu(id).get_device_id()) +
                                           ", new feature insertion failed.\n");
      }
    }
  }

};  // end of class DistributedSlotSparseEmbeddingHash

}  // namespace HugeCTR
