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

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embeddings/embedding.hpp"
#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {
/**
 * The LocalizedSlotSparseEmbeddingHash class inherits from Embedding class, which is the base
 * class for implementing all embedding layers. In this class, some of the slots in the embedding
 * table are assigned to a single GPU, which are called localized slots. For example, slot-0 on
 * GPU-0, slot-1 on GPU-1, slot-2 on GPU-0, slot-3 on GPU-1, etc. The embedding table is
 * encapsulated in a hash table. The key in the hash table is called as hash_table_key, and the
 * value in the hash table is called as hash_table_value_index that means it indicates the embedding
 * feature's row number in the embedding table, and the embedding feature is called as
 * hash_table_value. This class implements all the operations needed by the training process of
 * embedding layer, including forward propagation and backward propagation. The forward propagation
 * is corresponding to the API forward(). The backward propagation is divided into 2-stage APIs:
 * backward() and update_params(). The class also provides the operations for uploading hash
 * tables(including hash_table_key, hash_table_value_index and hash_table_value) from a host file to
 * GPUs(which named load_parameters()), and for downloading hash tables from GPUs to a host
 * file(which named dump_parameters()).
 */

template <typename TypeHashKey, typename TypeEmbeddingComp>
class LocalizedSlotSparseEmbeddingHash : public Embedding<TypeHashKey, TypeEmbeddingComp> {
  using Base = Embedding<TypeHashKey, TypeEmbeddingComp>;

  using NvHashTable = HashTable<TypeHashKey, size_t>;
  using comm_handler = GossipComm::FasterComm;

 private:
  std::vector<std::shared_ptr<NvHashTable>> hash_tables_; /**< Hash table.  */

  // define tensors
  Tensors2<float> hash_table_value_tensors_; /**< Hash table value. */
  std::vector<Tensors2<float>> value_table_tensors_;

  Tensors2<size_t> hash_table_slot_id_tensors_; /**< the tensors for storing slot ids */
  Tensors2<size_t> hash_value_index_tensors_;   /**< Hash value index. The index is corresponding to
                                                     the line number of the value. */
  Tensors2<TypeEmbeddingComp>
      embedding_feature_tensors_;             /**< the output tensor of the forward(). */
  Tensors2<TypeEmbeddingComp> wgrad_tensors_; /**< the input tensor of the backward(). */
  Tensors2<TypeEmbeddingComp>
      opt_m_tensors_; /**< The mi variable storage for adam optimizer in the update_params(). */
  Tensors2<TypeEmbeddingComp>
      opt_v_tensors_; /**< The vi variable storage for adam optimizer in the update_params(). */
  Tensors2<uint64_t> opt_prev_time_tensors_; /**< The previous update time storage for lazy adam
                                                  in update_params(). */
  Tensors2<TypeEmbeddingComp> opt_momentum_tensors_; /**< The momentum variable storage for the
                                           momentum optimizer in the update_params(). */
  Tensors2<TypeEmbeddingComp> opt_accm_tensors_;     /**< The accm variable storage for the nesterov
                                                         optimizer in the update_params(). */

  Tensors2<size_t> hash_value_index_sort_tensors_; /**< The temp memory to store the sorted hash
                                                        table value indexes in update_params(). */

  Tensors2<uint32_t>
      hash_value_index_count_offset_tensors_; /**< The temp memory to store the offset of each count
                                                 of hash table value indexes in update_params(). */

  Tensors2<uint32_t> new_hash_value_flag_tensors_;
  Tensors2<uint32_t> hash_value_flag_sumed_tensors_;

  Tensors2<uint32_t> hash_value_index_count_counter_tensors_; /**< The temp memory to store the
                                                                counter of the count of hash table
                                                                value indexes in update_params(). */
  Tensors2<TypeHashKey> sample_id_tensors_; /**< The temp memory to store the sample ids of hash
                                              table value in      update_params(). */
  Tensors2<TypeHashKey> sample_id_sort_tensors_; /**< The temp memory to store the sorted sample
                                                   ids of hash table value in update_params(). */
  Tensors2<void> temp_storage_sort_tensors_;     /**< The temp memory for the CUB lib sorting
                                                          API in update_params(). */
  Tensors2<void> temp_storage_scan_tensors_;     /**< The temp memory for the CUB lib scaning API
                                                          in update_params(). */

  size_t max_vocabulary_size_;
  size_t max_vocabulary_size_per_gpu_;   /**< Max vocabulary size for each GPU. */
  std::vector<size_t> slot_num_per_gpu_; /* slot_num per GPU */
  std::vector<size_t> slot_size_array_;

  SparseEmbeddingFunctors functors_;

#ifndef NCCL_A2A
  std::string plan_file_;                          /**< plan file for all2all */
  std::unique_ptr<comm_handler> all2all_forward_;  /**< obj of all2all for forward */
  std::unique_ptr<comm_handler> all2all_backward_; /**< obj of all2all for backward */
  std::unique_ptr<comm_handler>
      all2all_utest_; /**< obj of all2all for utest of getting backward results> */
#endif
  Tensors2<TypeEmbeddingComp> all2all_tensors_; /**< the temple buffer to store all2all results */

  Tensors2<TypeEmbeddingComp> utest_all2all_tensors_;
  Tensors2<TypeEmbeddingComp> utest_reorder_tensors_;
  Tensors2<TypeEmbeddingComp> utest_backward_temp_tensors_;
  Tensors2<TypeEmbeddingComp> utest_forward_temp_tensors_;

  /**
   * Calculate the max vocabulary size per GPU.
   * @param total_gpu_count total GPU count.
   * @param local_gpu_count local GPU count.
   * @param slot_sizes an array which stores the size of the slots to be intialized.
   * @param device_resources GPU device resources.
   */
  static size_t cal_max_voc_size_per_gpu(const std::vector<size_t> slot_sizes,
                                         const ResourceManager &resource_manager) {
    size_t local_gpu_count = resource_manager.get_local_gpu_count();
    size_t total_gpu_count = resource_manager.get_global_gpu_count();

    size_t max_voc_size = 0;
    for (size_t id = 0; id < local_gpu_count; id++) {
      size_t global_id = resource_manager.get_local_gpu(id)->get_global_gpu_id();

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
   * Initialize the embedding table on local GPUs.
   * @param max_vocabulary_size_per_gpu max vocabulary size per GPU.
   * @param embedding_vec_size embedding vector size.
   * @param hash_table_value_tensors embedding table tensors.
   * @param device_resources GPU device resources.
   */
  void init_embedding(size_t max_vocabulary_size_per_gpu, size_t embedding_vec_size,
                      Tensors2<float> &hash_table_value_tensors);

  /**
   * Initialize the hash table and embedding table on local GPUs. This function is only used
   * by LocalizedSparseEmbeddingHash.
   * @param slot_sizes an array which stores the size of the slots to be intialized.
   * @param embedding_vec_size embedding vector size.
   * @param hash_table_value_tensors embedding table tensors.
   * @param hash_tables GPU hash tables which stores <key, value_index>.
   * @param hash_table_slot_id_tensors slot ids tensors.
   * @param device_resources GPU device resources.
   */
  void init_embedding(const std::vector<size_t> &slot_sizes, size_t embedding_vec_size,
                      std::vector<Tensors2<float>> &hash_table_value_tensors,
                      Tensors2<size_t> &hash_table_slot_id_tensors);

  /**
   * load_parameters() for LocalizedSlotSparseEmbeddingHash
   * @param weight_stream weight file stream to read.
   * @param vocabulary_size the total row number of hash table.
   * @param embedding_vec_size embedding vector size.
   * @param max_vocabulary_size_per_gpu max vocabulary size for each GPU
   * @param hash_table_value_tensors the hash table value on multi GPUs.
   * @param hash_table_slot_id_tensors the hash table slot_ids on multi GPUs.
   * @param hash_tables the hash tables on multi GPUs.
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  void load_parameters(std::ifstream &weight_stream, size_t vocabulary_size,
                       size_t embedding_vec_size, size_t max_vocabulary_size_per_gpu,
                       Tensors2<float> &hash_table_value_tensors,
                       Tensors2<size_t> &hash_table_slot_id_tensors,
                       std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables);

  /**
   * dump_parameters for LocalizedSlotSparseEmbeddingHash.
   * @param weight_stream weight file stream to write.
   * @param vocabulary_size the total row number of hash table.
   * @param embedding_vec_size embedding vector size.
   * @param hash_table_value_tensors the hash table value on multi-GPU.
   * @param hash_table_slot_id_tensors the hash table slot_ids on multi-GPU
   * @param hash_tables the hash tables on multi GPUs
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  void dump_parameters(
      std::ofstream &weight_stream, size_t vocabulary_size, size_t embedding_vec_size,
      const Tensors2<float> &hash_table_value_tensors,
      const Tensors2<size_t> &hash_table_slot_id_tensors,
      const std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables) const;

 public:
  /**
   * The constructor of LocalizedSlotSparseEmbeddingHash.
   * @param row_offsets_tensors row offsets of the input tensor(refer to row offset vector in sparse
   * matrix CSR format).
   * @param hash_key_tensors hash keys of the input tensor(refer to value vector in sparse matrix
   * CSR format).
   * @param embedding_params embedding params for initialization.
   * @param resource_manager the GPU resource group
   */
  LocalizedSlotSparseEmbeddingHash(
      const Tensors2<TypeHashKey> &train_row_offsets_tensors,
      const Tensors2<TypeHashKey> &train_value_tensors,
      const std::vector<std::shared_ptr<size_t>> &train_nnz_array,
      const Tensors2<TypeHashKey> &evaluate_row_offsets_tensors,
      const Tensors2<TypeHashKey> &evaluate_value_tensors,
      const std::vector<std::shared_ptr<size_t>> &evaluate_nnz_array,
      const SparseEmbeddingHashParams<TypeEmbeddingComp> &embedding_params,
      const std::string plan_file, const std::shared_ptr<ResourceManager> &resource_manager);

  /**
   * The forward propagation of embedding layer.
   */
  void forward(bool is_train) override {
    CudaDeviceContext context;

    for (size_t i = 0; i < Base::get_resource_manager().get_local_gpu_count(); i++) {
      context.set_device(Base::get_local_gpu(i).get_device_id());  // set device
      functors_.forward_per_gpu(
          Base::get_batch_size(is_train), slot_num_per_gpu_[i], Base::get_embedding_vec_size(),
          Base::get_combiner(), is_train, Base::get_row_offsets_tensors(is_train)[i],
          Base::get_value_tensors(is_train)[i], *Base::get_nnz_array(is_train)[i], *hash_tables_[i],
          hash_table_value_tensors_[i], hash_value_index_tensors_[i], embedding_feature_tensors_[i],
          Base::get_local_gpu(i).get_stream());
    }

// do all-to-all
#ifdef NCCL_A2A

#ifndef ENABLE_MPI
    if (Base::get_resource_manager().get_global_gpu_count() > 1) {
      functors_.all2all_forward(Base::get_batch_size_per_gpu(is_train), slot_num_per_gpu_,
                                Base::get_embedding_vec_size(), embedding_feature_tensors_,
                                all2all_tensors_, Base::get_resource_manager());
    } else {
      CK_CUDA_THROW_(
          cudaMemcpyAsync(all2all_tensors_[0].get_ptr(), embedding_feature_tensors_[0].get_ptr(),
                          Base::get_batch_size_per_gpu(is_train) * slot_num_per_gpu_[0] *
                              Base::get_embedding_vec_size() * sizeof(TypeEmbeddingComp),
                          cudaMemcpyDeviceToDevice, Base::get_local_gpu(0).get_stream()));
    }
#else
    if (Base::get_resource_manager().get_global_gpu_count() > 1) {
      functors_.all2all_forward(Base::get_batch_size_per_gpu(is_train), Base::get_slot_num(),
                                Base::get_embedding_vec_size(), embedding_feature_tensors_,
                                all2all_tensors_, Base::get_resource_manager());
    } else {
      CK_CUDA_THROW_(
          cudaMemcpyAsync(all2all_tensors_[0].get_ptr(), embedding_feature_tensors_[0].get_ptr(),
                          (size_t)Base::get_batch_size_per_gpu(is_train) * slot_num_per_gpu_[0] *
                              Base::get_embedding_vec_size() * sizeof(TypeEmbeddingComp),
                          cudaMemcpyDeviceToDevice, Base::get_local_gpu(0).get_stream()));
    }
#endif

#else
    // sync: guarantee the data is ready for all2all
    functors_.sync_all_gpus(Base::get_resource_manager());
    functors_.all2all_exec(*all2all_forward_);
#endif

    // reorder
    functors_.forward_reorder(Base::get_batch_size_per_gpu(is_train), Base::get_slot_num(),
                              Base::get_embedding_vec_size(), all2all_tensors_,
                              Base::get_output_tensors(is_train), Base::get_resource_manager());

    // store slot ids
    functors_.store_slot_id(Base::get_batch_size(is_train), Base::get_slot_num(), slot_num_per_gpu_,
                            Base::get_row_offsets_tensors(is_train), hash_value_index_tensors_,
                            hash_table_slot_id_tensors_, Base::get_resource_manager());

    return;
  }

  /**
   * The first stage of backward propagation of embedding layer,
   * which computes the wgrad by the dgrad from the top layer.
   */
  void backward() override {
    // Read dgrad from output_tensors -> compute wgrad

    // reorder
    functors_.backward_reorder(Base::get_batch_size_per_gpu(true), Base::get_slot_num(),
                               Base::get_embedding_vec_size(), Base::get_output_tensors(true),
                               all2all_tensors_, Base::get_resource_manager());

// do all2all
#ifdef NCCL_A2A

#ifndef ENABLE_MPI
    if (Base::get_resource_manager().get_global_gpu_count() > 1) {
      functors_.all2all_backward(Base::get_batch_size_per_gpu(true), slot_num_per_gpu_,
                                 Base::get_embedding_vec_size(), all2all_tensors_,
                                 embedding_feature_tensors_, Base::get_resource_manager());

    } else {
      CudaDeviceContext context(Base::get_local_gpu(0).get_device_id());
      CK_CUDA_THROW_(
          cudaMemcpyAsync(embedding_feature_tensors_[0].get_ptr(), all2all_tensors_[0].get_ptr(),
                          Base::get_batch_size_per_gpu(true) * slot_num_per_gpu_[0] *
                              Base::get_embedding_vec_size() * sizeof(TypeEmbeddingComp),
                          cudaMemcpyDeviceToDevice, Base::get_local_gpu(0).get_stream()));
    }
#else
    if (Base::get_resource_manager().get_global_gpu_count() > 1) {
      functors_.all2all_backward(Base::get_batch_size_per_gpu(true), Base::get_slot_num(),
                                 Base::get_embedding_vec_size(), all2all_tensors_,
                                 embedding_feature_tensors_, Base::get_resource_manager());

    } else {
      CudaDeviceContext context(Base::get_local_gpu(0).get_device_id());
      CK_CUDA_THROW_(
          cudaMemcpyAsync(embedding_feature_tensors_[0].get_ptr(), all2all_tensors_[0].get_ptr(),
                          Base::get_batch_size_per_gpu(true) * slot_num_per_gpu_[0] *
                              Base::get_embedding_vec_size() * sizeof(TypeEmbeddingComp),
                          cudaMemcpyDeviceToDevice, Base::get_local_gpu(0).get_stream()));
    }
#endif

#else
    // sync: guarantee the data is ready for all2all
    functors_.sync_all_gpus(Base::get_resource_manager());
    functors_.all2all_exec(*all2all_backward_);
#endif

    // do backward
    functors_.backward(Base::get_batch_size(true), slot_num_per_gpu_,
                       Base::get_embedding_vec_size(), Base::get_combiner(),
                       Base::get_row_offsets_tensors(true), embedding_feature_tensors_,
                       wgrad_tensors_, Base::get_resource_manager());

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

      Base::get_opt_params(id).hyperparams.adam.times++;

      // do update params operation
      functors_.update_params(
          Base::get_batch_size(true), slot_num_per_gpu_[id], Base::get_embedding_vec_size(),
          max_vocabulary_size_per_gpu_, Base::get_opt_params(id), *Base::get_nnz_array(true)[id],
          Base::get_row_offsets_tensors(true)[id], hash_value_index_tensors_[id],
          sample_id_tensors_[id], sample_id_sort_tensors_[id], hash_value_index_sort_tensors_[id],
          hash_value_index_count_offset_tensors_[id], new_hash_value_flag_tensors_[id],
          hash_value_flag_sumed_tensors_[id], hash_value_index_count_counter_tensors_[id],
          temp_storage_sort_tensors_[id], temp_storage_scan_tensors_[id], wgrad_tensors_[id],
          hash_table_value_tensors_[id], Base::get_local_gpu(id).get_sm_count(),
          Base::get_local_gpu(id).get_stream());
    }
  }

  /**
   * Initialize the embedding table
   */
  void init_params() override {
    // do hash table value initialization
    if (slot_size_array_.empty()) {  // if no slot_sizes provided, use the old method to init
      init_embedding(max_vocabulary_size_per_gpu_, Base::get_embedding_vec_size(),
                     hash_table_value_tensors_);

    } else {
      if (slot_size_array_.size() == Base::get_slot_num()) {
#ifndef DATA_READING_TEST
        init_embedding(slot_size_array_, Base::get_embedding_vec_size(), value_table_tensors_,
                       hash_table_slot_id_tensors_);

#endif
      } else {
        throw std::runtime_error(
            std::string("[HCDEBUG][ERROR] Runtime error: the size of slot_sizes != slot_num\n"));
      }
    }
  }

  /**
   * Read the hash table from the weight_stream on the host, and
   * upload it onto multi-GPUs global memory.
   * @param weight_stream the host file stream for reading data from.
   */
  void load_parameters(std::ifstream &weight_stream) override {
    // check if file is opened successfully
    if (!weight_stream.is_open()) {
      CK_THROW_(Error_t::WrongInput, "Error: file not open for reading");
    }

    load_parameters(weight_stream, max_vocabulary_size_, Base::get_embedding_vec_size(),
                    max_vocabulary_size_per_gpu_, hash_table_value_tensors_,
                    hash_table_slot_id_tensors_, hash_tables_);

    return;
  }

  void load_parameters(const TensorBag2 &keys, const Tensor2<float> &embeddings,
                       size_t num) override {}

  /**
   * Download the hash table from multi-GPUs global memroy to CPU memory
   * and write it to the weight_stream on the host.
   * @param weight_stream the host file stream for writing data to.
   */
  void dump_parameters(std::ofstream &weight_stream) const override {
    // check if the file is opened successfully
    if (!weight_stream.is_open()) {
      CK_THROW_(Error_t::WrongInput, "Error: file not open for writing");
      return;
    }

    dump_parameters(weight_stream, max_vocabulary_size_, Base::get_embedding_vec_size(),
                    hash_table_value_tensors_, hash_table_slot_id_tensors_, hash_tables_);

    return;
  }

  void dump_parameters(TensorBag2 keys, Tensor2<float> &embeddings, size_t *num) const override {}

  /**
   * Reset the embedding
   */
  void reset() override;

  /**
   * Get the total size of hash tables on all GPUs.
   */
  size_t get_params_num() const override {
    // Read data from input_buffers_ -> look up -> write to output_tensors
    return get_vocabulary_size() * Base::get_embedding_vec_size();
  }

  size_t get_vocabulary_size() const override {
    size_t total_size = 0;

    // need to collect the <key, value> pair count from all GPUs and do sum reduction
    CudaDeviceContext context;
    for (size_t id = 0; id < Base::get_resource_manager().get_local_gpu_count(); id++) {
      context.set_device(Base::get_local_gpu(id).get_device_id());
      total_size += hash_tables_[id]->get_size(Base::get_local_gpu(id).get_stream());
    }

    return total_size;
  }

  size_t get_max_vocabulary_size() const override { return max_vocabulary_size_; }

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
   * Get the forward() results from GPUs and copy them to TensorFlow's tensor.
  */
  void get_forward_results_tf(const bool is_train, const bool on_gpu, void* const forward_result) override {
    size_t memcpy_size = Base::get_batch_size_per_gpu(is_train) * Base::get_slot_num() * 
                         Base::get_embedding_vec_size();
    functors_.get_forward_results(memcpy_size, Base::get_output_tensors(is_train),
                                  forward_result, utest_forward_temp_tensors_,
                                  Base::get_resource_manager(), on_gpu);
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
    CudaDeviceContext context(Base::get_local_gpu(0).get_device_id());

#ifdef NCCL_A2A

#ifndef ENABLE_MPI
    if (Base::get_resource_manager().get_global_gpu_count() > 1) {
      functors_.all2all_forward(Base::get_batch_size_per_gpu(true), slot_num_per_gpu_,
                                Base::get_embedding_vec_size(), wgrad_tensors_,
                                utest_all2all_tensors_, Base::get_resource_manager());
    } else {
      CK_CUDA_THROW_(
          cudaMemcpyAsync(utest_all2all_tensors_[0].get_ptr(), wgrad_tensors_[0].get_ptr(),
                          Base::get_batch_size_per_gpu(true) * slot_num_per_gpu_[0] *
                              Base::get_embedding_vec_size() * sizeof(TypeEmbeddingComp),
                          cudaMemcpyDeviceToDevice, Base::get_local_gpu(0).get_stream()));
    }
#else
    if (Base::get_resource_manager().get_global_gpu_count() > 1) {
      functors_.all2all_forward(Base::get_batch_size_per_gpu(true), Base::get_slot_num(),
                                Base::get_embedding_vec_size(), wgrad_tensors_,
                                utest_all2all_tensors_, Base::get_resource_manager());
    } else {
      CK_CUDA_THROW_(
          cudaMemcpyAsync(utest_all2all_tensors_[0].get_ptr(), wgrad_tensors_[0].get_ptr(),
                          (size_t)Base::get_batch_size_per_gpu(true) * slot_num_per_gpu_[0] *
                              Base::get_embedding_vec_size() * sizeof(TypeEmbeddingComp),
                          cudaMemcpyDeviceToDevice, Base::get_local_gpu(0).get_stream()));
    }
#endif

#else
    // sync: guarantee the data is ready for all2all
    functors_.sync_all_gpus(Base::get_resource_manager());
    functors_.all2all_exec(*all2all_utest_);
#endif

    // reorder
    functors_.forward_reorder(Base::get_batch_size_per_gpu(true), Base::get_slot_num(),
                              Base::get_embedding_vec_size(), utest_all2all_tensors_,
                              utest_reorder_tensors_, Base::get_resource_manager());

    // there are batch_size_per_gpu samples' wgard on each GPU
    size_t memcpy_size = (size_t)Base::get_batch_size_per_gpu(true) * Base::get_slot_num() *
                         Base::get_embedding_vec_size();

    // nccl gather
    functors_.all_gather(memcpy_size,
                         utest_reorder_tensors_,        // send
                         utest_backward_temp_tensors_,  // recv
                         Base::get_resource_manager());

    // memcpy H2D
    functors_.get_backward_results(
        devIndex, Base::get_resource_manager().get_global_gpu_count() * memcpy_size,
        utest_backward_temp_tensors_, wgrad, Base::get_resource_manager());

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

  /** only used in tf embedding plugin to distribute top_gradients to each GPUs' output tensor.
  */
  cudaError_t update_top_gradients(const bool on_gpu, const void* const top_gradients) override {
    auto output_tensors = Base::get_output_tensors(true);
    CudaDeviceContext context;

    const auto top_gradients_internel = reinterpret_cast<const TypeEmbeddingComp*>(top_gradients);
    cudaMemcpyKind direction = (on_gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice);

    cudaError_t error = cudaError_t::cudaSuccess;
    for (size_t dev_id = 0; dev_id < Base::get_resource_manager().get_local_gpu_count(); ++dev_id) {
      context.set_device(Base::get_local_gpu(dev_id).get_device_id());

      error = cudaMemcpyAsync(output_tensors[dev_id].get_ptr(), 
                              top_gradients_internel + dev_id * output_tensors[dev_id].get_num_elements(),
                              output_tensors[dev_id].get_size_in_bytes(),
                              direction, 
                              Base::get_local_gpu(dev_id).get_stream());
      if (error != cudaError_t::cudaSuccess) return error;
    }

    for (size_t dev_id = 0; dev_id < Base::get_resource_manager().get_local_gpu_count(); ++dev_id) {
      context.set_device(Base::get_local_gpu(dev_id).get_device_id());
      error = cudaStreamSynchronize(Base::get_local_gpu(dev_id).get_stream());
      if (error != cudaError_t::cudaSuccess) return error;
    }

    return cudaError_t::cudaSuccess;
  }

};  // end of class LocalizedSlotSparseEmbeddingHash

}  // namespace HugeCTR
