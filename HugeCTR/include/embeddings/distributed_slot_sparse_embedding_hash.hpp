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
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/embeddings/sparse_embedding_hash_functors.hpp"
#include "cub/cub/device/device_radix_sort.cuh"

#include "HugeCTR/include/hashtable/nv_hashtable.cuh"

#include <vector>

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

  using TypeHashValueIndex = TypeHashKey;  // use the hash key type as the hash value_index type(it
                                           // will be uint32 or int64)
  using NvHashTable =
      HashTable<TypeHashKey, TypeHashValueIndex, std::numeric_limits<TypeHashKey>::max()>;

 private:
  SparseEmbeddingHashParams<TypeEmbeddingComp>
      embedding_params_;                                  /**< Sparse embedding hash params. */
  std::vector<OptParams<TypeEmbeddingComp>> opt_params_;  /**< Optimizer params. */
  std::vector<std::shared_ptr<NvHashTable>> hash_tables_; /**< Hash table.  */
  std::vector<PinnedBuffer<TypeHashKey>>
      nnz_num_per_batch_; /**< non-zero feature number in one batch */

  // define tensors
  Tensors<float> hash_table_value_tensors_; /**< Hash table value. */
  Tensors<TypeHashValueIndex>
      hash_value_index_tensors_; /**< Hash table value index. The index is corresponding to the line
                                    number of the value. */
  Tensors<TypeEmbeddingComp> embedding_feature_tensors_; /**< the output tensor of the forward(). */
  Tensors<TypeEmbeddingComp> wgrad_tensors_;             /**< the input tensor of the backward(). */
  Tensors<TypeEmbeddingComp>
      opt_m_tensors_; /**< The mi variable storage for adam optimizer in the update_params(). */
  Tensors<TypeEmbeddingComp>
      opt_v_tensors_; /**< The vi variable storage for adam optimizer in the update_params(). */
  Tensors<TypeEmbeddingComp> opt_momentum_tensors_; /**< The momentum variable storage for the
                                           momentum optimizer in the update_params(). */
  Tensors<TypeEmbeddingComp> opt_accm_tensors_;     /**< The accm variable storage for the nesterov
                                                         optimizer in the update_params(). */
  Tensors<TypeHashKey>
      row_offset_allreduce_tensors_; /**< The temp memory to store the row_offset after all_reduce
                                        operation among multi-gpu in forward(). */
  Tensors<TypeHashValueIndex>
      hash_value_index_sort_tensors_; /**< The temp memory to store the sorted hash table value
                                         indexes in update_params(). */
  // Tensors<uint32_t> hash_value_index_count_tensors_; /**< The temp memory to store the count of
  // hash
  //                                                       table value indexes in update_params().
  //                                                       */
  Tensors<uint32_t>
      hash_value_index_count_offset_tensors_; /**< The temp memory to store the offset of each count
                                                 of hash table value indexes in update_params(). */
  Tensors<uint32_t> hash_value_index_count_counter_tensors_; /**< The temp memory to store the
                                                                counter of the count of hash table
                                                                value indexes in update_params(). */

  Tensors<uint32_t> new_hash_value_flag_tensors_;
  Tensors<uint32_t> hash_value_flag_sumed_tensors_;

  Tensors<TypeHashKey> sample_id_tensors_;      /**< The temp memory to store the sample ids of hash
                                                   table value in      update_params(). */
  Tensors<TypeHashKey> sample_id_sort_tensors_; /**< The temp memory to store the sorted sample ids
                                                   of hash table value in update_params(). */
  Tensors<TypeHashKey> temp_storage_sort_tensors_; /**< The temp memory for the CUB lib sorting API
                                                      in update_params(). */

  Tensors<uint32_t> temp_storage_scan_tensors_; /**< The temp memory for the CUB lib scaning API
                                                      in update_params(). */

  Tensors<TypeHashValueIndex>
      deltaw_hash_value_index_tensors_; /**< The temp memory to store the hash table indexes of
                                           deltaw in update_params(). */
  Tensors<float> deltaw_tensors_; /**< The temp memory to store the deltaw in update_params(). */

  // define GeneralBuffers
  GeneralBuffers<float> float_bufs_; /**< float type general buffer. */
  GeneralBuffers<TypeEmbeddingComp>
      fp_bufs_; /**< TypeEmbeddingComp(fp32 or fp16) type general buffer. */
  GeneralBuffers<uint32_t> uint32_bufs_; /**< uint32 type general buffer. */
  GeneralBuffers<TypeHashKey> key_bufs_; /**< TypeHashKey type general buffer. */
  GeneralBuffers<TypeHashValueIndex>
      value_index_bufs_; /**< TypeHashValueIndex type general buffer. */

  std::vector<size_t> temp_storage_sort_bytes_; /**< The temp variable for CUB lib sorting API. */
  std::vector<size_t> temp_storage_scan_bytes_; /**< The temp variable for CUB lib scaning API. */

  size_t max_vocabulary_size_;           /**< Max vocabulary size for each GPU. */
  size_t max_vocabulary_size_per_gpu_;   /**< Max vocabulary size for each GPU. */
  int batch_size_per_gpu_;               /*< batch_size per GPU */
  SparseEmbeddingHashFunctors functors_; /**< obj of SparseEmbeddingHashFunctors */

  Tensors<TypeEmbeddingComp> utest_forward_temp_tensors_;

  int total_gpu_count_;
  int local_gpu_count_;

  /**
   * The constructor of DistributedSlotSparseEmbeddingHash.
   * This ctor is only used when you already have a instant of DistributedSlotSparseEmbeddingHash
   * and you want to reuse the hash_table/embedding_table for evaluation.
   * @param row_offsets_tensors row offsets of the input tensor(refer to row offset vector in sparse
   * matrix CSR format).
   * @param hash_key_tensors hash keys of the input tensor(refer to value vector in sparse matrix
   * CSR format).
   * @param batchsize batsize. The batchsize for eval and train may be different.
   * @param gpu_resource_group the GPU resource group.
   * @param obj the current DistributedSlotSparseEmbeddingHash class object.
   */
  DistributedSlotSparseEmbeddingHash(const Tensors<TypeHashKey> &row_offsets_tensors,
                                     const Tensors<TypeHashKey> &value_tensors, size_t batchsize,
                                     const std::shared_ptr<GPUResourceGroup> &gpu_resource_group,
                                     const DistributedSlotSparseEmbeddingHash &obj)
      : embedding_params_(obj.embedding_params_),
        total_gpu_count_(obj.total_gpu_count_),
        local_gpu_count_(obj.local_gpu_count_),
        max_vocabulary_size_(obj.max_vocabulary_size_),
        max_vocabulary_size_per_gpu_(obj.max_vocabulary_size_per_gpu_),
        hash_tables_(obj.hash_tables_),
        hash_table_value_tensors_(obj.hash_table_value_tensors_),
        Base(row_offsets_tensors, value_tensors, batchsize, obj.embedding_params_.slot_num,
             obj.embedding_params_.embedding_vec_size, gpu_resource_group,
             obj.embedding_params_.opt_params.scaler) {
    try {
      CudaDeviceContext context((*Base::device_resources_)[0]->get_device_id());

      embedding_params_.batch_size = batchsize;
      batch_size_per_gpu_ = embedding_params_.batch_size / total_gpu_count_;

      for (int id = 0; id < local_gpu_count_; id++) {
        int cur_device = (*Base::device_resources_)[id]->get_device_id();
        context.set_device(cur_device);

        // new nnz vectors
        nnz_num_per_batch_.push_back(PinnedBuffer<TypeHashKey>(1));

        // new GeneralBuffer objects
        fp_bufs_.emplace_back(new GeneralBuffer<TypeEmbeddingComp>());
        value_index_bufs_.emplace_back(new GeneralBuffer<TypeHashValueIndex>());
        key_bufs_.emplace_back(new GeneralBuffer<TypeHashKey>());

        // new hash table value_index that get() from HashTable
        hash_value_index_tensors_.emplace_back(new Tensor<TypeHashValueIndex>(
            {1, embedding_params_.batch_size * embedding_params_.max_feature_num},
            value_index_bufs_.back(), TensorFormat_t::HW));

        // new embedding features reduced by hash table values(results of forward)
        embedding_feature_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
            {embedding_params_.batch_size * embedding_params_.slot_num,
             embedding_params_.embedding_vec_size},
            fp_bufs_.back(), TensorFormat_t::HW));

        row_offset_allreduce_tensors_.emplace_back(new Tensor<TypeHashKey>(
            {1, embedding_params_.batch_size * embedding_params_.slot_num + 1}, key_bufs_.back(),
            TensorFormat_t::HW));

        utest_forward_temp_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
            {embedding_params_.batch_size * embedding_params_.slot_num,
             embedding_params_.embedding_vec_size},
            fp_bufs_.back(), TensorFormat_t::HW));

        // init GenenralBuffers to do real allocation
#ifndef NDEBUG
        std::cout << " fp_bufs_:" << fp_bufs_.back()->get_size() << std::endl;
        std::cout << " value_index_bufs_:" << value_index_bufs_.back()->get_size() << std::endl;
#endif
        fp_bufs_.back()->init(cur_device);
        value_index_bufs_.back()->init(cur_device);
        key_bufs_.back()->init(cur_device);

      }  // end of for(int id = 0; id < local_gpu_count_; id++)

      // sync
      functors_.sync_all_gpus(Base::device_resources_, context);

    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }

    return;
  }

 public:
  /**
   * Clone the embedding object for evaluation.
   * @param row_offsets_tensors row offsets of the input tensor(refer to row offset vector in sparse
   * matrix CSR format).
   * @param hash_key_tensors hash keys of the input tensor(refer to value vector in sparse matrix
   * CSR format).
   * @param batchsize batsize. The batchsize for eval and train may be different.
   * @param gpu_resource_group the GPU resource group.
   */
  Embedding<TypeHashKey, TypeEmbeddingComp> *clone_eval(
      const Tensors<TypeHashKey> &row_offsets_tensors, const Tensors<TypeHashKey> &value_tensors,
      size_t batchsize, const std::shared_ptr<GPUResourceGroup> &gpu_resource_group) {
    Embedding<TypeHashKey, TypeEmbeddingComp> *new_embedding =
        new DistributedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>(
            row_offsets_tensors, value_tensors, batchsize, gpu_resource_group, *this);
    return new_embedding;
  }

  /**
   * The constructor of DistributedSlotSparseEmbeddingHash.
   * @param row_offsets_tensors row offsets of the input tensor(refer to row offset vector in sparse
   * matrix CSR format).
   * @param hash_key_tensors hash keys of the input tensor(refer to value vector in sparse matrix
   * CSR format).
   * @param embedding_params embedding params for initialization.
   * @param gpu_resource_group the GPU resource group
   */
  DistributedSlotSparseEmbeddingHash(const Tensors<TypeHashKey> &row_offsets_tensors,
                                     const Tensors<TypeHashKey> &hash_key_tensors,
                                     SparseEmbeddingHashParams<TypeEmbeddingComp> embedding_params,
                                     const std::shared_ptr<GPUResourceGroup> &gpu_resource_group)
      : embedding_params_(embedding_params),
        Base(row_offsets_tensors, hash_key_tensors, embedding_params.batch_size,
             embedding_params.slot_num, embedding_params.embedding_vec_size, gpu_resource_group,
             embedding_params.opt_params.scaler) {
    try {
      total_gpu_count_ = Base::device_resources_->get_total_gpu_count();
      local_gpu_count_ = Base::device_resources_->size();
      CudaDeviceContext context((*Base::device_resources_)[0]->get_device_id());

      // CAUSION: can not decide how many <key,value> pairs in each GPU, because the GPU
      // distribution is computed by (key%gpu_count). In order to not allocate the total size of
      // hash table on each GPU, meanwhile get a better performance by a unfull hash table, the
      // users need to set the param "load_factor"(load_factor<1).
      max_vocabulary_size_per_gpu_ = embedding_params_.max_vocabulary_size_per_gpu;
      max_vocabulary_size_ = max_vocabulary_size_per_gpu_ * total_gpu_count_;

      MESSAGE_("max_vocabulary_size_per_gpu_=" + std::to_string(max_vocabulary_size_per_gpu_));

      batch_size_per_gpu_ = embedding_params_.batch_size / total_gpu_count_;

      for (int id = 0; id < local_gpu_count_; id++) {
        int cur_device = (*Base::device_resources_)[id]->get_device_id();
        context.set_device(cur_device);

        // new nnz vectors
        nnz_num_per_batch_.push_back(PinnedBuffer<TypeHashKey>(1));

        // construct HashTable object: used to store hash table <key, value_index>
        hash_tables_.emplace_back(new NvHashTable(max_vocabulary_size_per_gpu_));

        // new GeneralBuffer objects
        float_bufs_.emplace_back(new GeneralBuffer<float>());
        fp_bufs_.emplace_back(new GeneralBuffer<TypeEmbeddingComp>());
        uint32_bufs_.emplace_back(new GeneralBuffer<uint32_t>());
        key_bufs_.emplace_back(new GeneralBuffer<TypeHashKey>());
        value_index_bufs_.emplace_back(new GeneralBuffer<TypeHashValueIndex>());

        // new hash table value vectors
        hash_table_value_tensors_.emplace_back(
            new Tensor<float>({max_vocabulary_size_per_gpu_, embedding_params_.embedding_vec_size},
                              float_bufs_.back(), TensorFormat_t::HW));

        // new hash table value_index that get() from HashTable
        hash_value_index_tensors_.emplace_back(new Tensor<TypeHashValueIndex>(
            {1, embedding_params_.batch_size * embedding_params_.max_feature_num},
            value_index_bufs_.back(), TensorFormat_t::HW));

        // new embedding features reduced by hash table values(results of forward)
        embedding_feature_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
            {embedding_params_.batch_size * embedding_params_.slot_num,
             embedding_params_.embedding_vec_size},
            fp_bufs_.back(), TensorFormat_t::HW));

        // new wgrad used by backward
        wgrad_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
            {embedding_params_.batch_size * embedding_params_.slot_num,
             embedding_params_.embedding_vec_size},
            fp_bufs_.back(), TensorFormat_t::HW));

        // new optimizer params used by update_params
        opt_params_.push_back(OptParams<TypeEmbeddingComp>());
        opt_params_[id].optimizer = embedding_params_.opt_params.optimizer;
        opt_params_[id].lr = embedding_params_.opt_params.lr;
        opt_params_[id].global_update = embedding_params_.opt_params.global_update;
        opt_params_[id].scaler = embedding_params_.opt_params.scaler;
        switch (embedding_params_.opt_params.optimizer) {
          case Optimizer_t::Adam:  // adam
            opt_m_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
                {max_vocabulary_size_per_gpu_, embedding_params_.embedding_vec_size},
                fp_bufs_.back(), TensorFormat_t::HW));
            opt_v_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
                {max_vocabulary_size_per_gpu_, embedding_params_.embedding_vec_size},
                fp_bufs_.back(), TensorFormat_t::HW));
            break;

          case Optimizer_t::MomentumSGD:  // momentum_sgd
            opt_momentum_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
                {max_vocabulary_size_per_gpu_, embedding_params_.embedding_vec_size},
                fp_bufs_.back(), TensorFormat_t::HW));
            break;

          case Optimizer_t::Nesterov:  // nesterov
            opt_accm_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
                {max_vocabulary_size_per_gpu_, embedding_params_.embedding_vec_size},
                fp_bufs_.back(), TensorFormat_t::HW));
            break;

          case Optimizer_t::SGD:
            break;

          default:
            throw std::runtime_error(
                std::string("[HCDEBUG][ERROR] Runtime error: Invalid optimizer type\n"));
        }

        // new temp tensors used by update_params
        row_offset_allreduce_tensors_.emplace_back(new Tensor<TypeHashKey>(
            {1, embedding_params_.batch_size * embedding_params_.slot_num + 1}, key_bufs_.back(),
            TensorFormat_t::HW));
        sample_id_tensors_.emplace_back(new Tensor<TypeHashKey>(
            {1, embedding_params_.batch_size * embedding_params_.max_feature_num}, key_bufs_.back(),
            TensorFormat_t::HW));
        sample_id_sort_tensors_.emplace_back(new Tensor<TypeHashKey>(
            {1, embedding_params_.batch_size * embedding_params_.max_feature_num}, key_bufs_.back(),
            TensorFormat_t::HW));
        hash_value_index_sort_tensors_.emplace_back(new Tensor<TypeHashValueIndex>(
            {1, embedding_params_.batch_size * embedding_params_.max_feature_num},
            value_index_bufs_.back(), TensorFormat_t::HW));
        hash_value_index_count_offset_tensors_.emplace_back(new Tensor<uint32_t>(
            {1, embedding_params_.batch_size * embedding_params_.max_feature_num + 1},
            uint32_bufs_.back(), TensorFormat_t::HW));

        new_hash_value_flag_tensors_.emplace_back(new Tensor<uint32_t>(
            {1, embedding_params_.batch_size * embedding_params_.max_feature_num},
            uint32_bufs_.back(), TensorFormat_t::HW));

        hash_value_flag_sumed_tensors_.emplace_back(new Tensor<uint32_t>(
            {1, embedding_params_.batch_size * embedding_params_.max_feature_num},
            uint32_bufs_.back(), TensorFormat_t::HW));

        hash_value_index_count_counter_tensors_.emplace_back(
            new Tensor<uint32_t>({1, 1}, uint32_bufs_.back(), TensorFormat_t::HW));
        deltaw_hash_value_index_tensors_.emplace_back(new Tensor<TypeHashValueIndex>(
            {1, embedding_params_.batch_size * embedding_params_.max_feature_num},
            value_index_bufs_.back(), TensorFormat_t::HW));
        deltaw_tensors_.emplace_back(
            new Tensor<float>({embedding_params_.batch_size * embedding_params_.max_feature_num,
                               embedding_params_.embedding_vec_size},
                              float_bufs_.back(), TensorFormat_t::HW));
        {
          // cal the temp storage bytes for CUB radix sort
          size_t temp = 0;
          cub::DeviceRadixSort::SortPairs(
              (void *)NULL, (size_t &)temp, (TypeHashKey *)NULL, (TypeHashKey *)NULL,
              (TypeHashKey *)NULL, (TypeHashKey *)NULL,
              embedding_params_.batch_size * embedding_params_.max_feature_num);
          temp_storage_sort_bytes_.push_back(temp);
          size_t size = (size_t)ceil((float)temp_storage_sort_bytes_[id] / sizeof(TypeHashKey));

          // new temp storage tensors for CUB radix sort
          temp_storage_sort_tensors_.emplace_back(
              new Tensor<TypeHashKey>({1, size}, key_bufs_.back(), TensorFormat_t::HW));
        }

        {
          size_t temp = 0;
          cub::DeviceScan::InclusiveSum(
              (void *)NULL, temp, (uint32_t *)NULL, (uint32_t *)NULL,
              embedding_params_.batch_size * embedding_params_.max_feature_num);
          temp_storage_scan_bytes_.push_back(temp);

          size_t size = (size_t)ceil((float)temp_storage_scan_bytes_[id] / sizeof(uint32_t));

          temp_storage_scan_tensors_.emplace_back(
              new Tensor<uint32_t>({1, size}, uint32_bufs_.back(), TensorFormat_t::HW));
        }

        utest_forward_temp_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
            {embedding_params_.batch_size * embedding_params_.slot_num,
             embedding_params_.embedding_vec_size},
            fp_bufs_.back(), TensorFormat_t::HW));

        // init GenenralBuffers to do real allocation
#ifndef NDEBUG
        std::cout << " max_feature_num_:" << embedding_params_.max_feature_num << std::endl;
        std::cout << " float_bufs_:" << float_bufs_.back()->get_size() << std::endl;
        std::cout << " fp_bufs_:" << fp_bufs_.back()->get_size() << std::endl;
        std::cout << " uint32_bufs_:" << uint32_bufs_.back()->get_size() << std::endl;
        std::cout << " key_bufs_:" << key_bufs_.back()->get_size() << std::endl;
        std::cout << " value_index_bufs_:" << value_index_bufs_.back()->get_size() << std::endl;
#endif
        float_bufs_.back()->init(cur_device);
        fp_bufs_.back()->init(cur_device);
        uint32_bufs_.back()->init(cur_device);
        key_bufs_.back()->init(cur_device);
        value_index_bufs_.back()->init(cur_device);

        switch (embedding_params_.opt_params.optimizer) {
          case Optimizer_t::Adam:  // adam
            CK_CUDA_THROW_(cudaMemsetAsync(opt_m_tensors_[id]->get_ptr(), 0,
                                           max_vocabulary_size_per_gpu_ *
                                               embedding_params_.embedding_vec_size *
                                               sizeof(TypeEmbeddingComp),
                                           (*Base::device_resources_)[id]->get_stream()));
            CK_CUDA_THROW_(cudaMemsetAsync(opt_v_tensors_[id]->get_ptr(), 0,
                                           max_vocabulary_size_per_gpu_ *
                                               embedding_params_.embedding_vec_size *
                                               sizeof(TypeEmbeddingComp),
                                           (*Base::device_resources_)[id]->get_stream()));
            opt_params_[id].hyperparams.adam.times = 0;
            opt_params_[id].hyperparams.adam.beta1 =
                embedding_params_.opt_params.hyperparams.adam.beta1;
            opt_params_[id].hyperparams.adam.beta2 =
                embedding_params_.opt_params.hyperparams.adam.beta2;
            opt_params_[id].hyperparams.adam.epsilon =
                embedding_params_.opt_params.hyperparams.adam.epsilon;
            opt_params_[id].hyperparams.adam.m_ptr = opt_m_tensors_[id]->get_ptr();
            opt_params_[id].hyperparams.adam.v_ptr = opt_v_tensors_[id]->get_ptr();

            break;

          case Optimizer_t::MomentumSGD:  // momentum_sgd
            CK_CUDA_THROW_(cudaMemsetAsync(opt_momentum_tensors_[id]->get_ptr(), 0,
                                           max_vocabulary_size_per_gpu_ *
                                               embedding_params_.embedding_vec_size *
                                               sizeof(TypeEmbeddingComp),
                                           (*Base::device_resources_)[id]->get_stream()));
            opt_params_[id].hyperparams.momentum.factor =
                embedding_params_.opt_params.hyperparams.momentum.factor;
            opt_params_[id].hyperparams.momentum.momentum_ptr =
                opt_momentum_tensors_[id]->get_ptr();
            break;

          case Optimizer_t::Nesterov:  // nesterov
            CK_CUDA_THROW_(cudaMemsetAsync(opt_accm_tensors_[id]->get_ptr(), 0,
                                           max_vocabulary_size_per_gpu_ *
                                               embedding_params_.embedding_vec_size *
                                               sizeof(TypeEmbeddingComp),
                                           (*Base::device_resources_)[id]->get_stream()));
            opt_params_[id].hyperparams.nesterov.mu =
                embedding_params_.opt_params.hyperparams.nesterov.mu;
            opt_params_[id].hyperparams.nesterov.accm_ptr = opt_accm_tensors_[id]->get_ptr();
            break;

          case Optimizer_t::SGD:
            break;

          default:
            throw std::runtime_error(
                std::string("[HCDEBUG][ERROR] Runtime error: Invalid optimizer type\n"));
        }

      }  // end of for(int id = 0; id < local_gpu_count_; id++)

      functors_.sync_all_gpus(Base::device_resources_, context);

    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }

    return;
  }

  /**
   * This function is used for implementing CPU multi-threads when doing
   * forward() on multi-GPUs. In this case, each CPU thread corresponding
   * to one GPU.
   * @param tid the CPU thread id.
   */
  void forward_per_thread(int tid) {
#ifndef NDEBUG
    MESSAGE_("forward_per_thread: this is thread: " + std::to_string(tid));
#endif

    CudaDeviceContext context((*Base::device_resources_)[tid]->get_device_id());  // set device

    // do update params operation
    functors_.forward_per_gpu(
        embedding_params_.batch_size, embedding_params_.slot_num,
        embedding_params_.embedding_vec_size, Base::row_offsets_tensors_[tid]->get_ptr(),
        Base::value_tensors_[tid]->get_ptr(), nnz_num_per_batch_[tid].get(),
        hash_tables_[tid].get(), hash_table_value_tensors_[tid]->get_ptr(),
        hash_value_index_tensors_[tid]->get_ptr(), embedding_feature_tensors_[tid]->get_ptr(),
        (*Base::device_resources_)[tid]->get_stream());

    return;
  }

  /**
   * The forward propagation of embedding layer.
   */
  void forward() override {
    // Read data from input_buffers_ -> look up -> write to output_tensors

    CudaDeviceContext context((*Base::device_resources_)[0]->get_device_id());

    // use multiple CPU threads to launch tasks on multiple GPUs
    if (local_gpu_count_ > 1) {
      // launch threads
      for (int id = 0; id < local_gpu_count_; id++) {
        Base::device_resources_->results[id] = Base::device_resources_->train_thread_pool.push(
            [this, id](int i) { this->forward_per_thread(id); });
      }

      // wait for threads completion
      for (int id = 0; id < local_gpu_count_; id++) {
        Base::device_resources_->results[id].get();
      }
    } else if (local_gpu_count_ == 1) {  // use current main thread to launch task on one GPU
      forward_per_thread(0);
    } else {
      throw std::runtime_error(
          std::string("[HCDEBUG][ERROR] Runtime error: local_gpu_count <= 0 \n"));
    }

    // do reduce scatter
    int recv_count =
        batch_size_per_gpu_ * embedding_params_.slot_num * embedding_params_.embedding_vec_size;
    functors_.reduce_scatter(recv_count, embedding_feature_tensors_, Base::output_tensors_,
                             Base::device_resources_, context);

    // scale for combiner=mean after reduction
    if (embedding_params_.combiner == 1) {
      int send_count = embedding_params_.batch_size * embedding_params_.slot_num + 1;
      functors_.all_reduce(send_count, Base::row_offsets_tensors_, row_offset_allreduce_tensors_,
                           Base::device_resources_, context);

      // do average
      functors_.forward_scale(embedding_params_.batch_size, embedding_params_.slot_num,
                              embedding_params_.embedding_vec_size, row_offset_allreduce_tensors_,
                              Base::output_tensors_, Base::device_resources_, context);
    }

    return;
  }

  /**
   * The first stage of backward propagation of embedding layer,
   * which only computes the wgrad by the dgrad from the top layer.
   */
  void backward() override {
    // Read dgrad from output_tensors -> compute wgrad

    CudaDeviceContext context((*Base::device_resources_)[0]->get_device_id());

    // do all-gather to collect the top_grad
    int send_count =
        batch_size_per_gpu_ * embedding_params_.slot_num * embedding_params_.embedding_vec_size;
    functors_.all_gather(send_count, Base::output_tensors_, embedding_feature_tensors_,
                         Base::device_resources_, context);

    // do backward
    functors_.backward(embedding_params_.batch_size, embedding_params_.slot_num,
                       embedding_params_.embedding_vec_size, embedding_params_.combiner,
                       row_offset_allreduce_tensors_, embedding_feature_tensors_, wgrad_tensors_,
                       Base::device_resources_, context);

    return;
  }

  /**
   * This function is used for implementing CPU multi-threads when doing
   * update_params() on multi-GPUs. In this case, each CPU thread corresponding
   * to one GPU.
   * @param tid the CPU thread id.
   */
  void update_params_per_thread(int tid) {
#ifndef NDEBUG
    MESSAGE_("update_params_per_thread: this is thread: " + std::to_string(tid));
#endif

    CudaDeviceContext context((*Base::device_resources_)[tid]->get_device_id());

    // accumulate times for adam optimizer
    opt_params_[tid].hyperparams.adam.times++;

    // do update params operation
    functors_.update_params(
        (*Base::device_resources_)[tid]->get_stream(), embedding_params_.batch_size,
        embedding_params_.slot_num, embedding_params_.embedding_vec_size,
        max_vocabulary_size_per_gpu_, opt_params_[tid], (int)nnz_num_per_batch_[tid].get()[0],
        Base::row_offsets_tensors_[tid]->get_ptr(), hash_value_index_tensors_[tid]->get_ptr(),
        sample_id_tensors_[tid]->get_ptr(), sample_id_sort_tensors_[tid]->get_ptr(),
        hash_value_index_sort_tensors_[tid]->get_ptr(),
        hash_value_index_count_offset_tensors_[tid]->get_ptr(),
        new_hash_value_flag_tensors_[tid]->get_ptr(),
        hash_value_flag_sumed_tensors_[tid]->get_ptr(),
        hash_value_index_count_counter_tensors_[tid]->get_ptr(),
        temp_storage_sort_tensors_[tid]->get_ptr(), temp_storage_sort_bytes_[tid],
        temp_storage_scan_tensors_[tid]->get_ptr(), temp_storage_scan_bytes_[tid],
        wgrad_tensors_[tid]->get_ptr(), deltaw_hash_value_index_tensors_[tid]->get_ptr(),
        deltaw_tensors_[tid]->get_ptr(), hash_table_value_tensors_[tid]->get_ptr());

    return;
  }

  /**
   * The second stage of backward propagation of embedding layer, which
   * updates the hash table by wgrad(from backward()) and optimizer.
   */
  void update_params() override {
    if (local_gpu_count_ > 1) {  // use multiple CPU threads to launch tasks on multiple GPUs
      // launch threads
      for (int id = 0; id < local_gpu_count_; id++) {
        Base::device_resources_->results[id] = Base::device_resources_->train_thread_pool.push(
            [this, id](int i) { this->update_params_per_thread(id); });
      }

      // wait for threads completion
      for (int id = 0; id < local_gpu_count_; id++) {
        Base::device_resources_->results[id].get();
      }
    } else if (local_gpu_count_ == 1) {  // use current main thread to launch task on one GPU
      update_params_per_thread(0);
    } else {
      throw std::runtime_error(
          std::string("[HCDEBUG][ERROR] Runtime error: local_gpu_count <= 0 \n"));
    }

    return;
  }

  /**
   * Initialize the embedding table
   */
  void init_params() override {
    functors_.init_embedding(max_vocabulary_size_per_gpu_, embedding_params_.embedding_vec_size,
                             hash_table_value_tensors_, Base::device_resources_);
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

    CudaDeviceContext context((*Base::device_resources_)[0]->get_device_id());

    functors_.upload_params_to_device<TypeHashKey, TypeHashValueIndex>(
        weight_stream, max_vocabulary_size_, embedding_params_.embedding_vec_size,
        max_vocabulary_size_per_gpu_, hash_table_value_tensors_, hash_tables_,
        Base::device_resources_, context);

    return;
  }

  /**
   * Download the hash table from multi-GPUs global memroy to CPU memory
   * and write it to the weight_stream on the host.
   * @param weight_stream the host file stream for writing data to.
   */
  void download_params_to_host(std::ofstream &weight_stream) override {
    // check if the file is opened successfully
    if (!weight_stream.is_open()) {
      CK_THROW_(Error_t::WrongInput, "Error: file not open for writing");
      return;
    }

    CudaDeviceContext context((*Base::device_resources_)[0]->get_device_id());

    functors_.download_params_to_host(
        weight_stream, max_vocabulary_size_, embedding_params_.embedding_vec_size,
        hash_table_value_tensors_, hash_tables_, Base::device_resources_, context);

    return;
  }

  /**
   * Get the total size of hash tables on all GPUs.
   */
  size_t get_params_num() override {
    // Read data from input_buffers_ -> look up -> write to output_tensors

    size_t total_size = 0;

    CudaDeviceContext context((*Base::device_resources_)[0]->get_device_id());

    // need to collect the <key, value> pair count from all GPUs and do sum reduction
    for (int id = 0; id < local_gpu_count_; id++) {
      context.set_device((*Base::device_resources_)[id]->get_device_id());
      total_size += hash_tables_[id]->get_size((*Base::device_resources_)[id]->get_stream());
      CK_CUDA_THROW_(cudaStreamSynchronize((*Base::device_resources_)[id]->get_stream()));
    }

    total_size *= (size_t)embedding_params_.embedding_vec_size;

    return total_size;
  }

  // only used for results check
  /**
   * Get the forward() results from GPUs and copy them to the host pointer
   * embedding_feature. This function is only used for unit test.
   * @param embedding_feature the host pointer for storing the forward()
   * results.
   */
  void get_forward_results(TypeEmbeddingComp *embedding_feature) override {
    CudaDeviceContext context((*Base::device_resources_)[0]->get_device_id());
    int memcpy_size =
        batch_size_per_gpu_ * embedding_params_.slot_num * embedding_params_.embedding_vec_size;

    functors_.get_forward_results(memcpy_size, Base::output_tensors_, embedding_feature,
                                  utest_forward_temp_tensors_, Base::device_resources_, context);

    return;
  }

  /**
   * Get the backward() results from GPUs and copy them to the host pointer
   * wgrad. The wgrad on each GPU should be the same. This function is only
   * used for unit test.
   * @param wgrad the host pointer for stroing the backward() results.
   * @param devIndex the GPU device id.
   */
  void get_backward_results(TypeEmbeddingComp *wgrad, int devIndex) override {
    CudaDeviceContext context((*Base::device_resources_)[0]->get_device_id());

    // wgard shuld be the same on multi-gpus after backward()
    int memcpy_size = embedding_params_.batch_size * embedding_params_.slot_num *
                      embedding_params_.embedding_vec_size;

    functors_.get_backward_results(devIndex, memcpy_size, wgrad_tensors_, wgrad,
                                   Base::device_resources_, context);

    return;
  }

  /**
   * Get the update_params() results(the hash table, including hash_table_keys
   * and hash_table_values) from GPUs and copy them to the host pointers.
   * This function is only used for unit test.
   * @param hash_table_key the host pointer for stroing the hash table keys.
   * @param hash_table_value the host pointer for stroing the hash table values.
   */
  void get_update_params_results(TypeHashKey *hash_table_key, float *hash_table_value) override {
    CudaDeviceContext context((*Base::device_resources_)[0]->get_device_id());

    functors_.get_update_params_results(embedding_params_.embedding_vec_size, max_vocabulary_size_,
                                        hash_table_value_tensors_, hash_tables_, hash_table_key,
                                        hash_table_value, Base::device_resources_, context);

    return;
  }

  /**
   * Set learning rate
   * @param lr
   */
  void set_learning_rate(float lr) override {
    for (int id = 0; id < local_gpu_count_; id++) {
      opt_params_[id].lr = lr;
    }
  }

  /**
   * Check overflow
   * @param lr
   */
  void check_overflow() const override {
    CudaDeviceContext context((*Base::device_resources_)[0]->get_device_id());

    for (int id = 0; id < local_gpu_count_; id++) {
      context.set_device((*Base::device_resources_)[id]->get_device_id());
      size_t count = hash_tables_[id]->get_size((*Base::device_resources_)[id]->get_stream());
      if (count > max_vocabulary_size_per_gpu_) {
        CK_THROW_(Error_t::OutOfBound,
                  "Runtime vocabulary size (" + std::to_string(count) +
                      ") exceeds max_vocabulary_size_per_gpu (" +
                      std::to_string(max_vocabulary_size_per_gpu_) + ") on GPU " +
                      std::to_string((*Base::device_resources_)[id]->get_device_id()) +
                      ", new feature insertion failed.\n");
      }
    }
  }

};  // end of class DistributedSlotSparseEmbeddingHash

}  // namespace HugeCTR
