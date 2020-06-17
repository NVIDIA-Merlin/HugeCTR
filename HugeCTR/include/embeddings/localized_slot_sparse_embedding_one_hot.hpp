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
#include <vector>
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/embeddings/sparse_embedding_hash_functors.hpp"
#include "cub/cub/device/device_radix_sort.cuh"

namespace HugeCTR {
/**
 * The LocalizedSlotSparseEmbeddingOneHot class inherits from Embedding class, which is the base
 * class for implementing all embedding layers. In this class, the slots in the embedding table
 * are assigned to a single GPU seperately, which are called localized slots. For example, slot-0 on
 * GPU-0, slot-1 on GPU-1, slot-2 on GPU-0, slot-3 on GPU-1, etc. This class is very simple to the
 * LocalizedSlotSparseEmbeddingHash, but optimized for performance according to the "one-hot"
 * feature. So, there are several assumptions in this class: 1) The mapping method from keys to
 * embedding row_indices is linear, so there is no hashtable in this class; 2) all the features are
 * one-hot, while multi-hot is not supported in this class; 3) Implement P2P access in forward prop,
 * fused forward_sum+all2all+reorder, so there is no all2all in forward and backward prop, and can
 * only support single node. 4) only support SGD optimizer by now.
 */

template <typename TypeHashKey, typename TypeEmbeddingComp>
class LocalizedSlotSparseEmbeddingOneHot : public Embedding<TypeHashKey, TypeEmbeddingComp> {
  using Base = Embedding<TypeHashKey, TypeEmbeddingComp>;

  using TypeHashValueIndex = TypeHashKey;  // use the hash key type as the hash value_index type(it
                                           // will be uint32 or int64)
  using NvHashTable =
      HashTable<TypeHashKey, TypeHashValueIndex, std::numeric_limits<TypeHashKey>::max()>;

 private:
  SparseEmbeddingHashParams<TypeEmbeddingComp>
      embedding_params_; /**< Sparse embedding hash params. */

  std::vector<OptParams<TypeEmbeddingComp>> opt_params_;  /**< Optimizer params. */
  std::vector<std::shared_ptr<NvHashTable>> hash_tables_; /**< Hash table.  */
  std::vector<PinnedBuffer<TypeHashKey>>
      nnz_num_per_batch_; /**< non-zero feature number in one batch */

  // define tensors
  Tensors<float> hash_table_value_tensors_;                /**< Hash table value. */
  Tensors<TypeHashValueIndex> hash_table_slot_id_tensors_; /**< the tensors for storing slot ids */
  Tensors<TypeHashValueIndex>
      hash_value_index_tensors_; /**< Hash value index. The index is corresponding to the line
                                    number of the value. */
  Tensors<TypeEmbeddingComp> embedding_feature_tensors_; /**< the output tensor of the forward(). */
  TypeEmbeddingComp **embedding_features_;
  Tensors<TypeEmbeddingComp> wgrad_tensors_; /**< the input tensor of the backward(). */

  // define GeneralBuffers
  GeneralBuffers<float> float_bufs_; /**< float type general buffer. */
  GeneralBuffers<TypeEmbeddingComp>
      fp_bufs_; /**< TypeEmbeddingComp(fp32 or fp16) type general buffer. */
  GeneralBuffers<uint32_t> uint32_bufs_; /**< uint32 type general buffer. */
  GeneralBuffers<TypeHashKey> key_bufs_; /**< TypeHashKey type general buffer. */
  GeneralBuffers<TypeHashValueIndex>
      value_index_bufs_; /**< TypeHashValueIndex type general buffer. */

  size_t max_vocabulary_size_per_gpu_; /**< Max vocabulary size for each GPU. */
  size_t max_hash_table_size_per_gpu_; /**< equal to max_vocabulary_size_per_gpu_ / load_factor. */
  int batch_size_per_gpu_;             /*< batch_size per GPU */
  std::vector<int> slot_num_per_gpu_;  /* slot_num per GPU */
  int total_gpu_count_;
  int local_gpu_count_;

  SparseEmbeddingHashFunctors functors_; /**< obj of SparseEmbeddingHashFunctors */

  Tensors<TypeEmbeddingComp> all2all_tensors_; /**< the temple buffer to store all2all results */
  Tensors<TypeEmbeddingComp> utest_all2all_tensors_;
  Tensors<TypeEmbeddingComp> utest_reorder_tensors_;
  Tensors<TypeEmbeddingComp> utest_backward_temp_tensors_;
  Tensors<TypeEmbeddingComp> utest_forward_temp_tensors_;

  Tensors<uint32_t> mapping_offsets_per_gpu_tensors_;

  /**
   * The constructor of LocalizedSlotSparseEmbeddingOneHot.
   * This ctor is only used when you already have a instant of LocalizedSlotSparseEmbeddingOneHot
   * and you want to reuse the hash_table/embedding_table for evaluation.
   * @param row_offsets_tensors row offsets of the input tensor(refer to row offset vector in sparse
   * matrix CSR format).
   * @param hash_key_tensors hash keys of the input tensor(refer to value vector in sparse matrix
   * CSR format).
   * @param batchsize batsize. The batchsize for eval and train may be different.
   * @param gpu_resource_group the GPU resource group.
   * @param obj the current LocalizedSlotSparseEmbeddingOneHot class object.
   */
  LocalizedSlotSparseEmbeddingOneHot(const Tensors<TypeHashKey> &row_offsets_tensors,
                                     const Tensors<TypeHashKey> &value_tensors, size_t batchsize,
                                     const std::shared_ptr<GPUResourceGroup> &gpu_resource_group,
                                     const LocalizedSlotSparseEmbeddingOneHot &obj);

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
        new LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>(
            row_offsets_tensors, value_tensors, batchsize, gpu_resource_group, *this);
    return new_embedding;
  }

  /**
   * The constructor of LocalizedSlotSparseEmbeddingOneHot.
   * @param row_offsets_tensors row offsets of the input tensor(refer to row offset vector in sparse
   * matrix CSR format).
   * @param hash_key_tensors hash keys of the input tensor(refer to value vector in sparse matrix
   * CSR format).
   * @param embedding_params embedding params for initialization.
   * @param gpu_resource_group the GPU resource group
   */
  LocalizedSlotSparseEmbeddingOneHot(const Tensors<TypeHashKey> &row_offsets_tensors,
                                     const Tensors<TypeHashKey> &hash_key_tensors,
                                     SparseEmbeddingHashParams<TypeEmbeddingComp> embedding_params,
                                     const std::string plan_file,
                                     const std::shared_ptr<GPUResourceGroup> &gpu_resource_group,
                                     std::vector<size_t> slot_sizes);
  /**
   * This function is used for implementing CPU multi-threads when doing
   * forward() on multi-GPUs. In this case, each CPU thread corresponding
   * to one GPU.
   * @param tid the CPU thread id.
   */
  void forward_per_thread(int tid);
  /**
   * The forward propagation of embedding layer.
   */
  void forward() override;
  /**
   * The first stage of backward propagation of embedding layer,
   * which computes the wgrad by the dgrad from the top layer.
   */
  void backward() override;
  /**
   * This function is used for implementing CPU multi-threads when doing
   * update_params() on multi-GPUs. In this case, each CPU thread corresponding
   * to one GPU.
   * @param tid the CPU thread id.
   */
  void update_params_per_thread(int tid);
  /**
   * The second stage of backward propagation of embedding layer, which
   * updates the hash table by wgrad(from backward()) and optimizer.
   */
  void update_params() override;
  /**
   * Read the hash table from the weight_stream on the host, and
   * upload it onto multi-GPUs global memory.
   * @param weight_stream the host file stream for reading data from.
   */
  void upload_params_to_device(std::ifstream &weight_stream) override;
  /**
   * Download the hash table from multi-GPUs global memroy to CPU memory
   * and write it to the weight_stream on the host.
   * @param weight_stream the host file stream for writing data to.
   */
  void download_params_to_host(std::ofstream &weight_stream) override;
  /**
   * Get the total size of hash tables on all GPUs.
   */
  size_t get_params_num() override;

  // only used for results check
  /**
   * Get the forward() results from GPUs and copy them to the host pointer
   * embedding_feature. This function is only used for unit test.
   * @param embedding_feature the host pointer for storing the forward()
   * results.
   */
  void get_forward_results(TypeEmbeddingComp *embedding_feature) override;
  /**
   * Get the backward() results from GPUs and copy them to the host pointer
   * wgrad. The wgrad on each GPU should be the same. This function is only
   * used for unit test.
   * @param wgrad the host pointer for stroing the backward() results.
   * @param devIndex the GPU device id.
   */
  void get_backward_results(TypeEmbeddingComp *wgrad, int devIndex) override;
  /**
   * Get the update_params() results(the hash table, including hash_table_keys
   * and hash_table_values) from GPUs and copy them to the host pointers.
   * This function is only used for unit test.
   * @param hash_table_key the host pointer for stroing the hash table keys.
   * @param hash_table_value the host pointer for stroing the hash table values.
   */
  void get_update_params_results(TypeHashKey *hash_table_key, float *hash_table_value) override;

  void set_learning_rate(float lr) override;

};  // end of class LocalizedSlotSparseEmbeddingOneHot

template <typename TypeHashKey, typename TypeEmbeddingComp>
LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::
    LocalizedSlotSparseEmbeddingOneHot(
        const Tensors<TypeHashKey> &row_offsets_tensors,
        const Tensors<TypeHashKey> &hash_key_tensors,
        SparseEmbeddingHashParams<TypeEmbeddingComp> embedding_params, const std::string plan_file,
        const std::shared_ptr<GPUResourceGroup> &gpu_resource_group, std::vector<size_t> slot_sizes)
    : embedding_params_(embedding_params),
      Base(row_offsets_tensors, hash_key_tensors, embedding_params.batch_size,
           embedding_params.slot_num, embedding_params.embedding_vec_size, gpu_resource_group,
           embedding_params.opt_params.scaler) {
  try {
    total_gpu_count_ = Base::device_resources_->get_total_gpu_count();
    local_gpu_count_ = Base::device_resources_->size();
    CudaDeviceContext context((*Base::device_resources_)[0]->get_device_id());

    if (slot_sizes.size() == 0) {
      // CAUSION: The users need to gaurantee the given vocabulary size is big enough since
      // the embedding table is ditrubuted by slot and the size of the slot maybe non-uniform.
      float scaler = (ceil((float)embedding_params_.slot_num / total_gpu_count_) /
                      floor((float)embedding_params_.slot_num / total_gpu_count_));
      max_vocabulary_size_per_gpu_ =
          (size_t)(ceil((float)embedding_params_.vocabulary_size / total_gpu_count_ * scaler));
    } else {
      max_vocabulary_size_per_gpu_ = functors_.cal_max_voc_size_per_gpu(
          total_gpu_count_, local_gpu_count_, slot_sizes, Base::device_resources_);
    }
    max_hash_table_size_per_gpu_ = max_vocabulary_size_per_gpu_ / embedding_params_.load_factor;

    MESSAGE_("max_vocabulary_size_per_gpu_=" + std::to_string(max_vocabulary_size_per_gpu_));

    batch_size_per_gpu_ = embedding_params_.batch_size / total_gpu_count_;

    for (int id = 0; id < local_gpu_count_; id++) {
      int cur_device = (*Base::device_resources_)[id]->get_device_id();
      context.set_device(cur_device);

      int gid = Base::device_resources_->get_global_id(cur_device);
      int slot_num_per_gpu = embedding_params_.slot_num / total_gpu_count_ +
                             ((gid < (int)(embedding_params_.slot_num % total_gpu_count_)) ? 1 : 0);
      slot_num_per_gpu_.push_back(slot_num_per_gpu);

      // new nnz vectors
      nnz_num_per_batch_.push_back(PinnedBuffer<TypeHashKey>(1));

      // construct HashTable object: used to store hash table <key, value_index>
      hash_tables_.emplace_back(new NvHashTable(max_hash_table_size_per_gpu_));

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
          {embedding_params_.batch_size * slot_num_per_gpu, embedding_params_.embedding_vec_size},
          fp_bufs_.back(), TensorFormat_t::HW));

      // new wgrad used by backward
      wgrad_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
          {embedding_params_.batch_size * slot_num_per_gpu, embedding_params_.embedding_vec_size},
          fp_bufs_.back(), TensorFormat_t::HW));

      // new optimizer params used by update_params
      opt_params_.push_back(OptParams<TypeEmbeddingComp>());
      opt_params_[id].optimizer = embedding_params_.opt_params.optimizer;
      opt_params_[id].lr = embedding_params_.opt_params.lr;
      opt_params_[id].global_update = embedding_params_.opt_params.global_update;
      opt_params_[id].scaler = embedding_params_.opt_params.scaler;
      switch (embedding_params_.opt_params.optimizer) {
        case Optimizer_t::SGD:
          break;

        default:
          throw std::runtime_error(
              std::string("[HCDEBUG][ERROR] Runtime error: Invalid optimizer type\n"));
      }

      // the tenosrs for storing slot ids
      // TODO: init to -1 ?
      hash_table_slot_id_tensors_.emplace_back(new Tensor<TypeHashValueIndex>(
          {max_vocabulary_size_per_gpu_, 1}, value_index_bufs_.back(), TensorFormat_t::HW));

      // temp tensors for all2all
      all2all_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
          {batch_size_per_gpu_ * embedding_params_.slot_num, embedding_params_.embedding_vec_size},
          fp_bufs_.back(), TensorFormat_t::HW));

      utest_forward_temp_tensors_.emplace_back(
          new Tensor<TypeEmbeddingComp>({embedding_params_.batch_size * embedding_params_.slot_num,
                                         embedding_params_.embedding_vec_size},
                                        fp_bufs_.back(), TensorFormat_t::HW));
      utest_all2all_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
          {batch_size_per_gpu_ * embedding_params_.slot_num, embedding_params_.embedding_vec_size},
          fp_bufs_.back(), TensorFormat_t::HW));
      utest_reorder_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
          {batch_size_per_gpu_ * embedding_params_.slot_num, embedding_params_.embedding_vec_size},
          fp_bufs_.back(), TensorFormat_t::HW));
      utest_backward_temp_tensors_.emplace_back(
          new Tensor<TypeEmbeddingComp>({embedding_params_.batch_size * embedding_params_.slot_num,
                                         embedding_params_.embedding_vec_size},
                                        fp_bufs_.back(), TensorFormat_t::HW));

      mapping_offsets_per_gpu_tensors_.emplace_back(new Tensor<uint32_t>(
          {1, (size_t)slot_num_per_gpu}, uint32_bufs_.back(), TensorFormat_t::HW));

      // init GenenralBuffers to do real allocation
#ifndef NDEBUG
      std::cout << " max_feature_num_:" << embedding_params_.max_feature_num;
      std::cout << " float_bufs_:" << float_bufs_.back()->get_size();
      std::cout << " fp_bufs_:" << fp_bufs_.back()->get_size();
      std::cout << " uint32_bufs_:" << uint32_bufs_.back()->get_size();
      std::cout << " key_bufs_:" << key_bufs_.back()->get_size();
      std::cout << " value_index_bufs_:" << value_index_bufs_.back()->get_size() << std::endl;
#endif
      float_bufs_.back()->init(cur_device);
      fp_bufs_.back()->init(cur_device);
      uint32_bufs_.back()->init(cur_device);
      key_bufs_.back()->init(cur_device);
      value_index_bufs_.back()->init(cur_device);

    }  // end of for(int id = 0; id < local_gpu_count_; id++)

    // sync
    functors_.sync_all_gpus(Base::device_resources_, context);

    // do hash table value initialization
    if (slot_sizes.size() == 0) {  // if no slot_sizes provided, use the old method to init
      functors_.init_embedding(max_vocabulary_size_per_gpu_, embedding_params_.embedding_vec_size,
                               hash_table_value_tensors_, Base::device_resources_);

    } else {
      if (slot_sizes.size() == embedding_params_.slot_num) {
        size_t total = 0;
        for (auto slot_size : slot_sizes) {
          total += slot_size;
        }
        if (total != embedding_params_.vocabulary_size) {
          throw std::runtime_error(std::string(
              "[HCDEBUG][ERROR] Runtime error: the total sum of slot_sizes != vocabulary_size\n"));
        }
        functors_.init_embedding(slot_sizes, embedding_params_.embedding_vec_size,
                                 hash_table_value_tensors_, hash_tables_,
                                 hash_table_slot_id_tensors_, Base::device_resources_);
      } else {
        throw std::runtime_error(
            std::string("[HCDEBUG][ERROR] Runtime error: the size of slot_sizes != slot_num\n"));
      }
    }

    // get the mapping table between local value_index and input value_index
    for (int id = 0; id < local_gpu_count_; id++) {
      uint32_t slot_sizes_prefix_sum = 0;
      uint32_t slot_sizes_prefix_sum_local = 0;
      int slot_num = 0;
      for (int i = 0; i < (int)(slot_sizes.size()); i++) {
        int device_id = (*Base::device_resources_)[id]->get_device_id();
        int global_id = Base::device_resources_->get_global_id(device_id);
        size_t slot_size = slot_sizes[i];
        if ((i % total_gpu_count_) == global_id) {
          uint32_t mapping_offset = slot_sizes_prefix_sum - slot_sizes_prefix_sum_local;
          CK_CUDA_THROW_(cudaMemcpy(&((mapping_offsets_per_gpu_tensors_[id]->get_ptr())[slot_num]),
                                    &mapping_offset, sizeof(uint32_t), cudaMemcpyHostToDevice));
          slot_sizes_prefix_sum_local += slot_size;
          slot_num++;
        }
        slot_sizes_prefix_sum += slot_size;
      }
    }

    // peer2peer access
    auto &device_list = gpu_resource_group->get_device_list();
    for (int dev_i : device_list) {
      CudaDeviceContext context(dev_i);  // set device
      for (int dev_j : device_list) {
        if (dev_j != dev_i) {
          cudaError_t ret =
              cudaDeviceEnablePeerAccess(dev_j, 0);  // enable p2p access between gpu i and j
          if (ret != cudaSuccess && ret != cudaErrorPeerAccessAlreadyEnabled) {
            CK_CUDA_THROW_(ret);
          } else {
            // cudaErrorPeerAccessAlreadyEnabled must not be handled as an error
            // so we reset it to cudaSuccess here
            cudaGetLastError();
          }
        }
      }
    }

    // unified memory for 2D pointer
    CK_CUDA_THROW_(
        cudaMallocManaged(&embedding_features_, sizeof(TypeEmbeddingComp *) * local_gpu_count_));
    for (int id = 0; id < local_gpu_count_; id++) {
      embedding_features_[id] = Base::output_tensors_[id]->get_ptr();
    }

    // warm up for nccl all2all
    // #ifdef NCCL_A2A
    //     if(total_gpu_count_ > 1) {
    //       functors_.all2all_forward(batch_size_per_gpu_, slot_num_per_gpu_,
    //                                 embedding_params_.embedding_vec_size,
    //                                 embedding_feature_tensors_, all2all_tensors_,
    //                                 Base::device_resources_);
    //     }
    // #endif

  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}  // end of LocalizedSlotSparseEmbeddingOneHot()

// Ctor only used for eval
template <typename TypeHashKey, typename TypeEmbeddingComp>
LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::
    LocalizedSlotSparseEmbeddingOneHot(const Tensors<TypeHashKey> &row_offsets_tensors,
                                       const Tensors<TypeHashKey> &value_tensors, size_t batchsize,
                                       const std::shared_ptr<GPUResourceGroup> &gpu_resource_group,
                                       const LocalizedSlotSparseEmbeddingOneHot &obj)
    : embedding_params_(obj.embedding_params_),
      total_gpu_count_(obj.total_gpu_count_),
      local_gpu_count_(obj.local_gpu_count_),
      max_vocabulary_size_per_gpu_(obj.max_vocabulary_size_per_gpu_),
      max_hash_table_size_per_gpu_(obj.max_hash_table_size_per_gpu_),
      slot_num_per_gpu_(obj.slot_num_per_gpu_),
      hash_tables_(obj.hash_tables_),
      hash_table_value_tensors_(obj.hash_table_value_tensors_),
      hash_table_slot_id_tensors_(obj.hash_table_slot_id_tensors_),
      mapping_offsets_per_gpu_tensors_(obj.mapping_offsets_per_gpu_tensors_),
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

      // new hash table value_index that get() from HashTable
      hash_value_index_tensors_.emplace_back(new Tensor<TypeHashValueIndex>(
          {1, embedding_params_.batch_size * embedding_params_.max_feature_num},
          value_index_bufs_.back(), TensorFormat_t::HW));

      // new embedding features reduced by hash table values(results of forward)
      embedding_feature_tensors_.emplace_back(
          new Tensor<TypeEmbeddingComp>({embedding_params_.batch_size * slot_num_per_gpu_[id],
                                         embedding_params_.embedding_vec_size},
                                        fp_bufs_.back(), TensorFormat_t::HW));

      // temp tensors for all2all
      all2all_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
          {batch_size_per_gpu_ * embedding_params_.slot_num, embedding_params_.embedding_vec_size},
          fp_bufs_.back(), TensorFormat_t::HW));

      utest_forward_temp_tensors_.emplace_back(
          new Tensor<TypeEmbeddingComp>({embedding_params_.batch_size * embedding_params_.slot_num,
                                         embedding_params_.embedding_vec_size},
                                        fp_bufs_.back(), TensorFormat_t::HW));

      // init GenenralBuffers to do real allocation
#ifndef NDEBUG
      std::cout << " fp_bufs_:" << fp_bufs_.back()->get_size();
      std::cout << " value_index_bufs_:" << value_index_bufs_.back()->get_size() << std::endl;
#endif
      fp_bufs_.back()->init(cur_device);
      value_index_bufs_.back()->init(cur_device);

    }  // end of for(int id = 0; id < local_gpu_count_; id++)

    // sync
    functors_.sync_all_gpus(Base::device_resources_, context);

    // unified memory for 2D pointer
    CK_CUDA_THROW_(
        cudaMallocManaged(&embedding_features_, sizeof(TypeEmbeddingComp *) * local_gpu_count_));
    for (int id = 0; id < local_gpu_count_; id++) {
      embedding_features_[id] = Base::output_tensors_[id]->get_ptr();
    }

  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
size_t LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::get_params_num() {
  return (embedding_params_.vocabulary_size * embedding_params_.embedding_vec_size);
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::forward_per_thread(
    int tid) {
#ifndef NDEBUG
  MESSAGE_("forward_per_thread: this is thread: " + std::to_string(tid));
#endif

  CudaDeviceContext context((*Base::device_resources_)[tid]->get_device_id());  // set device

  // functors_.forward_per_gpu(embedding_params_.batch_size, slot_num_per_gpu_[tid],
  //                           embedding_params_.embedding_vec_size, embedding_params_.combiner,
  //                           Base::row_offsets_tensors_[tid]->get_ptr(),
  //                           Base::value_tensors_[tid]->get_ptr(), nnz_num_per_batch_[tid].get(),
  //                           mapping_offsets_per_gpu_tensors_[tid]->get_ptr(),
  //                           hash_table_value_tensors_[tid]->get_ptr(),
  //                           hash_value_index_tensors_[tid]->get_ptr(),
  //                           embedding_feature_tensors_[tid]->get_ptr(),
  //                           (*Base::device_resources_)[tid]->get_stream());

  // for forward_fuse method
  functors_.forward_mapping_per_gpu(
      embedding_params_.batch_size, slot_num_per_gpu_[tid],
      Base::row_offsets_tensors_[tid]->get_ptr(), Base::value_tensors_[tid]->get_ptr(),
      nnz_num_per_batch_[tid].get(), mapping_offsets_per_gpu_tensors_[tid]->get_ptr(),
      hash_value_index_tensors_[tid]->get_ptr(), (*Base::device_resources_)[tid]->get_stream());
  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::forward() {
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

  //   // do all-to-all
  // #ifdef NCCL_A2A
  //   if(total_gpu_count_ > 1) {
  //     functors_.all2all_forward(batch_size_per_gpu_, slot_num_per_gpu_,
  //                               embedding_params_.embedding_vec_size,
  //                               embedding_feature_tensors_, all2all_tensors_,
  //                               Base::device_resources_);
  //   }
  //   else {
  //     CK_CUDA_THROW_(cudaMemcpyAsync(all2all_tensors_[0]->get_ptr(),
  //     embedding_feature_tensors_[0]->get_ptr(),
  //                                     (size_t)batch_size_per_gpu_ * slot_num_per_gpu_[0] *
  //                                     embedding_params_.embedding_vec_size * \
//                                     sizeof(TypeEmbeddingComp), cudaMemcpyDeviceToDevice,
  //                                     (*Base::device_resources_)[0]->get_stream()));
  //   }
  // #else
  //    // sync: guarantee the data is ready for all2all
  //   functors_.sync_all_gpus(Base::device_resources_, context);
  //   functors_.all2all_exec<TypeEmbeddingComp>(all2all_forward_);
  // #endif

  //   // reorder
  //   functors_.forward_reorder(batch_size_per_gpu_, embedding_params_.slot_num,
  //                             embedding_params_.embedding_vec_size, all2all_tensors_,
  //                             Base::output_tensors_, Base::device_resources_, context);

  // fuse forward+all2all+reorder into one kernel
  functors_.forward_fuse(embedding_params_.batch_size, embedding_params_.slot_num,
                         slot_num_per_gpu_, embedding_params_.embedding_vec_size,
                         embedding_params_.combiner, Base::row_offsets_tensors_,
                         hash_table_value_tensors_, hash_value_index_tensors_, embedding_features_,
                         Base::device_resources_);

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::backward() {
  // Read dgrad from output_tensors -> compute wgrad

  //   CudaDeviceContext context((*Base::device_resources_)[0]->get_device_id());

  //   // reorder
  //   functors_.backward_reorder(batch_size_per_gpu_, embedding_params_.slot_num,
  //                              embedding_params_.embedding_vec_size, Base::output_tensors_,
  //                              all2all_tensors_, Base::device_resources_, context);

  //   // do all2all
  // #ifdef NCCL_A2A
  //   if(total_gpu_count_ > 1) {
  //     functors_.all2all_backward(batch_size_per_gpu_, slot_num_per_gpu_,
  //                               embedding_params_.embedding_vec_size,
  //                               all2all_tensors_, wgrad_tensors_,
  //                               Base::device_resources_);
  //   }
  //   else {
  //     CK_CUDA_THROW_(cudaMemcpyAsync(wgrad_tensors_[0]->get_ptr(),
  //     all2all_tensors_[0]->get_ptr(),
  //                                     (size_t)batch_size_per_gpu_ *  slot_num_per_gpu_[0] *
  //                                     embedding_params_.embedding_vec_size * \
//                                     sizeof(TypeEmbeddingComp), cudaMemcpyDeviceToDevice,
  //                                     (*Base::device_resources_)[0]->get_stream()));
  //   }
  // #else
  //   // do not support gossip
  //   MESSAGE_("Error: Not support gossip in backward for one-hot");
  // #endif

  // fuse reorder+all2all+backward into one kernel
  functors_.backward_fuse(embedding_params_.batch_size, (int)embedding_params_.slot_num,
                          slot_num_per_gpu_, (int)embedding_params_.embedding_vec_size,
                          embedding_params_.combiner, embedding_features_, wgrad_tensors_,
                          Base::device_resources_);

  return;
}  // end of backward()

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::update_params_per_thread(
    int tid) {
#ifndef NDEBUG
  MESSAGE_("update_params_per_thread: this is thread: " + std::to_string(tid));
#endif

  CudaDeviceContext context((*Base::device_resources_)[tid]->get_device_id());

  // accumulate times for adam optimizer
  opt_params_[tid].hyperparams.adam.times++;

  // do update params operation: only support SGD
  functors_.update_params((*Base::device_resources_)[tid]->get_stream(),
                          embedding_params_.embedding_vec_size, opt_params_[tid],
                          (int)nnz_num_per_batch_[tid].get()[0],
                          hash_value_index_tensors_[tid]->get_ptr(), wgrad_tensors_[tid]->get_ptr(),
                          hash_table_value_tensors_[tid]->get_ptr());

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::update_params() {
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

// read hash_table_key and hash_table_value from host file, and write to GPU
template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::upload_params_to_device(
    std::ifstream &weight_stream) {
#ifndef NDEBUG
  MESSAGE_("upload_params_to_device");
#endif

  // check if file is opened successfully
  if (!weight_stream.is_open()) {
    CK_THROW_(Error_t::WrongInput, "Error: file not open for reading");
  }

  CudaDeviceContext context((*Base::device_resources_)[0]->get_device_id());

  // Note: not verify (without hashtable for one-hot)
  functors_.upload_params_to_device<TypeHashKey, TypeHashValueIndex>(
      weight_stream, embedding_params_.vocabulary_size, embedding_params_.embedding_vec_size,
      max_vocabulary_size_per_gpu_, hash_table_value_tensors_, hash_table_slot_id_tensors_,
      hash_tables_, Base::device_resources_, context);

  return;
}  // end of upload_params_to_device()

// read hash_table_key and hash_table_value from GPU, and write to the file on the host
template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::download_params_to_host(
    std::ofstream &weight_stream) {
  // check if the file is opened successfully
  if (!weight_stream.is_open()) {
    CK_THROW_(Error_t::WrongInput, "Error: file not open for writing");
    return;
  }

  CudaDeviceContext context((*Base::device_resources_)[0]->get_device_id());

  // Note: not verify (without hashtable for one-hot)
  functors_.download_params_to_host(
      weight_stream, embedding_params_.vocabulary_size, embedding_params_.embedding_vec_size,
      max_hash_table_size_per_gpu_, hash_table_value_tensors_, hash_table_slot_id_tensors_,
      hash_tables_, Base::device_resources_, context);

  return;
}  // end of download_params_to_host()

// only used for results check: copy forward results from output_tensors_ to embedding_feature(input
// CPU buffer)
template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::get_forward_results(
    TypeEmbeddingComp *embedding_feature) {
  CudaDeviceContext context((*Base::device_resources_)[0]->get_device_id());

  int memcpy_size =
      batch_size_per_gpu_ * embedding_params_.slot_num * embedding_params_.embedding_vec_size;

  functors_.get_forward_results(memcpy_size, Base::output_tensors_, embedding_feature,
                                utest_forward_temp_tensors_, Base::device_resources_, context);

  return;
}  // end of get_forward_results()

// only used for results check: copy backward() results from wgrad_tensors_ to wgrad(input CPU
// buffer)
template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::get_backward_results(
    TypeEmbeddingComp *wgrad, int devIndex) {
  CudaDeviceContext context((*Base::device_resources_)[0]->get_device_id());

#ifdef NCCL_A2A
  if (total_gpu_count_ > 1) {
    functors_.all2all_forward(batch_size_per_gpu_, slot_num_per_gpu_,
                              embedding_params_.embedding_vec_size, wgrad_tensors_,
                              utest_all2all_tensors_, Base::device_resources_);
  } else {
    CK_CUDA_THROW_(
        cudaMemcpyAsync(utest_all2all_tensors_[0]->get_ptr(), wgrad_tensors_[0]->get_ptr(),
                        (size_t)batch_size_per_gpu_ * slot_num_per_gpu_[0] *
                            embedding_params_.embedding_vec_size * sizeof(TypeEmbeddingComp),
                        cudaMemcpyDeviceToDevice, (*Base::device_resources_)[0]->get_stream()));
  }
#else
  // do not support gossip
  MESSAGE_("Error: Not support gossip in backward for one-hot");
#endif

  // reorder
  functors_.forward_reorder(batch_size_per_gpu_, embedding_params_.slot_num,
                            embedding_params_.embedding_vec_size, utest_all2all_tensors_,
                            utest_reorder_tensors_, Base::device_resources_, context);

  // there are batch_size_per_gpu samples' wgard on each GPU
  size_t memcpy_size = (size_t)batch_size_per_gpu_ * embedding_params_.slot_num *
                       embedding_params_.embedding_vec_size;

  // nccl gather
  functors_.all_gather(memcpy_size,
                       utest_reorder_tensors_,        // send
                       utest_backward_temp_tensors_,  // recv
                       Base::device_resources_, context);

  // memcpy H2D
  functors_.get_backward_results(devIndex, total_gpu_count_ * memcpy_size,
                                 utest_backward_temp_tensors_, wgrad, Base::device_resources_,
                                 context);

  return;
}  // end of get_backward_results()

// only used for results check: copy hash_tabale <key, value> from gpu to cpu
template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::get_update_params_results(
    TypeHashKey *hash_table_key, float *hash_table_value) {
  CudaDeviceContext context((*Base::device_resources_)[0]->get_device_id());

  // Note: not verify (without hashtable for one-hot)
  functors_.get_update_params_results(
      max_hash_table_size_per_gpu_, embedding_params_.embedding_vec_size,
      embedding_params_.vocabulary_size, hash_table_value_tensors_, hash_tables_, hash_table_key,
      hash_table_value, Base::device_resources_, context);

  return;

}  // end of get_update_params_results()

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::set_learning_rate(
    float lr) {
  for (int id = 0; id < local_gpu_count_; id++) {
    opt_params_[id].lr = lr;
  }
}

}  // namespace HugeCTR
