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
#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"

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

 private:
  // define tensors
  TensorPtrs<float> hash_table_value_tensors_; /**< Hash table value. */
  TensorPtrs<size_t>
      hash_table_slot_id_tensors_; /**< the tensors for storing slot ids */
  TensorPtrs<size_t>
      hash_value_index_tensors_; /**< Hash value index. The index is corresponding to the line
                                    number of the value. */
  TensorPtrs<TypeEmbeddingComp>
      embedding_feature_tensors_; /**< the output tensor of the forward(). */
  TypeEmbeddingComp **train_embedding_features_;
  TypeEmbeddingComp **evaluate_embedding_features_;
  TensorPtrs<TypeEmbeddingComp> wgrad_tensors_; /**< the input tensor of the backward(). */

  // define GeneralBuffers
  GeneralBufferPtrs<float> float_bufs_; /**< float type general buffer. */
  GeneralBufferPtrs<TypeEmbeddingComp>
      fp_bufs_; /**< TypeEmbeddingComp(fp32 or fp16) type general buffer. */
  GeneralBufferPtrs<uint32_t> uint32_bufs_; /**< uint32 type general buffer. */
  GeneralBufferPtrs<TypeHashKey> key_bufs_; /**< TypeHashKey type general buffer. */
  GeneralBufferPtrs<size_t>
      value_index_bufs_; /**< TypeHashValueIndex type general buffer. */

  size_t max_vocabulary_size_;
  size_t max_vocabulary_size_per_gpu_;   /**< Max vocabulary size for each GPU. */
  std::vector<size_t> slot_num_per_gpu_; /* slot_num per GPU */
  std::vector<size_t> slot_size_array_;

  SparseEmbeddingFunctors functors_;

  TensorPtrs<TypeEmbeddingComp> all2all_tensors_; /**< the temple buffer to store all2all results */
  TensorPtrs<TypeEmbeddingComp> utest_all2all_tensors_;
  TensorPtrs<TypeEmbeddingComp> utest_reorder_tensors_;
  TensorPtrs<TypeEmbeddingComp> utest_backward_temp_tensors_;
  TensorPtrs<TypeEmbeddingComp> utest_forward_temp_tensors_;

  TensorPtrs<uint32_t> mapping_offsets_per_gpu_tensors_;

  TypeEmbeddingComp **get_embedding_features() {
    if (Base::in_train()) {
      return train_embedding_features_;
    } else {
      return evaluate_embedding_features_;
    }
  }

  /**
   * Calculate the max vocabulary size per GPU.
   * @param total_gpu_count total GPU count.
   * @param local_gpu_count local GPU count.
   * @param slot_sizes an array which stores the size of the slots to be intialized.
   * @param device_resources GPU device resources.
   */
  size_t cal_max_voc_size_per_gpu(size_t total_gpu_count, size_t local_gpu_count,
                                  const std::vector<size_t> slot_sizes,
                                  const GPUResourceGroup &device_resources) {
    size_t max_voc_size = 0;
    for (size_t id = 0; id < local_gpu_count; id++) {
      size_t device_id = device_resources[id].get_device_id();
      size_t global_id = device_resources.get_global_id(device_id);

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
   * @param slot_sizes an array which stores the size of the slots to be intialized.
   * @param embedding_vec_size embedding vector size.
   * @param hash_table_value_tensors embedding table tensors.
   * @param hash_table_slot_id_tensors slot ids tensors.
   * @param device_resources GPU device resources.
   */
  void init_embedding(const std::vector<size_t> slot_sizes, size_t embedding_vec_size,
                      const TensorPtrs<float> &hash_table_value_tensors,
                      const TensorPtrs<size_t> &hash_table_slot_id_tensors,
                      const GPUResourceGroup &device_resources);

  /**
   * upload_params_to_device() for LocalizedSlotSparseEmbeddingOnehot
   * @param weight_stream weight file stream to read.
   * @param embedding_vec_size embedding vector size.
   * @param hash_table_value_tensors the hash table value on multi GPUs.
   * @param slot_sizes the size for each slot
   * @param mapping_offsets_per_gpu_tensors the mapping offset of each slot on every GPU
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  void upload_params_to_device(std::ifstream &weight_stream, size_t embedding_vec_size,
                               TensorPtrs<float> &hash_table_value_tensors,
                               const std::vector<size_t> &slot_sizes,
                               const TensorPtrs<uint32_t> &mapping_offsets_per_gpu_tensors,
                               const GPUResourceGroup &device_resources);

  /**
   * download_params_to_host for LocalizedSlotSparseEmbeddingOnehot.
   * @param weight_stream weight file stream to write.
   * @param embedding_vec_size embedding vector size.
   * @param hash_table_value_tensors the hash table value on multi-GPU.
   * @param slot_sizes the size for each slot
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  void download_params_to_host(std::ofstream &weight_stream, size_t embedding_vec_size,
                               const TensorPtrs<float> &hash_table_value_tensors,
                               const std::vector<size_t> &slot_sizes,
                               const GPUResourceGroup &device_resources) const;

 public:
  /**
   * The constructor of LocalizedSlotSparseEmbeddingOneHot.
   * @param row_offsets_tensors row offsets of the input tensor(refer to row offset vector in sparse
   * matrix CSR format).
   * @param hash_key_tensors hash keys of the input tensor(refer to value vector in sparse matrix
   * CSR format).
   * @param embedding_params embedding params for initialization.
   * @param gpu_resource_group the GPU resource group
   */
  LocalizedSlotSparseEmbeddingOneHot(
      const TensorPtrs<TypeHashKey> &train_row_offsets_tensors,
      const TensorPtrs<TypeHashKey> &train_value_tensors,
      const std::vector<std::shared_ptr<size_t>> &train_nnz_array,
      const TensorPtrs<TypeHashKey> &evaluate_row_offsets_tensors,
      const TensorPtrs<TypeHashKey> &evaluate_value_tensors,
      const std::vector<std::shared_ptr<size_t>> &evaluate_nnz_array,
      const SparseEmbeddingHashParams<TypeEmbeddingComp> &embedding_params,
      const std::string plan_file, const std::shared_ptr<GPUResourceGroup> &gpu_resource_group);

  /**
   * The forward propagation of embedding layer.
   */
  void forward() override {
    CudaDeviceContext context;

    for (size_t i = 0; i < Base::get_local_gpu_count(); i++) {
      context.set_device(Base::get_gpu_resource(i).get_device_id());  // set device
      // for forward_fuse method
      functors_.forward_mapping_per_gpu(
          Base::get_batch_size(), slot_num_per_gpu_[i], Base::get_value_tensors()[i]->get_ptr(),
          *Base::get_nnz_array()[i], mapping_offsets_per_gpu_tensors_[i]->get_ptr(),
          hash_value_index_tensors_[i]->get_ptr(), Base::get_gpu_resource(i).get_stream());

      // fuse forward+all2all+reorder into one kernel
      functors_.forward_fuse_per_gpu(
          i, Base::get_local_gpu_count(), Base::get_batch_size(), Base::get_batch_size_per_gpu(),
          Base::get_slot_num(), slot_num_per_gpu_[i], Base::get_embedding_vec_size(),
          Base::get_combiner(), Base::get_row_offsets_tensors()[i]->get_ptr(),
          hash_value_index_tensors_[i]->get_ptr(), hash_table_value_tensors_[i]->get_ptr(),
          get_embedding_features(), Base::get_gpu_resource_group().get_sm_count(),
          Base::get_gpu_resource(i).get_stream());
    }

    functors_.sync_all_gpus(Base::get_gpu_resource_group());

    return;
  }

  /**
   * The first stage of backward propagation of embedding layer,
   * which computes the wgrad by the dgrad from the top layer.
   */
  void backward() override {
    functors_.sync_all_gpus(Base::get_gpu_resource_group());

    CudaDeviceContext context;
    for (size_t i = 0; i < Base::get_local_gpu_count(); i++) {
      context.set_device(Base::get_gpu_resource(i).get_device_id());

      functors_.backward_fuse_per_gpu(
          i, Base::get_local_gpu_count(), Base::get_batch_size(), Base::get_batch_size_per_gpu(),
          Base::get_slot_num(), slot_num_per_gpu_[i], Base::get_embedding_vec_size(),
          Base::get_combiner(), get_embedding_features(), wgrad_tensors_[i]->get_ptr(),
          Base::get_gpu_resource_group().get_sm_count(), Base::get_gpu_resource(i).get_stream());
    }

    return;
  }

  /**
   * The second stage of backward propagation of embedding layer, which
   * updates the hash table by wgrad(from backward()) and optimizer.
   */
  void update_params() override {
    CudaDeviceContext context;
    for (size_t i = 0; i < Base::get_local_gpu_count(); i++) {
      context.set_device(Base::get_gpu_resource(i).get_device_id());

      // accumulate times for adam optimizer
      Base::get_opt_params(i).hyperparams.adam.times++;

      // do update params operation: only support SGD
      functors_.update_params(Base::get_embedding_vec_size(), Base::get_opt_params(i),
                              *Base::get_nnz_array()[i], hash_value_index_tensors_[i]->get_ptr(),
                              wgrad_tensors_[i]->get_ptr(), hash_table_value_tensors_[i]->get_ptr(),
                              Base::get_gpu_resource_group().get_sm_count(),
                              Base::get_gpu_resource(i).get_stream());
    }

    return;
  }

  /**
   * Initialize the embedding table
   */
  void init_params() override {
    // do hash table value initialization
    if (slot_size_array_.size() == Base::get_slot_num()) {
      init_embedding(slot_size_array_, Base::get_embedding_vec_size(), hash_table_value_tensors_,
                     hash_table_slot_id_tensors_, Base::get_gpu_resource_group());
    } else {
      throw std::runtime_error(
          std::string("[HCDEBUG][ERROR] Runtime error: the size of slot_sizes != slot_num\n"));
    }
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

    upload_params_to_device(weight_stream, Base::get_embedding_vec_size(),
                            hash_table_value_tensors_, slot_size_array_,
                            mapping_offsets_per_gpu_tensors_, Base::get_gpu_resource_group());

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

    download_params_to_host(weight_stream, Base::get_embedding_vec_size(),
                            hash_table_value_tensors_, slot_size_array_,
                            Base::get_gpu_resource_group());

    return;
  }

  /**
   * Get the total size of hash tables on all GPUs.
   */
  size_t get_params_num() const override {
    return (max_vocabulary_size_ * Base::get_embedding_vec_size());
  }

  // only used for results check
  /**
   * Get the forward() results from GPUs and copy them to the host pointer
   * embedding_feature. This function is only used for unit test.
   * @param embedding_feature the host pointer for storing the forward()
   * results.
   */
  void get_forward_results(TypeEmbeddingComp *embedding_feature) override {
    size_t memcpy_size =
        Base::get_batch_size_per_gpu() * Base::get_slot_num() * Base::get_embedding_vec_size();

    functors_.get_forward_results(memcpy_size, Base::get_output_tensors(), embedding_feature,
                                  utest_forward_temp_tensors_, Base::get_gpu_resource_group());

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
    CudaDeviceContext context(Base::get_gpu_resource(0).get_device_id());

#ifdef NCCL_A2A

#ifndef ENABLE_MPI
    if (Base::get_total_gpu_count() > 1) {
      functors_.all2all_forward(Base::get_batch_size_per_gpu(), slot_num_per_gpu_,
                                Base::get_embedding_vec_size(), wgrad_tensors_,
                                utest_all2all_tensors_, Base::get_gpu_resource_group());
    } else {
      CK_CUDA_THROW_(
          cudaMemcpyAsync(utest_all2all_tensors_[0]->get_ptr(), wgrad_tensors_[0]->get_ptr(),
                          Base::get_batch_size_per_gpu() * slot_num_per_gpu_[0] *
                              Base::get_embedding_vec_size() * sizeof(TypeEmbeddingComp),
                          cudaMemcpyDeviceToDevice, Base::get_gpu_resource(0).get_stream()));
    }
#else
    if (Base::get_total_gpu_count() > 1) {
      functors_.all2all_forward(Base::get_batch_size_per_gpu(), Base::get_slot_num(),
                                Base::get_embedding_vec_size(), wgrad_tensors_,
                                utest_all2all_tensors_, Base::get_gpu_resource_group());
    } else {
      CK_CUDA_THROW_(
          cudaMemcpyAsync(utest_all2all_tensors_[0]->get_ptr(), wgrad_tensors_[0]->get_ptr(),
                          Base::get_batch_size_per_gpu() * slot_num_per_gpu_[0] *
                              Base::get_embedding_vec_size() * sizeof(TypeEmbeddingComp),
                          cudaMemcpyDeviceToDevice, Base::get_gpu_resource(0).get_stream()));
    }
#endif

#else
    // do not support gossip
    MESSAGE_("Error: Not support gossip in backward for one-hot");
#endif

    // reorder
    functors_.forward_reorder(Base::get_batch_size_per_gpu(), Base::get_slot_num(),
                              Base::get_embedding_vec_size(), utest_all2all_tensors_,
                              utest_reorder_tensors_, Base::get_gpu_resource_group());

    // there are batch_size_per_gpu samples' wgard on each GPU
    size_t memcpy_size =
        Base::get_batch_size_per_gpu() * Base::get_slot_num() * Base::get_embedding_vec_size();

    // nccl gather
    functors_.all_gather(memcpy_size,
                         utest_reorder_tensors_,        // send
                         utest_backward_temp_tensors_,  // recv
                         Base::get_gpu_resource_group());

    // memcpy H2D
    functors_.get_backward_results(devIndex, Base::get_total_gpu_count() * memcpy_size,
                                   utest_backward_temp_tensors_, wgrad,
                                   Base::get_gpu_resource_group());

    return;
  }

  /**
   * Get the update_params() results(the hash table, including hash_table_keys
   * and hash_table_values) from GPUs and copy them to the host pointers.
   * This function is only used for unit test.
   * @param hash_table_key the host pointer for stroing the hash table keys.
   * @param hash_table_value the host pointer for stroing the hash table values.
   */
  void get_update_params_results(TypeHashKey *hash_table_key, float *hash_table_value) override {}

  void check_overflow() const override {}

};  // end of class LocalizedSlotSparseEmbeddingOneHot

}  // namespace HugeCTR
