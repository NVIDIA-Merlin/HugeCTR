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
#include <common.hpp>
#include <embedding.hpp>
#include <embeddings/sparse_embedding_hash_functors.hpp>
#include <cub/cub/device/device_radix_sort.cuh>

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
 private:
  // define tensors
  TensorPtrs<float> hash_table_value_tensors_; /**< Hash table value. */
  TensorPtrs<TypeHashValueIndex>
      hash_table_slot_id_tensors_; /**< the tensors for storing slot ids */
  TensorPtrs<TypeHashValueIndex>
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
  GeneralBufferPtrs<TypeHashValueIndex>
      value_index_bufs_; /**< TypeHashValueIndex type general buffer. */

  size_t max_vocabulary_size_;
  size_t max_vocabulary_size_per_gpu_;   /**< Max vocabulary size for each GPU. */
  std::vector<size_t> slot_num_per_gpu_; /* slot_num per GPU */
  std::vector<size_t> slot_size_array_;

  SparseEmbeddingHashFunctors functors_; /**< obj of SparseEmbeddingHashFunctors */

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
      const std::string plan_file, const std::shared_ptr<GPUResourceGroup> &gpu_resource_group)
      : Base(train_row_offsets_tensors, train_value_tensors, train_nnz_array,
             evaluate_row_offsets_tensors, evaluate_value_tensors, evaluate_nnz_array,
             embedding_params, gpu_resource_group),
        slot_size_array_(embedding_params.slot_size_array) {
    try {
      CudaDeviceContext context(Base::get_gpu_resource(0).get_device_id());

      max_vocabulary_size_ = 0;
      for (size_t slot_size : slot_size_array_) {
        max_vocabulary_size_ += slot_size;
      }

      max_vocabulary_size_per_gpu_ = functors_.cal_max_voc_size_per_gpu(
          Base::get_total_gpu_count(), Base::get_local_gpu_count(), slot_size_array_,
          Base::get_gpu_resource_group());

      MESSAGE_("max_vocabulary_size_per_gpu_=" + std::to_string(max_vocabulary_size_per_gpu_));

      for (size_t id = 0; id < Base::get_local_gpu_count(); id++) {
        int cur_device = Base::get_gpu_resource(id).get_device_id();
        context.set_device(cur_device);

        size_t gid = Base::get_gpu_resource_group().get_global_id(cur_device);
        size_t slot_num_per_gpu =
            Base::get_slot_num() / Base::get_total_gpu_count() +
            ((gid < Base::get_slot_num() % Base::get_total_gpu_count()) ? 1 : 0);
        slot_num_per_gpu_.push_back(slot_num_per_gpu);

        // new GeneralBuffer objects
        float_bufs_.emplace_back(new GeneralBuffer<float>());
        fp_bufs_.emplace_back(new GeneralBuffer<TypeEmbeddingComp>());
        uint32_bufs_.emplace_back(new GeneralBuffer<uint32_t>());
        key_bufs_.emplace_back(new GeneralBuffer<TypeHashKey>());
        value_index_bufs_.emplace_back(new GeneralBuffer<TypeHashValueIndex>());

        // new hash table value vectors
        hash_table_value_tensors_.emplace_back(
            new Tensor<float>({max_vocabulary_size_per_gpu_, Base::get_embedding_vec_size()},
                              float_bufs_.back(), TensorFormat_t::HW));

        // new hash table value_index that get() from HashTable
        hash_value_index_tensors_.emplace_back(new Tensor<TypeHashValueIndex>(
            {1, Base::get_universal_batch_size() * Base::get_max_feature_num()},
            value_index_bufs_.back(), TensorFormat_t::HW));

        // new embedding features reduced by hash table values(results of forward)
        embedding_feature_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
            {Base::get_universal_batch_size() * slot_num_per_gpu, Base::get_embedding_vec_size()},
            fp_bufs_.back(), TensorFormat_t::HW));

        // new wgrad used by backward
        wgrad_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
            {Base::get_train_only_batch_size() * slot_num_per_gpu, Base::get_embedding_vec_size()},
            fp_bufs_.back(), TensorFormat_t::HW));

        // new optimizer params used by update_params
        switch (Base::get_optimizer()) {
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
            {Base::get_universal_batch_size_per_gpu() * Base::get_slot_num(),
             Base::get_embedding_vec_size()},
            fp_bufs_.back(), TensorFormat_t::HW));

        utest_forward_temp_tensors_.emplace_back(
            new Tensor<TypeEmbeddingComp>({Base::get_universal_batch_size() * Base::get_slot_num(),
                                           Base::get_embedding_vec_size()},
                                          fp_bufs_.back(), TensorFormat_t::HW));
        utest_all2all_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
            {Base::get_batch_size_per_gpu() * Base::get_slot_num(), Base::get_embedding_vec_size()},
            fp_bufs_.back(), TensorFormat_t::HW));
        utest_reorder_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
            {Base::get_batch_size_per_gpu() * Base::get_slot_num(), Base::get_embedding_vec_size()},
            fp_bufs_.back(), TensorFormat_t::HW));
        utest_backward_temp_tensors_.emplace_back(
            new Tensor<TypeEmbeddingComp>({Base::get_train_only_batch_size() * Base::get_slot_num(),
                                           Base::get_embedding_vec_size()},
                                          fp_bufs_.back(), TensorFormat_t::HW));

        mapping_offsets_per_gpu_tensors_.emplace_back(new Tensor<uint32_t>(
            {1, (size_t)slot_num_per_gpu}, uint32_bufs_.back(), TensorFormat_t::HW));

        // init GenenralBuffers to do real allocation
#ifndef NDEBUG
        std::cout << " max_feature_num_:" << Base::get_max_feature_num() << std::endl;
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
      }  // end of for(int id = 0; id < Base::get_local_gpu_count(); id++)

      // sync
      functors_.sync_all_gpus(Base::get_gpu_resource_group());

      // get the mapping table between local value_index and input value_index
      for (size_t id = 0; id < Base::get_local_gpu_count(); id++) {
        uint32_t slot_sizes_prefix_sum = 0;
        uint32_t slot_sizes_prefix_sum_local = 0;
        int slot_num = 0;
        for (size_t i = 0; i < slot_size_array_.size(); i++) {
          size_t device_id = Base::get_gpu_resource(id).get_device_id();
          size_t global_id = Base::get_gpu_resource_group().get_global_id(device_id);
          size_t slot_size = slot_size_array_[i];
          if (i % Base::get_total_gpu_count() == global_id) {
            uint32_t mapping_offset = slot_sizes_prefix_sum - slot_sizes_prefix_sum_local;
            CK_CUDA_THROW_(
                cudaMemcpy(&((mapping_offsets_per_gpu_tensors_[id]->get_ptr())[slot_num]),
                           &mapping_offset, sizeof(uint32_t), cudaMemcpyHostToDevice));
            slot_sizes_prefix_sum_local += slot_size;
            slot_num++;
          }
          slot_sizes_prefix_sum += slot_size;
        }
      }

      // Check whether the P2P access can be enabled
      if (gpu_resource_group->get_local_gpu_count() > 1 && !gpu_resource_group->all_p2p_enabled()) {
        throw std::runtime_error(
            std::string("[HCDEBUG][ERROR] Runtime error: Localized_slot_sparse_embedding_one_hot "
                        "cannot be used on machine without GPU peer2peer access support. \n"));
      }
#ifdef ENABLE_MPI
      throw std::runtime_error(
          std::string("[HCDEBUG][ERROR] Runtime error: Localized_slot_sparse_embedding_one_hot "
                      "cannot support multi-node currently. \n"));
#endif

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
      CK_CUDA_THROW_(cudaMallocManaged(&train_embedding_features_,
                                       sizeof(TypeEmbeddingComp *) * Base::get_local_gpu_count()));

      for (size_t id = 0; id < Base::get_local_gpu_count(); id++) {
        train_embedding_features_[id] = Base::get_train_only_output_tensors()[id]->get_ptr();
      }

      CK_CUDA_THROW_(cudaMallocManaged(&evaluate_embedding_features_,
                                       sizeof(TypeEmbeddingComp *) * Base::get_local_gpu_count()));

      for (size_t id = 0; id < Base::get_local_gpu_count(); id++) {
        evaluate_embedding_features_[id] = Base::get_evaluate_only_output_tensors()[id]->get_ptr();
      }

    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }

    return;
  }

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
          get_embedding_features(), Base::get_gpu_resource(i).get_stream());
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
          Base::get_gpu_resource(i).get_stream());
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
      functors_.update_params(Base::get_gpu_resource(i).get_stream(),
                              Base::get_embedding_vec_size(), Base::get_opt_params(i),
                              *Base::get_nnz_array()[i], hash_value_index_tensors_[i]->get_ptr(),
                              wgrad_tensors_[i]->get_ptr(),
                              hash_table_value_tensors_[i]->get_ptr());
    }

    return;
  }

  /**
   * Initialize the embedding table
   */
  void init_params() override {
    // do hash table value initialization
    if (slot_size_array_.size() == Base::get_slot_num()) {
      functors_.init_embedding<TypeHashKey>(slot_size_array_, Base::get_embedding_vec_size(),
                                            hash_table_value_tensors_, hash_table_slot_id_tensors_,
                                            Base::get_gpu_resource_group());
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

    functors_.upload_params_to_device<TypeHashKey, TypeHashValueIndex>(
        weight_stream, Base::get_embedding_vec_size(), hash_table_value_tensors_, slot_size_array_,
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

    functors_.download_params_to_host<TypeHashKey, TypeHashValueIndex>(
        weight_stream, Base::get_embedding_vec_size(), hash_table_value_tensors_, slot_size_array_,
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
      functors_.all2all_forward(batch_size_per_gpu_, Base::get_slot_num(),
                                Base::get_embedding_vec_size(), wgrad_tensors_,
                                utest_all2all_tensors_, Base::get_gpu_resource_group());
    } else {
      CK_CUDA_THROW_(
          cudaMemcpyAsync(utest_all2all_tensors_[0]->get_ptr(), wgrad_tensors_[0]->get_ptr(),
                          (size_t)batch_size_per_gpu_ * slot_num_per_gpu_[0] *
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
