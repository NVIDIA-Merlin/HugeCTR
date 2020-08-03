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
#include <common.hpp>
#include <embedding.hpp>
#include <embeddings/sparse_embedding_hash_functors.hpp>
#include <cub/cub/device/device_radix_sort.cuh>
#include <hashtable/nv_hashtable.cuh>
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
  std::vector<std::shared_ptr<NvHashTable>> hash_tables_; /**< Hash table.  */

  // define tensors
  TensorPtrs<float> hash_table_value_tensors_; /**< Hash table value. */
  TensorPtrs<TypeHashValueIndex>
      hash_value_index_tensors_; /**< Hash table value index. The index is corresponding to the line
                                    number of the value. */
  TensorPtrs<TypeEmbeddingComp>
      embedding_feature_tensors_;               /**< the output tensor of the forward(). */
  TensorPtrs<TypeEmbeddingComp> wgrad_tensors_; /**< the input tensor of the backward(). */
  TensorPtrs<TypeEmbeddingComp>
      opt_m_tensors_; /**< The mi variable storage for adam optimizer in the update_params(). */
  TensorPtrs<TypeEmbeddingComp>
      opt_v_tensors_; /**< The vi variable storage for adam optimizer in the update_params(). */
  TensorPtrs<TypeEmbeddingComp> opt_momentum_tensors_; /**< The momentum variable storage for the
                                           momentum optimizer in the update_params(). */
  TensorPtrs<TypeEmbeddingComp> opt_accm_tensors_; /**< The accm variable storage for the nesterov
                                                     optimizer in the update_params(). */
  TensorPtrs<TypeHashKey>
      row_offset_allreduce_tensors_; /**< The temp memory to store the row_offset after all_reduce
                                        operation among multi-gpu in forward(). */
  TensorPtrs<TypeHashValueIndex>
      hash_value_index_sort_tensors_; /**< The temp memory to store the sorted hash table value
                                         indexes in update_params(). */

  TensorPtrs<uint32_t>
      hash_value_index_count_offset_tensors_; /**< The temp memory to store the offset of each count
                                                 of hash table value indexes in update_params(). */
  TensorPtrs<uint32_t> hash_value_index_count_counter_tensors_; /**< The temp memory to store the
                                                                counter of the count of hash table
                                                                value indexes in update_params(). */

  TensorPtrs<uint32_t> new_hash_value_flag_tensors_;
  TensorPtrs<uint32_t> hash_value_flag_sumed_tensors_;

  TensorPtrs<TypeHashKey> sample_id_tensors_; /**< The temp memory to store the sample ids of hash
                                              table value in      update_params(). */
  TensorPtrs<TypeHashKey> sample_id_sort_tensors_; /**< The temp memory to store the sorted sample
                                                   ids of hash table value in update_params(). */
  TensorPtrs<TypeHashKey> temp_storage_sort_tensors_; /**< The temp memory for the CUB lib sorting
                                                      API in update_params(). */

  TensorPtrs<uint32_t> temp_storage_scan_tensors_; /**< The temp memory for the CUB lib scaning API
                                                      in update_params(). */

  TensorPtrs<TypeHashValueIndex>
      deltaw_hash_value_index_tensors_; /**< The temp memory to store the hash table indexes of
                                           deltaw in update_params(). */
  TensorPtrs<float> deltaw_tensors_; /**< The temp memory to store the deltaw in update_params(). */

  // define GeneralBuffers
  GeneralBufferPtrs<float> float_bufs_; /**< float type general buffer. */
  GeneralBufferPtrs<TypeEmbeddingComp>
      fp_bufs_; /**< TypeEmbeddingComp(fp32 or fp16) type general buffer. */
  GeneralBufferPtrs<uint32_t> uint32_bufs_; /**< uint32 type general buffer. */
  GeneralBufferPtrs<TypeHashKey> key_bufs_; /**< TypeHashKey type general buffer. */
  GeneralBufferPtrs<TypeHashValueIndex>
      value_index_bufs_; /**< TypeHashValueIndex type general buffer. */

  std::vector<size_t> temp_storage_sort_bytes_; /**< The temp variable for CUB lib sorting API. */
  std::vector<size_t> temp_storage_scan_bytes_; /**< The temp variable for CUB lib scaning API. */

  size_t max_vocabulary_size_;         /**< Max vocabulary size for each GPU. */
  size_t max_vocabulary_size_per_gpu_; /**< Max vocabulary size for each GPU. */

  SparseEmbeddingHashFunctors functors_; /**< obj of SparseEmbeddingHashFunctors */

  TensorPtrs<TypeEmbeddingComp> utest_forward_temp_tensors_;

 public:
  /**
   * The constructor of DistributedSlotSparseEmbeddingHash.
   * @param row_offsets_tensors row offsets of the input tensor(refer to row offset vector in sparse
   * matrix CSR format).
   * @param hash_key_tensors hash keys of the input tensor(refer to value vector in sparse matrix
   * CSR format).
   * @param embedding_params embedding params for initialization.
   * @param gpu_resource_group the GPU resource group
   */
  DistributedSlotSparseEmbeddingHash(const TensorPtrs<TypeHashKey> &train_row_offsets_tensors,
                                     const TensorPtrs<TypeHashKey> &train_value_tensors,
                                     const std::vector<std::shared_ptr<size_t>> &train_nnz_array,
                                     const TensorPtrs<TypeHashKey> &evaluate_row_offsets_tensors,
                                     const TensorPtrs<TypeHashKey> &evaluate_value_tensors,
                                     const std::vector<std::shared_ptr<size_t>> &evaluate_nnz_array,
                                     SparseEmbeddingHashParams<TypeEmbeddingComp> embedding_params,
                                     const std::shared_ptr<GPUResourceGroup> &gpu_resource_group)
      : Base(train_row_offsets_tensors, train_value_tensors, train_nnz_array,
             evaluate_row_offsets_tensors, evaluate_value_tensors, evaluate_nnz_array,
             embedding_params, gpu_resource_group) {
    try {
      CudaDeviceContext context;

      // CAUSION: can not decide how many <key,value> pairs in each GPU, because the GPU
      // distribution is computed by (key%gpu_count). In order to not allocate the total size of
      // hash table on each GPU, meanwhile get a better performance by a unfull hash table, the
      // users need to set the param "load_factor"(load_factor<1).
      max_vocabulary_size_per_gpu_ = Base::get_max_vocabulary_size_per_gpu();
      max_vocabulary_size_ = max_vocabulary_size_per_gpu_ * Base::get_total_gpu_count();

      MESSAGE_("max_vocabulary_size_per_gpu_=" + std::to_string(max_vocabulary_size_per_gpu_));

      for (size_t id = 0; id < Base::get_local_gpu_count(); id++) {
        int cur_device = Base::get_gpu_resource(id).get_device_id();
        context.set_device(cur_device);

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
            new Tensor<float>({max_vocabulary_size_per_gpu_, Base::get_embedding_vec_size()},
                              float_bufs_.back(), TensorFormat_t::HW));

        // new hash table value_index that get() from HashTable
        hash_value_index_tensors_.emplace_back(new Tensor<TypeHashValueIndex>(
            {1, Base::get_universal_batch_size() * Base::get_max_feature_num()},
            value_index_bufs_.back(), TensorFormat_t::HW));

        // new embedding features reduced by hash table values(results of forward)
        embedding_feature_tensors_.emplace_back(
            new Tensor<TypeEmbeddingComp>({Base::get_universal_batch_size() * Base::get_slot_num(),
                                           Base::get_embedding_vec_size()},
                                          fp_bufs_.back(), TensorFormat_t::HW));

        // new wgrad used by backward
        wgrad_tensors_.emplace_back(
            new Tensor<TypeEmbeddingComp>({Base::get_train_only_batch_size() * Base::get_slot_num(),
                                           Base::get_embedding_vec_size()},
                                          fp_bufs_.back(), TensorFormat_t::HW));

        // new optimizer params used by update_params
        switch (Base::get_optimizer()) {
          case Optimizer_t::Adam:  // adam
            opt_m_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
                {max_vocabulary_size_per_gpu_, Base::get_embedding_vec_size()}, fp_bufs_.back(),
                TensorFormat_t::HW));
            opt_v_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
                {max_vocabulary_size_per_gpu_, Base::get_embedding_vec_size()}, fp_bufs_.back(),
                TensorFormat_t::HW));
            break;

          case Optimizer_t::MomentumSGD:  // momentum_sgd
            opt_momentum_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
                {max_vocabulary_size_per_gpu_, Base::get_embedding_vec_size()}, fp_bufs_.back(),
                TensorFormat_t::HW));
            break;

          case Optimizer_t::Nesterov:  // nesterov
            opt_accm_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
                {max_vocabulary_size_per_gpu_, Base::get_embedding_vec_size()}, fp_bufs_.back(),
                TensorFormat_t::HW));
            break;

          case Optimizer_t::SGD:
            break;

          default:
            throw std::runtime_error(
                std::string("[HCDEBUG][ERROR] Runtime error: Invalid optimizer type\n"));
        }

        // new temp tensors used by update_params
        row_offset_allreduce_tensors_.emplace_back(new Tensor<TypeHashKey>(
            {1, Base::get_universal_batch_size() * Base::get_slot_num() + 1}, key_bufs_.back(),
            TensorFormat_t::HW));
        sample_id_tensors_.emplace_back(new Tensor<TypeHashKey>(
            {1, Base::get_train_only_batch_size() * Base::get_max_feature_num()}, key_bufs_.back(),
            TensorFormat_t::HW));
        sample_id_sort_tensors_.emplace_back(new Tensor<TypeHashKey>(
            {1, Base::get_train_only_batch_size() * Base::get_max_feature_num()}, key_bufs_.back(),
            TensorFormat_t::HW));
        hash_value_index_sort_tensors_.emplace_back(new Tensor<TypeHashValueIndex>(
            {1, Base::get_train_only_batch_size() * Base::get_max_feature_num()},
            value_index_bufs_.back(), TensorFormat_t::HW));
        hash_value_index_count_offset_tensors_.emplace_back(new Tensor<uint32_t>(
            {1, Base::get_train_only_batch_size() * Base::get_max_feature_num() + 1},
            uint32_bufs_.back(), TensorFormat_t::HW));

        new_hash_value_flag_tensors_.emplace_back(new Tensor<uint32_t>(
            {1, Base::get_train_only_batch_size() * Base::get_max_feature_num()},
            uint32_bufs_.back(), TensorFormat_t::HW));

        hash_value_flag_sumed_tensors_.emplace_back(new Tensor<uint32_t>(
            {1, Base::get_train_only_batch_size() * Base::get_max_feature_num()},
            uint32_bufs_.back(), TensorFormat_t::HW));

        hash_value_index_count_counter_tensors_.emplace_back(
            new Tensor<uint32_t>({1, 1}, uint32_bufs_.back(), TensorFormat_t::HW));
        deltaw_hash_value_index_tensors_.emplace_back(new Tensor<TypeHashValueIndex>(
            {1, Base::get_train_only_batch_size() * Base::get_max_feature_num()},
            value_index_bufs_.back(), TensorFormat_t::HW));
        deltaw_tensors_.emplace_back(
            new Tensor<float>({Base::get_train_only_batch_size() * Base::get_max_feature_num(),
                               Base::get_embedding_vec_size()},
                              float_bufs_.back(), TensorFormat_t::HW));
        {
          // cal the temp storage bytes for CUB radix sort
          size_t temp = 0;
          cub::DeviceRadixSort::SortPairs(
              (void *)NULL, (size_t &)temp, (TypeHashKey *)NULL, (TypeHashKey *)NULL,
              (TypeHashKey *)NULL, (TypeHashKey *)NULL,
              Base::get_train_only_batch_size() * Base::get_max_feature_num());
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
              Base::get_train_only_batch_size() * Base::get_max_feature_num());
          temp_storage_scan_bytes_.push_back(temp);

          size_t size = (size_t)ceil((float)temp_storage_scan_bytes_[id] / sizeof(uint32_t));

          temp_storage_scan_tensors_.emplace_back(
              new Tensor<uint32_t>({1, size}, uint32_bufs_.back(), TensorFormat_t::HW));
        }

        utest_forward_temp_tensors_.emplace_back(
            new Tensor<TypeEmbeddingComp>({Base::get_universal_batch_size() * Base::get_slot_num(),
                                           Base::get_embedding_vec_size()},
                                          fp_bufs_.back(), TensorFormat_t::HW));

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

        const OptParams<TypeEmbeddingComp> &source_opt_param = Base::get_opt_params();
        OptParams<TypeEmbeddingComp> &target_opt_param = Base::get_opt_params(id);

        switch (Base::get_optimizer()) {
          case Optimizer_t::Adam:  // adam
            CK_CUDA_THROW_(
                cudaMemsetAsync(opt_m_tensors_[id]->get_ptr(), 0,
                                max_vocabulary_size_per_gpu_ * Base::get_embedding_vec_size() *
                                    sizeof(TypeEmbeddingComp),
                                Base::get_gpu_resource(id).get_stream()));
            CK_CUDA_THROW_(
                cudaMemsetAsync(opt_v_tensors_[id]->get_ptr(), 0,
                                max_vocabulary_size_per_gpu_ * Base::get_embedding_vec_size() *
                                    sizeof(TypeEmbeddingComp),
                                Base::get_gpu_resource(id).get_stream()));
            target_opt_param.hyperparams.adam.times = 0;
            target_opt_param.hyperparams.adam.beta1 = source_opt_param.hyperparams.adam.beta1;
            target_opt_param.hyperparams.adam.beta2 = source_opt_param.hyperparams.adam.beta2;
            target_opt_param.hyperparams.adam.epsilon = source_opt_param.hyperparams.adam.epsilon;
            target_opt_param.hyperparams.adam.m_ptr = opt_m_tensors_[id]->get_ptr();
            target_opt_param.hyperparams.adam.v_ptr = opt_v_tensors_[id]->get_ptr();

            break;

          case Optimizer_t::MomentumSGD:  // momentum_sgd
            CK_CUDA_THROW_(
                cudaMemsetAsync(opt_momentum_tensors_[id]->get_ptr(), 0,
                                max_vocabulary_size_per_gpu_ * Base::get_embedding_vec_size() *
                                    sizeof(TypeEmbeddingComp),
                                Base::get_gpu_resource(id).get_stream()));
            target_opt_param.hyperparams.momentum.factor =
                source_opt_param.hyperparams.momentum.factor;
            target_opt_param.hyperparams.momentum.momentum_ptr =
                opt_momentum_tensors_[id]->get_ptr();
            break;

          case Optimizer_t::Nesterov:  // nesterov
            CK_CUDA_THROW_(
                cudaMemsetAsync(opt_accm_tensors_[id]->get_ptr(), 0,
                                max_vocabulary_size_per_gpu_ * Base::get_embedding_vec_size() *
                                    sizeof(TypeEmbeddingComp),
                                Base::get_gpu_resource(id).get_stream()));
            target_opt_param.hyperparams.nesterov.mu = source_opt_param.hyperparams.nesterov.mu;
            target_opt_param.hyperparams.nesterov.accm_ptr = opt_accm_tensors_[id]->get_ptr();
            break;

          case Optimizer_t::SGD:
            break;

          default:
            throw std::runtime_error(
                std::string("[HCDEBUG][ERROR] Runtime error: Invalid optimizer type\n"));
        }

      }  // end of for(int id = 0; id < local_gpu_count_; id++)

      functors_.sync_all_gpus(Base::get_gpu_resource_group());

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
    // Read data from input_buffers_ -> look up -> write to output_tensors

    CudaDeviceContext context;

    for (size_t i = 0; i < Base::get_local_gpu_count(); i++) {
      context.set_device(Base::get_gpu_resource(i).get_device_id());
      functors_.forward_per_gpu(
          Base::get_batch_size(), Base::get_slot_num(), Base::get_embedding_vec_size(),
          Base::in_train(), Base::get_row_offsets_tensors()[i]->get_ptr(),
          Base::get_value_tensors()[i]->get_ptr(), *Base::get_nnz_array()[i], *hash_tables_[i],
          hash_table_value_tensors_[i]->get_ptr(), hash_value_index_tensors_[i]->get_ptr(),
          embedding_feature_tensors_[i]->get_ptr(), Base::get_gpu_resource(i).get_stream());
    }

    // do reduce scatter
    size_t recv_count =
        Base::get_batch_size_per_gpu() * Base::get_slot_num() * Base::get_embedding_vec_size();
    functors_.reduce_scatter(recv_count, embedding_feature_tensors_, Base::get_output_tensors(),
                             Base::get_gpu_resource_group());

    // scale for combiner=mean after reduction
    if (Base::get_combiner() == 1) {
      size_t send_count = Base::get_batch_size() * Base::get_slot_num() + 1;
      functors_.all_reduce(send_count, Base::get_row_offsets_tensors(),
                           row_offset_allreduce_tensors_, Base::get_gpu_resource_group());

      // do average
      functors_.forward_scale(Base::get_batch_size(), Base::get_slot_num(),
                              Base::get_embedding_vec_size(), row_offset_allreduce_tensors_,
                              Base::get_output_tensors(), Base::get_gpu_resource_group());
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
        Base::get_batch_size_per_gpu() * Base::get_slot_num() * Base::get_embedding_vec_size();
    functors_.all_gather(send_count, Base::get_output_tensors(), embedding_feature_tensors_,
                         Base::get_gpu_resource_group());

    // do backward
    functors_.backward(Base::get_batch_size(), Base::get_slot_num(), Base::get_embedding_vec_size(),
                       Base::get_combiner(), row_offset_allreduce_tensors_,
                       embedding_feature_tensors_, wgrad_tensors_, Base::get_gpu_resource_group());

    return;
  }

  /**
   * The second stage of backward propagation of embedding layer, which
   * updates the hash table by wgrad(from backward()) and optimizer.
   */
  void update_params() override {
    CudaDeviceContext context;
    for (size_t i = 0; i < Base::get_local_gpu_count(); i++) {
      CudaDeviceContext context(Base::get_gpu_resource(i).get_device_id());

      // accumulate times for adam optimizer
      Base::get_opt_params(i).hyperparams.adam.times++;

      // do update params operation
      functors_.update_params(
          Base::get_gpu_resource(i).get_stream(), Base::get_batch_size(), Base::get_slot_num(),
          Base::get_embedding_vec_size(), max_vocabulary_size_per_gpu_, Base::get_opt_params(i),
          *Base::get_nnz_array()[i], Base::get_row_offsets_tensors()[i]->get_ptr(),
          hash_value_index_tensors_[i]->get_ptr(), sample_id_tensors_[i]->get_ptr(),
          sample_id_sort_tensors_[i]->get_ptr(), hash_value_index_sort_tensors_[i]->get_ptr(),
          hash_value_index_count_offset_tensors_[i]->get_ptr(),
          new_hash_value_flag_tensors_[i]->get_ptr(), hash_value_flag_sumed_tensors_[i]->get_ptr(),
          hash_value_index_count_counter_tensors_[i]->get_ptr(),
          temp_storage_sort_tensors_[i]->get_ptr(), temp_storage_sort_bytes_[i],
          temp_storage_scan_tensors_[i]->get_ptr(), temp_storage_scan_bytes_[i],
          wgrad_tensors_[i]->get_ptr(), deltaw_hash_value_index_tensors_[i]->get_ptr(),
          deltaw_tensors_[i]->get_ptr(), hash_table_value_tensors_[i]->get_ptr());
    }

    return;
  }

  /**
   * Initialize the embedding table
   */
  void init_params() override {
    functors_.init_embedding(max_vocabulary_size_per_gpu_, Base::get_embedding_vec_size(),
                             hash_table_value_tensors_, Base::get_gpu_resource_group());
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

    functors_.upload_params_to_device(weight_stream, max_vocabulary_size_,
                                      Base::get_embedding_vec_size(), max_vocabulary_size_per_gpu_,
                                      hash_table_value_tensors_, hash_tables_,
                                      Base::get_gpu_resource_group());

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

    functors_.download_params_to_host(weight_stream, max_vocabulary_size_,
                                      Base::get_embedding_vec_size(), hash_table_value_tensors_,
                                      hash_tables_, Base::get_gpu_resource_group());

    return;
  }

  /**
   * Get the total size of hash tables on all GPUs.
   */
  size_t get_params_num() const override {
    // Read data from input_buffers_ -> look up -> write to output_tensors

    size_t total_size = 0;

    CudaDeviceContext context(Base::get_gpu_resource(0).get_device_id());

    // need to collect the <key, value> pair count from all GPUs and do sum reduction
    for (size_t id = 0; id < Base::get_local_gpu_count(); id++) {
      context.set_device(Base::get_gpu_resource(id).get_device_id());
      total_size += hash_tables_[id]->get_size(Base::get_gpu_resource(id).get_stream());
      CK_CUDA_THROW_(cudaStreamSynchronize(Base::get_gpu_resource(id).get_stream()));
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
    // wgard shuld be the same on multi-gpus after backward()
    size_t memcpy_size =
        Base::get_batch_size() * Base::get_slot_num() * Base::get_embedding_vec_size();

    functors_.get_backward_results(devIndex, memcpy_size, wgrad_tensors_, wgrad,
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
  void get_update_params_results(TypeHashKey *hash_table_key, float *hash_table_value) override {
    functors_.get_update_params_results(Base::get_embedding_vec_size(), max_vocabulary_size_,
                                        hash_table_value_tensors_, hash_tables_, hash_table_key,
                                        hash_table_value, Base::get_gpu_resource_group());

    return;
  }

  /**
   * Check overflow
   * @param lr
   */
  void check_overflow() const override {
    CudaDeviceContext context(Base::get_gpu_resource(0).get_device_id());

    for (size_t id = 0; id < Base::get_local_gpu_count(); id++) {
      context.set_device(Base::get_gpu_resource(id).get_device_id());
      size_t count = hash_tables_[id]->get_size(Base::get_gpu_resource(id).get_stream());
      if (count > max_vocabulary_size_per_gpu_) {
        CK_THROW_(Error_t::OutOfBound,
                  "Runtime vocabulary size (" + std::to_string(count) +
                      ") exceeds max_vocabulary_size_per_gpu (" +
                      std::to_string(max_vocabulary_size_per_gpu_) + ") on GPU " +
                      std::to_string(Base::get_gpu_resource(id).get_device_id()) +
                      ", new feature insertion failed.\n");
      }
    }
  }

};  // end of class DistributedSlotSparseEmbeddingHash

}  // namespace HugeCTR
