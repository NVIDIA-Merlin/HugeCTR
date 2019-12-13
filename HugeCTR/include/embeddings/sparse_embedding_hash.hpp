/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include "HugeCTR/include/embeddings/sparse_embedding_hash.cuh"
#include "cub/cub/device/device_radix_sort.cuh"

#include "HugeCTR/include/hashtable/nv_hashtable.cuh"

#include <vector>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {
/**
 * The SparseEmbeddingHash class inherits from Embedding class, which is the base class for
 * implementing all embedding layers. In this class, the embedding table is encapsulated in
 * a hash table. The key in the hash table is called as hash_table_key, and the value in
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
template <typename TypeHashKey>
class SparseEmbeddingHash : public Embedding<TypeHashKey> {
  using Base = Embedding<TypeHashKey>;

  using TypeHashValueIndex = TypeHashKey;  // use the hash key type as the hash value_index type(it
                                           // will be uint32 or int64)

 private:
  SparseEmbeddingHashParams embedding_params_; /**< Sparse embedding hash params. */

  std::vector<OptParams *> opt_params_; /**< Optimizer params. */
  std::vector<
      nv::HashTable<TypeHashKey, TypeHashValueIndex, std::numeric_limits<TypeHashKey>::max()> *>
      hash_tables_; /**< Hash table.  */

  // define tensors
  std::vector<Tensor<float> *> hash_table_value_tensors_; /**< Hash table value. */
  std::vector<Tensor<TypeHashValueIndex> *>
      hash_value_index_tensors_; /**< Hash table value index. The index is corresponding to the line
                                    number of the value. */
  std::vector<Tensor<float> *>
      embedding_feature_tensors_; /**< Embedding feature: the output tensor of the forward(). */
  std::vector<Tensor<float> *> wgrad_tensors_; /**< wgrad: the input tensor of the backward(). */
  std::vector<Tensor<float> *>
      opt_m_tensors_; /**< The mi variable storage for adam optimizer in the update_params(). */
  std::vector<Tensor<float> *>
      opt_v_tensors_; /**< The vi variable storage for adam optimizer in the update_params(). */
  std::vector<Tensor<float> *>
      opt_momentum_tensors_; /**< The momentum variable storage for the momentum optimizer in the
                                update_params(). */
  std::vector<Tensor<float> *> opt_accm_tensors_; /**< The accm variable storage for the nesterov
                                                     optimizer in the update_params(). */
  std::vector<Tensor<TypeHashKey> *>
      row_offset_allreduce_tensors_; /**< The temp memory to store the row_offset after all_reduce
                                        operation among multi-gpu in forward(). */
  std::vector<Tensor<TypeHashValueIndex> *>
      hash_value_index_sort_tensors_; /**< The temp memory to store the sorted hash table value
                                         indexes in update_params(). */
  std::vector<Tensor<uint32_t> *>
      hash_value_index_count_tensors_; /**< The temp memory to store the count of hash table value
                                          indexes in update_params(). */
  std::vector<Tensor<uint32_t> *>
      hash_value_index_count_offset_tensors_; /**< The temp memory to store the offset of each count
                                                 of hash table value indexes in update_params(). */
  std::vector<Tensor<uint32_t> *>
      hash_value_index_count_counter_tensors_; /**< The temp memory to store the counter of the
                                                  count of hash table value indexes in
                                                  update_params(). */
  std::vector<Tensor<TypeHashKey> *>
      sample_id_tensors_; /**< The temp memory to store the sample ids of hash table value in
                             update_params(). */
  std::vector<Tensor<TypeHashKey> *>
      sample_id_sort_tensors_; /**< The temp memory to store the sorted sample ids of hash table
                                  value in update_params(). */
  std::vector<Tensor<TypeHashKey> *>
      temp_storage_sort_tensors_; /**< The temp memory for the CUB lib sorting API in
                                     update_params(). */
  std::vector<Tensor<TypeHashValueIndex> *>
      deltaw_hash_value_index_tensors_; /**< The temp memory to store the hash table indexes of
                                           deltaw in update_params(). */
  std::vector<Tensor<float> *>
      deltaw_tensors_; /**< The temp memory to store the deltaw in update_params(). */

  // define GeneralBuffers
  std::vector<GeneralBuffer<float> *> float_bufs_;     /**< float type general buffer. */
  std::vector<GeneralBuffer<uint32_t> *> uint32_bufs_; /**< uint32 type general buffer. */
  std::vector<GeneralBuffer<TypeHashKey> *> key_bufs_; /**< TypeHashKey type general buffer. */
  std::vector<GeneralBuffer<TypeHashValueIndex> *>
      value_index_bufs_; /**< TypeHashValueIndex type general buffer. */

  std::vector<size_t> temp_storage_sort_bytes_;  // for CUB radix sort /**< The temp variable for
                                                 // CUB lib sorting API. */
  int max_vocabulary_size_per_gpu;               /**< Max vocabulary size for each GPU. */

 public:
  /**
   * The constructor of SparseEmbeddingHash.
   * @param row_offsets_tensors row offsets of the input tensor(refer to row offset vector in sparse
   * matrix CSR format).
   * @param hash_key_tensors hash keys of the input tensor(refer to value vector in sparse matrix
   * CSR format).
   * @param embedding_params embedding params for initialization.
   * @param gpu_resource_group the GPU resource group
   */
  SparseEmbeddingHash(const std::vector<Tensor<TypeHashKey> *> &row_offsets_tensors,
                      const std::vector<Tensor<TypeHashKey> *> &hash_key_tensors,
                      SparseEmbeddingHashParams embedding_params,
                      GPUResourceGroup &gpu_resource_group);
  /**
   * The destructor of SparseEmbeddingHash.
   */
  ~SparseEmbeddingHash();
  /**
   * The forward propagation of embedding layer.
   */
  void forward() override;
  /**
   * The first stage of backward propagation of embedding layer,
   * which only computes the wgrad by the dgrad from the top layer.
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
  long long get_params_num() override;

  // only used for results check
  /**
   * Get the forward() results from GPUs and copy them to the host pointer
   * embedding_feature. This function is only used for unit test.
   * @param embedding_feature the host pointer for storing the forward()
   * results.
   */
  float *get_embedding_feature_ptr(float *embedding_feature) override;
  /**
   * Get the backward() results from GPUs and copy them to the host pointer
   * wgrad. The wgrad on each GPU should be the same. This function is only
   * used for unit test.
   * @param wgrad the host pointer for stroing the backward() results.
   * @param devIndex the GPU device id.
   */
  float *get_wgrad_ptr(float *wgrad, int devIndex) override;
  /**
   * Get the update_params() results(the hash table, including hash_table_keys
   * and hash_table_values) from GPUs and copy them to the host pointers.
   * This function is only used for unit test.
   * @param hash_table_key the host pointer for stroing the hash table keys.
   * @param hash_table_value the host pointer for stroing the hash table values.
   */
  void get_hash_table_ptr(TypeHashKey *hash_table_key, float *hash_table_value) override;

};  // end of class SparseEmbeddingHash

template <typename TypeHashKey>
SparseEmbeddingHash<TypeHashKey>::SparseEmbeddingHash(
    const std::vector<Tensor<TypeHashKey> *> &row_offsets_tensors,
    const std::vector<Tensor<TypeHashKey> *> &hash_key_tensors,
    SparseEmbeddingHashParams embedding_params, GPUResourceGroup &gpu_resource_group)
    : embedding_params_(embedding_params),
      Base(row_offsets_tensors, hash_key_tensors, embedding_params.batch_size,
           embedding_params.slot_num, embedding_params.embedding_vec_size, gpu_resource_group) {
  try {
    int gpu_count = Base::device_resources_.size();
    CudaDeviceContext context(Base::device_resources_[0]->get_device_id());

    // CAUSION: can not decide how many <key,value> pairs in each GPU, because the GPU distribution
    // is computed by (key%gpu_count) In order to not allocate the total size of hash table on each
    // GPU, meanwhile get a better performance by a unfull hash table, the users need to set the
    // param "load_factor"(load_factor<1). The size of
    // max_vocabulary_size_per_gpu=vocabulary_size/gpu_count/load_factor, which should be more than
    // vocabulary_size/gpu_count.
    // TODO: Is "int" enough here?
    // int embedding_rows_per_gpu = (int)ceil((double)embedding_params_.vocabulary_size /
    // (double)gpu_count); int embedding_rows_per_gpu = (int)embedding_params_.vocabulary_size;
    max_vocabulary_size_per_gpu =
        (int)((float)embedding_params_.vocabulary_size /
              Base::device_resources_.get_total_gpu_count() / embedding_params_.load_factor);

#ifndef NDEBUG
    std::cout << "max_vocabulary_size_per_gpu:" << max_vocabulary_size_per_gpu;
#endif

    // for hash_table_value initialization
    HugeCTR::UnifiedDataSimulator<float> fdata_sim(-1.f / embedding_params_.embedding_vec_size,
                                                   1.f / embedding_params_.embedding_vec_size);
    float *h_hash_table_value;
    CK_CUDA_THROW_(cudaMallocHost(
        &h_hash_table_value,
        max_vocabulary_size_per_gpu * embedding_params_.embedding_vec_size * sizeof(float)));
    for (long long i = 0;
         i < ((long long)max_vocabulary_size_per_gpu * embedding_params_.embedding_vec_size); i++) {
      h_hash_table_value[i] = fdata_sim.get_num();
    }

    for (int id = 0; id < gpu_count; id++) {
      int cur_device = Base::device_resources_[id]->get_device_id();
      context.set_device(cur_device);

      // construct HashTable object: used to store hash table <key, value_index>
      hash_tables_.push_back(
          new nv::HashTable<TypeHashKey, TypeHashValueIndex,
                            std::numeric_limits<TypeHashKey>::max()>(max_vocabulary_size_per_gpu));

      // new GeneralBuffer objects
      float_bufs_.push_back(new GeneralBuffer<float>());
      uint32_bufs_.push_back(new GeneralBuffer<uint32_t>());
      key_bufs_.push_back(new GeneralBuffer<TypeHashKey>());
      value_index_bufs_.push_back(new GeneralBuffer<TypeHashValueIndex>());

      // new hash table value vectors
      hash_table_value_tensors_.push_back(
          new Tensor<float>({max_vocabulary_size_per_gpu, embedding_params_.embedding_vec_size},
                            (*float_bufs_.back()), TensorFormat_t::HW));

      // new hash table value_index that get() from HashTable
      hash_value_index_tensors_.push_back(new Tensor<TypeHashValueIndex>(
          {1, embedding_params_.batch_size * embedding_params_.max_feature_num},
          (*value_index_bufs_.back()), TensorFormat_t::HW));

      // new embedding features reduced by hash table values(results of forward)
      embedding_feature_tensors_.push_back(
          new Tensor<float>({embedding_params_.batch_size * embedding_params_.slot_num,
                             embedding_params_.embedding_vec_size},
                            (*float_bufs_.back()), TensorFormat_t::HW));

      // new wgrad used by backward
      wgrad_tensors_.push_back(
          new Tensor<float>({embedding_params_.batch_size * embedding_params_.slot_num,
                             embedding_params_.embedding_vec_size},
                            *(float_bufs_.back()), TensorFormat_t::HW));

      // new optimizer params used by update_params
      opt_params_.push_back(new OptParams);
      opt_params_[id]->optimizer = embedding_params_.opt_params.optimizer;
      opt_params_[id]->lr = embedding_params_.opt_params.lr;
      switch (embedding_params_.opt_params.optimizer) {
        case 0:  // adam
          opt_m_tensors_.push_back(
              new Tensor<float>({max_vocabulary_size_per_gpu, embedding_params_.embedding_vec_size},
                                *(float_bufs_.back()), TensorFormat_t::HW));
          opt_v_tensors_.push_back(
              new Tensor<float>({max_vocabulary_size_per_gpu, embedding_params_.embedding_vec_size},
                                *(float_bufs_.back()), TensorFormat_t::HW));
          break;

        case 1:  // momentum_sgd
          opt_momentum_tensors_.push_back(
              new Tensor<float>({max_vocabulary_size_per_gpu, embedding_params_.embedding_vec_size},
                                *(float_bufs_.back()), TensorFormat_t::HW));
          break;

        case 2:  // nesterov
          opt_accm_tensors_.push_back(
              new Tensor<float>({max_vocabulary_size_per_gpu, embedding_params_.embedding_vec_size},
                                *(float_bufs_.back()), TensorFormat_t::HW));
          break;

        default:
          throw std::runtime_error(
              std::string("[HCDEBUG][ERROR] Runtime error: Invalid optimizer type: ") +
              std::to_string(embedding_params_.opt_params.optimizer) + "\n");
      }

      // new temp tensors used by update_params
      row_offset_allreduce_tensors_.push_back(new Tensor<TypeHashKey>(
          {1, embedding_params_.batch_size * embedding_params_.slot_num + 1}, *(key_bufs_.back()),
          TensorFormat_t::HW));
      sample_id_tensors_.push_back(new Tensor<TypeHashKey>(
          {1, embedding_params_.batch_size * embedding_params_.max_feature_num},
          *(key_bufs_.back()), TensorFormat_t::HW));
      sample_id_sort_tensors_.push_back(new Tensor<TypeHashKey>(
          {1, embedding_params_.batch_size * embedding_params_.max_feature_num},
          *(key_bufs_.back()), TensorFormat_t::HW));
      hash_value_index_sort_tensors_.push_back(new Tensor<TypeHashValueIndex>(
          {1, embedding_params_.batch_size * embedding_params_.max_feature_num},
          *(value_index_bufs_.back()), TensorFormat_t::HW));
      hash_value_index_count_tensors_.push_back(new Tensor<uint32_t>(
          {1, embedding_params_.batch_size * embedding_params_.max_feature_num},
          *(uint32_bufs_.back()), TensorFormat_t::HW));
      hash_value_index_count_offset_tensors_.push_back(new Tensor<uint32_t>(
          {1, embedding_params_.batch_size * embedding_params_.max_feature_num},
          *(uint32_bufs_.back()), TensorFormat_t::HW));
      hash_value_index_count_counter_tensors_.push_back(
          new Tensor<uint32_t>({1, 1}, *(uint32_bufs_.back()), TensorFormat_t::HW));
      deltaw_hash_value_index_tensors_.push_back(new Tensor<TypeHashValueIndex>(
          {1, embedding_params_.batch_size * embedding_params_.max_feature_num},
          *(value_index_bufs_.back()), TensorFormat_t::HW));
      deltaw_tensors_.push_back(
          new Tensor<float>({embedding_params_.batch_size * embedding_params_.max_feature_num,
                             embedding_params_.embedding_vec_size},
                            *(float_bufs_.back()), TensorFormat_t::HW));

      // cal the temp storage bytes for CUB radix sort
      size_t temp = 0;
      cub::DeviceRadixSort::SortPairs(
          (void *)NULL, (size_t &)temp, (TypeHashKey *)NULL, (TypeHashKey *)NULL,
          (TypeHashKey *)NULL, (TypeHashKey *)NULL,
          embedding_params_.batch_size * embedding_params_.max_feature_num);
      temp_storage_sort_bytes_.push_back(temp);
      int size = (int)ceil((float)temp_storage_sort_bytes_[id] / sizeof(TypeHashKey));

      // new temp storage tensors for CUB radix sort
      temp_storage_sort_tensors_.push_back(
          new Tensor<TypeHashKey>({1, size}, *(key_bufs_.back()), TensorFormat_t::HW));

      // init GenenralBuffers to do real allocation
#ifndef NDEBUG
      std::cout << " max_feature_num_:" << embedding_params_.max_feature_num;
      std::cout << " float_bufs_:" << float_bufs_.back()->get_size();
      std::cout << " uint32_bufs_:" << uint32_bufs_.back()->get_size();
      std::cout << " key_bufs_:" << key_bufs_.back()->get_size();
      std::cout << " value_index_bufs_:" << value_index_bufs_.back()->get_size() << std::endl;
#endif
      float_bufs_.back()->init(cur_device);
      uint32_bufs_.back()->init(cur_device);
      key_bufs_.back()->init(cur_device);
      value_index_bufs_.back()->init(cur_device);

      // do initialization
      CK_CUDA_THROW_(cudaMemcpy(
          hash_table_value_tensors_[id]->get_ptr(), h_hash_table_value,
          max_vocabulary_size_per_gpu * embedding_params_.embedding_vec_size * sizeof(float),
          cudaMemcpyHostToDevice));

      switch (embedding_params_.opt_params.optimizer) {
        case 0:  // adam
          CK_CUDA_THROW_(cudaMemsetAsync(
              opt_m_tensors_[id]->get_ptr(), 0,
              max_vocabulary_size_per_gpu * embedding_params_.embedding_vec_size * sizeof(float),
              *Base::device_resources_[id]->get_stream_ptr()));
          CK_CUDA_THROW_(cudaMemsetAsync(
              opt_v_tensors_[id]->get_ptr(), 0,
              max_vocabulary_size_per_gpu * embedding_params_.embedding_vec_size * sizeof(float),
              *Base::device_resources_[id]->get_stream_ptr()));
          opt_params_[id]->hyperparams.adam.times = 0;
          opt_params_[id]->hyperparams.adam.alpha_t =
              embedding_params_.opt_params.hyperparams.adam.alpha_t;
          opt_params_[id]->hyperparams.adam.beta1 =
              embedding_params_.opt_params.hyperparams.adam.beta1;
          opt_params_[id]->hyperparams.adam.beta2 =
              embedding_params_.opt_params.hyperparams.adam.beta2;
          opt_params_[id]->hyperparams.adam.epsilon =
              embedding_params_.opt_params.hyperparams.adam.epsilon;
          opt_params_[id]->hyperparams.adam.m_ptr = opt_m_tensors_[id]->get_ptr();
          opt_params_[id]->hyperparams.adam.v_ptr = opt_v_tensors_[id]->get_ptr();
          break;

        case 1:  // momentum_sgd
          CK_CUDA_THROW_(cudaMemsetAsync(
              opt_momentum_tensors_[id]->get_ptr(), 0,
              max_vocabulary_size_per_gpu * embedding_params_.embedding_vec_size * sizeof(float),
              *Base::device_resources_[id]->get_stream_ptr()));
          opt_params_[id]->hyperparams.momentum.factor =
              embedding_params_.opt_params.hyperparams.momentum.factor;
          opt_params_[id]->hyperparams.momentum.momentum_ptr = opt_momentum_tensors_[id]->get_ptr();
          break;

        case 2:  // nesterov
          CK_CUDA_THROW_(cudaMemsetAsync(
              opt_accm_tensors_[id]->get_ptr(), 0,
              max_vocabulary_size_per_gpu * embedding_params_.embedding_vec_size * sizeof(float),
              *Base::device_resources_[id]->get_stream_ptr()));
          opt_params_[id]->hyperparams.nesterov.mu =
              embedding_params_.opt_params.hyperparams.nesterov.mu;
          opt_params_[id]->hyperparams.nesterov.accm_ptr = opt_accm_tensors_[id]->get_ptr();
          break;

        default:
          throw std::runtime_error(
              std::string("[HCDEBUG][ERROR] Runtime error: Invalid optimizer type: ") +
              std::to_string(embedding_params_.opt_params.optimizer) + "\n");
      }

    }  // end of for(int id = 0; id < gpu_count; id++)

    for (int id = 0; id < gpu_count; id++) {
      context.set_device(Base::device_resources_[id]->get_device_id());
      CK_CUDA_THROW_(cudaStreamSynchronize(*Base::device_resources_[id]->get_stream_ptr()));
    }

    CK_CUDA_THROW_(cudaFreeHost(h_hash_table_value));
  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}  // end of SparseEmbeddingHash()

template <typename TypeHashKey>
SparseEmbeddingHash<TypeHashKey>::~SparseEmbeddingHash() {
  try {
    for (auto hash_table : hash_tables_) {
      delete hash_table;
    }
    for (auto hash_table_value_tensor : hash_table_value_tensors_) {
      delete hash_table_value_tensor;
    }
    for (auto hash_value_index_tensor : hash_value_index_tensors_) {
      delete hash_value_index_tensor;
    }
    for (auto embedding_feature_tensor : embedding_feature_tensors_) {
      delete embedding_feature_tensor;
    }
    for (auto wgrad_tensor : wgrad_tensors_) {
      delete wgrad_tensor;
    }

    switch (embedding_params_.opt_params.optimizer) {
      case 0:
        for (auto opt_m_tensor : opt_m_tensors_) {
          delete opt_m_tensor;
        }
        for (auto opt_v_tensor : opt_v_tensors_) {
          delete opt_v_tensor;
        }
        break;
      case 1:
        for (auto opt_momentum_tensor : opt_momentum_tensors_) {
          delete opt_momentum_tensor;
        }
        break;
      case 2:
        for (auto opt_accm_tensor : opt_accm_tensors_) {
          delete opt_accm_tensor;
        }
        break;
    }

    for (auto row_offset_allreduce_tensor : row_offset_allreduce_tensors_) {
      delete row_offset_allreduce_tensor;
    }
    for (auto sample_id_tensor : sample_id_tensors_) {
      delete sample_id_tensor;
    }
    for (auto sample_id_sort_tensor : sample_id_sort_tensors_) {
      delete sample_id_sort_tensor;
    }
    for (auto hash_value_index_sort_tensor : hash_value_index_sort_tensors_) {
      delete hash_value_index_sort_tensor;
    }
    for (auto hash_value_index_count_tensor : hash_value_index_count_tensors_) {
      delete hash_value_index_count_tensor;
    }
    for (auto hash_value_index_count_offset_tensor : hash_value_index_count_offset_tensors_) {
      delete hash_value_index_count_offset_tensor;
    }
    for (auto hash_value_index_count_counter_tensor : hash_value_index_count_counter_tensors_) {
      delete hash_value_index_count_counter_tensor;
    }
    for (auto temp_storage_sort_tensor : temp_storage_sort_tensors_) {
      delete temp_storage_sort_tensor;
    }
    for (auto deltaw_hash_value_index_tensor : deltaw_hash_value_index_tensors_) {
      delete deltaw_hash_value_index_tensor;
    }
    for (auto deltaw_tensor : deltaw_tensors_) {
      delete deltaw_tensor;
    }

    // delete GenenralBuffers
    for (auto float_buf : float_bufs_) {
      delete float_buf;
    }
    for (auto uint32_buf : uint32_bufs_) {
      delete uint32_buf;
    }
    for (auto key_buf : key_bufs_) {
      delete key_buf;
    }
    for (auto value_index_buf : value_index_bufs_) {
      delete value_index_buf;
    }

  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }

}  // end of ~SparseEmbeddingHash()

template <typename TypeHashKey>
long long SparseEmbeddingHash<TypeHashKey>::get_params_num() {
  //  return ((long long)embedding_params_.embedding_vec_size * embedding_params_.vocabulary_size);

  // Read data from input_buffers_ -> look up -> write to output_tensors

  long long total_size = 0;

  CudaDeviceContext context(Base::device_resources_[0]->get_device_id());

  // need to collect the <key, value> pair count from all GPUs and do sum reduction
  int gpu_count = Base::device_resources_.size();
  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());
    total_size += hash_tables_[id]->get_size(*Base::device_resources_[id]->get_stream_ptr());
    CK_CUDA_THROW_(cudaStreamSynchronize(*Base::device_resources_[id]->get_stream_ptr()));
  }

  total_size *= (long long)embedding_params_.embedding_vec_size;

  return total_size;
}

template <typename TypeHashKey>
void SparseEmbeddingHash<TypeHashKey>::forward() {
  // Read data from input_buffers_ -> look up -> write to output_tensors

  CudaDeviceContext context(Base::device_resources_[0]->get_device_id());

  int local_gpu_count = Base::device_resources_.size();
  int total_gpu_count = Base::device_resources_.get_total_gpu_count();
  // launch kernels on GPUs: do embedding lookup on multi GPUs
  for (int id = 0; id < local_gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());

    // embedding lookup and reduction(sum)
    SparseEmbeddingHashKernels::do_forward(
        *Base::device_resources_[id]->get_stream_ptr(), embedding_params_.batch_size,
        embedding_params_.slot_num, embedding_params_.embedding_vec_size,
        Base::row_offsets_tensors_[id]->get_ptr(), Base::value_tensors_[id]->get_ptr(),
        hash_tables_[id], hash_table_value_tensors_[id]->get_ptr(),
        hash_value_index_tensors_[id]->get_ptr(), embedding_feature_tensors_[id]->get_ptr());
  }

  // sync
  for (int id = 0; id < local_gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());
    CK_CUDA_THROW_(cudaStreamSynchronize(*Base::device_resources_[id]->get_stream_ptr()));
  }

  // use NCCL to do Reduce-Scatter
  int batchsize_per_gpu = (int)(embedding_params_.batch_size / total_gpu_count);
  int recv_count =
      batchsize_per_gpu * embedding_params_.slot_num * embedding_params_.embedding_vec_size;
  if (total_gpu_count > 1) {
    CK_NCCL_THROW_(ncclGroupStart());
    for (int id = 0; id < local_gpu_count; id++) {
      auto input_tensor = embedding_feature_tensors_[id];
      auto output_tensor = Base::output_tensors_[id];

      CK_NCCL_THROW_(ncclReduceScatter(input_tensor->get_ptr(),   // send buf
                                       output_tensor->get_ptr(),  // recv buff
                                       recv_count, ncclFloat, ncclSum,
                                       *Base::device_resources_[id]->get_nccl_ptr(),
                                       *Base::device_resources_[id]->get_stream_ptr()));
    }
    CK_NCCL_THROW_(ncclGroupEnd());
  } else {  // total_gpu_count == 1
    context.set_device(Base::device_resources_[0]->get_device_id());
    Tensor<float> *output_tensor = Base::output_tensors_[0];
    CK_CUDA_THROW_(cudaMemcpyAsync(output_tensor->get_ptr(),
                                   embedding_feature_tensors_[0]->get_ptr(),
                                   recv_count * sizeof(float), cudaMemcpyDeviceToDevice,
                                   *Base::device_resources_[0]->get_stream_ptr()));
  }

  // sync
  for (int id = 0; id < local_gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());
    CK_CUDA_THROW_(cudaStreamSynchronize(*Base::device_resources_[id]->get_stream_ptr()));
  }

  // scale for combiner=mean after reduction
  if (embedding_params_.combiner == 1) {
    int send_count = embedding_params_.batch_size * embedding_params_.slot_num + 1;

    // use nccl all-reduce to get the row_offset_allreduce
    if (total_gpu_count > 1) {
      // need to know the type of TypeHashKey here
      ncclDataType_t type;
      switch (sizeof(TypeHashKey)) {
        case 4:
          type = ncclUint32;
          break;
        case 8:
          type = ncclUint64;
          break;
        default:
          CK_THROW_(Error_t::WrongInput, "Error: TypeHashKey not support by now");
      }

      CK_NCCL_THROW_(ncclGroupStart());
      for (int id = 0; id < local_gpu_count; id++) {
        CK_NCCL_THROW_(ncclAllReduce(Base::row_offsets_tensors_[id]->get_ptr(),
                                     row_offset_allreduce_tensors_[id]->get_ptr(), send_count, type,
                                     ncclSum, *Base::device_resources_[id]->get_nccl_ptr(),
                                     *Base::device_resources_[id]->get_stream_ptr()));
      }

      CK_NCCL_THROW_(ncclGroupEnd());
    } else {  // gpu == 1
      context.set_device(Base::device_resources_[0]->get_device_id());
      CK_CUDA_THROW_(cudaMemcpyAsync(row_offset_allreduce_tensors_[0]->get_ptr(),
                                     Base::row_offsets_tensors_[0]->get_ptr(),
                                     send_count * sizeof(TypeHashKey), cudaMemcpyDeviceToDevice,
                                     *Base::device_resources_[0]->get_stream_ptr()));
    }

    // sync
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device(Base::device_resources_[id]->get_device_id());
      CK_CUDA_THROW_(cudaStreamSynchronize(*Base::device_resources_[id]->get_stream_ptr()));
    }

    // do average
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device(Base::device_resources_[id]->get_device_id());

      TypeHashKey *row_offset =
          row_offset_allreduce_tensors_[id]->get_ptr() + id * batchsize_per_gpu;
      Tensor<float> *output_tensor = Base::output_tensors_[id];

      SparseEmbeddingHashKernels::do_forward_scale(*Base::device_resources_[id]->get_stream_ptr(),
                                                   batchsize_per_gpu, embedding_params_.slot_num,
                                                   embedding_params_.embedding_vec_size, row_offset,
                                                   output_tensor->get_ptr());
    }

    // sync
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device(Base::device_resources_[id]->get_device_id());
      CK_CUDA_THROW_(cudaStreamSynchronize(*Base::device_resources_[id]->get_stream_ptr()));
    }
  }

  return;
}  // end of forward()

template <typename TypeHashKey>
void SparseEmbeddingHash<TypeHashKey>::backward() {
  // Read dgrad from output_tensors -> compute wgrad

  CudaDeviceContext context(Base::device_resources_[0]->get_device_id());

  int local_gpu_count = Base::device_resources_.size();
  int total_gpu_count = Base::device_resources_.get_total_gpu_count();
  int batchsize_per_gpu = (int)(embedding_params_.batch_size / total_gpu_count);

  // use NCCL to do All-Gather
  // each recv buffer(embedding_feature_tensors_[i]) has the same top_grad data
  int send_count =
      batchsize_per_gpu * embedding_params_.slot_num * embedding_params_.embedding_vec_size;
  if (total_gpu_count > 1) {
    CK_NCCL_THROW_(ncclGroupStart());
    for (int id = 0; id < local_gpu_count; id++) {
      Tensor<float> *output_tensor = Base::output_tensors_[id];

      CK_NCCL_THROW_(ncclAllGather(output_tensor->get_ptr(),                   // send buff
                                   embedding_feature_tensors_[id]->get_ptr(),  // recv buff
                                   send_count, ncclFloat,
                                   *Base::device_resources_[id]->get_nccl_ptr(),
                                   *Base::device_resources_[id]->get_stream_ptr()));
    }
    CK_NCCL_THROW_(ncclGroupEnd());
  } else {  // total_gpu_count == 1
    context.set_device(Base::device_resources_[0]->get_device_id());
    Tensor<float> *output_tensor = Base::output_tensors_[0];
    CK_CUDA_THROW_(cudaMemcpyAsync(embedding_feature_tensors_[0]->get_ptr(),
                                   output_tensor->get_ptr(), send_count * sizeof(float),
                                   cudaMemcpyDeviceToDevice,
                                   *Base::device_resources_[0]->get_stream_ptr()));
  }

  // sync
  for (int id = 0; id < local_gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());
    CK_CUDA_THROW_(cudaStreamSynchronize(*Base::device_resources_[id]->get_stream_ptr()));
  }

  // do backward
  for (int id = 0; id < local_gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());

    // before backward, top diff data are already in embedding_feature_tensors_
    SparseEmbeddingHashKernels::do_backward(
        *Base::device_resources_[id]->get_stream_ptr(), embedding_params_.batch_size,
        embedding_params_.slot_num, embedding_params_.embedding_vec_size,
        embedding_params_.combiner, row_offset_allreduce_tensors_[id]->get_ptr(),
        embedding_feature_tensors_[id]
            ->get_ptr(),  // the output of forward, also the input of backward
        wgrad_tensors_[id]->get_ptr());
  }

  // sync
  for (int id = 0; id < local_gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());
    CK_CUDA_THROW_(cudaStreamSynchronize(*Base::device_resources_[id]->get_stream_ptr()));
  }

  return;
}  // end of backward()

template <typename TypeHashKey>
void SparseEmbeddingHash<TypeHashKey>::update_params_per_thread(int tid) {
#ifndef NDEBUG
  MESSAGE_("update_params_per_thread: this is thread: " + std::to_string(tid));
#endif

  CudaDeviceContext context(Base::device_resources_[tid]->get_device_id());

  // accumulate times for adam optimizer
  opt_params_[tid]->hyperparams.adam.times++;

  // do update params operation
  SparseEmbeddingHashKernels::do_update_params(
      *Base::device_resources_[tid]->get_stream_ptr(), embedding_params_.batch_size,
      embedding_params_.slot_num, embedding_params_.embedding_vec_size, max_vocabulary_size_per_gpu,
      *opt_params_[tid], Base::row_offsets_tensors_[tid]->get_ptr(),
      Base::value_tensors_[tid]->get_ptr(), hash_tables_[tid],
      hash_value_index_tensors_[tid]->get_ptr(), sample_id_tensors_[tid]->get_ptr(),
      sample_id_sort_tensors_[tid]->get_ptr(), hash_value_index_sort_tensors_[tid]->get_ptr(),
      hash_value_index_count_tensors_[tid]->get_ptr(),
      hash_value_index_count_offset_tensors_[tid]->get_ptr(),
      hash_value_index_count_counter_tensors_[tid]->get_ptr(),
      temp_storage_sort_tensors_[tid]->get_ptr(), temp_storage_sort_bytes_[tid],
      wgrad_tensors_[tid]->get_ptr(), deltaw_hash_value_index_tensors_[tid]->get_ptr(),
      deltaw_tensors_[tid]->get_ptr(), hash_table_value_tensors_[tid]->get_ptr());

  // stream sync
  CK_CUDA_THROW_(cudaStreamSynchronize(*Base::device_resources_[tid]->get_stream_ptr()));

  return;
}

template <typename TypeHashKey>
static void update_params_per_thread_wrapper_hash(
    int tid, SparseEmbeddingHash<TypeHashKey> *sparse_embedding_hash) {
  sparse_embedding_hash->update_params_per_thread(tid);
}

template <typename TypeHashKey>
void SparseEmbeddingHash<TypeHashKey>::update_params() {
  int local_gpu_count = Base::device_resources_.size();
  int total_gpu_count = Base::device_resources_.get_total_gpu_count();

  if (total_gpu_count > 1) {  // use multiple CPU threads to launch tasks on multiple GPUs
    // launch threads
    for (int id = 0; id < local_gpu_count; id++) {
      Base::device_resources_.results[id] = Base::device_resources_.train_thread_pool.push(
          std::ref(update_params_per_thread_wrapper_hash<TypeHashKey>), this);
    }

    // wait for threads completion
    for (int id = 0; id < local_gpu_count; id++) {
      Base::device_resources_.results[id].get();
    }
  } else if (total_gpu_count == 1) {  // use current main thread to launch task on one GPU
    update_params_per_thread(0);
  } else {
    throw std::runtime_error(
        std::string("[HCDEBUG][ERROR] Runtime error: total_gpu_count <= 0 \n"));
  }

  return;
}

// read hash_table_key and hash_table_value from host file, and write to GPU
template <typename TypeHashKey>
void SparseEmbeddingHash<TypeHashKey>::upload_params_to_device(std::ifstream &weight_stream) {
  // check if file is opened successfully
  if (!weight_stream.is_open()) {
    CK_THROW_(Error_t::WrongInput, "Error: file not open for reading");
  }

  // check file size and vocabulary_size (file size <=　hash_table_size)
  weight_stream.seekg(0, weight_stream.end);
  long long file_size_in_B = weight_stream.tellg();
  weight_stream.seekg(0, weight_stream.beg);
  long long hash_table_size_in_B =
      embedding_params_.vocabulary_size *
      ((long long)embedding_params_.embedding_vec_size * sizeof(float) +
       sizeof(TypeHashKey));  // hash_key size + hash_value size
  if (file_size_in_B > hash_table_size_in_B) {
    CK_THROW_(Error_t::WrongInput,
              "Error: hash table file size is larger than hash table vocabulary_size");
  }

  int my_rank = 0;
  int n_ranks = 1;
#ifdef ENABLE_MPI
  CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
  CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));
#endif

  CudaDeviceContext context(Base::device_resources_[0]->get_device_id());

  // define size
  int gpu_count = Base::device_resources_.size();
  int chunk_loop = 1000;
  int hash_table_key_tile_size =
      1;  // must be 1, because we need to cal (key&gpu_count) to decide gpu_id for each <key,value>
  int hash_table_key_tile_size_in_B = hash_table_key_tile_size * sizeof(TypeHashKey);
  int hash_table_key_chunk_size = hash_table_key_tile_size * chunk_loop;
  int hash_table_key_chunk_size_in_B = hash_table_key_chunk_size * sizeof(TypeHashKey);
  int hash_table_value_tile_size = embedding_params_.embedding_vec_size;
  int hash_table_value_tile_size_in_B = hash_table_value_tile_size * sizeof(float);
  int hash_table_value_chunk_size = hash_table_value_tile_size * chunk_loop;
  int hash_table_value_chunk_size_in_B = hash_table_value_chunk_size * sizeof(float);
  int hash_table_tile_size_in_B = hash_table_key_tile_size_in_B + hash_table_value_tile_size_in_B;
  int hash_table_chunk_size_in_B = hash_table_tile_size_in_B * chunk_loop;

  // CAUSION: can not decide how many values for each GPU, so need to allocate enough memory for
  // each GPU allocate GPU memory for hash_table_value_index
  long long *hash_table_value_index_count_per_gpu;  // <= hash_table_value_index_per_gpu_size
  hash_table_value_index_count_per_gpu = (long long *)malloc(sizeof(long long) * gpu_count);
  memset(hash_table_value_index_count_per_gpu, 0, sizeof(long long) * gpu_count);
  long long *hash_table_value_index_count_chunk_per_gpu;
  hash_table_value_index_count_chunk_per_gpu = (long long *)malloc(sizeof(long long) * gpu_count);
  memset(hash_table_value_index_count_chunk_per_gpu, 0, sizeof(long long) * gpu_count);
  TypeHashKey **hash_table_value_index_chunk_per_gpu_d;
  hash_table_value_index_chunk_per_gpu_d =
      (TypeHashKey **)malloc(sizeof(TypeHashKey *) * gpu_count);
  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());
    CK_CUDA_THROW_(
        cudaMalloc(&hash_table_value_index_chunk_per_gpu_d[id], hash_table_key_chunk_size_in_B));
    // initalize to zeros
    CK_CUDA_THROW_(cudaMemsetAsync(hash_table_value_index_chunk_per_gpu_d[id], 0,
                                   hash_table_key_chunk_size_in_B,
                                   *Base::device_resources_[id]->get_stream_ptr()));
  }

  // sync wait
  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());
    CK_CUDA_THROW_(cudaStreamSynchronize(*Base::device_resources_[id]->get_stream_ptr()));
  }

  // CAUSION: can not decide how many values for each GPU, so need to allocate enough memory for
  // each GPU allocate CPU/GPU memory for hash_table/key/value chunk
  char *hash_table_chunk;
  CK_CUDA_THROW_(cudaMallocHost(&hash_table_chunk, hash_table_chunk_size_in_B));
  TypeHashKey **hash_table_key_chunk_per_gpu;
  hash_table_key_chunk_per_gpu = (TypeHashKey **)malloc(gpu_count * sizeof(TypeHashKey *));
  for (int id = 0; id < gpu_count; id++) {
    CK_CUDA_THROW_(
        cudaMallocHost(&hash_table_key_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
  }
  TypeHashKey **hash_table_key_chunk_per_gpu_d;
  hash_table_key_chunk_per_gpu_d = (TypeHashKey **)malloc(gpu_count * sizeof(TypeHashKey *));
  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());
    CK_CUDA_THROW_(cudaMalloc(&hash_table_key_chunk_per_gpu_d[id], hash_table_key_chunk_size_in_B));
  }
  float **hash_table_value_chunk_per_gpu;
  hash_table_value_chunk_per_gpu = (float **)malloc(gpu_count * sizeof(float *));
  for (int id = 0; id < gpu_count; id++) {
    CK_CUDA_THROW_(
        cudaMallocHost(&hash_table_value_chunk_per_gpu[id], hash_table_value_chunk_size_in_B));
  }

  // do upload
  int loop_num = file_size_in_B / hash_table_chunk_size_in_B;
  for (int i = 0; i < loop_num; i++) {
    // read a chunk of data from file
    weight_stream.read(hash_table_chunk, hash_table_chunk_size_in_B);

    // memcpy from CPU to CPU
    char *src_buf = hash_table_chunk;
    TypeHashKey *key_dst_buf;
    float *value_dst_buf;
    for (int k = 0; k < chunk_loop; k++) {
      TypeHashKey key = *((TypeHashKey *)src_buf);
      int gid = key % Base::device_resources_.get_total_gpu_count();  // global GPU ID
      int id = Base::device_resources_.get_local_device_id(gid);      // local GPU ID
      int dst_rank = Base::device_resources_.get_pid(gid);

      if (my_rank == dst_rank) {
        // memcpy hash_table_key to corresponding GPU
        key_dst_buf =
            hash_table_key_chunk_per_gpu[id] + hash_table_value_index_count_chunk_per_gpu[id];
        CK_CUDA_THROW_(cudaMemcpyAsync(key_dst_buf, src_buf, hash_table_key_tile_size_in_B,
                                       cudaMemcpyHostToHost,
                                       *Base::device_resources_[id]->get_stream_ptr()));

        src_buf += hash_table_key_tile_size_in_B;

        // memcpy hash_table_value to corresponding GPU
        value_dst_buf =
            hash_table_value_chunk_per_gpu[id] +
            hash_table_value_index_count_chunk_per_gpu[id] * embedding_params_.embedding_vec_size;
        CK_CUDA_THROW_(cudaMemcpyAsync(value_dst_buf, src_buf, hash_table_value_tile_size_in_B,
                                       cudaMemcpyHostToHost,
                                       *Base::device_resources_[id]->get_stream_ptr()));

        src_buf += hash_table_value_tile_size_in_B;

        hash_table_value_index_count_chunk_per_gpu[id] += hash_table_key_tile_size;
      } else {
        break;
      }
    }  // end of for(int k = 0; k < (chunk_loop * gpu_count); k++)

    // do HashTable insert <key,value_index>
    for (int id = 0; id < gpu_count; id++) {
      context.set_device(Base::device_resources_[id]->get_device_id());

      long long value_index_chunk_size = hash_table_value_index_count_chunk_per_gpu[id];

      // memcpy hash_table_key from CPU to GPU
      CK_CUDA_THROW_(
          cudaMemcpyAsync(hash_table_key_chunk_per_gpu_d[id], hash_table_key_chunk_per_gpu[id],
                          value_index_chunk_size * sizeof(TypeHashKey), cudaMemcpyHostToDevice,
                          *Base::device_resources_[id]->get_stream_ptr()));

      long long value_index_offset = hash_table_value_index_count_per_gpu[id];
      TypeHashKey *value_index_buf = hash_table_value_index_chunk_per_gpu_d[id];
      // set hash_table_value_index on GPU
      SparseEmbeddingHashKernels::do_memset_liner(*Base::device_resources_[id]->get_stream_ptr(),
                                                  value_index_buf, (TypeHashKey)value_index_offset,
                                                  (TypeHashKey)1, value_index_chunk_size);

      // do hash table insert <key, value_index> on GPU
      hash_tables_[id]->insert(hash_table_key_chunk_per_gpu_d[id], value_index_buf,
                               value_index_chunk_size,
                               *Base::device_resources_[id]->get_stream_ptr());
      unsigned long long value_head = hash_tables_[id]->add_value_head(value_index_chunk_size);
    }

    // memcpy hash_table_value from CPU to GPU
    for (int id = 0; id < gpu_count; id++) {
      context.set_device(Base::device_resources_[id]->get_device_id());
      long long value_chunk_size =
          hash_table_value_index_count_chunk_per_gpu[id] * embedding_params_.embedding_vec_size;
      long long value_chunk_offset =
          hash_table_value_index_count_per_gpu[id] * embedding_params_.embedding_vec_size;
      float *src_buf = hash_table_value_chunk_per_gpu[id];
      float *dst_buf = hash_table_value_tensors_[id]->get_ptr() + value_chunk_offset;
      CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf, src_buf, value_chunk_size * sizeof(float),
                                     cudaMemcpyHostToDevice,
                                     *Base::device_resources_[id]->get_stream_ptr()));
    }

    // sync wait
    for (int id = 0; id < gpu_count; id++) {
      context.set_device(Base::device_resources_[id]->get_device_id());
      CK_CUDA_THROW_(cudaStreamSynchronize(*Base::device_resources_[id]->get_stream_ptr()));
    }

    // set counter value
    for (int id = 0; id < gpu_count; id++) {
      hash_table_value_index_count_per_gpu[id] += hash_table_value_index_count_chunk_per_gpu[id];
      hash_table_value_index_count_chunk_per_gpu[id] = 0;  // reset chunk counter to zero

      if (hash_table_value_index_count_per_gpu[id] > max_vocabulary_size_per_gpu) {
        char msg[100];
        sprintf(msg, "The size of hash table on GPU%d is out of range %d\n", id,
                max_vocabulary_size_per_gpu);
        CK_THROW_(Error_t::OutOfBound, msg);
      }
    }
  }  // end of for(int i = 0; i < loop_num; i++)

  // process the remaining data(less than a chunk)
  int remain_size_in_B = file_size_in_B - loop_num * hash_table_chunk_size_in_B;
  int remain_loop_num = remain_size_in_B / hash_table_tile_size_in_B;
  if (remain_loop_num) {
    // read all the remaining data
    weight_stream.read((char *)hash_table_chunk, remain_size_in_B);

    char *src_buf = hash_table_chunk;
    TypeHashKey *key_dst_buf;
    TypeHashKey *value_index_buf;
    float *value_dst_buf;
    for (int i = 0; i < remain_loop_num; i++) {
      TypeHashKey key = *((TypeHashKey *)src_buf);
      int gid = key % Base::device_resources_.get_total_gpu_count();  // global GPU ID
      int id = Base::device_resources_.get_local_device_id(gid);      // local GPU ID
      int dst_rank = Base::device_resources_.get_pid(gid);

      if (my_rank == dst_rank) {
        context.set_device(Base::device_resources_[id]->get_device_id());

        // memcpy hash_table_key from CPU to GPU
        key_dst_buf = hash_table_key_chunk_per_gpu_d[id];
        CK_CUDA_THROW_(cudaMemcpyAsync(key_dst_buf, src_buf, hash_table_key_tile_size_in_B,
                                       cudaMemcpyHostToDevice,
                                       *Base::device_resources_[id]->get_stream_ptr()));

        src_buf += hash_table_key_tile_size_in_B;

        // set value_index
        long long value_index_offset = hash_table_value_index_count_per_gpu[id];
        value_index_buf = hash_table_value_index_chunk_per_gpu_d[id];
        SparseEmbeddingHashKernels::do_memset_liner(
            *Base::device_resources_[id]->get_stream_ptr(), value_index_buf,
            (TypeHashKey)value_index_offset, (TypeHashKey)1, 1);

        // do hash table insert <key, value_index> on GPU
        hash_tables_[id]->insert(hash_table_key_chunk_per_gpu_d[id], value_index_buf,
                                 hash_table_key_tile_size,
                                 *Base::device_resources_[id]->get_stream_ptr());
        unsigned long long value_head = hash_tables_[id]->add_value_head(hash_table_key_tile_size);

        // memcpy hash_table_value from CPU to GPU
        long long value_offset =
            hash_table_value_index_count_per_gpu[id] * embedding_params_.embedding_vec_size;
        value_dst_buf = hash_table_value_tensors_[id]->get_ptr() + value_offset;
        CK_CUDA_THROW_(cudaMemcpyAsync(value_dst_buf, src_buf, hash_table_value_tile_size_in_B,
                                       cudaMemcpyHostToDevice,
                                       *Base::device_resources_[id]->get_stream_ptr()));
        src_buf += hash_table_value_tile_size_in_B;

        // set counter
        hash_table_value_index_count_per_gpu[id] += hash_table_key_tile_size;
      } else {
        break;
      }
    }

    // sync wait
    for (int id = 0; id < gpu_count; id++) {
      context.set_device(Base::device_resources_[id]->get_device_id());
      CK_CUDA_THROW_(cudaStreamSynchronize(*Base::device_resources_[id]->get_stream_ptr()));
    }

  }  // end of if(remain_loop_num)

  // release resources
  free(hash_table_value_index_count_per_gpu);
  free(hash_table_value_index_count_chunk_per_gpu);
  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());
    CK_CUDA_THROW_(cudaFree(hash_table_value_index_chunk_per_gpu_d[id]));
    CK_CUDA_THROW_(cudaFree(hash_table_key_chunk_per_gpu_d[id]));
  }
  free(hash_table_value_index_chunk_per_gpu_d);
  free(hash_table_key_chunk_per_gpu_d);
  CK_CUDA_THROW_(cudaFreeHost(hash_table_chunk));
  for (int id = 0; id < gpu_count; id++) {
    CK_CUDA_THROW_(cudaFreeHost(hash_table_key_chunk_per_gpu[id]));
    CK_CUDA_THROW_(cudaFreeHost(hash_table_value_chunk_per_gpu[id]));
  }
  free(hash_table_key_chunk_per_gpu);
  free(hash_table_value_chunk_per_gpu);

  return;
}  // end of upload_params_to_device()

// read hash_table_key and hash_table_value from GPU, and write to the file on the host
template <typename TypeHashKey>
void SparseEmbeddingHash<TypeHashKey>::download_params_to_host(std::ofstream &weight_stream) {
  // check if the file is opened successfully
  if (!weight_stream.is_open()) {
    CK_THROW_(Error_t::WrongInput, "Error: file not open for writing");
    return;
  }

  CudaDeviceContext context(Base::device_resources_[0]->get_device_id());

  int gpu_count = Base::device_resources_.size();

  // memory allocation
  unsigned long long *count;
  count = (unsigned long long *)malloc(gpu_count * sizeof(unsigned long long));
  unsigned long long max_count = 0;
  unsigned long long total_count = 0;
  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());
    auto count_tmp = hash_tables_[id]->get_size(*Base::device_resources_[id]->get_stream_ptr());
    if (count_tmp != hash_tables_[id]->get_value_head()) {
      CK_THROW_(Error_t::WrongInput,
                "Error: hash_table get_value_head() size not equal to get_size()");
    }
    count[id] = count_tmp;
    max_count = max(max_count, count[id]);
    total_count += count[id];
  }

#ifdef ENABLE_MPI
  CK_MPI_THROW_(MPI_Allreduce(MPI_IN_PLACE, &max_count, sizeof(unsigned long long), MPI_CHAR,
                              MPI_MAX, MPI_COMM_WORLD));
#endif

  if (total_count > (unsigned long long)embedding_params_.vocabulary_size) {
    CK_THROW_(Error_t::WrongInput,
              "Error: required download size is larger than hash table vocabulary_size");
  }

  TypeHashKey **h_hash_table_key, **d_hash_table_key;
  h_hash_table_key = (TypeHashKey **)malloc(gpu_count * sizeof(TypeHashKey *));
  d_hash_table_key = (TypeHashKey **)malloc(gpu_count * sizeof(TypeHashKey *));
  TypeHashValueIndex **d_hash_table_value_index;
  d_hash_table_value_index =
      (TypeHashValueIndex **)malloc(gpu_count * sizeof(TypeHashValueIndex *));
  float **h_hash_table_value, **d_hash_table_value;
  h_hash_table_value = (float **)malloc(gpu_count * sizeof(float *));
  d_hash_table_value = (float **)malloc(gpu_count * sizeof(float *));
  size_t **d_dump_counter;
  d_dump_counter = (size_t **)malloc(gpu_count * sizeof(size_t *));
  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());

    cudaMallocHost(&h_hash_table_key[id], count[id] * sizeof(TypeHashKey));
    cudaMalloc(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey));
    cudaMalloc(&d_hash_table_value_index[id], count[id] * sizeof(TypeHashValueIndex));
    cudaMallocHost(&h_hash_table_value[id],
                   count[id] * embedding_params_.embedding_vec_size * sizeof(float));
    cudaMalloc(&d_hash_table_value[id],
               count[id] * embedding_params_.embedding_vec_size * sizeof(float));
    cudaMalloc(&d_dump_counter[id], count[id] * sizeof(size_t));
  }

  // dump hash table on GPU
  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());

    hash_tables_[id]->dump(d_hash_table_key[id], d_hash_table_value_index[id], 0,
                           max_vocabulary_size_per_gpu, d_dump_counter[id],
                           *Base::device_resources_[id]->get_stream_ptr());

    CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_key[id], d_hash_table_key[id],
                                   count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost,
                                   *Base::device_resources_[id]->get_stream_ptr()));

    SparseEmbeddingHashKernels::do_get_hash_table_value(
        *Base::device_resources_[id]->get_stream_ptr(), count[id],
        embedding_params_.embedding_vec_size, d_hash_table_value_index[id],
        hash_table_value_tensors_[id]->get_ptr(), d_hash_table_value[id]);

    CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_value[id], d_hash_table_value[id],
                                   count[id] * embedding_params_.embedding_vec_size * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   *Base::device_resources_[id]->get_stream_ptr()));
  }

  // sync wait
  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());
    CK_CUDA_THROW_(cudaStreamSynchronize(*Base::device_resources_[id]->get_stream_ptr()));
  }

  int my_rank = 0;
  int n_ranks = 1;
#ifdef ENABLE_MPI
  CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
  CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));

#endif

  const int master_node = 0;
  const int base_tag = 0xed;
  // TODO: could be optimized ???
  unsigned long long max_size_in_B =
      max_count * (sizeof(TypeHashKey) + sizeof(float) * embedding_params_.embedding_vec_size);
  char *file_buf = (char *)malloc(max_size_in_B);
  size_t key_size = sizeof(TypeHashKey);
  size_t value_size = sizeof(float) * embedding_params_.embedding_vec_size;
  for (int id = 0; id < gpu_count; id++) {
    unsigned long long size_in_B =
        count[id] * (sizeof(TypeHashKey) + sizeof(float) * embedding_params_.embedding_vec_size);
    unsigned long long offset = 0;
    for (unsigned int k = 0; k < count[id]; k++) {
      memcpy(file_buf + offset, h_hash_table_key[id] + k, key_size);
      offset += key_size;
      memcpy(file_buf + offset, h_hash_table_value[id] + k * embedding_params_.embedding_vec_size,
             value_size);
      offset += value_size;
    }
    if (my_rank == master_node) {
      weight_stream.write(file_buf, size_in_B);
    }
#ifdef ENABLE_MPI
    else {
      int tag = (id << 8) | base_tag;
      CK_MPI_THROW_(MPI_Send(file_buf, size_in_B, MPI_CHAR, master_node, tag, MPI_COMM_WORLD));
    }
#endif
  }

#ifdef ENABLE_MPI
  if (my_rank == master_node) {
    for (int r = 1; r < n_ranks; r++) {
      for (int id = 0; id < gpu_count; id++) {
        int tag = (id << 8) | base_tag;
        MPI_Status status;
        CK_MPI_THROW_(MPI_Probe(r, tag, MPI_COMM_WORLD, &status));
        int size_in_B;
        CK_MPI_THROW_(MPI_Get_count(&status, MPI_CHAR, &size_in_B));
        CK_MPI_THROW_(
            MPI_Recv(file_buf, size_in_B, MPI_CHAR, r, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        weight_stream.write(file_buf, size_in_B);
      }
    }
  }
#endif

  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());

    CK_CUDA_THROW_(cudaFreeHost(h_hash_table_key[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_key[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_value_index[id]));
    CK_CUDA_THROW_(cudaFreeHost(h_hash_table_value[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_value[id]));
    CK_CUDA_THROW_(cudaFree(d_dump_counter[id]));
  }
  free(h_hash_table_key);
  free(d_hash_table_key);
  free(d_hash_table_value_index);
  free(h_hash_table_value);
  free(d_hash_table_value);
  free(d_dump_counter);
  free(count);
  free(file_buf);

  return;
}  // end of download_params_to_host()

// only used for results check: copy forward results from output_tensors_ to embedding_feature(input
// CPU buffer)
template <typename TypeHashKey>
float *SparseEmbeddingHash<TypeHashKey>::get_embedding_feature_ptr(float *embedding_feature) {
  int gpu_count = Base::device_resources_.size();

  CudaDeviceContext context(Base::device_resources_[0]->get_device_id());

  int batchsize_per_gpu =
      (int)(embedding_params_.batch_size / Base::device_resources_.get_total_gpu_count());
  int memcpy_size =
      batchsize_per_gpu * embedding_params_.slot_num * embedding_params_.embedding_vec_size;
  int offset = 0;
  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());
    Tensor<float> *output_tensor = Base::output_tensors_[id];
    CK_CUDA_THROW_(cudaMemcpyAsync(embedding_feature + offset, output_tensor->get_ptr(),
                                   memcpy_size * sizeof(float), cudaMemcpyDeviceToHost,
                                   *Base::device_resources_[id]->get_stream_ptr()));
    offset += memcpy_size;
  }

  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());
    CK_CUDA_THROW_(cudaStreamSynchronize(*Base::device_resources_[id]->get_stream_ptr()));
  }

  return embedding_feature;
}  // end of get_embedding_feature_ptr()

// only used for results check: copy backward() results from wgrad_tensors_ to wgrad(input CPU
// buffer)
template <typename TypeHashKey>
float *SparseEmbeddingHash<TypeHashKey>::get_wgrad_ptr(float *wgrad, int devIndex) {
  CudaDeviceContext context(Base::device_resources_[devIndex]->get_device_id());

  // wgard shuld be the same on multi-gpus
  int memcpy_size = embedding_params_.batch_size * embedding_params_.slot_num *
                    embedding_params_.embedding_vec_size;
  CK_CUDA_THROW_(cudaMemcpy(wgrad, wgrad_tensors_[devIndex]->get_ptr(), memcpy_size * sizeof(float),
                            cudaMemcpyDeviceToHost));

  return wgrad;
}  // end of get_wgrad_ptr()

// only used for results check: copy hash_tabale <key, value> from gpu to cpu
template <typename TypeHashKey>
void SparseEmbeddingHash<TypeHashKey>::get_hash_table_ptr(TypeHashKey *hash_table_key,
                                                          float *hash_table_value) {
  CudaDeviceContext context(Base::device_resources_[0]->get_device_id());

  int gpu_count = Base::device_resources_.size();

  // memory allocation
  unsigned long long *count;
  count = (unsigned long long *)malloc(gpu_count * sizeof(unsigned long long));
  unsigned long long max_count = 0;
  unsigned long long total_count = 0;
  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());
    if ((count[id] = hash_tables_[id]->get_value_head()) !=
        hash_tables_[id]->get_size(*Base::device_resources_[id]->get_stream_ptr())) {
      CK_THROW_(Error_t::WrongInput,
                "Error: hash_table get_value_head() size not equal to get_size()");
    }
    max_count = max(max_count, count[id]);
    total_count += count[id];

#ifndef NDEBUG
    std::cout << "GPU[%d]: " << id << "number of <key,value> pairs:%d" << count[id] << std::endl;
#endif
  }

  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());
    count[id] = hash_tables_[id]->get_size(*Base::device_resources_[id]->get_stream_ptr());
  }

  if (total_count > (unsigned long long)embedding_params_.vocabulary_size) {
    CK_THROW_(Error_t::WrongInput,
              "Error: required download size is larger than hash table vocabulary_size");
  }

  TypeHashKey **d_hash_table_key;
  d_hash_table_key = (TypeHashKey **)malloc(gpu_count * sizeof(TypeHashKey *));
  TypeHashValueIndex **d_hash_table_value_index;
  d_hash_table_value_index =
      (TypeHashValueIndex **)malloc(gpu_count * sizeof(TypeHashValueIndex *));
  float **d_hash_table_value;
  d_hash_table_value = (float **)malloc(gpu_count * sizeof(float *));
  size_t **d_dump_counter;
  d_dump_counter = (size_t **)malloc(gpu_count * sizeof(size_t *));
  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());

    cudaMalloc(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey));
    cudaMalloc(&d_hash_table_value_index[id], count[id] * sizeof(TypeHashValueIndex));
    cudaMalloc(&d_hash_table_value[id],
               count[id] * embedding_params_.embedding_vec_size * sizeof(float));
    cudaMalloc(&d_dump_counter[id], count[id] * sizeof(size_t));
  }

  // dump hash table on GPU
  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());

    hash_tables_[id]->dump(d_hash_table_key[id], d_hash_table_value_index[id], 0,
                           max_vocabulary_size_per_gpu, d_dump_counter[id],
                           *Base::device_resources_[id]->get_stream_ptr());

    SparseEmbeddingHashKernels::do_get_hash_table_value(
        *Base::device_resources_[id]->get_stream_ptr(), count[id],
        embedding_params_.embedding_vec_size, d_hash_table_value_index[id],
        hash_table_value_tensors_[id]->get_ptr(), d_hash_table_value[id]);
  }

  // sync wait
  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());
    CK_CUDA_THROW_(cudaStreamSynchronize(*Base::device_resources_[id]->get_stream_ptr()));
  }

  // memcpy from GPU to CPU memory
  long long key_offset = 0;
  long long value_offset = 0;
  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());

    CK_CUDA_THROW_(cudaMemcpy(hash_table_key + key_offset, d_hash_table_key[id],
                              count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost));
    key_offset += count[id];

    CK_CUDA_THROW_(cudaMemcpy(hash_table_value + value_offset, d_hash_table_value[id],
                              count[id] * embedding_params_.embedding_vec_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
    value_offset += count[id] * embedding_params_.embedding_vec_size;
  }

  for (int id = 0; id < gpu_count; id++) {
    context.set_device(Base::device_resources_[id]->get_device_id());

    CK_CUDA_THROW_(cudaFree(d_hash_table_key[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_value_index[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_value[id]));
    CK_CUDA_THROW_(cudaFree(d_dump_counter[id]));
  }
  free(d_hash_table_key);
  free(d_hash_table_value_index);
  free(d_hash_table_value);
  free(d_dump_counter);
  free(count);
}  // end of get_hash_table_value_ptr()

}  // namespace HugeCTR
