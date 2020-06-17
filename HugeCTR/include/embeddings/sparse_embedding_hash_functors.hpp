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
#include "HugeCTR/include/embeddings/sparse_embedding_kernels.cuh"
#ifndef NCCL_A2A
#include "HugeCTR/include/faster_gossip_comm/FasterGossipComm/FasterGossipComm.h"
#endif
#include <iomanip>
#include "HugeCTR/include/gpu_resource.hpp"
#include "HugeCTR/include/hashtable/nv_hashtable.cuh"
#include "HugeCTR/include/utils.hpp"
#include "cub/cub/device/device_radix_sort.cuh"
#include "cub/cub/device/device_scan.cuh"

#ifdef ENABLE_MPI
#include <mpi.h>
#ifndef NCCL_A2A
#include "HugeCTR/include/faster_gossip_comm/FasterGossipComm/FasterGossipCommMulti.h"
#endif
#endif

#define SGD_ATOMIC  // if define this macro, the atomic method for SGD optimizer will be used

namespace HugeCTR {

class SparseEmbeddingHashFunctors {
 private:
  int sm_count_;

 public:
  /**
   * Ctor of SparseEmbeddingHashFunctors. Copy construction and assigment are disabled.
   */
  SparseEmbeddingHashFunctors() {
    // Assumption: all the GPUs in the server are the same.
    CK_CUDA_THROW_(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, 0));
  }

  SparseEmbeddingHashFunctors(SparseEmbeddingHashFunctors &obj) = delete;
  SparseEmbeddingHashFunctors &operator=(const SparseEmbeddingHashFunctors &) = delete;

  /**
   * Dtor of SparseEmbeddingHashFunctors.
   */
  ~SparseEmbeddingHashFunctors() {}

  /**
   * stream sync on multi GPUs.
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device.
   */
  void sync_all_gpus(const std::shared_ptr<GPUResourceGroup> &device_resources,
                     const CudaDeviceContext &context) {
    int local_gpu_count = device_resources->size();

    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      CK_CUDA_THROW_(cudaStreamSynchronize((*device_resources)[id]->get_stream()));
    }
  }

  /**
   * forward propagation for DistributedSlotSparseEmbeddingHash
   * @param batch_size batch size for the current mini-batch computation.
   * @param slot_num the number of slots in hash table.
   * @param embedding_vec_size embedding vector size.
   * @param row_offsets_tensors row_offsets tensors of mulitple GPUs (CSR format of input sparse
   * tensors)
   * @param value_tensors value tensors of multi GPUs (CSR format of input sparse tensors)
   * @param nnz_num_per_batch non-zero feature number per batch of multi GPUs
   * @param hash_tables hash table of multi GPUs, pairs of <key, value_index>
   * @param hash_table_value_tensors hash table value tenosrs of multi GPUs, the value is
   * represented for embedding vector
   * @param hash_value_index_tensors hash table value_index tensors of multi GPUs
   * @param embedding_feature_tensors embedding feature tensors on multi GPUs
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  template <typename TypeHashKey, typename TypeHashValueIndex, typename TypeEmbeddingComp>
  void forward(
      int batch_size, int slot_num, int embedding_vec_size,
      const Tensors<TypeHashKey> &row_offsets_tensors, const Tensors<TypeHashKey> &value_tensors,
      std::vector<PinnedBuffer<TypeHashKey>> &nnz_num_per_batch,
      const std::vector<std::unique_ptr<
          HashTable<TypeHashKey, TypeHashValueIndex, std::numeric_limits<TypeHashKey>::max()>>>
          &hash_tables,
      const Tensors<float> &hash_table_value_tensors,
      const Tensors<TypeHashValueIndex> &hash_value_index_tensors,
      Tensors<TypeEmbeddingComp> &embedding_feature_tensors,
      const std::shared_ptr<GPUResourceGroup> &device_resources, const CudaDeviceContext &context) {
    int local_gpu_count = device_resources->size();

    // need to know the Type
    switch (sizeof(TypeEmbeddingComp)) {
      case 2:                                         // fp16
        embedding_vec_size = embedding_vec_size / 2;  // use __half2
        break;
      case 4:  // fp32
        embedding_vec_size = embedding_vec_size;
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
    }

    // launch kernels on GPUs: do embedding lookup on multi GPUs
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());

      const auto &row_offset = row_offsets_tensors[id]->get_ptr();
      const auto &hash_key = value_tensors[id]->get_ptr();
      auto nnz = nnz_num_per_batch[id].get();
      const auto &hash_table = hash_tables[id].get();
      const auto &hash_table_value = hash_table_value_tensors[id]->get_ptr();
      const auto &hash_value_index = hash_value_index_tensors[id]->get_ptr();
      auto embedding_feature = embedding_feature_tensors[id]->get_ptr();
      const cudaStream_t stream = (*device_resources)[id]->get_stream();

      try {
        // get hash_value_index from hash_table by hash_key
        CK_CUDA_THROW_(cudaMemcpyAsync(nnz, &row_offset[batch_size * slot_num], sizeof(TypeHashKey),
                                       cudaMemcpyDeviceToHost, stream));
        CK_CUDA_THROW_(cudaStreamSynchronize(stream));
        hash_table->get_insert(hash_key, hash_value_index, *nnz, stream);

        // do sum reduction
        dim3 blockSize(embedding_vec_size, 1,
                       1);  // each thread corresponds to one element in an embedding vector
        dim3 gridSize(batch_size, 1, 1);  // each block corresponds to a sample
        forward_sum_kernel<TypeHashKey, TypeHashValueIndex><<<gridSize, blockSize, 0, stream>>>(
            batch_size, slot_num, embedding_vec_size, row_offset, hash_value_index,
            hash_table_value, embedding_feature);
        // for combiner=mean, call forward_scale() after this forward() and NCCL all-reduce
        // operation

      } catch (const std::runtime_error &rt_err) {
        std::cerr << rt_err.what() << std::endl;
        throw;
      }
    }

    return;
  }

  /**
   * forward propagation on each GPU for DistributedSlotSparseEmbeddingHash
   * @param batch_size batch size for the current mini-batch computation.
   * @param slot_num the number of slots for current GPU
   * @param embedding_vec_size embedding vector size.
   * @param row_offset row_offset (CSR format of input sparse tensors)
   * @param hash_key value (CSR format of input sparse tensors)
   * @param nnz non-zero feature number per batch
   * @param hash_table hash table, pairs of <key, value_index>
   * @param hash_table_value hash table value, which represents embedding vector
   * @param hash_value_index hash table value_index(row index of embedding)
   * @param embedding_feature embedding feature (output)
   * @param stream cuda stream
   */
  template <typename TypeHashKey, typename TypeHashValueIndex, typename TypeEmbeddingComp>
  void forward_per_gpu(int batch_size, int slot_num, int embedding_vec_size,
                       const TypeHashKey *row_offset, const TypeHashKey *hash_key, TypeHashKey *nnz,
                       const HashTable<TypeHashKey, TypeHashValueIndex,
                                       std::numeric_limits<TypeHashKey>::max()> *hash_table,
                       const float *hash_table_value, TypeHashValueIndex *hash_value_index,
                       TypeEmbeddingComp *embedding_feature, cudaStream_t stream) {
    // need to know the Type
    switch (sizeof(TypeEmbeddingComp)) {
      case 2:                                         // fp16
        embedding_vec_size = embedding_vec_size / 2;  // use __half2
        break;
      case 4:  // fp32
        embedding_vec_size = embedding_vec_size;
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
    }

    try {
      // get hash_value_index from hash_table by hash_key
      CK_CUDA_THROW_(cudaMemcpyAsync(nnz, &row_offset[batch_size * slot_num], sizeof(TypeHashKey),
                                     cudaMemcpyDeviceToHost, stream));
      CK_CUDA_THROW_(cudaStreamSynchronize(stream));
      hash_table->get_insert(hash_key, hash_value_index, *nnz, stream);

      // do sum reduction
      dim3 blockSize(embedding_vec_size, 1,
                     1);  // each thread corresponds to one element in an embedding vector
      dim3 gridSize(batch_size, 1, 1);  // each block corresponds to a sample
      forward_sum_kernel<TypeHashKey, TypeHashValueIndex><<<gridSize, blockSize, 0, stream>>>(
          batch_size, slot_num, embedding_vec_size, row_offset, hash_value_index, hash_table_value,
          embedding_feature);
      // for combiner=mean, call forward_scale() after this forward() and NCCL all-reduce
      // operation

    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }

    return;
  }

  /**
   * forward propagation for LocalizedSlotSparseEmbeddingHash
   * @param batch_size batch size for the current mini-batch computation.
   * @param slot_num_per_gpu the number of slots for each local GPUs
   * @param embedding_vec_size embedding vector size.
   * @param combiner 0-sum; 1-mean
   * @param row_offsets_tensors row_offsets tensors of mulitple GPUs (CSR format of input sparse
   * tensors)
   * @param value_tensors value tensors of multi GPUs (CSR format of input sparse tensors)
   * @param nnz_num_per_batch non-zero feature number per batch of multi GPUs
   * @param hash_tables hash table of multi GPUs, pairs of <key, value_index>
   * @param hash_table_value_tensors hash table value tenosrs of multi GPUs, the value is
   * represented for embedding vector
   * @param hash_value_index_tensors hash table value_index tensors of multi GPUs
   * @param embedding_feature_tensors embedding feature tensors on multi GPUs
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  template <typename TypeHashKey, typename TypeHashValueIndex, typename TypeEmbeddingComp>
  void forward(
      int batch_size, const std::vector<int> &slot_num_per_gpu, int embedding_vec_size,
      int combiner, const Tensors<TypeHashKey> &row_offsets_tensors,
      const Tensors<TypeHashKey> &value_tensors,
      std::vector<PinnedBuffer<TypeHashKey>> &nnz_num_per_batch,
      const std::vector<std::unique_ptr<
          HashTable<TypeHashKey, TypeHashValueIndex, std::numeric_limits<TypeHashKey>::max()>>>
          &hash_tables,
      const Tensors<float> &hash_table_value_tensors,
      const Tensors<TypeHashValueIndex> &hash_value_index_tensors,
      Tensors<TypeEmbeddingComp> &embedding_feature_tensors,
      const std::shared_ptr<GPUResourceGroup> &device_resources, const CudaDeviceContext &context) {
    int local_gpu_count = device_resources->size();
    int total_gpu_count = device_resources->get_total_gpu_count();

    // need to know the Type
    switch (sizeof(TypeEmbeddingComp)) {
      case 2:                                         // fp16
        embedding_vec_size = embedding_vec_size / 2;  // use __half2
        break;
      case 4:  // fp32
        embedding_vec_size = embedding_vec_size;
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
    }

    // launch kernels on GPUs: do embedding lookup on multi GPUs
    for (int id = 0; id < local_gpu_count; id++) {
      if (slot_num_per_gpu[id] == 0) {
        continue;
      }

      int cur_device = (*device_resources)[id]->get_device_id();
      context.set_device(cur_device);

      const auto &row_offset = row_offsets_tensors[id]->get_ptr();
      const auto &hash_key = value_tensors[id]->get_ptr();
      auto nnz = nnz_num_per_batch[id].get();
      const auto &hash_table = hash_tables[id].get();
      const auto &hash_table_value = hash_table_value_tensors[id]->get_ptr();
      const auto &hash_value_index = hash_value_index_tensors[id]->get_ptr();
      auto embedding_feature = embedding_feature_tensors[id]->get_ptr();
      const cudaStream_t stream = (*device_resources)[id]->get_stream();

      try {
        // get hash_value_index from hash_table by hash_key
        CK_CUDA_THROW_(cudaMemcpyAsync(nnz, &row_offset[batch_size * slot_num_per_gpu[id]],
                                       sizeof(TypeHashKey), cudaMemcpyDeviceToHost, stream));
        CK_CUDA_THROW_(cudaStreamSynchronize(stream));
        hash_table->get_insert(hash_key, hash_value_index, *nnz, stream);

        // do sum reduction
        dim3 blockSize(embedding_vec_size, 1,
                       1);  // each thread corresponds to one element in an embedding vector
        dim3 gridSize(batch_size, 1, 1);  // each block corresponds to a sample
        if (combiner == 0) {
          forward_sum_kernel<TypeHashKey, TypeHashValueIndex><<<gridSize, blockSize, 0, stream>>>(
              batch_size, slot_num_per_gpu[id], embedding_vec_size, row_offset, hash_value_index,
              hash_table_value, embedding_feature);
        } else if (combiner == 1) {
          forward_mean_kernel<TypeHashKey, TypeHashValueIndex><<<gridSize, blockSize, 0, stream>>>(
              batch_size, slot_num_per_gpu[id], embedding_vec_size, row_offset, hash_value_index,
              hash_table_value, embedding_feature);
        } else {
          CK_THROW_(Error_t::WrongInput, "Invalid combiner type ");
        }
      } catch (const std::runtime_error &rt_err) {
        std::cerr << rt_err.what() << std::endl;
        throw;
      }
    }

    return;
  }

  /**
   * forward propagation on each GPU for LocalizedSlotSparseEmbeddingHash
   * @param batch_size batch size for the current mini-batch computation.
   * @param slot_num the number of slots for current GPU
   * @param embedding_vec_size embedding vector size.
   * @param combiner 0-sum; 1-mean
   * @param row_offset row_offset (CSR format of input sparse tensors)
   * @param hash_key value (CSR format of input sparse tensors)
   * @param nnz non-zero feature number per batch
   * @param hash_table hash table, pairs of <key, value_index>
   * @param hash_table_value hash table value, which represents embedding vector
   * @param hash_value_index hash table value_index(row index of embedding)
   * @param embedding_feature embedding feature (output)
   * @param stream cuda stream
   */
  template <typename TypeHashKey, typename TypeHashValueIndex, typename TypeEmbeddingComp>
  void forward_per_gpu(int batch_size, int slot_num, int embedding_vec_size, int combiner,
                       const TypeHashKey *row_offset, const TypeHashKey *hash_key, TypeHashKey *nnz,
                       const HashTable<TypeHashKey, TypeHashValueIndex,
                                       std::numeric_limits<TypeHashKey>::max()> *hash_table,
                       const float *hash_table_value, TypeHashValueIndex *hash_value_index,
                       TypeEmbeddingComp *embedding_feature, cudaStream_t stream) {
    // need to know the Type
    switch (sizeof(TypeEmbeddingComp)) {
      case 2:                                         // fp16
        embedding_vec_size = embedding_vec_size / 2;  // use __half2
        break;
      case 4:  // fp32
        embedding_vec_size = embedding_vec_size;
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
    }

    try {
      // get hash_value_index from hash_table by hash_key
      CK_CUDA_THROW_(cudaMemcpyAsync(nnz, &row_offset[batch_size * slot_num], sizeof(TypeHashKey),
                                     cudaMemcpyDeviceToHost, stream));
      CK_CUDA_THROW_(cudaStreamSynchronize(stream));
      hash_table->get_insert(hash_key, hash_value_index, *nnz, stream);

      // do sum reduction
      dim3 blockSize(embedding_vec_size, 1,
                     1);  // each thread corresponds to one element in an embedding vector
      dim3 gridSize(batch_size, 1, 1);  // each block corresponds to a sample
      if (combiner == 0) {
        forward_sum_kernel<TypeHashKey, TypeHashValueIndex><<<gridSize, blockSize, 0, stream>>>(
            batch_size, slot_num, embedding_vec_size, row_offset, hash_value_index,
            hash_table_value, embedding_feature);
      } else if (combiner == 1) {
        forward_mean_kernel<TypeHashKey, TypeHashValueIndex><<<gridSize, blockSize, 0, stream>>>(
            batch_size, slot_num, embedding_vec_size, row_offset, hash_value_index,
            hash_table_value, embedding_feature);
      } else {
        CK_THROW_(Error_t::WrongInput, "Invalid combiner type ");
      }
    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }

    return;
  }

  /**
   * forward propagation on each GPU for LocalizedSlotSparseEmbeddingOneHot
   * @param batch_size batch size for the current mini-batch computation.
   * @param slot_num the number of slots for current GPU
   * @param embedding_vec_size embedding vector size.
   * @param combiner 0-sum; 1-mean
   * @param row_offset row_offset (CSR format of input sparse tensors)
   * @param hash_key value (CSR format of input sparse tensors)
   * @param nnz non-zero feature number per batch
   * @param hash_table hash table, pairs of <key, value_index>
   * @param hash_table_value hash table value, which represents embedding vector
   * @param hash_value_index hash table value_index(row index of embedding)
   * @param embedding_feature embedding feature (output)
   * @param stream cuda stream
   */
  template <typename TypeHashKey, typename TypeHashValueIndex, typename TypeEmbeddingComp>
  void forward_per_gpu(int batch_size, int slot_num, int embedding_vec_size, int combiner,
                       const TypeHashKey *row_offset, const TypeHashKey *hash_key, TypeHashKey *nnz,
                       const uint32_t *mapping_offsets, const float *hash_table_value,
                       TypeHashValueIndex *hash_value_index, TypeEmbeddingComp *embedding_feature,
                       cudaStream_t stream) {
    // need to know the Type
    switch (sizeof(TypeEmbeddingComp)) {
      case 2:                                         // fp16
        embedding_vec_size = embedding_vec_size / 2;  // use __half2
        break;
      case 4:  // fp32
        embedding_vec_size = embedding_vec_size;
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
    }

    try {
      // get hash_value_index from hash_table by hash_key
      CK_CUDA_THROW_(cudaMemcpyAsync(nnz, &row_offset[batch_size * slot_num], sizeof(TypeHashKey),
                                     cudaMemcpyDeviceToHost, stream));
      CK_CUDA_THROW_(cudaStreamSynchronize(stream));

      // remove hashtable get_insert(), and do linear mapping between key and value_index
      hash_key_value_index_mapping_kernel<<<((*nnz) + 255) / 256, 256, 0, stream>>>(
          (*nnz), slot_num, mapping_offsets, hash_key, hash_value_index);

      // do sum reduction
      dim3 blockSize(embedding_vec_size, 1,
                     1);  // each thread corresponds to one element in an embedding vector
      dim3 gridSize(batch_size, 1, 1);  // each block corresponds to a sample
      if (combiner == 0) {
        forward_sum_kernel<TypeHashKey, TypeHashValueIndex><<<gridSize, blockSize, 0, stream>>>(
            batch_size, slot_num, embedding_vec_size, row_offset, hash_value_index,
            hash_table_value, embedding_feature);
      } else {
        CK_THROW_(Error_t::WrongInput, "Invalid combiner type ");
      }
    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }

    return;
  }

  /**
   * forward propagation on each GPU for LocalizedSlotSparseEmbeddingOneHot.
   * Because there is no hashtable in this class, so there must be a mapping table
   * between input valud_index and local value_index.
   * @param batch_size batch size for the current mini-batch computation.
   * @param slot_num the number of slots for current GPU
   * @param row_offset row_offset (CSR format of input sparse tensors)
   * @param hash_key value (CSR format of input sparse tensors)
   * @param nnz non-zero feature number per batch
   * @param mapping_offsets the mapping between input value_index and local value_index
   * @param hash_value_index hash table value_index(row index of embedding)
   * @param stream cuda stream
   */
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void forward_mapping_per_gpu(size_t batch_size, int slot_num, const TypeHashKey *row_offset,
                               const TypeHashKey *hash_key, TypeHashKey *nnz,
                               const uint32_t *mapping_offsets,
                               TypeHashValueIndex *hash_value_index, cudaStream_t stream) {
    try {
      CK_CUDA_THROW_(cudaMemcpyAsync(nnz, &row_offset[batch_size * slot_num], sizeof(TypeHashKey),
                                     cudaMemcpyDeviceToHost, stream));
      CK_CUDA_THROW_(cudaStreamSynchronize(stream));

      // remove hashtable get_insert(), and do linear mapping between key and value_index
      hash_key_value_index_mapping_kernel<<<((*nnz) + 255) / 256, 256, 0, stream>>>(
          (*nnz), slot_num, mapping_offsets, hash_key, hash_value_index);
    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }

    return;
  }

  /**
   * forward propagation for LocalizedSlotSparseEmbeddingOneHot.
   * fuse (forward_sum_kernel + all2all + forward_reorder) into one kernel.
   * Only support single node currently.
   * @param batch_size batch size for the current mini-batch computation.
   * @param slot_num total slots number
   * @param slot_num_per_gpu the number of slots for each GPU
   * @param embedding_vec_size embedding vector size.
   * @param combiner 0-sum; 1-mean
   * @param row_offsets_tensors row_offset (CSR format of input sparse tensors)
   * @param hash_table_value_tensors hash table value, which represents embedding vector
   * @param hash_value_index_tensors hash table value_index(row index of embedding)
   * @param embedding_features embedding features of all gpus (output)
   * @param device_resources device resources
   */
  template <typename TypeHashKey, typename TypeHashValueIndex, typename TypeEmbeddingComp>
  void forward_fuse(size_t batch_size, int slot_num, const std::vector<int> &slot_num_per_gpu,
                    int embedding_vec_size, int combiner,
                    const Tensors<TypeHashKey> &row_offsets_tensors,
                    const Tensors<float> &hash_table_value_tensors,
                    const Tensors<TypeHashValueIndex> &hash_value_index_tensors,
                    TypeEmbeddingComp **embedding_features,
                    const std::shared_ptr<GPUResourceGroup> &device_resources) {
    int local_gpu_count = device_resources->size();
    size_t batch_size_per_gpu = batch_size / local_gpu_count;

    // need to know the Type
    switch (sizeof(TypeEmbeddingComp)) {
      case 2:                                         // fp16
        embedding_vec_size = embedding_vec_size / 2;  // use __half2
        break;
      case 4:  // fp32
        embedding_vec_size = embedding_vec_size;
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
    }

    dim3 blockSize(embedding_vec_size, 1, 1);
    int maxActiveBlocks;
    switch (sizeof(TypeEmbeddingComp)) {
      case 2:  // fp16
        CK_CUDA_THROW_(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks, forward_sum_fuse_kernel_fp16<TypeHashKey, TypeHashValueIndex>,
            blockSize.x, 0));
        break;
      case 4:  // fp32
        CK_CUDA_THROW_(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks, forward_sum_fuse_kernel_fp32<TypeHashKey, TypeHashValueIndex>,
            blockSize.x, 0));
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
    }
    dim3 gridSize(min((int)batch_size, (sm_count_ * maxActiveBlocks)), 1, 1);

    // std::cout << "batch_size=" << batch_size << ", sm_count=" << sm_count_
    //     << ", maxActiveBlocks=" << maxActiveBlocks << ", girdSize=" << gridSize.x
    //     << std::endl;

    struct cudaLaunchParams params[local_gpu_count];
    void *kargs[local_gpu_count][11];
    int id_list[local_gpu_count];
    int slot_num_list[local_gpu_count];
    TypeHashKey *row_offsets_list[local_gpu_count];
    float *hash_table_value_list[local_gpu_count];
    TypeHashValueIndex *hash_value_index_list[local_gpu_count];

    for (int id = 0; id < local_gpu_count; id++) {
      if (slot_num_per_gpu[id] == 0) {
        continue;
      }

      id_list[id] = (*device_resources)[id]->get_device_id();
      slot_num_list[id] = slot_num_per_gpu[id];
      row_offsets_list[id] = row_offsets_tensors[id]->get_ptr();
      hash_value_index_list[id] = hash_value_index_tensors[id]->get_ptr();
      hash_table_value_list[id] = hash_table_value_tensors[id]->get_ptr();

      switch (sizeof(TypeEmbeddingComp)) {
        case 2:  // fp16
          params[id].func = (void *)forward_sum_fuse_kernel_fp16<TypeHashKey, TypeHashValueIndex>;
          break;
        case 4:  // fp32
          params[id].func = (void *)forward_sum_fuse_kernel_fp32<TypeHashKey, TypeHashValueIndex>;
          break;
        default:
          CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
      }

      params[id].gridDim = gridSize;
      params[id].blockDim = blockSize;
      params[id].sharedMem = 0;
      params[id].stream = (*device_resources)[id]->get_stream();
      kargs[id][0] = (void *)&(id_list[id]);
      kargs[id][1] = (void *)&local_gpu_count;
      kargs[id][2] = (void *)&batch_size;
      kargs[id][3] = (void *)&batch_size_per_gpu;
      kargs[id][4] = (void *)&slot_num;
      kargs[id][5] = (void *)&(slot_num_list[id]);
      kargs[id][6] = (void *)&embedding_vec_size;
      kargs[id][7] = (void *)&(row_offsets_list[id]);
      kargs[id][8] = (void *)&(hash_value_index_list[id]);
      kargs[id][9] = (void *)&(hash_table_value_list[id]);
      kargs[id][10] = (void *)&embedding_features;
      params[id].args = (void **)kargs[id];
    }

    try {
      if (combiner == 0) {
        CK_CUDA_THROW_(cudaLaunchCooperativeKernelMultiDevice(
            params, local_gpu_count,
            cudaCooperativeLaunchMultiDeviceNoPreSync));  // no pre-sync but has post-sync
      } else {
        CK_THROW_(Error_t::WrongInput, "Invalid combiner type ");
      }
    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }

    return;
  }

  /**
   * An additional function for the forward propagation when (combiner=mean).
   *  (only for DistributedSlotSparseEmbeddingHash)
   * @param batch_size batch size for the current mini-batch computation.
   * @param slot_num the number of slots
   * @param embedding_vec_size embedding vector size.
   * @param row_offset_allreduce_tensors row_offsets tensors after all_reduce of mulitple GPUs
   * @param output_tensors forward prop output tensors of multi GPUs
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  template <typename TypeHashKey, typename TypeEmbeddingComp>
  void forward_scale(int batch_size, int slot_num, int embedding_vec_size,
                     const Tensors<TypeHashKey> &row_offset_allreduce_tensors,
                     Tensors<TypeEmbeddingComp> &output_tensors,
                     const std::shared_ptr<GPUResourceGroup> &device_resources,
                     const CudaDeviceContext &context) {
    int local_gpu_count = device_resources->size();
    int total_gpu_count = device_resources->get_total_gpu_count();

    // need to know the Type
    switch (sizeof(TypeEmbeddingComp)) {
      case 2:                                         // fp16
        embedding_vec_size = embedding_vec_size / 2;  // use __half2
        break;
      case 4:  // fp32
        embedding_vec_size = embedding_vec_size;
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
    }

    int batchsize_per_gpu = (int)(batch_size / total_gpu_count);
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());

      const auto &row_offset =
          row_offset_allreduce_tensors[id]->get_ptr() + id * batchsize_per_gpu * slot_num;
      auto embedding_feature = output_tensors[id]->get_ptr();
      const auto &stream = (*device_resources)[id]->get_stream();

      try {
        dim3 blockSize(embedding_vec_size, 1, 1);
        dim3 gridSize(batchsize_per_gpu, 1, 1);

        forward_scale_kernel<<<gridSize, blockSize, 0, stream>>>(
            batchsize_per_gpu, slot_num, embedding_vec_size, row_offset, embedding_feature);

      } catch (const std::runtime_error &rt_err) {
        std::cerr << rt_err.what() << std::endl;
        throw;
      }
    }

    return;
  }

  /**
   * backward propagation for DistributedSlotSparseEmbeddingHash
   * The first step of backward propagation: computing the wgrad.
   * @param batch_size batch size for the current mini-batch computation.
   * @param slot_num the number of slots in hash table.
   * @param embedding_vec_size embedding vector size.
   * @param combiner combiner type: 0-sum, 1-mean
   * @param row_offset_allreduce_tensors row_offsets tensors after all_reduce of mulitple GPUs
   * @param embedding_feature_tensors embedding features tensors of multiplu GPUs, storing dgrad
   * from the top layer
   * @param wgrad_tensors wgrad tensors of multi GPUs, the output of this function.
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  template <typename TypeHashKey, typename TypeEmbeddingComp>
  void backward(int batch_size, int slot_num, int embedding_vec_size, int combiner,
                const Tensors<TypeHashKey> &row_offset_allreduce_tensors,
                const Tensors<TypeEmbeddingComp> &embedding_feature_tensors,
                Tensors<TypeEmbeddingComp> &wgrad_tensors,
                const std::shared_ptr<GPUResourceGroup> &device_resources,
                const CudaDeviceContext &context) {
    int local_gpu_count = device_resources->size();

    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      const auto &stream = (*device_resources)[id]->get_stream();
      const auto &top_grad = embedding_feature_tensors[id]->get_ptr();
      const auto &row_offset = row_offset_allreduce_tensors[id]->get_ptr();
      auto wgrad = wgrad_tensors[id]->get_ptr();

      try {
        dim3 blockSize(embedding_vec_size, 1,
                       1);  // each thread corresponds to one element in an embedding vetor
        dim3 gridSize(batch_size, 1, 1);  // each block corresponds to a sample

        if (combiner == 0)  // sum
        {
          backward_sum_kernel<TypeHashKey><<<gridSize, blockSize, 0, stream>>>(
              batch_size, slot_num, embedding_vec_size, top_grad, wgrad);
        } else if (combiner == 1)  // mean
        {
          backward_mean_kernel<<<gridSize, blockSize, 0, stream>>>(
              batch_size, slot_num, embedding_vec_size, row_offset, top_grad, wgrad);
        } else {
          CK_THROW_(Error_t::WrongInput, "Invalid combiner type ");
        }
      } catch (const std::runtime_error &rt_err) {
        std::cerr << rt_err.what() << std::endl;
        throw;
      }
    }

    return;
  }

  /**
   * backward propagation for LocalizedSlotSparseEmbeddingHash
   * The first step of backward propagation: computing the wgrad.
   * @param batch_size batch size for the current mini-batch computation.
   * @param slot_num_per_gpu slot_num per GPU.
   * @param embedding_vec_size embedding vector size.
   * @param combiner combiner type: 0-sum, 1-mean
   * @param row_offset_allreduce_tensors row_offsets tensors after all_reduce of mulitple GPUs
   * @param embedding_feature_tensors embedding features tensors of multiplu GPUs, storing dgrad
   * from the top layer
   * @param wgrad_tensors wgrad tensors of multi GPUs, the output of this function.
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  template <typename TypeHashKey, typename TypeEmbeddingComp>
  void backward(int batch_size, const std::vector<int> &slot_num_per_gpu, int embedding_vec_size,
                int combiner, const Tensors<TypeHashKey> &row_offset_allreduce_tensors,
                const Tensors<TypeEmbeddingComp> &embedding_feature_tensors,
                Tensors<TypeEmbeddingComp> &wgrad_tensors,
                const std::shared_ptr<GPUResourceGroup> &device_resources,
                const CudaDeviceContext &context) {
    int local_gpu_count = device_resources->size();

    // need to know the Type
    switch (sizeof(TypeEmbeddingComp)) {
      case 2:                                         // fp16
        embedding_vec_size = embedding_vec_size / 2;  // use __half2
        break;
      case 4:  // fp32
        embedding_vec_size = embedding_vec_size;
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
    }

    for (int id = 0; id < local_gpu_count; id++) {
      if (slot_num_per_gpu[id] == 0) {
        continue;
      }

      context.set_device((*device_resources)[id]->get_device_id());
      const auto &stream = (*device_resources)[id]->get_stream();
      const auto &top_grad = embedding_feature_tensors[id]->get_ptr();
      const auto &row_offset = row_offset_allreduce_tensors[id]->get_ptr();
      auto wgrad = wgrad_tensors[id]->get_ptr();

      try {
        dim3 blockSize(embedding_vec_size, 1,
                       1);  // each thread corresponds to one element in an embedding vetor
        dim3 gridSize(batch_size, 1, 1);  // each block corresponds to a sample

        if (combiner == 0)  // sum
        {
          backward_sum_kernel<TypeHashKey><<<gridSize, blockSize, 0, stream>>>(
              batch_size, slot_num_per_gpu[id], embedding_vec_size, top_grad, wgrad);
        } else if (combiner == 1)  // mean
        {
          backward_mean_kernel<<<gridSize, blockSize, 0, stream>>>(
              batch_size, slot_num_per_gpu[id], embedding_vec_size, row_offset, top_grad, wgrad);
        } else {
          CK_THROW_(Error_t::WrongInput, "Invalid combiner type ");
        }
      } catch (const std::runtime_error &rt_err) {
        std::cerr << rt_err.what() << std::endl;
        throw;
      }
    }

    return;
  }

  /**
   * backward propagation for LocalizedSlotSparseEmbeddingOneHot.
   * fuse (backward_reorder + all2all + backward_xxx_kernel) into one kernel.
   * Only support single node currently.
   * @param batch_size batch size for the current mini-batch computation.
   * @param slot_num total slots number
   * @param slot_num_per_gpu the number of slots for each GPU
   * @param embedding_vec_size embedding vector size.
   * @param combiner 0-sum; 1-mean
   * @param embedding_features embedding features of all gpus (output)
   * @param wgrad_tensors wgrad tensors of multi GPUs, the output of this function.
   * @param device_resources device resources
   */
  template <typename TypeEmbeddingComp>
  void backward_fuse(size_t batch_size, int slot_num, const std::vector<int> &slot_num_per_gpu,
                     int embedding_vec_size, int combiner, TypeEmbeddingComp **embedding_features,
                     Tensors<TypeEmbeddingComp> &wgrad_tensors,
                     const std::shared_ptr<GPUResourceGroup> &device_resources) {
    int local_gpu_count = device_resources->size();
    size_t batch_size_per_gpu = batch_size / local_gpu_count;

    // need to know the Type
    switch (sizeof(TypeEmbeddingComp)) {
      case 2:                                         // fp16
        embedding_vec_size = embedding_vec_size / 2;  // use __half2
        break;
      case 4:  // fp32
        embedding_vec_size = embedding_vec_size;
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
    }

    dim3 blockSize(embedding_vec_size, 1, 1);
    int maxActiveBlocks;
    switch (sizeof(TypeEmbeddingComp)) {
      case 2:  // fp16
        CK_CUDA_THROW_(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks, backward_sum_fuse_kernel_fp16, blockSize.x, 0));
        break;
      case 4:  // fp32
        CK_CUDA_THROW_(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks, backward_sum_fuse_kernel_fp32, blockSize.x, 0));
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
    }
    dim3 gridSize(min((int)batch_size, (sm_count_ * maxActiveBlocks)), 1, 1);

    // std::cout << "batch_size=" << batch_size << ", sm_count=" << sm_count_
    //     << ", maxActiveBlocks=" << maxActiveBlocks << ", girdSize=" << gridSize.x
    //     << std::endl;

    struct cudaLaunchParams params[local_gpu_count];
    void *kargs[local_gpu_count][9];
    int id_list[local_gpu_count];
    int slot_num_list[local_gpu_count];
    TypeEmbeddingComp *wgrad_list[local_gpu_count];

    for (int id = 0; id < local_gpu_count; id++) {
      if (slot_num_per_gpu[id] == 0) {
        continue;
      }

      id_list[id] = (*device_resources)[id]->get_device_id();
      slot_num_list[id] = slot_num_per_gpu[id];
      wgrad_list[id] = wgrad_tensors[id]->get_ptr();

      switch (sizeof(TypeEmbeddingComp)) {
        case 2:  // fp16
          params[id].func = (void *)backward_sum_fuse_kernel_fp16;
          break;
        case 4:  // fp32
          params[id].func = (void *)backward_sum_fuse_kernel_fp32;
          break;
        default:
          CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
      }

      params[id].gridDim = gridSize;
      params[id].blockDim = blockSize;
      params[id].sharedMem = 0;
      params[id].stream = (*device_resources)[id]->get_stream();
      kargs[id][0] = (void *)&(id_list[id]);
      kargs[id][1] = (void *)&local_gpu_count;
      kargs[id][2] = (void *)&batch_size;
      kargs[id][3] = (void *)&batch_size_per_gpu;
      kargs[id][4] = (void *)&slot_num;
      kargs[id][5] = (void *)&(slot_num_list[id]);
      kargs[id][6] = (void *)&embedding_vec_size;
      kargs[id][7] = (void *)&embedding_features;
      kargs[id][8] = (void *)&(wgrad_list[id]);
      params[id].args = (void **)kargs[id];
    }

    try {
      if (combiner == 0) {
        CudaDeviceContext context((*device_resources)[0]->get_device_id());
        sync_all_gpus(device_resources, context);

        CK_CUDA_THROW_(
            cudaLaunchCooperativeKernelMultiDevice(params, local_gpu_count,
                                                   (cudaCooperativeLaunchMultiDeviceNoPreSync |
                                                    cudaCooperativeLaunchMultiDeviceNoPostSync)));
      } else {
        CK_THROW_(Error_t::WrongInput, "Invalid combiner type ");
      }
    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }

    return;
  }

  /**
   * The second step of backward propagation: update embedding tables(weights)
   * @param stream cuda stream corresponding to the current GPU.
   * @param batch_size batch size for the current mini-batch computation.
   * @param slot_num the number of slots in hash table.
   * @param embedding_vec_size embedding vector size.
   * @param max_vocabulary_size_per_gpu the max row number of hash table for each GPU.
   * @param opt_params optimizer params.
   * @param nnz non-zero feature number in one batch
   * @param row_offset the pointer of row_offset
   * @param hash_value_index the pointer of hash value_index
   * @param sample_id the pointer of sample ids
   * @param sample_id_sort the pointer of sorted sample ids
   * @param hash_value_index_sort the pointer of sorted hash table value_index
   * @param hash_value_index_count the pointer of the count of each hash value_index
   * @param hash_value_index_count_offset the pointer of the offset for each count of hash
   * value_index
   * @param hash_value_index_count_counter the pointer of the counter of hash value_index count
   * @param temp_storage_sort the pointer of the temp buffer for the CUB lib sorting API
   * @param temp_storage_sort_bytes the bytes of the temp buffer for the CUB lib sorting API
   * @param temp_storage_scan the pointer of the temp buffer for the CUB lib scaning API
   * @param temp_storage_scan_bytes the bytes of the temp buffer for the CUB lib scaning API
   * @param wgrad the pointer of wgrad
   * @param deltaw_hash_value_index the pointer of deltaw's corresponding hash value_index
   * @param deltaw the pointer of deltaw, which is used to update the hash table value
   * @param hash_table_value the pointer of hash table value, which will be updated
   */
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void update_params(cudaStream_t stream, int batch_size, int slot_num, int embedding_vec_size,
                     size_t max_vocabulary_size_per_gpu, OptParams<float> &opt_params, int nnz,
                     const TypeHashKey *row_offset, TypeHashValueIndex *hash_value_index,
                     TypeHashKey *sample_id, TypeHashKey *sample_id_sort,
                     TypeHashValueIndex *hash_value_index_sort,
                     // uint32_t *hash_value_index_count,
                     uint32_t *hash_value_index_count_offset, uint32_t *new_hash_value_flag,
                     uint32_t *hash_value_flag_sumed, uint32_t *hash_value_index_count_counter,
                     void *temp_storage_sort, size_t temp_storage_sort_bytes,
                     void *temp_storage_scan, size_t temp_storage_scan_bytes, const float *wgrad,
                     TypeHashValueIndex *deltaw_hash_value_index, float *deltaw,
                     float *hash_table_value) {
    if (slot_num == 0) {
      return;
    }

    try {
      // step1: expand sample IDs
      dim3 blockSize(64, 1, 1);
      dim3 gridSize((batch_size * slot_num + blockSize.x - 1) / blockSize.x, 1, 1);
      sample_id_expand_kernel<<<gridSize, blockSize, 0, stream>>>(batch_size, slot_num, row_offset,
                                                                  sample_id);

#ifdef SGD_ATOMIC
      if (opt_params.optimizer == Optimizer_t::SGD) {  // for SGD, do atomic update
        dim3 gridSize(min(max(1, nnz), sm_count_ * 32), 1, 1);
        dim3 blockSize(embedding_vec_size, 1, 1);

        float lr_scale = (float)opt_params.lr / (float)opt_params.scaler;
        opt_sgd_atomic_kernel<<<gridSize, blockSize, 0, stream>>>(nnz, embedding_vec_size, lr_scale,
                                                                  hash_value_index, sample_id,
                                                                  wgrad, hash_table_value);
      } else
#endif
      {
        // step3: sort by hash_value_index
        int end_bit = (int)log2((float)max_vocabulary_size_per_gpu) + 1;
        CK_CUDA_THROW_(cub::DeviceRadixSort::SortPairs(
            (void *)temp_storage_sort, temp_storage_sort_bytes, hash_value_index,
            hash_value_index_sort, sample_id, sample_id_sort, nnz, 0, end_bit, stream, false));

        // step4: count the number for each unduplicated hash_value_index
        CK_CUDA_THROW_(
            cudaMemsetAsync(hash_value_index_count_counter, 0, sizeof(uint32_t), stream));
        blockSize.x = 256;
        const int target_grid_size = (nnz + (blockSize.x - 1)) / blockSize.x;
        const int MAX_GRID = 384;
        gridSize.x = target_grid_size < MAX_GRID ? target_grid_size : MAX_GRID;
        value_count_kernel_1<<<gridSize, blockSize, 0, stream>>>(nnz, hash_value_index_sort,
                                                                 new_hash_value_flag);

        // prefix_sum
        CK_CUDA_THROW_(cub::DeviceScan::InclusiveSum((void *)temp_storage_scan,
                                                     temp_storage_scan_bytes, new_hash_value_flag,
                                                     hash_value_flag_sumed, nnz, stream));

        value_count_kernel_2<<<gridSize, blockSize, 0, stream>>>(
            nnz, new_hash_value_flag, hash_value_flag_sumed, hash_value_index_count_offset,
            hash_value_index_count_counter);

        uint32_t hash_hash_value_index_count_num = 0;
        // this async memcpy will not perform as a async operation because the host memory is not a
        // pinned memroy
        CK_CUDA_THROW_(cudaMemcpyAsync(&hash_hash_value_index_count_num,
                                       hash_value_index_count_counter, sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, stream));

        // step5: use optimizer method to compute deltaw, and record corresponding
        // deltaw_hash_value_index
        blockSize.x = embedding_vec_size;
        gridSize.x = max(1, hash_hash_value_index_count_num);

        if (opt_params.global_update) {
          switch (opt_params.optimizer) {
            case Optimizer_t::Adam:  // adam
              opt_params.hyperparams.adam.alpha_t =
                  opt_params.lr *
                  sqrt(1 -
                       pow(opt_params.hyperparams.adam.beta2, opt_params.hyperparams.adam.times)) /
                  (1 - pow(opt_params.hyperparams.adam.beta1, opt_params.hyperparams.adam.times));
              // update target mi and vi
              opt_adam_kernel_global<<<gridSize, blockSize, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.hyperparams.adam,
                  sample_id_sort, hash_value_index_sort, hash_value_index_count_offset, wgrad,
                  opt_params.scaler);
              // all update according to the mi vi
              adam_update_kernel_global<<<1024, 256, 0, stream>>>(
                  embedding_vec_size, max_vocabulary_size_per_gpu, opt_params.hyperparams.adam,
                  hash_table_value);
              break;
            case Optimizer_t::MomentumSGD:  // momentum sgd
              opt_momentum_sgd_kernel_global<<<gridSize, blockSize, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                  opt_params.hyperparams.momentum, sample_id_sort, hash_value_index_sort,
                  hash_value_index_count_offset, wgrad, opt_params.scaler);
              momentum_sgd_update_kernel_global<<<1024, 256, 0, stream>>>(
                  embedding_vec_size, max_vocabulary_size_per_gpu, opt_params.hyperparams.momentum,
                  hash_table_value);
              break;
            case Optimizer_t::Nesterov:  // nesterov
              nesterov_global_update_kernel_global<<<1024, 256, 0, stream>>>(
                  embedding_vec_size, max_vocabulary_size_per_gpu, opt_params.hyperparams.nesterov,
                  hash_table_value);
              nesterov_local_update_kernel_global<<<gridSize, blockSize, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                  opt_params.hyperparams.nesterov, sample_id_sort, hash_value_index_sort,
                  hash_value_index_count_offset, wgrad, hash_table_value, opt_params.scaler);
              break;
#ifndef SGD_ATOMIC
            case Optimizer_t::SGD:
              opt_sgd_kernel_global<<<gridSize, blockSize, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                  sample_id_sort, hash_value_index_sort, hash_value_index_count_offset, wgrad,
                  hash_table_value, opt_params.scaler);
              break;
#endif
            default:
              CK_THROW_(Error_t::WrongInput, "Error: Invalid opitimizer type");
          }
        } else {
          switch (opt_params.optimizer) {
            case Optimizer_t::Adam:  // adam
              opt_params.hyperparams.adam.alpha_t =
                  opt_params.lr *
                  sqrt(1 -
                       pow(opt_params.hyperparams.adam.beta2, opt_params.hyperparams.adam.times)) /
                  (1 - pow(opt_params.hyperparams.adam.beta1, opt_params.hyperparams.adam.times));

              opt_adam_kernel<<<gridSize, blockSize, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.hyperparams.adam,
                  sample_id_sort, hash_value_index_sort, hash_value_index_count_offset, wgrad,
                  deltaw_hash_value_index, (float *)deltaw, opt_params.scaler);
              break;
            case Optimizer_t::MomentumSGD:  // momentum sgd
              opt_momentum_sgd_kernel<<<gridSize, blockSize, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                  opt_params.hyperparams.momentum, sample_id_sort, hash_value_index_sort,
                  hash_value_index_count_offset, wgrad, deltaw_hash_value_index, (float *)deltaw,
                  opt_params.scaler);
              break;
            case Optimizer_t::Nesterov:  // nesterov
              opt_nesterov_kernel<<<gridSize, blockSize, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                  opt_params.hyperparams.nesterov, sample_id_sort, hash_value_index_sort,
                  hash_value_index_count_offset, wgrad, deltaw_hash_value_index, (float *)deltaw,
                  opt_params.scaler);
              break;
#ifndef SGD_ATOMIC
            case Optimizer_t::SGD:
              opt_sgd_kernel<<<gridSize, blockSize, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                  sample_id_sort, hash_value_index_sort, hash_value_index_count_offset, wgrad,
                  deltaw_hash_value_index, (float *)deltaw, opt_params.scaler);
              break;
#endif
            default:
              CK_THROW_(Error_t::WrongInput, "Error: Invalid opitimizer type");
          }

          // step6: update hash_table_value by deltaw
          blockSize.x = embedding_vec_size;
          gridSize.x = max(1, hash_hash_value_index_count_num);
          update_kernel<TypeHashValueIndex><<<gridSize, blockSize, 0, stream>>>(
              hash_hash_value_index_count_num, embedding_vec_size, deltaw_hash_value_index, deltaw,
              hash_table_value);
        }  // else

      }  // else

    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }

    return;
  }

  /**
   * overload for fp16. Only support atmoic SGD currently.
   * The second step of backward propagation: update embedding tables(weights)
   * @param stream cuda stream corresponding to the current GPU.
   * @param batch_size batch size for the current mini-batch computation.
   * @param slot_num the number of slots for the current gpu.
   * @param embedding_vec_size embedding vector size.
   * @param max_vocabulary_size_per_gpu the max row number of hash table for each GPU.
   * @param opt_params optimizer params.
   * @param nnz non-zero feature number in one batch
   * @param row_offset the pointer of row_offset
   * @param hash_value_index the pointer of hash value_index
   * @param sample_id the pointer of sample ids
   * @param sample_id_sort the pointer of sorted sample ids
   * @param hash_value_index_sort the pointer of sorted hash table value_index
   * @param hash_value_index_count the pointer of the count of each hash value_index
   * @param hash_value_index_count_offset the pointer of the offset for each count of hash
   * value_index
   * @param hash_value_index_count_counter the pointer of the counter of hash value_index count
   * @param temp_storage_sort the pointer of the temp buffer for the CUB lib sorting API
   * @param temp_storage_sort_bytes the bytes of the temp buffer for the CUB lib sorting API
   * @param temp_storage_scan the pointer of the temp buffer for the CUB lib scaning API
   * @param temp_storage_scan_bytes the bytes of the temp buffer for the CUB lib scaning API
   * @param wgrad the pointer of wgrad
   * @param deltaw_hash_value_index the pointer of deltaw's corresponding hash value_index
   * @param deltaw the pointer of deltaw, which is used to update the hash table value
   * @param hash_table_value the pointer of hash table value, which will be updated
   */
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void update_params(cudaStream_t stream, int batch_size, int slot_num, int embedding_vec_size,
                     size_t max_vocabulary_size_per_gpu, OptParams<__half> &opt_params, int nnz,
                     const TypeHashKey *row_offset, TypeHashValueIndex *hash_value_index,
                     TypeHashKey *sample_id, TypeHashKey *sample_id_sort,
                     TypeHashValueIndex *hash_value_index_sort,
                     // uint32_t *hash_value_index_count,
                     uint32_t *hash_value_index_count_offset, uint32_t *new_hash_value_flag,
                     uint32_t *hash_value_flag_sumed, uint32_t *hash_value_index_count_counter,
                     void *temp_storage_sort, size_t temp_storage_sort_bytes,
                     void *temp_storage_scan, size_t temp_storage_scan_bytes, const __half *wgrad,
                     TypeHashValueIndex *deltaw_hash_value_index, __half *deltaw,
                     float *hash_table_value) {
    if (slot_num == 0) {
      return;
    }

    try {
      // step1: expand sample IDs
      {
        dim3 blockSize(64, 1, 1);
        dim3 gridSize((batch_size * slot_num + blockSize.x - 1) / blockSize.x, 1, 1);
        sample_id_expand_kernel<<<gridSize, blockSize, 0, stream>>>(batch_size, slot_num,
                                                                    row_offset, sample_id);
      }

#ifdef SGD_ATOMIC
      if (opt_params.optimizer == Optimizer_t::SGD) {  // for SGD, do atomic update
        dim3 gridSize(min(max(1, nnz), sm_count_ * 32), 1, 1);
        dim3 blockSize(embedding_vec_size, 1, 1);

        float lr_scale = (float)opt_params.lr / (float)opt_params.scaler;

        opt_sgd_atomic_kernel<<<gridSize, blockSize, 0, stream>>>(nnz, embedding_vec_size, lr_scale,
                                                                  hash_value_index, sample_id,
                                                                  wgrad, hash_table_value);
      } else {
        CK_THROW_(Error_t::WrongInput, "Error: Invalid opitimizer type");
      }
#else
      CK_THROW_(Error_t::WrongInput, "Error: Invalid opitimizer/method pattern");
#endif

    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }

    return;
  }

  /**
   * update_params for LocalizedSlotSparseEmbeddingOneHot.
   * overload for fp16. Only support atmoic SGD currently.
   * The second step of backward propagation: update embedding tables(weights)
   * @param stream cuda stream corresponding to the current GPU.
   * @param embedding_vec_size embedding vector size.
   * @param opt_params optimizer params.
   * @param nnz non-zero feature number in one batch
   * @param hash_value_index the pointer of hash value_index
   * @param wgrad the pointer of wgrad
   * @param hash_table_value the pointer of hash table value, which will be updated
   */
  template <typename TypeHashValueIndex, typename TypeEmbeddingComp>
  void update_params(cudaStream_t stream, int embedding_vec_size,
                     const OptParams<TypeEmbeddingComp> &opt_params, int nnz,
                     const TypeHashValueIndex *hash_value_index, const TypeEmbeddingComp *wgrad,
                     float *hash_table_value) {
    try {
#ifdef SGD_ATOMIC
      if (opt_params.optimizer == Optimizer_t::SGD) {  // for SGD, do atomic update
        dim3 gridSize(min(max(1, nnz), sm_count_ * 32), 1, 1);
        dim3 blockSize(embedding_vec_size, 1, 1);

        float lr_scale = (float)opt_params.lr / (float)opt_params.scaler;

        // for one-hot, the sample_id is dedicated.
        opt_sgd_atomic_kernel<<<gridSize, blockSize, 0, stream>>>(
            nnz, embedding_vec_size, lr_scale, hash_value_index, wgrad, hash_table_value);
      } else {
        CK_THROW_(Error_t::WrongInput, "Error: Invalid opitimizer type");
      }
#else
      CK_THROW_(Error_t::WrongInput, "Error: Invalid opitimizer/method pattern");
#endif

    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }

    return;
  }

  /**
   * collection communication: reduce_scatter f or DistributedSlotSparseEmbeddingHash
   * @param recv_count the count of elements will be received.
   * @param send_tensors the send tensors of multi GPUs.
   * @param recv_tensors the recv tensors of multi GPUs.
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device.
   */
  template <typename Type>
  void reduce_scatter(int recv_count, const Tensors<Type> &send_tensors,
                      Tensors<Type> &recv_tensors,
                      const std::shared_ptr<GPUResourceGroup> &device_resources,
                      const CudaDeviceContext &context) {
    int local_gpu_count = device_resources->size();

    // need to know the type of TypeHashKey here
    ncclDataType_t type;
    switch (sizeof(Type)) {
      case 2:
        type = ncclHalf;
        break;
      case 4:
        type = ncclFloat;
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: TypeHashKey not support by now");
    }

    CK_NCCL_THROW_(ncclGroupStart());
    for (int id = 0; id < local_gpu_count; id++) {
      CK_NCCL_THROW_(ncclReduceScatter(send_tensors[id]->get_ptr(),  // send buf
                                       recv_tensors[id]->get_ptr(),  // recv buff
                                       recv_count, type, ncclSum,
                                       *(*device_resources)[id]->get_nccl_ptr(),
                                       (*device_resources)[id]->get_stream()));
    }
    CK_NCCL_THROW_(ncclGroupEnd());

    return;
  }

  /**
   * collection communication: all_reduce.
   * @param send_count the count of elements will be sent.
   * @param send_tensors the send tensors of multi GPUs.
   * @param recv_tensors the recv tensors of multi GPUs.
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device.
   */
  template <typename Type>
  void all_reduce(int send_count, const Tensors<Type> &send_tensors, Tensors<Type> &recv_tensors,
                  const std::shared_ptr<GPUResourceGroup> &device_resources,
                  const CudaDeviceContext &context) {
    int local_gpu_count = device_resources->size();

    // need to know the type of Type here
    ncclDataType_t type;
    switch (sizeof(Type)) {
      case 4:
        type = ncclUint32;
        break;
      case 8:
        type = ncclUint64;
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
    }

    CK_NCCL_THROW_(ncclGroupStart());
    for (int id = 0; id < local_gpu_count; id++) {
      CK_NCCL_THROW_(ncclAllReduce(
          send_tensors[id]->get_ptr(), recv_tensors[id]->get_ptr(), send_count, type, ncclSum,
          *(*device_resources)[id]->get_nccl_ptr(), (*device_resources)[id]->get_stream()));
    }
    CK_NCCL_THROW_(ncclGroupEnd());

    return;
  }

  /**
   * collection communication: all_gather.
   * @param send_count the count of elements will be sent.
   * @param send_tensors the send tensors of multi GPUs.
   * @param recv_tensors the recv tensors of multi GPUs.
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device.
   */
  template <typename Type>
  void all_gather(int send_count, const Tensors<Type> &send_tensors, Tensors<Type> &recv_tensors,
                  const std::shared_ptr<GPUResourceGroup> &device_resources,
                  const CudaDeviceContext &context) {
    int local_gpu_count = device_resources->size();

    // need to know the Type
    ncclDataType_t type;
    switch (sizeof(Type)) {
      case 2:
        type = ncclHalf;
        break;
      case 4:
        type = ncclFloat;
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
    }

    CK_NCCL_THROW_(ncclGroupStart());
    for (int id = 0; id < local_gpu_count; id++) {
      CK_NCCL_THROW_(ncclAllGather(send_tensors[id]->get_ptr(),  // send buff
                                   recv_tensors[id]->get_ptr(),  // recv buff
                                   send_count, type, *(*device_resources)[id]->get_nccl_ptr(),
                                   (*device_resources)[id]->get_stream()));
    }
    CK_NCCL_THROW_(ncclGroupEnd());

    return;
  }

#ifdef NCCL_A2A  // use nccl all2all (currently, it only supports intra-node)

  /**
   * nccl all2all communication for forward.
   * CAUSION: Only support intra-node all2all currently
   * @param batch_size_per_gpu batch size per GPU
   * @param slot_num_per_gpu slot number for each local GPU
   * @param embedding_vec_size embedding vector size
   * @param send_tensors the send tensors of multi GPUs.
   * @param recv_tensors the recv tensors of multi GPUs.
   * @param device_resources all gpus device resources.
   */
  template <typename Type>
  void all2all_forward(int batch_size_per_gpu, const std::vector<int> &slot_num_per_gpu,
                       int embedding_vec_size, const Tensors<Type> &send_tensors,
                       Tensors<Type> &recv_tensors,
                       const std::shared_ptr<GPUResourceGroup> &device_resources) {
    std::vector<int> device_list = device_resources->get_device_list();
    int local_gpu_count = (int)device_list.size();

    // Fill in partition table, ith Topo GPU to jth Topo GPU
    std::vector<std::vector<size_t>> table(local_gpu_count, std::vector<size_t>(local_gpu_count));
    for (int i = 0; i < local_gpu_count; i++) {
      size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu[i] * embedding_vec_size;
      for (int j = 0; j < local_gpu_count; j++) {
        table[i][j] = element_per_send;
      }
    }

#ifndef NDEBUG
    std::cout << "nccl all2all forward table:" << std::endl;
    for (int i = 0; i < local_gpu_count; i++) {
      for (int j = 0; j < local_gpu_count; j++) {
        std::cout << table[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
#endif

    std::vector<Type *> src(local_gpu_count);
    std::vector<Type *> dst(local_gpu_count);
    for (int id = 0; id < local_gpu_count; id++) {
      src[id] = send_tensors[id]->get_ptr();
      dst[id] = recv_tensors[id]->get_ptr();
    }
    std::vector<std::vector<Type *>> src_pos(local_gpu_count, std::vector<Type *>(local_gpu_count));
    std::vector<std::vector<Type *>> dst_pos(local_gpu_count, std::vector<Type *>(local_gpu_count));
    // Calculate the src offset pointer from each GPU to each other
    for (int i = 0; i < local_gpu_count; i++) {
      size_t src_offset = 0;
      for (int j = 0; j < local_gpu_count; j++) {
        src_pos[i][j] = src[i] + src_offset;
        src_offset += table[i][j];
      }
    }
    // Calculate the dst offset pointer from each GPU to each other
    for (int i = 0; i < local_gpu_count; i++) {
      size_t dst_offset = 0;
      for (int j = 0; j < local_gpu_count; j++) {
        dst_pos[i][j] = dst[i] + dst_offset;
        dst_offset += table[j][i];
      }
    }

#ifndef NDEBUG
    std::cout << "nccl all2all forward src_pos:" << std::endl;
    for (int i = 0; i < local_gpu_count; i++) {
      for (int j = 0; j < local_gpu_count; j++) {
        std::cout << src_pos[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "nccl all2all forward dst_pos:" << std::endl;
    for (int i = 0; i < local_gpu_count; i++) {
      for (int j = 0; j < local_gpu_count; j++) {
        std::cout << dst_pos[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
#endif

    // need to know the Type
    ncclDataType_t type;
    switch (sizeof(Type)) {
      case 2:
        type = ncclHalf;
        break;
      case 4:
        type = ncclFloat;
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
    }

    // Do the all2all transfer
    CK_NCCL_THROW_(ncclGroupStart());
    for (int i = 0; i < local_gpu_count; i++) {
      for (int j = 0; j < local_gpu_count; j++) {
        CK_NCCL_THROW_(ncclSend(src_pos[i][j], table[i][j], type, j,
                                *(*device_resources)[i]->get_nccl_ptr(),
                                (*device_resources)[i]->get_stream()));
        CK_NCCL_THROW_(ncclRecv(dst_pos[i][j], table[j][i], type, j,
                                *(*device_resources)[i]->get_nccl_ptr(),
                                (*device_resources)[i]->get_stream()));
      }
    }
    CK_NCCL_THROW_(ncclGroupEnd());

    return;
  }

  /**
   * nccl all2all communication for backward
   * CAUSION: Only support intra-node all2all currently
   * @param batch_size_per_gpu batch size per GPU
   * @param slot_num_per_gpu slot number for each local GPU
   * @param embedding_vec_size embedding vector size
   * @param send_tensors the send tensors of multi GPUs.
   * @param recv_tensors the recv tensors of multi GPUs.
   * @param device_resources all gpus device resources.
   */
  template <typename Type>
  void all2all_backward(int batch_size_per_gpu, const std::vector<int> &slot_num_per_gpu,
                        int embedding_vec_size, const Tensors<Type> &send_tensors,
                        Tensors<Type> &recv_tensors,
                        const std::shared_ptr<GPUResourceGroup> &device_resources) {
    std::vector<int> device_list = device_resources->get_device_list();
    int local_gpu_count = (int)device_list.size();

    // Fill in partition table, ith Topo GPU to jth Topo GPU
    std::vector<std::vector<size_t>> table(local_gpu_count, std::vector<size_t>(local_gpu_count));
    for (int i = 0; i < local_gpu_count; i++) {
      size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu[i] * embedding_vec_size;
      for (int j = 0; j < local_gpu_count; j++) {
        table[j][i] = element_per_send;
      }
    }

#ifndef NDEBUG
    std::cout << "nccl all2all backward table:" << std::endl;
    for (int i = 0; i < local_gpu_count; i++) {
      for (int j = 0; j < local_gpu_count; j++) {
        std::cout << table[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
#endif

    std::vector<Type *> src(local_gpu_count);
    std::vector<Type *> dst(local_gpu_count);
    for (int id = 0; id < local_gpu_count; id++) {
      src[id] = send_tensors[id]->get_ptr();
      dst[id] = recv_tensors[id]->get_ptr();
    }
    std::vector<std::vector<Type *>> src_pos(local_gpu_count, std::vector<Type *>(local_gpu_count));
    std::vector<std::vector<Type *>> dst_pos(local_gpu_count, std::vector<Type *>(local_gpu_count));
    // Calculate the src offset pointer from each GPU to each other
    for (int i = 0; i < local_gpu_count; i++) {
      size_t src_offset = 0;
      for (int j = 0; j < local_gpu_count; j++) {
        src_pos[i][j] = src[i] + src_offset;
        src_offset += table[i][j];
      }
    }
    // Calculate the dst offset pointer from each GPU to each other
    for (int i = 0; i < local_gpu_count; i++) {
      size_t dst_offset = 0;
      for (int j = 0; j < local_gpu_count; j++) {
        dst_pos[i][j] = dst[i] + dst_offset;
        dst_offset += table[j][i];
      }
    }

#ifndef NDEBUG
    std::cout << "nccl all2all backward src_pos:" << std::endl;
    for (int i = 0; i < local_gpu_count; i++) {
      for (int j = 0; j < local_gpu_count; j++) {
        std::cout << src_pos[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "nccl all2all backward dst_pos:" << std::endl;
    for (int i = 0; i < local_gpu_count; i++) {
      for (int j = 0; j < local_gpu_count; j++) {
        std::cout << dst_pos[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
#endif

    // need to know the Type
    ncclDataType_t type;
    switch (sizeof(Type)) {
      case 2:
        type = ncclHalf;
        break;
      case 4:
        type = ncclFloat;
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
    }

    // Do the all2all transfer
    CK_NCCL_THROW_(ncclGroupStart());
    for (int i = 0; i < local_gpu_count; i++) {
      for (int j = 0; j < local_gpu_count; j++) {
        CK_NCCL_THROW_(ncclSend(src_pos[i][j], table[i][j], type, j,
                                *(*device_resources)[i]->get_nccl_ptr(),
                                (*device_resources)[i]->get_stream()));
        CK_NCCL_THROW_(ncclRecv(dst_pos[i][j], table[j][i], type, j,
                                *(*device_resources)[i]->get_nccl_ptr(),
                                (*device_resources)[i]->get_stream()));
      }
    }
    CK_NCCL_THROW_(ncclGroupEnd());

    return;
  }

#else  // use gossip all2all

#ifndef ENABLE_MPI  // without MPI (only for single node)
  /**
   * the initialization of collection communication: all2all
   * @param all2all all2all handler
   * @param plan_file plan file that describe the topo of GPUs
   * @param batch_size_per_gpu batch size per GPU
   * @param slot_num_per_gpu slot number for each local GPU
   * @param embedding_vec_size embedding vector size
   * @param send_tensors the send tensors of multi GPUs.
   * @param recv_tensors the recv tensors of multi GPUs.
   * @param device_resources all gpus device resources.
   */
  template <typename Type>
  void all2all_init_forward(
      std::unique_ptr<FasterGossipComm::FasterGossipComm<
          Type, FasterGossipComm::FasterGossipCommAll2AllTraits<Type>>> &all2all,
      const std::string &plan_file, int batch_size_per_gpu,
      const std::vector<int> &slot_num_per_gpu, int embedding_vec_size,
      const Tensors<Type> &send_tensors, Tensors<Type> &recv_tensors,
      const std::shared_ptr<GPUResourceGroup> &device_resources) {
    using transfer_plan_t =
        typename FasterGossipComm::FasterGossipCommAll2AllTraits<Type>::transfer_plan_t;
    transfer_plan_t *transfer_plan = new transfer_plan_t(parse_plan(plan_file.c_str()));
    int plan_gpu_count = transfer_plan->num_gpus();  // total number of GPUs in current node

    std::vector<int> device_list = device_resources->get_device_list();
    int local_gpu_count = (int)device_list.size();
    if (local_gpu_count != plan_gpu_count) {
      std::cout << "local_gpu_count=" << local_gpu_count << ", plan_gpu_count=" << plan_gpu_count
                << std::endl;
      CK_THROW_(Error_t::WrongInput,
                "Error: the local device_list doesn't match all2all plan_file");
    }
    std::vector<gossip::gpu_id_t> device_ids(device_list.begin(), device_list.end());
#ifndef NDEBUG
    std::cout << "gpu device list: { ";
    for (auto dev : device_ids) {
      std::cout << dev << " ";
    }
    std::cout << "}" << std::endl;
#endif

    all2all = std::unique_ptr<FasterGossipComm::FasterGossipComm<
        Type, FasterGossipComm::FasterGossipCommAll2AllTraits<Type>>>(
        new FasterGossipComm::FasterGossipComm<
            Type, FasterGossipComm::FasterGossipCommAll2AllTraits<Type>>(plan_file, device_ids));
    // The all2all communication class

    std::vector<Type *> src(local_gpu_count);
    std::vector<Type *> dst(local_gpu_count);
    for (int id = 0; id < local_gpu_count; id++) {
      src[id] = send_tensors[id]->get_ptr();
      dst[id] = recv_tensors[id]->get_ptr();
    }

    // Fill in partition table, ith Topo GPU to jth Topo GPU
    std::vector<std::vector<size_t>> table(local_gpu_count, std::vector<size_t>(local_gpu_count));
    for (int i = 0; i < local_gpu_count; i++) {
      size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu[i] * embedding_vec_size;
      for (int j = 0; j < local_gpu_count; j++) {
        table[i][j] = element_per_send;
      }
    }

#ifndef NDEBUG
    std::cout << "gossip all2all forward table:" << std::endl;
    for (int i = 0; i < local_gpu_count; i++) {
      for (int j = 0; j < local_gpu_count; j++) {
        std::cout << table[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
#endif

    all2all->Initialize(src, dst, table);

    return;
  }

  /**
   * the initialization of collection communication: all2all
   * @param all2all all2all handler
   * @param plan_file plan file that describe the topo of GPUs
   * @param batch_size_per_gpu batch size per GPU
   * @param slot_num_per_gpu slot number for each local GPU
   * @param embedding_vec_size embedding vector size
   * @param send_tensors the send tensors of multi GPUs.
   * @param recv_tensors the recv tensors of multi GPUs.
   * @param device_resources all gpus device resources.
   */
  template <typename Type>
  void all2all_init_backward(
      std::unique_ptr<FasterGossipComm::FasterGossipComm<
          Type, FasterGossipComm::FasterGossipCommAll2AllTraits<Type>>> &all2all,
      const std::string &plan_file, int batch_size_per_gpu,
      const std::vector<int> &slot_num_per_gpu, int embedding_vec_size,
      const Tensors<Type> &send_tensors, Tensors<Type> &recv_tensors,
      const std::shared_ptr<GPUResourceGroup> &device_resources) {
    using transfer_plan_t =
        typename FasterGossipComm::FasterGossipCommAll2AllTraits<Type>::transfer_plan_t;
    transfer_plan_t *transfer_plan = new transfer_plan_t(parse_plan(plan_file.c_str()));
    int plan_gpu_count = transfer_plan->num_gpus();  // total number of GPUs in current node

    std::vector<int> device_list = device_resources->get_device_list();
    int local_gpu_count = (int)device_list.size();
    if (local_gpu_count != plan_gpu_count) {
      std::cout << "local_gpu_count=" << local_gpu_count << ", plan_gpu_count=" << plan_gpu_count
                << std::endl;
      CK_THROW_(Error_t::WrongInput,
                "Error: the local device_list doesn't match all2all plan_file");
    }
    std::vector<gossip::gpu_id_t> device_ids(device_list.begin(), device_list.end());
#ifndef NDEBUG
    std::cout << "gpu device list: { ";
    for (auto dev : device_ids) {
      std::cout << dev << " ";
    }
    std::cout << "}" << std::endl;
#endif

    all2all = std::unique_ptr<FasterGossipComm::FasterGossipComm<
        Type, FasterGossipComm::FasterGossipCommAll2AllTraits<Type>>>(
        new FasterGossipComm::FasterGossipComm<
            Type, FasterGossipComm::FasterGossipCommAll2AllTraits<Type>>(plan_file, device_ids));
    // The all2all communication class

    std::vector<Type *> src(local_gpu_count);
    std::vector<Type *> dst(local_gpu_count);
    for (int id = 0; id < local_gpu_count; id++) {
      src[id] = send_tensors[id]->get_ptr();
      dst[id] = recv_tensors[id]->get_ptr();
    }

    // Fill in partition table, ith Topo GPU to jth Topo GPU
    std::vector<std::vector<size_t>> table(local_gpu_count, std::vector<size_t>(local_gpu_count));
    for (int i = 0; i < local_gpu_count; i++) {
      size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu[i] * embedding_vec_size;
      for (int j = 0; j < local_gpu_count; j++) {
        table[j][i] = element_per_send;
      }
    }

#ifndef NDEBUG
    std::cout << "gossip all2all backward table:" << std::endl;
    for (int i = 0; i < local_gpu_count; i++) {
      for (int j = 0; j < local_gpu_count; j++) {
        std::cout << table[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
#endif

    all2all->Initialize(src, dst, table);

    return;
  }

  /**
   * collection communication: all2all
   * @param all2all all2all handler
   */
  template <typename Type>
  void all2all_exec(const std::unique_ptr<FasterGossipComm::FasterGossipComm<
                        Type, FasterGossipComm::FasterGossipCommAll2AllTraits<Type>>> &all2all) {
    all2all->execAsync();
    all2all->sync();

    return;
  }

#else  // for mpirun (for single node or multiple node)

  /**
   * the initialization of collection communication: all2all
   * @param all2all all2all handler
   * @param plan_file plan file which demonstrates gpu topo
   * @param batch_size_per_gpu batch size per GPU
   * @param slot_num slot number
   * @param embedding_vec_size embedding vector size
   * @param send_tensors the send tensors of multi GPUs.
   * @param recv_tensors the recv tensors of multi GPUs.
   * @param device_resources all gpus device resources.
   */
  template <typename Type>
  void all2all_init_forward(
      std::unique_ptr<FasterGossipCommMulti::FasterGossipCommMulti<
          Type, FasterGossipCommMulti::FasterGossipCommMultiAll2AllTraits<Type>>> &all2all,
      const std::string &plan_file, int batch_size_per_gpu, int slot_num, int embedding_vec_size,
      const Tensors<Type> &send_tensors, Tensors<Type> &recv_tensors,
      const std::shared_ptr<GPUResourceGroup> &device_resources) {
    using transfer_plan_t =
        typename FasterGossipCommMulti::FasterGossipCommMultiAll2AllTraits<Type>::transfer_plan_t;
    transfer_plan_t *transfer_plan = new transfer_plan_t(parse_plan(plan_file.c_str()));
    int plan_gpu_count = transfer_plan->num_gpus();  // total number of GPUs in current node
    std::vector<int> device_list = device_resources->get_device_list();
    int local_gpu_count = (int)device_list.size();
    int total_gpu_count = device_resources->get_total_gpu_count();
    if (local_gpu_count != plan_gpu_count) {
      std::cout << "local_gpu_count=" << local_gpu_count << ", plan_gpu_count=" << plan_gpu_count
                << std::endl;
      CK_THROW_(Error_t::WrongInput,
                "Error: the local device_list doesn't match all2all plan_file");
    }

    int total_rank = 1;
    int my_rank = 0;
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &total_rank));
    int num_proc = device_resources->get_node_count();
    if (num_proc != total_rank) {
      CK_THROW_(Error_t::WrongInput, "Error: the MPI total rank doesn't match the node count");
    }
    if (total_gpu_count != (total_rank * local_gpu_count)) {
      CK_THROW_(Error_t::WrongInput, "Error: the total gpu count doesn't match");
    }

#ifndef NDEBUG
    std::cout << "total_rank=" << total_rank << ", my_rank=" << my_rank
              << ", total_gpu_count=" << total_gpu_count << ", local_gpu_count=" << local_gpu_count
              << std::endl;
#endif

    std::vector<gossip::gpu_id_t> device_ids(device_list.begin(), device_list.end());

#ifndef NDEBUG
    std::cout << "my_rank=" << my_rank << ", gpu device ids: { ";
    for (auto dev : device_ids) {
      std::cout << dev << " ";
    }
    std::cout << "}, gpu global ids: {";
    for (auto dev : device_ids) {
      std::cout << device_resources->get_global_id(dev) << " ";
    }
    std::cout << "}, gpu local ids: {";
    for (auto dev : device_ids) {
      std::cout << device_resources->get_local_id(device_resources->get_global_id(dev)) << " ";
    }
    std::cout << "}" << std::endl;
#endif

    // The all2all communication class
    all2all = std::unique_ptr<FasterGossipCommMulti::FasterGossipCommMulti<
        Type, FasterGossipCommMulti::FasterGossipCommMultiAll2AllTraits<Type>>>(
        new FasterGossipCommMulti::FasterGossipCommMulti<
            Type, FasterGossipCommMulti::FasterGossipCommMultiAll2AllTraits<Type>>(
            plan_file, device_ids, num_proc, my_rank, MPI_COMM_WORLD));

    std::vector<Type *> src(local_gpu_count);
    std::vector<Type *> dst(local_gpu_count);
    for (int id = 0; id < local_gpu_count; id++) {
      src[id] = send_tensors[id]->get_ptr();
      dst[id] = recv_tensors[id]->get_ptr();
    }

    std::vector<std::vector<size_t>> send_table(local_gpu_count,
                                                std::vector<size_t>(total_gpu_count));
    std::vector<std::vector<size_t>> recv_table(local_gpu_count,
                                                std::vector<size_t>(total_gpu_count));

    // Fill in sending partition table, ith Topo GPU send to jth global GPU
    for (int i = 0; i < local_gpu_count; i++) {
      int device_id = (*device_resources)[i]->get_device_id();
      int global_id = device_resources->get_global_id(device_id);
      int slot_num_per_gpu =
          slot_num / total_gpu_count + ((global_id < (slot_num % total_gpu_count)) ? 1 : 0);
      size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu * embedding_vec_size;

      for (int j = 0; j < total_gpu_count; j++) {
        send_table[i][j] = element_per_send;
      }
    }

    // Fill in receiving partition table, ith Topo GPU receive from jth global GPU
    for (int j = 0; j < total_gpu_count; j++) {
      int global_id = j;
      int slot_num_per_gpu =
          slot_num / total_gpu_count + ((global_id < (slot_num % total_gpu_count)) ? 1 : 0);
      size_t element_per_recv = batch_size_per_gpu * slot_num_per_gpu * embedding_vec_size;

      for (int i = 0; i < local_gpu_count; i++) {
        recv_table[i][j] = element_per_recv;
      }
    }

#ifndef NDEBUG
    std::cout << "my_rank=" << my_rank << ", gossip all2all forward send_table:" << std::endl;
    for (int i = 0; i < local_gpu_count; i++) {
      for (int j = 0; j < total_gpu_count; j++) {
        std::cout << send_table[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "my_rank=" << my_rank << ", gossip all2all forward recv_table:" << std::endl;
    for (int i = 0; i < local_gpu_count; i++) {
      for (int j = 0; j < total_gpu_count; j++) {
        std::cout << recv_table[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
#endif

    all2all->Initialize(src, dst, send_table, recv_table);

    return;
  }

  /**
   * the initialization of collection communication: all2all
   * @param all2all all2all handler
   * @param plan_file plan file which demonstrates gpu topo
   * @param batch_size_per_gpu batch size per GPU
   * @param slot_num slot number
   * @param embedding_vec_size embedding vector size
   * @param send_tensors the send tensors of multi GPUs.
   * @param recv_tensors the recv tensors of multi GPUs.
   * @param device_resources all gpus device resources.
   */
  template <typename Type>
  void all2all_init_backward(
      std::unique_ptr<FasterGossipCommMulti::FasterGossipCommMulti<
          Type, FasterGossipCommMulti::FasterGossipCommMultiAll2AllTraits<Type>>> &all2all,
      const std::string &plan_file, int batch_size_per_gpu, int slot_num, int embedding_vec_size,
      const Tensors<Type> &send_tensors, Tensors<Type> &recv_tensors,
      const std::shared_ptr<GPUResourceGroup> &device_resources) {
    using transfer_plan_t =
        typename FasterGossipCommMulti::FasterGossipCommMultiAll2AllTraits<Type>::transfer_plan_t;
    transfer_plan_t *transfer_plan = new transfer_plan_t(parse_plan(plan_file.c_str()));
    int plan_gpu_count = transfer_plan->num_gpus();  // total number of GPUs in current node
    std::vector<int> device_list = device_resources->get_device_list();
    int local_gpu_count = (int)device_list.size();
    int total_gpu_count = device_resources->get_total_gpu_count();
    if (local_gpu_count != plan_gpu_count) {
      std::cout << "local_gpu_count=" << local_gpu_count << ", plan_gpu_count=" << plan_gpu_count
                << std::endl;
      CK_THROW_(Error_t::WrongInput,
                "Error: the local device_list doesn't match all2all plan_file");
    }

    int total_rank = 1;
    int my_rank = 0;
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &total_rank));
    int num_proc = device_resources->get_node_count();
    if (num_proc != total_rank) {
      CK_THROW_(Error_t::WrongInput, "Error: the MPI total rank doesn't match the node count");
    }
    if (total_gpu_count != (total_rank * local_gpu_count)) {
      CK_THROW_(Error_t::WrongInput, "Error: the total gpu count doesn't match");
    }
#ifndef NDEBUG
    std::cout << "total_rank=" << total_rank << ", my_rank=" << my_rank
              << ", total_gpu_count=" << total_gpu_count << ", local_gpu_count=" << local_gpu_count
              << std::endl;
#endif

    std::vector<gossip::gpu_id_t> device_ids(device_list.begin(), device_list.end());
#ifndef NDEBUG
    std::cout << "gpu device list: { ";
    for (auto dev : device_ids) {
      std::cout << dev << " ";
    }
    std::cout << "}" << std::endl;
#endif

    // The all2all communication class
    all2all = std::unique_ptr<FasterGossipCommMulti::FasterGossipCommMulti<
        Type, FasterGossipCommMulti::FasterGossipCommMultiAll2AllTraits<Type>>>(
        new FasterGossipCommMulti::FasterGossipCommMulti<
            Type, FasterGossipCommMulti::FasterGossipCommMultiAll2AllTraits<Type>>(
            plan_file, device_ids, num_proc, my_rank, MPI_COMM_WORLD));

    std::vector<Type *> src(local_gpu_count);
    std::vector<Type *> dst(local_gpu_count);
    for (int id = 0; id < local_gpu_count; id++) {
      src[id] = send_tensors[id]->get_ptr();
      dst[id] = recv_tensors[id]->get_ptr();
    }

    std::vector<std::vector<size_t>> send_table(local_gpu_count,
                                                std::vector<size_t>(total_gpu_count));
    std::vector<std::vector<size_t>> recv_table(local_gpu_count,
                                                std::vector<size_t>(total_gpu_count));

    // Fill in receiving partition table, ith Topo GPU receive from jth global GPU
    for (int i = 0; i < local_gpu_count; i++) {
      int device_id = (*device_resources)[i]->get_device_id();
      int global_id = device_resources->get_global_id(device_id);
      int slot_num_per_gpu =
          slot_num / total_gpu_count + ((global_id < (slot_num % total_gpu_count)) ? 1 : 0);
      size_t element_per_recv = batch_size_per_gpu * slot_num_per_gpu * embedding_vec_size;

      for (int j = 0; j < total_gpu_count; j++) {
        recv_table[i][j] = element_per_recv;
      }
    }

    // Fill in sending partition table, ith Topo GPU send to jth global GPU
    for (int j = 0; j < total_gpu_count; j++) {
      int global_id = j;
      int slot_num_per_gpu =
          slot_num / total_gpu_count + ((global_id < (slot_num % total_gpu_count)) ? 1 : 0);
      size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu * embedding_vec_size;

      for (int i = 0; i < local_gpu_count; i++) {
        send_table[i][j] = element_per_send;
      }
    }

#ifndef NDEBUG
    std::cout << "my_rank=" << my_rank << ", gossip all2all backward send_table:" << std::endl;
    for (int i = 0; i < local_gpu_count; i++) {
      for (int j = 0; j < total_gpu_count; j++) {
        std::cout << send_table[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "my_rank=" << my_rank << ", gossip all2all backward recv_table:" << std::endl;
    for (int i = 0; i < local_gpu_count; i++) {
      for (int j = 0; j < total_gpu_count; j++) {
        std::cout << recv_table[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
#endif

    all2all->Initialize(src, dst, send_table, recv_table);

    return;
  }

  /**
   * collection communication: all2all
   * @param all2all all2all handler
   */
  template <typename Type>
  void all2all_exec(
      const std::unique_ptr<FasterGossipCommMulti::FasterGossipCommMulti<
          Type, FasterGossipCommMulti::FasterGossipCommMultiAll2AllTraits<Type>>> &all2all) {
    all2all->exec();
    return;
  }

#endif

#endif

  /**
   * reoder the sequence of data after all2all operation in forward propagation
   * @param batch_size_per_gpu batch size per GPU
   * @param slot_num the number of localized slots
   * @param embedding_vec_size embedding vector size.
   * @param src_tensors the source tensors before reorder
   * @param dst_tensors the destination tensors after reorder
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device.
   */
  template <typename Type>
  void forward_reorder(int batch_size_per_gpu, int slot_num, int embedding_vec_size,
                       Tensors<Type> &src_tensors, Tensors<Type> &dst_tensors,
                       const std::shared_ptr<GPUResourceGroup> &device_resources,
                       const CudaDeviceContext &context) {
    int local_gpu_count = device_resources->size();
    int total_gpu_count = device_resources->get_total_gpu_count();

    // need to know the Type
    switch (sizeof(Type)) {
      case 2:                                         // fp16
        embedding_vec_size = embedding_vec_size / 2;  // use __half2
        break;
      case 4:  // fp32
        embedding_vec_size = embedding_vec_size;
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
    }

    dim3 blockSize(embedding_vec_size, 1, 1);
    dim3 gridSize(batch_size_per_gpu, 1, 1);

    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      forward_reorder_kernel<Type>
          <<<gridSize, blockSize, 0, (*device_resources)[id]->get_stream()>>>(
              batch_size_per_gpu, slot_num, embedding_vec_size, total_gpu_count,
              src_tensors[id]->get_ptr(), dst_tensors[id]->get_ptr());
    }
  }

  /**
   * reoder the sequence of data before all2all operation in backward propagation
   * @param batch_size_per_gpu batch size per GPU
   * @param slot_num the number of localized slots
   * @param embedding_vec_size embedding vector size.
   * @param src_tensors the source tensors before reorder
   * @param dst_tensors the destination tensors after reorder
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device.
   */
  template <typename Type>
  void backward_reorder(int batch_size_per_gpu, int slot_num, int embedding_vec_size,
                        Tensors<Type> &src_tensors, Tensors<Type> &dst_tensors,
                        const std::shared_ptr<GPUResourceGroup> &device_resources,
                        const CudaDeviceContext &context) {
    int local_gpu_count = device_resources->size();
    int total_gpu_count = device_resources->get_total_gpu_count();

    // need to know the Type
    switch (sizeof(Type)) {
      case 2:                                         // fp16
        embedding_vec_size = embedding_vec_size / 2;  // use __half2
        break;
      case 4:  // fp32
        embedding_vec_size = embedding_vec_size;
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
    }

    dim3 blockSize(embedding_vec_size, 1, 1);
    dim3 gridSize(batch_size_per_gpu, 1, 1);

    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      backward_reorder_kernel<Type>
          <<<gridSize, blockSize, 0, (*device_resources)[id]->get_stream()>>>(
              batch_size_per_gpu, slot_num, embedding_vec_size, total_gpu_count,
              src_tensors[id]->get_ptr(), dst_tensors[id]->get_ptr());
    }
  }

  /**
   * set liner data for a buffer
   * @param stream cuda stream.
   * @param data the pointer of the data buffer which will be written.
   * @param start_value the start value of the liner data.
   * @param stride_value the stride value of the liner data.
   * @param n the number of the data.
   */
  template <typename Type>
  void memset_liner(Type *data, Type start_value, Type stride_value, size_t n,
                    cudaStream_t stream) {
    try {
      int blockSize = 256;
      int gridSize = (n + blockSize - 1) / blockSize;

      memset_liner_kernel<Type>
          <<<gridSize, blockSize, 0, stream>>>(data, start_value, stride_value, n);
    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
  }

  /**
   * set constant data for a buffer
   * @param stream cuda stream.
   * @param data the pointer of the data buffer which will be written.
   * @param value the setting value
   * @param n the number of the data.
   */
  template <typename Type>
  void memset_const(Type *data, Type value, size_t n, cudaStream_t stream) {
    try {
      int blockSize = 256;
      int gridSize = (n + blockSize - 1) / blockSize;

      memset_const_kernel<Type><<<gridSize, blockSize, 0, stream>>>(data, value, n);
    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
  }

  /**
   * get hash table value by value_index
   * @param stream cuda stream.
   * @param count total count of value which will be get from hash table.
   * @param embedding_vec_size embedding vector size, each value has the dim of embedding_vec_size.
   * @param value_index the pointer of value_index.
   * @param hash_table_value the pointer of hash table value.
   * @param value_retrieved the pointer of the retrived value.
   */
  template <typename TypeHashValueIndex>
  void get_hash_value(cudaStream_t stream, size_t count, int embedding_vec_size,
                      const TypeHashValueIndex *value_index, const float *hash_table_value,
                      float *value_retrieved) {
    try {
      int blockSize = embedding_vec_size;
      int gridSize = count;

      get_hash_value_kernel<<<gridSize, blockSize, 0, stream>>>(
          count, embedding_vec_size, value_index, hash_table_value, value_retrieved);

    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
  }

  /**
   * get hash table slot_id by value_index
   * @param stream cuda stream.
   * @param count total count of value which will be get from hash table.
   * @param value_index the pointer of value_index.
   * @param hash_table_slot_id the pointer of hash table slot id.
   * @param slot_id the pointer of the retrieved slot_id.
   */
  template <typename TypeHashValueIndex>
  void get_hash_slot_id(cudaStream_t stream, size_t count, const TypeHashValueIndex *value_index,
                        const TypeHashValueIndex *hash_table_slot_id, TypeHashValueIndex *slot_id) {
    try {
      dim3 blockSize(64, 1, 1);
      dim3 gridSize((count + blockSize.x - 1) / blockSize.x, 1, 1);

      get_hash_slot_id_kernel<<<gridSize, blockSize, 0, stream>>>(count, value_index,
                                                                  hash_table_slot_id, slot_id);

    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
  }

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
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void upload_params_to_device(
      std::ifstream &weight_stream, size_t vocabulary_size, int embedding_vec_size,
      size_t max_vocabulary_size_per_gpu, Tensors<float> &hash_table_value_tensors,
      std::vector<std::shared_ptr<HashTable<TypeHashKey, TypeHashValueIndex,
                                            std::numeric_limits<TypeHashKey>::max()>>> &hash_tables,
      const std::shared_ptr<GPUResourceGroup> &device_resources, const CudaDeviceContext &context) {
    // check file size and vocabulary_size (file size <=　hash_table_size)
    weight_stream.seekg(0, weight_stream.end);
    size_t file_size_in_B = weight_stream.tellg();
    weight_stream.seekg(0, weight_stream.beg);
    size_t hash_table_size_in_B =
        vocabulary_size * ((size_t)embedding_vec_size * sizeof(float) +
                           sizeof(TypeHashKey));  // hash_key size + hash_value size
    if (file_size_in_B > hash_table_size_in_B) {
      CK_THROW_(Error_t::WrongInput,
                "Error: hash table file size is larger than hash table vocabulary_size");
    }

    int my_rank = 0;
#ifdef ENABLE_MPI
    int n_ranks = 1;
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));
#endif

    // define size
    int local_gpu_count = device_resources->size();
    int chunk_loop = 1000;
    int tile_size = 1;  // must be 1, because we need to cal (key&local_gpu_count) to decide gpu_id
                        // for each <key,value>
    int hash_table_key_tile_size = tile_size;
    int hash_table_key_tile_size_in_B = hash_table_key_tile_size * sizeof(TypeHashKey);
    int hash_table_key_chunk_size = hash_table_key_tile_size * chunk_loop;
    int hash_table_key_chunk_size_in_B = hash_table_key_chunk_size * sizeof(TypeHashKey);
    int hash_table_value_tile_size = tile_size * embedding_vec_size;
    int hash_table_value_tile_size_in_B = hash_table_value_tile_size * sizeof(float);
    int hash_table_value_chunk_size = hash_table_value_tile_size * chunk_loop;
    int hash_table_value_chunk_size_in_B = hash_table_value_chunk_size * sizeof(float);
    int hash_table_tile_size_in_B = hash_table_key_tile_size_in_B + hash_table_value_tile_size_in_B;
    int hash_table_chunk_size_in_B = hash_table_tile_size_in_B * chunk_loop;

    // CAUSION: can not decide how many values for each GPU, so need to allocate enough memory for
    // each GPU allocate GPU memory for hash_table_value_index
    std::unique_ptr<size_t[]> tile_counter_per_gpu(
        new size_t[local_gpu_count]);  // <= hash_table_value_index_per_gpu_size
    memset(tile_counter_per_gpu.get(), 0, sizeof(size_t) * local_gpu_count);
    std::unique_ptr<size_t[]> tile_counter_in_chunk_per_gpu(new size_t[local_gpu_count]);
    memset(tile_counter_in_chunk_per_gpu.get(), 0, sizeof(size_t) * local_gpu_count);
    std::unique_ptr<TypeHashKey *[]> d_hash_table_value_index_chunk_per_gpu(
        new TypeHashKey *[local_gpu_count]);
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      CK_CUDA_THROW_(
          cudaMalloc(&d_hash_table_value_index_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
      // initalize to zeros
      CK_CUDA_THROW_(cudaMemsetAsync(d_hash_table_value_index_chunk_per_gpu[id], 0,
                                     hash_table_key_chunk_size_in_B,
                                     (*device_resources)[id]->get_stream()));
    }

    // sync wait
    sync_all_gpus(device_resources, context);

    // CAUSION: can not decide how many values for each GPU, so need to allocate enough memory for
    // each GPU allocate CPU/GPU memory for hash_table/key/value chunk
    char *hash_table_chunk;
    CK_CUDA_THROW_(cudaMallocHost(&hash_table_chunk, hash_table_chunk_size_in_B));
    std::unique_ptr<TypeHashKey *[]> h_hash_table_key_chunk_per_gpu(
        new TypeHashKey *[local_gpu_count]);
    for (int id = 0; id < local_gpu_count; id++) {
      CK_CUDA_THROW_(
          cudaMallocHost(&h_hash_table_key_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
    }
    std::unique_ptr<TypeHashKey *[]> d_hash_table_key_chunk_per_gpu(
        new TypeHashKey *[local_gpu_count]);
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      CK_CUDA_THROW_(
          cudaMalloc(&d_hash_table_key_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
    }
    std::unique_ptr<float *[]> h_hash_table_value_chunk_per_gpu(new float *[local_gpu_count]);
    for (int id = 0; id < local_gpu_count; id++) {
      CK_CUDA_THROW_(
          cudaMallocHost(&h_hash_table_value_chunk_per_gpu[id], hash_table_value_chunk_size_in_B));
    }

    // do upload
    int loop_num = file_size_in_B / hash_table_chunk_size_in_B;
    MESSAGE_("Start to upload embedding table file to GPUs, file size: " +
             std::to_string(file_size_in_B) +
             " Bytes, total loop_num: " + std::to_string(loop_num));
    for (int i = 0; i < loop_num; i++) {
      // read a chunk of data from file
      // one pair in hash table file includes: <key, value>
      weight_stream.read(hash_table_chunk, hash_table_chunk_size_in_B);

      // memcpy from CPU to CPU
      char *src_buf = hash_table_chunk;
      TypeHashKey *key_dst_buf;
      float *value_dst_buf;
      for (int k = 0; k < chunk_loop; k++) {  // process a tile in each loop
        TypeHashKey key = *((TypeHashKey *)src_buf);
        int gid = key % device_resources->get_total_gpu_count();  // global GPU ID
        int id = device_resources->get_local_id(gid);             // local GPU ID (not gpudevice id)
        int dst_rank = device_resources->get_pid(gid);            // node id

        if (my_rank == dst_rank) {
          // memcpy hash_table_key to corresponding GPU
          key_dst_buf = h_hash_table_key_chunk_per_gpu[id] +
                        tile_counter_in_chunk_per_gpu[id] * hash_table_key_tile_size;
          CK_CUDA_THROW_(cudaMemcpyAsync(key_dst_buf, src_buf, hash_table_key_tile_size_in_B,
                                         cudaMemcpyHostToHost,
                                         (*device_resources)[id]->get_stream()));

          src_buf += hash_table_key_tile_size_in_B;

          // memcpy hash_table_value to corresponding GPU
          value_dst_buf = h_hash_table_value_chunk_per_gpu[id] +
                          tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
          CK_CUDA_THROW_(cudaMemcpyAsync(value_dst_buf, src_buf, hash_table_value_tile_size_in_B,
                                         cudaMemcpyHostToHost,
                                         (*device_resources)[id]->get_stream()));

          src_buf += hash_table_value_tile_size_in_B;

          tile_counter_in_chunk_per_gpu[id] += tile_size;
        } else {
          src_buf += hash_table_key_tile_size_in_B;
          src_buf += hash_table_value_tile_size_in_B;
          continue;
        }
      }  // end of for(int k = 0; k < (chunk_loop * local_gpu_count); k++)

      // do HashTable insert <key,value_index>
      for (int id = 0; id < local_gpu_count; id++) {
        context.set_device((*device_resources)[id]->get_device_id());

        size_t tile_count = tile_counter_in_chunk_per_gpu[id];

        // memcpy hash_table_key from CPU to GPU
        CK_CUDA_THROW_(cudaMemcpyAsync(d_hash_table_key_chunk_per_gpu[id],
                                       h_hash_table_key_chunk_per_gpu[id],
                                       tile_count * sizeof(TypeHashKey), cudaMemcpyHostToDevice,
                                       (*device_resources)[id]->get_stream()));

        size_t value_index_offset = tile_counter_per_gpu[id];
        TypeHashKey *value_index_buf = d_hash_table_value_index_chunk_per_gpu[id];

        if (tile_count > 0) {
          // set hash_table_value_index on GPU
          memset_liner(value_index_buf, (TypeHashKey)value_index_offset, (TypeHashKey)1, tile_count,
                       (*device_resources)[id]->get_stream());
        }

        // do hash table insert <key, value_index> on GPU
        hash_tables[id]->insert(d_hash_table_key_chunk_per_gpu[id], value_index_buf, tile_count,
                                (*device_resources)[id]->get_stream());
        size_t value_head = hash_tables[id]->add_value_head(tile_count);
      }

      // memcpy hash_table_value from CPU to GPU
      for (int id = 0; id < local_gpu_count; id++) {
        context.set_device((*device_resources)[id]->get_device_id());
        size_t value_chunk_size = tile_counter_in_chunk_per_gpu[id] * embedding_vec_size;
        size_t value_chunk_offset = tile_counter_per_gpu[id] * embedding_vec_size;
        float *src_buf = h_hash_table_value_chunk_per_gpu[id];
        float *dst_buf = hash_table_value_tensors[id]->get_ptr() + value_chunk_offset;
        CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf, src_buf, value_chunk_size * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       (*device_resources)[id]->get_stream()));
      }

      sync_all_gpus(device_resources, context);

      // set counter value
      for (int id = 0; id < local_gpu_count; id++) {
        tile_counter_per_gpu[id] += tile_counter_in_chunk_per_gpu[id];
        tile_counter_in_chunk_per_gpu[id] = 0;  // reset chunk counter to zero

        if (tile_counter_per_gpu[id] > max_vocabulary_size_per_gpu) {
          char msg[100];
          sprintf(msg, "The size of hash table on GPU%d is out of range %zu\n", id,
                  max_vocabulary_size_per_gpu);
          CK_THROW_(Error_t::OutOfBound, msg);
        }
      }

      std::cout << "\rUploading " << std::fixed << std::setprecision(2)
                << (float)(i) / loop_num * 100.0f << "%, loop " << i << " of " << loop_num
                << std::flush;
    }  // end of for(int i = 0; i < loop_num; i++)

    // process the remaining data(less than a chunk)
    int remain_size_in_B = file_size_in_B - loop_num * hash_table_chunk_size_in_B;
    int remain_loop_num = remain_size_in_B / hash_table_tile_size_in_B;
    if (remain_loop_num) {
      MESSAGE_("Upload the remaining data");
      // read all the remaining data
      weight_stream.read((char *)hash_table_chunk, remain_size_in_B);

      char *src_buf = hash_table_chunk;
      TypeHashKey *key_dst_buf;
      TypeHashKey *value_index_buf;
      float *value_dst_buf;
      for (int i = 0; i < remain_loop_num; i++) {
        TypeHashKey key = *((TypeHashKey *)src_buf);
        int gid = key % device_resources->get_total_gpu_count();  // global GPU ID
        int id = device_resources->get_local_id(gid);             // local GPU ID (not gpudevice id)
        int dst_rank = device_resources->get_pid(gid);

        if (my_rank == dst_rank) {
          context.set_device((*device_resources)[id]->get_device_id());

          // memcpy hash_table_key from CPU to GPU
          key_dst_buf = d_hash_table_key_chunk_per_gpu[id];
          CK_CUDA_THROW_(cudaMemcpyAsync(key_dst_buf, src_buf, hash_table_key_tile_size_in_B,
                                         cudaMemcpyHostToDevice,
                                         (*device_resources)[id]->get_stream()));

          src_buf += hash_table_key_tile_size_in_B;

          // set value_index
          size_t value_index_offset = tile_counter_per_gpu[id];
          value_index_buf = d_hash_table_value_index_chunk_per_gpu[id];
          memset_liner(value_index_buf, (TypeHashKey)value_index_offset, (TypeHashKey)1, 1,
                       (*device_resources)[id]->get_stream());

          // do hash table insert <key, value_index> on GPU
          hash_tables[id]->insert(d_hash_table_key_chunk_per_gpu[id], value_index_buf,
                                  hash_table_key_tile_size, (*device_resources)[id]->get_stream());
          size_t value_head = hash_tables[id]->add_value_head(hash_table_key_tile_size);

          // memcpy hash_table_value from CPU to GPU
          size_t value_offset = tile_counter_per_gpu[id] * embedding_vec_size;
          value_dst_buf = hash_table_value_tensors[id]->get_ptr() + value_offset;
          CK_CUDA_THROW_(cudaMemcpyAsync(value_dst_buf, src_buf, hash_table_value_tile_size_in_B,
                                         cudaMemcpyHostToDevice,
                                         (*device_resources)[id]->get_stream()));
          src_buf += hash_table_value_tile_size_in_B;

          // set counter
          tile_counter_per_gpu[id] += hash_table_key_tile_size;
        } else {
          src_buf += hash_table_key_tile_size_in_B;
          src_buf += hash_table_value_tile_size_in_B;
          continue;
        }
      }

      // sync wait
      sync_all_gpus(device_resources, context);

    }  // end of if(remain_loop_num)

    MESSAGE_("Done");

    // release resources
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      CK_CUDA_THROW_(cudaFree(d_hash_table_value_index_chunk_per_gpu[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_key_chunk_per_gpu[id]));
    }
    CK_CUDA_THROW_(cudaFreeHost(hash_table_chunk));
    for (int id = 0; id < local_gpu_count; id++) {
      CK_CUDA_THROW_(cudaFreeHost(h_hash_table_key_chunk_per_gpu[id]));
      CK_CUDA_THROW_(cudaFreeHost(h_hash_table_value_chunk_per_gpu[id]));
    }
  }

  /**
   * upload_params_to_device() for LocalizedSlotSparseEmbeddingHash
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
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void upload_params_to_device(
      std::ifstream &weight_stream, size_t vocabulary_size, int embedding_vec_size,
      size_t max_vocabulary_size_per_gpu, Tensors<float> &hash_table_value_tensors,
      Tensors<TypeHashValueIndex> &hash_table_slot_id_tensors,
      std::vector<std::shared_ptr<HashTable<TypeHashKey, TypeHashValueIndex,
                                            std::numeric_limits<TypeHashKey>::max()>>> &hash_tables,
      const std::shared_ptr<GPUResourceGroup> &device_resources, const CudaDeviceContext &context) {
    // check file size and vocabulary_size (file size <=　hash_table_size)
    weight_stream.seekg(0, weight_stream.end);
    size_t file_size_in_B = weight_stream.tellg();
    weight_stream.seekg(0, weight_stream.beg);
    // size_t hash_table_size_in_B =
    //     vocabulary_size * (sizeof(TypeHashKey) + sizeof(TypeHashValueIndex) +
    //                        (size_t)embedding_vec_size * sizeof(float));  // key+ slot_id + value
    // if (file_size_in_B > hash_table_size_in_B) {
    //   CK_THROW_(Error_t::WrongInput,
    //             "Error: hash table file size " + std::to_string(file_size_in_B) +
    //             " is larger than hash table vocabulary_size " +
    //             std::to_string(hash_table_size_in_B));
    // }

    int my_rank = 0;
#ifdef ENABLE_MPI
    int n_ranks = 1;
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));
#endif

    // define size
    int local_gpu_count = device_resources->size();
    int chunk_loop = 1000;
    int tile_size = 1;  // must be 1, because we need to cal (key&local_gpu_count) to decide gpu_id
                        // for each <key,value>
    int hash_table_key_tile_size = tile_size;
    int hash_table_key_tile_size_in_B = hash_table_key_tile_size * sizeof(TypeHashKey);
    int hash_table_key_chunk_size = hash_table_key_tile_size * chunk_loop;
    int hash_table_key_chunk_size_in_B = hash_table_key_chunk_size * sizeof(TypeHashKey);
    int hash_table_value_tile_size = tile_size * embedding_vec_size;
    int hash_table_value_tile_size_in_B = hash_table_value_tile_size * sizeof(float);
    int hash_table_value_chunk_size = hash_table_value_tile_size * chunk_loop;
    int hash_table_value_chunk_size_in_B = hash_table_value_chunk_size * sizeof(float);
    int hash_table_slot_id_tile_size = tile_size;
    int hash_table_slot_id_tile_size_in_B =
        hash_table_slot_id_tile_size * sizeof(TypeHashValueIndex);
    int hash_table_slot_id_chunk_size = hash_table_slot_id_tile_size * chunk_loop;
    int hash_table_slot_id_chunk_size_in_B =
        hash_table_slot_id_chunk_size * sizeof(TypeHashValueIndex);
    int hash_table_tile_size_in_B = hash_table_key_tile_size_in_B +
                                    hash_table_slot_id_tile_size_in_B +
                                    hash_table_value_tile_size_in_B;
    int hash_table_chunk_size_in_B = hash_table_tile_size_in_B * chunk_loop;
    int total_gpu_count = device_resources->get_total_gpu_count();

    // CAUSION: can not decide how many values for each GPU, so need to allocate enough memory for
    // each GPU allocate GPU memory for hash_table_value_index
    std::unique_ptr<size_t[]> tile_counter_per_gpu(
        new size_t[local_gpu_count]);  // <= hash_table_value_index_per_gpu_size
    memset(tile_counter_per_gpu.get(), 0, sizeof(size_t) * local_gpu_count);
    std::unique_ptr<size_t[]> tile_counter_in_chunk_per_gpu(new size_t[local_gpu_count]);
    memset(tile_counter_in_chunk_per_gpu.get(), 0, sizeof(size_t) * local_gpu_count);
    std::unique_ptr<TypeHashKey *[]> d_hash_table_value_index_chunk_per_gpu(
        new TypeHashKey *[local_gpu_count]);
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      CK_CUDA_THROW_(
          cudaMalloc(&d_hash_table_value_index_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
      // initalize to zeros
      CK_CUDA_THROW_(cudaMemsetAsync(d_hash_table_value_index_chunk_per_gpu[id], 0,
                                     hash_table_key_chunk_size_in_B,
                                     (*device_resources)[id]->get_stream()));
    }

    // sync wait
    sync_all_gpus(device_resources, context);

    // CAUSION: can not decide how many values for each GPU, so need to allocate enough memory for
    // each GPU allocate CPU/GPU memory for hash_table/key/value chunk
    char *hash_table_chunk;
    CK_CUDA_THROW_(cudaMallocHost(&hash_table_chunk, hash_table_chunk_size_in_B));
    std::unique_ptr<TypeHashKey *[]> h_hash_table_key_chunk_per_gpu(
        new TypeHashKey *[local_gpu_count]);
    for (int id = 0; id < local_gpu_count; id++) {
      CK_CUDA_THROW_(
          cudaMallocHost(&h_hash_table_key_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
    }
    std::unique_ptr<TypeHashKey *[]> d_hash_table_key_chunk_per_gpu(
        new TypeHashKey *[local_gpu_count]);
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      CK_CUDA_THROW_(
          cudaMalloc(&d_hash_table_key_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
    }
    std::unique_ptr<TypeHashValueIndex *[]> h_hash_table_slot_id_chunk_per_gpu(
        new TypeHashValueIndex *[local_gpu_count]);
    for (int id = 0; id < local_gpu_count; id++) {
      CK_CUDA_THROW_(cudaMallocHost(&h_hash_table_slot_id_chunk_per_gpu[id],
                                    hash_table_slot_id_chunk_size_in_B));
    }
    std::unique_ptr<TypeHashValueIndex *[]> d_hash_table_slot_id_chunk_per_gpu(
        new TypeHashValueIndex *[local_gpu_count]);
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      CK_CUDA_THROW_(
          cudaMalloc(&d_hash_table_slot_id_chunk_per_gpu[id], hash_table_slot_id_chunk_size_in_B));
    }
    std::unique_ptr<float *[]> h_hash_table_value_chunk_per_gpu(new float *[local_gpu_count]);
    for (int id = 0; id < local_gpu_count; id++) {
      CK_CUDA_THROW_(
          cudaMallocHost(&h_hash_table_value_chunk_per_gpu[id], hash_table_value_chunk_size_in_B));
    }

    // do upload
    int loop_num = file_size_in_B / hash_table_chunk_size_in_B;
    MESSAGE_("Start to upload embedding table file to GPUs, file size: " +
             std::to_string(file_size_in_B) +
             " Bytes, total loop_num: " + std::to_string(loop_num));
    for (int i = 0; i < loop_num; i++) {
      // read a chunk of data from file
      // one pair in hash table file includes: <key, slot_id, value>
      weight_stream.read(hash_table_chunk, hash_table_chunk_size_in_B);

      // memcpy from CPU to CPU
      char *src_buf = hash_table_chunk;
      TypeHashKey *key_dst_buf;
      TypeHashValueIndex *slot_id_dst_buf;
      float *value_dst_buf;
      for (int k = 0; k < chunk_loop; k++) {  // process a tile in each loop
        TypeHashValueIndex slot_id =
            *((TypeHashValueIndex *)(src_buf + hash_table_key_tile_size_in_B));
        int gid = slot_id % total_gpu_count;            // global GPU ID
        int id = device_resources->get_local_id(gid);   // local GPU ID (not gpudevice id)
        int dst_rank = device_resources->get_pid(gid);  // node id

        if (my_rank == dst_rank) {
          // memcpy hash_table_key to corresponding GPU
          key_dst_buf = h_hash_table_key_chunk_per_gpu[id] +
                        tile_counter_in_chunk_per_gpu[id] * hash_table_key_tile_size;
          CK_CUDA_THROW_(cudaMemcpyAsync(key_dst_buf, src_buf, hash_table_key_tile_size_in_B,
                                         cudaMemcpyHostToHost,
                                         (*device_resources)[id]->get_stream()));

          src_buf += hash_table_key_tile_size_in_B;

          // memcpy hash_table_slot_id to corresponding GPU
          slot_id_dst_buf = h_hash_table_slot_id_chunk_per_gpu[id] +
                            tile_counter_in_chunk_per_gpu[id] * hash_table_slot_id_tile_size;
          CK_CUDA_THROW_(cudaMemcpyAsync(slot_id_dst_buf, src_buf,
                                         hash_table_slot_id_tile_size_in_B, cudaMemcpyHostToHost,
                                         (*device_resources)[id]->get_stream()));

          src_buf += hash_table_slot_id_tile_size_in_B;

          // memcpy hash_table_value to corresponding GPU
          value_dst_buf = h_hash_table_value_chunk_per_gpu[id] +
                          tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
          CK_CUDA_THROW_(cudaMemcpyAsync(value_dst_buf, src_buf, hash_table_value_tile_size_in_B,
                                         cudaMemcpyHostToHost,
                                         (*device_resources)[id]->get_stream()));

          src_buf += hash_table_value_tile_size_in_B;

          tile_counter_in_chunk_per_gpu[id] += tile_size;
        } else {
          src_buf += hash_table_key_tile_size_in_B;
          src_buf += hash_table_slot_id_tile_size_in_B;
          src_buf += hash_table_value_tile_size_in_B;
          continue;
        }
      }  // end of for(int k = 0; k < (chunk_loop * local_gpu_count); k++)

      // do HashTable insert <key,value_index>
      for (int id = 0; id < local_gpu_count; id++) {
        if (tile_counter_in_chunk_per_gpu[id] == 0) {
          continue;
        }

        context.set_device((*device_resources)[id]->get_device_id());

        size_t tile_count = tile_counter_in_chunk_per_gpu[id];

        // memcpy hash_table_key from CPU to GPU
        CK_CUDA_THROW_(cudaMemcpyAsync(d_hash_table_key_chunk_per_gpu[id],
                                       h_hash_table_key_chunk_per_gpu[id],
                                       tile_count * sizeof(TypeHashKey), cudaMemcpyHostToDevice,
                                       (*device_resources)[id]->get_stream()));

        size_t value_index_offset = tile_counter_per_gpu[id];
        TypeHashKey *value_index_buf = d_hash_table_value_index_chunk_per_gpu[id];

        if (tile_count > 0) {
          // set hash_table_value_index on GPU
          memset_liner(value_index_buf, (TypeHashKey)value_index_offset, (TypeHashKey)1, tile_count,
                       (*device_resources)[id]->get_stream());
        }

        // do hash table insert <key, value_index> on GPU
        hash_tables[id]->insert(d_hash_table_key_chunk_per_gpu[id], value_index_buf, tile_count,
                                (*device_resources)[id]->get_stream());
        size_t value_head = hash_tables[id]->add_value_head(tile_count);
      }

      // memcpy hash_table_slot_id and hash_table_value from CPU to GPU
      for (int id = 0; id < local_gpu_count; id++) {
        if (tile_counter_in_chunk_per_gpu[id] == 0) {
          continue;
        }

        context.set_device((*device_resources)[id]->get_device_id());

        size_t slot_id_chunk_size =
            tile_counter_in_chunk_per_gpu[id] * hash_table_slot_id_tile_size;
        size_t slot_id_offset = tile_counter_per_gpu[id] * hash_table_slot_id_tile_size;

        if ((slot_id_offset + slot_id_chunk_size) > max_vocabulary_size_per_gpu) {
          char msg[100];
          sprintf(msg, "The size of hash table on GPU%d is out of range %zu\n", id,
                  max_vocabulary_size_per_gpu);
          CK_THROW_(Error_t::OutOfBound, msg);
        }

        TypeHashValueIndex *src_buf_sid = h_hash_table_slot_id_chunk_per_gpu[id];
        TypeHashValueIndex *dst_buf_sid =
            hash_table_slot_id_tensors[id]->get_ptr() + slot_id_offset;
        CK_CUDA_THROW_(cudaMemcpyAsync(
            dst_buf_sid, src_buf_sid, slot_id_chunk_size * sizeof(TypeHashValueIndex),
            cudaMemcpyHostToDevice, (*device_resources)[id]->get_stream()));

        size_t value_chunk_size = tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
        size_t value_chunk_offset = tile_counter_per_gpu[id] * hash_table_value_tile_size;
        float *src_buf_value = h_hash_table_value_chunk_per_gpu[id];
        float *dst_buf_value = hash_table_value_tensors[id]->get_ptr() + value_chunk_offset;
        CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf_value, src_buf_value,
                                       value_chunk_size * sizeof(float), cudaMemcpyHostToDevice,
                                       (*device_resources)[id]->get_stream()));
      }

      sync_all_gpus(device_resources, context);

      // set counter value
      for (int id = 0; id < local_gpu_count; id++) {
        tile_counter_per_gpu[id] +=
            tile_counter_in_chunk_per_gpu[id];  // accumulate total tile counter
        tile_counter_in_chunk_per_gpu[id] = 0;  // reset chunk counter to zero

        if (tile_counter_per_gpu[id] > max_vocabulary_size_per_gpu) {
          char msg[100];
          sprintf(msg, "The size of hash table on GPU%d is out of range %zu\n", id,
                  max_vocabulary_size_per_gpu);
          CK_THROW_(Error_t::OutOfBound, msg);
        }
      }

      std::cout << "\rUploading " << std::fixed << std::setprecision(2)
                << (float)(i) / loop_num * 100.0f << "%, loop " << i << " of " << loop_num
                << std::flush;
    }  // end of for(int i = 0; i < loop_num; i++)

    std::cout << std::endl;

    // process the remaining data(less than a chunk)
    int remain_size_in_B = file_size_in_B - loop_num * hash_table_chunk_size_in_B;
    int remain_loop_num = remain_size_in_B / hash_table_tile_size_in_B;
    if (remain_loop_num) {
      MESSAGE_("Upload the remaining data");
      // read all the remaining data
      weight_stream.read((char *)hash_table_chunk, remain_size_in_B);

      char *src_buf = hash_table_chunk;
      TypeHashKey *key_dst_buf;
      TypeHashValueIndex *value_index_buf;
      TypeHashValueIndex *slot_id_dst_buf;
      float *value_dst_buf;
      for (int i = 0; i < remain_loop_num; i++) {  // process one tile in each loop

        TypeHashValueIndex slot_id =
            *((TypeHashValueIndex *)(src_buf + hash_table_key_tile_size_in_B));
        int gid = slot_id % total_gpu_count;            // global GPU ID
        int id = device_resources->get_local_id(gid);   // local GPU ID (not gpu devie id)
        int dst_rank = device_resources->get_pid(gid);  // node id

        if (my_rank == dst_rank) {
          context.set_device((*device_resources)[id]->get_device_id());

          // memcpy hash_table_key from CPU to GPU
          key_dst_buf = d_hash_table_key_chunk_per_gpu[id];
          CK_CUDA_THROW_(cudaMemcpyAsync(key_dst_buf, src_buf, hash_table_key_tile_size_in_B,
                                         cudaMemcpyHostToDevice,
                                         (*device_resources)[id]->get_stream()));
          src_buf += hash_table_key_tile_size_in_B;

          // set value_index
          size_t value_index_offset = tile_counter_per_gpu[id];
          value_index_buf = d_hash_table_value_index_chunk_per_gpu[id];
          memset_liner(value_index_buf, (TypeHashKey)value_index_offset, (TypeHashKey)1, 1,
                       (*device_resources)[id]->get_stream());

          // do hash table insert <key, value_index> on GPU
          hash_tables[id]->insert(d_hash_table_key_chunk_per_gpu[id], value_index_buf,
                                  hash_table_key_tile_size, (*device_resources)[id]->get_stream());
          size_t value_head = hash_tables[id]->add_value_head(hash_table_key_tile_size);

          // memcpy hash_table_slot_id to corresponding GPU
          size_t slot_id_offset = tile_counter_per_gpu[id];
          slot_id_dst_buf = hash_table_slot_id_tensors[id]->get_ptr() + slot_id_offset;
          CK_CUDA_THROW_(cudaMemcpyAsync(slot_id_dst_buf, src_buf,
                                         hash_table_slot_id_tile_size_in_B, cudaMemcpyHostToHost,
                                         (*device_resources)[id]->get_stream()));
          src_buf += hash_table_slot_id_tile_size_in_B;

          // memcpy hash_table_value from CPU to GPU
          size_t value_offset = tile_counter_per_gpu[id] * embedding_vec_size;
          value_dst_buf = hash_table_value_tensors[id]->get_ptr() + value_offset;
          CK_CUDA_THROW_(cudaMemcpyAsync(value_dst_buf, src_buf, hash_table_value_tile_size_in_B,
                                         cudaMemcpyHostToDevice,
                                         (*device_resources)[id]->get_stream()));
          src_buf += hash_table_value_tile_size_in_B;

          // set counter
          tile_counter_per_gpu[id] += tile_size;
        } else {
          src_buf += hash_table_key_tile_size_in_B;
          src_buf += hash_table_slot_id_tile_size_in_B;
          src_buf += hash_table_value_tile_size_in_B;
          continue;
        }
      }

      // sync wait
      sync_all_gpus(device_resources, context);

    }  // end of if(remain_loop_num)

    MESSAGE_("Done");

    // release resources
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      CK_CUDA_THROW_(cudaFree(d_hash_table_value_index_chunk_per_gpu[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_key_chunk_per_gpu[id]));
    }
    CK_CUDA_THROW_(cudaFreeHost(hash_table_chunk));
    for (int id = 0; id < local_gpu_count; id++) {
      CK_CUDA_THROW_(cudaFreeHost(h_hash_table_key_chunk_per_gpu[id]));
      CK_CUDA_THROW_(cudaFreeHost(h_hash_table_value_chunk_per_gpu[id]));
    }
  }

  /**
   * download_params_to_host for DistributedSlotSparseEmbeddingHash
   * download hash_table from GPUs to CPU.
   * @param weight_stream weight file stream to write.
   * @param vocabulary_size the total row number of hash table.
   * @param embedding_vec_size embedding vector size.
   * @param max_hash_table_size_per_gpu max vocabulary size for each GPU
   * @param hash_table_value_tensors the tensors of hash table value on multi GPUs.
   * @param hash_tables the hash tables on multi GPUs
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void download_params_to_host(
      std::ofstream &weight_stream, size_t vocabulary_size, int embedding_vec_size,
      size_t max_hash_table_size_per_gpu, Tensors<float> &hash_table_value_tensors,
      std::vector<std::shared_ptr<HashTable<TypeHashKey, TypeHashValueIndex,
                                            std::numeric_limits<TypeHashKey>::max()>>> &hash_tables,
      const std::shared_ptr<GPUResourceGroup> &device_resources, const CudaDeviceContext &context) {
    int local_gpu_count = device_resources->size();

    int my_rank = 0;
#ifdef ENABLE_MPI
    int n_ranks = 1;
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));
#endif

    // memory allocation
    std::unique_ptr<size_t[]> count(new size_t[local_gpu_count]);
    size_t max_count = 0;
    size_t total_count = 0;
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      auto count_tmp = hash_tables[id]->get_size((*device_resources)[id]->get_stream());
      if (count_tmp != hash_tables[id]->get_value_head()) {
        CK_THROW_(Error_t::WrongInput,
                  "Error: hash_table get_value_head() size not equal to get_size()");
      }
      count[id] = count_tmp;
      max_count = max(max_count, count[id]);
      total_count += count[id];
    }

#ifdef ENABLE_MPI
    CK_MPI_THROW_(
        MPI_Allreduce(MPI_IN_PLACE, &max_count, sizeof(size_t), MPI_CHAR, MPI_MAX, MPI_COMM_WORLD));
#endif

    if (total_count > (size_t)vocabulary_size) {
      CK_THROW_(Error_t::WrongInput,
                "Error: required download size is larger than hash table vocabulary_size");
    }

    std::unique_ptr<TypeHashKey *[]> h_hash_table_key(new TypeHashKey *[local_gpu_count]);
    std::unique_ptr<TypeHashKey *[]> d_hash_table_key(new TypeHashKey *[local_gpu_count]);
    std::unique_ptr<TypeHashValueIndex *[]> d_hash_table_value_index(
        new TypeHashValueIndex *[local_gpu_count]);
    std::unique_ptr<float *[]> h_hash_table_value(new float *[local_gpu_count]);
    std::unique_ptr<float *[]> d_hash_table_value(new float *[local_gpu_count]);
    std::unique_ptr<size_t *[]> d_dump_counter(new size_t *[local_gpu_count]);
    for (int id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      context.set_device((*device_resources)[id]->get_device_id());

      cudaMallocHost(&h_hash_table_key[id], count[id] * sizeof(TypeHashKey));
      cudaMalloc(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey));
      cudaMalloc(&d_hash_table_value_index[id], count[id] * sizeof(TypeHashValueIndex));
      cudaMallocHost(&h_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float));
      cudaMalloc(&d_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float));
      cudaMalloc(&d_dump_counter[id], count[id] * sizeof(size_t));
    }

    // dump hash table from GPUs
    for (int id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      MESSAGE_("Rank" + std::to_string(my_rank) + ": Dump hash table from GPU" +
               std::to_string(id));

      context.set_device((*device_resources)[id]->get_device_id());

      hash_tables[id]->dump(d_hash_table_key[id], d_hash_table_value_index[id], 0,
                            max_hash_table_size_per_gpu, d_dump_counter[id],
                            (*device_resources)[id]->get_stream());

      CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_key[id], d_hash_table_key[id],
                                     count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost,
                                     (*device_resources)[id]->get_stream()));

      get_hash_value((*device_resources)[id]->get_stream(), count[id], embedding_vec_size,
                     d_hash_table_value_index[id], hash_table_value_tensors[id]->get_ptr(),
                     d_hash_table_value[id]);

      CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_value[id], d_hash_table_value[id],
                                     count[id] * embedding_vec_size * sizeof(float),
                                     cudaMemcpyDeviceToHost,
                                     (*device_resources)[id]->get_stream()));
    }

    // sync wait
    sync_all_gpus(device_resources, context);

    const int master_node = 0;
#ifdef ENABLE_MPI
    const int base_tag = 0xed;
#endif
    // TODO: could be optimized ???
    size_t max_size_in_B = max_count * (sizeof(TypeHashKey) + sizeof(float) * embedding_vec_size);
    std::unique_ptr<char[]> file_buf(new char[max_size_in_B]);
    size_t key_size = sizeof(TypeHashKey);
    size_t value_size = sizeof(float) * embedding_vec_size;
    for (int id = 0; id < local_gpu_count; id++) {
      size_t size_in_B = count[id] * (sizeof(TypeHashKey) + sizeof(float) * embedding_vec_size);
      size_t offset = 0;
      for (unsigned int k = 0; k < count[id]; k++) {
        memcpy(file_buf.get() + offset, h_hash_table_key[id] + k, key_size);
        offset += key_size;
        memcpy(file_buf.get() + offset, h_hash_table_value[id] + k * embedding_vec_size,
               value_size);
        offset += value_size;

        std::cout << "\rRank" << my_rank << ": Seperate keys and values on GPU" << id << ", finish "
                  << k << " of total count " << count[id] << ", " << (float)k / count[id] * 100.0f
                  << "%" << std::flush;
      }
      std::cout << std::endl;
      if (my_rank == master_node) {
        MESSAGE_("Rank" + std::to_string(my_rank) + ": Write hash table <key,value> pairs to file");
        weight_stream.write(file_buf.get(), size_in_B);
      }
#ifdef ENABLE_MPI
      else {
        MESSAGE_("Rank" + std::to_string(my_rank) + ": Send hash table <key,value> pairs on GPU" +
                 std::to_string(id) + " to master node  ");
        int tag = (id << 8) | base_tag;
        CK_MPI_THROW_(
            MPI_Send(file_buf.get(), size_in_B, MPI_CHAR, master_node, tag, MPI_COMM_WORLD));
      }
#endif
    }

#ifdef ENABLE_MPI
    if (my_rank == master_node) {
      for (int r = 1; r < n_ranks; r++) {
        for (int id = 0; id < local_gpu_count; id++) {
          MESSAGE_("Rank" + std::to_string(my_rank) +
                   ": Recv hash table <key,value> pairs from rank" + std::to_string(r) + " on GPU" +
                   std::to_string(id) + ", and write to file ");
          int tag = (id << 8) | base_tag;
          MPI_Status status;
          CK_MPI_THROW_(MPI_Probe(r, tag, MPI_COMM_WORLD, &status));
          int size_in_B;
          CK_MPI_THROW_(MPI_Get_count(&status, MPI_CHAR, &size_in_B));
          CK_MPI_THROW_(MPI_Recv(file_buf.get(), size_in_B, MPI_CHAR, r, tag, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE));
          weight_stream.write(file_buf.get(), size_in_B);
        }
      }
    }
#endif

    MESSAGE_("Done");

    for (int id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      context.set_device((*device_resources)[id]->get_device_id());

      CK_CUDA_THROW_(cudaFreeHost(h_hash_table_key[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_key[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_value_index[id]));
      CK_CUDA_THROW_(cudaFreeHost(h_hash_table_value[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_value[id]));
      CK_CUDA_THROW_(cudaFree(d_dump_counter[id]));
    }

    return;
  }  // end of download_params_to_host()

  /**
   * download_params_to_host for LocalizedSlotSparseEmbeddingHash.
   * @param weight_stream weight file stream to write.
   * @param vocabulary_size the total row number of hash table.
   * @param embedding_vec_size embedding vector size.
   * @param max_hash_table_size_per_gpu max vocabulary size for each GPU
   * @param hash_table_value_tensors the hash table value on multi-GPU.
   * @param hash_table_slot_id_tensors the hash table slot_ids on multi-GPU
   * @param hash_tables the hash tables on multi GPUs
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void download_params_to_host(
      std::ofstream &weight_stream, size_t vocabulary_size, int embedding_vec_size,
      size_t max_hash_table_size_per_gpu, Tensors<float> &hash_table_value_tensors,
      Tensors<TypeHashValueIndex> &hash_table_slot_id_tensors,
      std::vector<std::shared_ptr<HashTable<TypeHashKey, TypeHashValueIndex,
                                            std::numeric_limits<TypeHashKey>::max()>>> &hash_tables,
      const std::shared_ptr<GPUResourceGroup> &device_resources, const CudaDeviceContext &context) {
    int local_gpu_count = device_resources->size();

    int my_rank = 0;
#ifdef ENABLE_MPI
    int n_ranks = 1;
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));
#endif

    // memory allocation
    std::unique_ptr<size_t[]> count(new size_t[local_gpu_count]);
    size_t max_count = 0;
    size_t total_count = 0;
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      auto count_tmp = hash_tables[id]->get_size((*device_resources)[id]->get_stream());
      if (count_tmp != hash_tables[id]->get_value_head()) {
        std::cout << "gpu" << id << ", get_size=" << count_tmp
                  << ", get_value_head=" << hash_tables[id]->get_value_head() << std::endl;
        CK_THROW_(Error_t::WrongInput,
                  "Error: hash_table get_value_head() is not equal to get_size()");
      }
      count[id] = count_tmp;
      max_count = max(max_count, count[id]);
      total_count += count[id];
    }

#ifdef ENABLE_MPI
    CK_MPI_THROW_(
        MPI_Allreduce(MPI_IN_PLACE, &max_count, sizeof(size_t), MPI_CHAR, MPI_MAX, MPI_COMM_WORLD));
#endif

    if (total_count > (size_t)vocabulary_size) {
      CK_THROW_(Error_t::WrongInput,
                "Error: required download size is larger than hash table vocabulary_size");
    }

    std::unique_ptr<TypeHashKey *[]> h_hash_table_key(new TypeHashKey *[local_gpu_count]);
    std::unique_ptr<TypeHashKey *[]> d_hash_table_key(new TypeHashKey *[local_gpu_count]);
    std::unique_ptr<TypeHashValueIndex *[]> d_hash_table_value_index(
        new TypeHashValueIndex *[local_gpu_count]);
    std::unique_ptr<TypeHashValueIndex *[]> h_hash_table_slot_id(
        new TypeHashValueIndex *[local_gpu_count]);
    std::unique_ptr<TypeHashValueIndex *[]> d_hash_table_slot_id(
        new TypeHashValueIndex *[local_gpu_count]);
    std::unique_ptr<float *[]> h_hash_table_value(new float *[local_gpu_count]);
    std::unique_ptr<float *[]> d_hash_table_value(new float *[local_gpu_count]);
    std::unique_ptr<size_t *[]> d_dump_counter(new size_t *[local_gpu_count]);
    for (int id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      context.set_device((*device_resources)[id]->get_device_id());

      cudaMallocHost(&h_hash_table_key[id], count[id] * sizeof(TypeHashKey));
      cudaMalloc(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey));
      cudaMalloc(&d_hash_table_value_index[id], count[id] * sizeof(TypeHashValueIndex));
      cudaMallocHost(&h_hash_table_slot_id[id], count[id] * sizeof(TypeHashValueIndex));
      cudaMalloc(&d_hash_table_slot_id[id], count[id] * sizeof(TypeHashValueIndex));
      cudaMallocHost(&h_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float));
      cudaMalloc(&d_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float));
      cudaMalloc(&d_dump_counter[id], count[id] * sizeof(size_t));
    }

    // dump hash table on GPU
    for (int id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      MESSAGE_("Rank" + std::to_string(my_rank) + ": Dump hash table from GPU" +
               std::to_string(id));

      context.set_device((*device_resources)[id]->get_device_id());

      hash_tables[id]->dump(d_hash_table_key[id], d_hash_table_value_index[id], 0,
                            max_hash_table_size_per_gpu, d_dump_counter[id],
                            (*device_resources)[id]->get_stream());

      CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_key[id], d_hash_table_key[id],
                                     count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost,
                                     (*device_resources)[id]->get_stream()));

      get_hash_value((*device_resources)[id]->get_stream(), count[id], embedding_vec_size,
                     d_hash_table_value_index[id], hash_table_value_tensors[id]->get_ptr(),
                     d_hash_table_value[id]);

      CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_value[id], d_hash_table_value[id],
                                     count[id] * embedding_vec_size * sizeof(float),
                                     cudaMemcpyDeviceToHost,
                                     (*device_resources)[id]->get_stream()));

      get_hash_slot_id((*device_resources)[id]->get_stream(), count[id],
                       d_hash_table_value_index[id], hash_table_slot_id_tensors[id]->get_ptr(),
                       d_hash_table_slot_id[id]);

      CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_slot_id[id], d_hash_table_slot_id[id],
                                     count[id] * sizeof(TypeHashValueIndex), cudaMemcpyDeviceToHost,
                                     (*device_resources)[id]->get_stream()));
    }

    // sync wait
    sync_all_gpus(device_resources, context);

    const int master_node = 0;
#ifdef ENABLE_MPI
    const int base_tag = 0xed;
#endif
    // TODO: could be optimized ???
    // one pair in the file includes <key,slot_id,value>
    size_t pair_size_in_B =
        sizeof(TypeHashKey) + sizeof(TypeHashValueIndex) + sizeof(float) * embedding_vec_size;
    size_t max_size_in_B = max_count * pair_size_in_B;
    std::unique_ptr<char[]> file_buf(new char[max_size_in_B]);
    size_t key_size = sizeof(TypeHashKey);
    size_t slot_id_size = sizeof(TypeHashValueIndex);
    size_t value_size = sizeof(float) * embedding_vec_size;
    for (int id = 0; id < local_gpu_count; id++) {
      size_t size_in_B = count[id] * pair_size_in_B;
      size_t offset = 0;
      for (unsigned int k = 0; k < count[id]; k++) {
        std::cout << "\rRank" << my_rank << ": Seperate keys, slot_ids and values on GPU" << id
                  << ", finish " << k << " of total count " << count[id] << ", "
                  << (float)k / count[id] * 100.0f << "%" << std::flush;

        memcpy(file_buf.get() + offset, h_hash_table_key[id] + k, key_size);
        offset += key_size;
        memcpy(file_buf.get() + offset, h_hash_table_slot_id[id] + k, slot_id_size);
        offset += slot_id_size;
        memcpy(file_buf.get() + offset, h_hash_table_value[id] + k * embedding_vec_size,
               value_size);
        offset += value_size;
      }
      std::cout << std::endl;
      if (my_rank == master_node) {
        MESSAGE_("Rank" + std::to_string(my_rank) + ": Write hash table <key,value> pairs to file");
        weight_stream.write(file_buf.get(), size_in_B);
      }
#ifdef ENABLE_MPI
      else {
        MESSAGE_("Rank" + std::to_string(my_rank) + ": Send hash table <key,value> pairs on GPU" +
                 std::to_string(id) + " to master node  ");
        int tag = (id << 8) | base_tag;
        CK_MPI_THROW_(
            MPI_Send(file_buf.get(), size_in_B, MPI_CHAR, master_node, tag, MPI_COMM_WORLD));
      }
#endif
    }

#ifdef ENABLE_MPI
    if (my_rank == master_node) {
      for (int r = 1; r < n_ranks; r++) {
        for (int id = 0; id < local_gpu_count; id++) {
          MESSAGE_("Rank" + std::to_string(my_rank) +
                   ": Recv hash table <key,value> pairs from rank" + std::to_string(r) + " on GPU" +
                   std::to_string(id) + ", and write to file ");
          int tag = (id << 8) | base_tag;
          MPI_Status status;
          CK_MPI_THROW_(MPI_Probe(r, tag, MPI_COMM_WORLD, &status));
          int size_in_B;
          CK_MPI_THROW_(MPI_Get_count(&status, MPI_CHAR, &size_in_B));
          CK_MPI_THROW_(MPI_Recv(file_buf.get(), size_in_B, MPI_CHAR, r, tag, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE));
          weight_stream.write(file_buf.get(), size_in_B);
        }
      }
    }
#endif

    MESSAGE_("Done");

    for (int id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      context.set_device((*device_resources)[id]->get_device_id());

      CK_CUDA_THROW_(cudaFreeHost(h_hash_table_key[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_key[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_value_index[id]));
      CK_CUDA_THROW_(cudaFreeHost(h_hash_table_slot_id[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_slot_id[id]));
      CK_CUDA_THROW_(cudaFreeHost(h_hash_table_value[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_value[id]));
      CK_CUDA_THROW_(cudaFree(d_dump_counter[id]));
    }

    return;
  }  // end of download_params_to_host()

  /**
   * get forward results from GPUs to CPU. This functin is just used for utest.
   * @param memcpy_size the number of elemments to do memcpy.
   * @param embedding_feature_tensors the source tensors of multi GPUs to copy from.
   * @param embedding_feature the destination CPU buffer pointer to copy to.
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  template <typename TypeEmbeddingComp>
  void get_forward_results(int memcpy_size,
                           const Tensors<TypeEmbeddingComp> &embedding_feature_tensors,
                           TypeEmbeddingComp *embedding_feature,
                           Tensors<TypeEmbeddingComp> &temp_tensors,
                           const std::shared_ptr<GPUResourceGroup> &device_resources,
                           const CudaDeviceContext &context) {
    int local_gpu_count = device_resources->size();
    int total_gpu_count = device_resources->get_total_gpu_count();

    if (total_gpu_count > 1) {
      // nccl allGather
      all_gather(memcpy_size,
                 embedding_feature_tensors,  // send
                 temp_tensors,               // recv
                 device_resources, context);
      // sync_all_gpus(device_resources, context);

      // memcpy D2H
      context.set_device((*device_resources)[0]->get_device_id());
      CK_CUDA_THROW_(cudaMemcpyAsync(embedding_feature, temp_tensors[0]->get_ptr(),
                                     total_gpu_count * memcpy_size * sizeof(TypeEmbeddingComp),
                                     cudaMemcpyDeviceToHost, (*device_resources)[0]->get_stream()));
      CK_CUDA_THROW_(cudaStreamSynchronize((*device_resources)[0]->get_stream()));
    } else {
      context.set_device((*device_resources)[0]->get_device_id());
      CK_CUDA_THROW_(cudaMemcpyAsync(embedding_feature, embedding_feature_tensors[0]->get_ptr(),
                                     memcpy_size * sizeof(TypeEmbeddingComp),
                                     cudaMemcpyDeviceToHost, (*device_resources)[0]->get_stream()));
      CK_CUDA_THROW_(cudaStreamSynchronize((*device_resources)[0]->get_stream()));
    }

    return;
  }

  /**
   * get backward results from GPU to CPU. This functin is just used for utest.
   * @param devId gpu device id to get backward resutls from.
   * @param memcpy_size the number of elemments to do memcpy.
   * @param wgrad_tensors the source tensors of multi GPUs to copy from.
   * @param wgrad the destination CPU buffer pointer to copy to.
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  template <typename TypeEmbeddingComp>
  void get_backward_results(int devId, int memcpy_size,
                            const Tensors<TypeEmbeddingComp> &wgrad_tensors,
                            TypeEmbeddingComp *wgrad,
                            const std::shared_ptr<GPUResourceGroup> &device_resources,
                            const CudaDeviceContext &context) {
    context.set_device((*device_resources)[devId]->get_device_id());
    CK_CUDA_THROW_(cudaMemcpyAsync(wgrad, wgrad_tensors[devId]->get_ptr(),
                                   memcpy_size * sizeof(TypeEmbeddingComp), cudaMemcpyDeviceToHost,
                                   (*device_resources)[devId]->get_stream()));
    CK_CUDA_THROW_(cudaStreamSynchronize((*device_resources)[devId]->get_stream()));

    return;
  }

  /**
   * get update_params results from GPU to CPU. This functin is just used for utest.
   * @param max_hash_table_size_per_gpu max vocabulary size for each GPU.
   * @param embedding_vec_size embedding vector size.
   * @param vocabulary_size the total number of rows in hash table
   * @param hash_table_value_tensors the tensors of hash table value on multi GPUs
   * @param hash_tables the hash tables on multi GPUs
   * @param hash_table_key the pointer of hash table key on CPU
   * @param hash_table_value the ponter of hash table value on CPU
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void get_update_params_results(
      size_t max_hash_table_size_per_gpu, int embedding_vec_size, size_t vocabulary_size,
      const Tensors<float> &hash_table_value_tensors,
      const std::vector<std::shared_ptr<
          HashTable<TypeHashKey, TypeHashValueIndex, std::numeric_limits<TypeHashKey>::max()>>>
          &hash_tables,
      TypeHashKey *hash_table_key, float *hash_table_value,
      const std::shared_ptr<GPUResourceGroup> &device_resources, const CudaDeviceContext &context) {
    int local_gpu_count = device_resources->size();

    // memory allocation
    std::unique_ptr<size_t[]> count(new size_t[local_gpu_count]);
    size_t total_count = 0;
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      if ((count[id] = hash_tables[id]->get_value_head()) !=
          hash_tables[id]->get_size((*device_resources)[id]->get_stream())) {
        std::cout << "hashtable: get_value_head()=" << hash_tables[id]->get_value_head()
                  << ", get_size()="
                  << hash_tables[id]->get_size((*device_resources)[id]->get_stream()) << std::endl;
        CK_THROW_(Error_t::WrongInput,
                  "Error: hash_table get_value_head() size not equal to get_size()");
      }
      total_count += count[id];

#ifndef NDEBUG
      std::cout << "GPU[" << id << "]: number of <key,value> pairs:" << count[id] << std::endl;
#endif
    }

#ifndef NDEBUG
    std::cout << "Total number of <key,value> pairs:" << total_count << std::endl;
#endif

    if (total_count > (size_t)vocabulary_size) {
      CK_THROW_(Error_t::WrongInput,
                "Error: required download size is larger than hash table vocabulary_size");
    }

    std::unique_ptr<TypeHashKey *[]> d_hash_table_key(new TypeHashKey *[local_gpu_count]);
    std::unique_ptr<TypeHashValueIndex *[]> d_hash_table_value_index(
        new TypeHashValueIndex *[local_gpu_count]);
    std::unique_ptr<float *[]> d_hash_table_value(new float *[local_gpu_count]);
    std::unique_ptr<size_t *[]> d_dump_counter(new size_t *[local_gpu_count]);
    for (int id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      context.set_device((*device_resources)[id]->get_device_id());

      cudaMalloc(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey));
      cudaMalloc(&d_hash_table_value_index[id], count[id] * sizeof(TypeHashValueIndex));
      cudaMalloc(&d_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float));
      cudaMalloc(&d_dump_counter[id], count[id] * sizeof(size_t));
    }

    // dump hash table on GPU
    for (int id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      context.set_device((*device_resources)[id]->get_device_id());

      hash_tables[id]->dump(d_hash_table_key[id], d_hash_table_value_index[id], 0,
                            max_hash_table_size_per_gpu, d_dump_counter[id],
                            (*device_resources)[id]->get_stream());

      get_hash_value((*device_resources)[id]->get_stream(), count[id], embedding_vec_size,
                     d_hash_table_value_index[id], hash_table_value_tensors[id]->get_ptr(),
                     d_hash_table_value[id]);
    }

    // sync wait
    sync_all_gpus(device_resources, context);

    // memcpy from GPU to CPU memory
    size_t key_offset = 0;
    size_t value_offset = 0;
    for (int id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      context.set_device((*device_resources)[id]->get_device_id());

      CK_CUDA_THROW_(cudaMemcpy(hash_table_key + key_offset, d_hash_table_key[id],
                                count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost));
      key_offset += count[id];

      CK_CUDA_THROW_(cudaMemcpy(hash_table_value + value_offset, d_hash_table_value[id],
                                count[id] * embedding_vec_size * sizeof(float),
                                cudaMemcpyDeviceToHost));
      value_offset += count[id] * embedding_vec_size;
    }

    for (int id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      context.set_device((*device_resources)[id]->get_device_id());

      CK_CUDA_THROW_(cudaFree(d_hash_table_key[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_value_index[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_value[id]));
      CK_CUDA_THROW_(cudaFree(d_dump_counter[id]));
    }

#ifdef ENABLE_MPI
    int my_rank = 0;
    int n_ranks = 1;
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));

    if (n_ranks > 1) {
      std::unique_ptr<int> displs(new int(n_ranks));
      std::unique_ptr<int> recv_count(new int(n_ranks));
      MPI_Gather(&total_count, 1, MPI_INT, recv_count.get(), 1, MPI_INT, 0, MPI_COMM_WORLD);

      if (my_rank == 0) {
        displs.get()[0] = 0;
        for (int i = 1; i < n_ranks; i++) {
          displs.get()[i] = displs.get()[i - 1] + recv_count.get()[i - 1];
        }
      }

      std::unique_ptr<int> displs_key(new int(n_ranks));
      std::unique_ptr<int> recv_count_key(new int(n_ranks));
      if (my_rank == 0) {
        for (int i = 0; i < n_ranks; i++) {
          recv_count_key.get()[i] = recv_count.get()[i] * sizeof(TypeHashKey);
          displs_key.get()[i] = displs.get()[i] * sizeof(TypeHashKey);
        }
      }

      MPI_Gatherv(hash_table_key, total_count * sizeof(TypeHashKey), MPI_CHAR, hash_table_key,
                  recv_count_key.get(), displs_key.get(), MPI_CHAR, 0, MPI_COMM_WORLD);

      std::unique_ptr<int> displs_value(new int(n_ranks));
      std::unique_ptr<int> recv_count_value(new int(n_ranks));
      if (my_rank == 0) {
        for (int i = 0; i < n_ranks; i++) {
          recv_count_value.get()[i] = recv_count.get()[i] * embedding_vec_size * sizeof(float);
          displs_value.get()[i] = displs.get()[i] * embedding_vec_size * sizeof(float);
        }
      }

      MPI_Gatherv(hash_table_value, total_count * embedding_vec_size * sizeof(float), MPI_CHAR,
                  hash_table_value, recv_count_value.get(), displs_value.get(), MPI_CHAR, 0,
                  MPI_COMM_WORLD);
    }
#endif

    return;
  }

  /**
   * store slot ids. This function is only used by LocalizedSparseEmbeddingHash.
   * @param batch_size batch size for the current mini-batch computation.
   * @param slot_num total slot number in hash table.
   * @param row_offsets_tensors row_offsets tensors of mulitple GPUs (CSR format of input sparse
   * tensors)
   * @param value_index_tensors hash value index tensors of multi GPUs
   * @param slot_id_tensors slot id tensors for multi GPUs
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  template <typename TypeKey>
  void store_slot_id(int batch_size, int slot_num, const std::vector<int> &slot_num_per_gpu,
                     const Tensors<TypeKey> &row_offset_tensors,
                     const Tensors<TypeKey> &value_index_tensors, Tensors<TypeKey> &slot_id_tensors,
                     const std::shared_ptr<GPUResourceGroup> &device_resources,
                     const CudaDeviceContext &context) {
    int local_gpu_count = device_resources->size();
    int total_gpu_count = device_resources->get_total_gpu_count();

    for (int id = 0; id < local_gpu_count; id++) {
      if (slot_num_per_gpu[id] == 0) {
        continue;
      }

      int local_device_id = (*device_resources)[id]->get_device_id();
      int global_id = device_resources->get_global_id(local_device_id);

      dim3 blockSize(64, 1, 1);
      dim3 gridSize((batch_size * slot_num_per_gpu[id] + blockSize.x - 1) / blockSize.x, 1, 1);

      context.set_device(local_device_id);
      store_slot_id_kernel<<<gridSize, blockSize, 0, (*device_resources)[id]->get_stream()>>>(
          batch_size, slot_num, slot_num_per_gpu[id], total_gpu_count, global_id,
          row_offset_tensors[id]->get_ptr(), value_index_tensors[id]->get_ptr(),
          slot_id_tensors[id]->get_ptr());
    }
  }

  /**
   * Initialize one hash table slot and embedding slot on GPU. This function is only used by
   * LocalizedSparseEmbeddingHash.
   * @param slot_id slot id.
   * @param slot_size the size of the slot to be intialized.
   * @param embedding_vec_size embedding vector size.
   * @param offset the start address of the slot in embedding table.
   * @param embedding_table the pointer to the embedding slot.
   * @param hash_table GPU hash table which stores <key, value_index>.
   * @param slot_ids the pointer to the slot ids.
   * @param stream cuda stream.
   */
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void init_embedding_per_slot(int slot_id, size_t slot_size, int embedding_vec_size,
                               size_t key_offset, size_t value_index_offset, float *embedding_table,
                               HashTable<TypeHashKey, TypeHashValueIndex,
                                         std::numeric_limits<TypeHashKey>::max()> *hash_table,
                               TypeHashValueIndex *slot_ids, cudaStream_t stream) {
    TypeHashKey *hash_keys;
    CK_CUDA_THROW_(cudaMalloc(&hash_keys, slot_size * sizeof(TypeHashKey)));
    TypeHashValueIndex *hash_value_indices;
    CK_CUDA_THROW_(cudaMalloc(&hash_value_indices, slot_size * sizeof(TypeHashValueIndex)));
    PinnedBuffer<float> embedding_init(slot_size * embedding_vec_size);

    float up_bound = sqrt(1.f / slot_size);
    HugeCTR::UnifiedDataSimulator<float> fdata_sim(-up_bound, up_bound);

    for (size_t i = 0; i < (slot_size * embedding_vec_size); i++) {
      embedding_init.get()[i] = fdata_sim.get_num();
      // embedding_init.get()[i] = key_offset+i/embedding_vec_size; // just for verificatoin
    }
    CK_CUDA_THROW_(cudaMemcpyAsync(embedding_table, embedding_init.get(),
                                   slot_size * embedding_vec_size * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));

    memset_liner(hash_keys, (TypeHashKey)key_offset, (TypeHashKey)1, slot_size, stream);
    memset_liner(hash_value_indices, (TypeHashValueIndex)value_index_offset, (TypeHashValueIndex)1,
                 slot_size, stream);
    hash_table->insert(hash_keys, hash_value_indices, slot_size, stream);
    size_t value_head = hash_table->add_value_head(slot_size);

    memset_const(slot_ids, (TypeHashValueIndex)slot_id, slot_size, stream);

    CK_CUDA_THROW_(cudaStreamSynchronize(stream));

    CK_CUDA_THROW_(cudaFree(hash_keys));
    CK_CUDA_THROW_(cudaFree(hash_value_indices));
  }

  /**
   * Initialize the hash table and embedding table on one GPU. This function is only used by
   * LocalizedSparseEmbeddingHash.
   * @param lid the gpu local id.
   * @param gid the gpu global id.
   * @param total_gpu_count total gpu count.
   * @param slot_sizes an array which stores the size of the slots to be intialized.
   * @param embedding_vec_size embedding vector size.
   * @param embedding_table the pointer to the embedding table.
   * @param hash_table GPU hash table which stores <key, value_index>.
   * @param slot_ids the pointer to the slot ids.
   * @param device_resources GPU device resources.
   */
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void init_embedding_per_gpu(int lid, int gid, int total_gpu_count,
                              const std::vector<size_t> slot_sizes, int embedding_vec_size,
                              float *embedding_table,
                              HashTable<TypeHashKey, TypeHashValueIndex,
                                        std::numeric_limits<TypeHashKey>::max()> *hash_table,
                              TypeHashValueIndex *slot_ids,
                              const std::shared_ptr<GPUResourceGroup> &device_resources) {
    CudaDeviceContext context((*device_resources)[lid]->get_device_id());

    size_t key_offset = 0;
    size_t value_index_offset = 0;
    for (int i = 0; i < (int)(slot_sizes.size()); i++) {
      size_t slot_size = slot_sizes[i];
      if ((i % total_gpu_count) == gid) {
        MESSAGE_("gpu" + std::to_string(gid) + " start to init embedding of slot" +
                 std::to_string(i) + " , slot_size=" + std::to_string(slot_size) +
                 ", key_offset=" + std::to_string(key_offset) +
                 ", value_index_offset=" + std::to_string(value_index_offset));
        init_embedding_per_slot(i, slot_size, embedding_vec_size, key_offset, value_index_offset,
                                embedding_table, hash_table, slot_ids,
                                (*device_resources)[lid]->get_stream());
        value_index_offset += slot_size;
        embedding_table += slot_size * embedding_vec_size;
        slot_ids += slot_size;
      }
      key_offset += slot_size;
    }
  }

  /**
   * Initialize the hash table and embedding table on local GPUs. This function is only used by
   * LocalizedSparseEmbeddingHash.
   * @param slot_sizes an array which stores the size of the slots to be intialized.
   * @param embedding_vec_size embedding vector size.
   * @param hash_table_value_tensors embedding table tensors.
   * @param hash_tables GPU hash tables which stores <key, value_index>.
   * @param hash_table_slot_id_tensors slot ids tensors.
   * @param device_resources GPU device resources.
   */
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void init_embedding(
      const std::vector<size_t> slot_sizes, int embedding_vec_size,
      Tensors<float> &hash_table_value_tensors,
      std::vector<std::shared_ptr<HashTable<TypeHashKey, TypeHashValueIndex,
                                            std::numeric_limits<TypeHashKey>::max()>>> &hash_tables,
      Tensors<TypeHashValueIndex> &hash_table_slot_id_tensors,
      const std::shared_ptr<GPUResourceGroup> &device_resources) {
    int local_gpu_count = device_resources->size();
    int total_gpu_count = device_resources->get_total_gpu_count();

#ifndef NDEBUG
    MESSAGE_("local_gpu_count=" + std::to_string(local_gpu_count) +
             ", total_gpu_count=" + std::to_string(total_gpu_count));
#endif

    for (int id = 0; id < local_gpu_count; id++) {
      int device_id = (*device_resources)[id]->get_device_id();
      int global_id = device_resources->get_global_id(device_id);

#ifndef NDEBUG
      MESSAGE_("id=" + std::to_string(id) + ", device_id=" + std::to_string(device_id) +
               ", global_id=" + std::to_string(global_id));
#endif

      init_embedding_per_gpu(id, global_id, total_gpu_count, slot_sizes, embedding_vec_size,
                             hash_table_value_tensors[id]->get_ptr(), hash_tables[id].get(),
                             hash_table_slot_id_tensors[id]->get_ptr(), device_resources);
    }

    return;
  }

  /**
   * Initialize the embedding table on local GPUs.
   * @param max_vocabulary_size_per_gpu max vocabulary size per GPU.
   * @param embedding_vec_size embedding vector size.
   * @param hash_table_value_tensors embedding table tensors.
   * @param device_resources GPU device resources.
   */
  void init_embedding(size_t max_vocabulary_size_per_gpu, int embedding_vec_size,
                      Tensors<float> &hash_table_value_tensors,
                      const std::shared_ptr<GPUResourceGroup> &device_resources) {
    size_t num = max_vocabulary_size_per_gpu * embedding_vec_size;
    PinnedBuffer<float> h_hash_table_value(num);

    HugeCTR::UnifiedDataSimulator<float> fdata_sim(-0.05, 0.05);
    for (size_t i = 0; i < num; i++) {
      h_hash_table_value[i] = fdata_sim.get_num();
    }

    CudaDeviceContext context((*device_resources)[0]->get_device_id());
    int local_gpu_count = device_resources->size();
    for (int id = 0; id < local_gpu_count; id++) {
      int cur_device = (*device_resources)[id]->get_device_id();
      context.set_device(cur_device);

      MESSAGE_("gpu" + std::to_string(id) + " start to init embedding");

      CK_CUDA_THROW_(cudaMemcpyAsync(
          hash_table_value_tensors[id]->get_ptr(), h_hash_table_value.get(), num * sizeof(float),
          cudaMemcpyHostToDevice, (*device_resources)[id]->get_stream()));
    }

    for (int id = 0; id < local_gpu_count; id++) {
      CK_CUDA_THROW_(cudaStreamSynchronize((*device_resources)[id]->get_stream()));
      MESSAGE_("gpu" + std::to_string(id) + " init embedding done");
    }
  }

  /**
   * Calculate the max vocabulary size per GPU.
   * @param total_gpu_count total GPU count.
   * @param local_gpu_count local GPU count.
   * @param slot_sizes an array which stores the size of the slots to be intialized.
   * @param device_resources GPU device resources.
   */
  size_t cal_max_voc_size_per_gpu(int total_gpu_count, int local_gpu_count,
                                  const std::vector<size_t> slot_sizes,
                                  const std::shared_ptr<GPUResourceGroup> &device_resources) {
    size_t max_voc_size = 0;
    for (int id = 0; id < local_gpu_count; id++) {
      int device_id = (*device_resources)[id]->get_device_id();
      int global_id = device_resources->get_global_id(device_id);

      size_t total_size = 0;
      for (int i = 0; i < (int)(slot_sizes.size()); i++) {
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
};  // end of SparseEmbeddingHashFunctors

}  // end of namespace HugeCTR
