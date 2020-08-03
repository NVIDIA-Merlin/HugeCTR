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
#include <embeddings/sparse_embedding_kernels.cuh>
#ifndef NCCL_A2A
#include <faster_gossip_comm/FasterGossipComm/FasterGossipComm.h>
#endif
#include <iomanip>

#include <diagnose.hpp>
#include <gpu_resource.hpp>
#include <hashtable/nv_hashtable.cuh>
#include <pinned_buffer.hpp>
#include <utils.hpp>
#include <cub/cub/device/device_radix_sort.cuh>
#include <cub/cub/device/device_scan.cuh>


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
  size_t sm_count_;

 public:
  /**
   * Ctor of SparseEmbeddingHashFunctors. Copy construction and assigment are disabled.
   */
  SparseEmbeddingHashFunctors() {
    // Assumption: all the GPUs in the server are the same.
    int sm_count;
    CK_CUDA_THROW_(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    sm_count_ = sm_count;
  }

  SparseEmbeddingHashFunctors(SparseEmbeddingHashFunctors &) = delete;
  SparseEmbeddingHashFunctors &operator=(const SparseEmbeddingHashFunctors &) = delete;
  ~SparseEmbeddingHashFunctors() {}

  /**
   * stream sync on multi GPUs.
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device.
   */
  void sync_all_gpus(const GPUResourceGroup &device_resources) const {
    CudaDeviceContext context;

    size_t local_gpu_count = device_resources.size();
    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());
      CK_CUDA_THROW_(cudaStreamSynchronize(device_resources[id].get_stream()));
    }
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
  void forward_per_gpu(size_t batch_size, size_t slot_num, size_t embedding_vec_size, bool train,
                       const TypeHashKey *row_offset, const TypeHashKey *hash_key, size_t nnz,
                       const HashTable<TypeHashKey, TypeHashValueIndex,
                                       std::numeric_limits<TypeHashKey>::max()> &hash_table,
                       const float *hash_table_value, TypeHashValueIndex *hash_value_index,
                       TypeEmbeddingComp *embedding_feature, cudaStream_t stream) {
    try {
      // get hash_value_index from hash_table by hash_key

      if (train) {
        hash_table.get_insert(hash_key, hash_value_index, nnz, stream);
      } else {
        hash_table.get(hash_key, hash_value_index, nnz, stream);
      }

      const size_t grid_size = batch_size;  // each block corresponds to a sample
      // do sum reduction
      if (std::is_same<TypeEmbeddingComp, __half>::value && embedding_vec_size % 2 == 0) {
        const size_t block_size = embedding_vec_size / 2;
        forward_sum_align2_kernel<<<grid_size, block_size, 0, stream>>>(
            batch_size, slot_num, embedding_vec_size / 2, row_offset, hash_value_index,
            hash_table_value, embedding_feature);
      } else {
        const size_t block_size =
            embedding_vec_size;  // each thread corresponds to one element in an embedding vector
        forward_sum_kernel<<<grid_size, block_size, 0, stream>>>(
            batch_size, slot_num, embedding_vec_size, row_offset, hash_value_index,
            hash_table_value, embedding_feature);
      }
      // for combiner=mean, call forward_scale() after this forward() and NCCL all-reduce operation

    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
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
  void forward_per_gpu(size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner,
                       bool train, const TypeHashKey *row_offset, const TypeHashKey *hash_key,
                       size_t nnz,
                       const HashTable<TypeHashKey, TypeHashValueIndex,
                                       std::numeric_limits<TypeHashKey>::max()> &hash_table,
                       const float *hash_table_value, TypeHashValueIndex *hash_value_index,
                       TypeEmbeddingComp *embedding_feature, cudaStream_t stream) {
    try {
      // get hash_value_index from hash_table by hash_key
      if (train) {
        hash_table.get_insert(hash_key, hash_value_index, nnz, stream);
      } else {
        hash_table.get(hash_key, hash_value_index, nnz, stream);
      }

      // do sum reduction
      const size_t grid_size = batch_size;  // each block corresponds to a sample
      if (combiner == 0) {
        if (std::is_same<TypeEmbeddingComp, __half>::value && embedding_vec_size % 2 == 0) {
          const size_t block_size =
              embedding_vec_size /
              2;  // each thread corresponds to one element in an embedding vector
          forward_sum_align2_kernel<<<grid_size, block_size, 0, stream>>>(
              batch_size, slot_num, embedding_vec_size / 2, row_offset, hash_value_index,
              hash_table_value, embedding_feature);
        } else {
          const size_t block_size = embedding_vec_size;
          forward_sum_kernel<<<grid_size, block_size, 0, stream>>>(
              batch_size, slot_num, embedding_vec_size, row_offset, hash_value_index,
              hash_table_value, embedding_feature);
        }
      } else if (combiner == 1) {
        if (std::is_same<TypeEmbeddingComp, __half>::value && embedding_vec_size % 2 == 0) {
          const size_t block_size = embedding_vec_size / 2;
          forward_mean_align2_kernel<<<grid_size, block_size, 0, stream>>>(
              batch_size, slot_num, embedding_vec_size / 2, row_offset, hash_value_index,
              hash_table_value, embedding_feature);
        } else {
          const size_t block_size = embedding_vec_size;
          forward_mean_kernel<<<grid_size, block_size, 0, stream>>>(
              batch_size, slot_num, embedding_vec_size, row_offset, hash_value_index,
              hash_table_value, embedding_feature);
        }
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
  void forward_mapping_per_gpu(size_t batch_size, size_t slot_num, const TypeHashKey *hash_key,
                               size_t nnz, const uint32_t *mapping_offsets,
                               TypeHashValueIndex *hash_value_index, cudaStream_t stream) {
    // remove hashtable get_insert(), and do linear mapping between key and value_index
    hash_key_value_index_mapping_kernel<<<(nnz + 255) / 256, 256, 0, stream>>>(
        nnz, slot_num, mapping_offsets, hash_key, hash_value_index);

    return;
  }

  /**
   * forward propagation for LocalizedSlotSparseEmbeddingOneHot (per GPU).
   * fuse (forward_sum_kernel + all2all + forward_reorder) into one kernel.
   * Only support single node currently.
   * @param id local gpu id
   * @param local_gpu_count local gpu count
   * @param batch_size batch size for the current mini-batch computation
   * @param batch_size_per_gpu batchsize per gpu
   * @param slot_num total slots number
   * @param slot_num_per_gpu the number of slots for each GPU
   * @param embedding_vec_size embedding vector size.
   * @param combiner 0-sum; 1-mean
   * @param row_offsets row_offset (CSR format of input sparse tensors)
   * @param hash_value_index hash table value_index(row index of embedding)
   * @param hash_table_value hash table value, which represents embedding vector
   * @param embedding_features embedding features of all gpus (output)
   * @param stream cuda stream
   */
  template <typename TypeHashKey, typename TypeHashValueIndex, typename TypeEmbeddingComp>
  void forward_fuse_per_gpu(size_t id, size_t local_gpu_count, size_t batch_size,
                            size_t batch_size_per_gpu, size_t slot_num, size_t slot_num_per_gpu,
                            size_t embedding_vec_size, int combiner, const TypeHashKey *row_offset,
                            const TypeHashValueIndex *hash_value_index,
                            const float *hash_table_value, TypeEmbeddingComp **embedding_features,
                            cudaStream_t stream) {
    int maxActiveBlocks;
    void *func;
    size_t embedding_vec_size_opt;

    if (std::is_same<TypeEmbeddingComp, __half>::value && embedding_vec_size % 2 == 0) {
      embedding_vec_size_opt = embedding_vec_size / 2;
      CK_CUDA_THROW_(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &maxActiveBlocks, forward_sum_fuse_align2_kernel<TypeHashKey, TypeHashValueIndex>,
          embedding_vec_size_opt, 0));
      func = (void *)forward_sum_fuse_align2_kernel<TypeHashKey, TypeHashValueIndex>;
    } else {
      embedding_vec_size_opt = embedding_vec_size;
      CK_CUDA_THROW_(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &maxActiveBlocks,
          forward_sum_fuse_kernel<TypeHashKey, TypeHashValueIndex, TypeEmbeddingComp>,
          embedding_vec_size_opt, 0));
      func = (void *)forward_sum_fuse_kernel<TypeHashKey, TypeHashValueIndex, TypeEmbeddingComp>;
    }

    const size_t block_size = embedding_vec_size_opt;
    const size_t grid_size = min(batch_size, static_cast<size_t>(sm_count_ * maxActiveBlocks));

    void *kargs[11];
    kargs[0] = (void *)&id;
    kargs[1] = (void *)&local_gpu_count;
    kargs[2] = (void *)&batch_size;
    kargs[3] = (void *)&batch_size_per_gpu;
    kargs[4] = (void *)&slot_num;
    kargs[5] = (void *)&slot_num_per_gpu;
    kargs[6] = (void *)&embedding_vec_size_opt;
    kargs[7] = (void *)&row_offset;
    kargs[8] = (void *)&hash_value_index;
    kargs[9] = (void *)&hash_table_value;
    kargs[10] = (void *)&embedding_features;

    try {
      if (combiner == 0) {
        CK_CUDA_THROW_(cudaLaunchKernel(func, grid_size, block_size, kargs, 0, stream));
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
  void forward_scale(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                     const TensorPtrs<TypeHashKey> &row_offset_allreduce_tensors,
                     const TensorPtrs<TypeEmbeddingComp> &output_tensors,
                     const GPUResourceGroup &device_resources) {
    CudaDeviceContext context;
    size_t local_gpu_count = device_resources.size();
    size_t total_gpu_count = device_resources.get_total_gpu_count();
    size_t batchsize_per_gpu = batch_size / total_gpu_count;

    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());

      const auto &row_offset =
          row_offset_allreduce_tensors[id]->get_ptr() + id * batchsize_per_gpu * slot_num;
      auto embedding_feature = output_tensors[id]->get_ptr();
      const auto &stream = device_resources[id].get_stream();

      const size_t grid_size = batchsize_per_gpu;
      if (std::is_same<TypeEmbeddingComp, __half>::value && embedding_vec_size % 2 == 0) {
        const size_t block_size = embedding_vec_size / 2;
        forward_scale_align2_kernel<<<grid_size, block_size, 0, stream>>>(
            batchsize_per_gpu, slot_num, embedding_vec_size / 2, row_offset, embedding_feature);
      } else {
        const size_t block_size = embedding_vec_size;
        forward_scale_kernel<<<grid_size, block_size, 0, stream>>>(
            batchsize_per_gpu, slot_num, embedding_vec_size, row_offset, embedding_feature);
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
  void backward(size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner,
                const TensorPtrs<TypeHashKey> &row_offset_allreduce_tensors,
                const TensorPtrs<TypeEmbeddingComp> &embedding_feature_tensors,
                const TensorPtrs<TypeEmbeddingComp> &wgrad_tensors,
                const GPUResourceGroup &device_resources) {
    CudaDeviceContext context;
    size_t local_gpu_count = device_resources.size();

    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());
      const auto &stream = device_resources[id].get_stream();
      const auto &top_grad = embedding_feature_tensors[id]->get_ptr();
      const auto &row_offset = row_offset_allreduce_tensors[id]->get_ptr();
      auto wgrad = wgrad_tensors[id]->get_ptr();

      const size_t grid_size = batch_size;  // each block corresponds to a sample

      if (combiner == 0)  // sum
      {
        if (std::is_same<TypeEmbeddingComp, __half>::value && embedding_vec_size % 2 == 0) {
          const size_t block_size =
              embedding_vec_size /
              2;  // each thread corresponds to one element in an embedding vetor
          backward_sum_align2_kernel<<<grid_size, block_size, 0, stream>>>(
              batch_size, slot_num, embedding_vec_size / 2, top_grad, wgrad);
        } else {
          const size_t block_size = embedding_vec_size;
          backward_sum_kernel<<<grid_size, block_size, 0, stream>>>(
              batch_size, slot_num, embedding_vec_size, top_grad, wgrad);
        }
      } else if (combiner == 1)  // mean
      {
        if (std::is_same<TypeEmbeddingComp, __half>::value && embedding_vec_size % 2 == 0) {
          const size_t block_size = embedding_vec_size / 2;
          backward_mean_align2_kernel<<<grid_size, block_size, 0, stream>>>(
              batch_size, slot_num, embedding_vec_size / 2, row_offset, top_grad, wgrad);
        } else {
          const size_t block_size = embedding_vec_size;
          backward_mean_kernel<<<grid_size, block_size, 0, stream>>>(
              batch_size, slot_num, embedding_vec_size, row_offset, top_grad, wgrad);
        }
      } else {
        CK_THROW_(Error_t::WrongInput, "Invalid combiner type ");
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
  void backward(size_t batch_size, const std::vector<size_t> &slot_num_per_gpu,
                size_t embedding_vec_size, int combiner,
                const TensorPtrs<TypeHashKey> &row_offset_allreduce_tensors,
                const TensorPtrs<TypeEmbeddingComp> &embedding_feature_tensors,
                TensorPtrs<TypeEmbeddingComp> &wgrad_tensors,
                const GPUResourceGroup &device_resources) {
    CudaDeviceContext context;
    size_t local_gpu_count = device_resources.size();

    for (size_t id = 0; id < local_gpu_count; id++) {
      if (slot_num_per_gpu[id] == 0) {
        continue;
      }

      context.set_device(device_resources[id].get_device_id());
      const auto &stream = device_resources[id].get_stream();
      const auto &top_grad = embedding_feature_tensors[id]->get_ptr();
      const auto &row_offset = row_offset_allreduce_tensors[id]->get_ptr();
      auto wgrad = wgrad_tensors[id]->get_ptr();

      const size_t grid_size = batch_size;  // each block corresponds to a sample

      if (combiner == 0)  // sum
      {
        if (std::is_same<TypeEmbeddingComp, __half>::value && embedding_vec_size % 2 == 0) {
          const size_t block_size =
              embedding_vec_size /
              2;  // each thread corresponds to one element in an embedding vetor
          backward_sum_align2_kernel<<<grid_size, block_size, 0, stream>>>(
              batch_size, slot_num_per_gpu[id], embedding_vec_size / 2, top_grad, wgrad);
        } else {
          const size_t block_size = embedding_vec_size;
          backward_sum_kernel<<<grid_size, block_size, 0, stream>>>(
              batch_size, slot_num_per_gpu[id], embedding_vec_size, top_grad, wgrad);
        }
      } else if (combiner == 1)  // mean
      {
        if (std::is_same<TypeEmbeddingComp, __half>::value && embedding_vec_size % 2 == 0) {
          const size_t block_size = embedding_vec_size / 2;
          backward_mean_align2_kernel<<<grid_size, block_size, 0, stream>>>(
              batch_size, slot_num_per_gpu[id], embedding_vec_size / 2, row_offset, top_grad,
              wgrad);
        } else {
          const size_t block_size = embedding_vec_size;
          backward_mean_kernel<<<grid_size, block_size, 0, stream>>>(
              batch_size, slot_num_per_gpu[id], embedding_vec_size, row_offset, top_grad, wgrad);
        }
      } else {
        CK_THROW_(Error_t::WrongInput, "Invalid combiner type ");
      }
    }

    return;
  }

  /**
   * backward propagation for LocalizedSlotSparseEmbeddingOneHot (per gpu).
   * fuse (backward_reorder + all2all + backward_xxx_kernel) into one kernel.
   * Only support single node currently.
   * @param id local gpu id
   * @param local_gpu_count local gpu count
   * @param batch_size batch size for the current mini-batch computation
   * @param batch_size_per_gpu batchsize per gpu
   * @param slot_num total slots number
   * @param slot_num_per_gpu the number of slots for each GPU
   * @param embedding_vec_size embedding vector size.
   * @param combiner 0-sum; 1-mean
   * @param embedding_features embedding features of all gpus (output)
   * @param wgrad wgrad, the output of this function.
   * @param stream cuda stream
   */
  template <typename TypeEmbeddingComp>
  void backward_fuse_per_gpu(size_t id, size_t local_gpu_count, size_t batch_size,
                             size_t batch_size_per_gpu, size_t slot_num, size_t slot_num_per_gpu,
                             size_t embedding_vec_size, int combiner,
                             TypeEmbeddingComp **embedding_features, TypeEmbeddingComp *wgrad,
                             cudaStream_t stream) {
    int maxActiveBlocks;
    void *func;
    size_t embedding_vec_size_opt;

    if (std::is_same<TypeEmbeddingComp, __half>::value && embedding_vec_size % 2 == 0) {
      embedding_vec_size_opt = embedding_vec_size / 2;
      CK_CUDA_THROW_(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &maxActiveBlocks, backward_sum_fuse_align2_kernel, embedding_vec_size_opt, 0));
      func = (void *)backward_sum_fuse_align2_kernel;
    } else {
      embedding_vec_size_opt = embedding_vec_size;
      CK_CUDA_THROW_(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &maxActiveBlocks, backward_sum_fuse_kernel<TypeEmbeddingComp>, embedding_vec_size_opt,
          0));
      func = (void *)backward_sum_fuse_kernel<TypeEmbeddingComp>;
    }

    const size_t block_size = embedding_vec_size_opt;
    const size_t grid_size = min(batch_size, static_cast<size_t>(sm_count_ * maxActiveBlocks));

    void *kargs[9];
    kargs[0] = (void *)&id;
    kargs[1] = (void *)&local_gpu_count;
    kargs[2] = (void *)&batch_size;
    kargs[3] = (void *)&batch_size_per_gpu;
    kargs[4] = (void *)&slot_num;
    kargs[5] = (void *)&slot_num_per_gpu;
    kargs[6] = (void *)&embedding_vec_size_opt;
    kargs[7] = (void *)&embedding_features;
    kargs[8] = (void *)&wgrad;

    try {
      if (combiner == 0) {
        CK_CUDA_THROW_(cudaLaunchKernel(func, grid_size, block_size, kargs, 0, stream));
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
  template <typename TypeHashKey, typename TypeHashValueIndex, typename TypeEmbeddingComp>
  void update_params(cudaStream_t stream, size_t batch_size, size_t slot_num,
                     size_t embedding_vec_size, size_t max_vocabulary_size_per_gpu,
                     OptParams<TypeEmbeddingComp> &opt_params, size_t nnz,
                     const TypeHashKey *row_offset, TypeHashValueIndex *hash_value_index,
                     TypeHashKey *sample_id, TypeHashKey *sample_id_sort,
                     TypeHashValueIndex *hash_value_index_sort,
                     uint32_t *hash_value_index_count_offset, uint32_t *new_hash_value_flag,
                     uint32_t *hash_value_flag_sumed, uint32_t *hash_value_index_count_counter,
                     void *temp_storage_sort, size_t temp_storage_sort_bytes,
                     void *temp_storage_scan, size_t temp_storage_scan_bytes,
                     const TypeEmbeddingComp *wgrad, TypeHashValueIndex *deltaw_hash_value_index,
                     float *deltaw, float *hash_table_value) {
    if (slot_num == 0) {
      return;
    }

    size_t block_size, grid_size;

    try {
      // step1: expand sample IDs
      block_size = 64;
      grid_size = (batch_size * slot_num - 1) / block_size + 1;
      sample_id_expand_kernel<<<grid_size, block_size, 0, stream>>>(batch_size, slot_num,
                                                                    row_offset, sample_id);

#ifdef SGD_ATOMIC
      if (opt_params.optimizer == Optimizer_t::SGD) {  // for SGD, do atomic update
        const size_t block_size = embedding_vec_size;
        const size_t grid_size = min(max(1ul, nnz), sm_count_ * 32);

        float lr_scale = opt_params.lr / opt_params.scaler;
        opt_sgd_atomic_kernel<<<grid_size, block_size, 0, stream>>>(
            nnz, embedding_vec_size, lr_scale, hash_value_index, sample_id, wgrad,
            hash_table_value);
      } else
#endif
      {
        // step3: sort by hash_value_index
        int end_bit = static_cast<int>(log2(static_cast<float>(max_vocabulary_size_per_gpu))) + 1;
        CK_CUDA_THROW_(cub::DeviceRadixSort::SortPairs(
            (void *)temp_storage_sort, temp_storage_sort_bytes, hash_value_index,
            hash_value_index_sort, sample_id, sample_id_sort, nnz, 0, end_bit, stream, false));

        // step4: count the number for each unduplicated hash_value_index
        CK_CUDA_THROW_(
            cudaMemsetAsync(hash_value_index_count_counter, 0, sizeof(uint32_t), stream));

        constexpr size_t max_grid_size = 384;
        block_size = 256;
        grid_size = min(max_grid_size, (nnz - 1) / block_size + 1);

        value_count_kernel_1<<<grid_size, block_size, 0, stream>>>(nnz, hash_value_index_sort,
                                                                   new_hash_value_flag);

        // prefix_sum
        CK_CUDA_THROW_(cub::DeviceScan::InclusiveSum((void *)temp_storage_scan,
                                                     temp_storage_scan_bytes, new_hash_value_flag,
                                                     hash_value_flag_sumed, nnz, stream));

        value_count_kernel_2<<<grid_size, block_size, 0, stream>>>(
            nnz, new_hash_value_flag, hash_value_flag_sumed, hash_value_index_count_offset,
            hash_value_index_count_counter);

        uint32_t hash_hash_value_index_count_num = 0;
        // this async memcpy will not perform as a async operation because the host memory is not
        // a pinned memroy
        CK_CUDA_THROW_(cudaMemcpyAsync(&hash_hash_value_index_count_num,
                                       hash_value_index_count_counter, sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, stream));

        // step5: use optimizer method to compute deltaw, and record corresponding
        // deltaw_hash_value_index
        block_size = embedding_vec_size;
        grid_size = max(1, hash_hash_value_index_count_num);

        if (opt_params.global_update) {
          switch (opt_params.optimizer) {
            case Optimizer_t::Adam:  // adam
            {
              float alpha_t =
                  opt_params.lr *
                  sqrt(1 -
                       pow(opt_params.hyperparams.adam.beta2, opt_params.hyperparams.adam.times)) /
                  (1 - pow(opt_params.hyperparams.adam.beta1, opt_params.hyperparams.adam.times));
              // update target mi and vi
              opt_adam_kernel_global<<<grid_size, block_size, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.hyperparams.adam,
                  sample_id_sort, hash_value_index_sort, hash_value_index_count_offset, wgrad,
                  opt_params.scaler);
              // all update according to the mi vi
              adam_update_kernel_global<<<1024, 256, 0, stream>>>(
                  embedding_vec_size, max_vocabulary_size_per_gpu, opt_params.hyperparams.adam,
                  alpha_t, hash_table_value);
              break;
            }
            case Optimizer_t::MomentumSGD:  // momentum sgd
              opt_momentum_sgd_kernel_global<<<grid_size, block_size, 0, stream>>>(
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
              nesterov_local_update_kernel_global<<<grid_size, block_size, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                  opt_params.hyperparams.nesterov, sample_id_sort, hash_value_index_sort,
                  hash_value_index_count_offset, wgrad, hash_table_value, opt_params.scaler);
              break;
#ifndef SGD_ATOMIC
            case Optimizer_t::SGD:
              opt_sgd_kernel_global<<<grid_size, block_size, 0, stream>>>(
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
            {
              float alpha_t =
                  opt_params.lr *
                  sqrt(1 -
                       pow(opt_params.hyperparams.adam.beta2, opt_params.hyperparams.adam.times)) /
                  (1 - pow(opt_params.hyperparams.adam.beta1, opt_params.hyperparams.adam.times));

              opt_adam_kernel<<<grid_size, block_size, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.hyperparams.adam,
                  alpha_t, sample_id_sort, hash_value_index_sort, hash_value_index_count_offset,
                  wgrad, deltaw_hash_value_index, deltaw, opt_params.scaler);
              break;
            }
            case Optimizer_t::MomentumSGD:  // momentum sgd
              opt_momentum_sgd_kernel<<<grid_size, block_size, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                  opt_params.hyperparams.momentum, sample_id_sort, hash_value_index_sort,
                  hash_value_index_count_offset, wgrad, deltaw_hash_value_index, deltaw,
                  opt_params.scaler);
              break;
            case Optimizer_t::Nesterov:  // nesterov
              opt_nesterov_kernel<<<grid_size, block_size, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                  opt_params.hyperparams.nesterov, sample_id_sort, hash_value_index_sort,
                  hash_value_index_count_offset, wgrad, deltaw_hash_value_index, deltaw,
                  opt_params.scaler);
              break;
#ifndef SGD_ATOMIC
            case Optimizer_t::SGD:
              opt_sgd_kernel<<<grid_size, block_size, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                  sample_id_sort, hash_value_index_sort, hash_value_index_count_offset, wgrad,
                  deltaw_hash_value_index, deltaw, opt_params.scaler);
              break;
#endif
            default:
              CK_THROW_(Error_t::WrongInput, "Error: Invalid opitimizer type");
          }

          // step6: update hash_table_value by deltaw
          block_size = embedding_vec_size;
          grid_size = max(1, hash_hash_value_index_count_num);
          update_kernel<TypeHashValueIndex><<<grid_size, block_size, 0, stream>>>(
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
  void update_params(cudaStream_t stream, size_t embedding_vec_size,
                     const OptParams<TypeEmbeddingComp> &opt_params, size_t nnz,
                     const TypeHashValueIndex *hash_value_index, const TypeEmbeddingComp *wgrad,
                     float *hash_table_value) {
    try {
#ifdef SGD_ATOMIC
      if (opt_params.optimizer == Optimizer_t::SGD) {  // for SGD, do atomic update
        const size_t grid_size = min(max(1ul, nnz), sm_count_ * 32);
        const size_t block_size = embedding_vec_size;

        float lr_scale = opt_params.lr / opt_params.scaler;

        // for one-hot, the sample_id is dedicated.
        opt_sgd_atomic_kernel<<<grid_size, block_size, 0, stream>>>(
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
   */
  template <typename Type>
  void reduce_scatter(size_t recv_count, const TensorPtrs<Type> &send_tensors,
                      const TensorPtrs<Type> &recv_tensors,
                      const GPUResourceGroup &device_resources) {
    size_t local_gpu_count = device_resources.size();
    size_t total_gpu_count = device_resources.get_total_gpu_count();

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

    // for multi GPUs, use NCCL to do Reduce-Scatter(supporting multi-node GPU servers)
    if (total_gpu_count > 1) {
      CK_NCCL_THROW_(ncclGroupStart());
      for (size_t id = 0; id < local_gpu_count; id++) {
        CK_NCCL_THROW_(ncclReduceScatter(send_tensors[id]->get_ptr(),  // send buf
                                         recv_tensors[id]->get_ptr(),  // recv buff
                                         recv_count, type, ncclSum, device_resources[id].get_nccl(),
                                         device_resources[id].get_stream()));
      }
      CK_NCCL_THROW_(ncclGroupEnd());
    }
    // for single GPU, just do memcpyD2D
    else {  // total_gpu_count == 1
      CudaDeviceContext context(device_resources[0].get_device_id());
      CK_CUDA_THROW_(cudaMemcpyAsync(recv_tensors[0]->get_ptr(), send_tensors[0]->get_ptr(),
                                     recv_count * sizeof(Type), cudaMemcpyDeviceToDevice,
                                     device_resources[0].get_stream()));
    }

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
  void all_reduce(size_t send_count, const TensorPtrs<Type> &send_tensors,
                  TensorPtrs<Type> &recv_tensors, const GPUResourceGroup &device_resources) {
    size_t local_gpu_count = device_resources.size();
    size_t total_gpu_count = device_resources.get_total_gpu_count();

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

    // for multi GPUs, use NCCL to do all_reduce (supporting multi-node GPU servers)
    if (total_gpu_count > 1) {
      CK_NCCL_THROW_(ncclGroupStart());
      for (size_t id = 0; id < local_gpu_count; id++) {
        CK_NCCL_THROW_(ncclAllReduce(send_tensors[id]->get_ptr(), recv_tensors[id]->get_ptr(),
                                     send_count, type, ncclSum, device_resources[id].get_nccl(),
                                     device_resources[id].get_stream()));
      }
      CK_NCCL_THROW_(ncclGroupEnd());
    }
    // for single GPU, just do memcpyD2D
    else {  // total_gpu_count == 1
      CudaDeviceContext context(device_resources[0].get_device_id());
      CK_CUDA_THROW_(cudaMemcpyAsync(recv_tensors[0]->get_ptr(), send_tensors[0]->get_ptr(),
                                     send_count * sizeof(Type), cudaMemcpyDeviceToDevice,
                                     device_resources[0].get_stream()));
    }

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
  void all_gather(size_t send_count, const TensorPtrs<Type> &send_tensors,
                  TensorPtrs<Type> &recv_tensors, const GPUResourceGroup &device_resources) {
    size_t local_gpu_count = device_resources.size();
    size_t total_gpu_count = device_resources.get_total_gpu_count();

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

    // for multi GPUs, use NCCL to do All-Gather
    if (total_gpu_count > 1) {
      CK_NCCL_THROW_(ncclGroupStart());
      for (size_t id = 0; id < local_gpu_count; id++) {
        CK_NCCL_THROW_(ncclAllGather(send_tensors[id]->get_ptr(),  // send buff
                                     recv_tensors[id]->get_ptr(),  // recv buff
                                     send_count, type, device_resources[id].get_nccl(),
                                     device_resources[id].get_stream()));
      }
      CK_NCCL_THROW_(ncclGroupEnd());
    }
    // for single GPU, just do memcpyD2D
    else {  // total_gpu_count == 1
      CudaDeviceContext context(device_resources[0].get_device_id());
      CK_CUDA_THROW_(cudaMemcpyAsync(recv_tensors[0]->get_ptr(), send_tensors[0]->get_ptr(),
                                     send_count * sizeof(Type), cudaMemcpyDeviceToDevice,
                                     device_resources[0].get_stream()));
    }

    return;
  }

#ifdef NCCL_A2A  // use nccl all2all

#ifndef ENABLE_MPI  // without MPI (only for single node)
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
  void all2all_forward(size_t batch_size_per_gpu, const std::vector<size_t> &slot_num_per_gpu,
                       size_t embedding_vec_size, const TensorPtrs<Type> &send_tensors,
                       const TensorPtrs<Type> &recv_tensors,
                       const GPUResourceGroup &device_resources) {
    std::vector<int> device_list = device_resources.get_device_list();
    size_t local_gpu_count = device_list.size();

    // Fill in partition table, ith Topo GPU to jth Topo GPU
    std::vector<std::vector<size_t>> table(local_gpu_count, std::vector<size_t>(local_gpu_count));
    for (size_t i = 0; i < local_gpu_count; i++) {
      size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu[i] * embedding_vec_size;
      for (size_t j = 0; j < local_gpu_count; j++) {
        table[i][j] = element_per_send;
      }
    }

#ifndef NDEBUG
    std::cout << "nccl all2all forward table:" << std::endl;
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < local_gpu_count; j++) {
        std::cout << table[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
#endif

    std::vector<Type *> src(local_gpu_count);
    std::vector<Type *> dst(local_gpu_count);
    for (size_t id = 0; id < local_gpu_count; id++) {
      src[id] = send_tensors[id]->get_ptr();
      dst[id] = recv_tensors[id]->get_ptr();
    }
    std::vector<std::vector<Type *>> src_pos(local_gpu_count, std::vector<Type *>(local_gpu_count));
    std::vector<std::vector<Type *>> dst_pos(local_gpu_count, std::vector<Type *>(local_gpu_count));
    // Calculate the src offset pointer from each GPU to each other
    for (size_t i = 0; i < local_gpu_count; i++) {
      size_t src_offset = 0;
      for (size_t j = 0; j < local_gpu_count; j++) {
        src_pos[i][j] = src[i] + src_offset;
        src_offset += table[i][j];
      }
    }
    // Calculate the dst offset pointer from each GPU to each other
    for (size_t i = 0; i < local_gpu_count; i++) {
      size_t dst_offset = 0;
      for (size_t j = 0; j < local_gpu_count; j++) {
        dst_pos[i][j] = dst[i] + dst_offset;
        dst_offset += table[j][i];
      }
    }

#ifndef NDEBUG
    std::cout << "nccl all2all forward src_pos:" << std::endl;
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < local_gpu_count; j++) {
        std::cout << src_pos[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "nccl all2all forward dst_pos:" << std::endl;
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < local_gpu_count; j++) {
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
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < local_gpu_count; j++) {
        CK_NCCL_THROW_(ncclSend(src_pos[i][j], table[i][j], type, j, device_resources[i].get_nccl(),
                                device_resources[i].get_stream()));
        CK_NCCL_THROW_(ncclRecv(dst_pos[i][j], table[j][i], type, j, device_resources[i].get_nccl(),
                                device_resources[i].get_stream()));
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
  void all2all_backward(size_t batch_size_per_gpu, const std::vector<size_t> &slot_num_per_gpu,
                        size_t embedding_vec_size, const TensorPtrs<Type> &send_tensors,
                        TensorPtrs<Type> &recv_tensors, const GPUResourceGroup &device_resources) {
    std::vector<int> device_list = device_resources.get_device_list();
    size_t local_gpu_count = device_list.size();

    // Fill in partition table, ith Topo GPU to jth Topo GPU
    std::vector<std::vector<size_t>> table(local_gpu_count, std::vector<size_t>(local_gpu_count));
    for (size_t i = 0; i < local_gpu_count; i++) {
      size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu[i] * embedding_vec_size;
      for (size_t j = 0; j < local_gpu_count; j++) {
        table[j][i] = element_per_send;
      }
    }

#ifndef NDEBUG
    std::cout << "nccl all2all backward table:" << std::endl;
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < local_gpu_count; j++) {
        std::cout << table[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
#endif

    std::vector<Type *> src(local_gpu_count);
    std::vector<Type *> dst(local_gpu_count);
    for (size_t id = 0; id < local_gpu_count; id++) {
      src[id] = send_tensors[id]->get_ptr();
      dst[id] = recv_tensors[id]->get_ptr();
    }
    std::vector<std::vector<Type *>> src_pos(local_gpu_count, std::vector<Type *>(local_gpu_count));
    std::vector<std::vector<Type *>> dst_pos(local_gpu_count, std::vector<Type *>(local_gpu_count));
    // Calculate the src offset pointer from each GPU to each other
    for (size_t i = 0; i < local_gpu_count; i++) {
      size_t src_offset = 0;
      for (size_t j = 0; j < local_gpu_count; j++) {
        src_pos[i][j] = src[i] + src_offset;
        src_offset += table[i][j];
      }
    }
    // Calculate the dst offset pointer from each GPU to each other
    for (size_t i = 0; i < local_gpu_count; i++) {
      size_t dst_offset = 0;
      for (size_t j = 0; j < local_gpu_count; j++) {
        dst_pos[i][j] = dst[i] + dst_offset;
        dst_offset += table[j][i];
      }
    }

#ifndef NDEBUG
    std::cout << "nccl all2all backward src_pos:" << std::endl;
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < local_gpu_count; j++) {
        std::cout << src_pos[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "nccl all2all backward dst_pos:" << std::endl;
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < local_gpu_count; j++) {
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
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < local_gpu_count; j++) {
        CK_NCCL_THROW_(ncclSend(src_pos[i][j], table[i][j], type, j, device_resources[i].get_nccl(),
                                device_resources[i].get_stream()));
        CK_NCCL_THROW_(ncclRecv(dst_pos[i][j], table[j][i], type, j, device_resources[i].get_nccl(),
                                device_resources[i].get_stream()));
      }
    }
    CK_NCCL_THROW_(ncclGroupEnd());

    return;
  }

#else  // for mpirun (for single node or multiple node)

  /**
   * nccl all2all communication for forward.
   * @param batch_size_per_gpu batch size per GPU
   * @param slot_num slot number
   * @param embedding_vec_size embedding vector size
   * @param send_tensors the send tensors of multi GPUs.
   * @param recv_tensors the recv tensors of multi GPUs.
   * @param device_resources all gpus device resources.
   */
  template <typename Type>
  void all2all_forward(size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
                       const TensorPtrs<Type> &send_tensors, TensorPtrs<Type> &recv_tensors,
                       const GPUResourceGroup &device_resources) {
    std::vector<int> device_list = device_resources.get_device_list();
    size_t local_gpu_count = device_list.size();
    size_t total_gpu_count = device_resources.get_total_gpu_count();

    int total_rank = 1;
    int my_rank = 0;
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &total_rank));
    size_t num_proc = device_resources.get_node_count();
    if (num_proc != total_rank) {
      CK_THROW_(Error_t::WrongInput, "Error: the MPI total rank doesn't match the node count");
    }
    if (total_gpu_count != (total_rank * local_gpu_count)) {
      CK_THROW_(Error_t::WrongInput, "Error: the total gpu count doesn't match");
    }

    std::vector<Type *> src(local_gpu_count);
    std::vector<Type *> dst(local_gpu_count);
    for (size_t id = 0; id < local_gpu_count; id++) {
      src[id] = send_tensors[id]->get_ptr();
      dst[id] = recv_tensors[id]->get_ptr();
    }

    std::vector<std::vector<size_t>> send_table(local_gpu_count,
                                                std::vector<size_t>(total_gpu_count));
    std::vector<std::vector<size_t>> recv_table(local_gpu_count,
                                                std::vector<size_t>(total_gpu_count));

    // Fill in sending partition table, ith Topo GPU send to jth global GPU
    for (size_t i = 0; i < local_gpu_count; i++) {
      size_t device_id = device_resources[i].get_device_id();
      size_t global_id = device_resources.get_global_id(device_id);
      size_t slot_num_per_gpu =
          slot_num / total_gpu_count + ((global_id < (slot_num % total_gpu_count)) ? 1 : 0);
      size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu * embedding_vec_size;

      for (size_t j = 0; j < total_gpu_count; j++) {
        send_table[i][j] = element_per_send;
      }
    }

    // Fill in receiving partition table, ith Topo GPU receive from jth global GPU
    for (size_t j = 0; j < total_gpu_count; j++) {
      size_t global_id = j;
      size_t slot_num_per_gpu =
          slot_num / total_gpu_count + ((global_id < (slot_num % total_gpu_count)) ? 1 : 0);
      size_t element_per_recv = batch_size_per_gpu * slot_num_per_gpu * embedding_vec_size;

      for (size_t i = 0; i < local_gpu_count; i++) {
        recv_table[i][j] = element_per_recv;
      }
    }

    std::vector<std::vector<Type *>> src_pos(local_gpu_count, std::vector<Type *>(total_gpu_count));
    std::vector<std::vector<Type *>> dst_pos(local_gpu_count, std::vector<Type *>(total_gpu_count));
    // Calculate the src offset pointer from each GPU to each other
    for (size_t i = 0; i < local_gpu_count; i++) {
      size_t src_offset = 0;
      for (size_t j = 0; j < total_gpu_count; j++) {
        src_pos[i][j] = src[i] + src_offset;
        src_offset += send_table[i][j];
      }
    }
    // Calculate the dst offset pointer from each GPU to each other
    for (size_t i = 0; i < local_gpu_count; i++) {
      size_t dst_offset = 0;
      for (size_t j = 0; j < total_gpu_count; j++) {
        dst_pos[i][j] = dst[i] + dst_offset;
        dst_offset += recv_table[i][j];
      }
    }

#ifndef NDEBUG
    std::cout << "nccl all2all forward src_pos:" << std::endl;
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < total_gpu_count; j++) {
        std::cout << src_pos[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "nccl all2all forward dst_pos:" << std::endl;
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < total_gpu_count; j++) {
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
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < total_gpu_count; j++) {
        CK_NCCL_THROW_(ncclSend(src_pos[i][j], send_table[i][j], type, j,
                                device_resources[i].get_nccl(), device_resources[i].get_stream()));
        CK_NCCL_THROW_(ncclRecv(dst_pos[i][j], recv_table[i][j], type, j,
                                device_resources[i].get_nccl(), device_resources[i].get_stream()));
      }
    }
    CK_NCCL_THROW_(ncclGroupEnd());

    return;
  }

  /**
   * nccl all2all communication for backward
   * @param batch_size_per_gpu batch size per GPU
   * @param slot_num slot number
   * @param embedding_vec_size embedding vector size
   * @param send_tensors the send tensors of multi GPUs.
   * @param recv_tensors the recv tensors of multi GPUs.
   * @param device_resources all gpus device resources.
   */
  template <typename Type>
  void all2all_backward(size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
                        const TensorPtrs<Type> &send_tensors, TensorPtrs<Type> &recv_tensors,
                        const GPUResourceGroup &device_resources) {
    std::vector<int> device_list = device_resources.get_device_list();
    size_t local_gpu_count = device_list.size();
    size_t total_gpu_count = device_resources.get_total_gpu_count();

    int total_rank = 1;
    int my_rank = 0;
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &total_rank));

    size_t num_proc = device_resources.get_node_count();
    if (num_proc != total_rank) {
      CK_THROW_(Error_t::WrongInput, "Error: the MPI total rank doesn't match the node count");
    }
    if (total_gpu_count != (total_rank * local_gpu_count)) {
      CK_THROW_(Error_t::WrongInput, "Error: the total gpu count doesn't match");
    }

    std::vector<Type *> src(local_gpu_count);
    std::vector<Type *> dst(local_gpu_count);
    for (size_t id = 0; id < local_gpu_count; id++) {
      src[id] = send_tensors[id]->get_ptr();
      dst[id] = recv_tensors[id]->get_ptr();
    }

    std::vector<std::vector<size_t>> send_table(local_gpu_count,
                                                std::vector<size_t>(total_gpu_count));
    std::vector<std::vector<size_t>> recv_table(local_gpu_count,
                                                std::vector<size_t>(total_gpu_count));

    // Fill in receiving partition table, ith Topo GPU receive from jth global GPU
    for (size_t i = 0; i < local_gpu_count; i++) {
      size_t device_id = device_resources[i].get_device_id();
      size_t global_id = device_resources.get_global_id(device_id);
      size_t slot_num_per_gpu =
          slot_num / total_gpu_count + ((global_id < (slot_num % total_gpu_count)) ? 1 : 0);
      size_t element_per_recv = batch_size_per_gpu * slot_num_per_gpu * embedding_vec_size;

      for (size_t j = 0; j < total_gpu_count; j++) {
        recv_table[i][j] = element_per_recv;
      }
    }

    // Fill in sending partition table, ith Topo GPU send to jth global GPU
    for (size_t j = 0; j < total_gpu_count; j++) {
      size_t global_id = j;
      size_t slot_num_per_gpu =
          slot_num / total_gpu_count + ((global_id < (slot_num % total_gpu_count)) ? 1 : 0);
      size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu * embedding_vec_size;

      for (size_t i = 0; i < local_gpu_count; i++) {
        send_table[i][j] = element_per_send;
      }
    }

    std::vector<std::vector<Type *>> src_pos(local_gpu_count, std::vector<Type *>(total_gpu_count));
    std::vector<std::vector<Type *>> dst_pos(local_gpu_count, std::vector<Type *>(total_gpu_count));
    // Calculate the src offset pointer from each GPU to each other
    for (size_t i = 0; i < local_gpu_count; i++) {
      size_t src_offset = 0;
      for (size_t j = 0; j < total_gpu_count; j++) {
        src_pos[i][j] = src[i] + src_offset;
        src_offset += send_table[i][j];
      }
    }
    // Calculate the dst offset pointer from each GPU to each other
    for (size_t i = 0; i < local_gpu_count; i++) {
      size_t dst_offset = 0;
      for (size_t j = 0; j < total_gpu_count; j++) {
        dst_pos[i][j] = dst[i] + dst_offset;
        dst_offset += recv_table[i][j];
      }
    }

#ifndef NDEBUG
    std::cout << "nccl all2all backward src_pos:" << std::endl;
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < total_gpu_count; j++) {
        std::cout << src_pos[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "nccl all2all backward dst_pos:" << std::endl;
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < total_gpu_count; j++) {
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
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < total_gpu_count; j++) {
        CK_NCCL_THROW_(ncclSend(src_pos[i][j], send_table[i][j], type, j,
                                device_resources[i].get_nccl(), device_resources[i]->get_stream()));
        CK_NCCL_THROW_(ncclRecv(dst_pos[i][j], recv_table[i][j], type, j,
                                device_resources[i].get_nccl(), device_resources[i]->get_stream()));
      }
    }
    CK_NCCL_THROW_(ncclGroupEnd());

    return;
  }

#endif

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
      const std::string &plan_file, size_t batch_size_per_gpu,
      const std::vector<size_t> &slot_num_per_gpu, size_t embedding_vec_size,
      const TensorPtrs<Type> &send_tensors, TensorPtrs<Type> &recv_tensors,
      const std::shared_ptr<GPUResourceGroup> &device_resources) {
    using transfer_plan_t =
        typename FasterGossipComm::FasterGossipCommAll2AllTraits<Type>::transfer_plan_t;
    transfer_plan_t *transfer_plan = new transfer_plan_t(parse_plan(plan_file.c_str()));
    size_t plan_gpu_count = transfer_plan->num_gpus();  // total number of GPUs in current node

    std::vector<int> device_list = device_resources->get_device_list();
    size_t local_gpu_count = device_list.size();
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
    for (size_t id = 0; id < local_gpu_count; id++) {
      src[id] = send_tensors[id]->get_ptr();
      dst[id] = recv_tensors[id]->get_ptr();
    }

    // Fill in partition table, ith Topo GPU to jth Topo GPU
    std::vector<std::vector<size_t>> table(local_gpu_count, std::vector<size_t>(local_gpu_count));
    for (size_t i = 0; i < local_gpu_count; i++) {
      size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu[i] * embedding_vec_size;
      for (size_t j = 0; j < local_gpu_count; j++) {
        table[i][j] = element_per_send;
      }
    }

#ifndef NDEBUG
    std::cout << "gossip all2all forward table:" << std::endl;
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < local_gpu_count; j++) {
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
      const std::string &plan_file, size_t batch_size_per_gpu,
      const std::vector<size_t> &slot_num_per_gpu, size_t embedding_vec_size,
      const TensorPtrs<Type> &send_tensors, TensorPtrs<Type> &recv_tensors,
      const std::shared_ptr<GPUResourceGroup> &device_resources) {
    using transfer_plan_t =
        typename FasterGossipComm::FasterGossipCommAll2AllTraits<Type>::transfer_plan_t;
    transfer_plan_t *transfer_plan = new transfer_plan_t(parse_plan(plan_file.c_str()));
    size_t plan_gpu_count = transfer_plan->num_gpus();  // total number of GPUs in current node

    std::vector<int> device_list = device_resources->get_device_list();
    size_t local_gpu_count = device_list.size();
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
    for (size_t id = 0; id < local_gpu_count; id++) {
      src[id] = send_tensors[id]->get_ptr();
      dst[id] = recv_tensors[id]->get_ptr();
    }

    // Fill in partition table, ith Topo GPU to jth Topo GPU
    std::vector<std::vector<size_t>> table(local_gpu_count, std::vector<size_t>(local_gpu_count));
    for (size_t i = 0; i < local_gpu_count; i++) {
      size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu[i] * embedding_vec_size;
      for (size_t j = 0; j < local_gpu_count; j++) {
        table[j][i] = element_per_send;
      }
    }

#ifndef NDEBUG
    std::cout << "gossip all2all backward table:" << std::endl;
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < local_gpu_count; j++) {
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
      const std::string &plan_file, size_t batch_size_per_gpu, size_t slot_num,
      size_t embedding_vec_size, const TensorPtrs<Type> &send_tensors,
      TensorPtrs<Type> &recv_tensors, const std::shared_ptr<GPUResourceGroup> &device_resources) {
    using transfer_plan_t =
        typename FasterGossipCommMulti::FasterGossipCommMultiAll2AllTraits<Type>::transfer_plan_t;
    transfer_plan_t *transfer_plan = new transfer_plan_t(parse_plan(plan_file.c_str()));
    size_t plan_gpu_count = transfer_plan->num_gpus();  // total number of GPUs in current node
    std::vector<int> device_list = device_resources->get_device_list();
    size_t local_gpu_count = device_list.size();
    size_t total_gpu_count = device_resources->get_total_gpu_count();
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
    size_t num_proc = device_resources->get_node_count();
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
    for (size_t id = 0; id < local_gpu_count; id++) {
      src[id] = send_tensors[id]->get_ptr();
      dst[id] = recv_tensors[id]->get_ptr();
    }

    std::vector<std::vector<size_t>> send_table(local_gpu_count,
                                                std::vector<size_t>(total_gpu_count));
    std::vector<std::vector<size_t>> recv_table(local_gpu_count,
                                                std::vector<size_t>(total_gpu_count));

    // Fill in sending partition table, ith Topo GPU send to jth global GPU
    for (size_t i = 0; i < local_gpu_count; i++) {
      size_t device_id = (*device_resources)[i].get_device_id();
      size_t global_id = device_resources->get_global_id(device_id);
      size_t slot_num_per_gpu =
          slot_num / total_gpu_count + ((global_id < (slot_num % total_gpu_count)) ? 1 : 0);
      size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu * embedding_vec_size;

      for (size_t j = 0; j < total_gpu_count; j++) {
        send_table[i][j] = element_per_send;
      }
    }

    // Fill in receiving partition table, ith Topo GPU receive from jth global GPU
    for (size_t j = 0; j < total_gpu_count; j++) {
      size_t global_id = j;
      size_t slot_num_per_gpu =
          slot_num / total_gpu_count + ((global_id < (slot_num % total_gpu_count)) ? 1 : 0);
      size_t element_per_recv = batch_size_per_gpu * slot_num_per_gpu * embedding_vec_size;

      for (size_t i = 0; i < local_gpu_count; i++) {
        recv_table[i][j] = element_per_recv;
      }
    }

#ifndef NDEBUG
    std::cout << "my_rank=" << my_rank << ", gossip all2all forward send_table:" << std::endl;
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < total_gpu_count; j++) {
        std::cout << send_table[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "my_rank=" << my_rank << ", gossip all2all forward recv_table:" << std::endl;
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < total_gpu_count; j++) {
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
      const std::string &plan_file, size_t batch_size_per_gpu, size_t slot_num,
      size_t embedding_vec_size, const TensorPtrs<Type> &send_tensors,
      TensorPtrs<Type> &recv_tensors, const std::shared_ptr<GPUResourceGroup> &device_resources) {
    using transfer_plan_t =
        typename FasterGossipCommMulti::FasterGossipCommMultiAll2AllTraits<Type>::transfer_plan_t;
    transfer_plan_t *transfer_plan = new transfer_plan_t(parse_plan(plan_file.c_str()));
    size_t plan_gpu_count = transfer_plan->num_gpus();  // total number of GPUs in current node
    std::vector<int> device_list = device_resources->get_device_list();
    size_t local_gpu_count = device_list.size();
    size_t total_gpu_count = device_resources->get_total_gpu_count();
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
    size_t num_proc = device_resources->get_node_count();
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
    for (size_t id = 0; id < local_gpu_count; id++) {
      src[id] = send_tensors[id]->get_ptr();
      dst[id] = recv_tensors[id]->get_ptr();
    }

    std::vector<std::vector<size_t>> send_table(local_gpu_count,
                                                std::vector<size_t>(total_gpu_count));
    std::vector<std::vector<size_t>> recv_table(local_gpu_count,
                                                std::vector<size_t>(total_gpu_count));

    // Fill in receiving partition table, ith Topo GPU receive from jth global GPU
    for (size_t i = 0; i < local_gpu_count; i++) {
      size_t device_id = (*device_resources)[i].get_device_id();
      size_t global_id = device_resources->get_global_id(device_id);
      size_t slot_num_per_gpu =
          slot_num / total_gpu_count + ((global_id < (slot_num % total_gpu_count)) ? 1 : 0);
      size_t element_per_recv = batch_size_per_gpu * slot_num_per_gpu * embedding_vec_size;

      for (size_t j = 0; j < total_gpu_count; j++) {
        recv_table[i][j] = element_per_recv;
      }
    }

    // Fill in sending partition table, ith Topo GPU send to jth global GPU
    for (size_t j = 0; j < total_gpu_count; j++) {
      size_t global_id = j;
      size_t slot_num_per_gpu =
          slot_num / total_gpu_count + ((global_id < (slot_num % total_gpu_count)) ? 1 : 0);
      size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu * embedding_vec_size;

      for (size_t i = 0; i < local_gpu_count; i++) {
        send_table[i][j] = element_per_send;
      }
    }

#ifndef NDEBUG
    std::cout << "my_rank=" << my_rank << ", gossip all2all backward send_table:" << std::endl;
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < total_gpu_count; j++) {
        std::cout << send_table[i][j] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "my_rank=" << my_rank << ", gossip all2all backward recv_table:" << std::endl;
    for (size_t i = 0; i < local_gpu_count; i++) {
      for (size_t j = 0; j < total_gpu_count; j++) {
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
  template <typename TypeEmbeddingComp>
  void forward_reorder(size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
                       const TensorPtrs<TypeEmbeddingComp> &src_tensors,
                       const TensorPtrs<TypeEmbeddingComp> &dst_tensors,
                       const GPUResourceGroup &device_resources) {
    CudaDeviceContext context;
    size_t local_gpu_count = device_resources.size();
    size_t total_gpu_count = device_resources.get_total_gpu_count();

    const size_t grid_size = batch_size_per_gpu;

    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());

      if (std::is_same<TypeEmbeddingComp, __half>::value && embedding_vec_size % 2 == 0) {
        const size_t block_size = embedding_vec_size / 2;
        forward_reorder_align2_kernel<<<grid_size, block_size, 0,
                                        device_resources[id].get_stream()>>>(
            batch_size_per_gpu, slot_num, embedding_vec_size / 2, total_gpu_count,
            src_tensors[id]->get_ptr(), dst_tensors[id]->get_ptr());
      } else {
        const size_t block_size = embedding_vec_size;
        forward_reorder_kernel<<<grid_size, block_size, 0, device_resources[id].get_stream()>>>(
            batch_size_per_gpu, slot_num, embedding_vec_size, total_gpu_count,
            src_tensors[id]->get_ptr(), dst_tensors[id]->get_ptr());
      }
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
  template <typename TypeEmbeddingComp>
  void backward_reorder(size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
                        const TensorPtrs<TypeEmbeddingComp> &src_tensors,
                        const TensorPtrs<TypeEmbeddingComp> &dst_tensors,
                        const GPUResourceGroup &device_resources) {
    CudaDeviceContext context;
    size_t local_gpu_count = device_resources.size();
    size_t total_gpu_count = device_resources.get_total_gpu_count();

    const size_t grid_size = batch_size_per_gpu;

    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());
      if (std::is_same<TypeEmbeddingComp, __half>::value && embedding_vec_size % 2 == 0) {
        const size_t block_size = embedding_vec_size / 2;
        backward_reorder_align2_kernel<<<grid_size, block_size, 0,
                                         device_resources[id].get_stream()>>>(
            batch_size_per_gpu, slot_num, embedding_vec_size / 2, total_gpu_count,
            src_tensors[id]->get_ptr(), dst_tensors[id]->get_ptr());
      } else {
        const size_t block_size = embedding_vec_size;
        backward_reorder_kernel<<<grid_size, block_size, 0, device_resources[id].get_stream()>>>(
            batch_size_per_gpu, slot_num, embedding_vec_size, total_gpu_count,
            src_tensors[id]->get_ptr(), dst_tensors[id]->get_ptr());
      }
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
                    cudaStream_t stream) const {
    const size_t block_size = 256;
    const size_t grid_size = (n + block_size - 1) / block_size;

    memset_liner_kernel<<<grid_size, block_size, 0, stream>>>(data, start_value, stride_value, n);
  }

  /**
   * set constant data for a buffer
   * @param stream cuda stream.
   * @param data the pointer of the data buffer which will be written.
   * @param value the setting value
   * @param n the number of the data.
   */
  template <typename Type>
  void memset_const(Type *data, Type value, size_t n, cudaStream_t stream) const {
    const size_t block_size = 256;
    const size_t grid_size = (n + block_size - 1) / block_size;

    memset_const_kernel<<<grid_size, block_size, 0, stream>>>(data, value, n);
  }

  /**
   * get hash table value by value_index
   * @param stream cuda stream.
   * @param count total count of value which will be get from hash table.
   * @param embedding_vec_size embedding vector size, each value has the dim of
   * embedding_vec_size.
   * @param value_index the pointer of value_index.
   * @param hash_table_value the pointer of hash table value.
   * @param value_retrieved the pointer of the retrived value.
   */
  template <typename TypeHashValueIndex>
  void get_hash_value(cudaStream_t stream, size_t count, size_t embedding_vec_size,
                      const TypeHashValueIndex *value_index, const float *hash_table_value,
                      float *value_retrieved) const {
    const size_t block_size = embedding_vec_size;
    const size_t grid_size = count;

    get_hash_value_kernel<<<grid_size, block_size, 0, stream>>>(
        count, embedding_vec_size, value_index, hash_table_value, value_retrieved);
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
                        const TypeHashValueIndex *hash_table_slot_id,
                        TypeHashValueIndex *slot_id) const {
    const size_t block_size = 64;
    const size_t grid_size = (count + block_size - 1) / block_size;

    get_hash_slot_id_kernel<<<grid_size, block_size, 0, stream>>>(count, value_index,
                                                                  hash_table_slot_id, slot_id);
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
      std::ifstream &weight_stream, size_t vocabulary_size, size_t embedding_vec_size,
      size_t max_vocabulary_size_per_gpu, TensorPtrs<float> &hash_table_value_tensors,
      std::vector<std::shared_ptr<HashTable<TypeHashKey, TypeHashValueIndex,
                                            std::numeric_limits<TypeHashKey>::max()>>> &hash_tables,
      const GPUResourceGroup &device_resources) {
    CudaDeviceContext context;
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
    size_t local_gpu_count = device_resources.size();
    size_t chunk_loop = 1000;
    size_t tile_size = 1;  // must be 1, because we need to cal (key&local_gpu_count) to decide
                           // gpu_id for each <key,value>
    size_t hash_table_key_tile_size = tile_size;
    size_t hash_table_key_tile_size_in_B = hash_table_key_tile_size * sizeof(TypeHashKey);
    size_t hash_table_key_chunk_size = hash_table_key_tile_size * chunk_loop;
    size_t hash_table_key_chunk_size_in_B = hash_table_key_chunk_size * sizeof(TypeHashKey);
    size_t hash_table_value_tile_size = tile_size * embedding_vec_size;
    size_t hash_table_value_tile_size_in_B = hash_table_value_tile_size * sizeof(float);
    size_t hash_table_value_chunk_size = hash_table_value_tile_size * chunk_loop;
    size_t hash_table_value_chunk_size_in_B = hash_table_value_chunk_size * sizeof(float);
    size_t hash_table_tile_size_in_B =
        hash_table_key_tile_size_in_B + hash_table_value_tile_size_in_B;
    size_t hash_table_chunk_size_in_B = hash_table_tile_size_in_B * chunk_loop;

    // CAUSION: can not decide how many values for each GPU, so need to allocate enough memory
    // for each GPU allocate GPU memory for hash_table_value_index
    std::unique_ptr<size_t[]> tile_counter_per_gpu(
        new size_t[local_gpu_count]);  // <= hash_table_value_index_per_gpu_size
    memset(tile_counter_per_gpu.get(), 0, sizeof(size_t) * local_gpu_count);
    std::unique_ptr<size_t[]> tile_counter_in_chunk_per_gpu(new size_t[local_gpu_count]);
    memset(tile_counter_in_chunk_per_gpu.get(), 0, sizeof(size_t) * local_gpu_count);
    std::unique_ptr<TypeHashKey *[]> d_hash_table_value_index_chunk_per_gpu(
        new TypeHashKey *[local_gpu_count]);

    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());
      CK_CUDA_THROW_(
          cudaMalloc(&d_hash_table_value_index_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
      // initalize to zeros
      CK_CUDA_THROW_(cudaMemsetAsync(d_hash_table_value_index_chunk_per_gpu[id], 0,
                                     hash_table_key_chunk_size_in_B,
                                     device_resources[id].get_stream()));
    }

    // sync wait
    sync_all_gpus(device_resources);

    // CAUSION: can not decide how many values for each GPU, so need to allocate enough memory
    // for each GPU allocate CPU/GPU memory for hash_table/key/value chunk
    char *hash_table_chunk;
    CK_CUDA_THROW_(cudaMallocHost(&hash_table_chunk, hash_table_chunk_size_in_B));
    std::unique_ptr<TypeHashKey *[]> h_hash_table_key_chunk_per_gpu(
        new TypeHashKey *[local_gpu_count]);
    for (size_t id = 0; id < local_gpu_count; id++) {
      CK_CUDA_THROW_(
          cudaMallocHost(&h_hash_table_key_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
    }
    std::unique_ptr<TypeHashKey *[]> d_hash_table_key_chunk_per_gpu(
        new TypeHashKey *[local_gpu_count]);
    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());
      CK_CUDA_THROW_(
          cudaMalloc(&d_hash_table_key_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
    }
    std::unique_ptr<float *[]> h_hash_table_value_chunk_per_gpu(new float *[local_gpu_count]);
    for (size_t id = 0; id < local_gpu_count; id++) {
      CK_CUDA_THROW_(
          cudaMallocHost(&h_hash_table_value_chunk_per_gpu[id], hash_table_value_chunk_size_in_B));
    }

    // do upload
    size_t loop_num = file_size_in_B / hash_table_chunk_size_in_B;
    MESSAGE_("Start to upload embedding table file to GPUs, file size: " +
             std::to_string(file_size_in_B) +
             " Bytes, total loop_num: " + std::to_string(loop_num));
    for (size_t i = 0; i < loop_num; i++) {
      // read a chunk of data from file
      // one pair in hash table file includes: <key, value>
      weight_stream.read(hash_table_chunk, hash_table_chunk_size_in_B);

      // memcpy from CPU to CPU
      char *src_buf = hash_table_chunk;
      TypeHashKey *key_dst_buf;
      float *value_dst_buf;
      for (size_t k = 0; k < chunk_loop; k++) {  // process a tile in each loop
        TypeHashKey key = *((TypeHashKey *)src_buf);
        size_t gid = key % device_resources.get_total_gpu_count();  // global GPU ID
        size_t id = device_resources.get_local_id(gid);   // local GPU ID (not gpudevice id)
        size_t dst_rank = device_resources.get_pid(gid);  // node id

        if (static_cast<size_t>(my_rank) == dst_rank) {
          // memcpy hash_table_key to corresponding GPU
          key_dst_buf = h_hash_table_key_chunk_per_gpu[id] +
                        tile_counter_in_chunk_per_gpu[id] * hash_table_key_tile_size;
          CK_CUDA_THROW_(cudaMemcpyAsync(key_dst_buf, src_buf, hash_table_key_tile_size_in_B,
                                         cudaMemcpyHostToHost, device_resources[id].get_stream()));

          src_buf += hash_table_key_tile_size_in_B;

          // memcpy hash_table_value to corresponding GPU
          value_dst_buf = h_hash_table_value_chunk_per_gpu[id] +
                          tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
          CK_CUDA_THROW_(cudaMemcpyAsync(value_dst_buf, src_buf, hash_table_value_tile_size_in_B,
                                         cudaMemcpyHostToHost, device_resources[id].get_stream()));

          src_buf += hash_table_value_tile_size_in_B;

          tile_counter_in_chunk_per_gpu[id] += tile_size;
        } else {
          src_buf += hash_table_key_tile_size_in_B;
          src_buf += hash_table_value_tile_size_in_B;
          continue;
        }
      }  // end of for(int k = 0; k < (chunk_loop * local_gpu_count); k++)

      // do HashTable insert <key,value_index>
      for (size_t id = 0; id < local_gpu_count; id++) {
        context.set_device(device_resources[id].get_device_id());

        size_t tile_count = tile_counter_in_chunk_per_gpu[id];

        // memcpy hash_table_key from CPU to GPU
        CK_CUDA_THROW_(cudaMemcpyAsync(d_hash_table_key_chunk_per_gpu[id],
                                       h_hash_table_key_chunk_per_gpu[id],
                                       tile_count * sizeof(TypeHashKey), cudaMemcpyHostToDevice,
                                       device_resources[id].get_stream()));

        size_t value_index_offset = tile_counter_per_gpu[id];
        TypeHashKey *value_index_buf = d_hash_table_value_index_chunk_per_gpu[id];

        if (tile_count > 0) {
          // set hash_table_value_index on GPU
          memset_liner(value_index_buf, (TypeHashKey)value_index_offset, (TypeHashKey)1, tile_count,
                       device_resources[id].get_stream());
        }

        // do hash table insert <key, value_index> on GPU
        hash_tables[id]->insert(d_hash_table_key_chunk_per_gpu[id], value_index_buf, tile_count,
                                device_resources[id].get_stream());
        size_t value_head = hash_tables[id]->add_value_head(tile_count);
      }

      // memcpy hash_table_value from CPU to GPU
      for (size_t id = 0; id < local_gpu_count; id++) {
        context.set_device(device_resources[id].get_device_id());
        size_t value_chunk_size = tile_counter_in_chunk_per_gpu[id] * embedding_vec_size;
        size_t value_chunk_offset = tile_counter_per_gpu[id] * embedding_vec_size;
        float *src_buf = h_hash_table_value_chunk_per_gpu[id];
        float *dst_buf = hash_table_value_tensors[id]->get_ptr() + value_chunk_offset;
        CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf, src_buf, value_chunk_size * sizeof(float),
                                       cudaMemcpyHostToDevice, device_resources[id].get_stream()));
      }

      sync_all_gpus(device_resources);

      // set counter value
      for (size_t id = 0; id < local_gpu_count; id++) {
        tile_counter_per_gpu[id] += tile_counter_in_chunk_per_gpu[id];
        tile_counter_in_chunk_per_gpu[id] = 0;  // reset chunk counter to zero

        if (tile_counter_per_gpu[id] > max_vocabulary_size_per_gpu) {
          char msg[100]{0};
          sprintf(msg, "The size of hash table on GPU %zu is out of range %zu\n", id,
                  max_vocabulary_size_per_gpu);
          CK_THROW_(Error_t::OutOfBound, msg);
        }
      }

      /*       std::cout << "\rUploading " << std::fixed << std::setprecision(2)
                      << (float)(i) / loop_num * 100.0f << "%, loop " << i << " of " << loop_num
                      << std::flush; */
    }  // end of for(int i = 0; i < loop_num; i++)

    // process the remaining data(less than a chunk)
    size_t remain_size_in_B = file_size_in_B - loop_num * hash_table_chunk_size_in_B;
    size_t remain_loop_num = remain_size_in_B / hash_table_tile_size_in_B;
    if (remain_loop_num != 0) {
      MESSAGE_("Upload the remaining data");
      // read all the remaining data
      weight_stream.read((char *)hash_table_chunk, remain_size_in_B);

      char *src_buf = hash_table_chunk;
      TypeHashKey *key_dst_buf;
      TypeHashKey *value_index_buf;
      float *value_dst_buf;
      for (size_t i = 0; i < remain_loop_num; i++) {
        TypeHashKey key = *((TypeHashKey *)src_buf);
        size_t gid = key % device_resources.get_total_gpu_count();  // global GPU ID
        size_t id = device_resources.get_local_id(gid);  // local GPU ID (not gpudevice id)
        size_t dst_rank = device_resources.get_pid(gid);

        if (static_cast<size_t>(my_rank) == dst_rank) {
          context.set_device(device_resources[id].get_device_id());

          // memcpy hash_table_key from CPU to GPU
          key_dst_buf = d_hash_table_key_chunk_per_gpu[id];
          CK_CUDA_THROW_(cudaMemcpyAsync(key_dst_buf, src_buf, hash_table_key_tile_size_in_B,
                                         cudaMemcpyHostToDevice,
                                         device_resources[id].get_stream()));

          src_buf += hash_table_key_tile_size_in_B;

          // set value_index
          size_t value_index_offset = tile_counter_per_gpu[id];
          value_index_buf = d_hash_table_value_index_chunk_per_gpu[id];
          memset_liner(value_index_buf, (TypeHashKey)value_index_offset, (TypeHashKey)1, 1,
                       device_resources[id].get_stream());

          // do hash table insert <key, value_index> on GPU
          hash_tables[id]->insert(d_hash_table_key_chunk_per_gpu[id], value_index_buf,
                                  hash_table_key_tile_size, device_resources[id].get_stream());
          size_t value_head = hash_tables[id]->add_value_head(hash_table_key_tile_size);

          // memcpy hash_table_value from CPU to GPU
          size_t value_offset = tile_counter_per_gpu[id] * embedding_vec_size;
          value_dst_buf = hash_table_value_tensors[id]->get_ptr() + value_offset;
          CK_CUDA_THROW_(cudaMemcpyAsync(value_dst_buf, src_buf, hash_table_value_tile_size_in_B,
                                         cudaMemcpyHostToDevice,
                                         device_resources[id].get_stream()));
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
      sync_all_gpus(device_resources);

    }  // end of if(remain_loop_num)

    MESSAGE_("Done");

    // release resources
    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());
      CK_CUDA_THROW_(cudaFree(d_hash_table_value_index_chunk_per_gpu[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_key_chunk_per_gpu[id]));
    }
    CK_CUDA_THROW_(cudaFreeHost(hash_table_chunk));
    for (size_t id = 0; id < local_gpu_count; id++) {
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
      std::ifstream &weight_stream, size_t vocabulary_size, size_t embedding_vec_size,
      size_t max_vocabulary_size_per_gpu, TensorPtrs<float> &hash_table_value_tensors,
      TensorPtrs<TypeHashValueIndex> &hash_table_slot_id_tensors,
      std::vector<std::shared_ptr<HashTable<TypeHashKey, TypeHashValueIndex,
                                            std::numeric_limits<TypeHashKey>::max()>>> &hash_tables,
      const GPUResourceGroup &device_resources) {
    CudaDeviceContext context;
    // check file size and vocabulary_size (file size <=　hash_table_size)
    weight_stream.seekg(0, weight_stream.end);
    size_t file_size_in_B = weight_stream.tellg();
    weight_stream.seekg(0, weight_stream.beg);

    int my_rank = 0;
#ifdef ENABLE_MPI
    int n_ranks = 1;
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));
#endif

    // define size
    size_t local_gpu_count = device_resources.size();
    size_t chunk_loop = 1000;
    size_t tile_size = 1;  // must be 1, because we need to cal (key&local_gpu_count) to decide
                           // gpu_id for each <key,value>
    size_t hash_table_key_tile_size = tile_size;
    size_t hash_table_key_tile_size_in_B = hash_table_key_tile_size * sizeof(TypeHashKey);
    size_t hash_table_key_chunk_size = hash_table_key_tile_size * chunk_loop;
    size_t hash_table_key_chunk_size_in_B = hash_table_key_chunk_size * sizeof(TypeHashKey);
    size_t hash_table_value_tile_size = tile_size * embedding_vec_size;
    size_t hash_table_value_tile_size_in_B = hash_table_value_tile_size * sizeof(float);
    size_t hash_table_value_chunk_size = hash_table_value_tile_size * chunk_loop;
    size_t hash_table_value_chunk_size_in_B = hash_table_value_chunk_size * sizeof(float);
    size_t hash_table_slot_id_tile_size = tile_size;
    size_t hash_table_slot_id_tile_size_in_B =
        hash_table_slot_id_tile_size * sizeof(TypeHashValueIndex);
    size_t hash_table_slot_id_chunk_size = hash_table_slot_id_tile_size * chunk_loop;
    size_t hash_table_slot_id_chunk_size_in_B =
        hash_table_slot_id_chunk_size * sizeof(TypeHashValueIndex);
    size_t hash_table_tile_size_in_B = hash_table_key_tile_size_in_B +
                                       hash_table_slot_id_tile_size_in_B +
                                       hash_table_value_tile_size_in_B;
    size_t hash_table_chunk_size_in_B = hash_table_tile_size_in_B * chunk_loop;
    size_t total_gpu_count = device_resources.get_total_gpu_count();

    // CAUSION: can not decide how many values for each GPU, so need to allocate enough memory
    // for each GPU allocate GPU memory for hash_table_value_index
    std::unique_ptr<size_t[]> tile_counter_per_gpu(
        new size_t[local_gpu_count]);  // <= hash_table_value_index_per_gpu_size
    memset(tile_counter_per_gpu.get(), 0, sizeof(size_t) * local_gpu_count);
    std::unique_ptr<size_t[]> tile_counter_in_chunk_per_gpu(new size_t[local_gpu_count]);
    memset(tile_counter_in_chunk_per_gpu.get(), 0, sizeof(size_t) * local_gpu_count);
    std::unique_ptr<TypeHashKey *[]> d_hash_table_value_index_chunk_per_gpu(
        new TypeHashKey *[local_gpu_count]);

    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());
      CK_CUDA_THROW_(
          cudaMalloc(&d_hash_table_value_index_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
      // initalize to zeros
      CK_CUDA_THROW_(cudaMemsetAsync(d_hash_table_value_index_chunk_per_gpu[id], 0,
                                     hash_table_key_chunk_size_in_B,
                                     device_resources[id].get_stream()));
    }

    // sync wait
    sync_all_gpus(device_resources);

    // CAUSION: can not decide how many values for each GPU, so need to allocate enough memory
    // for each GPU allocate CPU/GPU memory for hash_table/key/value chunk
    char *hash_table_chunk;
    CK_CUDA_THROW_(cudaMallocHost(&hash_table_chunk, hash_table_chunk_size_in_B));
    std::unique_ptr<TypeHashKey *[]> h_hash_table_key_chunk_per_gpu(
        new TypeHashKey *[local_gpu_count]);
    for (size_t id = 0; id < local_gpu_count; id++) {
      CK_CUDA_THROW_(
          cudaMallocHost(&h_hash_table_key_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
    }
    std::unique_ptr<TypeHashKey *[]> d_hash_table_key_chunk_per_gpu(
        new TypeHashKey *[local_gpu_count]);
    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());
      CK_CUDA_THROW_(
          cudaMalloc(&d_hash_table_key_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
    }
    std::unique_ptr<TypeHashValueIndex *[]> h_hash_table_slot_id_chunk_per_gpu(
        new TypeHashValueIndex *[local_gpu_count]);
    for (size_t id = 0; id < local_gpu_count; id++) {
      CK_CUDA_THROW_(cudaMallocHost(&h_hash_table_slot_id_chunk_per_gpu[id],
                                    hash_table_slot_id_chunk_size_in_B));
    }
    std::unique_ptr<TypeHashValueIndex *[]> d_hash_table_slot_id_chunk_per_gpu(
        new TypeHashValueIndex *[local_gpu_count]);
    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());
      CK_CUDA_THROW_(
          cudaMalloc(&d_hash_table_slot_id_chunk_per_gpu[id], hash_table_slot_id_chunk_size_in_B));
    }
    std::unique_ptr<float *[]> h_hash_table_value_chunk_per_gpu(new float *[local_gpu_count]);
    for (size_t id = 0; id < local_gpu_count; id++) {
      CK_CUDA_THROW_(
          cudaMallocHost(&h_hash_table_value_chunk_per_gpu[id], hash_table_value_chunk_size_in_B));
    }

    // do upload
    size_t loop_num = file_size_in_B / hash_table_chunk_size_in_B;
    MESSAGE_("Start to upload embedding table file to GPUs, file size: " +
             std::to_string(file_size_in_B) +
             " Bytes, total loop_num: " + std::to_string(loop_num));
    for (size_t i = 0; i < loop_num; i++) {
      // read a chunk of data from file
      // one pair in hash table file includes: <key, slot_id, value>
      weight_stream.read(hash_table_chunk, hash_table_chunk_size_in_B);

      // memcpy from CPU to CPU
      char *src_buf = hash_table_chunk;
      TypeHashKey *key_dst_buf;
      TypeHashValueIndex *slot_id_dst_buf;
      float *value_dst_buf;
      for (size_t k = 0; k < chunk_loop; k++) {  // process a tile in each loop
        TypeHashValueIndex slot_id =
            *((TypeHashValueIndex *)(src_buf + hash_table_key_tile_size_in_B));
        size_t gid = slot_id % total_gpu_count;           // global GPU ID
        size_t id = device_resources.get_local_id(gid);   // local GPU ID (not gpudevice id)
        size_t dst_rank = device_resources.get_pid(gid);  // node id

        if (static_cast<size_t>(my_rank) == dst_rank) {
          // memcpy hash_table_key to corresponding GPU
          key_dst_buf = h_hash_table_key_chunk_per_gpu[id] +
                        tile_counter_in_chunk_per_gpu[id] * hash_table_key_tile_size;
          CK_CUDA_THROW_(cudaMemcpyAsync(key_dst_buf, src_buf, hash_table_key_tile_size_in_B,
                                         cudaMemcpyHostToHost, device_resources[id].get_stream()));

          src_buf += hash_table_key_tile_size_in_B;

          // memcpy hash_table_slot_id to corresponding GPU
          slot_id_dst_buf = h_hash_table_slot_id_chunk_per_gpu[id] +
                            tile_counter_in_chunk_per_gpu[id] * hash_table_slot_id_tile_size;
          CK_CUDA_THROW_(cudaMemcpyAsync(slot_id_dst_buf, src_buf,
                                         hash_table_slot_id_tile_size_in_B, cudaMemcpyHostToHost,
                                         device_resources[id].get_stream()));

          src_buf += hash_table_slot_id_tile_size_in_B;

          // memcpy hash_table_value to corresponding GPU
          value_dst_buf = h_hash_table_value_chunk_per_gpu[id] +
                          tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
          CK_CUDA_THROW_(cudaMemcpyAsync(value_dst_buf, src_buf, hash_table_value_tile_size_in_B,
                                         cudaMemcpyHostToHost, device_resources[id].get_stream()));

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
      for (size_t id = 0; id < local_gpu_count; id++) {
        if (tile_counter_in_chunk_per_gpu[id] == 0) {
          continue;
        }

        context.set_device(device_resources[id].get_device_id());

        size_t tile_count = tile_counter_in_chunk_per_gpu[id];

        // memcpy hash_table_key from CPU to GPU
        CK_CUDA_THROW_(cudaMemcpyAsync(d_hash_table_key_chunk_per_gpu[id],
                                       h_hash_table_key_chunk_per_gpu[id],
                                       tile_count * sizeof(TypeHashKey), cudaMemcpyHostToDevice,
                                       device_resources[id].get_stream()));

        size_t value_index_offset = tile_counter_per_gpu[id];
        TypeHashKey *value_index_buf = d_hash_table_value_index_chunk_per_gpu[id];

        if (tile_count > 0) {
          // set hash_table_value_index on GPU
          memset_liner(value_index_buf, (TypeHashKey)value_index_offset, (TypeHashKey)1, tile_count,
                       device_resources[id].get_stream());
        }

        // do hash table insert <key, value_index> on GPU
        hash_tables[id]->insert(d_hash_table_key_chunk_per_gpu[id], value_index_buf, tile_count,
                                device_resources[id].get_stream());
        size_t value_head = hash_tables[id]->add_value_head(tile_count);
      }

      // memcpy hash_table_slot_id and hash_table_value from CPU to GPU
      for (size_t id = 0; id < local_gpu_count; id++) {
        if (tile_counter_in_chunk_per_gpu[id] == 0) {
          continue;
        }

        context.set_device(device_resources[id].get_device_id());

        size_t slot_id_chunk_size =
            tile_counter_in_chunk_per_gpu[id] * hash_table_slot_id_tile_size;
        size_t slot_id_offset = tile_counter_per_gpu[id] * hash_table_slot_id_tile_size;

        if ((slot_id_offset + slot_id_chunk_size) > max_vocabulary_size_per_gpu) {
          char msg[100]{0};
          sprintf(msg, "The size of hash table on GPU%zu is out of range %zu\n", id,
                  max_vocabulary_size_per_gpu);
          CK_THROW_(Error_t::OutOfBound, msg);
        }

        TypeHashValueIndex *src_buf_sid = h_hash_table_slot_id_chunk_per_gpu[id];
        TypeHashValueIndex *dst_buf_sid =
            hash_table_slot_id_tensors[id]->get_ptr() + slot_id_offset;
        CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf_sid, src_buf_sid,
                                       slot_id_chunk_size * sizeof(TypeHashValueIndex),
                                       cudaMemcpyHostToDevice, device_resources[id].get_stream()));

        size_t value_chunk_size = tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
        size_t value_chunk_offset = tile_counter_per_gpu[id] * hash_table_value_tile_size;
        float *src_buf_value = h_hash_table_value_chunk_per_gpu[id];
        float *dst_buf_value = hash_table_value_tensors[id]->get_ptr() + value_chunk_offset;
        CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf_value, src_buf_value,
                                       value_chunk_size * sizeof(float), cudaMemcpyHostToDevice,
                                       device_resources[id].get_stream()));
      }

      sync_all_gpus(device_resources);

      // set counter value
      for (size_t id = 0; id < local_gpu_count; id++) {
        tile_counter_per_gpu[id] +=
            tile_counter_in_chunk_per_gpu[id];  // accumulate total tile counter
        tile_counter_in_chunk_per_gpu[id] = 0;  // reset chunk counter to zero

        if (tile_counter_per_gpu[id] > max_vocabulary_size_per_gpu) {
          char msg[100];
          sprintf(msg, "The size of hash table on GPU%zu is out of range %zu\n", id,
                  max_vocabulary_size_per_gpu);
          CK_THROW_(Error_t::OutOfBound, msg);
        }
      }

      /*       std::cout << "\rUploading " << std::fixed << std::setprecision(2)
                      << (float)(i) / loop_num * 100.0f << "%, loop " << i << " of " << loop_num
                      << std::flush; */
    }  // end of for(int i = 0; i < loop_num; i++)

    // std::cout << std::endl;

    // process the remaining data(less than a chunk)
    size_t remain_size_in_B = file_size_in_B - loop_num * hash_table_chunk_size_in_B;
    size_t remain_loop_num = remain_size_in_B / hash_table_tile_size_in_B;
    if (remain_loop_num != 0) {
      MESSAGE_("Upload the remaining data");
      // read all the remaining data
      weight_stream.read((char *)hash_table_chunk, remain_size_in_B);

      char *src_buf = hash_table_chunk;
      TypeHashKey *key_dst_buf;
      TypeHashValueIndex *value_index_buf;
      TypeHashValueIndex *slot_id_dst_buf;
      float *value_dst_buf;
      for (size_t i = 0; i < remain_loop_num; i++) {  // process one tile in each loop

        TypeHashValueIndex slot_id =
            *((TypeHashValueIndex *)(src_buf + hash_table_key_tile_size_in_B));
        size_t gid = slot_id % total_gpu_count;           // global GPU ID
        size_t id = device_resources.get_local_id(gid);   // local GPU ID (not gpu devie id)
        size_t dst_rank = device_resources.get_pid(gid);  // node id

        if (static_cast<size_t>(my_rank) == dst_rank) {
          context.set_device(device_resources[id].get_device_id());

          // memcpy hash_table_key from CPU to GPU
          key_dst_buf = d_hash_table_key_chunk_per_gpu[id];
          CK_CUDA_THROW_(cudaMemcpyAsync(key_dst_buf, src_buf, hash_table_key_tile_size_in_B,
                                         cudaMemcpyHostToDevice,
                                         device_resources[id].get_stream()));
          src_buf += hash_table_key_tile_size_in_B;

          // set value_index
          size_t value_index_offset = tile_counter_per_gpu[id];
          value_index_buf = d_hash_table_value_index_chunk_per_gpu[id];
          memset_liner(value_index_buf, (TypeHashKey)value_index_offset, (TypeHashKey)1, 1,
                       device_resources[id].get_stream());

          // do hash table insert <key, value_index> on GPU
          hash_tables[id]->insert(d_hash_table_key_chunk_per_gpu[id], value_index_buf,
                                  hash_table_key_tile_size, device_resources[id].get_stream());
          size_t value_head = hash_tables[id]->add_value_head(hash_table_key_tile_size);

          // memcpy hash_table_slot_id to corresponding GPU
          size_t slot_id_offset = tile_counter_per_gpu[id];
          slot_id_dst_buf = hash_table_slot_id_tensors[id]->get_ptr() + slot_id_offset;
          CK_CUDA_THROW_(cudaMemcpyAsync(slot_id_dst_buf, src_buf,
                                         hash_table_slot_id_tile_size_in_B, cudaMemcpyHostToHost,
                                         device_resources[id].get_stream()));
          src_buf += hash_table_slot_id_tile_size_in_B;

          // memcpy hash_table_value from CPU to GPU
          size_t value_offset = tile_counter_per_gpu[id] * embedding_vec_size;
          value_dst_buf = hash_table_value_tensors[id]->get_ptr() + value_offset;
          CK_CUDA_THROW_(cudaMemcpyAsync(value_dst_buf, src_buf, hash_table_value_tile_size_in_B,
                                         cudaMemcpyHostToDevice,
                                         device_resources[id].get_stream()));
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
      sync_all_gpus(device_resources);

    }  // end of if(remain_loop_num)

    MESSAGE_("Done");

    // release resources
    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());
      CK_CUDA_THROW_(cudaFree(d_hash_table_value_index_chunk_per_gpu[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_key_chunk_per_gpu[id]));
    }
    CK_CUDA_THROW_(cudaFreeHost(hash_table_chunk));
    for (size_t id = 0; id < local_gpu_count; id++) {
      CK_CUDA_THROW_(cudaFreeHost(h_hash_table_key_chunk_per_gpu[id]));
      CK_CUDA_THROW_(cudaFreeHost(h_hash_table_value_chunk_per_gpu[id]));
    }
  }

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
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void upload_params_to_device(std::ifstream &weight_stream, size_t embedding_vec_size,
                               TensorPtrs<float> &hash_table_value_tensors,
                               const std::vector<size_t> &slot_sizes,
                               const TensorPtrs<uint32_t> &mapping_offsets_per_gpu_tensors,
                               const GPUResourceGroup &device_resources) {
    CudaDeviceContext context;
    // check file size and vocabulary_size (file size <=　hash_table_size)
    weight_stream.seekg(0, weight_stream.end);
    size_t file_size_in_B = weight_stream.tellg();
    weight_stream.seekg(0, weight_stream.beg);

    int my_rank = 0;
#ifdef ENABLE_MPI
    int n_ranks = 1;
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));
#endif

    // define size
    size_t local_gpu_count = device_resources.size();
    size_t chunk_loop = 1000;
    size_t tile_size = 1;  // must be 1, because we need to cal (key&local_gpu_count) to decide
                           // gpu_id for each <key,value>
    size_t hash_table_key_tile_size = tile_size;
    size_t hash_table_key_tile_size_in_B = hash_table_key_tile_size * sizeof(TypeHashKey);
    size_t hash_table_value_tile_size = tile_size * embedding_vec_size;
    size_t hash_table_value_tile_size_in_B = hash_table_value_tile_size * sizeof(float);
    size_t hash_table_value_chunk_size = hash_table_value_tile_size * chunk_loop;
    size_t hash_table_value_chunk_size_in_B = hash_table_value_chunk_size * sizeof(float);
    size_t hash_table_slot_id_tile_size = tile_size;
    size_t hash_table_slot_id_tile_size_in_B =
        hash_table_slot_id_tile_size * sizeof(TypeHashValueIndex);
    size_t hash_table_tile_size_in_B = hash_table_key_tile_size_in_B +
                                       hash_table_slot_id_tile_size_in_B +
                                       hash_table_value_tile_size_in_B;
    size_t hash_table_chunk_size_in_B = hash_table_tile_size_in_B * chunk_loop;
    size_t total_gpu_count = device_resources.get_total_gpu_count();

    // CAUSION: can not decide how many values for each GPU, so need to allocate enough memory for
    // each GPU allocate CPU/GPU memory for value/index chunk
    char *hash_table_chunk;
    CK_CUDA_THROW_(cudaMallocHost(&hash_table_chunk, hash_table_chunk_size_in_B));
    std::unique_ptr<float *[]> h_hash_table_value_chunk_per_gpu(new float *[local_gpu_count]);
    for (size_t id = 0; id < local_gpu_count; id++) {
      CK_CUDA_THROW_(
          cudaMallocHost(&h_hash_table_value_chunk_per_gpu[id], hash_table_value_chunk_size_in_B));
    }
    std::unique_ptr<float *[]> d_hash_table_value_chunk_per_gpu(new float *[local_gpu_count]);
    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());
      CK_CUDA_THROW_(
          cudaMalloc(&d_hash_table_value_chunk_per_gpu[id], hash_table_value_chunk_size_in_B));
    }
    std::unique_ptr<size_t *[]> h_hash_table_index_chunk_per_gpu(new size_t *[local_gpu_count]);
    for (size_t id = 0; id < local_gpu_count; id++) {
      CK_CUDA_THROW_(
          cudaMallocHost(&h_hash_table_index_chunk_per_gpu[id], chunk_loop * sizeof(size_t)));
    }
    std::unique_ptr<size_t *[]> d_hash_table_index_chunk_per_gpu(new size_t *[local_gpu_count]);
    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());
      CK_CUDA_THROW_(
          cudaMalloc(&d_hash_table_index_chunk_per_gpu[id], chunk_loop * sizeof(size_t)));
    }

    std::unique_ptr<size_t[]> tile_counter_in_chunk_per_gpu(new size_t[local_gpu_count]);
    memset(tile_counter_in_chunk_per_gpu.get(), 0, sizeof(size_t) * local_gpu_count);

    // The vector that store the relationship between slot_id and slot order on the specific GPU
    std::vector<size_t> local_slot_id(slot_sizes.size());
    std::vector<size_t> local_slot_num(local_gpu_count, 0);
    for (size_t i = 0; i < slot_sizes.size(); i++) {
      size_t gid = i % total_gpu_count;                 // global GPU ID
      size_t id = device_resources.get_local_id(gid);   // local GPU ID (not gpudevice id)
      size_t dst_rank = device_resources.get_pid(gid);  // node id
      if (static_cast<size_t>(my_rank) == dst_rank) {
        local_slot_id[i] = local_slot_num[id];
        local_slot_num[id]++;
      }
    }

    // Host buffer to keep mapping_offset
    std::vector<uint32_t *> h_mapping_offsets_per_gpu_tensors(local_gpu_count);
    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());
      CK_CUDA_THROW_(cudaMallocHost(&h_mapping_offsets_per_gpu_tensors[id],
                                    local_slot_num[id] * sizeof(uint32_t)));
      // Copy the mapping offset from GPU to Host
      cudaMemcpyAsync(h_mapping_offsets_per_gpu_tensors[id],
                      mapping_offsets_per_gpu_tensors[id]->get_ptr(),
                      local_slot_num[id] * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                      device_resources[id].get_stream());
    }

    // sync wait
    sync_all_gpus(device_resources);

    // do upload
    size_t loop_num = file_size_in_B / hash_table_chunk_size_in_B;
    MESSAGE_("Start to upload embedding table file to GPUs, file size: " +
             std::to_string(file_size_in_B) +
             " Bytes, total loop_num: " + std::to_string(loop_num));
    for (size_t i = 0; i < loop_num; i++) {
      // read a chunk of data from file
      // one pair in hash table file includes: <key, slot_id, value>
      weight_stream.read(hash_table_chunk, hash_table_chunk_size_in_B);

      // memcpy from CPU to CPU
      char *src_buf = hash_table_chunk;
      float *value_dst_buf;
      size_t *tensor_index_dst_buf;
      for (size_t k = 0; k < chunk_loop; k++) {  // process a tile in each loop
        TypeHashValueIndex slot_id =
            *((TypeHashValueIndex *)(src_buf + hash_table_key_tile_size_in_B));
        size_t gid = slot_id % total_gpu_count;           // global GPU ID
        size_t id = device_resources.get_local_id(gid);   // local GPU ID (not gpudevice id)
        size_t dst_rank = device_resources.get_pid(gid);  // node id

        if (static_cast<size_t>(my_rank) == dst_rank) {
          TypeHashKey tile_key = *((TypeHashKey *)src_buf);
          size_t tensor_index =
              tile_key - (h_mapping_offsets_per_gpu_tensors[id][local_slot_id[slot_id]]);

          src_buf += hash_table_key_tile_size_in_B;
          src_buf += hash_table_slot_id_tile_size_in_B;
          // memcpy hash_table_value to corresponding GPU
          value_dst_buf = h_hash_table_value_chunk_per_gpu[id] +
                          tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
          CK_CUDA_THROW_(cudaMemcpyAsync(value_dst_buf, src_buf, hash_table_value_tile_size_in_B,
                                         cudaMemcpyHostToHost, device_resources[id].get_stream()));
          src_buf += hash_table_value_tile_size_in_B;
          tensor_index_dst_buf =
              h_hash_table_index_chunk_per_gpu[id] + tile_counter_in_chunk_per_gpu[id];
          *tensor_index_dst_buf = tensor_index;
          tile_counter_in_chunk_per_gpu[id] += tile_size;

        } else {
          src_buf += hash_table_key_tile_size_in_B;
          src_buf += hash_table_slot_id_tile_size_in_B;
          src_buf += hash_table_value_tile_size_in_B;
          continue;
        }
      }  // end of for(int k = 0; k < (chunk_loop * local_gpu_count); k++)

      // memcpy hash_table_slot_id and hash_table_value from CPU to GPU
      for (size_t id = 0; id < local_gpu_count; id++) {
        if (tile_counter_in_chunk_per_gpu[id] == 0) {
          continue;
        }

        context.set_device(device_resources[id].get_device_id());

        // Copy value buffer and tensor_index buffer to GPU
        size_t value_chunk_size = tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
        float *src_buf_value = h_hash_table_value_chunk_per_gpu[id];
        float *dst_buf_value = d_hash_table_value_chunk_per_gpu[id];
        CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf_value, src_buf_value,
                                       value_chunk_size * sizeof(float), cudaMemcpyHostToDevice,
                                       device_resources[id].get_stream()));
        size_t *src_buf_index = h_hash_table_index_chunk_per_gpu[id];
        size_t *dst_buf_index = d_hash_table_index_chunk_per_gpu[id];
        value_chunk_size = tile_counter_in_chunk_per_gpu[id];
        CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf_index, src_buf_index,
                                       value_chunk_size * sizeof(size_t), cudaMemcpyHostToDevice,
                                       device_resources[id].get_stream()));

        // Call kernel to insert the value into embedding value tensor
        const size_t grid_size = (tile_counter_in_chunk_per_gpu[id] - 1) / 256 + 1;
        upload_value_tensor_kernel<<<grid_size, 256, 0, device_resources[id].get_stream()>>>(
            d_hash_table_value_chunk_per_gpu[id], d_hash_table_index_chunk_per_gpu[id],
            hash_table_value_tensors[id]->get_ptr(), hash_table_value_tile_size,
            tile_counter_in_chunk_per_gpu[id]);
      }

      sync_all_gpus(device_resources);

      // set counter value
      for (size_t id = 0; id < local_gpu_count; id++) {
        tile_counter_in_chunk_per_gpu[id] = 0;  // reset chunk counter to zero
      }

      /*       std::cout << "\rUploading " << std::fixed << std::setprecision(2)
                      << (float)(i) / loop_num * 100.0f << "%, loop " << i << " of " << loop_num
                      << std::flush; */
    }  // end of for(int i = 0; i < loop_num; i++)

    // std::cout << std::endl;

    // process the remaining data(less than a chunk)
    size_t remain_size_in_B = file_size_in_B - loop_num * hash_table_chunk_size_in_B;
    size_t remain_loop_num = remain_size_in_B / hash_table_tile_size_in_B;
    if (remain_loop_num) {
      MESSAGE_("Upload the remaining data");
      // read all the remaining data
      weight_stream.read((char *)hash_table_chunk, remain_size_in_B);

      char *src_buf = hash_table_chunk;
      float *value_dst_buf;
      size_t *tensor_index_dst_buf;
      for (size_t i = 0; i < remain_loop_num; i++) {  // process one tile in each loop

        TypeHashValueIndex slot_id =
            *((TypeHashValueIndex *)(src_buf + hash_table_key_tile_size_in_B));
        size_t gid = slot_id % total_gpu_count;           // global GPU ID
        size_t id = device_resources.get_local_id(gid);   // local GPU ID (not gpudevice id)
        size_t dst_rank = device_resources.get_pid(gid);  // node id

        if (static_cast<size_t>(my_rank) == dst_rank) {
          TypeHashKey tile_key = *((TypeHashKey *)src_buf);
          size_t tensor_index =
              tile_key - (h_mapping_offsets_per_gpu_tensors[id][local_slot_id[slot_id]]);

          src_buf += hash_table_key_tile_size_in_B;
          src_buf += hash_table_slot_id_tile_size_in_B;
          // memcpy hash_table_value to corresponding GPU
          value_dst_buf = h_hash_table_value_chunk_per_gpu[id] +
                          tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
          CK_CUDA_THROW_(cudaMemcpyAsync(value_dst_buf, src_buf, hash_table_value_tile_size_in_B,
                                         cudaMemcpyHostToHost, device_resources[id].get_stream()));
          src_buf += hash_table_value_tile_size_in_B;
          tensor_index_dst_buf =
              h_hash_table_index_chunk_per_gpu[id] + tile_counter_in_chunk_per_gpu[id];
          *tensor_index_dst_buf = tensor_index;
          tile_counter_in_chunk_per_gpu[id] += tile_size;

        } else {
          src_buf += hash_table_key_tile_size_in_B;
          src_buf += hash_table_slot_id_tile_size_in_B;
          src_buf += hash_table_value_tile_size_in_B;
          continue;
        }
      }

      // memcpy hash_table_slot_id and hash_table_value from CPU to GPU and insert into embedding
      // table
      for (size_t id = 0; id < local_gpu_count; id++) {
        if (tile_counter_in_chunk_per_gpu[id] == 0) {
          continue;
        }

        context.set_device(device_resources[id].get_device_id());

        // Copy value buffer and tensor_index buffer to GPU
        size_t value_chunk_size = tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
        float *src_buf_value = h_hash_table_value_chunk_per_gpu[id];
        float *dst_buf_value = d_hash_table_value_chunk_per_gpu[id];
        CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf_value, src_buf_value,
                                       value_chunk_size * sizeof(float), cudaMemcpyHostToDevice,
                                       device_resources[id].get_stream()));
        size_t *src_buf_index = h_hash_table_index_chunk_per_gpu[id];
        size_t *dst_buf_index = d_hash_table_index_chunk_per_gpu[id];
        value_chunk_size = tile_counter_in_chunk_per_gpu[id];
        CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf_index, src_buf_index,
                                       value_chunk_size * sizeof(size_t), cudaMemcpyHostToDevice,
                                       device_resources[id].get_stream()));

        // Call kernel to insert the value into embedding value tensor
        const size_t grid_size = (tile_counter_in_chunk_per_gpu[id] - 1) / 256 + 1;
        upload_value_tensor_kernel<<<grid_size, 256, 0, device_resources[id].get_stream()>>>(
            d_hash_table_value_chunk_per_gpu[id], d_hash_table_index_chunk_per_gpu[id],
            hash_table_value_tensors[id]->get_ptr(), hash_table_value_tile_size,
            tile_counter_in_chunk_per_gpu[id]);
      }

      // sync wait
      sync_all_gpus(device_resources);

    }  // end of if(remain_loop_num)

    MESSAGE_("Done");

    // release resources
    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());
      CK_CUDA_THROW_(cudaFree(d_hash_table_value_chunk_per_gpu[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_index_chunk_per_gpu[id]));
    }
    CK_CUDA_THROW_(cudaFreeHost(hash_table_chunk));
    for (size_t id = 0; id < local_gpu_count; id++) {
      CK_CUDA_THROW_(cudaFreeHost(h_hash_table_value_chunk_per_gpu[id]));
      CK_CUDA_THROW_(cudaFreeHost(h_hash_table_index_chunk_per_gpu[id]));
      CK_CUDA_THROW_(cudaFreeHost(h_mapping_offsets_per_gpu_tensors[id]));
    }
  }

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
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void download_params_to_host(
      std::ofstream &weight_stream, size_t vocabulary_size, size_t embedding_vec_size,
      const TensorPtrs<float> &hash_table_value_tensors,
      const std::vector<std::shared_ptr<
          HashTable<TypeHashKey, TypeHashValueIndex, std::numeric_limits<TypeHashKey>::max()>>>
          &hash_tables,
      const GPUResourceGroup &device_resources) const {
    CudaDeviceContext context;
    size_t local_gpu_count = device_resources.size();

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

    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());
      auto count_tmp = hash_tables[id]->get_size(device_resources[id].get_stream());
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
    for (size_t id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      context.set_device(device_resources[id].get_device_id());

      cudaMallocHost(&h_hash_table_key[id], count[id] * sizeof(TypeHashKey));
      cudaMalloc(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey));
      cudaMalloc(&d_hash_table_value_index[id], count[id] * sizeof(TypeHashValueIndex));
      cudaMallocHost(&h_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float));
      cudaMalloc(&d_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float));
      cudaMalloc(&d_dump_counter[id], count[id] * sizeof(size_t));
    }

    // dump hash table from GPUs
    for (size_t id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      MESSAGE_("Rank" + std::to_string(my_rank) + ": Dump hash table from GPU" +
               std::to_string(id));

      context.set_device(device_resources[id].get_device_id());

      hash_tables[id]->dump(d_hash_table_key[id], d_hash_table_value_index[id], d_dump_counter[id],
                            device_resources[id].get_stream());

      CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_key[id], d_hash_table_key[id],
                                     count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost,
                                     device_resources[id].get_stream()));

      get_hash_value(device_resources[id].get_stream(), count[id], embedding_vec_size,
                     d_hash_table_value_index[id], hash_table_value_tensors[id]->get_ptr(),
                     d_hash_table_value[id]);

      CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_value[id], d_hash_table_value[id],
                                     count[id] * embedding_vec_size * sizeof(float),
                                     cudaMemcpyDeviceToHost, device_resources[id].get_stream()));
    }

    // sync wait
    sync_all_gpus(device_resources);

    const int master_node = 0;
#ifdef ENABLE_MPI
    const int base_tag = 0xed;
#endif
    // TODO: could be optimized ???
    size_t max_size_in_B = max_count * (sizeof(TypeHashKey) + sizeof(float) * embedding_vec_size);
    std::unique_ptr<char[]> file_buf(new char[max_size_in_B]);
    size_t key_size = sizeof(TypeHashKey);
    size_t value_size = sizeof(float) * embedding_vec_size;
    for (size_t id = 0; id < local_gpu_count; id++) {
      size_t size_in_B = count[id] * (sizeof(TypeHashKey) + sizeof(float) * embedding_vec_size);
      size_t offset = 0;
      for (unsigned int k = 0; k < count[id]; k++) {
        memcpy(file_buf.get() + offset, h_hash_table_key[id] + k, key_size);
        offset += key_size;
        memcpy(file_buf.get() + offset, h_hash_table_value[id] + k * embedding_vec_size,
               value_size);
        offset += value_size;

        /*         std::cout << "\rRank" << my_rank << ": Seperate keys and values on GPU" << id <<
           ", finish "
                          << k << " of total count " << count[id] << ", " << (float)k / count[id] *
           100.0f
                          << "%" << std::flush; */
      }
      // std::cout << std::endl;
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
        for (size_t id = 0; id < local_gpu_count; id++) {
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

    for (size_t id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      context.set_device(device_resources[id].get_device_id());

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
   * @param hash_table_value_tensors the hash table value on multi-GPU.
   * @param hash_table_slot_id_tensors the hash table slot_ids on multi-GPU
   * @param hash_tables the hash tables on multi GPUs
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void download_params_to_host(
      std::ofstream &weight_stream, size_t vocabulary_size, size_t embedding_vec_size,
      const TensorPtrs<float> &hash_table_value_tensors,
      const TensorPtrs<TypeHashValueIndex> &hash_table_slot_id_tensors,
      const std::vector<std::shared_ptr<
          HashTable<TypeHashKey, TypeHashValueIndex, std::numeric_limits<TypeHashKey>::max()>>>
          &hash_tables,
      const GPUResourceGroup &device_resources) const {
    CudaDeviceContext context;
    size_t local_gpu_count = device_resources.size();

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

    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());
      auto count_tmp = hash_tables[id]->get_size(device_resources[id].get_stream());
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
    for (size_t id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      context.set_device(device_resources[id].get_device_id());

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
    for (size_t id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      MESSAGE_("Rank" + std::to_string(my_rank) + ": Dump hash table from GPU" +
               std::to_string(id));

      context.set_device(device_resources[id].get_device_id());

      hash_tables[id]->dump(d_hash_table_key[id], d_hash_table_value_index[id], d_dump_counter[id],
                            device_resources[id].get_stream());

      CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_key[id], d_hash_table_key[id],
                                     count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost,
                                     device_resources[id].get_stream()));

      get_hash_value(device_resources[id].get_stream(), count[id], embedding_vec_size,
                     d_hash_table_value_index[id], hash_table_value_tensors[id]->get_ptr(),
                     d_hash_table_value[id]);

      CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_value[id], d_hash_table_value[id],
                                     count[id] * embedding_vec_size * sizeof(float),
                                     cudaMemcpyDeviceToHost, device_resources[id].get_stream()));

      get_hash_slot_id(device_resources[id].get_stream(), count[id], d_hash_table_value_index[id],
                       hash_table_slot_id_tensors[id]->get_ptr(), d_hash_table_slot_id[id]);

      CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_slot_id[id], d_hash_table_slot_id[id],
                                     count[id] * sizeof(TypeHashValueIndex), cudaMemcpyDeviceToHost,
                                     device_resources[id].get_stream()));
    }

    // sync wait
    sync_all_gpus(device_resources);

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
    for (size_t id = 0; id < local_gpu_count; id++) {
      size_t size_in_B = count[id] * pair_size_in_B;
      size_t offset = 0;
      for (unsigned int k = 0; k < count[id]; k++) {
        /*         std::cout << "\rRank" << my_rank << ": Seperate keys, slot_ids and values on GPU"
           << id
                          << ", finish " << k << " of total count " << count[id] << ", "
                          << (float)k / count[id] * 100.0f << "%" << std::flush;
         */
        memcpy(file_buf.get() + offset, h_hash_table_key[id] + k, key_size);
        offset += key_size;
        memcpy(file_buf.get() + offset, h_hash_table_slot_id[id] + k, slot_id_size);
        offset += slot_id_size;
        memcpy(file_buf.get() + offset, h_hash_table_value[id] + k * embedding_vec_size,
               value_size);
        offset += value_size;
      }
      // std::cout << std::endl;
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
        for (size_t id = 0; id < local_gpu_count; id++) {
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

    for (size_t id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      context.set_device(device_resources[id].get_device_id());

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
   * download_params_to_host for LocalizedSlotSparseEmbeddingOnehot.
   * @param weight_stream weight file stream to write.
   * @param embedding_vec_size embedding vector size.
   * @param hash_table_value_tensors the hash table value on multi-GPU.
   * @param slot_sizes the size for each slot
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void download_params_to_host(std::ofstream &weight_stream, size_t embedding_vec_size,
                               const TensorPtrs<float> &hash_table_value_tensors,
                               const std::vector<size_t> &slot_sizes,
                               const GPUResourceGroup &device_resources) const {
    CudaDeviceContext context;
    size_t local_gpu_count = device_resources.size();

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
    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());
      count[id] = 0;
      for (size_t i = 0; i < slot_sizes.size(); i++) {
        size_t device_id = device_resources[id].get_device_id();
        size_t global_id = device_resources.get_global_id(device_id);
        if ((i % device_resources.get_total_gpu_count()) == global_id) {
          count[id] += slot_sizes[i];
        }
      }
      max_count = max(max_count, count[id]);
      total_count += count[id];
    }

#ifdef ENABLE_MPI
    CK_MPI_THROW_(
        MPI_Allreduce(MPI_IN_PLACE, &max_count, sizeof(size_t), MPI_CHAR, MPI_MAX, MPI_COMM_WORLD));
#endif

    /*if (total_count > (size_t)vocabulary_size) {
      CK_THROW_(Error_t::WrongInput,
                "Error: required download size is larger than hash table vocabulary_size");
    }*/

    std::unique_ptr<TypeHashKey *[]> h_hash_table_key(new TypeHashKey *[local_gpu_count]);
    std::unique_ptr<TypeHashKey *[]> d_hash_table_key(new TypeHashKey *[local_gpu_count]);
    std::unique_ptr<TypeHashValueIndex *[]> h_hash_table_slot_id(
        new TypeHashValueIndex *[local_gpu_count]);
    std::unique_ptr<TypeHashValueIndex *[]> d_hash_table_slot_id(
        new TypeHashValueIndex *[local_gpu_count]);
    std::unique_ptr<float *[]> h_hash_table_value(new float *[local_gpu_count]);
    for (size_t id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      context.set_device(device_resources[id].get_device_id());

      cudaMallocHost(&h_hash_table_key[id], count[id] * sizeof(TypeHashKey));
      cudaMalloc(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey));
      cudaMallocHost(&h_hash_table_slot_id[id], count[id] * sizeof(TypeHashValueIndex));
      cudaMalloc(&d_hash_table_slot_id[id], count[id] * sizeof(TypeHashValueIndex));
      cudaMallocHost(&h_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float));
    }

    // Generate key and slot_id tensor, dump value tensor on GPU
    for (size_t id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      MESSAGE_("Rank" + std::to_string(my_rank) + ": Dump embedding table from GPU" +
               std::to_string(id));

      context.set_device(device_resources[id].get_device_id());

      // Loop for each slot
      size_t buffer_offset = 0;
      for (size_t i = 0; i < slot_sizes.size(); i++) {
        size_t device_id = device_resources[id].get_device_id();
        size_t global_id = device_resources.get_global_id(device_id);
        if ((i % device_resources.get_total_gpu_count()) == global_id) {
          // Generate key buffer
          size_t key_offset = 0;
          for (size_t j = 0; j < i; j++) {
            key_offset += slot_sizes[j];
          }
          memset_liner(d_hash_table_key[id] + buffer_offset, (TypeHashKey)key_offset,
                       (TypeHashKey)1, slot_sizes[i], device_resources[id].get_stream());

          // Generate slot_id
          memset_const(d_hash_table_slot_id[id] + buffer_offset, (TypeHashValueIndex)i,
                       slot_sizes[i], device_resources[id].get_stream());

          buffer_offset += slot_sizes[i];
        }
      }
      // Copy key buffer to host
      CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_key[id], d_hash_table_key[id],
                                     count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost,
                                     device_resources[id].get_stream()));
      // Copy value buffer to host
      CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_value[id],
                                     hash_table_value_tensors[id]->get_ptr(),
                                     count[id] * embedding_vec_size * sizeof(float),
                                     cudaMemcpyDeviceToHost, device_resources[id].get_stream()));
      // Copy slot_id to host
      CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_slot_id[id], d_hash_table_slot_id[id],
                                     count[id] * sizeof(TypeHashValueIndex), cudaMemcpyDeviceToHost,
                                     device_resources[id].get_stream()));
    }

    // sync wait
    sync_all_gpus(device_resources);

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
    for (size_t id = 0; id < local_gpu_count; id++) {
      size_t size_in_B = count[id] * pair_size_in_B;
      size_t offset = 0;
      for (unsigned int k = 0; k < count[id]; k++) {
        /*         std::cout << "\rRank" << my_rank << ": Seperate keys, slot_ids and values on GPU"
           << id
                          << ", finish " << k << " of total count " << count[id] << ", "
                          << (float)k / count[id] * 100.0f << "%" << std::flush; */

        memcpy(file_buf.get() + offset, h_hash_table_key[id] + k, key_size);
        offset += key_size;
        memcpy(file_buf.get() + offset, h_hash_table_slot_id[id] + k, slot_id_size);
        offset += slot_id_size;
        memcpy(file_buf.get() + offset, h_hash_table_value[id] + k * embedding_vec_size,
               value_size);
        offset += value_size;
      }
      // std::cout << std::endl;
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
        for (size_t id = 0; id < local_gpu_count; id++) {
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

    for (size_t id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      context.set_device(device_resources[id].get_device_id());

      CK_CUDA_THROW_(cudaFreeHost(h_hash_table_key[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_key[id]));
      CK_CUDA_THROW_(cudaFreeHost(h_hash_table_slot_id[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_slot_id[id]));
      CK_CUDA_THROW_(cudaFreeHost(h_hash_table_value[id]));
    }

    return;
  }

  /**
   * get forward results from GPUs to CPU. This functin is just used for utest.
   * @param memcpy_size the number of elemments to do memcpy.
   * @param embedding_feature_tensors the source tensors of multi GPUs to copy from.
   * @param embedding_feature the destination CPU buffer pointer to copy to.
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  template <typename TypeEmbeddingComp>
  void get_forward_results(size_t memcpy_size,
                           const TensorPtrs<TypeEmbeddingComp> &embedding_feature_tensors,
                           TypeEmbeddingComp *embedding_feature,
                           TensorPtrs<TypeEmbeddingComp> &temp_tensors,
                           const GPUResourceGroup &device_resources) {
    CudaDeviceContext context;
    size_t local_gpu_count = device_resources.size();
    size_t total_gpu_count = device_resources.get_total_gpu_count();

    if (total_gpu_count > 1) {
      // nccl allGather
      all_gather(memcpy_size,
                 embedding_feature_tensors,  // send
                 temp_tensors,               // recv
                 device_resources);

      // memcpy D2H
      context.set_device(device_resources[0].get_device_id());
      CK_CUDA_THROW_(cudaMemcpyAsync(embedding_feature, temp_tensors[0]->get_ptr(),
                                     total_gpu_count * memcpy_size * sizeof(TypeEmbeddingComp),
                                     cudaMemcpyDeviceToHost, device_resources[0].get_stream()));
      CK_CUDA_THROW_(cudaStreamSynchronize(device_resources[0].get_stream()));
    } else {
      context.set_device(device_resources[0].get_device_id());
      CK_CUDA_THROW_(cudaMemcpyAsync(embedding_feature, embedding_feature_tensors[0]->get_ptr(),
                                     memcpy_size * sizeof(TypeEmbeddingComp),
                                     cudaMemcpyDeviceToHost, device_resources[0].get_stream()));
      CK_CUDA_THROW_(cudaStreamSynchronize(device_resources[0].get_stream()));
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
  void get_backward_results(size_t devId, size_t memcpy_size,
                            const TensorPtrs<TypeEmbeddingComp> &wgrad_tensors,
                            TypeEmbeddingComp *wgrad, const GPUResourceGroup &device_resources) {
    CudaDeviceContext context(device_resources[devId].get_device_id());
    CK_CUDA_THROW_(cudaMemcpyAsync(wgrad, wgrad_tensors[devId]->get_ptr(),
                                   memcpy_size * sizeof(TypeEmbeddingComp), cudaMemcpyDeviceToHost,
                                   device_resources[devId].get_stream()));
    CK_CUDA_THROW_(cudaStreamSynchronize(device_resources[devId].get_stream()));

    return;
  }

  /**
   * get update_params results from GPU to CPU. This functin is just used for utest.
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
      size_t embedding_vec_size, size_t vocabulary_size,
      const TensorPtrs<float> &hash_table_value_tensors,
      const std::vector<std::shared_ptr<
          HashTable<TypeHashKey, TypeHashValueIndex, std::numeric_limits<TypeHashKey>::max()>>>
          &hash_tables,
      TypeHashKey *hash_table_key, float *hash_table_value,
      const GPUResourceGroup &device_resources) {
    CudaDeviceContext context;

    size_t local_gpu_count = device_resources.size();

    // memory allocation
    std::unique_ptr<size_t[]> count(new size_t[local_gpu_count]);
    size_t total_count = 0;
    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(device_resources[id].get_device_id());
      if ((count[id] = hash_tables[id]->get_value_head()) !=
          hash_tables[id]->get_size(device_resources[id].get_stream())) {
        std::cout << "hashtable: get_value_head()=" << hash_tables[id]->get_value_head()
                  << ", get_size()=" << hash_tables[id]->get_size(device_resources[id].get_stream())
                  << std::endl;
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
    for (size_t id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      context.set_device(device_resources[id].get_device_id());

      cudaMalloc(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey));
      cudaMalloc(&d_hash_table_value_index[id], count[id] * sizeof(TypeHashValueIndex));
      cudaMalloc(&d_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float));
      cudaMalloc(&d_dump_counter[id], count[id] * sizeof(size_t));
    }

    // dump hash table on GPU
    for (size_t id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      context.set_device(device_resources[id].get_device_id());

      hash_tables[id]->dump(d_hash_table_key[id], d_hash_table_value_index[id], d_dump_counter[id],
                            device_resources[id].get_stream());

      get_hash_value(device_resources[id].get_stream(), count[id], embedding_vec_size,
                     d_hash_table_value_index[id], hash_table_value_tensors[id]->get_ptr(),
                     d_hash_table_value[id]);
    }

    // sync wait
    sync_all_gpus(device_resources);

    // memcpy from GPU to CPU memory
    size_t key_offset = 0;
    size_t value_offset = 0;
    for (size_t id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      context.set_device(device_resources[id].get_device_id());

      CK_CUDA_THROW_(cudaMemcpy(hash_table_key + key_offset, d_hash_table_key[id],
                                count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost));
      key_offset += count[id];

      CK_CUDA_THROW_(cudaMemcpy(hash_table_value + value_offset, d_hash_table_value[id],
                                count[id] * embedding_vec_size * sizeof(float),
                                cudaMemcpyDeviceToHost));
      value_offset += count[id] * embedding_vec_size;
    }

    for (size_t id = 0; id < local_gpu_count; id++) {
      if (count[id] == 0) {
        continue;
      }

      context.set_device(device_resources[id].get_device_id());

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
   * @param row_offsets_tensors row_offsets tensors of mulitple GPUs (CSR format of input
   * sparse tensors)
   * @param value_index_tensors hash value index tensors of multi GPUs
   * @param slot_id_tensors slot id tensors for multi GPUs
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  template <typename TypeKey>
  void store_slot_id(size_t batch_size, size_t slot_num,
                     const std::vector<size_t> &slot_num_per_gpu,
                     const TensorPtrs<TypeKey> &row_offset_tensors,
                     const TensorPtrs<TypeKey> &value_index_tensors,
                     TensorPtrs<TypeKey> &slot_id_tensors,
                     const GPUResourceGroup &device_resources) {
    CudaDeviceContext context;
    size_t local_gpu_count = device_resources.size();
    size_t total_gpu_count = device_resources.get_total_gpu_count();

    for (size_t id = 0; id < local_gpu_count; id++) {
      if (slot_num_per_gpu[id] == 0) {
        continue;
      }

      size_t local_device_id = device_resources[id].get_device_id();
      size_t global_id = device_resources.get_global_id(local_device_id);

      const size_t block_size = 64;
      const size_t grid_size = (batch_size * slot_num_per_gpu[id] + block_size - 1) / block_size;

      context.set_device(local_device_id);
      store_slot_id_kernel<<<grid_size, block_size, 0, device_resources[id].get_stream()>>>(
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
  void init_embedding_per_slot(size_t slot_id, size_t slot_size, size_t embedding_vec_size,
                               size_t key_offset, size_t value_index_offset, float *embedding_table,
                               HashTable<TypeHashKey, TypeHashValueIndex,
                                         std::numeric_limits<TypeHashKey>::max()> &hash_table,
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
    }
    CK_CUDA_THROW_(cudaMemcpyAsync(embedding_table, embedding_init.get(),
                                   slot_size * embedding_vec_size * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));

    memset_liner(hash_keys, (TypeHashKey)key_offset, (TypeHashKey)1, slot_size, stream);
    memset_liner(hash_value_indices, (TypeHashValueIndex)value_index_offset, (TypeHashValueIndex)1,
                 slot_size, stream);
    hash_table.insert(hash_keys, hash_value_indices, slot_size, stream);
    size_t value_head = hash_table.add_value_head(slot_size);

    memset_const(slot_ids, (TypeHashValueIndex)slot_id, slot_size, stream);

    CK_CUDA_THROW_(cudaStreamSynchronize(stream));

    CK_CUDA_THROW_(cudaFree(hash_keys));
    CK_CUDA_THROW_(cudaFree(hash_value_indices));
  }

  /**
   * Initialize one hash table slot and embedding slot on GPU. This function is only used by
   * LocalizedSparseEmbeddingHash.
   * @param slot_id slot id.
   * @param slot_size the size of the slot to be intialized.
   * @param embedding_vec_size embedding vector size.
   * @param offset the start address of the slot in embedding table.
   * @param embedding_table the pointer to the embedding slot.
   * @param slot_ids the pointer to the slot ids.
   * @param stream cuda stream.
   */
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void init_embedding_per_slot(size_t slot_id, size_t slot_size, size_t embedding_vec_size,
                               size_t key_offset, size_t value_index_offset, float *embedding_table,
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
    }
    CK_CUDA_THROW_(cudaMemcpyAsync(embedding_table, embedding_init.get(),
                                   slot_size * embedding_vec_size * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));

    memset_liner(hash_keys, (TypeHashKey)key_offset, (TypeHashKey)1, slot_size, stream);
    memset_liner(hash_value_indices, (TypeHashValueIndex)value_index_offset, (TypeHashValueIndex)1,
                 slot_size, stream);

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
  void init_embedding_per_gpu(size_t lid, size_t gid, size_t total_gpu_count,
                              const std::vector<size_t> &slot_sizes, size_t embedding_vec_size,
                              float *embedding_table,
                              HashTable<TypeHashKey, TypeHashValueIndex,
                                        std::numeric_limits<TypeHashKey>::max()> &hash_table,
                              TypeHashValueIndex *slot_ids,
                              const GPUResourceGroup &device_resources) {
    CudaDeviceContext context(device_resources[lid].get_device_id());

    size_t key_offset = 0;
    size_t value_index_offset = 0;
    for (size_t i = 0; i < slot_sizes.size(); i++) {
      size_t slot_size = slot_sizes[i];
      if ((i % total_gpu_count) == gid) {
        MESSAGE_("gpu" + std::to_string(gid) + " start to init embedding of slot" +
                 std::to_string(i) + " , slot_size=" + std::to_string(slot_size) +
                 ", key_offset=" + std::to_string(key_offset) +
                 ", value_index_offset=" + std::to_string(value_index_offset));
        init_embedding_per_slot(i, slot_size, embedding_vec_size, key_offset, value_index_offset,
                                embedding_table, hash_table, slot_ids,
                                device_resources[lid].get_stream());
        value_index_offset += slot_size;
        embedding_table += slot_size * embedding_vec_size;
        slot_ids += slot_size;
      }
      key_offset += slot_size;
    }
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
   * @param slot_ids the pointer to the slot ids.
   * @param device_resources GPU device resources.
   */
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void init_embedding_per_gpu(size_t lid, size_t gid, size_t total_gpu_count,
                              const std::vector<size_t> &slot_sizes, size_t embedding_vec_size,
                              float *embedding_table, TypeHashValueIndex *slot_ids,
                              const GPUResourceGroup &device_resources) {
    CudaDeviceContext context(device_resources[lid].get_device_id());

    size_t key_offset = 0;
    size_t value_index_offset = 0;
    for (size_t i = 0; i < slot_sizes.size(); i++) {
      size_t slot_size = slot_sizes[i];
      if ((i % total_gpu_count) == gid) {
        MESSAGE_("gpu" + std::to_string(gid) + " start to init embedding of slot" +
                 std::to_string(i) + " , slot_size=" + std::to_string(slot_size) +
                 ", key_offset=" + std::to_string(key_offset) +
                 ", value_index_offset=" + std::to_string(value_index_offset));
        init_embedding_per_slot<TypeHashKey>(i, slot_size, embedding_vec_size, key_offset,
                                             value_index_offset, embedding_table, slot_ids,
                                             device_resources[lid].get_stream());
        value_index_offset += slot_size;
        embedding_table += slot_size * embedding_vec_size;
        slot_ids += slot_size;
      }
      key_offset += slot_size;
    }
  }

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
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void init_embedding(
      const std::vector<size_t> slot_sizes, size_t embedding_vec_size,
      TensorPtrs<float> &hash_table_value_tensors,
      std::vector<std::shared_ptr<HashTable<TypeHashKey, TypeHashValueIndex,
                                            std::numeric_limits<TypeHashKey>::max()>>> &hash_tables,
      TensorPtrs<TypeHashValueIndex> &hash_table_slot_id_tensors,
      const GPUResourceGroup &device_resources) {
    size_t local_gpu_count = device_resources.size();
    size_t total_gpu_count = device_resources.get_total_gpu_count();

#ifndef NDEBUG
    MESSAGE_("local_gpu_count=" + std::to_string(local_gpu_count) +
             ", total_gpu_count=" + std::to_string(total_gpu_count));
#endif

    for (size_t id = 0; id < local_gpu_count; id++) {
      size_t device_id = device_resources[id].get_device_id();
      size_t global_id = device_resources.get_global_id(device_id);

#ifndef NDEBUG
      MESSAGE_("id=" + std::to_string(id) + ", device_id=" + std::to_string(device_id) +
               ", global_id=" + std::to_string(global_id));
#endif

      init_embedding_per_gpu(id, global_id, total_gpu_count, slot_sizes, embedding_vec_size,
                             hash_table_value_tensors[id]->get_ptr(), *hash_tables[id],
                             hash_table_slot_id_tensors[id]->get_ptr(), device_resources);
    }

    return;
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
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void init_embedding(const std::vector<size_t> slot_sizes, size_t embedding_vec_size,
                      const TensorPtrs<float> &hash_table_value_tensors,
                      const TensorPtrs<TypeHashValueIndex> &hash_table_slot_id_tensors,
                      const GPUResourceGroup &device_resources) {
    size_t local_gpu_count = device_resources.size();
    size_t total_gpu_count = device_resources.get_total_gpu_count();

#ifndef NDEBUG
    MESSAGE_("local_gpu_count=" + std::to_string(local_gpu_count) +
             ", total_gpu_count=" + std::to_string(total_gpu_count));
#endif

    for (size_t id = 0; id < local_gpu_count; id++) {
      size_t device_id = device_resources[id].get_device_id();
      size_t global_id = device_resources.get_global_id(device_id);

#ifndef NDEBUG
      MESSAGE_("id=" + std::to_string(id) + ", device_id=" + std::to_string(device_id) +
               ", global_id=" + std::to_string(global_id));
#endif

      init_embedding_per_gpu<TypeHashKey>(
          id, global_id, total_gpu_count, slot_sizes, embedding_vec_size,
          hash_table_value_tensors[id]->get_ptr(), hash_table_slot_id_tensors[id]->get_ptr(),
          device_resources);
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
  void init_embedding(size_t max_vocabulary_size_per_gpu, size_t embedding_vec_size,
                      const TensorPtrs<float> &hash_table_value_tensors,
                      const GPUResourceGroup &device_resources) {
    size_t num = max_vocabulary_size_per_gpu * embedding_vec_size;
    PinnedBuffer<float> h_hash_table_value(num);

    HugeCTR::UnifiedDataSimulator<float> fdata_sim(-0.05, 0.05);
    for (size_t i = 0; i < num; i++) {
      h_hash_table_value[i] = fdata_sim.get_num();
    }

    CudaDeviceContext context(device_resources[0].get_device_id());
    size_t local_gpu_count = device_resources.size();
    for (size_t id = 0; id < local_gpu_count; id++) {
      size_t cur_device = device_resources[id].get_device_id();
      context.set_device(cur_device);

      MESSAGE_("gpu" + std::to_string(id) + " start to init embedding");

      CK_CUDA_THROW_(cudaMemcpyAsync(hash_table_value_tensors[id]->get_ptr(),
                                     h_hash_table_value.get(), num * sizeof(float),
                                     cudaMemcpyHostToDevice, device_resources[id].get_stream()));
    }

    for (size_t id = 0; id < local_gpu_count; id++) {
      CK_CUDA_THROW_(cudaStreamSynchronize(device_resources[id].get_stream()));
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
};  // end of SparseEmbeddingHashFunctors

}  // end of namespace HugeCTR
