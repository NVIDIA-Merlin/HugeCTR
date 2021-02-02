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
#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/faster_gossip_comm/FasterComm.h"
#include "HugeCTR/include/hashtable/nv_hashtable.hpp"
#include "HugeCTR/include/resource_manager.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {

class SparseEmbeddingFunctors {
 public:
  /**
   * stream sync on multi GPUs.
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device.
   */
  void sync_all_gpus(const ResourceManager &resource_manager) const;

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
  void init_embedding_per_gpu(size_t gid, size_t total_gpu_count,
                              const std::vector<size_t> &slot_sizes, size_t embedding_vec_size,
                              Tensors2<float> &embedding_tables, Tensor2<size_t> &slot_ids,
                              const GPUResource &gpu_resource);

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
  template <typename TypeHashKey, typename TypeEmbeddingComp>
  void forward_per_gpu(size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner,
                       bool train, const Tensor2<TypeHashKey> &row_offset,
                       const Tensor2<TypeHashKey> &hash_key, size_t nnz,
                       HashTable<TypeHashKey, size_t> &hash_table,
                       const Tensor2<float> &hash_table_value, Tensor2<size_t> &hash_value_index,
                       Tensor2<TypeEmbeddingComp> &embedding_feature, cudaStream_t stream);

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
                     const Tensors2<TypeHashKey> &row_offset_allreduce_tensors,
                     Tensors2<TypeEmbeddingComp> &output_tensors,
                     const ResourceManager &resource_manager);

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
                       const Tensors2<TypeEmbeddingComp> &src_tensors,
                       Tensors2<TypeEmbeddingComp> &dst_tensors,
                       const ResourceManager &resource_manager);

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
  template <typename TypeHashKey>
  void forward_mapping_per_gpu(size_t batch_size, size_t slot_num,
                               const Tensor2<TypeHashKey> &hash_key, size_t nnz,
                               const Tensor2<uint32_t> &mapping_offsets,
                               Tensor2<size_t> &hash_value_index, cudaStream_t stream);

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
  template <typename TypeHashKey, typename TypeEmbeddingComp>
  void forward_fuse_per_gpu(size_t id, size_t local_gpu_count, size_t batch_size,
                            size_t batch_size_per_gpu, size_t slot_num, size_t slot_num_per_gpu,
                            size_t embedding_vec_size, int combiner,
                            const Tensor2<TypeHashKey> &row_offset,
                            const Tensor2<size_t> &hash_value_index,
                            const Tensor2<float> &hash_table_value,
                            Tensor2<TypeEmbeddingComp *> &embedding_features, size_t sm_count,
                            cudaStream_t stream);

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
                     const Tensors2<TypeKey> &row_offset_tensors,
                     const Tensors2<size_t> &value_index_tensors, Tensors2<size_t> &slot_id_tensors,
                     const ResourceManager &resource_manager);

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
                const Tensors2<TypeHashKey> &row_offset_allreduce_tensors,
                const Tensors2<TypeEmbeddingComp> &embedding_feature_tensors,
                Tensors2<TypeEmbeddingComp> &wgrad_tensors,
                const ResourceManager &resource_manager);

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
                const Tensors2<TypeHashKey> &row_offset_allreduce_tensors,
                const Tensors2<TypeEmbeddingComp> &embedding_feature_tensors,
                Tensors2<TypeEmbeddingComp> &wgrad_tensors,
                const ResourceManager &resource_manager);

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
                        const Tensors2<TypeEmbeddingComp> &src_tensors,
                        Tensors2<TypeEmbeddingComp> &dst_tensors,
                        const ResourceManager &resource_manager);

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
                             const Tensor2<TypeEmbeddingComp *> &embedding_features,
                             Tensor2<TypeEmbeddingComp> &wgrad, size_t sm, cudaStream_t stream);

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
   * @param hash_table_value the pointer of hash table value, which will be updated
   */
  template <typename TypeHashKey, typename TypeEmbeddingComp>
  void update_params(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                     size_t max_vocabulary_size_per_gpu, OptParams<TypeEmbeddingComp> &opt_params,
                     size_t nnz, const Tensor2<TypeHashKey> &row_offset,
                     Tensor2<size_t> &hash_value_index, Tensor2<TypeHashKey> &sample_id,
                     Tensor2<TypeHashKey> &sample_id_sort, Tensor2<size_t> &hash_value_index_sort,
                     Tensor2<uint32_t> &hash_value_index_count_offset,
                     Tensor2<uint32_t> &new_hash_value_flag,
                     Tensor2<uint32_t> &hash_value_flag_sumed,
                     Tensor2<uint32_t> &hash_value_index_count_counter,
                     Tensor2<void> &temp_storage_sort, Tensor2<void> &temp_storage_scan,
                     const Tensor2<TypeEmbeddingComp> &wgrad, Tensor2<float> &hash_table_value,
                     size_t sm_count, cudaStream_t stream);

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
  template <typename TypeEmbeddingComp>
  void update_params(size_t embedding_vec_size, const OptParams<TypeEmbeddingComp> &opt_params,
                     size_t nnz, const Tensor2<size_t> &hash_value_index,
                     const Tensor2<TypeEmbeddingComp> &wgrad, Tensor2<float> &hash_table_value,
                     size_t sm_count, cudaStream_t stream);

  /**
   * collection communication: reduce_scatter f or DistributedSlotSparseEmbeddingHash
   * @param recv_count the count of elements will be received.
   * @param send_tensors the send tensors of multi GPUs.
   * @param recv_tensors the recv tensors of multi GPUs.
   * @param device_resources all gpus device resources.
   */
  template <typename TypeEmbeddingComp>
  void reduce_scatter(size_t recv_count, const Tensors2<TypeEmbeddingComp> &send_tensors,
                      Tensors2<TypeEmbeddingComp> &recv_tensors,
                      const ResourceManager &resource_manager);

  /**
   * collection communication: all_reduce.
   * @param send_count the count of elements will be sent.
   * @param send_tensors the send tensors of multi GPUs.
   * @param recv_tensors the recv tensors of multi GPUs.
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device.
   */
  template <typename TypeHashKey>
  void all_reduce(size_t send_count, const Tensors2<TypeHashKey> &send_tensors,
                  Tensors2<TypeHashKey> &recv_tensors, const ResourceManager &resource_manager);

  /**
   * collection communication: all_gather.
   * @param send_count the count of elements will be sent.
   * @param send_tensors the send tensors of multi GPUs.
   * @param recv_tensors the recv tensors of multi GPUs.
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device.
   */
  template <typename TypeHashKey>
  void all_gather(size_t send_count, const Tensors2<TypeHashKey> &send_tensors,
                  Tensors2<TypeHashKey> &recv_tensors, const ResourceManager &resource_manager);

#ifdef NCCL_A2A
#ifdef ENABLE_MPI
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
                       const Tensors2<Type> &send_tensors, Tensors2<Type> &recv_tensors,
                       const ResourceManager &resource_manager);

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
                        const Tensors2<Type> &send_tensors, Tensors2<Type> &recv_tensors,
                        const ResourceManager &resource_manager);
#else
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
                       size_t embedding_vec_size, const Tensors2<Type> &send_tensors,
                       Tensors2<Type> &recv_tensors, const ResourceManager &resource_manager);

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
                        size_t embedding_vec_size, const Tensors2<Type> &send_tensors,
                        Tensors2<Type> &recv_tensors, const ResourceManager &resource_manager);
#endif
#else
#ifdef ENABLE_MPI
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
  void all2all_init_forward(std::unique_ptr<GossipComm::FasterComm> &all2all,
                            const std::string &plan_file, size_t batch_size_per_gpu,
                            size_t slot_num, size_t embedding_vec_size,
                            Tensors2<Type> &send_tensors, Tensors2<Type> &recv_tensors,
                            const ResourceManager &resource_manager);

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
  void all2all_init_backward(std::unique_ptr<GossipComm::FasterComm> &all2all,
                             const std::string &plan_file, size_t batch_size_per_gpu,
                             size_t slot_num, size_t embedding_vec_size,
                             Tensors2<Type> &send_tensors, Tensors2<Type> &recv_tensors,
                             const ResourceManager &resource_manager);

  /**
   * collection communication: all2all
   * @param all2all all2all handler
   */
  void all2all_exec(GossipComm::FasterComm &all2all);
#else
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
  void all2all_init_forward(std::unique_ptr<GossipComm::FasterComm> &all2all,
                            const std::string &plan_file, size_t batch_size_per_gpu,
                            const std::vector<size_t> &slot_num_per_gpu, size_t embedding_vec_size,
                            Tensors2<Type> &send_tensors, Tensors2<Type> &recv_tensors,
                            const ResourceManager &resource_manager);

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
  void all2all_init_backward(std::unique_ptr<GossipComm::FasterComm> &all2all,
                             const std::string &plan_file, size_t batch_size_per_gpu,
                             const std::vector<size_t> &slot_num_per_gpu, size_t embedding_vec_size,
                             Tensors2<Type> &send_tensors, Tensors2<Type> &recv_tensors,
                             const ResourceManager &resource_manager);

  /**
   * collection communication: all2all
   * @param all2all all2all handler
   */
  void all2all_exec(GossipComm::FasterComm &all2all);
#endif
#endif

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
                           const Tensors2<TypeEmbeddingComp> &embedding_feature_tensors,
                           Tensor2<TypeEmbeddingComp> &embedding_feature,
                           Tensors2<TypeEmbeddingComp> &temp_tensors,
                           const ResourceManager &resource_manager);

  /**
   * get forward results from GPUs to TensorFlow's tensor.
  */
  template <typename TypeEmbeddingComp>
  void get_forward_results(size_t memcpy_size,
                           const Tensors2<TypeEmbeddingComp> &embedding_feature_tensors,
                           void* const embedding_feature,
                           Tensors2<TypeEmbeddingComp> &temp_tensors,
                           const ResourceManager &resource_manager,
                           const bool on_gpu);

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
  void get_backward_results(size_t id, size_t memcpy_size,
                            const Tensors2<TypeEmbeddingComp> &wgrad_tensors,
                            Tensor2<TypeEmbeddingComp> &wgrad,
                            const ResourceManager &resource_manager);

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
  template <typename TypeHashKey>
  void get_update_params_results(
      size_t embedding_vec_size, size_t vocabulary_size,
      const Tensors2<float> &hash_table_value_tensors,
      const std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables,
      Tensor2<TypeHashKey> &hash_table_key, Tensor2<float> &hash_table_value,
      const ResourceManager &resource_manager);

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
                    cudaStream_t stream) const;

  /**
   * set constant data for a buffer
   * @param stream cuda stream.
   * @param data the pointer of the data buffer which will be written.
   * @param value the setting value
   * @param n the number of the data.
   */
  void memset_const(size_t *data, size_t value, size_t n, cudaStream_t stream) const;

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
  void get_hash_value(size_t count, size_t embedding_vec_size, const size_t *value_index,
                      const float *hash_table_value, float *value_retrieved,
                      cudaStream_t stream) const;

  template <typename TypeEmbeddingComp>
  void dump_opt_states(std::ofstream& stream, const ResourceManager &resource_manager,
                       std::vector<Tensors2<TypeEmbeddingComp>>& opt_states);
  template <typename TypeEmbeddingComp>
  void load_opt_states(std::ifstream& stream, const ResourceManager &resource_manager,
                       std::vector<Tensors2<TypeEmbeddingComp>>& opt_states);
};

}  // namespace HugeCTR
