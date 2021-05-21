/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include "HugeCTR/include/data_simulator.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/utils.cuh"
#include "HugeCTR/include/utils.hpp"
#include "cub/cub/device/device_radix_sort.cuh"
#include "cub/cub/device/device_scan.cuh"

namespace HugeCTR {

namespace {

// get slot_id from hash_table_slot_id vector by value_index
__global__ void get_hash_slot_id_kernel(size_t count, const size_t *value_index,
                                        const size_t *hash_table_slot_id, size_t *slot_id) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < count) {
    size_t index = value_index[gid];
    slot_id[gid] = hash_table_slot_id[index];
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
void get_hash_slot_id(size_t count, const size_t *value_index, const size_t *hash_table_slot_id,
                      size_t *slot_id, cudaStream_t stream) {
  const size_t block_size = 64;
  const size_t grid_size = (count + block_size - 1) / block_size;

  get_hash_slot_id_kernel<<<grid_size, block_size, 0, stream>>>(count, value_index,
                                                                hash_table_slot_id, slot_id);
}

}  // namespace

template <typename TypeHashKey, typename TypeEmbeddingComp>
LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::LocalizedSlotSparseEmbeddingHash(
    const Tensors2<TypeHashKey> &train_row_offsets_tensors,
    const Tensors2<TypeHashKey> &train_value_tensors,
    const std::vector<std::shared_ptr<size_t>> &train_nnz_array,
    const Tensors2<TypeHashKey> &evaluate_row_offsets_tensors,
    const Tensors2<TypeHashKey> &evaluate_value_tensors,
    const std::vector<std::shared_ptr<size_t>> &evaluate_nnz_array,
    const SparseEmbeddingHashParams<TypeEmbeddingComp> &embedding_params,
    const std::string plan_file, const std::shared_ptr<ResourceManager> &resource_manager)
    : Base(train_row_offsets_tensors, train_value_tensors, train_nnz_array,
           evaluate_row_offsets_tensors, evaluate_value_tensors, evaluate_nnz_array,
           embedding_params, resource_manager),
      slot_size_array_(embedding_params.slot_size_array)
#ifndef NCCL_A2A
      ,
      plan_file_(plan_file)
#endif
{
  try {
    CudaDeviceContext context;

    if (slot_size_array_.empty()) {
      max_vocabulary_size_per_gpu_ = Base::get_max_vocabulary_size_per_gpu();
      max_vocabulary_size_ = Base::get_max_vocabulary_size_per_gpu() *
                             Base::get_resource_manager().get_global_gpu_count();
    } else {
      max_vocabulary_size_per_gpu_ =
          cal_max_voc_size_per_gpu(slot_size_array_, Base::get_resource_manager());
      max_vocabulary_size_ = 0;
      for (size_t slot_size : slot_size_array_) {
        max_vocabulary_size_ += slot_size;
      }
    }

    MESSAGE_("max_vocabulary_size_per_gpu_=" + std::to_string(max_vocabulary_size_per_gpu_));

    for (size_t id = 0; id < Base::get_resource_manager().get_local_gpu_count(); id++) {
      int cur_device = Base::get_local_gpu(id).get_device_id();
      context.set_device(cur_device);

      size_t gid = Base::get_local_gpu(id).get_global_id();
      size_t slot_num_per_gpu =
          Base::get_slot_num() / Base::get_resource_manager().get_global_gpu_count() +
          ((gid < Base::get_slot_num() % Base::get_resource_manager().get_global_gpu_count()) ? 1
                                                                                              : 0);
      slot_num_per_gpu_.push_back(slot_num_per_gpu);

      // construct HashTable object: used to store hash table <key, value_index>
      hash_tables_.emplace_back(new NvHashTable(max_vocabulary_size_per_gpu_));

      // new GeneralBuffer objects
      const std::shared_ptr<GeneralBuffer2<CudaAllocator>> &buf = Base::get_buffer(id);

      // new hash table value vectors
      if (slot_size_array_.empty()) {
        Tensor2<float> tensor;
        buf->reserve({max_vocabulary_size_per_gpu_, Base::get_embedding_vec_size()}, &tensor);
        hash_table_value_tensors_.push_back(tensor);
      } else {
        const std::shared_ptr<BufferBlock2<float>> &block = buf->create_block<float>();
        Tensors2<float> tensors;
        for (size_t i = 0; i < slot_size_array_.size(); i++) {
          if ((i % Base::get_resource_manager().get_global_gpu_count()) == gid) {
            Tensor2<float> tensor;
            block->reserve({slot_size_array_[i], Base::get_embedding_vec_size()}, &tensor);
            tensors.push_back(tensor);
          }
        }
        value_table_tensors_.push_back(tensors);
        hash_table_value_tensors_.push_back(block->as_tensor());
      }

      // new hash table value_index that get() from HashTable
      {
        Tensor2<size_t> tensor;
        buf->reserve({1, Base::get_universal_batch_size() * Base::get_max_feature_num()}, &tensor);
        hash_value_index_tensors_.push_back(tensor);
      }

      // new embedding features reduced by hash table values(results of forward)
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve(
            {Base::get_universal_batch_size() * slot_num_per_gpu, Base::get_embedding_vec_size()},
            &tensor);
        embedding_feature_tensors_.push_back(tensor);
      }

      // new wgrad used by backward
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve(
            {Base::get_batch_size(true) * slot_num_per_gpu, Base::get_embedding_vec_size()},
            &tensor);
        wgrad_tensors_.push_back(tensor);
      }

      // new optimizer params used by update_params
      switch (Base::get_optimizer()) {
        case Optimizer_t::Adam:  // adam
        {
          {
            Tensor2<TypeEmbeddingComp> tensor;
            buf->reserve({max_vocabulary_size_per_gpu_, Base::get_embedding_vec_size()}, &tensor);
            opt_m_tensors_.push_back(tensor);
            buf->reserve({max_vocabulary_size_per_gpu_, Base::get_embedding_vec_size()}, &tensor);
            opt_v_tensors_.push_back(tensor);
          }
          if (Base::get_update_type() == Update_t::LazyGlobal) {
            Tensor2<uint64_t> tensor;
            buf->reserve({max_vocabulary_size_per_gpu_, Base::get_embedding_vec_size()}, &tensor);
            opt_prev_time_tensors_.push_back(tensor);
          }
          break;
        }

        case Optimizer_t::MomentumSGD:  // momentum_sgd
        {
          Tensor2<TypeEmbeddingComp> tensor;
          buf->reserve({max_vocabulary_size_per_gpu_, Base::get_embedding_vec_size()}, &tensor);
          opt_momentum_tensors_.push_back(tensor);
          break;
        }

        case Optimizer_t::Nesterov:  // nesterov
        {
          Tensor2<TypeEmbeddingComp> tensor;
          buf->reserve({max_vocabulary_size_per_gpu_, Base::get_embedding_vec_size()}, &tensor);
          opt_accm_tensors_.push_back(tensor);
          break;
        }

        case Optimizer_t::SGD:
          break;

        default:
          throw std::runtime_error(
              std::string("[HCDEBUG][ERROR] Runtime error: Invalid optimizer type\n"));
      }

      {
        Tensor2<TypeHashKey> tensor;
        buf->reserve({1, Base::get_batch_size(true) * Base::get_max_feature_num()}, &tensor);
        sample_id_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeHashKey> tensor;
        buf->reserve({1, Base::get_batch_size(true) * Base::get_max_feature_num()}, &tensor);
        sample_id_sort_tensors_.push_back(tensor);
      }
      {
        Tensor2<size_t> tensor;
        buf->reserve({1, Base::get_batch_size(true) * Base::get_max_feature_num()}, &tensor);
        hash_value_index_sort_tensors_.push_back(tensor);
      }
      {
        Tensor2<uint32_t> tensor;
        buf->reserve({1, Base::get_batch_size(true) * Base::get_max_feature_num() + 1}, &tensor);
        hash_value_index_count_offset_tensors_.push_back(tensor);
      }
      {
        Tensor2<uint32_t> tensor;
        buf->reserve({1, Base::get_batch_size(true) * Base::get_max_feature_num()}, &tensor);
        new_hash_value_flag_tensors_.push_back(tensor);
      }
      {
        Tensor2<uint32_t> tensor;
        buf->reserve({1, Base::get_batch_size(true) * Base::get_max_feature_num()}, &tensor);
        hash_value_flag_sumed_tensors_.push_back(tensor);
      }
      {
        Tensor2<uint32_t> tensor;
        buf->reserve({1, 1}, &tensor);
        hash_value_index_count_counter_tensors_.push_back(tensor);
      }
      {
        // cal the temp storage bytes for CUB radix sort
        size_t size = 0;
        cub::DeviceRadixSort::SortPairs((void *)nullptr, size, (size_t *)nullptr, (size_t *)nullptr,
                                        (TypeHashKey *)nullptr, (TypeHashKey *)nullptr,
                                        Base::get_batch_size(true) * Base::get_max_feature_num());

        // new temp storage tensors for CUB radix sort
        Tensor2<void> tensor;
        buf->reserve({size}, &tensor);
        temp_storage_sort_tensors_.push_back(tensor);
      }

      {
        size_t size = 0;
        cub::DeviceScan::InclusiveSum((void *)nullptr, size, (uint32_t *)nullptr,
                                      (uint32_t *)nullptr,
                                      Base::get_batch_size(true) * Base::get_max_feature_num());

        Tensor2<void> tensor;
        buf->reserve({size}, &tensor);
        temp_storage_scan_tensors_.push_back(tensor);
      }

      // the tenosrs for storing slot ids
      // TODO: init to -1 ?
      {
        Tensor2<size_t> tensor;
        buf->reserve({max_vocabulary_size_per_gpu_, 1}, &tensor);
        hash_table_slot_id_tensors_.push_back(tensor);
      }
      // temp tensors for all2all
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({Base::get_universal_batch_size_per_gpu() * Base::get_slot_num(),
                      Base::get_embedding_vec_size()},
                     &tensor);
        all2all_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({Base::get_universal_batch_size() * Base::get_slot_num(),
                      Base::get_embedding_vec_size()},
                     &tensor);
        utest_forward_temp_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({Base::get_batch_size_per_gpu(true) * Base::get_slot_num(),
                      Base::get_embedding_vec_size()},
                     &tensor);
        utest_all2all_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({Base::get_batch_size_per_gpu(true) * Base::get_slot_num(),
                      Base::get_embedding_vec_size()},
                     &tensor);
        utest_reorder_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve(
            {Base::get_batch_size(true) * Base::get_slot_num(), Base::get_embedding_vec_size()},
            &tensor);
        utest_backward_temp_tensors_.push_back(tensor);
      }

// init GenenralBuffers to do real allocation
#ifndef NDEBUG
      std::cout << " max_feature_num_:" << Base::get_max_feature_num() << std::endl;
#endif
      buf->allocate();

      const OptParams<TypeEmbeddingComp> &source_opt_param = Base::get_opt_params();
      OptParams<TypeEmbeddingComp> &target_opt_param = Base::get_opt_params(id);

      switch (Base::get_optimizer()) {
        case Optimizer_t::Adam:  // adam
          CK_CUDA_THROW_(cudaMemsetAsync(opt_m_tensors_[id].get_ptr(), 0,
                                         opt_m_tensors_[id].get_size_in_bytes(),
                                         Base::get_local_gpu(id).get_stream()));
          CK_CUDA_THROW_(cudaMemsetAsync(opt_v_tensors_[id].get_ptr(), 0,
                                         opt_v_tensors_[id].get_size_in_bytes(),
                                         Base::get_local_gpu(id).get_stream()));
          if (Base::get_update_type() == Update_t::LazyGlobal) {
            dim3 grid(Base::get_local_gpu(id).get_sm_count() * 4, 1, 1);
            dim3 block(512, 1, 1);
            initialize_array<<<grid, block, 0, Base::get_local_gpu(id).get_stream()>>>(
                opt_prev_time_tensors_[id].get_ptr(), opt_prev_time_tensors_[id].get_num_elements(),
                uint64_t(1));
            target_opt_param.hyperparams.adam.prev_time_ptr = opt_prev_time_tensors_[id].get_ptr();
          }
          target_opt_param.hyperparams.adam.times = 0;
          target_opt_param.hyperparams.adam.beta1 = source_opt_param.hyperparams.adam.beta1;
          target_opt_param.hyperparams.adam.beta2 = source_opt_param.hyperparams.adam.beta2;
          target_opt_param.hyperparams.adam.epsilon = source_opt_param.hyperparams.adam.epsilon;
          target_opt_param.hyperparams.adam.m_ptr = opt_m_tensors_[id].get_ptr();
          target_opt_param.hyperparams.adam.v_ptr = opt_v_tensors_[id].get_ptr();
          break;

        case Optimizer_t::MomentumSGD:  // momentum_sgd
          CK_CUDA_THROW_(cudaMemsetAsync(opt_momentum_tensors_[id].get_ptr(), 0,
                                         opt_momentum_tensors_[id].get_size_in_bytes(),
                                         Base::get_local_gpu(id).get_stream()));
          target_opt_param.hyperparams.momentum.factor =
              source_opt_param.hyperparams.momentum.factor;
          target_opt_param.hyperparams.momentum.momentum_ptr = opt_momentum_tensors_[id].get_ptr();
          break;

        case Optimizer_t::Nesterov:  // nesterov
          CK_CUDA_THROW_(cudaMemsetAsync(opt_accm_tensors_[id].get_ptr(), 0,
                                         opt_accm_tensors_[id].get_size_in_bytes(),
                                         Base::get_local_gpu(id).get_stream()));
          target_opt_param.hyperparams.nesterov.mu = source_opt_param.hyperparams.nesterov.mu;
          target_opt_param.hyperparams.nesterov.accm_ptr = opt_accm_tensors_[id].get_ptr();
          break;

        case Optimizer_t::SGD:
          break;

        default:
          throw std::runtime_error(
              std::string("[HCDEBUG][ERROR] Runtime error: Invalid optimizer type\n"));
      }

    }  // end of for(int id = 0; id < Base::get_local_gpu_count(); id++)

    // sync
    functors_.sync_all_gpus(Base::get_resource_manager());

#ifndef NCCL_A2A
// all2all init
#ifndef ENABLE_MPI  // without MPI
    functors_.all2all_init_forward(all2all_forward_, plan_file_, Base::get_batch_size_per_gpu(true),
                                   slot_num_per_gpu_, Base::get_embedding_vec_size(),
                                   embedding_feature_tensors_, all2all_tensors_,
                                   Base::get_resource_manager());
    functors_.all2all_init_backward(all2all_backward_, plan_file_,
                                    Base::get_batch_size_per_gpu(true), slot_num_per_gpu_,
                                    Base::get_embedding_vec_size(), all2all_tensors_,
                                    embedding_feature_tensors_, Base::get_resource_manager());
    functors_.all2all_init_forward(all2all_utest_, plan_file_, Base::get_batch_size_per_gpu(true),
                                   slot_num_per_gpu_, Base::get_embedding_vec_size(),
                                   wgrad_tensors_, utest_all2all_tensors_,
                                   Base::get_resource_manager());
#else
    functors_.all2all_init_forward(all2all_forward_, plan_file_, Base::get_batch_size_per_gpu(true),
                                   Base::get_slot_num(), Base::get_embedding_vec_size(),
                                   embedding_feature_tensors_, all2all_tensors_,
                                   Base::get_resource_manager());
    functors_.all2all_init_backward(all2all_backward_, plan_file_,
                                    Base::get_batch_size_per_gpu(true), Base::get_slot_num(),
                                    Base::get_embedding_vec_size(), all2all_tensors_,
                                    embedding_feature_tensors_, Base::get_resource_manager());
    functors_.all2all_init_forward(all2all_utest_, plan_file_, Base::get_batch_size_per_gpu(true),
                                   Base::get_slot_num(), Base::get_embedding_vec_size(),
                                   wgrad_tensors_, utest_all2all_tensors_,
                                   Base::get_resource_manager());
#endif

#endif

// warm up for nccl all2all
#ifdef NCCL_A2A
    MESSAGE_("All2All Warmup Start");
#ifndef ENABLE_MPI
    if (Base::get_resource_manager().get_global_gpu_count() > 1) {
      functors_.all2all_forward(Base::get_batch_size_per_gpu(true), slot_num_per_gpu_,
                                Base::get_embedding_vec_size(), embedding_feature_tensors_,
                                all2all_tensors_, Base::get_resource_manager());
    }
#else
    if (Base::get_resource_manager().get_global_gpu_count() > 1) {
      functors_.all2all_forward(Base::get_batch_size_per_gpu(true), Base::get_slot_num(),
                                Base::get_embedding_vec_size(), embedding_feature_tensors_,
                                all2all_tensors_, Base::get_resource_manager());
    }
#endif
    MESSAGE_("All2All Warmup End");
#endif

  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::load_parameters(
    std::istream &weight_stream, size_t vocabulary_size, size_t embedding_vec_size,
    size_t max_vocabulary_size_per_gpu, Tensors2<float> &hash_table_value_tensors,
    Tensors2<size_t> &hash_table_slot_id_tensors,
    std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables) {
  // check file size and vocabulary_size (file size <=　hash_table_size)
  weight_stream.seekg(0, weight_stream.end);
  size_t file_size_in_B = weight_stream.tellg();
  weight_stream.seekg(0, weight_stream.beg);

  // define size
  size_t local_gpu_count = Base::get_resource_manager().get_local_gpu_count();
  size_t chunk_loop = 1000;
  size_t tile_size = 1;  // must be 1, because we need to cal (key&local_gpu_count) to decide
                         // gpu_id for each <key,value>
  size_t hash_table_key_tile_size = tile_size;
  size_t hash_table_key_tile_size_in_B = hash_table_key_tile_size * sizeof(TypeHashKey);
  size_t hash_table_key_chunk_size = hash_table_key_tile_size * chunk_loop;
  size_t hash_table_key_chunk_size_in_B = hash_table_key_chunk_size * sizeof(TypeHashKey);
  size_t hash_table_value_index_chunk_size_in_B = hash_table_key_chunk_size * sizeof(size_t);
  size_t hash_table_value_tile_size = tile_size * embedding_vec_size;
  size_t hash_table_value_tile_size_in_B = hash_table_value_tile_size * sizeof(float);
  size_t hash_table_value_chunk_size = hash_table_value_tile_size * chunk_loop;
  size_t hash_table_value_chunk_size_in_B = hash_table_value_chunk_size * sizeof(float);
  size_t hash_table_slot_id_tile_size = tile_size;
  size_t hash_table_slot_id_tile_size_in_B = hash_table_slot_id_tile_size * sizeof(size_t);
  size_t hash_table_slot_id_chunk_size = hash_table_slot_id_tile_size * chunk_loop;
  size_t hash_table_slot_id_chunk_size_in_B = hash_table_slot_id_chunk_size * sizeof(size_t);
  size_t hash_table_tile_size_in_B = hash_table_key_tile_size_in_B +
                                     hash_table_slot_id_tile_size_in_B +
                                     hash_table_value_tile_size_in_B;
  size_t hash_table_chunk_size_in_B = hash_table_tile_size_in_B * chunk_loop;
  size_t total_gpu_count = Base::get_resource_manager().get_global_gpu_count();

  // CAUSION: can not decide how many values for each GPU, so need to allocate enough memory
  // for each GPU allocate GPU memory for hash_table_value_index
  std::unique_ptr<size_t[]> tile_counter_per_gpu(
      new size_t[local_gpu_count]);  // <= hash_table_value_index_per_gpu_size
  memset(tile_counter_per_gpu.get(), 0, sizeof(size_t) * local_gpu_count);
  std::unique_ptr<size_t[]> tile_counter_in_chunk_per_gpu(new size_t[local_gpu_count]);
  memset(tile_counter_in_chunk_per_gpu.get(), 0, sizeof(size_t) * local_gpu_count);
  std::unique_ptr<size_t *[]> d_hash_table_value_index_chunk_per_gpu(new size_t *[local_gpu_count]);

  CudaDeviceContext context;
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(Base::get_local_gpu(id).get_device_id());
    CK_CUDA_THROW_(cudaMalloc(&d_hash_table_value_index_chunk_per_gpu[id],
                              hash_table_value_index_chunk_size_in_B));
    // initalize to zeros
    CK_CUDA_THROW_(cudaMemsetAsync(d_hash_table_value_index_chunk_per_gpu[id], 0,
                                   hash_table_value_index_chunk_size_in_B,
                                   Base::get_local_gpu(id).get_stream()));
  }

  // sync wait
  functors_.sync_all_gpus(Base::get_resource_manager());

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
    context.set_device(Base::get_local_gpu(id).get_device_id());
    CK_CUDA_THROW_(cudaMalloc(&d_hash_table_key_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
  }
  std::unique_ptr<size_t *[]> h_hash_table_slot_id_chunk_per_gpu(new size_t *[local_gpu_count]);
  for (size_t id = 0; id < local_gpu_count; id++) {
    CK_CUDA_THROW_(cudaMallocHost(&h_hash_table_slot_id_chunk_per_gpu[id],
                                  hash_table_slot_id_chunk_size_in_B));
  }
  std::unique_ptr<size_t *[]> d_hash_table_slot_id_chunk_per_gpu(new size_t *[local_gpu_count]);
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(Base::get_local_gpu(id).get_device_id());
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
           std::to_string(file_size_in_B) + " Bytes, total loop_num: " + std::to_string(loop_num));
  for (size_t i = 0; i < loop_num; i++) {
    // read a chunk of data from file
    // one pair in hash table file includes: <key, slot_id, value>
    weight_stream.read(hash_table_chunk, hash_table_chunk_size_in_B);

    // memcpy from CPU to CPU
    char *src_buf = hash_table_chunk;
    TypeHashKey *key_dst_buf;
    size_t *slot_id_dst_buf;
    float *value_dst_buf;
    for (size_t k = 0; k < chunk_loop; k++) {  // process a tile in each loop
      size_t slot_id = *((size_t *)(src_buf + hash_table_key_tile_size_in_B));
      size_t gid = slot_id % total_gpu_count;  // global GPU ID
      size_t id = Base::get_resource_manager().get_gpu_local_id_from_global_id(
          gid);  // local GPU ID (not gpudevice id)
      int dst_rank =
          Base::get_resource_manager().get_process_id_from_gpu_global_id(gid);  // node id

      if (Base::get_resource_manager().get_process_id() == dst_rank) {
        // memcpy hash_table_key to corresponding GPU
        key_dst_buf = h_hash_table_key_chunk_per_gpu[id] +
                      tile_counter_in_chunk_per_gpu[id] * hash_table_key_tile_size;

        memcpy(key_dst_buf, src_buf, hash_table_key_tile_size_in_B);

        src_buf += hash_table_key_tile_size_in_B;

        // memcpy hash_table_slot_id to corresponding GPU
        slot_id_dst_buf = h_hash_table_slot_id_chunk_per_gpu[id] +
                          tile_counter_in_chunk_per_gpu[id] * hash_table_slot_id_tile_size;

        memcpy(slot_id_dst_buf, src_buf, hash_table_slot_id_tile_size_in_B);

        src_buf += hash_table_slot_id_tile_size_in_B;

        // memcpy hash_table_value to corresponding GPU
        value_dst_buf = h_hash_table_value_chunk_per_gpu[id] +
                        tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;

        memcpy(value_dst_buf, src_buf, hash_table_value_tile_size_in_B);

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

      context.set_device(Base::get_local_gpu(id).get_device_id());

      size_t tile_count = tile_counter_in_chunk_per_gpu[id];

      // memcpy hash_table_key from CPU to GPU
      CK_CUDA_THROW_(cudaMemcpyAsync(d_hash_table_key_chunk_per_gpu[id],
                                     h_hash_table_key_chunk_per_gpu[id],
                                     tile_count * sizeof(TypeHashKey), cudaMemcpyHostToDevice,
                                     Base::get_local_gpu(id).get_stream()));

      size_t value_index_offset = tile_counter_per_gpu[id];
      size_t *value_index_buf = d_hash_table_value_index_chunk_per_gpu[id];

      if (tile_count > 0) {
        // set hash_table_value_index on GPU
        functors_.memset_liner(value_index_buf, value_index_offset, 1ul, tile_count,
                               Base::get_local_gpu(id).get_stream());
      }

      // do hash table insert <key, value_index> on GPU
      hash_tables[id]->insert(d_hash_table_key_chunk_per_gpu[id], value_index_buf, tile_count,
                              Base::get_local_gpu(id).get_stream());
      size_t value_head =
          hash_tables[id]->get_and_add_value_head(tile_count, Base::get_local_gpu(id).get_stream());
    }

    // memcpy hash_table_slot_id and hash_table_value from CPU to GPU
    for (size_t id = 0; id < local_gpu_count; id++) {
      if (tile_counter_in_chunk_per_gpu[id] == 0) {
        continue;
      }

      context.set_device(Base::get_local_gpu(id).get_device_id());

      size_t slot_id_chunk_size = tile_counter_in_chunk_per_gpu[id] * hash_table_slot_id_tile_size;
      size_t slot_id_offset = tile_counter_per_gpu[id] * hash_table_slot_id_tile_size;

      if ((slot_id_offset + slot_id_chunk_size) > max_vocabulary_size_per_gpu) {
        char msg[100]{0};
        sprintf(msg, "The size of hash table on GPU%zu is out of range %zu\n", id,
                max_vocabulary_size_per_gpu);
        CK_THROW_(Error_t::OutOfBound, msg);
      }

      size_t *src_buf_sid = h_hash_table_slot_id_chunk_per_gpu[id];
      size_t *dst_buf_sid = hash_table_slot_id_tensors[id].get_ptr() + slot_id_offset;
      CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf_sid, src_buf_sid, slot_id_chunk_size * sizeof(size_t),
                                     cudaMemcpyHostToDevice, Base::get_local_gpu(id).get_stream()));

      size_t value_chunk_size = tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
      size_t value_chunk_offset = tile_counter_per_gpu[id] * hash_table_value_tile_size;
      float *src_buf_value = h_hash_table_value_chunk_per_gpu[id];
      float *dst_buf_value = hash_table_value_tensors[id].get_ptr() + value_chunk_offset;
      CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf_value, src_buf_value, value_chunk_size * sizeof(float),
                                     cudaMemcpyHostToDevice, Base::get_local_gpu(id).get_stream()));
    }

    functors_.sync_all_gpus(Base::get_resource_manager());

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
    size_t *value_index_buf;
    size_t *slot_id_dst_buf;
    float *value_dst_buf;
    for (size_t i = 0; i < remain_loop_num; i++) {  // process one tile in each loop

      size_t slot_id = *((size_t *)(src_buf + hash_table_key_tile_size_in_B));
      size_t gid = slot_id % total_gpu_count;  // global GPU ID
      size_t id = Base::get_resource_manager().get_gpu_local_id_from_global_id(
          gid);  // local GPU ID (not gpu devie id)
      int dst_rank =
          Base::get_resource_manager().get_process_id_from_gpu_global_id(gid);  // node id

      if (Base::get_resource_manager().get_process_id() == dst_rank) {
        context.set_device(Base::get_local_gpu(id).get_device_id());

        // memcpy hash_table_key from CPU to GPU
        key_dst_buf = d_hash_table_key_chunk_per_gpu[id];
        CK_CUDA_THROW_(cudaMemcpyAsync(key_dst_buf, src_buf, hash_table_key_tile_size_in_B,
                                       cudaMemcpyHostToDevice,
                                       Base::get_local_gpu(id).get_stream()));
        src_buf += hash_table_key_tile_size_in_B;

        // set value_index
        size_t value_index_offset = tile_counter_per_gpu[id];
        value_index_buf = d_hash_table_value_index_chunk_per_gpu[id];
        functors_.memset_liner(value_index_buf, value_index_offset, 1ul, 1ul,
                               Base::get_local_gpu(id).get_stream());

        // do hash table insert <key, value_index> on GPU
        hash_tables[id]->insert(d_hash_table_key_chunk_per_gpu[id], value_index_buf,
                                hash_table_key_tile_size, Base::get_local_gpu(id).get_stream());
        size_t value_head = hash_tables[id]->get_and_add_value_head(
            hash_table_key_tile_size, Base::get_local_gpu(id).get_stream());

        // memcpy hash_table_slot_id to corresponding GPU
        size_t slot_id_offset = tile_counter_per_gpu[id];
        slot_id_dst_buf = hash_table_slot_id_tensors[id].get_ptr() + slot_id_offset;
        CK_CUDA_THROW_(cudaMemcpyAsync(slot_id_dst_buf, src_buf, hash_table_slot_id_tile_size_in_B,
                                       cudaMemcpyHostToDevice,
                                       Base::get_local_gpu(id).get_stream()));
        src_buf += hash_table_slot_id_tile_size_in_B;

        // memcpy hash_table_value from CPU to GPU
        size_t value_offset = tile_counter_per_gpu[id] * embedding_vec_size;
        value_dst_buf = hash_table_value_tensors[id].get_ptr() + value_offset;
        CK_CUDA_THROW_(cudaMemcpyAsync(value_dst_buf, src_buf, hash_table_value_tile_size_in_B,
                                       cudaMemcpyHostToDevice,
                                       Base::get_local_gpu(id).get_stream()));
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
    functors_.sync_all_gpus(Base::get_resource_manager());

  }  // end of if(remain_loop_num)

  MESSAGE_("Done");

  // release resources
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(Base::get_local_gpu(id).get_device_id());
    CK_CUDA_THROW_(cudaFree(d_hash_table_value_index_chunk_per_gpu[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_key_chunk_per_gpu[id]));
  }
  CK_CUDA_THROW_(cudaFreeHost(hash_table_chunk));
  for (size_t id = 0; id < local_gpu_count; id++) {
    CK_CUDA_THROW_(cudaFreeHost(h_hash_table_key_chunk_per_gpu[id]));
    CK_CUDA_THROW_(cudaFreeHost(h_hash_table_value_chunk_per_gpu[id]));
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::dump_parameters(
    std::ostream &weight_stream, size_t vocabulary_size, size_t embedding_vec_size,
    const Tensors2<float> &hash_table_value_tensors,
    const Tensors2<size_t> &hash_table_slot_id_tensors,
    const std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables) const {
  size_t local_gpu_count = Base::get_resource_manager().get_local_gpu_count();

  // memory allocation
  std::unique_ptr<size_t[]> count(new size_t[local_gpu_count]);
  size_t max_count = 0;
  size_t total_count = 0;

  CudaDeviceContext context;
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(Base::get_local_gpu(id).get_device_id());
    auto count_tmp = hash_tables[id]->get_size(Base::get_local_gpu(id).get_stream());
    if (count_tmp != hash_tables[id]->get_value_head(Base::get_local_gpu(id).get_stream())) {
      std::cout << "gpu" << id << ", get_size=" << count_tmp << ", get_value_head="
                << hash_tables[id]->get_value_head(Base::get_local_gpu(id).get_stream())
                << std::endl;
      CK_THROW_(Error_t::WrongInput,
                "Error: hash_table get_value_head() is not equal to get_size()");
    }
    count[id] = count_tmp;
    max_count = max(max_count, count[id]);
    total_count += count[id];
  }

#ifdef ENABLE_MPI
  CK_MPI_THROW_(
      MPI_Allreduce(MPI_IN_PLACE, &max_count, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD));
#endif

  if (total_count > (size_t)vocabulary_size) {
    CK_THROW_(Error_t::WrongInput,
              "Error: required download size is larger than hash table vocabulary_size");
  }

  std::unique_ptr<TypeHashKey *[]> h_hash_table_key(new TypeHashKey *[local_gpu_count]);
  std::unique_ptr<TypeHashKey *[]> d_hash_table_key(new TypeHashKey *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_hash_table_value_index(new size_t *[local_gpu_count]);
  std::unique_ptr<size_t *[]> h_hash_table_slot_id(new size_t *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_hash_table_slot_id(new size_t *[local_gpu_count]);
  std::unique_ptr<float *[]> h_hash_table_value(new float *[local_gpu_count]);
  std::unique_ptr<float *[]> d_hash_table_value(new float *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_dump_counter(new size_t *[local_gpu_count]);

  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) {
      continue;
    }

    context.set_device(Base::get_local_gpu(id).get_device_id());

    cudaMallocHost(&h_hash_table_key[id], count[id] * sizeof(TypeHashKey));
    cudaMalloc(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey));
    cudaMalloc(&d_hash_table_value_index[id], count[id] * sizeof(size_t));
    cudaMallocHost(&h_hash_table_slot_id[id], count[id] * sizeof(size_t));
    cudaMalloc(&d_hash_table_slot_id[id], count[id] * sizeof(size_t));
    cudaMallocHost(&h_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float));
    cudaMalloc(&d_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float));
    cudaMalloc(&d_dump_counter[id], count[id] * sizeof(size_t));
  }

  // dump hash table on GPU
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) {
      continue;
    }

    MESSAGE_("Rank" + std::to_string(Base::get_resource_manager().get_process_id()) +
             ": Dump hash table from GPU" + std::to_string(id));

    context.set_device(Base::get_local_gpu(id).get_device_id());

    hash_tables[id]->dump(d_hash_table_key[id], d_hash_table_value_index[id], d_dump_counter[id],
                          Base::get_local_gpu(id).get_stream());

    CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_key[id], d_hash_table_key[id],
                                   count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost,
                                   Base::get_local_gpu(id).get_stream()));

    functors_.get_hash_value(count[id], embedding_vec_size, d_hash_table_value_index[id],
                             hash_table_value_tensors[id].get_ptr(), d_hash_table_value[id],
                             Base::get_local_gpu(id).get_stream());

    CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_value[id], d_hash_table_value[id],
                                   count[id] * embedding_vec_size * sizeof(float),
                                   cudaMemcpyDeviceToHost, Base::get_local_gpu(id).get_stream()));

    get_hash_slot_id(count[id], d_hash_table_value_index[id],
                     hash_table_slot_id_tensors[id].get_ptr(), d_hash_table_slot_id[id],
                     Base::get_local_gpu(id).get_stream());

    CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_slot_id[id], d_hash_table_slot_id[id],
                                   count[id] * sizeof(size_t), cudaMemcpyDeviceToHost,
                                   Base::get_local_gpu(id).get_stream()));
  }

  // sync wait
  functors_.sync_all_gpus(Base::get_resource_manager());

#ifdef ENABLE_MPI
  const int base_tag = 0xed;
#endif
  // TODO: could be optimized ???
  // one pair in the file includes <key,slot_id,value>
  size_t pair_size_in_B = sizeof(TypeHashKey) + sizeof(size_t) + sizeof(float) * embedding_vec_size;
  size_t max_size_in_B = max_count * pair_size_in_B;
  std::unique_ptr<char[]> file_buf(new char[max_size_in_B]);
  size_t key_size = sizeof(TypeHashKey);
  size_t slot_id_size = sizeof(size_t);
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
      memcpy(file_buf.get() + offset, h_hash_table_value[id] + k * embedding_vec_size, value_size);
      offset += value_size;
    }
    if (Base::get_resource_manager().is_master_process()) {
      MESSAGE_("Rank" + std::to_string(Base::get_resource_manager().get_process_id()) +
               ": Write hash table <key,value> pairs to file");
      weight_stream.write(file_buf.get(), size_in_B);
    }
#ifdef ENABLE_MPI
    else {
      MESSAGE_("Rank" + std::to_string(Base::get_resource_manager().get_process_id()) +
               ": Send hash table <key,value> pairs on GPU" + std::to_string(id) +
               " to master node  ");
      int tag = (id << 8) | base_tag;
      CK_MPI_THROW_(MPI_Send(file_buf.get(), size_in_B, MPI_CHAR,
                             Base::get_resource_manager().get_master_process_id(), tag,
                             MPI_COMM_WORLD));
    }
#endif
  }

#ifdef ENABLE_MPI
  if (Base::get_resource_manager().is_master_process()) {
    for (int r = 1; r < Base::get_resource_manager().get_num_process(); r++) {
      for (size_t id = 0; id < local_gpu_count; id++) {
        MESSAGE_("Rank" + std::to_string(Base::get_resource_manager().get_process_id()) +
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

    context.set_device(Base::get_local_gpu(id).get_device_id());

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
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::init_embedding(
    size_t max_vocabulary_size_per_gpu, size_t embedding_vec_size,
    Tensors2<float> &hash_table_value_tensors) {
  CudaDeviceContext context;
  size_t local_gpu_count = Base::get_resource_manager().get_local_gpu_count();
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(Base::get_local_gpu(id).get_device_id());

    MESSAGE_("gpu" + std::to_string(id) + " start to init embedding");

    HugeCTR::UniformGenerator::fill(
        hash_table_value_tensors[id].get_ptr(), 
        hash_table_value_tensors[id].get_num_elements(), 
        -0.05f, 0.05f,
        Base::get_local_gpu(id).get_sm_count(),
        Base::get_local_gpu(id).get_replica_variant_curand_generator(),
        Base::get_local_gpu(id).get_stream());
  }

  for (size_t id = 0; id < local_gpu_count; id++) {
    CK_CUDA_THROW_(cudaStreamSynchronize(Base::get_local_gpu(id).get_stream()));
    MESSAGE_("gpu" + std::to_string(id) + " init embedding done");
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::init_embedding(
    const std::vector<size_t> &slot_sizes, size_t embedding_vec_size,
    std::vector<Tensors2<float>> &hash_table_value_tensors,
    Tensors2<size_t> &hash_table_slot_id_tensors) {
  size_t local_gpu_count = Base::get_resource_manager().get_local_gpu_count();
  size_t total_gpu_count = Base::get_resource_manager().get_global_gpu_count();

#ifndef NDEBUG
  MESSAGE_("local_gpu_count=" + std::to_string(local_gpu_count) +
           ", total_gpu_count=" + std::to_string(total_gpu_count));
#endif

  for (size_t id = 0; id < local_gpu_count; id++) {
    size_t device_id = Base::get_local_gpu(id).get_device_id();
    size_t global_id = Base::get_local_gpu(id).get_global_id();

#ifndef NDEBUG
    MESSAGE_("id=" + std::to_string(id) + ", device_id=" + std::to_string(device_id) +
             ", global_id=" + std::to_string(global_id));
#endif

    functors_.init_embedding_per_gpu(global_id, total_gpu_count, slot_sizes, embedding_vec_size,
                                     hash_table_value_tensors[id], hash_table_slot_id_tensors[id],
                                     Base::get_local_gpu(id));
  }

  for (size_t id = 0; id < local_gpu_count; id++) {
    CK_CUDA_THROW_(cudaStreamSynchronize(Base::get_local_gpu(id).get_stream()));
    MESSAGE_("gpu" + std::to_string(id) + " init embedding done");
  }

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::reset() {
  CudaDeviceContext context;
  for (size_t i = 0; i < Base::get_resource_manager().get_local_gpu_count(); i++) {
    context.set_device(Base::get_local_gpu(i).get_device_id());
    hash_tables_[i]->clear(Base::get_local_gpu(i).get_stream());

    if (slot_size_array_.empty()) {
      HugeCTR::UniformGenerator::fill(
          hash_table_value_tensors_[i].get_ptr(),
          hash_table_value_tensors_[i].get_num_elements(),
          -0.05f, 0.05f,
          Base::get_local_gpu(i).get_sm_count(),
          Base::get_local_gpu(i).get_replica_variant_curand_generator(),
          Base::get_local_gpu(i).get_stream());
    } else {
      functors_.init_embedding_per_gpu(Base::get_local_gpu(i).get_global_id(),
                                       Base::get_resource_manager().get_global_gpu_count(),
                                       slot_size_array_, Base::get_embedding_vec_size(),
                                       value_table_tensors_[i], hash_table_slot_id_tensors_[i],
                                       Base::get_local_gpu(i));
    }
  }

  for (size_t i = 0; i < Base::get_resource_manager().get_local_gpu_count(); i++) {
    CK_CUDA_THROW_(cudaStreamSynchronize(Base::get_local_gpu(i).get_stream()));
  }
}

template class LocalizedSlotSparseEmbeddingHash<unsigned int, float>;
template class LocalizedSlotSparseEmbeddingHash<long long, float>;
template class LocalizedSlotSparseEmbeddingHash<unsigned int, __half>;
template class LocalizedSlotSparseEmbeddingHash<long long, __half>;

}  // namespace HugeCTR
