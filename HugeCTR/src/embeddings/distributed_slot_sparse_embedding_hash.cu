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
#include "HugeCTR/include/data_simulator.hpp"
#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/utils.cuh"
#include "cub/cub/device/device_radix_sort.cuh"
#include "cub/cub/device/device_scan.cuh"

namespace HugeCTR {

template <typename TypeHashKey, typename TypeEmbeddingComp>
DistributedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::
    DistributedSlotSparseEmbeddingHash(
        const Tensors2<TypeHashKey> &train_row_offsets_tensors,
        const Tensors2<TypeHashKey> &train_value_tensors,
        const std::vector<std::shared_ptr<size_t>> &train_nnz_array,
        const Tensors2<TypeHashKey> &evaluate_row_offsets_tensors,
        const Tensors2<TypeHashKey> &evaluate_value_tensors,
        const std::vector<std::shared_ptr<size_t>> &evaluate_nnz_array,
        const SparseEmbeddingHashParams<TypeEmbeddingComp> &embedding_params,
        const std::shared_ptr<ResourceManager> &resource_manager)
    : Base(train_row_offsets_tensors, train_value_tensors, train_nnz_array,
           evaluate_row_offsets_tensors, evaluate_value_tensors, evaluate_nnz_array,
           embedding_params, resource_manager) {
  try {
    CudaDeviceContext context;

    // CAUSION: can not decide how many <key,value> pairs in each GPU, because the GPU
    // distribution is computed by (key%gpu_count). In order to not allocate the total size of
    // hash table on each GPU, meanwhile get a better performance by a unfull hash table, the
    // users need to set the param "load_factor"(load_factor<1).
    max_vocabulary_size_per_gpu_ = Base::get_max_vocabulary_size_per_gpu();
    max_vocabulary_size_ =
        max_vocabulary_size_per_gpu_ * Base::get_resource_manager().get_global_gpu_count();

    MESSAGE_("max_vocabulary_size_per_gpu_=" + std::to_string(max_vocabulary_size_per_gpu_));

    for (size_t id = 0; id < Base::get_resource_manager().get_local_gpu_count(); id++) {
      int cur_device = Base::get_local_gpu(id).get_device_id();
      context.set_device(cur_device);

      // construct HashTable object: used to store hash table <key, value_index>
      hash_tables_.emplace_back(new NvHashTable(max_vocabulary_size_per_gpu_));

      // new GeneralBuffer objects
      const std::shared_ptr<GeneralBuffer2<CudaAllocator>> &buf = Base::get_buffer(id);

      // new hash table value vectors
      {
        Tensor2<float> tensor;
        buf->reserve({max_vocabulary_size_per_gpu_, Base::get_embedding_vec_size()}, &tensor);
        hash_table_value_tensors_.push_back(tensor);
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
        buf->reserve({Base::get_universal_batch_size() * Base::get_slot_num(),
                      Base::get_embedding_vec_size()},
                     &tensor);
        embedding_feature_tensors_.push_back(tensor);
      }

      // new wgrad used by backward
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve(
            {Base::get_batch_size(true) * Base::get_slot_num(), Base::get_embedding_vec_size()},
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

      // new temp tensors used by update_params
      {
        Tensor2<TypeHashKey> tensor;
        buf->reserve({1, Base::get_universal_batch_size() * Base::get_slot_num() + 1}, &tensor);
        row_offset_allreduce_tensors_.push_back(tensor);
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
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({Base::get_universal_batch_size() * Base::get_slot_num(),
                      Base::get_embedding_vec_size()},
                     &tensor);
        utest_forward_temp_tensors_.push_back(tensor);
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

    }  // end of for(int id = 0; id < local_gpu_count_; id++)

    functors_.sync_all_gpus(Base::get_resource_manager());

  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void DistributedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::load_parameters(
    std::ifstream &stream) {
  // check if file is opened successfully
  if (!stream.is_open()) {
    CK_THROW_(Error_t::WrongInput, "Error: file not open for reading");
  }

  stream.seekg(0, stream.end);
  size_t file_size = stream.tellg();
  stream.seekg(0, stream.beg);

  size_t row_size = sizeof(TypeHashKey) + sizeof(float) * Base::get_embedding_vec_size();
  size_t row_num = file_size / row_size;

  if (file_size % row_size != 0) {
    CK_THROW_(Error_t::WrongInput, "Error: file size is not correct");
  }

  std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> blobs_buff =
      GeneralBuffer2<CudaHostAllocator>::create();

  Tensor2<TypeHashKey> keys;
  blobs_buff->reserve({row_num}, &keys);

  Tensor2<float> embeddings;
  blobs_buff->reserve({row_num, Base::get_embedding_vec_size()}, &embeddings);

  blobs_buff->allocate();

  TypeHashKey *key_ptr = keys.get_ptr();
  float *embedding_ptr = embeddings.get_ptr();

  for (size_t i = 0; i < row_num; i++) {
    stream.read(reinterpret_cast<char *>(key_ptr + i), sizeof(TypeHashKey));
    stream.read(reinterpret_cast<char *>(embedding_ptr + i * Base::get_embedding_vec_size()),
                sizeof(float) * Base::get_embedding_vec_size());
  }

  load_parameters(keys, embeddings, row_num, max_vocabulary_size_, Base::get_embedding_vec_size(),
                  max_vocabulary_size_per_gpu_, hash_table_value_tensors_, hash_tables_);

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void DistributedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::load_parameters(
    const TensorBag2 &keys_bag, const Tensor2<float> &embeddings, size_t num) {
  Tensor2<TypeHashKey> keys = Tensor2<TypeHashKey>::stretch_from(keys_bag);
  load_parameters(keys, embeddings, num, max_vocabulary_size_, Base::get_embedding_vec_size(),
                  max_vocabulary_size_per_gpu_, hash_table_value_tensors_, hash_tables_);
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void DistributedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::load_parameters(
    const Tensor2<TypeHashKey> &keys, const Tensor2<float> &embeddings, size_t num,
    size_t vocabulary_size, size_t embedding_vec_size, size_t max_vocabulary_size_per_gpu,
    Tensors2<float> &embedding_tensors,
    std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables) {
  if (keys.get_dimensions()[0] < num || embeddings.get_dimensions()[0] < num) {
    CK_THROW_(Error_t::WrongInput, "The rows of keys and embeddings are not consistent.");
  }

  if (num > vocabulary_size) {
    CK_THROW_(Error_t::WrongInput,
              "Error: hash table file size is larger than hash table vocabulary_size");
  }

  const TypeHashKey *key_ptr = keys.get_ptr();
  const float *embedding_ptr = embeddings.get_ptr();

  int my_rank = Base::get_resource_manager().get_pid();
  int n_ranks = Base::get_resource_manager().get_num_process();

  // define size
  size_t local_gpu_count = Base::get_resource_manager().get_local_gpu_count();
  const size_t chunk_size = 1000;
  size_t hash_table_key_tile_size = 1;
  size_t hash_table_key_tile_size_in_B = hash_table_key_tile_size * sizeof(TypeHashKey);
  size_t hash_table_key_chunk_size = hash_table_key_tile_size * chunk_size;
  size_t hash_table_key_chunk_size_in_B = hash_table_key_chunk_size * sizeof(TypeHashKey);
  size_t hash_table_value_index_chunk_size_in_B = hash_table_key_chunk_size * sizeof(size_t);
  size_t hash_table_value_tile_size = embedding_vec_size;
  size_t hash_table_value_tile_size_in_B = hash_table_value_tile_size * sizeof(float);
  size_t hash_table_value_chunk_size = hash_table_value_tile_size * chunk_size;
  size_t hash_table_value_chunk_size_in_B = hash_table_value_chunk_size * sizeof(float);

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
  std::unique_ptr<float *[]> h_hash_table_value_chunk_per_gpu(new float *[local_gpu_count]);
  for (size_t id = 0; id < local_gpu_count; id++) {
    CK_CUDA_THROW_(
        cudaMallocHost(&h_hash_table_value_chunk_per_gpu[id], hash_table_value_chunk_size_in_B));
  }

  // do upload
  size_t loop_num = num / chunk_size;
  MESSAGE_("Start to upload embedding table file to GPUs, total loop_num: " +
           std::to_string(loop_num));
  for (size_t i = 0; i < loop_num; i++) {
    TypeHashKey *key_dst_buf;
    float *value_dst_buf;
    for (size_t k = 0; k < chunk_size; k++) {  // process a tile in each loop
      TypeHashKey key = key_ptr[i * chunk_size + k];
      size_t gid = key % Base::get_resource_manager().get_global_gpu_count();  // global GPU ID
      size_t id = Base::get_resource_manager().get_gpu_local_id_from_global_id(
          gid);  // local GPU ID (not gpudevice id)
      int dst_rank = Base::get_resource_manager().get_pid_from_gpu_global_id(gid);  // node id

      if (my_rank == dst_rank) {
        // memcpy hash_table_key to corresponding GPU
        key_dst_buf = h_hash_table_key_chunk_per_gpu[id] +
                      tile_counter_in_chunk_per_gpu[id] * hash_table_key_tile_size;

        *key_dst_buf = key;

        // memcpy hash_table_value to corresponding GPU
        value_dst_buf = h_hash_table_value_chunk_per_gpu[id] +
                        tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;

        memcpy(value_dst_buf, embedding_ptr + (i * chunk_size + k) * embedding_vec_size,
               hash_table_value_tile_size_in_B);

        tile_counter_in_chunk_per_gpu[id] += 1;
      } else {
        continue;
      }
    }  // end of for(int k = 0; k < (chunk_loop * local_gpu_count); k++)

    // do HashTable insert <key,value_index>
    for (size_t id = 0; id < local_gpu_count; id++) {
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

    // memcpy hash_table_value from CPU to GPU
    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(Base::get_local_gpu(id).get_device_id());
      size_t value_chunk_size = tile_counter_in_chunk_per_gpu[id] * embedding_vec_size;
      size_t value_chunk_offset = tile_counter_per_gpu[id] * embedding_vec_size;
      float *src_buf = h_hash_table_value_chunk_per_gpu[id];
      float *dst_buf = embedding_tensors[id].get_ptr() + value_chunk_offset;
      CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf, src_buf, value_chunk_size * sizeof(float),
                                     cudaMemcpyHostToDevice, Base::get_local_gpu(id).get_stream()));
    }

    functors_.sync_all_gpus(Base::get_resource_manager());

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
  size_t remain_loop_num = num - loop_num * chunk_size;
  TypeHashKey *key_dst_buf;
  size_t *value_index_buf;
  float *value_dst_buf;
  for (size_t i = 0; i < remain_loop_num; i++) {
    TypeHashKey key = key_ptr[loop_num * chunk_size + i];
    size_t gid = key % Base::get_resource_manager().get_global_gpu_count();  // global GPU ID
    size_t id = Base::get_resource_manager().get_gpu_local_id_from_global_id(
        gid);  // local GPU ID (not gpudevice id)
    int dst_rank = Base::get_resource_manager().get_pid_from_gpu_global_id(gid);

    if (my_rank == dst_rank) {
      context.set_device(Base::get_local_gpu(id).get_device_id());

      // memcpy hash_table_key from CPU to GPU
      key_dst_buf = d_hash_table_key_chunk_per_gpu[id];
      CK_CUDA_THROW_(cudaMemcpyAsync(key_dst_buf, &key, hash_table_key_tile_size_in_B,
                                     cudaMemcpyHostToDevice, Base::get_local_gpu(id).get_stream()));

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

      // memcpy hash_table_value from CPU to GPU
      size_t value_offset = tile_counter_per_gpu[id] * embedding_vec_size;
      value_dst_buf = embedding_tensors[id].get_ptr() + value_offset;
      CK_CUDA_THROW_(cudaMemcpyAsync(
          value_dst_buf, embedding_ptr + (loop_num * chunk_size + i) * embedding_vec_size,
          hash_table_value_tile_size_in_B, cudaMemcpyHostToDevice,
          Base::get_local_gpu(id).get_stream()));

      // set counter
      tile_counter_per_gpu[id] += hash_table_key_tile_size;
    } else {
      continue;
    }

    // sync wait
    functors_.sync_all_gpus(Base::get_resource_manager());

  }  // end of if(remain_loop_num)

  // MESSAGE_("Done");

  // release resources
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(Base::get_local_gpu(id).get_device_id());
    CK_CUDA_THROW_(cudaFree(d_hash_table_value_index_chunk_per_gpu[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_key_chunk_per_gpu[id]));
  }
  for (size_t id = 0; id < local_gpu_count; id++) {
    CK_CUDA_THROW_(cudaFreeHost(h_hash_table_key_chunk_per_gpu[id]));
    CK_CUDA_THROW_(cudaFreeHost(h_hash_table_value_chunk_per_gpu[id]));
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void DistributedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::dump_parameters(
    std::ofstream &stream) const {
  // check if the file is opened successfully
  if (!stream.is_open()) {
    CK_THROW_(Error_t::WrongInput, "Error: file not open for writing");
    return;
  }

  dump_parameters(stream, max_vocabulary_size_, Base::get_embedding_vec_size(),
                  hash_table_value_tensors_, hash_tables_);

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void DistributedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::dump_parameters(
    TensorBag2 keys_bag, Tensor2<float> &embeddings, size_t *num) const {
  Tensor2<TypeHashKey> keys = Tensor2<TypeHashKey>::stretch_from(keys_bag);
  dump_parameters(keys, embeddings, num, max_vocabulary_size_, Base::get_embedding_vec_size(),
                  hash_table_value_tensors_, hash_tables_);
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void DistributedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::dump_parameters(
    std::ofstream &weight_stream, size_t vocabulary_size, size_t embedding_vec_size,
    const Tensors2<float> &hash_table_value_tensors,
    const std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables) const {
  CudaDeviceContext context;
  size_t local_gpu_count = Base::get_resource_manager().get_local_gpu_count();

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
    context.set_device(Base::get_local_gpu(id).get_device_id());
    auto count_tmp = hash_tables[id]->get_size(Base::get_local_gpu(id).get_stream());
    if (count_tmp != hash_tables[id]->get_value_head(Base::get_local_gpu(id).get_stream())) {
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
  std::unique_ptr<size_t *[]> d_hash_table_value_index(new size_t *[local_gpu_count]);
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
    cudaMallocHost(&h_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float));
    cudaMalloc(&d_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float));
    cudaMalloc(&d_dump_counter[id], count[id] * sizeof(size_t));
  }

  // dump hash table from GPUs
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) {
      continue;
    }

    MESSAGE_("Rank" + std::to_string(my_rank) + ": Dump hash table from GPU" + std::to_string(id));

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
  }

  // sync wait
  functors_.sync_all_gpus(Base::get_resource_manager());

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
      memcpy(file_buf.get() + offset, h_hash_table_value[id] + k * embedding_vec_size, value_size);
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

    context.set_device(Base::get_local_gpu(id).get_device_id());

    CK_CUDA_THROW_(cudaFreeHost(h_hash_table_key[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_key[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_value_index[id]));
    CK_CUDA_THROW_(cudaFreeHost(h_hash_table_value[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_value[id]));
    CK_CUDA_THROW_(cudaFree(d_dump_counter[id]));
  }

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void DistributedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::dump_parameters(
    Tensor2<TypeHashKey> &keys, Tensor2<float> &embeddings, size_t *num, size_t vocabulary_size,
    size_t embedding_vec_size, const Tensors2<float> &embedding_tensors,
    const std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables) const {
  TypeHashKey *key_ptr = keys.get_ptr();
  float *embedding_ptr = embeddings.get_ptr();

  size_t local_gpu_count = Base::get_resource_manager().get_local_gpu_count();

  // memory allocation
  std::unique_ptr<size_t[]> count(new size_t[local_gpu_count]);
  size_t max_count = 0;
  size_t total_count = 0;

  CudaDeviceContext context;
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(Base::get_local_gpu(id).get_device_id());
    auto count_tmp_1 = hash_tables[id]->get_size(Base::get_local_gpu(id).get_stream());
    auto count_tmp_2 = hash_tables[id]->get_value_head(Base::get_local_gpu(id).get_stream());
    if (count_tmp_1 != count_tmp_2) {
      CK_THROW_(Error_t::WrongInput,
                "Error: hash_table get_value_head() size not equal to get_size()");
    }
    count[id] = count_tmp_1;
    max_count = max(max_count, count[id]);
    total_count += count[id];
  }

  if (total_count > (size_t)vocabulary_size) {
    CK_THROW_(Error_t::WrongInput,
              "Error: required download size is larger than hash table vocabulary_size");
  }

  std::unique_ptr<TypeHashKey *[]> h_hash_table_key(new TypeHashKey *[local_gpu_count]);
  std::unique_ptr<TypeHashKey *[]> d_hash_table_key(new TypeHashKey *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_hash_table_value_index(new size_t *[local_gpu_count]);
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
    cudaMallocHost(&h_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float));
    cudaMalloc(&d_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float));
    cudaMalloc(&d_dump_counter[id], count[id] * sizeof(size_t));
  }

  // dump hash table from GPUs
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) {
      continue;
    }

    MESSAGE_("Dump hash table from GPU" + std::to_string(id));

    context.set_device(Base::get_local_gpu(id).get_device_id());

    hash_tables[id]->dump(d_hash_table_key[id], d_hash_table_value_index[id], d_dump_counter[id],
                          Base::get_local_gpu(id).get_stream());

    CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_key[id], d_hash_table_key[id],
                                   count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost,
                                   Base::get_local_gpu(id).get_stream()));

    functors_.get_hash_value(count[id], embedding_vec_size, d_hash_table_value_index[id],
                             embedding_tensors[id].get_ptr(), d_hash_table_value[id],
                             Base::get_local_gpu(id).get_stream());

    CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_value[id], d_hash_table_value[id],
                                   count[id] * embedding_vec_size * sizeof(float),
                                   cudaMemcpyDeviceToHost, Base::get_local_gpu(id).get_stream()));
  }

  // sync wait
  functors_.sync_all_gpus(Base::get_resource_manager());

  // TODO: could be optimized ???

  const size_t key_size = sizeof(TypeHashKey);
  const size_t value_size = sizeof(float) * embedding_vec_size;

  size_t offset = 0;
  for (size_t id = 0; id < local_gpu_count; id++) {
    size_t size_in_B = count[id] * (sizeof(TypeHashKey) + sizeof(float) * embedding_vec_size);
    for (unsigned int k = 0; k < count[id]; k++) {
      memcpy(key_ptr + offset, h_hash_table_key[id] + k, key_size);
      memcpy(embedding_ptr + offset * embedding_vec_size,
             h_hash_table_value[id] + k * embedding_vec_size, value_size);
      offset += 1;
    }
    // std::cout << std::endl;
    MESSAGE_("Write hash table <key,value> pairs to file");
  }

  *num = offset;

  // MESSAGE_("Done");

  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) {
      continue;
    }

    context.set_device(Base::get_local_gpu(id).get_device_id());

    CK_CUDA_THROW_(cudaFreeHost(h_hash_table_key[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_key[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_value_index[id]));
    CK_CUDA_THROW_(cudaFreeHost(h_hash_table_value[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_value[id]));
    CK_CUDA_THROW_(cudaFree(d_dump_counter[id]));
  }

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void DistributedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::init_embedding(
    size_t max_vocabulary_size_per_gpu, size_t embedding_vec_size,
    Tensors2<float> &hash_table_value_tensors) {
  CudaDeviceContext context;
  size_t local_gpu_count = Base::get_resource_manager().get_local_gpu_count();
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(Base::get_local_gpu(id).get_device_id());

    MESSAGE_("gpu" + std::to_string(id) + " start to init embedding");

    HugeCTR::UniformGenerator::fill(hash_table_value_tensors[id], -0.05f, 0.05f,
                                    Base::get_local_gpu(id));
  }

  for (size_t id = 0; id < local_gpu_count; id++) {
    CK_CUDA_THROW_(cudaStreamSynchronize(Base::get_local_gpu(id).get_stream()));
    MESSAGE_("gpu" + std::to_string(id) + " init embedding done");
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void DistributedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::reset() {
  CudaDeviceContext context;
  for (size_t i = 0; i < Base::get_resource_manager().get_local_gpu_count(); i++) {
    context.set_device(Base::get_local_gpu(i).get_device_id());
    hash_tables_[i]->clear(Base::get_local_gpu(i).get_stream());
    HugeCTR::UniformGenerator::fill(hash_table_value_tensors_[i], -0.05f, 0.05f,
                                    Base::get_local_gpu(i));
  }

  for (size_t id = 0; id < Base::get_resource_manager().get_local_gpu_count(); id++) {
    CK_CUDA_THROW_(cudaStreamSynchronize(Base::get_local_gpu(id).get_stream()));
  }
}

template class DistributedSlotSparseEmbeddingHash<unsigned int, float>;
template class DistributedSlotSparseEmbeddingHash<long long, float>;
template class DistributedSlotSparseEmbeddingHash<unsigned int, __half>;
template class DistributedSlotSparseEmbeddingHash<long long, __half>;

}  // namespace HugeCTR
