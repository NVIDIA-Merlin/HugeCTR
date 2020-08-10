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
#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/pinned_buffer.hpp"
#include "cub/cub/device/device_radix_sort.cuh"
#include "cub/cub/device/device_scan.cuh"

namespace HugeCTR {

template <typename TypeHashKey, typename TypeEmbeddingComp>
DistributedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::
    DistributedSlotSparseEmbeddingHash(
        const TensorPtrs<TypeHashKey> &train_row_offsets_tensors,
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
      value_index_bufs_.emplace_back(new GeneralBuffer<size_t>());

      // new hash table value vectors
      hash_table_value_tensors_.emplace_back(
          new Tensor<float>({max_vocabulary_size_per_gpu_, Base::get_embedding_vec_size()},
                            float_bufs_.back(), TensorFormat_t::HW));

      // new hash table value_index that get() from HashTable
      hash_value_index_tensors_.emplace_back(
          new Tensor<size_t>({1, Base::get_universal_batch_size() * Base::get_max_feature_num()},
                             value_index_bufs_.back(), TensorFormat_t::HW));

      // new embedding features reduced by hash table values(results of forward)
      embedding_feature_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
          {Base::get_universal_batch_size() * Base::get_slot_num(), Base::get_embedding_vec_size()},
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
      row_offset_allreduce_tensors_.emplace_back(
          new Tensor<TypeHashKey>({1, Base::get_universal_batch_size() * Base::get_slot_num() + 1},
                                  key_bufs_.back(), TensorFormat_t::HW));
      sample_id_tensors_.emplace_back(new Tensor<TypeHashKey>(
          {1, Base::get_train_only_batch_size() * Base::get_max_feature_num()}, key_bufs_.back(),
          TensorFormat_t::HW));
      sample_id_sort_tensors_.emplace_back(new Tensor<TypeHashKey>(
          {1, Base::get_train_only_batch_size() * Base::get_max_feature_num()}, key_bufs_.back(),
          TensorFormat_t::HW));
      hash_value_index_sort_tensors_.emplace_back(
          new Tensor<size_t>({1, Base::get_train_only_batch_size() * Base::get_max_feature_num()},
                             value_index_bufs_.back(), TensorFormat_t::HW));
      hash_value_index_count_offset_tensors_.emplace_back(new Tensor<uint32_t>(
          {1, Base::get_train_only_batch_size() * Base::get_max_feature_num() + 1},
          uint32_bufs_.back(), TensorFormat_t::HW));

      new_hash_value_flag_tensors_.emplace_back(
          new Tensor<uint32_t>({1, Base::get_train_only_batch_size() * Base::get_max_feature_num()},
                               uint32_bufs_.back(), TensorFormat_t::HW));

      hash_value_flag_sumed_tensors_.emplace_back(
          new Tensor<uint32_t>({1, Base::get_train_only_batch_size() * Base::get_max_feature_num()},
                               uint32_bufs_.back(), TensorFormat_t::HW));

      hash_value_index_count_counter_tensors_.emplace_back(
          new Tensor<uint32_t>({1, 1}, uint32_bufs_.back(), TensorFormat_t::HW));
      deltaw_hash_value_index_tensors_.emplace_back(
          new Tensor<size_t>({1, Base::get_train_only_batch_size() * Base::get_max_feature_num()},
                             value_index_bufs_.back(), TensorFormat_t::HW));
      deltaw_tensors_.emplace_back(
          new Tensor<float>({Base::get_train_only_batch_size() * Base::get_max_feature_num(),
                             Base::get_embedding_vec_size()},
                            float_bufs_.back(), TensorFormat_t::HW));
      {
        // cal the temp storage bytes for CUB radix sort
        size_t temp = 0;
        cub::DeviceRadixSort::SortPairs(
            (void *)NULL, (size_t &)temp, (size_t *)NULL, (size_t *)NULL,
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

      utest_forward_temp_tensors_.emplace_back(new Tensor<TypeEmbeddingComp>(
          {Base::get_universal_batch_size() * Base::get_slot_num(), Base::get_embedding_vec_size()},
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
          target_opt_param.hyperparams.momentum.momentum_ptr = opt_momentum_tensors_[id]->get_ptr();
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

template <typename TypeHashKey, typename TypeEmbeddingComp>
void DistributedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::upload_params_to_device(
    std::ifstream &weight_stream, size_t vocabulary_size, size_t embedding_vec_size,
    size_t max_vocabulary_size_per_gpu, const TensorPtrs<float> &hash_table_value_tensors,
    std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables,
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
  size_t hash_table_value_index_chunk_size_in_B = hash_table_key_chunk_size * sizeof(size_t);
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
  std::unique_ptr<size_t *[]> d_hash_table_value_index_chunk_per_gpu(new size_t *[local_gpu_count]);

  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(device_resources[id].get_device_id());
    CK_CUDA_THROW_(cudaMalloc(&d_hash_table_value_index_chunk_per_gpu[id],
                              hash_table_value_index_chunk_size_in_B));
    // initalize to zeros
    CK_CUDA_THROW_(cudaMemsetAsync(d_hash_table_value_index_chunk_per_gpu[id], 0,
                                   hash_table_value_index_chunk_size_in_B,
                                   device_resources[id].get_stream()));
  }

  // sync wait
  functors_.sync_all_gpus(device_resources);

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
    CK_CUDA_THROW_(cudaMalloc(&d_hash_table_key_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
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
    // one pair in hash table file includes: <key, value>
    weight_stream.read(hash_table_chunk, hash_table_chunk_size_in_B);

    // memcpy from CPU to CPU
    char *src_buf = hash_table_chunk;
    TypeHashKey *key_dst_buf;
    float *value_dst_buf;
    for (size_t k = 0; k < chunk_loop; k++) {  // process a tile in each loop
      TypeHashKey key = *((TypeHashKey *)src_buf);
      size_t gid = key % device_resources.get_total_gpu_count();  // global GPU ID
      size_t id = device_resources.get_local_id(gid);             // local GPU ID (not gpudevice id)
      size_t dst_rank = device_resources.get_pid(gid);            // node id

      if (static_cast<size_t>(my_rank) == dst_rank) {
        // memcpy hash_table_key to corresponding GPU
        key_dst_buf = h_hash_table_key_chunk_per_gpu[id] +
                      tile_counter_in_chunk_per_gpu[id] * hash_table_key_tile_size;
        
        memcpy(key_dst_buf, src_buf, hash_table_key_tile_size_in_B);

        src_buf += hash_table_key_tile_size_in_B;

        // memcpy hash_table_value to corresponding GPU
        value_dst_buf = h_hash_table_value_chunk_per_gpu[id] +
                        tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
        
        memcpy(value_dst_buf, src_buf, hash_table_value_tile_size_in_B);

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
      size_t *value_index_buf = d_hash_table_value_index_chunk_per_gpu[id];

      if (tile_count > 0) {
        // set hash_table_value_index on GPU
        functors_.memset_liner(value_index_buf, value_index_offset, 1ul, tile_count,
                               device_resources[id].get_stream());
      }

      // do hash table insert <key, value_index> on GPU
      hash_tables[id]->insert(d_hash_table_key_chunk_per_gpu[id], value_index_buf, tile_count,
                              device_resources[id].get_stream());
      size_t value_head =
          hash_tables[id]->get_and_add_value_head(tile_count, device_resources[id].get_stream());
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

    functors_.sync_all_gpus(device_resources);

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
    size_t *value_index_buf;
    float *value_dst_buf;
    for (size_t i = 0; i < remain_loop_num; i++) {
      TypeHashKey key = *((TypeHashKey *)src_buf);
      size_t gid = key % device_resources.get_total_gpu_count();  // global GPU ID
      size_t id = device_resources.get_local_id(gid);             // local GPU ID (not gpudevice id)
      size_t dst_rank = device_resources.get_pid(gid);

      if (static_cast<size_t>(my_rank) == dst_rank) {
        context.set_device(device_resources[id].get_device_id());

        // memcpy hash_table_key from CPU to GPU
        key_dst_buf = d_hash_table_key_chunk_per_gpu[id];
        CK_CUDA_THROW_(cudaMemcpyAsync(key_dst_buf, src_buf, hash_table_key_tile_size_in_B,
                                       cudaMemcpyHostToDevice, device_resources[id].get_stream()));

        src_buf += hash_table_key_tile_size_in_B;

        // set value_index
        size_t value_index_offset = tile_counter_per_gpu[id];
        value_index_buf = d_hash_table_value_index_chunk_per_gpu[id];
        functors_.memset_liner(value_index_buf, value_index_offset, 1ul, 1ul,
                               device_resources[id].get_stream());

        // do hash table insert <key, value_index> on GPU
        hash_tables[id]->insert(d_hash_table_key_chunk_per_gpu[id], value_index_buf,
                                hash_table_key_tile_size, device_resources[id].get_stream());
        size_t value_head = hash_tables[id]->get_and_add_value_head(
            hash_table_key_tile_size, device_resources[id].get_stream());

        // memcpy hash_table_value from CPU to GPU
        size_t value_offset = tile_counter_per_gpu[id] * embedding_vec_size;
        value_dst_buf = hash_table_value_tensors[id]->get_ptr() + value_offset;
        CK_CUDA_THROW_(cudaMemcpyAsync(value_dst_buf, src_buf, hash_table_value_tile_size_in_B,
                                       cudaMemcpyHostToDevice, device_resources[id].get_stream()));
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
    functors_.sync_all_gpus(device_resources);

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

template <typename TypeHashKey, typename TypeEmbeddingComp>
void DistributedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::download_params_to_host(
    std::ofstream &weight_stream, size_t vocabulary_size, size_t embedding_vec_size,
    const TensorPtrs<float> &hash_table_value_tensors,
    const std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables,
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
    if (count_tmp != hash_tables[id]->get_value_head(device_resources[id].get_stream())) {
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

    context.set_device(device_resources[id].get_device_id());

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

    context.set_device(device_resources[id].get_device_id());

    hash_tables[id]->dump(d_hash_table_key[id], d_hash_table_value_index[id], d_dump_counter[id],
                          device_resources[id].get_stream());

    CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_key[id], d_hash_table_key[id],
                                   count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost,
                                   device_resources[id].get_stream()));

    functors_.get_hash_value(count[id], embedding_vec_size, d_hash_table_value_index[id],
                             hash_table_value_tensors[id]->get_ptr(), d_hash_table_value[id],
                             device_resources[id].get_stream());

    CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_value[id], d_hash_table_value[id],
                                   count[id] * embedding_vec_size * sizeof(float),
                                   cudaMemcpyDeviceToHost, device_resources[id].get_stream()));
  }

  // sync wait
  functors_.sync_all_gpus(device_resources);

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

    context.set_device(device_resources[id].get_device_id());

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
    const TensorPtrs<float> &hash_table_value_tensors, const GPUResourceGroup &device_resources) {
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

template class DistributedSlotSparseEmbeddingHash<unsigned int, float>;
template class DistributedSlotSparseEmbeddingHash<long long, float>;
template class DistributedSlotSparseEmbeddingHash<unsigned int, __half>;
template class DistributedSlotSparseEmbeddingHash<long long, __half>;

}  // namespace HugeCTR