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

#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_one_hot.hpp"

namespace HugeCTR {

namespace {

template <typename value_type>
__global__ void upload_value_tensor_kernel(value_type *value_buf, size_t *index_buf,
                                           value_type *dst_tensor, int emb_vec_size, size_t len) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < len) {
    size_t src_offset = gid * emb_vec_size;
    size_t dst_offset = index_buf[gid] * emb_vec_size;
    for (int i = 0; i < emb_vec_size; i++) {
      dst_tensor[dst_offset + i] = value_buf[src_offset + i];
    }
  }
}

}  // namespace

template <typename TypeHashKey, typename TypeEmbeddingComp>
LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::
    LocalizedSlotSparseEmbeddingOneHot(
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
      slot_size_array_(embedding_params.slot_size_array) {
  try {
    max_vocabulary_size_ = 0;
    for (size_t slot_size : slot_size_array_) {
      max_vocabulary_size_ += slot_size;
    }

    max_vocabulary_size_per_gpu_ =
        cal_max_voc_size_per_gpu(slot_size_array_, Base::get_resource_manager());

    MESSAGE_("max_vocabulary_size_per_gpu_=" + std::to_string(max_vocabulary_size_per_gpu_));

    CudaDeviceContext context;
    for (size_t id = 0; id < Base::get_resource_manager().get_local_gpu_count(); id++) {
      int cur_device = Base::get_local_gpu(id).get_device_id();
      context.set_device(cur_device);

      size_t gid = Base::get_local_gpu(id).get_global_gpu_id();
      size_t slot_num_per_gpu =
          Base::get_slot_num() / Base::get_resource_manager().get_global_gpu_count() +
          ((gid < Base::get_slot_num() % Base::get_resource_manager().get_global_gpu_count()) ? 1
                                                                                              : 0);
      slot_num_per_gpu_.push_back(slot_num_per_gpu);

      // new GeneralBuffer objects
      const std::shared_ptr<GeneralBuffer2<CudaAllocator>> &buf = Base::get_buffer(id);

      // new hash table value vectors
      {
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
        case Optimizer_t::SGD:
          break;

        default:
          throw std::runtime_error(
              std::string("[HCDEBUG][ERROR] Runtime error: Invalid optimizer type\n"));
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
      {
        Tensor2<uint32_t> tensor;
        buf->reserve({1, slot_num_per_gpu}, &tensor);
        mapping_offsets_per_gpu_tensors_.push_back(tensor);
      }

// init GenenralBuffers to do real allocation
#ifndef NDEBUG
      std::cout << " max_feature_num_:" << Base::get_max_feature_num() << std::endl;
#endif
      buf->allocate();

      const OptParams<TypeEmbeddingComp> &source_opt_param = Base::get_opt_params();
      OptParams<TypeEmbeddingComp> &target_opt_param = Base::get_opt_params(id);

      switch (Base::get_optimizer()) {
        case Optimizer_t::SGD:
          target_opt_param.hyperparams.sgd.atomic_update =
              source_opt_param.hyperparams.sgd.atomic_update;

          break;

        default:
          throw std::runtime_error(
              std::string("[HCDEBUG][ERROR] Runtime error: Invalid optimizer type\n"));
      }

    }  // end of for(int id = 0; id < Base::get_local_gpu_count(); id++)

    // sync
    functors_.sync_all_gpus(Base::get_resource_manager());

    // get the mapping table between local value_index and input value_index
    for (size_t id = 0; id < Base::get_resource_manager().get_local_gpu_count(); id++) {
      uint32_t slot_sizes_prefix_sum = 0;
      uint32_t slot_sizes_prefix_sum_local = 0;
      int slot_num = 0;
      for (size_t i = 0; i < slot_size_array_.size(); i++) {
        size_t global_id = Base::get_local_gpu(id).get_global_gpu_id();
        size_t slot_size = slot_size_array_[i];
        if (i % Base::get_resource_manager().get_global_gpu_count() == global_id) {
          uint32_t mapping_offset = slot_sizes_prefix_sum - slot_sizes_prefix_sum_local;
          CK_CUDA_THROW_(cudaMemcpy(&((mapping_offsets_per_gpu_tensors_[id].get_ptr())[slot_num]),
                                    &mapping_offset, sizeof(uint32_t), cudaMemcpyHostToDevice));
          slot_sizes_prefix_sum_local += slot_size;
          slot_num++;
        }
        slot_sizes_prefix_sum += slot_size;
      }
    }

    // Check whether the P2P access can be enabled
    if (Base::get_resource_manager().get_local_gpu_count() > 1 &&
        !Base::get_resource_manager().all_p2p_enabled()) {
      throw std::runtime_error(
          std::string("[HCDEBUG][ERROR] Runtime error: Localized_slot_sparse_embedding_one_hot "
                      "cannot be used on machine without GPU peer2peer access support. \n"));
    }
#ifdef ENABLE_MPI
    throw std::runtime_error(
        std::string("[HCDEBUG][ERROR] Runtime error: Localized_slot_sparse_embedding_one_hot "
                    "cannot support multi-node currently. \n"));
#endif

    std::shared_ptr<GeneralBuffer2<CudaManagedAllocator>> unified_buf =
        GeneralBuffer2<CudaManagedAllocator>::create();
    unified_buf->reserve({Base::get_resource_manager().get_local_gpu_count()},
                         &train_embedding_features_);
    unified_buf->reserve({Base::get_resource_manager().get_local_gpu_count()},
                         &evaluate_embedding_features_);
    unified_buf->allocate();

    for (size_t id = 0; id < Base::get_resource_manager().get_local_gpu_count(); id++) {
      train_embedding_features_.get_ptr()[id] = Base::get_output_tensors(true)[id].get_ptr();
      evaluate_embedding_features_.get_ptr()[id] = Base::get_output_tensors(false)[id].get_ptr();
    }

  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::load_parameters(
    std::ifstream &weight_stream, size_t embedding_vec_size,
    Tensors2<float> &hash_table_value_tensors, const std::vector<size_t> &slot_sizes,
    const Tensors2<uint32_t> &mapping_offsets_per_gpu_tensors) {
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
  size_t local_gpu_count = Base::get_resource_manager().get_local_gpu_count();
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
  size_t hash_table_slot_id_tile_size_in_B = hash_table_slot_id_tile_size * sizeof(size_t);
  size_t hash_table_tile_size_in_B = hash_table_key_tile_size_in_B +
                                     hash_table_slot_id_tile_size_in_B +
                                     hash_table_value_tile_size_in_B;
  size_t hash_table_chunk_size_in_B = hash_table_tile_size_in_B * chunk_loop;
  size_t total_gpu_count = Base::get_resource_manager().get_global_gpu_count();

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
    context.set_device(Base::get_local_gpu(id).get_device_id());
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
    context.set_device(Base::get_local_gpu(id).get_device_id());
    CK_CUDA_THROW_(cudaMalloc(&d_hash_table_index_chunk_per_gpu[id], chunk_loop * sizeof(size_t)));
  }

  std::unique_ptr<size_t[]> tile_counter_in_chunk_per_gpu(new size_t[local_gpu_count]);
  memset(tile_counter_in_chunk_per_gpu.get(), 0, sizeof(size_t) * local_gpu_count);

  // The vector that store the relationship between slot_id and slot order on the specific GPU
  std::vector<size_t> local_slot_id(slot_sizes.size());
  std::vector<size_t> local_slot_num(local_gpu_count, 0);
  for (size_t i = 0; i < slot_sizes.size(); i++) {
    size_t gid = i % total_gpu_count;  // global GPU ID
    size_t id = Base::get_resource_manager().get_gpu_local_id_from_global_id(
        gid);  // local GPU ID (not gpudevice id)
    int dst_rank = Base::get_resource_manager().get_pid_from_gpu_global_id(gid);  // node id
    if (my_rank == dst_rank) {
      local_slot_id[i] = local_slot_num[id];
      local_slot_num[id]++;
    }
  }

  // Host buffer to keep mapping_offset
  std::vector<uint32_t *> h_mapping_offsets_per_gpu_tensors(local_gpu_count);
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(Base::get_local_gpu(id).get_device_id());
    CK_CUDA_THROW_(cudaMallocHost(&h_mapping_offsets_per_gpu_tensors[id],
                                  local_slot_num[id] * sizeof(uint32_t)));
    // Copy the mapping offset from GPU to Host
    cudaMemcpyAsync(h_mapping_offsets_per_gpu_tensors[id],
                    mapping_offsets_per_gpu_tensors[id].get_ptr(),
                    local_slot_num[id] * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                    Base::get_local_gpu(id).get_stream());
  }

  // sync wait
  functors_.sync_all_gpus(Base::get_resource_manager());

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
    float *value_dst_buf;
    size_t *tensor_index_dst_buf;
    for (size_t k = 0; k < chunk_loop; k++) {  // process a tile in each loop
      size_t slot_id = *((size_t *)(src_buf + hash_table_key_tile_size_in_B));
      size_t gid = slot_id % total_gpu_count;  // global GPU ID
      size_t id = Base::get_resource_manager().get_gpu_local_id_from_global_id(
          gid);  // local GPU ID (not gpudevice id)
      int dst_rank = Base::get_resource_manager().get_pid_from_gpu_global_id(gid);  // node id

      if (my_rank == dst_rank) {
        TypeHashKey tile_key = *((TypeHashKey *)src_buf);
        size_t tensor_index =
            tile_key - (h_mapping_offsets_per_gpu_tensors[id][local_slot_id[slot_id]]);

        src_buf += hash_table_key_tile_size_in_B;
        src_buf += hash_table_slot_id_tile_size_in_B;
        // memcpy hash_table_value to corresponding GPU
        value_dst_buf = h_hash_table_value_chunk_per_gpu[id] +
                        tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
        memcpy(value_dst_buf, src_buf, hash_table_value_tile_size_in_B);
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

      context.set_device(Base::get_local_gpu(id).get_device_id());

      // Copy value buffer and tensor_index buffer to GPU
      size_t value_chunk_size = tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
      float *src_buf_value = h_hash_table_value_chunk_per_gpu[id];
      float *dst_buf_value = d_hash_table_value_chunk_per_gpu[id];
      CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf_value, src_buf_value, value_chunk_size * sizeof(float),
                                     cudaMemcpyHostToDevice, Base::get_local_gpu(id).get_stream()));
      size_t *src_buf_index = h_hash_table_index_chunk_per_gpu[id];
      size_t *dst_buf_index = d_hash_table_index_chunk_per_gpu[id];
      value_chunk_size = tile_counter_in_chunk_per_gpu[id];
      CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf_index, src_buf_index,
                                     value_chunk_size * sizeof(size_t), cudaMemcpyHostToDevice,
                                     Base::get_local_gpu(id).get_stream()));

      // Call kernel to insert the value into embedding value tensor
      const size_t grid_size = (tile_counter_in_chunk_per_gpu[id] - 1) / 256 + 1;
      upload_value_tensor_kernel<<<grid_size, 256, 0, Base::get_local_gpu(id).get_stream()>>>(
          d_hash_table_value_chunk_per_gpu[id], d_hash_table_index_chunk_per_gpu[id],
          hash_table_value_tensors[id].get_ptr(), hash_table_value_tile_size,
          tile_counter_in_chunk_per_gpu[id]);
    }

    functors_.sync_all_gpus(Base::get_resource_manager());

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

      size_t slot_id = *((size_t *)(src_buf + hash_table_key_tile_size_in_B));
      size_t gid = slot_id % total_gpu_count;  // global GPU ID
      size_t id = Base::get_resource_manager().get_gpu_local_id_from_global_id(
          gid);  // local GPU ID (not gpudevice id)
      int dst_rank = Base::get_resource_manager().get_pid_from_gpu_global_id(gid);  // node id

      if (my_rank == dst_rank) {
        TypeHashKey tile_key = *((TypeHashKey *)src_buf);
        size_t tensor_index =
            tile_key - (h_mapping_offsets_per_gpu_tensors[id][local_slot_id[slot_id]]);

        src_buf += hash_table_key_tile_size_in_B;
        src_buf += hash_table_slot_id_tile_size_in_B;
        // memcpy hash_table_value to corresponding GPU
        value_dst_buf = h_hash_table_value_chunk_per_gpu[id] +
                        tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
        memcpy(value_dst_buf, src_buf, hash_table_value_tile_size_in_B);
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

      context.set_device(Base::get_local_gpu(id).get_device_id());

      // Copy value buffer and tensor_index buffer to GPU
      size_t value_chunk_size = tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
      float *src_buf_value = h_hash_table_value_chunk_per_gpu[id];
      float *dst_buf_value = d_hash_table_value_chunk_per_gpu[id];
      CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf_value, src_buf_value, value_chunk_size * sizeof(float),
                                     cudaMemcpyHostToDevice, Base::get_local_gpu(id).get_stream()));
      size_t *src_buf_index = h_hash_table_index_chunk_per_gpu[id];
      size_t *dst_buf_index = d_hash_table_index_chunk_per_gpu[id];
      value_chunk_size = tile_counter_in_chunk_per_gpu[id];
      CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf_index, src_buf_index,
                                     value_chunk_size * sizeof(size_t), cudaMemcpyHostToDevice,
                                     Base::get_local_gpu(id).get_stream()));

      // Call kernel to insert the value into embedding value tensor
      const size_t grid_size = (tile_counter_in_chunk_per_gpu[id] - 1) / 256 + 1;
      upload_value_tensor_kernel<<<grid_size, 256, 0, Base::get_local_gpu(id).get_stream()>>>(
          d_hash_table_value_chunk_per_gpu[id], d_hash_table_index_chunk_per_gpu[id],
          hash_table_value_tensors[id].get_ptr(), hash_table_value_tile_size,
          tile_counter_in_chunk_per_gpu[id]);
    }

    // sync wait
    functors_.sync_all_gpus(Base::get_resource_manager());

  }  // end of if(remain_loop_num)

  MESSAGE_("Done");

  // release resources
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(Base::get_local_gpu(id).get_device_id());
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

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::dump_parameters(
    std::ofstream &weight_stream, size_t embedding_vec_size,
    const Tensors2<float> &hash_table_value_tensors, const std::vector<size_t> &slot_sizes) const {
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

  CudaDeviceContext context;
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(Base::get_local_gpu(id).get_device_id());
    count[id] = 0;
    for (size_t i = 0; i < slot_sizes.size(); i++) {
      size_t global_id = Base::get_local_gpu(id).get_global_gpu_id();
      if ((i % Base::get_resource_manager().get_global_gpu_count()) == global_id) {
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
  std::unique_ptr<size_t *[]> h_hash_table_slot_id(new size_t *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_hash_table_slot_id(new size_t *[local_gpu_count]);
  std::unique_ptr<float *[]> h_hash_table_value(new float *[local_gpu_count]);
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) {
      continue;
    }

    context.set_device(Base::get_local_gpu(id).get_device_id());

    cudaMallocHost(&h_hash_table_key[id], count[id] * sizeof(TypeHashKey));
    cudaMalloc(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey));
    cudaMallocHost(&h_hash_table_slot_id[id], count[id] * sizeof(size_t));
    cudaMalloc(&d_hash_table_slot_id[id], count[id] * sizeof(size_t));
    cudaMallocHost(&h_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float));
  }

  // Generate key and slot_id tensor, dump value tensor on GPU
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) {
      continue;
    }

    MESSAGE_("Rank" + std::to_string(my_rank) + ": Dump embedding table from GPU" +
             std::to_string(id));

    context.set_device(Base::get_local_gpu(id).get_device_id());

    // Loop for each slot
    size_t buffer_offset = 0;
    for (size_t i = 0; i < slot_sizes.size(); i++) {
      size_t global_id = Base::get_local_gpu(id).get_global_gpu_id();
      if ((i % Base::get_resource_manager().get_global_gpu_count()) == global_id) {
        // Generate key buffer
        size_t key_offset = 0;
        for (size_t j = 0; j < i; j++) {
          key_offset += slot_sizes[j];
        }
        functors_.memset_liner(d_hash_table_key[id] + buffer_offset, (TypeHashKey)key_offset,
                               (TypeHashKey)1, slot_sizes[i], Base::get_local_gpu(id).get_stream());

        // Generate slot_id
        functors_.memset_const(d_hash_table_slot_id[id] + buffer_offset, i, slot_sizes[i],
                               Base::get_local_gpu(id).get_stream());

        buffer_offset += slot_sizes[i];
      }
    }
    // Copy key buffer to host
    CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_key[id], d_hash_table_key[id],
                                   count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost,
                                   Base::get_local_gpu(id).get_stream()));
    // Copy value buffer to host
    CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_value[id], hash_table_value_tensors[id].get_ptr(),
                                   count[id] * embedding_vec_size * sizeof(float),
                                   cudaMemcpyDeviceToHost, Base::get_local_gpu(id).get_stream()));
    // Copy slot_id to host
    CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_slot_id[id], d_hash_table_slot_id[id],
                                   count[id] * sizeof(size_t), cudaMemcpyDeviceToHost,
                                   Base::get_local_gpu(id).get_stream()));
  }

  // sync wait
  functors_.sync_all_gpus(Base::get_resource_manager());

  const int master_node = 0;
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
                        << (float)k / count[id] * 100.0f << "%" << std::flush; */

      memcpy(file_buf.get() + offset, h_hash_table_key[id] + k, key_size);
      offset += key_size;
      memcpy(file_buf.get() + offset, h_hash_table_slot_id[id] + k, slot_id_size);
      offset += slot_id_size;
      memcpy(file_buf.get() + offset, h_hash_table_value[id] + k * embedding_vec_size, value_size);
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

    context.set_device(Base::get_local_gpu(id).get_device_id());

    CK_CUDA_THROW_(cudaFreeHost(h_hash_table_key[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_key[id]));
    CK_CUDA_THROW_(cudaFreeHost(h_hash_table_slot_id[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_slot_id[id]));
    CK_CUDA_THROW_(cudaFreeHost(h_hash_table_value[id]));
  }

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::init_embedding(
    const std::vector<size_t> slot_sizes, size_t embedding_vec_size,
    std::vector<Tensors2<float>> &hash_table_value_tensors,
    Tensors2<size_t> &hash_table_slot_id_tensors) {
  size_t local_gpu_count = Base::get_resource_manager().get_local_gpu_count();
  size_t total_gpu_count = Base::get_resource_manager().get_global_gpu_count();

#ifndef NDEBUG
  MESSAGE_("local_gpu_count=" + std::to_string(local_gpu_count) + ", total_gpu_count=" +
           std::to_string(total_gpu_count));
#endif

  for (size_t id = 0; id < local_gpu_count; id++) {
    size_t device_id = Base::get_local_gpu(id).get_device_id();
    size_t global_id = Base::get_local_gpu(id).get_global_gpu_id();

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
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::reset() {
  CudaDeviceContext context;
  for (size_t i = 0; i < Base::get_resource_manager().get_local_gpu_count(); i++) {
    functors_.init_embedding_per_gpu(Base::get_local_gpu(i).get_global_gpu_id(),
                                     Base::get_resource_manager().get_global_gpu_count(),
                                     slot_size_array_, Base::get_embedding_vec_size(),
                                     value_table_tensors_[i], hash_table_slot_id_tensors_[i],
                                     Base::get_local_gpu(i));
  }

  for (size_t i = 0; i < Base::get_resource_manager().get_local_gpu_count(); i++) {
    CK_CUDA_THROW_(cudaStreamSynchronize(Base::get_local_gpu(i).get_stream()));
  }
}

template class LocalizedSlotSparseEmbeddingOneHot<unsigned int, float>;
template class LocalizedSlotSparseEmbeddingOneHot<long long, float>;
template class LocalizedSlotSparseEmbeddingOneHot<unsigned int, __half>;
template class LocalizedSlotSparseEmbeddingOneHot<long long, __half>;

}  // namespace HugeCTR