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
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/utils.cuh"
#include "HugeCTR/include/utils.hpp"

#include <numeric>
#include <experimental/filesystem>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

namespace fs = std::experimental::filesystem;

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
    const SparseEmbeddingHashParams &embedding_params,
    const std::shared_ptr<ResourceManager> &resource_manager)
    : Base(train_row_offsets_tensors, train_value_tensors, train_nnz_array,
           evaluate_row_offsets_tensors, evaluate_value_tensors, evaluate_nnz_array,
           Embedding_t::LocalizedSlotSparseEmbeddingHash, embedding_params, resource_manager),
      slot_size_array_(embedding_params.slot_size_array)
{
  try {
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

    CudaDeviceContext context;
    for (size_t id = 0; id < Base::get_resource_manager().get_local_gpu_count(); id++) {
      context.set_device(Base::get_local_gpu(id).get_device_id());

      size_t gid = Base::get_local_gpu(id).get_global_id();
      size_t slot_num_per_gpu =
          Base::get_slot_num() / Base::get_resource_manager().get_global_gpu_count() +
          ((gid < Base::get_slot_num() % Base::get_resource_manager().get_global_gpu_count()) ? 1
                                                                                              : 0);
      slot_num_per_gpu_.push_back(slot_num_per_gpu);
      // new GeneralBuffer objects
      const std::shared_ptr<GeneralBuffer2<CudaAllocator>> &buf = Base::get_buffer(id);
      embedding_optimizers_.emplace_back(max_vocabulary_size_per_gpu_, Base::embedding_params_, buf);

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
    }

    hash_tables_.resize(Base::get_resource_manager().get_local_gpu_count());
#pragma omp parallel num_threads(Base::get_resource_manager().get_local_gpu_count())
    {
      size_t id = omp_get_thread_num();
      CudaDeviceContext context(Base::get_local_gpu(id).get_device_id());
      // construct HashTable object: used to store hash table <key, value_index>
      hash_tables_[id].reset(new NvHashTable(max_vocabulary_size_per_gpu_));
      Base::get_buffer(id)->allocate();
    }

    for (size_t id = 0; id < Base::get_resource_manager().get_local_gpu_count(); id++) {
      context.set_device(Base::get_local_gpu(id).get_device_id());
      embedding_optimizers_[id].initialize(Base::get_local_gpu(id));

    }  // end of for(int id = 0; id < Base::get_local_gpu_count(); id++)

    // sync
    functors_.sync_all_gpus(Base::get_resource_manager());

// warm up for nccl all2all
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

  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}


template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::load_parameters(
    std::string sparse_model) {
  if (!fs::exists(sparse_model)) {
    CK_THROW_(Error_t::WrongInput, std::string("Error: folder ") + sparse_model + " doesn't exist");
  }
  const std::string key_file(sparse_model + "/" + sparse_model + ".key");
  const std::string slot_file(sparse_model + "/" + sparse_model + ".slot");
  const std::string vec_file(sparse_model + "/" + sparse_model + ".vec");

  std::ifstream key_stream(key_file, std::ifstream::binary);
  std::ifstream slot_stream(slot_file, std::ifstream::binary);
  std::ifstream vec_stream(vec_file, std::ifstream::binary);
  // check if file is opened successfully
  if (!vec_stream.is_open() || !key_stream.is_open() || !slot_stream.is_open()) {
    CK_THROW_(Error_t::WrongInput, "Error: file not open for reading");
  }

  size_t key_file_size_in_byte = fs::file_size(key_file);
  size_t slot_file_size_in_byte = fs::file_size(slot_file);
  size_t vec_file_size_in_byte = fs::file_size(vec_file);

  size_t key_size = sizeof(TypeHashKey);
  size_t slot_size = sizeof(size_t);
  size_t vec_size = sizeof(float) * Base::get_embedding_vec_size();
  size_t key_num = key_file_size_in_byte / key_size;
  size_t slot_num = slot_file_size_in_byte / slot_size;
  size_t vec_num = vec_file_size_in_byte / vec_size;

  if (key_num != vec_num || key_file_size_in_byte % key_size != 0 || vec_file_size_in_byte % vec_size != 0 ||
      key_num != slot_num || slot_file_size_in_byte % slot_size != 0) {
    CK_THROW_(Error_t::WrongInput, "Error: file size is not correct");
  }

  auto blobs_buff = GeneralBuffer2<CudaHostAllocator>::create();

  Tensor2<TypeHashKey> keys;
  blobs_buff->reserve({key_num}, &keys);

  Tensor2<size_t> slot_id;
  blobs_buff->reserve({slot_num}, &slot_id);

  Tensor2<float> embeddings;
  blobs_buff->reserve({vec_num, Base::get_embedding_vec_size()}, &embeddings);

  blobs_buff->allocate();

  TypeHashKey *key_ptr = keys.get_ptr();
  size_t *slot_id_ptr = slot_id.get_ptr();
  float *embedding_ptr = embeddings.get_ptr();

  key_stream.read(reinterpret_cast<char *>(key_ptr), key_file_size_in_byte);
  slot_stream.read(reinterpret_cast<char *>(slot_id_ptr), slot_file_size_in_byte);
  vec_stream.read(reinterpret_cast<char *>(embedding_ptr), vec_file_size_in_byte);

  load_parameters(keys, slot_id, embeddings, key_num, max_vocabulary_size_,
                  Base::get_embedding_vec_size(), max_vocabulary_size_per_gpu_,
                  hash_table_value_tensors_, hash_table_slot_id_tensors_, hash_tables_);
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::load_parameters(
    BufferBag &buf_bag, size_t num) {
  const TensorBag2 &keys_bag = buf_bag.keys;
  const TensorBag2 &slot_id_bag = buf_bag.slot_id;
  const Tensor2<float> &embeddings = buf_bag.embedding;
  Tensor2<TypeHashKey> keys = Tensor2<TypeHashKey>::stretch_from(keys_bag);
  Tensor2<size_t> slot_id = Tensor2<size_t>::stretch_from(slot_id_bag);
  load_parameters(keys, slot_id, embeddings, num, max_vocabulary_size_,
                  Base::get_embedding_vec_size(), max_vocabulary_size_per_gpu_,
                  hash_table_value_tensors_, hash_table_slot_id_tensors_, hash_tables_);
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::load_parameters(
    const Tensor2<TypeHashKey> &keys, const Tensor2<size_t> &slot_id,
    const Tensor2<float> &embeddings, size_t num, size_t vocabulary_size, size_t embedding_vec_size,
    size_t max_vocabulary_size_per_gpu, Tensors2<float> &hash_table_value_tensors,
    Tensors2<size_t> &hash_table_slot_id_tensors,
    std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables) {
  if (keys.get_dimensions()[0] < num || slot_id.get_dimensions()[0] < num ||
      embeddings.get_dimensions()[0] < num) {
    CK_THROW_(Error_t::WrongInput, "The rows of keys and embeddings are not consistent.");
  }

  if (num > vocabulary_size) {
    CK_THROW_(Error_t::WrongInput,
              "Error: hash table file size is larger than hash table vocabulary_size");
  }

  const TypeHashKey *key_ptr = keys.get_ptr();
  const size_t *slot_id_ptr = slot_id.get_ptr();
  const float *embedding_ptr = embeddings.get_ptr();

  int my_rank = Base::get_resource_manager().get_process_id();
  int n_ranks = Base::get_resource_manager().get_num_process();

  // define size
  size_t local_gpu_count = Base::get_resource_manager().get_local_gpu_count();
  size_t chunk_size = 1000;
  size_t tile_size = 1;  // must be 1, because we need to cal (key&local_gpu_count) to decide
                         // gpu_id for each <key,value>
  size_t hash_table_key_tile_size = tile_size;
  size_t hash_table_key_tile_size_in_B = hash_table_key_tile_size * sizeof(TypeHashKey);
  size_t hash_table_key_chunk_size = hash_table_key_tile_size * chunk_size;
  size_t hash_table_key_chunk_size_in_B = hash_table_key_chunk_size * sizeof(TypeHashKey);
  size_t hash_table_value_index_chunk_size_in_B = hash_table_key_chunk_size * sizeof(size_t);
  size_t hash_table_value_tile_size = tile_size * embedding_vec_size;
  size_t hash_table_value_tile_size_in_B = hash_table_value_tile_size * sizeof(float);
  size_t hash_table_value_chunk_size = hash_table_value_tile_size * chunk_size;
  size_t hash_table_value_chunk_size_in_B = hash_table_value_chunk_size * sizeof(float);
  size_t hash_table_slot_id_tile_size = tile_size;
  size_t hash_table_slot_id_tile_size_in_B = hash_table_slot_id_tile_size * sizeof(size_t);
  size_t hash_table_slot_id_chunk_size = hash_table_slot_id_tile_size * chunk_size;
  size_t hash_table_slot_id_chunk_size_in_B = hash_table_slot_id_chunk_size * sizeof(size_t);
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
  std::unique_ptr<TypeHashKey *[]> h_hash_table_key_chunk_per_gpu(
      new TypeHashKey *[local_gpu_count]);
  for (size_t id = 0; id < local_gpu_count; id++) {
    CK_CUDA_THROW_(cudaMallocHost(&h_hash_table_key_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
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
  size_t loop_num = num / chunk_size;
  MESSAGE_("Start to upload embedding table file to GPUs, total loop_num: " +
           std::to_string(loop_num));
  for (size_t i = 0; i < loop_num; i++) {
    TypeHashKey *key_dst_buf;
    size_t *slot_id_dst_buf;
    float *value_dst_buf;
    for (size_t k = 0; k < chunk_size; k++) {  // process a tile in each loop
      TypeHashKey key = key_ptr[i * chunk_size + k];
      size_t slot_id = slot_id_ptr[i * chunk_size + k];
      size_t gid = slot_id % total_gpu_count;  // global GPU ID
      size_t id = Base::get_resource_manager().get_gpu_local_id_from_global_id(
          gid);  // local GPU ID (not gpudevice id)
      int dst_rank =
          Base::get_resource_manager().get_process_id_from_gpu_global_id(gid);  // node id

      if (Base::get_resource_manager().get_process_id() == dst_rank) {
        // memcpy hash_table_key to corresponding GPU
        key_dst_buf = h_hash_table_key_chunk_per_gpu[id] +
                      tile_counter_in_chunk_per_gpu[id] * hash_table_key_tile_size;

        *key_dst_buf = key;

        // memcpy hash_table_slot_id to corresponding GPU
        slot_id_dst_buf = h_hash_table_slot_id_chunk_per_gpu[id] +
                          tile_counter_in_chunk_per_gpu[id] * hash_table_slot_id_tile_size;

        *slot_id_dst_buf = slot_id;

        // memcpy hash_table_value to corresponding GPU
        value_dst_buf = h_hash_table_value_chunk_per_gpu[id] +
                        tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;

        memcpy(value_dst_buf, embedding_ptr + (i * chunk_size + k) * embedding_vec_size,
               hash_table_value_tile_size_in_B);

        tile_counter_in_chunk_per_gpu[id] += 1;
      } else {
        continue;
      }
    }  // end of for(int k = 0; k < (chunk_size * local_gpu_count); k++)

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
  }  // end of for(int i = 0; i < loop_num; i++)

  // process the remaining data(less than a chunk)
  const size_t remain_loop_num = num - loop_num * chunk_size;
  TypeHashKey *key_dst_buf;
  size_t *value_index_buf;
  size_t *slot_id_dst_buf;
  float *value_dst_buf;
  for (size_t i = 0; i < remain_loop_num; i++) {  // process one tile in each loop
    TypeHashKey key = key_ptr[loop_num * chunk_size + i];
    size_t slot_id = slot_id_ptr[loop_num * chunk_size + i];
    size_t gid = slot_id % total_gpu_count;  // global GPU ID
    size_t id = Base::get_resource_manager().get_gpu_local_id_from_global_id(
        gid);  // local GPU ID (not gpu devie id)
    int dst_rank = Base::get_resource_manager().get_process_id_from_gpu_global_id(gid);  // node id

    if (Base::get_resource_manager().get_process_id() == dst_rank) {
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

      // memcpy hash_table_slot_id to corresponding GPU
      size_t slot_id_offset = tile_counter_per_gpu[id];
      slot_id_dst_buf = hash_table_slot_id_tensors[id].get_ptr() + slot_id_offset;
      CK_CUDA_THROW_(cudaMemcpyAsync(slot_id_dst_buf, &slot_id, hash_table_slot_id_tile_size_in_B,
                                     cudaMemcpyHostToDevice, Base::get_local_gpu(id).get_stream()));

      // memcpy hash_table_value from CPU to GPU
      size_t value_offset = tile_counter_per_gpu[id] * embedding_vec_size;
      value_dst_buf = hash_table_value_tensors[id].get_ptr() + value_offset;
      CK_CUDA_THROW_(cudaMemcpyAsync(
          value_dst_buf, embedding_ptr + (loop_num * chunk_size + i) * embedding_vec_size,
          hash_table_value_tile_size_in_B, cudaMemcpyHostToDevice,
          Base::get_local_gpu(id).get_stream()));

      // set counter
      tile_counter_per_gpu[id] += hash_table_key_tile_size;
    } else {
      continue;
    }
  }

  // sync wait
  functors_.sync_all_gpus(Base::get_resource_manager());

  MESSAGE_("Done");

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
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::dump_parameters(
    std::string sparse_model) const {
  dump_parameters(sparse_model, max_vocabulary_size_, Base::get_embedding_vec_size(),
                  hash_table_value_tensors_, hash_table_slot_id_tensors_, hash_tables_);
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::dump_parameters(
    BufferBag &buf_bag, size_t *num) const {
  TensorBag2 keys_bag = buf_bag.keys;
  TensorBag2 slot_id_bag = buf_bag.slot_id;
  Tensor2<float> &embeddings = buf_bag.embedding;
  Tensor2<TypeHashKey> keys = Tensor2<TypeHashKey>::stretch_from(keys_bag);
  Tensor2<size_t> slot_id = Tensor2<size_t>::stretch_from(slot_id_bag);
  dump_parameters(keys, slot_id, embeddings, num, max_vocabulary_size_,
                  Base::get_embedding_vec_size(), hash_table_value_tensors_,
                  hash_table_slot_id_tensors_, hash_tables_);
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::dump_parameters(
    const std::string &sparse_model, size_t vocabulary_size, size_t embedding_vec_size,
    const Tensors2<float> &hash_table_value_tensors, const Tensors2<size_t> &hash_table_slot_id_tensors,
    const std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables) const {
  CudaDeviceContext context;
  size_t local_gpu_count = Base::get_resource_manager().get_local_gpu_count();

  if (!fs::exists(sparse_model)) {
    fs::create_directory(sparse_model);
  }
  const std::string key_file(sparse_model + "/" + sparse_model + ".key");
  const std::string slot_file(sparse_model + "/" + sparse_model + ".slot");
  const std::string vec_file(sparse_model + "/" + sparse_model + ".vec");

#ifdef ENABLE_MPI
  MPI_File key_fh, slot_fh, vec_fh;
  CK_MPI_THROW_(
    MPI_File_open(MPI_COMM_WORLD, key_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &key_fh));
  CK_MPI_THROW_(
    MPI_File_open(MPI_COMM_WORLD, slot_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &slot_fh));
  CK_MPI_THROW_(
    MPI_File_open(MPI_COMM_WORLD, vec_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &vec_fh));
#else
  std::ofstream key_stream(key_file, std::ofstream::binary | std::ofstream::trunc);
  std::ofstream slot_stream(slot_file, std::ofstream::binary | std::ofstream::trunc);
  std::ofstream vec_stream(vec_file, std::ofstream::binary | std::ofstream::trunc);
  // check if the file is opened successfully
  if (!vec_stream.is_open() || !key_stream.is_open() || !slot_stream.is_open()) {
    CK_THROW_(Error_t::WrongInput, "Error: file not open for writing");
    return;
  }
#endif

  // memory allocation
  std::unique_ptr<size_t[]> count(new size_t[local_gpu_count]);
  size_t total_count = 0;

  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(Base::get_local_gpu(id).get_device_id());
    auto count_tmp = hash_tables[id]->get_size(Base::get_local_gpu(id).get_stream());
    if (count_tmp != hash_tables[id]->get_value_head(Base::get_local_gpu(id).get_stream())) {
      CK_THROW_(Error_t::WrongInput,
                "Error: hash_table get_value_head() is not equal to get_size()");
    }
    count[id] = count_tmp;
    total_count += count[id];
  }

  if (total_count > (size_t)vocabulary_size) {
    CK_THROW_(Error_t::WrongInput,
              "Error: required download size is larger than hash table vocabulary_size");
  }

  std::vector<size_t> offset_host(local_gpu_count, 0);
  std::exclusive_scan(count.get(), count.get() + local_gpu_count, offset_host.begin(), 0);

  TypeHashKey *h_hash_table_key;
  size_t *h_hash_table_slot_id;
  float *h_hash_table_value;
  CK_CUDA_THROW_(cudaMallocHost(&h_hash_table_key, total_count * sizeof(TypeHashKey)));
  CK_CUDA_THROW_(cudaMallocHost(&h_hash_table_slot_id, total_count * sizeof(size_t)));
  CK_CUDA_THROW_(cudaMallocHost(&h_hash_table_value, total_count * embedding_vec_size * sizeof(float)));

  std::unique_ptr<TypeHashKey *[]> d_hash_table_key(new TypeHashKey *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_hash_table_value_index(new size_t *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_hash_table_slot_id(new size_t *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_dump_counter(new size_t *[local_gpu_count]);

  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) continue;
    context.set_device(Base::get_local_gpu(id).get_device_id());

    CK_CUDA_THROW_(cudaMallocManaged(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey)));
    CK_CUDA_THROW_(cudaMallocManaged(&d_hash_table_value_index[id], count[id] * sizeof(size_t)));
    CK_CUDA_THROW_(cudaMallocManaged(&d_hash_table_slot_id[id], count[id] * sizeof(size_t)));
    CK_CUDA_THROW_(cudaMallocManaged(&d_dump_counter[id], sizeof(size_t)));
  }

  // dump hash table from GPU
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) continue;
    context.set_device(Base::get_local_gpu(id).get_device_id());

    MESSAGE_("Rank" + std::to_string(Base::get_resource_manager().get_process_id()) +
             ": Dump hash table from GPU" + std::to_string(id), true);

    hash_tables[id]->dump(d_hash_table_key[id], d_hash_table_value_index[id], d_dump_counter[id],
                          Base::get_local_gpu(id).get_stream());

    CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_value + offset_host[id] * embedding_vec_size,
                                   hash_table_value_tensors[id].get_ptr(),
                                   count[id] * embedding_vec_size * sizeof(float),
                                   cudaMemcpyDeviceToHost, Base::get_local_gpu(id).get_stream()));
  }
  functors_.sync_all_gpus(Base::get_resource_manager());

  // sort key according to memory index
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) continue;
    context.set_device(Base::get_local_gpu(id).get_device_id());

    thrust::sort_by_key(thrust::device, d_hash_table_value_index[id],
                        d_hash_table_value_index[id] + count[id], d_hash_table_key[id]);

    CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_key + offset_host[id],
                                   d_hash_table_key[id],
                                   count[id] * sizeof(TypeHashKey),
                                   cudaMemcpyDeviceToHost, Base::get_local_gpu(id).get_stream()));

    CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_slot_id + offset_host[id],
                                   hash_table_slot_id_tensors[id].get_ptr(),
                                   count[id] * sizeof(size_t),
                                   cudaMemcpyDeviceToHost, Base::get_local_gpu(id).get_stream()));
  }
  functors_.sync_all_gpus(Base::get_resource_manager());

  const size_t key_size = sizeof(TypeHashKey);
  const size_t slot_size = sizeof(size_t);
  const size_t vec_size = sizeof(float) * embedding_vec_size;

  // write sparse model to file
  MESSAGE_("Rank" + std::to_string(Base::get_resource_manager().get_process_id()) +
           ": Write hash table <key,value> pairs to file", true);
#ifdef ENABLE_MPI
  int my_rank = Base::get_resource_manager().get_process_id();
  int n_ranks = Base::get_resource_manager().get_num_process();

  std::vector<size_t> offset_per_rank(n_ranks, 0);
  CK_MPI_THROW_(MPI_Allgather(&total_count, sizeof(size_t), MPI_CHAR, offset_per_rank.data(),
                              sizeof(size_t), MPI_CHAR, MPI_COMM_WORLD));
  std::exclusive_scan(offset_per_rank.begin(), offset_per_rank.end(), offset_per_rank.begin(), 0);

  size_t key_offset = offset_per_rank[my_rank] * key_size;
  size_t slot_offset = offset_per_rank[my_rank] * slot_size;
  size_t vec_offset = offset_per_rank[my_rank] * vec_size;

  CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
  MPI_Status status;
  CK_MPI_THROW_(MPI_File_write_at(key_fh, key_offset, h_hash_table_key, total_count * key_size, MPI_CHAR, &status));
  CK_MPI_THROW_(MPI_File_write_at(slot_fh, slot_offset, h_hash_table_slot_id, total_count * slot_size, MPI_CHAR, &status));
  CK_MPI_THROW_(MPI_File_write_at(vec_fh, vec_offset, h_hash_table_value, total_count * vec_size, MPI_CHAR, &status));

  CK_MPI_THROW_(MPI_File_close(&key_fh));
  CK_MPI_THROW_(MPI_File_close(&slot_fh));
  CK_MPI_THROW_(MPI_File_close(&vec_fh));
#else
  key_stream.write(reinterpret_cast<char*>(h_hash_table_key), total_count * key_size);
  slot_stream.write(reinterpret_cast<char*>(h_hash_table_slot_id), total_count * slot_size);
  vec_stream.write(reinterpret_cast<char*>(h_hash_table_value), total_count * vec_size);
#endif
  MESSAGE_("Done");

  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) continue;
    context.set_device(Base::get_local_gpu(id).get_device_id());

    CK_CUDA_THROW_(cudaFree(d_hash_table_key[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_value_index[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_slot_id[id]));
    CK_CUDA_THROW_(cudaFree(d_dump_counter[id]));
  }
  CK_CUDA_THROW_(cudaFreeHost(h_hash_table_key));
  CK_CUDA_THROW_(cudaFreeHost(h_hash_table_slot_id));
  CK_CUDA_THROW_(cudaFreeHost(h_hash_table_value));
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::dump_parameters(
    Tensor2<TypeHashKey> &keys, Tensor2<size_t> &slot_id, Tensor2<float> &embeddings, size_t *num,
    size_t vocabulary_size, size_t embedding_vec_size,
    const Tensors2<float> &hash_table_value_tensors,
    const Tensors2<size_t> &hash_table_slot_id_tensors,
    const std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables) const {
  TypeHashKey *key_ptr = keys.get_ptr();
  size_t *slot_id_ptr = slot_id.get_ptr();
  float *embedding_ptr = embeddings.get_ptr();

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

    CK_CUDA_THROW_(cudaMallocHost(&h_hash_table_key[id], count[id] * sizeof(TypeHashKey)));
    CK_CUDA_THROW_(cudaMallocManaged(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey)));
    CK_CUDA_THROW_(cudaMallocManaged(&d_hash_table_value_index[id], count[id] * sizeof(size_t)));
    CK_CUDA_THROW_(cudaMallocHost(&h_hash_table_slot_id[id], count[id] * sizeof(size_t)));
    CK_CUDA_THROW_(cudaMallocManaged(&d_hash_table_slot_id[id], count[id] * sizeof(size_t)));
    CK_CUDA_THROW_(cudaMallocHost(&h_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float)));
    CK_CUDA_THROW_(cudaMallocManaged(&d_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float)));
    CK_CUDA_THROW_(cudaMallocManaged(&d_dump_counter[id], count[id] * sizeof(size_t)));
  }

  // dump hash table on GPU
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) {
      continue;
    }

    MESSAGE_("Rank" + std::to_string(Base::get_resource_manager().get_process_id()) +
             ": Dump hash table from GPU" + std::to_string(id),
						 true);

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

  // TODO: could be optimized ???
  const size_t key_size = sizeof(TypeHashKey);
  const size_t slot_id_size = sizeof(size_t);
  const size_t value_size = sizeof(float) * embedding_vec_size;

  size_t offset = 0;
  for (size_t id = 0; id < local_gpu_count; id++) {
    for (unsigned int k = 0; k < count[id]; k++) {
      memcpy(key_ptr + offset, h_hash_table_key[id] + k, key_size);
      memcpy(slot_id_ptr + offset, h_hash_table_slot_id[id] + k, slot_id_size);
      memcpy(embedding_ptr + offset * embedding_vec_size,
             h_hash_table_value[id] + k * embedding_vec_size, value_size);
      offset += 1;
    }
    MESSAGE_("Write hash table <key,slot_id,value> pairs to file");
  }

  *num = offset;

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
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::dump_opt_states(
    std::ofstream& stream) {
  std::vector<OptimizerTensor<TypeEmbeddingComp>> opt_tensors_;
  for(auto &opt: embedding_optimizers_){
    opt_tensors_.push_back(opt.opt_tensors_);
  }
  auto opt_states = functors_.get_opt_states(opt_tensors_, Base::get_optimizer(), Base::get_resource_manager().get_local_gpu_count());

  functors_.dump_opt_states(stream, Base::get_resource_manager(), opt_states);
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::load_opt_states(
    std::ifstream& stream) {
  std::vector<OptimizerTensor<TypeEmbeddingComp>> opt_tensors_;
  for(auto &opt: embedding_optimizers_){
    opt_tensors_.push_back(opt.opt_tensors_);
  }
  auto opt_states = functors_.get_opt_states(opt_tensors_, Base::get_optimizer(), Base::get_resource_manager().get_local_gpu_count());

  functors_.load_opt_states(stream, Base::get_resource_manager(), opt_states);

}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::init_embedding(
    size_t max_vocabulary_size_per_gpu, size_t embedding_vec_size,
    Tensors2<float> &hash_table_value_tensors) {
#pragma omp parallel num_threads(Base::get_resource_manager().get_local_gpu_count())
  {
    size_t id = omp_get_thread_num();
    MESSAGE_("gpu" + std::to_string(id) + " start to init embedding");

    CudaDeviceContext context(Base::get_local_gpu(id).get_device_id());

    HugeCTR::UniformGenerator::fill(hash_table_value_tensors[id], -0.05f, 0.05f,
                                    Base::get_local_gpu(id).get_sm_count(),
                                    Base::get_local_gpu(id).get_replica_variant_curand_generator(),
                                    Base::get_local_gpu(id).get_stream());
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
      HugeCTR::UniformGenerator::fill(hash_table_value_tensors_[i], -0.05f, 0.05f,
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
