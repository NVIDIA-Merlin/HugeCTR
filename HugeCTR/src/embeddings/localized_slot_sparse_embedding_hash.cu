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
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <cub/cub.cuh>
#include <filesystem>
#include <numeric>

#include "HugeCTR/include/data_simulator.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/io/filesystem.hpp"
#include "HugeCTR/include/utils.cuh"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

namespace localized_filter_keys_kernel {

template <typename TypeKey>
__global__ void select_value_and_rowoffset_by_slot_id_kernel(
    const TypeKey *rowoffset_ptr, size_t num, char *flag, TypeKey *selected_rowoffset_ptr,
    size_t slot_num_per_gpu, size_t slot_num, size_t global_id, size_t global_num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num) {
    if (tid == 0) {
      selected_rowoffset_ptr[tid] = 0;
    }
    int batch_id = tid / slot_num;
    int slot_id = tid % slot_num;
    if (slot_id % global_num == global_id) {
      int res_slot_id = slot_id / global_num;
      selected_rowoffset_ptr[1 + batch_id * slot_num_per_gpu + res_slot_id] =
          rowoffset_ptr[tid + 1] - rowoffset_ptr[tid];
      for (TypeKey i = rowoffset_ptr[tid]; i < rowoffset_ptr[tid + 1]; ++i) {
        flag[i] = 1;
      }
    }
  }
}

}  // namespace localized_filter_keys_kernel

template <typename TypeHashKey>
LocalizedFilterKeyStorage<TypeHashKey>::LocalizedFilterKeyStorage(
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>> &buf, size_t max_nnz,
    size_t rowoffset_count) {
  buf->reserve({max_nnz}, &value_select_flag);
  buf->reserve({1}, &value_select_num);
  {
    size_t bytes_size = 0;
    cub::DeviceSelect::Flagged(nullptr, bytes_size, (TypeHashKey *)nullptr, (char *)nullptr,
                               (TypeHashKey *)nullptr, (size_t *)nullptr, max_nnz);
    buf->reserve({bytes_size}, &temp_value_select_storage);
  }

  buf->reserve({rowoffset_count}, &rowoffset_select);
  {
    size_t size_in_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, size_in_bytes, (TypeHashKey *)nullptr,
                                  (TypeHashKey *)nullptr, rowoffset_count);
    buf->reserve({size_in_bytes}, &temp_rowoffset_select_scan_storage);
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::filter_keys_per_gpu(
    bool is_train, size_t id, size_t global_id, size_t global_num) {
  const SparseTensor<TypeHashKey> &all_gather_key = embedding_data_.get_input_keys(is_train)[id];
  auto &local_gpu = embedding_data_.get_local_gpu(id);
  Tensor2<TypeHashKey> rowoffset_tensor = embedding_data_.get_row_offsets_tensors(is_train)[id];
  Tensor2<TypeHashKey> value_tensor = embedding_data_.get_value_tensors(is_train)[id];
  std::shared_ptr<size_t> nnz_ptr = embedding_data_.get_nnz_array(is_train)[id];
  auto &filter_keys_storage = filter_keys_storages_[id];

  if (all_gather_key.get_dimensions().size() != 2) {
    HCTR_OWN_THROW(Error_t::WrongInput, "localized embedding all gather key dimension != 2");
  }
  size_t batch_size = embedding_data_.embedding_params_.get_batch_size(is_train);
  size_t slot_num = (all_gather_key.rowoffset_count() - 1) / batch_size;
  size_t slot_num_per_gpu = slot_num_per_gpu_[id];
  size_t rowoffset_num_before_filter = batch_size * slot_num + 1;
  size_t rowoffset_num_without_zero = rowoffset_num_before_filter - 1;
  size_t rowoffset_num_after_filter = batch_size * slot_num_per_gpu + 1;
  if (rowoffset_tensor.get_num_elements() != rowoffset_num_after_filter) {
    HCTR_LOG_S(ERROR, WORLD) << "rowoffset_num_after_filter:" << rowoffset_num_after_filter << "!="
                             << "rowoffset_tensor num_elements:"
                             << rowoffset_tensor.get_num_elements() << std::endl;
    HCTR_OWN_THROW(Error_t::WrongInput, "filter rowoffset size not match.");
  }

  // select value and rowoffset
  {
    cudaMemsetAsync(filter_keys_storage.value_select_flag.get_ptr(), 0,
                    filter_keys_storage.value_select_flag.get_size_in_bytes(),
                    local_gpu.get_stream());
    {
      constexpr size_t block_size = 256;
      size_t grid_size = (rowoffset_num_without_zero - 1) / block_size + 1;
      localized_filter_keys_kernel::select_value_and_rowoffset_by_slot_id_kernel<<<
          grid_size, block_size, 0, local_gpu.get_stream()>>>(
          all_gather_key.get_rowoffset_ptr(), rowoffset_num_without_zero,
          filter_keys_storage.value_select_flag.get_ptr(),
          filter_keys_storage.rowoffset_select.get_ptr(), slot_num_per_gpu, slot_num, global_id,
          global_num);
    }
    {
      size_t size_in_bytes = filter_keys_storage.temp_value_select_storage.get_size_in_bytes();
      cub::DeviceSelect::Flagged(
          filter_keys_storage.temp_value_select_storage.get_ptr(), size_in_bytes,
          all_gather_key.get_value_ptr(), filter_keys_storage.value_select_flag.get_ptr(),
          value_tensor.get_ptr(), filter_keys_storage.value_select_num.get_ptr(),
          all_gather_key.nnz(), local_gpu.get_stream());
    }
    {
      size_t size_in_bytes =
          filter_keys_storage.temp_rowoffset_select_scan_storage.get_size_in_bytes();
      cub::DeviceScan::InclusiveSum(
          filter_keys_storage.temp_rowoffset_select_scan_storage.get_ptr(), size_in_bytes,
          filter_keys_storage.rowoffset_select.get_ptr(), rowoffset_tensor.get_ptr(),
          rowoffset_num_after_filter, local_gpu.get_stream());
    }
  }

  // select nnz
  {
    HCTR_LIB_THROW(cudaMemcpyAsync(nnz_ptr.get(), filter_keys_storage.value_select_num.get_ptr(),
                                   sizeof(size_t), cudaMemcpyDeviceToHost, local_gpu.get_stream()));
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::LocalizedSlotSparseEmbeddingHash(
    const SparseTensors<TypeHashKey> &train_keys, const SparseTensors<TypeHashKey> &evaluate_keys,
    const SparseEmbeddingHashParams &embedding_params,
    const std::shared_ptr<ResourceManager> &resource_manager)
    : embedding_data_(Embedding_t::LocalizedSlotSparseEmbeddingHash, train_keys, evaluate_keys,
                      embedding_params, resource_manager),
      slot_size_array_(embedding_params.slot_size_array) {
  try {
    if (slot_size_array_.empty()) {
      max_vocabulary_size_per_gpu_ = embedding_data_.embedding_params_.max_vocabulary_size_per_gpu;
      max_vocabulary_size_ = embedding_data_.embedding_params_.max_vocabulary_size_per_gpu *
                             embedding_data_.get_resource_manager().get_global_gpu_count();
    } else {
      max_vocabulary_size_per_gpu_ =
          cal_max_voc_size_per_gpu(slot_size_array_, embedding_data_.get_resource_manager());
      max_vocabulary_size_ = 0;
      for (size_t slot_size : slot_size_array_) {
        max_vocabulary_size_ += slot_size;
      }
    }

    HCTR_LOG_S(INFO, ROOT) << "max_vocabulary_size_per_gpu_=" << max_vocabulary_size_per_gpu_
                           << std::endl;

    CudaDeviceContext context;
    for (size_t id = 0; id < embedding_data_.get_resource_manager().get_local_gpu_count(); id++) {
      context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

      size_t gid = embedding_data_.get_local_gpu(id).get_global_id();
      size_t slot_num_per_gpu =
          embedding_data_.embedding_params_.slot_num /
              embedding_data_.get_resource_manager().get_global_gpu_count() +
          ((gid < embedding_data_.embedding_params_.slot_num %
                      embedding_data_.get_resource_manager().get_global_gpu_count())
               ? 1
               : 0);
      slot_num_per_gpu_.push_back(slot_num_per_gpu);
      // new GeneralBuffer objects
      const std::shared_ptr<GeneralBuffer2<CudaAllocator>> &buf = embedding_data_.get_buffer(id);
      embedding_optimizers_.emplace_back(max_vocabulary_size_per_gpu_,
                                         embedding_data_.embedding_params_, buf);

      // new hash table value vectors
      if (slot_size_array_.empty()) {
        Tensor2<float> tensor;
        buf->reserve(
            {max_vocabulary_size_per_gpu_, embedding_data_.embedding_params_.embedding_vec_size},
            &tensor);
        hash_table_value_tensors_.push_back(tensor);
      } else {
        const std::shared_ptr<BufferBlock2<float>> &block = buf->create_block<float>();
        Tensors2<float> tensors;
        size_t vocabulary_size_in_current_gpu = 0;
        for (size_t i = 0; i < slot_size_array_.size(); i++) {
          if ((i % embedding_data_.get_resource_manager().get_global_gpu_count()) == gid) {
            Tensor2<float> tensor;
            block->reserve(
                {slot_size_array_[i], embedding_data_.embedding_params_.embedding_vec_size},
                &tensor);
            tensors.push_back(tensor);
            vocabulary_size_in_current_gpu += slot_size_array_[i];
          }
        }
        value_table_tensors_.push_back(tensors);
        if (max_vocabulary_size_per_gpu_ > vocabulary_size_in_current_gpu) {
          Tensor2<float> padding_tensor_for_optimizer;
          block->reserve({max_vocabulary_size_per_gpu_ - vocabulary_size_in_current_gpu,
                          embedding_data_.embedding_params_.embedding_vec_size},
                         &padding_tensor_for_optimizer);
        }
        hash_table_value_tensors_.push_back(block->as_tensor());
      }
      {
        Tensor2<TypeHashKey> tensor;
        buf->reserve({embedding_data_.embedding_params_.get_batch_size(true),
                      embedding_data_.embedding_params_.max_feature_num},
                     &tensor);
        embedding_data_.train_value_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeHashKey> tensor;
        buf->reserve({embedding_data_.embedding_params_.get_batch_size(false),
                      embedding_data_.embedding_params_.max_feature_num},
                     &tensor);
        embedding_data_.evaluate_value_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeHashKey> tensor;
        buf->reserve(
            {embedding_data_.embedding_params_.get_batch_size(true) * slot_num_per_gpu + 1},
            &tensor);
        embedding_data_.train_row_offsets_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeHashKey> tensor;
        buf->reserve(
            {embedding_data_.embedding_params_.get_batch_size(false) * slot_num_per_gpu + 1},
            &tensor);
        embedding_data_.evaluate_row_offsets_tensors_.push_back(tensor);
      }
      { embedding_data_.train_nnz_array_.push_back(std::make_shared<size_t>(0)); }
      { embedding_data_.evaluate_nnz_array_.push_back(std::make_shared<size_t>(0)); }
      // new hash table value_index that get() from HashTable
      {
        Tensor2<size_t> tensor;
        buf->reserve({1, embedding_data_.embedding_params_.get_universal_batch_size() *
                             embedding_data_.embedding_params_.max_feature_num},
                     &tensor);
        hash_value_index_tensors_.push_back(tensor);
      }

      // new embedding features reduced by hash table values(results of forward)
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve(
            {embedding_data_.embedding_params_.get_universal_batch_size() * slot_num_per_gpu,
             embedding_data_.embedding_params_.embedding_vec_size},
            &tensor);
        embedding_feature_tensors_.push_back(tensor);
      }

      // new wgrad used by backward
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({embedding_data_.embedding_params_.get_batch_size(true) * slot_num_per_gpu,
                      embedding_data_.embedding_params_.embedding_vec_size},
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
        buf->reserve({embedding_data_.get_universal_batch_size_per_gpu() *
                          embedding_data_.embedding_params_.slot_num,
                      embedding_data_.embedding_params_.embedding_vec_size},
                     &tensor);
        all2all_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({embedding_data_.embedding_params_.get_universal_batch_size() *
                          embedding_data_.embedding_params_.slot_num,
                      embedding_data_.embedding_params_.embedding_vec_size},
                     &tensor);
        utest_forward_temp_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({embedding_data_.get_batch_size_per_gpu(true) *
                          embedding_data_.embedding_params_.slot_num,
                      embedding_data_.embedding_params_.embedding_vec_size},
                     &tensor);
        utest_all2all_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({embedding_data_.get_batch_size_per_gpu(true) *
                          embedding_data_.embedding_params_.slot_num,
                      embedding_data_.embedding_params_.embedding_vec_size},
                     &tensor);
        utest_reorder_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({embedding_data_.embedding_params_.get_batch_size(true) *
                          embedding_data_.embedding_params_.slot_num,
                      embedding_data_.embedding_params_.embedding_vec_size},
                     &tensor);
        utest_backward_temp_tensors_.push_back(tensor);
      }
      {
        size_t max_nnz = embedding_data_.embedding_params_.get_universal_batch_size() *
                         embedding_data_.embedding_params_.max_feature_num;
        size_t rowoffset_count = embedding_data_.embedding_params_.slot_num *
                                     embedding_data_.embedding_params_.get_universal_batch_size() +
                                 1;

        filter_keys_storages_.emplace_back(buf, max_nnz, rowoffset_count);
      }
// init GenenralBuffers to do real allocation
#ifndef NDEBUG
      HCTR_LOG_S(DEBUG, WORLD) << " max_feature_num_:"
                               << embedding_data_.embedding_params_.max_feature_num << std::endl;
#endif
    }

    hash_tables_.resize(embedding_data_.get_resource_manager().get_local_gpu_count());
#pragma omp parallel for num_threads(embedding_data_.get_resource_manager().get_local_gpu_count())
    for (size_t id = 0; id < embedding_data_.get_resource_manager().get_local_gpu_count(); id++) {
      CudaDeviceContext context(embedding_data_.get_local_gpu(id).get_device_id());
      // construct HashTable object: used to store hash table <key, value_index>
      hash_tables_[id].reset(new NvHashTable(max_vocabulary_size_per_gpu_));
      embedding_data_.get_buffer(id)->allocate();
    }

    for (size_t id = 0; id < embedding_data_.get_resource_manager().get_local_gpu_count(); id++) {
      context.set_device(embedding_data_.get_local_gpu(id).get_device_id());
      embedding_optimizers_[id].initialize(embedding_data_.get_local_gpu(id));

    }  // end of for(int id = 0; id < embedding_data_.get_local_gpu_count(); id++)

    if (!embedding_data_.embedding_params_.slot_size_array.empty()) {
      std::vector<TypeHashKey> embedding_offsets;
      TypeHashKey slot_sizes_prefix_sum = 0;
      for (size_t i = 0; i < embedding_data_.embedding_params_.slot_size_array.size(); i++) {
        embedding_offsets.push_back(slot_sizes_prefix_sum);
        slot_sizes_prefix_sum += embedding_data_.embedding_params_.slot_size_array[i];
      }
      for (size_t id = 0; id < embedding_data_.get_resource_manager().get_local_gpu_count(); ++id) {
        CudaDeviceContext context(embedding_data_.get_local_gpu(id).get_device_id());

        HCTR_LIB_THROW(
            cudaMemcpy(embedding_data_.embedding_offsets_[id].get_ptr(), embedding_offsets.data(),
                       embedding_offsets.size() * sizeof(TypeHashKey), cudaMemcpyHostToDevice));
      }
    }
    // sync
    functors_.sync_all_gpus(embedding_data_.get_resource_manager());

  } catch (const std::runtime_error &rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::load_parameters(
    std::string sparse_model) {
  const std::string key_file(sparse_model + "/key");
  const std::string slot_file(sparse_model + "/slot_id");
  const std::string vec_file(sparse_model + "/emb_vector");

  auto fs = FileSystemBuilder::build_unique_by_path(sparse_model);

  size_t key_file_size_in_byte = fs->get_file_size(key_file);
  size_t slot_file_size_in_byte = fs->get_file_size(slot_file);
  size_t vec_file_size_in_byte = fs->get_file_size(vec_file);

  size_t key_size = sizeof(long long);
  size_t slot_size = sizeof(size_t);
  size_t vec_size = sizeof(float) * embedding_data_.embedding_params_.embedding_vec_size;
  size_t key_num = key_file_size_in_byte / key_size;
  size_t slot_num = slot_file_size_in_byte / slot_size;
  size_t vec_num = vec_file_size_in_byte / vec_size;

  if (key_num != vec_num || key_file_size_in_byte % key_size != 0 ||
      vec_file_size_in_byte % vec_size != 0 || key_num != slot_num ||
      slot_file_size_in_byte % slot_size != 0) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Error: file size is not correct");
  }

  auto blobs_buff = GeneralBuffer2<CudaHostAllocator>::create();

  Tensor2<TypeHashKey> keys;
  blobs_buff->reserve({key_num}, &keys);

  Tensor2<size_t> slot_id;
  blobs_buff->reserve({slot_num}, &slot_id);

  Tensor2<float> embeddings;
  blobs_buff->reserve({vec_num, embedding_data_.embedding_params_.embedding_vec_size}, &embeddings);

  blobs_buff->allocate();

  TypeHashKey *key_ptr = keys.get_ptr();
  size_t *slot_id_ptr = slot_id.get_ptr();
  float *embedding_ptr = embeddings.get_ptr();

  if (std::is_same<TypeHashKey, long long>::value) {
    fs->read(key_file, reinterpret_cast<char *>(key_ptr), key_file_size_in_byte, 0);
  } else {
    std::vector<long long> i64_key_vec(key_num, 0);
    fs->read(key_file, reinterpret_cast<char *>(i64_key_vec.data()), key_file_size_in_byte, 0);
    std::transform(i64_key_vec.begin(), i64_key_vec.end(), key_ptr,
                   [](long long key) { return static_cast<unsigned>(key); });
  }
  fs->read(slot_file, reinterpret_cast<char *>(slot_id_ptr), slot_file_size_in_byte, 0);
  fs->read(vec_file, reinterpret_cast<char *>(embedding_ptr), vec_file_size_in_byte, 0);

  load_parameters(keys, slot_id, embeddings, key_num, max_vocabulary_size_,
                  embedding_data_.embedding_params_.embedding_vec_size,
                  max_vocabulary_size_per_gpu_, hash_table_value_tensors_,
                  hash_table_slot_id_tensors_, hash_tables_);
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::load_parameters(
    BufferBag &buf_bag, size_t num) {
  load_parameters(buf_bag, num, max_vocabulary_size_,
                  embedding_data_.embedding_params_.embedding_vec_size,
                  max_vocabulary_size_per_gpu_, hash_table_value_tensors_,
                  hash_table_slot_id_tensors_, hash_tables_);
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::load_parameters(
    BufferBag &buf_bag, size_t num, size_t vocabulary_size, size_t embedding_vec_size,
    size_t max_vocabulary_size_per_gpu, Tensors2<float> &hash_table_value_tensors,
    Tensors2<size_t> &hash_table_slot_id_tensors,
    std::vector<std::shared_ptr<NvHashTable>> &hash_tables) {
  if (num == 0) return;

  const TensorBag2 &keys_bag = buf_bag.keys;
  const TensorBag2 &slot_id_bag = buf_bag.slot_id;
  const Tensor2<float> &embeddings = buf_bag.embedding;
  Tensor2<TypeHashKey> keys = Tensor2<TypeHashKey>::stretch_from(keys_bag);
  Tensor2<size_t> slot_id = Tensor2<size_t>::stretch_from(slot_id_bag);

  if (keys.get_dimensions()[0] < num || embeddings.get_dimensions()[0] < num) {
    HCTR_OWN_THROW(Error_t::WrongInput, "The rows of keys and embeddings are not consistent.");
  }

  if (num > vocabulary_size) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "num_key to be loaded is larger than hash table vocabulary_size");
  }

  const TypeHashKey *key_ptr = keys.get_ptr();
  const size_t *slot_id_ptr = slot_id.get_ptr();
  const float *embedding_ptr = embeddings.get_ptr();

  int my_rank = embedding_data_.get_resource_manager().get_process_id();

  CudaDeviceContext context;
  const size_t local_gpu_count = embedding_data_.get_resource_manager().get_local_gpu_count();

  std::unique_ptr<size_t[]> counter_per_gpu(new size_t[local_gpu_count]);
  memset(counter_per_gpu.get(), 0, sizeof(size_t) * local_gpu_count);

  const size_t num_thread = std::thread::hardware_concurrency();
  std::vector<std::vector<std::vector<TypeHashKey>>> chunk_keys(num_thread);
  std::vector<std::vector<std::vector<size_t>>> chunk_slot_ids(num_thread);
  std::vector<std::vector<std::vector<size_t>>> chunk_src_indexs(num_thread);

  for (size_t tid = 0; tid < num_thread; tid++) {
    chunk_keys[tid].resize(local_gpu_count);
    chunk_slot_ids[tid].resize(local_gpu_count);
    chunk_src_indexs[tid].resize(local_gpu_count);
  }

#pragma omp parallel num_threads(num_thread)
  {
    const size_t tid = omp_get_thread_num();
    const size_t thread_num = omp_get_num_threads();
    size_t sub_chunk_size = num / thread_num;
    size_t res_chunk_size = num % thread_num;
    const size_t idx = tid * sub_chunk_size;

    if (tid == thread_num - 1) sub_chunk_size += res_chunk_size;

    for (size_t i = 0; i < sub_chunk_size; i++) {
      auto key = key_ptr[idx + i];
      auto gid = key % embedding_data_.get_resource_manager().get_global_gpu_count();
      auto id = embedding_data_.get_resource_manager().get_gpu_local_id_from_global_id(gid);
      auto dst_rank = embedding_data_.get_resource_manager().get_process_id_from_gpu_global_id(gid);

      if (dst_rank != my_rank) {
        HCTR_OWN_THROW(Error_t::UnspecificError, "ETC selected keys error");
      }

      chunk_keys[tid][id].push_back(key);
      chunk_slot_ids[tid][id].push_back(slot_id_ptr[idx + i]);
      chunk_src_indexs[tid][id].push_back(idx + i);
    }
  }

  std::vector<std::vector<size_t>> offset_per_thread(local_gpu_count);
#pragma omp parallel for num_threads(local_gpu_count)
  for (size_t id = 0; id < local_gpu_count; id++) {
    offset_per_thread[id].resize(num_thread, 0);

    std::vector<size_t> num_per_thread(num_thread, 0);
    for (size_t tid = 0; tid < num_thread; tid++) {
      counter_per_gpu[id] += chunk_keys[tid][id].size();
      num_per_thread[tid] = chunk_keys[tid][id].size();
    }
    std::exclusive_scan(num_per_thread.begin(), num_per_thread.end(), offset_per_thread[id].begin(),
                        0);
  }

  size_t total_count =
      std::accumulate(counter_per_gpu.get(), counter_per_gpu.get() + local_gpu_count, 0);
  if (total_count != num) {
    HCTR_OWN_THROW(Error_t::UnspecificError, "total_count != num_of_keys");
  }

  std::unique_ptr<TypeHashKey *[]> uvm_key_per_gpu(new TypeHashKey *[local_gpu_count]);
  std::unique_ptr<size_t *[]> h_slot_id_per_gpu(new size_t *[local_gpu_count]);
  std::unique_ptr<float *[]> h_value_per_gpu(new float *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_value_index_per_gpu(new size_t *[local_gpu_count]);

  for (size_t id = 0; id < local_gpu_count; id++) {
    uvm_key_per_gpu[id] =
        Tensor2<TypeHashKey>::stretch_from(buf_bag.uvm_key_tensor_bags[id]).get_ptr();
    h_slot_id_per_gpu[id] = buf_bag.h_slot_id_tensors[id].get_ptr();
    d_value_index_per_gpu[id] = buf_bag.d_value_index_tensors[id].get_ptr();
    h_value_per_gpu[id] = buf_bag.h_value_tensors[id].get_ptr();

    size_t value_index_size_in_B = counter_per_gpu[id] * sizeof(size_t);
    HCTR_LIB_THROW(cudaMemsetAsync(d_value_index_per_gpu[id], 0, value_index_size_in_B,
                                   embedding_data_.get_local_gpu(id).get_stream()));

    size_t key_size_in_B = counter_per_gpu[id] * sizeof(TypeHashKey);
    HCTR_LIB_THROW(cudaMemPrefetchAsync(uvm_key_per_gpu[id], key_size_in_B, cudaCpuDeviceId,
                                        embedding_data_.get_local_gpu(id).get_stream()));
  }
  functors_.sync_all_gpus(embedding_data_.get_resource_manager());

  std::vector<std::vector<size_t>> src_indexs(local_gpu_count);
  for (size_t id = 0; id < local_gpu_count; id++) {
    src_indexs[id].resize(counter_per_gpu[id]);

#pragma omp parallel for num_threads(num_thread)
    for (size_t tid = 0; tid < num_thread; tid++) {
      TypeHashKey *key_dst_ptr = uvm_key_per_gpu[id] + offset_per_thread[id][tid];
      TypeHashKey *key_src_ptr = chunk_keys[tid][id].data();
      size_t key_size_in_B = chunk_keys[tid][id].size() * sizeof(TypeHashKey);
      memcpy(key_dst_ptr, key_src_ptr, key_size_in_B);

      size_t *slot_id_dst_ptr = h_slot_id_per_gpu[id] + offset_per_thread[id][tid];
      size_t *slot_id_src_ptr = chunk_slot_ids[tid][id].data();
      size_t slot_id_size_in_B = chunk_slot_ids[tid][id].size() * sizeof(size_t);
      memcpy(slot_id_dst_ptr, slot_id_src_ptr, slot_id_size_in_B);

      size_t *idx_dst_ptr = src_indexs[id].data() + offset_per_thread[id][tid];
      size_t *idx_src_ptr = chunk_src_indexs[tid][id].data();
      size_t idx_size_in_B = chunk_src_indexs[tid][id].size() * sizeof(size_t);
      memcpy(idx_dst_ptr, idx_src_ptr, idx_size_in_B);
    }

#pragma omp parallel for num_threads(num_thread)
    for (size_t i = 0; i < src_indexs[id].size(); i++) {
      float *vec_dst_ptr = h_value_per_gpu[id] + i * embedding_vec_size;
      const float *vec_src_ptr = embedding_ptr + src_indexs[id][i] * embedding_vec_size;
      size_t vec_size_in_B = embedding_vec_size * sizeof(float);
      memcpy(vec_dst_ptr, vec_src_ptr, vec_size_in_B);
    }
  }

// do HashTable insert <key,value_index>
#pragma omp parallel for num_threads(local_gpu_count)
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (counter_per_gpu[id] == 0) continue;
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

    const size_t counter = counter_per_gpu[id];

    // memcpy hash_table_key from CPU to GPU
    size_t key_size_in_B = counter * sizeof(TypeHashKey);
    HCTR_LIB_THROW(cudaMemPrefetchAsync(uvm_key_per_gpu[id], key_size_in_B, id,
                                        embedding_data_.get_local_gpu(id).get_stream()));

    // set hash_table_value_index on GPU
    functors_.memset_liner(d_value_index_per_gpu[id], 0ul, 1ul, counter,
                           embedding_data_.get_local_gpu(id).get_stream());

    // do hash table insert <key, value_index> on GPU
    hash_tables[id]->insert(uvm_key_per_gpu[id], d_value_index_per_gpu[id], counter,
                            embedding_data_.get_local_gpu(id).get_stream());
    hash_tables[id]->set_value_head(counter, embedding_data_.get_local_gpu(id).get_stream());

    // memcpy slot_id from CPU to GPU
    size_t slot_id_size_in_B = counter * sizeof(size_t);
    HCTR_LIB_THROW(cudaMemcpyAsync(hash_table_slot_id_tensors[id].get_ptr(), h_slot_id_per_gpu[id],
                                   slot_id_size_in_B, cudaMemcpyHostToDevice,
                                   embedding_data_.get_local_gpu(id).get_stream()));

    // memcpy hash_table_value from CPU to GPU
    size_t vec_block_in_B = counter * embedding_vec_size * sizeof(float);
    HCTR_LIB_THROW(cudaMemcpyAsync(hash_table_value_tensors[id].get_ptr(), h_value_per_gpu[id],
                                   vec_block_in_B, cudaMemcpyHostToDevice,
                                   embedding_data_.get_local_gpu(id).get_stream()));
  }
  functors_.sync_all_gpus(embedding_data_.get_resource_manager());
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
    HCTR_OWN_THROW(Error_t::WrongInput, "The rows of keys and embeddings are not consistent.");
  }

  if (num > vocabulary_size) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "Error: hash table file size is larger than hash table vocabulary_size");
  }

  const TypeHashKey *key_ptr = keys.get_ptr();
  const size_t *slot_id_ptr = slot_id.get_ptr();
  const float *embedding_ptr = embeddings.get_ptr();

  int my_rank = embedding_data_.get_resource_manager().get_process_id();
  int n_ranks = embedding_data_.get_resource_manager().get_num_process();

  // define size
  size_t local_gpu_count = embedding_data_.get_resource_manager().get_local_gpu_count();
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
  size_t total_gpu_count = embedding_data_.get_resource_manager().get_global_gpu_count();

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
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());
    HCTR_LIB_THROW(cudaMalloc(&d_hash_table_value_index_chunk_per_gpu[id],
                              hash_table_value_index_chunk_size_in_B));
    // initalize to zeros
    HCTR_LIB_THROW(cudaMemsetAsync(d_hash_table_value_index_chunk_per_gpu[id], 0,
                                   hash_table_value_index_chunk_size_in_B,
                                   embedding_data_.get_local_gpu(id).get_stream()));
  }

  // sync wait
  functors_.sync_all_gpus(embedding_data_.get_resource_manager());

  // CAUSION: can not decide how many values for each GPU, so need to allocate enough memory
  // for each GPU allocate CPU/GPU memory for hash_table/key/value chunk
  std::unique_ptr<TypeHashKey *[]> h_hash_table_key_chunk_per_gpu(
      new TypeHashKey *[local_gpu_count]);
  for (size_t id = 0; id < local_gpu_count; id++) {
    HCTR_LIB_THROW(
        cudaMallocHost(&h_hash_table_key_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
  }
  std::unique_ptr<TypeHashKey *[]> d_hash_table_key_chunk_per_gpu(
      new TypeHashKey *[local_gpu_count]);
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());
    HCTR_LIB_THROW(cudaMalloc(&d_hash_table_key_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
  }
  std::unique_ptr<size_t *[]> h_hash_table_slot_id_chunk_per_gpu(new size_t *[local_gpu_count]);
  for (size_t id = 0; id < local_gpu_count; id++) {
    HCTR_LIB_THROW(cudaMallocHost(&h_hash_table_slot_id_chunk_per_gpu[id],
                                  hash_table_slot_id_chunk_size_in_B));
  }
  std::unique_ptr<size_t *[]> d_hash_table_slot_id_chunk_per_gpu(new size_t *[local_gpu_count]);
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());
    HCTR_LIB_THROW(
        cudaMalloc(&d_hash_table_slot_id_chunk_per_gpu[id], hash_table_slot_id_chunk_size_in_B));
  }
  std::unique_ptr<float *[]> h_hash_table_value_chunk_per_gpu(new float *[local_gpu_count]);
  for (size_t id = 0; id < local_gpu_count; id++) {
    HCTR_LIB_THROW(
        cudaMallocHost(&h_hash_table_value_chunk_per_gpu[id], hash_table_value_chunk_size_in_B));
  }

  // do upload
  const size_t loop_num = num / chunk_size;
  HCTR_LOG_S(INFO, ROOT) << "Start to upload embedding table file to GPUs, total loop_num: "
                         << loop_num << std::endl;
  for (size_t i = 0; i < loop_num; i++) {
    TypeHashKey *key_dst_buf;
    size_t *slot_id_dst_buf;
    float *value_dst_buf;
    for (size_t k = 0; k < chunk_size; k++) {  // process a tile in each loop
      TypeHashKey key = key_ptr[i * chunk_size + k];
      size_t slot_id = slot_id_ptr[i * chunk_size + k];
      size_t gid = slot_id % total_gpu_count;  // global GPU ID
      size_t id = embedding_data_.get_resource_manager().get_gpu_local_id_from_global_id(
          gid);  // local GPU ID (not gpudevice id)
      int dst_rank =
          embedding_data_.get_resource_manager().get_process_id_from_gpu_global_id(gid);  // node id

      if (embedding_data_.get_resource_manager().get_process_id() == dst_rank) {
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

      context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

      size_t tile_count = tile_counter_in_chunk_per_gpu[id];

      // memcpy hash_table_key from CPU to GPU
      HCTR_LIB_THROW(cudaMemcpyAsync(d_hash_table_key_chunk_per_gpu[id],
                                     h_hash_table_key_chunk_per_gpu[id],
                                     tile_count * sizeof(TypeHashKey), cudaMemcpyHostToDevice,
                                     embedding_data_.get_local_gpu(id).get_stream()));

      size_t value_index_offset = tile_counter_per_gpu[id];
      size_t *value_index_buf = d_hash_table_value_index_chunk_per_gpu[id];

      if (tile_count > 0) {
        // set hash_table_value_index on GPU
        functors_.memset_liner(value_index_buf, value_index_offset, 1ul, tile_count,
                               embedding_data_.get_local_gpu(id).get_stream());
      }

      // do hash table insert <key, value_index> on GPU
      hash_tables[id]->insert(d_hash_table_key_chunk_per_gpu[id], value_index_buf, tile_count,
                              embedding_data_.get_local_gpu(id).get_stream());
      size_t value_head = hash_tables[id]->get_and_add_value_head(
          tile_count, embedding_data_.get_local_gpu(id).get_stream());
    }

    // memcpy hash_table_slot_id and hash_table_value from CPU to GPU
    for (size_t id = 0; id < local_gpu_count; id++) {
      if (tile_counter_in_chunk_per_gpu[id] == 0) {
        continue;
      }

      context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

      size_t slot_id_chunk_size = tile_counter_in_chunk_per_gpu[id] * hash_table_slot_id_tile_size;
      size_t slot_id_offset = tile_counter_per_gpu[id] * hash_table_slot_id_tile_size;

      if ((slot_id_offset + slot_id_chunk_size) > max_vocabulary_size_per_gpu) {
        std::ostringstream os;
        os << "The size of hash table on GPU" << id << " is out of range "
           << max_vocabulary_size_per_gpu << '.' << std::endl;
        HCTR_OWN_THROW(Error_t::OutOfBound, os.str());
      }

      size_t *src_buf_sid = h_hash_table_slot_id_chunk_per_gpu[id];
      size_t *dst_buf_sid = hash_table_slot_id_tensors[id].get_ptr() + slot_id_offset;
      HCTR_LIB_THROW(cudaMemcpyAsync(dst_buf_sid, src_buf_sid, slot_id_chunk_size * sizeof(size_t),
                                     cudaMemcpyHostToDevice,
                                     embedding_data_.get_local_gpu(id).get_stream()));

      size_t value_chunk_size = tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
      size_t value_chunk_offset = tile_counter_per_gpu[id] * hash_table_value_tile_size;
      float *src_buf_value = h_hash_table_value_chunk_per_gpu[id];
      float *dst_buf_value = hash_table_value_tensors[id].get_ptr() + value_chunk_offset;
      HCTR_LIB_THROW(cudaMemcpyAsync(dst_buf_value, src_buf_value, value_chunk_size * sizeof(float),
                                     cudaMemcpyHostToDevice,
                                     embedding_data_.get_local_gpu(id).get_stream()));
    }

    functors_.sync_all_gpus(embedding_data_.get_resource_manager());

    // set counter value
    for (size_t id = 0; id < local_gpu_count; id++) {
      tile_counter_per_gpu[id] +=
          tile_counter_in_chunk_per_gpu[id];  // accumulate total tile counter
      tile_counter_in_chunk_per_gpu[id] = 0;  // reset chunk counter to zero

      if (tile_counter_per_gpu[id] > max_vocabulary_size_per_gpu) {
        std::ostringstream os;
        os << "The size of hash table on GPU" << id << " is out of range "
           << max_vocabulary_size_per_gpu << '.' << std::endl;
        HCTR_OWN_THROW(Error_t::OutOfBound, os.str());
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
    size_t id = embedding_data_.get_resource_manager().get_gpu_local_id_from_global_id(
        gid);  // local GPU ID (not gpu devie id)
    int dst_rank =
        embedding_data_.get_resource_manager().get_process_id_from_gpu_global_id(gid);  // node id

    if (embedding_data_.get_resource_manager().get_process_id() == dst_rank) {
      context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

      // memcpy hash_table_key from CPU to GPU
      key_dst_buf = d_hash_table_key_chunk_per_gpu[id];
      HCTR_LIB_THROW(cudaMemcpyAsync(key_dst_buf, &key, hash_table_key_tile_size_in_B,
                                     cudaMemcpyHostToDevice,
                                     embedding_data_.get_local_gpu(id).get_stream()));

      // set value_index
      size_t value_index_offset = tile_counter_per_gpu[id];
      value_index_buf = d_hash_table_value_index_chunk_per_gpu[id];
      functors_.memset_liner(value_index_buf, value_index_offset, 1ul, 1ul,
                             embedding_data_.get_local_gpu(id).get_stream());

      // do hash table insert <key, value_index> on GPU
      hash_tables[id]->insert(d_hash_table_key_chunk_per_gpu[id], value_index_buf,
                              hash_table_key_tile_size,
                              embedding_data_.get_local_gpu(id).get_stream());
      size_t value_head = hash_tables[id]->get_and_add_value_head(
          hash_table_key_tile_size, embedding_data_.get_local_gpu(id).get_stream());

      // memcpy hash_table_slot_id to corresponding GPU
      size_t slot_id_offset = tile_counter_per_gpu[id];
      slot_id_dst_buf = hash_table_slot_id_tensors[id].get_ptr() + slot_id_offset;
      HCTR_LIB_THROW(cudaMemcpyAsync(slot_id_dst_buf, &slot_id, hash_table_slot_id_tile_size_in_B,
                                     cudaMemcpyHostToDevice,
                                     embedding_data_.get_local_gpu(id).get_stream()));

      // memcpy hash_table_value from CPU to GPU
      size_t value_offset = tile_counter_per_gpu[id] * embedding_vec_size;
      value_dst_buf = hash_table_value_tensors[id].get_ptr() + value_offset;
      HCTR_LIB_THROW(cudaMemcpyAsync(
          value_dst_buf, embedding_ptr + (loop_num * chunk_size + i) * embedding_vec_size,
          hash_table_value_tile_size_in_B, cudaMemcpyHostToDevice,
          embedding_data_.get_local_gpu(id).get_stream()));

      // set counter
      tile_counter_per_gpu[id] += hash_table_key_tile_size;
    } else {
      continue;
    }
  }

  // sync wait
  functors_.sync_all_gpus(embedding_data_.get_resource_manager());

  HCTR_LOG(INFO, ROOT, "Done\n");

  // release resources
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());
    HCTR_LIB_THROW(cudaFree(d_hash_table_value_index_chunk_per_gpu[id]));
    HCTR_LIB_THROW(cudaFree(d_hash_table_key_chunk_per_gpu[id]));
  }
  for (size_t id = 0; id < local_gpu_count; id++) {
    HCTR_LIB_THROW(cudaFreeHost(h_hash_table_key_chunk_per_gpu[id]));
    HCTR_LIB_THROW(cudaFreeHost(h_hash_table_value_chunk_per_gpu[id]));
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::dump_parameters(
    std::string sparse_model) const {
  dump_parameters(sparse_model, max_vocabulary_size_,
                  embedding_data_.embedding_params_.embedding_vec_size, hash_table_value_tensors_,
                  hash_table_slot_id_tensors_, hash_tables_);
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
                  embedding_data_.embedding_params_.embedding_vec_size, hash_table_value_tensors_,
                  hash_table_slot_id_tensors_, hash_tables_);
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::dump_parameters(
    const std::string &sparse_model, size_t vocabulary_size, size_t embedding_vec_size,
    const Tensors2<float> &hash_table_value_tensors,
    const Tensors2<size_t> &hash_table_slot_id_tensors,
    const std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables) const {
  CudaDeviceContext context;
  size_t local_gpu_count = embedding_data_.get_resource_manager().get_local_gpu_count();

  auto fs = FileSystemBuilder::build_unique_by_path(sparse_model);
  const std::string key_file(sparse_model + "/key");
  const std::string slot_file(sparse_model + "/slot_id");
  const std::string vec_file(sparse_model + "/emb_vector");

#ifdef ENABLE_MPI
  MPI_File key_fh, slot_fh, vec_fh;
  HCTR_MPI_THROW(MPI_File_open(MPI_COMM_WORLD, key_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                               MPI_INFO_NULL, &key_fh));
  HCTR_MPI_THROW(MPI_File_open(MPI_COMM_WORLD, slot_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                               MPI_INFO_NULL, &slot_fh));
  HCTR_MPI_THROW(MPI_File_open(MPI_COMM_WORLD, vec_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                               MPI_INFO_NULL, &vec_fh));
#endif

  // memory allocation
  std::unique_ptr<size_t[]> count(new size_t[local_gpu_count]);
  size_t total_count = 0;

  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());
    auto count_tmp = hash_tables[id]->get_size(embedding_data_.get_local_gpu(id).get_stream());
    if (count_tmp !=
        hash_tables[id]->get_value_head(embedding_data_.get_local_gpu(id).get_stream())) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Error: hash_table get_value_head() is not equal to get_size()");
    }
    count[id] = count_tmp;
    total_count += count[id];
  }

  if (total_count > (size_t)vocabulary_size) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "Error: required download size is larger than hash table vocabulary_size");
  }

  std::vector<size_t> offset_host(local_gpu_count, 0);
  std::exclusive_scan(count.get(), count.get() + local_gpu_count, offset_host.begin(), 0);

  TypeHashKey *h_hash_table_key;
  size_t *h_hash_table_slot_id;
  float *h_hash_table_value;
  HCTR_LIB_THROW(cudaMallocHost(&h_hash_table_key, total_count * sizeof(TypeHashKey)));
  HCTR_LIB_THROW(cudaMallocHost(&h_hash_table_slot_id, total_count * sizeof(size_t)));
  HCTR_LIB_THROW(
      cudaMallocHost(&h_hash_table_value, total_count * embedding_vec_size * sizeof(float)));

  std::unique_ptr<TypeHashKey *[]> d_hash_table_key(new TypeHashKey *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_hash_table_value_index(new size_t *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_hash_table_slot_id(new size_t *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_dump_counter(new size_t *[local_gpu_count]);

  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) continue;
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

    HCTR_LIB_THROW(cudaMallocManaged(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey)));
    HCTR_LIB_THROW(cudaMallocManaged(&d_hash_table_value_index[id], count[id] * sizeof(size_t)));
    HCTR_LIB_THROW(cudaMallocManaged(&d_hash_table_slot_id[id], count[id] * sizeof(size_t)));
    HCTR_LIB_THROW(cudaMallocManaged(&d_dump_counter[id], sizeof(size_t)));
  }

  // dump hash table from GPU
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) continue;
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

    HCTR_LOG_S(INFO, WORLD) << "Rank" << embedding_data_.get_resource_manager().get_process_id()
                            << ": Dump hash table from GPU" << id << std::endl;

    hash_tables[id]->dump(d_hash_table_key[id], d_hash_table_value_index[id], d_dump_counter[id],
                          embedding_data_.get_local_gpu(id).get_stream());

    HCTR_LIB_THROW(cudaMemcpyAsync(
        h_hash_table_value + offset_host[id] * embedding_vec_size,
        hash_table_value_tensors[id].get_ptr(), count[id] * embedding_vec_size * sizeof(float),
        cudaMemcpyDeviceToHost, embedding_data_.get_local_gpu(id).get_stream()));
  }
  functors_.sync_all_gpus(embedding_data_.get_resource_manager());

  // sort key according to memory index
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) continue;
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

    thrust::sort_by_key(thrust::device, d_hash_table_value_index[id],
                        d_hash_table_value_index[id] + count[id], d_hash_table_key[id]);

    HCTR_LIB_THROW(cudaMemcpyAsync(h_hash_table_key + offset_host[id], d_hash_table_key[id],
                                   count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost,
                                   embedding_data_.get_local_gpu(id).get_stream()));

    HCTR_LIB_THROW(cudaMemcpyAsync(h_hash_table_slot_id + offset_host[id],
                                   hash_table_slot_id_tensors[id].get_ptr(),
                                   count[id] * sizeof(size_t), cudaMemcpyDeviceToHost,
                                   embedding_data_.get_local_gpu(id).get_stream()));
  }
  functors_.sync_all_gpus(embedding_data_.get_resource_manager());

  long long *h_key_ptr;
  std::vector<long long> i64_key_vec;
  if (std::is_same<TypeHashKey, long long>::value) {
    h_key_ptr = reinterpret_cast<long long *>(h_hash_table_key);
  } else {
    i64_key_vec.resize(total_count);
    std::transform(h_hash_table_key, h_hash_table_key + total_count, i64_key_vec.begin(),
                   [](unsigned key) { return static_cast<long long>(key); });
    h_key_ptr = i64_key_vec.data();
  }

  const size_t key_size = sizeof(long long);
  const size_t slot_size = sizeof(size_t);
  const size_t vec_size = sizeof(float) * embedding_vec_size;

  // write sparse model to file
  HCTR_LOG_S(INFO, WORLD) << "Rank" << embedding_data_.get_resource_manager().get_process_id()
                          << ": Write hash table <key,value> pairs to file" << std::endl;
#ifdef ENABLE_MPI
  MPI_Datatype TYPE_EMB_VECTOR;
  HCTR_MPI_THROW(MPI_Type_contiguous(embedding_vec_size, MPI_FLOAT, &TYPE_EMB_VECTOR));
  HCTR_MPI_THROW(MPI_Type_commit(&TYPE_EMB_VECTOR));

  int my_rank = embedding_data_.get_resource_manager().get_process_id();
  int n_ranks = embedding_data_.get_resource_manager().get_num_process();

  std::vector<size_t> offset_per_rank(n_ranks, 0);
  HCTR_MPI_THROW(MPI_Allgather(&total_count, sizeof(size_t), MPI_CHAR, offset_per_rank.data(),
                               sizeof(size_t), MPI_CHAR, MPI_COMM_WORLD));
  std::exclusive_scan(offset_per_rank.begin(), offset_per_rank.end(), offset_per_rank.begin(), 0);

  size_t key_offset = offset_per_rank[my_rank] * key_size;
  size_t slot_offset = offset_per_rank[my_rank] * slot_size;
  size_t vec_offset = offset_per_rank[my_rank] * vec_size;

  HCTR_MPI_THROW(MPI_Barrier(MPI_COMM_WORLD));
  MPI_Status status;
  HCTR_MPI_THROW(
      MPI_File_write_at(key_fh, key_offset, h_key_ptr, total_count, MPI_LONG_LONG_INT, &status));
  HCTR_MPI_THROW(MPI_File_write_at(slot_fh, slot_offset, h_hash_table_slot_id, total_count,
                                   MPI_SIZE_T, &status));
  HCTR_MPI_THROW(MPI_File_write_at(vec_fh, vec_offset, h_hash_table_value, total_count,
                                   TYPE_EMB_VECTOR, &status));

  HCTR_MPI_THROW(MPI_File_close(&key_fh));
  HCTR_MPI_THROW(MPI_File_close(&slot_fh));
  HCTR_MPI_THROW(MPI_File_close(&vec_fh));
  HCTR_MPI_THROW(MPI_Type_free(&TYPE_EMB_VECTOR));
#else
  fs->write(key_file, reinterpret_cast<char *>(h_key_ptr), total_count * key_size, true);
  fs->write(slot_file, reinterpret_cast<char *>(h_hash_table_slot_id), total_count * slot_size,
            true);
  fs->write(vec_file, reinterpret_cast<char *>(h_hash_table_value), total_count * vec_size, true);
#endif
  HCTR_LOG(INFO, ROOT, "Done\n");

  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) continue;
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

    HCTR_LIB_THROW(cudaFree(d_hash_table_key[id]));
    HCTR_LIB_THROW(cudaFree(d_hash_table_value_index[id]));
    HCTR_LIB_THROW(cudaFree(d_hash_table_slot_id[id]));
    HCTR_LIB_THROW(cudaFree(d_dump_counter[id]));
  }
  HCTR_LIB_THROW(cudaFreeHost(h_hash_table_key));
  HCTR_LIB_THROW(cudaFreeHost(h_hash_table_slot_id));
  HCTR_LIB_THROW(cudaFreeHost(h_hash_table_value));
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::dump_parameters(
    Tensor2<TypeHashKey> &keys, Tensor2<size_t> &slot_id, Tensor2<float> &embeddings, size_t *num,
    size_t vocabulary_size, size_t embedding_vec_size,
    const Tensors2<float> &hash_table_value_tensors,
    const Tensors2<size_t> &hash_table_slot_id_tensors,
    const std::vector<std::shared_ptr<NvHashTable>> &hash_tables) const {
  TypeHashKey *key_ptr = keys.get_ptr();
  size_t *slot_id_ptr = slot_id.get_ptr();
  float *embedding_ptr = embeddings.get_ptr();

  size_t local_gpu_count = embedding_data_.get_resource_manager().get_local_gpu_count();

  // memory allocation
  std::unique_ptr<size_t[]> count(new size_t[local_gpu_count]);
  size_t total_count = 0;

  CudaDeviceContext context;
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());
    auto count_tmp_1 = hash_tables[id]->get_size(embedding_data_.get_local_gpu(id).get_stream());
    auto count_tmp_2 =
        hash_tables[id]->get_value_head(embedding_data_.get_local_gpu(id).get_stream());
    if (count_tmp_1 != count_tmp_2) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Error: hash_table get_value_head() is not equal to get_size()");
    }
    count[id] = count_tmp_1;
    total_count += count[id];
  }

  if (total_count > (size_t)vocabulary_size) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Required download size > hash table vocabulary_size");
  }

  std::vector<size_t> offset_host(local_gpu_count, 0);
  std::exclusive_scan(count.get(), count.get() + local_gpu_count, offset_host.begin(), 0);
  *num = total_count;

  std::unique_ptr<TypeHashKey *[]> d_hash_table_key(new TypeHashKey *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_hash_table_value_index(new size_t *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_hash_table_slot_id(new size_t *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_dump_counter(new size_t *[local_gpu_count]);
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) continue;
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

    HCTR_LIB_THROW(cudaMalloc(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey)));
    HCTR_LIB_THROW(cudaMalloc(&d_hash_table_value_index[id], count[id] * sizeof(size_t)));
    HCTR_LIB_THROW(cudaMalloc(&d_hash_table_slot_id[id], count[id] * sizeof(size_t)));
    HCTR_LIB_THROW(cudaMalloc(&d_dump_counter[id], sizeof(size_t)));
  }

  // dump hash table from GPU
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) continue;
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

    hash_tables[id]->dump(d_hash_table_key[id], d_hash_table_value_index[id], d_dump_counter[id],
                          embedding_data_.get_local_gpu(id).get_stream());

    HCTR_LIB_THROW(cudaMemcpyAsync(
        embedding_ptr + offset_host[id] * embedding_vec_size,
        hash_table_value_tensors_[id].get_ptr(), count[id] * embedding_vec_size * sizeof(float),
        cudaMemcpyDeviceToHost, embedding_data_.get_local_gpu(id).get_stream()));
  }
  functors_.sync_all_gpus(embedding_data_.get_resource_manager());

  // sort key according to memory index
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) continue;
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

    thrust::sort_by_key(thrust::device, d_hash_table_value_index[id],
                        d_hash_table_value_index[id] + count[id], d_hash_table_key[id]);

    HCTR_LIB_THROW(cudaMemcpyAsync(key_ptr + offset_host[id], d_hash_table_key[id],
                                   count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost,
                                   embedding_data_.get_local_gpu(id).get_stream()));

    HCTR_LIB_THROW(cudaMemcpyAsync(slot_id_ptr + offset_host[id],
                                   hash_table_slot_id_tensors[id].get_ptr(),
                                   count[id] * sizeof(size_t), cudaMemcpyDeviceToHost,
                                   embedding_data_.get_local_gpu(id).get_stream()));
  }
  functors_.sync_all_gpus(embedding_data_.get_resource_manager());

  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) continue;
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

    HCTR_LIB_THROW(cudaFree(d_hash_table_key[id]));
    HCTR_LIB_THROW(cudaFree(d_hash_table_value_index[id]));
    HCTR_LIB_THROW(cudaFree(d_hash_table_slot_id[id]));
    HCTR_LIB_THROW(cudaFree(d_dump_counter[id]));
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::dump_opt_states(
    std::string write_path) {
  std::vector<OptimizerTensor<TypeEmbeddingComp>> opt_tensors_;
  for (auto &opt : embedding_optimizers_) {
    opt_tensors_.push_back(opt.opt_tensors_);
  }
  auto opt_states =
      functors_.get_opt_states(opt_tensors_, embedding_data_.embedding_params_.opt_params.optimizer,
                               embedding_data_.get_resource_manager().get_local_gpu_count());

  functors_.dump_opt_states(write_path, embedding_data_.get_resource_manager(), opt_states);
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::load_opt_states(
    std::string read_path) {
  std::vector<OptimizerTensor<TypeEmbeddingComp>> opt_tensors_;
  for (auto &opt : embedding_optimizers_) {
    opt_tensors_.push_back(opt.opt_tensors_);
  }
  auto opt_states =
      functors_.get_opt_states(opt_tensors_, embedding_data_.embedding_params_.opt_params.optimizer,
                               embedding_data_.get_resource_manager().get_local_gpu_count());

  functors_.load_opt_states(read_path, embedding_data_.get_resource_manager(), opt_states);
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::init_embedding(
    size_t max_vocabulary_size_per_gpu, size_t embedding_vec_size,
    Tensors2<float> &hash_table_value_tensors) {
#pragma omp parallel num_threads(embedding_data_.get_resource_manager().get_local_gpu_count())
  {
    const size_t id = omp_get_thread_num();
    HCTR_LOG_S(INFO, ROOT) << "gpu" << id << " start to init embedding" << std::endl;

    CudaDeviceContext context(embedding_data_.get_local_gpu(id).get_device_id());

    HugeCTR::UniformGenerator::fill(
        hash_table_value_tensors[id], -0.05f, 0.05f,
        embedding_data_.get_local_gpu(id).get_sm_count(),
        embedding_data_.get_local_gpu(id).get_replica_variant_curand_generator(),
        embedding_data_.get_local_gpu(id).get_stream());
    HCTR_LIB_THROW(cudaStreamSynchronize(embedding_data_.get_local_gpu(id).get_stream()));
    HCTR_LOG_S(INFO, ROOT) << "gpu" << id << " init embedding done" << std::endl;
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::init_embedding(
    const std::vector<size_t> &slot_sizes, size_t embedding_vec_size,
    std::vector<Tensors2<float>> &hash_table_value_tensors,
    Tensors2<size_t> &hash_table_slot_id_tensors) {
  size_t local_gpu_count = embedding_data_.get_resource_manager().get_local_gpu_count();
  size_t total_gpu_count = embedding_data_.get_resource_manager().get_global_gpu_count();

#ifndef NDEBUG
  HCTR_LOG_S(DEBUG, ROOT) << "local_gpu_count=" << local_gpu_count
                          << ", total_gpu_count=" << total_gpu_count << std::endl;
#endif

  for (size_t id = 0; id < local_gpu_count; id++) {
    size_t device_id = embedding_data_.get_local_gpu(id).get_device_id();
    size_t global_id = embedding_data_.get_local_gpu(id).get_global_id();

#ifndef NDEBUG
    HCTR_LOG_S(DEBUG, ROOT) << "id=" << id << ", device_id=" << device_id
                            << ", global_id=" << global_id << std::endl;
#endif

    functors_.init_embedding_per_gpu(global_id, total_gpu_count, slot_sizes, embedding_vec_size,
                                     hash_table_value_tensors[id], hash_table_slot_id_tensors[id],
                                     embedding_data_.get_local_gpu(id));
  }

  for (size_t id = 0; id < local_gpu_count; id++) {
    HCTR_LIB_THROW(cudaStreamSynchronize(embedding_data_.get_local_gpu(id).get_stream()));
    HCTR_LOG_S(INFO, ROOT) << "gpu" << id << " init embedding done" << std::endl;
  }

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::reset() {
  CudaDeviceContext context;
  for (size_t i = 0; i < embedding_data_.get_resource_manager().get_local_gpu_count(); i++) {
    context.set_device(embedding_data_.get_local_gpu(i).get_device_id());
    hash_tables_[i]->clear(embedding_data_.get_local_gpu(i).get_stream());

    if (slot_size_array_.empty()) {
      HugeCTR::UniformGenerator::fill(
          hash_table_value_tensors_[i], -0.05f, 0.05f,
          embedding_data_.get_local_gpu(i).get_sm_count(),
          embedding_data_.get_local_gpu(i).get_replica_variant_curand_generator(),
          embedding_data_.get_local_gpu(i).get_stream());
    } else {
      functors_.init_embedding_per_gpu(
          embedding_data_.get_local_gpu(i).get_global_id(),
          embedding_data_.get_resource_manager().get_global_gpu_count(), slot_size_array_,
          embedding_data_.embedding_params_.embedding_vec_size, value_table_tensors_[i],
          hash_table_slot_id_tensors_[i], embedding_data_.get_local_gpu(i));
    }
  }

  for (size_t i = 0; i < embedding_data_.get_resource_manager().get_local_gpu_count(); i++) {
    HCTR_LIB_THROW(cudaStreamSynchronize(embedding_data_.get_local_gpu(i).get_stream()));
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingHash<TypeHashKey, TypeEmbeddingComp>::reset_optimizer() {
  CudaDeviceContext context;
  auto local_gpu_count{embedding_data_.get_resource_manager().get_local_gpu_count()};
  for (size_t id{0}; id < local_gpu_count; id++) {
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());
    embedding_optimizers_[id].reset(embedding_data_.get_local_gpu(id));
  }
}

template class LocalizedSlotSparseEmbeddingHash<unsigned int, float>;
template class LocalizedSlotSparseEmbeddingHash<long long, float>;
template class LocalizedSlotSparseEmbeddingHash<unsigned int, __half>;
template class LocalizedSlotSparseEmbeddingHash<long long, __half>;

}  // namespace HugeCTR
