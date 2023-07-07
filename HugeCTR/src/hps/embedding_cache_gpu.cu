/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <embedding_cache_combined.cuh>
#include <hps/embedding_cache_gpu.hpp>
#include <nv_gpu_cache.hpp>
#include <utils.hpp>

namespace HugeCTR {

template <typename key_type>
EmbeddingCacheWrapper<key_type>::EmbeddingCacheWrapper(const size_t capacity_in_set,
                                                       const size_t embedding_vec_size) {
  config_.cacheSzInBytes =
      SLAB_SIZE * SET_ASSOCIATIVITY * capacity_in_set * embedding_vec_size * sizeof(float);
  config_.embedWidth = embedding_vec_size * sizeof(float);
  config_.maxUpdateSampleSz = capacity_in_set * EmbedCache<key_type, key_type>::NUM_WAYS;
  config_.numTables = 1;
  cache_ptr_ = std::make_shared<EmbedCache<key_type, key_type>>(&allocator_, &logger_, config_);
  cache_ptr_->Init();
}

// Dtor
template <typename key_type>
EmbeddingCacheWrapper<key_type>::~EmbeddingCacheWrapper() {
  {
    WriteLock(lookup_map_read_write_lock_);
    // destroy all lookup data
    for (auto li : lookup_handle_map_) {
      DestroyPerStreamLookupData(li.second);
    }
  }

  {
    WriteLock(update_map_read_write_lock_);
    for (auto ui : update_handle_map_) {
      DestroyPerStreamModifyData(ui.second);
    }
  }
}
template <typename key_type>
void EmbeddingCacheWrapper<key_type>::DestroyPerStreamLookupData(PerStreamLookupData& lookup) {
  CUDA_CHECK(cudaEventDestroy(lookup.event));
  cache_ptr_->LookupContextDestroy(lookup.hLookup);
}

template <typename key_type>
void EmbeddingCacheWrapper<key_type>::DestroyPerStreamModifyData(PerStreamModifyData& modify) {
  CUDA_CHECK(cudaEventDestroy(modify.wait_token));
  cache_ptr_->ModifyContextDestroy(modify.hUpdate);
  CUDA_CHECK(cudaFreeHost(modify.hIndices));
}

// Query API, i.e. A single read from the cache
template <typename key_type>
void EmbeddingCacheWrapper<key_type>::Query(const key_type* d_keys, const size_t len,
                                            float* d_values, uint64_t* d_missing_index,
                                            key_type* d_missing_keys, size_t* d_missing_len,
                                            cudaStream_t stream, const size_t task_per_warp_tile) {
  ReadLock l(read_write_lock_);
  auto hLookup = GetLookupData(stream);
  // call cache Query
  cache_ptr_->Lookup(hLookup.hLookup, d_keys, len, (int8_t*)d_values, d_missing_index,
                     d_missing_keys, d_missing_len, 0, config_.embedWidth, stream);
  CUDA_CHECK(cudaEventRecord(hLookup.event, stream));
}

// Replace API, i.e. Follow the Query API to update the content of the cache to Most Recent
template <typename key_type>
void EmbeddingCacheWrapper<key_type>::Replace(const key_type* d_keys, const size_t len,
                                              const float* d_values, cudaStream_t stream,
                                              const size_t task_per_warp_tile) {
  ModifyInternal(d_keys, len, d_values, stream, ModifyOp::MODIFY_REPLACE);
}

// Update API, i.e. update the embeddings which exist in the cache
template <typename key_type>
void EmbeddingCacheWrapper<key_type>::Update(const key_type* d_keys, const size_t len,
                                             const float* d_values, cudaStream_t stream,
                                             const size_t task_per_warp_tile) {
  ModifyInternal(d_keys, len, d_values, stream, ModifyOp::MODIFY_UPDATE);
}

template <typename key_type>
EmbeddingCacheWrapper<key_type>::PerStreamLookupData EmbeddingCacheWrapper<key_type>::GetLookupData(
    cudaStream_t stream) {
  // need to readlock
  // it = lookup_map.find(stream)
  // release lock
  // if (it == end)
  //   write lock
  //   search again see if we already have
  //   if (it == end)
  //       create new context
  PerStreamLookupData ret = {0};
  bool needToInit = false;
  {
    ReadLock l(lookup_map_read_write_lock_);
    if (lookup_handle_map_.find(stream) == lookup_handle_map_.end()) {
      needToInit = true;
    } else {
      ret = lookup_handle_map_[stream];
    }
  }

  if (needToInit) {
    WriteLock l(update_map_read_write_lock_);
    // search again if things were changed between locks
    if (lookup_handle_map_.find(stream) == lookup_handle_map_.end()) {
      cache_ptr_->LookupContextCreate(ret.hLookup, nullptr, 0);
      CUDA_CHECK(cudaEventCreate(&ret.event));
      lookup_handle_map_[stream] = ret;
    }
    ret = lookup_handle_map_[stream];
  }

  return ret;
}

template <typename key_type>
EmbeddingCacheWrapper<key_type>::PerStreamModifyData EmbeddingCacheWrapper<key_type>::GetModifyData(
    cudaStream_t stream) {
  PerStreamModifyData ret = {0};

  bool needToInit = false;
  {
    ReadLock l(update_map_read_write_lock_);
    if (update_handle_map_.find(stream) == update_handle_map_.end()) {
      needToInit = true;
    } else {
      ret = update_handle_map_[stream];
    }
  }

  if (needToInit) {
    WriteLock l(update_map_read_write_lock_);
    // search again if things were changed between locks
    if (update_handle_map_.find(stream) == update_handle_map_.end()) {
      cache_ptr_->ModifyContextCreate(ret.hUpdate, config_.maxUpdateSampleSz);
      CUDA_CHECK(cudaMallocHost(&ret.hIndices, config_.maxUpdateSampleSz * sizeof(key_type)));
      CUDA_CHECK(cudaEventCreate(&ret.wait_token));
      update_handle_map_[stream] = ret;
    }
    ret = update_handle_map_[stream];
  }
  // we should initialized ret.hIndices
  assert(ret.hIndices);
  return ret;
}

template <typename key_type>
void EmbeddingCacheWrapper<key_type>::ModifyInternal(const key_type* d_keys, const size_t len,
                                                     const float* d_values, cudaStream_t stream,
                                                     ModifyOp op) {
  auto hUpdate = GetModifyData(stream);
  auto modifiedLen = std::min(len, (size_t)config_.maxUpdateSampleSz);
  CUDA_CHECK(cudaMemcpyAsync(hUpdate.hIndices, d_keys, sizeof(key_type) * modifiedLen,
                             cudaMemcpyDefault, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  switch (op) {
    case ModifyOp::MODIFY_UPDATE:
      cache_ptr_->ModifyContextSetUpdateData(hUpdate.hUpdate, hUpdate.hIndices, modifiedLen,
                                             (const int8_t*)d_values, 0, config_.embedWidth, false);
      break;
    case ModifyOp::MODIFY_REPLACE:
      cache_ptr_->ModifyContextSetReplaceData(hUpdate.hUpdate, hUpdate.hIndices, modifiedLen,
                                              (const int8_t*)d_values, 0, config_.embedWidth,
                                              false);
      break;
    default:
      assert(0);
  }

  WriteLock l(read_write_lock_);
  SyncForModify();
  cache_ptr_->Modify(hUpdate.hUpdate, stream);
  CUDA_CHECK(cudaEventRecord(hUpdate.wait_token, stream));
  CUDA_CHECK(cudaEventSynchronize(hUpdate.wait_token));
}

template <typename key_type>
void EmbeddingCacheWrapper<key_type>::SyncForModify() const {
  // technically no need to lock here but since modify and query is mutually exclusive i don't know
  // who else is going to call me it better to be safe
  ReadLock l(lookup_map_read_write_lock_);
  for (auto& psl : lookup_handle_map_) {
    CUDA_CHECK(cudaEventSynchronize(psl.second.event));
  }
}

// Dump API, i.e. dump some slabsets' keys from the cache
template <typename key_type>
void EmbeddingCacheWrapper<key_type>::Dump(key_type* d_keys, size_t* d_dump_counter,
                                           const size_t start_set_index, const size_t end_set_index,
                                           cudaStream_t stream) {}

template class EmbeddingCacheWrapper<unsigned int>;
template class EmbeddingCacheWrapper<uint64_t>;
template class EmbeddingCacheWrapper<long long>;

}  // namespace HugeCTR

template class ecache::EmbedCache<long long, long long>;
template void ecache::callModifyKernel<long long, long long>(
    typename ecache::EmbedCache<long long, long long>::ModifyEntry* pEntries, uint32_t nEntries,
    uint32_t rowSizeInBytes, cudaStream_t stream);
template void ecache::callModifyKernel<unsigned int, unsigned int>(
    typename ecache::EmbedCache<unsigned int, unsigned int>::ModifyEntry* pEntries,
    uint32_t nEntries, uint32_t rowSizeInBytes, cudaStream_t stream);

template void
ecache::callCacheQueryUVM<long long, typename ecache::EmbedCache<long long, long long>::CacheData>(
    const long long* d_keys, const size_t len, int8_t* d_values, const int8_t* d_table,
    typename ecache::EmbedCache<long long, long long>::CacheData data, cudaStream_t stream,
    uint32_t currTable, size_t stride);

template void ecache::callCacheQueryUVM<
    unsigned int, typename ecache::EmbedCache<unsigned int, unsigned int>::CacheData>(
    const unsigned int* d_keys, const size_t len, int8_t* d_values, const int8_t* d_table,
    typename ecache::EmbedCache<unsigned int, unsigned int>::CacheData data, cudaStream_t stream,
    uint32_t currTable, size_t stride);