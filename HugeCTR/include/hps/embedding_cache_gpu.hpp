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

#pragma once
#include <embedding_cache_combined.h>

#include <gpu_cache_api.hpp>
#include <map>
#include <shared_mutex>
#include <vector>

using namespace ecache;
namespace HugeCTR {

template <typename key_type>
class EmbeddingCacheWrapper : public gpu_cache::gpu_cache_api<key_type> {
 public:
  using ModifyOp = typename EmbedCache<key_type, key_type>::ModifyOpType;

  // Ctor
  EmbeddingCacheWrapper(const size_t capacity_in_set, const size_t embedding_vec_size);

  // Dtor
  ~EmbeddingCacheWrapper() noexcept(false);

  // Query API, i.e. A single read from the cache
  void Query(const key_type* d_keys, const size_t len, float* d_values, uint64_t* d_missing_index,
             key_type* d_missing_keys, size_t* d_missing_len, cudaStream_t stream,
             const size_t task_per_warp_tile = 1) override;

  // Replace API, i.e. Follow the Query API to update the content of the cache to Most Recent
  void Replace(const key_type* d_keys, const size_t len, const float* d_values, cudaStream_t stream,
               const size_t task_per_warp_tile = 1) override;

  // Update API, i.e. update the embeddings which exist in the cache
  void Update(const key_type* d_keys, const size_t len, const float* d_values, cudaStream_t stream,
              const size_t task_per_warp_tile = 1) override;

  // Dump API, i.e. dump some slabsets' keys from the cache
  void Dump(key_type* d_keys, size_t* d_dump_counter, const size_t start_set_index,
            const size_t end_set_index, cudaStream_t stream) override;

 private:
  struct PerStreamLookupData {
    LookupContextHandle hLookup;
    cudaEvent_t event;
  };

  struct PerStreamModifyData {
    ModifyContextHandle hUpdate;
    key_type* hIndices;
    cudaEvent_t wait_token;
  };

  PerStreamModifyData GetModifyData(cudaStream_t stream);
  PerStreamLookupData GetLookupData(cudaStream_t stream);
  void SyncForModify() const;
  void ModifyInternal(const key_type* d_keys, const size_t len, const float* d_values,
                      cudaStream_t stream, ModifyOp op);

  void DestroyPerStreamLookupData(PerStreamLookupData& lookup);
  void DestroyPerStreamModifyData(PerStreamModifyData& modify);
  std::shared_ptr<EmbedCacheBase<key_type>> cache_ptr_;
  std::map<cudaStream_t, PerStreamLookupData> lookup_handle_map_;
  std::map<cudaStream_t, PerStreamModifyData> update_handle_map_;
  DefaultAllocator allocator_;
  DefaultLogger logger_;
  mutable std::shared_mutex
      read_write_lock_;  // mutable since query doesn't change cache but require a lock
  std::shared_mutex update_map_read_write_lock_;
  mutable std::shared_mutex
      lookup_map_read_write_lock_;  // mutable since query doesn't change cache but require a lock

  PerStreamLookupData m_gLData;
  typename EmbedCache<key_type, key_type>::CacheConfig config_;
  using WriteLock = std::unique_lock<std::shared_mutex>;
  using ReadLock = std::shared_lock<std::shared_mutex>;
};

}  // namespace HugeCTR