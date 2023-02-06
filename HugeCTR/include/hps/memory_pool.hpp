/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuda_runtime_api.h>

#include <base/debug/logger.hpp>
#include <hps/embedding_cache.hpp>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#define MAX_MEMORY_SIZE 256

namespace HugeCTR {

class MemoryPool;

class MemoryBlock {
 public:
  MemoryBlock* pNext;  // next mem block
  EmbeddingCacheWorkspace worker_buffer;
  EmbeddingCacheRefreshspace refresh_buffer;
  bool bUsed;        // occupied
  bool bBelong;      // belong to current pool
  MemoryPool* pMem;  // blong to which pool
  MemoryBlock() {
    this->bBelong = false;
    this->bUsed = false;
    this->pMem = nullptr;
    this->pNext = nullptr;
  };
};
class MemoryPool {
 public:
  MemoryPool(size_t nBlock, std::shared_ptr<EmbeddingCacheBase> embedding_cache,
             CACHE_SPACE_TYPE cache_type = CACHE_SPACE_TYPE::WORKER) {
    _nBlock = nBlock;
    _pHeader = nullptr;
    _pBuffer = nullptr;
    _embedding_cache = embedding_cache;
    _cache_type = cache_type;
    InitMemory(cache_type);
  }
  static MemoryPool* create(size_t nBlock, std::shared_ptr<EmbeddingCacheBase> embedding_cache,
                            CACHE_SPACE_TYPE cache_type = CACHE_SPACE_TYPE::WORKER) {
    return (new MemoryPool(nBlock, embedding_cache, cache_type));
  }
  virtual ~MemoryPool() {
    for (size_t i = 0; i < _nBlock; i++) {
      if (_Alloc[i] != nullptr) {
        delete _Alloc[i];
      }
    }
  }
  void* AllocMemory() {
    std::lock_guard<std::mutex> lock(_mutex);
    MemoryBlock* pRes = nullptr;
    if (_pHeader == nullptr) {
      HCTR_LOG(WARNING, WORLD, "memory pool is empty\n");
    } else {
      pRes = _pHeader;
      _pHeader = _pHeader->pNext;
      pRes->bUsed = true;
    }
    return (void*)(pRes);
  }

  void InitMemory(CACHE_SPACE_TYPE space_type = CACHE_SPACE_TYPE::WORKER) {
    if (_pBuffer) return;
    _Alloc[0] = (MemoryBlock*)(new MemoryBlock());
    _pBuffer = _Alloc[0];
    _pHeader = _pBuffer;
    _pHeader->bUsed = false;
    _pHeader->bBelong = true;
    if (space_type == CACHE_SPACE_TYPE::WORKER) {
      EmbeddingCacheWorkspace worker_buffer = _embedding_cache->create_workspace();
      _pHeader->worker_buffer = (worker_buffer);
    }
    if (space_type == CACHE_SPACE_TYPE::REFRESHER) {
      EmbeddingCacheRefreshspace refresh_buffer = _embedding_cache->create_refreshspace();
      _pHeader->refresh_buffer = (refresh_buffer);
    }
    _pHeader->pMem = this;
    MemoryBlock* tmp1 = _pHeader;
    for (size_t i = 1; i < _nBlock; i++) {
      _Alloc[i] = (MemoryBlock*)(new MemoryBlock());
      _Alloc[i]->bUsed = false;
      _Alloc[i]->pNext = NULL;
      if (space_type == CACHE_SPACE_TYPE::WORKER) {
        EmbeddingCacheWorkspace worker_buffer = _embedding_cache->create_workspace();
        _Alloc[i]->worker_buffer = (worker_buffer);
      }
      if (space_type == CACHE_SPACE_TYPE::REFRESHER) {
        EmbeddingCacheRefreshspace refresh_buffer = _embedding_cache->create_refreshspace();
        _Alloc[i]->refresh_buffer = (refresh_buffer);
      }
      _Alloc[i]->bBelong = true;
      _Alloc[i]->pMem = this;
      tmp1->pNext = _Alloc[i];
      tmp1 = _Alloc[i];
    }
  }

  void FreeMemory(void* p) {
    std::lock_guard<std::mutex> lock(_mutex);
    MemoryBlock* pBlock = (MemoryBlock*)p;
    if (pBlock->bBelong) {
      pBlock->bUsed = false;
      pBlock->pNext = _pHeader;
      _pHeader = pBlock;
    }
    return;
  }

  void DestoryMemoryPool(CACHE_SPACE_TYPE space_type = CACHE_SPACE_TYPE::WORKER) {
    std::lock_guard<std::mutex> lock(_mutex);
    for (size_t i = 0; i < _nBlock; i++) {
      if (_Alloc[i] != NULL) {
        if (space_type == CACHE_SPACE_TYPE::WORKER) {
          _embedding_cache->destroy_workspace(((MemoryBlock*)(_Alloc[i]))->worker_buffer);
        }
        if (space_type == CACHE_SPACE_TYPE::REFRESHER) {
          _embedding_cache->destroy_refreshspace(((MemoryBlock*)(_Alloc[i]))->refresh_buffer);
        }
      }
    }
  }

 public:
  MemoryBlock* _pBuffer;
  MemoryBlock* _pHeader;
  std::shared_ptr<EmbeddingCacheBase> _embedding_cache;
  size_t _nBlock;
  std::mutex _mutex;
  MemoryBlock* _Alloc[MAX_MEMORY_SIZE];
  CACHE_SPACE_TYPE _cache_type;
};

class ManagerPool {
 public:
  void* AllocBuffer(std::string model_name, int device_id,
                    CACHE_SPACE_TYPE space_type = CACHE_SPACE_TYPE::WORKER) {
    if (space_type == CACHE_SPACE_TYPE::WORKER &&
        _model_pool_map.find(model_name) != _model_pool_map.end()) {
      auto device_pool_map = _model_pool_map.find(model_name);
      auto cache = device_pool_map->second.find(device_id);
      if (cache != device_pool_map->second.end()) {
        return cache->second->AllocMemory();
      }
    }
    if (space_type == CACHE_SPACE_TYPE::REFRESHER &&
        _model_refresh_pool_map.find(model_name) != _model_refresh_pool_map.end()) {
      auto device_pool_map = _model_refresh_pool_map.find(model_name);
      auto cache = device_pool_map->second.find(device_id);
      if (cache != device_pool_map->second.end()) {
        return cache->second->AllocMemory();
      }
    }
    return NULL;
  }

  void FreeBuffer(void* p) {
    MemoryBlock* pBlock = (MemoryBlock*)((char*)p);
    if (pBlock->bBelong) {
      pBlock->pMem->FreeMemory(p);
    }
  }

  void DestoryManagerPool(CACHE_SPACE_TYPE space_type = CACHE_SPACE_TYPE::WORKER) {
    std::map<std::string, std::map<int64_t, std::shared_ptr<MemoryPool>>>::iterator iter;
    for (iter = _model_pool_map.begin(); iter != _model_pool_map.end();) {
      for (auto& f : iter->second) {
        f.second->DestoryMemoryPool(CACHE_SPACE_TYPE::WORKER);
      }
      iter = _model_pool_map.erase(iter);
    }
    _model_pool_map.clear();
    for (iter = _model_refresh_pool_map.begin(); iter != _model_refresh_pool_map.end();) {
      for (auto& f : iter->second) {
        f.second->DestoryMemoryPool(CACHE_SPACE_TYPE::REFRESHER);
      }
      iter = _model_refresh_pool_map.erase(iter);
    }
    _model_refresh_pool_map.clear();
  }

  void DestoryManagerPool(std::string model_name,
                          CACHE_SPACE_TYPE space_type = CACHE_SPACE_TYPE::WORKER) {
    if (_model_pool_map.find(model_name) != _model_pool_map.end()) {
      for (auto& f : _model_pool_map[model_name]) {
        f.second->DestoryMemoryPool(CACHE_SPACE_TYPE::WORKER);
      }
      _model_pool_map.erase(model_name);
    }

    if (_model_refresh_pool_map.find(model_name) != _model_refresh_pool_map.end()) {
      for (auto& f : _model_refresh_pool_map[model_name]) {
        f.second->DestoryMemoryPool(CACHE_SPACE_TYPE::REFRESHER);
      }
      _model_refresh_pool_map.erase(model_name);
    }
  }

  size_t _ncache = 1;
  ManagerPool(
      std::map<std::string, std::map<int64_t, std::shared_ptr<EmbeddingCacheBase>>> model_cache_map,
      inference_memory_pool_size_config memory_pool_config) {
    _model_cache_map = model_cache_map;
    _memory_pool_config = memory_pool_config;
    _create_memory_pool_map(&_model_pool_map, _memory_pool_config.num_woker_buffer_size_per_model,
                            CACHE_SPACE_TYPE::WORKER);
    _create_memory_pool_map(&_model_refresh_pool_map,
                            _memory_pool_config.num_refresh_buffer_size_per_model,
                            CACHE_SPACE_TYPE::REFRESHER);
    /*std::map<std::string, std::map<int64_t, std::shared_ptr<embedding_interface>>>::iterator iter;
      for (iter = _model_cache_map.begin(); iter != _model_cache_map.end(); ++iter) {
        _create_memory_pool_per_model(iter->first,_memory_pool_config.num_woker_buffer_size_per_model[iter->first],
      iter->second,CACHE_SPACE_TYPE::WORKER);
        _create_memory_pool_per_model(iter->first,_memory_pool_config.num_refresh_buffer_size_per_model[iter->first],
      iter->second,CACHE_SPACE_TYPE::REFRESHER);
      } */
  }

  void _create_memory_pool_per_model(
      std::string model_name, int pool_size,
      std::map<int64_t, std::shared_ptr<EmbeddingCacheBase>> embedding_cache_map,
      CACHE_SPACE_TYPE space_type) {
    std::map<int64_t, std::shared_ptr<MemoryPool>> device_mempool;
    for (auto& f : embedding_cache_map) {
      MemoryPool* tempmemorypool = MemoryPool::create(pool_size, f.second, space_type);
      device_mempool[f.first] = std::shared_ptr<MemoryPool>(tempmemorypool);
    }
    if (space_type == CACHE_SPACE_TYPE::WORKER) {
      _model_pool_map[model_name] = device_mempool;
    }
    if (space_type == CACHE_SPACE_TYPE::REFRESHER) {
      _model_refresh_pool_map[model_name] = device_mempool;
    }
  }

  void _create_memory_pool_map(
      std::map<std::string, std::map<int64_t, std::shared_ptr<MemoryPool>>>* model_cache_pool_map,
      std::map<std::string, int> num_cache_per_model, CACHE_SPACE_TYPE space_type) {
    std::map<std::string, std::map<int64_t, std::shared_ptr<EmbeddingCacheBase>>>::iterator iter;
    for (iter = _model_cache_map.begin(); iter != _model_cache_map.end(); ++iter) {
      std::map<int64_t, std::shared_ptr<MemoryPool>> device_mempool;
      for (auto& f : iter->second) {
        MemoryPool* tempmemorypool =
            MemoryPool::create(num_cache_per_model[iter->first], f.second, space_type);
        device_mempool[f.first] = std::shared_ptr<MemoryPool>(tempmemorypool);
      }
      (*model_cache_pool_map)[iter->first] = device_mempool;
    }
  }

  ~ManagerPool() {}

 private:
  std::map<std::string, std::map<int64_t, std::shared_ptr<EmbeddingCacheBase>>> _model_cache_map;
  std::map<std::string, std::map<int64_t, std::shared_ptr<MemoryPool>>> _model_pool_map;
  std::map<std::string, std::map<int64_t, std::shared_ptr<MemoryPool>>> _model_refresh_pool_map;
  inference_memory_pool_size_config _memory_pool_config;
};

}  // namespace HugeCTR