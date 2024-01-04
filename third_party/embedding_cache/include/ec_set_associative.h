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
#include "embed_cache.h"
#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <memory>
#include <assert.h>
#include <mutex>
#include <algorithm>
namespace ecache {

template<typename IndexT, typename TagT>
class EmbedCache;

template<typename IndexT, typename TagT>
void callCacheQuery(const IndexT* d_keys, const size_t len,
    int8_t* d_values, uint64_t* d_missing_index,
    IndexT* d_missing_keys, size_t* d_missing_len,
    typename EmbedCache<IndexT, TagT>::CacheData data,
    cudaStream_t stream, uint32_t currTable, size_t stride);

template<typename IndexT, typename CacheDataT>
void callCacheQueryUVM(const IndexT* d_keys, const size_t len,
    int8_t* d_values, const int8_t* d_table,
    CacheDataT data, cudaStream_t stream, uint32_t currTable, size_t stride);

template<typename IndexT, typename TagT>
void callMemUpdateKernel(typename EmbedCache<IndexT, TagT>::ModifyEntry* pEntries, uint32_t nEntries, uint32_t rowSizeInBytes, cudaStream_t stream);

template<typename IndexT, typename TagT>
void callTagUpdateKernel(typename EmbedCache<IndexT, TagT>::ModifyEntry* pEntries, uint32_t nEntries, TagT* pTags, cudaStream_t stream);

template<typename IndexT, typename TagT>
void callTagInvalidateKernel(typename EmbedCache<IndexT, TagT>::ModifyEntry* pEntries, uint32_t nEntries, TagT* pTags, cudaStream_t stream);

template<typename IndexT, typename TagT>
class EmbedCache : public EmbedCacheBase<IndexT>
{

public:
    using CounterT = uint32_t;
    using MissT = uint64_t;

    struct CacheData
    {
        int8_t* pCache; 
        const int8_t* pTags;
        uint32_t nSets;
        uint64_t rowSizeInBytes;
        bool bCountMisses;
        MissT* misses;
    };

    enum ModifyOpType
    {
        MODIFY_REPLACE,
        MODIFY_UPDATE,
        MODIFY_INVALIDATE
    };

    struct CacheConfig
    {
        size_t cacheSzInBytes = 0;
        uint64_t embedWidth = 0;
        uint64_t numTables = 1;
    };

    struct ModifyEntry
    {
        const int8_t* pSrc;
        int8_t* pDst;
        uint32_t set;
        uint32_t way;
        TagT tag;
    };

    struct ModifyContext
    {
        TagT* pTags; // host tags
        CounterT* pCounters;
        uint32_t* pWayUpdateMask; // host update mask - bit is set iff a way in the set require a replacement
        IndexT* pIndexMap;
        ModifyEntry* pEntries;
        ModifyEntry* pdEntries; // device allocated entries to modify
        uint32_t nEntriesToUpdate;
        uint32_t maxUpdateSz;
        uint64_t magicNumber; // our magic number
        uint64_t currentMagicNumber; // the magic number of the state we are working on
        uint32_t tableIndex;
        ModifyOpType op;
    };

public:
    static constexpr CACHE_IMPLEMENTATION_TYPE TYPE = CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE;

    ECError CalcAllocationSize(CacheAllocationSize& outAllocationSz) const
    {
        uint64_t nSets = CalcNumSets();
        
        if (nSets <= 0)
        {
            return ECERROR_NOT_ENOUGH_MEMORY_REQUESTED;
        }
        
        outAllocationSz.deviceAllocationSize = (nSets * GetDeviceSzPerSet()) * m_config.numTables;
        outAllocationSz.hostAllocationSize = (nSets * GetHostSzPerSet() + sizeof(m_pMagicNumber[0])) * m_config.numTables;

        // if our calculation are correct this shouldn't happen
        assert(outAllocationSz.deviceAllocationSize <= m_config.cacheSzInBytes);

        return ECERROR_SUCCESS;
    }

    EmbedCache(IAllocator* pAllocator, ILogger* pLogger, const CacheConfig& cfg) : EmbedCacheBase<IndexT>(pAllocator, pLogger, SET_ASSOCIATIVE), m_config(cfg), m_nSets(0), m_dpPool(nullptr), m_dpCache(nullptr), m_dpTags(nullptr),
                                                    m_hpTags(nullptr), m_pMagicNumber(nullptr)
    {

    }

    ECError LookupContextCreate(LookupContextHandle& outHandle, const PerformanceMetric* pMertics, size_t nMetrics) const override
    {
        try 
        {
            EmbedCache::CacheData* pData;
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostAllocate((void**)&pData, sizeof(EmbedCache::CacheData)), ECERROR_NOT_ENOUGH_MEMORY_REQUESTED);
            memset(pData, 0, sizeof(CacheData));
            pData->rowSizeInBytes = m_config.embedWidth;
            pData->pCache = m_dpCache;
            pData->pTags = (const int8_t*)m_dpTags;
            pData->nSets = m_nSets;
            for (size_t i = 0; i < nMetrics; i++)
            {
                if (pMertics->type == MERTIC_COUNT_MISSES)
                {
                    pData->bCountMisses = true;
                    pData->misses = pMertics[i].p_dVal;
                }
            }
            outHandle.handle = (uint64_t)pData;
            return ECERROR_SUCCESS;
        }
        catch(const ECExcption& e)
        {
            return e.m_err;
        }
    }


    ECError LookupContextDestroy(LookupContextHandle& handle) const override
    {
        EmbedCache::CacheData* p = (EmbedCache::CacheData*)handle.handle;
        handle.handle = 0;

        ECError ret = this->m_pAllocator->hostFree(p);
        return ret;
    }

    // performance 
    ECError PerformanceMetricCreate(PerformanceMetric& outMetric, PerformanceMerticTypes type) const override
    {
        try 
        {
            switch (type)
            {
            case MERTIC_COUNT_MISSES:
            {
                CHECK_ERR_AND_THROW(this->m_pAllocator->deviceAllocate((void**)&outMetric.p_dVal, sizeof(uint64_t)), ECERROR_NOT_ENOUGH_MEMORY_REQUESTED);
                outMetric.type = type;
                CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemset(outMetric.p_dVal, 0, sizeof(uint64_t)));
                return ECERROR_SUCCESS;
            }
            default:
                return ECERROR_NOT_IMPLEMENTED;
            }
        }
        catch(const ECExcption& e)
        {
            return e.m_err;
        }
        
    }
    
    ECError PerformanceMetricDestroy(PerformanceMetric& metric) const override
    {
        try
        {
            switch (metric.type)
            {
            case MERTIC_COUNT_MISSES:
            {
                ECError ret = this->m_pAllocator->deviceFree(metric.p_dVal);
                metric.p_dVal = nullptr;
                return ret;
            }
            default:
                return ECERROR_NOT_IMPLEMENTED;
            }
        }
        catch(const ECExcption& e)
        {
            return e.m_err;
        }
        
        
    }

    ECError PerformanceMetricGetValue(const PerformanceMetric& metric, uint64_t* pOutValue, cudaStream_t stream) const override
    {
        try
        {
            switch (metric.type)
            {
            case MERTIC_COUNT_MISSES:
            {
                if (!metric.p_dVal)
                {
                    throw ECExcption(ECERROR_INVALID_ARGUMENT);
                }
                CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(pOutValue, metric.p_dVal, sizeof(uint64_t), cudaMemcpyDefault, stream));
                return ECERROR_SUCCESS;
            }
            default:
                throw ECExcption(ECERROR_NOT_IMPLEMENTED);
            }
        }
        catch(const ECExcption& e)
        {
            return e.m_err;
        }
        
    }

    ECError PerformanceMetricReset(PerformanceMetric& pMetric, cudaStream_t stream) const override
    {
        try
        {
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemsetAsync(pMetric.p_dVal, 0, sizeof(uint32_t), stream));
            return ECERROR_SUCCESS;
        }
        catch(const ECExcption& e)
        {
            return e.m_err;
        }
    }

    ECError ModifyContextCreate(ModifyContextHandle& outHandle, uint32_t maxUpdateSize) const override
    {
        try
        {
            const size_t szCounterPerSet = sizeof(CounterT) * NUM_WAYS;
            const size_t szTagsPerSet = sizeof(TagT) * NUM_WAYS; // this already aligned to 16
            const size_t szWayUpdateMaskPerSet = sizeof(uint32_t);

            // check erros
            ModifyContext* pContext = nullptr;
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostAllocate((void**)&pContext, sizeof(ModifyContext)), ECERROR_NOT_ENOUGH_MEMORY_REQUESTED);
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostAllocate((void**)&pContext->pTags, m_nSets * szTagsPerSet), ECERROR_NOT_ENOUGH_MEMORY_REQUESTED);
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostAllocate((void**)&pContext->pWayUpdateMask, m_nSets * szWayUpdateMaskPerSet), ECERROR_NOT_ENOUGH_MEMORY_REQUESTED);
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostAllocate((void**)&pContext->pCounters, m_nSets * szCounterPerSet), ECERROR_NOT_ENOUGH_MEMORY_REQUESTED);
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostAllocate((void**)&pContext->pIndexMap, m_nSets * sizeof(IndexT) * NUM_WAYS), ECERROR_NOT_ENOUGH_MEMORY_REQUESTED);
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostAllocate((void**)&pContext->pEntries, maxUpdateSize * sizeof(ModifyEntry)), ECERROR_NOT_ENOUGH_MEMORY_REQUESTED);
            CHECK_ERR_AND_THROW(this->m_pAllocator->deviceAllocate((void**)&pContext->pdEntries, maxUpdateSize * sizeof(ModifyEntry)), ECERROR_NOT_ENOUGH_MEMORY_REQUESTED);
            pContext->maxUpdateSz = maxUpdateSize;
            pContext->nEntriesToUpdate = 0;
            pContext->magicNumber = (uint64_t)pContext;
            // Initialzie tags to invalid value
            std::fill(pContext->pTags, pContext->pTags + m_nSets * NUM_WAYS, -1);
            std::fill(pContext->pCounters, pContext->pCounters + m_nSets * NUM_WAYS, 0);

            outHandle.handle = (uint64_t)pContext;
            return ECERROR_SUCCESS;
        }
        catch(const ECExcption& e)
        {
            //deallocate everything
            // can i do multiple catches
            return e.m_err;
        }
    }

    ECError ModifyContextDestroy(ModifyContextHandle& outHandle) const override
    {
        try
        {
            EmbedCache::ModifyContext* pContext = (EmbedCache::ModifyContext*)outHandle.handle;
            if (!pContext)
            {
                return ECERROR_SUCCESS;
            }
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostFree(pContext->pTags), ECERROR_FREE_ERROR);
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostFree(pContext->pWayUpdateMask), ECERROR_FREE_ERROR);
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostFree(pContext->pCounters), ECERROR_FREE_ERROR);
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostFree(pContext->pIndexMap), ECERROR_FREE_ERROR);
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostFree(pContext->pEntries), ECERROR_FREE_ERROR);
            CHECK_ERR_AND_THROW(this->m_pAllocator->deviceFree(pContext->pdEntries), ECERROR_FREE_ERROR);
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostFree(pContext), ECERROR_FREE_ERROR);
            outHandle.handle = 0;
        }
        catch(const ECExcption& e)
        {
            return e.m_err;
        }
        return ECERROR_SUCCESS;
    }

    ECError ModifyContextGetProperty(const ModifyContextHandle& handle) const override
    {
        return ECERROR_NOT_IMPLEMENTED;
    }

    ECError ModifyContextSetInvalidateData(ModifyContextHandle& ModifyContextHandle, const IndexT* indices, uint32_t updateSz, uint32_t tableIndex) const override
    {
        return ECERROR_NOT_IMPLEMENTED;
    }

    ~EmbedCache()
    {
        this->m_pAllocator->deviceFree(m_dpPool);
        this->m_pAllocator->hostFree(m_hpTags);
        this->m_pAllocator->hostFree(m_pMagicNumber);
    }
    
    ECError Init() override
    {
        try
        {
            if (!this->m_pAllocator)
            {
                throw ECExcption(ECERROR_BAD_ALLOCATOR);
            }

            if (!this->m_pLogger)
            {
                throw ECExcption(ECERROR_BAD_LOGGER);
            }

            CacheAllocationSize allocSz = {0}; 
            CalcAllocationSize(allocSz);
            const size_t szTagsPerSet = CACHE_ALIGN(sizeof(TagT) * NUM_WAYS, 16); // this already aligned to 16

            m_nSets = CalcNumSets();
            if (m_nSets == 0)
            {
                throw ECExcption(ECERROR_MEMORY_ALLOCATED_TO_CACHE_TOO_SMALL);
            }
            // allocating pointer in device memory pool
            CHECK_ERR_AND_THROW(this->m_pAllocator->deviceAllocate((void**)&m_dpPool, allocSz.deviceAllocationSize), ECERROR_NOT_ENOUGH_MEMORY_REQUESTED);
            int8_t* p = m_dpPool;
            const int8_t* e = p + allocSz.deviceAllocationSize;

            // cactch bad alighments
            m_dpCache = AllocateInPool(p, e - p, m_nSets * NUM_WAYS * m_config.embedWidth * m_config.numTables, 16);
            m_dpTags = (TagT*)AllocateInPool(p, e - p, m_nSets * sizeof(TagT) * NUM_WAYS * m_config.numTables, 16);
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemset(m_dpTags, -1, m_nSets*sizeof(TagT) * NUM_WAYS* m_config.numTables));

            CHECK_ERR_AND_THROW(this->m_pAllocator->hostAllocate((void**)&m_hpTags, m_nSets * szTagsPerSet * m_config.numTables), ECERROR_NOT_ENOUGH_MEMORY_REQUESTED);
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostAllocate((void**)&m_pMagicNumber, sizeof(m_pMagicNumber[0]) * m_config.numTables), ECERROR_NOT_ENOUGH_MEMORY_REQUESTED);
            std::fill(m_hpTags, m_hpTags + m_nSets * NUM_WAYS * m_config.numTables, -1);
            std::fill(m_pMagicNumber, m_pMagicNumber + m_config.numTables, 0);
            return ECERROR_SUCCESS;
        }
        catch(const ECExcption& e)
        {
            return e.m_err;
        }
    }

    // return CacheData for Address Calcaulation each template should implement its own
    CacheData GetCacheData(const LookupContextHandle& handle) const
    {
        CacheData* cd = (CacheData*)handle.handle;
        // if cd is nullptr we have a problem, but i hate for this function to take a reference, it will create code ugly lines, please don't pass nullptr here
        return *cd;
    }

    ECError ModifyContextSetReplaceDataSparseData(ModifyContextHandle& modifyContextHandle, const IndexT* indices, uint32_t updateSz, const int8_t* data, uint32_t tableIndex, size_t stride) const override
    {
        return ModifyContextSetReplaceData(modifyContextHandle, indices, updateSz, data, tableIndex, stride, true);
    }

    ECError ModifyContextSetReplaceDataDenseData(ModifyContextHandle& modifyContextHandle, const IndexT* indices, uint32_t updateSz, const int8_t* data, uint32_t tableIndex, size_t stride) const override
    {
        return ModifyContextSetReplaceData(modifyContextHandle, indices, updateSz, data, tableIndex, stride, false);
    }
    ECError ModifyContextSetUpdateDataSparseData(ModifyContextHandle& modifyContextHandle, const IndexT* indices, uint32_t updateSz, const int8_t* data, uint32_t tableIndex, size_t stride) const override
    {
        return ModifyContextSetUpdateData(modifyContextHandle, indices, updateSz, data, tableIndex, stride, true);
    }
    ECError ModifyContextSetUpdateDataDenseData(ModifyContextHandle& modifyContextHandle, const IndexT* indices, uint32_t updateSz, const int8_t* data, uint32_t tableIndex, size_t stride) const override
    {
        return ModifyContextSetUpdateData(modifyContextHandle, indices, updateSz, data, tableIndex, stride, false);
    }

    // this function require synchronization with other worker threads needs to be atomic i.e no work that uses this cache can be called untill this function is returned and the event is waited
    // this code is non re-enternet
    ECError Modify(const ModifyContextHandle& modifyContextHandle, IECEvent* syncEvent, cudaStream_t stream) override
    {
        try
        {
            ModifyContext* pContext = (ModifyContext*)modifyContextHandle.handle;
            if (!pContext)
            {
                throw ECExcption(ECERROR_INVALID_ARGUMENT);
            }
            uint64_t tableIndex = pContext->tableIndex;
            std::lock_guard<std::mutex> modLock(m_modifyLock);
            // checking the magic number of the cache context that this modify context was based on
            // if it is different it means we looked at stale context and should not procedd with updating
            if (pContext->currentMagicNumber != m_pMagicNumber[tableIndex])
            {
                // context was calculated on bad state
                return ECERROR_STALE_MODIFY_CONTEXT;
            }

            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(pContext->pdEntries, pContext->pEntries, sizeof(ModifyEntry)*pContext->nEntriesToUpdate, cudaMemcpyHostToDevice, stream));
            auto pDstDeviceCurrTags = m_dpTags + tableIndex * m_nSets * NUM_WAYS;
            callTagInvalidateKernel<IndexT, TagT>(pContext->pdEntries, pContext->nEntriesToUpdate, pDstDeviceCurrTags, stream);

            // update doesn't change state
            if (pContext->op != MODIFY_UPDATE)
            {
                // critical section, preventing Analyze functions from copying ill posed state
                std::lock_guard<std::mutex> l(m_mutex);
                auto pSrcCurrTags = pContext->pTags;
                
                auto pDstHostCurrTags = m_hpTags + tableIndex * m_nSets * NUM_WAYS;

                //CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(pDstDeviceCurrTags, pSrcCurrTags, sizeof(TagT)*NUM_WAYS*m_nSets, cudaMemcpyHostToDevice, stream));
                // dispatch invalidateKernel
                // synchronize modifyContext Tags and cacheContext Tags
                memcpy(pDstHostCurrTags, pSrcCurrTags, sizeof(TagT)*NUM_WAYS *m_nSets);

            }

            CACHE_CUDA_ERR_CHK_AND_THROW(cudaStreamSynchronize(stream));
            CHECK_ERR_AND_THROW(syncEvent->EventRecord(), ECERROR_NOT_ENOUGH_MEMORY_REQUESTED);
            CHECK_ERR_AND_THROW(syncEvent->EventWaitStream(stream), ECERROR_NOT_ENOUGH_MEMORY_REQUESTED);
            callMemUpdateKernel<IndexT, TagT>(pContext->pdEntries, pContext->nEntriesToUpdate, m_config.embedWidth, stream);
            callTagUpdateKernel<IndexT, TagT>(pContext->pdEntries, pContext->nEntriesToUpdate, pDstDeviceCurrTags, stream);

            // update magic number
            m_pMagicNumber[tableIndex] = pContext->magicNumber;
            
            return ECERROR_SUCCESS;
        }
        catch(const ECExcption& e)
        {
            return e.m_err;
        }
    }

    ECError Lookup(const LookupContextHandle& hLookup, const IndexT* d_keys, const size_t len,
                                            int8_t* d_values, uint64_t* d_missing_index,
                                            IndexT* d_missing_keys, size_t* d_missing_len,
                                            uint32_t currTable, size_t stride, cudaStream_t stream) override
    {
        auto data = GetCacheData(hLookup);
        cudaMemsetAsync(d_missing_len, 0, sizeof(*d_missing_len), stream);
        if (len > 0)
        {
            callCacheQuery<IndexT, TagT>(d_keys, len, d_values, d_missing_index, d_missing_keys, d_missing_len, data, stream, currTable, stride);
        }
        return ECERROR_SUCCESS;
    }

    ECError Lookup(const LookupContextHandle& hLookup, const IndexT* d_keys, const size_t len,
                                            int8_t* d_values, const int8_t* d_table, uint32_t currTable, 
                                            size_t stride, cudaStream_t stream) override
    {
        auto data = GetCacheData(hLookup);
        if (len > 0)
        {
            callCacheQueryUVM<IndexT>(d_keys, len, d_values, d_table, data, stream, currTable, stride);
        }
        return ECERROR_SUCCESS;
    }

    size_t GetMaxNumEmbeddingVectorsInCache() const override
    {
        return m_nSets * NUM_WAYS;
    }
    
    CacheAllocationSize GetLookupContextSize() const override
    {
        CacheAllocationSize ret = {0};
        ret.hostAllocationSize = sizeof(CacheData);
        return ret;
    }

    CacheAllocationSize GetModifyContextSize(uint32_t maxUpdateSize) const override
    {
        constexpr size_t szTagsPerSet = CACHE_ALIGN(sizeof(TagT) * NUM_WAYS, 16); // this already aligned to 16
        constexpr size_t szCounterPerSet = sizeof(CounterT) * NUM_WAYS;
        constexpr size_t szWayUpdateMaskPerSet = sizeof(uint32_t);

        CacheAllocationSize ret = {0};
        size_t hostSize = 0;
        hostSize += sizeof(ModifyContext);
        hostSize += m_nSets * szTagsPerSet;
        hostSize += m_nSets * szWayUpdateMaskPerSet;
        hostSize += m_nSets * szCounterPerSet;
        hostSize += m_nSets * sizeof(IndexT) * NUM_WAYS;
        hostSize += maxUpdateSize * sizeof(ModifyEntry);
        ret.hostAllocationSize = hostSize;
        ret.deviceAllocationSize = maxUpdateSize * sizeof(ModifyEntry);
        return ret;
    }

protected:
    // assuming indices is host accessiable
    ECError ModifyContextSetReplaceData(ModifyContextHandle& modifyContextHandle, const IndexT* indices, uint32_t updateSz, const int8_t* data, uint32_t tableIndex, size_t stride, bool bLinearTable) const
    {
        try
        {
            ModifyContext* pContext = (ModifyContext*)modifyContextHandle.handle;
            if (!pContext)
            {
                throw ECExcption(ECERROR_INVALID_ARGUMENT);
            }
            pContext->op = MODIFY_REPLACE;
            auto pCurrTags = pContext->pTags;
            auto pCurrWayUpdateMask = pContext->pWayUpdateMask;
            auto pCurrCounters = pContext->pCounters;
            // critical section copying state to local context
            {
                std::lock_guard<std::mutex> l(m_mutex);
                if (m_pMagicNumber[tableIndex] != pContext->magicNumber)
                {
                    // someone else updated the state need copy latest version to local context
                    memcpy(pCurrTags, m_hpTags + tableIndex * m_nSets * NUM_WAYS, sizeof(TagT)*m_nSets*NUM_WAYS);
                }
                pContext->currentMagicNumber = m_pMagicNumber[tableIndex];
            }
            
            memset(pCurrWayUpdateMask, 0, sizeof(uint32_t)*m_nSets);
            for (uint32_t i = 0; i < updateSz; i++)
            {
                auto index = indices[i];
                uint32_t set = index % m_nSets;
                TagT* pSetWays = pCurrTags + set * NUM_WAYS;
                bool bFound = false;
                for (uint32_t j = 0; j < NUM_WAYS; j++)
                {
                    auto tag = pSetWays[j];
                    uint32_t key = tag * m_nSets + set;
                    if (key == index)
                    {
                        //hit
                        bFound = true;
                        pCurrCounters[set*NUM_WAYS + j]++;
                        break;
                    }
                }

                if (!bFound)
                {
                    uint32_t* pSetCounters = pCurrCounters + set*NUM_WAYS;
                    auto pSetIndexMap = pContext->pIndexMap  + set*NUM_WAYS;
                    auto candidate = std::min_element( pSetCounters, pSetCounters + NUM_WAYS );
                    auto candidateIndex = std::distance(pSetCounters, candidate);
                    *candidate = 1;
                    pSetWays[candidateIndex] = index / m_nSets;
                    pCurrWayUpdateMask[set] |= 1 << candidateIndex;
                    pSetIndexMap[candidateIndex] = bLinearTable ? index : i;
                }
            }
            int8_t* pCurrCache = m_dpCache + uint64_t(tableIndex) * m_nSets * NUM_WAYS * m_config.embedWidth;
            pContext->nEntriesToUpdate = 0;
            
            for (uint32_t i = 0; i < m_nSets; i++)
            {
                if (pCurrWayUpdateMask[i] == 0)
                {
                    continue;
                }
                auto pSetIndexMap = pContext->pIndexMap + i*NUM_WAYS;
                for (uint32_t j = 0; j < NUM_WAYS; j++)
                {
                    if ((pCurrWayUpdateMask[i] & (1 << j)) == 0)
                    {
                        continue;
                    }

                    assert(pContext->nEntriesToUpdate <= pContext->maxUpdateSz);
                    int8_t* dst = pCurrCache + ( i * NUM_WAYS + j ) * m_config.embedWidth;
                    const int8_t* src = data + pSetIndexMap[j] * stride;
                    TagT tag = *(pCurrTags + i * NUM_WAYS + j);
                    pContext->pEntries[pContext->nEntriesToUpdate++] = {src, dst, i, j, tag};
                }
            }
            pContext->tableIndex = tableIndex;
            return ECERROR_SUCCESS;
        }
        catch(const ECExcption& e)
        {
            return e.m_err;
        }
    }

    // assuming indices is host accessiable
    ECError ModifyContextSetUpdateData(ModifyContextHandle& modifyContextHandle, const IndexT* indices, uint64_t updateSz, const int8_t* data, uint32_t tableIndex, size_t stride, bool bLinearTable) const
    {
        try
        {
            ModifyContext* pContext = (ModifyContext*)modifyContextHandle.handle;
            if (!pContext)
            {
                throw ECExcption(ECERROR_INVALID_ARGUMENT);
            }
            pContext->op = MODIFY_UPDATE;
            auto pCurrTags = pContext->pTags;
            // critical section copying state to local context
            {
                std::lock_guard<std::mutex> l(m_mutex);
                if (m_pMagicNumber[tableIndex] != pContext->magicNumber)
                {
                    // someone else updated the state need copy latest version to local context
                    memcpy(pCurrTags, m_hpTags + tableIndex * m_nSets * NUM_WAYS, sizeof(TagT)*m_nSets*NUM_WAYS);
                }
                pContext->currentMagicNumber = m_pMagicNumber[tableIndex];
            }

            int8_t* pCurrCache = m_dpCache + tableIndex * m_nSets * NUM_WAYS * m_config.embedWidth;
            pContext->nEntriesToUpdate = 0;

            for (uint32_t i = 0; i < updateSz; i++)
            {
                auto index = indices[i];
                uint32_t set = index % m_nSets;
                TagT* pSetWays = pCurrTags + set * NUM_WAYS;
                for (uint32_t j = 0; j < NUM_WAYS; j++)
                {
                    auto tag = pSetWays[j];
                    uint32_t key = tag * m_nSets + set;
                    if (key == index)
                    {
                        //hit
                        assert(pContext->nEntriesToUpdate <= pContext->maxUpdateSz);
                        int8_t* dst = pCurrCache + ( set * NUM_WAYS + j ) * m_config.embedWidth;
                        const int8_t* src = data + ((bLinearTable) ? index : i) * stride;
                        TagT tag = *(pCurrTags + i * NUM_WAYS + j);
                        pContext->pEntries[pContext->nEntriesToUpdate++] = {src, dst, i, j, tag};
                        break;
                    }
                }
            }

            pContext->tableIndex = tableIndex;
            return ECERROR_SUCCESS;
        }
        catch(const ECExcption& e)
        {
            return e.m_err;
        }
    }

private:
    void Log(VERBOSITY_LEVEL verbosity, const char* format, ...) const
    {
        char buf[EC_MAX_STRING_BUF] = {0};
        va_list args;
        va_start(args, format);
        std::vsnprintf(buf, EC_MAX_STRING_BUF, format, args);
        va_end(args);
        assert(this->m_pLogger); // shouldn't be initilized if m_pLogger == nullptr
        this->m_pLogger->Log(verbosity, buf);
    }

    size_t GetDeviceSzPerSet() const
    {
        constexpr size_t szTagsPerSet = CACHE_ALIGN(sizeof(TagT) * NUM_WAYS, 16); // this already aligned to 16
        const size_t szEntryPerSet = m_config.embedWidth * NUM_WAYS;
        return szTagsPerSet + szEntryPerSet;
    }

    size_t GetHostSzPerSet() const
    {
        constexpr size_t szTagsPerSet = CACHE_ALIGN(sizeof(TagT) * NUM_WAYS, 16); // this already aligned to 16
        return szTagsPerSet;
    }

    uint64_t CalcNumSets() const
    {
        size_t szPerSet = GetDeviceSzPerSet();
        uint64_t nSets = (m_config.cacheSzInBytes / m_config.numTables ) / (szPerSet);
        return nSets;
    }


    int8_t* AllocateInPool(int8_t*& p, size_t space, size_t sz, size_t align = 0) const
    {
        int8_t* ret = nullptr;
        if (align > 0)
        {
            if (std::align(align, sz, (void*&)p, space))
            {
                ret = p;
                p += sz;
            }
            else
            {
                // handle error
            }
        }
        else
        {
            // should it be <= or < ?
            if (sz <= space)
            {
                ret = p;
                p += sz;
            }
            else
            {
                // handle error
            }
        }
        return ret;
    }

private:
    CacheConfig m_config;
    uint64_t m_nSets;
    int8_t* m_dpPool; // pointer to mem pool for cache internal buffers - one free to rule them all
    int8_t* m_dpCache; // cache storage
    TagT* m_dpTags; // device tags
    TagT* m_hpTags; // host copy of tags
    uint64_t* m_pMagicNumber; // per table magic number of last updated context
    mutable std::mutex m_mutex; // this should be per table
    mutable std::mutex m_modifyLock;

public:
    static const uint32_t NUM_WAYS = 8;
};
}