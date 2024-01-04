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

// need to include this for cudaCalls can we move this to another place
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace ecache {
#define EC_MAX_STRING_BUF 1024

// add name space 
enum CACHE_IMPLEMENTATION_TYPE
{
    API,
    SET_ASSOCIATIVE,

    NUM_IMPLEMENTATION_TYPES,
};

typedef uint32_t ECError;

#define ECERROR_SUCCESS 0x0
#define ECERROR_INVALID_ARGUMENT 0x1 
#define ECERROR_NOT_ENOUGH_MEMORY_REQUESTED 0x2
#define ECERROR_CUDA_ERROR 0x3
#define ECERROR_FREE_ERROR 0x4
#define ECERROR_NOT_IMPLEMENTED 0x5
#define ECERROR_STALE_MODIFY_CONTEXT 0x6 // returned when a ModifyContext was calculated on an old state and no modify operation can be done on it
#define ECERROR_BAD_ALLOCATOR 0x7 // return during init function if allocator is nullptr
#define ECERROR_BAD_LOGGER 0x8 // return during init function if logger is nullptr
#define ECERROR_MEMORY_ALLOCATED_TO_CACHE_TOO_SMALL 0x9 // return in init if memory allowed for cache is too small to initialize internal buffers

class ECExcption: public std::exception
{
public:
    ECExcption(ECError err) : m_err(err) {}
    virtual const char* what() const throw()
    {
        return "an err occured";
    }
    ECError m_err;
};

#ifndef CACHE_CUDA_ERR_CHK_AND_THROW
#define CACHE_CUDA_ERR_CHK_AND_THROW(ans) { if ((ans) != cudaSuccess) {throw ECExcption(ECERROR_CUDA_ERROR);}}
#endif

#ifndef CHECK_ERR_AND_THROW
#define CHECK_ERR_AND_THROW(ans, err_throw) { if ((ans) != ECERROR_SUCCESS) {throw ECExcption(err_throw);}}
#endif


// should this be a template function avoiding conflicts
#define CACHE_ALIGN(x,a)              CACHE__ALIGN_MASK(x,(a)-1)
#define CACHE__ALIGN_MASK(x,mask)    (((x)+(mask))&~(mask))

class IAllocator
{
public:
    virtual ECError deviceAllocate(void** ptr, size_t sz) noexcept = 0;
    virtual ECError deviceFree(void* ptr) = 0;
    virtual ECError hostAllocate(void** ptr, size_t sz) noexcept = 0;
    virtual ECError hostFree(void* ptr) = 0;
};

enum VERBOSITY_LEVEL
{
    DEBUG = 0,
    WARN = 1,
    INFO = 2,
    ERROR = 3,


    NONE
};

class ILogger
{
public:
    virtual void Log(uint32_t verbosity, const std::string msg) = 0;
};

class IECEvent
{
public:
    virtual ECError EventRecord() = 0;
    virtual ECError EventWaitStream(cudaStream_t stream) = 0;
};

class DefaultECEvent : public IECEvent
{
public:
    DefaultECEvent(const std::vector<cudaStream_t>& streams) : m_streams(streams)
    {
        for (size_t i = 0; i < m_streams.size(); i++)
        {
            cudaEvent_t nEvent;
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaEventCreate(&nEvent));
            m_events.push_back(nEvent);
        }
    }
    ~DefaultECEvent()
    {
        for (cudaEvent_t event : m_events)
        {
            cudaEventDestroy(event);
        }
    }

    ECError EventRecord() override
    {

        for (size_t i = 0; i < m_streams.size(); i++)
        {
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaEventRecord(m_events[i], m_streams[i]));
        } 
        return ECERROR_SUCCESS;
    }

    ECError EventWaitStream(cudaStream_t stream) override
    {

        for (size_t i = 0; i < m_events.size(); i++)
        {
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaStreamWaitEvent(stream, m_events[i]));
        } 
        return ECERROR_SUCCESS;
    }

private:
    std::vector<cudaStream_t> m_streams;
    std::vector<cudaEvent_t> m_events;
};

class DefaultLogger : public ILogger
{
public:
    DefaultLogger() {}
    ~DefaultLogger() {}
    void Log(uint32_t verbosity, const std::string msg) override
    {
    }
};

class DefaultAllocator : public IAllocator
{
public:
    ECError deviceAllocate(void** ptr, size_t sz) noexcept
    {
        auto ret = cudaMalloc(ptr, sz);
        if (ret != cudaSuccess)
        {
            return ECERROR_NOT_ENOUGH_MEMORY_REQUESTED;
        }
        else
        {
            return ECERROR_SUCCESS;
        }
    }

    ECError deviceFree(void* ptr)
    {
        auto ret = cudaFree(ptr);
        if (ret != cudaSuccess)
        {
            return ECERROR_FREE_ERROR;
        }
        else
        {
            return ECERROR_SUCCESS;
        }
    }
    ECError hostAllocate(void** ptr, size_t sz) noexcept
    {
        auto ret = cudaMallocHost(ptr, sz);
        if (ret != cudaSuccess)
        {
            return ECERROR_NOT_ENOUGH_MEMORY_REQUESTED;
        }
        else
        {
            return ECERROR_SUCCESS;
        }
    }
    ECError hostFree(void* ptr)
    {
        auto ret = cudaFreeHost(ptr);
        if (ret != cudaSuccess)
        {
            return ECERROR_FREE_ERROR;
        }
        else
        {
            return ECERROR_SUCCESS;
        }
    }
};

enum PerformanceMerticTypes
{
    MERTIC_COUNT_MISSES,
    
    NUM_PERFORMANCE_METRIC_TYPES,
};

struct CacheAllocationSize
{
    size_t deviceAllocationSize;
    size_t hostAllocationSize;
};

struct UpdateContextConfig
{
    bool bLinearTable = true;
};

struct ModifyContextHandle
{
    uint64_t handle;
};

struct LookupContextHandle
{
    uint64_t handle;
};

struct PerformanceMetric
{
    uint64_t* p_dVal;
    PerformanceMerticTypes type;
};

template<typename IndexT>
class EmbedCacheBase
{
public:
    EmbedCacheBase(IAllocator* pAllocator, ILogger* pLogger, CACHE_IMPLEMENTATION_TYPE type) : m_pAllocator(pAllocator), m_pLogger(pLogger), m_type(type){}

    struct CacheData
    {

    };

    CACHE_IMPLEMENTATION_TYPE GetType() const { return m_type; }

    // Init internal structures set buffers to memory pool
    virtual ECError Init() = 0;

    // readContext
    // cache Accessors
    // return CacheData for Address Calcaulation each template should implement its own
    virtual ECError LookupContextCreate(LookupContextHandle& outHandle, const PerformanceMetric* pMertics, size_t nMetrics) const = 0;
    virtual ECError LookupContextDestroy(LookupContextHandle& handle) const = 0;

    CacheData GetCacheData(const LookupContextHandle* pHandle) const;

    // return dense representation of the indices
    virtual ECError Lookup(const LookupContextHandle& hLookup, const IndexT* d_keys, const size_t len,
                                            int8_t* d_values, uint64_t* d_missing_index,
                                            IndexT* d_missing_keys, size_t* d_missing_len,
                                            uint32_t currTable, size_t stride, cudaStream_t stream) = 0;

    virtual ECError Lookup(const LookupContextHandle& hLookup, const IndexT* d_keys, const size_t len,
                                            int8_t* d_values, const int8_t* d_table, uint32_t currTable, size_t stride, cudaStream_t stream) = 0;

    // performance 
    virtual ECError PerformanceMetricCreate(PerformanceMetric& outMetric, PerformanceMerticTypes type) const = 0;
    virtual ECError PerformanceMetricDestroy(PerformanceMetric& metric) const = 0;
    virtual ECError PerformanceMetricGetValue(const PerformanceMetric& metric, uint64_t* pOutValue, cudaStream_t stream) const = 0;
    virtual ECError PerformanceMetricReset(PerformanceMetric& metric, cudaStream_t stream) const = 0;

    virtual ECError ModifyContextCreate(ModifyContextHandle& outHandle, uint32_t maxUpdatesize) const = 0;
    virtual ECError ModifyContextDestroy(ModifyContextHandle& handle) const = 0;

    // host only block operation, pre process on host
    // create invalidate state
    // create update state
    // create replace state
    virtual ECError ModifyContextSetReplaceDataSparseData(ModifyContextHandle& modifyContextHandle, const IndexT* indices, uint32_t updateSz, const int8_t* data, uint32_t tableIndex, size_t stride) const  = 0;
    virtual ECError ModifyContextSetReplaceDataDenseData(ModifyContextHandle& modifyContextHandle, const IndexT* indices, uint32_t updateSz, const int8_t* data, uint32_t tableIndex, size_t stride) const  = 0;
    virtual ECError ModifyContextSetUpdateDataSparseData(ModifyContextHandle& modifyContextHandle, const IndexT* indices, uint32_t updateSz, const int8_t* data, uint32_t tableIndex, size_t stride) const = 0;
    virtual ECError ModifyContextSetUpdateDataDenseData(ModifyContextHandle& modifyContextHandle, const IndexT* indices, uint32_t updateSz, const int8_t* data, uint32_t tableIndex, size_t stride) const = 0;
    virtual ECError ModifyContextSetInvalidateData(ModifyContextHandle& modifyContextHandle, const IndexT* indices, uint32_t updateSz, uint32_t tableIndex) const = 0;

    // includes device operation require synchornization
    // commit state
    virtual ECError Modify(const ModifyContextHandle& modifyContextHandle, IECEvent* syncEvent, cudaStream_t stream) = 0;

    // should replace
    virtual ECError ModifyContextGetProperty(const ModifyContextHandle& handle) const = 0;

    // inspectors
    virtual size_t GetMaxNumEmbeddingVectorsInCache() const = 0;

    virtual CacheAllocationSize GetLookupContextSize() const = 0;
    virtual CacheAllocationSize GetModifyContextSize(uint32_t maxUpdateSize) const = 0;

protected:
    CACHE_IMPLEMENTATION_TYPE m_type;
    IAllocator* m_pAllocator;
    mutable ILogger* m_pLogger;
};
}