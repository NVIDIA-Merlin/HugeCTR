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
#include "embed_cache.cuh"
#include "ec_set_associative.h"
namespace ecache {
template<typename IndexT, typename TagT>
static __device__ inline uint32_t EmbedCacheGetWayMask(IndexT laneIdx, uint32_t currTable, const typename EmbedCache<IndexT, TagT>::CacheData data)
{
    uint32_t cacheOffset = currTable * data.nSets * EmbedCache<IndexT, TagT>::NUM_WAYS;
    uint32_t setIdx = laneIdx % data.nSets;
    const TagT* pWays = (const TagT*)(data.pTags + (cacheOffset + setIdx * EmbedCache<IndexT, TagT>::NUM_WAYS) * sizeof(TagT));
    uint32_t out = 0;
    for (uint32_t i = 0; i < EmbedCache<IndexT, TagT>::NUM_WAYS; i++)
    {
        uint32_t way = pWays[i];
        uint32_t key = way * data.nSets + setIdx;
        uint32_t b = key == laneIdx;
        out |= (b << i); 
    }

    return out;
}

template<typename IndexT, typename TagT>
static __device__ inline uint64_t EmbedCacheGetAddress(IndexT laneIdx, const int8_t* pTable, uint32_t currTable, const typename EmbedCache<IndexT, TagT>::CacheData data)
{
    uint32_t cacheOffset = currTable * data.nSets * EmbedCache<IndexT, TagT>::NUM_WAYS;
    uint32_t setIdx = laneIdx % data.nSets;
    uint32_t out = EmbedCacheGetWayMask<IndexT, TagT>(laneIdx, currTable, data);
    uint32_t way = __ffs(out) - 1;
    uint64_t lanePtr = (out == 0) ? (uint64_t)pTable + (laneIdx)*(uint64_t)data.rowSizeInBytes : 
        (uint64_t)data.pCache + (cacheOffset + setIdx * EmbedCache<IndexT, TagT>::NUM_WAYS + way)*(uint64_t)data.rowSizeInBytes;

    if (out == 0 && data.bCountMisses)
    {
        atomicAdd((unsigned long long*)data.misses, 1);
    }

    return lanePtr;
}

template<typename IndexT>
class AddressFunctor<IndexT, typename EmbedCache<IndexT, uint16_t>::CacheData>
{
public:
    static __device__ inline uint64_t GetAddress(IndexT laneIdx, const int8_t* pTable, uint32_t currTable, const typename EmbedCache<IndexT, uint16_t>::CacheData data)
    {
        uint32_t cacheOffset = currTable * data.nSets * EmbedCache<IndexT, uint16_t>::NUM_WAYS;
        uint32_t setIdx = laneIdx % data.nSets;
        uint4 ways = *(uint4*)(data.pTags + (cacheOffset + setIdx * EmbedCache<IndexT, uint16_t>::NUM_WAYS) * sizeof(uint16_t));
        uint16_t* pWays = (uint16_t*)(&ways);
        uint32_t out = 0;
        for (uint32_t i = 0; i < EmbedCache<IndexT, uint16_t>::NUM_WAYS; i++)
        {
            uint32_t way = pWays[i];
            uint32_t key = way * data.nSets + setIdx;
            uint32_t b = key == laneIdx;
            out |= (b << i); 
        }
        uint32_t way = __ffs(out) - 1;
        uint64_t lanePtr = (out == 0) ? (uint64_t)pTable + (laneIdx)*(uint64_t)data.rowSizeInBytes : 
            (uint64_t)data.pCache + (cacheOffset + setIdx * EmbedCache<IndexT, uint16_t>::NUM_WAYS + way)*(uint64_t)data.rowSizeInBytes;

        if (out == 0 && data.bCountMisses)
        {
            atomicAdd((unsigned long long*)data.misses, 1llu);
        }

        return lanePtr;
    }
};




template<typename IndexT>
class AddressFunctor<IndexT, typename EmbedCache<IndexT, uint32_t>::CacheData>
{
public:
    static __device__ inline uint64_t GetAddress(IndexT laneIdx, const int8_t* pTable, uint32_t currTable, const typename EmbedCache<IndexT, uint32_t>::CacheData data)
    {
        return EmbedCacheGetAddress<IndexT, uint32_t>(laneIdx, pTable, currTable, data);
    }
};

template<typename IndexT>
class AddressFunctor<IndexT, typename EmbedCache<IndexT, uint64_t>::CacheData>
{
public:
    static __device__ inline uint64_t GetAddress(IndexT laneIdx, const int8_t* pTable, uint32_t currTable, const typename EmbedCache<IndexT, uint64_t>::CacheData data)
    {
        return EmbedCacheGetAddress<IndexT, uint64_t>(laneIdx, pTable, currTable, data);
    }
};

template<typename IndexT>
class AddressFunctor<IndexT, typename EmbedCache<IndexT, int64_t>::CacheData>
{
public:
    static __device__ inline uint64_t GetAddress(IndexT laneIdx, const int8_t* pTable, uint32_t currTable, const typename EmbedCache<IndexT, int64_t>::CacheData data)
    {
        return EmbedCacheGetAddress<IndexT, int64_t>(laneIdx, pTable, currTable, data);
    }
};

template<typename IndexT>
class AddressFunctor<IndexT, typename EmbedCache<IndexT, int32_t>::CacheData>
{
public:
    static __device__ inline uint64_t GetAddress(IndexT laneIdx, const int8_t* pTable, uint32_t currTable, const typename EmbedCache<IndexT, int32_t>::CacheData data)
    {
        return EmbedCacheGetAddress<IndexT, int32_t>(laneIdx, pTable, currTable, data);
    }
};

template<typename IndexT, typename TagT, typename DataType>
__global__ void MemUpdateKernel(typename EmbedCache<IndexT, TagT>::ModifyEntry* pEntries, uint32_t sz)
{
    typename EmbedCache<IndexT, TagT>::ModifyEntry e = pEntries[blockIdx.x];
    MemcpyWarp<32, DataType>(e.pDst, e.pSrc, sz);
}

template<typename IndexT, typename TagT>
__global__ void InvalidateTagKernel(typename EmbedCache<IndexT, TagT>::ModifyEntry* pEntries, uint32_t nEntries, TagT* pTags)
{
    auto tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < nEntries)
    {
        typename EmbedCache<IndexT, TagT>::ModifyEntry e = pEntries[tid];
        TagT* pToModTag = pTags + e.set * EmbedCache<IndexT, TagT>::NUM_WAYS + e.way;
        *pToModTag = (TagT)-1;
    }
}

template<typename IndexT, typename TagT>
__global__ void TagUpdateKernel(typename EmbedCache<IndexT, TagT>::ModifyEntry* pEntries, uint32_t nEntries, TagT* pTags)
{
    auto tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < nEntries)
    {
        typename EmbedCache<IndexT, TagT>::ModifyEntry e = pEntries[tid];
        TagT* pToModTag = pTags + e.set * EmbedCache<IndexT, TagT>::NUM_WAYS + e.way;
        *pToModTag = e.tag;
    }
}

template<typename IndexT, typename TagT, uint32_t SUBWARP_WIDTH, typename DataType>
__global__ void Query(const IndexT* d_keys, const size_t len,
    int8_t* d_values, uint64_t* d_missing_index,
    IndexT* d_missing_keys, size_t* d_missing_len,
    typename EmbedCache<IndexT, TagT>::CacheData data, uint32_t currTable, size_t stride)
{
    const uint32_t blockDims = blockDim.x * blockDim.y;
    uint32_t block_ptr = blockIdx.x * blockDims;
    uint32_t tid = block_ptr + threadIdx.x; // each tid search for one index, and then we do a "transpose" and copy them out if needed
    const uint32_t subwarp_idx = threadIdx.x / SUBWARP_WIDTH;
    const uint32_t subwarp_ptr = block_ptr + subwarp_idx * SUBWARP_WIDTH;
    const uint32_t intra_subwarp_idx = threadIdx.x % SUBWARP_WIDTH;

    uint64_t laneptr;
    if (tid >= len)
    {
        laneptr = 0;
    }
    else
    {
        IndexT laneIdx = d_keys[tid];
        uint32_t cacheOffset = currTable * data.nSets * EmbedCache<IndexT, TagT>::NUM_WAYS;
        uint32_t setIdx = laneIdx % data.nSets;
        uint32_t laneout = EmbedCacheGetWayMask<IndexT, TagT>(laneIdx, currTable, data);
        uint32_t laneway = __ffs(laneout) - 1;
        if (laneout == 0)
        {
            unsigned long long old = atomicAdd((unsigned long long*)d_missing_len, 1llu);
            d_missing_index[old] = tid;
            d_missing_keys[old] = laneIdx;
            laneptr = 0;
        }
        else
        {
            auto way = laneway;
            laneptr = (uint64_t)(data.pCache + (cacheOffset + setIdx * EmbedCache<IndexT, TagT>::NUM_WAYS + way)*(uint64_t)data.rowSizeInBytes);
        }
    }

    for (uint32_t s = 0; s < SUBWARP_WIDTH; s++)
    {
        const uint32_t ELEMENT_SIZE = sizeof(DataType);
        uint64_t src_ptr = __shfl_sync(0xffffffff, laneptr, s, SUBWARP_WIDTH);
        if (src_ptr == 0)
        {
            continue;
        }
        for (uint32_t k = 0; k < data.rowSizeInBytes; k += ELEMENT_SIZE*SUBWARP_WIDTH)
        {
            uint32_t offset = k + intra_subwarp_idx * ELEMENT_SIZE;
            if (offset < data.rowSizeInBytes)
            {
                DataType d = *(DataType*)(src_ptr + offset);
                DataType* dst_ptr = (DataType*)(d_values + (subwarp_ptr + s) * stride + offset);
                *dst_ptr = d;
            }
        }
        
    }
}

// need to have an argument explicity depend on IndexT type or the compiler gets confused
template<typename IndexT, typename TagT>
void callTagInvalidateKernel(typename EmbedCache<IndexT, TagT>::ModifyEntry* pEntries, uint32_t nEntries, TagT* pTags, cudaStream_t stream)
{
    dim3 gridSize((nEntries + 32 - 1)/32,1);
    dim3 blockSize(32, 1); 
    InvalidateTagKernel<IndexT, TagT><<<gridSize, blockSize, 0, stream>>>(pEntries, nEntries, pTags);
}

// need to have an argument explicity depend on IndexT type or the compiler gets confused
template<typename IndexT, typename TagT>
void callMemUpdateKernel(typename EmbedCache<IndexT, TagT>::ModifyEntry* pEntries, uint32_t nEntries, uint32_t rowSizeInBytes, cudaStream_t stream)
{
    dim3 gridSize(nEntries,1);
    dim3 blockSize(32, 1); 
    if (rowSizeInBytes % sizeof(uint4) == 0)
    {
        MemUpdateKernel<IndexT, TagT, uint4><<<gridSize, blockSize, 0, stream>>>(pEntries, rowSizeInBytes);
    }
    else if (rowSizeInBytes % sizeof(uint32_t) == 0)
    {
        MemUpdateKernel<IndexT, TagT, uint32_t><<<gridSize, blockSize, 0, stream>>>(pEntries, rowSizeInBytes);
    }
    else
    {
        MemUpdateKernel<IndexT, TagT, int8_t><<<gridSize, blockSize, 0, stream>>>(pEntries, rowSizeInBytes);
    }
}

// need to have an argument explicity depend on IndexT type or the compiler gets confused
template<typename IndexT, typename TagT>
void callTagUpdateKernel(typename EmbedCache<IndexT, TagT>::ModifyEntry* pEntries, uint32_t nEntries, TagT* pTags, cudaStream_t stream)
{
    dim3 gridSize((nEntries + 32 - 1)/32,1);
    dim3 blockSize(32, 1); 
    TagUpdateKernel<IndexT, TagT><<<gridSize, blockSize, 0, stream>>>(pEntries, nEntries, pTags);
}

template<typename IndexT, typename TagT>
void callCacheQuery(const IndexT* d_keys, const size_t len,
    int8_t* d_values, uint64_t* d_missing_index,
    IndexT* d_missing_keys, size_t* d_missing_len,
    typename EmbedCache<IndexT, TagT>::CacheData data,
    cudaStream_t stream, uint32_t currTable, size_t stride)
{
    const uint32_t blockX = 32;
    const uint32_t blockY = 4;
    const uint32_t blockSize = blockX * blockY;
    const uint32_t nBlock = len / blockSize + std::min(len % blockSize, (size_t)1);
    dim3 gridDims(nBlock);
    dim3 blockDims(blockSize);
    if (data.rowSizeInBytes % (sizeof(uint4)*32) == 0)
    {
        Query<IndexT, TagT, 32, uint4><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_missing_index, d_missing_keys, d_missing_len, data, currTable, stride);
    }
    else if (data.rowSizeInBytes % (sizeof(uint32_t)*32) == 0)
    {
        Query<IndexT, TagT, 32, uint32_t><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_missing_index, d_missing_keys, d_missing_len, data, currTable, stride);
    }
    else if (data.rowSizeInBytes % (sizeof(uint4)*4) == 0)
    {
        Query<IndexT, TagT, 4, uint4><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_missing_index, d_missing_keys, d_missing_len, data, currTable, stride);
    }
    else
    {
        Query<IndexT, TagT, blockX, uint8_t><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_missing_index, d_missing_keys, d_missing_len, data, currTable, stride);
    }
}
}
