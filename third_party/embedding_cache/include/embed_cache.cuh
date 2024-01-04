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

#include <cuda_runtime.h>
#include "embed_cache.h"
namespace ecache {

template<typename IndexT, typename CacheDataT>
class AddressFunctor
{
public:
    static __device__ inline uint64_t GetAddress(IndexT index, const int8_t* pTable, uint32_t currTable, const CacheDataT data)
    {
        return 0;
    }
};

template<uint32_t SUBWARP_WIDTH, typename DataType>
__device__ void MemcpyWarp(int8_t* pDst, __restrict__ const int8_t* pSrc, uint32_t sz)
{
    const uint32_t ELEMENT_SIZE = sizeof(DataType);
    for (uint32_t k = 0; k < sz; k += ELEMENT_SIZE*SUBWARP_WIDTH)
    {
        uint32_t offset = k + threadIdx.x * ELEMENT_SIZE;
        if (offset < sz)
        {
            DataType d = *(DataType*)((int8_t*)pSrc + offset);
            DataType* dst_ptr = (DataType*)((int8_t*)pDst + offset);
            *dst_ptr = d;
        }
    }
}

template<typename IndexT, uint32_t SUBWARP_WIDTH, uint32_t BLOCK_Y, typename DataType, typename CacheDataT>
__global__ void QueryUVM(const IndexT* d_keys, const size_t len,
    int8_t* d_values, const __restrict__ int8_t* d_table,
    CacheDataT data, uint32_t currTable, size_t stride)
{
    uint32_t tid_batch = blockIdx.x * SUBWARP_WIDTH * BLOCK_Y + threadIdx.y * SUBWARP_WIDTH;
    uint32_t tid = tid_batch + threadIdx.x; // each tid search for one index, and then we do a "transpose" and copy them out if needed
    uint64_t laneptr;
    if (tid >= len)
    {
        laneptr = 0;
    }
    else
    {
        IndexT laneIdx = d_keys[tid];
        laneptr = AddressFunctor<IndexT, CacheDataT>::GetAddress(laneIdx, d_table, currTable, data);
    }
    #pragma unroll
    for (uint32_t s = 0; s < SUBWARP_WIDTH; s++)
    {
        const uint32_t ELEMENT_SIZE = sizeof(DataType);
        uint64_t src_ptr = __shfl_sync(__activemask(), laneptr, s, SUBWARP_WIDTH);
        if (src_ptr == 0)
        {
            continue;
        }
        for (uint32_t k = 0; k < data.rowSizeInBytes; k += ELEMENT_SIZE*SUBWARP_WIDTH)
        {
            uint32_t offset = k + threadIdx.x * ELEMENT_SIZE;
            if (offset < data.rowSizeInBytes)
            {
                DataType d = __ldg((DataType*)(src_ptr + offset));
                DataType* dst_ptr = (DataType*)(d_values + (tid_batch + s) * stride + offset);
                __stcs(dst_ptr, d);
            }
        }
        
    }
}

template<typename IndexT, typename CacheDataT>
void callCacheQueryUVM(const IndexT* d_keys, const size_t len,
    int8_t* d_values, const int8_t* d_table,
    CacheDataT data, cudaStream_t stream, uint32_t currTable, size_t stride)
{
    const uint32_t blockX = 32;
    const uint32_t blockY = 4;
    const uint32_t blockSize = blockX * blockY;
    const uint32_t nBlock = len / blockSize + std::min(len % blockSize, (size_t)1);
    dim3 gridDims(nBlock);
    dim3 blockDims(blockX, blockY);
    if (data.rowSizeInBytes % sizeof(uint4) == 0)
    {
        QueryUVM<IndexT, blockX, blockY, uint4><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_table, data, currTable, stride);
    }
    else if (data.rowSizeInBytes % sizeof(uint32_t) == 0)
    {
        QueryUVM<IndexT, blockX, blockY, uint32_t><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_table, data, currTable, stride);
    }
    else
    {
        QueryUVM<IndexT, blockX, blockY, uint8_t><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_table, data, currTable, stride);
    }
}

}