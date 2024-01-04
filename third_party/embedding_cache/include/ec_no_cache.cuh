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
#include "ec_no_cache.h"
namespace ecache {
template<typename IndexT>
class AddressFunctor<IndexT, typename ECNoCache<IndexT>::CacheData>
{
public:
    static __device__ inline uint64_t GetAddress(IndexT laneIdx, const int8_t* pTable, uint32_t currTable, const typename ECNoCache<IndexT>::CacheData data)
    {
        if (data.bCountMisses)
        {
            atomicAdd((unsigned long long*)data.misses, 1llu);
        }

        return (uint64_t)pTable + laneIdx * (uint64_t)data.rowSizeInBytes;
    }
};
}
