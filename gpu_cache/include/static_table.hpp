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

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <nv_util.h>

#include <cstdio>
#include <limits>
#include <static_hash_table.hpp>

namespace gpu_cache {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename key_type, typename value_type, typename out_value_type>
class static_table {
 public:
  // Ctor
  static_table(const size_t table_size, const size_t embedding_vec_size,
               const out_value_type default_value = 0, bool enable_pagelock = false);

  // Dtor
  ~static_table(){};

  // Query API, i.e. A single read from the cache
  void Query(const key_type* d_keys, const size_t len, out_value_type* d_values,
             cudaStream_t stream);

  // Replace API, i.e. Follow the Query API to update the content of the cache to Most Recent
  void Init(const key_type* d_keys, const size_t len, const value_type* d_values,
            cudaStream_t stream);

  void Add(const key_type* d_keys, const size_t len, const value_type* d_values,
           const float* d_quant_scales, cudaStream_t stream = 0);

  void Clear(cudaStream_t stream);

 private:
  StaticHashTable<key_type, value_type, out_value_type> static_hash_table_;
  // Embedding vector size
  size_t embedding_vec_size_;
  size_t table_size_;
  out_value_type default_value_ = 0;
};

}  // namespace gpu_cache
