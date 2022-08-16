/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cuco/dynamic_map.cuh>
#include <cuco/initializer.cuh>

namespace det {

template <typename KeyType, typename ElementType>
class DynamicEmbeddingTable {
  const size_t initial_capacity_;
  const size_t num_classes_;
  std::vector<size_t> dimension_per_class_;
  curandState *curand_states_;
  std::vector<std::unique_ptr<cuco::dynamic_map<KeyType, ElementType, cuco::initializer>>> maps_;
  std::vector<cudaStream_t> stream_per_class_;
  std::vector<cudaEvent_t> event_per_class_;
  cudaEvent_t primary_event_;

  void reserve(size_t n);

 public:
  DynamicEmbeddingTable(size_t num_classes, size_t const *dimension_per_class,
                        size_t initial_capacity_for_class = 1048576);
  ~DynamicEmbeddingTable() {}

  void initialize(cudaStream_t stream = 0);
  void uninitialize(cudaStream_t stream = 0);

  void lookup(KeyType const *keys, ElementType *elements, size_t num_keys, size_t const *id_spaces,
              size_t const *id_space_offsets, size_t num_id_spaces, cudaStream_t stream = 0);
  void lookup_unsafe(KeyType const *keys, ElementType **elements, size_t num_keys,
                     size_t const *id_spaces, size_t const *id_space_offsets, size_t num_id_spaces,
                     cudaStream_t stream = 0);
  void scatter_add(KeyType const *keys, ElementType const *elements, size_t num_keys,
                   size_t const *id_spaces, size_t const *id_space_offsets, size_t num_id_spaces,
                   cudaStream_t stream = 0);
  void remove(KeyType const *keys, size_t num_keys, size_t const *id_spaces,
              size_t const *id_space_offsets, size_t num_id_spaces, cudaStream_t stream = 0);
  void eXport(size_t class_index, KeyType *keys, ElementType *values, size_t num_keys,
              cudaStream_t stream = 0);
  void clear(cudaStream_t stream = 0);

  size_t size() const;
  size_t capacity() const;
};

}  // namespace det
