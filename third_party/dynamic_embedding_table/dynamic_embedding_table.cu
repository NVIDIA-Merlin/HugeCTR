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
#include <random>

#include "dynamic_embedding_table.hpp"

namespace {
template <typename Seed>
__global__ void setup_kernel(Seed seed, curandState *state) {
  auto grid = cooperative_groups::this_grid();
  curand_init(seed, grid.thread_rank(), 0, &state[grid.thread_rank()]);
}
}  // namespace

namespace det {

template <typename KeyType, typename ValueType>
DynamicEmbeddingTable<KeyType, ValueType>::DynamicEmbeddingTable(size_t num_classes,
                                                                 size_t const *dimension_per_class,
                                                                 std::string initializer,
                                                                 size_t initial_capacity)
    : initial_capacity_(initial_capacity),
      num_classes_(num_classes),
      dimension_per_class_(dimension_per_class, dimension_per_class + num_classes),
      initializer_(initializer) {}

template <typename KeyType, typename ValueType>
void DynamicEmbeddingTable<KeyType, ValueType>::initialize(cudaStream_t stream) {
  std::random_device rd;
  auto seed = rd();

  int device;
  CUCO_CUDA_TRY(cudaGetDevice(&device));
  cudaDeviceProp deviceProp;
  CUCO_CUDA_TRY(cudaGetDeviceProperties(&deviceProp, device));
  CUCO_CUDA_TRY(cudaMallocAsync(
      &curand_states_, sizeof(curandState) * deviceProp.multiProcessorCount * 2048, stream));

  setup_kernel<<<deviceProp.multiProcessorCount * 2, 1024>>>(seed, curand_states_);

  for (size_t i = 0; i < num_classes_; i++) {
    cudaStream_t stream;
    CUCO_CUDA_TRY(cudaStreamCreate(&stream));
    stream_per_class_.push_back(stream);
  }

  for (size_t i = 0; i < num_classes_; i++) {
    cudaEvent_t event;
    CUCO_CUDA_TRY(cudaEventCreate(&event));
    event_per_class_.push_back(event);
  }

  CUCO_CUDA_TRY(cudaEventCreate(&primary_event_));

  for (size_t i = 0; i < num_classes_; i++) {
    bool use_const_initializer = false;
    float initial_val = 0.0;
    if (initializer_ != "") {
      if (initializer_ == "ones") {
        use_const_initializer = true;
        initial_val = 1.0;
      } else if (initializer_ == "zeros") {
        use_const_initializer = true;
        initial_val = 0.0;
      } else {
        try {
          initial_val = std::stof(initializer_);
          use_const_initializer = true;
        } catch (std::invalid_argument &err) {
          std::cout << "Using random initializer." << std::endl;
          use_const_initializer = false;
          initial_val = 0.0;
        }
      }
    }
    auto map = std::make_unique<cuco::dynamic_map<KeyType, ValueType, cuco::initializer>>(
        dimension_per_class_[i], initial_capacity_,
        cuco::initializer(curand_states_, use_const_initializer, initial_val));
    map->initialize(stream);
    maps_.push_back(std::move(map));
  }

  CUCO_CUDA_TRY(cudaGetLastError());
}

template <typename KeyType, typename ElementType>
void DynamicEmbeddingTable<KeyType, ElementType>::uninitialize(cudaStream_t stream) {
  for (size_t i = 0; i < num_classes_; i++) {
    maps_[i]->uninitialize(stream);
  }

  CUCO_CUDA_TRY(cudaEventDestroy(primary_event_));

  for (size_t i = 1; i < num_classes_; i++) {
    CUCO_CUDA_TRY(cudaEventDestroy(event_per_class_[i]));
  }

  for (size_t i = 1; i < num_classes_; i++) {
    CUCO_CUDA_TRY(cudaStreamDestroy(stream_per_class_[i]));
  }

  CUCO_CUDA_TRY(cudaFreeAsync(curand_states_, stream));
  CUCO_CUDA_TRY(cudaGetLastError());
}

template <typename KeyType, typename ElementType>
void DynamicEmbeddingTable<KeyType, ElementType>::lookup(
    KeyType const *keys, ElementType *output_elements, size_t num_keys, size_t const *id_spaces,
    size_t const *id_space_offsets, size_t num_id_spaces, cudaStream_t stream) {
  CUCO_CUDA_TRY(cudaEventRecord(primary_event_, stream));

  for (size_t i = 0; i < num_id_spaces; ++i) {
    size_t id_space = id_spaces[i];
    size_t id_space_offset = id_space_offsets[i];
    size_t id_space_size = id_space_offsets[i + 1] - id_space_offset;
    assert(id_space < num_classes_);
    assert(id_space_offset + id_space_size <= num_keys);

    CUCO_CUDA_TRY(cudaStreamWaitEvent(stream_per_class_[id_space], primary_event_));
    maps_[i]->lookup(keys + id_space_offset,
                     output_elements + id_space_offset * dimension_per_class_[id_space],
                     id_space_size, stream_per_class_[id_space]);
    CUCO_CUDA_TRY(cudaEventRecord(event_per_class_[id_space], stream_per_class_[id_space]));
  }

  for (size_t i = 0; i < num_classes_; ++i) {
    CUCO_CUDA_TRY(cudaStreamWaitEvent(stream, event_per_class_[i]));
  }

  CUCO_CUDA_TRY(cudaGetLastError());
}

template <typename KeyType, typename ElementType>
void DynamicEmbeddingTable<KeyType, ElementType>::lookup_unsafe(
    KeyType const *keys, ElementType **output_elements, size_t num_keys, size_t const *id_spaces,
    size_t const *id_space_offsets, size_t num_id_spaces, cudaStream_t stream) {
  CUCO_CUDA_TRY(cudaEventRecord(primary_event_, stream));

  for (size_t i = 0; i < num_id_spaces; ++i) {
    size_t id_space = id_spaces[i];
    size_t id_space_offset = id_space_offsets[i];
    size_t id_space_size = id_space_offsets[i + 1] - id_space_offset;

    assert(id_space < num_classes_);
    assert(id_space_offset + id_space_size <= num_keys);

    CUCO_CUDA_TRY(cudaStreamWaitEvent(stream_per_class_[id_space], primary_event_));
    maps_[id_space]->lookup_unsafe(keys + id_space_offset, output_elements + id_space_offset,
                                   id_space_size, stream_per_class_[id_space]);
    CUCO_CUDA_TRY(cudaEventRecord(event_per_class_[id_space], stream_per_class_[id_space]));
  }

  for (size_t i = 0; i < num_classes_; ++i) {
    CUCO_CUDA_TRY(cudaStreamWaitEvent(stream, event_per_class_[i]));
  }

  CUCO_CUDA_TRY(cudaGetLastError());
}

template <typename KeyType, typename ElementType>
void DynamicEmbeddingTable<KeyType, ElementType>::scatter_add(
    KeyType const *keys, ElementType const *update_elements, size_t num_keys,
    size_t const *id_spaces, size_t const *id_space_offsets, size_t num_id_spaces,
    cudaStream_t stream) {
  CUCO_CUDA_TRY(cudaEventRecord(primary_event_, stream));

  size_t update_offset = 0;
  for (size_t i = 0; i < num_id_spaces; ++i) {
    size_t id_space = id_spaces[i];
    size_t id_space_offset = id_space_offsets[i];
    size_t id_space_size = id_space_offsets[i + 1] - id_space_offset;
    assert(id_space < num_classes_);
    assert(id_space_offset + id_space_size <= num_keys);

    CUCO_CUDA_TRY(cudaStreamWaitEvent(stream_per_class_[id_space], primary_event_));
    maps_[id_space]->scatter_add(keys + id_space_offset, update_elements + update_offset,
                                 id_space_size, stream_per_class_[id_space]);
    update_offset += id_space_size * dimension_per_class_[id_space];
    CUCO_CUDA_TRY(cudaEventRecord(event_per_class_[id_space], stream_per_class_[id_space]));
  }

  for (size_t i = 0; i < num_classes_; ++i) {
    CUCO_CUDA_TRY(cudaStreamWaitEvent(stream, event_per_class_[i]));
  }

  CUCO_CUDA_TRY(cudaGetLastError());
}

template <typename KeyType, typename ElementType>
void DynamicEmbeddingTable<KeyType, ElementType>::scatter_update(
    KeyType const *keys, ElementType const *update_elements, size_t num_keys,
    size_t const *id_spaces, size_t const *id_space_offsets, size_t num_id_spaces,
    cudaStream_t stream) {
  CUCO_CUDA_TRY(cudaEventRecord(primary_event_, stream));
  size_t update_offset = 0;
  for (size_t i = 0; i < num_id_spaces; ++i) {
    size_t id_space = id_spaces[i];
    size_t id_space_offset = id_space_offsets[i];
    size_t id_space_size = id_space_offsets[i + 1] - id_space_offset;
    assert(id_space < num_classes_);
    assert(id_space_offset + id_space_size <= num_keys);

    CUCO_CUDA_TRY(cudaStreamWaitEvent(stream_per_class_[id_space], primary_event_));
    maps_[id_space]->scatter_update(keys + id_space_offset, update_elements + update_offset,
                                    id_space_size, stream_per_class_[id_space]);
    update_offset += id_space_size * dimension_per_class_[id_space];
    CUCO_CUDA_TRY(cudaEventRecord(event_per_class_[id_space], stream_per_class_[id_space]));
  }

  for (size_t i = 0; i < num_classes_; ++i) {
    CUCO_CUDA_TRY(cudaStreamWaitEvent(stream, event_per_class_[i]));
  }
  CUCO_CUDA_TRY(cudaGetLastError());
}

template <typename KeyType, typename ElementType>
void DynamicEmbeddingTable<KeyType, ElementType>::lookup_by_index(size_t class_index,
                                                                  KeyType const *d_keys,
                                                                  ElementType *d_values,
                                                                  size_t num_keys,
                                                                  cudaStream_t stream) {
  maps_[class_index]->lookup(d_keys, d_values, num_keys, stream);
  CUCO_CUDA_TRY(cudaGetLastError());
}

template <typename KeyType, typename ElementType>
void DynamicEmbeddingTable<KeyType, ElementType>::scatter_update_by_index(
    size_t class_index, KeyType const *d_keys, ElementType const *d_values, size_t num_keys,
    cudaStream_t stream) {
  maps_[class_index]->scatter_update(d_keys, d_values, num_keys, stream);
  CUCO_CUDA_TRY(cudaGetLastError());
}

template <typename KeyType, typename ElementType>
void DynamicEmbeddingTable<KeyType, ElementType>::remove(KeyType const *keys, size_t num_keys,
                                                         size_t const *id_spaces,
                                                         size_t const *id_space_offsets,
                                                         size_t num_id_spaces,
                                                         cudaStream_t stream) {
  CUCO_CUDA_TRY(cudaEventRecord(primary_event_, stream));

  for (size_t i = 0; i < num_id_spaces; ++i) {
    size_t id_space = id_spaces[i];
    size_t id_space_offset = id_space_offsets[i];
    size_t id_space_size = id_space_offsets[i + 1] - id_space_offset;
    assert(id_space < num_classes_);
    assert(id_space_offset + id_space_size <= num_keys);

    CUCO_CUDA_TRY(cudaStreamWaitEvent(stream_per_class_[id_space], primary_event_));
    maps_[id_space]->remove(keys + id_space_offset, id_space_size, stream_per_class_[id_space]);
    CUCO_CUDA_TRY(cudaEventRecord(event_per_class_[id_space], stream_per_class_[id_space]));
  }

  for (size_t i = 0; i < num_classes_; ++i) {
    CUCO_CUDA_TRY(cudaStreamWaitEvent(stream, event_per_class_[i]));
  }

  CUCO_CUDA_TRY(cudaGetLastError());
}

template <typename KeyType, typename ElementType>
void DynamicEmbeddingTable<KeyType, ElementType>::eXport(size_t class_index, KeyType *d_keys,
                                                         ElementType *d_values, size_t num_keys,
                                                         cudaStream_t stream) {
  maps_[class_index]->eXport(d_keys, d_values, num_keys, stream);
  CUCO_CUDA_TRY(cudaGetLastError());
}

template <typename KeyType, typename ElementType>
void DynamicEmbeddingTable<KeyType, ElementType>::clear(cudaStream_t stream) {
  for (auto &map : maps_) {
    map->clear(stream);
    CUCO_CUDA_TRY(cudaGetLastError());
  }
}

template <typename KeyType, typename ElementType>
size_t DynamicEmbeddingTable<KeyType, ElementType>::size() const {
  size_t n = 0;
  for (size_t i = 0; i < num_classes_; i++) {
    n += maps_[i]->get_size();
  }
  return n;
}

template <typename KeyType, typename ElementType>
size_t DynamicEmbeddingTable<KeyType, ElementType>::capacity() const {
  size_t n = 0;
  for (size_t i = 0; i < num_classes_; i++) {
    n += maps_[i]->get_capacity();
  }
  return n;
}

template <typename KeyType, typename ElementType>
std::vector<size_t> DynamicEmbeddingTable<KeyType, ElementType>::size_per_class() const {
  std::vector<size_t> sizes;
  for (size_t i = 0; i < num_classes_; i++) {
    sizes.push_back(maps_[i]->get_size());
  }
  return sizes;
}

template <typename KeyType, typename ElementType>
std::vector<size_t> DynamicEmbeddingTable<KeyType, ElementType>::capacity_per_class() const {
  std::vector<size_t> capacities;
  for (size_t i = 0; i < num_classes_; i++) {
    capacities.push_back(maps_[i]->get_capacity());
  }
  return capacities;
}

template class DynamicEmbeddingTable<uint64_t, float>;
template class DynamicEmbeddingTable<uint32_t, float>;
template class DynamicEmbeddingTable<int64_t, float>;
template class DynamicEmbeddingTable<int32_t, float>;

}  // namespace det