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
void DynamicEmbeddingTable<KeyType, ElementType>::lookup(KeyType const *keys,
                                                         ElementType *output_elements,
                                                         size_t num_keys,
                                                         size_t const *num_keys_per_class,
                                                         cudaStream_t stream) {
  CUCO_CUDA_TRY(cudaEventRecord(primary_event_, stream));

  size_t key_offset = 0;
  size_t element_offset = 0;
  for (size_t i = 0; i < num_classes_; i++) {
    CUCO_CUDA_TRY(cudaStreamWaitEvent(stream_per_class_[i], primary_event_));
    maps_[i]->lookup(keys + key_offset, output_elements + element_offset, num_keys_per_class[i],
                     stream_per_class_[i]);
    CUCO_CUDA_TRY(cudaEventRecord(event_per_class_[i], stream_per_class_[i]));

    key_offset += num_keys_per_class[i];
    element_offset += num_keys_per_class[i] * dimension_per_class_[i];
  }

  for (size_t i = 0; i < num_classes_; i++) {
    CUCO_CUDA_TRY(cudaStreamWaitEvent(stream, event_per_class_[i]));
  }

  CUCO_CUDA_TRY(cudaGetLastError());
}

template <typename KeyType, typename ElementType>
void DynamicEmbeddingTable<KeyType, ElementType>::scatter_add(KeyType const *keys,
                                                              ElementType const *update_elements,
                                                              size_t num_keys,
                                                              size_t const *num_keys_per_class,
                                                              cudaStream_t stream) {
  CUCO_CUDA_TRY(cudaEventRecord(primary_event_, stream));

  size_t key_offset = 0;
  size_t element_offset = 0;
  for (size_t i = 0; i < num_classes_; i++) {
    CUCO_CUDA_TRY(cudaStreamWaitEvent(stream_per_class_[i], primary_event_));
    maps_[i]->scatter_add(keys + key_offset, update_elements + element_offset,
                          num_keys_per_class[i], stream_per_class_[i]);
    CUCO_CUDA_TRY(cudaEventRecord(event_per_class_[i], stream_per_class_[i]));

    key_offset += num_keys_per_class[i];
    element_offset += num_keys_per_class[i] * dimension_per_class_[i];
  }

  for (size_t i = 0; i < num_classes_; i++) {
    CUCO_CUDA_TRY(cudaStreamWaitEvent(stream, event_per_class_[i]));
  }

  CUCO_CUDA_TRY(cudaGetLastError());
}

template <typename KeyType, typename ElementType>
void DynamicEmbeddingTable<KeyType, ElementType>::scatter_update(KeyType const *keys,
                                                                 ElementType const *update_elements,
                                                                 size_t num_keys,
                                                                 size_t const *num_keys_per_class,
                                                                 cudaStream_t stream) {
  CUCO_CUDA_TRY(cudaEventRecord(primary_event_, stream));

  size_t key_offset = 0;
  size_t element_offset = 0;
  for (size_t i = 0; i < num_classes_; i++) {
    CUCO_CUDA_TRY(cudaStreamWaitEvent(stream_per_class_[i], primary_event_));
    maps_[i]->scatter_update(keys + key_offset, update_elements + element_offset,
                             num_keys_per_class[i], stream_per_class_[i]);
    CUCO_CUDA_TRY(cudaEventRecord(event_per_class_[i], stream_per_class_[i]));

    key_offset += num_keys_per_class[i];
    element_offset += num_keys_per_class[i] * dimension_per_class_[i];
  }

  for (size_t i = 0; i < num_classes_; i++) {
    CUCO_CUDA_TRY(cudaStreamWaitEvent(stream, event_per_class_[i]));
  }

  CUCO_CUDA_TRY(cudaGetLastError());
}

template <typename KeyType, typename ElementType>
void DynamicEmbeddingTable<KeyType, ElementType>::remove(KeyType const *keys, size_t num_keys,
                                                         size_t const *num_keys_per_class,
                                                         cudaStream_t stream) {
  CUCO_CUDA_TRY(cudaEventRecord(primary_event_, stream));

  size_t key_offset = 0;
  for (size_t i = 0; i < num_classes_; i++) {
    CUCO_CUDA_TRY(cudaStreamWaitEvent(stream_per_class_[i], primary_event_));
    maps_[i]->remove(keys + key_offset, num_keys_per_class[i], stream_per_class_[i]);
    CUCO_CUDA_TRY(cudaEventRecord(event_per_class_[i], stream_per_class_[i]));

    key_offset += num_keys_per_class[i];
  }

  for (size_t i = 0; i < num_classes_; i++) {
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
size_t DynamicEmbeddingTable<KeyType, ElementType>::size() const {
  size_t n = 0;
  for (size_t i = 0; i < num_classes_; i++) {
    n += maps_[i]->get_size();
  }
  return n;
}

template <typename KeyType, typename ElementType>
size_t DynamicEmbeddingTable<KeyType, ElementType>::size(size_t idx) const {
  return maps_[idx]->get_size();
}

template <typename KeyType, typename ElementType>
size_t DynamicEmbeddingTable<KeyType, ElementType>::capacity() const {
  size_t n = 0;
  for (size_t i = 0; i < num_classes_; i++) {
    n += maps_[i]->get_capacity();
  }
  return n;
}

template class DynamicEmbeddingTable<int64_t, float>;
template class DynamicEmbeddingTable<int32_t, float>;