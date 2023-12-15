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

#include "common/check.h"
#include "variable/impl/hkv_variable.h"
#define SM_NUM 108
#define NTHREAD_PER_SM 2048

namespace sok {

__global__ static void setup_kernel(unsigned long long seed, curandState* states) {
  auto grid = cooperative_groups::this_grid();
  curand_init(seed, grid.thread_rank(), 0, &states[grid.thread_rank()]);
}

__device__ __forceinline__ unsigned int GlobalThreadId() {
  unsigned int smid;
  unsigned int warpid;
  unsigned int laneid;
  asm("mov.u32 %0, %%smid;" : "=r"(smid));
  asm("mov.u32 %0, %%warpid;" : "=r"(warpid));
  asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return smid * 2048 + warpid * 32 + laneid;
}

template <typename T>
__global__ void generate_uniform_kernel(curandState* state, T* result, size_t n) {
  auto id = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  /* Copy state to local memory for efficiency */
  curandState localState = state[GlobalThreadId()];
  /* Generate pseudo-random uniforms */
  for (size_t i = id; i < n; i += blockDim.x * gridDim.x) {
    result[i] = curand_uniform_double(&localState);
  }
  /* Copy state back to global memory */
  state[GlobalThreadId()] = localState;
}

template <typename T>
__global__ void generate_uniform_kernel(curandState* state, T** result, bool* d_found, size_t dim,
                                        size_t num_embedding) {
  auto id = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  size_t block_id = blockIdx.x;
  size_t emb_vec_id = threadIdx.x;
  /* Copy state to local memory for efficiency */
  curandState localState;
  bool load_state = false;
  for (size_t emb_id = block_id; emb_id < num_embedding; emb_id += gridDim.x) {
    if (!d_found[emb_id]) {
      if (!load_state) {
        localState = state[GlobalThreadId()];
        load_state = true;
      }
      for (size_t i = emb_vec_id; i < dim; i += blockDim.x) {
        result[emb_id][i] = curand_normal_double(&localState);
      }
    }
  }
  /* Copy state back to global memory */
  if (load_state) {
    state[GlobalThreadId()] = localState;
  }
}

template <typename T>
__global__ void const_initializer_kernel(float val, T* result, size_t n) {
  auto id = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  for (size_t i = id; i < n; i += blockDim.x * gridDim.x) {
    result[i] = static_cast<T>(val);
  }
}

template <typename T>
__global__ void const_initializer_kernel(float val, T** result, bool* d_found, size_t dim) {
  size_t id = threadIdx.x + blockIdx.x * blockDim.x;
  size_t emb_id = blockIdx.x;
  size_t emb_vec_id = threadIdx.x;
  if (!d_found[emb_id]) {
    for (size_t i = emb_vec_id; i < dim; i += blockDim.x) {
      result[emb_id][i] = static_cast<T>(val);
    }
  }
}

template <typename T>
__global__ void generate_normal_kernel(curandState* state, T* result, size_t n) {
  auto id = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  /* Copy state to local memory for efficiency */
  curandState localState = state[GlobalThreadId()];
  /* Generate pseudo-random normals */
  for (size_t i = id; i < n; i += blockDim.x * gridDim.x) {
    result[i] = curand_normal_double(&localState);
  }
  /* Copy state back to global memory */
  state[GlobalThreadId()] = localState;
}

template <typename T>
__global__ void generate_normal_kernel(curandState* state, T** result, bool* d_found, size_t dim,
                                       size_t num_embedding) {
  auto id = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  size_t block_id = blockIdx.x;
  size_t emb_vec_id = threadIdx.x;
  /* Copy state to local memory for efficiency */
  curandState localState;
  bool load_state = false;
  for (size_t emb_id = block_id; emb_id < num_embedding; emb_id += gridDim.x) {
    if (!d_found[emb_id]) {
      if (!load_state) {
        localState = state[GlobalThreadId()];
        load_state = true;
      }
      for (size_t i = emb_vec_id; i < dim; i += blockDim.x) {
        result[emb_id][i] = curand_normal_double(&localState);
      }
    }
  }

  /* Copy state back to global memory */
  if (load_state) {
    state[GlobalThreadId()] = localState;
  }
}

static void set_curand_states(curandState** states, cudaStream_t stream = 0) {
  int device;
  CUDACHECK(cudaGetDevice(&device));
  cudaDeviceProp deviceProp;
  CUDACHECK(cudaGetDeviceProperties(&deviceProp, device));
  // TODO: Use a more compatible way instead of `2048` to calculate the size.
  // CUDACHECK(
  //     cudaMallocAsync(states, sizeof(curandState) * deviceProp.multiProcessorCount * 2048,
  //     stream));
  CUDACHECK(cudaMalloc(states, sizeof(curandState) * deviceProp.multiProcessorCount * 2048));
  std::random_device rd;
  auto seed = rd();
  setup_kernel<<<deviceProp.multiProcessorCount * 2, 1024, 0, stream>>>(seed, *states);
  // To avoid unexpected errors caused by using `states` in other non-blocking streams.
  // It's OK to do synchronization here because this method should be called very few times.
  // CUDACHECK(cudaStreamSynchronize(stream));
}

static void parse_evict_strategy(const std::string& evict_strategy,
                                 nv::merlin::EvictStrategy& strategy) {
  if (evict_strategy == "kLru") {
    strategy = nv::merlin::EvictStrategy::kLru;
    return;
  } else if (evict_strategy == "kCustomized") {
    strategy = nv::merlin::EvictStrategy::kCustomized;
  } else {
    throw std::runtime_error("Unrecognized evict_strategy {" + evict_strategy + "}");
  }
}
template <typename KeyType, typename ValueType>
HKVVariable<KeyType, ValueType>::HKVVariable(int64_t dimension, int64_t initial_capacity,
                                             const std::string& initializer, size_t max_capacity,
                                             size_t max_hbm_for_vectors, size_t max_bucket_size,
                                             float max_load_factor, int block_size, int device_id,
                                             bool io_by_cpu, const std::string& evict_strategy,
                                             cudaStream_t stream)
    : dimension_(dimension),
      initial_capacity_(initial_capacity),
      initializer_(initializer),
      stream_(stream),
      curand_states_(nullptr) {
  if (dimension_ <= 0) {
    throw std::invalid_argument("dimension must > 0 but got " + std::to_string(dimension));
  }

  set_curand_states(&curand_states_, stream);

  hkv_table_option_.init_capacity = initial_capacity;
  hkv_table_option_.max_capacity = max_capacity;
  hkv_table_option_.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  hkv_table_option_.max_bucket_size = max_bucket_size;
  hkv_table_option_.max_load_factor = max_load_factor;
  hkv_table_option_.block_size = block_size;
  hkv_table_option_.device_id = device_id;
  hkv_table_option_.io_by_cpu = io_by_cpu;
  hkv_table_option_.dim = dimension;

  nv::merlin::EvictStrategy hkv_evict_strategy;
  parse_evict_strategy(evict_strategy, hkv_evict_strategy);
  hkv_table_option_.evict_strategy = hkv_evict_strategy;

  hkv_table_->init(hkv_table_option_);
}

template <typename KeyType, typename ValueType>
HKVVariable<KeyType, ValueType>::~HKVVariable() {
  if (curand_states_) {
    CUDACHECK(cudaFree(curand_states_));
  }
}

template <typename KeyType, typename ValueType>
int64_t HKVVariable<KeyType, ValueType>::rows() {
  return hkv_table_->size(stream_);
}

template <typename KeyType, typename ValueType>
int64_t HKVVariable<KeyType, ValueType>::cols() {
  return dimension_;
}

template <typename KeyType, typename ValueType>
void HKVVariable<KeyType, ValueType>::eXport(KeyType* keys, ValueType* values,
                                             cudaStream_t stream) {
  int64_t num_keys = rows();
  int64_t dim = cols();

  // `keys` and `values` are pointers of host memory
  KeyType* d_keys;
  CUDACHECK(cudaMallocManaged(&d_keys, sizeof(KeyType) * num_keys));
  ValueType* d_values;
  CUDACHECK(cudaMallocManaged(&d_values, sizeof(ValueType) * num_keys * dim));

  // KeyType* d_keys;
  // CUDACHECK(cudaMalloc(&d_keys, sizeof(KeyType) * num_keys));
  // ValueType* d_values;
  // CUDACHECK(cudaMalloc(&d_values, sizeof(ValueType) * num_keys * dim));
  hkv_table_->export_batch(hkv_table_option_.max_capacity, 0, d_keys, d_values, nullptr,
                           stream);  // Meta missing
  CUDACHECK(cudaStreamSynchronize(stream));

  // clang-format off
  std::memcpy(keys, d_keys, sizeof(KeyType) * num_keys);
  std::memcpy(values, d_values, sizeof(ValueType) * num_keys * dim);
  //CUDACHECK(cudaMemcpy(keys, d_keys, sizeof(KeyType) * num_keys,cudaMemcpyDeviceToHost));
  //CUDACHECK(cudaMemcpy(values, d_values, sizeof(ValueType) * num_keys * dim,cudaMemcpyDeviceToHost));
  //  clang-format on
  CUDACHECK(cudaFree(d_keys));
  CUDACHECK(cudaFree(d_values));
}

template <typename KeyType, typename ValueType>
void HKVVariable<KeyType, ValueType>::assign(const KeyType* keys, const ValueType* values,
                                             size_t num_keys, cudaStream_t stream) {
  int64_t dim = cols();
  // `keys` and `values` are pointers of host memory
  // KeyType* d_keys;
  // CUDACHECK(cudaMalloc(&d_keys, sizeof(KeyType) * num_keys));
  // ValueType* d_values;
  // CUDACHECK(cudaMalloc(&d_values, sizeof(ValueType) * num_keys * dim));

  KeyType* d_keys;
  CUDACHECK(cudaMallocManaged(&d_keys, sizeof(KeyType) * num_keys));
  ValueType* d_values;
  CUDACHECK(cudaMallocManaged(&d_values, sizeof(ValueType) * num_keys * dim));
  // clang-format off
  //CUDACHECK(cudaMemcpyAsync(d_keys, keys, sizeof(KeyType) * num_keys,
  //                          cudaMemcpyHostToDevice, stream));

  //CUDACHECK(cudaMemcpyAsync(d_values, values, sizeof(ValueType) * num_keys * dim,
  //                          cudaMemcpyHostToDevice, stream));

  //CUDACHECK(cudaStreamSynchronize(stream));
  std::memcpy(d_keys, keys, sizeof(KeyType) * num_keys);
  std::memcpy(d_values, values, sizeof(ValueType) * num_keys * dim);
  hkv_table_->insert_or_assign(num_keys, d_keys, d_values, nullptr, stream);

  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaFree(d_keys));
  CUDACHECK(cudaFree(d_values));
}

template <typename KeyType, typename ValueType>
void HKVVariable<KeyType, ValueType>::lookup(const KeyType* keys, ValueType* values,
                                             size_t num_keys, cudaStream_t stream) {
  bool* d_found;
  CUDACHECK(cudaMalloc(&d_found, num_keys * sizeof(bool)));
  CUDACHECK(cudaMemset(d_found, 0, num_keys * sizeof(bool)));

  int64_t dim = cols();

  uint32_t grid_dim = SM_NUM*(NTHREAD_PER_SM/1024);
  if ((num_keys * static_cast<size_t>(dim))/1024 < grid_dim) grid_dim = (num_keys * static_cast<size_t>(dim))/1024;
  if (initializer_ == "normal" || initializer_ == "random") {

    generate_normal_kernel<<<grid_dim, 1024, 0, stream>>>(curand_states_, values,num_keys * dim);
  } else if (initializer_ == "uniform") {
    generate_uniform_kernel<<<grid_dim , 1024, 0, stream>>>(curand_states_, values, num_keys * dim);
  } else {
    try {
      float val = std::stof(initializer_);
      const_initializer_kernel<<<num_keys, max(dim, static_cast<int64_t>(32)), 0, stream>>>(
          val, values, num_keys * dim);
    } catch (std::invalid_argument& err) {
      throw std::runtime_error("Unrecognized initializer {" + initializer_ + "}");
    }
  }

  hkv_table_->find_or_insert(num_keys, keys, values, nullptr, stream);
  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaFree(d_found));
}

template <typename KeyType, typename ValueType>
void HKVVariable<KeyType, ValueType>::lookup(const KeyType* keys, ValueType** values,
                                             size_t num_keys, cudaStream_t stream) {
  bool* d_found;
  CUDACHECK(cudaMalloc(&d_found, num_keys * sizeof(bool)));
  CUDACHECK(cudaMemset(d_found, 0, num_keys * sizeof(bool)));
  hkv_table_->find_or_insert(num_keys, keys, values, d_found, nullptr, stream);
  //CUDACHECK(cudaStreamSynchronize(stream));
  int64_t dim = cols();
  uint32_t block_dim = max(dim, static_cast<int64_t>(32));
  uint32_t grid_dim = SM_NUM*(NTHREAD_PER_SM/block_dim);
  if (num_keys<grid_dim) grid_dim = num_keys;
  if (initializer_ == "normal" || initializer_ == "random") {
    generate_normal_kernel<<<grid_dim, block_dim, 0, stream>>>(
        curand_states_, values, d_found, dim,num_keys);
  } else if (initializer_ == "uniform") {
    generate_uniform_kernel<<<grid_dim, block_dim, 0, stream>>>(
        curand_states_, values, d_found, dim,num_keys);
  } else {
    try {
      float val = std::stof(initializer_);
      const_initializer_kernel<<<num_keys, block_dim, 0, stream>>>(
          val, values, d_found, dim);
    } catch (std::invalid_argument& err) {
      throw std::runtime_error("Unrecognized initializer {" + initializer_ + "}");
    }
  }

  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaFree(d_found));
}

template <typename KeyType, typename ValueType>
void HKVVariable<KeyType, ValueType>::scatter_add(const KeyType* keys, const ValueType* values,
                                                  size_t num_keys, cudaStream_t stream) {
  int64_t dim = cols();
  bool* d_found;
  CUDACHECK(cudaMalloc(&d_found, num_keys * sizeof(bool)));
  CUDACHECK(cudaMemset(d_found, 0, num_keys * sizeof(bool)));
  
  ValueType* d_values;
  CUDACHECK(cudaMalloc(&d_values, sizeof(ValueType) * num_keys * dim));

  hkv_table_->find(num_keys, keys, d_values, d_found, nullptr, stream);
  hkv_table_->accum_or_assign(num_keys, keys, values, d_found, nullptr, stream);

  CUDACHECK(cudaFree(d_found));
  CUDACHECK(cudaFree(d_values));
}

template <typename KeyType, typename ValueType>
void HKVVariable<KeyType, ValueType>::scatter_update(const KeyType* keys, const ValueType* values,
                                                     size_t num_keys, cudaStream_t stream) {
  int64_t dim = cols();

  hkv_table_->assign(num_keys, keys, values, nullptr, stream);

  CUDACHECK(cudaStreamSynchronize(stream));
}

template class HKVVariable<int64_t, float>;
}  // namespace sok
