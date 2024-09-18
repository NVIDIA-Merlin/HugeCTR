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
        result[emb_id][i] = curand_uniform_double(&localState);
      }
    }
  }
  /* Copy state back to global memory */
  if (load_state) {
    state[GlobalThreadId()] = localState;
  }
}

template <typename T>
__global__ void generate_uniform_kernel(curandState* state, T* result, bool* d_found, size_t dim,
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
        T* tmp_reslut = result + emb_id * dim;
        tmp_reslut[i] = curand_uniform_double(&localState);
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
__global__ void const_initializer_kernel(float val, T* result, bool* d_found, size_t dim) {
  size_t id = threadIdx.x + blockIdx.x * blockDim.x;
  size_t emb_id = blockIdx.x;
  size_t emb_vec_id = threadIdx.x;
  if (!d_found[emb_id]) {
    for (size_t i = emb_vec_id; i < dim; i += blockDim.x) {
      T* tmp_reslut = result + emb_id * dim;
      tmp_reslut[i] = static_cast<T>(val);
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

template <typename T>
__global__ void generate_normal_kernel(curandState* state, T* result, bool* d_found, size_t dim,
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
        T* tmp_reslut = result + emb_id * dim;
        tmp_reslut[i] = curand_normal_double(&localState);
      }
    }
  }

  /* Copy state back to global memory */
  if (load_state) {
    state[GlobalThreadId()] = localState;
  }
}

template <typename KeyType, typename ValueType>
__global__ void select_no_found_kernel(const KeyType* keys, const ValueType* values, bool* d_found,
                                       uint64_t num_keys, uint64_t dim, KeyType* ret_keys,
                                       ValueType* ret_values, uint64_t* num_no_found) {
  auto id = static_cast<uint64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  __shared__ uint64_t smem[1];
  uint64_t block_id = blockIdx.x;
  uint64_t emb_vec_id = threadIdx.x;
  /* Copy state to local memory for efficiency */
  for (uint64_t emb_id = block_id; emb_id < num_keys; emb_id += gridDim.x) {
    if (!d_found[emb_id]) {
      if (emb_vec_id == 0) {
        uint64_t index = atomicAdd(num_no_found, 1);
        smem[0] = index;
        ret_keys[index] = keys[emb_id];
      }
      __syncthreads();
      uint64_t output_index = smem[0];

      for (uint64_t i = emb_vec_id; i < dim; i += blockDim.x) {
        const ValueType* tmp_values = values + emb_id * dim;
        ValueType* tmp_ret_values = ret_values + output_index * dim;
        tmp_ret_values[i] = tmp_values[i];
      }
    }
  }
}

template <class K, class S>
struct ExportIfPredFunctor {
  __forceinline__ __device__ bool operator()(const K& key, S& score, const K& pattern,
                                             const S& threshold) {
    return score > threshold;
  }
};

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
                                             cudaStream_t stream, float filter_ratio)
    : dimension_(dimension),
      initial_capacity_(initial_capacity),
      initializer_(initializer),
      stream_(stream),
      curand_states_(nullptr),
      filter_ratio_(filter_ratio) {
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

  hkv_table_->export_batch(hkv_table_option_.max_capacity, 0, d_keys, d_values, nullptr,
                           stream);  // Meta missing
  CUDACHECK(cudaStreamSynchronize(stream));

  std::memcpy(keys, d_keys, sizeof(KeyType) * num_keys);
  std::memcpy(values, d_values, sizeof(ValueType) * num_keys * dim);
  CUDACHECK(cudaFree(d_keys));
  CUDACHECK(cudaFree(d_values));
}

template <typename KeyType, typename ValueType>
void HKVVariable<KeyType, ValueType>::eXport_if(KeyType* keys, ValueType* values, size_t* counter,
                                                uint64_t threshold, cudaStream_t stream) {
  int64_t num_keys = rows();
  int64_t dim = cols();

  // `keys` and `values` are pointers of host memory
  KeyType* d_keys;
  CUDACHECK(cudaMallocManaged(&d_keys, sizeof(KeyType) * num_keys));
  ValueType* d_values;
  CUDACHECK(cudaMallocManaged(&d_values, sizeof(ValueType) * num_keys * dim));

  uint64_t* d_socre_type;
  CUDACHECK(cudaMallocManaged(&d_socre_type, sizeof(uint64_t) * num_keys));

  uint64_t* d_dump_counter;
  CUDACHECK(cudaMallocManaged(&d_dump_counter, sizeof(uint64_t)));
  // useless HKV need a input , but do nothing in the ExportIfPredFunctor
  KeyType pattern = 100;

  hkv_table_->template export_batch_if<ExportIfPredFunctor>(
      pattern, threshold, hkv_table_->capacity(), 0, d_dump_counter, d_keys, d_values, d_socre_type,
      stream);
  CUDACHECK(cudaStreamSynchronize(stream));
  // clang-format off
  std::memcpy(keys, d_keys, sizeof(KeyType) * (*d_dump_counter));
  std::memcpy(values, d_values, sizeof(ValueType) * (*d_dump_counter) * dim);
  counter[0] = (size_t)(*d_dump_counter);
  //  clang-format on
  CUDACHECK(cudaFree(d_keys));
  CUDACHECK(cudaFree(d_values));
  CUDACHECK(cudaFree(d_socre_type));
  CUDACHECK(cudaFree(d_dump_counter));
}

template <typename KeyType, typename ValueType>
void HKVVariable<KeyType, ValueType>::assign(const KeyType* keys, const ValueType* values,
                                             size_t num_keys, cudaStream_t stream) {
  int64_t dim = cols();

  KeyType* d_keys;
  CUDACHECK(cudaMallocManaged(&d_keys, sizeof(KeyType) * num_keys));
  ValueType* d_values;
  CUDACHECK(cudaMallocManaged(&d_values, sizeof(ValueType) * num_keys * dim));
  // clang-format off
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
void HKVVariable<KeyType, ValueType>::lookup_with_evict(const KeyType *keys,KeyType *tmp_keys, ValueType* tmp_values, ValueType *values,uint64_t* evict_num_keys,uint64_t num_keys,cudaStream_t stream){

  int64_t dim = cols();

  bool* d_found;
  KeyType* tmp_key_buffer;
  ValueType* tmp_value_buffer;
  uint64_t* tmp_counters;
  uint64_t* d_evict_num_keys;
  uint64_t h_tmp_counters[1];
  
  uint64_t tmp_buffer_size = 0;
  tmp_buffer_size += align_length(num_keys*sizeof(bool));
  tmp_buffer_size += align_length(num_keys*sizeof(KeyType));
  tmp_buffer_size += align_length(num_keys*dim*sizeof(ValueType));
  tmp_buffer_size += align_length(sizeof(uint64_t));
  tmp_buffer_size += align_length(sizeof(size_t));

  CUDACHECK(cudaMallocAsync(&d_found, tmp_buffer_size,stream));
  CUDACHECK(cudaMemsetAsync(d_found, 0, tmp_buffer_size,stream));

  CUDACHECK(cudaStreamSynchronize(stream));
  tmp_key_buffer = (KeyType*)(((char*)d_found)+align_length(num_keys*sizeof(bool))); 
  tmp_value_buffer = (ValueType*)(((char*)tmp_key_buffer)+align_length(num_keys*sizeof(KeyType)));
  tmp_counters = (uint64_t*)(((char*)tmp_value_buffer)+align_length(num_keys*dim*sizeof(ValueType)));
  d_evict_num_keys = (size_t*)(((char*)tmp_counters)+align_length(sizeof(uint64_t)));

  //found first 
  hkv_table_->find(num_keys, keys, values, d_found, nullptr, stream);

  //fill not found
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

  select_no_found_kernel<<<grid_dim, block_dim, 0, stream>>>(keys,values,d_found,num_keys,dim,tmp_key_buffer,tmp_value_buffer,tmp_counters);
  CUDACHECK(cudaMemcpyAsync(h_tmp_counters,tmp_counters,sizeof(uint64_t),cudaMemcpyDeviceToHost,stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  
  hkv_table_->insert_and_evict(h_tmp_counters[0], tmp_key_buffer, tmp_value_buffer, nullptr, tmp_keys,tmp_values,nullptr,d_evict_num_keys, stream);

  CUDACHECK(cudaMemcpyAsync(evict_num_keys,d_evict_num_keys,sizeof(size_t),cudaMemcpyDeviceToHost,stream));
  CUDACHECK(cudaFreeAsync(d_found,stream));
  CUDACHECK(cudaStreamSynchronize(stream));
}

template <typename KeyType, typename ValueType>
void HKVVariable<KeyType, ValueType>::copy_evict_keys(const KeyType* keys, const ValueType* values,size_t num_keys,size_t dim, KeyType* ret_keys, ValueType* ret_values, cudaStream_t stream) {

  CUDACHECK(cudaMemcpyAsync(ret_keys,keys,sizeof(KeyType)*num_keys,cudaMemcpyDeviceToDevice,stream));
  CUDACHECK(cudaMemcpyAsync(ret_values,values,sizeof(ValueType)*num_keys*dim,cudaMemcpyDeviceToDevice,stream));
  CUDACHECK(cudaStreamSynchronize(stream));

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

template <typename KeyType>
__global__ void ratio_filter_flag(curandState* state, const KeyType *keys, bool *filtered, size_t num_keys, float filter_ratio) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandState localState;
  localState = state[GlobalThreadId()];
  for (int i = idx; i < num_keys; i += blockDim.x * gridDim.x) {
    if (!filtered[i]) {
      auto ratio = curand_uniform(&localState);
      if (ratio < filter_ratio) {
        filtered[i] = true;
      }
    }
  }
  state[GlobalThreadId()] = localState;
}
template <typename KeyType, typename ValueType>
void HKVVariable<KeyType, ValueType>::ratio_filter(const KeyType *keys, bool *filtered,
                                            size_t num_keys, cudaStream_t stream) {
  // TODO: update hkv, use exist;
  ValueType** p_values;
  CUDACHECK(cudaMallocAsync(&p_values,  num_keys * sizeof(ValueType*),stream));
  hkv_table_->find(num_keys, keys, p_values, filtered, nullptr, stream);
  uint32_t grid_dim = SM_NUM * (NTHREAD_PER_SM/256);
  // filter
  ratio_filter_flag<<<grid_dim, 256, 0, stream>>>(curand_states_, keys, filtered, num_keys, filter_ratio_);
  CUDACHECK(cudaFreeAsync(p_values,stream));
  //CUDACHECK(cudaStreamSynchronize(stream));
}

template class HKVVariable<int64_t, float>;
}  // namespace sok
