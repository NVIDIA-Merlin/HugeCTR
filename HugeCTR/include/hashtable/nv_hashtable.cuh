/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#ifndef NV_HASHTABLE_H_
#define NV_HASHTABLE_H_
#include <mutex>
#include "cudf/concurrent_unordered_map.cuh"
#include "HugeCTR/include/common.hpp"
#include "thrust/pair.h"
//#define COUNTER_TYPE ValType

namespace HugeCTR {
  
template <typename value_type>
struct ReplaceOp {
  constexpr static value_type IDENTITY{0};

  __host__ __device__ value_type operator()(value_type new_value, value_type old_value) {
    return new_value;
  }
};

template <typename Table>
__global__ void insert_kernel(Table* table, const typename Table::key_type* const keys,
                              const typename Table::mapped_type* const vals, size_t len) {
  ReplaceOp<typename Table::mapped_type> op;
  thrust::pair<typename Table::key_type, typename Table::mapped_type> kv;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    kv.first = keys[i];
    kv.second = vals[i];
    auto it = table->insert(kv, op);
    assert(it != table->end() && "error: insert fails: table is full");
  }
}

template <typename Table>
__global__ void accum_kernel(Table* table, const typename Table::key_type* const keys,
                             const typename Table::mapped_type* const vals, size_t len) {
  thrust::pair<typename Table::key_type, typename Table::mapped_type> kv;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    kv.first = keys[i];
    kv.second = vals[i];
    auto it = table->accum(kv);
    assert(it != table->end() && "error: can't find key");
  }
}

template <typename Table, typename GradType, typename Optimizer>
__global__ void update_kernel(Table* table, const typename Table::key_type* const keys,
                              const GradType* const gradients, size_t len, Optimizer& op) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    assert(it != table->end() && "error: can't find key");
    op.update((it.getter())->second, gradients[i]);
  }
}

template <typename Table>
__global__ void search_kernel(Table* table, const typename Table::key_type* const keys,
                              typename Table::mapped_type* const vals, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    assert(it != table->end() && "error: can't find key");
    vals[i] = it->second;
  }
}

template <typename Table, typename counter_type>
__global__ void get_insert_kernel(Table* table, const typename Table::key_type* const keys,
                                  typename Table::mapped_type* const vals, size_t len,
                                  counter_type* d_counter) {
  ReplaceOp<typename Table::mapped_type> op;
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->get_insert(keys[i], op, d_counter);
    assert(it != table->end() && "error: get_insert fails: table is full");
    vals[i] = it->second;
  }
}

template <typename Table, typename KeyType>
__global__ void size_kernel(const Table* table, const size_t hash_capacity, size_t* table_size,
                            KeyType unused_key) {
  /* Per block accumulator */
  __shared__ size_t block_acc;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  /* Initialize */
  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  /* Whether the bucket mapping to the current thread is empty? do nothing : Atomically add to
   * counter */
  if (i < hash_capacity) {
    typename Table::value_type val = load_pair_vectorized(table->data() + i);
    if (val.first != unused_key) {
      atomicAdd(&block_acc, 1);
    }
  }
  __syncthreads();

  /* Atomically reduce block counter to global conuter */
  if (threadIdx.x == 0) {
    atomicAdd(table_size, block_acc);
  }
}

template <typename KeyType, typename ValType, typename Table>
__global__ void dump_kernel(KeyType* d_key, ValType* d_val, const Table* table, const size_t offset,
                            const size_t search_length, size_t* d_dump_counter,
                            KeyType unused_key) {
  // inter-block gathered key, value and counter. Global conuter for storing shared memory into
  // global memory.
  //__shared__ KeyType block_result_key[BLOCK_SIZE_];
  //__shared__ ValType block_result_val[BLOCK_SIZE_];
  extern __shared__ unsigned char s[];
  KeyType* smem = (KeyType*)s;
  KeyType* block_result_key = smem;
  ValType* block_result_val = (ValType*)&(smem[blockDim.x]);
  __shared__ size_t block_acc;
  __shared__ size_t global_acc;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  /* Initialize */
  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  // Each thread gather the key and value from bucket assigned to them and store them into shared
  // mem.
  if (i < search_length) {
    typename Table::value_type val = load_pair_vectorized(table->data() + offset + i);
    if (val.first != unused_key) {
      size_t local_index = atomicAdd(&block_acc, 1);
      block_result_key[local_index] = val.first;
      block_result_val[local_index] = val.second;
    }
  }
  __syncthreads();

  // Each block request a unique place in global memory buffer, this is the place where shared
  // memory store back to.
  if (threadIdx.x == 0) {
    global_acc = atomicAdd(d_dump_counter, block_acc);
  }
  __syncthreads();

  // Each thread store one bucket's data back to global memory, d_dump_counter is how many buckets
  // in total dumped.
  if (threadIdx.x < block_acc) {
    d_key[global_acc + threadIdx.x] = block_result_key[threadIdx.x];
    d_val[global_acc + threadIdx.x] = block_result_val[threadIdx.x];
  }
}

/**
 * The HashTable class is wrapped by cudf library for hash table operations on single GPU.
 * In this class, we implement the GPU version of the common used operations of hash table,
 * such as insert() / get() / set() / dump()...
 */
template <typename KeyType, typename ValType, KeyType empty_key,
          typename counter_type = unsigned long long int>
class HashTable {
 public:
  /**
   * The constructor of HashTable.
   * @param capacity the number of <key,value> pairs in the hash table.
   * @param count the existed number of <key,value> pairs in the hash table.
   */
  HashTable(size_t capacity, counter_type count = 0) {
    // assert(capacity <= std::numeric_limits<ValType>::max() && "error: Table is too large for the
    // value type");
    table_ = new Table(capacity, std::numeric_limits<ValType>::max());
    update_counter_ = 0;
    get_counter_ = 0;
    // Allocate device-side counter and copy user input to it
    CK_CUDA_THROW_(cudaMalloc((void**)&d_counter_, sizeof(*d_counter_)));
    CK_CUDA_THROW_(cudaMemcpy(d_counter_, &count, sizeof(*d_counter_), cudaMemcpyHostToDevice));
  }
  /**
   * The destructor of HashTable.
   */
  ~HashTable() {
    try {
      delete table_;
      // De-allocate device-side counter
      CK_CUDA_THROW_(cudaFree(d_counter_));
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
    }

  }
  /**
   * The declaration for indicating that there is no default copy construtor in this class.
   */
  HashTable(const HashTable&) = delete;
  /**
   * The declaration for indicating that there is no default operator "=" overloading in this class.
   */
  HashTable& operator=(const HashTable&) = delete;
  /**
   * The insert function for hash table. "insert" means putting some new <key,value> pairs
   * into the current hash table.
   * @param d_keys the device pointers for the keys.
   * @param d_vals the device pointers for the values.
   * @param len the number of <key,value> pairs to be inserted into the hash table.
   * @param stream the cuda stream for this operation.
   */
  void insert(const KeyType* d_keys, const ValType* d_vals, size_t len, cudaStream_t stream) {
    if (len == 0) {
      return;
    }
    const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
    insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, d_vals, len);
  }
  /**
   * The get function for hash table. "get" means fetching some values indexed
   * by the given keys.
   * @param d_keys the device pointers for the keys.
   * @param d_vals the device pointers for the values.
   * @param len the number of <key,value> pairs to be got from the hash table.
   * @param stream the cuda stream for this operation.
   */
  void get(const KeyType* d_keys, ValType* d_vals, size_t len, cudaStream_t stream) const {
    if (len == 0) {
      return;
    }
    const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
    search_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, d_vals, len);
  }
  /**
   * The set function for hash table. "set" means using the given values to
   * overwrite the values indexed by the given keys.
   * @param d_keys the device pointers for the keys.
   * @param d_vals the device pointers for the values.
   * @param len the number of <key,value> pairs to be set to the hash table.
   * @param stream the cuda stream for this operation.
   */
  void set(const KeyType* d_keys, const ValType* d_vals, size_t len, cudaStream_t stream) {
    if (len == 0) {
      return;
    }
    const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
    insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, d_vals, len);
  }
  /**
   * The accumulate function for hash table. "accumulate" means accumulating
   * the given values to the values indexed by the given keys.
   * @param d_keys the device pointers for the keys.
   * @param d_vals the device pointers for the values.
   * @param len the number of <key,value> pairs to be accumulated to the hash table.
   * @param stream the cuda stream for this operation.
   */
  void accum(const KeyType* d_keys, const ValType* d_vals, size_t len, cudaStream_t stream) {
    if (len == 0) {
      return;
    }
    const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
    accum_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, d_vals, len);
  }
  /**
   * Get the current size of the hash table. Size is also known as the number
   * of <key,value> pairs.
   * @param stream the cuda stream for this operation.
   */
  size_t get_size(cudaStream_t stream) const {
    /* size variable on Host and device, total capacity of the hashtable */
    size_t table_size;
    size_t* d_table_size;
    const size_t hash_capacity = table_->size();

    /* grid_size and allocating/initializing variable on dev, launching kernel*/
    const int grid_size = (hash_capacity - 1) / BLOCK_SIZE_ + 1;
    CK_CUDA_THROW_(cudaMalloc((void**)&d_table_size, sizeof(size_t)));
    CK_CUDA_THROW_(cudaMemset(d_table_size, 0, sizeof(size_t)));
    size_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, hash_capacity, d_table_size,
                                                       empty_key);
    CK_CUDA_THROW_(
        cudaMemcpyAsync(&table_size, d_table_size, sizeof(size_t), cudaMemcpyDeviceToHost, stream));
    CK_CUDA_THROW_(cudaStreamSynchronize(stream));

    /* Copy result back and do clean up*/
    CK_CUDA_THROW_(cudaFree(d_table_size));
    return table_size;
  }
  /**
   * The dump function for hash table. "dump" means getting some of the <key,value>
   * pairs from the hash table and copying them to the corresponding memory buffer.
   * @param d_keys the device pointers for the keys.
   * @param d_vals the device pointers for the values.
   * @param offset the start position of the dumped <key,value> pairs.
   * @param search_length the number of <key,value> pairs to be dumped.
   * @param d_dump_counter a temp device pointer to store the dump counter.
   * @param stream the cuda stream for this operation.
   */
  void dump(KeyType* d_key, ValType* d_val, const size_t offset, const size_t search_length,
            size_t* d_dump_counter, cudaStream_t stream) {
    // Before we call the kernel, set the global counter to 0
    CK_CUDA_THROW_(cudaMemset(d_dump_counter, 0, sizeof(size_t)));
    // grid size according to the searching length.
    const int grid_size = (search_length - 1) / BLOCK_SIZE_ + 1;
    // dump_kernel: dump bucket table_[offset, offset+search_length) to d_key and d_val, and report
    // how many buckets are dumped in d_dump_counter.
    size_t shared_size = sizeof(*d_key) * BLOCK_SIZE_ + sizeof(*d_val) * BLOCK_SIZE_;
    dump_kernel<<<grid_size, BLOCK_SIZE_, shared_size, stream>>>(
        d_key, d_val, table_, offset, search_length, d_dump_counter, empty_key);
  }
  /**
   * Get the capacity of the hash table. "capacity" is known as the number of
   * <key,value> pairs of the hash table.
   */
  size_t get_capacity() const { return (table_->size()); }
  /**
   * Get the head of the value from the device counter. It's equal to the
   * number of the <key,value> pairs in the hash table.
   */
  counter_type get_value_head() {
    counter_type counter;
    CK_CUDA_THROW_(cudaMemcpy(&counter, d_counter_, sizeof(*d_counter_), cudaMemcpyDeviceToHost));
    return counter;
  }
  /**
   * Set the head of the value. This will reset a new value to the device counter.
   * @param counter_value the new counter value to be set.
   */
  void set_value_head(counter_type counter_value) {
    CK_CUDA_THROW_(cudaMemcpy(d_counter_, &counter_value, sizeof(*d_counter_), cudaMemcpyHostToDevice));
  }
  /**
   * Add a number to the head of the value. This will add the given value to the
   * current value of the device counter.
   * @param counter_add the new counter value to be added.
   */
  counter_type add_value_head(counter_type counter_add) {
    counter_type counter;
    CK_CUDA_THROW_(cudaMemcpy(&counter, d_counter_, sizeof(*d_counter_), cudaMemcpyDeviceToHost));
    counter += counter_add;
    CK_CUDA_THROW_(cudaMemcpy(d_counter_, &counter, sizeof(*d_counter_), cudaMemcpyHostToDevice));
    return counter;
  }
  /**
   * The update function for hash table. "update" means using the given gradients to
   * update the values indexed by the input keys based on the optimizer "op".
   * @param d_keys the device pointers for the keys.
   * @param d_gradients the gradients for updating the values.
   * @param len the number of <key,value> pairs to be updated.
   * @param stream the cuda stream for this operation.
   * @param op the optimizer method to update the values with the gradients.
   */
  template <typename GradType, typename Optimizer>
  void update(const KeyType* d_keys, const GradType* d_gradients, size_t len, cudaStream_t stream,
              Optimizer& op) {
    if (len == 0) {
      return;
    }
    const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
    update_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, d_gradients, len, op);
  }
  /**
   * The get_insert function for hash table. "get_insert" means if we can find
   * the keys in the hash table, the values indexed by the keys will be returned,
   * which is known as a "get" operation; Otherwise, the not-found keys together
   * with the values computed by the current device counter automatically will be
   * inserted into the hash table.
   * @param d_keys the device pointers for the keys.
   * @param d_vals the device pointers for the values.
   * @param len the number of <key,value> pairs to be got or inserted into the hash table.
   * @param stream the cuda stream for this operation.
   */
  void get_insert(const KeyType* d_keys, ValType* d_vals, size_t len, cudaStream_t stream) const {
    if (len == 0) {
      return;
    }
    const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
    get_insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, d_vals, len,
                                                             d_counter_);
  }
  /**
   * Before any get API is called, call this function to check and update counter.
   */
  bool get_lock() {
    counter_mtx_.lock();
    bool ret_val;
    if (update_counter_ > 0) {
      ret_val = false;  // There are update APIs running, can't do get.
    } else {
      get_counter_++;
      ret_val = true;  // There is no update API running, can do get, increase counter.
    }
    counter_mtx_.unlock();
    return ret_val;
  }
  /**
   * Before any update API is called, call this function to check and update counter.
   */
  bool update_lock() {
    counter_mtx_.lock();
    bool ret_val;
    if (get_counter_ > 0) {
      ret_val = false;  // There are get APIs running, can't do update
    } else {
      update_counter_++;
      ret_val = true;  // There is no get API running, can do update, increase counter.
    }
    counter_mtx_.unlock();
    return ret_val;
  }
  /**
   * After each get API finished on this GPU's hash table, decrease the counter.
   */
  void get_release() {
    counter_mtx_.lock();
    get_counter_--;  // one get API finish, dec counter
    counter_mtx_.unlock();
  }
  /**
   * After each update API finished on this GPU's hash table, decrease the counter.
   */
  void update_release() {
    counter_mtx_.lock();
    update_counter_--;  // one update API finish, dec counter
    counter_mtx_.unlock();
  }

 private:
  static const int BLOCK_SIZE_ =
      256; /**< The block size of the CUDA kernels. The default value is 256. */
  using Table = concurrent_unordered_map<KeyType, ValType, empty_key>;

  Table* table_; /**< The object of the Table class which is defined in the concurrent_unordered_map
                    class. */

  // GPU-level lock and counters for get and update APIs
  std::mutex counter_mtx_;         /**< The mutex for protecting the counters. */
  volatile size_t update_counter_; /**< The counter to indicate how many update APIs are currently
                                      called on this GPU' hash table. */
  volatile size_t get_counter_; /**< The counter to indicate how many get APIs are currently called
                                   on this GPU's hash table. */

  // Counter for value index
  counter_type* d_counter_; /**< The device counter for value index. */
};

}  // namespace nv
#endif
