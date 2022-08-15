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

#include <curand.h>

#include <algorithm>
#include <chrono>
#include <random>

#include "gtest/gtest.h"
#include "hash_table.hpp"

#define CURAND_CALL(x)                                      \
  do {                                                      \
    if ((x) != CURAND_STATUS_SUCCESS) {                     \
      printf("Error %d at %s:%d\n", x, __FILE__, __LINE__); \
    }                                                       \
  } while (0)

#define TEST_ROUNDS 50

void benchmarkInsertOrAssign(const size_t num_keys, const size_t num_keys_space) {
  using KeyType = int64_t;
  using ValueType = size_t;

  std::random_device rd;
  std::mt19937 eng(rd());

  curandGenerator_t gen;
  CURAND_CALL(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));

  HashTable<KeyType, ValueType> hash_table(num_keys_space);

  std::vector<KeyType> h_keys_space(num_keys_space);
  CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned int *>(h_keys_space.data()),
                             num_keys_space * sizeof(KeyType) / 4));

  std::vector<KeyType> h_keys(num_keys);

  KeyType *d_keys;
  CUDA_RT_CALL(cudaMalloc(&d_keys, sizeof(KeyType) * num_keys));
  ValueType *d_values;
  CUDA_RT_CALL(cudaMalloc(&d_values, sizeof(ValueType) * num_keys));

  for (size_t i = 0; i < TEST_ROUNDS; i++) {
    std::shuffle(h_keys_space.begin(), h_keys_space.end(), eng);
    std::copy_n(h_keys_space.begin(), num_keys, h_keys.begin());

    CUDA_RT_CALL(
        cudaMemcpy(d_keys, h_keys.data(), sizeof(KeyType) * num_keys, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaDeviceSynchronize());

    auto start = std::chrono::steady_clock::now();
    hash_table.insert_or_assign(d_keys, d_values, num_keys);
    CUDA_RT_CALL(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("[%zu] %.2fms %.0fM %.0fM\n", i, diff.count() * 1000,
           1.0f * hash_table.get_size() / 1024 / 1024,
           1.0f * hash_table.get_capacity() / 1024 / 1024);
  }

  CUDA_RT_CALL(cudaFree(d_values));
  CUDA_RT_CALL(cudaFree(d_keys));

  CURAND_CALL(curandDestroyGenerator(gen));
}

void benchmarkLookupOrInsert(const size_t num_keys, const size_t num_keys_space) {
  using KeyType = int64_t;
  using ValueType = size_t;

  std::random_device rd;
  std::mt19937 eng(rd());

  curandGenerator_t gen;
  CURAND_CALL(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));

  HashTable<KeyType, ValueType> hash_table(num_keys_space);

  std::vector<KeyType> h_keys_space(num_keys_space);
  CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned int *>(h_keys_space.data()),
                             num_keys_space * sizeof(KeyType) / 4));

  std::vector<KeyType> h_keys(num_keys);

  KeyType *d_keys;
  CUDA_RT_CALL(cudaMalloc(&d_keys, sizeof(KeyType) * num_keys));
  ValueType *d_values;
  CUDA_RT_CALL(cudaMalloc(&d_values, sizeof(ValueType) * num_keys));

  for (size_t i = 0; i < TEST_ROUNDS; i++) {
    std::shuffle(h_keys_space.begin(), h_keys_space.end(), eng);
    std::copy_n(h_keys_space.begin(), num_keys, h_keys.begin());

    CUDA_RT_CALL(
        cudaMemcpy(d_keys, h_keys.data(), sizeof(KeyType) * num_keys, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaDeviceSynchronize());

    auto start = std::chrono::steady_clock::now();
    hash_table.lookup_or_insert(d_keys, d_values, num_keys);
    CUDA_RT_CALL(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("[%zu] %.2fms %.0fM %.0fM\n", i, diff.count() * 1000,
           1.0f * hash_table.get_size() / 1024 / 1024,
           1.0f * hash_table.get_capacity() / 1024 / 1024);
  }

  CUDA_RT_CALL(cudaFree(d_values));
  CUDA_RT_CALL(cudaFree(d_keys));

  CURAND_CALL(curandDestroyGenerator(gen));
}

TEST(HashTable, InsertOrAssign_Perf) { benchmarkInsertOrAssign(1048576 * 8, 1048576 * 100); }

TEST(HashTable, LookupOrInsert_Perf) { benchmarkLookupOrInsert(1048576 * 8, 1048576 * 100); }
