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
#include <random>

#include "dynamic_embedding_table.hpp"
#include "gtest/gtest.h"

#define CURAND_CALL(x)                                      \
  do {                                                      \
    if ((x) != CURAND_STATUS_SUCCESS) {                     \
      printf("Error %d at %s:%d\n", x, __FILE__, __LINE__); \
    }                                                       \
  } while (0)

#define FULL_INITIAL_CAPACITY 0
#define TEST_ROUNDS 50

void benchmarkLookup(const size_t dimension, const size_t num_keys, const size_t num_keys_space) {
  using KeyType = int64_t;
  using ValueType = float;

  std::random_device rd;
  std::mt19937 eng(rd());

  curandGenerator_t gen;
  CURAND_CALL(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));

  DynamicEmbeddingTable<KeyType, ValueType> embedding_table(1, &dimension, ""
#if FULL_INITIAL_CAPACITY == 1
                                                            ,
                                                            num_keys_space
#endif
  );

  embedding_table.initialize();

  std::vector<KeyType> h_keys_space(num_keys_space);
  CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned int *>(h_keys_space.data()),
                             num_keys_space * sizeof(KeyType) / 4));

  std::vector<KeyType> h_keys(num_keys);

  KeyType *d_keys;
  CUCO_CUDA_TRY(cudaMalloc(&d_keys, sizeof(KeyType) * num_keys));
  ValueType *d_values;
  CUCO_CUDA_TRY(cudaMalloc(&d_values, sizeof(ValueType) * dimension * num_keys));

  for (size_t i = 0; i < TEST_ROUNDS; i++) {
    std::shuffle(h_keys_space.begin(), h_keys_space.end(), eng);
    std::copy_n(h_keys_space.begin(), num_keys, h_keys.begin());

    CUCO_CUDA_TRY(
        cudaMemcpy(d_keys, h_keys.data(), sizeof(KeyType) * num_keys, cudaMemcpyHostToDevice));
    CUCO_CUDA_TRY(cudaDeviceSynchronize());

    auto start = std::chrono::steady_clock::now();
    embedding_table.lookup(d_keys, d_values, num_keys, &num_keys);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("[%zu] %.2fms %.0fM %.0fM\n", i, diff.count() * 1000,
           1.0f * embedding_table.size() / 1024 / 1024,
           1.0f * embedding_table.capacity() / 1024 / 1024);
  }

  embedding_table.uninitialize();

  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  CUCO_CUDA_TRY(cudaFree(d_values));
  CUCO_CUDA_TRY(cudaFree(d_keys));

  CURAND_CALL(curandDestroyGenerator(gen));
}

void benchmarkScatterAdd(const size_t dimension, const size_t num_keys,
                         const size_t num_keys_space) {
  using KeyType = int64_t;
  using ValueType = float;

  std::random_device rd;
  std::mt19937 eng(rd());

  curandGenerator_t gen;
  CURAND_CALL(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));

  DynamicEmbeddingTable<KeyType, ValueType> embedding_table(1, &dimension, ""
#if FULL_INITIAL_CAPACITY == 1
                                                            ,
                                                            num_keys_space
#endif
  );

  embedding_table.initialize();

  std::vector<KeyType> h_keys_space(num_keys_space);
  CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned int *>(h_keys_space.data()),
                             num_keys_space * sizeof(KeyType) / 4));

  std::vector<KeyType> h_keys(num_keys);

  KeyType *d_keys_space;
  CUCO_CUDA_TRY(cudaMalloc(&d_keys_space, sizeof(KeyType) * num_keys_space));
  ValueType *d_keys_space_values;
  CUCO_CUDA_TRY(cudaMalloc(&d_keys_space_values, sizeof(ValueType) * dimension * num_keys_space));
  KeyType *d_keys;
  CUCO_CUDA_TRY(cudaMalloc(&d_keys, sizeof(KeyType) * num_keys));
  ValueType *d_values;
  CUCO_CUDA_TRY(cudaMalloc(&d_values, sizeof(ValueType) * dimension * num_keys));

  CUCO_CUDA_TRY(cudaMemcpy(d_keys_space, h_keys_space.data(), sizeof(KeyType) * num_keys_space,
                           cudaMemcpyHostToDevice));

  size_t half_num_keys_space = num_keys_space / 2;
  embedding_table.lookup(d_keys_space, d_keys_space_values, half_num_keys_space,
                         &half_num_keys_space);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  for (size_t i = 0; i < TEST_ROUNDS; i++) {
    std::shuffle(h_keys_space.begin(), h_keys_space.end(), eng);
    std::copy_n(h_keys_space.begin(), num_keys, h_keys.begin());

    CUCO_CUDA_TRY(
        cudaMemcpy(d_keys, h_keys.data(), sizeof(KeyType) * num_keys, cudaMemcpyHostToDevice));
    CUCO_CUDA_TRY(cudaDeviceSynchronize());

    auto start = std::chrono::steady_clock::now();
    embedding_table.scatter_add(d_keys, d_values, num_keys, &num_keys);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("[%zu] %.2fms %.0fM %.0fM\n", i, diff.count() * 1000,
           1.0f * embedding_table.size() / 1024 / 1024,
           1.0f * embedding_table.capacity() / 1024 / 1024);
  }

  embedding_table.uninitialize();

  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  CUCO_CUDA_TRY(cudaFree(d_values));
  CUCO_CUDA_TRY(cudaFree(d_keys));
  CUCO_CUDA_TRY(cudaFree(d_keys_space_values));
  CUCO_CUDA_TRY(cudaFree(d_keys_space));

  CURAND_CALL(curandDestroyGenerator(gen));
}

void benchmarkRemove(const size_t dimension, const size_t num_keys, const size_t num_keys_space) {
  using KeyType = int64_t;
  using ValueType = float;

  std::random_device rd;
  std::mt19937 eng(rd());

  curandGenerator_t gen;
  CURAND_CALL(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));

  DynamicEmbeddingTable<KeyType, ValueType> embedding_table(1, &dimension, ""
#if FULL_INITIAL_CAPACITY == 1
                                                            ,
                                                            num_keys_space
#endif
  );

  embedding_table.initialize();

  std::vector<KeyType> h_keys_space(num_keys_space);
  CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned int *>(h_keys_space.data()),
                             num_keys_space * sizeof(KeyType) / 4));

  std::vector<KeyType> h_keys(num_keys);

  KeyType *d_keys_space;
  CUCO_CUDA_TRY(cudaMalloc(&d_keys_space, sizeof(KeyType) * num_keys_space));
  ValueType *d_keys_space_values;
  CUCO_CUDA_TRY(cudaMalloc(&d_keys_space_values, sizeof(ValueType) * dimension * num_keys_space));
  KeyType *d_keys;
  CUCO_CUDA_TRY(cudaMalloc(&d_keys, sizeof(KeyType) * num_keys));

  CUCO_CUDA_TRY(cudaMemcpy(d_keys_space, h_keys_space.data(), sizeof(KeyType) * num_keys_space,
                           cudaMemcpyHostToDevice));

  size_t half_num_keys_space = num_keys_space / 2;
  embedding_table.lookup(d_keys_space, d_keys_space_values, half_num_keys_space,
                         &half_num_keys_space);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  for (size_t i = 0; i < TEST_ROUNDS; i++) {
    std::shuffle(h_keys_space.begin(), h_keys_space.end(), eng);
    std::copy_n(h_keys_space.begin(), num_keys, h_keys.begin());

    CUCO_CUDA_TRY(
        cudaMemcpy(d_keys, h_keys.data(), sizeof(KeyType) * num_keys, cudaMemcpyHostToDevice));
    CUCO_CUDA_TRY(cudaDeviceSynchronize());

    auto start = std::chrono::steady_clock::now();
    embedding_table.remove(d_keys, num_keys, &num_keys);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("[%zu] %.2fms %.0fM %.0fM\n", i, diff.count() * 1000,
           1.0f * embedding_table.size() / 1024 / 1024,
           1.0f * embedding_table.capacity() / 1024 / 1024);
  }

  embedding_table.uninitialize();

  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  CUCO_CUDA_TRY(cudaFree(d_keys));
  CUCO_CUDA_TRY(cudaFree(d_keys_space_values));
  CUCO_CUDA_TRY(cudaFree(d_keys_space));

  CURAND_CALL(curandDestroyGenerator(gen));
}

void benchmarkLookupAndRemoveAlternately(const size_t dimension, const size_t num_keys,
                                         const size_t num_keys_space) {
  using KeyType = int64_t;
  using ValueType = float;

  std::random_device rd;
  std::mt19937 eng1(rd());
  std::mt19937 eng2(rd());

  curandGenerator_t gen;
  CURAND_CALL(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));

  DynamicEmbeddingTable<KeyType, ValueType> embedding_table(1, &dimension, ""
#if FULL_INITIAL_CAPACITY == 1
                                                            ,
                                                            num_keys_space
#endif
  );

  embedding_table.initialize();

  std::vector<KeyType> h_keys_space(num_keys_space);
  CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned int *>(h_keys_space.data()),
                             num_keys_space * sizeof(KeyType) / 4));

  std::vector<KeyType> h_keys(num_keys);

  KeyType *d_keys;
  CUCO_CUDA_TRY(cudaMalloc(&d_keys, sizeof(KeyType) * num_keys));
  ValueType *d_values;
  CUCO_CUDA_TRY(cudaMalloc(&d_values, sizeof(ValueType) * dimension * num_keys));

  for (size_t i = 0; i < TEST_ROUNDS / 10; i++) {
    for (size_t j = 0; j < 10; j++) {
      std::shuffle(h_keys_space.begin(), h_keys_space.end(), eng1);
      std::copy_n(h_keys_space.begin(), num_keys, h_keys.begin());

      CUCO_CUDA_TRY(
          cudaMemcpy(d_keys, h_keys.data(), sizeof(KeyType) * num_keys, cudaMemcpyHostToDevice));
      CUCO_CUDA_TRY(cudaDeviceSynchronize());

      auto start = std::chrono::steady_clock::now();
      embedding_table.lookup(d_keys, d_values, num_keys, &num_keys);
      CUCO_CUDA_TRY(cudaDeviceSynchronize());
      auto end = std::chrono::steady_clock::now();
      std::chrono::duration<double> diff = end - start;
      printf("LOOKUP [%zu] %.2fms %.0fM %.0fM\n", i * 10 + j, diff.count() * 1000,
             1.0f * embedding_table.size() / 1024 / 1024,
             1.0f * embedding_table.capacity() / 1024 / 1024);
    }
    for (size_t j = 0; j < 10; j++) {
      std::shuffle(h_keys_space.begin(), h_keys_space.end(), eng2);
      std::copy_n(h_keys_space.begin(), num_keys, h_keys.begin());

      CUCO_CUDA_TRY(
          cudaMemcpy(d_keys, h_keys.data(), sizeof(KeyType) * num_keys, cudaMemcpyHostToDevice));
      CUCO_CUDA_TRY(cudaDeviceSynchronize());

      auto start = std::chrono::steady_clock::now();
      embedding_table.remove(d_keys, num_keys, &num_keys);
      CUCO_CUDA_TRY(cudaDeviceSynchronize());
      auto end = std::chrono::steady_clock::now();
      std::chrono::duration<double> diff = end - start;
      printf("REMOVE [%zu] %.2fms %.0fM %.0fM\n", i * 10 + j, diff.count() * 1000,
             1.0f * embedding_table.size() / 1024 / 1024,
             1.0f * embedding_table.capacity() / 1024 / 1024);
    }
  }

  embedding_table.uninitialize();

  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  CUCO_CUDA_TRY(cudaFree(d_values));
  CUCO_CUDA_TRY(cudaFree(d_keys));

  CURAND_CALL(curandDestroyGenerator(gen));
}

TEST(DynamicEmbeddingTable, Lookup_Perf) { benchmarkLookup(4, 1048576, 1048576 * 50); }
TEST(DynamicEmbeddingTable, ScatterAdd_Perf) { benchmarkScatterAdd(4, 1048576, 1048576 * 50); }
TEST(DynamicEmbeddingTable, Remove_Perf) { benchmarkRemove(4, 1048576, 1048576 * 50); }
TEST(DynamicEmbeddingTable, benchmarkLookupAndRemoveAlternately_Perf) {
  benchmarkLookupAndRemoveAlternately(4, 1048576 * 4, 1048576 * 50);
}