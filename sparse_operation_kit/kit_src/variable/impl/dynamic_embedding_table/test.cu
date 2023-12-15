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
#include <unordered_map>

#include "dynamic_embedding_table.hpp"
#include "gtest/gtest.h"

#define CURAND_CALL(x)                                      \
  do {                                                      \
    if ((x) != CURAND_STATUS_SUCCESS) {                     \
      printf("Error %d at %s:%d\n", x, __FILE__, __LINE__); \
    }                                                       \
  } while (0)

template <typename T>
static void ASSERT_CONSISTENT(T const *a, T const *b, size_t num, T delta) {
  size_t inconsistent_cnt = 0;
  for (size_t i = 0; i < num; i++) {
    if (fabs(a[i] - (b[i] + delta)) > std::numeric_limits<T>::epsilon()) {
      inconsistent_cnt++;
    }
  }
  if (inconsistent_cnt != 0) {
    FAIL() << inconsistent_cnt << " numbers are inconsistent" << std::endl;
  }
}

void testSubsequentLookup(const size_t dimension, const size_t num_keys) {
  using KeyType = int64_t;
  using ElementType = float;

  const int64_t SPECIAL_KEY = 1075512610888153636;

  std::random_device rd;
  std::mt19937 eng(rd());
  std::uniform_int_distribution<size_t> index_dist(0, num_keys - 1);

  curandGenerator_t gen;
  CURAND_CALL(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));

  DynamicEmbeddingTable<KeyType, ElementType> embedding_table(1, &dimension);
  embedding_table.initialize();

  KeyType *keys;
  CUCO_CUDA_TRY(cudaMallocHost(&keys, sizeof(KeyType) * num_keys));
  ElementType *elements;
  CUCO_CUDA_TRY(cudaMallocHost(&elements, sizeof(ElementType) * dimension * num_keys));

  std::vector<ElementType> special_elements(dimension);

  for (size_t r = 0; r < 10; r++) {
    CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned int *>(keys),
                               num_keys * sizeof(KeyType) / sizeof(unsigned int)));
    size_t index = index_dist(eng);
    keys[index] = SPECIAL_KEY;

    embedding_table.lookup(keys, elements, num_keys, &num_keys);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());

    if (r == 0) {
      std::copy(elements + dimension * index, elements + dimension * (index + 1),
                special_elements.begin());
    } else {
      ASSERT_CONSISTENT(elements + dimension * index, special_elements.data(), dimension, 0.0f);
    }
  }

  CUCO_CUDA_TRY(cudaFreeHost(elements));
  CUCO_CUDA_TRY(cudaFreeHost(keys));

  embedding_table.uninitialize();

  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  CURAND_CALL(curandDestroyGenerator(gen));
}

void testCurrentLookup(const size_t dimension, const size_t num_keys) {
  using KeyType = int64_t;
  using ElementType = float;

  const int64_t SPECIAL_KEY = 1075512610888153636;

  std::random_device rd;
  std::mt19937 eng(rd());
  std::uniform_int_distribution<size_t> index_dist(0, num_keys - 1);

  std::vector<size_t> indices;
  for (size_t r = 0; r < 10; r++) {
    indices.push_back(index_dist(eng));
  }

  curandGenerator_t gen;
  CURAND_CALL(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));

  DynamicEmbeddingTable<KeyType, ElementType> embedding_table(1, &dimension);
  embedding_table.initialize();

  KeyType *keys;
  CUCO_CUDA_TRY(cudaMallocHost(&keys, sizeof(KeyType) * num_keys));
  ElementType *elements;
  CUCO_CUDA_TRY(cudaMallocHost(&elements, sizeof(ElementType) * dimension * num_keys));

  CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned int *>(keys),
                             num_keys * sizeof(KeyType) / sizeof(unsigned int)));

  for (size_t r = 0; r < 10; r++) {
    size_t index = indices[r];
    keys[index] = SPECIAL_KEY;
  }

  embedding_table.lookup(keys, elements, num_keys, &num_keys);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  for (size_t r = 1; r < 10; r++) {
    size_t index0 = indices[0];
    size_t index1 = indices[r];

    ASSERT_CONSISTENT(elements + dimension * index0, elements + dimension * index1, dimension,
                      0.0f);
  }

  CUCO_CUDA_TRY(cudaFreeHost(keys));
  CUCO_CUDA_TRY(cudaFreeHost(elements));

  embedding_table.uninitialize();

  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  CURAND_CALL(curandDestroyGenerator(gen));
}

void testCurrentLookupMultipleClasses(const size_t dimension, const size_t num_keys) {
  using KeyType = int64_t;
  using ElementType = float;

  const int64_t SPECIAL_KEY = 1075512610888153636;

  std::random_device rd;
  std::mt19937 eng(rd());
  std::uniform_int_distribution<size_t> index_dist(0, num_keys / 2 - 1);

  std::vector<size_t> left_indices;
  for (size_t r = 0; r < 10; r++) {
    left_indices.push_back(index_dist(eng));
  }

  std::vector<size_t> right_indices;
  for (size_t r = 0; r < 10; r++) {
    right_indices.push_back(index_dist(eng) + num_keys / 2);
  }

  curandGenerator_t gen;
  CURAND_CALL(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));

  size_t dimensions[] = {dimension, dimension};
  DynamicEmbeddingTable<KeyType, ElementType> embedding_table(2, dimensions);
  embedding_table.initialize();

  KeyType *keys;
  CUCO_CUDA_TRY(cudaMallocHost(&keys, sizeof(KeyType) * num_keys));
  ElementType *elements;
  CUCO_CUDA_TRY(cudaMallocHost(&elements, sizeof(ElementType) * dimension * num_keys));

  CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned int *>(keys),
                             num_keys * sizeof(KeyType) / sizeof(unsigned int)));

  for (size_t r = 0; r < 10; r++) {
    size_t index;
    index = left_indices[r];
    keys[index] = SPECIAL_KEY;
    index = right_indices[r];
    keys[index] = SPECIAL_KEY;
  }

  const size_t num_key_per_class[] = {num_keys / 2, num_keys / 2};
  embedding_table.lookup(keys, elements, num_keys, num_key_per_class);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  for (size_t r = 1; r < 10; r++) {
    size_t index0, index1;

    index0 = left_indices[0];
    index1 = left_indices[r];

    ASSERT_CONSISTENT(elements + dimension * index0, elements + dimension * index1, dimension,
                      0.0f);

    index0 = right_indices[0];
    index1 = right_indices[r];

    ASSERT_CONSISTENT(elements + dimension * index0, elements + dimension * index1, dimension,
                      0.0f);
  }

  CUCO_CUDA_TRY(cudaFreeHost(keys));
  CUCO_CUDA_TRY(cudaFreeHost(elements));

  embedding_table.uninitialize();

  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  CURAND_CALL(curandDestroyGenerator(gen));
}

void testSubsequentUpdate(const size_t dimension, const size_t num_keys) {
  using KeyType = int64_t;
  using ElementType = float;

  const int64_t SPECIAL_KEY = 1075512610888153636;

  std::random_device rd;
  std::mt19937 eng(rd());
  std::uniform_int_distribution<size_t> index_dist(0, num_keys - 1);

  curandGenerator_t gen;
  CURAND_CALL(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));

  DynamicEmbeddingTable<KeyType, ElementType> embedding_table(1, &dimension);
  embedding_table.initialize();

  KeyType *keys;
  CUCO_CUDA_TRY(cudaMallocHost(&keys, sizeof(KeyType) * num_keys));
  ElementType *elements;
  CUCO_CUDA_TRY(cudaMallocHost(&elements, sizeof(ElementType) * dimension * num_keys));

  std::vector<ElementType> special_elements(dimension);

  {
    CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned int *>(keys),
                               num_keys * sizeof(KeyType) / sizeof(unsigned int)));

    size_t index = index_dist(eng);
    keys[index] = SPECIAL_KEY;

    embedding_table.lookup(keys, elements, num_keys, &num_keys);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());

    std::copy(elements + dimension * index, elements + dimension * (index + 1),
              special_elements.begin());
  }

  for (size_t r = 0; r < 10; r++) {
    CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned int *>(keys),
                               num_keys * sizeof(KeyType) / sizeof(unsigned int)));
    size_t index = index_dist(eng);
    keys[index] = SPECIAL_KEY;

    for (size_t i = 0; i < dimension * num_keys; i++) {
      elements[i] = 0.125f;
    }

    embedding_table.scatter_add(keys, elements, num_keys, &num_keys);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());

    CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned int *>(keys),
                               num_keys * sizeof(KeyType) / sizeof(unsigned int)));
    index = index_dist(eng);
    keys[index] = SPECIAL_KEY;

    embedding_table.lookup(keys, elements, num_keys, &num_keys);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());

    ASSERT_CONSISTENT(elements + dimension * index, special_elements.data(), dimension,
                      0.125f * (r + 1));
  }

  CUCO_CUDA_TRY(cudaFreeHost(keys));
  CUCO_CUDA_TRY(cudaFreeHost(elements));

  embedding_table.uninitialize();

  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  CURAND_CALL(curandDestroyGenerator(gen));
}

void testCurrentUpdate(const size_t dimension, const size_t num_keys) {
  using KeyType = int64_t;
  using ElementType = float;

  const int64_t SPECIAL_KEY = 1075512610888153636;

  std::random_device rd;
  std::mt19937 eng(rd());
  std::uniform_int_distribution<size_t> index_dist(0, num_keys - 1);

  curandGenerator_t gen;
  CURAND_CALL(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));

  std::vector<size_t> indices;
  for (size_t r = 0; r < 10; r++) {
    indices.push_back(index_dist(eng));
  }

  DynamicEmbeddingTable<KeyType, ElementType> embedding_table(1, &dimension);
  embedding_table.initialize();

  KeyType *keys;
  CUCO_CUDA_TRY(cudaMallocHost(&keys, sizeof(KeyType) * num_keys));
  ElementType *elements;
  CUCO_CUDA_TRY(cudaMallocHost(&elements, sizeof(ElementType) * dimension * num_keys));

  std::vector<ElementType> special_elements(dimension);

  CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned int *>(keys),
                             num_keys * sizeof(KeyType) / sizeof(unsigned int)));
  size_t index = index_dist(eng);
  keys[index] = SPECIAL_KEY;

  embedding_table.lookup(keys, elements, num_keys, &num_keys);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  std::copy(elements + dimension * index, elements + dimension * (index + 1),
            special_elements.begin());

  CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned int *>(keys),
                             num_keys * sizeof(KeyType) / sizeof(unsigned int)));

  for (size_t r = 0; r < 10; r++) {
    size_t index = indices[r];
    keys[index] = SPECIAL_KEY;
  }

  for (size_t i = 0; i < dimension * num_keys; i++) {
    elements[i] = 0.125f;
  }

  embedding_table.scatter_add(keys, elements, num_keys, &num_keys);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned int *>(keys),
                             num_keys * sizeof(KeyType) / sizeof(unsigned int)));
  for (size_t r = 0; r < 10; r++) {
    index = indices[r];
    keys[index] = SPECIAL_KEY;
  }

  embedding_table.lookup(keys, elements, num_keys, &num_keys);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  for (size_t r = 0; r < 10; r++) {
    index = indices[r];

    ASSERT_CONSISTENT(elements + dimension * index, special_elements.data(), dimension,
                      0.125f * 10);
  }

  CUCO_CUDA_TRY(cudaFreeHost(keys));
  CUCO_CUDA_TRY(cudaFreeHost(elements));

  embedding_table.uninitialize();

  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  CURAND_CALL(curandDestroyGenerator(gen));
}

void testRemove(const size_t dimension, const size_t num_keys) {
  using KeyType = int64_t;
  using ElementType = float;

  const size_t num_keys_space = num_keys * 10;

  std::random_device rd;
  std::mt19937 eng(rd());

  curandGenerator_t gen;
  CURAND_CALL(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));

  DynamicEmbeddingTable<KeyType, ElementType> embedding_table(1, &dimension);
  std::set<KeyType> set;

  embedding_table.initialize();

  KeyType *keys_space;
  CUCO_CUDA_TRY(cudaMallocHost(&keys_space, sizeof(KeyType) * num_keys_space));
  CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned int *>(keys_space),
                             num_keys_space * sizeof(KeyType) / sizeof(unsigned int)));

  KeyType *keys;
  CUCO_CUDA_TRY(cudaMallocHost(&keys, sizeof(KeyType) * num_keys));
  ElementType *elements;
  CUCO_CUDA_TRY(cudaMallocHost(&elements, sizeof(ElementType) * dimension * num_keys));

  for (size_t r = 0; r < 10; r++) {
    std::shuffle(keys_space, keys_space + num_keys_space, eng);
    std::copy_n(keys_space, num_keys, keys);

    for (size_t i = 0; i < num_keys; i++) {
      set.insert(keys[i]);
    }
    embedding_table.lookup(keys, elements, num_keys, &num_keys);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());

    std::shuffle(keys_space, keys_space + num_keys_space, eng);
    std::copy_n(keys_space, num_keys, keys);

    for (size_t i = 0; i < num_keys; i++) {
      set.erase(keys[i]);
    }

    embedding_table.remove(keys, num_keys, &num_keys);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());

    ASSERT_EQ(set.size(), embedding_table.size());
  }

  embedding_table.uninitialize();

  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  CUCO_CUDA_TRY(cudaFreeHost(keys_space));
  CUCO_CUDA_TRY(cudaFreeHost(elements));
  CUCO_CUDA_TRY(cudaFreeHost(keys));

  CURAND_CALL(curandDestroyGenerator(gen));
}

void testExport(const size_t dimension, const size_t num_keys) {
  using KeyType = int64_t;
  using ElementType = float;

  std::random_device rd;

  curandGenerator_t gen;
  CURAND_CALL(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));

  DynamicEmbeddingTable<KeyType, ElementType> embedding_table(1, &dimension);

  embedding_table.initialize();

  KeyType *special_keys;
  CUCO_CUDA_TRY(cudaMallocHost(&special_keys, sizeof(KeyType) * num_keys));
  ElementType *special_elements;
  CUCO_CUDA_TRY(cudaMallocHost(&special_elements, sizeof(ElementType) * dimension * num_keys));

  CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned int *>(special_keys),
                             num_keys * sizeof(KeyType) / sizeof(unsigned int)));

  embedding_table.lookup(special_keys, special_elements, num_keys, &num_keys);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  KeyType *keys;
  CUCO_CUDA_TRY(cudaMallocHost(&keys, sizeof(KeyType) * num_keys));
  ElementType *elements;
  CUCO_CUDA_TRY(cudaMallocHost(&elements, sizeof(ElementType) * dimension * num_keys));

  for (size_t r = 0; r < 5; r++) {
    CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned int *>(keys),
                               num_keys * sizeof(KeyType) / sizeof(unsigned int)));
    embedding_table.lookup(keys, elements, num_keys, &num_keys);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());
  }

  size_t full_num_keys = embedding_table.size();
  KeyType *full_keys;
  CUCO_CUDA_TRY(cudaMallocHost(&full_keys, sizeof(KeyType) * full_num_keys));
  ElementType *full_elements;
  CUCO_CUDA_TRY(cudaMallocHost(&full_elements, sizeof(ElementType) * dimension * full_num_keys));

  embedding_table.eXport(0, full_keys, full_elements, full_num_keys);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  std::unordered_map<KeyType, size_t> map;
  for (size_t i = 0; i < full_num_keys; i++) {
    map.emplace(full_keys[i], i);
  }

  for (size_t i = 0; i < num_keys; i++) {
    KeyType key0 = special_keys[i];
    auto it = map.find(key0);
    ASSERT_NE(it, map.end());

    ASSERT_CONSISTENT(special_elements + dimension * i, full_elements + dimension * it->second,
                      dimension, 0.0f);
  }

  embedding_table.uninitialize();

  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  CUCO_CUDA_TRY(cudaFreeHost(full_elements));
  CUCO_CUDA_TRY(cudaFreeHost(full_keys));
  CUCO_CUDA_TRY(cudaFreeHost(elements));
  CUCO_CUDA_TRY(cudaFreeHost(keys));
  CUCO_CUDA_TRY(cudaFreeHost(special_elements));
  CUCO_CUDA_TRY(cudaFreeHost(special_keys));

  CURAND_CALL(curandDestroyGenerator(gen));
}

TEST(DynamicEmbeddingTable, SubsequentLookup_4_1M) { testSubsequentLookup(4, 1048576); }
TEST(DynamicEmbeddingTable, SubsequentLookup_8_1M) { testSubsequentLookup(8, 1048576); }
TEST(DynamicEmbeddingTable, SubsequentLookup_16_1M) { testSubsequentLookup(16, 1048576); }
TEST(DynamicEmbeddingTable, SubsequentLookup_32_1M) { testSubsequentLookup(32, 1048576); }
TEST(DynamicEmbeddingTable, SubsequentLookup_64_1M) { testSubsequentLookup(64, 1048576); }
TEST(DynamicEmbeddingTable, SubsequentLookup_128_1M) { testSubsequentLookup(128, 1048576); }
TEST(DynamicEmbeddingTable, SubsequentLookup_256_512K) { testSubsequentLookup(256, 524288); }
TEST(DynamicEmbeddingTable, SubsequentLookup_512_256K) { testSubsequentLookup(512, 262144); }
TEST(DynamicEmbeddingTable, SubsequentLookup_1024_128K) { testSubsequentLookup(1024, 131072); }
TEST(DynamicEmbeddingTable, CurrentLookup_4_1M) { testCurrentLookup(4, 1048576); }
TEST(DynamicEmbeddingTable, CurrentLookup_8_1M) { testCurrentLookup(8, 1048576); }
TEST(DynamicEmbeddingTable, CurrentLookup_16_1M) { testCurrentLookup(16, 1048576); }
TEST(DynamicEmbeddingTable, CurrentLookup_32_1M) { testCurrentLookup(32, 1048576); }
TEST(DynamicEmbeddingTable, CurrentLookup_64_1M) { testCurrentLookup(64, 1048576); }
TEST(DynamicEmbeddingTable, CurrentLookup_128_1M) { testCurrentLookup(128, 1048576); }
TEST(DynamicEmbeddingTable, CurrentLookup_256_512K) { testCurrentLookup(256, 524288); }
TEST(DynamicEmbeddingTable, CurrentLookup_512_256K) { testCurrentLookup(512, 262144); }
TEST(DynamicEmbeddingTable, CurrentLookup_1024_128K) { testCurrentLookup(1024, 131072); }
TEST(DynamicEmbeddingTable, CurrentLookup_1024_32K) { testCurrentLookup(1024, 32768); }
TEST(DynamicEmbeddingTable, CurrentLookup_1024_8K) { testCurrentLookup(1024, 8192); }
TEST(DynamicEmbeddingTable, CurrentLookup_1024_2K) { testCurrentLookup(1024, 2048); }
TEST(DynamicEmbeddingTable, CurrentLookup_1024_512) { testCurrentLookup(1024, 512); }
TEST(DynamicEmbeddingTable, CurrentLookup_1024_128) { testCurrentLookup(1024, 128); }
TEST(DynamicEmbeddingTable, CurrentLookup_1024_32) { testCurrentLookup(1024, 32); }
TEST(DynamicEmbeddingTable, CurrentLookupMultipleClasses_4_1M) {
  testCurrentLookupMultipleClasses(4, 1048576);
}
TEST(DynamicEmbeddingTable, CurrentLookupMultipleClasses_8_1M) {
  testCurrentLookupMultipleClasses(8, 1048576);
}
TEST(DynamicEmbeddingTable, CurrentLookupMultipleClasses_16_1M) {
  testCurrentLookupMultipleClasses(16, 1048576);
}
TEST(DynamicEmbeddingTable, CurrentLookupMultipleClasses_32_1M) {
  testCurrentLookupMultipleClasses(32, 1048576);
}
TEST(DynamicEmbeddingTable, CurrentLookupMultipleClasses_64_1M) {
  testCurrentLookupMultipleClasses(64, 1048576);
}
TEST(DynamicEmbeddingTable, CurrentLookupMultipleClasses_128_1M) {
  testCurrentLookupMultipleClasses(128, 1048576);
}
TEST(DynamicEmbeddingTable, CurrentLookupMultipleClasses_256_512K) {
  testCurrentLookupMultipleClasses(256, 524288);
}
TEST(DynamicEmbeddingTable, CurrentLookupMultipleClasses_512_256K) {
  testCurrentLookupMultipleClasses(512, 262144);
}
TEST(DynamicEmbeddingTable, CurrentLookupMultipleClasses_1024_128K) {
  testCurrentLookupMultipleClasses(1024, 131072);
}
TEST(DynamicEmbeddingTable, SubsequentUpdate_4_1M) { testSubsequentUpdate(4, 1048576); }
TEST(DynamicEmbeddingTable, SubsequentUpdate_8_1M) { testSubsequentUpdate(8, 1048576); }
TEST(DynamicEmbeddingTable, SubsequentUpdate_16_1M) { testSubsequentUpdate(16, 1048576); }
TEST(DynamicEmbeddingTable, SubsequentUpdate_32_1M) { testSubsequentUpdate(32, 1048576); }
TEST(DynamicEmbeddingTable, SubsequentUpdate_64_1M) { testSubsequentUpdate(64, 1048576); }
TEST(DynamicEmbeddingTable, SubsequentUpdate_128_1M) { testSubsequentUpdate(128, 1048576); }
TEST(DynamicEmbeddingTable, SubsequentUpdate_256_512K) { testSubsequentUpdate(256, 524288); }
TEST(DynamicEmbeddingTable, SubsequentUpdate_512_256K) { testSubsequentUpdate(512, 262144); }
TEST(DynamicEmbeddingTable, SubsequentUpdate_1024_128K) { testSubsequentUpdate(1024, 131072); }
TEST(DynamicEmbeddingTable, CurrentUpdate_4_1M) { testCurrentUpdate(4, 1048576); }
TEST(DynamicEmbeddingTable, CurrentUpdate_8_1M) { testCurrentUpdate(8, 1048576); }
TEST(DynamicEmbeddingTable, CurrentUpdate_16_1M) { testCurrentUpdate(16, 1048576); }
TEST(DynamicEmbeddingTable, CurrentUpdate_32_1M) { testCurrentUpdate(32, 1048576); }
TEST(DynamicEmbeddingTable, CurrentUpdate_64_1M) { testCurrentUpdate(64, 1048576); }
TEST(DynamicEmbeddingTable, CurrentUpdate_128_1M) { testCurrentUpdate(128, 1048576); }
TEST(DynamicEmbeddingTable, CurrentUpdate_256_512K) { testCurrentUpdate(256, 524288); }
TEST(DynamicEmbeddingTable, CurrentUpdate_512_256K) { testCurrentUpdate(512, 262144); }
TEST(DynamicEmbeddingTable, CurrentUpdate_1024_128K) { testCurrentUpdate(1024, 131072); }
TEST(DynamicEmbeddingTable, Remove_32_1M) { testRemove(32, 1048576); }
TEST(DynamicEmbeddingTable, Export_32_1M) { testExport(32, 1048576); }
