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

#pragma once

#include <stdio.h>

#include <cmath>
#include <fstream>
#include <type_traits>
#include <unordered_set>

#include "HugeCTR/include/common.hpp"
#include "utest/embedding/sparse_embedding_hash_cpu.hpp"

namespace HugeCTR {

namespace embedding_test {

inline bool compare_element(float a, float b, float epsilon) {
  // compare absolute error
  if (fabs(a - b) < epsilon) return true;

  // compare relative error
  if (fabs(a) >= fabs(b))
    if (fabs((a - b) / a) < epsilon)
      return true;
    else
      return false;
  else if (fabs((a - b) / b) < epsilon)
    return true;
  else
    return false;
}

inline bool compare_array(size_t len, const float *a, const float *b, float epsilon) {
  for (size_t i = 0; i < len; i++) {
    if (!compare_element(a[i], b[i], epsilon)) {
      printf("Error in compare_array: i=%zu, a=%.8f, b=%.8f\n", i, a[i], b[i]);
      return false;
    }
  }

  return true;
}

// overload for fp16 on GPU
inline bool compare_array(size_t len, const __half *a, const __half *b, float epsilon) {
  for (size_t i = 0; i < len; i++) {
    float fa = __half2float(a[i]);
    float fb = __half2float(b[i]);
    if (!compare_element(fa, fb, epsilon)) {
      printf("Error in compare_array: i=%zu, a=%.8f, b=%.8f\n", i, fa, fb);
      return false;
    }
  }

  return true;
}

template <typename T>
bool compare_file(std::string file1, std::string file2, float epsilon) {
  std::ifstream file_stream1(file1);
  std::ifstream file_stream2(file2);

  if (!file_stream1.is_open() || !file_stream2.is_open()) {
    ERROR_MESSAGE_("Error: file open failed");
    return false;
  }

  long long start_pos = file_stream1.tellg();
  file_stream1.seekg(0, file_stream1.end);
  long long end_pos = file_stream1.tellg();
  long long file_size1 = end_pos - start_pos;

  file_stream2.seekg(0, file_stream1.beg);
  start_pos = file_stream2.tellg();
  file_stream2.seekg(0, file_stream2.end);
  long long file_size2 = end_pos - start_pos;

  if (file_size1 != file_size2) {
    ERROR_MESSAGE_("Error: files size is not same");
    file_stream1.close();
    file_stream2.close();
    return false;
  }

  file_stream1.seekg(0, file_stream1.beg);
  file_stream2.seekg(0, file_stream2.beg);

  bool rtn = true;
  while (file_stream1.peek() != EOF) {
    T val1, val2;
    file_stream1.read((char *)&val1, sizeof(T));
    file_stream2.read((char *)&val2, sizeof(T));
    if (!compare_element(val1, val2, epsilon)) {
      rtn = false;
      break;
    }
  }

  file_stream1.close();
  file_stream2.close();

  return rtn;
}

// hash table files have same keys and values, but they may be unordered
template <typename TypeHashKey, typename TypeHashValue>
bool compare_distributed_hash_table_files(std::string file1, std::string file2, float epsilon) {
  bool rtn = true;

  std::ifstream file_stream1(file1);
  std::ifstream file_stream2(file2);

  if (!file_stream1.is_open() || !file_stream2.is_open()) {
    ERROR_MESSAGE_("Error: file open failed");
    return false;
  }

  long long start_pos = file_stream1.tellg();
  file_stream1.seekg(0, file_stream1.end);
  long long end_pos = file_stream1.tellg();
  long long file_size1 = end_pos - start_pos;

  file_stream2.seekg(0, file_stream2.beg);
  start_pos = file_stream2.tellg();
  file_stream2.seekg(0, file_stream2.end);
  long long file_size2 = end_pos - start_pos;

  if (file_size1 != file_size2) {
    ERROR_MESSAGE_("Error: files size is not same");
    file_stream1.close();
    file_stream2.close();
    return false;
  }

  file_stream1.seekg(0, file_stream1.beg);
  file_stream2.seekg(0, file_stream2.beg);

  size_t pair_size_in_B = sizeof(TypeHashKey) + sizeof(TypeHashValue);
  long long pair_num = file_size1 / pair_size_in_B;

  // CAUSION: file_stream1 is ordered, while file_stream2 is unordered
  // So, firstly, we read <key,value> pairs from file_stream2, and insert it into a hash table.
  char *buf = (char *)malloc(pair_size_in_B);
  TypeHashKey *key;
  TypeHashValue *value;
  HashTableCpu<TypeHashKey, TypeHashValue> *hash_table =
      new HashTableCpu<TypeHashKey, TypeHashValue>();
  while (file_stream2.peek() != EOF) {
    file_stream2.read(buf, pair_size_in_B);
    key = (TypeHashKey *)buf;
    value = (TypeHashValue *)(buf + sizeof(TypeHashKey));
    hash_table->insert(key, value, 1);
  }
  file_stream2.close();

  if (hash_table->get_size() != pair_num) {
    ERROR_MESSAGE_(
        "Error: The number of <key,value> pair inserting into hash table is not equal to hash "
        "table file size\n");
    return false;
  }

  // Then, we read <key,value1> pairs from file_stream1, and get(key,value2) from hash table, and
  // compare value1 and value2.
  TypeHashValue *value1;
  TypeHashValue *value2 = (TypeHashValue *)malloc(sizeof(TypeHashValue));
  size_t value_len = sizeof(TypeHashValue) / sizeof(float);
  while (file_stream1.peek() != EOF) {
    file_stream1.read(buf, pair_size_in_B);
    key = (TypeHashKey *)buf;
    value1 = (TypeHashValue *)(buf + sizeof(TypeHashKey));
    hash_table->get(key, value2, 1);

    if (!compare_array(value_len, (float *)value1, (float *)value2, epsilon)) {
      rtn = false;
      break;
    }
  }
  file_stream1.close();

  free(value2);

  return rtn;
}

// hash table files have same keys and values, but they may be unordered
template <typename TypeHashKey, typename TypeSlotId, typename TypeHashValue>
bool compare_localized_hash_table_files(std::string file1, std::string file2, float epsilon) {
  bool rtn = true;

  std::ifstream file_stream1(file1);
  std::ifstream file_stream2(file2);

  if (!file_stream1.is_open() || !file_stream2.is_open()) {
    ERROR_MESSAGE_("Error: file open failed");
    return false;
  }

  long long start_pos = file_stream1.tellg();
  file_stream1.seekg(0, file_stream1.end);
  long long end_pos = file_stream1.tellg();
  long long file_size1 = end_pos - start_pos;

  start_pos = file_stream2.tellg();
  file_stream2.seekg(0, file_stream2.end);
  end_pos = file_stream1.tellg();
  long long file_size2 = end_pos - start_pos;

  if (file_size1 != file_size2) {
    ERROR_MESSAGE_("Error: files size is not same");
    std::cout << "file_size1=" << file_size1 << ", file_size2=" << file_size2 << std::endl;
    file_stream1.close();
    file_stream2.close();
    return false;
  }

  file_stream1.seekg(0, file_stream1.beg);
  file_stream2.seekg(0, file_stream2.beg);

  size_t pair_size_in_B = sizeof(TypeHashKey) + sizeof(TypeSlotId) + sizeof(TypeHashValue);
  size_t pair_num = file_size1 / pair_size_in_B;

#ifndef NDEBUG
  std::cout << "pair_size_in_B=" << pair_size_in_B << std::endl;
  std::cout << "pair_num=" << pair_num << std::endl;
#endif

  // CAUSION: file_stream1 is ordered, while file_stream2 is unordered
  // So, firstly, we read <key,value> pairs from file_stream2, and insert it into a hash table.
  char *buf = (char *)malloc(pair_size_in_B);
  TypeHashKey *key;
  TypeHashValue *value;
  HashTableCpu<TypeHashKey, TypeHashValue> *hash_table =
      new HashTableCpu<TypeHashKey, TypeHashValue>();
  while (file_stream2.peek() != EOF) {
    file_stream2.read(buf, pair_size_in_B);
    key = (TypeHashKey *)buf;
    value = (TypeHashValue *)(buf + sizeof(TypeHashKey) +
                              sizeof(TypeSlotId));  // including slot_id and value
    hash_table->insert(key, value, 1);
  }
  file_stream2.close();

  size_t hash_table_size = hash_table->get_size();
  if (hash_table_size != pair_num) {
    ERROR_MESSAGE_(
        "Error: The number of <key,value> pair inserting into CPU hash table is not equal to hash "
        "table file size\n");
    std::cout << "CPU hash_table_size=" << hash_table_size << std::endl;
    return false;
  }

  // Then, we read <key,value1> pairs from file_stream1, and get(key,value2) from hash_table, and
  // compare value1 and value2.
  TypeHashValue *value1;
  TypeHashValue *value2 = (TypeHashValue *)malloc(sizeof(TypeHashValue));
  size_t value_len = sizeof(TypeHashValue) / sizeof(float);
  while (file_stream1.peek() != EOF) {
    file_stream1.read(buf, pair_size_in_B);
    key = (TypeHashKey *)buf;
    value1 = (TypeHashValue *)(buf + sizeof(TypeHashKey) +
                               sizeof(TypeSlotId));  // including slot_id and value

    hash_table->get(key, value2, 1);

    if (!compare_array(value_len, (float *)value1, (float *)value2, epsilon)) {
      rtn = false;
      break;
    }
  }
  file_stream1.close();

  free(value2);

  return rtn;
}

inline bool compare_embedding_feature(int num, float *embedding_feature_from_gpu,
                                      float *embedding_feature_from_cpu, float epsilon) {
  return compare_array(num, embedding_feature_from_gpu, embedding_feature_from_cpu, epsilon);
}

// overload for fp16 on GPU
inline bool compare_embedding_feature(int num, __half *embedding_feature_from_gpu,
                                      __half *embedding_feature_from_cpu, float epsilon) {
  return compare_array(num, embedding_feature_from_gpu, embedding_feature_from_cpu, epsilon);
}

inline bool compare_wgrad(int num, float *wgrad_from_gpu, float *wgrad_from_cpu, float epsilon) {
  return compare_array(num, wgrad_from_gpu, wgrad_from_cpu, epsilon);
}

// overlaod for fp16 on GPU
inline bool compare_wgrad(int num, __half *wgrad_from_gpu, __half *wgrad_from_cpu, float epsilon) {
  return compare_array(num, wgrad_from_gpu, wgrad_from_cpu, epsilon);
}

inline bool compare_embedding_table(long long num, float *embedding_table_from_gpu,
                                    float *embedding_table_from_cpu, float epsilon) {
  return compare_array(num, embedding_table_from_gpu, embedding_table_from_cpu, epsilon);
}

template <typename TypeHashKey, typename TypeHashValue>
bool compare_hash_table(long long capacity, TypeHashKey *hash_table_key_from_gpu,
                        TypeHashValue *hash_table_value_from_gpu,
                        TypeHashKey *hash_table_key_from_cpu,
                        TypeHashValue *hash_table_value_from_cpu, float epsilon) {
  bool rtn = true;

  // Since the <key1,value1> and <key2,value2> is not the same ordered, we need to insert <key1,
  // value1> into a hash_table, then compare value1=hash_table->get(key2) with value2
  HashTableCpu<TypeHashKey, TypeHashValue> *hash_table =
      new HashTableCpu<TypeHashKey, TypeHashValue>();
  hash_table->insert(hash_table_key_from_gpu, hash_table_value_from_gpu, capacity);

  TypeHashKey *key;
  TypeHashValue *value1 = (TypeHashValue *)malloc(sizeof(TypeHashValue));
  TypeHashValue *value2;
  size_t value_len = sizeof(TypeHashValue) / sizeof(float);
  for (long long i = 0; i < capacity; i++) {
    key = hash_table_key_from_cpu + i;
    value2 = hash_table_value_from_cpu + i;

    hash_table->get(key, value1, 1);

    if (!compare_array(value_len, (float *)value1, (float *)value2, epsilon)) {
      std::cout << "Error in compare_hash_table: <key, value> pair number=" << i << std::endl;
      rtn = false;
      break;
    }
  }

  free(value1);

  return rtn;
}

template <typename T>
class UnorderedKeyGenerator {
 public:
  UnorderedKeyGenerator() : gen_(rd_()) {}
  UnorderedKeyGenerator(T min, T max) : gen_(rd_()), dis_(min, max) {}

  // generate unduplicated dataset
  void fill_unique(T *data, size_t len) {
    if (len == 0) {
      return;
    }
    assert(dis_.max() - dis_.min() >= len - 1);

    std::unordered_set<T> set;
    size_t sz = 0;
    while (sz < len) {
      T x = dis_(gen_);
      auto res = set.insert(x);
      if (res.second) {
        data[sz++] = x;
      }
    }
    assert(sz == set.size());
    assert(sz == len);
  }

 private:
  std::random_device rd_;
  std::mt19937 gen_;
  std::uniform_int_distribution<T> dis_;
};

}  // namespace embedding_test

}  // namespace HugeCTR
