/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#ifndef CPUV_HASHTABLE_H_
#define CPUV_HASHTABLE_H_
#include <iostream>
#include <unordered_map>
template <typename KeyType, typename ValType>
class HashTableCpu {
 public:
  HashTableCpu() { table_ = new Table(); }
  ~HashTableCpu() { delete table_; }
  HashTableCpu(const HashTableCpu&) = delete;
  HashTableCpu& operator=(const HashTableCpu&) = delete;

  void insert(const KeyType* keys, const ValType* vals, size_t len) {
    if (len == 0) {
      return;
    }

    std::pair<KeyType, ValType> kv;

    for (size_t i = 0; i < len; i++) {
      kv.first = keys[i];
      kv.second = vals[i];
      auto pp = table_->insert(kv);
      if (pp.first == table_->end()) {
        assert(!"error: insert fails: table is full");
      }
    }
  }

  void get(const KeyType* keys, ValType* vals, size_t len) const {
    if (len == 0) {
      return;
    }
    for (size_t i = 0; i < len; i++) {
      auto it = table_->find(keys[i]);
      assert(it != table_->end() && "error: can't find key");
      vals[i] = it->second;
    }
  }

  size_t get_size() const { return table_->size(); }

  void dump(KeyType* key, ValType* val) {
    int i = 0;
    for (auto kv : (*table_)) {
      key[i] = kv.first;
      val[i] = kv.second;
      i++;
    }
  }

 private:
  using Table = std::unordered_map<KeyType, ValType>;
  Table* table_;
};

#endif
