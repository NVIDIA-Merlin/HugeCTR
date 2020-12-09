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

#include <model_oversubscriber/localized_parameter_server_delegate.hpp>

#include <cstring>
#include <memory>
#include <vector>
#include <iostream>
#include <map>
#include <omp.h>

namespace HugeCTR {

template <typename KeyType>
void LocalizedParameterServerDelegate<KeyType>::load(
    std::ofstream& embedding_table,
    std::ifstream& snapshot,
    const size_t file_size_in_byte,
    const size_t embedding_vector_size,
    HashTable& hash_table) {
  const size_t key_size_in_byte = sizeof(KeyType);
  const size_t embedding_vector_size_in_byte = sizeof(float) * embedding_vector_size;
  const size_t row_size_in_byte = key_size_in_byte + embedding_vector_size_in_byte;
  const size_t num_unit_rows = 1024;
  const size_t read_chunk_size = num_unit_rows * row_size_in_byte;

  const size_t write_emb_chunk_size = num_unit_rows * embedding_vector_size_in_byte;

  const size_t num_full_chunks = file_size_in_byte / read_chunk_size;
  const size_t num_remaining_rows_in_byte = file_size_in_byte % read_chunk_size;
  const size_t num_remaining_rows = num_remaining_rows_in_byte / row_size_in_byte;

  size_t cur_idx = 0;
  std::unique_ptr<char[]> read_chunk(new char[read_chunk_size]);
  std::unique_ptr<char[]> write_emb_chunk(new char[write_emb_chunk_size]);

  auto read_rows_op = [&embedding_table, &snapshot,
                       &write_emb_chunk, &read_chunk,
                       row_size_in_byte, &hash_table,
                       embedding_vector_size_in_byte]
      (const size_t num_rows, size_t& cur_idx) {
    char* cur_ptr = read_chunk.get();
    snapshot.read(cur_ptr, num_rows * row_size_in_byte);
    for (size_t k = 0; k < num_rows; k++) {
      KeyType key = *(KeyType*)(cur_ptr);
      hash_table.insert({key, {0, cur_idx}}); // default slot_id = 0 for distributed embedding

      float* dst_emb =
        (float*)(write_emb_chunk.get() + embedding_vector_size_in_byte * k);
      float* src_emb = (float*)(cur_ptr + key_size_in_byte);
      memcpy(dst_emb, src_emb, embedding_vector_size_in_byte);

      cur_ptr += row_size_in_byte;
      cur_idx++;
    }
    embedding_table.write(
        write_emb_chunk.get(), num_rows * embedding_vector_size_in_byte);
  };

  for (size_t ch = 0; ch < num_full_chunks; ch++) {
    read_rows_op(num_unit_rows, cur_idx);
  }
  read_rows_op(num_remaining_rows, cur_idx);
}

template <typename KeyType>
void LocalizedParameterServerDelegate<KeyType>::store(
    std::ofstream& snapshot,
    std::ifstream& embedding_table,
    const size_t file_size_in_byte,
    const size_t embedding_vector_size,
    HashTable& hash_table) {
  std::vector<KeyType> idx2key(hash_table.size()); // assume the indices are unique
  for (auto it = hash_table.begin(); it != hash_table.end(); ++it) {
    size_t idx = it->second.second;
    KeyType key = it->first;
    idx2key[idx] = key;
  }


  const size_t embedding_vector_size_in_byte = sizeof(float) * embedding_vector_size;
  const size_t num_unit_rows = 1024;
  const size_t read_emb_chunk_size = num_unit_rows * embedding_vector_size_in_byte;
  const size_t row_size_in_byte = sizeof(KeyType) + embedding_vector_size_in_byte;
  const size_t read_snapshot_chunk_size = num_unit_rows * row_size_in_byte;

  std::unique_ptr<char[]> read_emb_chunk(new char[read_emb_chunk_size]);
  std::unique_ptr<char[]> read_snapshot_chunk(new char[read_snapshot_chunk_size]);

  for (size_t pos = 0; pos < file_size_in_byte; pos += read_emb_chunk_size) {
    const size_t read_bytes = (pos + read_emb_chunk_size) < file_size_in_byte?
      read_emb_chunk_size : (file_size_in_byte - pos);

    embedding_table.read(read_emb_chunk.get(), read_bytes);
    const size_t base_idx = pos / embedding_vector_size_in_byte;
    const size_t num_embs = read_bytes / embedding_vector_size_in_byte;
    for(size_t o = 0; o < num_embs; o++) {
      const size_t idx = base_idx + o;
      const size_t src_key = idx2key[idx];
      char* dst_buf = read_snapshot_chunk.get() + row_size_in_byte * o;
      KeyType* dst_key = (KeyType*)dst_buf;
      *dst_key = src_key;
      float* dst_emb = (float*)(dst_buf + sizeof(KeyType));
      float* src_emb =
        (float*)(read_emb_chunk.get() + embedding_vector_size_in_byte * o);
      memcpy(dst_emb, src_emb, embedding_vector_size_in_byte);
    }
    snapshot.write(read_snapshot_chunk.get(), num_embs * (row_size_in_byte));
  }
}

template <typename KeyType>
void LocalizedParameterServerDelegate<KeyType>::load_from_embedding_file(
    float* mmaped_table,
    BufferBag& buf_bag,
    const std::vector<KeyType>& keyset,
    const size_t embedding_vec_size,
    const HashTable& hash_table,
    size_t& hit_size)
{
  KeyType* keys = Tensor2<KeyType>::stretch_from(buf_bag.keys).get_ptr();
  float* hash_table_val = buf_bag.embedding.get_ptr();

  std::vector<size_t> idx_exist;
  std::map<size_t, KeyType> pair_exist;
  for (size_t cnt = 0; cnt < keyset.size(); cnt++) {
    auto iter = hash_table.find(keyset[cnt]);
    if (iter == hash_table.end()) continue;
    pair_exist.insert({iter->second.second, iter->first});
  }

  size_t cnt_hit_keys = 0;
  idx_exist.reserve(pair_exist.size());
  for (auto& pair : pair_exist) {
    keys[cnt_hit_keys++] = pair.second;
    idx_exist.push_back(pair.first);
  }

  const size_t embedding_vector_size_in_byte = sizeof(float) * embedding_vec_size;

#pragma omp parallel num_threads(8)
  {
    const size_t tid = omp_get_thread_num();
    const size_t thread_num = omp_get_num_threads();
    size_t sub_chunk_size = idx_exist.size() / thread_num;
    size_t res_chunk_size = idx_exist.size() % thread_num;
    const size_t idx = tid * sub_chunk_size;

    if (tid == thread_num - 1) sub_chunk_size += res_chunk_size;

    for (size_t i = 0; i < sub_chunk_size; i++) {
      size_t src_idx = idx_exist[idx + i] * embedding_vec_size;
      size_t dst_idx = (idx + i) * embedding_vec_size;
      memcpy(&hash_table_val[dst_idx], &mmaped_table[src_idx], embedding_vector_size_in_byte);
    }
  }

  hit_size = cnt_hit_keys;
}

template <typename KeyType>
void LocalizedParameterServerDelegate<KeyType>::dump_to_embedding_file(
    float* mmaped_table,
    BufferBag& buf_bag,
    const size_t embedding_vec_size,
    const std::string& embedding_table_path,
    HashTable& hash_table,
    const size_t dump_size)
{
  const KeyType* keys = Tensor2<KeyType>::stretch_from(buf_bag.keys).get_ptr();
  const float* hash_table_val = buf_bag.embedding.get_ptr();
  
  size_t cnt_new_keys = 0;
  const size_t hash_table_size = hash_table.size();

  std::vector<size_t> idx_exist_src, idx_exist_dst, idx_miss_src;
  std::map<size_t, size_t> idx_exist;
  for (size_t cnt = 0; cnt < dump_size; cnt++) {
    auto iter = hash_table.find(keys[cnt]);
    if (iter == hash_table.end()) {
      hash_table.insert({keys[cnt], {0, hash_table_size + cnt_new_keys}});
      idx_miss_src.push_back(cnt);
      cnt_new_keys++;
    } else {
      idx_exist.insert({iter->second.second, cnt});
    }
  }

  idx_exist_src.reserve(idx_exist.size());
  idx_exist_dst.reserve(idx_exist.size());
  for (auto& pair : idx_exist) {
    idx_exist_src.push_back(pair.second);
    idx_exist_dst.push_back(pair.first);
  }

  const size_t embedding_vector_size_in_byte = sizeof(float) * embedding_vec_size;

#pragma omp parallel num_threads(8)
  {
    const size_t tid = omp_get_thread_num();
    const size_t thread_num = omp_get_num_threads();
    size_t sub_chunk_size = idx_exist_src.size() / thread_num;
    size_t res_chunk_size = idx_exist_src.size() % thread_num;
    const size_t idx = tid * sub_chunk_size;

    if (tid == thread_num - 1) sub_chunk_size += res_chunk_size;

    for (size_t i = 0; i < sub_chunk_size; i++) {
      size_t src_idx = idx_exist_src[idx + i] * embedding_vec_size;
      size_t dst_idx = idx_exist_dst[idx + i] * embedding_vec_size;
      memcpy(&mmaped_table[dst_idx], &hash_table_val[src_idx], embedding_vector_size_in_byte);
    }
  }

  // append new embedding to file
  std::ofstream embedding_file;
  embedding_file.open(embedding_table_path, std::ofstream::binary | std::ofstream::app);
  if (!embedding_file.is_open()) {
    CK_THROW_(Error_t::WrongInput, "Cannot open the file: " + embedding_table_path);
  }
  for (size_t cnt = 0; cnt < cnt_new_keys; cnt++) {
    size_t src_idx = idx_miss_src[cnt] * embedding_vec_size;
    embedding_file.write((char*)(&hash_table_val[src_idx]), embedding_vector_size_in_byte);
  }
}

template class LocalizedParameterServerDelegate<unsigned int>;
template class LocalizedParameterServerDelegate<long long>;

}  // namespace HugeCTR
