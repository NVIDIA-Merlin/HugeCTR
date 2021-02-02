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

namespace HugeCTR {

template <typename KeyType>
void LocalizedParameterServerDelegate<KeyType>::load_from_snapshot(
    std::ofstream& embedding_table,
    std::ifstream& snapshot,
    const size_t file_size_in_byte,
    const size_t embedding_vector_size,
    HashTable& hash_table) {
  const size_t key_size_in_byte = sizeof(KeyType);
  const size_t slot_id_size_in_byte = sizeof(size_t);
  const size_t embedding_vector_size_in_byte = sizeof(float) * embedding_vector_size;
  const size_t row_size_in_byte =
      key_size_in_byte + slot_id_size_in_byte + embedding_vector_size_in_byte;
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
      size_t slot_id = *(size_t*)(cur_ptr + key_size_in_byte);
      hash_table.insert({key, {slot_id, cur_idx}});

      float* dst_emb =
        (float*)(write_emb_chunk.get() + embedding_vector_size_in_byte * k);
      float* src_emb = (float*)(cur_ptr + key_size_in_byte + slot_id_size_in_byte);
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
void LocalizedParameterServerDelegate<KeyType>::store_to_snapshot(
    std::ofstream& snapshot,
    std::ifstream& embedding_table,
    const size_t file_size_in_byte,
    const size_t embedding_vector_size,
    HashTable& hash_table) {

  std::vector<std::pair<KeyType, size_t>> idx2key(hash_table.size()); // assume the indices are unique
  for (auto it = hash_table.begin(); it != hash_table.end(); ++it) {
    size_t idx = it->second.second;
    size_t slot_id = it->second.first;
    KeyType key = it->first;
    idx2key[idx] = {key, slot_id};
  }

  const size_t embedding_vector_size_in_byte = sizeof(float) * embedding_vector_size;
  const size_t num_unit_rows = 1024;
  const size_t read_emb_chunk_size = num_unit_rows * embedding_vector_size_in_byte;
  const size_t row_size_in_byte =
      sizeof(KeyType) + sizeof(size_t) + embedding_vector_size_in_byte;
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
      const size_t src_key = idx2key[idx].first;
      const size_t src_slot_id = idx2key[idx].second;
      char* dst_buf = read_snapshot_chunk.get() + row_size_in_byte * o;
      KeyType* dst_key = (KeyType*)dst_buf;
      *dst_key = src_key;
      size_t* dst_slot_id = (size_t*)(dst_buf + sizeof(KeyType));
      *dst_slot_id = src_slot_id;
      float* dst_emb = (float*)(dst_buf + sizeof(size_t) + sizeof(KeyType));
      float* src_emb =
        (float*)(read_emb_chunk.get() + embedding_vector_size_in_byte * o);
      memcpy(dst_emb, src_emb, embedding_vector_size_in_byte);
    }
    snapshot.write(read_snapshot_chunk.get(), num_embs * (row_size_in_byte));
  }
}

template class LocalizedParameterServerDelegate<unsigned int>;
template class LocalizedParameterServerDelegate<long long>;

}  // namespace HugeCTR
