/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <common.hpp>
#include <hps/inference_utils.hpp>
#include <hps/modelloader.hpp>
#include <io/filesystem.hpp>
#include <parser.hpp>
#include <unordered_set>
#include <utils.hpp>

namespace HugeCTR {

template <typename TKey, typename TValue>
RawModelLoader<TKey, TValue>::RawModelLoader() : IModelLoader() {
  HCTR_LOG_S(DEBUG, WORLD) << "Created raw model loader in local memory!" << std::endl;
  embedding_table_ = new UnifiedEmbeddingTable<TKey, TValue>();
}

template <typename TKey, typename TValue>
void RawModelLoader<TKey, TValue>::load(const std::string& table_name, const std::string& path) {
  const std::string emb_file_prefix = path + "/";
  const std::string key_file = emb_file_prefix + "key";
  const std::string vec_file = emb_file_prefix + "emb_vector";

  auto fs = FileSystemBuilder::build_unique_by_path(path);
  const size_t key_file_size_in_byte = fs->get_file_size(key_file);
  const size_t vec_file_size_in_byte = fs->get_file_size(vec_file);

  const size_t key_size_in_byte = sizeof(long long);
  const size_t vec_size_in_byte = sizeof(float);

  if (key_file_size_in_byte == 0) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Error: embeddings key file is empty");
  }
  if (vec_file_size_in_byte == 0) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Error: embeddings vector file is empty");
  }
  if (key_file_size_in_byte % key_size_in_byte != 0) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Error: embeddings key file size is not correct");
  }
  if (vec_file_size_in_byte % vec_size_in_byte != 0) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Error: embeddings vector file size is not correct");
  }

  const size_t num_key = key_file_size_in_byte / key_size_in_byte;
  embedding_table_->key_count = num_key;

  const size_t num_float_val_in_vec_file = vec_file_size_in_byte / vec_size_in_byte;

  // The temp embedding table
  embedding_table_->keys.resize(num_key);
  if (std::is_same<TKey, long long>::value) {
    fs->read(key_file, embedding_table_->keys.data(), key_file_size_in_byte, 0);
  } else {
    std::vector<long long> i64_key_vec(num_key, 0);
    fs->read(key_file, i64_key_vec.data(), key_file_size_in_byte, 0);
    std::transform(i64_key_vec.begin(), i64_key_vec.end(), embedding_table_->keys.begin(),
                   [](long long key) { return static_cast<unsigned>(key); });
  }

  embedding_table_->vectors.resize(num_float_val_in_vec_file);
  fs->read(vec_file, embedding_table_->vectors.data(), vec_file_size_in_byte, 0);
}

template <typename TKey, typename TValue>
void RawModelLoader<TKey, TValue>::delete_table() {
  std::vector<TKey>().swap(embedding_table_->keys);
  std::vector<TValue>().swap(embedding_table_->vectors);
  std::vector<TValue>().swap(embedding_table_->meta);
  delete embedding_table_;
}

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::getkeys() {
  return embedding_table_->keys.data();
}

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::getvectors() {
  return embedding_table_->vectors.data();
}

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::getmetas() {
  return embedding_table_->meta.data();
}

template <typename TKey, typename TValue>
size_t RawModelLoader<TKey, TValue>::getkeycount() {
  return embedding_table_->key_count;
}

template class RawModelLoader<long long, float>;
template class RawModelLoader<unsigned int, float>;

}  // namespace HugeCTR
