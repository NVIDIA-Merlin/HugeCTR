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

#include <algorithm>
#include <common.hpp>
#include <hps/inference_utils.hpp>
#include <hps/modelloader.hpp>
#include <parser.hpp>
#include <unordered_set>
#include <utils.hpp>

namespace HugeCTR {

template <typename TKey, typename TValue>
void* UnifiedEmbeddingTable<TKey, TValue>::get_cache_keys() {
  return this->keys.data();
}

template <typename TKey, typename TValue>
void* UnifiedEmbeddingTable<TKey, TValue>::get_caceh_vecs() {
  return this->vectors.data();
}

template <typename TKey, typename TValue>
void* UnifiedEmbeddingTable<TKey, TValue>::get_uvm_keys() {
  return this->uvm_keys.data();
}

template <typename TKey, typename TValue>
void* UnifiedEmbeddingTable<TKey, TValue>::get_uvm_vecs() {
  return this->uvm_vectors.data();
}

template <typename TKey, typename TValue>
size_t UnifiedEmbeddingTable<TKey, TValue>::get_cache_key_count() {
  return this->key_count;
}

template <typename TKey, typename TValue>
size_t UnifiedEmbeddingTable<TKey, TValue>::get_uvm_key_count() {
  return this->uvm_key_count;
}

template <typename TKey, typename TValue>
RawModelLoader<TKey, TValue>::RawModelLoader() : IModelLoader() {
  HCTR_LOG_S(DEBUG, WORLD) << "Created raw model loader in local memory!" << std::endl;
  embedding_table_ = new UnifiedEmbeddingTable<TKey, TValue>();
}

template <typename TKey, typename TValue>
void RawModelLoader<TKey, TValue>::load_fused_emb(const std::string& table_name,
                                                  const std::vector<std::string>& path_list) {
  embedding_table_->key_count = 0;
  embedding_table_->vec_elem_count = 0;
  for (auto path : path_list) {
    this->load_emb(table_name, path);
  }
}

template <typename TKey, typename TValue>
void RawModelLoader<TKey, TValue>::load_emb(const std::string& table_name,
                                            const std::string& path) {
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
  const size_t num_float_val_in_vec_file = vec_file_size_in_byte / vec_size_in_byte;

  size_t key_offset_in_elements = embedding_table_->key_count;
  size_t vec_offset_in_elements = embedding_table_->vec_elem_count;

  embedding_table_->key_count += num_key;
  embedding_table_->vec_elem_count += num_float_val_in_vec_file;

  embedding_table_->keys.resize(embedding_table_->key_count);
  embedding_table_->vectors.resize(embedding_table_->vec_elem_count);

  if (std::is_same<TKey, long long>::value) {
    fs->read(key_file, embedding_table_->keys.data() + key_offset_in_elements,
             key_file_size_in_byte, 0);
  } else {
    std::vector<long long> i64_key_vec(num_key, 0);
    fs->read(key_file, i64_key_vec.data(), key_file_size_in_byte, 0);
    std::transform(i64_key_vec.begin(), i64_key_vec.end(),
                   embedding_table_->keys.begin() + key_offset_in_elements,
                   [](long long key) { return static_cast<unsigned>(key); });
  }
  fs->read(vec_file, embedding_table_->vectors.data() + vec_offset_in_elements,
           vec_file_size_in_byte, 0);
}

template <typename TKey, typename TValue>
void RawModelLoader<TKey, TValue>::load(const std::string& table_name, const std::string& path,
                                        size_t key_num_per_iteration, size_t threshold) {
  embedding_folder_path = path;
  const std::string emb_file_prefix = path + "/";
  const std::string key_file = emb_file_prefix + "key";
  const std::string vec_file = emb_file_prefix + "emb_vector";
  const std::string meta_file = emb_file_prefix + "meta";

  fs_ = FileSystemBuilder::build_unique_by_path(path);
  const size_t key_file_size_in_byte = fs_->get_file_size(key_file);
  const size_t vec_file_size_in_byte = fs_->get_file_size(vec_file);

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
  embedding_table_->total_key_count = num_key;

  if (std::filesystem::exists(meta_file)) {
    const size_t meta_file_size_in_byte = fs_->get_file_size(meta_file);
    if (meta_file_size_in_byte == 0) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Error: embeddings meta file is empty");
    }
    if (meta_file_size_in_byte != key_file_size_in_byte) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Error: embeddings meta file size does not match embedding key file size");
    }
    if (threshold > 0) {
      embedding_table_->threshold = threshold;
    } else {
      embedding_table_->meta.resize(num_key);
      fs_->read(meta_file, embedding_table_->meta.data(), meta_file_size_in_byte, 0);
      sort(embedding_table_->meta.begin(), embedding_table_->meta.end());
      embedding_table_->threshold = embedding_table_->meta[num_key - key_num_per_iteration];
      embedding_table_->cache_capacity = key_num_per_iteration;
      std::vector<TKey>().swap(embedding_table_->meta);
    }
  }
  if (key_num_per_iteration == 0) {
    // todo: The number of iterations can be calculated based on the maximum memory size configured
    // by the user
    if (num_key % 10 == 0) {
      num_iterations = 10;
    } else {
      num_iterations = 11;
    }
    key_iteration = num_key / 10;
  } else {
    key_iteration = key_num_per_iteration;
    num_iterations = num_key % key_num_per_iteration == 0 ? num_key / key_num_per_iteration
                                                          : (num_key / key_num_per_iteration) + 1;
  }
}

template <typename TKey, typename TValue>
void RawModelLoader<TKey, TValue>::delete_table() {
  std::vector<TKey>().swap(embedding_table_->keys);
  std::vector<TValue>().swap(embedding_table_->vectors);
  std::vector<TKey>().swap(embedding_table_->meta);
  std::vector<TKey>().swap(embedding_table_->uvm_keys);
  std::vector<TValue>().swap(embedding_table_->uvm_vectors);
  delete embedding_table_;
}

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::getkeys() {
  return embedding_table_->keys.data();
}

template <typename TKey, typename TValue>
void RawModelLoader<TKey, TValue>::get_cache_uvm(size_t iteration, size_t emb_size,
                                                 size_t cache_capacity) {
  embedding_table_->cache_capacity = cache_capacity;
  const std::string key_file = embedding_folder_path + "/" + "key";
  const std::string vec_file = embedding_folder_path + "/" + "emb_vector";
  const std::string meta_file = embedding_folder_path + "meta";
  if (iteration <= cache_capacity / key_iteration) {
    embedding_table_->keys.resize(key_iteration);
    size_t iteration_key_reading_amount = key_iteration;
    size_t iteration_vec_reading_amount = key_iteration * emb_size;
    if ((iteration + 1) * key_iteration > embedding_table_->cache_capacity) {
      iteration_key_reading_amount = embedding_table_->cache_capacity - iteration * key_iteration;
      iteration_vec_reading_amount =
          embedding_table_->cache_capacity * emb_size - iteration * key_iteration * emb_size;
    }
    embedding_table_->key_count = iteration_key_reading_amount;
    embedding_table_->uvm_key_count = 0;
    if (std::is_same<TKey, long long>::value) {
      fs_->read(key_file, embedding_table_->keys.data(),
                iteration_key_reading_amount * sizeof(TKey),
                iteration * key_iteration * sizeof(TKey));
    } else {
      std::vector<long long> i64_key_vec(iteration_key_reading_amount, 0);
      fs_->read(key_file, i64_key_vec.data(), iteration_key_reading_amount * sizeof(long long),
                iteration * key_iteration * sizeof(long long));
      std::transform(i64_key_vec.begin(), i64_key_vec.end(), embedding_table_->keys.begin(),
                     [](long long key) { return static_cast<unsigned>(key); });
    }
    embedding_table_->vectors.resize(key_iteration * emb_size);
    fs_->read(vec_file, embedding_table_->vectors.data(),
              iteration_vec_reading_amount * sizeof(TValue),
              key_iteration * emb_size * iteration * sizeof(TValue));
  }
  if (iteration >= cache_capacity / key_iteration) {
    embedding_table_->keys.resize(key_iteration);
    size_t iteration_key_reading_amount = key_iteration;
    size_t iteration_vec_reading_amount = key_iteration * emb_size;
    size_t offset = 0;
    if (iteration == cache_capacity / key_iteration) {
      iteration_key_reading_amount =
          (iteration + 1) * key_iteration - embedding_table_->cache_capacity;
      iteration_vec_reading_amount =
          ((iteration + 1) * key_iteration - embedding_table_->cache_capacity) * emb_size;
      offset = embedding_table_->cache_capacity;
    } else {
      embedding_table_->key_count = 0;
      offset = key_iteration * iteration;
    }
    if ((iteration + 1) * key_iteration > embedding_table_->total_key_count) {
      iteration_key_reading_amount = embedding_table_->total_key_count - iteration * key_iteration;
      iteration_vec_reading_amount =
          embedding_table_->total_key_count * emb_size - iteration * key_iteration * emb_size;
    }
    embedding_table_->uvm_key_count = iteration_key_reading_amount;
    if (std::is_same<TKey, long long>::value) {
      fs_->read(key_file, embedding_table_->uvm_keys.data(),
                iteration_key_reading_amount * sizeof(TKey), offset * sizeof(TKey));
    } else {
      std::vector<long long> i64_key_vec(iteration_key_reading_amount, 0);
      fs_->read(key_file, i64_key_vec.data(), iteration_key_reading_amount * sizeof(long long),
                offset * sizeof(long long));
      std::transform(i64_key_vec.begin(), i64_key_vec.end(), embedding_table_->uvm_keys.begin(),
                     [](long long key) { return static_cast<unsigned>(key); });
    }
    embedding_table_->vectors.resize(key_iteration * emb_size);
    fs_->read(vec_file, embedding_table_->uvm_vectors.data(),
              iteration_vec_reading_amount * sizeof(TValue), offset * emb_size * sizeof(TValue));
  }
}

template <typename TKey, typename TValue>
std::pair<void*, size_t> RawModelLoader<TKey, TValue>::getkeys(size_t iteration) {
  const std::string key_file = embedding_folder_path + "/" + "key";
  embedding_table_->keys.resize(key_iteration);
  size_t iteration_reading_amount = key_iteration;
  if ((iteration + 1) * key_iteration > embedding_table_->total_key_count) {
    iteration_reading_amount = embedding_table_->total_key_count - iteration * key_iteration;
  }

  if (std::is_same<TKey, long long>::value) {
    fs_->read(key_file, embedding_table_->keys.data(), iteration_reading_amount * sizeof(TKey),
              iteration * key_iteration * sizeof(TKey));
  } else {
    std::vector<long long> i64_key_vec(iteration_reading_amount, 0);
    fs_->read(key_file, i64_key_vec.data(), iteration_reading_amount * sizeof(long long),
              iteration * key_iteration * sizeof(long long));
    std::transform(i64_key_vec.begin(), i64_key_vec.end(), embedding_table_->keys.begin(),
                   [](long long key) { return static_cast<unsigned>(key); });
  }
  return std::make_pair(embedding_table_->keys.data(), iteration_reading_amount);
}

template <typename TKey, typename TValue>
std::pair<void*, size_t> RawModelLoader<TKey, TValue>::getvectors(size_t iteration,
                                                                  size_t emb_size) {
  const std::string vec_file = embedding_folder_path + "/" + "emb_vector";
  embedding_table_->vectors.resize(key_iteration * emb_size);
  size_t iteration_reading_amount = key_iteration * emb_size;
  if ((iteration + 1) * key_iteration * emb_size > embedding_table_->total_key_count * emb_size) {
    iteration_reading_amount =
        embedding_table_->total_key_count * emb_size - iteration * key_iteration * emb_size;
  }
  fs_->read(vec_file, embedding_table_->vectors.data(), iteration_reading_amount * sizeof(TValue),
            key_iteration * emb_size * iteration * sizeof(TValue));
  return std::make_pair(embedding_table_->vectors.data(), iteration_reading_amount);
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
  return embedding_table_->total_key_count;
}

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::get_cache_keys() {
  return embedding_table_->get_cache_keys();
};

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::get_caceh_vecs() {
  return embedding_table_->get_caceh_vecs();
};

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::get_uvm_keys() {
  return embedding_table_->get_uvm_keys();
};

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::get_uvm_vecs() {
  return embedding_table_->get_uvm_vecs();
};

template <typename TKey, typename TValue>
size_t RawModelLoader<TKey, TValue>::get_cache_key_count() {
  return embedding_table_->get_cache_key_count();
};

template <typename TKey, typename TValue>
size_t RawModelLoader<TKey, TValue>::get_uvm_key_count() {
  return embedding_table_->get_uvm_key_count();
};

template <typename TKey, typename TValue>
size_t RawModelLoader<TKey, TValue>::get_num_iterations() {
  return num_iterations;
}

template class RawModelLoader<long long, float>;
template class RawModelLoader<unsigned int, float>;

}  // namespace HugeCTR
