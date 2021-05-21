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

#include <fcntl.h>
#include <omp.h>
#include <sys/io.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <model_oversubscriber/distributed_parameter_server_delegate.hpp>
#include <model_oversubscriber/parameter_server.hpp>
#include <random>

namespace HugeCTR {

namespace {

std::string generate_random_file_name() {
  std::string ch_set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<> ch_dist(0, ch_set.size() - 1);
  std::uniform_int_distribution<> len_dist(ch_set.size() / 5, ch_set.size() / 3);

  int length = len_dist(rng);
  auto get_ch = [&ch_set, &ch_dist, &rng]() { return ch_set[ch_dist(rng)]; };

  std::string ret(length, 0);
  std::generate_n(ret.begin(), length, get_ch);
  return ret;
}

void open_and_get_size(const std::string& file_name, std::ifstream& stream,
                       size_t& file_size_in_byte) {
  if (file_name.empty()) {
    std::string random_src_snapshot = "./" + generate_random_file_name();
    stream.open(random_src_snapshot, std::ofstream::binary | std::ofstream::trunc);
    file_size_in_byte = 0;
    return;
  }

  stream.open(file_name, std::ifstream::binary);
  if (!stream.is_open()) {
    CK_THROW_(Error_t::WrongInput, "Cannot open the file: " + file_name);
  }

  // get the file size in byte
  stream.seekg(0, stream.end);
  file_size_in_byte = stream.tellg();
  stream.seekg(0, stream.beg);
}

}  // namespace

template <typename TypeHashKey, typename TypeEmbeddingComp>
void ParameterServer<TypeHashKey, TypeEmbeddingComp>::map_embedding_to_memory_() {
  try {
    if (embedding_table_path_.empty()) {
      CK_THROW_(Error_t::WrongInput, "Temp embedding filename is empty");
    }

    fd_ = open(embedding_table_path_.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
    if (fd_ == -1) {
      CK_THROW_(Error_t::FileCannotOpen, "Cannot open the file: " + embedding_table_path_);
    }

    std::ifstream ifstream;
    open_and_get_size(embedding_table_path_, ifstream, file_size_in_byte_);
    if (file_size_in_byte_ < 0) {
      CK_THROW_(Error_t::WrongInput, "Temp embedding file size error");
    }
    ifstream.close();

    // handle empty file for trainning from scratch
    // but writting to empty file may cause error because only one page is mapped
    size_t mmaped_file_size = (!file_size_in_byte_) ? 128 : file_size_in_byte_;
    mmaped_table_ =
        (float*)mmap(NULL, mmaped_file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (mmaped_table_ == MAP_FAILED) {
      close(fd_);
      fd_ = -1;
      mmaped_table_ = nullptr;
      CK_THROW_(Error_t::WrongInput, "Mmap file " + embedding_table_path_ + " failed");
    }

    maped_to_memory_ = true;
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void ParameterServer<TypeHashKey, TypeEmbeddingComp>::unmap_embedding_from_memory_() {
  try {
    if (maped_to_memory_ && fd_ != -1 && mmaped_table_ != nullptr) {
      munmap(mmaped_table_, file_size_in_byte_);
      close(fd_);
      fd_ = -1;
      mmaped_table_ = nullptr;
      maped_to_memory_ = false;
    } else {
      CK_THROW_(Error_t::WrongInput, embedding_table_path_ + " not mapped to host memory");
    }
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
ParameterServer<TypeHashKey, TypeEmbeddingComp>::ParameterServer(
    const SparseEmbeddingHashParams<TypeEmbeddingComp>& embedding_params,
    const std::string& snapshot_src_file, const std::string& temp_embedding_dir)
    : embedding_params_(embedding_params),
      embedding_table_path_(temp_embedding_dir + "/" + generate_random_file_name()),
      // TODO(minseok, 10282020): handle the different types of Embeddings
      parameter_server_delegate_(new DistributedParameterServerDelegate<TypeHashKey>()),
      file_size_in_byte_{0},
      mmaped_table_{nullptr},
      fd_{-1},
      maped_to_memory_{false} {
  try {
    std::ifstream snapshot_stream;
    size_t file_size_in_byte = 0;
    open_and_get_size(snapshot_src_file, snapshot_stream, file_size_in_byte);

    std::ofstream embedding_table_stream(embedding_table_path_,
                                         std::ofstream::binary | std::ofstream::trunc);
    if (!embedding_table_stream.is_open()) {
      CK_THROW_(Error_t::WrongInput, "Cannot open the file: " + embedding_table_path_);
    }

    // let the delegate fill the hash table
    parameter_server_delegate_->load(embedding_table_stream, snapshot_stream, file_size_in_byte,
                                     embedding_params_.embedding_vec_size, hash_table_);

  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
ParameterServer<TypeHashKey, TypeEmbeddingComp>::~ParameterServer() {
  if (maped_to_memory_) {
    unmap_embedding_from_memory_();
  }
  if (!embedding_table_path_.empty()) {
    std::remove(embedding_table_path_.c_str());
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void ParameterServer<TypeHashKey, TypeEmbeddingComp>::load_keyset_from_file(
    std::string keyset_file) {
  try {
    std::ifstream keyset_stream;
    size_t file_size_in_byte = 0;
    open_and_get_size(keyset_file, keyset_stream, file_size_in_byte);

    size_t num_keys_in_file = file_size_in_byte / sizeof(TypeHashKey);
    keyset_.resize(num_keys_in_file);
    keyset_stream.read((char*)keyset_.data(), file_size_in_byte);
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void ParameterServer<TypeHashKey, TypeEmbeddingComp>::load_param_from_embedding_file(
    float* hash_table_val, TypeHashKey* keys, size_t* hit_size) {
  try {
    if (!keyset_.size()) {
      CK_THROW_(Error_t::WrongInput, "Keyset_ is empty");
    }

    std::vector<size_t> idx_exist;
    std::map<size_t, TypeHashKey> pair_exist;
    for (size_t cnt = 0; cnt < keyset_.size(); cnt++) {
      auto iter = hash_table_.find(keyset_[cnt]);
      if (iter == hash_table_.end()) continue;
      pair_exist.insert({iter->second, iter->first});
    }

    size_t cnt_hit_keys = 0;
    idx_exist.reserve(pair_exist.size());
    for (auto& pair : pair_exist) {
      keys[cnt_hit_keys++] = pair.second;
      idx_exist.push_back(pair.first);
    }

    const size_t embedding_vec_size = embedding_params_.embedding_vec_size;
    const size_t embedding_vector_size_in_byte = sizeof(float) * embedding_vec_size;

    if (!maped_to_memory_) {
      map_embedding_to_memory_();
    }

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
        memcpy(&hash_table_val[dst_idx], &mmaped_table_[src_idx], embedding_vector_size_in_byte);
      }
    }

    *hit_size = cnt_hit_keys;
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void ParameterServer<TypeHashKey, TypeEmbeddingComp>::dump_param_to_embedding_file(
    float* hash_table_val, TypeHashKey* keys, size_t dump_size) {
  try {
    size_t cnt_new_keys = 0;
    const size_t hash_table_size = hash_table_.size();

    std::vector<size_t> idx_exist_src, idx_exist_dst, idx_miss_src;
    std::map<size_t, size_t> idx_exist;
    for (size_t cnt = 0; cnt < dump_size; cnt++) {
      auto iter = hash_table_.find(keys[cnt]);
      if (iter == hash_table_.end()) {
        hash_table_.insert({keys[cnt], hash_table_size + cnt_new_keys});
        idx_miss_src.push_back(cnt);
        cnt_new_keys++;
      } else {
        idx_exist.insert({iter->second, cnt});
      }
    }

    idx_exist_src.reserve(idx_exist.size());
    idx_exist_dst.reserve(idx_exist.size());
    for (auto& pair : idx_exist) {
      idx_exist_src.push_back(pair.second);
      idx_exist_dst.push_back(pair.first);
    }

    const size_t embedding_vec_size = embedding_params_.embedding_vec_size;
    const size_t embedding_vector_size_in_byte = sizeof(float) * embedding_vec_size;

    // update existed embedding
    if (!maped_to_memory_) {
      map_embedding_to_memory_();
    }

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
        memcpy(&mmaped_table_[dst_idx], &hash_table_val[src_idx], embedding_vector_size_in_byte);
      }
    }
    unmap_embedding_from_memory_();

    // append new embedding to file
    std::ofstream embedding_file;
    embedding_file.open(embedding_table_path_, std::ofstream::binary | std::ofstream::app);
    if (!embedding_file.is_open()) {
      CK_THROW_(Error_t::WrongInput, "Cannot open the file: " + embedding_table_path_);
    }
    for (size_t cnt = 0; cnt < cnt_new_keys; cnt++) {
      size_t src_idx = idx_miss_src[cnt] * embedding_vec_size;
      embedding_file.write((char*)(&hash_table_val[src_idx]), embedding_vector_size_in_byte);
    }

  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void ParameterServer<TypeHashKey, TypeEmbeddingComp>::dump_to_snapshot(
    const std::string& snapshot_dst_file) {
  try {
    std::ofstream snapshot(snapshot_dst_file, std::ofstream::binary | std::ofstream::trunc);
    if (!snapshot.is_open()) {
      CK_THROW_(Error_t::WrongInput, "Cannot open the file: " + snapshot_dst_file);
    }

    std::ifstream embedding_table;
    size_t file_size_in_byte = 0;
    open_and_get_size(embedding_table_path_, embedding_table, file_size_in_byte);

    parameter_server_delegate_->store(snapshot, embedding_table, file_size_in_byte,
                                      embedding_params_.embedding_vec_size, hash_table_);
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
std::vector<TypeHashKey> ParameterServer<TypeHashKey, TypeEmbeddingComp>::get_keys_from_hash_table()
    const {
  auto get_key_op = [](auto e) { return e.first; };
  std::vector<TypeHashKey> keys(hash_table_.size());
  transform(hash_table_.begin(), hash_table_.end(), keys.begin(), get_key_op);
  return keys;
}

template class ParameterServer<long long, __half>;
template class ParameterServer<long long, float>;
template class ParameterServer<unsigned, __half>;
template class ParameterServer<unsigned, float>;

}  // namespace HugeCTR
