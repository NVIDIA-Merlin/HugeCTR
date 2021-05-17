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

#include <model_oversubscriber/parameter_server.hpp>

#include <map>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/io.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <experimental/filesystem>
#include <omp.h>

namespace fs = std::experimental::filesystem;

namespace HugeCTR {

namespace {

void open_and_get_size(
    const std::string& file_name, std::ifstream& stream, size_t& file_size_in_byte) {
  stream.open(file_name, std::ifstream::binary);
  if (!stream.is_open()) {
    CK_THROW_(Error_t::WrongInput, "Cannot open the file: " + file_name);
  }

  file_size_in_byte = fs::file_size(file_name);
}

} // namespace

template <typename TypeHashKey>
struct ParameterServer<TypeHashKey>::SparseModelFile {
  std::string folder_name;
  std::string key_file;
  std::string slot_file;
  std::string vec_file;

  SparseModelFile(std::string sparse_model) : folder_name(sparse_model) {
    auto remove_prefix = [](std::string& path) {
      size_t found = path.rfind("/");
      if (found != std::string::npos)
        return std::string(path, found + 1);
      else
        return path;
    };
    key_file = sparse_model + "/" + remove_prefix(sparse_model) + ".key";
    slot_file = sparse_model + "/" + remove_prefix(sparse_model) + ".slot";
    vec_file = sparse_model + "/" + remove_prefix(sparse_model) + ".vec";
  }
};

template <typename TypeHashKey>
void ParameterServer<TypeHashKey>::map_embedding_to_memory_() {
  try {
    fd_ = open(sparse_model_->vec_file.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
    if (fd_ == -1) {
      CK_THROW_(Error_t::FileCannotOpen, "Cannot open the file: " + sparse_model_->vec_file);
    }

    size_t vec_file_size_in_byte = fs::file_size(sparse_model_->vec_file);
    if (vec_file_size_in_byte == 0) {
      CK_THROW_(Error_t::WrongInput, "Cannot mmap empty file: " + sparse_model_->vec_file);
    }

    mmaped_table_ = (float *)mmap(NULL, vec_file_size_in_byte, PROT_READ|PROT_WRITE,
                                  MAP_SHARED, fd_, 0);
    if (mmaped_table_ == MAP_FAILED) {
      close(fd_);
      fd_ = -1;
      mmaped_table_ = nullptr;
      CK_THROW_(Error_t::WrongInput, "Mmap file " + sparse_model_->vec_file + " failed");
    }

    maped_to_memory_ = true;
    if (fd_ != -1) {
      close(fd_);
      fd_ = -1;
    }
  }
  catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
  catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeHashKey>
void ParameterServer<TypeHashKey>::unmap_embedding_from_memory_() {
  try {
    if (maped_to_memory_ && mmaped_table_ != nullptr) {
      size_t vec_file_size_in_byte = fs::file_size(sparse_model_->vec_file);
      munmap(mmaped_table_, vec_file_size_in_byte);
      mmaped_table_ = nullptr;
      maped_to_memory_ = false;
    } else {
      CK_THROW_(Error_t::WrongInput, sparse_model_->vec_file + " not mapped to host memory");
    }
  }
  catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
  catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeHashKey>
ParameterServer<TypeHashKey>::ParameterServer(
    const SparseEmbeddingHashParams& embedding_params,
    const std::string& sparse_model_name, const Embedding_t embedding_type)
  : sparse_model_(new SparseModelFile(sparse_model_name)),
    embedding_vec_size_(embedding_params.embedding_vec_size),
    is_distributed_(embedding_type == Embedding_t::DistributedSlotSparseEmbeddingHash
                    ? true : false),
    mmaped_table_{nullptr},
    fd_{-1},
    maped_to_memory_{false} {
  try {
    if (!fs::exists(sparse_model_->folder_name)) {
      MESSAGE_("Train from scratch, no trained embedding table");
      fs::create_directory(sparse_model_->folder_name);

      int ret;
      std::string command("touch ");
      ret = std::system((command + sparse_model_->key_file).c_str());
      ret = std::system((command + sparse_model_->vec_file).c_str());
      if (!is_distributed_) {
        ret = std::system((command + sparse_model_->slot_file).c_str());
      }
      (void)ret;

      return;
    }

    std::ifstream key_stream;
    size_t key_file_size_in_byte;
    open_and_get_size(sparse_model_->key_file, key_stream, key_file_size_in_byte);

    size_t num_key = key_file_size_in_byte / sizeof(TypeHashKey);
    size_t num_vec = fs::file_size(sparse_model_->vec_file) / (sizeof(float) * embedding_vec_size_);
    if (num_key != num_vec) {
      CK_THROW_(Error_t::BrokenFile, "num of vec and num of key do not equal");
    }

    std::ifstream slot_stream;
    size_t slot_file_size_in_byte;
    if (!is_distributed_) {
      open_and_get_size(sparse_model_->slot_file, slot_stream, slot_file_size_in_byte);

      size_t num_slot = slot_file_size_in_byte / sizeof(size_t);
      if (num_key != num_slot) {
        CK_THROW_(Error_t::BrokenFile, "num of key and num of slot_id do not equal");
      }
    }

    std::vector<TypeHashKey> key_vec(num_key);
    std::vector<size_t> slot_id_vec(num_key);
    key_stream.read(reinterpret_cast<char *>(key_vec.data()), key_file_size_in_byte);
    if (!is_distributed_) {
      slot_stream.read(reinterpret_cast<char *>(slot_id_vec.data()), slot_file_size_in_byte);
    }

    for (size_t i = 0; i < num_key; i++) {
      size_t slot_id = is_distributed_ ? 0 : slot_id_vec[i];
      hash_table_.insert({key_vec[i], {slot_id, i}});
    }
  }
  catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
  catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeHashKey>
ParameterServer<TypeHashKey>::~ParameterServer() {
  if (maped_to_memory_) {
    unmap_embedding_from_memory_();
  }
}

template <typename TypeHashKey>
void ParameterServer<TypeHashKey>::load_keyset_from_file(
	std::string keyset_file) {
  try {
    std::ifstream keyset_stream;
    size_t file_size_in_byte = 0;
    open_and_get_size(keyset_file, keyset_stream, file_size_in_byte);

    size_t num_keys_in_file = file_size_in_byte / sizeof(TypeHashKey);
    keyset_.resize(num_keys_in_file);
    keyset_stream.read((char*)keyset_.data(), file_size_in_byte);
  }
  catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
  catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeHashKey>
void ParameterServer<TypeHashKey>::load_param_from_embedding_file(
     BufferBag& buf_bag, size_t& hit_size) {
  try {
    if (!keyset_.size()) {
      CK_THROW_(Error_t::WrongInput, "Keyset_ is empty");
    }

    TypeHashKey *keys = Tensor2<TypeHashKey>::stretch_from(buf_bag.keys).get_ptr();
    float *hash_table_val = buf_bag.embedding.get_ptr();
    size_t *slot_id = nullptr;
    if (!is_distributed_) {
      slot_id = Tensor2<size_t>::stretch_from(buf_bag.slot_id).get_ptr();
    }

    std::vector<size_t> idx_exist;
    idx_exist.reserve(keyset_.size());

    size_t cnt_hit_keys = 0;
    for (size_t cnt = 0; cnt < keyset_.size(); cnt++) {
      auto iter = hash_table_.find(keyset_[cnt]);
      if (iter == hash_table_.end()) continue;
      keys[cnt_hit_keys] = iter->first;
      if (!is_distributed_) {
        slot_id[cnt_hit_keys] = iter->second.first;
      }
      idx_exist.push_back(iter->second.second);
      cnt_hit_keys++;
    }

    MESSAGE_("Load embedding: load " + std::to_string(keyset_.size()) +
             " keys, hit " + std::to_string(cnt_hit_keys) +
             " (" + std::to_string(cnt_hit_keys * 100.0 / keyset_.size()) +
             "%) in trained table");

    if (cnt_hit_keys == 0) return;

    const size_t embedding_vector_size_in_byte = sizeof(float) * embedding_vec_size_;

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
        size_t src_idx = idx_exist[idx + i] * embedding_vec_size_;
        size_t dst_idx = (idx + i) * embedding_vec_size_;
        memcpy(&hash_table_val[dst_idx], &mmaped_table_[src_idx], embedding_vector_size_in_byte);
      }
    }

    hit_size = cnt_hit_keys;

  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}
  
template <typename TypeHashKey>
void ParameterServer<TypeHashKey>::dump_param_to_embedding_file(
     BufferBag &buf_bag, const size_t dump_size) {
  try {
    if (dump_size == 0) return;

    const TypeHashKey *keys = Tensor2<TypeHashKey>::stretch_from(buf_bag.keys).get_ptr();
    const float *hash_table_val = buf_bag.embedding.get_ptr();
    size_t *slot_id = nullptr;
    if (!is_distributed_) {
      slot_id = Tensor2<size_t>::stretch_from(buf_bag.slot_id).get_ptr();
    }

    size_t cnt_new_keys = 0;
    const size_t hash_table_size = hash_table_.size();

    std::vector<size_t> idx_exist_src, idx_exist_dst;
    std::vector<TypeHashKey> key_miss;
    std::vector<size_t> slot_id_miss;

    idx_exist_src.reserve(dump_size);
    idx_exist_dst.reserve(dump_size);

    for (size_t cnt = 0; cnt < dump_size; cnt++) {
      auto iter = hash_table_.find(keys[cnt]);
      if (iter == hash_table_.end()) {
        size_t slot_id_temp = is_distributed_ ? 0 : slot_id[cnt];
        hash_table_.insert({keys[cnt], {slot_id_temp, hash_table_size + cnt_new_keys}});

        key_miss.push_back(keys[cnt]);
        if (!is_distributed_) {
          slot_id_miss.push_back(slot_id[cnt]);
        }

        idx_exist_src.push_back(cnt);
        idx_exist_dst.push_back(hash_table_size + cnt_new_keys);

        cnt_new_keys++;
      } else {
        idx_exist_src.push_back(cnt);
        idx_exist_dst.push_back(iter->second.second);
      }
    }

    MESSAGE_("Dump embedding: dump " + std::to_string(dump_size) + " keys, hit " +
             std::to_string(idx_exist_src.size() - cnt_new_keys) +
             " (" + std::to_string((idx_exist_src.size() - cnt_new_keys) * 100.0 / dump_size) +
             "%) in trained table");

    const size_t embedding_vector_size_in_byte = sizeof(float) * embedding_vec_size_;

    if (maped_to_memory_) {
      unmap_embedding_from_memory_();
    }

    // resize the embedding vector file to hold vectors corresponding to new keys
    size_t extended_vec_file_size =
        fs::file_size(sparse_model_->vec_file) + cnt_new_keys * embedding_vector_size_in_byte;

    fs::resize_file(sparse_model_->vec_file, extended_vec_file_size);

    // update embedding vectors
    map_embedding_to_memory_();

#pragma omp parallel num_threads(8)
    {
      const size_t tid = omp_get_thread_num();
      const size_t thread_num = omp_get_num_threads();
      size_t sub_chunk_size = idx_exist_src.size() / thread_num;
      size_t res_chunk_size = idx_exist_src.size() % thread_num;
      const size_t idx = tid * sub_chunk_size;

      if (tid == thread_num - 1) sub_chunk_size += res_chunk_size;

      for (size_t i = 0; i < sub_chunk_size; i++) {
        size_t src_idx = idx_exist_src[idx + i] * embedding_vec_size_;
        size_t dst_idx = idx_exist_dst[idx + i] * embedding_vec_size_;
        memcpy(&mmaped_table_[dst_idx], &hash_table_val[src_idx], embedding_vector_size_in_byte);
      }
    }

    // append new key & slot_id to file
    std::ofstream key_stream(sparse_model_->key_file, std::ofstream::binary | std::ofstream::app);
    if (!key_stream.is_open()) {
      CK_THROW_(Error_t::WrongInput, "Cannot open " + sparse_model_->key_file);
    }
    key_stream.write(reinterpret_cast<char *>(key_miss.data()), cnt_new_keys * sizeof(TypeHashKey));

    if (!is_distributed_) {
      std::ofstream slot_stream(sparse_model_->slot_file, std::ofstream::binary | std::ofstream::app);
      if (!slot_stream.is_open()) {
        CK_THROW_(Error_t::WrongInput, "Cannot open " + sparse_model_->slot_file);
      }
      slot_stream.write(reinterpret_cast<char *>(slot_id_miss.data()), cnt_new_keys * sizeof(size_t));
    }
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }

}

template <typename TypeHashKey>
std::vector<TypeHashKey>
ParameterServer<TypeHashKey>::get_keys_from_hash_table() const {
  auto get_key_op = [](auto e){ return e.first; };
  std::vector<TypeHashKey> keys(hash_table_.size());
  transform(hash_table_.begin(), hash_table_.end(), keys.begin(), get_key_op);
  return keys;
}

template class ParameterServer<long long>;
template class ParameterServer<unsigned>;

}  // namespace HugeCTR
