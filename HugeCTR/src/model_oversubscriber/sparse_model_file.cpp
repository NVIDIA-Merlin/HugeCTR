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
#include <cstdlib>
#include <experimental/filesystem>
#include <fstream>
#include <map>
#include <model_oversubscriber/sparse_model_file.hpp>

namespace fs = std::experimental::filesystem;

namespace HugeCTR {

namespace {

void open_and_get_size(const std::string& file_name, std::ifstream& stream,
                       size_t& file_size_in_byte) {
  stream.open(file_name, std::ifstream::binary);
  if (!stream.is_open()) {
    CK_THROW_(Error_t::WrongInput, "Cannot open the file: " + file_name);
  }

  file_size_in_byte = fs::file_size(file_name);
}

}  // namespace

template <typename TypeKey>
struct SparseModelFile<TypeKey>::EmbeddingTableFile {
  std::string folder_name;
  std::string key_file;
  std::string slot_file;
  std::string vec_file;

  EmbeddingTableFile(std::string sparse_model) : folder_name(sparse_model) {
    key_file = sparse_model + "/key";
    slot_file = sparse_model + "/slot_id";
    vec_file = sparse_model + "/emb_vector";
  }
};

template <typename TypeKey>
void SparseModelFile<TypeKey>::map_embedding_to_memory_() {
  try {
    const char* emb_vec_file = mmap_handler_.get_vec_file();
    int fd = open(emb_vec_file, O_RDWR, S_IRUSR | S_IWUSR);
    if (fd == -1) {
      CK_THROW_(Error_t::FileCannotOpen, std::string("Cannot open the file: ") + emb_vec_file);
    }

    size_t vec_file_size_in_byte = fs::file_size(emb_vec_file);
    if (vec_file_size_in_byte == 0) {
      CK_THROW_(Error_t::WrongInput, std::string("Cannot mmap empty file: ") + emb_vec_file);
    }

    mmap_handler_.mmaped_table_ =
        (float*)mmap(NULL, vec_file_size_in_byte, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mmap_handler_.mmaped_table_ == MAP_FAILED) {
      close(fd);
      fd = -1;
      mmap_handler_.mmaped_table_ = nullptr;
      CK_THROW_(Error_t::WrongInput, std::string("Mmap file ") + emb_vec_file + " failed");
    }

    mmap_handler_.maped_to_memory_ = true;
    if (fd != -1) {
      close(fd);
    }
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
void SparseModelFile<TypeKey>::sync_mmaped_embedding_with_disk_() {
  try {
    if (!mmap_handler_.maped_to_memory_) {
      CK_THROW_(Error_t::IllegalCall,
                std::string(mmap_handler_.get_vec_file()) + " not mapped to HMEM");
    }
    const char* emb_vec_file = mmap_handler_.get_vec_file();
    size_t vec_file_size_in_byte = fs::file_size(emb_vec_file);
    int ret = msync(mmap_handler_.mmaped_table_, vec_file_size_in_byte, MS_SYNC);
    if (ret != 0) {
      CK_THROW_(Error_t::WrongInput, "Mmap sync error");
    }
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
void SparseModelFile<TypeKey>::unmap_embedding_from_memory_() {
  try {
    const char* emb_vec_file = mmap_handler_.get_vec_file();
    if (mmap_handler_.maped_to_memory_ && mmap_handler_.mmaped_table_ != nullptr) {
      size_t vec_file_size_in_byte = fs::file_size(emb_vec_file);
      munmap(mmap_handler_.mmaped_table_, vec_file_size_in_byte);
      mmap_handler_.mmaped_table_ = nullptr;
      mmap_handler_.maped_to_memory_ = false;
    } else {
      CK_THROW_(Error_t::IllegalCall,
                std::string(mmap_handler_.get_vec_file()) + " not mapped to HMEM");
    }
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
SparseModelFile<TypeKey>::SparseModelFile(const std::string& sparse_model_file,
                                          Embedding_t embedding_type, size_t emb_vec_size,
                                          std::shared_ptr<ResourceManager> resource_manager)
    : is_distributed_(embedding_type == Embedding_t::DistributedSlotSparseEmbeddingHash),
      emb_vec_size_(emb_vec_size),
      resource_manager_(resource_manager) {
  try {
    mmap_handler_.emb_tbl_.reset(new EmbeddingTableFile(sparse_model_file));
    if (!fs::exists(mmap_handler_.get_folder_name())) {
#ifdef ENABLE_MPI
      CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
#endif
      if (resource_manager_->is_master_process()) {
        MESSAGE_(sparse_model_file + " not exist, create and train from scratch");
        fs::create_directory(mmap_handler_.get_folder_name());

        int ret;
        std::string command("touch ");
        ret = std::system((command + mmap_handler_.get_key_file()).c_str());
        ret = std::system((command + mmap_handler_.get_vec_file()).c_str());
        if (!is_distributed_) {
          ret = std::system((command + mmap_handler_.get_slot_file()).c_str());
        }
        (void)ret;
      }
      return;
    }

    std::ifstream key_stream;
    size_t key_file_size_in_byte;
    open_and_get_size(mmap_handler_.get_key_file(), key_stream, key_file_size_in_byte);

    size_t num_key = key_file_size_in_byte / sizeof(long long);
    size_t num_vec = fs::file_size(mmap_handler_.get_vec_file()) / (sizeof(float) * emb_vec_size_);
    if (num_key != num_vec) {
      CK_THROW_(Error_t::BrokenFile, "num of vec and num of key do not equal");
    }

    std::ifstream slot_stream;
    size_t slot_file_size_in_byte;
    if (!is_distributed_) {
      open_and_get_size(mmap_handler_.get_slot_file(), slot_stream, slot_file_size_in_byte);
      size_t num_slot = slot_file_size_in_byte / sizeof(size_t);
      if (num_key != num_slot) {
        CK_THROW_(Error_t::BrokenFile, "num of key and num of slot_id do not equal");
      }
    }

    std::vector<TypeKey> key_vec(num_key);
    std::vector<size_t> slot_id_vec(num_key);
    if (std::is_same<TypeKey, long long>::value) {
      key_stream.read(reinterpret_cast<char*>(key_vec.data()), key_file_size_in_byte);
    } else {
      std::vector<long long> i64_key_vec(num_key, 0);
      key_stream.read(reinterpret_cast<char*>(i64_key_vec.data()), key_file_size_in_byte);
      std::transform(i64_key_vec.begin(), i64_key_vec.end(), key_vec.begin(),
                     [](long long key) { return static_cast<unsigned>(key); });
    }
    if (!is_distributed_) {
      slot_stream.read(reinterpret_cast<char*>(slot_id_vec.data()), slot_file_size_in_byte);
    }

    // each rank stores a subset of embedding table
    int my_rank = resource_manager_->get_process_id();
    for (size_t i = 0; i < num_key; i++) {
      int dst_rank;
      if (is_distributed_) {
        TypeKey key = key_vec[i];
        size_t gid = key % resource_manager_->get_global_gpu_count();
        dst_rank = resource_manager_->get_process_id_from_gpu_global_id(gid);
      } else {
        size_t slot_id = slot_id_vec[i];
        size_t gid = slot_id % resource_manager_->get_global_gpu_count();
        dst_rank = resource_manager_->get_process_id_from_gpu_global_id(gid);
      }
      if (my_rank == dst_rank) {
        size_t slot_id = is_distributed_ ? 0 : slot_id_vec[i];
        key_idx_map_.insert({key_vec[i], {slot_id, i}});
      }
    }
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
void SparseModelFile<TypeKey>::load_exist_vec_by_key(const std::vector<TypeKey>& keys,
                                                     std::vector<size_t>& slots,
                                                     std::vector<float>& vecs) {
  try {
    if (keys.size() == 0) return;

    if (!is_distributed_) {
      slots.resize(keys.size());
    }
    vecs.resize(keys.size() * emb_vec_size_);
    const size_t emb_vec_size_in_byte = emb_vec_size_ * sizeof(float);

    map_embedding_to_memory_();
#pragma omp parallel num_threads(8)
    {
      const size_t tid = omp_get_thread_num();
      const size_t thread_num = omp_get_num_threads();
      size_t sub_chunk_size = keys.size() / thread_num;
      size_t res_chunk_size = keys.size() % thread_num;
      const size_t idx = tid * sub_chunk_size;

      if (tid == thread_num - 1) sub_chunk_size += res_chunk_size;

      for (size_t i = 0; i < sub_chunk_size; i++) {
        const auto& pair = key_idx_map_.at(keys[idx + i]);
        if (!is_distributed_) slots[idx + i] = pair.first;
        size_t src_vec_idx = pair.second * emb_vec_size_;
        size_t dst_vec_idx = (idx + i) * emb_vec_size_;
        memcpy(&vecs[dst_vec_idx], &(mmap_handler_.mmaped_table_[src_vec_idx]),
               emb_vec_size_in_byte);
      }
    }
    sync_mmaped_embedding_with_disk_();
    unmap_embedding_from_memory_();
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
void SparseModelFile<TypeKey>::dump_exist_vec_by_key(const std::vector<TypeKey>& keys,
                                                     const std::vector<size_t>& vec_indices,
                                                     const float* vecs) {
  try {
    if (keys.size() != vec_indices.size()) {
      CK_THROW_(Error_t::WrongInput, "keys.size() != vec_indices.size()");
    }
    if (keys.size() == 0) return;

    const size_t emb_vec_size_in_byte = emb_vec_size_ * sizeof(float);

    map_embedding_to_memory_();
#pragma omp parallel num_threads(8)
    {
      const size_t tid = omp_get_thread_num();
      const size_t thread_num = omp_get_num_threads();
      size_t sub_chunk_size = keys.size() / thread_num;
      size_t res_chunk_size = keys.size() % thread_num;
      const size_t idx = tid * sub_chunk_size;

      if (tid == thread_num - 1) sub_chunk_size += res_chunk_size;

      for (size_t i = 0; i < sub_chunk_size; i++) {
        size_t src_vec_idx = vec_indices[idx + i] * emb_vec_size_;
        const auto& pair = key_idx_map_.at(keys[idx + i]);
        size_t dst_vec_idx = pair.second * emb_vec_size_;
        memcpy(&(mmap_handler_.mmaped_table_[dst_vec_idx]), &vecs[src_vec_idx],
               emb_vec_size_in_byte);
      }
    }
    sync_mmaped_embedding_with_disk_();
    unmap_embedding_from_memory_();
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
void SparseModelFile<TypeKey>::append_new_vec_and_key(const std::vector<TypeKey>& keys,
                                                      const size_t* slots,
                                                      const std::vector<size_t>& vec_indices,
                                                      const float* vecs) {
  try {
    if (keys.size() != vec_indices.size()) {
      CK_THROW_(Error_t::WrongInput, "keys.size() != vec_indices.size()");
    }
    auto check_key_exists_op = [this](auto key) {
      if (this->key_idx_map_.find(key) != this->key_idx_map_.end())
        CK_THROW_(Error_t::WrongInput, std::to_string(key) + " exists in key_idx_map_!");
    };
    std::for_each(keys.begin(), keys.end(), check_key_exists_op);

    long long* key_ptr = nullptr;
    std::vector<long long> i64_keys;
    if (std::is_same<TypeKey, long long>::value) {
      key_ptr = const_cast<long long*>(reinterpret_cast<const long long*>(keys.data()));
    } else {
      i64_keys.resize(keys.size());
      std::transform(keys.begin(), keys.end(), i64_keys.begin(),
                     [](unsigned key) { return static_cast<long long>(key); });
      key_ptr = i64_keys.data();
    }

    const size_t emb_vec_size_in_byte = emb_vec_size_ * sizeof(float);
    const size_t num_vec_in_file =
        fs::file_size(mmap_handler_.get_vec_file()) / emb_vec_size_in_byte;
    // write keys, slots, vectors to file
    std::ofstream key_ofs(mmap_handler_.get_key_file(), std::ofstream::out | std::ofstream::app);
    if (!key_ofs.is_open()) {
      CK_THROW_(Error_t::FileCannotOpen, "Cannot open key file");
    }
    key_ofs.write(reinterpret_cast<const char*>(key_ptr), keys.size() * sizeof(long long));

    if (!is_distributed_) {
      std::ofstream slot_ofs(mmap_handler_.get_slot_file(),
                             std::ofstream::out | std::ofstream::app);
      if (!slot_ofs.is_open()) {
        CK_THROW_(Error_t::FileCannotOpen, "Cannot open slot file");
      }
      slot_ofs.write(reinterpret_cast<const char*>(slots), keys.size() * sizeof(size_t));
    }

    size_t extended_vec_file_size =
        fs::file_size(mmap_handler_.get_vec_file()) + keys.size() * emb_vec_size_in_byte;
    fs::resize_file(mmap_handler_.get_vec_file(), extended_vec_file_size);

    // update key_idx_map_
    for (size_t i = 0; i < keys.size(); i++) {
      size_t slot_id = is_distributed_ ? 0 : slots[i];
      key_idx_map_.insert({keys[i], {slot_id, num_vec_in_file + i}});
    }

    // write embedding vector to disk
    dump_exist_vec_by_key(keys, vec_indices, vecs);
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
void SparseModelFile<TypeKey>::load_emb_tbl_to_mem(HashTableType& mem_key_index_map,
                                                   std::vector<float>& vecs) {
  try {
    const size_t num_vecs = key_idx_map_.size();
    mem_key_index_map.clear();
    mem_key_index_map.reserve(num_vecs);
    vecs.resize(num_vecs * emb_vec_size_);

    std::vector<TypeKey> exist_key;
    exist_key.reserve(num_vecs);
    size_t counter = 0;
    for (auto key_idx_pair : key_idx_map_) {
      exist_key.push_back(key_idx_pair.first);
      key_idx_pair.second.second = counter++;
      mem_key_index_map.emplace(std::move(key_idx_pair));
    }

    std::vector<size_t> temp_slots;
    load_exist_vec_by_key(exist_key, temp_slots, vecs);
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template class SparseModelFile<long long>;
template class SparseModelFile<unsigned>;

}  // namespace HugeCTR
