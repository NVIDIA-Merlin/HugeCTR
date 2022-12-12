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
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <embedding_training_cache/hmem_cache/sparse_model_file_ts.hpp>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>

namespace HugeCTR {

namespace {

inline void open_and_get_size(const std::string& file_name, std::ifstream& stream,
                              size_t& file_size_in_byte) {
  stream.open(file_name, std::ifstream::binary);
  if (!stream.is_open()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Cannot open the file: " + file_name);
  }
  file_size_in_byte = std::filesystem::file_size(file_name);
}

template <typename T>
std::vector<T> load_data_from_file(std::string file_path) {
  if (!std::filesystem::exists(file_path)) {
    HCTR_OWN_THROW(Error_t::FileCannotOpen, file_path + " doesn't exist.");
  }
  std::ifstream ifs;
  size_t file_size;
  open_and_get_size(file_path, ifs, file_size);

  std::vector<T> data_vec(std::filesystem::file_size(file_path) / sizeof(T));
  ifs.read(reinterpret_cast<char*>(data_vec.data()), file_size);

  return data_vec;
}

template <typename T>
void mmap_file_to_memory(T** mmaped_ptr, std::string file_name) {
  if (!std::filesystem::exists(file_name)) {
    HCTR_OWN_THROW(Error_t::FileCannotOpen, file_name + " doesn't exist");
  }
  int fd{open(file_name.c_str(), O_RDWR, S_IRUSR | S_IWUSR)};
  if (fd == -1) {
    HCTR_OWN_THROW(Error_t::FileCannotOpen, "Can't open " + file_name);
  }
  size_t file_size_in_byte{std::filesystem::file_size(file_name)};
  if (file_size_in_byte == 0) {
    return;
    // TODO: Unreachable code! Why?
    HCTR_OWN_THROW(Error_t::WrongInput, "Can't mmap empty file " + file_name);
  }
  *mmaped_ptr = (T*)mmap(NULL, file_size_in_byte, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (*mmaped_ptr == MAP_FAILED) {
    close(fd);
    HCTR_OWN_THROW(Error_t::WrongInput, "Fail mmap file " + file_name);
  }
  if (fd != -1) {
    close(fd);
  }
}

template <typename T>
void sync_mmap_with_disk(T** mmaped_ptr, std::string file_name) {
  if (*mmaped_ptr == nullptr) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Invalid pointer");
  }
  size_t file_size_in_byte{std::filesystem::file_size(file_name)};
  auto ret{msync(*mmaped_ptr, file_size_in_byte, MS_SYNC)};
  if (ret != 0) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Mmap sync error");
  }
}

template <typename T>
void unmap_file_from_memory(T** mmaped_ptr, std::string file_name) {
  if (*mmaped_ptr == nullptr) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Can't unmap nullptr");
  }
  size_t file_size_in_byte{std::filesystem::file_size(file_name)};
  munmap(*mmaped_ptr, file_size_in_byte);
  *mmaped_ptr = nullptr;
}

}  // namespace

template <typename TypeKey>
size_t SparseModelFileTS<TypeKey>::num_instance = 0;

template <typename TypeKey>
struct SparseModelFileTS<TypeKey>::EmbeddingTableFile {
  std::string folder_name;
  std::string key_file;
  std::string slot_file;
  std::vector<std::string> data_files;

  EmbeddingTableFile(std::string local_model, Optimizer_t opt_type) : folder_name(local_model) {
    key_file = local_model + "/key";
    slot_file = local_model + "/slot_id";

    data_files.clear();
    data_files.emplace_back(local_model + "/emb_vector");

    switch (opt_type) {
      case Optimizer_t::Ftrl:
        data_files.emplace_back(local_model + "/Ftrl.n");
        data_files.emplace_back(local_model + "/Ftrl.z");
        break;
      case Optimizer_t::Adam:
        // Adam.m & Adam.v & (Update_t == LazyGlobal is not supported)
        data_files.emplace_back(local_model + "/Adam.m");
        data_files.emplace_back(local_model + "/Adam.v");
        break;
      case Optimizer_t::AdaGrad:
        data_files.emplace_back(local_model + "/AdaGrad.accm");
        break;
      case Optimizer_t::MomentumSGD:
        data_files.emplace_back(local_model + "/MomentumSGD.momtentum");
        break;
      case Optimizer_t::Nesterov:
        data_files.emplace_back(local_model + "/Nesterov.accm");
        break;
      case Optimizer_t::SGD:
        break;
      default:
        HCTR_OWN_THROW(Error_t::WrongInput, "Wrong optimizer type");
    }
  }
};

template <typename TypeKey>
void SparseModelFileTS<TypeKey>::mmap_to_memory_() {
  try {
    if (mmap_handler_.mapped_to_file_) {
      HCTR_OWN_THROW(Error_t::IllegalCall, "data files are already mapped to memory");
    }
    const auto& data_files{mmap_handler_.get_data_files()};
    {
      std::vector<bool> is_empty(data_files.size());
      auto cnt{0};
      for (auto& data_file : data_files) {
        is_empty[cnt++] = (std::filesystem::file_size(data_file) == 0);
      }
      auto has_empty{std::any_of(is_empty.begin(), is_empty.end(), [](auto val) { return val; })};
      if (has_empty) {
        auto all_empty{std::all_of(is_empty.begin(), is_empty.end(), [](auto val) { return val; })};
        if (!all_empty) {
          HCTR_OWN_THROW(Error_t::BrokenFile,
                         "Broken sparse model, some files are empty while others are not");
        }
        return;
      }
    }
    for (size_t i{0}; i < data_files.size(); i++) {
      mmap_file_to_memory(&(mmap_handler_.mmaped_ptrs_[i]), data_files[i]);
    }
    mmap_handler_.mapped_to_file_ = true;
  } catch (const internal_runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
void SparseModelFileTS<TypeKey>::flush_mmap_to_disk_() {
  try {
    if (!mmap_handler_.mapped_to_file_) {
      HCTR_OWN_THROW(Error_t::IllegalCall, "data files doesn't mapped to memory");
    }
    const auto& data_files{mmap_handler_.get_data_files()};
    for (size_t i{0}; i < data_files.size(); i++) {
      sync_mmap_with_disk(&(mmap_handler_.mmaped_ptrs_[i]), data_files[i]);
    }
  } catch (const internal_runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
void SparseModelFileTS<TypeKey>::unmap_from_memory_() {
  try {
    if (!mmap_handler_.mapped_to_file_) {
      HCTR_OWN_THROW(Error_t::IllegalCall, "data files doesn't mapped to memory");
    }
    const auto& data_files{mmap_handler_.get_data_files()};
    for (size_t i{0}; i < data_files.size(); i++) {
      unmap_file_from_memory(&(mmap_handler_.mmaped_ptrs_[i]), data_files[i]);
    }
    mmap_handler_.mapped_to_file_ = false;
  } catch (const internal_runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, ROOT) << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    HCTR_LOG_S(ERROR, ROOT) << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
void SparseModelFileTS<TypeKey>::expand_(size_t expand_size) {
  try {
    if (expand_size == 0) return;
    if (mmap_handler_.mapped_to_file_) {
      flush_mmap_to_disk_();
      unmap_from_memory_();
    }
    size_t file_size_expand{expand_size * sizeof(float) * emb_vec_size_};
    for (const auto& file_name : mmap_handler_.get_data_files()) {
      std::filesystem::resize_file(file_name,
                                   std::filesystem::file_size(file_name) + file_size_expand);
    }
    mmap_to_memory_();

    size_t const expanded_num{key_idx_map_.size() + expand_size};
    if (slot_ids.size() < expanded_num) slot_ids.resize(expanded_num);
  } catch (const internal_runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
SparseModelFileTS<TypeKey>::SparseModelFileTS(std::string sparse_model_path,
                                              std::string local_temp_path, bool use_slot_id,
                                              Optimizer_t opt_type, size_t emb_vec_size,
                                              std::shared_ptr<ResourceManager> resource_manager)
    : global_model_path_(sparse_model_path),
      opt_type_(opt_type),
      use_slot_id_(use_slot_id),
      emb_vec_size_(emb_vec_size),
      resource_manager_(resource_manager) {
  try {
    auto check_integrate_and_init = [&](EmbeddingTableFile& etf) {
      auto const num_key{std::filesystem::file_size(etf.key_file) / sizeof(long long)};
      if (use_slot_id_) {
        if (!std::filesystem::exists(etf.slot_file)) {
          HCTR_OWN_THROW(Error_t::BrokenFile, "Can't find " + etf.folder_name + "/slot_id");
        }
        auto num_slot_id{std::filesystem::file_size(etf.slot_file) / sizeof(size_t)};
        if (num_slot_id != num_key) {
          std::ostringstream os;
          os << "Num of keys(" << num_key << ") != num of slot_id(" << num_slot_id << ")";
          HCTR_OWN_THROW(Error_t::BrokenFile, os.str());
        }
      }
      auto num_vec{std::filesystem::file_size(etf.data_files[0]) / emb_vec_size_ / sizeof(float)};
      if (num_vec != num_key) {
        std::ostringstream os;
        os << "Num of keys(" << num_key << ") != num of embedding vectors(" << num_vec << ")";
        HCTR_OWN_THROW(Error_t::BrokenFile, os.str());
      }
      // check whether the opt state file exists, create if not exist
      for (size_t i{1}; i < etf.data_files.size(); i++) {
        auto file_name{etf.data_files[i]};
        if (std::filesystem::exists(file_name)) {
          auto num_state{std::filesystem::file_size(file_name) / emb_vec_size_ / sizeof(float)};
          if (num_state != num_key) {
            std::ostringstream os;
            os << "Num of keys(" << num_key << ") != num of opt states(" << num_vec << ") in "
               << file_name;
            HCTR_OWN_THROW(Error_t::BrokenFile, os.str());
          }
        } else {
          HCTR_LOG_S(INFO, ROOT) << file_name << " doesn't exist, create and initialize with zeros"
                                 << std::endl;
          auto ret = std::system((std::string("touch ") + file_name).c_str());
          (void)ret;
          std::filesystem::resize_file(file_name, std::filesystem::file_size(etf.data_files[0]));
        }
      }
    };

    auto create_file_in_ssd = [](std::string file_path) {
      HCTR_LOG_S(INFO, ROOT) << file_path << " doesn't exist, created" << std::endl;
      auto ret = std::system((std::string("touch ") + file_path).c_str());
      (void)ret;
    };

    auto create_sparse_model = [&](EmbeddingTableFile& emb_tbl) {
      if (!std::filesystem::exists(emb_tbl.folder_name)) {
        std::filesystem::create_directories(emb_tbl.folder_name);
      }
      create_file_in_ssd(emb_tbl.key_file);
      if (use_slot_id_) create_file_in_ssd(emb_tbl.slot_file);
      for (const auto& file_name : emb_tbl.data_files) {
        create_file_in_ssd(file_name);
      }
    };

    bool const from_scratch{!(std::filesystem::exists(sparse_model_path) &&
                              std::filesystem::is_directory(sparse_model_path))};
    bool const localized_train{(resource_manager_->get_num_process() == 1)};
    EmbeddingTableFile global_sparse_model(sparse_model_path, opt_type);

    // Train using a single node, read/write directly to the global_sparse_model
    if (localized_train) {
      mmap_handler_.emb_tbl_.reset(new EmbeddingTableFile(sparse_model_path, opt_type));
      mmap_handler_.mmaped_ptrs_ =
          std::vector<float*>(mmap_handler_.get_data_files().size(), nullptr);
      if (from_scratch) {
        create_sparse_model(global_sparse_model);
      } else {
        check_integrate_and_init(global_sparse_model);
        // initialize the key<-->ssd mapping
        auto keys{load_data_from_file<long long>(mmap_handler_.get_key_file())};
        for (size_t i{0}; i < keys.size(); i++) {
          key_idx_map_.insert({static_cast<TypeKey>(keys[i]), i});
        }
        if (use_slot_id_) {
          slot_ids = load_data_from_file<size_t>(mmap_handler_.get_slot_file());
        }
      }
      return;
    }

    // Train using multiple nodes, read/write to the local copy of sparse model
    auto const my_rank{resource_manager_->get_process_id()};
    {
      std::ostringstream os;
      os << local_temp_path << "/tmp_hctr." << num_instance++ << '.' << my_rank;
      mmap_handler_.emb_tbl_.reset(new EmbeddingTableFile(os.str(), opt_type));
    }
    mmap_handler_.mmaped_ptrs_ =
        std::vector<float*>(mmap_handler_.get_data_files().size(), nullptr);

    std::vector<bool> data_exists(mmap_handler_.get_data_files().size(), true);

    // check the existance of sparse model file to be used
    if (std::filesystem::exists(mmap_handler_.get_folder_name()) && my_rank == 0) {
      HCTR_LOG_S(INFO, ROOT) << "Remove existing " << mmap_handler_.get_folder_name() << std::endl;
      std::filesystem::remove_all(mmap_handler_.get_folder_name());
    }

    if (!std::filesystem::exists(sparse_model_path)) {
      data_exists = std::vector(mmap_handler_.get_data_files().size(), false);
#ifdef ENABLE_MPI
      HCTR_MPI_THROW(MPI_Barrier(MPI_COMM_WORLD));
#endif
      if (resource_manager_->is_master_process()) {
        create_sparse_model(global_sparse_model);
      }
    } else {
      auto data_files{global_sparse_model.data_files};
      for (size_t i{1}; i < data_files.size(); i++) {
        if (!std::filesystem::exists(data_files[i])) {
          data_exists[i] = false;
#ifdef ENABLE_MPI
          HCTR_MPI_THROW(MPI_Barrier(MPI_COMM_WORLD));
#endif
          if (resource_manager_->is_master_process()) {
            create_file_in_ssd(data_files[i]);
          }
        }
      }
    }
    create_sparse_model(*(mmap_handler_.emb_tbl_.get()));
#ifdef ENABLE_MPI
    HCTR_MPI_THROW(MPI_Barrier(MPI_COMM_WORLD));
#endif

    if (from_scratch) return;

    // check whether num_key, num_slot_id, num_vec, num_opt_state are equal
    std::ifstream key_stream;
    size_t key_file_size_in_byte;
    open_and_get_size(global_sparse_model.key_file, key_stream, key_file_size_in_byte);
    size_t num_key{key_file_size_in_byte / sizeof(long long)};

    std::ifstream slot_stream;
    size_t slot_file_size_in_byte;
    if (use_slot_id_) {
      open_and_get_size(global_sparse_model.slot_file, slot_stream, slot_file_size_in_byte);
      size_t num_slot{slot_file_size_in_byte / sizeof(size_t)};
      if (num_key != num_slot) {
        HCTR_OWN_THROW(Error_t::BrokenFile, "key and slot_id num do not equal");
      }
    }

    std::vector<size_t> num_of_data;
    auto global_data_files{global_sparse_model.data_files};
    for (size_t i{0}; i < global_data_files.size(); i++) {
      if (data_exists[i]) {
        size_t num_bytes{std::filesystem::file_size(global_data_files[i])};
        num_of_data.push_back(num_bytes / (sizeof(float) * emb_vec_size_));
      }
    }
    auto is_broken_file{std::all_of(num_of_data.begin(), num_of_data.end(),
                                    [=](auto elem) { return elem == num_key; })};
    if (!is_broken_file) {
      HCTR_OWN_THROW(Error_t::BrokenFile, "num of vec/opt_states and num of key do not equal");
    }

    // load key & slot_id from the global sparse model
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
    if (use_slot_id_) {
      slot_stream.read(reinterpret_cast<char*>(slot_id_vec.data()), slot_file_size_in_byte);
    }

    // filter keys belongs to the current processor
    std::unordered_map<TypeKey, size_t> global_key_idx_map_;
    slot_ids.resize(slot_id_vec.size());
    size_t counter{0};
    for (size_t i{0}; i < num_key; i++) {
      int dst_rank;
      if (!use_slot_id_) {
        TypeKey key = key_vec[i];
        size_t gid = key % resource_manager_->get_global_gpu_count();
        dst_rank = resource_manager_->get_process_id_from_gpu_global_id(gid);
      } else {
        size_t slot_id = slot_id_vec[i];
        size_t gid = slot_id % resource_manager_->get_global_gpu_count();
        dst_rank = resource_manager_->get_process_id_from_gpu_global_id(gid);
      }
      if (my_rank == dst_rank) {
        global_key_idx_map_.insert({key_vec[i], i});
        if (use_slot_id_) slot_ids[counter] = slot_id_vec[i];
        key_idx_map_.insert({key_vec[i], counter++});
      }
    }

    // load data from the global sparse model file to the local SSD
    auto local_num_key{global_key_idx_map_.size()};
    auto local_data_file_size{local_num_key * sizeof(float) * emb_vec_size_};
    for (auto file : mmap_handler_.get_data_files()) {
      std::filesystem::resize_file(file, local_data_file_size);
    }

    mmap_to_memory_();

    for (size_t i{0}; i < global_data_files.size(); i++) {
      if (!data_exists[i]) continue;
      float* temp_mmaped_ptr;
      mmap_file_to_memory(&temp_mmaped_ptr, global_data_files[i]);
      for (auto pair : global_key_idx_map_) {
        auto src_idx{pair.second * emb_vec_size_};

        auto it{key_idx_map_.find(pair.first)};
        if (it == key_idx_map_.end()) {
          HCTR_OWN_THROW(Error_t::UnspecificError, "Key doesn't found in init");
        }
        auto dst_idx{it->second * emb_vec_size_};
        memcpy(mmap_handler_.mmaped_ptrs_[i] + dst_idx, temp_mmaped_ptr + src_idx,
               emb_vec_size_ * sizeof(float));
      }
      unmap_file_from_memory(&temp_mmaped_ptr, global_data_files[i]);
    }
#ifdef ENABLE_MPI
    HCTR_MPI_THROW(MPI_Barrier(MPI_COMM_WORLD));
#endif
  } catch (const internal_runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
SparseModelFileTS<TypeKey>::~SparseModelFileTS() {
  if (mmap_handler_.mapped_to_file_) unmap_from_memory_();
  bool const localized_train{(resource_manager_->get_num_process() == 1)};
  if (!localized_train && std::filesystem::exists(mmap_handler_.get_folder_name())) {
    std::filesystem::remove_all(mmap_handler_.get_folder_name());
  }
}

template <typename TypeKey>
void SparseModelFileTS<TypeKey>::load(std::vector<size_t> const& mem_src_idx, size_t* slot_id_ptr,
                                      std::vector<float*>& data_ptrs) {
  try {
    if (!mmap_handler_.mapped_to_file_) {
      mmap_to_memory_();
    }
    if (data_ptrs.size() != mmap_handler_.mmaped_ptrs_.size()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Num of data files and pointers doesn't equal");
    }
    size_t const len{mem_src_idx.size()};
#pragma omp parallel for num_threads(24)
    for (size_t cnt = 0; cnt < len; cnt++) {
      auto src_idx{mem_src_idx[cnt]};
      if (use_slot_id_) slot_id_ptr[cnt] = slot_ids[src_idx];
      for (size_t i{0}; i < data_ptrs.size(); i++) {
        float* src_ptr{mmap_handler_.mmaped_ptrs_[i] + src_idx * emb_vec_size_};
        float* dst_ptr{data_ptrs[i] + cnt * emb_vec_size_};
        memcpy(dst_ptr, src_ptr, sizeof(float) * emb_vec_size_);
      }
    }
  } catch (const internal_runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
void SparseModelFileTS<TypeKey>::dump_update(HashTableType& dump_key_idx_map,
                                             std::vector<size_t>& slot_id_vec,
                                             std::vector<std::vector<float>>& data_vecs) {
  try {
    if (dump_key_idx_map.size() == 0) return;
    if (!mmap_handler_.mapped_to_file_) {
      mmap_to_memory_();
    }
    if (data_vecs.size() != mmap_handler_.mmaped_ptrs_.size()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Num of data files and pointers doesn't equal");
    }
    std::vector<TypeKey> keys_vec;
    keys_vec.reserve(dump_key_idx_map.size());
    std::vector<size_t> mem_idx_vec;
    mem_idx_vec.reserve(dump_key_idx_map.size());
    std::for_each(dump_key_idx_map.begin(), dump_key_idx_map.end(), [&](auto& pair) {
      keys_vec.push_back(pair.first);
      mem_idx_vec.push_back(pair.second);
    });

    size_t const num_thread(24);
    std::vector<std::vector<std::vector<size_t>>> sub_idx_vecs(num_thread,
                                                               std::vector<std::vector<size_t>>(2));
    size_t len{keys_vec.size()};
#pragma omp parallel num_threads(num_thread)
    {
      size_t const tid(omp_get_thread_num());
      auto sub_chunk_size{len / num_thread};
      auto const idx{sub_chunk_size * tid};
      if (tid == num_thread - 1) sub_chunk_size += (len % num_thread);

      for (size_t i{0}; i < 2; i++) {
        sub_idx_vecs[tid][i].reserve(sub_chunk_size);
      }
      for (size_t i{idx}; i < idx + sub_chunk_size; i++) {
        auto it{key_idx_map_.find(keys_vec[i])};
        if (it == key_idx_map_.end()) {
          HCTR_OWN_THROW(Error_t::WrongInput, "key not found");
        } else {
          sub_idx_vecs[tid][0].push_back(mem_idx_vec[i]);
          sub_idx_vecs[tid][1].push_back(it->second);
        }
      }
    }
    std::vector<std::vector<size_t>> idx_vecs(2);
#pragma omp parallel for num_threads(2)
    for (size_t i = 0; i < 2; i++) {
      size_t total_elem{0};
      for (size_t tid{0}; tid < num_thread; tid++) {
        total_elem += sub_idx_vecs[tid][i].size();
      }
      if (total_elem == 0) continue;
      idx_vecs[i].resize(total_elem);
      size_t offset{0};
      for (size_t tid{0}; tid < num_thread; tid++) {
        auto num_elem{sub_idx_vecs[tid][i].size()};
        if (num_elem == 0) continue;

        auto src_id_ptr{sub_idx_vecs[tid][i].data()};
        auto dst_id_ptr{idx_vecs[i].data() + offset};
        memcpy(dst_id_ptr, src_id_ptr, num_elem * sizeof(size_t));
        offset += num_elem;
      }
    }

    std::vector<float*> data_ptrs(data_vecs.size());
    for (size_t i{0}; i < data_vecs.size(); i++) {
      data_ptrs[i] = data_vecs[i].data();
    }
    dump_update(idx_vecs[1], idx_vecs[0], slot_id_vec.data(), data_ptrs);
  } catch (const internal_runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
void SparseModelFileTS<TypeKey>::dump_update(std::vector<size_t> const& ssd_idx_vec,
                                             std::vector<size_t> const& mem_idx_vec,
                                             size_t const* slot_id_ptr,
                                             std::vector<float*>& data_ptrs) {
  try {
    if (!mmap_handler_.mapped_to_file_) {
      mmap_to_memory_();
    }
    if (data_ptrs.size() != mmap_handler_.mmaped_ptrs_.size()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Num of data files and pointers doesn't equal");
    }
    if (ssd_idx_vec.size() != mem_idx_vec.size()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "ssd_idx_vec.size() != mem_idx_vec.size()");
    }
    size_t const len{ssd_idx_vec.size()};
#pragma omp parallel for num_threads(24)
    for (size_t cnt = 0; cnt < len; cnt++) {
      auto mem_idx{mem_idx_vec[cnt]};
      auto ssd_idx{ssd_idx_vec[cnt]};
      if (use_slot_id_) slot_ids[ssd_idx] = slot_id_ptr[mem_idx];
      for (size_t i{0}; i < data_ptrs.size(); i++) {
        float* mem_ptr{data_ptrs[i] + mem_idx * emb_vec_size_};
        float* ssd_ptr{mmap_handler_.mmaped_ptrs_[i] + ssd_idx * emb_vec_size_};
        memcpy(ssd_ptr, mem_ptr, sizeof(float) * emb_vec_size_);
      }
    }
  } catch (const internal_runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
void SparseModelFileTS<TypeKey>::dump_insert(TypeKey const* key_ptr,
                                             std::vector<size_t> const& mem_idx_vec,
                                             size_t const* slot_id_ptr,
                                             std::vector<float*>& data_ptrs) {
  try {
    if (!mmap_handler_.mapped_to_file_) {
      mmap_to_memory_();
    }
    if (data_ptrs.size() != mmap_handler_.mmaped_ptrs_.size()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Num of data files and pointers doesn't equal");
    }
    size_t const len{mem_idx_vec.size()}, num_exist_keys{key_idx_map_.size()};
    omp_set_nested(2);
#pragma omp parallel sections
    {
#pragma omp section
      {
        size_t cnt_new_keys{0};
        key_idx_map_.reserve(num_exist_keys + len);
        for (auto idx : mem_idx_vec) {
          key_idx_map_.insert({key_ptr[idx], num_exist_keys + cnt_new_keys++});
        }
      }
#pragma omp section
      {
        std::vector<size_t> ssd_idx_vec(len);
        std::iota(ssd_idx_vec.begin(), ssd_idx_vec.end(), num_exist_keys);
        expand_(len);
        dump_update(ssd_idx_vec, mem_idx_vec, slot_id_ptr, data_ptrs);
      }
    }
  } catch (const internal_runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
void SparseModelFileTS<TypeKey>::update_local_model_() {
  try {
    flush_mmap_to_disk_();
    unmap_from_memory_();

    std::vector<std::pair<size_t, std::pair<TypeKey, size_t>>> idx_key_vec;
    idx_key_vec.reserve(key_idx_map_.size());
    for (const auto& pair : key_idx_map_) {
      auto idx{pair.second};
      size_t slot_id = use_slot_id_ ? slot_ids[idx] : 0;
      idx_key_vec.push_back({idx, {pair.first, slot_id}});
    }
    std::sort(idx_key_vec.begin(), idx_key_vec.end());

    std::vector<TypeKey> key_vec(idx_key_vec.size());
    std::vector<size_t> slot_id_vec(idx_key_vec.size());
    std::transform(idx_key_vec.begin(), idx_key_vec.end(), key_vec.begin(),
                   [](const auto& pair) { return pair.second.first; });
    if (use_slot_id_) {
      std::transform(idx_key_vec.begin(), idx_key_vec.end(), slot_id_vec.begin(),
                     [](const auto& pair) { return pair.second.second; });
    }

    long long* key_ptr{nullptr};
    std::vector<long long> i64_keys;
    if (std::is_same<TypeKey, long long>::value) {
      key_ptr = const_cast<long long*>(reinterpret_cast<const long long*>(key_vec.data()));
    } else {
      i64_keys.resize(key_vec.size());
      std::transform(key_vec.begin(), key_vec.end(), i64_keys.begin(),
                     [](unsigned key) { return static_cast<long long>(key); });
      key_ptr = i64_keys.data();
    }

    std::ofstream key_ofs(mmap_handler_.get_key_file(), std::ofstream::trunc);
    if (!key_ofs.is_open()) {
      HCTR_OWN_THROW(Error_t::FileCannotOpen, "Cannot open key file");
    }
    key_ofs.write(reinterpret_cast<const char*>(key_ptr), key_vec.size() * sizeof(long long));

    if (use_slot_id_) {
      std::ofstream slot_ofs(mmap_handler_.get_slot_file(), std::ofstream::trunc);
      if (!slot_ofs.is_open()) {
        HCTR_OWN_THROW(Error_t::FileCannotOpen, "Cannot open slot file");
      }
      slot_ofs.write(reinterpret_cast<const char*>(slot_id_vec.data()),
                     key_vec.size() * sizeof(size_t));
    }

    mmap_to_memory_();
  } catch (const internal_runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
void SparseModelFileTS<TypeKey>::update_global_model() {
  try {
    update_local_model_();
    bool const localized_train{(resource_manager_->get_num_process() == 1)};
    if (localized_train) return;
#ifdef ENABLE_MPI
    HCTR_MPI_THROW(MPI_Barrier(MPI_COMM_WORLD));

    EmbeddingTableFile global_model(global_model_path_, opt_type_);
    size_t local_num_keys{key_idx_map_.size()};
    size_t global_num_keys;
    HCTR_MPI_THROW(
        MPI_Allreduce(&local_num_keys, &global_num_keys, 1, MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD));
    if (resource_manager_->is_master_process()) {
      std::filesystem::remove_all(global_model_path_);
      std::filesystem::create_directories(global_model_path_);
    }
    HCTR_MPI_THROW(MPI_Barrier(MPI_COMM_WORLD));

    MPI_File key_fh, slot_id_fh;
    std::vector<MPI_File> data_fhs(global_model.data_files.size());

    HCTR_MPI_THROW(MPI_File_open(MPI_COMM_WORLD, global_model.key_file.c_str(),
                                 MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &key_fh));
    if (use_slot_id_) {
      HCTR_MPI_THROW(MPI_File_open(MPI_COMM_WORLD, global_model.slot_file.c_str(),
                                   MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &slot_id_fh));
    }
    const auto data_files{global_model.data_files};
    for (size_t i{0}; i < data_files.size(); i++) {
      HCTR_MPI_THROW(MPI_File_open(MPI_COMM_WORLD, data_files[i].c_str(),
                                   MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                                   &(data_fhs[i])));
    }
    MPI_Datatype TYPE_EMB_VECTOR;
    HCTR_MPI_THROW(MPI_Type_contiguous(emb_vec_size_, MPI_FLOAT, &TYPE_EMB_VECTOR));
    HCTR_MPI_THROW(MPI_Type_commit(&TYPE_EMB_VECTOR));

    int my_rank{resource_manager_->get_process_id()};
    int n_ranks{resource_manager_->get_num_process()};

    std::vector<size_t> offset_per_rank(n_ranks, 0);
    HCTR_MPI_THROW(MPI_Allgather(&local_num_keys, sizeof(size_t), MPI_CHAR, offset_per_rank.data(),
                                 sizeof(size_t), MPI_CHAR, MPI_COMM_WORLD));
    std::exclusive_scan(offset_per_rank.begin(), offset_per_rank.end(), offset_per_rank.begin(), 0);

    const size_t key_offset{offset_per_rank[my_rank] * sizeof(long long)};
    const size_t slot_id_offset{offset_per_rank[my_rank] * sizeof(size_t)};
    const size_t vec_offset{offset_per_rank[my_rank] * sizeof(float) * emb_vec_size_};

    long long* h_key_ptr{nullptr};
    mmap_file_to_memory(&h_key_ptr, mmap_handler_.get_key_file());
    size_t* h_slot_id_ptr{nullptr};
    if (use_slot_id_) mmap_file_to_memory(&h_slot_id_ptr, mmap_handler_.get_slot_file());
    if (!mmap_handler_.mapped_to_file_) mmap_to_memory_();

    HCTR_MPI_THROW(MPI_Barrier(MPI_COMM_WORLD));
    MPI_Status status;
    HCTR_MPI_THROW(MPI_File_write_at(key_fh, key_offset, h_key_ptr, local_num_keys,
                                     MPI_LONG_LONG_INT, &status));
    if (use_slot_id_) {
      HCTR_MPI_THROW(MPI_File_write_at(slot_id_fh, slot_id_offset, h_slot_id_ptr, local_num_keys,
                                       MPI_SIZE_T, &status));
    }
    for (size_t i{0}; i < data_fhs.size(); i++) {
      HCTR_MPI_THROW(MPI_File_write_at(data_fhs[i], vec_offset, mmap_handler_.mmaped_ptrs_[i],
                                       local_num_keys, TYPE_EMB_VECTOR, &status));
    }

    if (use_slot_id_) unmap_file_from_memory(&h_slot_id_ptr, mmap_handler_.get_slot_file());
    unmap_file_from_memory(&h_key_ptr, mmap_handler_.get_key_file());

    HCTR_MPI_THROW(MPI_File_close(&key_fh));
    if (use_slot_id_) HCTR_MPI_THROW(MPI_File_close(&slot_id_fh));
    for (auto& fh : data_fhs) HCTR_MPI_THROW(MPI_File_close(&fh));
    HCTR_MPI_THROW(MPI_Type_free(&TYPE_EMB_VECTOR));
#endif
  } catch (const internal_runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
    throw;
  }
}

template class SparseModelFileTS<long long>;
template class SparseModelFileTS<unsigned>;

}  // namespace HugeCTR
