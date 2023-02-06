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

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <io/file_loader.hpp>

namespace HugeCTR {

Error_t FileLoader::set_file(const std::string& file_name) noexcept {
  cur_file_name_ = file_name;
  if (use_mmap_) {
    in_file_stream_.open(cur_file_name_, std::ifstream::binary);
    if (!in_file_stream_.is_open()) {
      HCTR_LOG_S(ERROR, WORLD) << "in_file_stream_.is_open() failed: " << cur_file_name_ << ' '
                               << HCTR_LOCATION() << std::endl;
      return Error_t::FileCannotOpen;
    }
    // evaluate parquet file size
    in_file_stream_.seekg(0, std::ios::end);
    cur_file_size_ = in_file_stream_.tellg();
    in_file_stream_.close();
    return Error_t::Success;
  } else {
    size_t temp = file_system_->get_file_size(cur_file_name_);
    if (temp > 0) {
      cur_file_size_ = temp;
      return Error_t::Success;
    } else {
      HCTR_LOG_S(ERROR, WORLD) << "data_source_backend failed to open: " << cur_file_name_ << ' '
                               << HCTR_LOCATION() << std::endl;
      return Error_t::FileCannotOpen;
    }
  }
}

FileLoader::FileLoader(const DataSourceParams& data_source_params)
    : cur_file_size_(0), data_source_params_(data_source_params), fd_(-1), data_(nullptr) {
  use_mmap_ = data_source_params_.type == FileSystemType_t::Local;
  if (!use_mmap_) {
    file_system_ = FileSystemBuilder::build_unique_by_data_source_params(data_source_params);
  }
}

FileLoader::~FileLoader() { clean(); }

Error_t FileLoader::load(const std::string& file_name) noexcept {
  Error_t err = set_file(file_name);
  if (err != Error_t::Success) {
    HCTR_LOG_S(ERROR, WORLD) << "Error open file for read " << HCTR_LOCATION() << std::endl;
    return err;
  }
  if (use_mmap_) {
    fd_ = open(cur_file_name_.c_str(), O_RDONLY, 0);
    if (fd_ == -1) {
      HCTR_LOG_S(ERROR, WORLD) << "Error open file for read " << HCTR_LOCATION() << std::endl;
      return Error_t::BrokenFile;
    }
    data_ = (char*)mmap(0, cur_file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
      close(fd_);
      fd_ = -1;
      HCTR_LOG_S(ERROR, WORLD) << "Error mmapping the file " << HCTR_LOCATION() << std::endl;
      return Error_t::BrokenFile;
    }
    return Error_t::Success;
  } else {
    data_ = new char[cur_file_size_ / sizeof(char)];
    int bytes_read = file_system_->read(cur_file_name_, data_, cur_file_size_, 0);
    if (bytes_read < 0) {
      delete[] data_;
      data_ = nullptr;
      HCTR_LOG_S(ERROR, WORLD) << "Error reading the file from dfs " << HCTR_LOCATION()
                               << std::endl;
      return Error_t::BrokenFile;
    }
    return Error_t::Success;
  }
}

void FileLoader::clean() {
  if (use_mmap_ && fd_ != -1) {
    munmap(data_, cur_file_size_);
    close(fd_);
    fd_ = -1;
  } else if (!use_mmap_ && data_ != nullptr) {
    delete[] data_;
    data_ = nullptr;
  }
}

}  // namespace HugeCTR