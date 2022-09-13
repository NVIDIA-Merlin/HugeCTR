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
#include <sys/mman.h>
#include <unistd.h>

#include <fstream>

#include "common.hpp"
#include "io/filesystem.hpp"
#include "io/hadoop_filesystem.hpp"

namespace HugeCTR {

class FileLoader {
 private:
  std::unique_ptr<FileSystem> file_system_; /**< data source backend of distributed file systems **/
  std::string cur_file_name_; /**< the file name of the current file for file loader to load **/
  size_t cur_file_size_;      /**< the file size of the current file **/
  DataSourceParams data_source_params_; /**< the configurations of the data source **/
  bool use_mmap_; /**< whether to use mmap or not, true for local file system, false for distributed
                     file systems **/
  int fd_;        /**< File descriptor for mapped file */
  std::ifstream in_file_stream_; /**< file stream of data set file */
  char* data_;                   /**< loaded data */

  /**
   * @brief private helper function to get the current file information
   *
   * @param file_name
   * @return 'Success', 'FileCannotOpen'
   */
  Error_t set_file(std::string file_name) noexcept {
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

 public:
  FileLoader(DataSourceParams data_source_params)
      : cur_file_size_(0), data_source_params_(data_source_params), fd_(-1), data_(nullptr) {
    use_mmap_ = data_source_params_.type == DataSourceType_t::Local;

    if (data_source_params_.type == DataSourceType_t::HDFS) {
      file_system_ = data_source_params.create_unique();
      HCTR_LOG_S(INFO, WORLD) << "Using Hadoop Cluster " << data_source_params.server << ":"
                              << data_source_params.port << std::endl;
    } else if (data_source_params_.type == DataSourceType_t::S3) {
      // TODO: Migrate to a suitable DataSourceBackend implementation.
      HCTR_LOG_S(INFO, WORLD) << "S3 is currently not supported. Use Local instead." << std::endl;
      use_mmap_ = true;
    } else if (data_source_params_.type == DataSourceType_t::Other) {
      // TODO: Migrate to a suitable DataSourceBackend implementation.
      HCTR_LOG_S(INFO, WORLD) << "Other filesystems are not supported. Use Local instead."
                              << std::endl;
      use_mmap_ = true;
    }
  }

  ~FileLoader() { clean(); }

  /**
   * @brief Load the file from file system to CPU memory
   *
   * @param file_name
   * @return 'Success', 'BrokenFile', 'FileCannotOpen'
   */
  Error_t load(std::string file_name) noexcept {
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
      if (bytes_read <= 0) {
        delete[] data_;
        data_ = nullptr;
        HCTR_LOG_S(ERROR, WORLD) << "Error reading the file from dfs " << HCTR_LOCATION()
                                 << std::endl;
        return Error_t::BrokenFile;
      }
      return Error_t::Success;
    }
  }

  /**
   * @brief clean the loaded data and set corresponding flags
   *
   */
  void clean() {
    if (use_mmap_ && fd_ != -1) {
      munmap(data_, cur_file_size_);
      close(fd_);
      fd_ = -1;
    } else if (!use_mmap_ && data_ != nullptr) {
      delete[] data_;
      data_ = nullptr;
    }
  }

  /**
   * @brief Get the loaded data
   *
   * @return ptr to the data
   */
  char* get_loaded_data() { return data_; }

  /**
   * @brief Get the current file size
   *
   * @return the file size
   */
  size_t get_current_file_size() { return cur_file_size_; }
};  // namespace HugeCTR
}  // namespace HugeCTR