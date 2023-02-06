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
#pragma once

#include <common.hpp>
#include <fstream>
#include <io/filesystem.hpp>
#include <memory>
#include <string>

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
  Error_t set_file(const std::string& file_name) noexcept;

 public:
  FileLoader(const DataSourceParams& data_source_params);

  ~FileLoader();

  /**
   * @brief Load the file from file system to CPU memory
   *
   * @param file_name
   * @return 'Success', 'BrokenFile', 'FileCannotOpen'
   */
  Error_t load(const std::string& file_name) noexcept;

  /**
   * @brief clean the loaded data and set corresponding flags
   *
   */
  void clean();

  /**
   * @brief Get the loaded data
   *
   * @return ptr to the data
   */
  inline char* get_loaded_data() noexcept { return data_; }

  /**
   * @brief Get the current file size
   *
   * @return the file size
   */
  inline size_t get_current_file_size() noexcept { return cur_file_size_; }
};

}  // namespace HugeCTR