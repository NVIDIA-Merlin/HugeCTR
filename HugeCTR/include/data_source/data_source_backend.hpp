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

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace HugeCTR {

class DataSourceBackend {
 public:
  DataSourceBackend() = default;

  DataSourceBackend(const DataSourceBackend&) = delete;

  virtual ~DataSourceBackend() = default;

  DataSourceBackend& operator=(const DataSourceBackend&) = delete;

  /**
   * @brief Get the file size of target file from the file system
   *
   * @param path HDFS file path.
   * @return File size in bytes.
   */
  virtual size_t get_file_size(const std::string& path) const = 0;

  /**
   * @brief Write file from butter to the specified path in the file system
   *
   * @param path HDFS path of the file to write into.
   * @param data The data stream to write.
   * @param data_size The size of the data stream.
   * @param overwrite Whether to overwrite or append.
   * @return Number of successfully written bytes.
   */
  virtual int write(const std::string& path, const void* data, size_t data_size,
                    bool overwrite) = 0;

  /**
   * @brief Read file from the file system to the buffer
   *
   * @param path HDFS path of the file from which to read.
   * @param buffer Buffer to hold the read data.
   * @param buffer_size The number of bytes to read.
   * @param offset Offset within the file from which to start reading.
   * @return Number of successfully read bytes.
   */
  virtual int read(const std::string& path, void* buffer, size_t buffer_size, size_t offset) = 0;

  /**
   * @brief Copy a specific file from the one file system to the other.
   *
   * @param source_file Source file path.
   * @param target_file Target file path.
   * @param to_local If true, copy from DFS to local FS; if false, vice versa.
   */
  virtual void copy(const std::string& source_file, const std::string& target_file,
                    bool to_local) = 0;

  /**
   * @brief Copy all files under the source directory to target directory from one filesystem to the
   * other.
   *
   * @param source_dir Source dir path.
   * @param target_dir Target dir path.
   * @param to_local If true, copy from DFS to local FS; if false, vice versa.
   * @return Number of files copied.
   */
  virtual int batch_copy(const std::string& source_dir, const std::string& target_dir,
                         bool to_local) = 0;
};

enum class DataSourceType_t { Local, HDFS, S3, Other };

struct DataSourceParams {
  DataSourceType_t type;
  std::string server;
  int port;

  DataSourceParams(const DataSourceType_t type, const std::string& server, const int port)
      : type(type), server(server), port(port){};
  DataSourceParams() : type(DataSourceType_t::Local), server("localhost"), port(9000){};

  DataSourceBackend* create() const;

  std::unique_ptr<DataSourceBackend> create_unique() const {
    return std::unique_ptr<DataSourceBackend>{create()};
  }
};

}  // namespace HugeCTR