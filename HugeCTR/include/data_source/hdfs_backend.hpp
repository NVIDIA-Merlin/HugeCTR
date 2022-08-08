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

#include <data_source/data_source_backend.hpp>
#include <string>
#include <vector>
#ifdef ENABLE_HDFS
#include "hdfs.h"
#endif

namespace HugeCTR {

class HdfsService final : public DataSourceBackend {
 public:
  HdfsService();
  /**
   * @brief Construct a new Hdfs Service:: Hdfs Service object
   *
   * @param name_node The host-address of the HDFS name node.
   * @param port The port number.
   */

  HdfsService(const std::string& name_node, const int port);

  /**
   * @brief Destroy the Hdfs Service object
   *
   */
  virtual ~HdfsService();

  /**
   * @brief Get the File Size object
   *
   * @param path Path to the file
   * @return size_t
   */
  size_t getFileSize(const std::string& path) const override;
  /**
   * @brief Write to HDFS
   *
   * @param writepath The HDFS path to write.
   * @param data The data stream to write.
   * @param dataSize The size of the data stream.
   * @param overwrite Whether to overwrite or append.
   * @return int
   */

  int write(const std::string& writepath, const void* data, size_t dataSize,
            bool overwrite) override;
  /**
   * @brief Read from HDFS
   *
   * @param readpath The HDFS path to read data from.
   * @param buffer The buffer used to store data read.
   * @param dataSize The number of bytes to read.
   * @param offset The offset to start read.
   * @return int
   */

  int read(const std::string& readpath, const void* buffer, size_t data_size,
           size_t offset) override;

  /**
   * @brief Copy source file to target from one filesystem to the other.
   *
   * @param source_file the source file path
   * @param target_file the target file path
   * @param to_local copy from dfs to local fs or the other way
   * @return int
   */
  int copy(const std::string& source_file, const std::string& target_file, bool to_local) override;

  /**
   * @brief Copy all files under the source directory to target directory from one filesystem to the
   * other.
   *
   * @param source_dir the source dir path
   * @param target_dir the target dir path
   * @param to_local copy from dfs to local fs or the other way
   * @return int
   */
  int batchCopy(const std::string& source_dir, const std::string& target_dir, bool to_local);

 private:
#ifdef ENABLE_HDFS
  hdfsFS fs_;
  hdfsFS local_fs_;
#endif
  std::string name_node_;
  int hdfs_port_;

  /**
   * @brief Connect to HDFS server
   *
   * @return hdfsFS The FS handler for HDFS.
   */
#ifdef ENABLE_HDFS
  hdfsFS connect();
#endif
  /**
   * @brief Connect to local File system.
   *
   * @return hdfsFS
   */
#ifdef ENABLE_HDFS
  hdfsFS connectToLocal();
#endif
  /**
   * @brief Disconnect to HDFS server.
   *
   */
  void disconnect();
};

}  // namespace HugeCTR