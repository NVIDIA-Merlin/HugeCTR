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

#ifdef ENABLE_HDFS
#include "hdfs.h"
#endif

namespace HugeCTR {

#ifdef ENABLE_HDFS
class HdfsService final : public DataSourceBackend {
 public:
  /**
   * @brief Construct a new Hdfs Service:: Hdfs Service object
   *
   * @param name_node The host-address of the HDFS name node.
   * @param port The port number.
   */
  HdfsService(const std::string& name_node, int port);

  virtual ~HdfsService();

  size_t get_file_size(const std::string& path) const override;

  int write(const std::string& writepath, const void* data, size_t data_size,
            bool overwrite) override;

  int read(const std::string& readpath, void* buffer, size_t buffer_size, size_t offset) override;

  void copy(const std::string& source_file, const std::string& target_file, bool to_local) override;

  int batch_copy(const std::string& source_dir, const std::string& target_dir,
                 bool to_local) override;

 private:
  std::string name_node_;
  int hdfs_port_;
  hdfsFS fs_;
  hdfsFS local_fs_;

  /**
   * @brief Connect to HDFS server
   */
  void connect();

  /**
   * @brief Connect to local File system.
   */
  void connect_to_local();

  /**
   * @brief Disconnect to HDFS server.
   */
  void disconnect();
};

#endif  // HDFS_ENABLE

}  // namespace HugeCTR