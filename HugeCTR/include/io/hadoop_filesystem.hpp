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

#include <io/filesystem.hpp>

#ifdef ENABLE_HDFS
#include <hdfs.h>
#endif

namespace HugeCTR {

#ifdef ENABLE_HDFS
struct HdfsConfigs {
  HdfsConfigs() = default;
  ~HdfsConfigs() = default;

  int32_t buffer_size = 0;
  int16_t replication = 0;
  int64_t block_size = 0;

  std::string namenode;
  int port;
  std::string user_name;
  bool ready_to_connect = false;

  void set_buffer_size(int32_t buffer_size);
  void set_replication(int16_t replication);
  void set_block_size(int64_t block_size);
  void set_user_name(std::string& user_name);
  void set_connection_configs(std::string namenode, int port);
  void set_all_from_json(const std::string& path);

  static HdfsConfigs FromDataSourceParams(const DataSourceParams& data_source_params);
  static HdfsConfigs FromUrl(const std::string& url);
  static HdfsConfigs FromJSON(const std::string& url);
};

class HadoopFileSystem final : public FileSystem {
 public:
  /**
   * @brief Construct a new HadoopFileSystem:: HadoopFileSystem object
   *
   * @param name_node The host-address of the HDFS name node.
   * @param port The port number.
   */
  HadoopFileSystem(const std::string& name_node, const int port);

  HadoopFileSystem(const HdfsConfigs& configs);

  virtual ~HadoopFileSystem();

  size_t get_file_size(const std::string& path) const override;

  void create_dir(const std::string& path) override;

  void delete_file(const std::string& path) override;

  void fetch(const std::string& source_path, const std::string& target_path) override;

  void upload(const std::string& source_path, const std::string& target_path) override;

  int write(const std::string& path, const void* data, size_t data_size, bool overwrite) override;

  int read(const std::string& path, void* buffer, size_t buffer_size, size_t offset) override;

  void copy(const std::string& source_file, const std::string& target_file) override;

  void batch_fetch(const std::string& source_dir, const std::string& target_dir) override;

  void batch_upload(const std::string& source_dir, const std::string& target_dir) override;

 private:
  HdfsConfigs configs_;

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

#endif

}  // namespace HugeCTR