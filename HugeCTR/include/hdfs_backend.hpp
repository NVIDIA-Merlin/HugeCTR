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

#include <string>
#include <vector>

#ifdef ENABLE_HDFS
#include "hdfs.h"
#endif

namespace HugeCTR {

class HdfsService {
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
  size_t getFileSize(const std::string& path);
  /**
   * @brief Write to HDFS
   *
   * @param writepath The HDFS path to write.
   * @param data The data stream to write.
   * @param dataSize The size of the data stream.
   * @param overwrite Whether to overwrite or append.
   * @return int
   */

  int write(const std::string& writepath, const void* data, size_t dataSize, bool overwrite);
  /**
   * @brief Read from HDFS
   *
   * @param readpath The HDFS path to read data from.
   * @param buffer The buffer used to store data read.
   * @param dataSize The number of bytes to read.
   * @param offset The offset to start read.
   * @return int
   */

  int read(const std::string& readpath, const void* buffer, size_t data_size, size_t offset);

  /**
   * @brief copy the single file from HDFS to Local
   *
   * @param hdfs_path The HDFS path to copy from.
   * @param local_path The local path to copy to.
   * @return int
   */
  int copyToLocal(const std::string& hdfs_path, const std::string& local_path);

  /**
   * @brief Copy ALL files of the given HDFS path to Local
   *
   * @param hdfs_dir_path The HDFS path to copy from.
   * @param local_path The local path to copy to.
   * @return int
   */
  int batchCopyToLocal(const std::string& hdfs_dir_path, const std::string& local_path);

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

struct DataSourceParams {
  bool use_hdfs;

  std::string namenode;
  int port;

  std::string hdfs_train_source;
  std::string hdfs_train_filelist;
  std::string hdfs_eval_source;
  std::string hdfs_eval_filelist;
  std::string hdfs_dense_model;
  std::string hdfs_dense_opt_states;
  std::vector<std::string> hdfs_sparse_model;
  std::vector<std::string> hdfs_sparse_opt_states;

  std::string local_train_source;
  std::string local_train_filelist;
  std::string local_eval_source;
  std::string local_eval_filelist;
  std::string local_dense_model;
  std::string local_dense_opt_states;
  std::vector<std::string> local_sparse_model;
  std::vector<std::string> local_sparse_opt_states;

  std::string hdfs_model_home;
  std::string local_model_home;
  DataSourceParams(const bool use_hdfs, const std::string& namenode, const int port,
                   const std::string& hdfs_train_source, const std::string& hdfs_train_filelist,
                   const std::string& hdfs_eval_source, const std::string& hdfs_eval_filelist,
                   const std::string& hdfs_dense_model, const std::string& hdfs_dense_opt_states,
                   const std::vector<std::string>& hdfs_sparse_model,
                   const std::vector<std::string>& hdfs_sparse_opt_states,
                   const std::string& local_train_source, const std::string& local_train_filelist,
                   const std::string& local_eval_source, const std::string& local_eval_filelist,
                   const std::string& local_dense_model, const std::string& local_dense_opt_states,
                   const std::vector<std::string>& local_sparse_model,
                   const std::vector<std::string>& local_sparse_opt_states,
                   const std::string& hdfs_model_home, const std::string& local_model_home);
  DataSourceParams();
};

class DataSource {
 public:
  ~DataSource();
  DataSource(const DataSourceParams& data_source_params);
  DataSource(const DataSource&) = delete;
  DataSource& operator=(const DataSource&) = delete;
  void move_to_local();

 private:
  DataSourceParams data_source_params_;
};

}  // namespace HugeCTR