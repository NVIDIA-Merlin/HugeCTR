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

#include "hdfs_backend.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <exception>
#include <iostream>

namespace HugeCTR {

HdfsService::HdfsService() {
#ifdef ENABLE_HDFS
  fs_ = NULL;
  local_fs_ = NULL;
#endif
}

HdfsService::HdfsService(const std::string& name_node, const int port) {
  name_node_ = name_node;
  hdfs_port_ = port;
#ifdef ENABLE_HDFS
  fs_ = connect();
  local_fs_ = connectToLocal();
#endif
#ifndef ENABLE_HDFS
  std::cout << "[HDFS][WARN]: Please install Hadoop and compile HugeCTR with ENABLE_HDFS to use "
               "HDFS functionalities."
            << std::endl;
#endif
  // if (fs_ && local_fs_) {
  //   std::cout << "[HDFS][INFO]: HDFS service start successfully!" << std::endl;
  // }
}

HdfsService::~HdfsService() { disconnect(); }

void HdfsService::disconnect() {
#ifdef ENABLE_HDFS
  if (fs_) {
    int result = hdfsDisconnect(fs_);
    if (result != 0) {
      std::cout << "[HDFS][ERROR]: Unable to disconnect HDFS" << std::endl;
      exit(1);
    }
  }
  if (local_fs_) {
    int result_local = hdfsDisconnect(local_fs_);
    if (result_local != 0) {
      std::cout << "[HDFS][ERROR]: Unable to disconnect local FS" << std::endl;
      exit(1);
    }
  }
#endif
  // std::cout << "[HDFS][INFO]: HDFS service disconnect successfully!" << std::endl;
}
#ifdef ENABLE_HDFS
hdfsFS HdfsService::connect() {
  struct hdfsBuilder* bld = hdfsNewBuilder();
  if (!bld) {
    std::cout << "[HDFS][ERROR]: Unable to create HDFS builder" << std::endl;
    exit(1);
  }
  hdfsBuilderSetNameNode(bld, name_node_.c_str());
  hdfsBuilderSetNameNodePort(bld, hdfs_port_);
  hdfsBuilderSetForceNewInstance(bld);
  hdfsFS fs = hdfsBuilderConnect(bld);
  if (!fs) {
    std::cout << "[HDFS][ERROR]: Unable to connect to HDFS" << std::endl;
    exit(1);
  }
  return fs;
}
#endif

#ifdef ENABLE_HDFS
hdfsFS HdfsService::connectToLocal() {
  struct hdfsBuilder* bld = hdfsNewBuilder();
  if (!bld) {
    std::cout << "[HDFS][ERROR]: Unable to create HDFS builder" << std::endl;
    exit(1);
  }
  hdfsBuilderSetNameNode(bld, NULL);
  hdfsBuilderSetNameNodePort(bld, 0);
  hdfsBuilderSetForceNewInstance(bld);
  hdfsFS fs = hdfsBuilderConnect(bld);
  if (!fs) {
    std::cout << "[HDFS][ERROR]: Unable to connect to Local FS" << std::endl;
    exit(1);
  }
  return fs;
}
#endif

size_t HdfsService::getFileSize(const std::string& path) {
#ifdef ENABLE_HDFS
  hdfsFileInfo* info = hdfsGetPathInfo(fs_, path.c_str());
  if (info == NULL) {
    std::cout << "[HDFS][ERROR]: File does not exist" << std::endl;
    exit(1);
  }
  size_t file_size = (info->mSize);
  return file_size;
#endif
  return 0;
}

int HdfsService::write(const std::string& writepath, const void* data, size_t dataSize,
                       bool overwrite) {
#ifdef ENABLE_HDFS
  if (!fs_) {
    std::cout << "[HDFS][ERROR]: Not connected to HDFS" << std::endl;
    exit(1);
  }
  int a = hdfsExists(fs_, writepath.c_str());
  hdfsFile writeFile;
  if (a == 0) {
    if (overwrite) {
      writeFile = hdfsOpenFile(fs_, writepath.c_str(), O_WRONLY, 0, 0, 0);
    } else {
      writeFile = hdfsOpenFile(fs_, writepath.c_str(), O_WRONLY | O_APPEND, 0, 0, 0);
    }
  } else {
    writeFile = hdfsOpenFile(fs_, writepath.c_str(), O_WRONLY | O_CREAT, 0, 0, 0);
  }
  if (!writeFile) {
    std::cout << "[HDFS][ERROR]: Failed to open the file" << std::endl;
    exit(1);
  }
  tSize result = hdfsWrite(fs_, writeFile, (void*)data, dataSize);
  if (hdfsFlush(fs_, writeFile)) {
    std::cout << "[HDFS][ERROR]: Failed to flush" << std::endl;
    exit(1);
  }
  std::cout << "[HDFS][INFO]: Write to HDFS " << writepath << " successfully!" << std::endl;
  hdfsCloseFile(fs_, writeFile);
  return result;
#endif
  return 0;
}

int HdfsService::read(const std::string& readpath, const void* buffer, size_t data_size,
                      size_t offset) {
#ifdef ENABLE_HDFS
  if (buffer == NULL) {
    std::cout << "[HDFS][ERROR]: Buffer error" << std::endl;
    exit(1);
  } else {
    hdfsFile handle_hdfsFile_w = hdfsOpenFile(fs_, readpath.c_str(), O_RDONLY, 0, 0, 0);
    if (handle_hdfsFile_w == NULL) {
      std::cout << "[HDFS][ERROR]: Failed to open file!" << std::endl;
      exit(1);
    } else {
      hdfsFileInfo* info = hdfsGetPathInfo(fs_, readpath.c_str());
      size_t file_size = (info->mSize);
      if (file_size - offset < data_size) {
        std::cout << "[HDFS][ERROR]: No enough bytes to read!" << std::endl;
        exit(1);
      }
      tSize num_read_bytes = hdfsPread(fs_, handle_hdfsFile_w, offset, (void*)buffer, data_size);
      if ((int)data_size == num_read_bytes) {
        std::cout << "[HDFS][INFO]: Read file " << readpath << " successfully!" << std::endl;
      } else {
        std::cout << "[HDFS][ERROR]: Failed to read file!" << std::endl;
        exit(1);
      }
      hdfsCloseFile(fs_, handle_hdfsFile_w);
      return num_read_bytes;
    }
  }
#endif
  return 0;
}

int HdfsService::copyToLocal(const std::string& hdfs_path, const std::string& local_path) {
#ifdef ENABLE_HDFS
  if (!local_fs_) {
    std::cout << "[HDFS][ERROR]: Not connected to local FS" << std::endl;
    exit(1);
  }
  int result = hdfsCopy(fs_, hdfs_path.c_str(), local_fs_, local_path.c_str());
  if (result == 0) {
    std::cout << "[HDFS][INFO]: Copied " << hdfs_path << " to local successfully!" << std::endl;
  } else {
    std::cout << "[HDFS][ERROR]: Unable to copy to local" << std::endl;
    exit(1);
  }
  return result;
#endif
  return 0;
}

int HdfsService::batchCopyToLocal(const std::string& hdfs_path, const std::string& local_path) {
#ifdef ENABLE_HDFS
  int hdfs_exist = hdfsExists(fs_, hdfs_path.c_str());
  int local_exist = hdfsExists(local_fs_, local_path.c_str());
  if (hdfs_exist != 0) {
    std::cout << "[HDFS][ERROR]: hdfs_path does not exist" << std::endl;
    exit(1);
  }
  if (local_exist != 0) {
    hdfsCreateDirectory(local_fs_, local_path.c_str());
    std::cout << "[HDFS][INFO]: local_path does not exist, just created" << std::endl;
  }
  if (!local_fs_) {
    std::cout << "[HDFS][ERROR]: Not connected to local FS" << std::endl;
    exit(1);
  }

  hdfsFileInfo* file_info = hdfsGetPathInfo(fs_, hdfs_path.c_str());
  if (file_info->mKind == tObjectKind::kObjectKindFile) {
    std::cout << "[HDFS][ERROR]: not a directory" << std::endl;
    exit(1);
  } else {
    int count = -1;
    hdfsFileInfo* info_ptr_list = hdfsListDirectory(fs_, hdfs_path.c_str(), &count);
    auto info_ptr = info_ptr_list;
    for (int i = 0; i < count; ++i, ++info_ptr) {
      auto cur_path = std::string(info_ptr->mName);
      if (info_ptr->mKind == kObjectKindFile) {
        copyToLocal(cur_path, local_path);
      }
    }
    hdfsFreeFileInfo(info_ptr_list, count);
    std::cout << "[HDFS][INFO]: Batch copy done !" << std::endl;
  }
  hdfsFreeFileInfo(file_info, 1);
  return 0;
#endif
  return 0;
}

DataSourceParams::DataSourceParams(
    const bool use_hdfs, const std::string& namenode, const int port,
    const std::string& hdfs_train_source, const std::string& hdfs_train_filelist,
    const std::string& hdfs_eval_source, const std::string& hdfs_eval_filelist,
    const std::string& hdfs_dense_model, const std::string& hdfs_dense_opt_states,
    const std::vector<std::string>& hdfs_sparse_model,
    const std::vector<std::string>& hdfs_sparse_opt_states, const std::string& local_train_source,
    const std::string& local_train_filelist, const std::string& local_eval_source,
    const std::string& local_eval_filelist, const std::string& local_dense_model,
    const std::string& local_dense_opt_states, const std::vector<std::string>& local_sparse_model,
    const std::vector<std::string>& local_sparse_opt_states, const std::string& hdfs_model_home,
    const std::string& local_model_home)
    : use_hdfs(use_hdfs),
      namenode(namenode),
      port(port),
      hdfs_train_source(hdfs_train_source),
      hdfs_train_filelist(hdfs_train_filelist),
      hdfs_eval_source(hdfs_eval_source),
      hdfs_eval_filelist(hdfs_eval_filelist),
      hdfs_dense_model(hdfs_dense_model),
      hdfs_dense_opt_states(hdfs_dense_opt_states),
      hdfs_sparse_model(hdfs_sparse_model),
      hdfs_sparse_opt_states(hdfs_sparse_opt_states),
      local_train_source(local_train_source),
      local_train_filelist(local_train_filelist),
      local_eval_source(local_eval_source),
      local_eval_filelist(local_eval_filelist),
      local_dense_model(local_dense_model),
      local_dense_opt_states(local_dense_opt_states),
      local_sparse_model(local_sparse_model),
      local_sparse_opt_states(local_sparse_opt_states),
      hdfs_model_home(hdfs_model_home),
      local_model_home(local_model_home) {}

DataSourceParams::DataSourceParams()
    : use_hdfs(false),
      namenode("localhost"),
      port(9000),
      hdfs_train_source(""),
      hdfs_train_filelist(""),
      hdfs_eval_source(""),
      hdfs_eval_filelist(""),
      hdfs_dense_model(""),
      hdfs_dense_opt_states(""),
      hdfs_sparse_model(std::vector<std::string>()),
      hdfs_sparse_opt_states(std::vector<std::string>()),
      local_train_source(""),
      local_train_filelist(""),
      local_eval_source(""),
      local_eval_filelist(""),
      local_dense_model(""),
      local_dense_opt_states(""),
      local_sparse_model(std::vector<std::string>()),
      local_sparse_opt_states(std::vector<std::string>()),
      hdfs_model_home(""),
      local_model_home("") {}

DataSource::~DataSource() {}

DataSource::DataSource(const DataSourceParams& data_source_params)
    : data_source_params_(data_source_params) {}

void DataSource::move_to_local() {
  HdfsService hs = HdfsService(data_source_params_.namenode, data_source_params_.port);

  hs.batchCopyToLocal(data_source_params_.hdfs_train_source,
                      data_source_params_.local_train_source);
  hs.copyToLocal(data_source_params_.hdfs_train_filelist, data_source_params_.local_train_filelist);
  hs.batchCopyToLocal(data_source_params_.hdfs_eval_source, data_source_params_.local_eval_source);
  hs.copyToLocal(data_source_params_.hdfs_eval_filelist, data_source_params_.local_eval_filelist);
  hs.copyToLocal(data_source_params_.hdfs_dense_model, data_source_params_.local_dense_model);
  hs.copyToLocal(data_source_params_.hdfs_dense_opt_states,
                 data_source_params_.local_dense_opt_states);
  if (!data_source_params_.hdfs_sparse_model.empty() &&
      !data_source_params_.local_sparse_model.empty()) {
    if (data_source_params_.hdfs_sparse_model.size() !=
        data_source_params_.local_sparse_model.size()) {
      std::cout << "[HDFS][ERROR]: Number of sparse models is not consistent" << std::endl;
      exit(1);
    }
    for (int i = 0; i < (int)data_source_params_.hdfs_sparse_model.size(); ++i) {
      hs.batchCopyToLocal(data_source_params_.hdfs_sparse_model[i],
                          data_source_params_.local_sparse_model[i]);
    }
  }
  if (!data_source_params_.hdfs_sparse_opt_states.empty() &&
      !data_source_params_.local_sparse_opt_states.empty()) {
    if (data_source_params_.hdfs_sparse_opt_states.size() !=
        data_source_params_.local_sparse_opt_states.size()) {
      std::cout << "[HDFS][ERROR]: Number of sparse opt models is not consistent" << std::endl;
      exit(1);
    }
    for (int i = 0; i < (int)data_source_params_.hdfs_sparse_opt_states.size(); ++i) {
      hs.copyToLocal(data_source_params_.hdfs_sparse_opt_states[i],
                     data_source_params_.local_sparse_opt_states[i]);
    }
  }
}
}  // namespace HugeCTR