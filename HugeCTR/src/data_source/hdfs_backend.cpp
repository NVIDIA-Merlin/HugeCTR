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

#include "data_source/hdfs_backend.hpp"

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
#else
  std::cout << "[HDFS][WARN]: Please install Hadoop and compile HugeCTR with ENABLE_HDFS to use "
               "HDFS functionalities."
            << std::endl;
#endif
}

HdfsService::~HdfsService() { disconnect(); }

void HdfsService::disconnect() {
#ifdef ENABLE_HDFS
  if (fs_) {
    int result = hdfsDisconnect(fs_);
    if (result != 0) {
      std::cout << "[HDFS][ERROR]: Unable to disconnect HDFS" << std::endl;
    }
  }
  if (local_fs_) {
    int result_local = hdfsDisconnect(local_fs_);
    if (result_local != 0) {
      std::cout << "[HDFS][ERROR]: Unable to disconnect local FS" << std::endl;
    }
  }
#endif
}
#ifdef ENABLE_HDFS
hdfsFS HdfsService::connect() {
  struct hdfsBuilder* bld = hdfsNewBuilder();
  if (!bld) {
    std::cout << "[HDFS][ERROR]: Unable to create HDFS builder" << std::endl;
    return NULL;
  }
  hdfsBuilderSetNameNode(bld, name_node_.c_str());
  hdfsBuilderSetNameNodePort(bld, hdfs_port_);
  hdfsBuilderSetForceNewInstance(bld);
  hdfsFS fs = hdfsBuilderConnect(bld);
  if (!fs) {
    std::cout << "[HDFS][ERROR]: Unable to connect to HDFS" << std::endl;
    return NULL;
  }
  return fs;
}
#endif

#ifdef ENABLE_HDFS
hdfsFS HdfsService::connectToLocal() {
  struct hdfsBuilder* bld = hdfsNewBuilder();
  if (!bld) {
    std::cout << "[HDFS][ERROR]: Unable to create HDFS builder" << std::endl;
    return NULL;
  }
  hdfsBuilderSetNameNode(bld, NULL);
  hdfsBuilderSetNameNodePort(bld, 0);
  hdfsBuilderSetForceNewInstance(bld);
  hdfsFS fs = hdfsBuilderConnect(bld);
  if (!fs) {
    std::cout << "[HDFS][ERROR]: Unable to connect to Local FS" << std::endl;
    return NULL;
  }
  return fs;
}
#endif

size_t HdfsService::getFileSize(const std::string& path) const {
#ifdef ENABLE_HDFS
  if (!fs_) {
    std::cout << "[HDFS][ERROR]: Not connected to HDFS" << std::endl;
    return 0;
  }
  hdfsFileInfo* info = hdfsGetPathInfo(fs_, path.c_str());
  if (info == NULL) {
    std::cout << "[HDFS][ERROR]: File does not exist" << std::endl;
    return 0;
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
    return -1;
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
    return -1;
  }
  tSize result = hdfsWrite(fs_, writeFile, (void*)data, dataSize);
  if (hdfsFlush(fs_, writeFile)) {
    std::cout << "[HDFS][ERROR]: Failed to flush" << std::endl;
    return -1;
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
  if (!fs_) {
    std::cout << "[HDFS][ERROR]: Not connected to HDFS" << std::endl;
    return -1;
  }
  if (buffer == NULL) {
    std::cout << "[HDFS][ERROR]: Buffer error" << std::endl;
    return -1;
  } else {
    hdfsFile handle_hdfsFile_w = hdfsOpenFile(fs_, readpath.c_str(), O_RDONLY, 0, 0, 0);
    if (handle_hdfsFile_w == NULL) {
      std::cout << "[HDFS][ERROR]: Failed to open file!" << std::endl;
      return -1;
    } else {
      hdfsFileInfo* info = hdfsGetPathInfo(fs_, readpath.c_str());
      size_t file_size = (info->mSize);
      if (file_size - offset < data_size) {
        std::cout << "[HDFS][ERROR]: No enough bytes to read!" << std::endl;
        return -1;
      }
      tSize num_read_bytes = hdfsPread(fs_, handle_hdfsFile_w, offset, (void*)buffer, data_size);
      if ((int)data_size != num_read_bytes) {
        std::cout << "[HDFS][ERROR]: Failed to read file!" << std::endl;
        return -1;
      }
      hdfsCloseFile(fs_, handle_hdfsFile_w);
      return num_read_bytes;
    }
  }
#endif
  return 0;
}

int HdfsService::copy(const std::string& source_path, const std::string& target_path,
                      bool to_local) {
#ifdef ENABLE_HDFS
  if (!fs_) {
    std::cout << "[HDFS][ERROR]: Not connected to HDFS" << std::endl;
    return -1;
  }
  if (!local_fs_) {
    std::cout << "[HDFS][ERROR]: Not connected to local FS" << std::endl;
    return -1;
  }
  int result = -1;
  if (to_local) {
    result = hdfsCopy(fs_, source_path.c_str(), local_fs_, target_path.c_str());
  } else {
    result = hdfsCopy(local_fs_, source_path.c_str(), fs_, target_path.c_str());
  }
  if (result == 0) {
    std::cout << "[HDFS][INFO]: Copied " << source_path << " successfully!" << std::endl;
  } else {
    std::cout << "[HDFS][ERROR]: Unable to copy." << std::endl;
    return -1;
  }
  return result;
#endif
  return 0;
}

int HdfsService::batchCopy(const std::string& source_dir, const std::string& target_dir,
                           bool to_local) {
#ifdef ENABLE_HDFS
  if (!fs_) {
    std::cout << "[HDFS][ERROR]: Not connected to HDFS" << std::endl;
    return -1;
  }
  if (!local_fs_) {
    std::cout << "[HDFS][ERROR]: Not connected to local FS" << std::endl;
    return -1;
  }
  hdfsFS source_fs;
  hdfsFS target_fs;
  if (to_local) {
    source_fs = fs_;
    target_fs = local_fs_;
  } else {
    source_fs = local_fs_;
    target_fs = fs_;
  }

  int source_exist = hdfsExists(source_fs, source_dir.c_str());
  int target_exist = hdfsExists(target_fs, target_dir.c_str());

  if (source_exist != 0) {
    std::cout << "[HDFS][ERROR]: source_dir does not exist" << std::endl;
    return -1;
  }
  if (target_exist != 0) {
    hdfsCreateDirectory(target_fs, target_dir.c_str());
    std::cout << "[HDFS][INFO]: target directory does not exist, just created" << std::endl;
  }

  hdfsFileInfo* file_info = hdfsGetPathInfo(source_fs, source_dir.c_str());
  if (file_info->mKind == tObjectKind::kObjectKindFile) {
    std::cout << "[HDFS][ERROR]: not a directory" << std::endl;
    return -1;
  } else {
    int count = -1;
    hdfsFileInfo* info_ptr_list = hdfsListDirectory(source_fs, source_dir.c_str(), &count);
    auto info_ptr = info_ptr_list;
    for (int i = 0; i < count; ++i, ++info_ptr) {
      auto cur_path = std::string(info_ptr->mName);
      if (info_ptr->mKind == kObjectKindFile) {
        copy(cur_path, target_dir, to_local);
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

}  // namespace HugeCTR