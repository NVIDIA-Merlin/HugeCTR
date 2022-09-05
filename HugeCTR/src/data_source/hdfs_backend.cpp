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

#include <base/debug/logger.hpp>
#include <data_source/hdfs_backend.hpp>

namespace HugeCTR {

#ifdef ENABLE_HDFS

HdfsService::HdfsService(const std::string& name_node, const int port) {
  name_node_ = name_node;
  hdfs_port_ = port;
  connect();
  connect_to_local();
}

HdfsService::~HdfsService() { disconnect(); }

void HdfsService::disconnect() {
  if (fs_) {
    const int res = hdfsDisconnect(fs_);
    HCTR_CHECK_HINT(!res, "Unable to disconnect HDFS.");
    fs_ = nullptr;
  }

  if (local_fs_) {
    const int res = hdfsDisconnect(local_fs_);
    HCTR_CHECK_HINT(!res, "Unable to disconnect local FS.");
    local_fs_ = nullptr;
  }
}

void HdfsService::connect() {
  HCTR_CHECK_HINT(!fs_, "Already connected to HDFS.");

  hdfsBuilder* const bld = hdfsNewBuilder();
  HCTR_CHECK_HINT(bld, "Unable to create HDFS builder.");

  hdfsBuilderSetNameNode(bld, name_node_.c_str());
  hdfsBuilderSetNameNodePort(bld, hdfs_port_);
  hdfsBuilderSetForceNewInstance(bld);

  fs_ = hdfsBuilderConnect(bld);
  HCTR_CHECK_HINT(fs_, "Unable to connect to HDFS.");
}

void HdfsService::connect_to_local() {
  HCTR_CHECK_HINT(!local_fs_, "Already connected to HDFS.");

  hdfsBuilder* const bld = hdfsNewBuilder();
  HCTR_CHECK_HINT(bld, "Unable to create HDFS builder.");

  hdfsBuilderSetNameNode(bld, nullptr);
  hdfsBuilderSetNameNodePort(bld, 0);
  hdfsBuilderSetForceNewInstance(bld);

  local_fs_ = hdfsBuilderConnect(bld);
  HCTR_CHECK_HINT(local_fs_, "Unable to connect to Local FS.");
}

size_t HdfsService::get_file_size(const std::string& path) const {
  HCTR_CHECK_HINT(fs_, "Not connected to HDFS.");

  hdfsFileInfo* const fi = hdfsGetPathInfo(fs_, path.c_str());

  size_t file_size;
  if (fi) {
    file_size = fi[0].mSize;
    hdfsFreeFileInfo(fi, 1);
  } else {
    file_size = 0;
    HCTR_LOG_S(ERROR, WORLD) << "HDFS file '" << path << "' does not exist." << std::endl;
  }

  return file_size;
}

int HdfsService::write(const std::string& path, const void* const data, const size_t data_size,
                       const bool overwrite) {
  HCTR_CHECK_HINT(fs_, "Not connected to HDFS.");
  HCTR_CHECK(data_size <= std::numeric_limits<tSize>::max());

  hdfsFile file;
  {
    const int path_exists = hdfsExists(fs_, path.c_str());
    if (path_exists) {
      file = hdfsOpenFile(fs_, path.c_str(), O_WRONLY | O_CREAT, 0, 0, 0);
    } else if (overwrite) {
      file = hdfsOpenFile(fs_, path.c_str(), O_WRONLY, 0, 0, 0);
    } else {
      file = hdfsOpenFile(fs_, path.c_str(), O_WRONLY | O_APPEND, 0, 0, 0);
    }
  }
  HCTR_CHECK_HINT(file, "Failed to open/create HDFS file.");

  const tSize num_written = hdfsWrite(fs_, file, data, data_size);
  HCTR_CHECK_HINT(num_written == data_size, "Writing HDFS file failed.");
  HCTR_CHECK_HINT(!hdfsFlush(fs_, file), "Flushing HDFS file failed.");
  HCTR_CHECK_HINT(!hdfsCloseFile(fs_, file), "Closing HDFS file failed.");

  HCTR_LOG_S(INFO, WORLD) << "Successfully wrote " << num_written << " bytes to HDFS file '" << path
                          << "'." << std::endl;
  return num_written;
}

int HdfsService::read(const std::string& path, void* const buffer, const size_t buffer_size,
                      const size_t offset) {
  HCTR_CHECK_HINT(fs_, "Not connected to HDFS.");
  HCTR_CHECK_HINT(buffer, "Buffer pointer is invalid.");

  hdfsFile file = hdfsOpenFile(fs_, path.c_str(), O_RDONLY, 0, 0, 0);
  HCTR_CHECK_HINT(file, "Failed to open HDFS file.");

  const tSize num_read = hdfsPread(fs_, file, offset, buffer, buffer_size);
  HCTR_CHECK_HINT(num_read == buffer_size, "Reading HDFS file failed.");
  HCTR_CHECK_HINT(!hdfsCloseFile(fs_, file), "Closing HDFS file failed.");

  return num_read;
}

void HdfsService::copy(const std::string& source_path, const std::string& target_path,
                       const bool to_local) {
  HCTR_CHECK_HINT(fs_, "Not connected to HDFS.");
  HCTR_CHECK_HINT(local_fs_, "Not connected to local FS.");

  int res;
  if (to_local) {
    res = hdfsCopy(fs_, source_path.c_str(), local_fs_, target_path.c_str());
  } else {
    res = hdfsCopy(local_fs_, source_path.c_str(), fs_, target_path.c_str());
  }
  HCTR_CHECK_HINT(!res, "HDFS copy operation failed.");

  HCTR_LOG_S(INFO, WORLD) << "Successfully copied " << source_path << " to " << target_path << '.'
                          << std::endl;
}

int HdfsService::batch_copy(const std::string& source_path, const std::string& target_path,
                            const bool to_local) {
  HCTR_CHECK_HINT(fs_, "Not connected to HDFS.");
  HCTR_CHECK_HINT(local_fs_, "Not connected to local FS.");

  hdfsFS source_fs;
  hdfsFS target_fs;
  if (to_local) {
    source_fs = fs_;
    target_fs = local_fs_;
  } else {
    source_fs = local_fs_;
    target_fs = fs_;
  }

  // Ensure source exists.
  const int source_exists = hdfsExists(source_fs, source_path.c_str());
  HCTR_CHECK_HINT(!source_exists, "HDFS source directory does not exist.");

  // Create target directory if it doesn't exist yet.
  const int target_exists = hdfsExists(target_fs, target_path.c_str());
  if (target_exists) {
    HCTR_LOG_S(INFO, WORLD) << "Creating target HDFS directory because it didn't exist."
                            << std::endl;

    const int res = hdfsCreateDirectory(target_fs, target_path.c_str());
    HCTR_CHECK_HINT(!res, "Unable to create target HDFS directory.");
  }

  hdfsFileInfo* fi;

  // Make sure we have a directory.
  fi = hdfsGetPathInfo(source_fs, source_path.c_str());
  HCTR_CHECK_HINT(fi && fi[0].mKind == kObjectKindDirectory, "Target is not a HDFS directory.");
  hdfsFreeFileInfo(fi, 1);

  // Iterate over directory
  int fi_count;
  fi = hdfsListDirectory(source_fs, source_path.c_str(), &fi_count);
  HCTR_CHECK_HINT(fi, "Listing HDFS directory failed.");

  for (int i = 0; i < fi_count; ++i) {
    if (fi[i].mKind == kObjectKindFile) {
      copy(fi[i].mName, target_path, to_local);
    }
  }

  hdfsFreeFileInfo(fi, fi_count);

  HCTR_LOG_S(INFO, WORLD) << "HDFS batch copy is complete!" << std::endl;
  return fi_count;
}

#endif  // HDFS_ENABLE

}  // namespace HugeCTR