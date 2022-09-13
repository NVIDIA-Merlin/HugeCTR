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

#include "io/hadoop_filesystem.hpp"

#include <base/debug/logger.hpp>
#include <exception>
#include <fstream>
#include <iostream>

#include "nlohmann/json.hpp"

namespace HugeCTR {
#ifdef ENABLE_HDFS
void HdfsConfigs::set_buffer_size(int32_t buffer_size) { this->buffer_size = buffer_size; }

void HdfsConfigs::set_replication(int16_t replication) { this->replication = replication; }

void HdfsConfigs::set_block_size(int64_t block_size) { this->block_size = block_size; }

void HdfsConfigs::set_user_name(std::string& user_name) { this->user_name = user_name; }

void HdfsConfigs::set_connection_configs(std::string namenode, int port) {
  this->namenode = namenode;
  this->port = port;
  this->ready_to_connect = true;
}

void HdfsConfigs::set_all_from_json(const std::string& path) {
  nlohmann::json config;
  std::ifstream file_stream(path);

  if (!file_stream.is_open()) {
    HCTR_OWN_THROW(Error_t::FileCannotOpen, "file_stream.is_open() failed: " + path);
  }
  try {
    file_stream >> config;
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
  }
  auto fs_type = config.find("fs_type").value();
  if (fs_type != "HDFS") {
    std::cout << "[HDFS][WARNING]: Not a HDFS configuration file, skipped" << std::endl;
    return;
  } else {
    std::string namenode = config.find("namenode").value();
    int port = config.find("port").value();
    this->set_connection_configs(namenode, port);
    int32_t buffer_size = config.find("buffer_size").value();
    this->set_block_size(buffer_size);
    int16_t replication = config.find("replication").value();
    this->set_block_size(replication);
    int64_t block_size = config.find("block_size").value();
    this->set_block_size(block_size);
    std::string user_name = config.find("user_name").value();
    this->set_user_name(user_name);
  }
  file_stream.close();
  return;
}

HdfsConfigs HdfsConfigs::FromDataSourceParams(const DataSourceParams& data_source_params) {
  HdfsConfigs configs;
  configs.set_connection_configs(data_source_params.server, data_source_params.port);
  return configs;
}

HdfsConfigs HdfsConfigs::FromUrl(const std::string& url) {
  HdfsConfigs configs;
  size_t first_colon = url.find_first_of(":");
  std::string server = url.substr(0, first_colon + 1);
  std::string body = url.substr(first_colon + 3);
  if (server != "hdfs") {
    std::cout << "[HDFS][WARN]: Not a valid HDFS URL, connection configs not created." << std::endl;
  }
  std::string ip = body.substr(0, body.find_first_of(":") + 1);
  std::string port =
      body.substr(body.find_first_of(":") + 1, body.find_first_of("/") - body.find_first_of(":"));
  configs.set_connection_configs(ip, std::stoi(port));
  return configs;
}

HdfsConfigs HdfsConfigs::FromJSON(const std::string& json_path) {
  HdfsConfigs configs;
  configs.set_all_from_json(json_path);
  return configs;
}

HadoopFileSystem::HadoopFileSystem(const std::string& name_node, const int port) {
  configs_ = HdfsConfigs();
  configs_.set_connection_configs(name_node, port);
  connect();
  connect_to_local();
}

HadoopFileSystem::~HadoopFileSystem() { disconnect(); }

void HadoopFileSystem::disconnect() {
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
void HadoopFileSystem::connect() {
  hdfsBuilder* const bld = hdfsNewBuilder();
  HCTR_CHECK_HINT(bld, "Unable to create HDFS builder.");

  hdfsBuilderSetNameNode(bld, configs_.namenode.c_str());
  hdfsBuilderSetNameNodePort(bld, configs_.port);
  hdfsBuilderSetForceNewInstance(bld);

  fs_ = hdfsBuilderConnect(bld);
  HCTR_CHECK_HINT(fs_, "Unable to connect to HDFS.");
}

void HadoopFileSystem::connect_to_local() {
  hdfsBuilder* const bld = hdfsNewBuilder();
  HCTR_CHECK_HINT(bld, "Unable to create HDFS builder.");

  hdfsBuilderSetNameNode(bld, nullptr);
  hdfsBuilderSetNameNodePort(bld, 0);
  hdfsBuilderSetForceNewInstance(bld);

  local_fs_ = hdfsBuilderConnect(bld);
  HCTR_CHECK_HINT(local_fs_, "Unable to connect to Local FS.");
}

size_t HadoopFileSystem::get_file_size(const std::string& path) const {
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

void HadoopFileSystem::create_dir(const std::string& path) {
  HCTR_CHECK_HINT(fs_, "Not connected to HDFS.");

  int res = hdfsCreateDirectory(fs_, path.c_str());

  HCTR_CHECK_HINT(res == 0, "Failed to create the directory in HDFS.");
}

void HadoopFileSystem::delete_file(const std::string& path, bool recursive) {
  HCTR_CHECK_HINT(fs_, "Not connected to HDFS.");

  int res = hdfsDelete(fs_, path.c_str(), static_cast<int>(recursive));

  HCTR_CHECK_HINT(res == 0, "Failed to delete the file in HDFS.");
}

void HadoopFileSystem::fetch(const std::string& source_path, const std::string& target_path) {
  HCTR_CHECK_HINT(fs_, "Not connected to HDFS.");
  HCTR_CHECK_HINT(local_fs_, "Not connected to local FS.");

  int res = hdfsCopy(fs_, source_path.c_str(), local_fs_, target_path.c_str());

  HCTR_CHECK_HINT(res == 0, "Failed to fetch the file from HDFS to local.");
}

void HadoopFileSystem::upload(const std::string& source_path, const std::string& target_path) {
  HCTR_CHECK_HINT(fs_, "Not connected to HDFS.");
  HCTR_CHECK_HINT(local_fs_, "Not connected to local FS.");

  int res = hdfsCopy(local_fs_, source_path.c_str(), fs_, target_path.c_str());

  HCTR_CHECK_HINT(res == 0, "Failed to upload the file from Local to HDFS.");
}

int HadoopFileSystem::write(const std::string& path, const void* const data, const size_t data_size,
                            const bool overwrite) {
  HCTR_CHECK_HINT(fs_, "Not connected to HDFS.");
  HCTR_CHECK(data_size <= std::numeric_limits<tSize>::max());

  hdfsFile file;
  {
    const int path_exists = hdfsExists(fs_, path.c_str());
    if (path_exists) {
      file = hdfsOpenFile(fs_, path.c_str(), O_WRONLY | O_CREAT, configs_.buffer_size,
                          configs_.replication, configs_.block_size);
    } else if (overwrite) {
      file = hdfsOpenFile(fs_, path.c_str(), O_WRONLY, configs_.buffer_size, configs_.replication,
                          configs_.block_size);
    } else {
      file = hdfsOpenFile(fs_, path.c_str(), O_WRONLY | O_APPEND, configs_.buffer_size,
                          configs_.replication, configs_.block_size);
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

int HadoopFileSystem::read(const std::string& path, void* const buffer, const size_t buffer_size,
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

void HadoopFileSystem::copy(const std::string& source_path, const std::string& target_path) {
  HCTR_CHECK_HINT(fs_, "Not connected to HDFS.");

  int res = hdfsCopy(fs_, source_path.c_str(), fs_, target_path.c_str());

  HCTR_CHECK_HINT(res == 0, "HDFS copy operation failed.");

  HCTR_LOG_S(INFO, WORLD) << "Successfully copied " << source_path << " to " << target_path << '.'
                          << std::endl;
}

int HadoopFileSystem::batch_fetch(const std::string& source_path, const std::string& target_path) {
  HCTR_CHECK_HINT(fs_, "Not connected to HDFS.");
  HCTR_CHECK_HINT(local_fs_, "Not connected to local FS.");

  // Ensure source exists.
  const int source_exists = hdfsExists(fs_, source_path.c_str());
  HCTR_CHECK_HINT(!source_exists, "HDFS source directory does not exist.");

  // Create target directory if it doesn't exist yet.
  const int target_exists = hdfsExists(local_fs_, target_path.c_str());
  if (target_exists) {
    HCTR_LOG_S(INFO, WORLD) << "Creating target HDFS directory because it didn't exist."
                            << std::endl;

    const int res = hdfsCreateDirectory(local_fs_, target_path.c_str());
    HCTR_CHECK_HINT(res == 0, "Unable to create target HDFS directory.");
  }

  hdfsFileInfo* fi;

  // Make sure we have a directory.
  fi = hdfsGetPathInfo(fs_, source_path.c_str());
  HCTR_CHECK_HINT(fi && fi[0].mKind == kObjectKindDirectory, "Target is not a HDFS directory.");
  hdfsFreeFileInfo(fi, 1);

  // Iterate over directory
  int fi_count;
  fi = hdfsListDirectory(fs_, source_path.c_str(), &fi_count);
  HCTR_CHECK_HINT(fi, "Listing HDFS directory failed.");

  for (int i = 0; i < fi_count; ++i) {
    if (fi[i].mKind == kObjectKindFile) {
      fetch(fi[i].mName, target_path);
    }
  }

  hdfsFreeFileInfo(fi, fi_count);

  HCTR_LOG_S(INFO, WORLD) << "HDFS batch fetch is complete!" << std::endl;
  return fi_count;
}

int HadoopFileSystem::batch_upload(const std::string& source_path, const std::string& target_path) {
  HCTR_CHECK_HINT(fs_, "Not connected to HDFS.");
  HCTR_CHECK_HINT(local_fs_, "Not connected to local FS.");

  // Ensure source exists.
  const int source_exists = hdfsExists(local_fs_, source_path.c_str());
  HCTR_CHECK_HINT(!source_exists, "Source directory does not exist.");

  // Create target directory if it doesn't exist yet.
  const int target_exists = hdfsExists(fs_, target_path.c_str());
  if (target_exists) {
    HCTR_LOG_S(INFO, WORLD) << "Creating target directory because it didn't exist." << std::endl;

    const int res = hdfsCreateDirectory(fs_, target_path.c_str());
    HCTR_CHECK_HINT(res == 0, "Unable to create target directory.");
  }

  hdfsFileInfo* fi;

  // Make sure we have a directory.
  fi = hdfsGetPathInfo(local_fs_, source_path.c_str());
  HCTR_CHECK_HINT(fi && fi[0].mKind == kObjectKindDirectory, "Target is not a directory.");
  hdfsFreeFileInfo(fi, 1);

  // Iterate over directory
  int fi_count;
  fi = hdfsListDirectory(local_fs_, source_path.c_str(), &fi_count);
  HCTR_CHECK_HINT(fi, "Listing HDFS directory failed.");

  for (int i = 0; i < fi_count; ++i) {
    if (fi[i].mKind == kObjectKindFile) {
      upload(fi[i].mName, target_path);
    }
  }

  hdfsFreeFileInfo(fi, fi_count);

  HCTR_LOG_S(INFO, WORLD) << "HDFS batch upload is complete!" << std::endl;
  return fi_count;
}
#endif
}  // namespace HugeCTR