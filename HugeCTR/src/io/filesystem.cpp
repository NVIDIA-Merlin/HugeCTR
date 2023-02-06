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

#include <base/debug/logger.hpp>
#include <io/filesystem.hpp>
#include <io/gcs_filesystem.hpp>
#include <io/hadoop_filesystem.hpp>
#include <io/io_utils.hpp>
#include <io/local_filesystem.hpp>
#include <io/s3_filesystem.hpp>

namespace HugeCTR {

FileSystem* FileSystemBuilder::build_by_path(const std::string& file_path) {
  std::string scheme = IOUtils::get_path_scheme(file_path);
  FileSystemType_t fs_type;
  if (scheme == "") {
    fs_type = FileSystemType_t::Local;
  } else if (scheme == "hdfs") {
    fs_type = FileSystemType_t::HDFS;
  } else if (scheme == "S3" || scheme == "s3") {
    fs_type = FileSystemType_t::S3;
  } else if (scheme == "GS" || scheme == "gs") {
    fs_type = FileSystemType_t::GCS;
  } else if (scheme == "https") {
    if (IOUtils::is_valid_s3_https_url(file_path)) {
      fs_type = FileSystemType_t::S3;
    } else if (IOUtils::is_valid_gcs_https_url(file_path)) {
      fs_type = FileSystemType_t::GCS;
    } else {
      fs_type = FileSystemType_t::Other;
    }
  } else {
    fs_type = FileSystemType_t::Other;
  }
  switch (fs_type) {
    case FileSystemType_t::Local:
      return new LocalFileSystem{};
    case FileSystemType_t::HDFS:
#ifdef ENABLE_HDFS
      return new HadoopFileSystem{HdfsConfigs::FromUrl(file_path)};
#else
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Please install Hadoop and compile HugeCTR with ENABLE_HDFS to use HDFS "
                     "functionalities.");
#endif
    case FileSystemType_t::S3:
#ifdef ENABLE_S3
      return new S3FileSystem{S3Configs::FromUrl(file_path)};
#else
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Please install AWS s3 sdk and compile HugeCTR with ENABLE_S3 to use S3 "
                     "functionalities.");
#endif
    case FileSystemType_t::GCS:
#ifdef ENABLE_GCS
      return new GCSFileSystem{GCSConfigs::FromUrl(file_path)};
#else
      HCTR_OWN_THROW(
          Error_t::WrongInput,
          "Please install Google Cloud sdk and compile HugeCTR with ENABLE_GCS to use GCS "
          "functionalities.");
#endif
    default:
      HCTR_OWN_THROW(Error_t::WrongInput, "Unsupproted filesystem.");
  }

  return nullptr;
}

FileSystem* FileSystemBuilder::build_by_data_source_params(
    const DataSourceParams& data_source_params) {
  switch (data_source_params.type) {
    case FileSystemType_t::Local:
      return new LocalFileSystem{};
    case FileSystemType_t::HDFS:
#ifdef ENABLE_HDFS
      return new HadoopFileSystem{HdfsConfigs::FromDataSourceParams(data_source_params)};
#else
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Please install Hadoop and compile HugeCTR with ENABLE_HDFS to use HDFS "
                     "functionalities.");
#endif
    case FileSystemType_t::S3:
#ifdef ENABLE_S3
      return new S3FileSystem{S3Configs::FromDataSourceParams(data_source_params)};
#else
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Please install AWS s3 sdk and compile HugeCTR with ENABLE_S3 to use S3 "
                     "functionalities.");
#endif
    case FileSystemType_t::GCS:
#ifdef ENABLE_GCS
      return new GCSFileSystem{GCSConfigs::FromDataSourceParams(data_source_params)};
#else
      HCTR_OWN_THROW(
          Error_t::WrongInput,
          "Please install Google Cloud sdk and compile HugeCTR with ENABLE_GCS to use GCS "
          "functionalities.");
#endif
    default:
      HCTR_OWN_THROW(Error_t::WrongInput, "Unsupproted filesystem.");
  }

  return nullptr;
}

FileSystem* FileSystemBuilder::build_by_config(const std::string& config_path) {
  std::string fs_type_string = IOUtils::get_fs_type_from_json(config_path);
  FileSystemType_t fs_type;
  if (fs_type_string == "Local") {
    fs_type = FileSystemType_t::Local;
  } else if (fs_type_string == "HDFS") {
    fs_type = FileSystemType_t::HDFS;
  } else if (fs_type_string == "S3") {
    fs_type = FileSystemType_t::S3;
  } else if (fs_type_string == "GCS") {
    fs_type = FileSystemType_t::GCS;
  } else {
    fs_type = FileSystemType_t::Other;
  }
  switch (fs_type) {
    case FileSystemType_t::Local:
      return new LocalFileSystem{};
    case FileSystemType_t::HDFS:
#ifdef ENABLE_HDFS
      return new HadoopFileSystem{HdfsConfigs::FromJSON(config_path)};
#else
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Please install Hadoop and compile HugeCTR with ENABLE_HDFS to use HDFS "
                     "functionalities.");
#endif
    case FileSystemType_t::S3:
#ifdef ENABLE_S3
      return new S3FileSystem{S3Configs::FromJSON(config_path)};
#else
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Please install AWS s3 sdk and compile HugeCTR with ENABLE_S3 to use S3 "
                     "functionalities.");
#endif
    case FileSystemType_t::GCS:
#ifdef ENABLE_GCS
      return new GCSFileSystem{GCSConfigs::FromJSON(config_path)};
#else
      HCTR_OWN_THROW(
          Error_t::WrongInput,
          "Please install Google Cloud sdk and compile HugeCTR with ENABLE_GCS to use GCS "
          "functionalities.");
#endif
    default:
      HCTR_OWN_THROW(Error_t::WrongInput, "Unsupproted filesystem.");
  }
  return nullptr;
}

}  // namespace HugeCTR
