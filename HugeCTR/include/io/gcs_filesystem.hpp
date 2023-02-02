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

#ifdef ENABLE_GCS
#include <google/cloud/storage/client.h>

#include <io/filesystem.hpp>
#endif

#include <memory>
#include <string>
#include <vector>

namespace HugeCTR {

#ifdef ENABLE_GCS
struct GCSCredentialsHolder {
  explicit GCSCredentialsHolder(std::shared_ptr<google::cloud::Credentials> credentials)
      : credentials(std::move(credentials)) {}
  std::shared_ptr<google::cloud::Credentials> credentials;
};

/**
 * @brief To get the bucket, object, and endpoint information by parsing GCS URL or URI.
 *
 */
struct GCSPath {
  std::string bucket;
  std::string object;
  std::string endpoint_override;

  static GCSPath FromString(const std::string& s);

  bool has_bucket_and_object() const { return !bucket.empty() && !object.empty(); }
};

/**
 * @brief GCS essential configurations for initiating client.
 *
 */
struct GCSConfigs {
  std::shared_ptr<GCSCredentialsHolder> credentials_holder;
  std::string endpoint_override;
  std::string scheme;
  std::string default_bucket;
  std::optional<double> retry_limit_time;

  GCSConfigs();

  void set_endpoint_override(const std::string& endpoint);

  static GCSConfigs Default();

  static GCSConfigs FromUrl(const std::string& url);

  static GCSConfigs FromDataSourceParams(const DataSourceParams& data_source_params);

  static GCSConfigs FromJSON(const std::string& path);
};

class GCSFileSystem final : public FileSystem {
 public:
  GCSFileSystem(const GCSConfigs& configs);

  virtual ~GCSFileSystem();

  size_t get_file_size(const std::string& path) const override;

  void create_dir(const std::string& path) override;

  void delete_file(const std::string& path) override;

  void fetch(const std::string& source_path, const std::string& target_path) override;

  void upload(const std::string& source_path, const std::string& target_path) override;

  int write(const std::string& path, const void* data, size_t data_size, bool overwrite) override;

  int read(const std::string& path, void* buffer, size_t buffer_size, size_t offset) override;

  void copy(const std::string& source_path, const std::string& target_path) override;

  void batch_fetch(const std::string& source_path, const std::string& target_path) override;

  void batch_upload(const std::string& source_dir, const std::string& target_dir) override;

 private:
  std::unique_ptr<google::cloud::storage::Client> client_;
};
#endif
}  // namespace HugeCTR