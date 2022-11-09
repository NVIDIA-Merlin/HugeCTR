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

#ifdef ENABLE_S3
#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/s3/S3Client.h>

#include <io/filesystem.hpp>
#include <io/s3_utils.hpp>
#include <memory>
#include <string>
#include <vector>
#endif

namespace HugeCTR {
#ifdef ENABLE_S3
enum class S3CredentialsType : int8_t {
  // No credentials
  Anonymous,
  // Use default AWS credential
  Default,
  // Use key pair
  KeyPair
};

struct S3Configs {
  std::string region;            // AWS region. Default depends on the AWS SDK.
  std::string scheme = "https";  // Connection transport

  double connect_timeout = -1;  // -1 means the AWS SDK default value will be used.

  double request_timeout = -1;  // -1 means the AWS SDK default value will be used.

  std::string endpoint_override;  // if not empty, with be the AWS S3 server ip + port.

  S3CredentialsType credentials_type = S3CredentialsType::Default;

  bool ready_to_connect = false;

  S3Configs();

  void configure_default(const std::string& region);

  void set_all_from_json(const std::string& path);

  static S3Configs Default(const std::string& region);

  static S3Configs FromUrl(const std::string& url);

  static S3Configs FromDataSourceParams(const DataSourceParams& data_source_params);

  static S3Configs FromJSON(const std::string& url);
};

void start_aws_sdk();

void stop_aws_sdk();

void ensure_aws_sdk_running();

class S3ClientBuilder {
 public:
  S3ClientBuilder(S3Configs configs) : configs_(std::move(configs)) {}

  const Aws::Client::ClientConfiguration& config() const { return client_configs_; }

  std::unique_ptr<Aws::S3::S3Client> build_s3client();

 protected:
  S3Configs configs_;
  Aws::Client::ClientConfiguration client_configs_;
  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> credentials_provider_;
};

class S3FileSystem final : public FileSystem {
 public:
  S3FileSystem(const S3Configs& configs);

  virtual ~S3FileSystem();

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
  std::unique_ptr<Aws::S3::S3Client> client_;
};
#endif
}  // namespace HugeCTR