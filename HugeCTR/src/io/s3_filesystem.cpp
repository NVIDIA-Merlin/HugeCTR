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
#ifdef ENABLE_S3
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/utils/logging/ConsoleLogSystem.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/s3/model/CopyObjectRequest.h>
#include <aws/s3/model/CreateBucketRequest.h>
#include <aws/s3/model/DeleteBucketRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/s3/model/DeleteObjectsRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#endif
#include <sys/stat.h>

#include <base/debug/logger.hpp>
#include <fstream>
#include <io/s3_filesystem.hpp>
#include <io/s3_utils.hpp>

#include "nlohmann/json.hpp"

namespace HugeCTR {

#ifdef ENABLE_S3

Aws::SDKOptions sdk_options;
static bool sdk_running = false;

void start_aws_sdk() {
  Aws::Utils::Logging::LogLevel sdk_log_level = Aws::Utils::Logging::LogLevel::Off;
  sdk_options.loggingOptions.logLevel = sdk_log_level;

  // Move logging to console instead of file
  sdk_options.loggingOptions.logger_create_fn = [] {
    return std::make_shared<Aws::Utils::Logging::ConsoleLogSystem>(
        sdk_options.loggingOptions.logLevel);
  };
  Aws::InitAPI(sdk_options);
  sdk_running = true;
}

void stop_aws_sdk() {
  Aws::ShutdownAPI(sdk_options);
  sdk_running = false;
}

void ensure_aws_sdk_running() {
  if (!sdk_running) {
    start_aws_sdk();
  }
}

S3Configs::S3Configs() {}

void S3Configs::configure_default(const std::string& region) {
  this->region = region;
  this->credentials_type = S3CredentialsType::Default;
  this->ready_to_connect = true;
}

void S3Configs::set_all_from_json(const std::string& path) {
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
  std::string fs_type = (std::string)config.find("fs_type").value();
  HCTR_CHECK_HINT(fs_type == "S3", "Not a valid S3 configuration file.");
  // TODO: parse more configs
  file_stream.close();
  return;
}

S3Configs S3Configs::Default(const std::string& region) {
  S3Configs configs;
  configs.configure_default(region);
  return configs;
}

S3Configs S3Configs::FromUrl(const std::string& url) {
  S3Configs configs;
  configs.configure_default(get_region_from_url(url));
  return configs;
}

S3Configs S3Configs::FromDataSourceParams(const DataSourceParams& data_source_params) {
  S3Configs configs;
  configs.configure_default(data_source_params.server);
  return configs;
}

S3Configs S3Configs::FromJSON(const std::string& json_path) {
  S3Configs configs;
  configs.set_all_from_json(json_path);
  return configs;
}

std::unique_ptr<Aws::S3::S3Client> S3ClientBuilder::build_s3client() {
  HCTR_CHECK_HINT(configs_.ready_to_connect,
                  "Need to provide region information to build the client.");
  switch (configs_.credentials_type) {
    case S3CredentialsType::Default:
      credentials_provider_ = std::make_shared<Aws::Auth::DefaultAWSCredentialsProviderChain>();
      break;
    default:
      HCTR_OWN_THROW(Error_t::WrongInput, "Not supported credential chain.");
  }
  if (!configs_.region.empty()) {
    client_configs_.region = S3Utils::to_aws_string(configs_.region);
  }
  if (configs_.request_timeout > 0) {
    client_configs_.requestTimeoutMs = static_cast<long>(ceil(configs_.request_timeout * 1000));
  }
  if (configs_.connect_timeout > 0) {
    client_configs_.connectTimeoutMs = static_cast<long>(ceil(configs_.connect_timeout * 1000));
  }
  client_configs_.endpointOverride = S3Utils::to_aws_string(configs_.endpoint_override);
  if (configs_.scheme == "http") {
    client_configs_.scheme = Aws::Http::Scheme::HTTP;
  } else if (configs_.scheme == "https") {
    client_configs_.scheme = Aws::Http::Scheme::HTTPS;
  } else {
    HCTR_OWN_THROW(Error_t::WrongInput, "Invalid AWS scheme");
  }
  auto s3client = std::make_unique<Aws::S3::S3Client>(
      credentials_provider_, client_configs_,
      Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, false);
  return s3client;
}

class StringViewStream : Aws::Utils::Stream::PreallocatedStreamBuf, public std::iostream {
 public:
  StringViewStream(const void* data, size_t nbytes)
      : Aws::Utils::Stream::PreallocatedStreamBuf(
            reinterpret_cast<unsigned char*>(const_cast<void*>(data)), nbytes),
        std::iostream(this) {}
};

Aws::IOStreamFactory AwsWriteableStreamFactory(void* data, size_t nbytes) {
  return [=]() { return Aws::New<StringViewStream>("", data, nbytes); };
}

S3FileSystem::S3FileSystem(const S3Configs& configs) {
  ensure_aws_sdk_running();
  S3ClientBuilder builder = S3ClientBuilder(configs);
  client_ = builder.build_s3client();
}

S3FileSystem::~S3FileSystem() {}

size_t S3FileSystem::get_file_size(const std::string& path) const {
  S3Path s3_path = S3Path::FromString(path);
  HCTR_CHECK_HINT(s3_path.has_bucket_and_key(),
                  "This S3 path does not contain bucket or key information.");
  Aws::S3::Model::HeadObjectRequest head_request;
  head_request.SetBucket(S3Utils::to_aws_string(s3_path.bucket));
  head_request.SetKey(S3Utils::to_aws_string(s3_path.key));
  auto outcome = client_->HeadObject(head_request);
  HCTR_CHECK_HINT(outcome.IsSuccess(), "Failed to open the file in S3.");
  size_t content_length = outcome.GetResult().GetContentLength();
  return content_length;
}

void S3FileSystem::create_dir(const std::string& path) {
  S3Path s3_path = S3Path::FromString(path);
  HCTR_CHECK_HINT(s3_path.has_bucket_and_key(),
                  "This S3 path does not contain bucket or key information.");
  Aws::S3::Model::PutObjectRequest request;
  request.SetBucket(S3Utils::to_aws_string(s3_path.bucket));
  request.SetKey(S3Utils::to_aws_string(s3_path.key + '/'));
  request.SetBody(std::make_shared<std::stringstream>(""));
  Aws::S3::Model::PutObjectOutcome outcome = client_->PutObject(request);
  HCTR_CHECK_HINT(outcome.IsSuccess(), "Failed to create the directory in S3.");
}

void S3FileSystem::delete_file(const std::string& path, bool recursive) {
  // S3 deletion is always recursive
  S3Path s3_path = S3Path::FromString(path);
  HCTR_CHECK_HINT(s3_path.has_bucket_and_key(),
                  "This S3 path does not contain bucket or key information.");
  Aws::S3::Model::DeleteObjectRequest request;
  request.SetBucket(S3Utils::to_aws_string(s3_path.bucket));
  request.SetKey(S3Utils::to_aws_string(s3_path.key));
  Aws::S3::Model::DeleteObjectOutcome outcome = client_->DeleteObject(request);
  HCTR_CHECK_HINT(outcome.IsSuccess(), "Failed to delete the file/dir in S3.");
}

void S3FileSystem::fetch(const std::string& source_path, const std::string& target_path) {
  // TODO
  HCTR_DIE("Not implemented yet!");
}

void S3FileSystem::upload(const std::string& source_path, const std::string& target_path) {
  // TODO
  HCTR_DIE("Not implemented yet!");
}

int S3FileSystem::write(const std::string& path, const void* const data, const size_t data_size,
                        const bool overwrite) {
  S3Path s3_path = S3Path::FromString(path);
  HCTR_CHECK_HINT(s3_path.has_bucket_and_key(),
                  "This S3 path does not contain bucket or key information.");
  Aws::S3::Model::PutObjectRequest request;
  request.SetBucket(S3Utils::to_aws_string(s3_path.bucket));
  request.SetKey(S3Utils::to_aws_string(s3_path.key));

  auto aws_stream = Aws::MakeShared<Aws::StringStream>(
      "WriteObjectInputStream",
      std::stringstream::in | std::stringstream::out | std::stringstream::binary);
  aws_stream->write(reinterpret_cast<char*>(const_cast<void*>(data)), data_size);
  request.SetBody(aws_stream);

  Aws::S3::Model::PutObjectOutcome outcome = client_->PutObject(request);
  HCTR_CHECK_HINT(outcome.IsSuccess(), "Failed to write to S3.");
  HCTR_LOG_S(DEBUG, WORLD) << "Successfully write to AWS S3 location:  " << path << std::endl;
  return data_size;
}

int S3FileSystem::read(const std::string& path, void* const buffer, const size_t buffer_size,
                       const size_t offset) {
  size_t content_length = get_file_size(path);
  size_t nbytes = std::min(buffer_size, content_length - offset);
  S3Path s3_path = S3Path::FromString(path);
  Aws::S3::Model::GetObjectRequest get_request;
  get_request.SetBucket(S3Utils::to_aws_string(s3_path.bucket));
  get_request.SetKey(S3Utils::to_aws_string(s3_path.key));
  std::stringstream ss;
  ss << "bytes=" << (int64_t)offset << "-" << (int64_t)offset + (int64_t)nbytes - 1;
  get_request.SetRange(S3Utils::to_aws_string(ss.str()));
  get_request.SetResponseStreamFactory(AwsWriteableStreamFactory(buffer, nbytes));
  Aws::S3::Model::GetObjectOutcome outcome = client_->GetObject(get_request);
  HCTR_CHECK_HINT(outcome.IsSuccess(), "Failed to read the file.");
  Aws::S3::Model::GetObjectResult result = std::move(outcome).GetResultWithOwnership();
  auto& stream = result.GetBody();
  return stream.gcount();
}

void S3FileSystem::copy(const std::string& source_path, const std::string& target_path) {
  S3Path source = S3Path::FromString(source_path);
  HCTR_CHECK_HINT(source.has_bucket_and_key(),
                  "This source path does not contain bucket or key information.");
  S3Path dest = S3Path::FromString(target_path);
  HCTR_CHECK_HINT(dest.has_bucket_and_key(),
                  "This source path does not contain bucket or key information.");
  Aws::S3::Model::CopyObjectRequest request;
  request.SetBucket(S3Utils::to_aws_string(dest.bucket));
  request.SetKey(S3Utils::to_aws_string(dest.key));
  request.SetCopySource(source.to_aws_string());
  Aws::S3::Model::CopyObjectOutcome outcome = client_->CopyObject(request);
  HCTR_CHECK_HINT(outcome.IsSuccess(), "Failed to copy the files.");
}

int S3FileSystem::batch_fetch(const std::string& source_path, const std::string& target_path) {
  fetch(source_path, target_path);
  return 1;
}

int S3FileSystem::batch_upload(const std::string& source_path, const std::string& target_path) {
  upload(source_path, target_path);
  return 1;
}
#endif
}  // namespace HugeCTR