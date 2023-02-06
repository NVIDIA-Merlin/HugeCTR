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

#include <gtest/gtest.h>

#include <fstream>
#include <io/gcs_filesystem.hpp>
#include <io/io_utils.hpp>
#include <utest/test_utils.hpp>

using namespace HugeCTR;

namespace {

void gcs_path_match_test() {
  std::string url1 = "https://storage.googleapis.com/hugectr-io-test/ci_test/";
  std::string url2 = "https://storage.googleapis.com";
  std::string url3 = "https://storage.goo.com";
  EXPECT_EQ(IOUtils::is_valid_gcs_https_url(url1), true);
  EXPECT_EQ(IOUtils::is_valid_s3_https_url(url1), false);
  EXPECT_EQ(IOUtils::is_valid_gcs_https_url(url2), true);
  EXPECT_EQ(IOUtils::is_valid_gcs_https_url(url3), false);
}

void gcs_configs_test() {
  auto dsp = DataSourceParams{FileSystemType_t::GCS, "storage.googleapis.com", 9000};
  auto configs = GCSConfigs::FromDataSourceParams(dsp);
  EXPECT_EQ(configs.endpoint_override, "storage.googleapis.com");
  EXPECT_EQ(configs.scheme, "https");
}

void gcs_path_test() {
  GCSPath gcs_path_1 =
      GCSPath::FromString("https://storage.googleapis.com/hugectr-io-test/ci_test/");
  EXPECT_EQ(gcs_path_1.bucket, "hugectr-io-test");
  EXPECT_EQ(gcs_path_1.object, "ci_test/");
  EXPECT_EQ(gcs_path_1.endpoint_override, "storage.googleapis.com");

  GCSPath gcs_path_2 = GCSPath::FromString("gs://hugectr-io-test/ci_test/data1.txt");
  EXPECT_EQ(gcs_path_2.bucket, "hugectr-io-test");
  EXPECT_EQ(gcs_path_2.object, "ci_test/data1.txt");
  EXPECT_EQ(gcs_path_2.endpoint_override.empty(), true);
}

void gcs_simple_read_write_test_with_dsp() {
  auto dsp = DataSourceParams{FileSystemType_t::GCS, "storage.googleapis.com", 8888};
  auto hs = FileSystemBuilder::build_unique_by_data_source_params(dsp);
  std::string writepath1 = "gs://hugectr-io-test/ci_test/data1.txt";
  const char* buffer1 = "Hello, World!\n";

  hs->write(writepath1, buffer1, strlen(buffer1), true);
  char* buffer_for_read1 = new char[hs->get_file_size(writepath1)];
  hs->read(writepath1, buffer_for_read1, hs->get_file_size(writepath1), 0);
  EXPECT_EQ(*buffer1, *buffer_for_read1);
  delete[] buffer_for_read1;
}

void gcs_simple_read_write_test_with_path() {
  auto hs = FileSystemBuilder::build_unique_by_path(
      "https://storage.googleapis.com/hugectr-io-test/ci_test/data2.txt");
  std::string writepath1 = "https://storage.googleapis.com/hugectr-io-test/ci_test/data2.txt";
  const char* buffer1 = "Hello, World!\n";

  hs->write(writepath1, buffer1, strlen(buffer1), true);
  char* buffer_for_read1 = new char[hs->get_file_size(writepath1)];
  hs->read(writepath1, buffer_for_read1, hs->get_file_size(writepath1), 0);
  EXPECT_EQ(*buffer1, *buffer_for_read1);
  delete[] buffer_for_read1;
}

TEST(gcs_backend_test, gcs_path_match_test) { gcs_path_match_test(); }

TEST(gcs_backend_test, gcs_configs_test) { gcs_configs_test(); }

TEST(gcs_backend_test, gcs_path_test) { gcs_path_test(); }

TEST(gcs_backend_test, gcs_builder_with_dsp_test) { gcs_simple_read_write_test_with_dsp(); }

TEST(gcs_backend_test, gcs_builder_with_file_test) { gcs_simple_read_write_test_with_path(); }

}  // namespace