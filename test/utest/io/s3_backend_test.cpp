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
#include <io/s3_filesystem.hpp>
#include <utest/test_utils.hpp>

using namespace HugeCTR;

namespace {

void s3_configs_test() {
  auto dsp = DataSourceParams{FileSystemType_t::S3, "us-east-1", 9000};
  auto configs = S3Configs::FromDataSourceParams(dsp);
  EXPECT_EQ(configs.region, "us-east-1");
  EXPECT_EQ(configs.ready_to_connect, true);
}

void s3_path_test() {
  S3Path s3_path_1 =
      S3Path::FromString("https://s3.us-east-1.amazonaws.com/hugectr-io-test/ci_test/");
  EXPECT_EQ(s3_path_1.bucket, "hugectr-io-test");
  EXPECT_EQ(s3_path_1.key, "ci_test/");
  EXPECT_EQ(s3_path_1.region, "us-east-1");

  S3Path s3_path_2 = S3Path::FromString("s3://hugectr-io-test/ci_test/data1.txt");
  EXPECT_EQ(s3_path_2.bucket, "hugectr-io-test");
  EXPECT_EQ(s3_path_2.key, "ci_test/data1.txt");
  EXPECT_EQ(s3_path_2.region, "");

  S3Path s3_path_3 =
      S3Path::FromString("https://hugectr-io-test.s3.us-east-1.amazonaws.com/ci_test/");
  EXPECT_EQ(s3_path_3.bucket, "hugectr-io-test");
  EXPECT_EQ(s3_path_3.key, "ci_test/");
  EXPECT_EQ(s3_path_3.region, "us-east-1");
}

void s3_simple_read_write_test_with_dsp() {
  auto dsp = DataSourceParams{FileSystemType_t::S3, "us-east-1", 8888};
  auto hs = FileSystemBuilder::build_unique_by_data_source_params(dsp);
  std::string writepath1 = "s3://hugectr-io-test/ci_test/data1.txt";
  const char* buffer1 = "Hello, World!\n";

  hs->write(writepath1, buffer1, strlen(buffer1), true);
  char* buffer_for_read1 = new char[hs->get_file_size(writepath1)];
  hs->read(writepath1, buffer_for_read1, hs->get_file_size(writepath1), 0);
  EXPECT_EQ(*buffer1, *buffer_for_read1);
  delete[] buffer_for_read1;
}

void s3_simple_read_write_test_with_path() {
  auto hs = FileSystemBuilder::build_unique_by_path(
      "https://s3.us-east-1.amazonaws.com/hugectr-io-test/ci_test/data2.txt");
  std::string writepath1 = "https://s3.us-east-1.amazonaws.com/hugectr-io-test/ci_test/data2.txt";
  const char* buffer1 = "Hello, World!\n";

  hs->write(writepath1, buffer1, strlen(buffer1), true);
  char* buffer_for_read1 = new char[hs->get_file_size(writepath1)];
  hs->read(writepath1, buffer_for_read1, hs->get_file_size(writepath1), 0);
  EXPECT_EQ(*buffer1, *buffer_for_read1);
  delete[] buffer_for_read1;
}

TEST(s3_backend_test, s3_configs_test) { s3_configs_test(); }

TEST(s3_backend_test, s3_path_test) { s3_path_test(); }

TEST(s3_backend_test, s3_builder_with_dsp_test) { s3_simple_read_write_test_with_dsp(); }

TEST(s3_backend_test, s3_builder_with_file_test) { s3_simple_read_write_test_with_path(); }

}  // namespace