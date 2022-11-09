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

#include "HugeCTR/include/data_generator.hpp"
#include "HugeCTR/include/io/hadoop_filesystem.hpp"
#include "fstream"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace HugeCTR;
namespace {

void hdfs_configs_test() {
  auto configs1 = HdfsConfigs::FromUrl("hdfs://localhost:9000/my/dir/to/data");
  EXPECT_EQ(configs1.namenode, "localhost");
  EXPECT_EQ(configs1.port, 9000);
  EXPECT_EQ(configs1.ready_to_connect, true);

  auto dsp = DataSourceParams{FileSystemType_t::HDFS, "localhost", 9000};
  auto configs3 = HdfsConfigs::FromDataSourceParams(dsp);
  EXPECT_EQ(configs3.namenode, "localhost");
  EXPECT_EQ(configs3.port, 9000);
  EXPECT_EQ(configs3.ready_to_connect, true);
}

void simple_read_write_test_with_builder() {
  auto hs = FileSystemBuilder::build_unique_by_path("hdfs://localhost:9000/my/dir/to/data");
  std::string writepath1 = "/tmp/batch_copy/data1.txt";
  const char* buffer1 = "Hello, World!\n";

  hs->write(writepath1, buffer1, strlen(buffer1), true);
  char* buffer_for_read1 = new char[hs->get_file_size(writepath1)];
  hs->read(writepath1, buffer_for_read1, hs->get_file_size(writepath1), 0);
  EXPECT_EQ(*buffer1, *buffer_for_read1);
  delete[] buffer_for_read1;
}

void simple_read_write_test(const std::string server, const int port) {
  std::string writepath1 = "/tmp/batch_copy/data1.txt";
  const char* buffer1 = "Hello, World!\n";

  std::string writepath2 = "/tmp/batch_copy/data2.txt";
  const char* buffer2 = "Hello, HDFS!\n";

  std::string writepath3 = "/tmp/batch_copy/data3.txt";
  const char* buffer3 = "Hello, HugeCTR!\n";

  auto hs = FileSystemBuilder::build_unique_by_data_source_params(
      DataSourceParams{FileSystemType_t::HDFS, server, port});

  hs->write(writepath1, buffer1, strlen(buffer1), true);
  hs->write(writepath2, buffer2, strlen(buffer2), true);
  hs->write(writepath3, buffer3, strlen(buffer3), true);

  char* buffer_for_read1 = new char[hs->get_file_size(writepath1)];
  char* buffer_for_read2 = new char[hs->get_file_size(writepath2)];
  char* buffer_for_read3 = new char[hs->get_file_size(writepath3)];

  hs->read(writepath1, buffer_for_read1, hs->get_file_size(writepath1), 0);
  hs->read(writepath2, buffer_for_read2, hs->get_file_size(writepath2), 0);
  hs->read(writepath3, buffer_for_read3, hs->get_file_size(writepath3), 0);

  EXPECT_EQ(*buffer1, *buffer_for_read1);
  EXPECT_EQ(*buffer2, *buffer_for_read2);
  EXPECT_EQ(*buffer3, *buffer_for_read3);

  delete[] buffer_for_read1;
  delete[] buffer_for_read2;
  delete[] buffer_for_read3;
}

void upload2hdfs_test(const std::string path, const int num_rows_per_file, const int num_files) {
  auto hs = FileSystemBuilder::build_unique_by_data_source_params(
      DataSourceParams{FileSystemType_t::HDFS, "localhost", 9000});

  std::vector<size_t> slot_size_array = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> nnz_array = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  // generate a set of parquet file
  data_generation_for_parquet<int64_t>("../" + path + "/file_list.txt", "." + path + "gen_",
                                       num_files, num_rows_per_file, 26, 1, 13, slot_size_array,
                                       nnz_array);

  // copy the files to HDFS
  hs->batch_upload("." + path, path);
}

void copy_test(const std::string source_path, const std::string target_path) {
  auto hs = FileSystemBuilder::build_unique_by_data_source_params(
      DataSourceParams{FileSystemType_t::HDFS, "localhost", 9000});

  hs->copy(source_path, target_path);
}

void delete_test(const std::string path) {
  auto hs = FileSystemBuilder::build_unique_by_data_source_params(
      DataSourceParams{FileSystemType_t::HDFS, "localhost", 9000});

  hs->delete_file(path);
}

void fetch2local_test(const std::string path, const std::string local_path, const int num_files) {
  auto hs = FileSystemBuilder::build_unique_by_data_source_params(
      DataSourceParams{FileSystemType_t::HDFS, "localhost", 9000});

  hs->batch_fetch(path, "." + path);
}

TEST(hdfs_backend_test, hdfs_configs_test) { hdfs_configs_test(); }

TEST(hdfs_backend_test, hdfs_builder_test) { simple_read_write_test_with_builder(); }

TEST(hdfs_backend_test, read_write_test_docker) { simple_read_write_test("localhost", 9000); }

TEST(hdfs_backend_test, upload2hdfs_test_small) {
  upload2hdfs_test("/dlrm_parquet_test_small/", 20000, 40);
}
TEST(hdfs_backend_test, upload2hdfs_test_big) {
  upload2hdfs_test("/dlrm_parquet_test_big/", 200000, 4);
}

TEST(hdfs_backend_test, copy_small) {
  copy_test("/dlrm_parquet_test_small/", "/dlrm_parquet_test_small_copy/");
}
TEST(hdfs_backend_test, copy_big) {
  copy_test("/dlrm_parquet_test_big/", "/dlrm_parquet_test_big_copy/");
}

TEST(hdfs_backend_test, delete_remote) {
  delete_test("/dlrm_parquet_test_small_copy/");
  delete_test("/dlrm_parquet_test_big_copy/");
}

TEST(hdfs_backend_test, fetch2local_small) {
  fetch2local_test("/dlrm_parquet_test_small/", "../dlrm_parquet_test_small", 40);
}
TEST(hdfs_backend_test, fetch2local_big) {
  fetch2local_test("/dlrm_parquet_test_big/", "../dlrm_parquet_test_big", 4);
}
}  // namespace