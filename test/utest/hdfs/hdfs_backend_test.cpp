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

#include "HugeCTR/include/data_source/hdfs_backend.hpp"

#include "HugeCTR/include/data_generator.hpp"
#include "fstream"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace HugeCTR;
namespace {

void read_write_test(const std::string server, const int port) {
  std::string writepath1 = "/tmp/batch_copy/data1.txt";
  const char* buffer1 = "Hello, World!\n";

  std::string writepath2 = "/tmp/batch_copy/data2.txt";
  const char* buffer2 = "Hello, HDFS!\n";

  std::string writepath3 = "/tmp/batch_copy/data3.txt";
  const char* buffer3 = "Hello, HugeCTR!\n";

  auto hs = DataSourceParams{DataSourceType_t::HDFS, server, port}.create_unique();

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
  hs->copy("/tmp/batch_copy/data1.txt", "/tmp/local_batch_copy/data1.txt", true);
  hs->batch_copy("/tmp/batch_copy/", "/tmp/local_batch_copy/", true);

  delete[] buffer_for_read1;
  delete[] buffer_for_read2;
  delete[] buffer_for_read3;
}

void copy2hdfs_test(const std::string path, const int num_rows_per_file, const int num_files) {
  auto hs = DataSourceParams{DataSourceType_t::HDFS, "localhost", 9000}.create_unique();

  std::vector<size_t> slot_size_array = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> nnz_array = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  // generate a set of parquet file
  data_generation_for_parquet<int64_t>("../" + path + "/file_list.txt", "." + path + "gen_",
                                       num_files, num_rows_per_file, 26, 1, 13, slot_size_array,
                                       nnz_array);

  // copy the files to HDFS
  int result = hs->batch_copy("." + path, path, false);
  EXPECT_EQ(result, 0);
}

TEST(hdfs_backend_test, read_write_test_docker) { read_write_test("localhost", 9000); }

TEST(hdfs_backend_test, copy2hdfs_test_small) {
  copy2hdfs_test("/dlrm_parquet_test_small/", 20000, 40);
}
TEST(hdfs_backend_test, copy2hdfs_test_big) {
  copy2hdfs_test("/dlrm_parquet_test_big/", 200000, 4);
}
}  // namespace