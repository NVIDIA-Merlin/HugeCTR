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

#include "fstream"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace HugeCTR;
namespace {

void hdfs_backend_test(const std::string server, const int port) {
  std::string writepath1 = "/tmp/batch_copy/data1.txt";
  const char* buffer1 = "Hello, World!\n";

  std::string writepath2 = "/tmp/batch_copy/data2.txt";
  const char* buffer2 = "Hello, HDFS!\n";

  std::string writepath3 = "/tmp/batch_copy/data3.txt";
  const char* buffer3 = "Hello, HugeCTR!\n";

  HdfsService hs = HdfsService(server, port);
  hs.write(writepath1, buffer1, strlen(buffer1), true);
  hs.write(writepath2, buffer2, strlen(buffer2), true);
  hs.write(writepath3, buffer3, strlen(buffer3), true);

  char* buffer_for_read1 = (char*)malloc(hs.getFileSize(writepath1));
  char* buffer_for_read2 = (char*)malloc(hs.getFileSize(writepath2));
  char* buffer_for_read3 = (char*)malloc(hs.getFileSize(writepath3));
  hs.read(writepath1, buffer_for_read1, hs.getFileSize(writepath1), 0);
  hs.read(writepath2, buffer_for_read2, hs.getFileSize(writepath2), 0);
  hs.read(writepath3, buffer_for_read3, hs.getFileSize(writepath3), 0);

  EXPECT_EQ(*buffer1, *buffer_for_read1);
  EXPECT_EQ(*buffer2, *buffer_for_read2);
  EXPECT_EQ(*buffer3, *buffer_for_read3);
  hs.copyToLocal("/tmp/batch_copy/data1.txt", "/tmp/local_batch_copy/");
  hs.batchCopyToLocal("/tmp/batch_copy", "/tmp/local_batch_copy");
}

std::string server = "localhost";
int port = 9000;
TEST(hdfs_backend_test, read_write_test) { hdfs_backend_test(server, port); }
}  // namespace