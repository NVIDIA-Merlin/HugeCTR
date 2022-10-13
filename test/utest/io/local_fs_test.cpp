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
#include "HugeCTR/include/io/filesystem.hpp"
#include "fstream"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace HugeCTR;
namespace {

void simple_read_write_test_with_builder() {
  auto hs = FileSystemBuilder::build_unique_by_path("./tmp/data1.txt");
  std::string writepath1 = "./tmp/data1.txt";
  const char* buffer1 = "Hello, World!\n";

  hs->write(writepath1, buffer1, strlen(buffer1), true);
  char* buffer_for_read1 = new char[hs->get_file_size(writepath1)];
  hs->read(writepath1, buffer_for_read1, hs->get_file_size(writepath1), 0);
  EXPECT_EQ(*buffer1, *buffer_for_read1);
  delete[] buffer_for_read1;
}

void simple_read_write_test() {
  std::string writepath1 = "./tmp/data1.txt";
  const char* buffer1 = "Hello, World!\n";

  std::string writepath2 = "./tmp/data2.txt";
  const char* buffer2 = "Hello, LocalFS!\n";

  std::string writepath3 = "./tmp/data3.txt";
  const char* buffer3 = "Hello, HugeCTR!\n";

  auto hs = FileSystemBuilder::build_unique_by_data_source_params(
      DataSourceParams{FileSystemType_t::Local, "", 8888});

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

void append_test() {
  std::string writepath = "./tmp/append/data.txt";
  const char* buffer1 = "Hello, World!\n";

  const char* buffer2 = "Hello, LocalFS!\n";

  const char* buffer3 = "Hello, HugeCTR!\n";

  const char* buffer_all = "Hello, World!\nHello, LocalFS!\nHello, HugeCTR!\n";

  auto hs = FileSystemBuilder::build_unique_by_data_source_params(
      DataSourceParams{FileSystemType_t::Local, "", 8888});

  hs->write(writepath, buffer1, strlen(buffer1), true);
  hs->write(writepath, buffer2, strlen(buffer2), false);
  hs->write(writepath, buffer3, strlen(buffer3), false);

  char* buffer_for_read = new char[hs->get_file_size(writepath)];

  hs->read(writepath, buffer_for_read, hs->get_file_size(writepath), 0);

  EXPECT_EQ(*buffer_all, *buffer_for_read);

  delete[] buffer_for_read;
}

TEST(local_fs_test, fs_builder_test) { simple_read_write_test_with_builder(); }

TEST(local_fs_test, read_write_test) { simple_read_write_test(); }

TEST(local_fs_test, local_append_test) { append_test(); }

}  // namespace