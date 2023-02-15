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

#include <io/file_loader.hpp>
#include <utest/test_utils.hpp>

using namespace HugeCTR;

void file_loader_hdfs_test_impl(const std::string hdfs_dir, const int num_files) {
  const DataSourceParams data_source_params(FileSystemType_t::HDFS, "localhost", 9000);

  FileLoader *file_loader = new FileLoader(data_source_params);
  std::vector<std::string> filelist;
  for (int i = 0; i < num_files; i++) {
    filelist.push_back(hdfs_dir + "/gen_" + std::to_string(i) + ".parquet");
  }
  for (std::string file : filelist) {
    file_loader->clean();
    Error_t err = file_loader->load(file);
    EXPECT_EQ(err, Error_t::Success);
  }
}

TEST(file_loader_test, file_loder_test_small) {
  file_loader_hdfs_test_impl("/dlrm_parquet_test_small/", 40);
}

TEST(file_loader_test, file_loder_test_big) {
  file_loader_hdfs_test_impl("/dlrm_parquet_test_big/", 4);
}