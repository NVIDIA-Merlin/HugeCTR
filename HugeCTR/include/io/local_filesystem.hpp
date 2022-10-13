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

#include <io/filesystem.hpp>

namespace HugeCTR {
class LocalFileSystem final : public FileSystem {
 public:
  LocalFileSystem();

  virtual ~LocalFileSystem();

  size_t get_file_size(const std::string& path) const override;

  void create_dir(const std::string& path) override;

  void delete_file(const std::string& path, bool recursive) override;

  void fetch(const std::string& source_path, const std::string& target_path) override;

  void upload(const std::string& source_path, const std::string& target_path) override;

  int write(const std::string& path, const void* data, size_t data_size, bool overwrite) override;

  int read(const std::string& path, void* buffer, size_t buffer_size, size_t offset) override;

  void copy(const std::string& source_file, const std::string& target_file) override;

  int batch_fetch(const std::string& source_dir, const std::string& target_dir) override;

  int batch_upload(const std::string& source_dir, const std::string& target_dir) override;
};
}  // namespace HugeCTR