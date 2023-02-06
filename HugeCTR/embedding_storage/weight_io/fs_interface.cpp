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

#include <embedding_storage/weight_io/fs_interface.hpp>

using namespace HugeCTR;
namespace embedding {

#ifdef ENABLE_MPI
EmbeddingWeightIOMpi::EmbeddingWeightIOMpi(const std::string& file_name) {}

void EmbeddingWeightIOMpi::write_to(const std::string& path, const void* write_buffer,
                                    size_t start_offset, size_t write_size, bool overwrite) {
  HCTR_MPI_THROW(MPI_Barrier(MPI_COMM_WORLD));
  MPI_File file_handle;
  if (!overwrite) {
    MPI_File_open(MPI_COMM_WORLD, path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_APPEND,
                  MPI_INFO_NULL, &file_handle);
    MPI_Offset new_offset;
    MPI_File_get_position(file_handle, &new_offset);
    start_offset += new_offset;
  } else {
    MPI_File_open(MPI_COMM_WORLD, path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                  &file_handle);
  }

  MPI_Status status;
  HCTR_MPI_THROW(
      MPI_File_write_at(file_handle, start_offset, write_buffer, write_size, MPI_CHAR, &status));
  MPI_File_close(&file_handle);
}

void EmbeddingWeightIOMpi::read_from(const std::string& path, void* read_buffer, size_t read_size,
                                     size_t start_offset) {
  // TODO::don't think it will use for now.
  HCTR_MPI_THROW(MPI_Barrier(MPI_COMM_WORLD));
  MPI_File file_handle;
  MPI_File_open(MPI_COMM_WORLD, path.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file_handle);
  MPI_Status status;
  HCTR_MPI_THROW(
      MPI_File_read_at(file_handle, start_offset, read_buffer, read_size, MPI_CHAR, &status));
  MPI_File_close(&file_handle);
}

void EmbeddingWeightIOMpi::make_dir(const std::string& path) {
  if (!std::filesystem::exists(path)) {
    std::filesystem::create_directories(path);
  }
}

void EmbeddingWeightIOMpi::delete_dir(const std::string& path) {
  if (std::filesystem::exists(path)) {
    std::filesystem::remove_all(path);
  }
}

size_t EmbeddingWeightIOMpi::get_file_size(const std::string& path) {
  std::ifstream file_stream(path);
  HCTR_CHECK_HINT(file_stream.is_open(), std::string("File not open: " + path).c_str());
  file_stream.close();
  return std::filesystem::file_size(path);
}

#endif

EmbeddingWeightIOFS::EmbeddingWeightIOFS(const std::string& file_name) {
  hs_ = HugeCTR::FileSystemBuilder::build_unique_by_path(file_name);
}

void EmbeddingWeightIOFS::write_to(const std::string& path, const void* write_buffer,
                                   size_t start_offset, size_t write_size, bool overwrite) {
  if (write_size > 0) {
    hs_->write(path, write_buffer, write_size, overwrite);
  }
}

void EmbeddingWeightIOFS::read_from(const std::string& path, void* read_buffer, size_t read_size,
                                    size_t start_offset) {
  if (read_size > 0) {
    hs_->read(path, read_buffer, read_size, start_offset);
  }
}

void EmbeddingWeightIOFS::make_dir(const std::string& path) {
  if (!std::filesystem::exists(path)) {
    hs_->create_dir(path);
  }
}

void EmbeddingWeightIOFS::delete_dir(const std::string& path) {
  if (std::filesystem::exists(path)) {
    hs_->delete_file(path);
  }
}

size_t EmbeddingWeightIOFS::get_file_size(const std::string& path) {
  return hs_->get_file_size(path);
}

}  // namespace embedding
