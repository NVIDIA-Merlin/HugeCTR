#pragma once

#include <embedding_storage/weight_io/data_info.hpp>
#include <filesystem>
#include <io/filesystem.hpp>
#include <string>

#ifdef ENABLE_MPI
#include "mpi.h"
#endif

namespace embedding {

class EmbeddingWeightIO {
 public:
  virtual void write_to(const std::string& path, const void* write_buffer, size_t start_offset,
                        size_t write_size, bool overwrite = true) = 0;
  virtual void read_from(const std::string& path, void* read_buffer, size_t read_size,
                         size_t start_offset) = 0;
  virtual void make_dir(const std::string& path) = 0;
  virtual void delete_dir(const std::string& path) = 0;
  virtual size_t get_file_size(const std::string& path) = 0;
};

#ifdef ENABLE_MPI
class EmbeddingWeightIOMpi : public EmbeddingWeightIO {
 public:
  EmbeddingWeightIOMpi(const std::string& file_name);
  virtual void write_to(const std::string& path, const void* write_buffer, size_t start_offset,
                        size_t write_size, bool overwrite = true) override;
  virtual void read_from(const std::string& path, void* read_buffer, size_t read_size,
                         size_t start_offset = 0) override;
  virtual void make_dir(const std::string& path) override;
  virtual void delete_dir(const std::string& path) override;
  virtual size_t get_file_size(const std::string& path) override;
};
#endif

class EmbeddingWeightIOFS : public EmbeddingWeightIO {
 public:
  EmbeddingWeightIOFS(const std::string& file_name);
  virtual void write_to(const std::string& path, const void* write_buffer, size_t start_offset,
                        size_t write_size, bool overwrite = true) override;
  virtual void read_from(const std::string& path, void* read_buffer, size_t read_size,
                         size_t start_offset = 0) override;
  virtual void make_dir(const std::string& path) override;
  virtual void delete_dir(const std::string& path) override;
  virtual size_t get_file_size(const std::string& path) override;

 private:
  std::unique_ptr<HugeCTR::FileSystem> hs_;
};

}  // namespace embedding
