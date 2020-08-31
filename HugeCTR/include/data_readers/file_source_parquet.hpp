/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <fstream>
#include <vector>
#include "common.hpp"
#include "data_readers/file_list.hpp"
#include "data_readers/source.hpp"
#include <cudf/io/functions.hpp>
#include <cudf/copying.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include "data_readers/metadata.hpp"
#include <nvToolsExt.h>
#include <fcntl.h>
#include <sys/io.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace HugeCTR {
using namespace cudf;
namespace cudf_io = cudf::io;
class ParquetFileSource : public Source {
 private:
  FileList file_list_;           /**< file list of data set */
  std::ifstream in_file_stream_; /**< file stream of data set file */
  const long long offset_;
  const long long stride_;
  unsigned int counter_{0};
  long long curr_row_idx_;
  long long file_total_rows_;   /**< Total rows in current file, read from Metadata */
  std::string file_name_;        /**< file name of current file */
  cudf_io::read_parquet_args parquet_args{cudf_io::source_info{""}};
  bool can_read_file_;      /**< Flag if parquet file is readable/reachable */
  Metadata file_metadata_;  /**< Metadata object for the file */
  std::unique_ptr<cudf_io::table_with_metadata> cached_row_group_table_;
  int curr_row_group_;
  int row_group_index_;
  int row_group_size_;
  cudaStream_t slice_stream_;  /**< worker stream for slicing row_group */
  size_t file_size_;    /**< Size of parquet file in bytes */
  char* mmapped_data_;  /**< Memory mapped file pointer */
  int fd_;  /**< File descriptor for mapped file */

 /**
  * Private Helper function to get metdata file address
  */
  std::string get_metada_filename(std::string path) {
    std::size_t found = path.find_last_of("/\\");
    std::string metadata_path = path.substr(0, found);
    metadata_path.append("/_metadata.json");
    return metadata_path;
  }

/**
  * Private Helper function to get parquet file name for metadata query
  */
  std::string get_filename(std::string path) {
    std::size_t found = path.find_last_of("/\\");
    std::string file_name = path.substr(found + 1);
    return file_name;
  }

 public:
 /**
   * Ctor
  */
  ParquetFileSource(unsigned int offset, unsigned int stride, const std::string& file_list)
      : file_list_(file_list), offset_(offset), stride_(stride), can_read_file_(false),
      file_metadata_(), cached_row_group_table_(), curr_row_group_(0), file_size_(0),
      mmapped_data_(nullptr), fd_(-1) {
        slice_stream_ = NULL;
      }

  ~ParquetFileSource() {
    cudaStreamDestroy(slice_stream_);
    slice_stream_ = NULL;
    if (fd_ != -1) {
      munmap(mmapped_data_, file_size_);
      close(fd_);
      fd_ = -1;
    }
  }

  /**
   * Read "bytes_to_read" byte to the memory associated to ptr.
   * @param ptr pointer to user located buffer
   * @param bytes_to_read bytes to read
   * @return `FileCannotOpen` `OutOfBound` `Success` `UnspecificError`
   */
  Error_t read(char* ptr, size_t bytes_to_read) noexcept {
    return Error_t::IllegalCall; 
  }

  /**
   * Start a new file to read.
   * @return `Success`, `FileCannotOpen` or `UnspecificError`
   */
  Error_t next_source() noexcept {
    try {
      if (fd_ != -1) {
        munmap(mmapped_data_, file_size_);
        close(fd_);
        fd_ = -1;
        can_read_file_ = false;
      }
      file_name_ = file_list_.get_a_file_with_id(offset_ + counter_ * stride_);

      // check if file exists
      in_file_stream_.open(file_name_, std::ifstream::binary);
      if (!in_file_stream_.is_open()) {
        CK_RETURN_(Error_t::FileCannotOpen, "in_file_stream_.is_open() failed: " + file_name_);
      }
      in_file_stream_.seekg(0, std::ios::end);
      file_size_ = in_file_stream_.tellg();
      in_file_stream_.close();

      fd_ = open(file_name_.c_str(), O_RDONLY, 0);
      if (fd_ == -1) {
        CK_RETURN_(Error_t::BrokenFile, "Error open file for read");
      }

      mmapped_data_ = (char*)mmap(0, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
      if (mmapped_data_ == MAP_FAILED) {
        close(fd_);
        fd_ = -1;
        CK_RETURN_(Error_t::BrokenFile, "Error mmapping the file");
      }

      parquet_args.source = cudf_io::source_info{mmapped_data_, file_size_};
      counter_++;  // counter_ should be accum for every source.
      curr_row_idx_ = 0;  // set row to zero id
      file_total_rows_ = 0;
      curr_row_group_ = 0;
      row_group_index_ = 0;
      row_group_size_ = 0;

      can_read_file_ = true;

      if (file_list_.get_file_type().compare("parquet") == 0) {
        std::string metadata_file_name = get_metada_filename(file_name_);
        // single metadata json file, dont need to read again if init'd
        // if required - realloc Metadata obj then read
        if (!(file_metadata_.get_metadata_status()))
          file_metadata_.get_parquet_metadata(metadata_file_name);
        file_total_rows_ = (long long)(file_metadata_.get_file_stats(
                                                get_filename(file_name_)).num_rows );
      } else {
        CK_RETURN_(Error_t::IllegalCall, "Parquet files not found - check file extensions");
      }

      return Error_t::Success;
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      return Error_t::UnspecificError;
    }
  }

  bool is_open() noexcept { return can_read_file_; }

/**
 * Read "num_rows" from the memory-mapped parquet file
 * @param num_rows number of rows to read from Parquet file, -1 is single row_group
 * @param mr device memory resource for Parquet Reader
 * @return `cudf:table_with_metdata struct`
 */
  cudf_io::table_with_metadata read(long long num_rows,
                                        rmm::mr::device_memory_resource *mr) noexcept {
    nvtxRangePushA("load_DF");
    if (slice_stream_ == NULL) {
      try {
        CK_CUDA_THROW_(cudaStreamCreate(&slice_stream_));
      } catch (const std::runtime_error& rt_err) {
        std::cerr << rt_err.what() << std::endl;
      }
    }
    cudf_io::table_with_metadata x;
    bool use_cache = false;
    if (!use_cache) {

      if (num_rows == -1) {
        // read and inc row_group and send back
        std::vector<cudf::size_type> row_list = {curr_row_group_};
        std::vector<std::vector<cudf::size_type>> rgrps = {row_list};
        parquet_args.row_groups = rgrps;
        parquet_args.skip_rows = -1;
        parquet_args.num_rows = -1;
        parquet_args.timestamp_type = cudf::data_type(cudf::type_id::EMPTY);
        auto tbl_w_metadata = cudf_io::read_parquet(parquet_args, mr);
        curr_row_group_++;
        curr_row_idx_ += tbl_w_metadata.tbl->num_rows();
        nvtxRangePop();
        return tbl_w_metadata;
      } else {
        //parquet_args.row_group = -1; // set zero to use num_rows and skip_rows param
        std::vector<std::vector<cudf::size_type>> rgrps;
        parquet_args.row_groups = rgrps;
        parquet_args.skip_rows = curr_row_idx_;
        parquet_args.num_rows = num_rows;
        parquet_args.timestamp_type = cudf::data_type(cudf::type_id::EMPTY);
        auto tbl_w_metadata = cudf_io::read_parquet(parquet_args, mr);

        curr_row_idx_ += num_rows;
        nvtxRangePop();
        return tbl_w_metadata;
      }
    }
    else {
      cudf_io::table_with_metadata slice;
      int rows_to_read = (int)num_rows;

      while(rows_to_read > 0) {
        if (row_group_index_ == row_group_size_) {
          // first run will be reset as both are zero
          cached_row_group_table_.reset();
        }
        if (cached_row_group_table_.get() == nullptr) {
          std::vector<cudf::size_type> row_list = {curr_row_group_};
          std::vector<std::vector<cudf::size_type>> rgrps = {row_list};
          parquet_args.row_groups = rgrps; // set zero to use num_rows and skip_rows param
          //parquet_args.row_group_count = 1; // set zero to use num_rows and skip_rows param
          parquet_args.skip_rows = -1;
          parquet_args.num_rows = -1;
          parquet_args.timestamp_type = cudf::data_type(cudf::type_id::EMPTY);
          curr_row_group_++;
          //auto tbl_w_metadata = cudf_io::read_parquet(parquet_args, mr);
          cached_row_group_table_ = std::make_unique<cudf_io::table_with_metadata>(
                                                        cudf_io::read_parquet(parquet_args, mr));
          row_group_size_ = cached_row_group_table_.get()->tbl->num_rows();
          row_group_index_ = 0;
        }

        std::vector<cudf::table_view> tbl_view_vector;
        // send DF with right start-end idx
        if ((row_group_size_ - row_group_index_) >  rows_to_read) {
          std::vector<cudf::size_type> slice_indices{
                                              (cudf::size_type)row_group_index_,
                                              (cudf::size_type)(row_group_index_ + rows_to_read) };
          tbl_view_vector = cudf::slice(cached_row_group_table_.get()->tbl->view(), slice_indices);
          curr_row_idx_ += rows_to_read;
          row_group_index_ += rows_to_read;
          rows_to_read = 0;
        } else {
          int curr_rows_to_read = row_group_size_ - row_group_index_; // read remaining DF rows
          std::vector<cudf::size_type> slice_indices{
                                          (cudf::size_type)row_group_index_,
                                          (cudf::size_type)(row_group_index_ + curr_rows_to_read)};
          tbl_view_vector = cudf::slice(cached_row_group_table_.get()->tbl->view(), slice_indices);
          curr_row_idx_ += curr_rows_to_read;
          row_group_index_ += curr_rows_to_read;
          rows_to_read -= curr_rows_to_read;
        }

        assert(tbl_view_vector.size() == 1);

        // cant release row_group before new table_view copy is constructed to send back
        if (slice.tbl.get() == nullptr) {
          // copy is made here
          slice.tbl = std::make_unique<cudf::table>(tbl_view_vector[0], slice_stream_, mr);
          slice.metadata = cached_row_group_table_.get()->metadata;
          // blocking bcoz parquet_read is blocking - mimic same behavior
          cudaStreamSynchronize(slice_stream_);
        }
        else {
          // concat table/table_views
          std::vector<cudf::table_view> table_view_for_concat{slice.tbl->view(),
                                                                tbl_view_vector[0]};
          (cudf::concatenate(table_view_for_concat, mr)).swap(slice.tbl);
        }
      }
      nvtxRangePop();
      return slice;
    }
    nvtxRangePop();
    return x;
  }

  const Metadata& get_file_metadata() { return file_metadata_; }
  long long get_num_rows() { return file_total_rows_; }
};

}  // namespace HugeCTR
