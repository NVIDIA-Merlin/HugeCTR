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
#pragma once

#include <nvToolsExt.h>

#include <common.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <data_readers/file_list.hpp>
#include <data_readers/metadata.hpp>
#include <data_readers/source.hpp>
#include <fstream>
#include <io/file_loader.hpp>
#include <utils.hpp>
#include <vector>

namespace HugeCTR {

using namespace cudf;
namespace cudf_io = cudf::io;
class ParquetFileSource : public Source {
 private:
  FileList file_list_; /**< file list of data set */
  const long long offset_;
  const unsigned int worker_id_;
  const long long stride_;  // num_workers
  unsigned int counter_{0};
  long long curr_row_idx_;                  // current row offset within current file
  long long row_group_offset_;              // first row offset of current group
  long long file_total_rows_;               /**< Total rows in current file, read from Metadata */
  std::vector<long long> rows_file_offset_; /**< Total rows in all files, read from Metadata */
  std::string file_name_;                   /**< file name of current file */
  cudf_io::parquet_reader_options parquet_args_;
  bool can_read_file_;     /**< Flag if parquet file is readable/reachable */
  Metadata file_metadata_; /**< Metadata object for the file */
  std::unique_ptr<cudf_io::table_with_metadata> cached_row_group_table_;
  int curr_row_group_;                      // current row_group
  int num_row_groups_;                      // number of row_group
  int row_group_index_;                     // current row offset within current row_group
  int row_group_size_;                      // size of current row_group
  cudaStream_t slice_stream_;               /**< worker stream for slicing row_group */
  std::unique_ptr<FileLoader> file_loader_; /**< loader to load data from file system to memory */

  const bool repeat_;
  const bool sequential_file_consumption_;
  /**
   * Private Helper function to get metdata file address
   */
  std::string get_metada_filename(std::string path) {
    auto first_colon = path.find_first_of(":");
    if (first_colon != std::string::npos) {
      path.erase(0, first_colon + 1);
    }
    std::size_t found = path.find_last_of("/\\");
    std::string metadata_path = path.substr(0, found);
    metadata_path.append("/_metadata.json");
    return metadata_path;
  }

  /**
   * Private Helper function to get parquet file name for metadata query
   * return basename(<parquet_path>)
   */
  std::string get_filename(std::string path) {
    std::size_t found = path.find_last_of("/\\");
    std::string file_name = path.substr(found + 1);
    return file_name;
  }
  Error_t find_next_file_and_group(long long expected_num_row_group) noexcept;

 public:
  /**
   * Ctor
   */
  ParquetFileSource(unsigned int worker_id, unsigned int stride, const std::string& file_list,
                    bool sequtial_file_consumption, bool repeat,
                    const DataSourceParams& data_source_params);

  ~ParquetFileSource();
  /**
   * Read "bytes_to_read" byte to the memory associated to ptr.
   * @param ptr pointer to user located buffer
   * @param bytes_to_read bytes to read
   * @return `FileCannotOpen` `OutOfBound` `Success` `UnspecificError`
   */
  Error_t read(char* ptr, size_t bytes_to_read);

  /**
   * Start a new file to read. mmap() parquet to memory
   * @return `Success`, `FileCannotOpen` or `UnspecificError`
   */
  // counter_ always points to next file name
  Error_t next_source(long long expected_num_row_group) noexcept;
  bool is_open() noexcept;
  cudf_io::table_with_metadata read_group(size_t row_group_id,
                                          rmm::mr::device_memory_resource* mr) noexcept;
  /*
    jump to specific parquet file caclulated through global_record_id;
    if dst == cur, dont need to load again
  */
  const Metadata& get_file_metadata();
  int get_cur_file_id();
  int get_row_group();
  int get_num_row_groups();
  bool reach_eof();
  bool row_group_eof();
  long long get_num_rows();
  long long get_offset_to_read_within_file();
  long long get_row_group_to_read();
  long long get_offset_to_read_within_group();
  long long get_group_offset_to_read_within_file();
  std::string get_cur_file();
};

}  // namespace HugeCTR
