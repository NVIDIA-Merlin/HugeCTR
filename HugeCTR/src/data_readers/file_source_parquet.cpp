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

#include <data_readers/file_source_parquet.hpp>

namespace HugeCTR {
using namespace cudf;
namespace cudf_io = cudf::io;

ParquetFileSource::ParquetFileSource(unsigned int worker_id, unsigned int stride,
                                     const std::string& file_list, bool sequtial_file_consumption,
                                     bool repeat, const DataSourceParams& data_source_params)
    : file_list_(file_list),
      offset_(worker_id * !(sequtial_file_consumption)),
      worker_id_(worker_id),
      stride_(stride),
      row_group_offset_(0),
      can_read_file_(false),
      file_metadata_(),
      cached_row_group_table_(),
      curr_row_group_(0),
      num_row_groups_(0),
      repeat_(repeat),
      sequential_file_consumption_(sequtial_file_consumption) {
  slice_stream_ = NULL;
  file_loader_ = std::make_unique<FileLoader>(data_source_params);
  // load _metadata.json
  std::string metadata_file_name = get_metada_filename(file_list_.get_a_file_with_id(0, true));
  if (!(file_metadata_.get_metadata_status())) {
    file_metadata_.get_parquet_metadata(metadata_file_name);
    rows_file_offset_ = std::move(file_metadata_.get_rows_file_offset());
  }

  if (file_list_.get_num_of_files() < stride_ && !sequtial_file_consumption) {
    HCTR_LOG(WARNING, ROOT,
             "The number of data reader workers should be no greater than the "
             "number of files in "
             "concurrent mode \n");
  }
}

ParquetFileSource::~ParquetFileSource() {
  cudaStreamDestroy(slice_stream_);
  slice_stream_ = NULL;
  file_loader_->clean();
}

/**
 * Read "bytes_to_read" byte to the memory associated to ptr.
 * @param ptr pointer to user located buffer
 * @param bytes_to_read bytes to read
 * @return `FileCannotOpen` `OutOfBound` `Success` `UnspecificError`
 */
Error_t ParquetFileSource::read(char* ptr, size_t bytes_to_read) { return Error_t::IllegalCall; }
Error_t ParquetFileSource::find_next_file_and_group(long long expected_num_row_group) noexcept {
#ifdef ENABLE_ARROW_PARQUET
  while (expected_num_row_group > 0) {
    if (sequential_file_consumption_) {
      // counter_ % num_files = file_id
      file_name_ = file_list_.get_a_file_with_id(counter_, repeat_);
    } else {
      // (offset_ + counter_ * stride_) % num_files = file_id
      file_name_ = file_list_.get_a_file_with_id(offset_ + counter_ * stride_, repeat_);
    }
    counter_++;  // counter_ should be accum for every source.
    if (file_name_.empty()) {
      return Error_t::EndOfFile;
    }
    if (file_list_.get_file_type().compare("parquet") == 0) {
      HCTR_CHECK_HINT(file_metadata_.get_metadata_status(), "Please load _metadata.json first\n");
      auto fs = file_metadata_.get_file_stats(get_filename(file_name_));
      this->file_total_rows_ = (long long)(fs.num_rows);
      this->num_row_groups_ = fs.num_groups;
      expected_num_row_group -= fs.num_groups;
    } else {
      HCTR_LOG_S(ERROR, WORLD) << "Parquet files not found - check file extensions "
                               << HCTR_LOCATION() << std::endl;
      return Error_t::IllegalCall;
    }
  }
  curr_row_group_ = expected_num_row_group + this->num_row_groups_ - 1;
  std::vector<long long> row_groups_offset =
      file_metadata_.get_file_stats(get_filename(file_name_)).row_groups_offset;

  curr_row_idx_ = row_groups_offset[curr_row_group_];

  return Error_t::Success;
#else
  return Error_t::InvalidEnv;
#endif
}
/**
 * Start a new file to read. mmap() parquet to memory
 * @return `Success`, `FileCannotOpen` or `UnspecificError`
 */
// counter_ always points to next file name
Error_t ParquetFileSource::next_source(long long expected_num_row_group) noexcept {
  try {
    file_loader_->clean();
    can_read_file_ = false;
    auto res = this->find_next_file_and_group(expected_num_row_group);
    if (res == Error_t::EndOfFile) {
      return Error_t::EndOfFile;
    }
    if (res != Error_t::Success) {
      HCTR_OWN_THROW(res, "Library Dependency Error. Rebuild with Arrow::Parquet Library");
    }
    // check if file exists
    Error_t err = file_loader_->load(file_name_);
    if (err != Error_t::Success) {
      return err;
    }
    parquet_args_ = cudf_io::parquet_reader_options::builder(cudf_io::source_info{
        file_loader_->get_loaded_data(), file_loader_->get_current_file_size()});
    curr_row_idx_ = 0;  // set row to zero id
    row_group_index_ = 0;
    row_group_offset_ = 0;
    can_read_file_ = true;
    return Error_t::Success;
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    return Error_t::UnspecificError;
  }
}
bool ParquetFileSource::is_open() noexcept { return can_read_file_; }
cudf_io::table_with_metadata ParquetFileSource::read_group(
    size_t row_group_id, rmm::mr::device_memory_resource* mr) noexcept {
  nvtxRangePushA("load_DF");
  if (slice_stream_ == nullptr) {
    try {
      HCTR_LIB_THROW(cudaStreamCreate(&slice_stream_));
    } catch (const std::runtime_error& rt_err) {
      HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    }
  }
  cudf_io::table_with_metadata x;
  curr_row_idx_ = 0;
  curr_row_group_ = row_group_id;
  std::vector<cudf::size_type> row_list = {curr_row_group_};
  std::vector<std::vector<cudf::size_type>> rgrps = {row_list};
  parquet_args_.set_row_groups(rgrps);
  parquet_args_.set_skip_rows(0);
  parquet_args_.set_num_rows(-1);
  parquet_args_.set_timestamp_type(cudf::data_type(cudf::type_id::EMPTY));
  auto tbl_w_metadata = cudf_io::read_parquet(parquet_args_, mr);

  if (!counter_) {
    HCTR_OWN_THROW(Error_t::UnspecificError, "Read parquet file first\n");
  }
  if (sequential_file_consumption_) {
    // counter_ % num_files = file_id
    file_name_ = file_list_.get_a_file_with_id(counter_ - 1, repeat_);
  } else {
    // (offset_ + counter_ * stride_) % num_files = file_id
    file_name_ = file_list_.get_a_file_with_id(offset_ + (counter_ - 1) * stride_, repeat_);
  }
  // Must have Parquet library
  std::vector<long long> row_groups_offset =
      file_metadata_.get_file_stats(get_filename(file_name_)).row_groups_offset;
  // always points to next ready-to-read row
  curr_row_idx_ = row_groups_offset[row_group_id + 1];
  curr_row_group_++;

  nvtxRangePop();

  return tbl_w_metadata;
}

const Metadata& ParquetFileSource::get_file_metadata() { return file_metadata_; }
int ParquetFileSource::get_cur_file_id() {
  // counter_ always points to the next file to be read
  int num_files = file_list_.get_num_of_files();
  if (sequential_file_consumption_) {
    return (counter_ - 1) % num_files;
  } else {
    return offset_ + (counter_ - 1) * stride_ % num_files;
  }
}
bool ParquetFileSource::reach_eof() { return curr_row_idx_ >= file_total_rows_; }
bool ParquetFileSource::row_group_eof() {
#ifdef ENABLE_ARROW_PARQUET
  return curr_row_group_ >= num_row_groups_;
#else
  HCTR_LOG(ERROR, WORLD, "row_group_eof() not supported! \n");
  return false;
#endif
}

int ParquetFileSource::get_row_group() { return curr_row_group_; }
int ParquetFileSource::get_num_row_groups() { return num_row_groups_; }

long long ParquetFileSource::get_num_rows() { return file_total_rows_; }
long long ParquetFileSource::get_offset_to_read_within_file() { return curr_row_idx_; }
long long ParquetFileSource::get_row_group_to_read() { return curr_row_group_; }
long long ParquetFileSource::get_offset_to_read_within_group() { return row_group_index_; }
long long ParquetFileSource::get_group_offset_to_read_within_file() { return row_group_offset_; }
std::string ParquetFileSource::get_cur_file() { return file_name_; };

}  // namespace HugeCTR
