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

 public:
  /**
   * Ctor
   */
  ParquetFileSource(unsigned int worker_id, unsigned int stride, const std::string& file_list,
                    bool sequtial_file_consumption, bool repeat,
                    const DataSourceParams& data_source_params)
      : file_list_(file_list),
        repeat_(repeat),
        sequential_file_consumption_(sequtial_file_consumption),
        worker_id_(worker_id),
        stride_(stride),
        can_read_file_(false),
        file_metadata_(),
        cached_row_group_table_(),
        curr_row_group_(0),
        row_group_offset_(0),
        offset_(worker_id * !(sequtial_file_consumption)) {
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
               "The number of data reader workers should be no greater than the number of files in "
               "concurrent mode \n");
    }
    // HCTR_CHECK_HINT(
    //     file_list_.get_num_of_files() >= stride_,
    //     "The number of data reader workers should be no greater than the number of files in the "
    //     "file list. There is one worker on each GPU for Parquet dataset, please re-configure
    //     vvgpu " "within CreateSolver or guarantee enough files in the file list.");
  }

  ~ParquetFileSource() {
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
  Error_t read(char* ptr, size_t bytes_to_read) noexcept { return Error_t::IllegalCall; }

  /**
   * Start a new file to read. mmap() parquet to memory
   * @return `Success`, `FileCannotOpen` or `UnspecificError`
   */
  // counter_ always points to next file name
  Error_t next_source() noexcept {
    try {
      file_loader_->clean();
      can_read_file_ = false;
      if (sequential_file_consumption_) {
        // counter_ % num_files = file_id
        file_name_ = file_list_.get_a_file_with_id(counter_, repeat_);
      } else {
        // (offset_ + counter_ * stride_) % num_files = file_id
        file_name_ = file_list_.get_a_file_with_id(offset_ + counter_ * stride_, repeat_);
      }
      counter_++;  // counter_ should be accum for every source.
      // check if file exists
      if (file_name_.empty()) {
        return Error_t::EndOfFile;
      }

      Error_t err = file_loader_->load(file_name_);
      if (err != Error_t::Success) {
        return err;
      }

      parquet_args_ = cudf_io::parquet_reader_options::builder(cudf_io::source_info{
          file_loader_->get_loaded_data(), file_loader_->get_current_file_size()});
      curr_row_idx_ = 0;  // set row to zero id
      file_total_rows_ = 0;
      curr_row_group_ = 0;
      row_group_index_ = 0;
      row_group_size_ = 0;
      row_group_offset_ = 0;
      can_read_file_ = true;

      if (file_list_.get_file_type().compare("parquet") == 0) {
        HCTR_CHECK_HINT(file_metadata_.get_metadata_status(), "Please load _metadata.json first\n");
        file_total_rows_ =
            (long long)(file_metadata_.get_file_stats(get_filename(file_name_)).num_rows);
        // HCTR_LOG_S(INFO, WORLD) << "file_name_ " << file_name_ << " file_total_rows_ "
        //                          << file_total_rows_ << std::endl;
      } else {
        HCTR_LOG_S(ERROR, WORLD) << "Parquet files not found - check file extensions "
                                 << HCTR_LOCATION() << std::endl;
        return Error_t::IllegalCall;
      }

      return Error_t::Success;
    } catch (const std::runtime_error& rt_err) {
      HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
      return Error_t::UnspecificError;
    }
  }
  bool is_open() noexcept { return can_read_file_; }
  void set_row_idx(long long idx) { curr_row_idx_ = idx; }
  /**
   * Read "num_rows" from the memory-mapped parquet file
   * @param num_rows number of rows to read from Parquet file, -1 is single row_group
   * @param mr device memory resource for Parquet Reader
   * @return `cudf:table_with_metdata struct`
   */
  cudf_io::table_with_metadata read(long long num_rows,
                                    rmm::mr::device_memory_resource* mr) noexcept {
    nvtxRangePushA("load_DF");
    if (slice_stream_ == nullptr) {
      try {
        HCTR_LIB_THROW(cudaStreamCreate(&slice_stream_));
      } catch (const std::runtime_error& rt_err) {
        HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
      }
    }
    cudf_io::table_with_metadata x;
    bool use_cache = false;
    if (!use_cache) {
      if (num_rows == -1) {
        // read and inc row_group and send back
        std::vector<cudf::size_type> row_list = {curr_row_group_};
        std::vector<std::vector<cudf::size_type>> rgrps = {row_list};
        parquet_args_.set_row_groups(rgrps);
        parquet_args_.set_skip_rows(0);
        parquet_args_.set_num_rows(-1);
        parquet_args_.set_timestamp_type(cudf::data_type(cudf::type_id::EMPTY));
        auto tbl_w_metadata = cudf_io::read_parquet(parquet_args_, mr);
        curr_row_group_++;
        curr_row_idx_ += tbl_w_metadata.tbl->num_rows();
        row_group_offset_ += tbl_w_metadata.tbl->num_rows();
        nvtxRangePop();
        return tbl_w_metadata;
      } else {
        // parquet_args_.row_group = -1; // set zero to use num_rows and skip_rows param
        std::vector<std::vector<cudf::size_type>> rgrps;
        parquet_args_.set_row_groups(rgrps);
        parquet_args_.set_skip_rows(curr_row_idx_);
        parquet_args_.set_num_rows(num_rows);
        parquet_args_.set_timestamp_type(cudf::data_type(cudf::type_id::EMPTY));
        auto tbl_w_metadata = cudf_io::read_parquet(parquet_args_, mr);

        curr_row_idx_ += num_rows;
        nvtxRangePop();
        return tbl_w_metadata;
      }
    } else {
      cudf_io::table_with_metadata slice;
      int rows_to_read = (int)num_rows;

      while (rows_to_read > 0) {
        if (row_group_index_ == row_group_size_) {
          // first run will be reset as both are zero
          cached_row_group_table_.reset();
        }
        if (cached_row_group_table_.get() == nullptr) {
          std::vector<cudf::size_type> row_list = {curr_row_group_};
          std::vector<std::vector<cudf::size_type>> rgrps = {row_list};
          parquet_args_.set_row_groups(rgrps);  // set zero to use num_rows and skip_rows param
          // parquet_args_.row_group_count = 1; // set zero to use num_rows and skip_rows param
          parquet_args_.set_skip_rows(0);
          parquet_args_.set_num_rows(-1);
          parquet_args_.set_timestamp_type(cudf::data_type(cudf::type_id::EMPTY));
          curr_row_group_++;
          cached_row_group_table_ = std::make_unique<cudf_io::table_with_metadata>(
              cudf_io::read_parquet(parquet_args_, mr));
          row_group_size_ = cached_row_group_table_.get()->tbl->num_rows();
          row_group_index_ = 0;
        }

        std::vector<cudf::table_view> tbl_view_vector;
        // send DF with right start-end idx
        if ((row_group_size_ - row_group_index_) > rows_to_read) {
          std::vector<cudf::size_type> slice_indices{
              (cudf::size_type)row_group_index_,
              (cudf::size_type)(row_group_index_ + rows_to_read)};
          tbl_view_vector = cudf::slice(cached_row_group_table_.get()->tbl->view(), slice_indices);
          curr_row_idx_ += rows_to_read;
          row_group_index_ += rows_to_read;
          rows_to_read = 0;
        } else {
          int curr_rows_to_read = row_group_size_ - row_group_index_;  // read remaining DF rows
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
        } else {
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

  /*
    jump to specific parquet file caclulated through global_record_id;
    if dst == cur, dont need to load again
  */
  Error_t seek_by_records(long long global_record_id, long long batchsize) {
#ifdef ENABLE_ARROW_PARQUET

    HCTR_CHECK_HINT(global_record_id >= 0,
                    "Parquet file source: seek_by_records() requires global_record_id > 0");

    std::string file_name("");
    int num_files = file_list_.get_num_of_files();
    long long num_rows_total_files = file_metadata_.get_num_rows_total_files();
    if (!repeat_ && global_record_id >= num_rows_total_files) {
      return Error_t::EndOfFile;
    }
    if (sequential_file_consumption_) {
      long long round = global_record_id / num_rows_total_files;
      long long file_offset_round = global_record_id % num_rows_total_files;
      // which file input record lies in ?
      auto upper =
          std::upper_bound(rows_file_offset_.begin(), rows_file_offset_.end(), file_offset_round);

      if (upper == rows_file_offset_.end()) {
        HCTR_OWN_THROW(Error_t::UnspecificError, "Parquet file source: global_record_id overflow");
      }
      upper = upper - 1;
      long long calculated_file_id = std::distance(rows_file_offset_.begin(), upper);

      std::string calculated_file_name = file_list_.get_a_file_with_id(calculated_file_id, repeat_);
      // get row_group_size of cur file
      std::vector<long long> row_groups_offset =
          file_metadata_.get_file_stats(get_filename(calculated_file_name)).row_groups_offset;
      // binary search... find which row_group lies in
      auto upper_row_group =
          std::upper_bound(row_groups_offset.begin(), row_groups_offset.end(), curr_row_idx_);
      upper_row_group -= 1;

      // get file where current record lies in
      // in case current file is dst file to seek
      // if ((calculated_file_id + 1) % num_files != (counter_ % num_files)) {
      if (calculated_file_id % num_files != (counter_ - 1) % num_files) {
        // HCTR_LOG(INFO, ROOT,
        //        "calculated_file_id reset !cal:act %d: %d\n",calculated_file_id,counter_ %
        //        num_files - 1);
        // next_soure will read file[counter_]
        counter_ = static_cast<unsigned int>(round * num_files + calculated_file_id);
        // will reset current row_group
        this->next_source();
      } else {
        counter_ = static_cast<unsigned int>(round * num_files + calculated_file_id + 1);
      }
      curr_row_idx_ = file_offset_round - *(upper);
      curr_row_group_ = std::distance(row_groups_offset.begin(), (upper_row_group));
      row_group_index_ = curr_row_idx_ - *upper_row_group;
      row_group_offset_ = *upper_row_group;
      // HCTR_LOG(INFO, ROOT,"file_offset_round is %d,upper is %d\n",file_offset_round,*(upper));

      HCTR_LOG(INFO, ROOT,
               "workerid %d global_record_id %d round is %ld reset calculated_file_id is %ld, "
               "counter_ is "
               "%d, curr_row_idx_ is %ld, row_group_id is %ld, row_group_index_ is %ld\n",
               worker_id_, global_record_id, round, calculated_file_id, counter_, curr_row_idx_,
               curr_row_group_, row_group_index_);

    } else {
      long long global_batch_id = global_record_id / batchsize;
      long long batchmod = global_record_id % batchsize;

      long long dst_worker_id = global_batch_id % stride_;
      long long local_batch_id = global_batch_id / stride_;
      long long local_rows = local_batch_id * batchsize;
      HCTR_CHECK_HINT(batchmod == 0,
                      "Parquet file source: seek_by_records() requires global_record_id aligned "
                      "with batchsize");

      HCTR_CHECK_HINT(dst_worker_id == worker_id_,
                      "Parquet file source: cannot seek parquet rows to other workers' files");
      long long cnt = offset_;
      long long record_iter = 0;
      long long file_id_iter = cnt % num_files;
      // linear lookup... find which file record lies in
      while (record_iter <= local_rows) {
        file_id_iter = cnt % num_files;
        cnt += stride_;
        record_iter += rows_file_offset_[file_id_iter + 1] - rows_file_offset_[file_id_iter];
      }
      record_iter -= rows_file_offset_[file_id_iter + 1] - rows_file_offset_[file_id_iter];
      cnt -= stride_;
      long long calculated_file_id = cnt % num_files;
      std::string calculated_file_name = file_list_.get_a_file_with_id(calculated_file_id, repeat_);
      // get row_group_size of cur file
      std::vector<long long> row_groups_offset =
          file_metadata_.get_file_stats(get_filename(calculated_file_name)).row_groups_offset;
      // binary search... find which row_group lies in

      auto upper = std::upper_bound(row_groups_offset.begin(), row_groups_offset.end(),
                                    local_rows - record_iter);
      upper -= 1;
      // if(local_rows - record_iter != *upper){
      //   HCTR_LOG(ERROR, ROOT,
      //          "Parquet file source:In concurrent file loading mode, seek_by_records must align
      //          with row_group\n");
      //   return Error_t::WrongInput;
      // }
      curr_row_idx_ = local_rows - record_iter;
      curr_row_group_ = std::distance(row_groups_offset.begin(), (upper));
      row_group_index_ = curr_row_idx_ - *upper;
      row_group_offset_ = *upper;
      // HCTR_LOG(INFO, ROOT,
      //          "calculated_file_id %ld cur_row_group_ %ld row_group_index_ %ld row_group_offset_
      //          %ld\n",calculated_file_id,curr_row_group_,row_group_index_,row_group_offset_);
      // in case current file is dst file to seek
      if (calculated_file_id != (offset_ + (counter_ - 1) * stride_) % num_files) {
        counter_ = cnt;
        // HCTR_LOG(INFO, ROOT,
        //        "calculated_file_id cal:act %d: %d \n",calculated_file_id,(offset_ + (counter_ -
        //        1)* stride_) % num_files);
        // next_soure will inc counter_;
        this->next_source();
      } else {
        counter_ = cnt + 1;
      }

      HCTR_LOG(INFO, ROOT,
               "workerid %d global_record_id %d reset calculated_file_id is %ld, counter_ is "
               "%d, curr_row_idx_ is %ld, row_group_id is %ld, row_group_index_ is %ld\n",
               worker_id_, global_record_id, calculated_file_id, counter_, curr_row_idx_,
               curr_row_group_, row_group_index_);
    }
    return Error_t::Success;

#else
    return Error_t::InvalidEnv;
#endif
  }
  const Metadata& get_file_metadata() { return file_metadata_; }
  int get_cur_file_id() {
    // counter_ always points to the next file to be read
    int num_files = file_list_.get_num_of_files();
    if (sequential_file_consumption_) {
      return (counter_ - 1) % num_files;
    } else {
      return offset_ + (counter_ - 1) * stride_ % num_files;
    }
  }
  long long get_num_rows() { return file_total_rows_; }
  long long get_offset_to_read_within_file() { return curr_row_idx_; }
  long long get_row_group_to_read() { return curr_row_group_; }
  long long get_offset_to_read_within_group() { return row_group_index_; }
  long long get_group_offset_to_read_within_file() { return row_group_offset_; }
};

}  // namespace HugeCTR
