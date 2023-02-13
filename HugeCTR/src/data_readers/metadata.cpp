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

#include <data_readers/metadata.hpp>
// arrow::libparquet
#ifdef ENABLE_ARROW_PARQUET
#include <parquet/file_reader.h>
#include <parquet/metadata.h>
#endif

#include <iostream>
#include <nlohmann/json.hpp>

namespace HugeCTR {
void Metadata::reset_metadata(std::string file_name) {
  this->loaded_ = false;
  cat_names_.clear();
  cont_names_.clear();
  label_names_.clear();
  file_stats_.clear();
  num_rows_total_files_ = 0;
  rows_file_offset_.clear();
  get_parquet_metadata(file_name);
}
// filename is xxxx/_metadata.json
void Metadata::get_parquet_metadata(std::string file_name) {
  if (this->loaded_) {
    return;
  }
  std::size_t found_json_dir = file_name.find_last_of("/\\");
  std::string dirname = file_name.substr(0, found_json_dir + 1);
  nlohmann::json config;
  std::ifstream file_stream(file_name);
  if (!file_stream.is_open()) {
    HCTR_OWN_THROW(Error_t::FileCannotOpen, "file_stream.is_open() failed: " + file_name);
  }
  // TODO add specific exception check?
  try {
    file_stream >> config;
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
  }
  try {
    auto fstats = config.find("file_stats").value();
    num_rows_total_files_ = 0;
    rows_file_offset_.clear();
#ifdef ENABLE_ARROW_PARQUET
    long long global_num_row_groups = 0;
#endif

    for (unsigned int i = 0; i < fstats.size(); i++) {
      // maybe relative path, fname::parquet
      std::string fname = (std::string)fstats[i].find("file_name").value();
      std::size_t found = fname.find_last_of("/\\");
      std::string parquet_name = fname.substr(found + 1);
      std::string parquet_path = dirname + "/" + parquet_name;
      if (found != std::string::npos) {
        fname = fname.substr(found + 1);
      }
#ifdef ENABLE_ARROW_PARQUET
      long long max_row_group_file_ = 0;
      std::ifstream fexist(parquet_path);
      if (!fexist.good()) {
        continue;
      }
      long long group_offset = 0;
      std::vector<long long> row_groups_offset{0};
      std::unique_ptr<parquet::ParquetFileReader> reader =
          parquet::ParquetFileReader::OpenFile(parquet_path, false);
      const parquet::FileMetaData* file_metadata = reader->metadata().get();
      long long num_row_groups = file_metadata->num_row_groups();
      global_num_row_groups += num_row_groups;
      long long num_rows_file = file_metadata->num_rows();
      for (long long r = 0; r < num_row_groups; r++) {
        std::unique_ptr<parquet::RowGroupMetaData> group_metadata = file_metadata->RowGroup(r);
        group_offset += group_metadata->num_rows();
        if (max_row_group_file_ < group_metadata->num_rows()) {
          max_row_group_file_ = group_metadata->num_rows();
        }
        row_groups_offset.push_back(group_offset);
      }

      HCTR_CHECK_HINT(num_rows_file == long(fstats[i].find("num_rows").value()),
                      "Parquet file number of rows mismatch with _metadata.json\n");
      FileStats fs(num_rows_file, num_row_groups, row_groups_offset, max_row_group_file_);
      if (max_row_group_ < max_row_group_file_) {
        max_row_group_ = max_row_group_file_;
      }
#else
      long long num_rows_file = long(fstats[i].find("num_rows").value());
      FileStats fs(num_rows_file);
#endif
      file_stats_.insert({fname, fs});
      rows_file_offset_.push_back(num_rows_total_files_);
      num_rows_total_files_ += num_rows_file;
    }
    rows_file_offset_.push_back(num_rows_total_files_);
    auto cats = config.find("cats").value();
    for (unsigned int i = 0; i < cats.size(); i++) {
      Cols c;
      c.col_name = (std::string)cats[i].find("col_name").value();
      c.index = (int)cats[i].find("index").value();
      this->cat_names_.push_back(c);
    }
    auto conts = config.find("conts").value();
    for (unsigned int i = 0; i < conts.size(); i++) {
      Cols c;
      c.col_name = (std::string)conts[i].find("col_name").value();
      c.index = (int)conts[i].find("index").value();
      this->cont_names_.push_back(c);
    }
    auto labels = config.find("labels").value();
    for (unsigned int i = 0; i < labels.size(); i++) {
      Cols c;
      c.col_name = (std::string)labels[i].find("col_name").value();
      c.index = (int)labels[i].find("index").value();
      this->label_names_.push_back(c);
    }
    loaded_ = true;
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
  }

  file_stream.close();
}

}  // namespace HugeCTR
