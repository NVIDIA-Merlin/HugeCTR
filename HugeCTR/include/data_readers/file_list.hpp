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
#include <atomic>
#include <fstream>
#include <vector>
#include "data_readers/metadata.hpp"

namespace HugeCTR {

/**
 * @brief A threads safe file list implementation.
 *
 * FileList reads file list from text file, and maintains a vector of file name. It supports
 * getting file names with multiple threads. All the threads will get the names in order.
 * Text file begins with the number of files, and then the list of file names.
 * @verbatim
 * Text file example:
 * 3
 * 1.txt
 * 2.txt
 * 3.txt
 * @endverbatim
 */
class FileList {
 private:
  int num_of_files_;                     /**< num of files read from text file. */
  std::vector<std::string> file_vector_; /**< the vector of file names. */
  std::atomic<unsigned int> counter_{0};
  std::string file_type_;

  std::string get_file_type(std::string file_name) {
    std::string type = "None";
    std::size_t found = file_name.find_last_of(".");
    type = file_name.substr(found + 1);

    return type;
  }

 public:
  /*
   * Ctor
   */
  FileList(const std::string& file_list_name) {
    try {
      std::ifstream read_stream(file_list_name, std::ifstream::in);
      if (!read_stream.is_open()) {
        CK_THROW_(Error_t::FileCannotOpen, "file list open failed: " + file_list_name);
      }

      std::string buff;
      std::getline(read_stream, buff);
      num_of_files_ = std::stoi(buff);
      assert(file_vector_.empty());
      if (num_of_files_ > 0) {
        for (int i = 0; i < num_of_files_; i++) {
          std::getline(read_stream, buff);
          if (i == 0) {
            file_type_ = get_file_type(buff);
          }
          file_vector_.push_back(buff);
        }
        read_stream.close();
      } else {
        CK_THROW_(Error_t::UnSupportedFormat,
                  "The file format is invalid or there is no data file in the file list.");
      }
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
  }

  /**
   * Get a file name from the list.
   * @return the file name.
   */
  std::string get_a_file() {
    unsigned int counter = counter_;
    int current_file_idx = counter % num_of_files_;
    while (!counter_.compare_exchange_weak(counter, counter + 1))
      ;

    return file_vector_[current_file_idx];
  }

  /**
   * Get a file name from the list.
   * @return the file name and id.
   */
  std::string get_a_file_with_id(unsigned int id, bool repeat) {
    if (repeat) {
      int current_file_idx = id % num_of_files_;
      return file_vector_[current_file_idx];
    }
    else {
      if (static_cast<int>(id) < num_of_files_) {
        return file_vector_[id];
      } else {
        return std::string();
      }
    }
  }

  std::string get_file_type() { return file_type_; }
};

}  // namespace HugeCTR
