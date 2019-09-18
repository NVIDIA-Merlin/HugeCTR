/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <mutex>
#include <vector>

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
  int current_file_idx_{0};              /**< the current index of file name getting */
  std::mutex mtx_;                       /**< mutex for threads safty */
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
          file_vector_.push_back(buff);
        }
        read_stream.close();
      } else {
        CK_THROW_(Error_t::UnSupportedFormat, "Unsupported file format");
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
    std::lock_guard<std::mutex> lock(mtx_);
    std::string file_name = file_vector_[current_file_idx_];
    current_file_idx_++;
    if (current_file_idx_ == num_of_files_) {
      current_file_idx_ = 0;
    }
    return file_name;
  }
};

}  // namespace HugeCTR
