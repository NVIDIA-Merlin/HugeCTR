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

#include <cuda_runtime_api.h>
#include <sys/stat.h>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/data_parser.hpp"

namespace HugeCTR {

/**
 * CPU Timer.
 */
class Timer {
 public:
  void start() {
    m_StartTime_ = std::chrono::steady_clock::now();
    m_bRunning_ = true;
  }
  void stop() {
    m_EndTime_ = std::chrono::steady_clock::now();
    m_bRunning_ = false;
  }
  double elapsedMilliseconds() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed()).count();
  }
  double elapsedMicroseconds() {
    return elapsed().count();
  }
  double elapsedSeconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(elapsed()).count();
  }

 private:
  std::chrono::microseconds elapsed() {
    std::chrono::time_point<std::chrono::steady_clock> endTime;
    if (m_bRunning_) {
      endTime = std::chrono::steady_clock::now();
    } else {
      endTime = m_EndTime_;
    }
    return std::chrono::duration_cast<std::chrono::microseconds>(endTime - m_StartTime_);
  }

  std::chrono::time_point<std::chrono::steady_clock> m_StartTime_{};
  std::chrono::time_point<std::chrono::steady_clock> m_EndTime_{};
  bool m_bRunning_ = false;
};

/**
 * Pop current cuda device and set new device.
 * @param i_device device ID to set
 * @param o_device device ID to pop, if o_device is NULL just set device to i_device.
 * @return the same as cudaError_t
 */
inline cudaError_t get_set_device(int i_device, int* o_device = NULL) {
  int current_dev_id = 0;
  cudaError_t err = cudaSuccess;

  if (o_device != NULL) {
    err = cudaGetDevice(&current_dev_id);
    if (err != cudaSuccess) return err;
    if (current_dev_id == i_device) {
      *o_device = i_device;
    } else {
      err = cudaSetDevice(i_device);
      if (err != cudaSuccess) {
        return err;
      }
      *o_device = current_dev_id;
    }
  } else {
    err = cudaSetDevice(i_device);
    if (err != cudaSuccess) {
      return err;
    }
  }

  return cudaSuccess;
}

/**
 * Get total product from dims.
 */
inline size_t get_size_from_dims(std::vector<int> dims) {
  size_t matrix_size = 1;
  for (auto iter = dims.begin(); iter != dims.end(); iter++) {
    matrix_size = matrix_size * iter[0];
  }
  return matrix_size;
}

/**
 * Check if file exist.
 */
inline bool file_exist(const std::string& name) {
  if (FILE* file = fopen(name.c_str(), "r")) {
    fclose(file);
    return true;
  } else {
    return false;
  }
}

/**
 * Check if file path exist if not create it.
 */
inline void check_make_dir(std::string finalpath) {
  if (mkdir(finalpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
    if (errno == EEXIST) {
      MESSAGE_(finalpath + " exist");
    } else {
      // something else
      ERROR_MESSAGE_("cannot create" + finalpath + ": unexpected error");
    }
  }
}

/**
 * Generate random dataset for HugeCTR test.
 */
template <typename T>
void data_generation(std::string file_list_name, std::string data_prefix, int num_files,
                     int num_records_per_file, int slot_num, int vocabulary_size, int label_dim,
                     int max_nnz) {
  if (file_exist(file_list_name)) {
    return;
  }
  std::string directory;
  const size_t last_slash_idx = data_prefix.rfind('/');
  if (std::string::npos != last_slash_idx) {
    directory = data_prefix.substr(0, last_slash_idx);
  }
  check_make_dir(directory);

  std::ofstream file_list_stream(file_list_name, std::ofstream::out);
  file_list_stream << (std::to_string(num_files) + "\n");
  for (int k = 0; k < num_files; k++) {
    std::string tmp_file_name(data_prefix + std::to_string(k) + ".data");
    file_list_stream << (tmp_file_name + "\n");

    // data generation;
    std::ofstream out_stream(tmp_file_name, std::ofstream::binary);
    DataSetHeader header = {num_records_per_file, label_dim, slot_num, 0};
    out_stream.write(reinterpret_cast<char*>(&header), sizeof(DataSetHeader));
    for (int i = 0; i < num_records_per_file; i++) {
      UnifiedDataSimulator<int> idata_sim(0, max_nnz - 1);  // both inclusive
      UnifiedDataSimulator<T> ldata_sim(0, vocabulary_size);
      for (int j = 0; j < label_dim; j++) {
        int label = idata_sim.get_num();
        out_stream.write(reinterpret_cast<char*>(&label), sizeof(int));
      }
      for (int k = 0; k < slot_num; k++) {
        int nnz = idata_sim.get_num();
        out_stream.write(reinterpret_cast<char*>(&nnz), sizeof(int));
        for (int j = 0; j < nnz; j++) {
          T value = ldata_sim.get_num();
          out_stream.write(reinterpret_cast<char*>(&value), sizeof(T));
        }
      }
    }
    out_stream.close();
  }
  file_list_stream.close();
  return;
}

/**
 * Find the item from a map according to str and pass by opt.
 * @return true: found / false: not found
 **/
template <typename ITEM_TYPE>
bool find_item_in_map(ITEM_TYPE* item, const std::string& str,
                      const std::map<std::string, ITEM_TYPE>& item_map) {
  typename std::map<std::string, ITEM_TYPE>::const_iterator it = item_map.find(str);
  if (it == item_map.end()) {
    return false;
  } else {
    *item = it->second;
    return true;
  }
}



}  // namespace HugeCTR
