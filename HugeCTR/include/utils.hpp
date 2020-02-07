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
  double elapsedMilliseconds() { return elapsed().count() / 1000.0; }
  double elapsedMicroseconds() { return elapsed().count(); }
  double elapsedSeconds() { return elapsed().count() / 1000000.0; }

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
 * Helper class for switching device
 */
class CudaDeviceContext {
  int original_device;

  /**
   * Pop current cuda device and set new device.
   * @param i_device device ID to set
   * @param o_device device ID to pop, if o_device is NULL just set device to i_device.
   * @return the same as cudaError_t
   */
  static inline cudaError_t get_set_device(int i_device, int* o_device = nullptr) {
    int current_device = 0;
    cudaError_t err = cudaSuccess;

    err = cudaGetDevice(&current_device);
    if (err != cudaSuccess) return err;

    if (current_device != i_device) {
      err = cudaSetDevice(i_device);
      if (err != cudaSuccess) return err;
    }

    if (o_device) {
      *o_device = current_device;
    }

    return cudaSuccess;
  }

 public:
  CudaDeviceContext(int device) { CK_CUDA_THROW_(get_set_device(device, &original_device)); }
  ~CudaDeviceContext() noexcept(false) { CK_CUDA_THROW_(get_set_device(original_device)); }

  void set_device(int device) const { CK_CUDA_THROW_(get_set_device(device)); }
};

/**
 * Get total product from dims.
 */
inline size_t get_size_from_dims(const std::vector<int>& dims) {
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
inline void check_make_dir(const std::string& finalpath) {
  if (mkdir(finalpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
    if (errno == EEXIST) {
      std::cout << (finalpath + " exist") << std::endl;
    } else {
      // something else
      std::cerr << ("cannot create" + finalpath + ": unexpected error") << std::endl;
    }
  }
}

/**
 * Generate random dataset for HugeCTR test.
 */

template<Check_t T>
class Checker_Traits;

template<>
class Checker_Traits<Check_t::Sum>{
public:
  static char zero(){
    return 0;
  }
  
  static char accum(char pre, char x){
    return pre + x;
  }

  static void write(int N, char* array, char chk_bits, std::ofstream& stream){
    stream.write(reinterpret_cast<char*>(&N), sizeof(int));
    stream.write(reinterpret_cast<char*>(array), N);
    stream.write(reinterpret_cast<char*>(&chk_bits), sizeof(char));
  }

};

template<>
class Checker_Traits<Check_t::None>{
public:
  static char zero(){
    return 0;
  }
  
  static char accum(char pre, char x){
    return 0;
  }

  static void write(int N, char* array, char chk_bits, std::ofstream& stream){
    stream.write(reinterpret_cast<char*>(array), N);
  }
};


template<Check_t T>
class DataWriter{
  std::vector<char> array_;
  std::ofstream& stream_;
  char check_char_{0};
public:
  DataWriter(std::ofstream& stream): stream_(stream){
    check_char_ = Checker_Traits<T>::zero();
  }
  void append(char* array, int N){
    for(int i=0; i<N; i++){
      array_.push_back(array[i]);
      check_char_ = Checker_Traits<T>::accum(check_char_, array[i]);
    }
  }
  void write(){
    // if(array_.size() == 0){
    //   std::cout << "array_.size() == 0" << std::endl;;
    // }
    Checker_Traits<T>::write(static_cast<int>(array_.size()), array_.data(), check_char_, stream_);
    check_char_ = Checker_Traits<T>::zero();
    array_.clear();
  }
};


template <typename T, Check_t CK_T>
void data_generation(std::string file_list_name, std::string data_prefix, int num_files,
                     int num_records_per_file, int slot_num, int vocabulary_size, int label_dim, 
			int dense_dim, int max_nnz) {
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

    DataWriter<CK_T> data_writer(out_stream);

    DataSetHeader header = {1, num_records_per_file, label_dim, dense_dim, slot_num, 0, 0, 0};

    data_writer.append(reinterpret_cast<char*>(&header), sizeof(DataSetHeader));
    data_writer.write();

    for (int i = 0; i < num_records_per_file; i++) {
      UnifiedDataSimulator<int> idata_sim(0, max_nnz - 1);  // both inclusive
      UnifiedDataSimulator<float> fdata_sim(0, 1);  
      UnifiedDataSimulator<T> ldata_sim(0, vocabulary_size - 1);
      for (int j = 0; j < label_dim + dense_dim; j++) {
        float label_dense = fdata_sim.get_num();
	data_writer.append(reinterpret_cast<char*>(&label_dense), sizeof(float));
      }
      for (int k = 0; k < slot_num; k++) {
        int nnz = idata_sim.get_num();
	data_writer.append(reinterpret_cast<char*>(&nnz), sizeof(int));
	//        out_stream.write(reinterpret_cast<char*>(&nnz), sizeof(int));
        for (int j = 0; j < nnz; j++) {
          T value = ldata_sim.get_num();
	  data_writer.append(reinterpret_cast<char*>(&value), sizeof(T));
	  //          out_stream.write(reinterpret_cast<char*>(&value), sizeof(T));
        }
      }
      data_writer.write();
    }
    out_stream.close();
  }
  file_list_stream.close();
  return;
}

// Add a new data_generation function for LocalizedSparseEmbedding testing
// In this function, the relationship between key and slot_id is: key's slot_id=(key%slot_num)
template <typename T, Check_t CK_T>
void data_generation_for_localized_test(std::string file_list_name, std::string data_prefix, int num_files,
                     int num_records_per_file, int slot_num, int vocabulary_size, int label_dim, 
			int dense_dim, int max_nnz) {
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

    DataWriter<CK_T> data_writer(out_stream);

    DataSetHeader header = {1, num_records_per_file, label_dim, dense_dim, slot_num, 0, 0, 0};

    data_writer.append(reinterpret_cast<char*>(&header), sizeof(DataSetHeader));
    data_writer.write();

    for (int i = 0; i < num_records_per_file; i++) {
      UnifiedDataSimulator<int> idata_sim(0, max_nnz - 1);  // for nnz
      UnifiedDataSimulator<float> fdata_sim(0, 1);  // for lable and dense
      UnifiedDataSimulator<T> ldata_sim(0, vocabulary_size - 1); // for key
      for (int j = 0; j < label_dim + dense_dim; j++) {
        float label_dense = fdata_sim.get_num();
        data_writer.append(reinterpret_cast<char*>(&label_dense), sizeof(float));
      }
      for (int k = 0; k < slot_num; k++) {
        int nnz = idata_sim.get_num();
	      data_writer.append(reinterpret_cast<char*>(&nnz), sizeof(int));
        for (int j = 0; j < nnz; j++) {
          T key = ldata_sim.get_num();
          while((key % slot_num) != k) { // guarantee the key belongs to the current slot_id(=k)
            key = ldata_sim.get_num();
          }
	        data_writer.append(reinterpret_cast<char*>(&key), sizeof(T));
        }
      }
      data_writer.write();
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
bool find_item_in_map(ITEM_TYPE& item, const std::string& str,
                      const std::map<std::string, ITEM_TYPE>& item_map) {
  typename std::map<std::string, ITEM_TYPE>::const_iterator it = item_map.find(str);
  if (it == item_map.end()) {
    return false;
  } else {
    item = it->second;
    return true;
  }
}

}  // namespace HugeCTR
