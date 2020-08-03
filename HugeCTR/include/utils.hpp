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

#include <cuda_runtime_api.h>
#include <sys/stat.h>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>
#include <common.hpp>
#include <data_parser.hpp>

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
  int original_device_;

 public:
  CudaDeviceContext() { CK_CUDA_THROW_(cudaGetDevice(&original_device_)); }
  CudaDeviceContext(int device) : CudaDeviceContext() {
    if (device != original_device_) {
      set_device(device);
    }
  }
  ~CudaDeviceContext() noexcept(false) { set_device(original_device_); }

  void set_device(int device) const { CK_CUDA_THROW_(cudaSetDevice(device)); }
};

/**
 * Get total product from dims.
 */
inline size_t get_size_from_dims(const std::vector<size_t>& dims) {
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

template <Check_t T>
class Checker_Traits;

template <>
class Checker_Traits<Check_t::Sum> {
 public:
  static char zero() { return 0; }

  static char accum(char pre, char x) { return pre + x; }

  static void write(int N, char* array, char chk_bits, std::ofstream& stream) {
    stream.write(reinterpret_cast<char*>(&N), sizeof(int));
    stream.write(reinterpret_cast<char*>(array), N);
    stream.write(reinterpret_cast<char*>(&chk_bits), sizeof(char));
  }

  static long long ID() { return 1; }
};

template <>
class Checker_Traits<Check_t::None> {
 public:
  static char zero() { return 0; }

  static char accum(char pre, char x) { return 0; }

  static void write(int N, char* array, char chk_bits, std::ofstream& stream) {
    stream.write(reinterpret_cast<char*>(array), N);
  }

  static long long ID() { return 0; }
};

template <Check_t T>
class DataWriter {
  std::vector<char> array_;
  std::ofstream& stream_;
  char check_char_{0};

 public:
  DataWriter(std::ofstream& stream) : stream_(stream) { check_char_ = Checker_Traits<T>::zero(); }
  void append(char* array, int N) {
    for (int i = 0; i < N; i++) {
      array_.push_back(array[i]);
      check_char_ = Checker_Traits<T>::accum(check_char_, array[i]);
    }
  }
  void write() {
    // if(array_.size() == 0){
    //   std::cout << "array_.size() == 0" << std::endl;;
    // }
    Checker_Traits<T>::write(static_cast<int>(array_.size()), array_.data(), check_char_, stream_);
    check_char_ = Checker_Traits<T>::zero();
    array_.clear();
  }
};

template <typename T, Check_t CK_T>
void data_generation_for_test(std::string file_list_name, std::string data_prefix, int num_files,
                              int num_records_per_file, int slot_num, int vocabulary_size,
                              int label_dim, int dense_dim, int max_nnz) {
  if (file_exist(file_list_name)) {
    std::cout << "File (" + file_list_name +
                     ") exist. To generate new dataset plesae remove this file."
              << std::endl;
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
    std::cout << tmp_file_name << std::endl;
    // data generation;
    std::ofstream out_stream(tmp_file_name, std::ofstream::binary);

    DataWriter<CK_T> data_writer(out_stream);

    DataSetHeader header = {
        Checker_Traits<CK_T>::ID(), num_records_per_file, label_dim, dense_dim, slot_num, 0, 0, 0};

    data_writer.append(reinterpret_cast<char*>(&header), sizeof(DataSetHeader));
    data_writer.write();

    for (int i = 0; i < num_records_per_file; i++) {
      UnifiedDataSimulator<int> idata_sim(1, max_nnz);            // for nnz
      UnifiedDataSimulator<float> fdata_sim(0, 1);                // for lable and dense
      UnifiedDataSimulator<T> ldata_sim(0, vocabulary_size - 1);  // for key
      for (int j = 0; j < label_dim + dense_dim; j++) {
        float label_dense = fdata_sim.get_num();
        data_writer.append(reinterpret_cast<char*>(&label_dense), sizeof(float));
      }
      for (int k = 0; k < slot_num; k++) {
        int nnz = idata_sim.get_num();
        data_writer.append(reinterpret_cast<char*>(&nnz), sizeof(int));
        for (int j = 0; j < nnz; j++) {
          T key = ldata_sim.get_num();
          while ((key % static_cast<T>(slot_num)) !=
                 static_cast<T>(k)) {  // guarantee the key belongs to the current slot_id(=k)
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
  std::cout << file_list_name << " done!" << std::endl;
  return;
}

// Add a new data_generation function for LocalizedSparseEmbedding testing
// In this function, the relationship between key and slot_id is: key's slot_id=(key%slot_num)
// Add a new data_generation function for LocalizedSparseEmbedding testing
// In this function, the relationship between key and slot_id is: key's slot_id=(key%slot_num)
template <typename T, Check_t CK_T>
void data_generation_for_localized_test(std::string file_list_name, std::string data_prefix,
                                        int num_files, int num_records_per_file, int slot_num,
                                        int vocabulary_size, int label_dim, int dense_dim,
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

    DataWriter<CK_T> data_writer(out_stream);

    DataSetHeader header = {1, num_records_per_file, label_dim, dense_dim, slot_num, 0, 0, 0};

    data_writer.append(reinterpret_cast<char*>(&header), sizeof(DataSetHeader));
    data_writer.write();

    for (int i = 0; i < num_records_per_file; i++) {
      UnifiedDataSimulator<int> idata_sim(1, max_nnz);            // for nnz
      UnifiedDataSimulator<float> fdata_sim(0, 1);                // for lable and dense
      UnifiedDataSimulator<T> ldata_sim(0, vocabulary_size - 1);  // for key
      for (int j = 0; j < label_dim + dense_dim; j++) {
        float label_dense = fdata_sim.get_num();
        data_writer.append(reinterpret_cast<char*>(&label_dense), sizeof(float));
      }
      for (int k = 0; k < slot_num; k++) {
        int nnz = idata_sim.get_num();
        data_writer.append(reinterpret_cast<char*>(&nnz), sizeof(int));
        for (int j = 0; j < nnz; j++) {
          T key = ldata_sim.get_num();
          while ((key % slot_num) != k) {  // guarantee the key belongs to the current slot_id(=k)
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

template <typename T, Check_t CK_T>
void data_generation_for_localized_test(std::string file_list_name, std::string data_prefix,
                                        int num_files, int num_records_per_file, int slot_num,
                                        int vocabulary_size, int label_dim, int dense_dim,
                                        int max_nnz, const std::vector<size_t> slot_sizes) {
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
      UnifiedDataSimulator<int> idata_sim(1, max_nnz);  // for nnz per slot
      UnifiedDataSimulator<float> fdata_sim(0, 1);      // for lable and dense
      for (int j = 0; j < label_dim + dense_dim; j++) {
        float label_dense = fdata_sim.get_num();
        data_writer.append(reinterpret_cast<char*>(&label_dense), sizeof(float));
      }
      size_t offset = 0;
      for (int k = 0; k < slot_num; k++) {
        // int nnz = idata_sim.get_num();
        int nnz = max_nnz;  // for test
        data_writer.append(reinterpret_cast<char*>(&nnz), sizeof(int));
        size_t slot_size = slot_sizes[k];
        UnifiedDataSimulator<T> ldata_sim(0, slot_size - 1);  // for key
        for (int j = 0; j < nnz; j++) {
          T key = ldata_sim.get_num() + offset;
          data_writer.append(reinterpret_cast<char*>(&key), sizeof(T));
        }
        offset += slot_size;
      }
      data_writer.write();
    }
    out_stream.close();
  }
  file_list_stream.close();
  return;
}

inline void data_generation_for_raw(
    std::string file_name, long long num_samples, int label_dim = 1, int dense_dim = 13,
    int sparse_dim = 26, const std::vector<long long> slot_size = std::vector<long long>()) {
  std::ofstream out_stream(file_name, std::ofstream::binary);
  for (long long i = 0; i < num_samples; i++) {
    for (int j = 0; j < label_dim; j++) {
      int label = i % 2;
      out_stream.write(reinterpret_cast<char*>(&label), sizeof(int));
    }
    for (int j = 0; j < dense_dim; j++) {
      int dense = j;
      out_stream.write(reinterpret_cast<char*>(&dense), sizeof(int));
    }
    for (int j = 0; j < sparse_dim; j++) {
      int sparse = 0;
      if (slot_size.size() != 0) {
        UnifiedDataSimulator<long long> temp_sim(
            0, (slot_size[j] - 1) < 0 ? 0 : (slot_size[j] - 1));  // range = [0, slot_size[j])
        long long num_ = temp_sim.get_num();
        sparse = num_ > std::numeric_limits<int>::max() ? std::numeric_limits<int>::max() : num_;
      } else {
        sparse = j;
      }
      out_stream.write(reinterpret_cast<char*>(&sparse), sizeof(int));
    }
  }
  out_stream.close();
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

template <typename T>
void print_cuda_buff(T* buf, int start, int end) {
  T h_buf[end - start];
  cudaMemcpy(h_buf, buf, (end - start) * sizeof(T), cudaMemcpyDeviceToHost);
  std::cout << "Cuda Buff Print " << start << "-" << end << ": " << std::endl;
  for (int i = 0; i < (end - start); i++) {
    std::cout << h_buf[i] << ",";
  }
  std::cout << std::endl;
}

template <typename T>
void print_cuda_buff_sum(T* buf, int num) {
  T sum;
  T h_buf[num];
  cudaMemcpy(h_buf, buf, (num) * sizeof(T), cudaMemcpyDeviceToHost);
  for (int i = 0; i < num; i++) {
    sum += h_buf[i];
  }
  std::cout << "Cuda Buff Sum: " << sum << " size:" << num << std::endl;
}

template <typename TIN, typename TOUT>
std::vector<std::shared_ptr<TOUT>> sp_vec_dynamic_cast(std::vector<std::shared_ptr<TIN>>& vin) {
  std::vector<std::shared_ptr<TOUT>> vout;
  for (auto& i : vin) {
    vout.emplace_back(std::dynamic_pointer_cast<TOUT>(i));
  }
  return vout;
}

inline void set_affinity(std::thread& t, int core) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  int rc = pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    CK_THROW_(Error_t::WrongInput, "Error calling pthread_setaffinity_np: " + std::to_string(rc));
  }
  return;
}

inline void set_affinity(std::thread& t, std::set<int> set, bool excluded) {
  if (set.empty()) {
    auto get_core_id = [](int i) { return (i % 8) * 16 + (i / 8) % 2 + (i / 16) * 128; };
    for (int i = 0; i < 31; i++) {
      int core = get_core_id(i);
      set.insert(core);
    }
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  for (int core = 0; core < 256; core++) {
    if (excluded) {
      if (set.find(core) == set.end()) {
        CPU_SET(core, &cpuset);
      }
    } else {
      if (set.find(core) != set.end()) {
        CPU_SET(core, &cpuset);
      }
    }
  }
  int rc = pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    CK_THROW_(Error_t::WrongInput, "Error calling pthread_setaffinity_np: " + std::to_string(rc));
  }
  return;
}

template <typename T>
struct TypeConvert {
  static __host__ T convert(const float val) { return (T)val; }
};

template <>
struct TypeConvert<__half> {
  static __host__ __half convert(const float val) { return __float2half(val); }
};

}  // namespace HugeCTR
