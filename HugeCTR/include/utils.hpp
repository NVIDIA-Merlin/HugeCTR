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
#include <cudnn.h>
#include <chrono>
#include <cmath>
#include <common.hpp>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>
#include <unistd.h>
#include <getopt.h>
#include <sstream>
#include <algorithm>

namespace HugeCTR {

// const static char* data_generator_options = "";
// const static struct option data_generator_long_options[] = {
//       {"files", required_argument, NULL, 'f'},
//       {"samples",  required_argument, NULL, 's'},
//       {"long-tail", required_argument, NULL, 'l'},
//       {NULL, 0, NULL, 0}
// };
// class ArgParser {
// public:
//   static void parse_data_generator_args(int argc, char* argv[], int& files, int& samples, std::string& tail, bool& use_long_tail) {
//     int opt;
//     int option_index;
//     while ( (opt = getopt_long(argc,
//                                argv,
//                                data_generator_options,
//                                data_generator_long_options,
//                                &option_index)) != EOF) {
//       if (optarg == NULL) {
//         std::string opt_temp = argv[optind-1];
//         CK_THROW_(Error_t::WrongInput, "Unrecognized option for data generator: " + opt_temp);
//       }
//       switch (opt)
//       {
//       case 'f': {
//         files = std::stoi(optarg);
//         break;
//       }
//       case 's': {
//         samples = std::stoi(optarg);
//         break;
//       }
//       case 'l': {
//         tail = optarg;
//         use_long_tail = true;
//         break;
//       }
//       default:
//         assert(!"Error: no such option && should never get here!!");
//       }
//     }
//   }
// };

template <typename T> inline
void ArgConvertor(std::string arg, T& ret);

template <> inline
void ArgConvertor(std::string arg, int& ret){
  ret = std::stoi(arg);
}

template <> inline
void ArgConvertor(std::string arg, size_t& ret){
  ret = std::stoul(arg);
}

template <> inline
void ArgConvertor(std::string arg, float& ret){
  ret = std::stof(arg);
}


template<> inline
void ArgConvertor(std::string arg, std::vector<int>& ret){
  ret.clear();
  std::stringstream ss(arg);
  for (int i; ss >> i;) {
    ret.push_back(i);    
    if (ss.peek() == ',')
      ss.ignore();
  }
}

template<> inline
void ArgConvertor(std::string arg, std::vector<size_t>& ret){
  ret.clear();
  std::stringstream ss(arg);
  for (size_t i; ss >> i;) {
    ret.push_back(i);    
    if (ss.peek() == ',')
      ss.ignore();
  }
}


template<> inline
void ArgConvertor(std::string arg, std::string& ret){
  ret = arg;
}

struct ArgParser {

private:
  static std::string get_arg_(const std::string target, int argc, char** argv){

    std::vector <std::string> tokens;
    for (int i=1; i < argc; ++i)
      tokens.push_back(std::string(argv[i]));
    std::vector<std::string>::const_iterator itr;
    const std::string option = "--" + target;
    itr =  std::find(tokens.begin(), tokens.end(), option);
    if (itr != tokens.end() && ++itr != tokens.end()){
      return *itr;
    }
    static const std::string empty_string("");
    return empty_string;
  }

public:
  template <typename T>
  static T get_arg(const std::string target, int argc, char** argv){
    auto arg = get_arg_(target, argc, argv);
    if(arg.empty()){
      CK_THROW_(Error_t::WrongInput, "Cannot find target string: " + target);
    }
    T ret;
    ArgConvertor<T>(arg, ret);
    return ret;
  }
  template <typename T>
  static T get_arg(const std::string target, int argc, char** argv, T default_val){
    auto arg = get_arg_(target, argc, argv);
    if(arg.empty()){
      MESSAGE_("Cannot find target string: " + target + " use default value:");
      return default_val;
    }
    T ret;
    ArgConvertor<T>(arg, ret);
    return ret;
  }
  static bool has_arg(const std::string target, int argc, char** argv){
    std::vector <std::string> tokens;
    for (int i=1; i < argc; ++i)
      tokens.push_back(std::string(argv[i]));
    const std::string option = "--" + target;
    return std::find(tokens.begin(), tokens.end(), option)
      != tokens.end();
  }
};

template <typename T>
std::string vec_to_string(std::vector<T> vec){
  std::string ret;
  for(auto& elem: vec){
    ret = ret + std::to_string(elem) + ", ";
  }
  return ret.substr(0, ret.size()-2);
}


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
 * GPU Timer
 */
class GPUTimer {
 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
  cudaStream_t stream_;

 public:
  GPUTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~GPUTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start(cudaStream_t st = 0) {
    stream_ = st;
    cudaEventRecord(start_, stream_);
  }

  float stop() {
    float milliseconds = 0;
    cudaEventRecord(stop_, stream_);
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&milliseconds, start_, stop_);
    return milliseconds;
  }
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

template <typename TOUT, typename TIN>
struct TypeConvert;

template <>
struct TypeConvert<float, float> {
  static __host__ float convert(const float val) { return val;}
};

template <>
struct TypeConvert<__half, float> {
  static __host__ __half convert(const float val) { return __float2half(val); }
};

template <>
struct TypeConvert<float, __half> {
  static __host__ float convert(const __half val) { return __half2float(val); }
};

template <typename T>
struct CudnnDataType;

template <>
struct CudnnDataType<float> {
  static cudnnDataType_t getType() { return CUDNN_DATA_FLOAT; }
};

template <>
struct CudnnDataType<__half> {
  static cudnnDataType_t getType() { return CUDNN_DATA_HALF; }
};



  
}  // namespace HugeCTR
