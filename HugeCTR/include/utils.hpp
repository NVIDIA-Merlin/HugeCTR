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

#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <getopt.h>
#include <numa.h>
#include <unistd.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <common.hpp>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>
#include <unistd.h>
#include <getopt.h>
#include <sstream>
#include <algorithm>
#include <nccl.h>
#include <numa.h>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif
namespace HugeCTR {

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
 * Helper class for switching device and the associated NUMA domain.
 * Sticky: thread will remember the context and affinity.
 */
class CudaCPUDeviceContext {
 public:
  CudaCPUDeviceContext(int device_id) {
    auto node_it = device_id_to_numa_node_.find(device_id);
    assert(node_it != device_id_to_numa_node_.end());
    CK_CUDA_THROW_(cudaSetDevice(device_id));

    int node = node_it->second;
    if (node >= 0) {
      numa_run_on_node(node);
      numa_set_preferred(node);
    }
  }

  static void init_cpu_mapping(std::vector<int> device_ids) {
    constexpr int pci_id_len = 16;
    char pci_id[pci_id_len];

    std::stringstream ss;
    ss << "Device to NUMA mapping:" << std::endl;

    device_id_to_numa_node_.clear();
    auto cpu_mask = numa_allocate_cpumask();

    auto select_node = [](const bitmask* nvml_cpus) -> int {
      for (int cpu = 0; cpu < numa_num_possible_cpus(); cpu++) {
        if (numa_bitmask_isbitset(nvml_cpus, cpu)) {
          return numa_node_of_cpu(cpu);
        }
      }
      return -1;
    };

    for (auto device_id : device_ids) {
      nvmlDevice_t handle;
      CK_CUDA_THROW_(cudaDeviceGetPCIBusId(pci_id, pci_id_len, device_id));
      CK_NVML_THROW_(nvmlDeviceGetHandleByPciBusId_v2(pci_id, &handle));
      CK_NVML_THROW_(nvmlDeviceGetCpuAffinity(handle, cpu_mask->size / (sizeof(unsigned long) * 8),
                                              cpu_mask->maskp));
      int node = select_node(cpu_mask);
      device_id_to_numa_node_[device_id] = node;
      ss << "  GPU " << device_id << " -> "
         << " node " << node << std::endl;
    }

    MESSAGE_(ss.str());

    numa_bitmask_free(cpu_mask);
  }

 public:
  static std::unordered_map<int, int> device_id_to_numa_node_;
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

#ifdef ENABLE_MPI
template <typename TypeKey>
struct ToMpiType;

template <>
struct ToMpiType<long long> {
  static MPI_Datatype T() { return MPI_LONG_LONG; }
};

template <>
struct ToMpiType<unsigned int> {
  static MPI_Datatype T() { return MPI_UNSIGNED; }
};

template <>
struct ToMpiType<float> {
  static MPI_Datatype T() { return MPI_FLOAT; }
};

#endif
template <typename TIN, typename TOUT>
void convert_array_on_device(TOUT* out, const TIN* in, size_t num_elements, const cudaStream_t& stream);
template <typename T> struct NcclDataType;
template <> struct NcclDataType<int> { static ncclDataType_t getType() { return ncclInt32; } };
template <> struct NcclDataType<long long> { static ncclDataType_t getType() { return ncclInt64; } };
template <> struct NcclDataType<unsigned int> { static ncclDataType_t getType() { return ncclUint32; } };
template <> struct NcclDataType<unsigned long long> { static ncclDataType_t getType() { return ncclUint64; } };
template <> struct NcclDataType<float> { static ncclDataType_t getType() { return ncclFloat32; } };
template <> struct NcclDataType<__half> { static ncclDataType_t getType() { return ncclHalf; } };

template <typename TypeKey>
void data_to_unique_categories(TypeKey *value, const TypeKey *rowoffset,
                               const TypeKey *emmbedding_offsets, int num_tables,
                               int num_rowoffsets, const cudaStream_t &stream);

template <typename TypeKey>
void data_to_unique_categories(TypeKey *value, const TypeKey *emmbedding_offsets, int num_tables,
                               int nnz, const cudaStream_t &stream);

template <typename TypeKey>
void distribute_keys_for_inference(TypeKey* out, TypeKey* in, size_t batchsize, std::vector<size_t>& max_feature_num_for_tables) {
  size_t num_tables = max_feature_num_for_tables.size();
  std::vector<size_t> batch_keys_offset(num_tables, 0);
  std::vector<size_t> sample_keys_offset(num_tables, 0);
  size_t num_keys_per_sample = 0;
  for (size_t i = 0; i < num_tables; i++) {
    num_keys_per_sample += max_feature_num_for_tables[i];
    if (i > 0) {
      batch_keys_offset[i] = batch_keys_offset[i-1] + batchsize*max_feature_num_for_tables[i-1];
      sample_keys_offset[i] = sample_keys_offset[i-1] + max_feature_num_for_tables[i-1];
    }
  }
  for (size_t i = 0; i < batchsize; i++) {
    for (size_t j = 0; j < num_tables; j++) {
      for (size_t k = 0; k < max_feature_num_for_tables[j]; k++) {
        out[i*num_keys_per_sample+sample_keys_offset[j]+k] = in[batch_keys_offset[j]+i*max_feature_num_for_tables[j]+k];
      }
    }
  }
}

}  // namespace HugeCTR
