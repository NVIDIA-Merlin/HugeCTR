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

/*
 * This is the header file for the set of HugeCTR debugging features.
 *
 * 1. Multi-level, easy-to-redirect logging:
 * Instead of using std::cout or printf, we recommend you use our HCTR_LOG or HCTR_PRINT.
 * By using them, you can easily adjust which levels of messages are printed at.
 * We have 5 reserved levels or channels where logs are printed, but you can define your own
 * level N >= 4 as well. The message level is specfied as the first argument of log function.
     ERROR (-1): error messages. (stderr)
     SILENCE (0): messages which are never printed.
     INFO (1): non-error & non-warning informative messages. (stdout)
     WARNING (2): warning messages (stdout)
     DEBUG (3): debug, verbose messages (stdout)

 * 1.1. Examples:
     HCTR_LOG(INFO, ROOT, "the current value: %d\n" val); // only the root or rank0 prints the
 message. HCTR_PRINT(INFO, "the current value: %d\n" val); // the same as the call above except
 there is no header. HCTR_LOG(ERROR, WORLD, "the current value: %d\n" val); // all the ranks print
 the message to stderr. HCTR_LOG(INFO, ROOT, "the current value: %d\n" val); // only the root or
 rank0 prints the message. HCTR_LOG_AT(0, ROOT, "the current value: %d\n" val); // the level is
 specified as a number, e.g., INFO == 0.

 * If the HugeCTR is built in the release mode, the default maximum log level is 2 (or WARNING).
 * Thus, all the meesages which are at a level N <= 2, except SILENCE, are shown to users.
 * On the other hand, in the debug mode, the default log level is changed to 3 (or DEBUG).
 *
 * You can also change the maximum log level, without rebuild, by setting an env variable
 'HUGECTR_LOG_LEVEL'.
 * 1.2. Examples:
     $ HUGECTR_LOG_LEVEL=3 python dcn_norm_train.py

 * The default file streams for log messages are stdout and stderr, but you can redirect them to
 files
 * by setting an env variable 'HUGECTR_LOG_TO_FILE' to 1 whilst still print them to screen. You can
 find them in your execution directory.
 * The messages to different levels and MPI ranks are written to different files. If
 'HUGECTR_LOG_TO_FILE' is set to 2,
 * nothing is printed on screen, but all messages are written to those files.
 * 1.2. Examples:
     $ HUGECTR_LOG_TO_FILE=1 python dcn_norm_train.py
     $ ls
     hctr_3374842_0_error.log
     hctr_3374842_0_info.log
     hctr_3374842_0_warning.log
     hctr_3374842_0_debug.log

 * 2. Exception handling:
 * For HugeCTR's own errors, HCTR_OWN_THROW is used.
 * For MPI, use HCTR_MPI_THROW. For the other libraries including CUDA, cuBLAS, NCCL,etc, use
 HCTR_LIB_THROW.
 * The throwed exception records where the error has occured (or caught) and what the error is
 about.
 * If you add a new library, to track its error, it is recommended that you specialize getErrorType
 and getErrorString below.
 * 2.1. Examples:
     HCTR_OWN_THROW(Error_t::WrongInput, "device is not avaliable");
     HCTR_LIB_THROW(cudaDeviceSynchronize());
     HCTR_LIB_THROW(cublasGemmEx(...));
     HCTR_MPI_THROW(MPI_Gather(...));

 * If you want to print the nested exception message at a catch statement, call
 'Logger::print_exception(e, 0)'.
 * Then, they will be printed at the ERROR level.
 *
 * 3. Error check:
 * You sometimes want to terminate the HugeCTR immediately rather than throwing an exception.
 * Like the HCTR_*_THROW, the error message shows where the error has occured and which expression
 is failed.
 * 3.1. Host error check:
 * To check if an expression is valid on the host side, use HCTR_CHECK (always executed) or
 HCTR_ASSERT (debug build only).
 * 3.1.1. Examples:
     HCTR_CHECK(mixed_precision_mode == true);
     HCTR_ASSERT(emd_vec_size >= 16);

 * 3.2. Device error check:
 * To check if a device API all is failed, use HCTR_CUDA_CHECK.
 * 3.2.1. Examples:
     HCTR_CUDA_CHECK(BLOCKING, cudaLaunchKernel(...));
     HCTR_CUDA_CHECK(BLOCKING, cudaMemcpyAsync(...));
     HCTR_CUDA_CHECK(ASYNC, cudaGetDevice(...));

 * If you specify its first argument as 'BLOCKING', it will insert a cudaDeviceSynchronize() for
 you,
 * which can be useful in debugging asynchronous kernel launches or cudaMemcpys.
 */

#pragma once

#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <nccl.h>
#include <nvml.h>

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {

enum class Error_t {
  Success,
  FileCannotOpen,
  BrokenFile,
  OutOfMemory,
  OutOfBound,
  WrongInput,
  IllegalCall,
  NotInitialized,
  UnSupportedFormat,
  InvalidEnv,
  DataCheckError,
  UnspecificError,
  EndOfFile,
  MpiError,
  CudaError,
  CublasError,
  CudnnError,
  CurandError,
  NcclError,
  NvmlError
};

// We have five reserved verbosity levels for users' convenience.
#define LOG_ERROR_LEVEL -1
#define LOG_SILENCE_LEVEL 0  // print nothing
#define LOG_INFO_LEVEL 1
#define LOG_WARNING_LEVEL 2
#define LOG_DEBUG_LEVEL 3  // If you build in debug mode, it is the default mode

#define LOG_LEVEL(NAME) LOG_##NAME##_LEVEL

#define LOG_RANK_ROOT false
#define LOG_RANK_WORLD true

#define LOG_RANK(TYPE) LOG_RANK_##TYPE

#ifndef NDEBUG
#define DEFAULT_LOG_LEVEL LOG_LEVEL(WARNING)
#else
#define DEFAULT_LOG_LEVEL LOG_LEVEL(DEBUG)
#endif

#define LEVEL_MAP(MAP, NAME) MAP[LOG_LEVEL(NAME)] = #NAME

#define HCTR_LOG(NAME, TYPE, ...) \
  Logger::get().log(LOG_LEVEL(NAME), LOG_RANK(TYPE), true, __VA_ARGS__)
#define HCTR_LOG_AT(LEVEL, TYPE, ...) Logger::get().log(LEVEL, LOG_RANK(TYPE), true, __VA_ARGS__)

#define HCTR_LOG_S(NAME, TYPE) Logger::get().log(LOG_LEVEL(NAME), LOG_RANK(TYPE), true)

#define HCTR_PRINT(NAME, ...) Logger::get().log(LOG_LEVEL(NAME), LOG_RANK_ROOT, false, __VA_ARGS__)
#define HCTR_PRINT_AT(LEVEL, ...) Logger::get().log(LEVEL, LOG_RANK_ROOT, false, __VA_ARGS__)

struct SrcLoc {
  const char* file;
  unsigned line;
  const char* func;
  const char* expr;
};

#define CUR_SRC_LOC(EXPR) \
  SrcLoc { __FILE__, __LINE__, __func__, #EXPR }

template <typename SrcType>
Error_t getErrorType(SrcType err);
template <>
inline Error_t getErrorType(cudaError_t err) {
  return (err == cudaSuccess) ? Error_t::Success : Error_t::CudaError;
}
template <>
inline Error_t getErrorType(nvmlReturn_t err) {
  return (err == NVML_SUCCESS) ? Error_t::Success : Error_t::NvmlError;
}
template <>
inline Error_t getErrorType(cublasStatus_t err) {
  return (err == CUBLAS_STATUS_SUCCESS) ? Error_t::Success : Error_t::CublasError;
}
template <>
inline Error_t getErrorType(ncclResult_t err) {
  return (err == ncclSuccess) ? Error_t::Success : Error_t::NcclError;
}
template <>
inline Error_t getErrorType(cudnnStatus_t err) {
  return (err == CUDNN_STATUS_SUCCESS) ? Error_t::Success : Error_t::CudnnError;
}
template <>
inline Error_t getErrorType(curandStatus_t err) {
  return (err == CURAND_STATUS_SUCCESS) ? Error_t::Success : Error_t::CurandError;
}

template <typename SrcType>
std::string getErrorString(SrcType err);
template <>
inline std::string getErrorString(cudaError_t err) {
  return std::string(cudaGetErrorString(err));
}
template <>
inline std::string getErrorString(nvmlReturn_t err) {
  return std::string(nvmlErrorString(err));
}
template <>
inline std::string getErrorString(cublasStatus_t err) {
  switch (err) {
    case CUBLAS_STATUS_SUCCESS:
      return std::string("cuBLAS operation completed successfully. What are you doing here?");
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return std::string("cuBLAS was not initialized.");
    case CUBLAS_STATUS_ALLOC_FAILED:
      return std::string("cuBLAS internal resource allocation failed.");
    case CUBLAS_STATUS_INVALID_VALUE:
      return std::string("cuBLAS got an upsopported value or parameter.");
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return std::string("cuBLAS feature is unavailable on the current device arch.");
    case CUBLAS_STATUS_MAPPING_ERROR:
      return std::string("cuBLAS failed to access GPU memory space.");
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return std::string("cuBLAS failed execute the GPU program or launch the kernel.");
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return std::string("cuBLAS internal operation failed.");
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return std::string("cuBLAS doen't support the requested functionality.");
    case CUBLAS_STATUS_LICENSE_ERROR:
      return std::string("cuBLAS failed to check the current licencing.");
    default:
      return std::string("cuBLAS unkown error.");
  }
}

template <>
inline std::string getErrorString(ncclResult_t err) {
  return std::string(ncclGetErrorString(err));
}
template <>
inline std::string getErrorString(cudnnStatus_t err) {
  return std::string(cudnnGetErrorString(err));
}
template <>
inline std::string getErrorString(curandStatus_t err) {
  switch (err) {
    case CURAND_STATUS_SUCCESS:
      std::string("cuRAND no errors.");
    case CURAND_STATUS_VERSION_MISMATCH:
      std::string("cuRAND header file and linked library version do not match.");
    case CURAND_STATUS_NOT_INITIALIZED:
      std::string("cuRAND generator not initialized.");
    case CURAND_STATUS_ALLOCATION_FAILED:
      std::string("cuRAND memory allocation failed.");
    case CURAND_STATUS_TYPE_ERROR:
      std::string("cuRAND generator is wrong type.");
    case CURAND_STATUS_OUT_OF_RANGE:
      std::string("cuRAND argument out of range.");
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      std::string("cuRAND length requested is not a multple of dimension.");
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      std::string("cuRAND GPU does not have double precision required by MRG32k3a.");
    case CURAND_STATUS_LAUNCH_FAILURE:
      std::string("cuRAND kernel launch failure.");
    case CURAND_STATUS_PREEXISTING_FAILURE:
      std::string("cuRAND preexisting failure on library entry.");
    case CURAND_STATUS_INITIALIZATION_FAILED:
      std::string("cuRAND initialization of CUDA failed.");
    case CURAND_STATUS_ARCH_MISMATCH:
      std::string("cuRAND architecture mismatch, GPU does not support requested feature.");
    case CURAND_STATUS_INTERNAL_ERROR:
      std::string("cuRAND Internal library error.");
    default:
      return std::string("cuRAND unkown error.");
  }
}

// For HugeCTR own error types, it is up to users to define the msesage.
#define HCTR_OWN_THROW(EXPR, MSG)                                           \
  do {                                                                      \
    Error_t err_thr = (EXPR);                                               \
    if (err_thr != Error_t::Success) {                                      \
      Logger::get().do_throw(err_thr, CUR_SRC_LOC(EXPR), std::string(MSG)); \
    }                                                                       \
  } while (0);

#ifdef ENABLE_MPI
// Because MPI error code is in int, it is safe to have a separate macro for MPI,
// rather than reserving `int` as MPI error type.
// We don't want this set of macros to become another source of errors.
#define HCTR_MPI_THROW(EXPR)                                                              \
  do {                                                                                    \
    auto err_thr = (EXPR);                                                                \
    if (err_thr != MPI_SUCCESS) {                                                         \
      char err_str[MPI_MAX_ERROR_STRING];                                                 \
      int err_len = MPI_MAX_ERROR_STRING;                                                 \
      MPI_Error_string(err_thr, err_str, &err_len);                                       \
      Logger::get().do_throw(Error_t::MpiError, CUR_SRC_LOC(EXPR), std::string(err_str)); \
    }                                                                                     \
  } while (0);
#endif

// For other library calls such as CUDA, cuBLAS and NCCL, use this macro
#define HCTR_LIB_THROW(EXPR)                                        \
  do {                                                              \
    auto ret_thr = (EXPR);                                          \
    Error_t err_type = getErrorType(ret_thr);                       \
    if (err_type != Error_t::Success) {                             \
      std::string err_msg = getErrorString(ret_thr);                \
      Logger::get().do_throw(err_type, CUR_SRC_LOC(EXPR), err_msg); \
    }                                                               \
  } while (0);

#define HCTR_THROW_IF(EXPR, ERROR, MSG)                                     \
  do {                                                                      \
    const auto& expr = (EXPR);                                              \
    if (expr) {                                                             \
      Logger::get().do_throw((ERROR), CUR_SRC_LOC(EXPR), std::string(MSG)); \
    }                                                                       \
  } while (0)

#define CHECK_CALL(MODE) CHECK_##MODE##_CALL

#define CHECK_BLOCKING_CALL true
#define CHECK_ASYNC_CALL false

#define HCTR_CHECK(EXPR)                          \
  do {                                            \
    Logger::get().check(EXPR, CUR_SRC_LOC(EXPR)); \
  } while (0)

#define HCTR_CHECK_HINT(EXPR, HINT, ...)                               \
  do {                                                                 \
    Logger::get().check(EXPR, CUR_SRC_LOC(EXPR), HINT, ##__VA_ARGS__); \
  } while (0)

#define HCTR_DIE(HINT, ...) HCTR_CHECK_HINT(false, HINT, ##__VA_ARGS__)

// TODO: print the cuda error string
#define HCTR_CUDA_CHECK(SYNC_MODE, FUNC)                            \
  do {                                                              \
    auto ret_err = (FUNC);                                          \
    if (CHECK_CALL(SYNC_MODE)) {                                    \
      ret_err = cudaDeviceSynchronize();                            \
    }                                                               \
    Logger::get().check(ret_err == cudaSuccess, CUR_SRC_LOC(EXPR)); \
  } while (0);

#ifndef NDEBUG
#define HCTR_ASSERT(EXPR)                                              \
  do {                                                                 \
    Logger::get().check_lazy([&] { return EXPR; }, CUR_SRC_LOC(EXPR)); \
  } while (0);
#else
#define HCTR_ASSERT(EXPR)
#endif

class DeferredLogEntry {
 public:
  inline DeferredLogEntry(const bool bypass,
                          std::function<void(std::ostringstream&)> make_log_entry)
      : bypass_{bypass}, make_log_entry_{make_log_entry} {}

  inline ~DeferredLogEntry() { make_log_entry_(ss_); }

  DeferredLogEntry(const DeferredLogEntry&) = delete;
  DeferredLogEntry(const DeferredLogEntry&&) = delete;
  DeferredLogEntry& operator=(const DeferredLogEntry&) = delete;
  DeferredLogEntry&& operator=(const DeferredLogEntry&&) = delete;

  template <typename T>
  inline DeferredLogEntry& operator<<(const T& value) {
    if (!bypass_) {
      ss_ << value;
    }
    return *this;
  }

  inline DeferredLogEntry& operator<<(std::ostream& (*fn)(std::ostream&)) {
    if (!bypass_) {
      fn(ss_);
    }
    return *this;
  }

 private:
  bool bypass_;
  std::ostringstream ss_;
  std::function<void(std::ostringstream&)> make_log_entry_;
};

class Logger final {
 public:
  static void print_exception(const std::exception& e, int depth);
  static Logger& get();
  ~Logger();
  void log(const int level, bool per_rank, bool with_prefix, const char* format, ...) const;
  DeferredLogEntry log(const int level, bool per_rank, bool with_prefix) const;
  void check(bool condition, const SrcLoc& loc, const char* format = nullptr, ...) const;
  template <typename Condition>
  void check_lazy(const Condition& condition, const SrcLoc& loc) {
    check(condition(), loc);
  }
  void do_throw(HugeCTR::Error_t error_type, const SrcLoc& loc, const std::string& message) const;
  int get_rank();

 private:
  Logger();
  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

  FILE* get_file_stream(int level);
  std::string get_log_prefix(int level) const;

  int rank_;
  int max_level_;
  bool log_to_std_;
  bool log_to_file_;
  std::map<int, FILE*> log_std_;
  std::map<int, FILE*> log_file_;
  std::map<int, std::string> level_name_;

  static std::unique_ptr<Logger> g_instance;
  static std::once_flag g_once_flag;
};

// TODO: Make fully templated and find better location for this?
template <typename TTarget>
inline static TTarget hctr_safe_cast(const size_t value) {
  HCTR_CHECK(value <= std::numeric_limits<TTarget>::max());
  return static_cast<TTarget>(value);
}

template <typename TTarget>
inline static TTarget hctr_safe_cast(const double value) {
  HCTR_CHECK(value >= std::numeric_limits<TTarget>::lowest() &&
             value <= std::numeric_limits<TTarget>::max());
  return static_cast<TTarget>(value);
}

}  // namespace HugeCTR
