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

#include <cublas_v2.h>
#include <cuda.h>
#include <cudnn.h>
#include <curand.h>
#include <nccl.h>
#include <nvml.h>

#include <core/macro.hpp>
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
  CudaDriverError,
  CudaRuntimeError,
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
#define LOG_TRACE_LEVEL 9

#define LOG_LEVEL(NAME) LOG_##NAME##_LEVEL

#define LOG_RANK_ROOT false
#define LOG_RANK_WORLD true

#define LOG_RANK(TYPE) LOG_RANK_##TYPE

#ifndef NDEBUG
#define DEFAULT_LOG_LEVEL LOG_LEVEL(WARNING)
#else
#define DEFAULT_LOG_LEVEL LOG_LEVEL(DEBUG)
#endif

#define HCTR_LOG(NAME, TYPE, ...) \
  HugeCTR::Logger::get().log(LOG_LEVEL(NAME), LOG_RANK(TYPE), true, __VA_ARGS__)
#define HCTR_LOG_AT(LEVEL, TYPE, ...) \
  HugeCTR::Logger::get().log(LEVEL, LOG_RANK(TYPE), true, __VA_ARGS__)

#define HCTR_LOG_S(NAME, TYPE) HugeCTR::Logger::get().log(LOG_LEVEL(NAME), LOG_RANK(TYPE), true)

#define HCTR_LOG_C(NAME, TYPE, ...)                                          \
  do {                                                                       \
    const HugeCTR::Logger& logger = HugeCTR::Logger::get();                  \
    if (logger.can_log_at(LOG_LEVEL(NAME), LOG_RANK(TYPE))) {                \
      logger.log(LOG_LEVEL(NAME), LOG_RANK(TYPE), true).append(__VA_ARGS__); \
    }                                                                        \
  } while (0)

#define HCTR_PRINT(NAME, ...) \
  HugeCTR::Logger::get().log(LOG_LEVEL(NAME), LOG_RANK(ROOT), false, __VA_ARGS__)
#define HCTR_PRINT_AT(LEVEL, ...) \
  HugeCTR::Logger::get().log(LEVEL, LOG_RANK(ROOT), false, __VA_ARGS__)

struct SrcLoc {
  const char* file;
  unsigned line;
  const char* func;
  const char* expr;
};

#define CUR_SRC_LOC(EXPR) \
  HugeCTR::SrcLoc { __FILE__, __LINE__, __func__, #EXPR }

#define HCTR_LOCATION() '(' << __FILE__ << ':' << __LINE__ << ')'

template <typename SrcType>
Error_t getErrorType(SrcType err);
template <>
inline Error_t getErrorType(CUresult err) {
  return (err == CUDA_SUCCESS) ? Error_t::Success : Error_t::CudaDriverError;
}
template <>
inline Error_t getErrorType(cudaError_t err) {
  return (err == cudaSuccess) ? Error_t::Success : Error_t::CudaRuntimeError;
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
inline std::string getErrorString(CUresult err) {
  const char* ptr;
  if (cuGetErrorString(err, &ptr) != CUDA_SUCCESS) {
    ptr = "CUDA driver: Unknown error.";
  }
  return ptr;
}
template <>
inline std::string getErrorString(cudaError_t err) {
  return cudaGetErrorString(err);
}
template <>
inline std::string getErrorString(nvmlReturn_t err) {
  return nvmlErrorString(err);
}
template <>
inline std::string getErrorString(cublasStatus_t err) {
  switch (err) {
    case CUBLAS_STATUS_SUCCESS:
      return "cuBLAS operation completed successfully. What are you doing here?";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "cuBLAS was not initialized.";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "cuBLAS internal resource allocation failed.";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "cuBLAS got an upsopported value or parameter.";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "cuBLAS feature is unavailable on the current device arch.";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "cuBLAS failed to access GPU memory space.";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "cuBLAS failed execute the GPU program or launch the kernel.";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "cuBLAS internal operation failed.";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "cuBLAS doen't support the requested functionality.";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "cuBLAS failed to check the current licencing.";
    default:
      return "cuBLAS unkown error.";
  }
}

template <>
inline std::string getErrorString(ncclResult_t err) {
  return ncclGetErrorString(err);
}
template <>
inline std::string getErrorString(cudnnStatus_t err) {
  return cudnnGetErrorString(err);
}
template <>
inline std::string getErrorString(curandStatus_t err) {
  switch (err) {
    case CURAND_STATUS_SUCCESS:
      return "cuRAND no errors.";
    case CURAND_STATUS_VERSION_MISMATCH:
      return "cuRAND header file and linked library version do not match.";
    case CURAND_STATUS_NOT_INITIALIZED:
      return "cuRAND generator not initialized.";
    case CURAND_STATUS_ALLOCATION_FAILED:
      return "cuRAND memory allocation failed.";
    case CURAND_STATUS_TYPE_ERROR:
      return "cuRAND generator is wrong type.";
    case CURAND_STATUS_OUT_OF_RANGE:
      return "cuRAND argument out of range.";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "cuRAND length requested is not a multple of dimension.";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "cuRAND GPU does not have double precision required by MRG32k3a.";
    case CURAND_STATUS_LAUNCH_FAILURE:
      return "cuRAND kernel launch failure.";
    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "cuRAND preexisting failure on library entry.";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "cuRAND initialization of CUDA failed.";
    case CURAND_STATUS_ARCH_MISMATCH:
      return "cuRAND architecture mismatch, GPU does not support requested feature.";
    case CURAND_STATUS_INTERNAL_ERROR:
      return "cuRAND Internal library error.";
    default:
      return "cuRAND unkown error.";
  }
}

// For HugeCTR own error types, it is up to users to define the msesage.
#define HCTR_OWN_THROW(EXPR, MSG)                                                    \
  do {                                                                               \
    HugeCTR::Error_t err_thr = (EXPR);                                               \
    if (err_thr != HugeCTR::Error_t::Success) {                                      \
      HugeCTR::Logger::get().do_throw(err_thr, CUR_SRC_LOC(EXPR), std::string(MSG)); \
    }                                                                                \
  } while (0);

#ifdef ENABLE_MPI
// Because MPI error code is in int, it is safe to have a separate macro for MPI,
// rather than reserving `int` as MPI error type.
// We don't want this set of macros to become another source of errors.
#define HCTR_MPI_THROW(EXPR)                                                 \
  do {                                                                       \
    const int err_code = (EXPR);                                             \
    if (err_code != MPI_SUCCESS) {                                           \
      char err_str[MPI_MAX_ERROR_STRING];                                    \
      int err_len = MPI_MAX_ERROR_STRING;                                    \
      MPI_Error_string(err_code, err_str, &err_len);                         \
      Logger::get().do_throw(Error_t::MpiError, CUR_SRC_LOC(EXPR), err_str); \
    }                                                                        \
  } while (0)
#endif

// For other library calls such as CUDA, cuBLAS and NCCL, use this macro
#define HCTR_LIB_THROW(EXPR)                                                 \
  do {                                                                       \
    auto ret_thr = (EXPR);                                                   \
    HugeCTR::Error_t err_type = HugeCTR::getErrorType(ret_thr);              \
    if (err_type != HugeCTR::Error_t::Success) {                             \
      std::string err_msg = HugeCTR::getErrorString(ret_thr);                \
      HugeCTR::Logger::get().do_throw(err_type, CUR_SRC_LOC(EXPR), err_msg); \
    }                                                                        \
  } while (0);

#define HCTR_THROW_IF(EXPR, ERROR, MSG)                                              \
  do {                                                                               \
    const auto& expr = (EXPR);                                                       \
    if (expr) {                                                                      \
      HugeCTR::Logger::get().do_throw((ERROR), CUR_SRC_LOC(EXPR), std::string(MSG)); \
    }                                                                                \
  } while (0)

#define CHECK_CALL(MODE) CHECK_##MODE##_CALL

#define CHECK_BLOCKING_CALL true
#define CHECK_ASYNC_CALL false

#define HCTR_CHECK(EXPR)                               \
  do {                                                 \
    const auto& expr = (EXPR);                         \
    if (!expr) {                                       \
      HugeCTR::Logger::get().abort(CUR_SRC_LOC(EXPR)); \
    }                                                  \
  } while (0)

#define HCTR_CHECK_HINT(EXPR, HINT, ...)                                      \
  do {                                                                        \
    const auto& expr = (EXPR);                                                \
    if (!expr) {                                                              \
      HugeCTR::Logger::get().abort(CUR_SRC_LOC(EXPR), (HINT), ##__VA_ARGS__); \
    }                                                                         \
  } while (0)

#define HCTR_DIE(HINT, ...) HCTR_CHECK_HINT(false, HINT, ##__VA_ARGS__)

// TODO: print the cuda error string
#define HCTR_CUDA_CHECK(SYNC_MODE, FUNC)                                       \
  do {                                                                         \
    auto ret_err = (FUNC);                                                     \
    if (CHECK_CALL(SYNC_MODE)) {                                               \
      ret_err = cudaDeviceSynchronize();                                       \
    }                                                                          \
    if (ret_err != cudaSuccess) {                                              \
      HugeCTR::Logger::get().check(ret_err == cudaSuccess, CUR_SRC_LOC(EXPR)); \
    }                                                                          \
  } while (0)

#ifndef NDEBUG
#define HCTR_ASSERT(EXPR)                                                       \
  do {                                                                          \
    HugeCTR::Logger::get().check_lazy([&] { return EXPR; }, CUR_SRC_LOC(EXPR)); \
  } while (0)
#else
#define HCTR_ASSERT(EXPR)
#endif

class Logger final {
 public:
  class DeferredEntry final {
   public:
    HCTR_DISALLOW_COPY_AND_MOVE(DeferredEntry);

    inline DeferredEntry(const Logger* logger, const int level, const bool with_prefix)
        : logger_{logger}, level_{level}, with_prefix_{with_prefix} {}

    ~DeferredEntry();

    template <typename... Args>
    inline DeferredEntry& append(Args&&... args) {
      if (logger_) {
        (os_ << ... << args);
      }
      return *this;
    }

    template <typename T>
    inline DeferredEntry& operator<<(const T& value) {
      if (logger_) {
        os_ << value;
      }
      return *this;
    }

    inline DeferredEntry& operator<<(std::ostream& (*fn)(std::ostream&)) {
      if (logger_) {
        fn(os_);
      }
      return *this;
    }

   private:
    const Logger* logger_;
    const int level_;
    const bool with_prefix_;
    std::ostringstream os_;
  };

  static constexpr size_t MAX_PREFIX_LENGTH = 96;

  static void print_exception(const std::exception& e, int depth);

  static Logger& get();

  HCTR_DISALLOW_COPY_AND_MOVE(Logger);

  ~Logger();

  inline bool can_log_at(const int level, const bool per_rank) const {
    return level != LOG_LEVEL(SILENCE) && level <= max_level_ && (rank_ == 0 || per_rank);
  }

  void log(int level, bool per_rank, bool with_prefix, const char* format, ...) const;

  DeferredEntry log(int level, bool per_rank, bool with_prefix) const;

  void abort(const SrcLoc& loc, const char* format = nullptr, ...) const;

  template <typename Condition>
  void check_lazy(const Condition& condition, const SrcLoc& loc) {
    if (condition() == false) {
      abort(loc);
    }
  }

  void do_throw(HugeCTR::Error_t error_type, const SrcLoc& loc, const std::string& message) const;

  inline int get_rank() const { return rank_; }

 private:
  Logger();

  FILE* get_file_stream(int level);

  size_t write_log_prefix(bool with_prefix, char (&buffer)[Logger::MAX_PREFIX_LENGTH],
                          int level) const;

 private:
  const int rank_;
  int max_level_{DEFAULT_LOG_LEVEL};
  bool log_to_std_{true};
  bool log_to_file_{false};

  std::map<int, FILE*> log_std_;
  std::map<int, FILE*> log_file_;
  std::map<int, std::string> level_name_;
};

bool hctr_has_thread_name();
const char* hctr_get_thread_name();
void hctr_set_thread_name(const char* name);
inline void hctr_set_thread_name(const std::string& name) {
  return hctr_set_thread_name(name.c_str());
}

}  // namespace HugeCTR
