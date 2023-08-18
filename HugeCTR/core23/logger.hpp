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
 * level N >= 4 as well. The message level is specified as the first argument of log function.
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
 * The thrown exception records where the error has occurred (or caught) and what the error is
 about.
 * If you add a new library, to track its error, it is recommended that you specialize getErrorType
 and getErrorString below.
 * 2.1. Examples:
     HCTR_OWN_THROW(Error_t::WrongInput, "device is not available");
     HCTR_LIB_THROW(cudaDeviceSynchronize());
     HCTR_LIB_THROW(cublasGemmEx(...));
     HCTR_MPI_THROW(MPI_Gather(...));

 * If you want to print the nested exception message at a catch statement, call
 'Logger::get().print(e)'.
 * Then, they will be printed at the ERROR level.
 *
 * 3. Error check:
 * You sometimes want to terminate the HugeCTR immediately rather than throwing an exception.
 * Like the HCTR_*_THROW, the error message shows where the error has occurred and which expression
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

#include <cerrno>
#include <core/macro.hpp>
#include <core23/error.hpp>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

#ifdef HCTR_ASSERT_
#error HCTR_ASSERT_ already defined. Potential naming conflict!
#endif
#ifdef NDEBUG
#define HCTR_ASSERT_(EXPR) \
  do {                     \
  } while (0)
#else
#define HCTR_ASSERT_(EXPR) HCTR_CHECK_(EXPR, "Assertion.")
#endif

#ifdef HCTR_CHECK_
#error HCTR_CHECK_ already defined. Potential naming conflict!
#endif
#define HCTR_CHECK_(EXPR, ...)                                                                   \
  do {                                                                                           \
    if (!(EXPR)) {                                                                               \
      HugeCTR::Logger::get().print_error("Expression check failed!", HCTR_CODE_REFERENCE_(EXPR), \
                                         HugeCTR::core23::hctr_render_string(__VA_ARGS__));      \
      std::abort();                                                                              \
    }                                                                                            \
  } while (0)

#ifdef HCTR_LIB_CHECK_
#error HCTR_LIB_CHECK_ already defined. Potential naming conflict!
#endif
#define HCTR_LIB_CHECK_(EXPR)                                                                \
  do {                                                                                       \
    const auto _expr_eval = (EXPR);                                                          \
    const auto _expr_eval_err = HugeCTR::core23::to_error(_expr_eval);                       \
    if (_expr_eval_err != HugeCTR::Error_t::Success) {                                       \
      HugeCTR::Logger::get().print_error("Library call failed!", HCTR_CODE_REFERENCE_(EXPR), \
                                         HugeCTR::core23::hctr_render_string(_expr_eval));   \
      std::abort();                                                                          \
    }                                                                                        \
  } while (0)

#ifdef HCTR_SHARP_CHECK_
#error HCTR_SHARP_CHECK_
#endif
#define HCTR_SHARP_CHECK_(EXPR)                                                            \
  do {                                                                                     \
    const auto _expr_eval = (EXPR);                                                        \
    static_assert(std::is_same<decltype(_expr_eval), const int>::value);                   \
    if (_expr_eval != SHARP_COLL_SUCCESS) {                                                \
      HugeCTR::Logger::get().print_error("SHARP call failed!", HCTR_CODE_REFERENCE_(EXPR), \
                                         sharp_coll_strerror(_expr_eval));                 \
      std::abort();                                                                        \
    }                                                                                      \
  } while (0)

/**
 * Legacy macros.
 */
#ifdef HCTR_CHECK
#error HCTR_CHECK already defined. Potential naming conflict!
#endif
#define HCTR_CHECK(EXPR) HCTR_CHECK_(EXPR)

#ifdef HCTR_ASSERT
#error HCTR_ASSERT already defined. Potential naming conflict!
#endif
#define HCTR_ASSERT(EXPR) HCTR_ASSERT_(EXPR)

#ifdef HCTR_CHECK_HINT
#error HCTR_CHECK_HINT already defined. Potential naming conflict!
#endif
#define HCTR_CHECK_HINT(EXPR, ...) HCTR_CHECK_(EXPR, __VA_ARGS__)

#ifdef HCTR_DIE
#error HCTR_DIE already defined. Potential naming conflict!
#endif
#define HCTR_DIE(...) HCTR_CHECK_(false, __VA_ARGS__)

namespace HugeCTR {

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

#define HCTR_LOG(NAME, TYPE, ...) HCTR_LOG_AT(LOG_LEVEL(NAME), TYPE, __VA_ARGS__)

#define HCTR_LOG_AT(LEVEL, TYPE, ...)                \
  do {                                               \
    HugeCTR::Logger& logger{HugeCTR::Logger::get()}; \
    if (logger.enabled_at(LEVEL, LOG_RANK(TYPE))) {  \
      logger.printf(LEVEL, true, __VA_ARGS__);       \
    }                                                \
  } while (0)

#define HCTR_LOG_S(NAME, TYPE) \
  HugeCTR::LogEntry(HugeCTR::Logger::get(), LOG_LEVEL(NAME), LOG_RANK(TYPE), true)

#define HCTR_LOG_C(NAME, TYPE, ...)                                                          \
  do {                                                                                       \
    HugeCTR::Logger& logger{HugeCTR::Logger::get()};                                         \
    if (logger.enabled_at(LOG_LEVEL(NAME), LOG_RANK(TYPE))) {                                \
      logger.print(LOG_LEVEL(NAME), true, HugeCTR::core23::hctr_render_string(__VA_ARGS__)); \
    }                                                                                        \
  } while (0)

#define HCTR_PRINT(NAME, ...) HCTR_PRINT_AT(LOG_LEVEL(NAME), __VA_ARGS__)

#define HCTR_PRINT_AT(LEVEL, ...)                    \
  do {                                               \
    HugeCTR::Logger& logger{HugeCTR::Logger::get()}; \
    if (logger.enabled_at(LEVEL, LOG_RANK_ROOT)) {   \
      logger.printf(LEVEL, false, __VA_ARGS__);      \
    }                                                \
  } while (0)

#define CHECK_CALL(MODE) CHECK_##MODE##_CALL

#define CHECK_BLOCKING_CALL true
#define CHECK_ASYNC_CALL false

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

/**
 * The logger class shouldn't be used directly. Instead use the below HCTR_LOG_* and
 * HCTR_*_CHECK_* macros.
 */
class Logger final {
 public:
  HCTR_DISALLOW_COPY_AND_MOVE(Logger);

  static constexpr int default_level{
#ifndef NDEBUG
      LOG_LEVEL(WARNING)
#else
      LOG_LEVEL(DEBUG)
#endif
  };
  static constexpr size_t MAX_PREFIX_LENGTH = 96;

  inline int get_rank() const { return rank_; }

  ~Logger();

  inline bool enabled_at(const int level, const bool per_rank) const {
    return level != LOG_LEVEL(SILENCE) && level <= max_level_ && (rank_ == 0 || per_rank);
  }

  void print(const std::exception& e, size_t depth = 0);

  inline void print(const int level, const bool with_prefix, const char* const message) {
    printf(level, with_prefix, "%s", message);
  }

  inline void print(const int level, const bool with_prefix, const std::string& message) {
    printf(level, with_prefix, "%s", message.c_str());
  }

  void printf(int level, bool with_prefix, const char* format, ...)
      __attribute__((format(printf, 4, 5)));

  inline void print_error(const char* const reason, const core23::CodeReference& ref,
                          const char* const hint) {
    printf(LOG_LEVEL(ERROR), true,
           "%s\n\tFile: %s:%zu\n\tFunction: %s\n\tExpression: %s\n\tHint: %s\n", reason, ref.file,
           ref.line, ref.function, ref.expression, hint);
  }

  inline void print_error(const char* const reason, const core23::CodeReference& ref,
                          const std::string& hint) {
    print_error(reason, ref, hint.c_str());
  }

 private:
  Logger();

  size_t write_log_prefix(bool with_prefix, char (&buffer)[Logger::MAX_PREFIX_LENGTH],
                          int level) const;

 private:
  int rank_;
  int max_level_{default_level};
  bool log_to_std_{true};
  bool log_to_file_{false};

  std::map<int, FILE*> log_std_;
  std::map<int, FILE*> log_file_;
  std::map<int, std::string> level_name_;

 public:
  static Logger& get();

  static bool has_thread_name();
  static const char* get_thread_name();
  static void set_thread_name(const char* name);
  static void set_thread_name(const std::string& name);
};

class LogEntry final {
 public:
  HCTR_DISALLOW_COPY_AND_MOVE(LogEntry);

  inline LogEntry(Logger& logger, const int level, const bool per_rank, const bool with_prefix)
      : logger_{logger.enabled_at(level, per_rank) ? &logger : nullptr},
        level_{level},
        with_prefix_{with_prefix} {}

  inline ~LogEntry() {
    if (logger_) {
      logger_->print(level_, with_prefix_, message_.str());
    }
  }

  template <typename... Args>
  inline LogEntry& append(Args&&... args) {
    if (logger_) {
      (message_ << ... << args);
    }
    return *this;
  }

  template <typename T>
  inline LogEntry& operator<<(const T& value) {
    if (logger_) {
      message_ << value;
    }
    return *this;
  }

  inline LogEntry& operator<<(std::ostream& (*fn)(std::ostream&)) {
    if (logger_) {
      fn(message_);
    }
    return *this;
  }

 private:
  Logger* const logger_;
  const int level_;
  const bool with_prefix_;
  std::ostringstream message_;
};

}  // namespace HugeCTR
