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

#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <core23/logger.hpp>
#ifdef ENABLE_MPI
#include <core23/mpi_init_service.hpp>
#endif
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

namespace HugeCTR {

Logger::~Logger() {
  // if stdout and stderr are in use, we don't do fclose to prevent the situations where
  //   (1) the fds are taken in opening other files or
  //   (2) writing to the closed fds occurs, which is UB.
  // Due to the same reason, we don't wrap FILE* with a smart pointer.
  if (log_to_file_) {
    for (int level = LOG_ERROR_LEVEL; level <= max_level_; level++) {
      if (level != LOG_SILENCE_LEVEL) {
        fclose(log_file_[level]);
      }
    }
  }
}

void Logger::print(const std::exception& e, const size_t depth) {
  printf(LOG_ERROR_LEVEL, false, "%zu. %s\n", depth, e.what());
  try {
    std::rethrow_if_nested(e);
  } catch (const std::exception& e) {
    print(e, depth + 1);
  } catch (...) {
  }
}

void Logger::printf(const int level, bool with_prefix, const char* format, ...) {
  char prefix[MAX_PREFIX_LENGTH];
  write_log_prefix(with_prefix, prefix, level);

  if (log_to_std_) {
    std::va_list args;
    va_start(args, format);

    FILE* const file = log_std_.at(level);
    std::fputs(prefix, file);
    std::vfprintf(file, format, args);
    std::fflush(file);

    va_end(args);
  }

  if (log_to_file_) {
    std::va_list args;
    va_start(args, format);

    FILE* const file = log_file_.at(level);
    std::fputs(prefix, file);
    std::vfprintf(file, format, args);
    std::fflush(file);

    va_end(args);
  }
}

#ifdef HCTR_LEVEL_MAP_
#error HCTR_LEVEL_MAP_ already defined!
#else
#define HCTR_LEVEL_MAP_(MAP, NAME) MAP[LOG_LEVEL(NAME)] = #NAME
#endif

Logger::Logger() : rank_{0} {
#ifdef ENABLE_MPI
  rank_ = core23::MpiInitService::get().world_rank();
#endif
  set_thread_name("main");

  const char* const max_level_str = std::getenv("HUGECTR_LOG_LEVEL");
  if (max_level_str != nullptr && max_level_str[0] != '\0') {
    int max_level;
    if (std::sscanf(max_level_str, "%d", &max_level) == 1) {
      max_level_ = max_level;
    }
  }

  const char* const log_to_file_str = std::getenv("HUGECTR_LOG_TO_FILE");
  if (log_to_file_str != nullptr && log_to_file_str[0] != '\0') {
    int log_to_file_val = 0;
    if (std::sscanf(log_to_file_str, "%d", &log_to_file_val) == 1) {
      log_to_std_ = log_to_file_val < 2;
      log_to_file_ = log_to_file_val > 0;
    }
  }

  HCTR_LEVEL_MAP_(level_name_, ERROR);
  HCTR_LEVEL_MAP_(level_name_, SILENCE);
  HCTR_LEVEL_MAP_(level_name_, INFO);
  HCTR_LEVEL_MAP_(level_name_, WARNING);
  HCTR_LEVEL_MAP_(level_name_, DEBUG);
  HCTR_LEVEL_MAP_(level_name_, TRACE);

  if (log_to_file_) {
    for (int level = LOG_ERROR_LEVEL; level <= max_level_; level++) {
      if (level != LOG_SILENCE_LEVEL) {
        std::string level_name = level_name_[level];
        std::transform(level_name.begin(), level_name.end(), level_name.begin(),
                       [](unsigned char ch) { return std::tolower(ch); });
        std::ostringstream log_fname;
        log_fname << "hctr_" << getpid() << '_' << rank_ << '_' << level_name << ".log";
        log_file_[level] = std::fopen(log_fname.str().c_str(), "w");
      } else {
        log_file_[LOG_SILENCE_LEVEL] = nullptr;
      }
    }
  }

  if (log_to_std_) {
    log_std_[LOG_ERROR_LEVEL] = stderr;
    log_std_[LOG_SILENCE_LEVEL] = nullptr;
    for (int level = LOG_INFO_LEVEL; level <= max_level_; level++) {
      log_std_[level] = stdout;
    }
  }
}

size_t Logger::write_log_prefix(const bool with_prefix, char (&buffer)[Logger::MAX_PREFIX_LENGTH],
                                const int level) const {
  if (!with_prefix) {
    buffer[0] = '\0';
    return 0;
  }

  // "[HCTR][08:00:57.622][WARNING][RK0][redis background thread]: " << typical
  // "[HCTR][08:00:57.622][LV2147483647][RK2147483647][1234567890123456789012345678901]: " << worst
  //  ^         ^         ^         ^         ^         ^         ^         ^         ^         ^
  //            1         2         3         4         5         6         7         8         9
  //  0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890
  char* p = buffer;

  // HCTR prefix + Time.
  {
    struct timeval now;
    gettimeofday(&now, nullptr);
    std::tm now_local;
    localtime_r(&now.tv_sec, &now_local);

    // %H:%M:%S = [00-23]:[00-59]:[00-60] == e.g., 23:59:60 = 8 bytes + 1 zero terminate.
    // (60 = for second-time-shift years)
    p += std::strftime(p, sizeof(buffer), "[HCTR][%T", &now_local);
    p += std::sprintf(p, ".%03ld][", now.tv_usec / 1000);
  }

  // Level
  {
    const auto& level_it = level_name_.find(level);
    if (level_it != level_name_.end()) {
      const size_t n = level_it->second.size();
      level_it->second.copy(p, n);
      p += n;
    } else {
      p += std::sprintf(p, "LV%d", level);
    }
  }

  // Assign thread name if not already set.
  if (!has_thread_name()) {
    std::ostringstream os;
    os << "tid #" << std::this_thread::get_id();
    set_thread_name(os.str());
  }

  // Thread + Rank + Prompt
  p += std::sprintf(p, "][RK%d][%s]: ", rank_, get_thread_name());

  return p - buffer;
}

Logger& Logger::get() {
  // That is sufficient in C++-11 and later. See also
  // https://en.cppreference.com/w/cpp/language/storage_duration#Static_local_variables
  // or
  // https://stackoverflow.com/questions/8102125/is-local-static-variable-initialization-thread-safe-in-c11
  // or
  // https://stackoverflow.com/questions/1270927/are-function-static-variables-thread-safe-in-gcc/1270948
  // .
  static Logger instance;
  return instance;
}

static thread_local std::string thread_name_;

bool Logger::has_thread_name() { return !thread_name_.empty(); }
const char* Logger::get_thread_name() { return thread_name_.c_str(); }
void Logger::set_thread_name(const char* const name) { thread_name_ = name; }
void Logger::set_thread_name(const std::string& name) { thread_name_ = name; }

}  // namespace HugeCTR
