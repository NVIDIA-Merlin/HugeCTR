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

#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <base/debug/logger.hpp>
#include <chrono>
#include <common.hpp>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {

thread_local std::string THREAD_NAME;

const std::string& hctr_get_thread_name() { return THREAD_NAME; }

void hctr_set_thread_name(const std::string& name) { THREAD_NAME = name; }

void Logger::print_exception(const std::exception& e, int depth) {
  Logger::get().log(LOG_ERROR_LEVEL, true, false, "%d. %s\n", depth, e.what());
  try {
    std::rethrow_if_nested(e);
  } catch (const std::exception& e) {
    print_exception(e, depth + 1);
  } catch (...) {
  }
}

Logger& Logger::get() {
  static std::unique_ptr<Logger> instance;
  static std::once_flag once_flag;

  call_once(once_flag, []() { instance.reset(new Logger()); });
  return *instance;
}

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

void Logger::log(const int level, bool per_rank, bool with_prefix, const char* format, ...) const {
  if (level == LOG_SILENCE_LEVEL || level > max_level_) {
    return;
  }

  if (rank_ == 0 || per_rank) {
    std::string new_format(format);
    if (with_prefix) {
      new_format.insert(0, get_log_prefix(level));
    }

    if (log_to_std_) {
      va_list args;
      va_start(args, format);
      auto& file = log_std_.at(level);
      vfprintf(file, new_format.c_str(), args);
      va_end(args);
      fflush(file);
    }

    if (log_to_file_) {
      va_list args;
      va_start(args, format);
      auto& file = log_file_.at(level);
      vfprintf(file, new_format.c_str(), args);
      va_end(args);
      fflush(file);
    }
  }
}

DeferredLogEntry Logger::log(const int level, bool per_rank, bool with_prefix) const {
  if (level == LOG_SILENCE_LEVEL || level > max_level_) {
    return {true, [](std::ostringstream&) {}};
  } else if (rank_ == 0 || per_rank) {
    return {false, [level, with_prefix, this](std::ostringstream& ss) {
              if (log_to_std_) {
                auto& file = log_std_.at(level);
                if (with_prefix) {
                  fputs(get_log_prefix(level).c_str(), file);
                }
                fputs(ss.str().c_str(), file);
                fflush(file);
              }

              if (log_to_file_) {
                auto& file = log_file_.at(level);
                if (with_prefix) {
                  fputs(get_log_prefix(level).c_str(), file);
                }
                fputs(ss.str().c_str(), file);
                fflush(file);
              }
            }};
  } else {
    return {true, [](std::ostringstream&) {}};
  }
}

void Logger::check(bool condition, const SrcLoc& loc, const char* format, ...) const {
  if (condition == false) {
    if (format) {
      std::string hint;
      {
        va_list args;
        va_start(args, format);
        hint.resize(vsnprintf(nullptr, 0, format, args) + 1);
        va_end(args);
      }
      {
        va_list args;
        va_start(args, format);
        vsprintf(hint.data(), format, args);
        va_end(args);
      }
      log(-1, true, true,
          "Check Failed!\n"
          "\tFile: %s:%u\n"
          "\tFunction: %s\n"
          "\tExpression: %s\n"
          "\tHint: %s\n",
          loc.file, loc.line, loc.func, loc.expr, hint.c_str());
    } else {
      log(-1, true, true,
          "Check Failed!\n"
          "\tFile: %s:%u\n"
          "\tFunction: %s\n"
          "\tExpression: %s\n",
          loc.file, loc.line, loc.func, loc.expr);
    }
    std::abort();
  }
}

void Logger::do_throw(HugeCTR::Error_t error_type, const SrcLoc& loc,
                      const std::string& message) const {
  std::string error_message = "Runtime error: " + message + "\n" + "\t" + loc.expr + " at " +
                              loc.func + "(" + loc.file + ":" + std::to_string(loc.line) + ")";
  std::throw_with_nested(internal_runtime_error(error_type, error_message));
}

int Logger::get_rank() { return rank_; }

Logger::Logger() : rank_(0), max_level_(DEFAULT_LOG_LEVEL), log_to_std_(true), log_to_file_(false) {
  hctr_set_thread_name("main");

#ifdef ENABLE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
#endif

  const char* max_level_str = std::getenv("HUGECTR_LOG_LEVEL");
  if (max_level_str != nullptr && max_level_str[0] != '\0') {
    int max_level;
    if (sscanf(max_level_str, "%d", &max_level) == 1) {
      max_level_ = max_level;
    }
  }

  const char* log_to_file_str = std::getenv("HUGECTR_LOG_TO_FILE");
  if (log_to_file_str != nullptr && log_to_file_str[0] != '\0') {
    int log_to_file_val = 0;
    if (sscanf(log_to_file_str, "%d", &log_to_file_val) == 1) {
      log_to_std_ = log_to_file_val < 2;
      log_to_file_ = log_to_file_val > 0;
    }
  }

  LEVEL_MAP(level_name_, ERROR);
  LEVEL_MAP(level_name_, SILENCE);
  LEVEL_MAP(level_name_, INFO);
  LEVEL_MAP(level_name_, WARNING);
  LEVEL_MAP(level_name_, DEBUG);
  LEVEL_MAP(level_name_, TRACE);

  if (log_to_file_) {
    for (int level = LOG_ERROR_LEVEL; level <= max_level_; level++) {
      if (level != LOG_SILENCE_LEVEL) {
        std::string level_name = level_name_[level];
        std::transform(level_name.begin(), level_name.end(), level_name.begin(),
                       [](unsigned char ch) { return std::tolower(ch); });
        std::string log_fname = "hctr_" + std::to_string(getpid()) + "_" + std::to_string(rank_) +
                                "_" + level_name + ".log";
        log_file_[level] = fopen(log_fname.c_str(), "w");
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

std::string Logger::get_log_prefix(int level) const {
  std::ostringstream prefix;

  // Base & time
  prefix << "[HCTR][";
  {
    const time_t now = std::time(nullptr);
    std::tm now_local;
    localtime_r(&now, &now_local);

    // %H:%M:%S = [00-23]:[00-59]:[00-60] == e.g., 23:59:60 = 8 bytes + 1 zero terminate.
    // (60 = for second-time-shift years)
    char buffer[8 + 1];
    std::strftime(buffer, sizeof(buffer), "%T", &now_local);
    prefix << buffer;
  }

  // Level
  prefix << "][";
  {
    const auto level_it = level_name_.find(level);
    if (level_it != level_name_.end()) {
      prefix << level_it->second;
    } else {
      prefix << "LEVEL" << level;
    }
  }

  // Rank
  prefix << "][RK" << rank_;

  // Thread
  prefix << "][";
  const std::string& thread_name = hctr_get_thread_name();
  if (thread_name.empty()) {
    prefix << "tid #" << std::this_thread::get_id();
  } else {
    prefix << thread_name;
  }

  // Prompt & return.
  prefix << "]: ";
  return prefix.str();
}

}  // namespace HugeCTR
