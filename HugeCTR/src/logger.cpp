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

#include <logger.hpp>

#include <algorithm>
#include <chrono>
#include <common.hpp>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <exception>

#include <unistd.h>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {

std::unique_ptr<Logger> Logger::g_instance;
std::once_flag Logger::g_once_flag;

void Logger::print_exception(const std::exception& e, int depth) {
  Logger::get().log(LOG_ERROR_LEVEL, true, false, "%d. %s\n", depth, e.what());
  try {
    std::rethrow_if_nested(e);
  } catch (const std::exception& e) {
    print_exception(e, depth + 1);
  } catch (...) {}
}

Logger& Logger::get() {
  call_once(Logger::g_once_flag, []() { g_instance.reset(new Logger()); });
  return *(g_instance.get());
}

Logger::~Logger() {
  // if stdout and stderr are in use, we dont 'do fclose to prevent the situations where
  // (1) the fds are taken in opening other files or
  // (2) writing to the closed fds occurs, which is UB.
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
    FILE* file = log_file_.at(level);
    if (with_prefix) {
      fprintf(file, "%s", get_log_prefix(level).c_str());
    }
    va_list args;
    va_start(args, format);
    vfprintf(file, format, args);
    va_end(args);
    fflush(file);
  }
}

void Logger::check(bool condition, const SrcLoc& loc) const {
  if (condition == false) {
    log(-1, true, true,
        "Check Failed!\n"
        "\tFile: %s:%u\n"
        "\tFunction: %s\n"
        "\tExpression: %s\n",
        loc.file, loc.line, loc.func, loc.expr);
    std::abort();
  }
}

void Logger::do_throw(HugeCTR::Error_t error_type, const SrcLoc& loc,
                      const std::string& message) const {
  std::string error_message = "Runtime error: " + message + "\n" + "\t" +
                              loc.expr + " at " + loc.func + "(" + loc.file + ":" +
                              std::to_string(loc.line) + ")";
  std::throw_with_nested(internal_runtime_error(error_type, error_message));
}

Logger::Logger() : rank_(0), max_level_(DEFAULT_LOG_LEVEL), log_to_file_(false) {
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
    int log_to_file = 0;
    if (sscanf(log_to_file_str , "%d", &log_to_file) == 1) {
      log_to_file_ = static_cast<bool>(log_to_file);
    }
  }

  LEVEL_MAP(level_name_, ERROR);
  LEVEL_MAP(level_name_, SILENCE);
  LEVEL_MAP(level_name_, INFO);
  LEVEL_MAP(level_name_, WARNING);
  LEVEL_MAP(level_name_, DEBUG);

  log_file_[LOG_SILENCE_LEVEL] = nullptr;
  if (log_to_file_) {
    for (int level = LOG_ERROR_LEVEL; level <= max_level_; level++) {
      if (level != LOG_SILENCE_LEVEL) {
        std::string level_name = level_name_[level];
        std::transform(level_name.begin(), level_name.end(), level_name.begin(),
                       [](unsigned char ch) { return std::tolower(ch); });
        std::string log_fname = "hctr_" + std::to_string(getpid()) + "_" + std::to_string(rank_) + "_" + level_name + ".log";
        log_file_[level] = fopen(log_fname.c_str(), "w");
      }
    }
  }
  else {
    log_file_[LOG_ERROR_LEVEL] = stderr;
    for (int level = LOG_INFO_LEVEL; level <= max_level_; level++) {
      log_file_[level] = stdout;
    }
  }
}

std::string Logger::get_log_prefix(int level) const {
  using std::chrono::system_clock;

  system_clock::time_point time_now = system_clock::now();
  auto tt = system_clock::to_time_t(time_now);
  char time[18];
  std::strftime(time, sizeof(time), "%T", std::localtime(&tt));

  std::string prefix_str = std::string("[HUGECTR][") + std::string(time) + "]";

  std::string level_name;
  auto it = level_name_.find(level);
  if (it != level_name_.end()) {
    level_name = it->second;
  } else {
    level_name = "LEVEL" + std::to_string(level);
  }
  prefix_str += "[" + level_name + "]";

  prefix_str += "[RANK" + std::to_string(rank_) + "]";

  prefix_str += ": ";

  return prefix_str;
}

}  // namespace HugeCTR
