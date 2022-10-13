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

#include <base/debug/logger.hpp>
#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace HugeCTR {

class IOUtils {
 public:
  static std::string get_path_scheme(const std::string& path) {
    auto first_colon = path.find_first_of(":");
    if (first_colon == std::string::npos) {
      return "";
    }
    std::string scheme = path.substr(0, first_colon);
    return scheme;
  }

  static std::string get_fs_type_from_json(const std::string& config_path) {
    nlohmann::json config;
    std::ifstream file_stream(config_path);
    if (!file_stream.is_open()) {
      HCTR_OWN_THROW(Error_t::FileCannotOpen, "file_stream.is_open() failed: " + config_path);
    }
    try {
      file_stream >> config;
    } catch (const std::runtime_error& rt_err) {
      HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    }
    HCTR_CHECK_HINT(config.contains("fs_type"),
                    "The user provided config file does not contain fs_type");
    HCTR_CHECK_HINT((std::string)config.find("fs_type").value() != "",
                    "The user provided config file does not contain fs_type");
    return (std::string)config.find("fs_type").value();
  }

  static std::string get_parent_dir(const std::string& path) {
    auto last_forward_slash = path.find_last_of('/');
    if (last_forward_slash != std::string::npos) {
      return path.substr(0, last_forward_slash);
    }
    return "";
  }

  static bool is_local_path(const std::string& path) { return get_path_scheme(path) == ""; }

  static bool is_valid_s3_https_url(const std::string& url) {
    // TODO: add the correct logic when enable aws s3
    return true;
  }
};
}  // namespace HugeCTR