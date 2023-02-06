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
#include <cstring>
#include <string>

#include "NvInferPlugin.h"

namespace nvinfer1 {

namespace plugin {

typedef enum {
  STATUS_SUCCESS = 0,
  STATUS_FAILURE = 1,
  STATUS_BAD_PARAM = 2,
  STATUS_NOT_SUPPORTED = 3,
  STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

template <typename T>
void write(char*& buffer, const T& val) {
  std::memcpy(buffer, &val, sizeof(T));
  buffer += sizeof(T);
}

template <typename T>
T read(const char*& buffer) {
  T val{};
  std::memcpy(&val, buffer, sizeof(T));
  buffer += sizeof(T);
  return val;
}

void write_string(char*& buffer, const std::string& val) {
  std::memcpy(buffer, val.data(), val.size());
  buffer += val.size();
}

std::string read_string(const char*& buffer, size_t str_size) {
  std::string val(str_size, 0);
  std::memcpy(val.data(), buffer, str_size);
  buffer += str_size;
  return val;
}

void validateRequiredAttributesExist(std::set<std::string> requiredFieldNames,
                                     PluginFieldCollection const* fc) {
  for (int32_t i = 0; i < fc->nbFields; i++) {
    requiredFieldNames.erase(fc->fields[i].name);
  }
  if (!requiredFieldNames.empty()) {
    auto log = HCTR_LOG_S(ERROR, WORLD);
    log << "PluginFieldCollection missing required fields: {";
    char const* seperator = "";
    for (auto const& field : requiredFieldNames) {
      log << seperator << field;
      seperator = ", ";
    }
    log << "}" << std::endl;
  }
  HCTR_CHECK_HINT(requiredFieldNames.empty(), "There are unspecified plugin fields");
}

}  // namespace plugin

}  // namespace nvinfer1