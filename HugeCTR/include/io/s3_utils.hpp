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

#ifdef ENABLE_S3
#include <aws/core/Aws.h>
#include <aws/core/utils/StringUtils.h>
#endif

#include <base/debug/logger.hpp>
#include <string>
#include <string_view>

namespace HugeCTR {

#ifdef ENABLE_S3

class S3Utils {
 public:
  /**
   * @brief Turn std::string into Aws::String
   *
   * @param str
   * @return Aws::String
   */
  static Aws::String to_aws_string(const std::string& str) {
    return Aws::String(str.begin(), str.end());
  }
};

/**
 * @brief Split the path string by separator
 *
 * @param str
 * @param sep
 * @return std::vector<std::string>
 */
inline std::vector<std::string> split_str(std::string& str, const std::string& sep) {
  std::vector<std::string> res;
  size_t pos = 0;
  std::string token;
  while ((pos = str.find(sep)) != std::string::npos) {
    token = str.substr(0, pos);
    res.push_back(token);
    str.erase(0, pos + sep.length());
  }
  res.push_back(str);
  return res;
}

/**
 * @brief A stuct used to represent a path in s3 file system
 *
 */
struct S3Path {
  std::string region = "";
  std::string bucket = "";
  std::string key = "";

  static S3Path FromString(const std::string& s) {
    std::string_view sv(s);
    auto exist_colon = sv.find_first_of(':');
    HCTR_CHECK_HINT(exist_colon != std::string::npos,
                    "This is not a valid s3 path. Please provide a correct S3 object url.");
    std::string_view scheme = sv.substr(0, exist_colon);
    HCTR_CHECK_HINT(scheme == "https" || scheme == "s3",
                    "The path format is not correct. Please provide either a https url or s3 url.");
    std::string_view body = sv.substr(exist_colon + 3);
    auto first_slash = body.find_first_of('/');
    if (scheme == "s3" || scheme == "S3") {
      if (first_slash == std::string::npos) {
        return S3Path{"", std::string(body), ""};
      } else {
        return S3Path{"", std::string(body.substr(0, first_slash)),
                      std::string(body.substr(first_slash + 1))};
      }
    } else {
      S3Path path;
      auto host_end = body.find_first_of("/");
      HCTR_CHECK_HINT(host_end != std::string::npos,
                      "This is not a valid s3 path. Please provide a correct S3 object url.");
      std::string_view host = body.substr(0, host_end);
      std::string_view remaining = body.substr(host_end + 1);
      std::string host_cpy = std::string(host);
      std::vector<std::string> dot_sep_parts = split_str(host_cpy, ".");
      if (dot_sep_parts.size() == 4 && dot_sep_parts[0] == "s3" &&
          dot_sep_parts[2] == "amazonaws" && dot_sep_parts[3] == "com") {
        path.region = dot_sep_parts[1];
        auto bucket_key_split = remaining.find_first_of('/');
        HCTR_CHECK_HINT(bucket_key_split != std::string::npos,
                        "This is not a valid s3 path. Please provide a correct S3 object url.");
        path.bucket = std::string(remaining.substr(0, bucket_key_split));
        path.key = std::string(remaining.substr(bucket_key_split + 1));
      } else if (dot_sep_parts.size() == 5 && dot_sep_parts[1] == "s3" &&
                 dot_sep_parts[3] == "amazonaws" && dot_sep_parts[4] == "com") {
        path.region = std::string(dot_sep_parts[2]);
        path.bucket = std::string(dot_sep_parts[0]);
        path.key = std::string(remaining);
      } else {
        HCTR_OWN_THROW(Error_t::FileCannotOpen, "This is not a valid AWS S3 path: " + s);
      }
      return path;
    }
  }

  Aws::String to_aws_string() const {
    Aws::String res(bucket.begin(), bucket.end());
    res.reserve(bucket.size() + key.size() + 1);
    res += '/';
    res.append(key.begin(), key.end());
    return res;
  }

  bool has_bucket_and_key() const { return bucket != "" && key != ""; }
};

inline std::string get_region_from_url(const std::string& str) {
  S3Path s3_path = S3Path::FromString(str);
  HCTR_CHECK_HINT(s3_path.region != "", "This S3 url does not contain region information.");
  return s3_path.region;
}
#endif
}  // namespace HugeCTR