/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "HugeCTR/include/metadata.hpp"
#include "nlohmann/json.hpp"
#include <iostream>

namespace HugeCTR {

void Metadata::get_parquet_metadata(std::string file_name) {

  nlohmann::json config;
  std::ifstream file_stream(file_name);
  if (!file_stream.is_open()) {
    CK_THROW_(Error_t::FileCannotOpen, "file_stream.is_open() failed: " + file_name);
  }
  file_stream >> config;

  try {
    auto fstats = config.find("file_stats").value();
    for (unsigned int i=0; i<fstats.size(); i++) {
      FileStats fs;
      std::string fname = (std::string)fstats[i].find("file_name").value();
      fs.num_rows = long(fstats[i].find("num_rows").value());
	  file_stats_.insert({fname, fs});
    }

    auto cats = config.find("cats").value();
    for (unsigned int i=0; i<cats.size(); i++) {
      Cols c;
      c.col_name = (std::string)cats[i].find("col_name").value();
      c.index = (int)cats[i].find("index").value();
      this->cat_names_.push_back(c);
    }

    auto conts = config.find("conts").value();
    for (unsigned int i=0; i<conts.size(); i++) {
      Cols c;
      c.col_name = (std::string)conts[i].find("col_name").value();
      c.index = (int)conts[i].find("index").value();
      this->cont_names_.push_back(c);
    }

    auto labels = config.find("labels").value();
    for (unsigned int i=0; i<labels.size(); i++) {
      Cols c;
      c.col_name = (std::string)labels[i].find("col_name").value();
      c.index = (int)labels[i].find("index").value();
      this->label_names_.push_back(c);
    }
    loaded_ = true;
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }

  file_stream.close();
}

}  // namespace HugeCTR
