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

#include <parser.hpp>

namespace HugeCTR {

InferenceParser::InferenceParser(const nlohmann::json& config) {
  auto j = get_json(config, "inference");
  if (has_key_(j, "max_batchsize")) {
    max_batchsize = get_value_from_json<unsigned long long>(j, "max_batchsize");
  } else {
    max_batchsize = 1024;
  }

  bool has_dense_model_file = has_key_(j, "dense_model_file");
  bool has_sparse_model_files = has_key_(j, "sparse_model_file");
  if (!(has_dense_model_file && has_sparse_model_files)) {
    CK_THROW_(Error_t::WrongInput, "dense_model_file and sparse_model_file must be specified");
  }
  dense_model_file = get_value_from_json<std::string>(j, "dense_model_file");
  auto j_sparse_model_files = get_json(j, "sparse_model_file");
  if (j_sparse_model_files.is_array()) {
    for (auto j_embedding_tmp : j_sparse_model_files) {
      sparse_model_files.push_back(j_embedding_tmp.get<std::string>());
    }
  } else {
    sparse_model_files.push_back(get_value_from_json<std::string>(j, "sparse_model_file"));
  }

  if (has_key_(j, "mixed_precision")) {
      use_mixed_precision = true;
      int i_scaler = get_value_from_json<int>(j, "mixed_precision");
      if (i_scaler != 128 && i_scaler != 256 && i_scaler != 512 && i_scaler != 1024) {
        CK_THROW_(Error_t::WrongInput,
                  "Scaler of mixed_precision training should be either 128/256/512/1024");
      }
      scaler = i_scaler;
      std::stringstream ss;
      ss << "Mixed Precision training with scaler: " << i_scaler << " is enabled." << std::endl;
      MESSAGE_(ss.str());

    } else {
      use_mixed_precision = false;
      scaler = 1.f;
    }
    
    use_algorithm_search = get_value_from_json_soft<bool>(j, "algorithm_search", true);
    use_cuda_graph = get_value_from_json_soft<bool>(j, "cuda_graph", true);
}

}  // namespace HugeCTR
