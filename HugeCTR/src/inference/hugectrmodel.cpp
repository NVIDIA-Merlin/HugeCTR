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

#include "HugeCTR/include/inference/hugectrmodel.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optimizer.hpp>
#include <parser.hpp>
#include <utils.hpp>

#include "HugeCTR/include/inference/session_inference.hpp"

namespace HugeCTR {
HugeCTRModel::HugeCTRModel() {}

HugeCTRModel::~HugeCTRModel() {}

HugeCTRModel* HugeCTRModel::load_model(INFER_TYPE Infer_type, const std::string& model_config_path,
                                       const InferenceParams& inference_params,
                                       std::shared_ptr<embedding_interface>& embedding_cache) {
  HugeCTRModel* model;

  switch (Infer_type) {
    case TRITON:
      model = new InferenceSession(model_config_path, inference_params, embedding_cache);
      break;
    default:
      std::cout << "wrong type!" << std::endl;
      return NULL;
  }

  return model;
}
}  // namespace HugeCTR
