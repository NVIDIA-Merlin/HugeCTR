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

#include <inference/inference_utils.hpp>
#include <inference/parameter_server.hpp>

namespace HugeCTR {
template <typename TypeHashKey>
HugectrUtility<TypeHashKey>::HugectrUtility(){}

template <typename TypeHashKey>
HugectrUtility<TypeHashKey>::~HugectrUtility(){}

template <typename TypeHashKey>
HugectrUtility<TypeHashKey>* HugectrUtility<TypeHashKey>::Create_Parameter_Server(INFER_TYPE Infer_type,
                                                                                const std::vector<std::string>& model_config_path,
                                                                                const std::vector<InferenceParams>& inference_params_array){
  HugectrUtility<TypeHashKey>* new_parameter_server;

  switch(Infer_type){
    case TRITON:
      new_parameter_server = new parameter_server<TypeHashKey>("TRITON", model_config_path, inference_params_array);
      break;
    default:
      CK_THROW_(Error_t::WrongInput, "Error: unknown framework name.");
      break;
  }

  return new_parameter_server;
}

template class HugectrUtility<unsigned int>;
template class HugectrUtility<long long>;
}  // namespace HugeCTR
