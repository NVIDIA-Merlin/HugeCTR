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

#ifndef WRAPPER_VARIABLES_V2_H
#define WRAPPER_VARIABLES_V2_H

#include "embedding_wrapper.h"

namespace HugeCTR {
namespace Version2 {

extern std::string key_type;
extern std::string value_type;

extern const std::set<std::string> KEY_TYPE_SET;
extern const std::set<std::string> VALUE_TYPE_SET;
extern const std::set<float> SCALER_SET;
extern const std::map<std::string, HugeCTR::Embedding_t> EMBEDDING_TYPE_MAP;
extern const std::map<std::string, HugeCTR::Optimizer_t> OPTIMIZER_TYPE_MAP;
extern const std::map<std::string, int> COMBINER_MAP;
extern const std::map<std::string, HugeCTR::Update_t> UPDATE_TYPE_MAP;

extern std::unique_ptr<HugeCTR::Version2::Wrapper> wrapper;

} // namespace Version2
} // namespace HugeCTR
#endif // WRAPPER_VARIABLES_V2_H