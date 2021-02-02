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

#include "wrapper_variables.h"

namespace wrapper_vars {

std::unique_ptr<HugeCTR::Wrapper> wrapper;
std::string key_type;
std::string value_type;

const std::set<std::string> KEY_TYPE_SET = {"uint32", "int64"};
const std::set<std::string> VALUE_TYPE_SET = {"float", "half"};
const std::set<float> SCALER_SET = {1.0, 128.0, 256.0, 512.0, 1024.0};
const std::map<std::string, HugeCTR::Embedding_t> EMBEDDING_TYPE_MAP = {
    {"distributed", HugeCTR::Embedding_t::DistributedSlotSparseEmbeddingHash},
    {"localized", HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingHash}};
const std::map<std::string, HugeCTR::Optimizer_t> OPTIMIZER_TYPE_MAP = {
    {"Adam", HugeCTR::Optimizer_t::Adam},
    {"MomentumSGD", HugeCTR::Optimizer_t::MomentumSGD},
    {"Nesterov", HugeCTR::Optimizer_t::Nesterov},
    {"SGD", HugeCTR::Optimizer_t::SGD}};
const std::map<std::string, int> COMBINER_MAP = {
    {"sum", 0}, {"mean", 1}};
const std::map<std::string, HugeCTR::Update_t> UPDATE_TYPE_MAP = {
    {"Local", HugeCTR::Update_t::Local},
    {"Global", HugeCTR::Update_t::Global},
    {"LazyGlobal", HugeCTR::Update_t::LazyGlobal}};

} // namespace wrapper_vars