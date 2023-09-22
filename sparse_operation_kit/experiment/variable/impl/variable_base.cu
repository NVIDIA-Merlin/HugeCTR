/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <nlohmann/json.hpp>

#include "variable/impl/det_variable.h"
#include "variable/impl/hkv_variable.h"
#include "variable/impl/variable_base.h"

namespace sok {

template <typename KeyType, typename ValueType>
std::shared_ptr<VariableBase<KeyType, ValueType>> VariableFactory::create(
    int64_t rows, int64_t cols, const std::string &type, const std::string &initializer,
    const std::string &config, cudaStream_t stream) {
  if (type == "hbm") {
    return std::make_shared<DETVariable<KeyType, ValueType>>(cols, 2E4, initializer, stream);
  }
  if (type == "hybrid") {
    nlohmann::json config_json = nlohmann::json::parse(config);
    int64_t init_capacity = 64 * 1024 * 1024UL;  ///< The initial capacity of the hash table.
    auto init_capacity_it = config_json.find("init_capacity");
    if (init_capacity_it != config_json.end()) {
      init_capacity = init_capacity_it->get<int64_t>();
    }
    int64_t max_capacity = 64 * 1024 * 1024UL;  ///< The maximum capacity of the hash table.
    auto max_capacity_it = config_json.find("max_capacity");
    if (max_capacity_it != config_json.end()) {
      max_capacity = max_capacity_it->get<int64_t>();
    }
    size_t max_hbm_for_vectors = 16;  ///< The maximum HBM for vectors, in giga-bytes.
    auto max_hbm_for_vectors_it = config_json.find("max_hbm_for_vectors");
    if (max_hbm_for_vectors_it != config_json.end()) {
      max_hbm_for_vectors = max_hbm_for_vectors_it->get<size_t>();
    }
    float max_load_factor = 0.5f;  ///< The max load factor before rehashing.
    auto max_load_factor_it = config_json.find("max_load_factor");
    if (max_load_factor_it != config_json.end()) {
      max_load_factor = max_load_factor_it->get<float>();
    }

    size_t max_bucket_size = 128;  ///< The length of each bucket.
    auto max_bucket_size_it = config_json.find("max_bucket_size");
    if (max_bucket_size_it != config_json.end()) {
      max_bucket_size = max_bucket_size_it->get<float>();
    }
    int block_size = 1024;  ///< The default block size for CUDA kernels.
    auto block_size_it = config_json.find("block_size");
    if (block_size_it != config_json.end()) {
      block_size = block_size_it->get<int>();
    }
    int device_id = -1;  ///< The ID of device.
    auto device_id_it = config_json.find("device_id");
    if (device_id_it != config_json.end()) {
      device_id = block_size_it->get<int>();
    }
    bool io_by_cpu = false;  ///< The flag indicating if the CPU handles IO.
    auto io_by_cpu_it = config_json.find("io_by_cpu");
    if (io_by_cpu_it != config_json.end()) {
      io_by_cpu = io_by_cpu_it->get<bool>();
    }
    std::string evict_strategy = "kLru";
    auto evict_strategy_it = config_json.find("evict_strategy");
    if (evict_strategy_it != config_json.end()) {
      evict_strategy = io_by_cpu_it->get<std::string>();
    }
    return std::make_shared<HKVVariable<KeyType, ValueType>>(
        cols, init_capacity, initializer, max_capacity, max_hbm_for_vectors, max_bucket_size,
        max_load_factor, block_size, device_id, io_by_cpu, evict_strategy, stream);
  }
}
template <>
std::shared_ptr<VariableBase<int32_t, float>> VariableFactory::create(
    int64_t rows, int64_t cols, const std::string &type, const std::string &initializer,
    const std::string &config, cudaStream_t stream) {
  if (type == "hbm") {
    return std::make_shared<DETVariable<int32_t, float>>(cols, 2E4, initializer, stream);
  }
  if (type == "hybrid") {
    throw std::runtime_error("int32_t Keytype for hkv is not implemented yet.");
  }
}

template std::shared_ptr<VariableBase<int32_t, float>> VariableFactory::create<int32_t, float>(
    int64_t rows, int64_t cols, const std::string &type, const std::string &initializer,
    const std::string &config, cudaStream_t stream);

template std::shared_ptr<VariableBase<int64_t, float>> VariableFactory::create<int64_t, float>(
    int64_t rows, int64_t cols, const std::string &type, const std::string &initializer,
    const std::string &config, cudaStream_t stream);

}  // namespace sok
