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
#include "hps/plugin/lookup_manager.hpp"

namespace HierarchicalParameterServer {

using namespace HugeCTR;

class Facade final {
 private:
  Facade();
  ~Facade() = default;
  Facade(const Facade&) = delete;
  Facade& operator=(const Facade&) = delete;
  Facade(Facade&&) = delete;
  Facade& operator=(Facade&&) = delete;

  std::once_flag lookup_manager_init_once_flag_;
  std::shared_ptr<LookupManager> lookup_manager_;

 public:
  static Facade* instance();
  void operator delete(void*);

  // global_batch_size and num_replicas_in_sync only valid for TensorFlow plugin
  void init(const char* ps_config_file, const pluginType_t plugin_type,
            const int32_t global_batch_size = 1, const int32_t num_replicas_in_sync = 1);
  void forward(const char* model_name, const int32_t table_id, const int32_t global_replica_id,
               const size_t num_keys, const size_t emb_vec_size, const void* d_keys,
               void* d_vectors);
};

}  // namespace HierarchicalParameterServer