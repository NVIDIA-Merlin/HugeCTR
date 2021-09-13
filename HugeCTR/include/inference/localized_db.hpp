/**Copyright(c) 2021, NVIDIA CORPORATION.**Licensed under the Apache License,
    Version 2.0(the "License");
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

#include <string.h>  //strcmpcase

#include <cstdlib>
#include <inference/database.hpp>
#include <inference/inference_utils.hpp>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace HugeCTR {

template <typename TypeHashKey>
class localdb : public DataBase<TypeHashKey> {
 public:
  localdb();
  localdb(parameter_server_config ps_config);
  virtual ~localdb();

  void load_data(std::vector<TypeHashKey> keys, std::vector<float> values, size_t embedding_size);

  void look_up(const TypeHashKey* embeddingcolumns_ptr, size_t num_samples,
               float* h_embeddingoutputvector, const std::string& model_name,
               size_t embedding_table_id);
  std::vector<std::vector<std::unordered_map<TypeHashKey, std::vector<float>>>> GetDB();

  void SetDB(std::vector<std::unordered_map<TypeHashKey, std::vector<float>>> model_emb_table);

 private:
  // The framework name
  std::string framework_name_;
  // Currently, embedding tables are implemented as CPU hashtable, 1 hashtable per embedding
  // table per model
  std::vector<std::vector<std::unordered_map<TypeHashKey, std::vector<float>>>>
      cpu_embedding_table_;
  std::vector<std::unordered_map<TypeHashKey, std::vector<float>>> model_emb_table;
  std::unordered_map<TypeHashKey, std::vector<float>> emb_table;
  // The parameter server configuration
  parameter_server_config ps_config_;
};
}  // namespace HugeCTR