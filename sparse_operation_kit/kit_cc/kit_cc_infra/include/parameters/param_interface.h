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

#ifndef PARAM_INTERFACE_H
#define PARAM_INTERFACE_H

#include "hashtable/hashtable.h"
#include "hashtable/simple_hashtable.h"
#include "parameters/state_interface.h"
#include "common.h"

namespace SparseOperationKit {

class EmbeddingLayer;

/*
 * This class represents the variables shared to multiple GPUs.
 */
class ParamInterface : public States {
 public:
  virtual ~ParamInterface() {}
  ParamInterface(const size_t max_vocabulary_size_per_gpu, const size_t embedding_vec_size,
                 const bool trainable, const std::string var_name,
                 const DataType dtype);
  virtual std::shared_ptr<HashTable>& get_hashtable(const size_t local_replica_id) = 0;
  virtual std::shared_ptr<Tensor>& get_embedding_table_tensor(const size_t local_replica_id) = 0;
  std::shared_ptr<Tensor>& get_tensor(const size_t local_replica_id) override;
  virtual void set_initial_value(const size_t local_replica_id,
                                 const std::shared_ptr<Tensor>& initial_value) = 0;
  virtual void dump_to_file(const std::string filepath) = 0;
  virtual void restore_from_file(const std::string filepath) = 0;
  virtual void load_embedding_values(const std::vector<std::shared_ptr<Tensor>>& tensors) = 0;
  // It is not compulsory for the subclass to override this function.
  virtual size_t get_max_vocabulary_size_per_gpu() const;
  virtual size_t get_embedding_vec_size() const;
  virtual bool trainable() const;
  virtual std::string get_var_name() const;
  virtual DataType dtype() const;

  virtual void set_user(std::shared_ptr<EmbeddingLayer>& embedding);
  virtual void let_user_dump_to_file(const std::string filepath);
  virtual void let_user_restore_from_file(const std::string filepath);
  virtual void let_user_load_embedding_values(
      const std::vector<std::shared_ptr<Tensor>>& tensor_list);
  virtual void set_hashtable(std::shared_ptr<BaseSimpleHashtable> hashtable);

 private:
  const size_t max_vocabulary_size_per_gpu_;
  const size_t embedding_vec_size_;
  const bool trainable_;
  const std::string var_name_;
  DataType dtype_;
};

}  // namespace SparseOperationKit

#endif  // PARAM_INTERFACE_H