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
#pragma once
#include <any>
#include <vector>

#include "embedding_table.hpp"

namespace embedding {

class ContextContainer {
  std::map<std::string, std::any> data_;

 public:
  template <typename T, typename = typename std::enable_if_t<std::is_copy_constructible_v<T>>>
  void pack(const std::string &key, T t) {
    if (data_.find(key) != data_.end()) {
      HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                     "context container do not support pack same key.");
    }
    data_[key] = t;
  }

  template <typename T, typename = typename std::enable_if_t<std::is_copy_constructible_v<T>>>
  T unpack(const std::string &key) {
    return std::any_cast<T>(data_[key]);
  }
};

class IEmbeddingCollectionForward {
 public:
  virtual ~IEmbeddingCollectionForward() = default;

  virtual void forward_per_gpu(const Tensor &key, const Tensor &bucket_range, size_t num_keys,
                               const Tensor &sparse_weight,
                               std::vector<ILookup *> &embedding_tables, Tensor &output_buffer,
                               std::vector<ContextContainer *> *context_container_list) = 0;
};

class IEmbeddingCollectionBackward {
 public:
  virtual ~IEmbeddingCollectionBackward() = default;

  virtual void backward_per_gpu(std::vector<ContextContainer *> &context_container_list,
                                Tensor &top_grad, std::vector<Tensor> *grad_key_list,
                                std::vector<size_t> *num_grad_key_list,
                                std::vector<Tensor> *grad_key_id_space_offset_list,
                                std::vector<size_t> *num_grad_key_id_space_offset_list,
                                std::vector<Tensor> *grad_ev_list,
                                std::vector<Tensor> *grad_ev_offset_list,
                                std::vector<Tensor> *grad_id_space_list_list,
                                bool do_allreduce) = 0;
};

class IGroupedEmbeddingForward {
 public:
  virtual ~IGroupedEmbeddingForward() = default;

  virtual void forward_per_gpu(const Tensor &keys, const Tensor &bucket_range, size_t num_keys,
                               const Tensor &sparse_weight, ILookup *embedding_table,
                               Tensor &output_buffer, ContextContainer *context_container) = 0;
};

class IGroupedEmbeddingBackward {
 public:
  virtual ~IGroupedEmbeddingBackward() = default;

  virtual void backward_per_gpu(ContextContainer *context_container, const Tensor &top_grad,
                                bool do_allreduce, Tensor *grad_key, size_t *num_grad_key,
                                Tensor *grad_key_id_space_offset,
                                size_t *num_grad_key_id_space_offset, Tensor *grad_ev,
                                Tensor *grad_ev_offset) = 0;
};
}  // namespace embedding
