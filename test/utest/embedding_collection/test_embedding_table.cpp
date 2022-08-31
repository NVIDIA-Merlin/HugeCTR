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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "HugeCTR/core/hctr_impl/hctr_backend.hpp"
#include "HugeCTR/embedding_storage/dynamic_embedding.hpp"
#include "HugeCTR/embedding_storage/ragged_static_embedding.hpp"
#include "HugeCTR/include/resource_managers/resource_manager_ext.hpp"

using namespace embedding;

template <typename key_t, typename index_t>
void test_ragged_static_embedding_table(int device_id) {
  std::vector<int> device_list{device_id};
  auto resource_manager = HugeCTR::ResourceManagerExt::create({device_list}, 0);
  auto core = std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, 0);

  auto key_type = HugeCTR::TensorScalarTypeFunc<key_t>::get_type();
  auto index_type = HugeCTR::TensorScalarTypeFunc<index_t>::get_type();

  std::vector<EmbeddingTableParam> param_list;
  int num_embedding_table = 3;
  std::vector<int> max_vocabulary_size_list{504663, 504663, 504663};
  std::vector<int> ev_size_list{8, 16, 16};
  for (int id_space = 0; id_space < num_embedding_table; ++id_space) {
    EmbeddingTableParam param;
    param.table_id = id_space;
    param.max_vocabulary_size = max_vocabulary_size_list[id_space];
    param.ev_size = ev_size_list[id_space];
    param.min_key = 0;
    param.max_key = max_vocabulary_size_list[id_space];
    param.opt_param.optimizer = HugeCTR::Optimizer_t::SGD;
    param.opt_param.lr = 1e-1;
    param_list.push_back(param);
  }

  EmbeddingShardingParam sharding_param;
  sharding_param.local_embedding_list = {0, 1, 2};
  sharding_param.global_embedding_list = {{0, 1, 2}};
  sharding_param.shard_id = 0;
  sharding_param.shards_count = 1;
  sharding_param.table_placement_strategy = TablePlacementStrategy::ModelParallel;

  EmbeddingCollectionParam ebc_param;
  ebc_param.num_embedding = num_embedding_table;
  for (int id_space = 0; id_space < num_embedding_table; ++id_space) {
    EmbeddingParam p;
    p.embedding_id = id_space;
    p.id_space = id_space;
    p.ev_size = ev_size_list[id_space];
    ebc_param.embedding_params.push_back(std::move(p));
  }
  ebc_param.universal_batch_size = 1024;
  ebc_param.key_type = key_type;
  ebc_param.index_type = index_type;

  auto ragged_static_embedding_table = std::make_shared<RaggedStaticEmbeddingTable>(
      *resource_manager->get_local_gpu(0), core, param_list, ebc_param, sharding_param,
      param_list[0].opt_param);

  Device device{DeviceType::GPU, core->get_device_id()};
  std::vector<key_t> cpu_searched_key{504663, 2, 0, 2, 0, 2, 10, 12};
  std::vector<uint32_t> cpu_searched_id_space_offset{0, 2, 4, 6, 8};
  std::vector<int> cpu_searched_id_space_list{0, 0, 1, 2};
  TensorList emb_vec{core.get(), 10, device, TensorScalarType::Float32};

  auto buffer_ptr = GetBuffer(core);
  auto searched_keys = buffer_ptr->reserve(cpu_searched_key.size(), device, key_type);
  auto searched_id_space_offset =
      buffer_ptr->reserve(cpu_searched_id_space_offset.size(), device, TensorScalarType::UInt32);
  auto searched_id_space_list =
      buffer_ptr->reserve(cpu_searched_id_space_list.size(), device, TensorScalarType::Int32);
  buffer_ptr->allocate();

  searched_keys.copy_from(cpu_searched_key);
  searched_id_space_offset.copy_from(cpu_searched_id_space_offset);
  searched_id_space_list.copy_from(cpu_searched_id_space_list);

  ragged_static_embedding_table->lookup(
      searched_keys, searched_keys.get_num_elements(), searched_id_space_offset,
      searched_id_space_offset.get_num_elements(), searched_id_space_list, emb_vec);
  {
    HCTR_LIB_THROW(cudaStreamSynchronize(core->get_local_gpu()->get_stream()));

    float** cpu_emb_vec = new float*[cpu_searched_key.size()];

    HCTR_LIB_THROW(cudaMemcpy(cpu_emb_vec, emb_vec.get<float>(),
                              cpu_searched_key.size() * sizeof(void*), cudaMemcpyDeviceToHost));

    for (size_t idx = 0; idx < cpu_searched_id_space_offset.size() - 1; ++idx) {
      uint32_t start = cpu_searched_id_space_offset[idx];
      uint32_t end = cpu_searched_id_space_offset[idx + 1];
      int id_space = cpu_searched_id_space_list[idx];
      int ev_size = ev_size_list[id_space];

      for (uint32_t i = start; i < end; ++i) {
        float* ev = new float[ev_size];
        cudaMemcpy(ev, cpu_emb_vec[i], ev_size * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "key:" << cpu_searched_key[i] << ", ev:";
        for (int t = 0; t < ev_size; ++t) {
          std::cout << ev[t] << " ";
        }
        std::cout << std::endl;
      }
    }
  }
}

template <typename key_t, typename index_t>
void test_dynamic_embedding_table(int device_id) {
  std::vector<int> device_list{device_id};
  auto resource_manager = HugeCTR::ResourceManagerExt::create({device_list}, 0);
  auto core = std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, 0);

  auto key_type = HugeCTR::TensorScalarTypeFunc<key_t>::get_type();
  auto index_type = HugeCTR::TensorScalarTypeFunc<index_t>::get_type();

  std::vector<EmbeddingTableParam> param_list;
  int num_embedding_table = 3;
  std::vector<int> max_vocabulary_size_list{504663, 504663, 504663};
  std::vector<int> ev_size_list{8, 16, 16};
  std::vector<int> id_space_list{0, 1, 2};
  for (int id_space = 0; id_space < num_embedding_table; ++id_space) {
    EmbeddingTableParam param;
    param.table_id = id_space_list[id_space];
    param.max_vocabulary_size = -1;
    param.ev_size = ev_size_list[id_space];
    param.min_key = 0;
    param.max_key = max_vocabulary_size_list[id_space];
    param_list.push_back(param);
  }

  EmbeddingShardingParam sharding_param;
  sharding_param.local_embedding_list = {0, 1, 2};
  sharding_param.global_embedding_list = {{0, 1, 2}};
  sharding_param.shard_id = 0;
  sharding_param.shards_count = 1;
  sharding_param.table_placement_strategy = TablePlacementStrategy::ModelParallel;

  EmbeddingCollectionParam ebc_param;
  ebc_param.num_embedding = num_embedding_table;
  for (int id_space = 0; id_space < num_embedding_table; ++id_space) {
    EmbeddingParam p;
    p.embedding_id = id_space;
    p.id_space = id_space_list[id_space];
    p.ev_size = ev_size_list[id_space];
    ebc_param.embedding_params.push_back(std::move(p));
  }
  ebc_param.universal_batch_size = 1024;
  ebc_param.key_type = key_type;
  ebc_param.index_type = index_type;

  auto dynamic_embedding_table =
      std::make_shared<DynamicEmbeddingTable>(core, param_list, ebc_param, sharding_param);

  Device device{DeviceType::GPU, core->get_device_id()};
  std::vector<key_t> cpu_searched_key{504663, 2, 0, 2, 0, 2, 10, 12};
  std::vector<index_t> cpu_searched_id_space_offset{0, 2, 4, 6, 8};
  std::vector<index_t> cpu_searched_id_space_list{0, 1, 2, 2};
  TensorList emb_vec{core.get(), 10, device, TensorScalarType::Float32};

  auto buffer_ptr = GetBuffer(core);
  auto searched_keys = buffer_ptr->reserve(cpu_searched_key.size(), device, key_type);
  auto searched_id_space_offset =
      buffer_ptr->reserve(cpu_searched_id_space_offset.size(), device, index_type);
  auto searched_id_space_list =
      buffer_ptr->reserve(cpu_searched_id_space_list.size(), device, index_type);
  buffer_ptr->allocate();

  searched_keys.copy_from(cpu_searched_key);
  searched_id_space_offset.copy_from(cpu_searched_id_space_offset);
  searched_id_space_list.copy_from(cpu_searched_id_space_list);

  dynamic_embedding_table->lookup(
      searched_keys, searched_keys.get_num_elements(), searched_id_space_offset,
      searched_id_space_offset.get_num_elements(), searched_id_space_list, emb_vec);
  {
    HCTR_LIB_THROW(cudaStreamSynchronize(core->get_local_gpu()->get_stream()));

    float** cpu_emb_vec = new float*[cpu_searched_key.size()];

    HCTR_LIB_THROW(cudaMemcpy(cpu_emb_vec, emb_vec.get<float>(),
                              cpu_searched_key.size() * sizeof(void*), cudaMemcpyDeviceToHost));

    for (size_t idx = 0; idx < cpu_searched_id_space_offset.size() - 1; ++idx) {
      uint32_t start = cpu_searched_id_space_offset[idx];
      uint32_t end = cpu_searched_id_space_offset[idx + 1];
      int id_space = cpu_searched_id_space_list[idx];
      int ev_size = ev_size_list[id_space];

      for (uint32_t i = start; i < end; ++i) {
        float* ev = new float[ev_size];
        HCTR_LIB_THROW(
            cudaMemcpy(ev, cpu_emb_vec[i], ev_size * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "key:" << cpu_searched_key[i] << ", ev:";
        for (int t = 0; t < ev_size; ++t) {
          std::cout << ev[t] << " ";
        }
        std::cout << std::endl;
      }
    }
  }
}

TEST(ragged_static_embedding_table, ragged_static_embedding_table) {
  test_ragged_static_embedding_table<int32_t, uint32_t>(0);
}

TEST(dynamic_embedding_table, dynamic_embedding_table) {
  test_dynamic_embedding_table<int32_t, size_t>(0);
  test_dynamic_embedding_table<int32_t, int32_t>(0);
  test_dynamic_embedding_table<int64_t, int32_t>(0);
}