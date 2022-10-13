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
int num_embedding_table = 3;
std::vector<EmbeddingTableParam> table_param_list = {
    {0, 504663, 8, std::nullopt, {}},
    {0, 12345, 16, std::nullopt, {}},
    {0, 122334, 16, std::nullopt, {}},
};

const std::vector<std::vector<int>> shard_matrix = {{1, 1, 1}};

const std::vector<GroupedEmbeddingParam> grouped_emb_params = {
    {TablePlacementStrategy::ModelParallel, {0, 1, 2}}};

const std::vector<LookupParam> lookup_params = {
    {0, 0, Combiner::Sum, 8, table_param_list[0].ev_size},
    {1, 1, Combiner::Average, 20, table_param_list[1].ev_size},
    {2, 2, Combiner::Sum, 10, table_param_list[2].ev_size},
    {3, 2, Combiner::Average, 5, table_param_list[2].ev_size},
};
int universal_batch_size = 1024;

template <typename key_t, typename index_t>
void test_embedding_table(int device_id, int table_type) {
  std::vector<int> device_list{device_id};
  auto resource_manager = HugeCTR::ResourceManagerExt::create({device_list}, 0);
  auto core = std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, 0);

  auto key_type = HugeCTR::TensorScalarTypeFunc<key_t>::get_type();
  auto index_type = HugeCTR::TensorScalarTypeFunc<index_t>::get_type();

  EmbeddingCollectionParam ebc_param{static_cast<int>(table_param_list.size()),
                                     static_cast<int>(lookup_params.size()),
                                     lookup_params,
                                     shard_matrix,
                                     grouped_emb_params,
                                     universal_batch_size,
                                     key_type,
                                     index_type,
                                     HugeCTR::TensorScalarTypeFunc<uint32_t>::get_type(),
                                     HugeCTR::TensorScalarTypeFunc<float>::get_type(),
                                     EmbeddingLayout::BatchMajor};

  IGroupedEmbeddingTable* embedding_table;
  if (table_type == 0) {
    embedding_table =
        new RaggedStaticEmbeddingTable(*resource_manager->get_local_gpu(0), core, table_param_list,
                                       ebc_param, 0, table_param_list[0].opt_param);
  } else {
    embedding_table =
        new DynamicEmbeddingTable(*resource_manager->get_local_gpu(0), core, table_param_list,
                                  ebc_param, 0, table_param_list[0].opt_param);
  }

  Device device{DeviceType::GPU, core->get_device_id()};
  std::vector<key_t> cpu_searched_key{504662, 2, 0, 2, 0, 2, 10, 12};
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

  embedding_table->lookup(searched_keys, searched_keys.get_num_elements(), searched_id_space_offset,
                          searched_id_space_offset.get_num_elements(), searched_id_space_list,
                          emb_vec);
  {
    HCTR_LIB_THROW(cudaStreamSynchronize(core->get_local_gpu()->get_stream()));

    float** cpu_emb_vec = new float*[cpu_searched_key.size()];

    HCTR_LIB_THROW(cudaMemcpy(cpu_emb_vec, emb_vec.get<float>(),
                              cpu_searched_key.size() * sizeof(void*), cudaMemcpyDeviceToHost));

    for (size_t idx = 0; idx < cpu_searched_id_space_offset.size() - 1; ++idx) {
      uint32_t start = cpu_searched_id_space_offset[idx];
      uint32_t end = cpu_searched_id_space_offset[idx + 1];
      int id_space = cpu_searched_id_space_list[idx];
      int ev_size = table_param_list[id_space].ev_size;

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
  delete embedding_table;
}

TEST(ragged_static_embedding_table, ragged_static_embedding_table) {
  test_embedding_table<int32_t, uint32_t>(0, 0);
}

TEST(dynamic_embedding_table, dynamic_embedding_table) {
  test_embedding_table<int32_t, size_t>(0, 1);
  test_embedding_table<int32_t, int32_t>(0, 1);
  test_embedding_table<int64_t, int32_t>(0, 1);
}