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

#include <random>

#include "HugeCTR/core/hctr_impl/hctr_backend.hpp"
#include "HugeCTR/embedding_storage/dynamic_embedding.hpp"
#include "HugeCTR/embedding_storage/dynamic_embedding_cpu.hpp"
#include "HugeCTR/embedding_storage/ragged_static_embedding.hpp"
#include "HugeCTR/include/resource_managers/resource_manager_ext.hpp"

using namespace embedding;

const std::vector<std::vector<int>> shard_matrix = {{1, 1, 1}};

const std::vector<GroupedEmbeddingParam> grouped_emb_params = {
    {TablePlacementStrategy::ModelParallel, {0, 1, 2}}};

const size_t universal_batch_size = 1024;

template <class Key, typename Index>
void test_embedding_table_optimizer(int device_id, const char table_type[],
                                    const HugeCTR::Optimizer_t& opt_type,
                                    const size_t num_iterations) {
  std::vector<int> device_list{device_id};
  auto resource_manager = HugeCTR::ResourceManagerExt::create({device_list}, 0);
  auto core = std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, 0);

  const auto key_type = HugeCTR::TensorScalarTypeFunc<Key>::get_type();
  const auto index_type = HugeCTR::TensorScalarTypeFunc<Index>::get_type();

  const HugeCTR::OptParams opt_params{opt_type, 0.1f, {}, HugeCTR::Update_t::Local, 1.f};
  const std::vector<EmbeddingTableParam> table_params{
      {0, 10000, 8, {opt_params}, {}},
      {0, 20000, 10, {opt_params}, {}},
      {0, 4711, 6, {opt_params}, {}},
  };
  size_t max_ev_size = 0;
  for (const auto& tp : table_params) {
    max_ev_size = std::max(max_ev_size, static_cast<size_t>(tp.ev_size));
  }

  const std::vector<LookupParam> lookup_params{
      {0, 0, Combiner::Average, 10, table_params[0].ev_size},
      {1, 1, Combiner::Average, 10, table_params[1].ev_size},
      {2, 2, Combiner::Average, 10, table_params[2].ev_size},
  };

  bool indices_only = false;
  EmbeddingCollectionParam ebc_param{static_cast<int>(table_params.size()),
                                     {10000, 20000, 4711},
                                     static_cast<int>(lookup_params.size()),
                                     lookup_params,
                                     shard_matrix,
                                     grouped_emb_params,
                                     universal_batch_size,
                                     key_type,
                                     index_type,
                                     HugeCTR::TensorScalarTypeFunc<uint32_t>::get_type(),
                                     HugeCTR::TensorScalarTypeFunc<float>::get_type(),
                                     EmbeddingLayout::BatchMajor,
                                     EmbeddingLayout::FeatureMajor,
                                     indices_only};

  // Implementation to test.
  std::unique_ptr<IGroupedEmbeddingTable> test_table;
  if (!strcmp(table_type, "RaggedStatic")) {
    HCTR_LOG_S(INFO, WORLD) << "Creating `RaggedStaticEmbeddingTable`..." << std::endl;
    test_table = std::make_unique<RaggedStaticEmbeddingTable>(*resource_manager->get_local_gpu(0),
                                                              core, table_params, ebc_param, 0,
                                                              table_params[0].opt_param);
  } else if (!strcmp(table_type, "Dynamic")) {
    HCTR_LOG_S(INFO, WORLD) << "Creating `DynamicEmbeddingTable`..." << std::endl;
    test_table = std::make_unique<DynamicEmbeddingTable>(*resource_manager->get_local_gpu(0), core,
                                                         table_params, ebc_param, 0,
                                                         table_params[0].opt_param);
  } else if (!strcmp(table_type, "Dynamic_CPU")) {
    HCTR_LOG_S(INFO, WORLD) << "Creating `DynamicEmbeddingTableCPU<Key>`..." << std::endl;
    test_table = std::make_unique<DynamicEmbeddingTableCPU<Key>>(table_params, ebc_param, 0,
                                                                 table_params[0].opt_param);
  } else {
    HCTR_DIE("Unsupported table_type!");
  }

  // CPU reference implementation.
  std::unique_ptr<IGroupedEmbeddingTable> ref_table;
  ref_table = std::make_unique<DynamicEmbeddingTableCPU<Key>>(table_params, ebc_param, 0,
                                                              table_params[0].opt_param);

  Device device{DeviceType::GPU, core->get_device_id()};

  auto dump = [&](std::unique_ptr<IGroupedEmbeddingTable>& table)
      -> std::vector<std::tuple<int32_t, Key, std::vector<float>>> {
    // Indices that we are going to check.
    const std::vector<Key> keys_vec{
        0,     1000,  2000, 4710,  // Table 0 [0 1 2 3]
        0,     1000,  2000, 4710,  // Table 1 [4 5 6 7]
        0,     1000,  2000, 4710,  // Table 2 [8 9 10 11]
        9999,                      // Table 0 [12]
        10000, 19999,              // Table 1 [13 14]
    };
    const std::vector<uint32_t> id_space_offsets_vec{0, 4, 8, 12, 13, 15};
    const std::vector<int32_t> id_spaces_vec{0, 1, 2, 0, 1};

    // Copy to GPU memory.
    auto buffer_ptr = GetBuffer(core);
    auto keys_buf = buffer_ptr->reserve(keys_vec.size(), device, key_type);
    auto id_space_offsets_buf =
        buffer_ptr->reserve(id_space_offsets_vec.size(), device, TensorScalarType::UInt32);
    auto id_spaces_buf = buffer_ptr->reserve(id_spaces_vec.size(), device, TensorScalarType::Int32);

    std::vector<Tensor> emb_buf_;
    if (!strcmp(table_type, "Dynamic_CPU") || table.get() == ref_table.get()) {
      emb_buf_.resize(keys_vec.size());
      for (size_t i = 0; i < emb_buf_.size(); ++i) {
        emb_buf_[i] =
            buffer_ptr->reserve(max_ev_size * keys_vec.size(), device, TensorScalarType::Float32);
      }
    }
    buffer_ptr->allocate();

    // CPU Impl needs to have valid storage destination.
    std::unique_ptr<TensorList> embs_ptrs_buf;
    if (!strcmp(table_type, "Dynamic_CPU") || table.get() == ref_table.get()) {
      embs_ptrs_buf =
          std::make_unique<TensorList>(core.get(), emb_buf_, device, TensorScalarType::Float32);
    } else {
      embs_ptrs_buf = std::make_unique<TensorList>(core.get(), keys_vec.size(), device,
                                                   TensorScalarType::Float32);
    }

    keys_buf.copy_from(keys_vec);
    id_space_offsets_buf.copy_from(id_space_offsets_vec);
    id_spaces_buf.copy_from(id_spaces_vec);

    // Lookup and download values.
    table->lookup(keys_buf, keys_buf.get_num_elements(), id_space_offsets_buf,
                  id_space_offsets_buf.get_num_elements(), id_spaces_buf, *embs_ptrs_buf);

    HCTR_LIB_THROW(cudaStreamSynchronize(core->get_local_gpu()->get_stream()));
    std::vector<float*> embs_ptrs_vec(keys_vec.size());
    HCTR_LIB_THROW(cudaMemcpy(embs_ptrs_vec.data(), embs_ptrs_buf->get<float>(),
                              embs_ptrs_vec.size() * sizeof(float*), cudaMemcpyDeviceToHost));

    std::vector<std::tuple<int32_t, Key, std::vector<float>>> records;

    for (size_t idx = 0; idx < id_spaces_vec.size(); ++idx) {
      const int32_t id_space = id_spaces_vec[idx];
      const uint32_t ev_size = table_params[id_space].ev_size;

      const uint32_t off0 = id_space_offsets_vec[idx];
      const uint32_t off1 = id_space_offsets_vec[idx + 1];

      for (uint32_t off = off0; off < off1; ++off) {
        std::vector<float> ev(ev_size);
        HCTR_LIB_THROW(cudaMemcpy(ev.data(), embs_ptrs_vec[off], ev_size * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        records.emplace_back(id_space, keys_vec[off], ev);
      }
    }

    return records;
  };

  auto load = [&](std::unique_ptr<IGroupedEmbeddingTable>& table,
                  const std::vector<std::tuple<int32_t, Key, std::vector<float>>>& records) {
    // TODO: Replace hardcoded values with input.

    // Indices that we are going to check.
    const std::vector<Key> keys_vec{
        0,     1000,  2000, 4710,  // Table 0 [0 1 2 3]
        0,     1000,  2000, 4710,  // Table 1 [4 5 6 7]
        0,     1000,  2000, 4710,  // Table 2 [8 9 10 11]
        9999,                      // Table 0 [12]
        10000, 19999,              // Table 1 [13 14]
    };
    HCTR_CHECK(keys_vec.size() == records.size());
    const std::vector<uint32_t> id_space_offsets_vec{0, 4, 8, 12, 13, 15};
    const std::vector<int32_t> id_spaces_vec{0, 1, 2, 0, 1};

    std::vector<float> values_vec;
    std::vector<uint32_t> value_sizes_vec;
    value_sizes_vec.reserve(records.size());
    for (size_t i = 0; i < records.size(); ++i) {
      const auto& r = records[i];
      HCTR_CHECK(keys_vec[i] == std::get<1>(r));
      values_vec.insert(values_vec.end(), std::get<2>(r).begin(), std::get<2>(r).end());
      value_sizes_vec.emplace_back(table_params[std::get<0>(r)].ev_size);
    }

    // Copy to GPU memory.
    auto buffer_ptr = GetBuffer(core);
    auto keys_buf = buffer_ptr->reserve(keys_vec.size(), device, key_type);
    auto id_space_offsets_buf =
        buffer_ptr->reserve(id_space_offsets_vec.size(), device, TensorScalarType::UInt32);
    auto id_spaces_buf = buffer_ptr->reserve(id_spaces_vec.size(), device, TensorScalarType::Int32);
    auto values_buf = buffer_ptr->reserve(values_vec.size(), device, TensorScalarType::Float32);
    auto value_sizes_buf =
        buffer_ptr->reserve(value_sizes_vec.size(), device, TensorScalarType::UInt32);
    buffer_ptr->allocate();

    keys_buf.copy_from(keys_vec);
    id_space_offsets_buf.copy_from(id_space_offsets_vec);
    id_spaces_buf.copy_from(id_spaces_vec);
    values_buf.copy_from(values_vec);
    value_sizes_buf.copy_from(value_sizes_vec);

    // Call load function.
    table->load(keys_buf, id_space_offsets_buf, values_buf, value_sizes_buf, id_spaces_buf);
  };

  auto print = [&](const std::vector<std::tuple<int32_t, Key, std::vector<float>>>& records) {
    auto log = HCTR_LOG_S(INFO, WORLD);

    // Divider.
    log << std::endl;
    for (size_t i = 0; i < 80; ++i) {
      log << '=';
    }
    log << std::endl;

    // Actual data.
    for (const auto& [s, k, ev] : records) {
      log << "key: " << std::noshowpos << s << '/' << std::setw(5) << k << ", ev:";
      for (const float value : ev) {
        log << ' ' << std::fixed << std::showpos << std::setprecision(4) << value;
      }
      log << std::endl;
    }

    // Divider
    for (size_t i = 0; i < 80; ++i) {
      log << '=';
    }
    log << std::endl;
  };

  auto diff = [&](const std::vector<std::tuple<int32_t, Key, std::vector<float>>>& records0,
                  const std::vector<std::tuple<int32_t, Key, std::vector<float>>>& records1)
      -> std::vector<std::tuple<int32_t, Key, std::vector<float>>> {
    HCTR_CHECK(records0.size() == records1.size());

    std::vector<std::tuple<int32_t, Key, std::vector<float>>> diff;
    diff.reserve(records0.size());

    for (auto r_it0 = records0.begin(), r_it1 = records1.begin(); r_it0 != records0.end();
         ++r_it0, ++r_it1) {
      HCTR_CHECK(std::get<0>(*r_it0) == std::get<0>(*r_it1));
      HCTR_CHECK(std::get<1>(*r_it0) == std::get<1>(*r_it1));
      HCTR_CHECK(std::get<2>(*r_it0).size() == std::get<2>(*r_it1).size());

      std::vector<float> values;
      values.reserve(std::get<2>(*r_it0).size());

      auto v_it0 = std::get<2>(*r_it0).begin();
      auto v_it1 = std::get<2>(*r_it1).begin();
      for (; v_it0 != std::get<2>(*r_it0).end(); ++v_it0, ++v_it1) {
        values.emplace_back(*v_it1 - *v_it0);
      }

      diff.emplace_back(std::get<0>(*r_it0), std::get<1>(*r_it0), values);
    }

    return diff;
  };

  auto update = [&](std::unique_ptr<IGroupedEmbeddingTable>& table, const bool print,
                    const int seed) {
    // Fiddle together a couple of gradients.
    const std::vector<Key> keys_vec{
        0,    1000, 2000,  // Table 0 [0 1 2]
        2000, 4710,        // Table 1 [3 4]
        4710,              // Table 2 [5]
    };
    const std::vector<size_t> num_keys_vec{keys_vec.size()};
    const std::vector<int> table_ids_vec{
        0, 0, 0,  // Table 0
        1, 1,     // Table 1
        2         // Table 2
    };

    std::vector<float> grad_vec;
    std::vector<uint32_t> grad_idx_vec;
    grad_idx_vec.reserve(keys_vec.size());

    if (print) {
      std::cout << std::endl;
    }

    std::seed_seq seed_seq{seed, 1, seed, 2, seed, 3, seed, 4};
    std::mt19937 update_generator_(seed_seq);
    std::normal_distribution<float> update_distribution_{0.0f, 0.01f};

    float grad_value = 0.1f;
    for (size_t off = 0; off < keys_vec.size(); ++off) {
      const int table_id = table_ids_vec[off];
      const uint32_t ev_size = table_params[table_id].ev_size;

      grad_idx_vec.emplace_back(grad_vec.size());

      if (print) {
        std::cout << "key: " << std::noshowpos << table_id << '/' << std::setw(5) << keys_vec[off]
                  << ", upd:";
      }

      grad_vec.reserve(grad_vec.size() + ev_size);
      for (uint32_t i = 0; i < ev_size; ++i) {
        float upd = grad_value + i * 0.01f;
        upd = update_distribution_(update_generator_);
        // upd = 0.2f;
        grad_vec.emplace_back(upd);
        if (print) {
          std::cout << " " << std::fixed << std::showpos << std::setprecision(4) << upd;
        }
      }
      if (print) {
        std::cout << std::endl;
      }

      grad_value += 0.1f;
    }
    grad_idx_vec.emplace_back(grad_vec.size());

    // Get GPU memory.
    auto buffer_ptr = GetBuffer(core);
    auto keys_buf = buffer_ptr->reserve(keys_vec.size(), device, key_type);
    auto num_keys_buf = buffer_ptr->reserve({1}, device, core::TensorScalarType::Size_t);
    auto table_ids_buf =
        buffer_ptr->reserve(table_ids_vec.size(), device, core::TensorScalarType::Int32);
    auto grad_buf = buffer_ptr->reserve(grad_vec.size(), device, TensorScalarType::Float32);
    auto grad_idx_buf = buffer_ptr->reserve(grad_idx_vec.size(), device, TensorScalarType::UInt32);
    buffer_ptr->allocate();

    keys_buf.copy_from(keys_vec);
    num_keys_buf.copy_from(num_keys_vec);
    table_ids_buf.copy_from(table_ids_vec);
    grad_buf.copy_from(grad_vec);
    grad_idx_buf.copy_from(grad_idx_vec);

    // Inject into optimizer.
    table->update(keys_buf, num_keys_buf, table_ids_buf, grad_idx_buf, grad_buf);
  };

  auto eval = [&](const std::vector<std::tuple<int32_t, Key, std::vector<float>>>& records0,
                  const std::vector<std::tuple<int32_t, Key, std::vector<float>>>& records1) {
    ASSERT_EQ(records0.size(), records1.size());

    for (size_t i = 0; i < records0.size(); ++i) {
      const auto& r0 = records0[i];
      const auto& r1 = records1[i];

      ASSERT_EQ(std::get<0>(r0), std::get<0>(r1));
      ASSERT_EQ(std::get<1>(r0), std::get<1>(r1));

      const auto& v0 = std::get<2>(r0);
      const auto& v1 = std::get<2>(r1);
      ASSERT_EQ(v0.size(), v1.size());

      for (size_t j = 0; j < v0.size(); ++j) {
        EXPECT_NEAR(v0[j], v1[j], 1e-6);
      }
    }
  };

  HCTR_LOG_S(INFO, WORLD) << "=== Optimizer: " << static_cast<int>(opt_type) << " ===" << std::endl;

  // Test load: GPU -> CPU
  auto test0 = dump(test_table);
  //  print(test0);
  load(ref_table, test0);
  auto ref0 = dump(ref_table);
  // print(ref0);
  eval(test0, ref0);

  // Perform a couple of updates and compare.
  for (size_t i = 0; i < num_iterations; ++i) {
    update(test_table, false, i);
    update(ref_table, false, i);
  }
  auto test1 = dump(test_table);
  auto ref1 = dump(ref_table);

  if (false) {
    auto diff01 = diff(test0, test1);
    print(diff01);
  }

  auto diff1 = diff(test1, ref1);
  // print(test1);
  // print(ref1);
  print(diff1);
  eval(test1, ref1);
}

TEST(dynamic_embedding_table, optimizer) {
  // Self test.
  test_embedding_table_optimizer<int64_t, uint32_t>(0, "Dynamic_CPU", HugeCTR::Optimizer_t::SGD,
                                                    10);

  // DET vs. CPU mock implementation.
  test_embedding_table_optimizer<int64_t, uint32_t>(0, "Dynamic", HugeCTR::Optimizer_t::SGD, 10);
  test_embedding_table_optimizer<int64_t, uint32_t>(0, "Dynamic", HugeCTR::Optimizer_t::MomentumSGD,
                                                    10);
  test_embedding_table_optimizer<int64_t, uint32_t>(0, "Dynamic", HugeCTR::Optimizer_t::Nesterov,
                                                    10);
  test_embedding_table_optimizer<int64_t, uint32_t>(0, "Dynamic", HugeCTR::Optimizer_t::AdaGrad,
                                                    10);
  test_embedding_table_optimizer<int64_t, uint32_t>(0, "Dynamic", HugeCTR::Optimizer_t::RMSProp,
                                                    10);
  test_embedding_table_optimizer<int64_t, uint32_t>(0, "Dynamic", HugeCTR::Optimizer_t::Adam, 10);
  test_embedding_table_optimizer<int64_t, uint32_t>(0, "Dynamic", HugeCTR::Optimizer_t::Ftrl, 10);
}

TEST(static_embedding_table, optimizer) {
  test_embedding_table_optimizer<int64_t, uint32_t>(0, "RaggedStatic", HugeCTR::Optimizer_t::SGD,
                                                    10);
  test_embedding_table_optimizer<int64_t, uint32_t>(0, "RaggedStatic",
                                                    HugeCTR::Optimizer_t::AdaGrad, 10);
}