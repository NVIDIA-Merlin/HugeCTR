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
#include <sys/time.h>

#include <numeric>

#include "../embedding_collection_utils.hpp"
#include "HugeCTR/core/hctr_impl/hctr_backend.hpp"
#include "HugeCTR/embedding/embedding.hpp"
#include "HugeCTR/embedding/embedding_collection.hpp"
#include "HugeCTR/include/resource_managers/resource_manager_ext.hpp"
#include "embedding_collection_cpu.hpp"
using namespace embedding;

namespace meta {

// table params
const int num_table = 4;
// const std::vector<int> table_ev_size_list = {128, 64, 32, 16};
const std::vector<int> table_ev_size_list = {1, 2, 3, 4};
const std::vector<int> table_min_key_list = {0, 0, 0, 0};
const std::vector<int> table_max_key_list = {1000, 1000, 1000, 1000};

// lookup params
const std::vector<LookupParam> lookup_params0 = {
    {0, 0, Combiner::Sum, 2, table_ev_size_list[0]},
    {1, 1, Combiner::Sum, 2, table_ev_size_list[1]},
    {2, 2, Combiner::Sum, 2, table_ev_size_list[2]},
    {3, 3, Combiner::Sum, 2, table_ev_size_list[3]},
};

const std::vector<LookupParam> lookup_params1 = {
    {0, 0, Combiner::Sum, 8, table_ev_size_list[0]},
    {1, 1, Combiner::Average, 20, table_ev_size_list[1]},
    {2, 2, Combiner::Sum, 10, table_ev_size_list[2]},
    {3, 3, Combiner::Average, 5, table_ev_size_list[3]},
};

const std::vector<LookupParam> lookup_params2 = {
    {0, 0, Combiner::Sum, 8, table_ev_size_list[0]},
    {1, 1, Combiner::Average, 20, table_ev_size_list[1]},
    {2, 2, Combiner::Sum, 10, table_ev_size_list[2]},
    {3, 3, Combiner::Average, 5, table_ev_size_list[3]},
    {4, 3, Combiner::Average, 5, table_ev_size_list[3]},
};

const int batch_size = 8;
const std::vector<int> device_list = {0, 1};

std::vector<EmbeddingTableParam> get_table_param_list(core::DataType emb_type) {
  std::vector<EmbeddingTableParam> table_param_list;
  for (int table_id = 0; table_id < num_table; ++table_id) {
    EmbeddingTableParam table_param;
    table_param.table_id = table_id;
    table_param.max_vocabulary_size =
        static_cast<int>(table_max_key_list[table_id] - table_min_key_list[table_id]);
    table_param.ev_size = table_ev_size_list[table_id];
    table_param.min_key = table_min_key_list[table_id];
    table_param.max_key = table_max_key_list[table_id];
    HugeCTR::OptParams opt_param;
    opt_param.optimizer = HugeCTR::Optimizer_t::SGD;
    opt_param.lr = 1e-1;
    opt_param.scaler = (emb_type == TensorScalarType::Float16) ? 1024 : 1;
    table_param.opt_param = opt_param;
    table_param_list.push_back(std::move(table_param));
  }
  return table_param_list;
}

template <typename key_t, typename offset_t, typename index_t, typename emb_t>
void embedding_collection_e2e(const std::vector<LookupParam> &lookup_params,
                              const std::vector<std::vector<int>> &shard_matrix,
                              const std::vector<GroupedEmbeddingParam> &emb_params) {
  EmbeddingCollectionParam ebc_param;
  ebc_param.num_lookup = static_cast<int>(lookup_params.size());
  ebc_param.lookup_params = lookup_params;
  ebc_param.universal_batch_size = batch_size;
  ebc_param.shard_matrix = shard_matrix;
  ebc_param.emb_params = emb_params;

  ebc_param.is_table_first_input = true;
  ebc_param.key_type = HugeCTR::TensorScalarTypeFunc<key_t>::get_type();
  ebc_param.index_type = HugeCTR::TensorScalarTypeFunc<index_t>::get_type();
  ebc_param.offset_type = HugeCTR::TensorScalarTypeFunc<offset_t>::get_type();
  ebc_param.emb_type = HugeCTR::TensorScalarTypeFunc<emb_t>::get_type();
  auto table_param_list = get_table_param_list(ebc_param.emb_type);

  auto resource_manager = HugeCTR::ResourceManagerExt::create({device_list}, 0);
  int num_gpus = static_cast<int>(device_list.size());

  std::vector<key_t> key_list;
  std::vector<offset_t> bucket_range;
  auto prepare_input = [&] {
    timeval t1;
    gettimeofday(&t1, NULL);
    srand(t1.tv_usec * t1.tv_sec);
    key_list.clear();
    bucket_range.clear();
    bucket_range.push_back(0);

    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      auto &lookup_param = ebc_param.lookup_params[lookup_id];
      int table_id = lookup_param.table_id;
      int max_hotness = lookup_param.max_hotness;
      auto &table_param = table_param_list[table_id];

      for (int b = 0; b < ebc_param.universal_batch_size; ++b) {
        // int nnz = max_hotness;
        int nnz = (lookup_param.combiner == Combiner::Concat)
                      ? max_hotness
                      : 1 + rand() % max_hotness;  // TODO: support nnz=0
        bucket_range.push_back(nnz);
        for (int i = 0; i < nnz; ++i) {
          key_t key = rand() % (table_param.max_key - table_param.min_key) + table_param.min_key;
          key_list.push_back(key);
        }
      }
    }
    std::inclusive_scan(bucket_range.begin(), bucket_range.end(), bucket_range.begin());
  };

  std::vector<std::vector<emb_t>> top_grads;
  auto prepare_top_grads = [&] {
    top_grads.resize(num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      top_grads[gpu_id].clear();
      for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
        auto &lookup_param = ebc_param.lookup_params[lookup_id];
        int num_ev = (lookup_param.combiner == Combiner::Concat) ? lookup_param.max_hotness : 1;
        for (int b = 0;
             b < ebc_param.universal_batch_size * lookup_param.ev_size * num_ev / num_gpus; ++b) {
          float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
          top_grads[gpu_id].push_back(HugeCTR::TypeConvert<emb_t, float>::convert(r));
        }
      }
    }
  };

  std::vector<std::shared_ptr<core::CoreResourceManager>> core_resource_manager_list;
  std::vector<std::vector<std::unique_ptr<IGroupedEmbeddingTable>>> grouped_emb_table_list;
  // std::vector<std::unique_ptr<IEmbeddingCollection>> ebc_list;

  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    auto core = std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, gpu_id);

    core_resource_manager_list.push_back(core);
    //   ebc_list.push_back(std::make_unique<EmbeddingCollection>(core, ebc_param));
  }

  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    grouped_emb_table_list.push_back(create_grouped_embedding_table(
        resource_manager, core_resource_manager_list[gpu_id], ebc_param, table_param_list));
  }

  // std::vector<core::Tensor> ebc_key_list;
  // std::vector<core::Tensor> ebc_bucket_range_list;
  // std::vector<size_t *> ebc_num_keys_list;
  // std::vector<core::Tensor> ebc_top_grads;
  // std::vector<core::Tensor> ebc_outptut;
  // std::vector<std::vector<core::Tensor>> ebc_unique_keys_list;
  // std::vector<std::vector<size_t>> ebc_num_unique_keys_list;
  // std::vector<std::vector<core::Tensor>> ebc_num_unique_key_per_table_offset_list;
  // std::vector<std::vector<size_t>> ebc_num_table_list;
  // std::vector<std::vector<core::Tensor>> ebc_wgrad_list;
  // std::vector<std::vector<core::Tensor>> ebc_wgrad_ev_offset_list;
  // std::vector<std::vector<core::Tensor>> ebc_unique_table_ids_list;
  // ebc_unique_keys_list.resize(num_gpus);
  // ebc_num_unique_keys_list.resize(num_gpus);
  // ebc_num_unique_key_per_table_offset_list.resize(num_gpus);
  // ebc_num_table_list.resize(num_gpus);
  // ebc_wgrad_list.resize(num_gpus);
  // ebc_wgrad_ev_offset_list.resize(num_gpus);
  // ebc_unique_table_ids_list.resize(num_gpus);
  // for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
  //   HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());
  //   auto buffer = GetBuffer(core_resource_manager_list[gpu_id]);

  //   int max_hotness_sum = 0;
  //   for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
  //     auto &lookup_param = ebc_param.lookup_params[lookup_id];
  //     int max_hotness = lookup_param.max_hotness;
  //     max_hotness_sum += max_hotness;
  //   }

  //   ebc_key_list.push_back(buffer->reserve({ebc_param.universal_batch_size, max_hotness_sum},
  //                                          DeviceType::GPU, ebc_param.key_type));
  //   ebc_bucket_range_list.push_back(
  //       buffer->reserve({ebc_param.universal_batch_size * ebc_param.num_lookup + 1},
  //                       DeviceType::GPU, ebc_param.offset_type));
  //   ebc_num_keys_list.push_back(new size_t);

  //   int64_t num_ev = 0;
  //   for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
  //     auto &lookup_param = ebc_param.lookup_params[lookup_id];
  //     num_ev += (lookup_param.combiner == Combiner::Concat)
  //                   ? lookup_param.ev_size * lookup_param.max_hotness
  //                   : lookup_param.ev_size;
  //   }
  //   num_ev *= (ebc_param.universal_batch_size / num_gpus);
  //   ebc_top_grads.push_back(buffer->reserve(num_ev, DeviceType::GPU, ebc_param.emb_type));
  //   ebc_outptut.push_back(buffer->reserve(num_ev, DeviceType::GPU, ebc_param.emb_type));
  //   buffer->allocate();
  // }

  // auto prepare_gpu_input = [&] {
  //   for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
  //     HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());

  //     ebc_key_list[gpu_id].copy_from(key_list);
  //     ebc_bucket_range_list[gpu_id].copy_from(bucket_range);
  //     *(ebc_num_keys_list[gpu_id]) = key_list.size();
  //     ebc_top_grads[gpu_id].copy_from(top_grads[gpu_id]);
  //   }
  // };

  auto prepare_data = [&] {
    prepare_input();
    prepare_top_grads();
    // prepare_gpu_input();
  };

  auto sync_gpus = [&]() {
    for (auto core : core_resource_manager_list) {
      HCTR_LIB_THROW(cudaStreamSynchronize(core->get_local_gpu()->get_stream()));
    }
  };
  // sync for emb table init
  sync_gpus();

  std::vector<std::vector<IGroupedEmbeddingTable *>> grouped_emb_table_ptr_list;
  grouped_emb_table_ptr_list.resize(num_gpus);
  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    for (size_t i = 0; i < grouped_emb_table_list[gpu_id].size(); ++i) {
      grouped_emb_table_ptr_list[gpu_id].push_back(grouped_emb_table_list[gpu_id][i].get());
    }
  }

  EmbeddingCollectionCPU<key_t, offset_t, index_t, emb_t> ebc_cpu{
      num_gpus, ebc_param, num_table, table_param_list, grouped_emb_table_ptr_list};

  EmbeddingReferenceCPU<key_t, offset_t, index_t, emb_t> emb_ref{
      num_gpus, ebc_param, num_table, table_param_list, grouped_emb_table_ptr_list};

  auto check_forward_result = [&] {
    std::cout << "compare ebc cpu emb output vs. emb reference emb output.\n";
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      ASSERT_EQ(ebc_cpu.embedding_vec_[gpu_id].size(), emb_ref.embedding_vec_[gpu_id].size());
      // std::cout << "forward cpu output:\n";
      // print_array(ebc_cpu.embedding_vec_[gpu_id].size(), ebc_cpu.embedding_vec_[gpu_id]);
      // std::cout << "forward ref output:\n";
      // print_array(emb_ref.embedding_vec_[gpu_id].size(), emb_ref.embedding_vec_[gpu_id]);
      assert_array_eq(ebc_cpu.embedding_vec_[gpu_id].size(), ebc_cpu.embedding_vec_[gpu_id],
                      emb_ref.embedding_vec_[gpu_id]);
    }
    std::cout << "\t>pass compare ebc cpu emb output vs. emb reference emb output.\n";

    // std::cout << "compare ebc gpu emb output vs. emb reference emb output.\n";
    // for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    //   std::vector<emb_t> gpu_emb_output;
    //   ebc_outptut[gpu_id].to(&gpu_emb_output);
    //   ASSERT_EQ(gpu_emb_output.size(), emb_ref.embedding_vec_[gpu_id].size());
    //   std::cout << "forward ref output:\n";
    //   print_array(emb_ref.embedding_vec_[gpu_id].size(), emb_ref.embedding_vec_[gpu_id]);
    //   std::cout << "forward gpu output:\n";
    //   print_array(gpu_emb_output.size(), gpu_emb_output);
    //   assert_array_eq(gpu_emb_output.size(), gpu_emb_output, ebc_cpu.embedding_vec_[gpu_id]);
    // }
    // std::cout << "\t>pass compare ebc gpu emb output vs. emb reference emb output.\n";
  };
  auto check_backward_result = [&] {
    // std::vector<std::unordered_map<key_t, std::vector<float>>> gpu_grad_info;
    // gpu_grad_info.resize(num_table);
    // for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    //   for (size_t emb_id = 0; emb_id < ebc_param.emb_params.size(); ++emb_id) {
    //     const auto &unique_keys = ebc_unique_keys_list[gpu_id][emb_id];
    //     const auto &num_unique_keys = ebc_num_unique_keys_list[gpu_id][emb_id];
    //     const auto &num_unique_key_per_table_offset =
    //     ebc_num_unique_key_per_table_offset_list[gpu_id][emb_id]; const auto &num_table =
    //     ebc_num_table_list[gpu_id][emb_id]; const auto &wgrad = ebc_wgrad_list[gpu_id][emb_id];
    //     const auto &wgrad_ev_offset = ebc_wgrad_ev_offset_list[gpu_id][emb_id];
    //     const auto &unique_table_ids = ebc_unique_table_ids_list[gpu_id][emb_id];

    //     std::vector<key_t> gpu_unique_keys;
    //     unique_keys.to(&gpu_unique_keys);
    //     std::cout << "gpu_unique_keys:\n";
    //     print_array(gpu_unique_keys.size(), gpu_unique_keys);

    //     std::vector<uint32_t> gpu_num_unique_key_per_table_offset;
    //     num_unique_key_per_table_offset.to(&gpu_num_unique_key_per_table_offset);
    //     std::cout << "gpu_num_unique_key_per_table_offset:\n";
    //     print_array(gpu_num_unique_key_per_table_offset.size(),
    //     gpu_num_unique_key_per_table_offset); std::cout << "num_unique_keys:\n"; std::cout <<
    //     num_unique_keys << "\n";

    //     std::vector<float> gpu_wgrad;
    //     wgrad.to(&gpu_wgrad);
    //     std::cout << "gpu_wgrad:\n";
    //     print_array(gpu_wgrad.size(), gpu_wgrad);

    //     std::vector<uint32_t> gpu_wgrad_ev_offset;
    //     wgrad_ev_offset.to(&gpu_wgrad_ev_offset);
    //     std::cout << "gpu_wgrad_ev_offset:\n";
    //     print_array(gpu_wgrad_ev_offset.size(), gpu_wgrad_ev_offset);

    //     std::vector<int> gpu_unique_table_ids;
    //     unique_table_ids.to(&gpu_unique_table_ids);
    //     std::cout << "gpu_unique_table_ids:\n";
    //     print_array(gpu_unique_table_ids.size(), gpu_unique_table_ids);

    //     ASSERT_EQ(num_unique_keys, gpu_num_unique_key_per_table_offset.back());
    //     for (size_t i_table = 0; i_table < num_table; ++i_table) {
    //       int table_id = gpu_unique_table_ids[i_table];
    //       ASSERT_TRUE(table_id < gpu_grad_info.size());
    //       uint32_t start = gpu_num_unique_key_per_table_offset[i_table];
    //       uint32_t end = gpu_num_unique_key_per_table_offset[i_table + 1];
    //       for (uint32_t r = 0; r < (end - start); ++r) {
    //         key_t k = gpu_unique_keys[r + start];
    //         uint32_t ev_start = gpu_wgrad_ev_offset[r + start];
    //         uint32_t ev_end = gpu_wgrad_ev_offset[r + start + 1];
    //         int ev_size= static_cast<int>(ev_end - ev_start);
    //         ArrayView<float> ev{&gpu_wgrad[ev_start], ev_size};
    //         if (gpu_grad_info[table_id].find(k) == gpu_grad_info[table_id].end()) {
    //           for (int e = 0; e < ev_size; ++e) {
    //             gpu_grad_info[table_id][k].push_back(ev[e]);
    //           }
    //         } else {
    //           for (int e = 0; e < ev_size; ++e) {
    //             gpu_grad_info[table_id][k][e] += ev[e];
    //           }
    //         }
    //       }
    //     }
    //   }
    // }

    auto compare_grad_in_table = [](const std::unordered_map<key_t, std::vector<float>> &lhs,
                                    const std::unordered_map<key_t, std::vector<float>> &rhs,
                                    bool verbose = false) {
      ASSERT_EQ(lhs.size(), rhs.size());

      for (auto p : lhs) {
        auto &k = p.first;
        auto &lhs_ev = p.second;
        ASSERT_TRUE(rhs.find(k) != rhs.end());
        auto &rhs_ev = rhs.at(k);
        ASSERT_EQ(lhs_ev.size(), rhs_ev.size());
        if (verbose) {
          std::cout << "lhs output:\n";
          print_array(lhs_ev.size(), lhs_ev);
          std::cout << "rhs output:\n";
          print_array(rhs_ev.size(), rhs_ev);
        }
        assert_array_eq(lhs_ev.size(), lhs_ev, rhs_ev);
      }
    };

    std::cout << "compare ref grad info vs. ebc cpu grad info.\n";
    ASSERT_EQ(ebc_cpu.grad_info_.size(), emb_ref.accumulate_grad_map_.size());
    for (int table_id = 0; table_id < num_table; ++table_id) {
      ASSERT_TRUE(table_id < static_cast<int>(ebc_cpu.grad_info_.size()));
      auto &cpu_grad_in_table = ebc_cpu.grad_info_.at(table_id);
      auto &ref_grad_in_table = emb_ref.accumulate_grad_map_.at(table_id);
      compare_grad_in_table(cpu_grad_in_table, ref_grad_in_table);
    }
    std::cout << "\t>pass compare ref grad info vs. ebc cpu grad info.\n";

    // std::cout << "compare ref grad info vs. ebc gpu grad info.\n";
    // for (int table_id = 0; table_id < num_table; ++table_id) {
    //   auto &ref_grad_in_table = emb_ref.accumulate_grad_map_[table_id];
    //   auto &gpu_grad_in_table = gpu_grad_info[table_id];
    //   compare_grad_in_table(ref_grad_in_table, gpu_grad_in_table, true);
    // }
    // std::cout << "\t>pass compare ebc gpu grad info vs. ebc ref grad info.\n";
  };

  auto check_embedding_table = [&] {
    std::cout << "compare ref emb table vs. ebc cpu emb table.\n";
    const auto &cpu_emb_table = ebc_cpu.emb_table_cpu_.emb_table_list_;
    const auto &ref_emb_table = emb_ref.emb_table_cpu_.emb_table_list_;
    ASSERT_TRUE(cpu_emb_table.size() == ref_emb_table.size());

    for (size_t table_id = 0; table_id < cpu_emb_table.size(); ++table_id) {
      ASSERT_EQ(cpu_emb_table[table_id].size(), ref_emb_table[table_id].size());

      for (auto &[k, cpu_ev] : cpu_emb_table[table_id]) {
        ASSERT_TRUE(cpu_emb_table[table_id].find(k) != ref_emb_table[table_id].end());
        auto ref_ev = ref_emb_table[table_id].at(k);

        ASSERT_EQ(cpu_ev.size(), ref_ev.size());
        assert_array_eq(cpu_ev.size(), cpu_ev, ref_ev);
      }
    }
    std::cout << "\t>pass compare ref emb table vs. ebc cpu emb table.\n";

    // EmbeddingTableCPU<key_t, index_t> copy_gpu_emb_table{num_table,
    // table_major_ebc_table_ptr_list,
    //                                                      table_param_list};
    // const auto &gpu_emb_table = copy_gpu_emb_table.emb_table_list_;

    // std::cout << "compare ref emb table vs. ebc gpu emb table.\n";
    // ASSERT_TRUE(gpu_emb_table.size() == ref_emb_table.size());

    // for (size_t id_space = 0; id_space < gpu_emb_table.size(); ++id_space) {
    //   ASSERT_EQ(gpu_emb_table[id_space].size(), ref_emb_table[id_space].size());

    //   for (auto &[k, gpu_ev] : gpu_emb_table[id_space]) {
    //     ASSERT_TRUE(gpu_emb_table[id_space].find(k) != ref_emb_table[id_space].end());
    //     auto ref_ev = ref_emb_table[id_space].at(k);

    //     ASSERT_EQ(gpu_ev.size(), ref_ev.size());
    //     assert_array_eq(gpu_ev.size(), gpu_ev, ref_ev);
    //   }
    // }
    // std::cout << "\t>pass compare ref emb table vs. ebc gpu emb table.\n";
  };

  int num_iteration = 10;
  for (int iter = 0; iter < num_iteration; ++iter) {
    std::cout << "iter:" << iter << "\n";
    prepare_data();
    sync_gpus();

    // forward
    ebc_cpu.embedding_forward_cpu(key_list, bucket_range);
    emb_ref.embedding_forward_cpu(key_list, bucket_range);

    sync_gpus();
    check_forward_result();

    // backward
    ebc_cpu.embedding_backward_cpu(top_grads, batch_size);
    emb_ref.embedding_backward_cpu(top_grads, key_list, bucket_range);

    sync_gpus();
    check_backward_result();

    // update
    ebc_cpu.embedding_update_cpu();
    emb_ref.embedding_update_cpu();

    sync_gpus();

    check_embedding_table();
  }
}

// dp
namespace dp {
const std::vector<std::vector<int>> shard_matrix = {
    {0, 0, 0, 0},
    {0, 0, 0, 0},
};

const std::vector<GroupedEmbeddingParam> emb_params = {
    {TablePlacementStrategy::DataParallel, {0, 1, 2, 3}}};

TEST(test_embedding_collection, dp_plan0) {
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, float>(lookup_params0, shard_matrix,
                                                                emb_params);
}

TEST(test_embedding_collection, dp_plan1) {
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, float>(lookup_params1, shard_matrix,
                                                                emb_params);
}
}  // namespace dp

namespace mp {
const std::vector<std::vector<int>> shard_matrix = {
    {1, 0, 1, 1},
    {0, 1, 1, 1},
};

const std::vector<GroupedEmbeddingParam> emb_params = {
    {TablePlacementStrategy::ModelParallel, {0, 1, 2, 3}}};

TEST(test_embedding_collection, mp_plan0) {
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, float>(lookup_params0, shard_matrix,
                                                                emb_params);
}

TEST(test_embedding_collection, mp_plan1) {
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, float>(lookup_params1, shard_matrix,
                                                                emb_params);
}

TEST(test_embedding_collection, mp_plan2) {
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, float>(lookup_params2, shard_matrix,
                                                                emb_params);
}
}  // namespace mp

namespace dp_and_mp {
const std::vector<std::vector<int>> shard_matrix = {
    {1, 0, 0, 1},
    {0, 1, 0, 1},
};

const std::vector<GroupedEmbeddingParam> emb_params = {
    {TablePlacementStrategy::DataParallel, {2}},
    {TablePlacementStrategy::ModelParallel, {0, 1, 3}}};

TEST(test_embedding_collection, dp_and_mp_plan0) {
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, float>(lookup_params0, shard_matrix,
                                                                emb_params);
}

TEST(test_embedding_collection, dp_and_mp_plan1) {
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, float>(lookup_params1, shard_matrix,
                                                                emb_params);
}

TEST(test_embedding_collection, dp_and_mp_plan2) {
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, float>(lookup_params2, shard_matrix,
                                                                emb_params);
}
}  // namespace dp_and_mp
}  // namespace meta
