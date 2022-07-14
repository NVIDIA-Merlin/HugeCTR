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

#include "HugeCTR/core/hctr_impl/hctr_backend.hpp"
#include "HugeCTR/embedding/embedding.hpp"
#include "HugeCTR/embedding/embedding_planner.hpp"
#include "HugeCTR/include/resource_managers/resource_manager_ext.hpp"
#include "embedding_collection_cpu.hpp"

using namespace embedding;

template <typename type>
void print_array(size_t num, const std::vector<type> &a) {
  ASSERT_GE(a.size(), num);
  for (size_t i = 0; i < num; ++i) {
    std::cout << a[i] << " ";
  }
  std::cout << "\n";
}

template <>
void print_array<__half>(size_t num, const std::vector<__half> &a) {
  ASSERT_GE(a.size(), num);
  for (size_t i = 0; i < num; ++i) {
    std::cout << HugeCTR::TypeConvert<float, __half>::convert(a[i]) << " ";
  }
  std::cout << "\n";
}

// template <typename type>
template <typename type, typename = std::enable_if_t<std::is_integral_v<type>>>
void assert_array_eq(size_t num, const std::vector<type> &a, const std::vector<type> &b) {
  ASSERT_GE(a.size(), num);
  ASSERT_GE(b.size(), num);
  for (size_t i = 0; i < num; ++i) {
    ASSERT_EQ(a[i], b[i]) << "idx:" << i;
  }
}

void assert_array_eq(size_t num, const std::vector<float> &a, const std::vector<float> &b,
                     float threshold = 1e-1) {
  ASSERT_GE(a.size(), num);
  ASSERT_GE(b.size(), num);
  float max_error = 0.f;
  for (size_t i = 0; i < num; ++i) {
    float error = std::abs(a[i] - b[i]);
    max_error = std::max(max_error, error);
  }
  ASSERT_LE(max_error, threshold) << "max error:" << max_error << ",threshold:" << threshold;
}

void assert_array_eq(size_t num, const std::vector<__half> &a, const std::vector<__half> &b,
                     float threshold = 1e-1) {
  ASSERT_GE(a.size(), num);
  ASSERT_GE(b.size(), num);
  float max_error = 0.f;
  for (size_t i = 0; i < num; ++i) {
    float lhs = HugeCTR::TypeConvert<float, __half>::convert(a[i]);
    float rhs = HugeCTR::TypeConvert<float, __half>::convert(b[i]);
    float error = std::abs(lhs - rhs);
    ASSERT_LE(error, threshold) << ",lhs:" << lhs << ",rhs:" << rhs << "\n";
    max_error = std::max(max_error, error);
  }
  ASSERT_LE(max_error, threshold) << "max error:" << max_error << ",threshold:" << threshold;
}

std::vector<EmbeddingTableParam> get_table_param_list(int num_table,
                                                      const std::vector<int> &table_ev_size_list,
                                                      const std::vector<int> &table_min_key_list,
                                                      const std::vector<int> &table_max_key_list,
                                                      core::DataType emb_type) {
  std::vector<EmbeddingTableParam> table_param_list;
  for (int id = 0; id < num_table; ++id) {
    EmbeddingTableParam table_param;
    table_param.id_space = id;
    table_param.max_vocabulary_size = 5;
    table_param.ev_size = table_ev_size_list[id];
    table_param.min_key = table_min_key_list[id];
    table_param.max_key = table_max_key_list[id];
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
void embedding_collection_e2e(const std::vector<int> device_list, const int &batch_size,
                              const int &num_table, const std::vector<int> &table_ev_size_list,
                              const int &num_embedding, const std::vector<int> &id_space_list,
                              const std::vector<int> &hotness_list,
                              const std::vector<Combiner> &combiner_list,
                              const std::vector<int> table_min_key_list,
                              const std::vector<int> table_max_key_list,
                              const std::string &plan_file) {
  ASSERT_TRUE(static_cast<size_t>(num_table) == table_ev_size_list.size());
  ASSERT_TRUE(static_cast<size_t>(num_table) == table_min_key_list.size());
  ASSERT_TRUE(static_cast<size_t>(num_table) == table_max_key_list.size());
  ASSERT_TRUE(static_cast<size_t>(num_embedding) == id_space_list.size());
  ASSERT_TRUE(static_cast<size_t>(num_embedding) == hotness_list.size());
  ASSERT_TRUE(static_cast<size_t>(num_embedding) == combiner_list.size());
  EmbeddingCollectionParam ebc_param;
  ebc_param.num_embedding = num_embedding;
  for (int embedding_id = 0; embedding_id < num_embedding; ++embedding_id) {
    EmbeddingParam emb_param;
    emb_param.embedding_id = embedding_id;
    emb_param.id_space = id_space_list[embedding_id];
    emb_param.combiner = combiner_list[embedding_id];
    emb_param.hotness = hotness_list[embedding_id];
    emb_param.ev_size = table_ev_size_list[id_space_list[embedding_id]];
    ebc_param.embedding_params.push_back(std::move(emb_param));
  }
  ebc_param.universal_batch_size = batch_size;
  ebc_param.is_table_first_input = true;
  ebc_param.is_utest = true;
  ebc_param.key_type = HugeCTR::TensorScalarTypeFunc<key_t>::get_type();
  ebc_param.index_type = HugeCTR::TensorScalarTypeFunc<index_t>::get_type();
  ebc_param.offset_type = HugeCTR::TensorScalarTypeFunc<offset_t>::get_type();
  ebc_param.emb_type = HugeCTR::TensorScalarTypeFunc<emb_t>::get_type();
  auto table_param_list = get_table_param_list(num_table, table_ev_size_list, table_min_key_list,
                                               table_max_key_list, ebc_param.emb_type);

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

    for (int embedding_id = 0; embedding_id < ebc_param.num_embedding; ++embedding_id) {
      auto &embedding_param = ebc_param.embedding_params[embedding_id];
      int id_space = embedding_param.id_space;
      int hotness = embedding_param.hotness;
      auto &table_param = table_param_list[id_space];

      for (int b = 0; b < ebc_param.universal_batch_size; ++b) {
        int nnz = (ebc_param.embedding_params[embedding_id].combiner == Combiner::Concat)
                      ? hotness
                      : 1 + rand() % hotness;  // TODO: support nnz=0
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
      for (int embedding_id = 0; embedding_id < ebc_param.num_embedding; ++embedding_id) {
        auto &embedding_param = ebc_param.embedding_params[embedding_id];
        int hotness = (embedding_param.combiner == Combiner::Concat) ? embedding_param.hotness : 1;
        for (int b = 0;
             b < ebc_param.universal_batch_size * embedding_param.ev_size * hotness / num_gpus;
             ++b) {
          float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
          top_grads[gpu_id].push_back(HugeCTR::TypeConvert<emb_t, float>::convert(r));
        }
      }
    }
  };

  EmbeddingPlanner planner{ebc_param};
  planner.generate_embedding_plan_from_json_file(plan_file);

  std::vector<std::shared_ptr<core::CoreResourceManager>> core_resource_manager_list;
  std::vector<std::vector<std::unique_ptr<IEmbeddingTable>>> table_major_ebc_table_list;
  std::vector<std::unique_ptr<IEmbeddingCollectionForward>> ebc_forward_list;
  std::vector<std::unique_ptr<IEmbeddingCollectionBackward>> ebc_backward_list;
  std::vector<std::shared_ptr<core::CoreResourceManager>> core_list;

  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    auto core = std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, gpu_id);
    core_list.push_back(core);

    core_resource_manager_list.push_back(core);
    ebc_forward_list.push_back(std::move(planner.create_embedding_collection_forward(core)));
    ebc_backward_list.push_back(std::move(planner.create_embedding_collection_backward(core)));
  }

  auto table_major_global_embedding_sharding_param_list =
      planner.get_table_major_global_embedding_sharding_param_list();
  for (size_t table_id = 0; table_id < table_major_global_embedding_sharding_param_list.size();
       ++table_id) {
    table_major_ebc_table_list.push_back(
        create_embedding_table(resource_manager, core_list, ebc_param, table_param_list,
                               table_major_global_embedding_sharding_param_list[table_id]));
  }

  std::vector<core::Tensor> ebc_key_list;
  std::vector<core::Tensor> ebc_bucket_range_list;
  std::vector<size_t *> ebc_num_keys_list;
  std::vector<core::Tensor> ebc_top_grads;
  std::vector<core::Tensor> ebc_outptut;
  std::vector<std::vector<core::Tensor>> ebc_unique_key_list;
  std::vector<std::vector<size_t>> ebc_num_unique_key_list;
  std::vector<std::vector<core::Tensor>> ebc_unique_id_space_offset_list;
  std::vector<std::vector<size_t>> ebc_num_unique_id_space_offset_list;
  std::vector<std::vector<core::Tensor>> ebc_grad_ev_list;
  std::vector<std::vector<core::Tensor>> ebc_unique_dst_idx_list;
  std::vector<std::vector<core::Tensor>> ebc_unique_id_space_list_list;
  std::vector<std::vector<ContextContainer *>> context_container_list;
  ebc_unique_key_list.resize(num_gpus);
  ebc_num_unique_key_list.resize(num_gpus);
  ebc_unique_id_space_offset_list.resize(num_gpus);
  ebc_num_unique_id_space_offset_list.resize(num_gpus);
  ebc_grad_ev_list.resize(num_gpus);
  ebc_unique_dst_idx_list.resize(num_gpus);
  ebc_unique_id_space_list_list.resize(num_gpus);
  context_container_list.resize(num_gpus);
  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());
    auto buffer = GetBuffer(core_resource_manager_list[gpu_id]);

    int hotness_sum = 0;
    for (int embedding_id = 0; embedding_id < ebc_param.num_embedding; ++embedding_id) {
      auto &embedding_param = ebc_param.embedding_params[embedding_id];
      int hotness = embedding_param.hotness;
      hotness_sum += hotness;
    }

    ebc_key_list.push_back(buffer->reserve({ebc_param.universal_batch_size, hotness_sum},
                                           DeviceType::GPU, ebc_param.key_type));
    ebc_bucket_range_list.push_back(
        buffer->reserve({ebc_param.universal_batch_size * ebc_param.num_embedding + 1},
                        DeviceType::GPU, ebc_param.offset_type));
    ebc_num_keys_list.push_back(new size_t);

    int64_t num_emb_vec = 0;
    for (size_t i = 0; i < ebc_param.embedding_params.size(); ++i) {
      auto &emb_param = ebc_param.embedding_params[i];
      if (emb_param.combiner == Combiner::Concat) {
        num_emb_vec +=
            ebc_param.universal_batch_size * emb_param.ev_size * emb_param.hotness / num_gpus;
      } else {
        num_emb_vec += ebc_param.universal_batch_size * emb_param.ev_size / num_gpus;
      }
    }
    ebc_top_grads.push_back(buffer->reserve(num_emb_vec, DeviceType::GPU, ebc_param.emb_type));
    ebc_outptut.push_back(buffer->reserve(num_emb_vec, DeviceType::GPU, ebc_param.emb_type));
    buffer->allocate();
  }
  auto prepare_gpu_input = [&] {
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());

      ebc_key_list[gpu_id].copy_from(key_list);
      ebc_bucket_range_list[gpu_id].copy_from(bucket_range);
      *(ebc_num_keys_list[gpu_id]) = key_list.size();
      ebc_top_grads[gpu_id].copy_from(top_grads[gpu_id]);
    }
  };

  auto prepare_data = [&] {
    prepare_input();
    prepare_top_grads();
    prepare_gpu_input();
  };

  auto sync_gpus = [&]() {
    for (auto core : core_resource_manager_list) {
      cudaStreamSynchronize(core->get_local_gpu()->get_stream());
    }
  };
  // sync for emb table init
  sync_gpus();

  std::vector<std::vector<IEmbeddingTable *>> table_major_ebc_table_ptr_list;
  for (size_t i = 0; i < table_major_ebc_table_list.size(); ++i) {
    std::vector<IEmbeddingTable *> local_ebc_table_ptr_list;
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      local_ebc_table_ptr_list.push_back(table_major_ebc_table_list[i][gpu_id].get());
    }
    table_major_ebc_table_ptr_list.push_back(local_ebc_table_ptr_list);
  }

  EmbeddingCollectionCPU<key_t, offset_t, index_t, emb_t> ebc_cpu{
      num_gpus,
      num_table,
      ebc_param,
      planner.get_gpu_major_global_embedding_sharding_param_list(),
      table_major_ebc_table_ptr_list,
      table_param_list};

  EmbeddingReferenceCPU<key_t, offset_t, index_t, emb_t> emb_ref{
      num_gpus, num_table, ebc_param, table_major_ebc_table_ptr_list, table_param_list};

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

    std::cout << "compare ebc gpu emb output vs. emb reference emb output.\n";
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      std::vector<emb_t> gpu_emb_output;
      ebc_outptut[gpu_id].to(&gpu_emb_output);
      ASSERT_EQ(gpu_emb_output.size(), emb_ref.embedding_vec_[gpu_id].size());
      // std::cout << "forward ref output:\n";
      // print_array(emb_ref.embedding_vec_[gpu_id].size(), emb_ref.embedding_vec_[gpu_id]);
      // std::cout << "forward gpu output:\n";
      // print_array(gpu_emb_output.size(), gpu_emb_output);
      assert_array_eq(gpu_emb_output.size(), gpu_emb_output, ebc_cpu.embedding_vec_[gpu_id]);
    }
    std::cout << "\t>pass compare ebc gpu emb output vs. emb reference emb output.\n";
  };
  auto check_backward_result = [&] {
    std::cout << "compare ref grad info vs. ebc cpu grad info.\n";
    ASSERT_EQ(ebc_cpu.grad_info_.size(), emb_ref.accumulate_grad_map_.size());
    for (size_t i = 0; i < ebc_cpu.grad_info_.size(); ++i) {
      auto &cpu_grad_in_id_space = ebc_cpu.grad_info_.at(i);
      auto &ref_grad_in_id_space = emb_ref.accumulate_grad_map_.at(i);
      ASSERT_EQ(cpu_grad_in_id_space.size(), cpu_grad_in_id_space.size());

      for (auto p : cpu_grad_in_id_space) {
        auto &k = p.first;
        auto &cpu_gi = p.second;
        // std::cout << "id_space:" << i << ",key:" << k << "\n";
        ASSERT_TRUE(ref_grad_in_id_space.find(k) != ref_grad_in_id_space.end());
        auto &ref_gi = ref_grad_in_id_space.at(k);
        ASSERT_EQ(cpu_gi.size(), ref_gi.size());
        // std::cout << "cpu_gi:\n";
        // print_array(cpu_gi.size(), cpu_gi);
        // std::cout << "ref_gi:\n";
        // print_array(ref_gi.size(), ref_gi);
        if (ebc_param.emb_type.type() == HugeCTR::TensorScalarType::Float16) {
          assert_array_eq(cpu_gi.size(), cpu_gi, ref_gi);
        } else {
          assert_array_eq(cpu_gi.size(), cpu_gi, ref_gi);
        }
      }
    }
    std::cout << "\t>pass compare ref grad info vs. ebc cpu grad info.\n";

    std::cout << "compare ebc gpu grad info vs. ebc ref grad info.\n";
    std::unordered_set<int> unique_id_space;
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      for (size_t emb_table_id = 0; emb_table_id < table_major_ebc_table_list.size();
           ++emb_table_id) {
        std::vector<int> gpu_ebc_unique_id_space_list;
        ebc_unique_id_space_list_list[gpu_id][emb_table_id].to(&gpu_ebc_unique_id_space_list);
        for (size_t i = 0; i < gpu_ebc_unique_id_space_list.size(); ++i) {
          int id_space = gpu_ebc_unique_id_space_list[i];
          unique_id_space.insert(id_space);
        }
      }
    }

    ASSERT_EQ(unique_id_space.size(), emb_ref.accumulate_grad_map_.size());

    std::vector<std::unordered_map<key_t, std::vector<float>>> gpu_grad;
    gpu_grad.resize(unique_id_space.size());
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      for (size_t emb_table_id = 0; emb_table_id < table_major_ebc_table_list.size();
           ++emb_table_id) {
        std::vector<key_t> gpu_ebc_unique_key;
        ebc_unique_key_list[gpu_id][emb_table_id].to(&gpu_ebc_unique_key);
        // std::cout << "gpu_ebc_unique_key:\n";
        // print_array(gpu_ebc_unique_key.size(), gpu_ebc_unique_key);

        std::vector<uint32_t> gpu_ebc_unique_id_space_offset;
        ebc_unique_id_space_offset_list[gpu_id][emb_table_id].to(&gpu_ebc_unique_id_space_offset);
        // std::cout << "gpu_ebc_unique_id_space_offset:\n";
        // print_array(gpu_ebc_unique_id_space_offset.size(), gpu_ebc_unique_id_space_offset);

        std::vector<int> gpu_ebc_unique_id_space_list;
        ebc_unique_id_space_list_list[gpu_id][emb_table_id].to(&gpu_ebc_unique_id_space_list);
        // std::cout << "gpu_ebc_unique_id_space_list:\n";
        // print_array(gpu_ebc_unique_id_space_list.size(), gpu_ebc_unique_id_space_list);

        std::vector<float> gpu_ebc_grad_ev;
        ebc_grad_ev_list[gpu_id][emb_table_id].to(&gpu_ebc_grad_ev);
        // std::cout << "gpu_ebc_grad_ev:\n";
        // print_array(gpu_ebc_grad_ev.size(), gpu_ebc_grad_ev);

        std::vector<uint32_t> gpu_ebc_unique_dst_idx;
        ebc_unique_dst_idx_list[gpu_id][emb_table_id].to(&gpu_ebc_unique_dst_idx);
        // std::cout << "gpu_ebc_unique_dst_idx:\n";
        // print_array(gpu_ebc_unique_dst_idx.size(), gpu_ebc_unique_dst_idx);

        for (size_t i = 0; i < gpu_ebc_unique_id_space_list.size(); ++i) {
          int id_space = gpu_ebc_unique_id_space_list[i];

          uint32_t id_space_start = gpu_ebc_unique_id_space_offset[i];
          uint32_t id_space_end = gpu_ebc_unique_id_space_offset[i + 1];
          for (uint32_t r = id_space_start; r < id_space_end; ++r) {
            key_t k = gpu_ebc_unique_key[r];
            uint32_t dst_idx_start = gpu_ebc_unique_dst_idx[r];
            uint32_t dst_idx_end = gpu_ebc_unique_dst_idx[r + 1];
            std::vector<float> insert_ev;
            for (uint32_t e = 0; e < (dst_idx_end - dst_idx_start); ++e) {
              insert_ev.push_back(gpu_ebc_grad_ev[dst_idx_start + e]);
            }

            if (gpu_grad[id_space].find(k) == gpu_grad[id_space].end()) {
              gpu_grad[id_space][k] = insert_ev;
            } else {
              auto existed_ev = gpu_grad[id_space][k];
              ASSERT_EQ(existed_ev.size(), (dst_idx_end - dst_idx_start));
              assert_array_eq(dst_idx_end - dst_idx_start, existed_ev, insert_ev);
            }
          }
        }
      }
    }

    for (int id_space = 0; id_space < static_cast<int>(gpu_grad.size()); ++id_space) {
      auto &ref_grad_in_id_space = emb_ref.accumulate_grad_map_[id_space];
      auto &gpu_grad_in_id_space = gpu_grad[id_space];

      // std::cout << "ref grad in id_sapce:" << id_space << "\n";
      // for (auto p : ref_grad_in_id_space) {
      //   std::cout << "k:" << p.first << "\n";
      //   print_array(p.second.size(), p.second);
      // }
      // std::cout << "gpu grad in id_sapce:" << id_space << "\n";
      // for (auto p : gpu_grad_in_id_space) {
      //   std::cout << "k:" << p.first << "\n";
      //   print_array(p.second.size(), p.second);
      // }
      ASSERT_EQ(ref_grad_in_id_space.size(), gpu_grad_in_id_space.size());
      for (auto p : ref_grad_in_id_space) {
        auto &k = p.first;
        auto &cpu_gi = p.second;
        auto &gpu_gi = gpu_grad_in_id_space[k];
        ASSERT_EQ(cpu_gi.size(), gpu_gi.size());
        // std::cout << "id_space:" << id_space << ",k:" << k << "\n";
        // std::cout << "\tcpu_gi:";
        // print_array(cpu_gi.size(), cpu_gi);
        // std::cout << "\tgpu_gi:";
        // print_array(gpu_gi.size(), gpu_gi);
        if (ebc_param.emb_type.type() == HugeCTR::TensorScalarType::Float16) {
          assert_array_eq(cpu_gi.size(), cpu_gi, gpu_gi, 1e-1);
        } else {
          assert_array_eq(cpu_gi.size(), cpu_gi, gpu_gi,
                          1e-1);  // TODO: for DP the grad looks not exactly match. May need to take
                                  // a deeper look.
        }
      }
    }
    std::cout << "\t>pass compare ebc gpu grad info vs. ebc ref grad info.\n";
  };

  auto check_embedding_table = [&] {
    std::cout << "compare ref emb table vs. ebc cpu emb table.\n";
    const auto &cpu_emb_table = ebc_cpu.emb_table_cpu_.emb_table_list_;
    const auto &ref_emb_table = emb_ref.emb_table_cpu_.emb_table_list_;
    ASSERT_TRUE(cpu_emb_table.size() == ref_emb_table.size());

    for (size_t id_space = 0; id_space < cpu_emb_table.size(); ++id_space) {
      ASSERT_EQ(cpu_emb_table[id_space].size(), ref_emb_table[id_space].size());

      for (auto &[k, cpu_ev] : cpu_emb_table[id_space]) {
        ASSERT_TRUE(cpu_emb_table[id_space].find(k) != ref_emb_table[id_space].end());
        auto ref_ev = ref_emb_table[id_space].at(k);

        ASSERT_EQ(cpu_ev.size(), ref_ev.size());
        assert_array_eq(cpu_ev.size(), cpu_ev, ref_ev);
      }
    }
    std::cout << "\t>pass compare ref emb table vs. ebc cpu emb table.\n";

    EmbeddingTableCPU<key_t, index_t> copy_gpu_emb_table{num_table, table_major_ebc_table_ptr_list,
                                                         table_param_list};
    const auto &gpu_emb_table = copy_gpu_emb_table.emb_table_list_;

    std::cout << "compare ref emb table vs. ebc gpu emb table.\n";
    ASSERT_TRUE(gpu_emb_table.size() == ref_emb_table.size());

    for (size_t id_space = 0; id_space < gpu_emb_table.size(); ++id_space) {
      ASSERT_EQ(gpu_emb_table[id_space].size(), ref_emb_table[id_space].size());

      for (auto &[k, gpu_ev] : gpu_emb_table[id_space]) {
        ASSERT_TRUE(gpu_emb_table[id_space].find(k) != ref_emb_table[id_space].end());
        auto ref_ev = ref_emb_table[id_space].at(k);

        ASSERT_EQ(gpu_ev.size(), ref_ev.size());
        assert_array_eq(gpu_ev.size(), gpu_ev, ref_ev);
      }
    }
    std::cout << "\t>pass compare ref emb table vs. ebc gpu emb table.\n";
  };

  int num_iteration = 10;
  for (int iter = 0; iter < num_iteration; ++iter) {
    std::cout << "iter:" << iter << "\n";
    prepare_data();
    sync_gpus();

    // forward
    ebc_cpu.embedding_forward_cpu(key_list, bucket_range);
    emb_ref.embedding_forward_cpu(key_list, bucket_range);

#pragma omp parallel for num_threads(num_gpus)
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());

      std::vector<ILookup *> lookup_list;
      for (size_t table_id = 0; table_id < table_major_ebc_table_ptr_list.size(); ++table_id) {
        lookup_list.push_back(
            dynamic_cast<ILookup *>(table_major_ebc_table_ptr_list[table_id][gpu_id]));
      }
      ebc_forward_list[gpu_id]->forward_per_gpu(
          ebc_key_list[gpu_id], ebc_bucket_range_list[gpu_id], *ebc_num_keys_list[gpu_id], Tensor(),
          lookup_list, ebc_outptut[gpu_id], &context_container_list[gpu_id]);
    }

    sync_gpus();
    check_forward_result();

    // backward
    ebc_cpu.embedding_backward_cpu(top_grads, batch_size);
    emb_ref.embedding_backward_cpu(top_grads, key_list, bucket_range);

#pragma omp parallel for num_threads(num_gpus)
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());

      ebc_backward_list[gpu_id]->backward_per_gpu(
          context_container_list[gpu_id], ebc_top_grads[gpu_id], &ebc_unique_key_list[gpu_id],
          &ebc_num_unique_key_list[gpu_id], &ebc_unique_id_space_offset_list[gpu_id],
          &ebc_num_unique_id_space_offset_list[gpu_id], &ebc_grad_ev_list[gpu_id],
          &ebc_unique_dst_idx_list[gpu_id], &ebc_unique_id_space_list_list[gpu_id], true);
    }
    sync_gpus();
    check_backward_result();

    // update
    ebc_cpu.embedding_update_cpu();
    emb_ref.embedding_update_cpu();

#pragma omp parallel for num_threads(num_gpus)
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());
      for (size_t table_id = 0; table_id < table_major_ebc_table_list.size(); ++table_id) {
        table_major_ebc_table_list[table_id][gpu_id]->update(
            ebc_unique_key_list[gpu_id][table_id], ebc_num_unique_key_list[gpu_id][table_id],
            ebc_unique_id_space_offset_list[gpu_id][table_id],
            ebc_num_unique_id_space_offset_list[gpu_id][table_id],
            ebc_unique_id_space_list_list[gpu_id][table_id], ebc_grad_ev_list[gpu_id][table_id],
            ebc_unique_dst_idx_list[gpu_id][table_id]);
      }
    }
    sync_gpus();

    check_embedding_table();

    {
      for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        for (ContextContainer *context_container : context_container_list[gpu_id]) {
          delete context_container;
        }
      }
    }
  }

  // legacy intermidia result check. but the code is broken now.
  {
    // {
    //   std::cout << "key_list:\n";
    //   print_array(key_list.size(), key_list);

    //   std::cout << "bucket_range:\n";
    //   print_array(bucket_range.size(), bucket_range);

    //   std::cout << "top grads:\n";
    //   for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    //     std::cout << "gpu:" << gpu_id << "\n";
    //     print_array(top_grads[gpu_id].size(), top_grads[gpu_id]);
    //   }

    //   std::cout << "cpu emb vec:\n";
    //   auto &cpu_emb_vec = ebc_cpu.embedding_vec_;
    //   for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    //     std::cout << "gpu:" << gpu_id << "\n";
    //     print_array(cpu_emb_vec[gpu_id].size(), cpu_emb_vec[gpu_id]);
    //   }

    //   std::cout << "ref emb vec:\n";
    //   auto &ref_emb_vec = emb_ref.embedding_vec_;
    //   for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    //     std::cout << "gpu:" << gpu_id << "\n";
    //     print_array(ref_emb_vec[gpu_id].size(), ref_emb_vec[gpu_id]);
    //   }
    // }

    // auto compare_for_all_gpus = [&](const std::string &compare_info, TablePlacementStrategy tps,
    //                                 auto compare_func) {
    //   std::cout << "compare " << compare_info << ".\n";
    //   for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    //     int shard_idx = ebc_cpu.get_shard_idx(gpu_id, tps);
    //     if (shard_idx < 0) continue;
    //     compare_func(shard_idx, gpu_id);
    //   }
    //   std::cout << "pass compare " << compare_info << ".\n";
    // };
    // {
    //   compare_for_all_gpus(
    //       "model idx", TablePlacementStrategy::Localized, [&](int shard_idx, int gpu_id) {
    //         ContextContainer *context = context_container_list[gpu_id][shard_idx];
    //         Tensor model_key = context->unpack<core::Tensor>("model_key");
    //         size_t num_model_key = context->unpack<size_t>("num_model_key");

    //         // std::cout << "ebc gpu num_model_key:" << num_model_key << "\n";
    //         // {
    //         //   std::cout << "ebc gpu model_key:\n";
    //         //   std::vector<key_t> cpu_model_key;
    //         //   model_key.to(&cpu_model_key);
    //         //   print_array(num_model_key, cpu_model_key);
    //         // }

    //         // std::cout << "ebc cpu num_model_key:" << ebc_cpu.mp_model_key_list_[gpu_id].size()
    //         //           << "\n";
    //         // {
    //         //   std::cout << "ebc cpu model_key:\n";
    //         //   print_array(ebc_cpu.mp_model_key_list_[gpu_id].size(),
    //         //               ebc_cpu.mp_model_key_list_[gpu_id]);
    //         // }

    //         ASSERT_EQ(num_model_key, ebc_cpu.mp_model_key_list_[gpu_id].size());
    //         std::vector<key_t> cpu_model_key;
    //         model_key.to(&cpu_model_key);
    //         assert_array_eq(num_model_key, cpu_model_key, ebc_cpu.mp_model_key_list_[gpu_id]);
    //       });

    //   compare_for_all_gpus("model offsets", TablePlacementStrategy::Localized,
    //                        [&](int shard_idx, int gpu_id) {
    //                          ContextContainer *context =
    //                          context_container_list[gpu_id][shard_idx]; Tensor model_offsets =
    //                          context->unpack<core::Tensor>("model_offsets");
    //                          std::vector<uint32_t> cpu_model_offsets;
    //                          model_offsets.to(&cpu_model_offsets);
    //                          assert_array_eq(cpu_model_offsets.size(), cpu_model_offsets,
    //                                          ebc_cpu.mp_model_offset_list_[gpu_id]);
    //                        });

    //   compare_for_all_gpus("num key in bucket", TablePlacementStrategy::Localized,
    //                        [&](int shard_idx, int gpu_id) {
    //                          ContextContainer *context =
    //                          context_container_list[gpu_id][shard_idx]; Tensor
    //                          num_key_in_bucket_for_combiner =
    //                              context->unpack<core::Tensor>("num_key_in_bucket_for_combiner");

    //                          std::vector<uint32_t> cpu_num_key_in_bucket;
    //                          num_key_in_bucket_for_combiner.to(&cpu_num_key_in_bucket);
    //                          assert_array_eq(cpu_num_key_in_bucket.size(), cpu_num_key_in_bucket,
    //                                          ebc_cpu.mp_model_num_key_in_bucket_list_[gpu_id]);
    //                        });

    //   compare_for_all_gpus("model comm buffer", TablePlacementStrategy::Localized,
    //                        [&](int shard_idx, int gpu_id) {
    //                          ContextContainer *context =
    //                          context_container_list[gpu_id][shard_idx]; auto
    //                          model_comm_buffer_list =
    //                              context->unpack<std::vector<core::Tensor>>("model_comm_buffer_list");
    //                          auto model_comm_buffer_size =
    //                              context->unpack<std::vector<size_t>>("model_comm_buffer_size");

    //                          // std::cout << "gpu " << gpu_id << ", gpu model comm buffer:\n";
    //                          // for (int dst_gpu_id = 0; dst_gpu_id < num_gpus; ++dst_gpu_id) {
    //                          //   std::cout << "\tdst gpu id:" << dst_gpu_id << "\n";
    //                          //   std::cout << "\tdata:";
    //                          //   std::vector<emb_t> cpu_comm_vec;
    //                          //   model_comm_buffer_list[dst_gpu_id].to(&cpu_comm_vec);
    //                          //   print_array(cpu_comm_vec.size(), cpu_comm_vec);
    //                          // }

    //                          // std::cout << "gpu " << gpu_id << ", cpu model comm buffer:\n";
    //                          // for (int dst_gpu_id = 0; dst_gpu_id < num_gpus; ++dst_gpu_id) {
    //                          //   std::cout << "\tdst gpu id:" << dst_gpu_id << "\n";
    //                          //   std::cout << "\tdata:";
    //                          //   std::vector<emb_t> &cpu_comm_vec =
    //                          //   ebc_cpu.model_comm_buffer_list_[gpu_id][dst_gpu_id];
    //                          //   print_array(cpu_comm_vec.size(), cpu_comm_vec);
    //                          // }

    //                          for (int dst_gpu_id = 0; dst_gpu_id < num_gpus; ++dst_gpu_id) {
    //                            std::vector<emb_t> &ref_cpu_comm_vec =
    //                                ebc_cpu.model_comm_buffer_list_[gpu_id][dst_gpu_id];
    //                            std::vector<emb_t> cpu_comm_vec;
    //                            model_comm_buffer_list[dst_gpu_id].to(&cpu_comm_vec);

    //                            ASSERT_EQ(model_comm_buffer_size[dst_gpu_id],
    //                            ref_cpu_comm_vec.size()); assert_array_eq(cpu_comm_vec.size(),
    //                            ref_cpu_comm_vec, cpu_comm_vec);
    //                          }
    //                        });

    //   compare_for_all_gpus(
    //       "mp backward idx", TablePlacementStrategy::Localized, [&](int shard_idx, int gpu_id)
    //       {
    //         ContextContainer *context = context_container_list[gpu_id][shard_idx];
    //         auto unique_key = context->unpack<core::Tensor>("unique_key");
    //         auto unique_dst_idx = context->unpack<core::Tensor>("unique_dst_idx");
    //         auto sorted_bucket_id_list = context->unpack<core::Tensor>("sorted_bucket_id_list");
    //         auto sorted_bucket_id_offset =
    //         context->unpack<core::Tensor>("sorted_bucket_id_offset"); auto unique_id_space_list =
    //         context->unpack<core::Tensor>("unique_id_space_list"); auto unique_id_space_offset =
    //         context->unpack<core::Tensor>("unique_id_space_offset");

    //         std::vector<key_t> gpu_unique_key;
    //         unique_key.to(&gpu_unique_key);
    //         // std::cout << "gpu_unique_key:\n";
    //         // auto num_unique_key = context->unpack<size_t>("num_unique_key");
    //         // print_array(num_unique_key, gpu_unique_key);

    //         std::vector<uint32_t> gpu_unique_dst_idx;
    //         unique_dst_idx.to(&gpu_unique_dst_idx);
    //         // std::cout << "gpu_unique_dst_idx:\n";
    //         // print_array(gpu_unique_dst_idx.size(), gpu_unique_dst_idx);

    //         std::vector<uint32_t> gpu_sorted_bucket_id_list;
    //         sorted_bucket_id_list.to(&gpu_sorted_bucket_id_list);
    //         // std::cout << "gpu_sorted_bucket_id_list:\n";
    //         // print_array(gpu_sorted_bucket_id_list.size(), gpu_sorted_bucket_id_list);

    //         std::vector<uint32_t> gpu_sorted_bucket_id_offset;
    //         sorted_bucket_id_offset.to(&gpu_sorted_bucket_id_offset);
    //         // std::cout << "gpu_sorted_bucket_id_offset:\n";
    //         // print_array(gpu_sorted_bucket_id_offset.size(), gpu_sorted_bucket_id_offset);

    //         std::vector<int> gpu_mp_unique_id_space_list;
    //         unique_id_space_list.to(&gpu_mp_unique_id_space_list);
    //         // std::cout << "gpu_mp_unique_id_space_list:\n";
    //         // print_array(gpu_mp_unique_id_space_list.size(), gpu_mp_unique_id_space_list);

    //         std::vector<uint32_t> gpu_unique_id_space_offset;
    //         unique_id_space_offset.to(&gpu_unique_id_space_offset);
    //         // std::cout << "gpu_unique_id_space_offset:\n";
    //         // print_array(gpu_unique_id_space_offset.size(), gpu_unique_id_space_offset);

    //         ASSERT_EQ(ebc_cpu.mp_unique_id_space_list_[gpu_id].size(),
    //                   gpu_mp_unique_id_space_list.size());
    //         uint32_t dst_idx = 0;
    //         for (size_t idx = 0; idx < gpu_mp_unique_id_space_list.size(); ++idx) {
    //           int id_space = gpu_mp_unique_id_space_list[idx];
    //           ASSERT_EQ(id_space, ebc_cpu.mp_unique_id_space_list_[gpu_id][idx]);
    //           auto &backward_idx_info = ebc_cpu.mp_backward_info_[gpu_id][idx];
    //           uint32_t start = gpu_unique_id_space_offset[idx];
    //           uint32_t end = gpu_unique_id_space_offset[idx + 1];
    //           int ev_size = ebc_cpu.mp_unique_ev_size_list_[gpu_id][idx];
    //           ASSERT_EQ(backward_idx_info.size(), end - start);

    //           for (uint32_t r = 0; r < (end - start); ++r) {
    //             key_t k = gpu_unique_key[r + start];
    //             uint32_t bucket_id_start = gpu_sorted_bucket_id_offset[r + start];
    //             uint32_t bucket_id_end = gpu_sorted_bucket_id_offset[r + start + 1];
    //             ASSERT_EQ(dst_idx, gpu_unique_dst_idx[r + start]);
    //             dst_idx += ev_size;
    //             ASSERT_EQ(backward_idx_info[k].size(), (bucket_id_end - bucket_id_start));
    //             // std::cout << "backward_idx_info[k]:" << k << "\n";
    //             // print_array(backward_idx_info[k].size(), backward_idx_info[k]);
    //             for (uint32_t b = 0; b < (bucket_id_end - bucket_id_start); ++b) {
    //               uint32_t bucket_id = gpu_sorted_bucket_id_list[b + bucket_id_start];
    //               ASSERT_TRUE(std::find(backward_idx_info[k].begin(), backward_idx_info[k].end(),
    //                                     bucket_id) != backward_idx_info[k].end());
    //             }
    //           }
    //         }
    //       });

    //   compare_for_all_gpus(
    //       "network comm buffer", TablePlacementStrategy::Localized,
    //       [&](int shard_idx, int gpu_id) {
    //         ContextContainer *context = context_container_list[gpu_id][shard_idx];
    //         auto network_comm_buffer_list =
    //             context->unpack<std::vector<core::Tensor>>("network_comm_buffer_list");
    //         auto network_comm_buffer_size =
    //             context->unpack<std::vector<size_t>>("network_comm_buffer_size");

    //         // std::cout << "gpu " << gpu_id << ", gpu network comm buffer:\n";
    //         // for (int src_gpu_id = 0; src_gpu_id < num_gpus; ++src_gpu_id) {
    //         //   std::cout << "\tsrc gpu id:" << src_gpu_id << "\n";
    //         //   std::cout << "\tdata:";
    //         //   std::vector<emb_t> cpu_comm_vec;
    //         //   network_comm_buffer_list[src_gpu_id].to(&cpu_comm_vec);
    //         //   print_array(cpu_comm_vec.size(), cpu_comm_vec);
    //         // }

    //         // std::cout << "gpu " << gpu_id << ", cpu network comm buffer:\n";
    //         // for (int src_gpu_id = 0; src_gpu_id < num_gpus; ++src_gpu_id) {
    //         //   std::cout << "\tsrc gpu id:" << src_gpu_id << "\n";
    //         //   std::cout << "\tdata:";
    //         //   std::vector<emb_t> &cpu_comm_vec =
    //         //       ebc_cpu.network_comm_buffer_list_[gpu_id][src_gpu_id];
    //         //   print_array(cpu_comm_vec.size(), cpu_comm_vec);
    //         // }

    //         for (int dst_gpu_id = 0; dst_gpu_id < num_gpus; ++dst_gpu_id) {
    //           std::vector<emb_t> &ref_cpu_comm_vec =
    //               ebc_cpu.network_comm_buffer_list_[gpu_id][dst_gpu_id];
    //           std::vector<emb_t> cpu_comm_vec;
    //           network_comm_buffer_list[dst_gpu_id].to(&cpu_comm_vec);

    //           ASSERT_EQ(network_comm_buffer_size[dst_gpu_id], ref_cpu_comm_vec.size());
    //           assert_array_eq(cpu_comm_vec.size(), ref_cpu_comm_vec, cpu_comm_vec);
    //         }
    //       });

    //   compare_for_all_gpus(
    //       "network idx", TablePlacementStrategy::Localized, [&](int shard_idx, int gpu_id) {
    //         ContextContainer *context = context_container_list[gpu_id][shard_idx];
    //         auto dst_embedding_id_list = context->unpack<core::Tensor>("dst_embedding_id_list");
    //         auto num_dst_embedding_id = context->unpack<size_t>("num_dst_embedding_id");
    //         auto network_idx = context->unpack<core::Tensor>("network_idx");
    //         auto network_gpu_idx = context->unpack<core::Tensor>("network_gpu_idx");
    //         auto network_offset = context->unpack<core::Tensor>("network_offset");

    //         ASSERT_EQ(num_dst_embedding_id, ebc_cpu.dst_embedding_id_list_[gpu_id].size());
    //         std::vector<int> cpu_dst_embedding_id_list;
    //         dst_embedding_id_list.to(&cpu_dst_embedding_id_list);
    //         assert_array_eq(num_dst_embedding_id, cpu_dst_embedding_id_list,
    //                         ebc_cpu.dst_embedding_id_list_[gpu_id]);

    //         std::vector<int> cpu_network_idx;
    //         network_idx.to(&cpu_network_idx);
    //         ASSERT_EQ(cpu_network_idx.size(), ebc_cpu.network_idx_[gpu_id].size());
    //         assert_array_eq(cpu_network_idx.size(), cpu_network_idx,
    //         ebc_cpu.network_idx_[gpu_id]);

    //         std::vector<int> cpu_network_gpu_idx;
    //         network_gpu_idx.to(&cpu_network_gpu_idx);
    //         ASSERT_EQ(cpu_network_gpu_idx.size(), ebc_cpu.network_gpu_idx_[gpu_id].size());
    //         assert_array_eq(cpu_network_gpu_idx.size(), cpu_network_gpu_idx,
    //                         ebc_cpu.network_gpu_idx_[gpu_id]);

    //         std::vector<int> cpu_network_offset;
    //         network_offset.to(&cpu_network_offset);
    //         ASSERT_EQ(cpu_network_offset.size(), ebc_cpu.network_offset_[gpu_id].size());
    //         assert_array_eq(cpu_network_offset.size(), cpu_network_offset,
    //                         ebc_cpu.network_offset_[gpu_id]);
    //       });

    //   compare_for_all_gpus(
    //       "dp idx", TablePlacementStrategy::DataParallel, [&](int shard_idx, int gpu_id) {
    //         ContextContainer *context = context_container_list[gpu_id][shard_idx];
    //         auto dp_key = context->unpack<core::Tensor>("dp_key");
    //         auto num_dp_key = context->unpack<size_t>("num_dp_key");
    //         auto dp_offset = context->unpack<core::Tensor>("dp_offset");

    //         ASSERT_EQ(ebc_cpu.dp_key_list_[gpu_id].size(), num_dp_key);
    //         std::vector<key_t> cpu_dp_key;
    //         dp_key.to(&cpu_dp_key);
    //         assert_array_eq(num_dp_key, cpu_dp_key, ebc_cpu.dp_key_list_[gpu_id]);

    //         std::vector<uint32_t> cpu_dp_offset;
    //         dp_offset.to(&cpu_dp_offset);
    //         ASSERT_EQ(cpu_dp_offset.size(), ebc_cpu.dp_offset_list_[gpu_id].size());
    //         assert_array_eq(cpu_dp_offset.size(), cpu_dp_offset,
    //         ebc_cpu.dp_offset_list_[gpu_id]);
    //       });

    //   compare_for_all_gpus(
    //       "dp backward idx", TablePlacementStrategy::DataParallel, [&](int shard_idx, int
    //       gpu_id) {
    //         ContextContainer *context = context_container_list[gpu_id][shard_idx];
    //         auto unique_key = context->unpack<core::Tensor>("unique_key");
    //         auto num_unique_key = context->unpack<size_t>("num_unique_key");
    //         auto unique_dst_idx = context->unpack<core::Tensor>("unique_dst_idx");
    //         auto sorted_bucket_id_list = context->unpack<core::Tensor>("sorted_bucket_id_list");
    //         auto sorted_bucket_id_offset =
    //         context->unpack<core::Tensor>("sorted_bucket_id_offset"); auto unique_id_space_list =
    //         context->unpack<core::Tensor>("unique_id_space_list"); auto unique_id_space_offset =
    //         context->unpack<core::Tensor>("unique_id_space_offset");

    //         std::vector<key_t> gpu_unique_key;
    //         unique_key.to(&gpu_unique_key);

    //         std::vector<key_t> gpu_unique_id_space_list;
    //         unique_id_space_list.to(&gpu_unique_id_space_list);

    //         std::vector<uint32_t> gpu_unique_id_space_offset;
    //         unique_id_space_offset.to(&gpu_unique_id_space_offset);

    //         std::vector<uint32_t> gpu_unique_dst_idx;
    //         unique_dst_idx.to(&gpu_unique_dst_idx);

    //         std::vector<uint32_t> gpu_sorted_bucket_id_list;
    //         sorted_bucket_id_list.to(&gpu_sorted_bucket_id_list);

    //         std::vector<uint32_t> gpu_sorted_bucket_id_offset;
    //         sorted_bucket_id_offset.to(&gpu_sorted_bucket_id_offset);

    //         auto local_embedding_list = ebc_cpu.get_local_embedding_list(shard_idx, gpu_id);
    //         int dst_idx = 0;
    //         ASSERT_EQ(gpu_unique_id_space_offset.back(), num_unique_key);
    //         for (int idx = 0; idx < static_cast<int>(gpu_unique_id_space_offset.size()) - 1;
    //         ++idx)
    //         {
    //           uint32_t id_space_start = gpu_unique_id_space_offset[idx];
    //           uint32_t id_space_end = gpu_unique_id_space_offset[idx + 1];
    //           int id_space = gpu_unique_id_space_list[idx];
    //           int embedding_id = local_embedding_list[idx];
    //           int ev_size = ebc_cpu.ev_size_list_[embedding_id];
    //           ASSERT_EQ(id_space, ebc_cpu.get_local_id_space_list(shard_idx, gpu_id)[idx]);

    //           auto &dp_backward_info_in_current_id_space = ebc_cpu.dp_backward_info_[idx];

    //           std::vector<key_t> dp_key_in_id_space;
    //           for (auto p : dp_backward_info_in_current_id_space) {
    //             dp_key_in_id_space.push_back(p.first);
    //           }
    //           std::sort(dp_key_in_id_space.begin(), dp_key_in_id_space.end());

    //           std::vector<key_t> gpu_unique_key_in_id_space;
    //           for (uint32_t r = id_space_start; r < id_space_end; ++r) {
    //             key_t k = gpu_unique_key[r];
    //             gpu_unique_key_in_id_space.push_back(k);
    //           }
    //           std::sort(gpu_unique_key_in_id_space.begin(), gpu_unique_key_in_id_space.end());

    //           // std::cout << "dp_key_in_id_space:\n";
    //           // print_array(dp_key_in_id_space.size(), dp_key_in_id_space);
    //           // std::cout << "gpu_unique_key_in_id_space:\n";
    //           // print_array(gpu_unique_key_in_id_space.size(), gpu_unique_key_in_id_space);

    //           ASSERT_EQ(dp_key_in_id_space.size(), gpu_unique_key_in_id_space.size());
    //           assert_array_eq(dp_key_in_id_space.size(), dp_key_in_id_space,
    //                           gpu_unique_key_in_id_space);

    //           for (uint32_t r = id_space_start; r < id_space_end; ++r) {
    //             ASSERT_EQ(dst_idx, gpu_unique_dst_idx[r]);
    //             dst_idx += ev_size;
    //           }

    //           std::unordered_map<key_t, std::vector<uint32_t>>
    //               gpu_unique_key_and_bucket_id_list_in_id_space;
    //           for (uint32_t r = id_space_start; r < id_space_end; ++r) {
    //             key_t k = gpu_unique_key[r];
    //             uint32_t start = gpu_sorted_bucket_id_offset[r];
    //             uint32_t end = gpu_sorted_bucket_id_offset[r + 1];
    //             for (uint32_t i = start; i < end; ++i) {
    //               int local_bucket_id = gpu_sorted_bucket_id_list[i];
    //               int embedding_id = local_bucket_id / batch_size_per_gpu;
    //               int local_batch_id = local_bucket_id % batch_size_per_gpu;
    //               int bucket_id =
    //                   embedding_id * batch_size + (gpu_id * batch_size_per_gpu + local_batch_id);
    //               gpu_unique_key_and_bucket_id_list_in_id_space[k].push_back(bucket_id);
    //             }
    //           }

    //           for (auto &p : gpu_unique_key_and_bucket_id_list_in_id_space) {
    //             key_t k = p.first;
    //             auto &bucket_id_list = p.second;
    //             auto &ref_bucket_id_list = dp_backward_info_in_current_id_space[k];
    //             // ASSERT_EQ(bucket_id_list.size(), ref_bucket_id_list.size());
    //             for (auto bucket_id : bucket_id_list) {
    //               ASSERT_TRUE(std::find(ref_bucket_id_list.begin(), ref_bucket_id_list.end(),
    //                                     bucket_id) != ref_bucket_id_list.end());
    //             }
    //           }
    //         }
    //       });
    // }
    //   compare_for_all_gpus("backward model comm buffer", TablePlacementStrategy::Localized,
    //                        [&](int shard_idx, int gpu_id) {
    //                          ContextContainer *context =
    //                          context_container_list[gpu_id][shard_idx]; auto
    //                          model_comm_buffer_list =
    //                              context->unpack<std::vector<core::Tensor>>("model_comm_buffer_list");
    //                          auto model_comm_buffer_size =
    //                              context->unpack<std::vector<size_t>>("model_comm_buffer_size");

    //                          //  std::cout << "gpu " << gpu_id << ", gpu model comm buffer:\n";
    //                          //  for (int dst_gpu_id = 0; dst_gpu_id < num_gpus; ++dst_gpu_id) {
    //                          //    std::cout << "\tdst gpu id:" << dst_gpu_id << "\n";
    //                          //    std::cout << "\tdata:";
    //                          //    std::vector<emb_t> cpu_comm_vec;
    //                          //    model_comm_buffer_list[dst_gpu_id].to(&cpu_comm_vec);
    //                          //    print_array(cpu_comm_vec.size(), cpu_comm_vec);
    //                          //  }

    //                          //  std::cout << "gpu " << gpu_id << ", cpu model comm buffer:\n";
    //                          //  for (int dst_gpu_id = 0; dst_gpu_id < num_gpus; ++dst_gpu_id) {
    //                          //    std::cout << "\tdst gpu id:" << dst_gpu_id << "\n";
    //                          //    std::cout << "\tdata:";
    //                          //    std::vector<emb_t> &cpu_comm_vec =
    //                          //        ebc_cpu.model_comm_buffer_list_[gpu_id][dst_gpu_id];
    //                          //    print_array(cpu_comm_vec.size(), cpu_comm_vec);
    //                          //  }

    //                          for (int dst_gpu_id = 0; dst_gpu_id < num_gpus; ++dst_gpu_id) {
    //                            std::vector<emb_t> &ref_cpu_comm_vec =
    //                                ebc_cpu.model_comm_buffer_list_[gpu_id][dst_gpu_id];
    //                            std::vector<emb_t> cpu_comm_vec;
    //                            model_comm_buffer_list[dst_gpu_id].to(&cpu_comm_vec);

    //                            ASSERT_EQ(model_comm_buffer_size[dst_gpu_id],
    //                            ref_cpu_comm_vec.size()); assert_array_eq(cpu_comm_vec.size(),
    //                            ref_cpu_comm_vec, cpu_comm_vec);
    //                          }
    //                        });

    //   compare_for_all_gpus(
    //       "backward network comm buffer", TablePlacementStrategy::Localized,
    //       [&](int shard_idx, int gpu_id) {
    //         ContextContainer *context = context_container_list[gpu_id][shard_idx];
    //         auto network_comm_buffer_list =
    //             context->unpack<std::vector<core::Tensor>>("network_comm_buffer_list");
    //         auto network_comm_buffer_size =
    //             context->unpack<std::vector<size_t>>("network_comm_buffer_size");

    //         for (int dst_gpu_id = 0; dst_gpu_id < num_gpus; ++dst_gpu_id) {
    //           std::vector<emb_t> &ref_cpu_comm_vec =
    //               ebc_cpu.network_comm_buffer_list_[gpu_id][dst_gpu_id];
    //           std::vector<emb_t> cpu_comm_vec;
    //           network_comm_buffer_list[dst_gpu_id].to(&cpu_comm_vec);

    //           ASSERT_EQ(network_comm_buffer_size[dst_gpu_id], ref_cpu_comm_vec.size());
    //           assert_array_eq(cpu_comm_vec.size(), ref_cpu_comm_vec, cpu_comm_vec);
    //         }
    //       });
  }
}

const std::vector<int> gpus = {0, 1};

namespace normal {
const int batch_size = 1024;
const int num_table = 5;
const std::vector<int> table_ev_size_list = {128, 32, 64, 16, 8};
const int num_embedding = 5;
const std::vector<int> id_space_list = {0, 1, 2, 3, 4};
const std::vector<int> hotness_list = {8, 20, 10, 5, 8};
// const std::vector<Combiner> combiner_list = {
//     Combiner::Average, Combiner::Average, Combiner::Average, Combiner::Average,
//     Combiner::Average};
const std::vector<Combiner> combiner_list = {Combiner::Sum, Combiner::Sum, Combiner::Sum,
                                             Combiner::Sum, Combiner::Sum};
const std::vector<int> table_min_key_list = {0, 0, 0, 0, 0};
const std::vector<int> table_max_key_list = {100, 100, 100, 100, 100};

TEST(test_embedding_collection, plan_0) {
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, float>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_0.json");
}

TEST(test_embedding_collection, plan_0_i64) {
  embedding_collection_e2e<int64_t, int64_t, uint64_t, float>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_0.json");
}

TEST(test_embedding_collection, plan_0_half) {
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, __half>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_0.json");
}

TEST(test_embedding_collection, plan_0_i64_half) {
  embedding_collection_e2e<int64_t, int64_t, uint64_t, __half>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_0.json");
}

TEST(test_embedding_collection, plan_1) {
  embedding_collection_e2e<uint32_t, uint32_t, uint32_t, float>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_1.json");
}

TEST(test_embedding_collection, plan_1_half) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, __half>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_1.json");
}

TEST(test_embedding_collection, plan_2) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, float>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_2.json");
}

TEST(test_embedding_collection, plan_2_half) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, __half>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_2.json");
}

TEST(test_embedding_collection, plan_3) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, float>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_3.json");
}

TEST(test_embedding_collection, plan_3_half) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, __half>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_3.json");
}

}  // namespace normal

namespace concat_combiner {
const int batch_size = 1024;
const int num_table = 5;
const std::vector<int> table_ev_size_list = {128, 32, 64, 16, 8};
const int num_embedding = 5;
const std::vector<int> id_space_list = {0, 1, 2, 3, 4};
const std::vector<int> hotness_list = {8, 20, 10, 5, 8};
const std::vector<Combiner> combiner_list = {Combiner::Concat, Combiner::Average, Combiner::Concat,
                                             Combiner::Sum, Combiner::Sum};
const std::vector<int> table_min_key_list = {0, 0, 0, 0, 0};
const std::vector<int> table_max_key_list = {1000, 1000, 1000, 1000, 1000};

TEST(test_embedding_collection, plan_0_concat) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, float>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_0.json");
}

TEST(test_embedding_collection, plan_0_concat_half) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, __half>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_0.json");
}

TEST(test_embedding_collection, plan_2_concat) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, float>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_2.json");
}

TEST(test_embedding_collection, plan_2_concat_half) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, __half>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_2.json");
}

TEST(test_embedding_collection, plan_3_concat) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, float>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_3.json");
}

TEST(test_embedding_collection, plan_3_concat_half) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, __half>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_3.json");
}
}  // namespace concat_combiner

namespace share_embedding_table {
const int batch_size = 1024;
const int num_table = 4;
const std::vector<int> table_ev_size_list = {128, 32, 64, 16};
const int num_embedding = 5;
const std::vector<int> id_space_list = {0, 1, 0, 2, 3};
const std::vector<int> hotness_list = {8, 20, 10, 5, 8};
const std::vector<Combiner> combiner_list = {Combiner::Sum, Combiner::Average, Combiner::Sum,
                                             Combiner::Sum, Combiner::Sum};
const std::vector<int> table_min_key_list = {0, 0, 0, 0};
const std::vector<int> table_max_key_list = {1000, 1000, 1000, 1000};

TEST(test_embedding_collection, plan_0_share_id_space) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, float>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_0.json");
}

TEST(test_embedding_collection, plan_0_share_id_space_half) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, __half>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_0.json");
}

TEST(test_embedding_collection, plan_2_share_id_space) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, float>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_2.json");
}

TEST(test_embedding_collection, plan_2_share_id_space_half) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, __half>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_2.json");
}

TEST(test_embedding_collection, plan_3_share_id_space) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, float>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_3.json");
}

TEST(test_embedding_collection, plan_3_share_id_space_half) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, __half>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_3.json");
}
}  // namespace share_embedding_table

namespace share_embedding_table_and_concat_combiner {
const int batch_size = 1024;
const int num_table = 4;
const std::vector<int> table_ev_size_list = {128, 32, 64, 16};
const int num_embedding = 5;
const std::vector<int> id_space_list = {0, 1, 0, 2, 3};
const std::vector<int> hotness_list = {8, 20, 10, 5, 8};
const std::vector<Combiner> combiner_list = {Combiner::Concat, Combiner::Average, Combiner::Sum,
                                             Combiner::Sum, Combiner::Sum};
const std::vector<int> table_min_key_list = {0, 0, 0, 0};
const std::vector<int> table_max_key_list = {1000, 1000, 1000, 1000};

TEST(test_embedding_collection, plan_0_share_id_space_and_concat) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, float>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_0.json");
}

TEST(test_embedding_collection, plan_0_share_id_space_and_concat_half) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, __half>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_0.json");
}

TEST(test_embedding_collection, plan_2_share_id_space_and_concat) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, float>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_2.json");
}

TEST(test_embedding_collection, plan_2_share_id_space_and_concat_half) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, __half>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_2.json");
}

TEST(test_embedding_collection, plan_3_share_id_space_and_concat) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, float>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_3.json");
}

TEST(test_embedding_collection, plan_3_share_id_space_and_concat_half) {
  embedding_collection_e2e<int32_t, uint32_t, uint32_t, __half>(
      gpus, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_3.json");
}
}  // namespace share_embedding_table_and_concat_combiner

namespace criteo {
const std::vector<int> gpus8 = {0, 1, 2, 3, 4, 5, 6, 7};
const int batch_size = 8;
const int num_table = 26;
const std::vector<int> table_ev_size_list = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                                             8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
const int num_embedding = 26;
const std::vector<int> id_space_list = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
const std::vector<int> hotness_list = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
// const std::vector<Combiner> combiner_list = {
//     Combiner::Average, Combiner::Average, Combiner::Average, Combiner::Average,
//     Combiner::Average};
const std::vector<Combiner> combiner_list = {
    Combiner::Sum, Combiner::Sum, Combiner::Sum, Combiner::Sum, Combiner::Sum, Combiner::Sum,
    Combiner::Sum, Combiner::Sum, Combiner::Sum, Combiner::Sum, Combiner::Sum, Combiner::Sum,
    Combiner::Sum, Combiner::Sum, Combiner::Sum, Combiner::Sum, Combiner::Sum, Combiner::Sum,
    Combiner::Sum, Combiner::Sum, Combiner::Sum, Combiner::Sum, Combiner::Sum, Combiner::Sum,
    Combiner::Sum, Combiner::Sum};
const std::vector<int> table_min_key_list = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
const std::vector<int> table_max_key_list = {
    203931, 18598, 14092, 7012, 18977, 4,  6385,   1245,   49,     186213, 71328, 67288, 11,
    2168,   7338,  61,    4,    932,   15, 204515, 141526, 199433, 60919,  9137,  71,    34};

TEST(test_embedding_collection, plan) {
  embedding_collection_e2e<int64_t, int64_t, uint64_t, float>(
      gpus8, batch_size, num_table, table_ev_size_list, num_embedding, id_space_list, hotness_list,
      combiner_list, table_min_key_list, table_max_key_list,
      "/workdir/test/utest/embedding_collection/plan_criteo_8gpu.json");
}
}  // namespace criteo
