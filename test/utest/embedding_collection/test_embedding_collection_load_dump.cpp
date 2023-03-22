/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <core/hctr_impl/hctr_backend.hpp>
#include <embedding/embedding.hpp>
#include <embedding_storage/weight_io/parameter_IO.hpp>
#include <embeddings/embedding_collection.hpp>
#include <include/embeddings/embedding_collection.hpp>
#include <numeric>
#include <resource_managers/resource_manager_ext.hpp>
#include <utest/embedding_collection/embedding_collection_cpu.hpp>
#include <utest/embedding_collection/embedding_collection_utils.hpp>

using namespace embedding;

/*
this unit test is same with test_embedding_collection.cpp
this unit test is add embedding load dump during iteration
will remove to a new unit test soon
*/
class EmbeddingIO {
 public:
  EmbeddingIO(std::shared_ptr<HugeCTR::ResourceManager> resource_manager) {
    resource_manager_ = resource_manager;
    embedding_para_io_ = std::shared_ptr<embedding::EmbeddingParameterIO>(
        new embedding::EmbeddingParameterIO(resource_manager_));
  }

  void add_embedding_collection(EmbeddingCollection* embedding_collection) {
    ebc_list_.push_back(embedding_collection);
    embedding_para_io_->add_embedding_collection(embedding_collection);
  }
  // same with model.embedding_load function
  void embedding_load(const std::string& path, const std::map<int, int>& table_id_map_raw,
                      int embedding_collection_id) {
    int embedding_collection_nums = ebc_list_.size();
    if (embedding_collection_id < 0 || embedding_collection_id >= embedding_collection_nums) {
      HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput, "embedding_collection_id is out of range");
    }
    std::map<int, int> table_id_map;
    auto& tmp_embedding_collection = ebc_list_[embedding_collection_id];
    auto& tmp_ebc_param = tmp_embedding_collection->ebc_param_;
    auto& tmp_shard_matrix = tmp_ebc_param.shard_matrix;
    int num_total_gpus = resource_manager_->get_global_gpu_count();
    int num_local_gpus = resource_manager_->get_local_gpu_count();
    std::vector<std::shared_ptr<core::CoreResourceManager>> core_list;

    for (int local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
      auto core_resource_manager =
          std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager_, local_gpu_id);
      core_list.push_back(core_resource_manager);
    }

    struct embedding::EmbeddingParameterInfo tmp_epi = embedding::EmbeddingParameterInfo();
    embedding_para_io_->load_metadata(path, embedding_collection_id, tmp_epi);
    if (table_id_map_raw.empty()) {
      int tmp_table_num = tmp_ebc_param.num_table;
      for (int i = 0; i < tmp_table_num; ++i) {
        table_id_map[i] = i;
      }
    } else {
      for (auto tmp_iter : table_id_map_raw) {
        table_id_map[tmp_iter.first] = tmp_iter.second;
      }
    }

    for (auto table_id_iter = table_id_map.begin(); table_id_iter != table_id_map.end();
         ++table_id_iter) {
      int file_table_id = table_id_iter->first;
      int model_table_id = table_id_iter->second;
      int target_grouped_id = -1;
      embedding::TablePlacementStrategy target_placement;
      for (int grouped_id = 0; grouped_id < tmp_ebc_param.grouped_emb_params.size(); ++grouped_id) {
        auto& tmp_table_ids = tmp_ebc_param.grouped_emb_params[grouped_id].table_ids;

        auto tmp_it = std::find(tmp_table_ids.begin(), tmp_table_ids.end(), model_table_id);
        if (tmp_it != tmp_table_ids.end()) {
          target_grouped_id = grouped_id;
          target_placement = tmp_ebc_param.grouped_emb_params[grouped_id].table_placement_strategy;
          break;
        }
      }
      if (target_grouped_id == -1) {
        HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput,
                       "can not find table_id in model table_ids,please check your input");
      }

      if (target_placement == embedding::TablePlacementStrategy::DataParallel) {
        auto tmp_filter = [=](size_t key) { return true; };
        core23::Tensor keys;
        core23::Tensor embedding_weights;
        auto& target_key_type = tmp_ebc_param.key_type;
        auto& target_value_type = tmp_ebc_param.emb_type;
        embedding_para_io_->load_embedding_weight(tmp_epi, file_table_id, keys, embedding_weights,
                                                  tmp_filter, core_list[0], target_key_type,
                                                  target_value_type);
        for (size_t local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
          HugeCTR::CudaDeviceContext context(core_list[local_gpu_id]->get_device_id());
          auto& grouped_table =
              tmp_embedding_collection->embedding_tables_[local_gpu_id][target_grouped_id];
          grouped_table->load_by_id(&keys, &embedding_weights, model_table_id);
        }
      } else if (target_placement == embedding::TablePlacementStrategy::ModelParallel) {
        for (size_t local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
          HugeCTR::CudaDeviceContext context(core_list[local_gpu_id]->get_device_id());
          size_t global_id = resource_manager_->get_gpu_global_id_from_local_id(local_gpu_id);
          auto& target_key_type = tmp_ebc_param.key_type;
          auto& target_value_type = tmp_ebc_param.emb_type;
          std::vector<int> shard_gpu_list;
          for (int gpu_id = 0; gpu_id < num_total_gpus; ++gpu_id) {
            HCTR_CHECK_HINT(model_table_id < static_cast<int>(tmp_shard_matrix[gpu_id].size()),
                            "table_id is out of range");
            if (tmp_ebc_param.shard_matrix[gpu_id][model_table_id] == 1) {
              shard_gpu_list.push_back(gpu_id);
            }
          }
          int num_shards = static_cast<int>(shard_gpu_list.size());
          auto find_shard_id_iter =
              std::find(shard_gpu_list.begin(), shard_gpu_list.end(), global_id);
          if (find_shard_id_iter == shard_gpu_list.end()) {
            continue;
          }
          int shard_id =
              static_cast<int>(std::distance(shard_gpu_list.begin(), find_shard_id_iter));

          auto tmp_filter = [=](size_t key) { return key % num_shards == shard_id; };
          core23::Tensor keys;
          core23::Tensor embedding_weights;
          embedding_para_io_->load_embedding_weight(tmp_epi, file_table_id, keys, embedding_weights,
                                                    tmp_filter, core_list[0], target_key_type,
                                                    target_value_type);

          auto& grouped_table =
              tmp_embedding_collection->embedding_tables_[local_gpu_id][target_grouped_id];
          grouped_table->load_by_id(&keys, &embedding_weights, model_table_id);
        }
      } else {
        HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError, "unsupport parallel mode");
      }
    }
  }

  // same with model.embedding_dump function
  void embedding_dump(const std::string& path, const std::map<int, std::vector<int>>& table_ids) {
    std::vector<struct embedding::EmbeddingParameterInfo> epis;

    embedding_para_io_->get_parameter_info_from_model(path, epis);
    for (int i = 0; i < epis.size(); ++i) {
      epis[i].gemb_distribution->print_info();
    }

    if (table_ids.empty()) {
      int collection_num = ebc_list_.size();
      for (int cid = 0; cid < collection_num; ++cid) {
        auto& tmp_embedding_collection = ebc_list_[cid];
        auto& tmp_ebc_param = tmp_embedding_collection->ebc_param_;
        int tmp_table_num = tmp_ebc_param.num_table;
        std::vector<int> tmp_table_ids;
        for (int i = 0; i < tmp_table_num; ++i) {
          tmp_table_ids.push_back(i);
        }
        embedding_para_io_->dump_metadata(path, epis[cid], tmp_table_ids);
        embedding_para_io_->dump_embedding_weight(path, epis[cid], tmp_table_ids);
      }
    } else {
      for (auto collection_id_iter = table_ids.begin(); collection_id_iter != table_ids.end();
           ++collection_id_iter) {
        auto& cid = collection_id_iter->first;
        auto& raw_table_ids = collection_id_iter->second;
        std::vector<int> tmp_table_ids = std::vector<int>();
        if (raw_table_ids.size() == 0) {
          auto& tmp_embedding_collection = ebc_list_[cid];
          auto& tmp_ebc_param = tmp_embedding_collection->ebc_param_;
          int tmp_table_num = tmp_ebc_param.num_table;
          for (int i = 0; i < tmp_table_num; ++i) {
            tmp_table_ids.push_back(i);
          }
        } else {
          for (int i = 0; i < raw_table_ids.size(); ++i) {
            tmp_table_ids.push_back(raw_table_ids[i]);
          }
        }
        embedding_para_io_->dump_metadata(path, epis[cid], tmp_table_ids);
        embedding_para_io_->dump_embedding_weight(path, epis[cid], tmp_table_ids);
      }
    }
  }

 private:
  std::vector<EmbeddingCollection*> ebc_list_;
  std::shared_ptr<HugeCTR::ResourceManager> resource_manager_;
  std::shared_ptr<EmbeddingParameterIO> embedding_para_io_;
};

const int batch_size = 8192;
// table params
const int num_table = 4;
const std::vector<int> table_ev_size_list = {128, 64, 32, 16};
const std::vector<int> table_max_vocabulary_list = {39884, 3904, 1728, 12434};

// lookup params
const std::vector<LookupParam> lookup_params = {
    {0, 0, Combiner::Sum, 8, table_ev_size_list[0]},
    {1, 1, Combiner::Average, 20, table_ev_size_list[1]},
    {2, 2, Combiner::Sum, 10, table_ev_size_list[2]},
    {3, 3, Combiner::Average, 5, table_ev_size_list[3]},
};

const std::vector<LookupParam> lookup_params_with_shared_table = {
    {0, 0, Combiner::Sum, 8, table_ev_size_list[0]},
    {1, 1, Combiner::Average, 20, table_ev_size_list[1]},
    {2, 2, Combiner::Sum, 10, table_ev_size_list[2]},
    {3, 3, Combiner::Average, 5, table_ev_size_list[3]},
};

const std::vector<int> device_list = {0, 1};
bool debug_verbose_io = false;

std::vector<EmbeddingTableParam> get_table_param_list_io(core23::DataType emb_type) {
  std::vector<EmbeddingTableParam> table_param_list;

  HugeCTR::OptParams opt_param;
  // FIXME: We need to initialize all variable or we will trigger uninitialized error in
  // EmbeddingTableParam ctor because the copy constructor of HugeCTR::OptParams trys to copy all
  // members
  opt_param.optimizer = HugeCTR::Optimizer_t::SGD;
  opt_param.lr = 1e-1;
  opt_param.scaler = (emb_type == core23::ScalarType::Half) ? 1024 : 1;
  opt_param.hyperparams = HugeCTR::OptHyperParams{};
  opt_param.update_type = HugeCTR::Update_t::Local;

  InitParams init_param;
  for (int table_id = 0; table_id < num_table; ++table_id) {
    EmbeddingTableParam table_param{table_id, table_max_vocabulary_list[table_id],
                                    table_ev_size_list[table_id], opt_param, init_param};
    table_param_list.push_back(std::move(table_param));
  }
  return table_param_list;
}

template <typename key_t, typename offset_t, typename index_t, typename emb_t>
void embedding_collection_e2e_io(const std::vector<LookupParam>& lookup_params,
                                 const std::vector<std::vector<int>>& shard_matrix,
                                 const std::vector<GroupedEmbeddingParam>& grouped_emb_params) {
  ASSERT_EQ(table_max_vocabulary_list.size(), num_table);
  ASSERT_EQ(table_ev_size_list.size(), num_table);
  EmbeddingCollectionParam ebc_param{num_table,
                                     table_max_vocabulary_list,
                                     static_cast<int>(lookup_params.size()),
                                     lookup_params,
                                     shard_matrix,
                                     grouped_emb_params,
                                     batch_size,
                                     HugeCTR::core23::ToScalarType<key_t>::value,
                                     HugeCTR::core23::ToScalarType<index_t>::value,
                                     HugeCTR::core23::ToScalarType<offset_t>::value,
                                     HugeCTR::core23::ToScalarType<emb_t>::value,
                                     HugeCTR::core23::ToScalarType<emb_t>::value,
                                     EmbeddingLayout::FeatureMajor,
                                     EmbeddingLayout::FeatureMajor,
                                     embedding::SortStrategy::Segmented,
                                     embedding::KeysPreprocessStrategy::AddOffset,
                                     embedding::AllreduceStrategy::Sparse,
                                     CommunicationStrategy::Uniform};
  auto table_param_list = get_table_param_list_io(ebc_param.emb_type);

  auto resource_manager = HugeCTR::ResourceManagerExt::create({device_list}, 0);
  EmbeddingIO emb_io = EmbeddingIO(resource_manager);
  int num_gpus = static_cast<int>(device_list.size());
  int batch_size_per_gpu = batch_size / num_gpus;

  std::vector<key_t> key_list;
  std::vector<offset_t> bucket_range;
  std::vector<std::vector<std::vector<key_t>>> dp_keys;
  std::vector<std::vector<std::vector<offset_t>>> dp_bucket_range;
  auto prepare_input = [&] {
    timeval t1;
    gettimeofday(&t1, NULL);
    srand(t1.tv_usec * t1.tv_sec);
    key_list.clear();
    bucket_range.clear();
    dp_keys.clear();
    dp_bucket_range.clear();

    bucket_range.push_back(0);
    dp_keys.resize(num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      dp_keys[gpu_id].resize(ebc_param.num_lookup);
    }
    dp_bucket_range.resize(num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      dp_bucket_range[gpu_id].resize(ebc_param.num_lookup);
      for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
        dp_bucket_range[gpu_id][lookup_id].push_back(0);
      }
    }

    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      auto& lookup_param = ebc_param.lookup_params[lookup_id];
      int table_id = lookup_param.table_id;
      int max_hotness = lookup_param.max_hotness;
      auto& table_param = table_param_list[table_id];

      std::vector<std::vector<key_t>> dp_keys_on_one_gpu;
      for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        for (int b = 0; b < batch_size_per_gpu; ++b) {
          int nnz = max_hotness;  // FIXME: static nnz
          /*
              int nnz = (lookup_param.combiner == Combiner::Concat)
                            ? max_hotness
                            : 1 + rand() % max_hotness;  // TODO: support nnz=0
          */

          dp_bucket_range[gpu_id][lookup_id].push_back(nnz);

          bucket_range.push_back(nnz);
          for (int i = 0; i < nnz; ++i) {
            key_t key = rand() % table_param.max_vocabulary_size;
            key_list.push_back(key);
            dp_keys[gpu_id][lookup_id].push_back(key);
          }
        }
      }
    }
    std::inclusive_scan(bucket_range.begin(), bucket_range.end(), bucket_range.begin());
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
        std::inclusive_scan(dp_bucket_range[gpu_id][lookup_id].begin(),
                            dp_bucket_range[gpu_id][lookup_id].end(),
                            dp_bucket_range[gpu_id][lookup_id].begin());
      }
    }
  };

  std::vector<std::vector<emb_t>> top_grads;
  auto prepare_top_grads = [&] {
    top_grads.resize(num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      top_grads[gpu_id].clear();
      for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
        auto& lookup_param = ebc_param.lookup_params[lookup_id];
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

  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    auto core = std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, gpu_id);

    core_resource_manager_list.push_back(core);
  }

  std::shared_ptr<HugeCTR::DataDistributor> data_distributor =
      std::make_shared<HugeCTR::DataDistributor>(ebc_param.universal_batch_size, ebc_param.key_type,
                                                 resource_manager, core_resource_manager_list,
                                                 ebc_param, table_param_list);

  std::vector<HugeCTR::DataDistributor::Result> data_distributor_outputs;
  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    data_distributor_outputs.push_back(HugeCTR::allocate_output_for_data_distributor(
        core_resource_manager_list[gpu_id], ebc_param));
  }
  std::unique_ptr<embedding::EmbeddingCollection> ebc =
      std::make_unique<embedding::EmbeddingCollection>(resource_manager, core_resource_manager_list,
                                                       ebc_param, ebc_param, table_param_list);
  emb_io.add_embedding_collection(ebc.get());

  std::vector<std::vector<core23::Tensor>> sparse_dp_tensors;
  std::vector<std::vector<core23::Tensor>> sparse_dp_bucket_ranges;
  std::vector<size_t*> ebc_num_keys_list;
  std::vector<core23::Tensor> ebc_top_grads;
  std::vector<core23::Tensor> ebc_outptut;
  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());
    core23::Device device(core23::DeviceType::GPU,
                          core_resource_manager_list[gpu_id]->get_device_id());
    core23::TensorParams params = core23::TensorParams().device(device);

    std::vector<core23::Tensor> sparse_dp_tensors_on_current_gpu;
    std::vector<core23::Tensor> sparse_dp_bucket_range_on_current_gpu;
    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      auto& lookup_param = ebc_param.lookup_params[lookup_id];
      int max_hotness = lookup_param.max_hotness;

      sparse_dp_tensors_on_current_gpu.emplace_back(
          params.shape({ebc_param.universal_batch_size / num_gpus, max_hotness})
              .data_type(ebc_param.key_type));
      sparse_dp_bucket_range_on_current_gpu.emplace_back(
          params.shape({ebc_param.universal_batch_size / num_gpus})
              .data_type(ebc_param.offset_type));
    }
    sparse_dp_tensors.push_back(sparse_dp_tensors_on_current_gpu);
    sparse_dp_bucket_ranges.push_back(sparse_dp_bucket_range_on_current_gpu);

    int64_t num_ev = 0;
    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      auto& lookup_param = ebc_param.lookup_params[lookup_id];
      num_ev += (lookup_param.combiner == Combiner::Concat)
                    ? lookup_param.ev_size * lookup_param.max_hotness
                    : lookup_param.ev_size;
    }
    num_ev *= (ebc_param.universal_batch_size / num_gpus);
    ebc_top_grads.emplace_back(params.shape({num_ev}).data_type(ebc_param.emb_type));
    ebc_outptut.emplace_back(params.shape({num_ev}).data_type(ebc_param.emb_type));
  }

  auto prepare_gpu_input = [&] {
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());

      core23::copy_sync(ebc_top_grads[gpu_id], top_grads[gpu_id]);

      for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
        core23::copy_sync(sparse_dp_tensors[gpu_id][lookup_id], dp_keys[gpu_id][lookup_id]);
      }
    }
  };

  auto prepare_data = [&] {
    prepare_input();
    prepare_top_grads();
    prepare_gpu_input();
  };

  auto sync_gpus = [&]() {
    for (auto core : core_resource_manager_list) {
      HCTR_LIB_THROW(cudaStreamSynchronize(core->get_local_gpu()->get_stream()));
    }
  };
  // sync for emb table init
  sync_gpus();

  std::vector<std::vector<IGroupedEmbeddingTable*>> grouped_emb_table_ptr_list =
      ebc->get_grouped_embedding_tables();

  EmbeddingCollectionCPU<key_t, offset_t, index_t, emb_t> ebc_cpu{
      num_gpus, ebc_param, num_table, table_param_list, grouped_emb_table_ptr_list};

  EmbeddingReferenceCPU<key_t, offset_t, index_t, emb_t> emb_ref{num_gpus,
                                                                 ebc_param,
                                                                 num_table,
                                                                 table_param_list,
                                                                 grouped_emb_table_ptr_list,
                                                                 EmbeddingLayout::FeatureMajor};

  auto check_forward_result = [&] {
    std::cout << "compare ebc cpu emb output vs. emb reference emb output.\n";
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      ASSERT_EQ(ebc_cpu.embedding_vec_[gpu_id].size(), emb_ref.embedding_vec_[gpu_id].size());
      // std::cout << "forward cpu output:\n";
      // print_array(ebc_cpu.embedding_vec_[gpu_id].size(),
      // ebc_cpu.embedding_vec_[gpu_id]); std::cout << "forward ref output:\n";
      // print_array(emb_ref.embedding_vec_[gpu_id].size(),
      // emb_ref.embedding_vec_[gpu_id]);
      assert_array_eq(ebc_cpu.embedding_vec_[gpu_id].size(), ebc_cpu.embedding_vec_[gpu_id],
                      emb_ref.embedding_vec_[gpu_id]);
    }
    std::cout << "\t>pass compare ebc cpu emb output vs. emb reference emb output.\n";

    std::cout << "compare ebc gpu emb output vs. emb reference emb output.\n";
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      std::vector<emb_t> gpu_emb_output(ebc_outptut[gpu_id].num_elements());
      core23::copy_sync(gpu_emb_output, ebc_outptut[gpu_id]);
      ASSERT_EQ(gpu_emb_output.size(), emb_ref.embedding_vec_[gpu_id].size());
      if (debug_verbose_io) {
        std::cout << "forward ref output:\n";
        print_array(emb_ref.embedding_vec_[gpu_id].size(), emb_ref.embedding_vec_[gpu_id]);
        std::cout << "forward gpu output:\n";
        print_array(gpu_emb_output.size(), gpu_emb_output);
      }
      assert_array_eq(gpu_emb_output.size(), gpu_emb_output, ebc_cpu.embedding_vec_[gpu_id]);
    }
    std::cout << "\t>pass compare ebc gpu emb output vs. emb reference emb output.\n";
  };
  auto check_backward_result = [&] {
    auto compare_grad_in_table = [](const std::unordered_map<key_t, std::vector<float>>& lhs,
                                    const std::unordered_map<key_t, std::vector<float>>& rhs) {
      ASSERT_EQ(lhs.size(), rhs.size());

      for (auto p : lhs) {
        auto& k = p.first;
        auto& lhs_ev = p.second;
        ASSERT_TRUE(rhs.find(k) != rhs.end());
        auto& rhs_ev = rhs.at(k);
        ASSERT_EQ(lhs_ev.size(), rhs_ev.size());
        // if (debug_verbose_io) {
        //   std::cout << "lhs output:\n";
        //   print_array(lhs_ev.size(), lhs_ev);
        //   std::cout << "rhs output:\n";
        //   print_array(rhs_ev.size(), rhs_ev);
        // }
        assert_array_eq(lhs_ev.size(), lhs_ev, rhs_ev);
      }
    };

    std::cout << "compare ref grad info vs. ebc cpu grad info.\n";
    ASSERT_EQ(ebc_cpu.grad_info_.size(), emb_ref.accumulate_grad_map_.size());
    for (int table_id = 0; table_id < num_table; ++table_id) {
      ASSERT_TRUE(table_id < static_cast<int>(ebc_cpu.grad_info_.size()));
      auto& cpu_grad_in_table = ebc_cpu.grad_info_.at(table_id);
      auto& ref_grad_in_table = emb_ref.accumulate_grad_map_.at(table_id);
      compare_grad_in_table(cpu_grad_in_table, ref_grad_in_table);
    }
    std::cout << "\t>pass compare ref grad info vs. ebc cpu grad info.\n";
  };

  auto check_embedding_table = [&] {
    std::cout << "compare ref emb table vs. ebc cpu emb table.\n";
    const auto& cpu_emb_table = ebc_cpu.emb_table_cpu_.emb_table_list_;
    const auto& ref_emb_table = emb_ref.emb_table_cpu_.emb_table_list_;
    ASSERT_TRUE(cpu_emb_table.size() == ref_emb_table.size());

    for (size_t table_id = 0; table_id < cpu_emb_table.size(); ++table_id) {
      ASSERT_EQ(cpu_emb_table[table_id].size(), ref_emb_table[table_id].size());

      for (auto& [k, cpu_ev] : cpu_emb_table[table_id]) {
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
    //   ASSERT_EQ(gpu_emb_table[id_space].size(),
    //   ref_emb_table[id_space].size());

    //   for (auto &[k, gpu_ev] : gpu_emb_table[id_space]) {
    //     ASSERT_TRUE(gpu_emb_table[id_space].find(k) !=
    //     ref_emb_table[id_space].end()); auto ref_ev =
    //     ref_emb_table[id_space].at(k);

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
#pragma omp parallel for num_threads(num_gpus)
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      data_distributor->distribute(gpu_id, sparse_dp_tensors[gpu_id],
                                   sparse_dp_bucket_ranges[gpu_id],
                                   data_distributor_outputs[gpu_id], batch_size);
      ebc->forward_per_gpu(true, gpu_id, data_distributor_outputs[gpu_id], ebc_outptut[gpu_id],
                           batch_size);
    }
    sync_gpus();
    // try to dump data to file system , and load it from file systems
    // if value don't change , it load dump can be work , and result is correct
    std::map<int, std::vector<int>> dump_table_ids_map;
    emb_io.embedding_dump("./embedding_io_test", dump_table_ids_map);

    std::map<int, int> load_table_id_map;
    emb_io.embedding_load("./embedding_io_test", load_table_id_map, 0);
    sync_gpus();

    check_forward_result();

    // backward
    ebc_cpu.embedding_backward_cpu(top_grads, batch_size);
    emb_ref.embedding_backward_cpu(top_grads, key_list, bucket_range);
#pragma omp parallel for num_threads(num_gpus)
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      ebc->backward_per_gpu(gpu_id, data_distributor_outputs[gpu_id], ebc_top_grads[gpu_id],
                            batch_size);
    }
    sync_gpus();

    check_backward_result();

    // update
    ebc_cpu.embedding_update_cpu();
    emb_ref.embedding_update_cpu();
#pragma omp parallel for num_threads(num_gpus)
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      ebc->update_per_gpu(gpu_id);
    }
    sync_gpus();

    check_embedding_table();
  }
}

// dp
namespace dp {
const std::vector<std::vector<int>> shard_matrix = {
    {1, 1, 1, 1},
    {1, 1, 1, 1},
};

const std::vector<GroupedEmbeddingParam> grouped_emb_params = {
    {TablePlacementStrategy::DataParallel, {0, 1, 2, 3}}};

TEST(test_embedding_collection_load_dump, dp_plan0) {
  embedding_collection_e2e_io<uint32_t, uint32_t, uint32_t, float>(lookup_params, shard_matrix,
                                                                   grouped_emb_params);
}
}  // namespace dp

namespace mp {
const std::vector<std::vector<int>> shard_matrix = {
    {1, 0, 1, 1},
    {0, 1, 1, 1},
};

const std::vector<GroupedEmbeddingParam> grouped_emb_params = {
    {TablePlacementStrategy::ModelParallel, {0, 1, 2, 3}}};

TEST(test_embedding_collection_load_dump, mp_plan0) {
  embedding_collection_e2e_io<uint32_t, uint32_t, uint32_t, float>(lookup_params, shard_matrix,
                                                                   grouped_emb_params);
}

TEST(test_embedding_collection_load_dump, mp_plan1) {
  embedding_collection_e2e_io<uint32_t, uint32_t, uint32_t, float>(
      lookup_params_with_shared_table, shard_matrix, grouped_emb_params);
}
}  // namespace mp

namespace dp_and_mp {
const std::vector<std::vector<int>> shard_matrix = {
    {1, 0, 1, 1},
    {0, 1, 1, 1},
};

const std::vector<GroupedEmbeddingParam> grouped_emb_params = {
    {TablePlacementStrategy::DataParallel, {2}},
    {TablePlacementStrategy::ModelParallel, {0, 1, 3}}};

TEST(test_embedding_collection_load_dump, dp_and_mp_plan0) {
  embedding_collection_e2e_io<uint32_t, uint32_t, uint32_t, float>(lookup_params, shard_matrix,
                                                                   grouped_emb_params);
}

TEST(test_embedding_collection_load_dump, dp_and_mp_plan1) {
  embedding_collection_e2e_io<uint32_t, uint32_t, uint32_t, float>(
      lookup_params_with_shared_table, shard_matrix, grouped_emb_params);
}
}  // namespace dp_and_mp
