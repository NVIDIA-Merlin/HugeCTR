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
#include <gtest/gtest.h>
#include <omp.h>

#include <core/hctr_impl/hctr_backend.hpp>
#include <embedding/data_distributor/data_distributor.hpp>
#include <iostream>
#include <resource_managers/resource_manager_ext.hpp>

using namespace HugeCTR;
using namespace embedding;

const int batch_size = 10;
// table params
const int num_table = 4;
const std::vector<int> table_ev_size_list = {128, 64, 32, 16};
const std::vector<int> table_max_vocabulary_list = {398844, 39043, 17289, 124345};

// lookup params
const std::vector<LookupParam> lookup_params0 = {
    {0, 0, Combiner::Sum, 2, table_ev_size_list[0]},
    {1, 1, Combiner::Average, 4, table_ev_size_list[1]},
    {2, 2, Combiner::Sum, 3, table_ev_size_list[2]},
    {3, 3, Combiner::Average, 1, table_ev_size_list[3]},
};

// const std::vector<LookupParam> lookup_params0 = {
//        {0, 0, Combiner::Sum, 1, table_ev_size_list[0]},
//        {1, 1, Combiner::Average, 1, table_ev_size_list[1]},
//        {2, 2, Combiner::Sum, 1, table_ev_size_list[2]},
//        {3, 3, Combiner::Average, 1, table_ev_size_list[3]},
//};

const std::vector<LookupParam> lookup_params_with_shared_table = {
    {0, 0, Combiner::Sum, 8, table_ev_size_list[0]},
    {1, 1, Combiner::Average, 20, table_ev_size_list[1]},
    {2, 2, Combiner::Sum, 10, table_ev_size_list[2]},
    {3, 3, Combiner::Average, 5, table_ev_size_list[3]},
    {4, 1, Combiner::Average, 8, table_ev_size_list[1]},
};

// lookup params
const std::vector<LookupParam> dense_lookup_params = {
    {0, 0, Combiner::Concat, 2, table_ev_size_list[0]},
    {1, 1, Combiner::Average, 4, table_ev_size_list[1]},
    {2, 2, Combiner::Concat, 3, table_ev_size_list[2]},
    {3, 3, Combiner::Average, 1, table_ev_size_list[3]},
};

static auto get_core_resource_managers(std::shared_ptr<ResourceManager> resource_manager) {
  std::vector<std::shared_ptr<core::CoreResourceManager>> core_list;
  for (int local_gpu_id = 0; local_gpu_id < resource_manager->get_local_gpu_count();
       ++local_gpu_id) {
    auto core_resource_manager =
        std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, local_gpu_id);
    core_list.push_back(core_resource_manager);
  }
  return core_list;
}

template <typename key_t, typename offset_t>
void test_data_distributor(const std::vector<int>& device_list,
                           const std::vector<LookupParam>& lookup_params,
                           const std::vector<std::vector<int>>& shard_matrix,
                           const std::vector<GroupedTableParam>& grouped_emb_params,
                           bool incomplete_batch = false) {
  static_assert(
      std::disjunction<std::is_same<key_t, uint32_t>, std::is_same<key_t, long long>>::value);

  auto key_type = core23::ToScalarType<key_t>::value;
  auto index_type = core23::ToScalarType<uint32_t>::value;
  auto offset_type = core23::ToScalarType<offset_t>::value;
  auto emb_type = core23::ToScalarType<float>::value;             // doesn't matter
  auto wgrad_type = HugeCTR::core23::ToScalarType<float>::value;  // doesn't matter

  auto resource_manager = ResourceManagerExt::create({device_list}, 424242);
  auto core_list = get_core_resource_managers(resource_manager);
  int num_gpus = device_list.size();
  int num_lookup = lookup_params.size();
  int batch_size_per_dev = batch_size / num_gpus;
  int current_batch_size = incomplete_batch ? batch_size - 3 : batch_size;

  EmbeddingCollectionParam ebc_param{num_table,
                                     static_cast<int>(num_lookup),
                                     lookup_params,
                                     shard_matrix,
                                     grouped_emb_params,
                                     batch_size,
                                     key_type,
                                     index_type,
                                     offset_type,
                                     emb_type,
                                     wgrad_type,
                                     EmbeddingLayout::FeatureMajor,
                                     EmbeddingLayout::FeatureMajor,  // output layout
                                     embedding::SortStrategy::Radix,
                                     embedding::KeysPreprocessStrategy::None,
                                     AllreduceStrategy::Dense,
                                     CommunicationStrategy::Uniform,
                                     {}};

  std::vector<EmbeddingTableParam> table_param_list;
  for (int id = 0; id < num_table; ++id) {
    EmbeddingTableParam table_param{
        id, table_max_vocabulary_list[id], table_ev_size_list[id], {}, {}};
    table_param_list.push_back(std::move(table_param));
  }

  std::vector<int> dr_lookup_ids(ebc_param.num_lookup);
  std::iota(dr_lookup_ids.begin(), dr_lookup_ids.end(), 0);
  HugeCTR::DataDistributor distributor(core_list, ebc_param, table_param_list, dr_lookup_ids);

  // --- allocate resulting output ---
  std::vector<DataDistributor::Result> results;
  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    results.emplace_back(allocate_output_for_data_distributor(core_list[gpu_id], ebc_param));
  }

  // --- init keys and bucket_range tensors ---
  std::vector<std::vector<core23::Tensor>> dp_keys(num_gpus);
  std::vector<std::vector<core23::Tensor>> dp_bucket_range(num_gpus);

  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    auto core = core_list[gpu_id];
    CudaDeviceContext ctx(core->get_device_id());

    core23::Device device(core23::DeviceType::GPU, core->get_device_id());
    core23::BufferParams buffer_params;
    buffer_params.unitary = false;
    core23::TensorParams params =
        core23::TensorParams().device(device).buffer_params(buffer_params);

    for (int lookup_id = 0; lookup_id < num_lookup; ++lookup_id) {
      int max_hotness = lookup_params[lookup_id].max_hotness;
      int num_keys = max_hotness * batch_size_per_dev;
      int num_buckets = batch_size_per_dev + 1;

      auto d_keys =
          core23::Tensor(params.shape({static_cast<int64_t>(num_keys)}).data_type(key_type));
      auto d_bucket_range =
          core23::Tensor(params.shape({static_cast<int64_t>(num_buckets)}).data_type(offset_type));

      std::vector<key_t> h_keys(num_keys, lookup_id);
      std::iota(h_keys.begin(), h_keys.end(), 0);

      std::vector<offset_t> h_bucket_range(num_buckets, 0);
      for (int bucket_id = 1; bucket_id < num_buckets; ++bucket_id) {
        int num_keys_per_bucket = max_hotness;  // TODO: (rand() % max_hotness) + 1;
        h_bucket_range[bucket_id] = num_keys_per_bucket;
      }
      std::inclusive_scan(h_bucket_range.begin() + 1, h_bucket_range.end(),
                          h_bucket_range.begin() + 1);

      //      printf("gpu: %d, lookup: %d\n", gpu_id, lookup_id);
      //      printf("keys: ");
      //      for (auto x : h_keys) printf("%d ", x);
      //      printf("\n");
      //      printf("bucket_range: ");
      //      for (auto x : h_bucket_range) printf("%d ", x);
      //      printf("\n");

      core23::copy_sync(d_keys, h_keys);
      //      core23::copy_sync(d_bucket_range, h_bucket_range);

      dp_keys[gpu_id].push_back(d_keys);
      dp_bucket_range[gpu_id].push_back(d_bucket_range);
    }
  }

  // --- all-to-all ---
  // iterate 3 times to ensure internal state does not propagate to next iteration
  for (int i = 0; i < 3; ++i) {
#pragma omp parallel num_threads(num_gpus)
    {
      size_t id = omp_get_thread_num();
      CudaDeviceContext context(core_list[id]->get_device_id());
      distributor.distribute(id, dp_keys[id], dp_bucket_range[id], results[id], current_batch_size);
      HCTR_LIB_THROW(cudaStreamSynchronize(core_list[id]->get_local_gpu()->get_stream()));
    }
  }

  // --- check results ---
  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    int remaining = current_batch_size - gpu_id * batch_size_per_dev;
    int num_valid_samples = std::max(std::min(remaining, batch_size_per_dev), 0);

    std::vector<offset_t> expected_keys_per_bucket;
    for (int lookup_id = 0; lookup_id < num_lookup; ++lookup_id) {
      expected_keys_per_bucket.insert(expected_keys_per_bucket.end(), num_valid_samples,
                                      ebc_param.lookup_params[lookup_id].max_hotness);
      expected_keys_per_bucket.insert(expected_keys_per_bucket.end(),
                                      batch_size_per_dev - num_valid_samples, 0);
    }

    for (size_t group_id = 0; group_id < ebc_param.grouped_lookup_params.size(); ++group_id) {
      auto group_result = results[gpu_id][group_id];

      std::vector<key_t> result_keys(group_result.keys.num_elements());
      core23::copy_sync(result_keys, group_result.keys);

      // --- check keys per bucket

      auto embedding_group_type = ebc_param.grouped_lookup_params[group_id].embedding_group_type;
      switch (embedding_group_type) {
        case (embedding::EmbeddingGroupType::SparseModelParallel): {
          std::vector<offset_t> result_bucket_range(group_result.bucket_range.num_elements());
          core23::copy_sync(result_bucket_range, group_result.bucket_range);

          std::vector<offset_t> result_keys_per_bucket(
              group_result.num_keys_per_bucket.num_elements());
          core23::copy_sync(result_keys_per_bucket, group_result.num_keys_per_bucket);
          ASSERT_EQ(result_keys_per_bucket, expected_keys_per_bucket);
          // expected feature-major output
          int shard_id = 0;
          for (int lookup_id = 0; lookup_id < num_lookup; ++lookup_id) {
            int num_shards = 0;
            for (int i = 0; i < num_gpus; ++i) {
              if (ebc_param.has_table_shard(i, group_id, lookup_id)) {
                num_shards++;
              }
            }

            if (ebc_param.has_table_shard(gpu_id, group_id, lookup_id)) {
              size_t num_keys = result_bucket_range[(shard_id + 1) * batch_size] -
                                result_bucket_range[shard_id * batch_size];

              // --- calculate expected keys for this gpu ---
              int table_id = ebc_param.lookup_params[lookup_id].table_id;
              std::vector<int> shard_gpus;
              for (size_t i = 0; i < num_gpus; ++i) {
                if (ebc_param.shard_matrix[i][table_id] == 1) {
                  shard_gpus.push_back(i);
                }
              }
              auto find_shard_id_iter = std::find(shard_gpus.begin(), shard_gpus.end(), gpu_id);
              HCTR_CHECK_HINT(find_shard_id_iter != shard_gpus.end(),
                              "ModelParallelEmbeddingMeta does not find shard id");
              int this_gpu_shard_id = std::distance(shard_gpus.begin(), find_shard_id_iter);

              std::vector<key_t> expected_keys;
              for (int i = 0; i < current_batch_size * lookup_params[lookup_id].max_hotness; ++i) {
                key_t key = i % (batch_size_per_dev * lookup_params[lookup_id].max_hotness);
                if (key % num_shards == this_gpu_shard_id) {
                  expected_keys.push_back(key);
                }
              }

              // --- check ---
              ASSERT_EQ(num_keys, expected_keys.size());

              auto begin = result_keys.begin() + result_bucket_range[shard_id * batch_size];
              auto end = begin + num_keys;
              std::vector<key_t> result_shard_keys(begin, end);

              ASSERT_EQ(result_shard_keys, expected_keys);

              //          printf("num_expected_keys: %zu\n", expected_keys.size());
              //          printf("expected keys: ");
              //          for (auto x : expected_keys) {
              //            printf("%d ", x);
              //          }
              //          printf("\n");
              //          printf("result_shard keys: ");
              //          for (auto x : result_shard_keys) {
              //            printf("%d ", x);
              //          }
              //          printf("\n");

              shard_id++;
            }
          }
        } break;
        case (embedding::EmbeddingGroupType::DataParallel): {
          std::vector<offset_t> result_bucket_range(group_result.bucket_range.num_elements());
          core23::copy_sync(result_bucket_range, group_result.bucket_range);

          std::vector<offset_t> result_keys_per_bucket(
              group_result.num_keys_per_bucket.num_elements());
          core23::copy_sync(result_keys_per_bucket, group_result.num_keys_per_bucket);
          ASSERT_EQ(result_keys_per_bucket, expected_keys_per_bucket);
          std::vector<key_t> expected_keys;
          int num_shards = 0;
          for (int lookup_id = 0; lookup_id < num_lookup; ++lookup_id) {
            if (ebc_param.has_table_shard(gpu_id, group_id, lookup_id)) {
              for (int i = 0;
                   i < ebc_param.lookup_params[lookup_id].max_hotness * num_valid_samples; ++i) {
                expected_keys.push_back(i);
              }
              num_shards++;
            }
          }

          //        printf("Skipping DP group: %zu\n", group_id);
          //        printf("DP Keys size: %d\n", (int)result_keys.size());
          //        printf("DP bucket range size: %d\n", (int)result_bucket_range.size());
          //
          //        printf("DP bucket range: ");
          //        for (auto x : result_bucket_range) {
          //          printf("%d ", (int)x);
          //        }
          //        printf("\n");
          //        printf("DP keys: ");
          //        for (auto x : result_keys) {
          //          printf("%d ", (int)x);
          //        }
          //        printf("\n");

          // --- check bucket range and num_keys
          size_t num_keys = result_bucket_range[num_shards * batch_size_per_dev];
          ASSERT_EQ(num_keys, expected_keys.size());
          ASSERT_EQ(results[gpu_id][group_id].h_num_keys, expected_keys.size());

          // --- check keys
          result_keys.resize(num_keys);
          ASSERT_EQ(result_keys, expected_keys);

        } break;
        default:
          break;
      }
    }
  }
}

const std::vector<int> device_list = {0, 1};

namespace mp {

const std::vector<std::vector<int>> shard_matrix = {
    {1, 0, 1, 1},
    {0, 1, 1, 1},
};

const std::vector<GroupedTableParam> grouped_emb_params = {
    {TablePlacementStrategy::ModelParallel, {0, 1}},
    {TablePlacementStrategy::ModelParallel, {2, 3}}};

TEST(data_distributor, mp_plan0_uint32) {
  test_data_distributor<uint32_t, uint32_t>(device_list, lookup_params0, shard_matrix,
                                            grouped_emb_params);
}

TEST(data_distributor, mp_plan1_uint32) {
  test_data_distributor<uint32_t, uint32_t>(device_list, lookup_params_with_shared_table,
                                            shard_matrix, grouped_emb_params);
}

TEST(data_distributor, mp_plan0_uint32_incomplete_batch) {
  test_data_distributor<uint32_t, uint32_t>(device_list, lookup_params0, shard_matrix,
                                            grouped_emb_params, true);
}

TEST(data_distributor, mp_plan1_uint32_incomplete_batch) {
  test_data_distributor<uint32_t, uint32_t>(device_list, lookup_params_with_shared_table,
                                            shard_matrix, grouped_emb_params, true);
}
}  // namespace mp

namespace dense_mp {

const std::vector<std::vector<int>> shard_matrix = {
    {1, 0, 1, 1},
    {0, 1, 1, 1},
};

const std::vector<GroupedTableParam> grouped_emb_params = {
    {TablePlacementStrategy::ModelParallel, {0, 1, 2, 3}}};

TEST(data_distributor, dense_mp_plan0) {
  test_data_distributor<uint32_t, uint32_t>(device_list, dense_lookup_params, shard_matrix,
                                            grouped_emb_params);
}
}  // namespace dense_mp

namespace dense_dp {

const std::vector<std::vector<int>> shard_matrix = {
    {1, 0, 1, 1},
    {1, 1, 1, 1},
};

const std::vector<GroupedTableParam> grouped_emb_params = {
    {TablePlacementStrategy::DataParallel, {0, 1, 2, 3}}};

TEST(data_distributor, dense_dp_plan0) {
  test_data_distributor<uint32_t, uint32_t>(device_list, dense_lookup_params, shard_matrix,
                                            grouped_emb_params);
}
}  // namespace dense_dp

namespace dp_and_mp {

const std::vector<std::vector<int>> shard_matrix = {
    {1, 0, 1, 1},
    {0, 1, 1, 1},
};

const std::vector<GroupedTableParam> grouped_emb_params = {
    {TablePlacementStrategy::DataParallel, {2, 3}},
    {TablePlacementStrategy::ModelParallel, {0, 1}}};

TEST(data_distributor, dp_and_mp_plan0_uint32) {
  test_data_distributor<uint32_t, uint32_t>(device_list, lookup_params0, shard_matrix,
                                            grouped_emb_params);
}

TEST(data_distributor, dp_and_mp_plan1_uint32) {
  test_data_distributor<uint32_t, uint32_t>(device_list, lookup_params_with_shared_table,
                                            shard_matrix, grouped_emb_params);
}

TEST(data_distributor, dp_and_mp_plan0_uint32_incomplete_batch) {
  test_data_distributor<uint32_t, uint32_t>(device_list, lookup_params0, shard_matrix,
                                            grouped_emb_params, true);
}

}  // namespace dp_and_mp