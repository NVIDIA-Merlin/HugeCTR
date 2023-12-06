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

#include <sys/time.h>

#include <core/hctr_impl/hctr_backend.hpp>
#include <embedding/embedding.hpp>
#include <embeddings/embedding_collection.hpp>
#include <numeric>
#include <resource_managers/resource_manager_ext.hpp>
#include <utest/embedding_collection/embedding_collection_utils.hpp>
#include <utest/embedding_collection/reference_embedding.hpp>

namespace {
// 1. optimizer
static HugeCTR::OptParams sgd_opt{.optimizer = HugeCTR::Optimizer_t::SGD, .lr = 1e-3, .scaler = 1};
static HugeCTR::OptParams adagrad_opt{
    .optimizer = HugeCTR::Optimizer_t::AdaGrad, .lr = 1e-3, .scaler = 1};

// 2. runtime configuration
struct RuntimeConfiguration {
  int batch_size_per_gpu;
  int num_gpus_per_node;
  int num_node;
};

static RuntimeConfiguration single_node{
    .batch_size_per_gpu = 8192, .num_gpus_per_node = 8, .num_node = 1};

static RuntimeConfiguration two_node{
    .batch_size_per_gpu = 8192, .num_gpus_per_node = 8, .num_node = 2};

static RuntimeConfiguration eight_node{
    .batch_size_per_gpu = 8192, .num_gpus_per_node = 8, .num_node = 8};

static RuntimeConfiguration sixteen_node{
    .batch_size_per_gpu = 8192, .num_gpus_per_node = 8, .num_node = 16};

// 3. table configuration
struct EmbeddingConfiguration {
  int num_table;
  std::vector<int> max_hotness_list;
  int64_t max_vocabulary_size;
  int emb_vec_size;
  Combiner combiner = Combiner::Concat;
};

static std::vector<EmbeddingConfiguration> tiny_embedding{
    {1, {1, 10}, 10000, 8, Combiner::Sum},
    {1, {1, 10}, 1000000, 16, Combiner::Sum},
    {1, {1, 10}, 25000000, 16, Combiner::Sum},
    {1, {1, 10}, 25000000, 16},
    {16, {1}, 10, 8},
    {10, {1}, 1000, 8},
    {4, {1}, 10000, 8},
    {2, {1}, 100000, 16},
    {19, {1}, 1000000, 16},
};

static std::vector<EmbeddingConfiguration> small_embedding{
    {5, {1, 30}, 10000, 16, Combiner::Sum},
    {3, {1, 30}, 4000000, 32, Combiner::Sum},
    {1, {1, 30}, 50000000, 32, Combiner::Sum},
    {1, {1}, 50000000, 32},
    {30, {1}, 10, 16},
    {30, {1}, 1000, 16},
    {5, {1}, 10000, 16},
    {5, {1}, 100000, 32},
    {27, {1}, 4000000, 32},
};

static std::vector<EmbeddingConfiguration> medium_embedding{
    {20, {1, 50}, 100000, 64, Combiner::Sum},
    {5, {1, 50}, 10000000, 64, Combiner::Sum},
    {1, {1, 50}, 100000000, 128, Combiner::Sum},
    {1, {1}, 100000000, 128},
    {80, {1}, 10, 32},
    {60, {1}, 1000, 32},
    {80, {1}, 100000, 64},
    {24, {1}, 200000, 64},
    {40, {1}, 10000000, 64},
};

static std::vector<EmbeddingConfiguration> large_embedding{
    {40, {1, 100}, 100000, 64, Combiner::Sum},
    {16, {1, 100}, 15000000, 64, Combiner::Sum},
    {1, {1, 100}, 200000000, 128, Combiner::Sum},
    {1, {1}, 200000000, 128},
    {100, {1}, 10, 32},
    {100, {1}, 10000, 32},
    {160, {1}, 100000, 64},
    {50, {1}, 500000, 64},
    {144, {1}, 15000000, 64},
};

static std::vector<EmbeddingConfiguration> jumbo_embedding{
    {50, {1, 200}, 100000, 128, Combiner::Sum},
    {24, {1, 200}, 20000000, 128, Combiner::Sum},
    {1, {1, 200}, 400000000, 256, Combiner::Sum},
    {1, {1}, 400000000, 256},
    {100, {1}, 10, 32},
    {200, {1}, 10000, 64},
    {350, {1}, 100000, 128},
    {80, {1}, 1000000, 128},
    {216, {1}, 20000000, 128},
};

static std::vector<EmbeddingConfiguration> colossal_embedding{
    {100, {1, 300}, 100000, 128, Combiner::Sum},
    {50, {1, 300}, 40000000, 256, Combiner::Sum},
    {1, {1, 300}, 2000000000, 256, Combiner::Sum},
    {1, {1}, 1000000000, 256},
    {100, {1}, 10, 32},
    {400, {1}, 10000, 128},
    {100, {1}, 100000, 128},
    {800, {1}, 1000000, 128},
    {450, {1}, 40000000, 256},
};

static std::vector<EmbeddingConfiguration> criteo_embedding{
    {1, {1}, 39884406, 128}, {1, {1}, 39043, 128},    {1, {1}, 17289, 128},
    {1, {1}, 7420, 128},     {1, {1}, 20263, 128},    {1, {1}, 3, 128},
    {1, {1}, 7120, 128},     {1, {1}, 1543, 128},     {1, {1}, 63, 128},
    {1, {1}, 38532951, 128}, {1, {1}, 2953546, 128},  {1, {1}, 403346, 128},
    {1, {1}, 10, 128},       {1, {1}, 2208, 128},     {1, {1}, 11938, 128},
    {1, {1}, 155, 128},      {1, {1}, 4, 128},        {1, {1}, 976, 128},
    {1, {1}, 14, 128},       {1, {1}, 39979771, 128}, {1, {1}, 25641295, 128},
    {1, {1}, 39664984, 128}, {1, {1}, 585935, 128},   {1, {1}, 12972, 128},
    {1, {1}, 108, 128},      {1, {1}, 36, 128},
};

static std::vector<EmbeddingConfiguration> criteo_multi_hot_embedding{
    {1, {3}, 40000000, 128, Combiner::Sum},
    {1, {2}, 39060, 128, Combiner::Sum},
    {1, {1}, 17295, 128},
    {1, {2}, 7424, 128, Combiner::Sum},
    {1, {6}, 20265, 128, Combiner::Sum},
    {1, {1}, 3, 128},
    {1, {1}, 7122, 128},
    {1, {1}, 1543, 128},
    {1, {1}, 63, 128},
    {1, {7}, 40000000, 128, Combiner::Sum},
    {1, {3}, 3067956, 128, Combiner::Sum},
    {1, {8}, 405282, 128, Combiner::Sum},
    {1, {1}, 10, 128},
    {1, {6}, 2209, 128, Combiner::Sum},
    {1, {9}, 11938, 128, Combiner::Sum},
    {1, {5}, 155, 128, Combiner::Sum},
    {1, {1}, 4, 128},
    {1, {1}, 976, 128},
    {1, {1}, 14, 128},
    {1, {12}, 40000000, 128, Combiner::Sum},
    {1, {100}, 40000000, 128, Combiner::Sum},
    {1, {27}, 40000000, 128, Combiner::Sum},
    {1, {10}, 590152, 128, Combiner::Sum},
    {1, {3}, 12973, 128, Combiner::Sum},
    {1, {1}, 108, 128},
    {1, {1}, 36, 128},
};

// 4. shard_matrix
struct ShardConfiguration {
  std::vector<std::vector<int>> shard_matrix;  // num_gpus * num_table
  std::vector<embedding::GroupedTableParam> grouped_table_params;
  embedding::CompressionParam compression_param;
};

namespace sharding {
int get_num_table(const std::vector<EmbeddingConfiguration> &embedding_config) {
  int num_table = 0;
  for (auto &config : embedding_config) {
    num_table += config.num_table;
  }
  return num_table;
}

ShardConfiguration table_wise_sharding(const RuntimeConfiguration &runtime_config,
                                       const std::vector<EmbeddingConfiguration> &embedding_config,
                                       bool use_dense_reduction = false) {
  int num_global_gpus = runtime_config.num_node * runtime_config.num_gpus_per_node;
  int num_table = get_num_table(embedding_config);

  std::vector<std::vector<int>> shard_matrix(num_global_gpus, std::vector<int>(num_table, 0));
  std::vector<int64_t> mp_table_size(num_global_gpus, 0ul);
  int table_id = 0;
  for (auto &config : embedding_config) {
    float table_size = config.max_vocabulary_size * config.emb_vec_size;
    for (int i = 0; i < config.num_table; ++i) {
      shard_matrix[table_id % num_global_gpus][table_id] = 1;
      mp_table_size[table_id % num_global_gpus] += static_cast<int64_t>(table_size);
      table_id++;
    }
  }
  std::vector<int> table_ids(num_table);
  std::iota(table_ids.begin(), table_ids.end(), 0);

  std::cout << "table wise sharding mp table size per gpu:\n";
  for (int gpu_id = 0; gpu_id < num_global_gpus; ++gpu_id) {
    std::cout << "gpu_id:" << gpu_id
              << ", table size:" << static_cast<float>(mp_table_size[gpu_id] * 4) / 1e9
              << "GBytes\n";
  }

  embedding::CompressionParam compression_param;
  if (use_dense_reduction) {
    compression_param.compression_strategy_to_table_ids[embedding::CompressionStrategy::Unique] =
        std::set<int>(table_ids.begin(), table_ids.end());
  }
  return {shard_matrix, {{TablePlacementStrategy::ModelParallel, table_ids}}};
}

ShardConfiguration row_wise_sharding(const RuntimeConfiguration &runtime_config,
                                     const std::vector<EmbeddingConfiguration> &embedding_config,
                                     bool use_dense_reduction = false) {
  int num_global_gpus = runtime_config.num_node * runtime_config.num_gpus_per_node;
  int num_table = get_num_table(embedding_config);

  std::vector<std::vector<int>> shard_matrix(num_global_gpus, std::vector<int>(num_table, 1));
  std::vector<int64_t> mp_table_size(num_global_gpus, 0ul);
  for (auto &config : embedding_config) {
    float table_size = config.max_vocabulary_size * config.emb_vec_size;
    for (int i = 0; i < config.num_table; ++i) {
      for (int gpu_id = 0; gpu_id < num_global_gpus; ++gpu_id) {
        mp_table_size[gpu_id] += static_cast<int64_t>(table_size / num_global_gpus);
      }
    }
  }
  std::vector<int> table_ids(num_table);
  std::iota(table_ids.begin(), table_ids.end(), 0);

  std::cout << "row wise sharding mp table size per gpu:\n";
  for (int gpu_id = 0; gpu_id < num_global_gpus; ++gpu_id) {
    std::cout << "gpu_id:" << gpu_id
              << ", table size:" << static_cast<float>(mp_table_size[gpu_id] * 4) / 1e9
              << "GBytes\n";
  }

  embedding::CompressionParam compression_param;
  if (use_dense_reduction) {
    compression_param.compression_strategy_to_table_ids[embedding::CompressionStrategy::Unique] =
        std::set<int>(table_ids.begin(), table_ids.end());
  }
  return {shard_matrix, {{TablePlacementStrategy::ModelParallel, table_ids}}, compression_param};
}

ShardConfiguration hybrid_sharding(const RuntimeConfiguration &runtime_config,
                                   const std::vector<EmbeddingConfiguration> &embedding_config) {
  int num_local_gpus = runtime_config.num_gpus_per_node;
  int num_global_gpus = runtime_config.num_node * runtime_config.num_gpus_per_node;
  int num_table = get_num_table(embedding_config);

  std::vector<std::vector<int>> shard_matrix(num_global_gpus, std::vector<int>(num_table, 0));

  float total_table_size = 0;
  for (auto &config : embedding_config) {
    total_table_size += config.num_table * config.max_vocabulary_size * config.emb_vec_size;
  }
  int table_id = 0;
  int target_node_id = 0;
  int target_gpu_id = 0;
  std::vector<int> dp_table_ids;
  std::vector<int> mp_table_ids;
  int64_t dp_table_size = 0ul;
  std::vector<int64_t> mp_table_size(num_global_gpus, 0ul);
  for (auto &config : embedding_config) {
    float table_size = config.max_vocabulary_size * config.emb_vec_size;
    for (int i = 0; i < config.num_table; ++i) {
      bool has_dense_lookup = (config.combiner == Combiner::Concat);
      if (table_size < total_table_size / num_global_gpus / num_table && !has_dense_lookup) {
        // 1. shard on every gpu with dp
        for (int gpu_id = 0; gpu_id < num_global_gpus; ++gpu_id) {
          shard_matrix[gpu_id][table_id] = 1;
        }
        dp_table_ids.push_back(table_id);
        dp_table_size += static_cast<int64_t>(table_size);
      } else if (table_size < total_table_size / num_global_gpus) {
        // 2. shard on single gpu
        shard_matrix[target_gpu_id][table_id] = 1;
        target_gpu_id += 1;
        target_gpu_id %= num_global_gpus;
        mp_table_ids.push_back(table_id);
        mp_table_size[target_gpu_id] += static_cast<int64_t>(table_size);
      } else if (table_size < total_table_size / num_local_gpus) {
        // 3. shard on single node
        for (int gpu_id = 0; gpu_id < num_local_gpus; ++gpu_id) {
          shard_matrix[target_node_id + gpu_id][table_id] = 1;
          mp_table_size[target_node_id + gpu_id] +=
              static_cast<int64_t>(table_size / num_local_gpus);
        }
        target_node_id += 1;
        target_node_id %= runtime_config.num_node;
        mp_table_ids.push_back(table_id);

      } else {
        // 4. shard on all gpus
        for (int gpu_id = 0; gpu_id < num_global_gpus; ++gpu_id) {
          shard_matrix[gpu_id][table_id] = 1;
          mp_table_size[gpu_id] += static_cast<int64_t>(table_size / num_global_gpus);
        }
        mp_table_ids.push_back(table_id);
      }
      ++table_id;
    }
  }
  std::vector<embedding::GroupedTableParam> grouped_table_params;
  std::cout << "hybrid sharding table size:\n";
  if (!mp_table_ids.empty()) {
    grouped_table_params.push_back({TablePlacementStrategy::ModelParallel, mp_table_ids});
    std::cout << "mp table size per gpu:\n";
    for (int gpu_id = 0; gpu_id < num_global_gpus; ++gpu_id) {
      std::cout << "gpu_id:" << gpu_id
                << ", table size:" << static_cast<float>(mp_table_size[gpu_id] * 4) / 1e9
                << "GBytes\n";
    }
  }
  if (!dp_table_ids.empty()) {
    grouped_table_params.push_back({TablePlacementStrategy::DataParallel, dp_table_ids});
    std::cout << "dp table size per gpu: " << static_cast<float>(dp_table_size * 4) / 1e9
              << "GBytes\n";
  }
  return {shard_matrix, grouped_table_params, {}};
}
}  // namespace sharding

// 5. optimization option
struct EmbeddingCollectionOption {
  embedding::EmbeddingLayout input_layout;
  embedding::EmbeddingLayout output_layout;
  embedding::KeysPreprocessStrategy keys_preprocess_strategy;
  embedding::SortStrategy sort_strategy;
  embedding::AllreduceStrategy allreduce_strategy;
  embedding::CommunicationStrategy comm_strategy;
};
std::ostream &operator<<(std::ostream &os, const EmbeddingCollectionOption &p) {
  os << "\n\tinput_layout:" << p.input_layout << "\n\toutput_layout:" << p.output_layout
     << "\n\tkeys_preprocess_strategy:" << p.keys_preprocess_strategy
     << "\n\tsort_strategy:" << p.sort_strategy << "\n\tallreduce_strategy:" << p.allreduce_strategy
     << "\n\tcomm_strategy:" << p.comm_strategy << std::endl;
  return os;
}

// 6. input data configuration
enum class InputDataType { Uniform, Powerlaw, RawFormat };
struct InputDataConfiguration {
  bool fixed_hotness;
  InputDataType input_data_type;

  struct PowerlawParam {
    float alpha;
  } powerlaw_param;
  struct RawFormatParam {
    std::string input_file;
    int label_dim;
    int dense_dim;
  } raw_format_param;
};

InputDataConfiguration synthetic_uniform_dataset = {.fixed_hotness = true,
                                                    .input_data_type = InputDataType::Uniform};

InputDataConfiguration criteo_dataset = {
    .fixed_hotness = true,
    .input_data_type = InputDataType::RawFormat,
    .raw_format_param = {
        .input_file = "/raid/datasets/criteo/mlperf/40m.limit_preshuffled/train_data.bin",
        .label_dim = 1,
        .dense_dim = 13}};

InputDataConfiguration criteo_multi_hot_dataset = {
    .fixed_hotness = true,
    .input_data_type = InputDataType::RawFormat,
    .raw_format_param = {
        .input_file = "/lustre/fsw/mlperf/mlperft-dlrm/datasets/criteo_multihot_raw/train_data.bin",
        .label_dim = 1,
        .dense_dim = 13}};

struct Configuration {
  std::vector<EmbeddingConfiguration> embedding_config;

  HugeCTR::OptParams opt;

  ShardConfiguration shard_configuration;

  RuntimeConfiguration runtime_configuration;

  InputDataConfiguration input_data_configuration;

  std::vector<EmbeddingCollectionOption> options;

  bool reference_check;
  int niters = 1;
};

// detailed configuration
std::vector<Configuration> get_ebc_single_node_utest_configuration() {
  std::vector<EmbeddingCollectionOption> options{
      EmbeddingCollectionOption{
          embedding::EmbeddingLayout::FeatureMajor, embedding::EmbeddingLayout::FeatureMajor,
          embedding::KeysPreprocessStrategy::AddOffset, embedding::SortStrategy::Radix,
          embedding::AllreduceStrategy::Dense, embedding::CommunicationStrategy::Uniform},
      EmbeddingCollectionOption{
          embedding::EmbeddingLayout::FeatureMajor, embedding::EmbeddingLayout::BatchMajor,
          embedding::KeysPreprocessStrategy::AddOffset, embedding::SortStrategy::Radix,
          embedding::AllreduceStrategy::Dense, embedding::CommunicationStrategy::Uniform},
      EmbeddingCollectionOption{
          embedding::EmbeddingLayout::FeatureMajor, embedding::EmbeddingLayout::FeatureMajor,
          embedding::KeysPreprocessStrategy::AddOffset, embedding::SortStrategy::Segmented,
          embedding::AllreduceStrategy::Dense, embedding::CommunicationStrategy::Uniform},
  };

  std::vector<Configuration> configurations{
      Configuration{
          .embedding_config = tiny_embedding,
          .opt = sgd_opt,
          .shard_configuration = sharding::table_wise_sharding(single_node, tiny_embedding),
          .runtime_configuration = single_node,
          .input_data_configuration = synthetic_uniform_dataset,
          .options = options,
          .reference_check = true,
      },
      Configuration{
          .embedding_config = tiny_embedding,
          .opt = sgd_opt,
          .shard_configuration = sharding::table_wise_sharding(single_node, tiny_embedding, true),
          .runtime_configuration = single_node,
          .input_data_configuration = synthetic_uniform_dataset,
          .options = options,
          .reference_check = true,
      },
      Configuration{
          .embedding_config = criteo_embedding,
          .opt = sgd_opt,
          .shard_configuration = sharding::row_wise_sharding(single_node, criteo_embedding),
          .runtime_configuration = single_node,
          .input_data_configuration = synthetic_uniform_dataset,
          .options = options,
          .reference_check = true,
      },
      Configuration{
          .embedding_config = criteo_embedding,
          .opt = sgd_opt,
          .shard_configuration = sharding::row_wise_sharding(single_node, criteo_embedding, true),
          .runtime_configuration = single_node,
          .input_data_configuration = synthetic_uniform_dataset,
          .options = options,
          .reference_check = true,
      },
      Configuration{
          .embedding_config = criteo_multi_hot_embedding,
          .opt = adagrad_opt,
          .shard_configuration = sharding::hybrid_sharding(single_node, criteo_multi_hot_embedding),
          .runtime_configuration = single_node,
          .input_data_configuration = synthetic_uniform_dataset,
          .options = options,
          .reference_check = true,
      },
  };
  return configurations;
}

std::vector<Configuration> get_ebc_two_node_utest_configuration() {
  std::vector<EmbeddingCollectionOption> options{
      EmbeddingCollectionOption{
          embedding::EmbeddingLayout::FeatureMajor, embedding::EmbeddingLayout::FeatureMajor,
          embedding::KeysPreprocessStrategy::AddOffset, embedding::SortStrategy::Radix,
          embedding::AllreduceStrategy::Dense, embedding::CommunicationStrategy::Hierarchical},
      EmbeddingCollectionOption{
          embedding::EmbeddingLayout::FeatureMajor, embedding::EmbeddingLayout::BatchMajor,
          embedding::KeysPreprocessStrategy::AddOffset, embedding::SortStrategy::Radix,
          embedding::AllreduceStrategy::Dense, embedding::CommunicationStrategy::Hierarchical},
      EmbeddingCollectionOption{
          embedding::EmbeddingLayout::FeatureMajor, embedding::EmbeddingLayout::FeatureMajor,
          embedding::KeysPreprocessStrategy::AddOffset, embedding::SortStrategy::Segmented,
          embedding::AllreduceStrategy::Dense, embedding::CommunicationStrategy::Hierarchical},
  };

  std::vector<Configuration> configurations{
      Configuration{
          .embedding_config = tiny_embedding,
          .opt = sgd_opt,
          .shard_configuration = sharding::table_wise_sharding(two_node, tiny_embedding),
          .runtime_configuration = two_node,
          .input_data_configuration = synthetic_uniform_dataset,
          .options = options,
          .reference_check = true,
      },
      Configuration{
          .embedding_config = tiny_embedding,
          .opt = sgd_opt,
          .shard_configuration = sharding::table_wise_sharding(two_node, tiny_embedding, true),
          .runtime_configuration = two_node,
          .input_data_configuration = synthetic_uniform_dataset,
          .options = options,
          .reference_check = true,
      },
      Configuration{
          .embedding_config = criteo_embedding,
          .opt = sgd_opt,
          .shard_configuration = sharding::row_wise_sharding(two_node, criteo_embedding),
          .runtime_configuration = two_node,
          .input_data_configuration = synthetic_uniform_dataset,
          .options = options,
          .reference_check = true,
      },
      Configuration{
          .embedding_config = criteo_embedding,
          .opt = sgd_opt,
          .shard_configuration = sharding::row_wise_sharding(two_node, criteo_embedding, true),
          .runtime_configuration = two_node,
          .input_data_configuration = synthetic_uniform_dataset,
          .options = options,
          .reference_check = true,
      },
      Configuration{
          .embedding_config = criteo_multi_hot_embedding,
          .opt = adagrad_opt,
          .shard_configuration = sharding::hybrid_sharding(two_node, criteo_multi_hot_embedding),
          .runtime_configuration = two_node,
          .input_data_configuration = synthetic_uniform_dataset,
          .options = options,
          .reference_check = true,
      },
  };
  return configurations;
}
}  // namespace
