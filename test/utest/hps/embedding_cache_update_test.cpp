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

#include <cuda_profiler_api.h>
#include <gtest/gtest.h>

#include <cassert>
#include <common.hpp>
#include <filesystem>
#include <fstream>
#include <hps/embedding_cache.hpp>
#include <hps/hier_parameter_server.hpp>
#include <utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

template <typename TypeHashKey>
void get_embedding_per_table(const std::string& model_config_path,
                             const InferenceParams& inference_params,
                             const std::vector<size_t>& embedding_vec_size,
                             std::vector<std::vector<TypeHashKey>>& key_vec_per_table,
                             std::vector<std::vector<float>>& vec_vec_per_table) {
  // Create input file stream to read the embedding file
  for (unsigned int j = 0; j < inference_params.sparse_model_files.size(); j++) {
    const std::string emb_file_prefix = inference_params.sparse_model_files[j] + "/";
    const std::string key_file = emb_file_prefix + "key";
    const std::string vec_file = emb_file_prefix + "emb_vector";
    std::ifstream key_stream(key_file);
    std::ifstream vec_stream(vec_file);
    // Check if file is opened successfully
    if (!key_stream.is_open() || !vec_stream.is_open()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Error: embeddings file not open for reading");
    }
    const size_t key_file_size_in_byte = std::filesystem::file_size(key_file);
    const size_t vec_file_size_in_byte = std::filesystem::file_size(vec_file);

    const size_t key_size_in_byte = sizeof(long long);
    const size_t embedding_size = embedding_vec_size[j];
    const size_t vec_size_in_byte = sizeof(float) * embedding_size;

    const size_t num_key = key_file_size_in_byte / key_size_in_byte;
    const size_t num_vec = vec_file_size_in_byte / vec_size_in_byte;
    if (num_key != num_vec) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Error: num_key != num_vec in embedding file");
    }
    size_t num_float_val_in_vec_file = vec_file_size_in_byte / sizeof(float);

    // The temp embedding table
    std::vector<TypeHashKey> key_vec(num_key, 0);
    if (std::is_same<TypeHashKey, long long>::value) {
      key_stream.read(reinterpret_cast<char*>(key_vec.data()), key_file_size_in_byte);
    } else {
      std::vector<long long> i64_key_vec(num_key, 0);
      key_stream.read(reinterpret_cast<char*>(i64_key_vec.data()), key_file_size_in_byte);
      std::transform(i64_key_vec.begin(), i64_key_vec.end(), key_vec.begin(),
                     [](long long key) { return static_cast<unsigned>(key); });
    }

    std::vector<float> vec_vec(num_float_val_in_vec_file, 0.0f);
    vec_stream.read(reinterpret_cast<char*>(vec_vec.data()), vec_file_size_in_byte);

    key_vec_per_table.push_back(key_vec);
    vec_vec_per_table.push_back(vec_vec);
    key_stream.close();
    vec_stream.close();
  }
}

void compare_lookup(float* h_embeddingvector_gt, float* h_embeddingvector, size_t vec_length,
                    float thres) {
  for (size_t i = 0; i < vec_length; i++) {
    if (abs(h_embeddingvector_gt[i] - h_embeddingvector[i]) > thres) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Embedding cache lookup results are not consistent with the ground truth");
    }
  }
}

template <typename TypeHashKey>
void parameter_server_test(const std::string& config_file, const std::string& model,
                           const std::string& dense_model, std::vector<std::string>& sparse_models,
                           const std::vector<size_t>& embedding_vec_size,
                           const std::vector<size_t>& embedding_feature_num,
                           const std::vector<size_t>& slot_num_per_table, size_t max_batch_size,
                           float hit_rate_threshold, size_t max_iterations,
                           DatabaseType_t database_t = DatabaseType_t::ParallelHashMap) {
  VolatileDatabaseParams dis_database;
  PersistentDatabaseParams per_database;
  switch (database_t) {
    case DatabaseType_t::ParallelHashMap:
      dis_database.type = DatabaseType_t::ParallelHashMap;
      break;
    case DatabaseType_t::MultiProcessHashMap:
      dis_database.type = DatabaseType_t::MultiProcessHashMap;
      break;
    case DatabaseType_t::RedisCluster:
      dis_database.type = DatabaseType_t::RedisCluster;
      dis_database.address = "127.0.0.1:7000,127.0.0.1:7001,127.0.0.1:7002";
      break;
    case DatabaseType_t::RocksDB:
      dis_database.type = DatabaseType_t::Disabled;
      per_database.type = DatabaseType_t::RocksDB;
      per_database.path = "/hugectr/Test_Data/rockdb";
      break;
    default:
      break;
  }
  InferenceParams infer_param(model, max_batch_size, hit_rate_threshold, dense_model, sparse_models,
                              0, true, 1, true);
  infer_param.volatile_db = dis_database;
  infer_param.persistent_db = per_database;
  std::vector<InferenceParams> inference_params{infer_param};
  std::vector<std::string> model_config_path{config_file};

  // Create parameter server and get embedding cache
  parameter_server_config ps_config{model_config_path, inference_params};
  std::shared_ptr<HierParameterServerBase> parameter_server =
      HierParameterServerBase::create(ps_config, inference_params);
  auto embedding_cache = parameter_server->get_embedding_cache(model, 0);

  // Read embedding files as ground truth
  std::vector<std::vector<TypeHashKey>> key_vec_per_table;
  std::vector<std::vector<float>> vec_vec_per_table;
  get_embedding_per_table(config_file, infer_param, embedding_vec_size, key_vec_per_table,
                          vec_vec_per_table);

  // Allocate the resources for embedding cache lookup
  size_t num_emb_table = infer_param.sparse_model_files.size();
  size_t key_per_batch = max_batch_size * std::accumulate(embedding_feature_num.begin(),
                                                          embedding_feature_num.end(), 0);
  size_t vec_length_per_batch{0};
  for (size_t i = 0; i < num_emb_table; i++) {
    vec_length_per_batch += max_batch_size * embedding_feature_num[i] * embedding_vec_size[i];
  }

  size_t slot_num = std::accumulate(slot_num_per_table.begin(), slot_num_per_table.end(), 0);
  std::vector<std::vector<int>> h_row_ptrs_per_table(num_emb_table);
  for (size_t i = 0; i < num_emb_table; i++) {
    h_row_ptrs_per_table[i].resize(max_batch_size * slot_num_per_table[i] + 1);
    // All slots are one-hot
    std::iota(h_row_ptrs_per_table[i].begin(), h_row_ptrs_per_table[i].end(), 0);
  }
  TypeHashKey* h_embeddingcolumns;  // sample first layout
  TypeHashKey* h_keys;              // table first layout
  int* h_row_ptrs;
  float* d_embeddingvector;
  float* h_embeddingvector;
  float* h_embeddingvector_gt;
  HCTR_LIB_THROW(cudaHostAlloc((void**)&h_embeddingcolumns, key_per_batch * sizeof(TypeHashKey),
                               cudaHostAllocPortable));
  HCTR_LIB_THROW(
      cudaHostAlloc((void**)&h_keys, key_per_batch * sizeof(TypeHashKey), cudaHostAllocPortable));
  HCTR_LIB_THROW(cudaHostAlloc((void**)&h_row_ptrs,
                               (max_batch_size * slot_num + num_emb_table) * sizeof(int),
                               cudaHostAllocPortable));
  HCTR_LIB_THROW(cudaHostAlloc((void**)&h_embeddingvector, vec_length_per_batch * sizeof(float),
                               cudaHostAllocPortable));
  HCTR_LIB_THROW(cudaHostAlloc((void**)&h_embeddingvector_gt, vec_length_per_batch * sizeof(float),
                               cudaHostAllocPortable));
  HCTR_LIB_THROW(cudaMalloc((void**)&d_embeddingvector, vec_length_per_batch * sizeof(float)));
  std::vector<cudaStream_t> query_streams(num_emb_table);
  for (size_t i = 0; i < num_emb_table; i++) {
    HCTR_LIB_THROW(cudaStreamCreate(&query_streams[i]));
  }

  // Copy from h_row_ptrs_per_table to h_row_ptrs
  size_t acc_row_ptrs_offset{0};
  for (size_t i = 0; i < num_emb_table; i++) {
    memcpy(h_row_ptrs + acc_row_ptrs_offset, h_row_ptrs_per_table[i].data(),
           h_row_ptrs_per_table[i].size() * sizeof(int));
    acc_row_ptrs_offset += h_row_ptrs_per_table[i].size();
  }

  // Embedding cache lookup for multiple time
  std::vector<size_t> vector_layout_offset(num_emb_table, 0);
  for (size_t i = 1; i < num_emb_table; i++) {
    vector_layout_offset[i] = vector_layout_offset[i - 1] + max_batch_size *
                                                                embedding_feature_num[i - 1] *
                                                                embedding_vec_size[i - 1];
  }

  auto embedding_cache_look_up = [&]() {
    // Redistribute keys ：from sample first to table first
    distribute_keys_per_table(h_keys, h_embeddingcolumns, h_row_ptrs, max_batch_size,
                              slot_num_per_table);
    size_t acc_vectors_offset{0};
    size_t acc_keys_offset{0};
    size_t num_keys{0};
    for (size_t table_id = 0; table_id < num_emb_table; table_id++) {
      num_keys = h_row_ptrs_per_table[table_id].back();
      embedding_cache->lookup(table_id, d_embeddingvector + acc_vectors_offset,
                              h_keys + acc_keys_offset, num_keys, hit_rate_threshold,
                              query_streams[table_id]);
      acc_keys_offset += num_keys;
      acc_vectors_offset +=
          max_batch_size * embedding_feature_num[table_id] * embedding_vec_size[table_id];
    }
    for (size_t table_id = 0; table_id < num_emb_table; table_id++) {
      HCTR_LIB_THROW(cudaStreamSynchronize(query_streams[table_id]));
    }
  };

  for (size_t i = 0; i < max_iterations; i++) {
    size_t key_offset{0};
    size_t vec_offset{0};
    for (size_t j = 0; j < max_batch_size; j++) {
      for (size_t k = 0; k < num_emb_table; k++) {
        for (size_t l = 0; l < embedding_feature_num[k]; l++) {
          int randomIndex = rand() % key_vec_per_table[k].size();
          h_embeddingcolumns[key_offset++] = key_vec_per_table[k][randomIndex];
          if (i == max_iterations - 1) {
            vec_offset = vector_layout_offset[k] +
                         j * embedding_feature_num[k] * embedding_vec_size[k] +
                         l * embedding_vec_size[k];
            memcpy(h_embeddingvector_gt + vec_offset,
                   vec_vec_per_table[k].data() + randomIndex * embedding_vec_size[k],
                   embedding_vec_size[k] * sizeof(float));
          }
        }
      }
    }
    embedding_cache_look_up();
  }

  // Embedding cache finalize, will join the asynchronous insert threads if there are any joinable
  // threads
  embedding_cache->finalize();

  // Embedding cache lookup and check results with ground truth
  embedding_cache_look_up();
  HCTR_LIB_THROW(cudaMemcpy(h_embeddingvector, d_embeddingvector,
                            vec_length_per_batch * sizeof(float), cudaMemcpyDeviceToHost));
  compare_lookup(h_embeddingvector_gt, h_embeddingvector, vec_length_per_batch, 0.01f);

  // Refresh embedding cache
  parameter_server->refresh_embedding_cache(model, 0);

  // Embedding cache lookup and check results with ground truth
  embedding_cache_look_up();
  HCTR_LIB_THROW(cudaMemcpy(h_embeddingvector, d_embeddingvector,
                            vec_length_per_batch * sizeof(float), cudaMemcpyDeviceToHost));
  compare_lookup(h_embeddingvector_gt, h_embeddingvector, vec_length_per_batch, 0.01f);

  // Release CUDA resources
  for (size_t i = 0; i < num_emb_table; i++) {
    HCTR_LIB_THROW(cudaStreamSynchronize(query_streams[i]));
  }
  for (size_t i = 0; i < num_emb_table; i++) {
    HCTR_LIB_THROW(cudaStreamDestroy(query_streams[i]));
  }
  HCTR_LIB_THROW(cudaFreeHost(h_embeddingcolumns));
  HCTR_LIB_THROW(cudaFreeHost(h_keys));
  HCTR_LIB_THROW(cudaFreeHost(h_row_ptrs));
  HCTR_LIB_THROW(cudaFreeHost(h_embeddingvector));
  HCTR_LIB_THROW(cudaFreeHost(h_embeddingvector_gt));
  HCTR_LIB_THROW(cudaFree(d_embeddingvector));
}

}  // end namespace

std::string dense_model{"/models/wdl/1/wdl_dense_20000.model"};
std::string network{"/models/wdl/1/wdl.json"};
std::string model_name = "wdl";
std::vector<std::string> sparse_models{"/models/wdl/1/wdl0_sparse_20000.model",
                                       "/models/wdl/1/wdl1_sparse_20000.model"};
std::vector<size_t> embedding_vec_size_wdl{1, 16};
std::vector<size_t> embedding_featre_num_wdl{2, 26};
std::vector<size_t> slot_num_per_table_wdl{2, 26};
TEST(parameter_server, CPU_look_up_1x00x1) {
  parameter_server_test<long long>(
      network, model_name, dense_model, sparse_models, embedding_vec_size_wdl,
      embedding_featre_num_wdl, slot_num_per_table_wdl, 1, 0.f, 1, DatabaseType_t::ParallelHashMap);
}
TEST(parameter_server, CPU_look_up_1x10x1) {
  parameter_server_test<long long>(
      network, model_name, dense_model, sparse_models, embedding_vec_size_wdl,
      embedding_featre_num_wdl, slot_num_per_table_wdl, 1, 1.f, 1, DatabaseType_t::ParallelHashMap);
}
TEST(parameter_server, CPU_look_up_1x00x100) {
  parameter_server_test<long long>(network, model_name, dense_model, sparse_models,
                                   embedding_vec_size_wdl, embedding_featre_num_wdl,
                                   slot_num_per_table_wdl, 1, 0.f, 100,
                                   DatabaseType_t::ParallelHashMap);
}
TEST(parameter_server, CPU_look_up_1x10x100) {
  parameter_server_test<long long>(network, model_name, dense_model, sparse_models,
                                   embedding_vec_size_wdl, embedding_featre_num_wdl,
                                   slot_num_per_table_wdl, 1, 1.f, 100,
                                   DatabaseType_t::ParallelHashMap);
}
TEST(parameter_server, CPU_look_up_1024x00x100) {
  parameter_server_test<long long>(network, model_name, dense_model, sparse_models,
                                   embedding_vec_size_wdl, embedding_featre_num_wdl,
                                   slot_num_per_table_wdl, 1024, 0.f, 100,
                                   DatabaseType_t::ParallelHashMap);
}
TEST(parameter_server, CPU_look_up_1024x10x100) {
  parameter_server_test<long long>(network, model_name, dense_model, sparse_models,
                                   embedding_vec_size_wdl, embedding_featre_num_wdl,
                                   slot_num_per_table_wdl, 1024, 1.f, 100,
                                   DatabaseType_t::ParallelHashMap);
}
TEST(parameter_server, CPU_look_up_1024x05x100) {
  parameter_server_test<long long>(network, model_name, dense_model, sparse_models,
                                   embedding_vec_size_wdl, embedding_featre_num_wdl,
                                   slot_num_per_table_wdl, 1024, 0.5f, 100,
                                   DatabaseType_t::ParallelHashMap);
}
TEST(parameter_server, Rocksdb_look_up_1024x00x100) {
  parameter_server_test<long long>(network, model_name, dense_model, sparse_models,
                                   embedding_vec_size_wdl, embedding_featre_num_wdl,
                                   slot_num_per_table_wdl, 1024, 0.f, 100, DatabaseType_t::RocksDB);
}
TEST(parameter_server, Redis_look_up_1024x00x100) {
  parameter_server_test<long long>(network, model_name, dense_model, sparse_models,
                                   embedding_vec_size_wdl, embedding_featre_num_wdl,
                                   slot_num_per_table_wdl, 1024, 0.f, 100,
                                   DatabaseType_t::RedisCluster);
}