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

#include "HugeCTR/include/data_generator.hpp"
#include "HugeCTR/include/inference/session_inference.hpp"
#include "HugeCTR/include/inference/embedding_interface.hpp"
#include "HugeCTR/include/common.hpp"
#include "experimental/filesystem"
#include "HugeCTR/include/general_buffer2.hpp"
#include "HugeCTR/include/utils.hpp"
#include "vector"
#include "gtest/gtest.h"
#include "utest/test_utils.h"
#include "fstream"
#include "cuda_profiler_api.h"
#include "cassert"

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
      CK_THROW_(Error_t::WrongInput, "Error: embeddings file not open for reading");
    }
    const size_t key_file_size_in_byte = std::experimental::filesystem::file_size(key_file);
    const size_t vec_file_size_in_byte = std::experimental::filesystem::file_size(vec_file);

    const size_t key_size_in_byte = sizeof(long long);
    const size_t embedding_size = embedding_vec_size[j];
    const size_t vec_size_in_byte = sizeof(float) * embedding_size;

    const size_t num_key = key_file_size_in_byte / key_size_in_byte;
    const size_t num_vec = vec_file_size_in_byte / vec_size_in_byte;
    if (num_key != num_vec) {
      CK_THROW_(Error_t::WrongInput, "Error: num_key != num_vec in embedding file");
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

void compare_lookup(float* h_embeddingvector_gt, float* h_embeddingvector, size_t vec_length, float thres) {
  for (size_t i = 0; i < vec_length; i++) {
    if (abs(h_embeddingvector_gt[i] - h_embeddingvector[i]) > thres) {
      CK_THROW_(Error_t::WrongInput, "Embedding cache lookup results are not consistent with the ground truth");
    }
  }
}

template <typename TypeHashKey>
void parameter_server_test(const std::string& config_file, const std::string& model, const std::string& dense_model,
                          std::vector<std::string>& sparse_models, const std::vector<size_t>& embedding_vec_size,
                          const std::vector<size_t>& embedding_feature_num, size_t max_batch_size,
                          float hit_rate_threshold, size_t max_iterations,
                          DatabaseType_t database_t = DatabaseType_t::ParallelHashMap) {
    VolatileDatabaseParams dis_database;
    PersistentDatabaseParams per_database;
    switch (database_t) {
        case DatabaseType_t::ParallelHashMap:
            dis_database.type = DatabaseType_t::ParallelHashMap;
            break;
        case DatabaseType_t::RedisCluster :
            dis_database.type = DatabaseType_t::RedisCluster;
            dis_database.address = "127.0.0.1:7000,127.0.0.1:7001,127.0.0.1:7002";
            break;
        case DatabaseType_t::RocksDB:
            dis_database.type = DatabaseType_t::Disabled;
            per_database.type = DatabaseType_t::RocksDB;
            per_database.path="/hugectr/Test_Data/rockdb" ;
            break;
        default:
            break;
    }
    InferenceParams infer_param(model, max_batch_size, hit_rate_threshold, dense_model, sparse_models, 0, true, 1, true);
    infer_param.volatile_db = dis_database;
    infer_param.persistent_db = per_database;
    std::vector<InferenceParams> inference_params{infer_param};
    std::vector<std::string> model_config_path{config_file};
    
    // Create parameter server and get embedding cache
    std::shared_ptr<HugectrUtility<TypeHashKey>> parameter_server;
    parameter_server.reset(HugectrUtility<TypeHashKey>::Create_Parameter_Server(
      INFER_TYPE::TRITON, model_config_path, inference_params));
    auto embedding_cache = parameter_server->GetEmbeddingCache(model, 0);
    
    // Read embedding files as ground truth
    std::vector<std::vector<TypeHashKey>> key_vec_per_table;
    std::vector<std::vector<float>> vec_vec_per_table;
    get_embedding_per_table(config_file, infer_param, embedding_vec_size, key_vec_per_table, vec_vec_per_table);

    // Allocate the resources for embedding cache lookup
    size_t num_emb_table = infer_param.sparse_model_files.size();
    size_t key_per_batch = max_batch_size * std::accumulate(embedding_feature_num.begin(), embedding_feature_num.end(), 0);
    size_t vec_length_per_batch{0};
    for (size_t i = 0; i < num_emb_table; i++) {
      vec_length_per_batch += max_batch_size * embedding_feature_num[i] * embedding_vec_size[i];
    }
    std::vector<size_t> h_embedding_offset(max_batch_size * num_emb_table + 1, 0);
    for (size_t i = 0; i < max_batch_size; i++) {
      for (size_t j = 0; j < num_emb_table; j++) {
        h_embedding_offset[i*num_emb_table+j+1] = h_embedding_offset[i*num_emb_table+j] + embedding_feature_num[j];
      }
    }
    TypeHashKey* h_embeddingcolumns;
    float* d_embeddingvector;
    float* h_embeddingvector;
    float* h_embeddingvector_gt;
    CK_CUDA_THROW_(cudaHostAlloc((void**)&h_embeddingcolumns,
                                key_per_batch * sizeof(TypeHashKey), cudaHostAllocPortable));
    CK_CUDA_THROW_(cudaHostAlloc((void**)&h_embeddingvector,
                                vec_length_per_batch * sizeof(float), cudaHostAllocPortable));
    CK_CUDA_THROW_(cudaHostAlloc((void**)&h_embeddingvector_gt,
                                vec_length_per_batch * sizeof(float), cudaHostAllocPortable));
    CK_CUDA_THROW_(cudaMalloc((void**)&d_embeddingvector,
                                vec_length_per_batch * sizeof(float)));
    std::vector<cudaStream_t> query_streams(num_emb_table);
    for (size_t i = 0; i < num_emb_table; i++) {
      CK_CUDA_THROW_(cudaStreamCreate(&query_streams[i]));
    }

    // Embedding cache lookup for multiple time
    MemoryBlock* memory_block;
    bool sync_flag;
    std::vector<size_t> vector_layout_offset(num_emb_table, 0);
    for (size_t i = 1; i < num_emb_table; i++) {
      vector_layout_offset[i] = vector_layout_offset[i-1] + max_batch_size * embedding_feature_num[i-1] * embedding_vec_size[i-1];
    }
    auto embedding_cache_look_up = [&]() {
      memory_block = NULL;
      while (memory_block == NULL) {
        memory_block = reinterpret_cast<MemoryBlock*>(embedding_cache->get_worker_space(
            infer_param.model_name, infer_param.device_id, CACHE_SPACE_TYPE::WORKER));
      }
      sync_flag = embedding_cache->look_up((void*)h_embeddingcolumns, h_embedding_offset,
                                                d_embeddingvector, memory_block, 
                                                query_streams, hit_rate_threshold);
      if (sync_flag) {
        embedding_cache->free_worker_space(memory_block);
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
            if (i == max_iterations-1) {
              vec_offset = vector_layout_offset[k] + j * embedding_feature_num[k] * embedding_vec_size[k] + l * embedding_vec_size[k]; 
              memcpy(h_embeddingvector_gt + vec_offset, 
                    vec_vec_per_table[k].data() + randomIndex * embedding_vec_size[k],
                    embedding_vec_size[k] * sizeof(float));
            }
          }
        }
      }
      embedding_cache_look_up();
    }

    // Embedding cache finalize, will join the asynchronous insert threads if there are any joinable threads
    embedding_cache->finalize();

    // Embedding cache lookup and check results with ground truth
    embedding_cache_look_up();
    CK_CUDA_THROW_(cudaMemcpy(h_embeddingvector, d_embeddingvector, 
                            vec_length_per_batch * sizeof(float), cudaMemcpyDeviceToHost));
    compare_lookup(h_embeddingvector_gt, h_embeddingvector, vec_length_per_batch, 0.01f);

    // Refresh embedding cache
    parameter_server->refresh_embedding_cache(model, 0);

    // Embedding cache lookup and check results with ground truth
    embedding_cache_look_up();
    CK_CUDA_THROW_(cudaMemcpy(h_embeddingvector, d_embeddingvector, 
                            vec_length_per_batch * sizeof(float), cudaMemcpyDeviceToHost));
    compare_lookup(h_embeddingvector_gt, h_embeddingvector, vec_length_per_batch, 0.01f);

    // Release CUDA resources
    for (size_t i = 0; i < num_emb_table; i++) {
      CK_CUDA_THROW_(cudaStreamSynchronize(query_streams[i]));
    }
    for (size_t i = 0; i < num_emb_table; i++) {
      CK_CUDA_THROW_(cudaStreamDestroy(query_streams[i]));
    }
    CK_CUDA_THROW_(cudaFreeHost(h_embeddingcolumns));
    CK_CUDA_THROW_(cudaFreeHost(h_embeddingvector));
    CK_CUDA_THROW_(cudaFreeHost(h_embeddingvector_gt));
    CK_CUDA_THROW_(cudaFree(d_embeddingvector));
}

} // end namespace

std::string dense_model{"/models/wdl/1/wdl_dense_20000.model"};
std::string network{"/models/wdl/1/wdl.json"};
std::string model_name = "wdl";
std::vector<std::string> sparse_models{"/models/wdl/1/wdl0_sparse_20000.model","/models/wdl/1/wdl1_sparse_20000.model"};
std::vector<size_t> embedding_vec_size_wdl{1, 16};
std::vector<size_t> embedding_featre_num_wdl{2, 26};
TEST(parameter_server, CPU_look_up_1x00x1) { parameter_server_test<long long>(network, model_name, dense_model, sparse_models, embedding_vec_size_wdl, embedding_featre_num_wdl, 1, 0.f, 1, DatabaseType_t::ParallelHashMap); }
TEST(parameter_server, CPU_look_up_1x10x1) { parameter_server_test<long long>(network, model_name, dense_model, sparse_models, embedding_vec_size_wdl, embedding_featre_num_wdl, 1, 1.f, 1, DatabaseType_t::ParallelHashMap); }
TEST(parameter_server, CPU_look_up_1x00x100) { parameter_server_test<long long>(network, model_name, dense_model, sparse_models, embedding_vec_size_wdl, embedding_featre_num_wdl, 1, 0.f, 100, DatabaseType_t::ParallelHashMap); }
TEST(parameter_server, CPU_look_up_1x10x100) { parameter_server_test<long long>(network, model_name, dense_model, sparse_models, embedding_vec_size_wdl, embedding_featre_num_wdl, 1, 1.f, 100, DatabaseType_t::ParallelHashMap); }
TEST(parameter_server, CPU_look_up_1024x00x100) { parameter_server_test<long long>(network, model_name, dense_model, sparse_models, embedding_vec_size_wdl, embedding_featre_num_wdl, 1024, 0.f, 100, DatabaseType_t::ParallelHashMap); }
TEST(parameter_server, CPU_look_up_1024x10x100) { parameter_server_test<long long>(network, model_name, dense_model, sparse_models, embedding_vec_size_wdl, embedding_featre_num_wdl, 1024, 1.f, 100, DatabaseType_t::ParallelHashMap); }
TEST(parameter_server, CPU_look_up_1024x05x100) { parameter_server_test<long long>(network, model_name, dense_model, sparse_models, embedding_vec_size_wdl, embedding_featre_num_wdl, 1024, 0.5f, 100, DatabaseType_t::ParallelHashMap); }
TEST(parameter_server, Rocksdb_look_up_1024x00x100) { parameter_server_test<long long>(network, model_name, dense_model, sparse_models, embedding_vec_size_wdl, embedding_featre_num_wdl, 1024, 0.f, 100, DatabaseType_t::RocksDB); }
TEST(parameter_server, Redis_look_up_1024x00x100) { parameter_server_test<long long>(network, model_name, dense_model, sparse_models, embedding_vec_size_wdl, embedding_featre_num_wdl, 1024, 0.f, 100, DatabaseType_t::RedisCluster); }