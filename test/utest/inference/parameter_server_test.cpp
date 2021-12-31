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
#include "inference/embedding_feature_combiner.hpp"
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
void validate_lookup_result_per_table(const std::string model_config_path, const InferenceParams inference_params,const std::vector<size_t> embedding_vec_size, HugectrUtility<TypeHashKey>* parameter_server_) {
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

    float* h_missing_emb_vec_;
    cudaMallocHost(&h_missing_emb_vec_, embedding_size * sizeof(float));
    for(int i = 0; i < 10; i++){
        int index=rand() % num_key;
        parameter_server_->look_up(key_vec.data()+index, 1, h_missing_emb_vec_, inference_params.model_name, j);
        EXPECT_EQ(vec_vec[index*embedding_size], h_missing_emb_vec_[0]);
        EXPECT_EQ(vec_vec[(index+1)*embedding_size-1], h_missing_emb_vec_[embedding_size-1]);
    }    
    cudaFreeHost(h_missing_emb_vec_);
  }
}


template <typename TypeHashKey>
void parameter_server_test(const std::string& config_file,const std::string& model, const std::string& dense_model, std::vector<std::string> sparse_models,
 const std::vector<size_t> embedding_vec_size, DatabaseType_t database_t=DatabaseType_t::ParallelHashMap) {
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
    InferenceParams infer_param(model, 1, 0.5, dense_model, sparse_models, 0, true, 0.8, false);
    infer_param.sparse_model_files.swap(sparse_models);
    infer_param.volatile_db=dis_database;
    infer_param.persistent_db=per_database;
    std::vector<InferenceParams> inference_params{infer_param};
    std::vector<std::string> model_config_path{config_file};
    HugectrUtility<TypeHashKey>* parameter_server = HugectrUtility<TypeHashKey>::Create_Parameter_Server(INFER_TYPE::TRITON, model_config_path, inference_params);
    validate_lookup_result_per_table<long long>(model_config_path[0], infer_param, embedding_vec_size,parameter_server);

}
}

std::string dense_model{"/models/wdl/1/wdl_dense_20000.model"};
std::string network{"/models/wdl/1/wdl.json"};
std::string model_name = "wdl";
std::vector<std::string> sparse_models{"/models/wdl/1/wdl0_sparse_20000.model","/models/wdl/1/wdl1_sparse_20000.model"};
std::vector<size_t> embedding_vec_size_wdl{1,16};
TEST(parameter_server, CPU_look_up) { parameter_server_test<long long>(network,model_name,dense_model, sparse_models, embedding_vec_size_wdl, DatabaseType_t::ParallelHashMap); }
TEST(parameter_server, Rocksdb_look_up) { parameter_server_test<long long>(network,model_name,dense_model, sparse_models, embedding_vec_size_wdl, DatabaseType_t::RocksDB); }
TEST(parameter_server, Redis_look_up) { parameter_server_test<long long>(network,model_name,dense_model, sparse_models, embedding_vec_size_wdl, DatabaseType_t::RedisCluster); }