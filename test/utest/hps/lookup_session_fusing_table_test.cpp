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
#include <hps/lookup_session.hpp>
#include <io/filesystem.hpp>
#include <utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

template <typename TypeHashKey>
void get_embedding_per_table(
    const std::vector<std::string>& sparse_files,
    const std::vector<size_t>& embedding_vecsize_per_table,
    std::map<size_t, std::map<TypeHashKey, std::vector<float>>>& embeddings_per_table) {
  // Create input file stream to read the embedding file
  for (unsigned int j = 0; j < sparse_files.size(); j++) {
    const std::string emb_file_prefix = sparse_files[j] + "/";
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
    const size_t embedding_size = embedding_vecsize_per_table[j];
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

    std::map<TypeHashKey, std::vector<float>> embeddings;
    size_t vec_vec_offset{0};
    for (auto key : key_vec) {
      std::vector<float> vec(embedding_vecsize_per_table[j]);
      memcpy(vec.data(), vec_vec.data() + vec_vec_offset,
             embedding_vecsize_per_table[j] * sizeof(float));
      vec_vec_offset += embedding_vecsize_per_table[j];
      embeddings[key] = vec;
    }
    embeddings_per_table[j] = embeddings;
    key_stream.close();
    vec_stream.close();
  }
}

template <typename TypeHashKey>
void lookup_from_ground_truth(float* h_vectors, const TypeHashKey* h_keys, size_t num_keys,
                              const std::map<TypeHashKey, std::vector<float>>& embeddings) {
  size_t offset{0};
  for (size_t i{0}; i < num_keys; ++i) {
    std::vector<float> vector_gt = embeddings.at(h_keys[i]);
    memcpy(h_vectors + offset, vector_gt.data(), vector_gt.size() * sizeof(float));
    offset += vector_gt.size();
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

void generate_config_file(const std::string& ps_config_file, bool i64_input_key,
                          const std::vector<std::string>& sparse_files,
                          const std::vector<size_t>& embedding_vecsize_per_table,
                          const std::vector<size_t>& maxnum_catfeature_query_per_table_per_sample) {
  EXPECT_EQ(sparse_files.size(), embedding_vecsize_per_table.size());
  EXPECT_EQ(sparse_files.size(), maxnum_catfeature_query_per_table_per_sample.size());

  nlohmann::json ps_config;
  ps_config["supportlonglong"] = i64_input_key;
  ps_config["fuse_embedding_table"] = true;

  nlohmann::json model_config;
  {
    size_t num_original_tables = sparse_files.size();
    std::vector<std::string> embedding_table_names(num_original_tables);
    for (size_t i{0}; i < num_original_tables; ++i) {
      embedding_table_names[i] = "table" + std::to_string(i);
    }
    model_config["num_of_worker_buffer_in_pool"] = num_original_tables;
    model_config["sparse_files"] = sparse_files;
    model_config["embedding_table_names"] = embedding_table_names;
    model_config["embedding_vecsize_per_table"] = embedding_vecsize_per_table;
    model_config["maxnum_catfeature_query_per_table_per_sample"] =
        maxnum_catfeature_query_per_table_per_sample;

    model_config["model"] = "test_fusing_table";
    model_config["default_value_for_each_table"] = std::vector<float>{0.f};
    model_config["deployed_device_list"] = std::vector<int>{0};
    model_config["max_batch_size"] = 256;
    model_config["cache_refresh_percentage_per_iteration"] = 1.0;
    model_config["hit_rate_threshold"] = 1.0;
    model_config["gpucacheper"] = 1.0;
    model_config["gpucache"] = true;
    model_config["use_static_table"] = false;
    model_config["use_context_stream"] = true;
  }

  ps_config["models"] = std::vector<nlohmann::json>{model_config};
  std::ofstream file_stream(ps_config_file);
  file_stream << std::setw(2) << ps_config;
  file_stream.close();
}

void generate_embedding_tables(const std::vector<std::string>& sparse_files,
                               const std::vector<size_t>& embedding_vecsize_per_table,
                               const std::vector<long long>& key_offset_per_table) {
  EXPECT_EQ(sparse_files.size(), embedding_vecsize_per_table.size());
  EXPECT_EQ(sparse_files.size() + 1, key_offset_per_table.size());
  size_t num_original_tables = sparse_files.size();

  for (size_t i{0}; i < num_original_tables; ++i) {
    auto fs = FileSystemBuilder::build_unique_by_path(sparse_files[i]);
    const std::string emb_file_prefix = sparse_files[i] + "/";
    const std::string key_file = emb_file_prefix + "key";
    const std::string vec_file = emb_file_prefix + "emb_vector";

    size_t num_keys = key_offset_per_table[i + 1] - key_offset_per_table[i];
    const size_t key_file_size_in_byte = num_keys * sizeof(long long);
    const size_t vec_file_size_in_byte = num_keys * embedding_vecsize_per_table[i] * sizeof(float);
    std::vector<long long> key_buff;
    std::vector<float> vec_buff;
    for (long long key{key_offset_per_table[i]}; key < key_offset_per_table[i + 1]; ++key) {
      key_buff.push_back(key);
      for (size_t vec_idx{0}; vec_idx < embedding_vecsize_per_table[i]; ++vec_idx) {
        float vec_value = float(rand() % 100) / 100.f;
        vec_buff.push_back(vec_value);
      }
    }
    fs->write(key_file, reinterpret_cast<char*>(key_buff.data()), key_file_size_in_byte, true);
    fs->write(vec_file, reinterpret_cast<char*>(vec_buff.data()), vec_file_size_in_byte, true);
  }
}

template <typename TypeHashKey>
void lookup_session_fusing_table_test(
    const std::string& ps_config_file, const std::vector<std::string>& sparse_files,
    const std::vector<long long>& key_offset_per_table,
    const std::vector<size_t>& embedding_vecsize_per_table,
    const std::vector<size_t>& maxnum_catfeature_query_per_table_per_sample,
    bool test_internal_multithreading) {
  bool i64_input_key = std::is_same<long long, TypeHashKey>::value;
  generate_embedding_tables(sparse_files, embedding_vecsize_per_table, key_offset_per_table);
  generate_config_file(ps_config_file, i64_input_key, sparse_files, embedding_vecsize_per_table,
                       maxnum_catfeature_query_per_table_per_sample);

  // Parse configuration file
  parameter_server_config ps_config{ps_config_file};
  auto inference_params = ps_config.inference_params_array[0];
  auto model_name = inference_params.model_name;
  auto device_id = inference_params.deployed_devices[0];
  size_t num_original_tables = sparse_files.size();

  // Create parameter server and get embedding cache
  auto parameter_server = HierParameterServerBase::create(ps_config);
  auto embedding_cache = parameter_server->get_embedding_cache(model_name, device_id);
  auto lookup_session = LookupSessionBase::create(inference_params, embedding_cache);

  // Read embedding files as ground truth
  std::map<size_t, std::map<TypeHashKey, std::vector<float>>> embeddings_per_table;
  get_embedding_per_table(sparse_files, embedding_vecsize_per_table, embeddings_per_table);

  // Allocate the resources for embedding cache lookup
  CudaDeviceContext context(device_id);
  std::vector<TypeHashKey*> h_keys_per_table(num_original_tables);
  std::vector<float*> h_vectors_per_table(num_original_tables);
  std::vector<float*> h_vectors_per_table_gt(num_original_tables);
  std::vector<TypeHashKey*> d_keys_per_table(num_original_tables);
  std::vector<float*> d_vectors_per_table(num_original_tables);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  for (size_t i{0}; i < num_original_tables; ++i) {
    // Over-allocated
    size_t max_num_key_per_sample = maxnum_catfeature_query_per_table_per_sample[i];
    size_t emb_vec_size = embedding_vecsize_per_table[i];
    HCTR_LIB_THROW(cudaMallocHost(
        reinterpret_cast<void**>(&h_keys_per_table[i]),
        inference_params.max_batchsize * max_num_key_per_sample * sizeof(TypeHashKey)));
    HCTR_LIB_THROW(cudaMallocHost(
        reinterpret_cast<void**>(&h_vectors_per_table[i]),
        inference_params.max_batchsize * max_num_key_per_sample * emb_vec_size * sizeof(float)));
    HCTR_LIB_THROW(cudaMallocHost(
        reinterpret_cast<void**>(&h_vectors_per_table_gt[i]),
        inference_params.max_batchsize * max_num_key_per_sample * emb_vec_size * sizeof(float)));
    HCTR_LIB_THROW(
        cudaMalloc(reinterpret_cast<void**>(&d_keys_per_table[i]),
                   inference_params.max_batchsize * max_num_key_per_sample * sizeof(TypeHashKey)));
    HCTR_LIB_THROW(cudaMalloc(
        reinterpret_cast<void**>(&d_vectors_per_table[i]),
        inference_params.max_batchsize * max_num_key_per_sample * emb_vec_size * sizeof(float)));
  }

  size_t max_iters = 100;
  for (size_t iter{0}; iter < max_iters; ++iter) {
    HCTR_LOG_S(INFO, WORLD) << "iteraton " << iter << std::endl;
    for (size_t i{0}; i < num_original_tables; ++i) {
      TypeHashKey* h_key = h_keys_per_table[i];
      TypeHashKey begin_value = static_cast<TypeHashKey>(key_offset_per_table[i]);
      TypeHashKey end_value = static_cast<TypeHashKey>(key_offset_per_table[i + 1]);
      for (size_t j{0};
           j < inference_params.max_batchsize * maxnum_catfeature_query_per_table_per_sample[i];
           ++j) {
        h_key[j] = rand() % (end_value - begin_value) + begin_value;
      }
      HCTR_LIB_THROW(cudaMemcpy(d_keys_per_table[i], h_keys_per_table[i],
                                inference_params.max_batchsize *
                                    maxnum_catfeature_query_per_table_per_sample[i] *
                                    sizeof(TypeHashKey),
                                cudaMemcpyHostToDevice));
    }

    if (test_internal_multithreading) {
      std::vector<size_t> num_keys_per_table;
      std::vector<const void*> d_keys_per_table_void;
      for (size_t i{0}; i < num_original_tables; ++i) {
        num_keys_per_table.emplace_back(inference_params.max_batchsize *
                                        maxnum_catfeature_query_per_table_per_sample[i]);
        d_keys_per_table_void.emplace_back(reinterpret_cast<const void*>(d_keys_per_table[i]));
      }
      lookup_session->lookup_from_device(d_keys_per_table_void, d_vectors_per_table,
                                         num_keys_per_table);
    } else {
      std::vector<std::thread> multi_lookup_threads;
      for (size_t i{0}; i < num_original_tables; ++i) {
        size_t num_keys =
            inference_params.max_batchsize * maxnum_catfeature_query_per_table_per_sample[i];
        multi_lookup_threads.emplace_back(std::thread([=] {
          lookup_session->lookup_from_device(d_keys_per_table[i], d_vectors_per_table[i], num_keys,
                                             i, stream);
        }));
      }
      for (size_t i{0}; i < num_original_tables; i++) {
        multi_lookup_threads[i].join();
      }
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    }

    for (size_t i{0}; i < num_original_tables; i++) {
      HCTR_LIB_THROW(cudaMemcpy(h_vectors_per_table[i], d_vectors_per_table[i],
                                inference_params.max_batchsize *
                                    maxnum_catfeature_query_per_table_per_sample[i] *
                                    embedding_vecsize_per_table[i] * sizeof(float),
                                cudaMemcpyDeviceToHost));
      lookup_from_ground_truth(
          h_vectors_per_table_gt[i], h_keys_per_table[i],
          inference_params.max_batchsize * maxnum_catfeature_query_per_table_per_sample[i],
          embeddings_per_table[i]);
      compare_lookup(h_vectors_per_table_gt[i], h_vectors_per_table[i],
                     inference_params.max_batchsize *
                         maxnum_catfeature_query_per_table_per_sample[i] *
                         embedding_vecsize_per_table[i],
                     0.001);
    }
  }

  // Release CUDA resources
  HCTR_LIB_THROW(cudaStreamDestroy(stream));
  for (size_t i{0}; i < num_original_tables; ++i) {
    HCTR_LIB_THROW(cudaFreeHost(h_keys_per_table[i]));
    HCTR_LIB_THROW(cudaFreeHost(h_vectors_per_table[i]));
    HCTR_LIB_THROW(cudaFree(d_keys_per_table[i]));
    HCTR_LIB_THROW(cudaFree(d_vectors_per_table[i]));
  }
}

}  // end namespace

TEST(lookup_session, unfused_table_1) {
  lookup_session_fusing_table_test<unsigned int>("fusion_utest.json", {"fusion_utest/table0"},
                                                 {0, 10000}, {128}, {10}, false);
}

TEST(lookup_session, unfused_table_1_i64) {
  lookup_session_fusing_table_test<long long>("fusion_utest.json", {"fusion_utest/table0"},
                                              {0, 10000}, {128}, {10}, false);
}

TEST(lookup_session, unfused_table_8) {
  lookup_session_fusing_table_test<unsigned int>(
      "fusion_utest.json",
      {"fusion_utest/table0", "fusion_utest/table1", "fusion_utest/table2", "fusion_utest/table3",
       "fusion_utest/table4", "fusion_utest/table5", "fusion_utest/table6", "fusion_utest/table7"},
      {0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000},
      {128, 32, 32, 128, 128, 128, 32, 128}, {10, 20, 10, 10, 30, 20, 10, 30}, false);
}

TEST(lookup_session, unfused_table_8_i64) {
  lookup_session_fusing_table_test<long long>(
      "fusion_utest.json",
      {"fusion_utest/table0", "fusion_utest/table1", "fusion_utest/table2", "fusion_utest/table3",
       "fusion_utest/table4", "fusion_utest/table5", "fusion_utest/table6", "fusion_utest/table7"},
      {0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000},
      {128, 32, 32, 128, 128, 128, 32, 128}, {10, 20, 10, 10, 30, 20, 10, 30}, false);
}

TEST(lookup_session, unfused_table_1_internal_multithreading) {
  lookup_session_fusing_table_test<unsigned int>("fusion_utest.json", {"fusion_utest/table0"},
                                                 {0, 10000}, {128}, {10}, true);
}

TEST(lookup_session, unfused_table_1_i64_internal_multithreading) {
  lookup_session_fusing_table_test<long long>("fusion_utest.json", {"fusion_utest/table0"},
                                              {0, 10000}, {128}, {10}, true);
}

TEST(lookup_session, unfused_table_8_internal_multithreading) {
  lookup_session_fusing_table_test<unsigned int>(
      "fusion_utest.json",
      {"fusion_utest/table0", "fusion_utest/table1", "fusion_utest/table2", "fusion_utest/table3",
       "fusion_utest/table4", "fusion_utest/table5", "fusion_utest/table6", "fusion_utest/table7"},
      {0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000},
      {128, 32, 32, 128, 128, 128, 32, 128}, {10, 20, 10, 10, 30, 20, 10, 30}, true);
}

TEST(lookup_session, unfused_table_8_i64_internal_multithreading) {
  lookup_session_fusing_table_test<long long>(
      "fusion_utest.json",
      {"fusion_utest/table0", "fusion_utest/table1", "fusion_utest/table2", "fusion_utest/table3",
       "fusion_utest/table4", "fusion_utest/table5", "fusion_utest/table6", "fusion_utest/table7"},
      {0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000},
      {128, 32, 32, 128, 128, 128, 32, 128}, {10, 20, 10, 10, 30, 20, 10, 30}, true);
}