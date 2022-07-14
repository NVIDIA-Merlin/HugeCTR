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

#pragma once
#include <cuda_runtime_api.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace HugeCTR {

enum INFER_TYPE { TRITON, OTHER };
enum CACHE_SPACE_TYPE { WORKER, REFRESHER };

enum class DatabaseType_t {
  Disabled,
  HashMap,
  ParallelHashMap,
  RedisCluster,
  RocksDB,
};
enum class DatabaseOverflowPolicy_t {
  EvictOldest,
  EvictRandom,
};
enum class UpdateSourceType_t {
  Null,
  KafkaMessageQueue,
};

constexpr const char* hctr_enum_to_c_str(const DatabaseType_t value) {
  // Remark: Dependent functions assume lower-case, and underscore separated.
  switch (value) {
    case DatabaseType_t::Disabled:
      return "disabled";
    case DatabaseType_t::HashMap:
      return "hash_map";
    case DatabaseType_t::ParallelHashMap:
      return "parallel_hash_map";
    case DatabaseType_t::RedisCluster:
      return "redis_cluster";
    case DatabaseType_t::RocksDB:
      return "rocks_db";
    default:
      return "<unknown DatabaseType_t value>";
  }
}
constexpr const char* hctr_enum_to_c_str(const DatabaseOverflowPolicy_t value) {
  // Remark: Dependent functions assume lower-case, and underscore separated.
  switch (value) {
    case DatabaseOverflowPolicy_t::EvictOldest:
      return "evict_oldest";
    case DatabaseOverflowPolicy_t::EvictRandom:
      return "evict_random";
    default:
      return "<unknown DatabaseOverflowPolicy_t value>";
  }
}
constexpr const char* hctr_enum_to_c_str(const UpdateSourceType_t value) {
  // Remark: Dependent functions assume lower-case, and underscore separated.
  switch (value) {
    case UpdateSourceType_t::Null:
      return "null";
    case UpdateSourceType_t::KafkaMessageQueue:
      return "kafka_message_queue";
    default:
      return "<unknown UpdateSourceType_t value>";
  }
}

std::ostream& operator<<(std::ostream& os, DatabaseType_t value);
std::ostream& operator<<(std::ostream& os, DatabaseOverflowPolicy_t value);
std::ostream& operator<<(std::ostream& os, UpdateSourceType_t value);
DatabaseType_t get_hps_database_type(const nlohmann::json& json, const std::string key);
UpdateSourceType_t get_hps_updatesource_type(const nlohmann::json& json, const std::string key);
DatabaseOverflowPolicy_t get_hps_overflow_policy(const nlohmann::json& json, const std::string key);

struct VolatileDatabaseParams {
  DatabaseType_t type;

  // Backend specific.
  std::string address;    // hostname[:port][[;hostname[:port]]...]
  std::string user_name;  // "default" = Standard user for Redis!
  std::string password;
  size_t num_partitions;
  size_t allocation_rate;  // Only used with HashMap type backends.
  size_t max_get_batch_size;
  size_t max_set_batch_size;

  // Overflow handling related.
  bool refresh_time_after_fetch;
  size_t overflow_margin;
  DatabaseOverflowPolicy_t overflow_policy;
  double overflow_resolution_target;

  // Chaching behavior related.
  double initial_cache_rate;
  bool cache_missed_embeddings;

  // Real-time update mechanism related.
  std::vector<std::string> update_filters;  // Should be a regex for Kafka.

  VolatileDatabaseParams(
      DatabaseType_t type = DatabaseType_t::ParallelHashMap,
      // Backend specific.
      const std::string& address = "127.0.0.1:7000", const std::string& user_name = "default",
      const std::string& password = "",
      size_t num_partitions = std::min(16u, std::thread::hardware_concurrency()),
      size_t allocation_rate = 256 * 1024 * 1024, size_t max_get_batch_size = 10'000,
      size_t max_set_batch_size = 10'000,
      // Overflow handling related.
      bool refresh_time_after_fetch = false,
      size_t overflow_margin = std::numeric_limits<size_t>::max(),
      DatabaseOverflowPolicy_t overflow_policy = DatabaseOverflowPolicy_t::EvictOldest,
      double overflow_resolution_target = 0.8,
      // Caching behavior related.
      double initial_cache_rate = 1.0, bool cache_missed_embeddings = false,
      // Real-time update mechanism related.
      const std::vector<std::string>& update_filters = {"^hps_.+$"});

  bool operator==(const VolatileDatabaseParams& p) const;
  bool operator!=(const VolatileDatabaseParams& p) const;
};

struct PersistentDatabaseParams {
  DatabaseType_t type;

  // Backend specific.
  std::string path;
  size_t num_threads;  // 16 = Default for RocksDB.
  bool read_only = false;
  size_t max_get_batch_size;
  size_t max_set_batch_size;

  // Real-time update mechanism related.
  std::vector<std::string> update_filters;  // Should be a regex for Kafka.

  PersistentDatabaseParams(DatabaseType_t type = DatabaseType_t::Disabled,
                           // Backend specific.
                           const std::string& path = std::filesystem::temp_directory_path() /
                                                     "rocksdb",
                           size_t num_threads = 16, bool read_only = false,
                           size_t max_get_batch_size = 10'000, size_t max_set_batch_size = 10'000,
                           // Real-time update mechanism related.
                           const std::vector<std::string>& update_filters = {"^hps_.+$"});

  bool operator==(const PersistentDatabaseParams& p) const;
  bool operator!=(const PersistentDatabaseParams& p) const;
};

struct UpdateSourceParams {
  UpdateSourceType_t type;

  // Backend specific.
  std::string brokers;  // Kafka: The IP[:Port][[;IP[:Port]]...] of the brokers.
  size_t metadata_refresh_interval_ms;
  size_t receive_buffer_size;
  size_t poll_timeout_ms;
  size_t max_batch_size;
  size_t failure_backoff_ms;
  size_t max_commit_interval;

  UpdateSourceParams(UpdateSourceType_t type = UpdateSourceType_t::Null,
                     // Backend specific.
                     const std::string& brokers = "127.0.0.1:9092",
                     size_t metadata_refresh_interval_ms = 30'000,
                     size_t receive_buffer_size = 256 * 1024, size_t poll_timeout_ms = 500,
                     size_t max_batch_size = 8 * 1024, size_t failure_backoff_ms = 50,
                     size_t max_commit_interval = 32);

  bool operator==(const UpdateSourceParams& p) const;
  bool operator!=(const UpdateSourceParams& p) const;
};

enum class PSUpdateSource_t { None, Kafka };

struct InferenceParams {
  std::string model_name;
  size_t max_batchsize;
  float hit_rate_threshold;
  std::string dense_model_file;
  std::vector<std::string> sparse_model_files;
  int device_id;
  bool use_gpu_embedding_cache;
  float cache_size_percentage;
  bool i64_input_key;
  bool use_mixed_precision;
  float scaler;
  bool use_algorithm_search;
  bool use_cuda_graph;
  int number_of_worker_buffers_in_pool;
  int number_of_refresh_buffers_in_pool;
  int thread_pool_size;
  float cache_refresh_percentage_per_iteration;
  std::vector<int> deployed_devices;
  std::vector<float> default_value_for_each_table;
  // Database backend.
  VolatileDatabaseParams volatile_db;
  PersistentDatabaseParams persistent_db;
  UpdateSourceParams update_source;
  // HPS required parameters
  int maxnum_des_feature_per_sample;
  float refresh_delay;
  float refresh_interval;
  std::vector<size_t> maxnum_catfeature_query_per_table_per_sample;
  std::vector<size_t> embedding_vecsize_per_table;
  std::vector<std::string> embedding_table_names;
  std::string network_file;
  size_t label_dim;
  size_t slot_num;

  InferenceParams(const std::string& model_name, size_t max_batchsize, float hit_rate_threshold,
                  const std::string& dense_model_file,
                  const std::vector<std::string>& sparse_model_files, int device_id,
                  bool use_gpu_embedding_cache, float cache_size_percentage, bool i64_input_key,
                  bool use_mixed_precision = false, float scaler = 1.0,
                  bool use_algorithm_search = true, bool use_cuda_graph = true,
                  int number_of_worker_buffers_in_pool = 1,
                  int number_of_refresh_buffers_in_pool = 1, int thread_pool_size = 16,
                  float cache_refresh_percentage_per_iteration = 0.1,
                  const std::vector<int>& deployed_devices = {0},
                  const std::vector<float>& default_value_for_each_table = {0.0f},
                  // Database backend.
                  const VolatileDatabaseParams& volatile_db = {},
                  const PersistentDatabaseParams& persistent_db = {},
                  const UpdateSourceParams& update_source = {},
                  // HPS required parameters
                  int maxnum_des_feature_per_sample = 26, float refresh_delay = 0.0f,
                  float refresh_interval = 0.0f,
                  const std::vector<size_t>& maxnum_catfeature_query_per_table_per_sample = {26},
                  const std::vector<size_t>& embedding_vecsize_per_table = {128},
                  const std::vector<std::string>& embedding_table_names = {""},
                  const std::string& network_file = "", size_t label_dim = 1, size_t slot_num = 10);
};

struct parameter_server_config {
  std::map<std::string, size_t> model_name_id_map_;
  // Each vector should have size of M(# of models), where each element in the vector should be a
  // vector with size E(# of embedding tables in that model)
  std::map<std::string, std::vector<std::string>>
      emb_file_name_;  // The sparse embedding table file path per embedding table per model
  std::map<std::string, std::vector<std::string>>
      emb_table_name_;  // The table name per embedding table per model
  std::map<std::string, std::vector<size_t>>
      embedding_vec_size_;  // The emb_vec_size per embedding table per model
  std::map<std::string, std::vector<size_t>>
      max_feature_num_per_sample_per_emb_table_;  // The max # of keys in each sample per table per
                                                  // model
  std::vector<std::vector<bool>>
      distributed_emb_;  // The file format flag per embedding table per model
  std::vector<std::vector<float>>
      default_emb_vec_value_;  // The defualt emb_vec value when emb_id cannot be found, per
                               // embedding table per model
  std::vector<InferenceParams>
      inference_params_array;  //// model configuration of all models deployed on HPS, e.g.,
                               ///{dcn_inferenceParamesStruct}

  // Database backend.
  VolatileDatabaseParams volatile_db;
  PersistentDatabaseParams persistent_db;
  UpdateSourceParams update_source;
  parameter_server_config(
      std::map<std::string, std::vector<std::string>> emb_table_name,
      std::map<std::string, std::vector<size_t>> embedding_vec_size,
      std::map<std::string, std::vector<size_t>> max_feature_num_per_sample_per_emb_table,
      const std::vector<InferenceParams>& inference_params_array,
      const VolatileDatabaseParams& volatile_db, const PersistentDatabaseParams& persistent_db,
      const UpdateSourceParams& update_source);
  parameter_server_config(const std::vector<std::string>& model_config_path_array,
                          const std::vector<InferenceParams>& inference_params_array);
  parameter_server_config(const std::string& hps_json_config_file);
  parameter_server_config(const char* hps_json_config_file);
  void init(const std::string& hps_json_config_file);
  std::optional<size_t> find_model_id(const std::string& model_name) const;
};

struct inference_memory_pool_size_config {
  std::map<std::string, int> num_woker_buffer_size_per_model;
  std::map<std::string, int> num_refresh_buffer_size_per_model;
};

struct embedding_cache_config {
  size_t num_emb_table_;  // # of embedding table in this model
  float cache_size_percentage_;
  float cache_refresh_percentage_per_iteration = 0.1;
  size_t num_set_in_refresh_workspace_;
  std::vector<float> default_value_for_each_table;
  std::string model_name_;        // Which model this cache belongs to
  int cuda_dev_id_;               // Which CUDA device this cache belongs to
  bool use_gpu_embedding_cache_;  // Whether enable GPU embedding cache or not
  // Each vector will have the size of E(# of embedding tables in the model)
  std::vector<size_t> embedding_vec_size_;  // # of float in emb_vec
  std::vector<size_t> num_set_in_cache_;    // # of cache set in the cache
  std::vector<std::string>
      embedding_table_name_;  // ## of embedding tables be cached by current embedding cache
  std::vector<size_t>
      max_query_len_per_emb_table_;  // The max # of embeddingcolumns each inference instance(batch)
                                     // will query from a embedding table
};

struct EmbeddingCacheWorkspace {
  std::vector<float*> h_missing_emb_vec_;  // The buffer to hold retrieved missing emb_vec from PS
                                           // on host, same size as d_missing_emb_vec_
  std::vector<void*> h_embeddingcolumns_;  // The embeeding keys buffer on host
  std::vector<void*> d_embeddingcolumns_;  // The embeeding keys buffer on device
  std::vector<uint64_t*>
      d_unique_output_index_;  // The output index for each emb_id in d_shuffled_embeddingcolumns_
                               // after unique on device, same size as h_embeddingcolumns
  std::vector<void*> d_unique_output_embeddingcolumns_;  // The output unique emb_id buffer on
                                                         // device, same size as h_embeddingcolumns
  std::vector<float*> d_hit_emb_vec_;  // The buffer to hold hit emb_vec on device, same size as
                                       // d_shuffled_embeddingoutputvector
  std::vector<void*>
      d_missing_embeddingcolumns_;  // The buffer to hold missing emb_id for each emb_table on
                                    // device, same size as h_embeddingcolumns
  std::vector<void*>
      h_missing_embeddingcolumns_;  // The buffer to hold missing emb_id for each emb_table on
                                    // host, same size as h_embeddingcolumns
  std::vector<uint64_t*> d_missing_index_;  // The buffer to hold missing index for each emb_table
                                            // on device, same size as h_embeddingcolumns
  std::vector<float*> d_missing_emb_vec_;   // The buffer to hold retrieved missing emb_vec on
                                            // device, same size as d_shuffled_embeddingoutputvector
  std::vector<void*> unique_op_obj_;  // The unique op object for to de-duplicate queried emb_id to
                                      // each emb_table, size = # of emb_table
  size_t* d_missing_length_;  // The buffer to hold missing length for each emb_table on device,
                              // size = # of emb_table
  size_t* h_missing_length_;  // The buffer to hold missing length for each emb_table on host, size
                              // = # of emb_table
  size_t* d_unique_length_;   // The # of emb_id after the unique operation for each emb_table on
                              // device, size = # of emb_table
  size_t* h_unique_length_;   // The # of emb_id after the unique operation for each emb_table on
                              // host, size = # of emb_table
  double* h_hit_rate_;        // The hit rate for each emb_table on host, size = # of emb_table
  bool use_gpu_embedding_cache_;  // whether to use gpu embedding cache
};

struct EmbeddingCacheRefreshspace {
  void* d_refresh_embeddingcolumns_;
  void* h_refresh_embeddingcolumns_;
  float* d_refresh_emb_vec_;
  float* h_refresh_emb_vec_;
  size_t* d_length_;
  size_t* h_length_;
};

void merge_emb_vec_async(float* d_vals_merge_dst_ptr, const float* d_vals_retrieved_ptr,
                         const uint64_t* d_missing_index_ptr, const size_t missing_len,
                         const size_t emb_vec_size, const size_t BLOCK_SIZE, cudaStream_t stream);

void fill_default_emb_vec_async(float* d_vals_merge_dst_ptr, const float default_emb_vec,
                                const uint64_t* d_missing_index_ptr, const size_t missing_len,
                                const size_t emb_vec_size, const size_t BLOCK_SIZE,
                                cudaStream_t stream);

void decompress_emb_vec_async(const float* d_unique_src_ptr, const uint64_t* d_unique_index_ptr,
                              float* d_decompress_dst_ptr, const size_t decompress_len,
                              const size_t emb_vec_size, const size_t BLOCK_SIZE,
                              cudaStream_t stream);

}  // namespace HugeCTR