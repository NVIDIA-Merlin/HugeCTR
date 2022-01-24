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

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
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
enum class DatabaseHashMapAlgorithm_t {
  STL,
  PHM,
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
constexpr const char* hctr_enum_to_c_str(const DatabaseHashMapAlgorithm_t value) {
  // Remark: Dependent functions assume lower-case, and underscore separated.
  switch (value) {
    case DatabaseHashMapAlgorithm_t::STL:
      return "stl";
    case DatabaseHashMapAlgorithm_t::PHM:
      return "phm";
    default:
      return "<unknown DatabaseHashMapAlgorithm_t value>";
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
std::ostream& operator<<(std::ostream& os, DatabaseHashMapAlgorithm_t value);
std::ostream& operator<<(std::ostream& os, DatabaseOverflowPolicy_t value);
std::ostream& operator<<(std::ostream& os, UpdateSourceType_t value);

struct VolatileDatabaseParams {
  DatabaseType_t type;

  // Backend specific.
  std::string address;    // hostname[:port][[;hostname[:port]]...]
  std::string user_name;  // "default" = Standard user for Redis!
  std::string password;
  DatabaseHashMapAlgorithm_t algorithm;  // Only used with HashMap type backends.
  size_t num_partitions;
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
      DatabaseHashMapAlgorithm_t algorithm = DatabaseHashMapAlgorithm_t::PHM,
      size_t num_partitions = std::min(16u, std::thread::hardware_concurrency()),
      size_t max_get_batch_size = 10'000, size_t max_set_batch_size = 10'000,
      // Overflow handling related.
      bool refresh_time_after_fetch = false,
      size_t overflow_margin = std::numeric_limits<size_t>::max(),
      DatabaseOverflowPolicy_t overflow_policy = DatabaseOverflowPolicy_t::EvictOldest,
      double overflow_resolution_target = 0.8,
      // Caching behavior related.
      double initial_cache_rate = 1.0, bool cache_missed_embeddings = false,
      // Real-time update mechanism related.
      const std::vector<std::string>& update_filters = {".+"});

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
                           const std::vector<std::string>& update_filters = {".+"});

  bool operator==(const PersistentDatabaseParams& p) const;
  bool operator!=(const PersistentDatabaseParams& p) const;
};

struct UpdateSourceParams {
  UpdateSourceType_t type;

  // Backend specific.
  std::string brokers;  // Kafka: The IP[:Port][[;IP[:Port]]...] of the brokers.
  size_t poll_timeout_ms;
  size_t max_receive_buffer_size;
  size_t max_batch_size;
  size_t failure_backoff_ms;

  UpdateSourceParams(UpdateSourceType_t type = UpdateSourceType_t::Null,
                     // Backend specific.
                     const std::string& brokers = "127.0.0.1:9092", size_t poll_timeout_ms = 500,
                     size_t max_receive_buffer_size = 2000, size_t max_batch_size = 1000,
                     size_t failure_backoff_ms = 50);

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
  float cache_refresh_percentage_per_iteration;
  std::vector<int> deployed_devices;
  std::vector<float> default_value_for_each_table;
  // Database backend.
  VolatileDatabaseParams volatile_db;
  PersistentDatabaseParams persistent_db;
  UpdateSourceParams update_source;

  InferenceParams(const std::string& model_name, size_t max_batchsize, float hit_rate_threshold,
                  const std::string& dense_model_file,
                  const std::vector<std::string>& sparse_model_files, int device_id,
                  bool use_gpu_embedding_cache, float cache_size_percentage, bool i64_input_key,
                  bool use_mixed_precision = false, float scaler = 1.0,
                  bool use_algorithm_search = true, bool use_cuda_graph = true,
                  int number_of_worker_buffers_in_pool = 2,
                  int number_of_refresh_buffers_in_pool = 1,
                  float cache_refresh_percentage_per_iteration = 0.1,
                  const std::vector<int>& deployed_devices = {0},
                  const std::vector<float>& default_value_for_each_table = {0.0f},
                  // Database backend.
                  const VolatileDatabaseParams& volatile_db = {},
                  const PersistentDatabaseParams& persistent_db = {},
                  const UpdateSourceParams& update_source = {});
};

struct parameter_server_config {
  std::map<std::string, size_t> model_name_id_map_;
  // Each vector should have size of M(# of models), where each element in the vector should be a
  // vector with size E(# of embedding tables in that model)
  std::map<std::string, std::vector<std::string>>
      emb_file_name_;  // The sparse embedding table file path per embedding table per model
  std::map<std::string, std::vector<std::string>>
      emb_table_name_;  // The table name per embedding table per model
  std::vector<std::vector<bool>>
      distributed_emb_;  // The file format flag per embedding table per model
  std::map<std::string, std::vector<size_t>>
      embedding_vec_size_;  // The emb_vec_size per embedding table per model
  std::vector<std::vector<float>>
      default_emb_vec_value_;  // The defualt emb_vec value when emb_id cannot be found, per
                               // embedding table per model

  std::optional<size_t> find_model_id(const std::string& model_name) const;
};

struct inference_memory_pool_size_config {
  std::map<std::string, int> num_woker_buffer_size_per_model;
  std::map<std::string, int> num_refresh_buffer_size_per_model;
};

}  // namespace HugeCTR
