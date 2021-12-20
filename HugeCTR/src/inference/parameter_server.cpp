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

#include <limits.h>

#include <chrono>
#include <cstdio>
#include <experimental/filesystem>
#include <inference/hash_map_backend.hpp>
#include <inference/kafka_message.hpp>
#include <inference/parameter_server.hpp>
#include <inference/redis_backend.hpp>
#include <inference/rocksdb_backend.hpp>
#include <regex>

namespace fs = std::experimental::filesystem;

namespace HugeCTR {

parameter_server_base::~parameter_server_base() {}

std::string parameter_server_base::make_tag_name(const std::string& model_name,
                                                 const std::string& embedding_table) {
  static std::regex syntax{"[a-zA-Z0-9_\\-]{1,120}"};
  HCTR_CHECK_HINT(std::regex_match(model_name, syntax), "The provided 'model_name' is invalid!");
  HCTR_CHECK_HINT(std::regex_match(embedding_table, syntax),
                  "The provided 'embedding_table' is invalid!");

  std::ostringstream ss;
  ss << PS_EMBEDDING_TABLE_TAG_PREFIX << '.';
  ss << model_name << '.' << embedding_table;
  return ss.str();
}

template <typename TypeHashKey>
parameter_server<TypeHashKey>::parameter_server(
    const std::string& framework_name, const std::vector<std::string>& model_config_path,
    std::vector<InferenceParams>& inference_params_array) {
  // Store the configuration
  framework_name_ = framework_name;
  if (model_config_path.size() != inference_params_array.size()) {
    CK_THROW_(Error_t::WrongInput, "Wrong input: The size of input args are not consistent.");
  }

  parse_networks_per_model_(model_config_path, inference_params_array);

  if (ps_config_.distributed_emb_.size() != model_config_path.size() ||
      ps_config_.embedding_vec_size_.size() != model_config_path.size() ||
      ps_config_.default_emb_vec_value_.size() != model_config_path.size()) {
    CK_THROW_(Error_t::WrongInput,
              "Wrong input: The size of parameter server parameters are not correct.");
  }

  // Connect to CPU memory database.
  {
    const auto& conf = inference_params_array[0].cpu_memory_db;
    switch (conf.type) {
      case DatabaseType_t::Disabled:
        break;  // No CPU memory database.
      case DatabaseType_t::HashMap:
        HCTR_LOG(INFO, WORLD, "Creating HashMap CPU database backend...\n");
        if (conf.num_partitions > 1) {
          HCTR_LOG(WARNING, WORLD,
                   "Setting 'num_partitions' = %d is not supported by the non-parallelized "
                   "HashTable backend and will be ignored.\n",
                   conf.num_partitions);
        }
        switch (conf.algorithm) {
          case CPUMemoryHashMapAlgorithm_t::STL:
            cpu_memory_db_ = std::make_shared<HCTR_DB_HASH_MAP_STL_(HashMapBackend, TypeHashKey)>(
                conf.overflow_margin, conf.overflow_policy, conf.overflow_resolution_target);
            break;
          case CPUMemoryHashMapAlgorithm_t::PHM:
            cpu_memory_db_ = std::make_shared<HCTR_DB_HASH_MAP_PHM_(HashMapBackend, TypeHashKey)>(
                conf.overflow_margin, conf.overflow_policy, conf.overflow_resolution_target);
            break;
          default:
            HCTR_DIE("Selected algorithm (cpu_memory_db.algorithm = %d) is not supported!",
                     conf.type);
            break;
        }
        break;
      case DatabaseType_t::ParallelHashMap:
        HCTR_LOG(INFO, WORLD, "Creating ParallelHashMap CPU database backend...\n");
        if (conf.num_partitions < 2) {
          HCTR_LOG(WARNING, WORLD,
                   "ParallelHashMap configured with 'num_partitions' = %d, which will likely "
                   "result in poor performance. Consider using 'HashMap' backend.\n",
                   conf.num_partitions);
        }
        switch (conf.algorithm) {
          case CPUMemoryHashMapAlgorithm_t::STL:
            cpu_memory_db_ =
                std::make_shared<HCTR_DB_HASH_MAP_STL_(ParallelHashMapBackend, TypeHashKey)>(
                    conf.num_partitions, conf.overflow_margin, conf.overflow_policy,
                    conf.overflow_resolution_target);
            break;
          case CPUMemoryHashMapAlgorithm_t::PHM:
            cpu_memory_db_ =
                std::make_shared<HCTR_DB_HASH_MAP_PHM_(ParallelHashMapBackend, TypeHashKey)>(
                    conf.num_partitions, conf.overflow_margin, conf.overflow_policy,
                    conf.overflow_resolution_target);
            break;
          default:
            HCTR_DIE("Selected algorithm (cpu_memory_db.algorithm = %d) is not supported!",
                     conf.type);
            break;
        }
        break;
      default:
        HCTR_DIE("Selected backend (cpu_memory_db.type = %d) is not supported!", conf.type);
        break;
    }
    cpu_memory_db_cache_rate_ = conf.initial_cache_rate;
  }

  // Connect to distributed database.
  {
    const auto& conf = inference_params_array[0].distributed_db;
    switch (conf.type) {
      case DatabaseType_t::Disabled:
        break;  // No distributed database.
      case DatabaseType_t::RedisCluster:
        HCTR_LOG(INFO, WORLD, "Creating RedisCluster backend...\n");
        distributed_db_ = std::make_shared<RedisClusterBackend<TypeHashKey>>(
            conf.address, conf.user_name, conf.password, conf.num_partitions,
            conf.max_get_batch_size, conf.max_set_batch_size, conf.overflow_margin,
            conf.overflow_policy, conf.overflow_resolution_target);
        break;
      default:
        HCTR_DIE("Selected backend (distributed_db.type = %d) is not supported!", conf.type);
        break;
    }
    distributed_db_cache_rate_ = conf.initial_cache_rate;
  }

  // Connect to persistent database.
  {
    const auto& conf = inference_params_array[0].persistent_db;
    switch (conf.type) {
      case DatabaseType_t::Disabled:
        break;  // No persistent database.
      case DatabaseType_t::RocksDB:
        HCTR_LOG(INFO, WORLD, "Creating RocksDB backend...\n");
        persistent_db_ = std::make_shared<RocksDBBackend<TypeHashKey>>(
            conf.path, conf.num_threads, conf.read_only, conf.max_get_batch_size,
            conf.max_set_batch_size);
        break;
      default:
        HCTR_DIE("Selected backend (persistent_db.type = %d) is not supported!", conf.type);
        break;
    }
  }

  // Load embeddings for each embedding table from each model
  for (unsigned int i = 0; i < model_config_path.size(); i++) {
    update_database_per_model(model_config_path[i], inference_params_array[i]);
  }

  // Initilize embedding cache for each embedding table of each model
  for (unsigned int i = 0; i < model_config_path.size(); i++) {
    //*********
    // The operation here is just to keep the logic of device_id in the python api
    // unchanged
    //*********
    if (inference_params_array[i].deployed_devices.empty()) {
      CK_THROW_(Error_t::WrongInput, "The list of deployed devices is empty.");
    }
    if (std::find(inference_params_array[i].deployed_devices.begin(),
                  inference_params_array[i].deployed_devices.end(),
                  inference_params_array[i].device_id) ==
        inference_params_array[i].deployed_devices.end()) {
      CK_THROW_(Error_t::WrongInput, "The device id is not in the list of deployed devices.");
    }
    std::map<int64_t, std::shared_ptr<embedding_interface>> embedding_cache_map;
    for (auto device_id : inference_params_array[i].deployed_devices) {
      HCTR_LOG(INFO, WORLD, "Create embedding cache in device %d.\n", device_id);
      inference_params_array[i].device_id = device_id;
      embedding_cache_map[device_id] =
          std::shared_ptr<embedding_interface>(embedding_interface::Create_Embedding_Cache(
              model_config_path[i], inference_params_array[i], this));
    }
    model_cache_map[inference_params_array[i].model_name] = embedding_cache_map;
    memory_pool_config.num_woker_buffer_size_per_model[inference_params_array[i].model_name] =
        inference_params_array[i].number_of_worker_buffers_in_pool;
    memory_pool_config.num_refresh_buffer_size_per_model[inference_params_array[i].model_name] =
        inference_params_array[i].number_of_refresh_buffers_in_pool;
  }

  // Populate DB stack.
  if (cpu_memory_db_) {
    db_stack_.push_back(cpu_memory_db_);
  }
  if (distributed_db_) {
    db_stack_.push_back(distributed_db_);
  }
  if (persistent_db_) {
    db_stack_.push_back(persistent_db_);
  }

  bufferpool.reset(new ManagerPool(model_cache_map, memory_pool_config));
}

template <typename TypeHashKey>
void parameter_server<TypeHashKey>::parse_networks_per_model_(
    const std::vector<std::string>& model_config_path,
    std::vector<InferenceParams>& inference_params_array) {
  // Initialize <model_name, id> map
  for (unsigned int i = 0; i < inference_params_array.size(); i++) {
    ps_config_.model_name_id_map_.emplace(inference_params_array[i].model_name, (size_t)i);
  }

  // Initialize for each model
  for (unsigned int i = 0; i < model_config_path.size(); i++) {
    HCTR_THROW_IF(
        inference_params_array[i].cpu_memory_db != inference_params_array[0].cpu_memory_db ||
            inference_params_array[i].distributed_db != inference_params_array[0].distributed_db ||
            inference_params_array[i].persistent_db != inference_params_array[0].persistent_db,
        Error_t::WrongInput,
        "Inconsistent database setup. HugeCTR paramter server does currently not support hybrid "
        "database deployment.");

    // Open model config file and input model json config
    nlohmann::json model_config(read_json_file(model_config_path[i]));

    // Read inference config
    std::vector<std::string> emb_file_path;
    if (inference_params_array[i].sparse_model_files.size() > 1) {
      for (unsigned int j = 0; j < inference_params_array[i].sparse_model_files.size(); j++) {
        emb_file_path.emplace_back(inference_params_array[i].sparse_model_files[j]);
      }
    } else {
      emb_file_path.emplace_back(inference_params_array[i].sparse_model_files[0]);
    }
    ps_config_.emb_file_name_[inference_params_array[i].model_name] = (emb_file_path);

    // Read embedding layer config
    const nlohmann::json& j_layers = get_json(model_config, "layers");
    std::vector<bool> distributed_emb;
    std::vector<size_t> embedding_vec_size;
    std::vector<float> default_emb_vec_value;
    std::vector<std::string> emb_table_name;
    // Search for all embedding layers
    for (unsigned int j = 1; j < j_layers.size(); j++) {
      const nlohmann::json& j_single_layer = j_layers[j];
      std::string embedding_type = get_value_from_json<std::string>(j_single_layer, "type");
      if (embedding_type.compare("DistributedSlotSparseEmbeddingHash") == 0) {
        distributed_emb.emplace_back(true);
        // parse embedding table name from network json file
        emb_table_name.emplace_back(get_value_from_json<std::string>(j_single_layer, "top"));
        const nlohmann::json& embedding_hparam =
            get_json(j_single_layer, "sparse_embedding_hparam");
        embedding_vec_size.emplace_back(
            get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size"));
        default_emb_vec_value.emplace_back(
            get_value_from_json_soft<float>(embedding_hparam, "default_emb_vec_value", 0.0f));
      } else if (embedding_type.compare("LocalizedSlotSparseEmbeddingHash") == 0 ||
                 embedding_type.compare("LocalizedSlotSparseEmbeddingOneHot") == 0) {
        distributed_emb.emplace_back(false);
        emb_table_name.emplace_back(get_value_from_json<std::string>(j_single_layer, "top"));
        const nlohmann::json& embedding_hparam =
            get_json(j_single_layer, "sparse_embedding_hparam");
        embedding_vec_size.emplace_back(
            get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size"));
        default_emb_vec_value.emplace_back(
            get_value_from_json_soft<float>(embedding_hparam, "default_emb_vec_value", 0.0f));
      } else {
        break;
      }
    }
    ps_config_.distributed_emb_.emplace_back(distributed_emb);
    ps_config_.emb_table_name_[inference_params_array[i].model_name] = emb_table_name;
    ps_config_.embedding_vec_size_[inference_params_array[i].model_name] = embedding_vec_size;
    ps_config_.default_emb_vec_value_.emplace_back(
        inference_params_array[i].default_value_for_each_table);
  }
}

template <typename TypeHashKey>
void parameter_server<TypeHashKey>::update_database_per_model(
    const std::string& model_config_path, const InferenceParams& inference_params) {
  // Create input file stream to read the embedding file
  for (unsigned int j = 0; j < inference_params.sparse_model_files.size(); j++) {
    if (ps_config_.embedding_vec_size_[inference_params.model_name].size() !=
        inference_params.sparse_model_files.size()) {
      CK_THROW_(Error_t::WrongInput,
                "Wrong input: The number of embedding tables in network json file for model " +
                    inference_params.model_name +
                    " doesn't match the size of 'sparse_model_files' in configuration.");
    }
    const std::string emb_file_prefix = inference_params.sparse_model_files[j] + "/";
    const std::string key_file = emb_file_prefix + "key";
    const std::string vec_file = emb_file_prefix + "emb_vector";
    std::ifstream key_stream(key_file);
    std::ifstream vec_stream(vec_file);
    // Check if file is opened successfully
    if (!key_stream.is_open() || !vec_stream.is_open()) {
      CK_THROW_(Error_t::WrongInput, "Error: embeddings file not open for reading");
    }
    const size_t key_file_size_in_byte = fs::file_size(key_file);
    const size_t vec_file_size_in_byte = fs::file_size(vec_file);

    const size_t key_size_in_byte = sizeof(long long);
    const size_t embedding_size = ps_config_.embedding_vec_size_[inference_params.model_name][j];
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

    const size_t cpu_cache_amount = std::min(
        hctr_safe_cast<size_t>(cpu_memory_db_cache_rate_ * static_cast<double>(num_key) + 0.5),
        num_key);
    const size_t dist_cache_amount = std::min(
        hctr_safe_cast<size_t>(distributed_db_cache_rate_ * static_cast<double>(num_key) + 0.5),
        num_key);

    std::string current_emb_table_name = ps_config_.emb_table_name_[inference_params.model_name][j];
    const std::string tag_name = make_tag_name(inference_params.model_name, current_emb_table_name);

    // Populate databases. CPU is static.
    if (cpu_memory_db_) {
      HCTR_CHECK(cpu_memory_db_->insert(tag_name, cpu_cache_amount, key_vec.data(),
                                        reinterpret_cast<const char*>(vec_vec.data()),
                                        embedding_size * sizeof(float)));
      HCTR_LOG(INFO, WORLD, "Table: %s; cached %d / %d embeddings in CPU memory database!\n",
               tag_name.c_str(), cpu_cache_amount, num_key);
    }

    // Distributed could be static but does not have to be.
    if (distributed_db_) {
      HCTR_CHECK(distributed_db_->insert(tag_name, dist_cache_amount, key_vec.data(),
                                         reinterpret_cast<const char*>(vec_vec.data()),
                                         embedding_size * sizeof(float)));
      HCTR_LOG(INFO, WORLD, "Table: %s; cached %d / %d embeddings in distributed database!\n",
               tag_name.c_str(), dist_cache_amount, num_key);
    }

    // Persistent database - by definition - always gets all keys.
    if (persistent_db_) {
      HCTR_CHECK(persistent_db_->insert(tag_name, num_key, key_vec.data(),
                                        reinterpret_cast<const char*>(vec_vec.data()),
                                        embedding_size * sizeof(float)));
      HCTR_LOG(INFO, WORLD, "Table: %s; cached %d embeddings in persistent database!\n",
               tag_name.c_str(), num_key);
    }
  }

  // Connect to online update service (if configured).
  // TODO: Maybe need to change the location where this is initialized.
  const std::string kafka_group_prefix = "hctr_ps.";

  auto kafka_prepare_filter = [](const std::string& s) -> std::string {
    std::ostringstream ss;
    ss << '^' << PS_EMBEDDING_TABLE_TAG_PREFIX << "\\." << s << "\\..+$";
    return ss.str();
  };

  char host_name[HOST_NAME_MAX + 1];
  HCTR_CHECK_HINT(!gethostname(host_name, sizeof(host_name)), "Unable to determine hostname.\n");

  switch (inference_params.update_source.type) {
    case UpdateSourceType_t::Null:
      break;  // Disabled
    case UpdateSourceType_t::KafkaMessageQueue:
      // CPU memory updates.
      if (cpu_memory_db_ && !inference_params.cpu_memory_db.update_filters.empty()) {
        std::vector<std::string> tag_filters;
        std::transform(inference_params.cpu_memory_db.update_filters.begin(),
                       inference_params.cpu_memory_db.update_filters.end(),
                       std::back_inserter(tag_filters), kafka_prepare_filter);
        cpu_memory_db_source_ = std::make_unique<KafkaMessageSource<TypeHashKey>>(
            inference_params.update_source.brokers, kafka_group_prefix + "cpu_memory." + host_name,
            tag_filters, inference_params.update_source.poll_timeout_ms,
            inference_params.update_source.max_receive_buffer_size,
            inference_params.update_source.max_batch_size,
            inference_params.update_source.failure_backoff_ms);
      }
      // Distributed updates.
      if (distributed_db_ && !inference_params.distributed_db.update_filters.empty()) {
        std::vector<std::string> tag_filters;
        std::transform(inference_params.distributed_db.update_filters.begin(),
                       inference_params.distributed_db.update_filters.end(),
                       std::back_inserter(tag_filters), kafka_prepare_filter);
        distributed_db_source_ = std::make_unique<KafkaMessageSource<TypeHashKey>>(
            inference_params.update_source.brokers, kafka_group_prefix + "distributed", tag_filters,
            inference_params.update_source.poll_timeout_ms,
            inference_params.update_source.max_receive_buffer_size,
            inference_params.update_source.max_batch_size,
            inference_params.update_source.failure_backoff_ms);
      }
      // Persistent updates.
      if (persistent_db_ && !inference_params.persistent_db.update_filters.empty()) {
        std::vector<std::string> tag_filters;
        std::transform(inference_params.persistent_db.update_filters.begin(),
                       inference_params.persistent_db.update_filters.end(),
                       std::back_inserter(tag_filters), kafka_prepare_filter);
        persistent_db_source_ = std::make_unique<KafkaMessageSource<TypeHashKey>>(
            inference_params.update_source.brokers, kafka_group_prefix + "persistent." + host_name,
            tag_filters, inference_params.update_source.poll_timeout_ms,
            inference_params.update_source.max_receive_buffer_size,
            inference_params.update_source.max_batch_size,
            inference_params.update_source.failure_backoff_ms);
      }
      break;
    default:
      HCTR_DIE("Unsupported update source!\n");
      break;
  }

  HCTR_LOG(DEBUG, WORLD, "Real-time subscribers created!\n");

  auto insert_fn = [&](const std::shared_ptr<DatabaseBackend<TypeHashKey>>& db,
                       const std::string& tag, const size_t num_pairs, const TypeHashKey* keys,
                       const char* values, const size_t value_size) -> bool {
    HCTR_LOG(DEBUG, WORLD,
             "Database \"%s\" update for tag: \"%s\", num_pairs: %d, value_size: %d bytes\n",
             db->get_name(), tag.c_str(), num_pairs, value_size);
    return db->insert(tag, num_pairs, keys, values, value_size);
  };

  // TODO: Update embedding cache!

  // Turn on background updates.
  if (cpu_memory_db_source_) {
    cpu_memory_db_source_->enable([&](const std::string& tag, const size_t num_pairs,
                                      const TypeHashKey* keys, const char* values,
                                      const size_t value_size) -> bool {
      // Try a search. If we can find the value, override it. If not, do nothing.
      return insert_fn(cpu_memory_db_, tag, num_pairs, keys, values, value_size);
    });
  }

  if (distributed_db_source_) {
    distributed_db_source_->enable([&](const std::string& tag, const size_t num_pairs,
                                       const TypeHashKey* keys, const char* values,
                                       const size_t value_size) -> bool {
      // Try a search. If we can find the value, override it. If not, do nothing.
      return insert_fn(distributed_db_, tag, num_pairs, keys, values, value_size);
    });
  }

  if (persistent_db_source_) {
    persistent_db_source_->enable([&](const std::string& tag, const size_t num_pairs,
                                      const TypeHashKey* keys, const char* values,
                                      const size_t value_size) -> bool {
      // For persistent, we always insert.
      return insert_fn(persistent_db_, tag, num_pairs, keys, values, value_size);
    });
  }
}

template <typename TypeHashKey>
parameter_server<TypeHashKey>::~parameter_server() {
  for (auto it = model_cache_map.begin(); it != model_cache_map.end(); it++) {
    for (auto& v : it->second) {
      v.second->finalize();
    }
  }
  bufferpool->DestoryManagerPool();
}

template <typename TypeHashKey>
void* parameter_server<TypeHashKey>::ApplyBuffer(const std::string& modelname, int deviceid,
                                                 CACHE_SPACE_TYPE cache_type) {
  return bufferpool->AllocBuffer(modelname, deviceid, cache_type);
}

template <typename TypeHashKey>
void parameter_server<TypeHashKey>::FreeBuffer(void* p) {
  bufferpool->FreeBuffer(p);
  return;
}

template <typename TypeHashKey>
std::shared_ptr<embedding_interface> parameter_server<TypeHashKey>::GetEmbeddingCache(
    const std::string& model_name, int device_id) {
  auto it = model_cache_map.find(model_name);
  if (it == model_cache_map.end()) {
    CK_THROW_(Error_t::WrongInput, "No such model: " + model_name);
  } else {
    auto f = it->second.find(device_id);
    if (f == it->second.end()) {
      CK_THROW_(Error_t::WrongInput, "No embedding cache on device " + std::to_string(device_id) +
                                         " for model " + model_name);
    }
  }
  return model_cache_map[model_name][device_id];
}

template <typename TypeHashKey>
void parameter_server<TypeHashKey>::look_up(const TypeHashKey* h_embeddingcolumns, size_t length,
                                            float* h_embeddingoutputvector,
                                            const std::string& model_name,
                                            size_t embedding_table_id) {
  if (!length) {
    return;
  }

  const auto start_time = std::chrono::high_resolution_clock::now();
  const auto& model_id = ps_config_.find_model_id(model_name);
  HCTR_CHECK_HINT(
      static_cast<bool>(model_id),
      "Error: parameter server unknown model name. Note that this error will also come "
      "out with "
      "using Triton LOAD/UNLOAD APIs which haven't been supported in HugeCTR backend.\n");

  const size_t embedding_size = ps_config_.embedding_vec_size_[model_name][embedding_table_id];
  const std::string& tag_name =
      make_tag_name(model_name, ps_config_.emb_table_name_[model_name][embedding_table_id]);
  float defalut_vec_value =
      ps_config_
          .default_emb_vec_value_[ps_config_.model_name_id_map_[model_name]][embedding_table_id];
  HCTR_LOG(INFO, WORLD, "Looking up %d embeddings (each with %d values)...\n", length,
           embedding_size);

  size_t hit_count = 0;
  auto db = db_stack_.begin();

  switch (db_stack_.size()) {
    case 0: {
      // Everything is default.
      std::fill_n(h_embeddingoutputvector, length * embedding_size, defalut_vec_value);
      HCTR_LOG(INFO, WORLD, "No database. All embeddings set to default.\n");
      break;
    }
    case 1: {
      // Query database, and fill in default value for unavailable embeddings.
      MissingKeyCallback fill_default_fn = [&](size_t index) -> void {
        std::fill_n(&h_embeddingoutputvector[index * embedding_size], embedding_size,
                    defalut_vec_value);
      };
      hit_count += (*db)->fetch(tag_name, length, h_embeddingcolumns,
                                reinterpret_cast<char*>(h_embeddingoutputvector),
                                embedding_size * sizeof(float), fill_default_fn);
      HCTR_LOG(DEBUG, WORLD, "%s: %d hits, %d missing!\n", (*db)->get_name(), hit_count,
               length - hit_count);
      break;
    }
    default: {
      // Layer 0: Do a sequential lookup. Remember missing keys.
      std::vector<size_t> indices;
      std::vector<size_t> missing;
      std::mutex missing_guard;

      MissingKeyCallback record_missing_fn = [&missing, &missing_guard](size_t index) -> void {
        std::lock_guard<std::mutex> lock(missing_guard);
        missing.push_back(index);
      };

      hit_count += (*db)->fetch(tag_name, length, h_embeddingcolumns,
                                reinterpret_cast<char*>(h_embeddingoutputvector),
                                embedding_size * sizeof(float), record_missing_fn);
      HCTR_LOG(DEBUG, WORLD, "%s: %d hits, %d missing!\n", (*db)->get_name(), hit_count,
               missing.size());
      db++;

      // Layers 1 thru N-2: Do a sparse lookup. Remember missing keys
      for (size_t i = 2; i < db_stack_.size(); i++, db++) {
        indices.clear();
        indices.swap(missing);
        hit_count += (*db)->fetch(tag_name, indices.size(), indices.data(), h_embeddingcolumns,
                                  reinterpret_cast<char*>(h_embeddingoutputvector),
                                  embedding_size * sizeof(float), record_missing_fn);
        HCTR_LOG(DEBUG, WORLD, "%s: %d hits, %d missing!\n", (*db)->get_name(), hit_count,
                 missing.size());
      }

      // Layer N-1: Do a sparse lookup. Fill in default values.
      MissingKeyCallback fill_default_fn = [&](size_t index) -> void {
        std::fill_n(&h_embeddingoutputvector[index * embedding_size], embedding_size,
                    defalut_vec_value);
      };
      hit_count += (*db)->fetch(tag_name, missing.size(), missing.data(), h_embeddingcolumns,
                                reinterpret_cast<char*>(h_embeddingoutputvector),
                                embedding_size * sizeof(float), fill_default_fn);
      HCTR_LOG(DEBUG, WORLD, "%s: %d hits, %d missing!\n", (*db)->get_name(), hit_count,
               length - hit_count);
    }
  }

  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  HCTR_LOG(INFO, WORLD, "Parameter server lookup of %d / %d embeddings took %d us.\n", hit_count,
           length, duration.count());
}

template <typename TypeHashKey>
void parameter_server<TypeHashKey>::refresh_embedding_cache(const std::string& model_name,
                                                            int device_id) {
  HCTR_LOG(INFO, WORLD, "*****Refresh embedding cache of model %s on device %d*****\n",
           model_name.c_str(), device_id);
  HugeCTR::Timer timer_refresh;
  timer_refresh.start();
  std::shared_ptr<embedding_interface> embedding_cache = GetEmbeddingCache(model_name, device_id);
  if (!embedding_cache->use_gpu_embedding_cache()) {
    HCTR_LOG(WARNING, WORLD, "GPU embedding cache is not enabled and cannot be refreshed!\n");
    return;
  }
  std::vector<cudaStream_t> streams = embedding_cache->get_refresh_streams();
  embedding_cache_config cache_config = embedding_cache->get_cache_config();
  // apply the memory block for embedding cache refresh workspace
  MemoryBlock* memory_block = NULL;
  while (memory_block == NULL) {
    memory_block = reinterpret_cast<struct MemoryBlock*>(
        this->ApplyBuffer(model_name, device_id, CACHE_SPACE_TYPE::REFRESHER));
  }
  embedding_cache_refreshspace refreshspace_handler = memory_block->refresh_buffer;
  // Refresh the embedding cache for each table
  size_t stride_set = cache_config.num_set_in_refresh_workspace_;
  HugeCTR::Timer timer;
  for (unsigned int i = 0; i < cache_config.num_emb_table_; i++) {
    for (unsigned int idx_set = 0; idx_set < cache_config.num_set_in_cache_[i];
         idx_set += stride_set) {
      size_t end_idx = (idx_set + stride_set > cache_config.num_set_in_cache_[i])
                           ? cache_config.num_set_in_cache_[i]
                           : idx_set + stride_set;
      timer.start();
      embedding_cache->Dump(i, refreshspace_handler.d_refresh_embeddingcolumns_,
                            refreshspace_handler.d_length_, idx_set, end_idx, streams[i]);

      CK_CUDA_THROW_(cudaMemcpyAsync(refreshspace_handler.h_length_, refreshspace_handler.d_length_,
                                     sizeof(size_t), cudaMemcpyDeviceToHost, streams[i]));
      CK_CUDA_THROW_(cudaStreamSynchronize(streams[i]));
      CK_CUDA_THROW_(cudaMemcpyAsync(refreshspace_handler.h_refresh_embeddingcolumns_,
                                     refreshspace_handler.d_refresh_embeddingcolumns_,
                                     *refreshspace_handler.h_length_ * sizeof(TypeHashKey),
                                     cudaMemcpyDeviceToHost, streams[i]));
      CK_CUDA_THROW_(cudaStreamSynchronize(streams[i]));
      timer.stop();
      MESSAGE_("Embedding Cache dumping the number of " + std::to_string(stride_set) +
               " sets takes: " + std::to_string(timer.elapsedSeconds()) + "s");
      timer.start();
      this->look_up((TypeHashKey*)refreshspace_handler.h_refresh_embeddingcolumns_,
                    *refreshspace_handler.h_length_, refreshspace_handler.h_refresh_emb_vec_,
                    model_name, i);
      CK_CUDA_THROW_(cudaMemcpyAsync(
          refreshspace_handler.d_refresh_emb_vec_, refreshspace_handler.h_refresh_emb_vec_,
          *refreshspace_handler.h_length_ * cache_config.embedding_vec_size_[i] * sizeof(float),
          cudaMemcpyHostToDevice, streams[i]));
      CK_CUDA_THROW_(cudaStreamSynchronize(streams[i]));
      timer.stop();
      MESSAGE_("Parameter Server looking up the number of " +
               std::to_string(*refreshspace_handler.h_length_) +
               " keys takes: " + std::to_string(timer.elapsedSeconds()) + "s");
      timer.start();
      embedding_cache->Refresh(i, refreshspace_handler.d_refresh_embeddingcolumns_,
                               refreshspace_handler.d_refresh_emb_vec_,
                               *refreshspace_handler.h_length_, streams[i]);
      timer.stop();
      MESSAGE_("Embedding Cache refreshing the number of " +
               std::to_string(*refreshspace_handler.h_length_) +
               " keys takes: " + std::to_string(timer.elapsedSeconds()) + "s");
    }
  }
  for (auto& stream : streams) {
    CK_CUDA_THROW_(cudaStreamSynchronize(stream));
  }
  // apply the memory block for embedding cache refresh workspace
  this->FreeBuffer(memory_block);
  timer_refresh.stop();
  MESSAGE_("The total Time of embedding cache refresh is : " +
           std::to_string(timer_refresh.elapsedSeconds()) + "s");
}

template <typename TypeHashKey>
void parameter_server<TypeHashKey>::insert_embedding_cache(
    embedding_interface* embedding_cache, embedding_cache_config& cache_config,
    embedding_cache_workspace& workspace_handler, const std::vector<cudaStream_t>& streams) {
  HCTR_LOG(INFO, WORLD, "*****Insert embedding cache of model %s on device %d*****\n",
           cache_config.model_name_.c_str(), cache_config.cuda_dev_id_);
  // Copy the missing embeddingcolumns to host
  for (unsigned int i = 0; i < cache_config.num_emb_table_; i++) {
    TypeHashKey* d_missing_key_ptr = (TypeHashKey*)(workspace_handler.d_missing_embeddingcolumns_) +
                                     workspace_handler.h_shuffled_embedding_offset_[i];
    TypeHashKey* h_missing_key_ptr = (TypeHashKey*)(workspace_handler.h_missing_embeddingcolumns_) +
                                     workspace_handler.h_shuffled_embedding_offset_[i];
    CK_CUDA_THROW_(cudaStreamSynchronize(streams[i]));
    CK_CUDA_THROW_(cudaMemcpyAsync(h_missing_key_ptr, d_missing_key_ptr,
                                   workspace_handler.h_missing_length_[i] * sizeof(TypeHashKey),
                                   cudaMemcpyDeviceToHost, streams[i]));
  }
  // Query the missing embeddingcolumns from Parameter Server
  size_t acc_emb_vec_offset = 0;
  for (unsigned int i = 0; i < cache_config.num_emb_table_; i++) {
    TypeHashKey* h_missing_key_ptr = (TypeHashKey*)(workspace_handler.h_missing_embeddingcolumns_) +
                                     workspace_handler.h_shuffled_embedding_offset_[i];
    size_t query_length = workspace_handler.h_shuffled_embedding_offset_[i + 1] -
                          workspace_handler.h_shuffled_embedding_offset_[i];
    float* h_vals_retrieved_ptr = workspace_handler.h_missing_emb_vec_ + acc_emb_vec_offset;
    CK_CUDA_THROW_(cudaStreamSynchronize(streams[i]));
    this->look_up(h_missing_key_ptr, workspace_handler.h_missing_length_[i], h_vals_retrieved_ptr,
                  cache_config.model_name_, i);
    acc_emb_vec_offset += query_length * cache_config.embedding_vec_size_[i];
  }
  // Copy missing emb_vec to device
  acc_emb_vec_offset = 0;
  for (unsigned int i = 0; i < cache_config.num_emb_table_; i++) {
    float* h_vals_retrieved_ptr = workspace_handler.h_missing_emb_vec_ + acc_emb_vec_offset;
    float* d_vals_retrieved_ptr = workspace_handler.d_missing_emb_vec_ + acc_emb_vec_offset;
    size_t missing_len_in_float =
        workspace_handler.h_missing_length_[i] * cache_config.embedding_vec_size_[i];
    size_t missing_len_in_byte = missing_len_in_float * sizeof(float);
    size_t query_length = workspace_handler.h_shuffled_embedding_offset_[i + 1] -
                          workspace_handler.h_shuffled_embedding_offset_[i];
    acc_emb_vec_offset += query_length * cache_config.embedding_vec_size_[i];
    CK_CUDA_THROW_(cudaMemcpyAsync(d_vals_retrieved_ptr, h_vals_retrieved_ptr, missing_len_in_byte,
                                   cudaMemcpyHostToDevice, streams[i]));
  }
  // Insert the vectors for missing keys into embedding cache
  embedding_cache->update(workspace_handler, streams);
}

template class parameter_server<unsigned int>;
template class parameter_server<long long>;

}  // namespace HugeCTR
