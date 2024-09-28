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

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <hps/hash_map_backend.hpp>
#include <hps/hier_parameter_server.hpp>
#include <hps/kafka_message.hpp>
#include <hps/modelloader.hpp>
#include <hps/mp_hash_map_backend.hpp>
#include <hps/redis_backend.hpp>
#include <hps/rocksdb_backend.hpp>
#include <regex>

namespace HugeCTR {

std::string HierParameterServerBase::make_tag_name(const std::string& model_name,
                                                   const std::string& embedding_table_name,
                                                   const bool check_arguments) {
  static const std::regex syntax{"[a-zA-Z0-9_\\-]{1,120}"};
  if (check_arguments) {
    HCTR_CHECK_HINT(std::regex_match(model_name, syntax), "The provided 'model_name' is invalid!");
    HCTR_CHECK_HINT(std::regex_match(embedding_table_name, syntax),
                    "The provided 'embedding_table_name' is invalid!");
  }

  std::ostringstream os;
  os << PS_EMBEDDING_TABLE_TAG_PREFIX << '.';
  os << model_name << '.' << embedding_table_name;
  return os.str();
}

// TODO: remove this static method after merging hugectr_inference_backend
std::shared_ptr<HierParameterServerBase> HierParameterServerBase::create(
    const parameter_server_config& ps_config,
    const std::vector<InferenceParams>& inference_params_array) {
  return HierParameterServerBase::create(ps_config);
}

std::shared_ptr<HierParameterServerBase> HierParameterServerBase::create(
    const parameter_server_config& ps_config) {
  HCTR_CHECK_HINT(ps_config.inference_params_array.size() > 0,
                  "ps_config.inference_params_array should not be empty");
  for (size_t i = 0; i < ps_config.inference_params_array.size(); i++) {
    if (ps_config.inference_params_array[i].i64_input_key !=
        ps_config.inference_params_array[0].i64_input_key) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Inconsistent key types for different models. Parameter server does not "
                     "support hybrid key types.");
    }
  }
  if (ps_config.inference_params_array[0].i64_input_key) {
    return std::make_shared<HierParameterServer<long long>>(ps_config);
  } else {
    return std::make_shared<HierParameterServer<unsigned int>>(ps_config);
  }
}

std::shared_ptr<HierParameterServerBase> HierParameterServerBase::create(
    const std::string& hps_json_config_file) {
  parameter_server_config ps_config{hps_json_config_file};
  return HierParameterServerBase::create(ps_config);
}

HierParameterServerBase::~HierParameterServerBase() = default;

template <typename TypeHashKey>
HierParameterServer<TypeHashKey>::HierParameterServer(const parameter_server_config& ps_config)
    : HierParameterServerBase(), ps_config_(ps_config) {
  HCTR_PRINT(INFO,
             "====================================================HPS "
             "Create====================================================\n");
  auto inference_params_array = ps_config.inference_params_array;
  for (size_t i = 0; i < inference_params_array.size(); i++) {
    if (inference_params_array[i].volatile_db != inference_params_array[0].volatile_db ||
        inference_params_array[i].persistent_db != inference_params_array[0].persistent_db) {
      HCTR_OWN_THROW(
          Error_t::WrongInput,
          "Inconsistent database setup. HugeCTR parameter server does currently not support hybrid "
          "database deployment.");
    }
  }

  if (ps_config_.embedding_vec_size_.size() != inference_params_array.size() ||
      ps_config_.default_emb_vec_value_.size() != inference_params_array.size()) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "Wrong input: The size of parameter server parameters are not correct.");
  }

  // Connect to volatile database.
  {
    const auto& conf = inference_params_array[0].volatile_db;
    switch (conf.type) {
      case DatabaseType_t::Disabled:
        break;  // No volatile database.

      case DatabaseType_t::HashMap:
      case DatabaseType_t::ParallelHashMap: {
        HCTR_LOG_S(INFO, WORLD) << "Creating HashMap CPU database backend..." << std::endl;
        HashMapBackendParams params{
            conf.max_batch_size,
            conf.num_partitions,
            conf.overflow_margin,
            conf.overflow_policy,
            conf.overflow_resolution_target,
            conf.allocation_rate,
        };
        volatile_db_ = std::make_unique<HashMapBackend<TypeHashKey>>(params);
      } break;

      case DatabaseType_t::MultiProcessHashMap: {
        HCTR_LOG_S(INFO, WORLD) << "Creating Multi-Process HashMap CPU database backend..."
                                << std::endl;
        MultiProcessHashMapBackendParams params{
            conf.max_batch_size,
            conf.num_partitions,
            conf.overflow_margin,
            conf.overflow_policy,
            conf.overflow_resolution_target,
            conf.allocation_rate,
            conf.shared_memory_size,
            conf.shared_memory_name,
            std::chrono::milliseconds{100},  // heart_beat_frequency
            conf.shared_memory_auto_remove,
        };
        volatile_db_ = std::make_unique<MultiProcessHashMapBackend<TypeHashKey>>(params);
      } break;

#ifdef HCTR_USE_REDIS
      case DatabaseType_t::RedisCluster: {
        HCTR_LOG_S(INFO, WORLD) << "Creating RedisCluster backend..." << std::endl;
        RedisClusterBackendParams params{
            conf.max_batch_size,
            conf.num_partitions,
            conf.overflow_margin,
            conf.overflow_policy,
            conf.overflow_resolution_target,
            conf.address,
            conf.user_name,
            conf.password,
            conf.num_node_connections,
            conf.enable_tls,
            conf.tls_ca_certificate,
            conf.tls_client_certificate,
            conf.tls_client_key,
            conf.tls_server_name_identification,
        };
        volatile_db_ = std::make_unique<RedisClusterBackend<TypeHashKey>>(params);
      } break;
#endif  // HCTR_USE_REDIS

      default:
        HCTR_DIE("Selected backend (volatile_db.type = %d) is not supported!", conf.type);
        break;
    }
    volatile_db_initialize_after_startup_ = conf.initialize_after_startup;
    volatile_db_cache_rate_ = conf.initial_cache_rate;
    volatile_db_cache_missed_embeddings_ = conf.cache_missed_embeddings;
    HCTR_LOG_S(INFO, WORLD) << "Volatile DB: initial cache rate = " << volatile_db_cache_rate_
                            << std::endl;
    HCTR_LOG_S(INFO, WORLD) << "Volatile DB: cache missed embeddings = "
                            << volatile_db_cache_missed_embeddings_ << std::endl;
  }

  // Connect to persistent database.
  {
    const auto& conf = inference_params_array[0].persistent_db;
    switch (conf.type) {
      case DatabaseType_t::Disabled:
        break;  // No persistent database.

#ifdef HCTR_USE_ROCKS_DB
      case DatabaseType_t::RocksDB: {
        HCTR_LOG_S(INFO, WORLD) << "Creating RocksDB backend..." << std::endl;
        RocksDBBackendParams params{
            conf.max_batch_size,
            conf.path,
            conf.num_threads,
            conf.read_only,
        };
        persistent_db_ = std::make_unique<RocksDBBackend<TypeHashKey>>(params);
      } break;
#endif  // HCTR_USE_ROCKS_DB

      default:
        HCTR_DIE("Selected backend (persistent_db.type = %d) is not supported!", conf.type);
        break;
    }
    persistent_db_initialize_after_startup_ = conf.initialize_after_startup;
  }

  // initialize the profiler
  hps_profiler = std::make_unique<profiler>(ProfilerTarget_t::HPSBACKEND);

  // Load embeddings for each embedding table from each model
  for (size_t i = 0; i < inference_params_array.size(); i++) {
    update_database_per_model(inference_params_array[i]);
  }

  // Initialize embedding cache for each embedding table of each model
  for (size_t i = 0; i < inference_params_array.size(); i++) {
    create_embedding_cache_per_model(inference_params_array[i]);
    inference_params_map_.emplace(inference_params_array[i].model_name, inference_params_array[i]);
  }
  buffer_pool_.reset(new ManagerPool(model_cache_map_, memory_pool_config_));

  // Insert embeddings to embedding cache for each embedding table of each mode
  for (size_t i = 0; i < inference_params_array.size(); i++) {
    if ((inference_params_array[i].use_gpu_embedding_cache &&
         inference_params_array[i].cache_refresh_percentage_per_iteration > 0 &&
         inference_params_array[i].init_ec) ||
        inference_params_array[i].embedding_cache_type != HugeCTR::EmbeddingCacheType_t::Dynamic) {
      HCTR_LOG_S(INFO, ROOT) << "Initialize the embedding cache by by inserting the same size "
                                "model file with embedding cache from beginning"
                             << std::endl;
      init_ec(inference_params_array[i], model_cache_map_[inference_params_array[i].model_name]);
    }
  }
}

template <typename TypeHashKey>
HierParameterServer<TypeHashKey>::~HierParameterServer() {
  // Await all pending volatile database transactions.
  volatile_db_async_inserter_.await_idle();

  for (auto it = model_cache_map_.begin(); it != model_cache_map_.end(); it++) {
    for (auto& v : it->second) {
      CudaDeviceContext dev_restorer{v.second->get_cache_config().cuda_dev_id_};
      v.second->finalize();
    }
  }
  buffer_pool_->DestoryManagerPool();
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::update_database_per_model(
    const InferenceParams& inference_params) {
  IModelLoader* rawreader =
      ModelLoader<TypeHashKey, float>::CreateLoader(DatabaseTableDumpFormat_t::Raw);
  size_t num_tables = inference_params.fuse_embedding_table
                          ? inference_params.fused_sparse_model_files.size()
                          : inference_params.sparse_model_files.size();
  // Create input file stream to read the embedding file
  for (size_t j = 0; j < num_tables; j++) {
    if (ps_config_.embedding_vec_size_[inference_params.model_name].size() != num_tables) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Wrong input: The number of embedding tables in network json file for model " +
                         inference_params.model_name +
                         " doesn't match the number of model files in configuration.");
    }
    // Get raw format model loader
    size_t num_key = 0;
    if (inference_params.fuse_embedding_table) {
      for (int table_id = 0; table_id < inference_params.fused_sparse_model_files[j].size();
           table_id++) {
        rawreader->load(inference_params.embedding_table_names[j],
                        inference_params.fused_sparse_model_files[j][table_id]);
        num_key += rawreader->getkeycount();
      }
    } else {
      rawreader->load(inference_params.embedding_table_names[j],
                      inference_params.sparse_model_files[j]);
      num_key = rawreader->getkeycount();
    }
    ps_config_.embedding_key_count_.at(inference_params.model_name).emplace_back(num_key);
    const std::string tag_name = make_tag_name(
        inference_params.model_name, ps_config_.emb_table_name_[inference_params.model_name][j]);
    const size_t embedding_size = ps_config_.embedding_vec_size_[inference_params.model_name][j];
    // Populate volatile database(s).
    if (volatile_db_ && volatile_db_initialize_after_startup_ &&
        inference_params.embedding_cache_type == HugeCTR::EmbeddingCacheType_t::Dynamic) {
      const size_t volatile_capacity = volatile_db_->capacity(tag_name);
      const size_t volatile_cache_amount =
          (num_key <= volatile_capacity)
              ? num_key
              : static_cast<size_t>(
                    volatile_db_cache_rate_ * static_cast<double>(volatile_capacity) + 0.5);

      volatile_db_async_inserter_.await_idle();
      if (!inference_params.fuse_embedding_table) {
        for (size_t i = 0; i < rawreader->get_num_iterations(); i++) {
          std::pair<void*, size_t> key_result = rawreader->getkeys(i);
          std::pair<void*, size_t> vec_result = rawreader->getvectors(i, embedding_size);
          volatile_db_->insert(tag_name, key_result.second,
                               reinterpret_cast<const TypeHashKey*>(key_result.first),
                               reinterpret_cast<const char*>(vec_result.first),
                               embedding_size * sizeof(float), embedding_size * sizeof(float));
        }
      } else {
        for (int table_id = 0; table_id < inference_params.fused_sparse_model_files[j].size();
             table_id++) {
          rawreader->load(inference_params.embedding_table_names[j],
                          inference_params.fused_sparse_model_files[j][table_id]);
          for (size_t i = 0; i < rawreader->get_num_iterations(); i++) {
            std::pair<void*, size_t> key_result = rawreader->getkeys(i);
            std::pair<void*, size_t> vec_result = rawreader->getvectors(i, embedding_size);
            volatile_db_->insert(tag_name, key_result.second,
                                 reinterpret_cast<const TypeHashKey*>(key_result.first),
                                 reinterpret_cast<const char*>(vec_result.first),
                                 embedding_size * sizeof(float), embedding_size * sizeof(float));
          }
        }
      }

      HCTR_LOG_S(INFO, WORLD) << "Table: " << tag_name << "; cached " << volatile_cache_amount
                              << " / " << num_key << " embeddings in volatile database ("
                              << volatile_db_->get_name()
                              << "); load: " << volatile_db_->size(tag_name) << " / "
                              << volatile_capacity << " (" << std::fixed << std::setprecision(2)
                              << (static_cast<double>(volatile_db_->size(tag_name)) * 100.0 /
                                  static_cast<double>(volatile_capacity))
                              << "%)." << std::endl;
    }

    // Persistent database - by definition - always gets all keys.
    if (persistent_db_ && persistent_db_initialize_after_startup_ &&
        inference_params.embedding_cache_type == HugeCTR::EmbeddingCacheType_t::Dynamic) {
      if (!inference_params.fuse_embedding_table) {
        for (size_t i = 0; i < rawreader->get_num_iterations(); i++) {
          std::pair<void*, size_t> key_result = rawreader->getkeys(i);
          std::pair<void*, size_t> vec_result = rawreader->getvectors(i, embedding_size);
          persistent_db_->insert(tag_name, key_result.second,
                                 reinterpret_cast<const TypeHashKey*>(key_result.first),
                                 reinterpret_cast<const char*>(vec_result.first),
                                 embedding_size * sizeof(float), embedding_size * sizeof(float));
        }
      } else {
        for (int table_id = 0; table_id < inference_params.fused_sparse_model_files[j].size();
             table_id++) {
          rawreader->load(inference_params.embedding_table_names[j],
                          inference_params.fused_sparse_model_files[j][table_id]);
          for (size_t i = 0; i < rawreader->get_num_iterations(); i++) {
            std::pair<void*, size_t> key_result = rawreader->getkeys(i);
            std::pair<void*, size_t> vec_result = rawreader->getvectors(i, embedding_size);
            persistent_db_->insert(tag_name, key_result.second,
                                   reinterpret_cast<const TypeHashKey*>(key_result.first),
                                   reinterpret_cast<const char*>(vec_result.first),
                                   embedding_size * sizeof(float), embedding_size * sizeof(float));
          }
        }
        HCTR_LOG_S(INFO, WORLD) << "Table: " << tag_name << "; cached " << num_key
                                << " embeddings in persistent database ("
                                << persistent_db_->get_name() << ")." << std::endl;
      }
    }
  }
  rawreader->delete_table();

  // Connect to online update service (if configured).
  // TODO: Maybe need to change the location where this is initialized.
  const char kafka_group_prefix[] = "hps.";

  auto kafka_prepare_filter = [](const std::string& s) -> std::string {
    std::ostringstream os;
    os << '^' << PS_EMBEDDING_TABLE_TAG_PREFIX << "\\." << s << "\\..+$";
    return os.str();
  };

  char host_name[HOST_NAME_MAX + 1];
  HCTR_CHECK_HINT(!::gethostname(host_name, sizeof(host_name)), "Unable to determine hostname.\n");

  switch (inference_params.update_source.type) {
    case UpdateSourceType_t::Null:
      break;  // Disabled

    case UpdateSourceType_t::KafkaMessageQueue:
      // Volatile database updates.
      if (volatile_db_ && !inference_params.volatile_db.update_filters.empty()) {
        std::ostringstream consumer_group;
        consumer_group << kafka_group_prefix << "volatile";
        if (!volatile_db_->is_shared()) {
          consumer_group << '.' << host_name;
        }

        std::vector<std::string> tag_filters;
        std::transform(inference_params.volatile_db.update_filters.begin(),
                       inference_params.volatile_db.update_filters.end(),
                       std::back_inserter(tag_filters), kafka_prepare_filter);

        volatile_db_source_ = std::make_unique<KafkaMessageSource<TypeHashKey>>(
            inference_params.update_source.brokers, consumer_group.str(), tag_filters,
            inference_params.update_source.metadata_refresh_interval_ms,
            inference_params.update_source.receive_buffer_size,
            inference_params.update_source.poll_timeout_ms,
            inference_params.update_source.max_batch_size,
            inference_params.update_source.failure_backoff_ms,
            inference_params.update_source.max_commit_interval);
      }
      // Persistent database updates.
      if (persistent_db_ && !inference_params.persistent_db.update_filters.empty()) {
        std::ostringstream consumer_group;
        consumer_group << kafka_group_prefix << "persistent";
        if (!persistent_db_->is_shared()) {
          consumer_group << '.' << host_name;
        }

        std::vector<std::string> tag_filters;
        std::transform(inference_params.persistent_db.update_filters.begin(),
                       inference_params.persistent_db.update_filters.end(),
                       std::back_inserter(tag_filters), kafka_prepare_filter);

        persistent_db_source_ = std::make_unique<KafkaMessageSource<TypeHashKey>>(
            inference_params.update_source.brokers, consumer_group.str(), tag_filters,
            inference_params.update_source.metadata_refresh_interval_ms,
            inference_params.update_source.receive_buffer_size,
            inference_params.update_source.poll_timeout_ms,
            inference_params.update_source.max_batch_size,
            inference_params.update_source.failure_backoff_ms,
            inference_params.update_source.max_commit_interval);
      }
      break;

    default:
      HCTR_DIE("Unsupported update source!\n");
      break;
  }

  HCTR_LOG(DEBUG, WORLD, "Real-time subscribers created!\n");

  // TODO: Update embedding cache!

  // Turn on background updates.
  if (volatile_db_source_) {
    volatile_db_source_->engage([&](const std::string& tag, const size_t num_pairs,
                                    const TypeHashKey* keys, const char* values,
                                    const size_t value_size) {
      HCTR_LOG_C(TRACE, WORLD, "Volatile DB update for tag: '", tag, "', num_pairs: ", num_pairs,
                 ", value_size: ", value_size, " bytes\n");
      volatile_db_->insert(tag, num_pairs, keys, values, value_size, value_size);
    });
  }

  if (persistent_db_source_) {
    persistent_db_source_->engage([&](const std::string& tag, const size_t num_pairs,
                                      const TypeHashKey* keys, const char* values,
                                      const size_t value_size) {
      HCTR_LOG_C(TRACE, WORLD, "Persistent DB update for tag: '", tag, "', num_pairs: ", num_pairs,
                 ", value_size: ", value_size, " bytes\n");
      persistent_db_->insert(tag, num_pairs, keys, values, value_size, value_size);
    });
  }
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::init_ec(
    InferenceParams& inference_params,
    std::map<int64_t, std::shared_ptr<EmbeddingCacheBase>> embedding_cache_map) {
  IModelLoader* rawreader =
      ModelLoader<TypeHashKey, float>::CreateLoader(DatabaseTableDumpFormat_t::Raw);
  size_t num_tables = inference_params.fuse_embedding_table
                          ? inference_params.fused_sparse_model_files.size()
                          : inference_params.sparse_model_files.size();

  for (size_t j = 0; j < num_tables; j++) {
    const std::string tag_name = make_tag_name(
        inference_params.model_name, ps_config_.emb_table_name_[inference_params.model_name][j]);
    for (auto device_id : inference_params.deployed_devices) {
      CudaDeviceContext dev_restorer{device_id};
      HCTR_LOG_S(INFO, ROOT) << "EC initialization on device " << device_id << " for " << tag_name
                             << std::endl;
      cudaStream_t stream = embedding_cache_map[device_id]->get_refresh_streams()[j];
      embedding_cache_config cache_config = embedding_cache_map[device_id]->get_cache_config();
      // apply the memory block for embedding cache refresh workspace
      MemoryBlock* memory_block = nullptr;
      while (memory_block == nullptr) {
        memory_block = reinterpret_cast<struct MemoryBlock*>(this->apply_buffer(
            inference_params.model_name, device_id, CACHE_SPACE_TYPE::REFRESHER));
      }
      EmbeddingCacheRefreshspace refreshspace_handler = memory_block->refresh_buffer;

      if (inference_params.embedding_cache_type == EmbeddingCacheType_t::Dynamic ||
          (inference_params.embedding_cache_type == EmbeddingCacheType_t::Stochastic)) {
        // initialize the embedding cache for each table
        const size_t stride_set =
            std::max(1.0f, floor(cache_config.num_set_in_cache_[j] *
                                 cache_config.cache_refresh_percentage_per_iteration));
        size_t length = SLAB_SIZE * SET_ASSOCIATIVITY * stride_set;
        refreshspace_handler.h_length_ = &length;
        size_t num_iteration = 0;
        std::pair<void*, size_t> key_result;
        std::pair<void*, size_t> vec_result;
        for (size_t idx_set = 0; idx_set + stride_set < cache_config.num_set_in_cache_[j];
             idx_set += stride_set) {
          if (inference_params.fuse_embedding_table) {
            size_t table_id = idx_set % inference_params.fused_sparse_model_files[j].size();
            rawreader->load(inference_params.embedding_table_names[j],
                            inference_params.fused_sparse_model_files[j][table_id], length);
            size_t iter_id = (idx_set / inference_params.fused_sparse_model_files[j].size()) %
                             rawreader->get_num_iterations();
            key_result = rawreader->getkeys(iter_id);
            vec_result = rawreader->getvectors(iter_id, cache_config.embedding_vec_size_[j]);
          } else {
            rawreader->load(inference_params.embedding_table_names[j],
                            inference_params.sparse_model_files[j], length);
            // copy the embedding keys from reader to refresh space
            key_result = rawreader->getkeys(idx_set / stride_set);
            vec_result =
                rawreader->getvectors(idx_set / stride_set, cache_config.embedding_vec_size_[j]);
          }
          HCTR_LIB_THROW(cudaMemcpyAsync(refreshspace_handler.d_refresh_embeddingcolumns_,
                                         reinterpret_cast<const TypeHashKey*>(key_result.first),
                                         *refreshspace_handler.h_length_ * sizeof(TypeHashKey),
                                         cudaMemcpyHostToDevice, stream));
          // copy the embedding vectors from reader to refresh space
          HCTR_LIB_THROW(cudaMemcpyAsync(
              refreshspace_handler.d_refresh_emb_vec_,
              reinterpret_cast<const float*>(vec_result.first),
              *refreshspace_handler.h_length_ * cache_config.embedding_vec_size_[j] * sizeof(float),
              cudaMemcpyHostToDevice, stream));
          HCTR_LOG_S(INFO, ROOT) << "Initialize the embedding table " << j << " for iteration "
                                 << idx_set / stride_set << " with number of " << length << " keys."
                                 << std::endl;
          embedding_cache_map[device_id]->init(j, refreshspace_handler, stream);
          HCTR_LIB_THROW(cudaStreamSynchronize(stream));
          num_iteration++;
        }
      }
      // For UVM/Static solution
      else if (inference_params.embedding_cache_type == EmbeddingCacheType_t::UVM ||
               inference_params.embedding_cache_type == EmbeddingCacheType_t::Static) {
        HCTR_LOG(INFO, WORLD,
                 "To achieve the best performance, when using static table, the pointers of keys "
                 "and vectors in HPS lookup should preferably be aligned to at least 16 Bytes.\n");
        float ratio_per_ini_iteration = 0.1;
        size_t length = cache_config.num_set_in_cache_[j] * ratio_per_ini_iteration;
        size_t num_fused_tables = 1;
        size_t total_emb_keys = cache_config.num_set_in_cache_[j];
        std::vector<size_t> num_emb_keys_per_table;
        std::pair<void*, size_t> key_result;
        std::pair<void*, size_t> vec_result;
        if (inference_params.fuse_embedding_table) {
          num_fused_tables = inference_params.fused_sparse_model_files[j].size();
          total_emb_keys = 0;
          for (int table_id = 0; table_id < inference_params.fused_sparse_model_files[j].size();
               table_id++) {
            rawreader->load(inference_params.embedding_table_names[j],
                            inference_params.fused_sparse_model_files[j][table_id]);
            // Get the number of keys in each table
            num_emb_keys_per_table.emplace_back(rawreader->getkeycount());
            // Get the total number of keys in fused table
            total_emb_keys += rawreader->getkeycount();
          }
        } else {
          // Get the total number of keys in non-fused table
          num_emb_keys_per_table.emplace_back(cache_config.num_set_in_cache_[j]);
        }
        // Calculate the number of iterations required to initialize ec
        size_t num_iterations = 0;
        // The number of keys that need to be inserted into the cache for each table
        size_t numkeys_in_EC_pertable = 0;
        for (size_t table_id = 0; table_id < num_fused_tables; table_id++) {
          if (inference_params.fuse_embedding_table) {
            // Get the number of keys in the cache for  the current table
            numkeys_in_EC_pertable = ((float)num_emb_keys_per_table[table_id] / total_emb_keys) *
                                     cache_config.num_set_in_cache_[j];
            length = ratio_per_ini_iteration * numkeys_in_EC_pertable;
            rawreader->load(inference_params.embedding_table_names[j],
                            inference_params.fused_sparse_model_files[j][table_id], length);
            num_iterations = (numkeys_in_EC_pertable - 1) / length + 1;
          } else {
            rawreader->load(inference_params.embedding_table_names[j],
                            inference_params.sparse_model_files[j], length, 1,
                            inference_params.fp8_quant);
            num_iterations = (cache_config.num_set_in_cache_[j] - 1) / length + 1;
            if (inference_params.fp8_quant) {
              num_iterations = rawreader->get_num_iterations();
            }
          }
          // initializing ec by iteratively inserting keys
          for (size_t it = 0; it < num_iterations; it++) {
            key_result = rawreader->getkeys(it);
            vec_result = rawreader->getvectors(it, cache_config.embedding_vec_size_[j],
                                               inference_params.fp8_quant);
            HCTR_LOG_S(INFO, ROOT)
                << "Initialize the embedding table " << j << " for iteration " << it
                << " with number of " << key_result.second << " keys." << std::endl;
            embedding_cache_map[device_id]->init(
                j, key_result.first, reinterpret_cast<float*>(vec_result.first),
                reinterpret_cast<float*>(rawreader->getmetas(inference_params.fp8_quant)),
                key_result.second, stream);
            HCTR_LIB_THROW(cudaStreamSynchronize(stream));
          }
        }

      } else {
        *refreshspace_handler.h_length_ = cache_config.num_set_in_cache_[j];
        std::pair<void*, size_t> key_result;
        std::pair<void*, size_t> vec_result;
        if (inference_params.fuse_embedding_table) {
          rawreader->load_fused_emb(inference_params.embedding_table_names[j],
                                    inference_params.fused_sparse_model_files[j]);
          key_result = std::make_pair(rawreader->getkeys(), *refreshspace_handler.h_length_);
          vec_result =
              std::make_pair(rawreader->getvectors(),
                             *refreshspace_handler.h_length_ * cache_config.embedding_vec_size_[j]);
        } else {
          rawreader->load(inference_params.embedding_table_names[j],
                          inference_params.sparse_model_files[j], *refreshspace_handler.h_length_);
          key_result = rawreader->getkeys(0);
          vec_result = rawreader->getvectors(0, cache_config.embedding_vec_size_[j]);
        }

        HCTR_LIB_THROW(cudaMemcpyAsync(refreshspace_handler.d_refresh_embeddingcolumns_,
                                       reinterpret_cast<const TypeHashKey*>(key_result.first),
                                       *refreshspace_handler.h_length_ * sizeof(TypeHashKey),
                                       cudaMemcpyHostToDevice, stream));
        HCTR_LIB_THROW(cudaMemcpyAsync(
            refreshspace_handler.d_refresh_emb_vec_,
            reinterpret_cast<const TypeHashKey*>(vec_result.first),
            *refreshspace_handler.h_length_ * cache_config.embedding_vec_size_[j] * sizeof(float),
            cudaMemcpyHostToDevice, stream));
        embedding_cache_map[device_id]->init(j, refreshspace_handler, stream);
        HCTR_LIB_THROW(cudaStreamSynchronize(stream));
      }
      this->free_buffer(memory_block);
    }
  }
  rawreader->delete_table();
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::create_embedding_cache_per_model(
    InferenceParams& inference_params) {
  if (inference_params.deployed_devices.empty()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "The list of deployed devices is empty.");
  }
  if (std::find(inference_params.deployed_devices.begin(), inference_params.deployed_devices.end(),
                inference_params.device_id) == inference_params.deployed_devices.end()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "The device id is not in the list of deployed devices.");
  }
  std::map<int64_t, std::shared_ptr<EmbeddingCacheBase>> embedding_cache_map;
  for (auto device_id : inference_params.deployed_devices) {
    HCTR_LOG(INFO, WORLD, "Creating embedding cache in device %d.\n", device_id);
    inference_params.device_id = device_id;
    embedding_cache_map[device_id] = EmbeddingCacheBase::create(inference_params, ps_config_, this);
  }
  model_cache_map_[inference_params.model_name] = embedding_cache_map;
  memory_pool_config_.num_woker_buffer_size_per_model[inference_params.model_name] =
      inference_params.number_of_worker_buffers_in_pool;
  memory_pool_config_.num_refresh_buffer_size_per_model[inference_params.model_name] =
      inference_params.number_of_refresh_buffers_in_pool;
  if (buffer_pool_ != nullptr) {
    buffer_pool_->_create_memory_pool_per_model(inference_params.model_name,
                                                inference_params.number_of_worker_buffers_in_pool,
                                                embedding_cache_map, CACHE_SPACE_TYPE::WORKER);
    buffer_pool_->_create_memory_pool_per_model(inference_params.model_name,
                                                inference_params.number_of_refresh_buffers_in_pool,
                                                embedding_cache_map, CACHE_SPACE_TYPE::REFRESHER);
  }
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::destory_embedding_cache_per_model(
    const std::string& model_name) {
  if (model_cache_map_.find(model_name) != model_cache_map_.end()) {
    for (auto& f : model_cache_map_[model_name]) {
      f.second->finalize();
    }
    model_cache_map_.erase(model_name);
  }
  buffer_pool_->DestoryManagerPool(model_name);
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::erase_model_from_hps(const std::string& model_name) {
  if (volatile_db_) {
    const std::vector<std::string>& table_names = volatile_db_->find_tables(model_name);
    volatile_db_->evict(table_names);
  }
  if (persistent_db_) {
    const std::vector<std::string>& table_names = persistent_db_->find_tables(model_name);
    persistent_db_->evict(table_names);
  }
}

template <typename TypeHashKey>
std::shared_ptr<EmbeddingCacheBase> HierParameterServer<TypeHashKey>::get_embedding_cache(
    const std::string& model_name, const int device_id) {
  const auto it = model_cache_map_.find(model_name);
  if (it == model_cache_map_.end()) {
    HCTR_LOG_C(WARNING, WORLD, "No embedding cache for model " + model_name + "\n");
    return nullptr;
  }
  if (it->second.find(device_id) == it->second.end()) {
    std::ostringstream os;
    os << "No embedding cache on device " << device_id << " for model " << model_name;
    HCTR_LOG_C(WARNING, WORLD, os.str());
    return nullptr;
  }
  return model_cache_map_[model_name][device_id];
}

template <typename TypeHashKey>
std::map<std::string, InferenceParams>
HierParameterServer<TypeHashKey>::get_hps_model_configuration_map() {
  return inference_params_map_;
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::parse_hps_configuraion(
    const std::string& hps_json_config_file) {
  parameter_server_config ps_config{hps_json_config_file};

  for (auto infer_param : ps_config.inference_params_array) {
    inference_params_map_.emplace(infer_param.model_name, infer_param);
  }
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::profiler_print() {
  hps_profiler->print();
}

template <typename TypeHashKey>
void* HierParameterServer<TypeHashKey>::apply_buffer(const std::string& model_name, int device_id,
                                                     CACHE_SPACE_TYPE cache_type) {
  return buffer_pool_->AllocBuffer(model_name, device_id, cache_type);
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::free_buffer(void* p) {
  buffer_pool_->FreeBuffer(p);
  return;
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::lookup(const void* const h_keys, const size_t length,
                                              float* const h_vectors, const std::string& model_name,
                                              const size_t table_id) {
  if (!length) {
    return;
  }

  const auto start_time = std::chrono::high_resolution_clock::now();
  BaseUnit* start = profiler::start();
  const auto& model_id = ps_config_.find_model_id(model_name);
  HCTR_CHECK_HINT(
      static_cast<bool>(model_id),
      "Error: parameter server unknown model name. Note that this error will also come out with "
      "using Triton LOAD/UNLOAD APIs which haven't been supported in HPS backend.\n");

  const size_t embedding_size = ps_config_.embedding_vec_size_[model_name][table_id];
  const size_t expected_value_size = embedding_size * sizeof(float);
  const std::string& embedding_table_name = ps_config_.emb_table_name_[model_name][table_id];
  const std::string& tag_name = make_tag_name(model_name, embedding_table_name);
  const float default_vec_value = ps_config_.default_emb_vec_value_[*model_id][table_id];

#ifdef ENABLE_INFERENCE
  HCTR_LOG_S(TRACE, WORLD) << "Looking up " << length << " embeddings (each with " << embedding_size
                           << " values)..." << std::endl;
#endif
  size_t hit_count = 0;

  DatabaseMissCallback fill_default{[&](const size_t index) {
    std::fill_n(&h_vectors[index * embedding_size], embedding_size, default_vec_value);
  }};

  // If have volatile and persistent database.
  if (volatile_db_ && persistent_db_) {
    // Do a sequential lookup in the volatile DB, and remember the missing keys.
    constexpr size_t invalid_index{std::numeric_limits<size_t>::max()};
    std::vector<size_t> indices(length, invalid_index);

    start = profiler::start();
    hit_count += volatile_db_->fetch(tag_name, length, reinterpret_cast<const TypeHashKey*>(h_keys),
                                     reinterpret_cast<char*>(h_vectors), expected_value_size,
                                     [&](const size_t index) { indices[index] = index; });
    hps_profiler->end(start, "Lookup the embedding key from VDB");

    HCTR_LOG_C(TRACE, WORLD, volatile_db_->get_name(), ": ", hit_count, " hits, ",
               length - hit_count, " missing!\n");

    if (hit_count != length) {
      // Compress indices (Erase-remove idiom).
      indices.erase(std::remove(indices.begin(), indices.end(), invalid_index), indices.end());

      // Do a sparse lookup in the persisent DB, to fill gaps and set others to default.
      start = profiler::start();
      hit_count += persistent_db_->fetch(
          tag_name, indices.size(), indices.data(), reinterpret_cast<const TypeHashKey*>(h_keys),
          reinterpret_cast<char*>(h_vectors), expected_value_size, fill_default);
      hps_profiler->end(start, "Lookup the missing embedding key from the PDB");

      HCTR_LOG_C(TRACE, WORLD, persistent_db_->get_name(), ": ", hit_count, " hits, ",
                 length - hit_count, " still missing!\n");

      // Elevate KV pairs if desired and possible.
      if (volatile_db_cache_missed_embeddings_) {
        // If the layer 0 cache should be optimized as we go, elevate missed keys.
        auto keys_to_elevate{std::make_shared<std::vector<TypeHashKey>>(indices.size())};
        auto values_to_elevate{
            std::make_shared<std::vector<float>>(indices.size() * embedding_size)};

        start = profiler::start();
        for (size_t i{}; i != indices.size(); ++i) {
          const size_t index{indices[i]};

          (*keys_to_elevate)[i] = reinterpret_cast<const TypeHashKey*>(h_keys)[index];
          std::copy_n(&h_vectors[index * embedding_size], embedding_size,
                      &(*values_to_elevate)[i * embedding_size]);
        }
        hps_profiler->end(start, "Insert the missing embedding key into the VDB");

        HCTR_LOG_C(DEBUG, WORLD, "Attempting to migrate ", keys_to_elevate->size(),
                   " embeddings from ", persistent_db_->get_name(), " to ",
                   volatile_db_->get_name(), ".\n");

        start = profiler::start();
        volatile_db_async_inserter_.submit([this, tag_name, keys_to_elevate, values_to_elevate,
                                            expected_value_size, start]() {
          volatile_db_->insert(tag_name, keys_to_elevate->size(), keys_to_elevate->data(),
                               reinterpret_cast<char*>(values_to_elevate->data()),
                               expected_value_size, expected_value_size);
          hps_profiler->end(
              start, "Insert the missing embedding key from the PDB into the VDB asynchronously");
        });
      }
    }
  } else {
    // If any database.
    DatabaseBackendBase<TypeHashKey>* const db =
        volatile_db_ ? static_cast<DatabaseBackendBase<TypeHashKey>*>(volatile_db_.get())
                     : static_cast<DatabaseBackendBase<TypeHashKey>*>(persistent_db_.get());
    if (db) {
      start = profiler::start();
      // Do a sequential lookup in the volatile DB, but fill gaps with a default value.
      hit_count += db->fetch(tag_name, length, reinterpret_cast<const TypeHashKey*>(h_keys),
                             reinterpret_cast<char*>(h_vectors), expected_value_size, fill_default);
      hps_profiler->end(start, "Lookup the embedding key from default HPS database Backend");
      HCTR_LOG_C(TRACE, WORLD, db->get_name(), ": ", hit_count, " hits, ", length - hit_count,
                 " missing!\n");
    } else {
      // Without a database, set everything to default.
      std::fill_n(h_vectors, length * embedding_size, default_vec_value);
      HCTR_LOG_C(WARNING, WORLD, "No database. All embeddings set to default.\n");
    }
  }

  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
#ifdef ENABLE_INFERENCE
  HCTR_LOG_S(TRACE, WORLD) << "Parameter server lookup of " << hit_count << " / " << length
                           << " embeddings took " << duration.count() << " us." << std::endl;
#endif
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::refresh_embedding_cache(const std::string& model_name,
                                                               const int device_id) {
  HCTR_LOG(TRACE, WORLD, "*****Refresh embedding cache of model %s on device %d*****\n",
           model_name.c_str(), device_id);
  HugeCTR::Timer timer_refresh;

  std::shared_ptr<EmbeddingCacheBase> embedding_cache = get_embedding_cache(model_name, device_id);
  if (!embedding_cache->use_gpu_embedding_cache()) {
    HCTR_LOG(WARNING, WORLD, "GPU embedding cache is not enabled and cannot be refreshed!\n");
    return;
  }

  embedding_cache_config cache_config = embedding_cache->get_cache_config();
  if (cache_config.cache_refresh_percentage_per_iteration <= 0) {
    HCTR_LOG(WARNING, WORLD,
             "The configuration of cache refresh percentage per iteration must be greater than 0 "
             "to refresh the GPU embedding cache!\n");
    return;
  }
  timer_refresh.start();
  std::vector<cudaStream_t> streams = embedding_cache->get_refresh_streams();
  // apply the memory block for embedding cache refresh workspace
  MemoryBlock* memory_block = nullptr;
  while (memory_block == nullptr) {
    memory_block = reinterpret_cast<struct MemoryBlock*>(
        this->apply_buffer(model_name, device_id, CACHE_SPACE_TYPE::REFRESHER));
  }
  EmbeddingCacheRefreshspace refreshspace_handler = memory_block->refresh_buffer;
  // Refresh the embedding cache for each table
  const size_t stride_set = cache_config.num_set_in_refresh_workspace_;
  HugeCTR::Timer timer;
  for (size_t i = 0; i < cache_config.num_emb_table_; i++) {
    for (size_t idx_set = 0; idx_set < cache_config.num_set_in_cache_[i]; idx_set += stride_set) {
      const size_t end_idx = (idx_set + stride_set > cache_config.num_set_in_cache_[i])
                                 ? cache_config.num_set_in_cache_[i]
                                 : idx_set + stride_set;
      timer.start();
      embedding_cache->dump(i, refreshspace_handler.d_refresh_embeddingcolumns_,
                            refreshspace_handler.d_length_, idx_set, end_idx, streams[i]);

      HCTR_LIB_THROW(cudaMemcpyAsync(refreshspace_handler.h_length_, refreshspace_handler.d_length_,
                                     sizeof(size_t), cudaMemcpyDeviceToHost, streams[i]));
      HCTR_LIB_THROW(cudaStreamSynchronize(streams[i]));
      HCTR_LIB_THROW(cudaMemcpyAsync(refreshspace_handler.h_refresh_embeddingcolumns_,
                                     refreshspace_handler.d_refresh_embeddingcolumns_,
                                     *refreshspace_handler.h_length_ * sizeof(TypeHashKey),
                                     cudaMemcpyDeviceToHost, streams[i]));
      HCTR_LIB_THROW(cudaStreamSynchronize(streams[i]));
      timer.stop();
      HCTR_LOG_S(TRACE, ROOT) << "Embedding Cache dumping the number of " << stride_set
                              << " sets takes: " << timer.elapsedSeconds() << "s" << std::endl;
      timer.start();
      this->lookup(
          reinterpret_cast<const TypeHashKey*>(refreshspace_handler.h_refresh_embeddingcolumns_),
          *refreshspace_handler.h_length_, refreshspace_handler.h_refresh_emb_vec_, model_name, i);
      HCTR_LIB_THROW(cudaMemcpyAsync(
          refreshspace_handler.d_refresh_emb_vec_, refreshspace_handler.h_refresh_emb_vec_,
          *refreshspace_handler.h_length_ * cache_config.embedding_vec_size_[i] * sizeof(float),
          cudaMemcpyHostToDevice, streams[i]));
      HCTR_LIB_THROW(cudaStreamSynchronize(streams[i]));
      timer.stop();
      HCTR_LOG_S(TRACE, ROOT) << "Parameter Server looking up the number of "
                              << *refreshspace_handler.h_length_
                              << " keys takes: " << timer.elapsedSeconds() << "s" << std::endl;
      timer.start();
      embedding_cache->refresh(
          static_cast<int>(i), refreshspace_handler.d_refresh_embeddingcolumns_,
          refreshspace_handler.d_refresh_emb_vec_, *refreshspace_handler.h_length_, streams[i]);
      timer.stop();
      HCTR_LOG_S(TRACE, ROOT) << "Embedding Cache refreshing the number of "
                              << *refreshspace_handler.h_length_
                              << " keys takes: " << timer.elapsedSeconds() << "s" << std::endl;
      HCTR_LIB_THROW(cudaStreamSynchronize(streams[i]));
    }
  }
  // apply the memory block for embedding cache refresh workspace
  this->free_buffer(memory_block);
  timer_refresh.stop();
  HCTR_LOG_S(TRACE, ROOT) << "The total Time of embedding cache refresh is : "
                          << timer_refresh.elapsedSeconds() << "s" << std::endl;
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::insert_embedding_cache(
    const size_t table_id, std::shared_ptr<EmbeddingCacheBase> embedding_cache,
    EmbeddingCacheWorkspace& workspace_handler, cudaStream_t stream) {
  auto cache_config = embedding_cache->get_cache_config();
#ifdef ENABLE_INFERENCE
  HCTR_LOG(TRACE, WORLD, "*****Insert embedding cache of model %s on device %d*****\n",
           cache_config.model_name_.c_str(), cache_config.cuda_dev_id_);
#endif
  // Copy the missing embeddingcolumns to host
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  HCTR_LIB_THROW(
      cudaMemcpyAsync(workspace_handler.h_missing_embeddingcolumns_[table_id],
                      workspace_handler.d_missing_embeddingcolumns_[table_id],
                      workspace_handler.h_missing_length_[table_id] * sizeof(TypeHashKey),
                      cudaMemcpyDeviceToHost, stream));

  // Query the missing embeddingcolumns from Parameter Server
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  this->lookup(workspace_handler.h_missing_embeddingcolumns_[table_id],
               workspace_handler.h_missing_length_[table_id],
               workspace_handler.h_missing_emb_vec_[table_id], cache_config.model_name_, table_id);

  // Copy missing emb_vec to device

  const size_t missing_len_in_byte = workspace_handler.h_missing_length_[table_id] *
                                     cache_config.embedding_vec_size_[table_id] * sizeof(float);
  HCTR_LIB_THROW(cudaMemcpyAsync(workspace_handler.d_missing_emb_vec_[table_id],
                                 workspace_handler.h_missing_emb_vec_[table_id],
                                 missing_len_in_byte, cudaMemcpyHostToDevice, stream));
  // Insert the vectors for missing keys into embedding cache
  embedding_cache->insert(table_id, workspace_handler, stream);
}

template class HierParameterServer<long long>;
template class HierParameterServer<unsigned int>;

}  // namespace HugeCTR
