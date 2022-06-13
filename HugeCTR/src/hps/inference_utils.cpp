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

#include <common.hpp>
#include <hps/inference_utils.hpp>
#include <parser.hpp>
#include <unordered_set>
#include <utils.hpp>

namespace HugeCTR {

std::ostream& operator<<(std::ostream& os, const DatabaseType_t value) {
  return os << hctr_enum_to_c_str(value);
}
std::ostream& operator<<(std::ostream& os, const DatabaseOverflowPolicy_t value) {
  return os << hctr_enum_to_c_str(value);
}
std::ostream& operator<<(std::ostream& os, const UpdateSourceType_t value) {
  return os << hctr_enum_to_c_str(value);
}

std::optional<size_t> parameter_server_config::find_model_id(const std::string& model_name) const {
  const auto it = model_name_id_map_.find(model_name);
  if (it != model_name_id_map_.end()) {
    return it->second;
  } else {
    return std::nullopt;
  }
}

bool VolatileDatabaseParams::operator==(const VolatileDatabaseParams& p) const {
  return type == p.type &&
         // Backend specific.
         address == p.address && user_name == p.user_name && password == p.password &&
         num_partitions == p.num_partitions && allocation_rate == p.allocation_rate &&
         max_get_batch_size == p.max_get_batch_size && max_set_batch_size == p.max_set_batch_size &&
         // Overflow handling related.
         refresh_time_after_fetch == p.refresh_time_after_fetch &&
         overflow_margin == p.overflow_margin && overflow_policy == p.overflow_policy &&
         overflow_resolution_target == p.overflow_resolution_target &&
         // Caching behavior related.
         initial_cache_rate == p.initial_cache_rate &&
         cache_missed_embeddings == p.cache_missed_embeddings &&
         // Real-time update mechanism related.
         update_filters == p.update_filters;
}
bool VolatileDatabaseParams::operator!=(const VolatileDatabaseParams& p) const {
  return !operator==(p);
}

bool PersistentDatabaseParams::operator==(const PersistentDatabaseParams& p) const {
  return type == p.type &&
         // Backend specific.
         path == p.path && num_threads == p.num_threads && read_only == p.read_only &&
         max_get_batch_size == p.max_get_batch_size && max_set_batch_size == p.max_set_batch_size &&
         // Real-time update mechanism related.
         update_filters == p.update_filters;
}
bool PersistentDatabaseParams::operator!=(const PersistentDatabaseParams& p) const {
  return !operator==(p);
}

bool UpdateSourceParams::operator==(const UpdateSourceParams& p) const {
  return type == p.type &&
         // Backend specific.
         brokers == p.brokers && metadata_refresh_interval_ms == p.metadata_refresh_interval_ms &&
         receive_buffer_size == p.receive_buffer_size && poll_timeout_ms == p.poll_timeout_ms &&
         max_batch_size == p.max_batch_size && failure_backoff_ms == p.failure_backoff_ms &&
         max_commit_interval == p.max_commit_interval;
}
bool UpdateSourceParams::operator!=(const UpdateSourceParams& p) const { return !operator==(p); }

VolatileDatabaseParams::VolatileDatabaseParams(
    const DatabaseType_t type,
    // Backend specific.
    const std::string& address, const std::string& user_name, const std::string& password,
    const size_t num_partitions, const size_t allocation_rate, const size_t max_get_batch_size,
    const size_t max_set_batch_size,
    // Overflow handling related.
    const bool refresh_time_after_fetch, const size_t overflow_margin,
    const DatabaseOverflowPolicy_t overflow_policy, const double overflow_resolution_target,
    // Caching behavior related.
    const double initial_cache_rate, const bool cache_missed_embeddings,
    // Real-time update mechanism related.
    const std::vector<std::string>& update_filters)
    : type(type),
      // Backend specific.
      address(address),
      user_name(user_name),
      password(password),
      num_partitions(num_partitions),
      allocation_rate(allocation_rate),
      max_get_batch_size(max_get_batch_size),
      max_set_batch_size(max_set_batch_size),
      // Overflow handling related.
      refresh_time_after_fetch(refresh_time_after_fetch),
      overflow_margin(overflow_margin),
      overflow_policy(overflow_policy),
      overflow_resolution_target(overflow_resolution_target),
      // Caching behavior related.
      initial_cache_rate(initial_cache_rate),
      cache_missed_embeddings(cache_missed_embeddings),
      // Real-time update mechanism related.
      update_filters(update_filters) {}

PersistentDatabaseParams::PersistentDatabaseParams(const DatabaseType_t type,
                                                   // Backend specific.
                                                   const std::string& path,
                                                   const size_t num_threads, const bool read_only,
                                                   const size_t max_get_batch_size,
                                                   const size_t max_set_batch_size,
                                                   // Real-time update mechanism related.
                                                   const std::vector<std::string>& update_filters)
    : type(type),
      // Backend specific.
      path(path),
      num_threads(num_threads),
      read_only(read_only),
      max_get_batch_size(max_get_batch_size),
      max_set_batch_size(max_set_batch_size),
      // Real-time update mechanism related.
      update_filters(update_filters) {}

UpdateSourceParams::UpdateSourceParams(const UpdateSourceType_t type,
                                       // Backend specific.
                                       const std::string& brokers,
                                       const size_t metadata_refresh_interval_ms,
                                       const size_t receive_buffer_size,
                                       const size_t poll_timeout_ms, const size_t max_batch_size,
                                       const size_t failure_backoff_ms,
                                       const size_t max_commit_interval)
    : type(type),
      // Backend specific.
      brokers(brokers),
      metadata_refresh_interval_ms(metadata_refresh_interval_ms),
      receive_buffer_size(receive_buffer_size),
      poll_timeout_ms(poll_timeout_ms),
      max_batch_size(max_batch_size),
      failure_backoff_ms(failure_backoff_ms),
      max_commit_interval(max_commit_interval) {}

InferenceParams::InferenceParams(
    const std::string& model_name, const size_t max_batchsize, const float hit_rate_threshold,
    const std::string& dense_model_file, const std::vector<std::string>& sparse_model_files,
    const int device_id, const bool use_gpu_embedding_cache, const float cache_size_percentage,
    const bool i64_input_key, const bool use_mixed_precision, const float scaler,
    const bool use_algorithm_search, const bool use_cuda_graph,
    const int number_of_worker_buffers_in_pool, const int number_of_refresh_buffers_in_pool,
    const int thread_pool_size, const float cache_refresh_percentage_per_iteration,
    const std::vector<int>& deployed_devices,
    const std::vector<float>& default_value_for_each_table,
    // Database backend.
    const VolatileDatabaseParams& volatile_db, const PersistentDatabaseParams& persistent_db,
    const UpdateSourceParams& update_source,
    // HPS required
    const int maxnum_des_feature_per_sample, const float refresh_delay,
    const float refresh_interval,
    const std::vector<size_t>& maxnum_catfeature_query_per_table_per_sample,
    const std::vector<size_t>& embedding_vecsize_per_table,
    const std::vector<std::string>& embedding_table_names, const std::string& network_file,
    const size_t label_dim, const size_t slot_num)
    : model_name(model_name),
      max_batchsize(max_batchsize),
      hit_rate_threshold(hit_rate_threshold),
      dense_model_file(dense_model_file),
      sparse_model_files(sparse_model_files),
      device_id(device_id),
      use_gpu_embedding_cache(use_gpu_embedding_cache),
      cache_size_percentage(cache_size_percentage),
      i64_input_key(i64_input_key),
      use_mixed_precision(use_mixed_precision),
      scaler(scaler),
      use_algorithm_search(use_algorithm_search),
      use_cuda_graph(use_cuda_graph),
      number_of_worker_buffers_in_pool(number_of_worker_buffers_in_pool),
      number_of_refresh_buffers_in_pool(number_of_refresh_buffers_in_pool),
      thread_pool_size(thread_pool_size),
      cache_refresh_percentage_per_iteration(cache_refresh_percentage_per_iteration),
      deployed_devices(deployed_devices),
      default_value_for_each_table(default_value_for_each_table),
      // Database backend.
      volatile_db(volatile_db),
      persistent_db(persistent_db),
      update_source(update_source),
      // HPS required
      maxnum_des_feature_per_sample(maxnum_des_feature_per_sample),
      refresh_delay(refresh_delay),
      refresh_interval(refresh_interval),
      maxnum_catfeature_query_per_table_per_sample(maxnum_catfeature_query_per_table_per_sample),
      embedding_vecsize_per_table(embedding_vecsize_per_table),
      embedding_table_names(embedding_table_names),
      network_file(network_file),
      label_dim(label_dim),
      slot_num(slot_num) {
  if (this->default_value_for_each_table.size() != this->sparse_model_files.size()) {
    HCTR_LOG(
        WARNING, ROOT,
        "default_value_for_each_table.size() is not equal to the number of embedding tables\n");
    float default_value =
        this->default_value_for_each_table.size() ? this->default_value_for_each_table[0] : 0.f;
    this->default_value_for_each_table.resize(this->sparse_model_files.size());
    fill_n(this->default_value_for_each_table.begin(), this->sparse_model_files.size(),
           default_value);
  }
}

parameter_server_config::parameter_server_config(const std::string& hps_json_config_file) {
  // Initialize for each model
  // Open model config file and input model json config
  nlohmann::json hps_config(read_json_file(hps_json_config_file));

  // Parsing HPS Databse backend
  //****Update source parameters.
  UpdateSourceParams update_source_params;
  if (hps_config.find("update_source") != hps_config.end()) {
    const nlohmann::json& update_source = get_json(hps_config, "update_source");
    auto& params = update_source_params;

    params.type = get_hps_updatesource_type(update_source, "type");

    // Backend specific.
    params.brokers =
        get_value_from_json_soft<std::string>(update_source, "brokers", "127.0.0.1:9092");

    params.metadata_refresh_interval_ms =
        get_value_from_json_soft<size_t>(update_source, "metadata_refresh_interval_ms", 30'000);

    params.receive_buffer_size =
        get_value_from_json_soft<size_t>(update_source, "receive_buffer_size", 256 * 1024);

    params.poll_timeout_ms =
        get_value_from_json_soft<size_t>(update_source, "poll_timeout_ms", 500);

    params.max_batch_size =
        get_value_from_json_soft<size_t>(update_source, "max_batch_size", 8 * 1024);

    params.failure_backoff_ms =
        get_value_from_json_soft<size_t>(update_source, "failure_backoff_ms", 50);

    params.max_commit_interval =
        get_value_from_json_soft<size_t>(update_source, "max_commit_interval", 32);
  }
  // Persistent database parameters.
  PersistentDatabaseParams persistent_db_params;
  if (hps_config.find("persistent_db") != hps_config.end()) {
    const nlohmann::json& persistent_db = get_json(hps_config, "persistent_db");
    auto& params = persistent_db_params;
    params.type = get_hps_database_type(persistent_db, "type");

    // Backend specific.
    params.path = get_value_from_json_soft<std::string>(
        persistent_db, "path", std::filesystem::temp_directory_path() / "rocksdb");

    params.num_threads = get_value_from_json_soft<size_t>(persistent_db, "num_threads", 16);

    params.read_only = get_value_from_json_soft<bool>(persistent_db, "read_only", false);

    params.max_get_batch_size =
        get_value_from_json_soft<size_t>(persistent_db, "max_get_batch_size", 10'000);

    params.max_set_batch_size =
        get_value_from_json_soft<size_t>(persistent_db, "max_set_batch_size", 10'000);

    if (persistent_db.find("update_filters") != persistent_db.end()) {
      params.update_filters.clear();
      auto update_filters = get_json(persistent_db, "update_filters");
      for (size_t filter_index = 0; filter_index < update_filters.size(); ++filter_index) {
        params.update_filters.emplace_back(update_filters[filter_index].get<std::string>());
      }
    }
  }
  // Volatile database parameters.
  HugeCTR::VolatileDatabaseParams volatile_db_params;
  if (hps_config.find("volatile_db") != hps_config.end()) {
    const nlohmann::json& volatile_db = get_json(hps_config, "volatile_db");
    auto& params = volatile_db_params;
    params.type = get_hps_database_type(volatile_db, "type");

    // Backend specific.
    params.address =
        get_value_from_json_soft<std::string>(volatile_db, "address", "127.0.0.1:7000");

    params.user_name = get_value_from_json_soft<std::string>(volatile_db, "user_name", "default");

    params.password = get_value_from_json_soft<std::string>(volatile_db, "password", "");

    params.num_partitions = get_value_from_json_soft<size_t>(
        volatile_db, "num_partitions", std::min(16u, std::thread::hardware_concurrency()));

    params.allocation_rate =
        get_value_from_json_soft<size_t>(volatile_db, "allocation_rate", 256 * 1024 * 1024);

    params.max_get_batch_size =
        get_value_from_json_soft<size_t>(volatile_db, "max_get_batch_size", 10'000);

    params.max_set_batch_size =
        get_value_from_json_soft<size_t>(volatile_db, "max_set_batch_size", 10'000);

    // Overflow handling related.
    params.refresh_time_after_fetch =
        get_value_from_json_soft<bool>(volatile_db, "refresh_time_after_fetch", false);

    params.overflow_margin = get_value_from_json_soft<size_t>(volatile_db, "overflow_margin",
                                                              std::numeric_limits<size_t>::max());
    params.overflow_policy = get_hps_overflow_policy(volatile_db, "overflow_policy");

    params.overflow_resolution_target =
        get_value_from_json_soft<double>(volatile_db, "overflow_resolution_target", 0.8);

    // Caching behavior related.
    params.initial_cache_rate =
        get_value_from_json_soft<double>(volatile_db, "initial_cache_rate", 1.0);

    params.cache_missed_embeddings =
        get_value_from_json_soft<bool>(volatile_db, "cache_missed_embeddings", false);

    // Real-time update mechanism related.
    if (volatile_db.find("update_filters") != volatile_db.end()) {
      params.update_filters.clear();
      auto update_filters = get_json(volatile_db, "update_filters");
      for (size_t filter_index = 0; filter_index < update_filters.size(); ++filter_index) {
        params.update_filters.emplace_back(update_filters[filter_index].get<std::string>());
      }
    }
  }

  this->volatile_db = volatile_db_params;
  this->persistent_db = persistent_db_params;
  this->update_source = update_source_params;
  // Search for all model configuration
  const nlohmann::json& models = get_json(hps_config, "models");
  HCTR_CHECK_HINT(models.size() > 0,
                  "No model configurations in JSON. Is the file formatted correctly?");
  for (size_t j = 0; j < models.size(); j++) {
    const nlohmann::json& model = models[j];
    // [0] model_name -> std::string
    std::string model_name = get_value_from_json_soft<std::string>(model, "model", "");
    // Initialize <model_name, id> map
    if (model_name_id_map_.count(model_name) == 0) {
      model_name_id_map_.emplace(model_name, (size_t)model_name_id_map_.size());
    }
    // [1] max_batch_size -> size_t
    size_t max_batch_size = get_value_from_json_soft<size_t>(model, "max_batch_size", 64);
    // [2] sparse_model_files -> std::vector<std::string>
    auto sparse_model_files = get_json(model, "sparse_files");
    std::vector<std::string> sparse_files;
    if (sparse_model_files.is_array()) {
      for (size_t sparse_id = 0; sparse_id < sparse_model_files.size(); ++sparse_id) {
        sparse_files.emplace_back(sparse_model_files[sparse_id].get<std::string>());
      }
    }
    // [3] use_gpu_embedding_cache -> bool
    bool use_gpu_embedding_cache = get_value_from_json_soft<bool>(model, "gpucache", true);
    // [4] hit_rate_threshold -> float
    float hit_rate_threshold = get_value_from_json_soft<float>(model, "hit_rate_threshold", 0.9);
    // [5] cache _size_percentage -> float
    float cache_size_percentage = get_value_from_json_soft<float>(model, "gpucacheper", 0.2);

    // [6] dense_file -> std::string
    std::string dense_file = get_value_from_json_soft<std::string>(model, "dense_file", "");

    // [7] device_id -> int
    const int device_id = 0;

    InferenceParams params(model_name, max_batch_size, hit_rate_threshold, dense_file, sparse_files,
                           device_id, use_gpu_embedding_cache, cache_size_percentage, true);
    // [8] number_of_worker_buffers_in_pool ->int
    params.number_of_worker_buffers_in_pool =
        get_value_from_json_soft<int>(model, "num_of_worker_buffer_in_pool", 1);

    // [9] num_of_refresher_buffer_in_pool
    params.number_of_refresh_buffers_in_pool =
        get_value_from_json_soft<int>(model, "num_of_refresher_buffer_in_pool", 1);

    // [10] cache_refresh_percentage_per_iteration
    params.cache_refresh_percentage_per_iteration =
        get_value_from_json_soft<float>(model, "cache_refresh_percentage_per_iteration", 0);

    // [11] deployed_device_list -> std::vector<int>
    auto deployed_device_list = get_json(model, "deployed_device_list");
    params.deployed_devices.clear();
    if (deployed_device_list.is_array()) {
      for (size_t device_index = 0; device_index < deployed_device_list.size(); ++device_index) {
        params.deployed_devices.emplace_back(deployed_device_list[device_index].get<int>());
      }
    }
    params.device_id = params.deployed_devices.back();
    // [12] default_value_for_each_table -> std::vector<float>
    auto default_value_for_each_table = get_json(model, "default_value_for_each_table");
    params.default_value_for_each_table.clear();
    if (default_value_for_each_table.is_array()) {
      for (size_t default_index = 0; default_index < default_value_for_each_table.size();
           ++default_index) {
        params.default_value_for_each_table.emplace_back(
            default_value_for_each_table[default_index].get<float>());
      }
    }
    // [13] maxnum_catfeature_query_per_table_per_sample -> std::vector<int>
    auto maxnum_catfeature_query_per_table_per_sample =
        get_json(model, "maxnum_catfeature_query_per_table_per_sample");
    params.maxnum_catfeature_query_per_table_per_sample.clear();
    if (maxnum_catfeature_query_per_table_per_sample.is_array()) {
      for (size_t cat_index = 0; cat_index < maxnum_catfeature_query_per_table_per_sample.size();
           ++cat_index) {
        params.maxnum_catfeature_query_per_table_per_sample.emplace_back(
            maxnum_catfeature_query_per_table_per_sample[cat_index].get<size_t>());
      }
    }

    // [14] embedding_vecsize_per_table -> std::vector<size_t>
    auto embedding_vecsize_per_table = get_json(model, "embedding_vecsize_per_table");
    params.embedding_vecsize_per_table.clear();
    if (embedding_vecsize_per_table.is_array()) {
      for (size_t vecsize_index = 0; vecsize_index < embedding_vecsize_per_table.size();
           ++vecsize_index) {
        params.embedding_vecsize_per_table.emplace_back(
            embedding_vecsize_per_table[vecsize_index].get<size_t>());
      }
    }

    // [15] maxnum_des_feature_per_sample -> int
    params.maxnum_des_feature_per_sample =
        get_value_from_json_soft<int>(model, "maxnum_des_feature_per_sample", 26);

    // [16] refresh_delay -> float
    params.refresh_delay = get_value_from_json_soft<float>(model, "refresh_delay", 0.0);

    // [17] refresh_interval -> float
    params.refresh_interval = get_value_from_json_soft<int>(model, "refresh_interval", 0.0);

    // [18] embedding_table_names ->  std::vector<string>
    if (model.find("embedding_table_names") != model.end()) {
      auto embedding_table_names = get_json(model, "embedding_table_names");
      params.embedding_table_names.clear();
      if (embedding_table_names.is_array()) {
        for (size_t name_index = 0; name_index < embedding_table_names.size(); ++name_index) {
          params.embedding_table_names.emplace_back(
              embedding_table_names[name_index].get<std::string>());
        }
      }
    } else {
      params.embedding_table_names.clear();
      for (size_t i = 0; i < sparse_model_files.size(); ++i) {
        params.embedding_table_names.emplace_back("sparse_embedding" + std::to_string(i));
      }
    }

    params.volatile_db = volatile_db_params;
    params.persistent_db = persistent_db_params;
    params.update_source = update_source_params;
    inference_params_array.emplace_back(params);

    // Fill the ps required parameters
    emb_file_name_[params.model_name] = params.sparse_model_files;
    emb_table_name_[params.model_name] = params.embedding_table_names;
    embedding_vec_size_[params.model_name] = params.embedding_vecsize_per_table;
    max_feature_num_per_sample_per_emb_table_[params.model_name] =
        params.maxnum_catfeature_query_per_table_per_sample;
    default_emb_vec_value_.emplace_back(params.default_value_for_each_table);
  }
}

parameter_server_config::parameter_server_config(
    std::map<std::string, std::vector<std::string>> emb_table_name,
    std::map<std::string, std::vector<size_t>> embedding_vec_size,
    std::map<std::string, std::vector<size_t>> max_feature_num_per_sample_per_emb_table,
    const std::vector<InferenceParams>& inference_params_array,
    const VolatileDatabaseParams& volatile_db, const PersistentDatabaseParams& persistent_db,
    const UpdateSourceParams& update_source) {
  if (emb_table_name.size() != inference_params_array.size() ||
      embedding_vec_size.size() != inference_params_array.size() ||
      max_feature_num_per_sample_per_emb_table.size() != inference_params_array.size()) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "Wrong input: The number of model names and inference_params_array "
                   "are not consistent.");
  }
  for (size_t i = 0; i < inference_params_array.size(); i++) {
    const auto& inference_params = inference_params_array[i];
    if (emb_table_name.find(inference_params.model_name) == emb_table_name.end() ||
        embedding_vec_size.find(inference_params.model_name) == embedding_vec_size.end() ||
        max_feature_num_per_sample_per_emb_table.find(inference_params.model_name) ==
            max_feature_num_per_sample_per_emb_table.end()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Wrong input: The model_name does not exist in the map.");
    }
    if (emb_table_name[inference_params.model_name].size() !=
            inference_params.default_value_for_each_table.size() ||
        embedding_vec_size[inference_params.model_name].size() !=
            inference_params.default_value_for_each_table.size() ||
        max_feature_num_per_sample_per_emb_table[inference_params.model_name].size() !=
            inference_params.default_value_for_each_table.size()) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Wrong input: The number of embedding tables are not consistent for model " +
                         inference_params.model_name);
    }

    // Initialize <model_name, id> map
    if (model_name_id_map_.count(inference_params.model_name) == 0) {
      model_name_id_map_.emplace(inference_params.model_name, (size_t)model_name_id_map_.size());
    }
    // Read inference config
    std::vector<std::string> emb_file_path;
    for (size_t j = 0; j < inference_params.sparse_model_files.size(); j++) {
      emb_file_path.emplace_back(inference_params.sparse_model_files[j]);
    }
    emb_file_name_[inference_params.model_name] = (emb_file_path);
    emb_table_name_[inference_params.model_name] = emb_table_name[inference_params.model_name];
    embedding_vec_size_[inference_params.model_name] =
        embedding_vec_size[inference_params.model_name];
    max_feature_num_per_sample_per_emb_table_[inference_params.model_name] =
        max_feature_num_per_sample_per_emb_table[inference_params.model_name];
    default_emb_vec_value_.emplace_back(inference_params.default_value_for_each_table);
  }  // end for
  this->inference_params_array = inference_params_array;
  this->volatile_db = volatile_db;
  this->persistent_db = persistent_db;
  this->update_source = update_source;
  for (auto& inference_params : this->inference_params_array) {
    inference_params.volatile_db = volatile_db;
    inference_params.persistent_db = persistent_db;
    inference_params.update_source = update_source;
  }
}

parameter_server_config::parameter_server_config(
    const std::vector<std::string>& model_config_path_array,
    const std::vector<InferenceParams>& inference_params_array) {
  if (model_config_path_array.size() != inference_params_array.size()) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "Wrong input: The size of model_config_path_array and inference_params_array "
                   "are not consistent.");
  }
  for (size_t i = 0; i < model_config_path_array.size(); i++) {
    const auto& model_config_path = model_config_path_array[i];
    const auto& inference_params = inference_params_array[i];
    // Initialize <model_name, id> map
    if (model_name_id_map_.count(inference_params.model_name) == 0) {
      model_name_id_map_.emplace(inference_params.model_name, (size_t)model_name_id_map_.size());
    }

    // Initialize for each model
    // Open model config file and input model json config
    nlohmann::json model_config(read_json_file(model_config_path));

    // Read inference config
    std::vector<std::string> emb_file_path;
    for (size_t j = 0; j < inference_params.sparse_model_files.size(); j++) {
      emb_file_path.emplace_back(inference_params.sparse_model_files[j]);
    }

    emb_file_name_[inference_params.model_name] = (emb_file_path);

    // Read embedding layer config
    std::vector<std::string> emb_table_name;
    std::vector<size_t> embedding_vec_size;
    std::vector<size_t> max_feature_num_per_sample_per_emb_table;
    std::vector<bool> distributed_emb;
    std::vector<float> default_emb_vec_value;

    // Search for all embedding layers
    const nlohmann::json& layers = get_json(model_config, "layers");
    for (size_t j = 0; j < layers.size(); j++) {
      const nlohmann::json& layer = layers[j];
      std::string layer_type = get_value_from_json<std::string>(layer, "type");
      if (layer_type.compare("Data") == 0) {
        const nlohmann::json& sparse_inputs = get_json(layer, "sparse");
        for (size_t k = 0; k < sparse_inputs.size(); k++) {
          max_feature_num_per_sample_per_emb_table.push_back(
              get_max_feature_num_per_sample_from_nnz_per_slot(sparse_inputs[k]));
        }
      } else if (layer_type.compare("DistributedSlotSparseEmbeddingHash") == 0) {
        distributed_emb.emplace_back(true);
        // parse embedding table name from network json file
        emb_table_name.emplace_back(get_value_from_json<std::string>(layer, "top"));
        const nlohmann::json& embedding_hparam = get_json(layer, "sparse_embedding_hparam");
        embedding_vec_size.emplace_back(
            get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size"));
        default_emb_vec_value.emplace_back(
            get_value_from_json_soft<float>(embedding_hparam, "default_emb_vec_value", 0.0f));
      } else if (layer_type.compare("LocalizedSlotSparseEmbeddingHash") == 0 ||
                 layer_type.compare("LocalizedSlotSparseEmbeddingOneHot") == 0) {
        distributed_emb.emplace_back(false);
        emb_table_name.emplace_back(get_value_from_json<std::string>(layer, "top"));
        const nlohmann::json& embedding_hparam = get_json(layer, "sparse_embedding_hparam");
        embedding_vec_size.emplace_back(
            get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size"));
        default_emb_vec_value.emplace_back(
            get_value_from_json_soft<float>(embedding_hparam, "default_emb_vec_value", 0.0f));
      } else {
        break;
      }
    }
    emb_table_name_[inference_params.model_name] = emb_table_name;
    embedding_vec_size_[inference_params.model_name] = embedding_vec_size;
    max_feature_num_per_sample_per_emb_table_[inference_params.model_name] =
        max_feature_num_per_sample_per_emb_table;
    distributed_emb_.emplace_back(distributed_emb);
    default_emb_vec_value_.emplace_back(default_emb_vec_value);
  }  // end for
  this->inference_params_array = inference_params_array;
}

HugeCTR::DatabaseType_t get_hps_database_type(const nlohmann::json& json, const std::string key) {
  if (json.find(key) == json.end()) {
    return HugeCTR::DatabaseType_t::Disabled;
  }
  std::string tmp = get_value_from_json<std::string>(json, key);
  HugeCTR::DatabaseType_t enum_value;
  std::unordered_set<const char*> names;

  enum_value = HugeCTR::DatabaseType_t::Disabled;
  names = {hctr_enum_to_c_str(enum_value), "disable", "none"};
  for (const char* name : names)
    if (tmp == name) {
      return enum_value;
    }

  enum_value = HugeCTR::DatabaseType_t::HashMap;
  names = {hctr_enum_to_c_str(enum_value), "hashmap", "hash", "map"};
  for (const char* name : names)
    if (tmp == name) {
      return enum_value;
    }

  enum_value = HugeCTR::DatabaseType_t::ParallelHashMap;
  names = {hctr_enum_to_c_str(enum_value), "parallel_hashmap", "parallel_hash", "parallel_map"};
  for (const char* name : names)
    if (tmp == name) {
      return enum_value;
    }

  enum_value = HugeCTR::DatabaseType_t::RedisCluster;
  names = {hctr_enum_to_c_str(enum_value), "redis"};
  for (const char* name : names)
    if (tmp == name) {
      return enum_value;
    }

  enum_value = HugeCTR::DatabaseType_t::RocksDB;
  names = {hctr_enum_to_c_str(enum_value), "rocksdb", "rocks"};
  for (const char* name : names)
    if (tmp == name) {
      return enum_value;
    }
  return HugeCTR::DatabaseType_t::Disabled;
}

UpdateSourceType_t get_hps_updatesource_type(const nlohmann::json& json, const std::string key) {
  if (json.find(key) == json.end()) {
    return HugeCTR::UpdateSourceType_t::KafkaMessageQueue;
  }
  std::string tmp = get_value_from_json<std::string>(json, key);
  HugeCTR::UpdateSourceType_t enum_value;
  std::unordered_set<const char*> names;

  enum_value = HugeCTR::UpdateSourceType_t::Null;
  names = {hctr_enum_to_c_str(enum_value), "none"};
  for (const char* name : names)
    if (tmp == name) {
      return enum_value;
    }

  enum_value = HugeCTR::UpdateSourceType_t::KafkaMessageQueue;
  names = {hctr_enum_to_c_str(enum_value), "kafka_mq", "kafka"};
  for (const char* name : names)
    if (tmp == name) {
      return enum_value;
    }

  return HugeCTR::UpdateSourceType_t::KafkaMessageQueue;
}

DatabaseOverflowPolicy_t get_hps_overflow_policy(const nlohmann::json& json,
                                                 const std::string key) {
  if (json.find(key) == json.end()) {
    return HugeCTR::DatabaseOverflowPolicy_t::EvictOldest;
  }
  std::string tmp = get_value_from_json<std::string>(json, key);
  HugeCTR::DatabaseOverflowPolicy_t enum_value;
  std::unordered_set<const char*> names;

  enum_value = HugeCTR::DatabaseOverflowPolicy_t::EvictOldest;
  names = {hctr_enum_to_c_str(enum_value), "oldest"};
  for (const char* name : names)
    if (tmp == name) {
      return enum_value;
    }

  enum_value = HugeCTR::DatabaseOverflowPolicy_t::EvictRandom;
  names = {hctr_enum_to_c_str(enum_value), "random"};
  for (const char* name : names)
    if (tmp == name) {
      return enum_value;
    }

  return HugeCTR::DatabaseOverflowPolicy_t::EvictOldest;
}

}  // namespace HugeCTR
