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

#include <chrono>
#include <cstdio>
#include <experimental/filesystem>
#include <inference/local_memory_backend.hpp>
#include <inference/parameter_server.hpp>
#include <inference/redis_backend.hpp>
#include <inference/rocksdb_backend.hpp>

#define DEBUG                                                  \
  std::cout << "[DEBUG]" << __FILE__ << "|" << __func__ << "|" \
            << "LINE" << __LINE__ << " "

namespace fs = std::experimental::filesystem;

namespace HugeCTR {

parameter_server_base::~parameter_server_base() {}

template <typename TypeHashKey>
parameter_server<TypeHashKey>::parameter_server(
    const std::string& framework_name, const std::vector<std::string>& model_config_path,
    std::vector<InferenceParams>& inference_params_array) {
  // Store the configuration

  framework_name_ = framework_name;
  if (model_config_path.size() != inference_params_array.size()) {
    CK_THROW_(Error_t::WrongInput, "Wrong input: The size of input args are not consistent.");
  }
  db_type_ = inference_params_array[0].db_type;
  // Initialize <model_name, id> map
  for (unsigned int i = 0; i < inference_params_array.size(); i++) {
    ps_config_.model_name_id_map_.emplace(inference_params_array[i].model_name, (size_t)i);
  }

  // Initialize for each model
  for (unsigned int i = 0; i < model_config_path.size(); i++) {
    if (db_type_ != inference_params_array[0].db_type) {
      CK_THROW_(
          Error_t::WrongInput,
          "Wrong Type of DB: Current HugeCTR PS does not support hybrid database deployment.");
    }
    if (inference_params_array[i].redis_ip != inference_params_array[0].redis_ip) {
      CK_THROW_(Error_t::WrongInput,
                "Please checke redis_ip of each model : All models must be deployed in the same "
                "Redis cluster .");
    }
    if (inference_params_array[i].rocksdb_path != inference_params_array[0].rocksdb_path) {
      CK_THROW_(Error_t::WrongInput,
                "Please checke rocksdb_path of each model : All models must be deployed in the "
                "same path.");
    }
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
    ps_config_.emb_file_name_.emplace_back(emb_file_path);

    // Read embedding layer config
    const nlohmann::json& j_layers = get_json(model_config, "layers");
    std::vector<bool> distributed_emb;
    std::vector<size_t> embedding_vec_size;
    std::vector<float> default_emb_vec_value;
    // Search for all embedding layers
    for (unsigned int j = 1; j < j_layers.size(); j++) {
      const nlohmann::json& j_single_layer = j_layers[j];
      std::string embedding_type = get_value_from_json<std::string>(j_single_layer, "type");
      if (embedding_type.compare("DistributedSlotSparseEmbeddingHash") == 0) {
        distributed_emb.emplace_back(true);
        const nlohmann::json& embedding_hparam =
            get_json(j_single_layer, "sparse_embedding_hparam");
        embedding_vec_size.emplace_back(
            get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size"));
        default_emb_vec_value.emplace_back(
            get_value_from_json_soft<float>(embedding_hparam, "default_emb_vec_value", 0.0f));
      } else if (embedding_type.compare("LocalizedSlotSparseEmbeddingHash") == 0 ||
                 embedding_type.compare("LocalizedSlotSparseEmbeddingOneHot") == 0) {
        distributed_emb.emplace_back(false);
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
    ps_config_.embedding_vec_size_.emplace_back(embedding_vec_size);
    ps_config_.default_emb_vec_value_.emplace_back(default_emb_vec_value);
  }

  if (ps_config_.distributed_emb_.size() != model_config_path.size() ||
      ps_config_.embedding_vec_size_.size() != model_config_path.size() ||
      ps_config_.default_emb_vec_value_.size() != model_config_path.size()) {
    CK_THROW_(Error_t::WrongInput,
              "Wrong input: The size of parameter server parameters are not correct.");
  }

  // Connect to database(s).
  switch (db_type_) {
    case DATABASE_TYPE::LOCAL: {
      // cpu_memory_db_ = std::make_shared<LocalMemoryBackend<TypeHashKey>>();
      cpu_memory_db_ = std::make_shared<ParallelLocalMemoryBackend<TypeHashKey>>(16);
      cpu_memory_db_cache_rate_ = 1;
      break;
    }
    case DATABASE_TYPE::REDIS: {
      DEBUG << "Creating Redis backend..." << std::endl;
      distributed_db_ = std::make_shared<RedisClusterBackend<TypeHashKey>>(
          inference_params_array[0].redis_ip, "");
      distributed_db_cache_rate_ = 1;
      break;
    }
    case DATABASE_TYPE::ROCKSDB: {
      DEBUG << "Creating RocksDB backend..." << std::endl;
      persistent_db_ =
          std::make_shared<RocksDBBackend<TypeHashKey>>(inference_params_array[0].rocksdb_path);
      break;
    }
    case DATABASE_TYPE::HIERARCHY: {
      DEBUG << "Creating Hierarchy backend..." << std::endl;
      distributed_db_ = std::make_shared<RedisClusterBackend<TypeHashKey>>(
          inference_params_array[0].redis_ip, "");
      distributed_db_cache_rate_ = inference_params_array[0].cache_size_percentage_redis;

      persistent_db_ =
          std::make_shared<RocksDBBackend<TypeHashKey>>(inference_params_array[0].rocksdb_path);
      break;
    }
    default:
      DEBUG << "wrong database type!" << std::endl;
  }

  // Load embeddings for each embedding table from each model
  for (unsigned int i = 0; i < model_config_path.size(); i++) {
    size_t num_emb_table = (ps_config_.emb_file_name_[i]).size();

    for (unsigned int j = 0; j < num_emb_table; j++) {
      // Create input file stream to read the embedding file
      const std::string emb_file_prefix = ps_config_.emb_file_name_[i][j] + "/";
      const std::string key_file = emb_file_prefix + "key";
      const std::string vec_file = emb_file_prefix + "emb_vector";
      std::ifstream key_stream(key_file);
      std::ifstream vec_stream(vec_file);
      // Check if file is opened successfully
      if (!key_stream.is_open() || !vec_stream.is_open()) {
        CK_THROW_(Error_t::WrongInput, "Error: embeddings file not open for reading");
      }
      size_t key_file_size_in_byte = fs::file_size(key_file);
      size_t vec_file_size_in_byte = fs::file_size(vec_file);

      const size_t key_size_in_byte = sizeof(long long);
      const size_t embedding_size = ps_config_.embedding_vec_size_[i][j];
      const size_t vec_size_in_byte = sizeof(float) * embedding_size;

      size_t num_key = key_file_size_in_byte / key_size_in_byte;
      size_t num_vec = vec_file_size_in_byte / vec_size_in_byte;
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

      const size_t cpu_cache_amount =
          std::min(static_cast<size_t>(cpu_memory_db_cache_rate_ * num_key + 0.5), num_key);
      const size_t dist_cache_amount =
          std::min(static_cast<size_t>(distributed_db_cache_rate_ * num_key + 0.5), num_key);
      assert(cpu_cache_amount + dist_cache_amount <= num_key);

      const std::string table_name = inference_params_array[i].model_name + "#" + std::to_string(j);

      // Populate databases. CPU is static.
      if (cpu_memory_db_) {
        cpu_memory_db_->insert(table_name, cpu_cache_amount, key_vec.data(),
                               reinterpret_cast<const char*>(vec_vec.data()),
                               embedding_size * sizeof(float));
        DEBUG << "Cached " << cpu_cache_amount << " embeddings in CPU memory database!"
              << std::endl;
      }

      // Distributed could be static but does not have to be.
      if (distributed_db_) {
        distributed_db_->insert(table_name, dist_cache_amount, &key_vec[cpu_cache_amount],
                                reinterpret_cast<const char*>(&vec_vec[cpu_cache_amount]),
                                embedding_size * sizeof(float));
        DEBUG << "Cached " << cpu_cache_amount << " embeddings in distributed database!"
              << std::endl;
      }

      // Persistent database - by definition - always gets all keys.
      if (persistent_db_) {
        persistent_db_->insert(table_name, num_key, key_vec.data(),
                               reinterpret_cast<const char*>(vec_vec.data()),
                               embedding_size * sizeof(float));
        DEBUG << "Cached " << num_key << " embeddings in persistent database!" << std::endl;
      }
    }

    //*********
    // The operation here is just to keep the logic of device_id in the python api
    // unchanged
    //*********
    std::map<int64_t, std::shared_ptr<embedding_interface>> embedding_cache_map;
    if (inference_params_array[i].depolyed_devices.empty()) {
      inference_params_array[i].depolyed_devices.push_back(inference_params_array[i].device_id);
    }
    for (auto device_id : inference_params_array[i].depolyed_devices) {
      DEBUG << "Create embedding cache in device " << device_id << "." << std::endl;
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

  bufferpool = new ManagerPool(model_cache_map, memory_pool_config);
}

template <typename TypeHashKey>
parameter_server<TypeHashKey>::~parameter_server() {
  MESSAGE_("***********Dtor of parameter server***********");
  bufferpool->DestoryManagerPool();
  delete bufferpool;
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
  const auto start_time = std::chrono::high_resolution_clock::now();

  const auto& model_id = ps_config_.find_model_id(model_name);
  if (!model_id) {
    std::cout << "Error: parameter server unknown model name. Note that this error will also come "
                 "out with "
                 "using Triton LOAD/UNLOAD APIs which haven't been supported in HugeCTR backend."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  const size_t embedding_size = ps_config_.embedding_vec_size_[*model_id][embedding_table_id];
  const std::string table_name = model_name + "#" + std::to_string(embedding_table_id);

  DEBUG << "Looking up " << length << " embeddings..." << std::endl;

  size_t hit_count = 0;
  auto db = db_stack_.begin();

  switch (db_stack_.size()) {
    case 0: {
      // Everything is default.
      std::fill_n(h_embeddingoutputvector, length * embedding_size, 0);
      DEBUG << "No database. All embeddings set to default." << std::endl;
      break;
    }
    case 1: {
      // Query database, and fill in default value for unavailable embeddings.
      MissingKeyCallback fill_default_fn = [&](size_t index) -> void {
        std::fill_n(&h_embeddingoutputvector[index * embedding_size], embedding_size, 0);
      };
      hit_count += (*db)->fetch(table_name, length, h_embeddingcolumns,
                                reinterpret_cast<char*>(h_embeddingoutputvector),
                                embedding_size * sizeof(float), fill_default_fn);
      DEBUG << (*db)->get_name() << ": " << hit_count << " hits, " << (length - hit_count)
            << " missing!" << std::endl;
      break;
    }
    default: {
      // Layer 0: Do a sequential lookup. Remember missing keys.
      std::vector<size_t> indices;
      std::vector<size_t> missing;

      MissingKeyCallback record_missing_fn = [&missing](size_t index) -> void {
        // DEBUG << "Key " << h_embeddingcolumns[index] << " at index " << index << " was not
        // found!"
        // << std::endl;
        missing.push_back(index);
      };

      hit_count += (*db)->fetch(table_name, length, h_embeddingcolumns,
                                reinterpret_cast<char*>(h_embeddingoutputvector),
                                embedding_size * sizeof(float), record_missing_fn);
      DEBUG << (*db)->get_name() << ": " << hit_count << " hits, " << missing.size() << " missing!"
            << std::endl;
      db++;

      // Layers 1 thru N-2: Do a sparse lookup. Remember missing keys
      for (size_t i = 2; i < db_stack_.size(); i++, db++) {
        indices.clear();
        indices.swap(missing);
        hit_count += (*db)->fetch(table_name, indices.size(), indices.data(), h_embeddingcolumns,
                                  reinterpret_cast<char*>(h_embeddingoutputvector),
                                  embedding_size * sizeof(float), record_missing_fn);
        DEBUG << (*db)->get_name() << ": " << hit_count << " hits, " << missing.size()
              << " missing!" << std::endl;
      }

      // Layer N-1: Do a sparse lookup. Fill in default values.
      MissingKeyCallback fill_default_fn = [&](size_t index) -> void {
        std::fill_n(&h_embeddingoutputvector[index * embedding_size], embedding_size, 0);
      };
      hit_count += (*db)->fetch(table_name, missing.size(), missing.data(), h_embeddingcolumns,
                                reinterpret_cast<char*>(h_embeddingoutputvector),
                                embedding_size * sizeof(float), fill_default_fn);
      DEBUG << (*db)->get_name() << ": " << hit_count << " hits, " << (length - hit_count)
            << " missing!" << std::endl;
    }
  }

  DEBUG << "Parameter server lookup; total hit count: " << hit_count << " / " << length
        << std::endl;

  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  std::cout << "Lookup of " << hit_count << " / " << length << " embeddings took "
            << duration.count() << " us" << std::endl;
}

template <typename TypeHashKey>
void parameter_server<TypeHashKey>::refresh_embedding_cache(const std::string& model_name,
                                                            int device_id) {
  std::cout << "refresh embedding cache" << std::endl;
  std::shared_ptr<embedding_interface> embedding_cache = GetEmbeddingCache(model_name, device_id);
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
  size_t length{0};
  for (unsigned int i = 0; i < cache_config.num_emb_table_; i++) {
    for (unsigned int idx_set = 0; idx_set < cache_config.num_set_in_cache_[i];
         idx_set += stride_set) {
      embedding_cache->Dump(i, refreshspace_handler.d_refresh_embeddingcolumns_, &length, idx_set,
                            idx_set + stride_set, streams[i]);
      CK_CUDA_THROW_(cudaMemcpyAsync(refreshspace_handler.h_refresh_embeddingcolumns_,
                                     refreshspace_handler.d_refresh_embeddingcolumns_,
                                     length * sizeof(TypeHashKey), cudaMemcpyDeviceToHost,
                                     streams[i]));
      CK_CUDA_THROW_(cudaStreamSynchronize(streams[i]));
      this->look_up((TypeHashKey*)refreshspace_handler.h_refresh_embeddingcolumns_, length,
                    refreshspace_handler.h_refresh_emb_vec_, model_name, i);
      CK_CUDA_THROW_(cudaMemcpyAsync(refreshspace_handler.d_refresh_emb_vec_,
                                     refreshspace_handler.h_refresh_emb_vec_,
                                     length * cache_config.embedding_vec_size_[i] * sizeof(float),
                                     cudaMemcpyHostToDevice, streams[i]));
      embedding_cache->Refresh(i, refreshspace_handler.d_refresh_embeddingcolumns_,
                               refreshspace_handler.d_refresh_emb_vec_, length, streams[i]);
    }
  }
  for (auto& stream : streams) {
    CK_CUDA_THROW_(cudaStreamSynchronize(stream));
  }
  // apply the memory block for embedding cache refresh workspace
  this->FreeBuffer(memory_block);
}

template <typename TypeHashKey>
void parameter_server<TypeHashKey>::insert_embedding_cache(
    embedding_interface* embedding_cache, embedding_cache_config& cache_config,
    embedding_cache_workspace& workspace_handler, const std::vector<cudaStream_t>& streams) {
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
