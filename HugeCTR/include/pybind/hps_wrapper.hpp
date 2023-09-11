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
#pragma once

#include <hps/dlpack.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <hps/embedding_cache.hpp>
#include <hps/hier_parameter_server.hpp>
#include <hps/lookup_session.hpp>
#include <pybind/hpsconversion.hpp>

namespace HugeCTR {

namespace python_lib {

/**
 * @brief Main HPS class
 *
 * This is a class supporting HPS lookup in Python, which consolidates HierParameterServer
 * and LookupSession. A HPS object currently supports one lookup session on a specific GPU
 * for each model, e.g., {"dcn": [0], "deepfm": [1], "wdl": [0], "dlrm": [2]}. To support
 * multiple models deployed on multiple GPUs, e.g., {"dcn": [0, 1, 2, 3], "deepfm": [0, 1],
 * "wdl": [0], "dlrm": [2, 3]}, this class needs to be modified in the future.
 */
class HPS {
 public:
  ~HPS();
  HPS(parameter_server_config& ps_config);
  HPS(const std::string& hps_json_config_file);
  HPS(HPS const&) = delete;
  HPS& operator=(HPS const&) = delete;

  pybind11::array_t<float> lookup(pybind11::array_t<size_t>& h_keys, const std::string& model_name,
                                  size_t table_id, int64_t device_id);
  void lookup_fromdlpack(pybind11::capsule& keys, pybind11::capsule& out_tensor,
                         const std::string& model_name, size_t table_id, int64_t device_id);

 private:
  void initialize();
  parameter_server_config ps_config_;

  std::shared_ptr<HierParameterServerBase>
      parameter_server_;  // Hierarchical parameter server that manages database backends and
                          // embedding caches of all models on all deployed devices
  std::map<std::string, std::map<int64_t, std::shared_ptr<LookupSessionBase>>>
      lookup_session_map_;  // Lookup sessions of all models deployed on all devices, currently only
                            // the first session on the first device will be used during lookup,
                            // i.e., there will be no batching or scheduling
  std::map<std::string, std::map<int64_t, std::vector<float*>>> d_vectors_per_table_map_;

  std::map<std::string, std::vector<unsigned int*>> h_keys_per_table_map_;
  std::map<std::string, std::vector<unsigned int*>> d_keys_per_table_map_;
};

HPS::~HPS() {
  for (auto it = d_vectors_per_table_map_.begin(); it != d_vectors_per_table_map_.end(); ++it) {
    for (auto f = it->second.begin(); f != it->second.end(); ++f) {
      auto d_vectors_per_table = f->second;
      for (size_t i{0}; i < d_vectors_per_table.size(); ++i) {
        HCTR_LIB_CHECK_(cudaFree(d_vectors_per_table[i]));
      }
    }
  }
  for (auto it = h_keys_per_table_map_.begin(); it != h_keys_per_table_map_.end(); ++it) {
    auto h_keys_per_table = it->second;
    for (size_t i{0}; i < h_keys_per_table.size(); ++i) {
      HCTR_LIB_CHECK_(cudaFreeHost(h_keys_per_table[i]));
    }
  }
  for (auto it = d_keys_per_table_map_.begin(); it != d_keys_per_table_map_.end(); ++it) {
    auto d_keys_per_table = it->second;
    for (size_t i{0}; i < d_keys_per_table.size(); ++i) {
      HCTR_LIB_CHECK_(cudaFree(d_keys_per_table[i]));
    }
  }
}

HPS::HPS(const std::string& hps_json_config_file) : ps_config_{hps_json_config_file} {
  initialize();
}

HPS::HPS(parameter_server_config& ps_config) : ps_config_(ps_config) { initialize(); }

void HPS::initialize() {
  parameter_server_ = HierParameterServerBase::create(ps_config_);
  for (auto& inference_params : ps_config_.inference_params_array) {
    std::map<int64_t, std::shared_ptr<LookupSessionBase>> lookup_sessions;
    for (const auto& device_id : inference_params.deployed_devices) {
      inference_params.device_id = device_id;
      auto embedding_cache =
          parameter_server_->get_embedding_cache(inference_params.model_name, device_id);
      auto lookup_session = LookupSessionBase::create(inference_params, embedding_cache);
      lookup_sessions.emplace(device_id, lookup_session);
    }
    lookup_session_map_.emplace(inference_params.model_name, lookup_sessions);

    const auto& max_keys_per_sample_per_table =
        ps_config_.max_feature_num_per_sample_per_emb_table_.at(inference_params.model_name);
    const auto& embedding_size_per_table =
        ps_config_.embedding_vec_size_.at(inference_params.model_name);
    std::map<int64_t, std::vector<float*>> d_vectors_per_table_per_device;
    for (const auto& device_id : inference_params.deployed_devices) {
      CudaDeviceContext context(device_id);
      std::vector<float*> d_vectors_per_table(inference_params.sparse_model_files.size());
      for (size_t id{0}; id < inference_params.sparse_model_files.size(); ++id) {
        HCTR_LIB_THROW(
            cudaMalloc((void**)&d_vectors_per_table[id],
                       inference_params.max_batchsize * max_keys_per_sample_per_table[id] *
                           embedding_size_per_table[id] * sizeof(float)));
      }
      d_vectors_per_table_per_device.emplace(device_id, d_vectors_per_table);
    }

    std::vector<unsigned int*> h_keys_per_table(inference_params.sparse_model_files.size());
    for (size_t id{0}; id < inference_params.sparse_model_files.size(); ++id) {
      HCTR_LIB_THROW(cudaHostAlloc(
          (void**)&h_keys_per_table[id],
          inference_params.max_batchsize * max_keys_per_sample_per_table[id] * sizeof(unsigned int),
          cudaHostAllocPortable));
    }

    std::vector<unsigned int*> d_keys_per_table(inference_params.sparse_model_files.size());
    for (size_t id{0}; id < inference_params.sparse_model_files.size(); ++id) {
      HCTR_LIB_THROW(
          cudaMallocManaged((void**)&d_keys_per_table[id], inference_params.max_batchsize *
                                                               max_keys_per_sample_per_table[id] *
                                                               sizeof(unsigned int)));
    }

    d_vectors_per_table_map_.emplace(inference_params.model_name, d_vectors_per_table_per_device);
    h_keys_per_table_map_.emplace(inference_params.model_name, h_keys_per_table);
    d_keys_per_table_map_.emplace(inference_params.model_name, d_keys_per_table);
  }
}

void HPS::lookup_fromdlpack(pybind11::capsule& keys, pybind11::capsule& vectors,
                            const std::string& model_name, size_t table_id, int64_t device_id) {
  HPSTensor hps_key = fromDLPack(keys);
  size_t num_keys = 1;
  for (int i = 0; i < hps_key.ndim; i++) {
    num_keys *= *(reinterpret_cast<size_t*>(hps_key.shape + i));
  }
  HPSTensor hps_vet = fromDLPack(vectors);
  size_t num_vectors = 1;
  for (int i = 0; i < hps_vet.ndim; i++) {
    num_vectors *= *(reinterpret_cast<size_t*>(hps_vet.shape + i));
  }

  if (lookup_session_map_.find(model_name) == lookup_session_map_.end()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "The model name does not exist in HPS.");
  }
  const auto& max_keys_per_sample_per_table =
      ps_config_.max_feature_num_per_sample_per_emb_table_.at(model_name);
  const auto& embedding_size_per_table = ps_config_.embedding_vec_size_.at(model_name);
  const auto& inference_params =
      parameter_server_->get_hps_model_configuration_map().at(model_name);

  HCTR_THROW_IF(num_keys > max_keys_per_sample_per_table[table_id] * inference_params.max_batchsize,
                HugeCTR::Error_t::DataCheckError,
                "The number of keys to be queried should be no large than "
                "max_keys_per_sample_per_table[table_id] * inference_params.max_batchsize.");

  HCTR_THROW_IF(num_vectors < num_keys * embedding_size_per_table[table_id],
                HugeCTR::Error_t::DataCheckError,
                "The number of vectors to be queried should be equal to or larger than "
                "embedding vector size * number of embedding keys");

  // Handle both keys of both long long and unsigned int
  void* key_ptr;
  if (inference_params.i64_input_key) {
    key_ptr = static_cast<void*>(hps_key.data);
  } else {
    unsigned int* keys;
    if (hps_key.device == DeviceType::CPU) {
      keys = h_keys_per_table_map_.find(model_name)->second[table_id];
    } else {
      keys = d_keys_per_table_map_.find(model_name)->second[table_id];
    }
    auto transform = [](unsigned int* out, long long* in, size_t count) {
      for (size_t i{0}; i < count; ++i) {
        out[i] = static_cast<unsigned int>(in[i]);
      }
    };
    transform(keys, static_cast<long long*>(hps_key.data), num_keys);
    key_ptr = static_cast<void*>(keys);
  }

  // TODO: batching or scheduling for lookup sessions on multiple GPUs
  const auto& lookup_session = lookup_session_map_.find(model_name)->second.find(device_id)->second;
  auto& d_vectors_per_table =
      d_vectors_per_table_map_.find(model_name)->second.find(device_id)->second;

  if (hps_key.device == DeviceType::CPU) {
    lookup_session->lookup(key_ptr, d_vectors_per_table[table_id], num_keys, table_id);
  } else {
    lookup_session->lookup_from_device(key_ptr, d_vectors_per_table[table_id], num_keys, table_id);
  }

  float* vec_ptr = static_cast<float*>(hps_vet.data);
  if (hps_vet.device == DeviceType::CPU) {
    HCTR_LIB_THROW(cudaMemcpy(vec_ptr, d_vectors_per_table[table_id],
                              num_keys * embedding_size_per_table[table_id] * sizeof(float),
                              cudaMemcpyDeviceToHost));
  } else {
    HCTR_LIB_THROW(cudaMemcpy(vec_ptr, d_vectors_per_table[table_id],
                              num_keys * embedding_size_per_table[table_id] * sizeof(float),
                              cudaMemcpyDeviceToDevice));
  }
}
pybind11::array_t<float> HPS::lookup(pybind11::array_t<size_t>& h_keys,
                                     const std::string& model_name, size_t table_id,
                                     int64_t device_id) {
  if (lookup_session_map_.find(model_name) == lookup_session_map_.end()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "The model name does not exist in HPS.");
  }
  const auto& max_keys_per_sample_per_table =
      ps_config_.max_feature_num_per_sample_per_emb_table_.at(model_name);
  const auto& embedding_size_per_table = ps_config_.embedding_vec_size_.at(model_name);
  const auto& inference_params =
      parameter_server_->get_hps_model_configuration_map().at(model_name);
  pybind11::buffer_info key_buf = h_keys.request();
  size_t num_keys = key_buf.size;
  HCTR_CHECK_HINT(key_buf.ndim == 1, "Number of dimensions of h_keys must be one.");
  HCTR_CHECK_HINT(
      num_keys <= max_keys_per_sample_per_table[table_id] * inference_params.max_batchsize,
      "The number of keys to be queried should be no large than "
      "max_keys_per_sample_per_table[table_id] * inference_params.max_batchsize.");

  // Handle both keys of both long long and unsigned int
  void* key_ptr;
  if (inference_params.i64_input_key) {
    key_ptr = static_cast<void*>(key_buf.ptr);
  } else {
    unsigned int* h_keys = h_keys_per_table_map_.find(model_name)->second[table_id];
    auto transform = [](unsigned int* out, long long* in, size_t count) {
      for (size_t i{0}; i < count; ++i) {
        out[i] = static_cast<unsigned int>(in[i]);
      }
    };
    transform(h_keys, static_cast<long long*>(key_buf.ptr), num_keys);
    key_ptr = static_cast<void*>(h_keys);
  }

  // TODO: batching or scheduling for lookup sessions on multiple GPUs
  const auto& lookup_session = lookup_session_map_.find(model_name)->second.find(device_id)->second;
  auto& d_vectors_per_table =
      d_vectors_per_table_map_.find(model_name)->second.find(device_id)->second;
  lookup_session->lookup(key_ptr, d_vectors_per_table[table_id], num_keys, table_id);
  std::vector<size_t> vector_shape{static_cast<size_t>(key_buf.shape[0]),
                                   embedding_size_per_table[table_id]};
  pybind11::array_t<float> h_vectors(vector_shape);
  pybind11::buffer_info vector_buf = h_vectors.request();
  float* vec_ptr = static_cast<float*>(vector_buf.ptr);
  HCTR_LIB_THROW(cudaMemcpy(vec_ptr, d_vectors_per_table[table_id],
                            num_keys * embedding_size_per_table[table_id] * sizeof(float),
                            cudaMemcpyDeviceToHost));
  return h_vectors;
}

void HPSPybind(pybind11::module& m) {
  pybind11::module infer = m.def_submodule("inference", "inference submodule of hugectr");

  pybind11::class_<HugeCTR::VolatileDatabaseParams,
                   std::shared_ptr<HugeCTR::VolatileDatabaseParams>>(infer,
                                                                     "VolatileDatabaseParams")
      .def(
          pybind11::init<DatabaseType_t,
                         // Backend specific.
                         const std::string&, const std::string&, const std::string&, size_t, size_t,
                         size_t, const std::string&, bool, size_t, size_t, bool, const std::string&,
                         const std::string&, const std::string&, const std::string&,
                         // Overflow handling related.
                         size_t, DatabaseOverflowPolicy_t, double,
                         // Caching behavior related.
                         bool, double, bool,
                         // Real-time update mechanism related.
                         const std::vector<std::string>&>(),
          pybind11::arg("type") = DatabaseType_t::ParallelHashMap,
          // Backend specific.
          pybind11::arg("address") = "127.0.0.1:7000", pybind11::arg("user_name") = "default",
          pybind11::arg("password") = "",
          pybind11::arg("num_partitions") = std::min(16u, std::thread::hardware_concurrency()),
          pybind11::arg("allocation_rate") = 256L * 1024L * 1024L,
          pybind11::arg("shared_memory_size") = 16L * 1024L * 1024L * 1024L,
          pybind11::arg("shared_memory_name") = "hctr_mp_hash_map_database",
          pybind11::arg("shared_memory_auto_remove") = true,
          pybind11::arg("num_node_connections") = 5, pybind11::arg("max_batch_size") = 64L * 1024L,
          pybind11::arg("enable_tls") = false,
          pybind11::arg("tls_ca_certificate") = "cacertbundle.crt",
          pybind11::arg("tls_client_certificate") = "client_cert.pem",
          pybind11::arg("tls_client_key") = "client_key.pem",
          pybind11::arg("tls_server_name_identification") = "redis.localhost",
          // Overflow handling related.
          pybind11::arg("overflow_margin") = std::numeric_limits<size_t>::max(),
          pybind11::arg("overflow_policy") = DatabaseOverflowPolicy_t::EvictRandom,
          pybind11::arg("overflow_resolution_target") = 0.8,
          // Caching behavior related.
          pybind11::arg("initialize_after_startup") = true,
          pybind11::arg("initial_cache_rate") = 1.0,
          pybind11::arg("cache_missed_embeddings") = false,
          // Real-time update mechanism related.
          pybind11::arg("update_filters") = std::vector<std::string>{"^hps_.+$"});

  pybind11::class_<HugeCTR::PersistentDatabaseParams,
                   std::shared_ptr<HugeCTR::PersistentDatabaseParams>>(infer,
                                                                       "PersistentDatabaseParams")
      .def(pybind11::init<DatabaseType_t,
                          // Backend specific.
                          const std::string&, size_t, bool, size_t,
                          // Caching behavior related.
                          bool,
                          // Real-time update mechanism related.
                          const std::vector<std::string>&>(),
           pybind11::arg("backend") = DatabaseType_t::Disabled,
           // Backend specific.
           pybind11::arg("path") = (std::filesystem::temp_directory_path() / "rocksdb").string(),
           pybind11::arg("num_threads") = 16, pybind11::arg("read_only") = false,
           pybind11::arg("max_batch_size") = 64L * 1024L,
           // Caching behavior related.
           pybind11::arg("initialize_after_startup") = true,
           // Real-time update mechanism related.
           pybind11::arg("update_filters") = std::vector<std::string>{"^hps_.+$"});

  pybind11::class_<HugeCTR::UpdateSourceParams, std::shared_ptr<HugeCTR::UpdateSourceParams>>(
      infer, "UpdateSourceParams")
      .def(pybind11::init<UpdateSourceType_t,
                          // Backend specific.
                          const std::string&, size_t, size_t, size_t, size_t, size_t, size_t>(),
           pybind11::arg("type") = UpdateSourceType_t::Null,
           // Backend specific.
           pybind11::arg("brokers") = "127.0.0.1:9092",
           pybind11::arg("metadata_refresh_interval_ms") = 30'000,
           pybind11::arg("receive_buffer_size") = 256 * 1024,
           pybind11::arg("poll_timeout_ms") = 500, pybind11::arg("max_batch_size") = 8 * 1024,
           pybind11::arg("failure_backoff_ms") = 50, pybind11::arg("max_commit_interval") = 32);

  pybind11::enum_<EmbeddingCacheType_t>(infer, "EmbeddingCacheType_t")
      .value("Dynamic", EmbeddingCacheType_t::Dynamic)
      .value("UVM", EmbeddingCacheType_t::UVM)
      .value("Static", EmbeddingCacheType_t::Static)
      .value(hctr_enum_to_c_str(EmbeddingCacheType_t::Stochastic), EmbeddingCacheType_t::Stochastic)
      .export_values();

  pybind11::class_<HugeCTR::InferenceParams, std::shared_ptr<HugeCTR::InferenceParams>>(
      infer, "InferenceParams")
      .def(pybind11::init<const std::string&, const size_t, const float, const std::string&,
                          const std::vector<std::string>&, const int, const bool, const float,
                          const bool, const bool, const float, const bool, const bool,
                          // HugeCTR::DATABASE_TYPE, const std::string&, const std::string&,
                          // const float,
                          const int, const int, const int, const float, const std::vector<int>&,
                          const std::vector<float>&, const VolatileDatabaseParams&,
                          const PersistentDatabaseParams&, const UpdateSourceParams&, const int,
                          const float, const float, const std::vector<size_t>&,
                          const std::vector<size_t>&, const std::vector<std::string>&,
                          const std::string&, const size_t, const size_t, const std::string&, bool,
                          const EmbeddingCacheType_t&, bool, bool, bool, bool, bool, bool>(),

           pybind11::arg("model_name"), pybind11::arg("max_batchsize"),
           pybind11::arg("hit_rate_threshold"), pybind11::arg("dense_model_file"),
           pybind11::arg("sparse_model_files"), pybind11::arg("device_id") = 0,
           pybind11::arg("use_gpu_embedding_cache"), pybind11::arg("cache_size_percentage"),
           pybind11::arg("i64_input_key"), pybind11::arg("use_mixed_precision") = false,
           pybind11::arg("scaler") = 1.0, pybind11::arg("use_algorithm_search") = true,
           pybind11::arg("use_cuda_graph") = true,
           pybind11::arg("number_of_worker_buffers_in_pool") = 2,
           pybind11::arg("number_of_refresh_buffers_in_pool") = 1,
           pybind11::arg("thread_pool_size") = 16,
           pybind11::arg("cache_refresh_percentage_per_iteration") = 0.0,
           pybind11::arg("deployed_devices") = std::vector<int>{0},
           pybind11::arg("default_value_for_each_table") = std::vector<float>{0.0f},
           // Database backend.
           pybind11::arg("volatile_db") = VolatileDatabaseParams{},
           pybind11::arg("persistent_db") = PersistentDatabaseParams{},
           pybind11::arg("update_source") = UpdateSourceParams{},
           // HPS required
           pybind11::arg("maxnum_des_feature_per_sample") = 26,
           pybind11::arg("refresh_delay") = 0.0f, pybind11::arg("refresh_interval") = 0.0f,
           pybind11::arg("maxnum_catfeature_query_per_table_per_sample") = std::vector<int>{26},
           pybind11::arg("embedding_vecsize_per_table") = std::vector<int>{128},
           pybind11::arg("embedding_table_names") = std::vector<std::string>{""},
           pybind11::arg("network_file") = "", pybind11::arg("label_dim") = 1,
           pybind11::arg("slot_num") = 10, pybind11::arg("non_trainable_params_file") = "",
           pybind11::arg("use_static_table") = false,
           pybind11::arg("embedding_cache_type") = EmbeddingCacheType_t::Dynamic,
           pybind11::arg("use_context_stream") = true,
           pybind11::arg("fuse_embedding_table") = false,
           pybind11::arg("use_hctr_cache_implementation") = true, pybind11::arg("init_ec") = true,
           pybind11::arg("enable_pagelock") = false, pybind11::arg("fp8_quant") = false);

  pybind11::class_<HugeCTR::parameter_server_config,
                   std::shared_ptr<HugeCTR::parameter_server_config>>(infer,
                                                                      "ParameterServerConfig")
      .def(pybind11::init<std::map<std::string, std::vector<std::string>>,
                          std::map<std::string, std::vector<size_t>>,
                          std::map<std::string, std::vector<size_t>>,
                          const std::vector<InferenceParams>&, const VolatileDatabaseParams&,
                          const PersistentDatabaseParams&, const UpdateSourceParams&>(),
           pybind11::arg("emb_table_name"), pybind11::arg("embedding_vec_size"),
           pybind11::arg("max_feature_num_per_sample_per_emb_table"),
           pybind11::arg("inference_params_array"),
           pybind11::arg("volatile_db") = VolatileDatabaseParams{},
           pybind11::arg("persistent_db") = PersistentDatabaseParams{},
           pybind11::arg("update_source") = UpdateSourceParams{});

  pybind11::class_<HugeCTR::python_lib::HPS, std::shared_ptr<HugeCTR::python_lib::HPS>>(infer,
                                                                                        "HPS")
      .def(pybind11::init<parameter_server_config&>(), pybind11::arg("ps_config"))
      .def(pybind11::init<const std::string&>(), pybind11::arg("hps_json_config_file"))
      .def("lookup", &HugeCTR::python_lib::HPS::lookup, pybind11::arg("h_keys"),
           pybind11::arg("model_name"), pybind11::arg("table_id"), pybind11::arg("device_id") = 0)
      .def("lookup_fromdlpack", &HugeCTR::python_lib::HPS::lookup_fromdlpack, pybind11::arg("keys"),
           pybind11::arg("out_tensor"), pybind11::arg("model_name"), pybind11::arg("table_id"),
           pybind11::arg("device_id") = 0);
}

}  // namespace python_lib

}  // namespace HugeCTR
