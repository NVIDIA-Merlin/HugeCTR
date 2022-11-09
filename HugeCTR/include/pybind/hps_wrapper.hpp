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
      parameter_server_;  // Hierarchical paramter server that manages database backends and
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
        HCTR_LIB_THROW(cudaFree(d_vectors_per_table[i]));
      }
    }
  }
  for (auto it = h_keys_per_table_map_.begin(); it != h_keys_per_table_map_.end(); ++it) {
    auto h_keys_per_table = it->second;
    for (size_t i{0}; i < h_keys_per_table.size(); ++i) {
      HCTR_LIB_THROW(cudaFreeHost(h_keys_per_table[i]));
    }
  }
  for (auto it = d_keys_per_table_map_.begin(); it != d_keys_per_table_map_.end(); ++it) {
    auto d_keys_per_table = it->second;
    for (size_t i{0}; i < d_keys_per_table.size(); ++i) {
      HCTR_LIB_THROW(cudaFree(d_keys_per_table[i]));
    }
  }
}

HPS::HPS(const std::string& hps_json_config_file) : ps_config_{hps_json_config_file} {
  initialize();
}

HPS::HPS(parameter_server_config& ps_config) : ps_config_(ps_config) { initialize(); }

void HPS::initialize() {
  parameter_server_ =
      HierParameterServerBase::create(ps_config_, ps_config_.inference_params_array);
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
