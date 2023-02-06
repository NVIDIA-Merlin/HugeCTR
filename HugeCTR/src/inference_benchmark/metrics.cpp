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

#include <omp.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <argparse/argparse.hpp>
#include <cassert>
#include <cmath>
#include <common.hpp>
#include <cstdlib>
#include <hps/embedding_cache.hpp>
#include <hps/hier_parameter_server.hpp>
#include <hps/inference_utils.hpp>
#include <hps/lookup_session.hpp>
#include <inference/inference_session.hpp>
#include <inference_benchmark/profiler.hpp>
#include <inference_key_generator.hpp>
#include <iostream>
#include <random>
#include <string>
#include <thread_pool.hpp>
#include <unordered_map>
#include <unordered_set>
#include <utils.hpp>
#include <vector>

using namespace HugeCTR;

template <typename T>
class IntGenerator_normal {
 public:
  IntGenerator_normal() : gen_(rd_()) {}
  IntGenerator_normal(double mean, double dev) : gen_(rd_()), distribution_(mean, dev) {}

  void fill_unique(T* data, size_t len, T min, T max) {
    if (len == 0) {
      return;
    }

    std::unordered_set<T> set;
    size_t sz = 0;
    while (sz < len) {
      T x = (T)(abs(distribution_(gen_)));
      if (x < min || x > max) {
        continue;
      }
      auto res = set.insert(x);
      if (res.second) {
        data[sz++] = x;
      }
    }
    assert(sz == set.size());
    assert(sz == len);
  }

 private:
  std::random_device rd_;
  std::mt19937 gen_;
  std::normal_distribution<double> distribution_;
};

/**
 * @brief Main HPS class
 *
 * This is a class supporting HPS lookup in Python, which consolidates HierParameterServer
 * and LookupSession. A HPS object currently supports one lookup session on a specific GPU
 * for each model, e.g., {"dcn": [0], "deepfm": [1], "wdl": [0], "dlrm": [2]}. To support
 * multiple models deployed on multiple GPUs, e.g., {"dcn": [0, 1, 2, 3], "deepfm": [0, 1],
 * "wdl": [0], "dlrm": [2, 3]}, this class needs to be modified in the future.
 */
class HPS_Metrics {
 public:
  ~HPS_Metrics();
  HPS_Metrics(parameter_server_config& ps_config);
  HPS_Metrics(const std::string& hps_json_config_file, metrics_argues metrics_config);
  HPS_Metrics(HPS_Metrics const&) = delete;
  HPS_Metrics& operator=(HPS_Metrics const&) = delete;

  void lookup(void* h_keys, size_t num_keys);
  void refresh_embeddingcache(int interations);
  void refresh_async(int iteration);
  void print();
  profiler* profile;

 private:
  parameter_server_config ps_config_;
  std::vector<void*> d_reader_keys_list_;
  std::vector<void*> d_reader_row_ptrs_list_;

  std::shared_ptr<HierParameterServerBase>
      parameter_server_;  // Hierarchical paramter server that manages database backends and
                          // embedding caches of all models on all deployed devices
  std::map<std::string, std::map<int64_t, std::shared_ptr<LookupSessionBase>>>
      lookup_session_map_;  // Lookup sessions of all models deployed on all devices, currently only
                            // the first session on the first device will be used during lookup,
                            // i.e., there will be no batching or scheduling
  std::map<std::string, std::map<int64_t, std::vector<float*>>> d_vectors_per_table_map_;

  metrics_argues metrics_config_;

  ThreadPool* refresh_thread_;

  void initialize();
};

HPS_Metrics::~HPS_Metrics() {
  // Join refresh threads
  refresh_thread_->await_idle();
  for (auto it = d_vectors_per_table_map_.begin(); it != d_vectors_per_table_map_.end(); ++it) {
    for (auto f = it->second.begin(); f != it->second.end(); ++f) {
      auto d_vectors_per_table = f->second;
      for (size_t i{0}; i < d_vectors_per_table.size(); ++i) {
        HCTR_LIB_THROW(cudaFree(d_vectors_per_table[i]));
      }
    }
  }
}

HPS_Metrics::HPS_Metrics(const std::string& hps_json_config_file, metrics_argues metrics_config)
    : ps_config_{hps_json_config_file} {
  metrics_config_ = metrics_config;
  initialize();
  refresh_thread_ = new ThreadPool("EC refresh", 16);
  profile = new profiler();
}

void HPS_Metrics::initialize() {
  parameter_server_ =
      HierParameterServerBase::create(ps_config_, ps_config_.inference_params_array);
  if (metrics_config_.database_backend) {
    parameter_server_->set_profiler(metrics_config_.iterations, metrics_config_.warmup, true);
  }

  for (auto& inference_params : ps_config_.inference_params_array) {
    std::map<int64_t, std::shared_ptr<LookupSessionBase>> lookup_sessions;
    for (const auto& device_id : inference_params.deployed_devices) {
      inference_params.device_id = device_id;
      auto embedding_cache =
          parameter_server_->get_embedding_cache(inference_params.model_name, device_id);
      if (metrics_config_.embedding_cache || metrics_config_.refresh_ec) {
        embedding_cache->set_profiler(metrics_config_.iterations, metrics_config_.warmup, true);
      }
      auto lookup_session = LookupSessionBase::create(inference_params, embedding_cache);
      if (metrics_config_.lookup_session) {
        lookup_session->set_profiler(metrics_config_.iterations, metrics_config_.warmup, true);
      }
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
    d_vectors_per_table_map_.emplace(inference_params.model_name, d_vectors_per_table_per_device);
  }
}

void HPS_Metrics::refresh_async(int iteration) {
  refresh_thread_->submit([this, iteration]() { refresh_embeddingcache(iteration); });
}

void HPS_Metrics::refresh_embeddingcache(int iteration) {
  int refresh_iteration =
      ps_config_.inference_params_array[0].cache_refresh_percentage_per_iteration * iteration;
  for (int i = 0; i < refresh_iteration; i++) {
    parameter_server_->refresh_embedding_cache(ps_config_.inference_params_array[0].model_name,
                                               ps_config_.inference_params_array[0].device_id);
  }
}

void HPS_Metrics::lookup(void* h_keys, size_t num_keys) {
  std::string model_name = ps_config_.inference_params_array[0].model_name;
  int device_id = ps_config_.inference_params_array[0].device_id;
  int table_id = ps_config_.inference_params_array[0].sparse_model_files.size() - 1;
  if (lookup_session_map_.find(model_name) == lookup_session_map_.end()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "The model name does not exist in HPS.");
  }
  const auto& embedding_size_per_table = ps_config_.embedding_vec_size_.at(model_name);

  // TODO: batching or scheduling for lookup sessions on multiple GPUs
  const auto& lookup_session = lookup_session_map_.find(model_name)->second.find(device_id)->second;
  auto& d_vectors_per_table =
      d_vectors_per_table_map_.find(model_name)->second.find(device_id)->second;
  lookup_session->lookup(h_keys, d_vectors_per_table[table_id], num_keys, table_id);
  std::vector<float> h_vectors;
  h_vectors.resize(num_keys * embedding_size_per_table[table_id]);
  float* vec_ptr = (h_vectors.data());
  HCTR_LIB_THROW(cudaMemcpy(vec_ptr, d_vectors_per_table[table_id],
                            num_keys * embedding_size_per_table[table_id] * sizeof(float),
                            cudaMemcpyDeviceToHost));
}

void HPS_Metrics::print() {
  parameter_server_->profiler_print();
  for (auto& inference_params : ps_config_.inference_params_array) {
    for (const auto& device_id : inference_params.deployed_devices) {
      inference_params.device_id = device_id;
      parameter_server_->get_embedding_cache(inference_params.model_name, device_id)
          ->profiler_print();
    }
  }

  for (auto it : lookup_session_map_) {
    for (auto lookup_session : it.second) {
      lookup_session.second->profiler_print();
    }
  }
}

int main(int argc, char** argv) {
  argparse::ArgumentParser args("HPS_Profiler");

  args.add_argument("--config")
      .help("The path of the HPS json configuration file")
      .required()
      .action([](const std::string& value) { return value; });

  args.add_argument("--powerlaw")
      .help("Generate the queried key that  in each iteration based on the power distribution")
      .default_value(false)
      .implicit_value(true);

  args.add_argument("--table_size")
      .help("The number of keys in the embedded table")
      .default_value(100000)
      .action([](const std::string& value) { return std::stoi(value); });
  args.add_argument("--alpha")
      .help("Alpha of power distribution")
      .default_value<float>(1.2)
      .action([](const std::string& value) { return std::stof(value); });

  args.add_argument("--hot_key_percentage")
      .help("Percentage of hot keys in embedding tables")
      .default_value<float>(0.2)
      .action([](const std::string& value) { return std::stof(value); });

  args.add_argument("--hot_key_coverage")
      .help("The probability of the hot key in each iteration")
      .default_value<float>(0.8)
      .action([](const std::string& value) { return std::stof(value); });

  args.add_argument("--num_key")
      .help("The number of keys to query for each iteration")
      .default_value(1000)
      .action([](const std::string& value) { return std::stoi(value); });

  args.add_argument("--iterations")
      .help("The number of iterations of the test")
      .default_value(1000)
      .action([](const std::string& value) { return std::stoi(value); });

  args.add_argument("--warmup_iterations")
      .help("Performance results in warmup stage will be discarded")
      .default_value(0)
      .action([](const std::string& value) { return std::stoi(value); });

  args.add_argument("--embedding_cache")
      .help("Enable embedding cache profiler, including the performance of lookup, insert, etc.")
      .default_value(false)
      .implicit_value(true);

  args.add_argument("--database_backend")
      .help("Enable database backend profiler, which is to get the lookup performance of VDB/PDB")
      .default_value(false)
      .implicit_value(true);

  args.add_argument("--refresh_embeddingcache")
      .help(
          "Enable refreshing embedding cache. If the embedding cache tool is also enabled, the "
          "refresh will be performed asynchronously")
      .default_value(false)
      .implicit_value(true);

  args.add_argument("--lookup_session")
      .help(
          "Enable lookup_session profiler, which is E2E profiler, including embedding cache and "
          "data backend query delay")
      .default_value(false)
      .implicit_value(true);

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << args;
    exit(1);
  }
  metrics_argues metrics_config;
  metrics_config.table_size = args.get<int>("--table_size") - 1;

  if (args["--embedding_cache"] == true) {
    metrics_config.embedding_cache = true;
  }
  if (args["--database_backend"] == true) {
    metrics_config.database_backend = true;
  }
  if (args["--lookup_session"] == true) {
    metrics_config.lookup_session = true;
  }
  if (args["--refresh_embeddingcache"] == true) {
    metrics_config.refresh_ec = true;
  }

  metrics_config.iterations = args.get<int>("--iterations");
  metrics_config.num_keys = args.get<int>("--num_key");
  metrics_config.warmup = args.get<int>("--warmup_iterations");
  HPS_Metrics metrics = HPS_Metrics(args.get<std::string>("--config"), metrics_config);

  if (metrics_config.lookup_session || metrics_config.database_backend ||
      metrics_config.embedding_cache) {
    if (metrics_config.refresh_ec) {
      metrics.refresh_async(metrics_config.iterations);
    }
    long long* h_query_keys_index;  // Buffer holding index for keys to be queried
    h_query_keys_index = (long long*)malloc(metrics_config.num_keys * sizeof(long long));
    for (int i = 0; i < metrics_config.iterations; i++) {
      if (args.get<bool>("--powerlaw")) {
        batch_key_generator_by_powerlaw(h_query_keys_index, metrics_config.num_keys,
                                        metrics_config.table_size, args.get<float>("--alpha"));
      } else {
        batch_key_generator_by_hotkey(
            h_query_keys_index, metrics_config.num_keys, metrics_config.table_size,
            args.get<float>("--hot_key_percentage"), args.get<float>("--hot_key_coverage"));
      }

      metrics.lookup(h_query_keys_index, metrics_config.num_keys);
    }
    std::cout << "*** Measurement Results ***" << std::endl;
    metrics.print();
  } else {
    if (metrics_config.refresh_ec) {
      metrics.refresh_embeddingcache(metrics_config.iterations);
      std::cout << "*** Measurement Results ***" << std::endl;
      metrics.print();
    } else {
      std::cout << "Please add at least one test item(embedding cache, lookup_session, "
                   "refresh_embeddingcache or database_backend)"
                << std::endl;
    }
  }

  return 0;
}
