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

#include <parser.hpp>

namespace HugeCTR {

SolverParser::SolverParser(const std::string& file) {
  try {
    /* file read to json */
    nlohmann::json config = read_json_file(file);

    const std::map<std::string, LrPolicy_t> LR_POLICY = {{"fixed", LrPolicy_t::fixed}};

    /* parse the solver */
    auto j = get_json(config, "solver");

    if (has_key_(j, "seed")) {
      seed = get_value_from_json<unsigned long long>(j, "seed");
    } else {
      seed = 0ull;
    }

    auto lr_policy_string = get_value_from_json<std::string>(j, "lr_policy");
    if (!find_item_in_map(lr_policy, lr_policy_string, LR_POLICY)) {
      CK_THROW_(Error_t::WrongInput, "No such poliicy: " + lr_policy_string);
    }

    display = get_value_from_json<int>(j, "display");

    bool has_max_iter = has_key_(j, "max_iter");
    bool has_num_epochs = has_key_(j, "num_epochs");
    if (has_max_iter && has_num_epochs) {
      CK_THROW_(Error_t::WrongInput, "max_iter and num_epochs cannot be used together.");
    } else {
      if (has_max_iter) {
        max_iter = get_value_from_json<int>(j, "max_iter");
        num_epochs = 0;
      } else if (has_num_epochs) {
        max_iter = 0;
        num_epochs = get_value_from_json<int>(j, "num_epochs");
      } else {
        max_iter = 0;
        num_epochs = 1;
      }
    }

    snapshot = get_value_from_json<int>(j, "snapshot");
    batchsize = get_value_from_json<int>(j, "batchsize");
    batchsize_eval = get_value_from_json_soft<int>(j, "batchsize_eval", batchsize);
    snapshot_prefix = get_value_from_json<std::string>(j, "snapshot_prefix");
    if (has_key_(j, "dense_model_file")) {
      model_file = get_value_from_json<std::string>(j, "dense_model_file");
    }
    if (has_key_(j, "dense_opt_states_file")) {
      dense_opt_states_file = get_value_from_json<std::string>(j, "dense_opt_states_file");
    }
    FIND_AND_ASSIGN_INT_KEY(eval_interval, j);

    if (has_key_(j, "eval_batches")) {
      CK_THROW_(Error_t::WrongInput,
                "eval_batches is deprecated, use max_eval_batches or max_eval_samples.");
    }

    bool has_max_eval_batches = has_key_(j, "max_eval_batches");
    bool has_max_eval_samples = has_key_(j, "max_eval_samples");
    if (has_max_eval_batches && has_max_eval_samples) {
      CK_THROW_(Error_t::WrongInput,
                "max_eval_batches and max_eval_samples cannot be used together.");
    } else {
      if (has_max_eval_batches) {
        max_eval_batches = get_value_from_json<int>(j, "max_eval_batches");
      } else if (has_max_eval_samples) {
        int max_eval_samples = get_value_from_json<int>(j, "max_eval_samples");
        int rem = max_eval_samples % batchsize_eval;
        if (rem) {
          MESSAGE_("max_eval_samples(" + std::to_string(max_eval_samples) +
                   ") is not divisible by batchsize_eval(" + std::to_string(batchsize_eval) +
                   ". The remainder is truncated.");
        }
        max_eval_batches = max_eval_samples / batchsize_eval;
      } else {
        CK_THROW_(Error_t::WrongInput,
                  "Either max_eval_batches or max_eval_samples must be specified.");
      }
    }

    if (has_key_(j, "sparse_model_file")) {
      auto j_embedding_files = get_json(j, "sparse_model_file");
      if (j_embedding_files.is_array()) {
        for (auto j_embedding_tmp : j_embedding_files) {
          embedding_files.push_back(j_embedding_tmp.get<std::string>());
        }
      } else {
        embedding_files.push_back(get_value_from_json<std::string>(j, "sparse_model_file"));
      }
    }

    if (has_key_(j, "sparse_opt_states_file")) {
      auto j_sparse_opt_states_files = get_json(j, "sparse_opt_states_file");
      if (j_sparse_opt_states_files.is_array()) {
        for (auto j_embedding_tmp : j_sparse_opt_states_files) {
          sparse_opt_states_files.push_back(j_embedding_tmp.get<std::string>());
        }
      } else {
        sparse_opt_states_files.push_back(
            get_value_from_json<std::string>(j, "sparse_opt_states_file"));
      }
    }

    if (has_key_(j, "mixed_precision") && has_key_(j, "enable_tf32_compute")) {
      CK_THROW_(Error_t::WrongInput, "mixed_precision and enable_tf32_compute MUST not coexist");
    }

    if (has_key_(j, "mixed_precision")) {
      use_mixed_precision = true;
      int i_scaler = get_value_from_json<int>(j, "mixed_precision");
      if (i_scaler != 128 && i_scaler != 256 && i_scaler != 512 && i_scaler != 1024) {
        CK_THROW_(Error_t::WrongInput,
                  "Scaler of mixed_precision training should be either 128/256/512/1024");
      }
      scaler = i_scaler;
      std::stringstream ss;
      ss << "Mixed Precision training with scaler: " << i_scaler << " is enabled." << std::endl;
      MESSAGE_(ss.str());

    } else {
      use_mixed_precision = false;
      scaler = 1.f;
    }

    enable_tf32_compute = get_value_from_json_soft<bool>(j, "enable_tf32_compute", false);
    MESSAGE_("TF32 Compute: " + std::string(enable_tf32_compute ? "ON" : "OFF"));

    auto gpu_array = get_json(j, "gpu");
    assert(vvgpu.empty());
    // todo: output the device map
    if (gpu_array[0].is_array()) {
      for (auto gpu : gpu_array) {
        std::vector<int> vgpu;
        assert(vgpu.empty());
        for (auto gpu_tmp : gpu) {
          int gpu_id = gpu_tmp.get<int>();
          vgpu.push_back(gpu_id);
          if (gpu_id < 0) {
            CK_THROW_(Error_t::WrongInput, "gpu_id < 0");
          }
        }
        vvgpu.push_back(vgpu);
      }
    } else {
      std::vector<int> vgpu;
      for (auto gpu_tmp : gpu_array) {
        int gpu_id = gpu_tmp.get<int>();
        vgpu.push_back(gpu_id);
        if (gpu_id < 0) {
          CK_THROW_(Error_t::WrongInput, "gpu_id < 0");
        }
      }
      vvgpu.push_back(vgpu);
    }

    if (has_key_(j, "device_layout")) {
      std::string device_layout_str = get_value_from_json<std::string>(j, "device_layout");
      if (device_layout_str == "LocalFirst") {
        device_layout = DeviceMap::LOCAL_FIRST;
      } else if (device_layout_str == "NodeFirst") {
        device_layout = DeviceMap::NODE_FIRST;
      } else {
        CK_THROW_(Error_t::WrongInput, "Invalid device layout. Options are: LocalFirst, NodeFirst");
      }
    } else {
      device_layout = DeviceMap::LOCAL_FIRST;
    }

    const std::map<std::string, metrics::Type> metrics_map = {
        {"AverageLoss", metrics::Type::AverageLoss},
        {"AUC", metrics::Type::AUC},
        {"HitRate", metrics::Type::HitRate}};

    if (has_key_(j, "eval_metrics")) {
      auto eval_metrics = get_json(j, "eval_metrics");
      if (eval_metrics.is_array()) {
        for (auto metric : eval_metrics) {
          std::stringstream ss(metric.get<std::string>());
          std::string elem;
          std::vector<std::string> metric_strs;
          while (std::getline(ss, elem, ':')) {
            metric_strs.push_back(std::move(elem));
          }
          auto it = metrics_map.find(metric_strs[0]);
          if (it != metrics_map.end()) {
            auto type = it->second;
            switch (type) {
              case metrics::Type::AverageLoss: {
                metrics_spec[metrics::Type::AverageLoss] = 0.f;
                break;
              }
              case metrics::Type::AUC: {
                float val = (metric_strs.size() == 1) ? 1.f : std::stof(metric_strs[1]);
                if (val < 0.0 || val > 1.0) {
                  CK_THROW_(Error_t::WrongInput, "0 <= AUC threshold <= 1 is not true");
                }
                metrics_spec[metrics::Type::AUC] = val;
                break;
              }
              case metrics::Type::HitRate: {
                float val = (metric_strs.size() == 1) ? 1.f : std::stof(metric_strs[1]);
                if (val < 0.0 || val > 1.0) {
                  CK_THROW_(Error_t::WrongInput, "0 <= HitRate threshold <= 1 is not true");
                }
                metrics_spec[metrics::Type::HitRate] = val;
                break;
              }
              default: {
                CK_THROW_(Error_t::WrongInput, "Unreachable");
                break;
              }
            }
          } else {
            CK_THROW_(Error_t::WrongInput, metric_strs[0] + " is a unsupported metric");
          }
        }
      } else {
        CK_THROW_(Error_t::WrongInput, "metrics must be in the form of list");
      }
    } else {
      // Default is AUC without the threshold
      MESSAGE_("Default evaluation metric is AUC without threshold value");
      metrics_spec[metrics::Type::AUC] = 1.f;
    }

    if (has_key_(j, "input_key_type")) {
      auto str = get_value_from_json<std::string>(j, "input_key_type");
      if (str.compare("I64") == 0) {
        i64_input_key = true;
      } else if (str.compare("I32") == 0) {
        i64_input_key = false;
      } else {
        CK_THROW_(Error_t::WrongInput, "input_key_type is I64 or I32");
      }
    } else {
      i64_input_key = false;
    }

    use_algorithm_search = get_value_from_json_soft<bool>(j, "algorithm_search", true);
    MESSAGE_("Algorithm search: " + std::string(use_algorithm_search ? "ON" : "OFF"));

    use_cuda_graph = get_value_from_json_soft<bool>(j, "cuda_graph", true);
    MESSAGE_("CUDA Graph: " + std::string(use_cuda_graph ? "ON" : "OFF"));

    use_overlapped_pipeline = get_value_from_json_soft<bool>(j, "enable_overlap", false);
    MESSAGE_("Overlapped pipeline: " + std::string(use_overlapped_pipeline ? "ON" : "OFF"));

    use_holistic_cuda_graph = get_value_from_json_soft<bool>(j, "holistic_cuda_graph", false);
    if (use_holistic_cuda_graph) {
      MESSAGE_("Holistic CUDA Graph: ON");
      use_cuda_graph = false;
    }

    if (has_key_(j, "export_predictions_prefix")) {
      export_predictions_prefix = get_value_from_json<std::string>(j, "export_predictions_prefix");
      MESSAGE_("Export prediction prefix: " + export_predictions_prefix);
    } else {
      export_predictions_prefix = "";
      MESSAGE_("Export prediction: OFF");
    }

    async_mlp_wgrad = get_value_from_json_soft<bool>(j, "async_mlp_wgrad", true);
    MESSAGE_("Asynchronous Wgrad computation of MLP: " +
             std::string(async_mlp_wgrad ? "ON" : "OFF"));

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

}  // namespace HugeCTR
