/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "HugeCTR/include/parser.hpp"

namespace HugeCTR {


SolverParser::SolverParser(std::string configure_file) {
  try {
    int num_procs = 1, pid = 0;
#ifdef ENABLE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif

    /* file read to json */
    nlohmann::json config;
    std::ifstream file_stream(configure_file);
    if (!file_stream.is_open()) {
      CK_THROW_(Error_t::FileCannotOpen, "file_stream.is_open() failed: " + configure_file);
    }
    file_stream >> config;
    file_stream.close();

    const std::map<std::string, LrPolicy_t> LR_POLICY = {{"fixed", LrPolicy_t::fixed}};

    /* parse the solver */
    auto j = get_json(config, "solver");

    if (has_key_(j, "seed")) {
      seed = get_value_from_json<unsigned int>(j, "seed");
    } else {
      std::random_device rd;
      seed = rd();
    }

    auto lr_policy_string = get_value_from_json<std::string>(j, "lr_policy");
    if (!find_item_in_map(lr_policy, lr_policy_string, LR_POLICY)) {
      CK_THROW_(Error_t::WrongInput, "No such poliicy: " + lr_policy_string);
    }

    display = get_value_from_json<int>(j, "display");
    max_iter = get_value_from_json<int>(j, "max_iter");
    snapshot = get_value_from_json<int>(j, "snapshot");
    batchsize = get_value_from_json<int>(j, "batchsize");
    //batchsize_eval = get_value_from_json<int>(j, "batchsize_eval");
    batchsize_eval = get_value_from_json_soft<int>(j, "batchsize_eval", batchsize);
    snapshot_prefix = get_value_from_json<std::string>(j, "snapshot_prefix");
    if (has_key_(j, "dense_model_file")) {
      model_file = get_value_from_json<std::string>(j, "dense_model_file");
    }
    FIND_AND_ASSIGN_INT_KEY(eval_interval, j);
    FIND_AND_ASSIGN_INT_KEY(eval_batches, j);
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

    if (has_key_(j, "mixed_precision")) {
      use_mixed_precision = true;
      int i_scaler = get_value_from_json<int>(j, "mixed_precision");
      if (i_scaler != 128 && i_scaler != 256 && i_scaler != 512 && i_scaler != 1024) {
        CK_THROW_(Error_t::WrongInput,
                  "Scaler of mixed_precision training should be either 128/256/512/1024");
      }
      scaler = i_scaler;
      if (pid == 0) {
        std::cout << "Mixed Precision training with scaler: " << i_scaler << " is enabled."
                  << std::endl;
      }

    } else {
      use_mixed_precision = false;
      scaler = 1.f;
    }

    auto gpu_array = get_json(j, "gpu");
    assert(device_list.empty());
    std::vector<std::vector<int>> vvgpu;
    // todo: output the device map
    if (gpu_array[0].is_array()) {
      int num_nodes = gpu_array.size();
      if (num_nodes != num_procs) {
        CK_THROW_(Error_t::WrongInput, "num_nodes != num_procs");
      } else {
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
      }
    } else {
      if (num_procs > 1) {
        CK_THROW_(Error_t::WrongInput, "num_procs > 1");
      }
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

    device_map.reset(new DeviceMap(vvgpu, pid));
    device_list = device_map->get_device_list();

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

} //namespace HugeCTR
