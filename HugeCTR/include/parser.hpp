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

#pragma once
#include <fstream>
#include <functional>
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/data_reader.hpp"
#include "HugeCTR/include/device_map.hpp"
#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/gpu_resource.hpp"
#include "HugeCTR/include/network.hpp"
#include "nlohmann/json.hpp"

namespace HugeCTR {

/**
 * @brief The parser of configure file (in json format).
 *
 * The builder of each layer / optimizer in HugeCTR.
 * Please see User Guide to learn how to write a configure file.
 * @verbatim
 * Some Restrictions:
 *  1. Embedding should be the first element of layers.
 *  2. layers should be listed from bottom to top.
 * @endverbatim
 */
class Parser {
 private:
  nlohmann::json config_; /**< configure file. */
  int batch_size_;        /**< batch size. */
  const bool use_mixed_precision_{false};
  const float scaler_{1.f};

 public:
  /**
   * Ctor.
   * Ctor only verify the configure file, doesn't create pipeline.
   */
  Parser(const std::string& configure_file, int batch_size, bool use_mixed_precision = false,
         float scaler = 1.f)
      : batch_size_(batch_size), use_mixed_precision_(use_mixed_precision), scaler_(scaler) {
    try {
      std::ifstream file(configure_file);
      if (!file.is_open()) {
        CK_THROW_(Error_t::FileCannotOpen, "file.is_open() failed: " + configure_file);
      }
      file >> config_;
      file.close();
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
    return;
  }
  typedef long long TYPE_1;
  typedef unsigned int TYPE_2;

  /**
   * Create the pipeline, which includes data reader, embedding.
   */
  void create_pipeline(std::unique_ptr<DataReader<TYPE_1>>& data_reader,
                       std::unique_ptr<DataReader<TYPE_1>>& data_reader_eval,
                       std::vector<std::unique_ptr<Embedding<TYPE_1>>>& embedding,
                       std::vector<std::unique_ptr<Network>>& network,
                       const std::shared_ptr<GPUResourceGroup>& gpu_resource_group);

  /**
   * Create the pipeline, which includes data reader, embedding.
   */
  void create_pipeline(std::unique_ptr<DataReader<TYPE_2>>& data_reader,
                       std::unique_ptr<DataReader<TYPE_2>>& data_reader_eval,
                       std::vector<std::unique_ptr<Embedding<TYPE_2>>>& embedding,
                       std::vector<std::unique_ptr<Network>>& network,
                       const std::shared_ptr<GPUResourceGroup>& gpu_resource_group);
};

/**
 * Solver Parser.
 * This class is designed to parse the solver clause of the configure file.
 */
struct SolverParser {
  LrPolicy_t lr_policy;                        /**< the only fixed lr is supported now. */
  int display;                                 /**< the interval of loss display. */
  int max_iter;                                /**< the number of iterations for training */
  int snapshot;                                /**< the number of iterations for a snapshot */
  std::string snapshot_prefix;                 /**< naming prefix of snapshot file */
  int eval_interval;                           /**< the interval of evaluations */
  int eval_batches;                            /**< the number of batches for evaluations */
  int batchsize;                               /**< batchsize */
  std::string model_file;                      /**< name of model file */
  std::vector<std::string> embedding_files;    /**< name of embedding file */
  std::vector<int> device_list;                /**< device_list */
  std::shared_ptr<const DeviceMap> device_map; /**< device map */
  bool use_mixed_precision;
  float scaler;
  SolverParser(std::string configure_file);
};

template <typename T>
struct SparseInput {
  Tensors<T> row;
  Tensors<T> value;
  int slot_num;
  int max_feature_num_per_sample;
  SparseInput(int slot_num_in, int max_feature_num_per_sample_in)
      : slot_num(slot_num_in), max_feature_num_per_sample(max_feature_num_per_sample_in) {}
  SparseInput() {}
};

}  // namespace HugeCTR
