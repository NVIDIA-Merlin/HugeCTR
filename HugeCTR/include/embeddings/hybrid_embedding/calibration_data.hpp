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

#include <cuda_runtime.h>

#include <common.hpp>
#include <embeddings/hybrid_embedding/data.hpp>
#include <embeddings/hybrid_embedding/statistics.hpp>
#include <embeddings/hybrid_embedding/utils.hpp>
#include <tensor2.hpp>
#include <vector>

namespace HugeCTR {

namespace hybrid_embedding {

// depends on : Statistics, Data

///
/// This class contains the calibrated measurements for all-to-all and all-reduce
/// for different data sizes. Each calibration consists of two arrays,
/// ._data_size array and the ._time array which represent a mapping.
///
/// This class will be executed on the cpu instead of the gpu if no
/// gpu memory is allocated for the calibration data.
struct CalibrationData {
  CalibrationData(size_t num_nodes_in, double p_dup_max_in, double max_all_reduce_bandwidth_in,
                  double max_all_to_all_bandwidth_in, double efficiency_bandwidth_ratio_in)
      : num_nodes(num_nodes_in),
        p_dup_max(p_dup_max_in),
        max_all_reduce_bandwidth(max_all_reduce_bandwidth_in),
        max_all_to_all_bandwidth(max_all_to_all_bandwidth_in),
        efficiency_bandwidth_ratio(efficiency_bandwidth_ratio_in){
            // TBD
        };
  ~CalibrationData() {}

  size_t num_nodes;

  // Calibration all-to-all :
  //   the following two arrays map data sizes to all-to-all times / latencies.
  std::vector<double> h_all_to_all_data_size;
  std::vector<double> h_all_to_all_times;
  Tensor2<float> all_to_all_data_size;  // data size of message per gpu
  Tensor2<float> all_to_all_times;      // calibrated all-to-all times

  // Calibration all-reduce :
  //   the following two arrays map data sizes to all-to-all times / latencies.
  std::vector<double> h_all_reduce_data_size;
  std::vector<double> h_all_reduce_times;
  Tensor2<float> all_reduce_data_size;  // data size of message per gpu
  Tensor2<float> all_reduce_times;      // calibrated all-reduce times

  // Alternative calibration: (if no calibration provided)
  //   the threshold for frequent categories is calculated from maximum bandwidths
  //   for the all-reduce and all-to-all respectively.
  //   This approximation assumes that the communications are bandwidth limited.
  double p_dup_max;
  double max_all_reduce_bandwidth;  // algorithm bandwidth all-reduce [data size message per gpu in
                                    // bytes / sec]
  double max_all_to_all_bandwidth;  // algorithm bandwidth all-to-all [data size message per gpu in
                                    // bytes / sec]
  double efficiency_bandwidth_ratio;
  // cpu functions
  double interpolate(const std::vector<double> &calibrated_data_size,
                     const std::vector<double> &calibrated_times,
                     const std::vector<double> &data_size,
                     std::vector<double> &communication_times);
  double interpolate_all_reduce(const std::vector<double> &data_size,
                                std::vector<double> &communication_times);
  double interpolate_all_to_all(const std::vector<double> &data_size,
                                std::vector<double> &communication_times);

  // gpu functions
  void interpolate(const Tensor2<float> &calibrated_data_size,
                   const Tensor2<float> &calibrated_times, const Tensor2<float> &data_size,
                   Tensor2<float> &communication_times);
  void interpolate_all_reduce(const Tensor2<float> &data_size, Tensor2<float> &communication_times);
  void interpolate_all_to_all(const Tensor2<float> &data_size, Tensor2<float> &communication_times);
};

template <typename dtype>
class ModelInitializationFunctors {
 public:
  static double calculate_threshold(const CommunicationType communication_type, double p_dup_max,
                                    double all_to_all_bandwidth, double all_reduce_bandwidth,
                                    double efficiency_bandwidth_ratio, size_t num_nodes,
                                    size_t batch_size, size_t num_networks, size_t num_iterations,
                                    size_t num_tables);
  static dtype calculate_num_frequent_categories(const CommunicationType &communication_type,
                                                 const size_t num_networks,
                                                 const CalibrationData &calibration,
                                                 const Statistics<dtype> &statistics,
                                                 const Data<dtype> &data, dtype *d_num_frequent,
                                                 cudaStream_t stream);
  static double calculate_frequent_probability(const Statistics<dtype> &statistics,
                                               const dtype num_frequent,
                                               uint32_t *d_total_frequent_count,
                                               cudaStream_t stream);
};

}  // namespace hybrid_embedding

}  // namespace HugeCTR