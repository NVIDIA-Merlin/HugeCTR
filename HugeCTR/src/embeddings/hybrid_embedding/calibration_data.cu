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

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <utils.cuh>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/calibration_data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

namespace calibration_data_kernels {

template <typename CountsT, typename ThresholdT, typename IdxT>
__global__ void binary_threshold_search(const CountsT *__restrict__ counts, ThresholdT threshold,
                                        IdxT *out_idx, IdxT n_elem) {
  if (threadIdx.x == 0) {
    IdxT start = 0;
    IdxT end = n_elem;
    while (start < end) {
      IdxT mid = (start + end) / 2;
      CountsT count = counts[mid];

      if (count >= threshold)
        start = mid + 1;
      else
        end = mid;
    }

    *out_idx = start;
  }
}

template <typename CountsT>
__global__ void sum_counts(const CountsT *__restrict__ counts, CountsT *result, size_t n_elem) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  CountsT val = 0;
  if (tid < n_elem) {
    val = counts[tid];
  }
  CountsT local_res = blockReduceSum(val);
  if (threadIdx.x == 0) {
    atomicAdd(result, local_res);
  }
}

}  // namespace calibration_data_kernels

///
/// interpolate data_size using the two calibration data
///   calibrated_data_size, calibrated_times
///   return communication_times
///
void CalibrationData::interpolate(const Tensor2<float> &calibrated_data_size,
                                  const Tensor2<float> &calibrated_times,
                                  const Tensor2<float> &data_size,
                                  Tensor2<float> &communication_times) {
  // TODO: implement this
}

///
/// Convenience function for interpolating all-to-all communication times from
/// calibrated data
///
void CalibrationData::interpolate_all_reduce(const Tensor2<float> &data_size,
                                             Tensor2<float> &communication_times) {
  interpolate(all_reduce_data_size, all_reduce_times, data_size, communication_times);
}

///
/// Convenience function for interpolating all-to-all communication times from
/// calibrated data
///
void CalibrationData::interpolate_all_to_all(const Tensor2<float> &data_size,
                                             Tensor2<float> &communication_times) {
  interpolate(all_to_all_data_size, all_to_all_times, data_size, communication_times);
}

// Calculate threshold such that for the worst case distribution there will
// be one duplication per network on average
template <typename dtype>
double ModelInitializationFunctors<dtype>::calculate_threshold(
    const CommunicationType communication_type, double p_dup_max, double all_to_all_bandwidth,
    double all_reduce_bandwidth, double efficiency_bandwidth_ratio, size_t num_nodes,
    size_t batch_size, size_t num_networks, size_t num_iterations, size_t num_tables) {
  float count_threshold = 1.f;

  // for NVLink capture effectively all duplications with number of categories
  double M = (double)batch_size / (double)num_networks;
  // double p_dup_max = 1.0 / 100.;  // maximum 1 % of samples the category will be duplicated
  switch (communication_type) {
    case CommunicationType::IB_NVLink:
    case CommunicationType::IB_NVLink_Hier:
      count_threshold = (double)num_iterations * (double)num_nodes * all_to_all_bandwidth /
                        all_reduce_bandwidth * efficiency_bandwidth_ratio * (double)num_nodes /
                        ((double)num_nodes - 1.);
      break;
    case CommunicationType::NVLink_SingleNode:
      // count threshold such that the probability of duplication is less than p_dup_max
      //   even if there are batch size number of categories that occur more often,
      //   there will be a duplication at most once every iteration per gpu
      //
      // p_duplication(category) \approx 1/2 M (M-1) \left( \frac{count}{batch_size x
      // num_iterations} \right)^2
      count_threshold = (double)((double)batch_size * (double)num_iterations *
                                 sqrt(2.0 * p_dup_max / (M * (M - 1))));
      break;
    default:
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Unknown communication type, expecting IB_NVLink or NVLink");
  }

  return count_threshold;
}

///
/// Calculate the number of frequent categories from data
///
template <typename dtype>
dtype ModelInitializationFunctors<dtype>::calculate_num_frequent_categories(
    const CommunicationType &communication_type, const size_t num_networks,
    const CalibrationData &calibration, const Statistics<dtype> &statistics,
    const Data<dtype> &data, dtype *d_num_frequent, cudaStream_t stream) {
  dtype num_frequent;
  dtype num_top_categories = (dtype)statistics.num_unique_categories;

  if (calibration.all_to_all_times.get_size_in_bytes() > 0) {
    // calibration is given, perform fully optimized hybrid model
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "initialization hybrid model from communication calibration not available yet");
  } else {
    size_t num_nodes = calibration.num_nodes;
    size_t batch_size = data.batch_size;
    size_t num_iterations = data.num_iterations;
    size_t num_tables = data.table_sizes.size();

    // Use threshold to determine number of frequent categories,
    // calculates optimal number of frequent categories when the all-to-all
    // and all-reduce are both bandwidth limited.
    double count_threshold = ModelInitializationFunctors::calculate_threshold(
        communication_type, calibration.p_dup_max, calibration.max_all_to_all_bandwidth,
        calibration.max_all_reduce_bandwidth, calibration.efficiency_bandwidth_ratio, num_nodes,
        batch_size, num_networks, num_iterations, num_tables);

    calibration_data_kernels::binary_threshold_search<<<1, 1, 0, stream>>>(
        statistics.counts_sorted.get_ptr(), count_threshold, d_num_frequent,
        (dtype)num_top_categories);

    HCTR_LIB_THROW(cudaMemcpyAsync(&num_frequent, d_num_frequent, sizeof(dtype),
                                   cudaMemcpyDeviceToHost, stream));
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  }
  if (num_frequent > 0) {
    num_frequent = ((num_frequent - 1) / num_networks + 1) * num_networks;
  }
  if (num_frequent > num_top_categories) {
    num_frequent -= num_networks;
  }
  return num_frequent;
}

///
/// Calculate the number of frequent categories from data
///
template <typename dtype>
double ModelInitializationFunctors<dtype>::calculate_frequent_probability(
    const Statistics<dtype> &statistics, const dtype num_frequent, uint32_t *d_total_frequent_count,
    cudaStream_t stream) {
  uint32_t total_frequent_count;

  HCTR_LIB_THROW(cudaMemsetAsync(d_total_frequent_count, 0, sizeof(uint32_t), stream));
  calibration_data_kernels::sum_counts<<<num_frequent / 128 + 1, 128, 0, stream>>>(
      statistics.counts_sorted.get_ptr(), d_total_frequent_count, num_frequent);
  HCTR_LIB_THROW(cudaMemcpyAsync(&total_frequent_count, d_total_frequent_count, sizeof(dtype),
                                 cudaMemcpyDeviceToHost, stream));
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  return (double)total_frequent_count / (double)statistics.num_samples;
}

template class ModelInitializationFunctors<uint32_t>;
template class ModelInitializationFunctors<long long>;
}  // namespace hybrid_embedding

}  // namespace HugeCTR
