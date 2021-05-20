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

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <resource_manager.hpp>
#include <string>
#include <tensor2.hpp>
#include <utils.hpp>
#include <vector>

namespace HugeCTR {

namespace metrics {

using CountType = u_int32_t;
enum class RawType { Loss, Pred, Label };
enum class Type { AUC, AverageLoss };

using RawMetricMap = std::map<RawType, TensorBag2>;

class Metric {
 public:
  static std::unique_ptr<Metric> Create(const Type type, bool use_mixed_precision,
                                        int batch_size_eval, int n_batches,
                                        const std::shared_ptr<ResourceManager>& resource_manager);
  Metric();
  virtual ~Metric();
  virtual void local_reduce(int local_gpu_id, RawMetricMap raw_metrics) = 0;
  virtual void global_reduce(int n_nets) = 0;
  virtual float finalize_metric() = 0;
  virtual std::string name() const = 0;
  void set_current_batch_size(int batch_size) { current_batch_size_ = batch_size; }

 protected:
  int current_batch_size_;
};

using Metrics = std::vector<std::unique_ptr<metrics::Metric>>;

template <typename T>
class AverageLoss : public Metric {
 public:
  AverageLoss(const std::shared_ptr<ResourceManager>& resource_manager);
  ~AverageLoss() override;

  void local_reduce(int local_gpu_id, RawMetricMap raw_metrics) override;
  void global_reduce(int n_nets) override;
  float finalize_metric() override;
  std::string name() const override { return "AverageLoss"; };

 private:
  std::shared_ptr<ResourceManager> resource_manager_;
  std::vector<float> loss_local_;
  float loss_global_;
  int n_batches_;
};

class AUCStorage {
 public:
  float* d_preds() const { return ptr_preds_1_; }
  float* d_labels() const { return ptr_labels_1_; }
  CountType* d_local_bins() const { return ptr_local_bins_; }
  CountType* d_global_bins() const { return ptr_global_bins_; }
  CountType* d_global_bins_sum() const { return ptr_global_bins_sum_; }
  CountType* d_local_bins_sum() const { return ptr_local_bins_sum_; }
  int* d_pivots() const { return ptr_pivots_; }
  CountType* d_partition_offsets() const { return ptr_partition_offsets_; }
  float* d_partitioned_preds() const { return ptr_preds_2_; }
  float* d_partitioned_labels() const { return ptr_labels_2_; }
  CountType* d_all_partition_offsets() const { return ptr_all_partition_offsets_; }
  CountType* d_recv_offsets() const { return ptr_recv_offsets_; }
  float* d_presorted_preds() const { return ptr_preds_1_; }
  float* d_presorted_labels() const { return ptr_labels_1_; }
  float* d_sorted_preds() const { return ptr_preds_2_; }
  float* d_sorted_labels() const { return ptr_labels_2_; }
  float* d_tp() const { return ptr_preds_1_; }
  float* d_fp() const { return ptr_labels_1_; }
  float* d_pos_per_gpu() const { return ptr_pos_per_gpu_; }
  float* d_neg_per_gpu() const { return ptr_neg_per_gpu_; }
  float* d_tpr() const { return ptr_preds_2_; }
  float* d_fpr() const { return ptr_labels_2_; }
  CountType* d_identical_pred_starts() const { return ptr_identical_pred_starts_; }
  CountType* d_identical_pred_lengths() const { return ptr_identical_pred_lengths_; }
  int* d_num_identical_segments() const { return ptr_num_identical_segments_; }
  float* d_halo_tpr() const { return ptr_halo_tpr_; }
  float* d_halo_fpr() const { return ptr_halo_fpr_; }
  CountType* d_tp_offsets() const { return ptr_tp_offsets_; }
  CountType* d_fp_offsets() const { return ptr_fp_offsets_; }
  float* d_auc() const { return ptr_auc_; }

  void* d_workspace() const { return workspace_; }

  size_t& temp_storage_bytes() { return allocated_temp_storage_; }

  void alloc_main(size_t num_local_samples, size_t num_bins, size_t num_partitions,
                  size_t num_global_gpus);
  void realloc_redistributed(size_t num_redistributed_samples, cudaStream_t stream);
  void realloc_workspace(size_t temp_storage);
  void free_all();

 private:
  const float reallocate_factor_ = 1.2f;
  size_t allocated_temp_storage_ = 0;
  size_t num_allocated_redistributed_ = 0;

  float* ptr_preds_1_ = nullptr;
  float* ptr_labels_1_ = nullptr;
  float* ptr_preds_2_ = nullptr;
  float* ptr_labels_2_ = nullptr;
  CountType* ptr_local_bins_ = nullptr;
  CountType* ptr_global_bins_ = nullptr;
  CountType* ptr_global_bins_sum_ = nullptr;
  CountType* ptr_local_bins_sum_ = nullptr;
  int* ptr_pivots_ = nullptr;
  CountType* ptr_partition_offsets_ = nullptr;
  CountType* ptr_all_partition_offsets_ = nullptr;
  CountType* ptr_recv_offsets_ = nullptr;
  float* ptr_pos_per_gpu_ = nullptr;
  float* ptr_neg_per_gpu_ = nullptr;
  CountType* ptr_identical_pred_starts_ = nullptr;
  CountType* ptr_identical_pred_lengths_ = nullptr;
  int* ptr_num_identical_segments_ = nullptr;
  float* ptr_halo_tpr_ = nullptr;
  float* ptr_halo_fpr_ = nullptr;
  CountType* ptr_tp_offsets_ = nullptr;
  CountType* ptr_fp_offsets_ = nullptr;
  float* ptr_auc_ = nullptr;

  void* workspace_ = nullptr;

  void realloc_ptr(void** ptr, size_t old_size, size_t new_size, cudaStream_t stream);
};

template <typename T>
class AUC : public Metric {
 public:
  using PredType = T;
  using LabelType = float;
  AUC(int batch_size_per_gpu, int n_batches,
      const std::shared_ptr<ResourceManager>& resource_manager);
  ~AUC() override;

  void local_reduce(int local_gpu_id, RawMetricMap raw_metrics) override;
  void global_reduce(int n_nets) override;
  float finalize_metric() override;
  std::string name() const override { return "AUC"; };

  // Public in order to use device lambda
  float _finalize_metric_per_gpu(int device_id);

 private:
  void warm_up(size_t num_local_samples);

  const float pred_min_ = 0.0f;
  const float pred_max_ = 1.0f;
  const int num_bins_per_gpu_ = 10000;

  std::shared_ptr<ResourceManager> resource_manager_;

  int n_batches_;
  int num_local_gpus_;
  int num_global_gpus_;
  int batch_size_per_gpu_;
  int num_bins_;
  int num_partitions_;
  size_t num_total_samples_;

  std::vector<size_t> offsets_;
  std::vector<AUCStorage> storage_;
};

}  // namespace metrics

}  // namespace HugeCTR
