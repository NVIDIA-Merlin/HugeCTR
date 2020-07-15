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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "HugeCTR/include/gpu_resource.hpp"
#include "HugeCTR/include/tensor.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

namespace metrics {

enum class RawType { Loss = 0, Pred = 1, Label = 2 };
enum class Type { AUC = 0, AverageLoss = 1 };

using RawMetricMap = std::map<RawType, std::shared_ptr<ITensor>>;

class Metric {
 public:
  static std::unique_ptr<Metric> Create(const Type type, bool use_mixed_precision,
                                 int batch_size_eval, int n_batches,
                                 std::shared_ptr<GPUResourceGroup> gpu_resource_group);
  Metric();
  virtual ~Metric();
  virtual void local_reduce(RawMetricMap raw_metrics) = 0;
  virtual void global_reduce(int n_nets) = 0;
  virtual float finalize_metric() = 0;
  virtual std::string name() const = 0;
  void set_current_batch_size(int batch_size) { current_batch_size_ = batch_size; }

 protected:
  int num_procs_;
  int pid_;
  int current_batch_size_;
};

using Metrics = std::vector<std::unique_ptr<metrics::Metric>>;

template <typename T>
class AverageLoss : public Metric {
 public:
  using TensorType = Tensor<T>;
  AverageLoss(int num_gpus);
  ~AverageLoss() override;

  void local_reduce(RawMetricMap raw_metrics) override;
  void global_reduce(int n_nets) override;
  float finalize_metric() override;
  std::string name() const override { return "AverageLoss"; };

 private:
  std::vector<float> loss_local_;
  float loss_global_;
  int n_batches_;
};

template <typename T>
class AUC : public Metric {
 public:
  using PredType = T;
  using LabelType = float;
  AUC(int batch_size_per_gpu, int n_batches, int root_device_id, int num_gpus,
      std::shared_ptr<GPUResourceGroup> gpu_resource_group);
  ~AUC() override;

  void local_reduce(RawMetricMap raw_metrics) override;
  void global_reduce(int n_nets) override;
  float finalize_metric() override;
  std::string name() const override { return "AUC"; };

 private:
  void set_max_temp_storage_bytes(size_t& new_val);
  float* d_pred() const { return (float*)temp0_; }
  float* d_label() const { return (float*)temp1_; }
  float* d_pred_sort() const { return (float*)temp2_; }
  float* d_label_sort() const { return (float*)temp3_; }
  float* d_inc_sum() const { return (float*)temp0_; }
  float* tpr() const { return (float*)temp2_; }
  int* d_flag_inc_sum() const { return (int*)temp0_; }
  int* d_unique_index() const { return (int*)temp1_; }
  float* fpr() const { return (float*)temp3_; }
  float* d_auc() const { return (float*)temp0_; }

  void num_active_gpu_and_r(int& num_active_gpu, int& r);

  int batch_size_per_gpu_;
  int n_batches_;
  int root_device_id_;
  int num_gpus_;
  int offset_;
  void* temp0_;
  void* temp1_;
  void* temp2_;
  void* temp3_;
  void* workspace_;
  size_t temp_storage_bytes_;
  int num_sms_;
  std::shared_ptr<GPUResourceGroup> gpu_resource_group_;
};

}  // namespace metrics

}  // namespace HugeCTR
