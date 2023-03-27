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

#include <condition_variable>
#include <core23/tensor.hpp>
#include <general_buffer2.hpp>
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
enum class Type { AUC, AverageLoss, HitRate, NDCG, SMAPE };

using RawMetricMap = std::map<RawType, TensorBag2>;
using Core23RawMetricMap = std::map<RawType, core23::Tensor>;
using MultiLossMetricMap = std::map<std::string, RawMetricMap>;
using Core23MultiLossMetricMap = std::map<std::string, Core23RawMetricMap>;

void get_raw_metric_as_host_float_tensor(Core23RawMetricMap metric_map, RawType raw_type,
                                         bool mixed_precision, float* rst, size_t num);

void get_raw_metric_as_host_float_tensor(RawMetricMap metric_map, RawType raw_type,
                                         bool mixed_precision, float* rst, size_t num);

class Metric {
 public:
  static std::unique_ptr<Metric> Create(const Type type, bool use_mixed_precision,
                                        int batch_size_eval, int n_batches, int label_dim,
                                        const std::shared_ptr<ResourceManager>& resource_manager);
  Metric();
  virtual ~Metric();
  virtual void local_reduce(int local_gpu_id, RawMetricMap raw_metrics) = 0;
  virtual void local_reduce(int local_gpu_id, Core23RawMetricMap raw_metrics) = 0;
  virtual void global_reduce(int n_nets) = 0;
  virtual float finalize_metric() = 0;
  virtual std::vector<float> get_per_class_metric() const {
    HCTR_CHECK_HINT(false, "Not implemented");
    return std::vector<float>(0.0, 1);
  }
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
  void local_reduce(int local_gpu_id, Core23RawMetricMap raw_metrics) override;
  void global_reduce(int n_nets) override;
  float finalize_metric() override;
  std::string name() const override { return "AverageLoss"; };

 private:
  std::shared_ptr<ResourceManager> resource_manager_;
  std::vector<float*> loss_local_;
  float loss_global_;
  int n_batches_;
};

template <typename T>
class HitRate : public Metric {
 public:
  using PredType = T;
  using LabelType = float;
  HitRate(int batch_size_per_gpu, const std::shared_ptr<ResourceManager>& resource_manager);
  ~HitRate() override;

  void local_reduce(int local_gpu_id, RawMetricMap raw_metrics) override;
  void local_reduce(int local_gpu_id, Core23RawMetricMap raw_metrics) override;
  void global_reduce(int n_nets) override;
  float finalize_metric() override;
  std::string name() const override { return "HitRate"; };

 private:
  void free_all();

  std::shared_ptr<ResourceManager> resource_manager_;
  int batch_size_per_gpu_;
  int n_batches_;
  int num_local_gpus_;

  std::vector<int*> checked_count_;
  std::vector<int*> hit_count_;
  std::vector<int> checked_local_;
  std::vector<int> hits_local_;
  int checked_global_;
  int hits_global_;
};

template <typename T>
class SMAPE : public Metric {
 public:
  using PredType = T;
  using LabelType = float;
  SMAPE(int batch_size_per_gpu, const std::shared_ptr<ResourceManager>& resource_manager);
  ~SMAPE() override;

  void local_reduce(int local_gpu_id, RawMetricMap raw_metrics) override;
  void local_reduce(int local_gpu_id, Core23RawMetricMap raw_metrics) override;
  void global_reduce(int n_nets) override;
  float finalize_metric() override;
  std::string name() const override { return "SMAPE"; }

 private:
  void free_all();

  std::shared_ptr<ResourceManager> resource_manager_;
  int batch_size_per_gpu_;
  int n_batches_;
  int num_local_gpus_;

  std::vector<float*> error_;       // Device variable
  std::vector<int> checked_local_;  // Host variable
  std::vector<float> error_local_;  // Host variable
  int checked_global_;
  float error_global_;
};

enum class ReallocType_t { NO_COPY, MMAP, DEFAULT };

template <typename T, ReallocType_t U>
class ReallocBuffer {
 public:
  ReallocBuffer();
  ~ReallocBuffer();

  void realloc(size_t new_num_elements, cudaStream_t stream = 0);
  T* get_ptr() { return ptr_; }
  void init_access_desc(const std::vector<CUmemAccessDesc>* access_desc);

 private:
  size_t num_elements_;
  T* ptr_;

  // Explicit virtual memory management
  CUmemAllocationProp prop_;
  const std::vector<CUmemAccessDesc>* access_desc_;  // Virtual memory access descriptor
  std::vector<std::pair<CUmemGenericAllocationHandle, size_t>> pm_handles_;
  std::vector<std::pair<CUdeviceptr, size_t>> vm_ranges_;    // Virtual allocation info
  std::vector<std::pair<CUdeviceptr, size_t>> mmap_ranges_;  // Memmap VA->PA info

  size_t chunk_size_;
  void get_aligned_size(size_t& size) const {
    size = ((size + chunk_size_ - 1) / chunk_size_) * chunk_size_;
  }
  void realloc_ptr_mmap(void** ptr, size_t old_size, size_t new_size);
  void release_mmap_memory();
};

class AUCStorageOld {
 public:
  float* d_class_preds(size_t class_id) { return class_preds_[class_id].get_ptr(); }
  float* d_class_labels(size_t class_id) { return class_labels_[class_id].get_ptr(); }

  float* d_lr_unsorted_preds() { return lr_unsorted_preds_.get_ptr(); }
  float* d_lr_sorted_preds() { return lr_sorted_preds_.get_ptr(); }
  float* d_lr_sorted_labels() { return lr_sorted_labels_.get_ptr(); }
  int* d_lr_class_ids() { return lr_class_ids_.get_ptr(); }
  int* d_lr_sorted_class_ids() { return lr_sorted_class_ids_.get_ptr(); }

  void* d_workspace(size_t stream_id) { return workspace_[stream_id].get_ptr(); }
  size_t& temp_storage_bytes(size_t stream_id) { return allocated_temp_storage_[stream_id]; }

  void alloc_main(size_t num_local_samples, size_t num_bins, size_t num_partitions,
                  size_t num_global_gpus, size_t label_dim, size_t num_streams,
                  const std::vector<int>& peers, cudaStream_t stream);
  void realloc_redistributed(size_t num_redistributed_samples, cudaStream_t stream,
                             size_t stream_id);
  void realloc_workspace(size_t temp_storage, size_t stream_id);
  bool realloc_local_reduce_storage(size_t input_size);
  void free_all();

 private:
  const float reallocate_factor_ = 1.2f;
  std::vector<size_t> allocated_temp_storage_;
  std::vector<size_t> num_allocated_redistributed_;
  std::vector<CUmemAccessDesc> access_desc_;  // Access descriptors used by ReallocBuffers

  // Raw per-class data
  size_t num_classes_ = 1;
  std::vector<Tensor2<float>> class_preds_;
  std::vector<Tensor2<float>> class_labels_;

  // Local reduce storage
  size_t allocated_lr_input_size_ = 0;
  ReallocBuffer<float, ReallocType_t::NO_COPY> lr_unsorted_preds_;
  ReallocBuffer<float, ReallocType_t::NO_COPY> lr_sorted_preds_;
  ReallocBuffer<float, ReallocType_t::NO_COPY> lr_sorted_labels_;
  ReallocBuffer<int, ReallocType_t::NO_COPY> lr_class_ids_;
  ReallocBuffer<int, ReallocType_t::NO_COPY> lr_sorted_class_ids_;

  // Workspace for CUB functions
  std::vector<ReallocBuffer<int8_t, ReallocType_t::NO_COPY>> workspace_;

  struct FinalizeStorage {
    ReallocBuffer<float, ReallocType_t::MMAP> preds_1_;
    ReallocBuffer<float, ReallocType_t::MMAP> labels_1_;
    ReallocBuffer<float, ReallocType_t::MMAP> preds_2_;
    ReallocBuffer<float, ReallocType_t::MMAP> labels_2_;
    ReallocBuffer<CountType, ReallocType_t::MMAP> identical_pred_starts_;
    ReallocBuffer<CountType, ReallocType_t::MMAP> identical_pred_lengths_;

    Tensor2<CountType> local_bins_;
    Tensor2<CountType> global_bins_;
    Tensor2<CountType> local_bins_sum_;
    Tensor2<CountType> global_bins_sum_;
    Tensor2<CountType> partition_offsets_;
    Tensor2<int> pivots_;
    Tensor2<CountType> all_partition_offsets_;
    Tensor2<CountType> recv_offsets_;
    Tensor2<float> pos_per_gpu_;
    Tensor2<float> neg_per_gpu_;
    Tensor2<int> num_identical_segments_;
    Tensor2<float> halo_tpr_;
    Tensor2<float> halo_fpr_;
    Tensor2<CountType> tp_offsets_;
    Tensor2<CountType> fp_offsets_;
    Tensor2<float> auc_;

    size_t num_redistributed_samples;
    std::vector<size_t> all_num_redistributed_samples;

    float* d_labels() { return labels_1_.get_ptr(); }
    float* d_preds() { return preds_1_.get_ptr(); }
    CountType* d_local_bins() { return local_bins_.get_ptr(); }
    CountType* d_global_bins() { return global_bins_.get_ptr(); }
    CountType* d_local_bins_sum() { return local_bins_sum_.get_ptr(); }
    CountType* d_global_bins_sum() { return global_bins_sum_.get_ptr(); }
    int* d_pivots() { return pivots_.get_ptr(); }
    CountType* d_partition_offsets() { return partition_offsets_.get_ptr(); }
    float* d_partitioned_preds() { return preds_2_.get_ptr(); }
    float* d_partitioned_labels() { return labels_2_.get_ptr(); }
    CountType* d_all_partition_offsets() { return all_partition_offsets_.get_ptr(); }
    CountType* d_recv_offsets() { return recv_offsets_.get_ptr(); }
    float* d_presorted_preds() { return preds_1_.get_ptr(); }
    float* d_presorted_labels() { return labels_1_.get_ptr(); }
    float* d_sorted_preds() { return preds_2_.get_ptr(); }
    float* d_sorted_labels() { return labels_2_.get_ptr(); }
    float* d_tp() { return preds_1_.get_ptr(); }
    float* d_fp() { return labels_1_.get_ptr(); }
    float* d_pos_per_gpu() { return pos_per_gpu_.get_ptr(); }
    float* d_neg_per_gpu() { return neg_per_gpu_.get_ptr(); }
    float* d_tpr() { return preds_2_.get_ptr(); }
    float* d_fpr() { return labels_2_.get_ptr(); }
    CountType* d_identical_pred_starts() { return identical_pred_starts_.get_ptr(); }
    CountType* d_identical_pred_lengths() { return identical_pred_lengths_.get_ptr(); }
    int* d_num_identical_segments() { return num_identical_segments_.get_ptr(); }
    float* d_halo_tpr() { return halo_tpr_.get_ptr(); }
    float* d_halo_fpr() { return halo_fpr_.get_ptr(); }
    CountType* d_tp_offsets() { return tp_offsets_.get_ptr(); }
    CountType* d_fp_offsets() { return fp_offsets_.get_ptr(); }
    float* d_auc() { return auc_.get_ptr(); }
  };

  // Intermediate storage needed in finalize metric, one element per stream
  std::vector<FinalizeStorage> finalize_storage_;

 public:
  FinalizeStorage& fst(size_t stream_id) { return finalize_storage_[stream_id]; }
};

class AUCStorage {
 public:
  float* d_class_preds(size_t class_id) { return class_preds_[class_id].data<float>(); }
  float* d_class_labels(size_t class_id) { return class_labels_[class_id].data<float>(); }

  float* d_lr_unsorted_preds() { return lr_unsorted_preds_.get_ptr(); }
  float* d_lr_sorted_preds() { return lr_sorted_preds_.get_ptr(); }
  float* d_lr_sorted_labels() { return lr_sorted_labels_.get_ptr(); }
  int* d_lr_class_ids() { return lr_class_ids_.get_ptr(); }
  int* d_lr_sorted_class_ids() { return lr_sorted_class_ids_.get_ptr(); }

  void* d_workspace(size_t stream_id) { return workspace_[stream_id].get_ptr(); }
  size_t& temp_storage_bytes(size_t stream_id) { return allocated_temp_storage_[stream_id]; }

  void alloc_main(size_t num_local_samples, size_t num_bins, size_t num_partitions,
                  size_t num_global_gpus, size_t label_dim, size_t num_streams,
                  const std::vector<int>& peers, cudaStream_t stream);
  void realloc_redistributed(size_t num_redistributed_samples, cudaStream_t stream,
                             size_t stream_id);
  void realloc_workspace(size_t temp_storage, size_t stream_id);
  bool realloc_local_reduce_storage(size_t input_size);
  void free_all();

 private:
  const float reallocate_factor_ = 1.2f;
  std::vector<size_t> allocated_temp_storage_;
  std::vector<size_t> num_allocated_redistributed_;
  std::vector<CUmemAccessDesc> access_desc_;  // Access descriptors used by ReallocBuffers

  // Raw per-class data
  size_t num_classes_ = 1;
  std::vector<core23::Tensor> class_preds_;
  std::vector<core23::Tensor> class_labels_;

  // Local reduce storage
  size_t allocated_lr_input_size_ = 0;
  ReallocBuffer<float, ReallocType_t::NO_COPY> lr_unsorted_preds_;
  ReallocBuffer<float, ReallocType_t::NO_COPY> lr_sorted_preds_;
  ReallocBuffer<float, ReallocType_t::NO_COPY> lr_sorted_labels_;
  ReallocBuffer<int, ReallocType_t::NO_COPY> lr_class_ids_;
  ReallocBuffer<int, ReallocType_t::NO_COPY> lr_sorted_class_ids_;

  // Workspace for CUB functions
  std::vector<ReallocBuffer<int8_t, ReallocType_t::NO_COPY>> workspace_;

  struct FinalizeStorage {
    ReallocBuffer<float, ReallocType_t::MMAP> preds_1_;
    ReallocBuffer<float, ReallocType_t::MMAP> labels_1_;
    ReallocBuffer<float, ReallocType_t::MMAP> preds_2_;
    ReallocBuffer<float, ReallocType_t::MMAP> labels_2_;
    ReallocBuffer<CountType, ReallocType_t::MMAP> identical_pred_starts_;
    ReallocBuffer<CountType, ReallocType_t::MMAP> identical_pred_lengths_;

    core23::Tensor local_bins_;
    core23::Tensor global_bins_;
    core23::Tensor local_bins_sum_;
    core23::Tensor global_bins_sum_;
    core23::Tensor partition_offsets_;
    core23::Tensor pivots_;
    core23::Tensor all_partition_offsets_;
    core23::Tensor recv_offsets_;
    core23::Tensor pos_per_gpu_;
    core23::Tensor neg_per_gpu_;
    core23::Tensor num_identical_segments_;
    core23::Tensor halo_tpr_;
    core23::Tensor halo_fpr_;
    core23::Tensor tp_offsets_;
    core23::Tensor fp_offsets_;
    core23::Tensor auc_;

    size_t num_redistributed_samples;
    std::vector<size_t> all_num_redistributed_samples;

    float* d_labels() { return labels_1_.get_ptr(); }
    float* d_preds() { return preds_1_.get_ptr(); }
    CountType* d_local_bins() { return local_bins_.data<CountType>(); }
    CountType* d_global_bins() { return global_bins_.data<CountType>(); }
    CountType* d_local_bins_sum() { return local_bins_sum_.data<CountType>(); }
    CountType* d_global_bins_sum() { return global_bins_sum_.data<CountType>(); }
    int* d_pivots() { return pivots_.data<int>(); }
    CountType* d_partition_offsets() { return partition_offsets_.data<CountType>(); }
    float* d_partitioned_preds() { return preds_2_.get_ptr(); }
    float* d_partitioned_labels() { return labels_2_.get_ptr(); }
    CountType* d_all_partition_offsets() { return all_partition_offsets_.data<CountType>(); }
    CountType* d_recv_offsets() { return recv_offsets_.data<CountType>(); }
    float* d_presorted_preds() { return preds_1_.get_ptr(); }
    float* d_presorted_labels() { return labels_1_.get_ptr(); }
    float* d_sorted_preds() { return preds_2_.get_ptr(); }
    float* d_sorted_labels() { return labels_2_.get_ptr(); }
    float* d_tp() { return preds_1_.get_ptr(); }
    float* d_fp() { return labels_1_.get_ptr(); }
    float* d_pos_per_gpu() { return pos_per_gpu_.data<float>(); }
    float* d_neg_per_gpu() { return neg_per_gpu_.data<float>(); }
    float* d_tpr() { return preds_2_.get_ptr(); }
    float* d_fpr() { return labels_2_.get_ptr(); }
    CountType* d_identical_pred_starts() { return identical_pred_starts_.get_ptr(); }
    CountType* d_identical_pred_lengths() { return identical_pred_lengths_.get_ptr(); }
    int* d_num_identical_segments() { return num_identical_segments_.data<int>(); }
    float* d_halo_tpr() { return halo_tpr_.data<float>(); }
    float* d_halo_fpr() { return halo_fpr_.data<float>(); }
    CountType* d_tp_offsets() { return tp_offsets_.data<CountType>(); }
    CountType* d_fp_offsets() { return fp_offsets_.data<CountType>(); }
    float* d_auc() { return auc_.data<float>(); }
  };

  // Intermediate storage needed in finalize metric, one element per stream
  std::vector<FinalizeStorage> finalize_storage_;

 public:
  FinalizeStorage& fst(size_t stream_id) { return finalize_storage_[stream_id]; }
};

template <typename T>
class AUC : public Metric {
 public:
  using PredType = T;
  using LabelType = float;
  AUC(int batch_size_per_gpu, int n_batches, int label_dim,
      const std::shared_ptr<ResourceManager>& resource_manager, bool use_old_tensor = true);
  ~AUC() override;

  void local_reduce(int local_gpu_id, RawMetricMap raw_metrics) override;
  void local_reduce(int local_gpu_id, Core23RawMetricMap raw_metrics) override;
  void global_reduce(int n_nets) override;
  float finalize_metric() override;
  std::string name() const override { return "AUC"; };
  std::vector<float> get_per_class_metric() const { return per_class_aucs_; }

  // Public in order to use device lambda
  void run_finalize_step(float* d_preds, float* d_labels, int local_id, size_t num_local_samples,
                         size_t stream_id, size_t step_id);

 private:
  void warm_up(size_t num_local_samples);

  float finalize_metric_per_gpu(int device_id);
  float finalize_class_metric(float* preds, float* labels, int local_id, size_t num_local_samples);
  float finalize_class_metric_multi_stream(int local_id, int num_local_samples);

  const float pred_min_ = 0.0f;
  const float pred_max_ = 1.0f;
  const int num_bins_per_gpu_ = 10000;
  const size_t num_finalize_steps_ = 3;
  const size_t num_classes_;

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
  std::vector<AUCStorageOld> storage_old_;
  bool use_old_tensor_;
  std::vector<std::vector<cudaStream_t>> streams_;
  std::vector<float> per_class_aucs_;
};

class NDCGStorageOld {
 public:
  void alloc_main(size_t num_local_samples, size_t num_bins, size_t num_partitions,
                  size_t num_global_gpus, const std::vector<int>& peers);
  void realloc_redistributed(size_t num_redistributed_samples, cudaStream_t stream);
  void realloc_workspace(size_t temp_storage);
  void* d_workspace() { return workspace_.get_ptr(); }
  size_t& temp_storage_bytes() { return allocated_temp_storage_; }

 private:
  const float reallocate_factor_ = 1.2f;
  size_t allocated_temp_storage_;
  size_t num_allocated_redistributed_;
  std::vector<CUmemAccessDesc> access_desc_;  // Access descriptors used by ReallocBuffers

  // Workspace for CUB functions
  ReallocBuffer<int8_t, ReallocType_t::NO_COPY> workspace_;

  ReallocBuffer<float, ReallocType_t::MMAP> preds_1_;
  ReallocBuffer<float, ReallocType_t::MMAP> labels_1_;
  ReallocBuffer<float, ReallocType_t::MMAP> preds_2_;
  ReallocBuffer<float, ReallocType_t::MMAP> labels_2_;
  ReallocBuffer<float, ReallocType_t::MMAP> scaled_labels_;

  Tensor2<CountType> local_bins_;
  Tensor2<CountType> global_bins_;
  Tensor2<CountType> local_bins_sum_;
  Tensor2<CountType> global_bins_sum_;
  Tensor2<CountType> partition_offsets_;
  Tensor2<int> pivots_;
  Tensor2<CountType> all_partition_offsets_;
  Tensor2<CountType> recv_offsets_;
  Tensor2<float> dcg_;
  Tensor2<CountType> label_count_;
  Tensor2<float> ideal_dcg_;

 public:
  float* d_labels() { return labels_1_.get_ptr(); }
  float* d_preds() { return preds_1_.get_ptr(); }
  CountType* d_local_bins() { return local_bins_.get_ptr(); }
  CountType* d_global_bins() { return global_bins_.get_ptr(); }
  CountType* d_local_bins_sum() { return local_bins_sum_.get_ptr(); }
  CountType* d_global_bins_sum() { return global_bins_sum_.get_ptr(); }
  int* d_pivots() { return pivots_.get_ptr(); }
  CountType* d_partition_offsets() { return partition_offsets_.get_ptr(); }
  float* d_partitioned_preds() { return preds_2_.get_ptr(); }
  float* d_partitioned_labels() { return labels_2_.get_ptr(); }
  CountType* d_all_partition_offsets() { return all_partition_offsets_.get_ptr(); }
  CountType* d_recv_offsets() { return recv_offsets_.get_ptr(); }
  float* d_presorted_preds() { return preds_1_.get_ptr(); }
  float* d_presorted_labels() { return labels_1_.get_ptr(); }
  float* d_sorted_preds() { return preds_2_.get_ptr(); }
  float* d_sorted_labels() { return labels_2_.get_ptr(); }
  float* d_scaled_labels() { return scaled_labels_.get_ptr(); }
  float* d_dcg() { return dcg_.get_ptr(); }
  CountType* d_label_count() { return label_count_.get_ptr(); }
  float* d_ideal_dcg() { return ideal_dcg_.get_ptr(); }

  size_t num_redistributed_samples;
  std::vector<size_t> all_num_redistributed_samples;
};

class NDCGStorage {
 public:
  void alloc_main(size_t num_local_samples, size_t num_bins, size_t num_partitions,
                  size_t num_global_gpus, const std::vector<int>& peers);
  void realloc_redistributed(size_t num_redistributed_samples, cudaStream_t stream);
  void realloc_workspace(size_t temp_storage);
  void* d_workspace() { return workspace_.get_ptr(); }
  size_t& temp_storage_bytes() { return allocated_temp_storage_; }

 private:
  const float reallocate_factor_ = 1.2f;
  size_t allocated_temp_storage_;
  size_t num_allocated_redistributed_;
  std::vector<CUmemAccessDesc> access_desc_;  // Access descriptors used by ReallocBuffers

  // Workspace for CUB functions
  ReallocBuffer<int8_t, ReallocType_t::NO_COPY> workspace_;

  ReallocBuffer<float, ReallocType_t::MMAP> preds_1_;
  ReallocBuffer<float, ReallocType_t::MMAP> labels_1_;
  ReallocBuffer<float, ReallocType_t::MMAP> preds_2_;
  ReallocBuffer<float, ReallocType_t::MMAP> labels_2_;
  ReallocBuffer<float, ReallocType_t::MMAP> scaled_labels_;

  core23::Tensor local_bins_;
  core23::Tensor global_bins_;
  core23::Tensor local_bins_sum_;
  core23::Tensor global_bins_sum_;
  core23::Tensor partition_offsets_;
  core23::Tensor pivots_;
  core23::Tensor all_partition_offsets_;
  core23::Tensor recv_offsets_;
  core23::Tensor dcg_;
  core23::Tensor label_count_;
  core23::Tensor ideal_dcg_;

 public:
  float* d_labels() { return labels_1_.get_ptr(); }
  float* d_preds() { return preds_1_.get_ptr(); }
  CountType* d_local_bins() { return local_bins_.data<CountType>(); }
  CountType* d_global_bins() { return global_bins_.data<CountType>(); }
  CountType* d_local_bins_sum() { return local_bins_sum_.data<CountType>(); }
  CountType* d_global_bins_sum() { return global_bins_sum_.data<CountType>(); }
  int* d_pivots() { return pivots_.data<int>(); }
  CountType* d_partition_offsets() { return partition_offsets_.data<CountType>(); }
  float* d_partitioned_preds() { return preds_2_.get_ptr(); }
  float* d_partitioned_labels() { return labels_2_.get_ptr(); }
  CountType* d_all_partition_offsets() { return all_partition_offsets_.data<CountType>(); }
  CountType* d_recv_offsets() { return recv_offsets_.data<CountType>(); }
  float* d_presorted_preds() { return preds_1_.get_ptr(); }
  float* d_presorted_labels() { return labels_1_.get_ptr(); }
  float* d_sorted_preds() { return preds_2_.get_ptr(); }
  float* d_sorted_labels() { return labels_2_.get_ptr(); }
  float* d_scaled_labels() { return scaled_labels_.get_ptr(); }
  float* d_dcg() { return dcg_.data<float>(); }
  CountType* d_label_count() { return label_count_.data<CountType>(); }
  float* d_ideal_dcg() { return ideal_dcg_.data<float>(); }

  size_t num_redistributed_samples;
  std::vector<size_t> all_num_redistributed_samples;
};

template <typename T>
class NDCG : public Metric {
 public:
  using PredType = T;
  using LabelType = float;
  NDCG(int batch_size_per_gpu, int n_batches,
       const std::shared_ptr<ResourceManager>& resource_manager, bool use_old_tensor = true);
  ~NDCG() override;

  void local_reduce(int local_gpu_id, RawMetricMap raw_metrics) override;
  void local_reduce(int local_gpu_id, Core23RawMetricMap raw_metrics) override;
  void global_reduce(int n_nets) override;
  float finalize_metric() override;
  float finalize_metric_per_gpu(int device_id);
  std::string name() const override { return "NDCG"; };

 private:
  float finalize_metric_single_gpu(int device_id);
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
  std::vector<NDCGStorageOld> storage_old_;
  std::vector<NDCGStorage> storage_;
  bool use_old_tensor_;
};

/*
template <typename T>
class HitRate: public Metric {
 public:
  using PredType = T;
  using LabelType = float;
  HitRate(int batch_size_per_gpu, int n_batches,
      const std::shared_ptr<ResourceManager>& resource_manager);
  ~HitRate() override;

  void local_reduce(int local_gpu_id, RawMetricMap raw_metrics) override;
  void global_reduce(int n_nets) override;
  float finalize_metric() override;
  std::string name() const override { return "HitRate"; };

  // Public in order to use device lambda
  float _finalize_metric_per_gpu(int device_id);

 private:
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

  AUCBarrier barrier_;

  std::vector<size_t> offsets_;
  std::vector<AUCStorageOld> storage_old_;
};
*/

/*
template <typename T>
class HitRate: public Metric {
 public:
  using PredType = T;
  using LabelType = float;
  HitRate(int batch_size_per_gpu, int n_batches,
      const std::shared_ptr<ResourceManager>& resource_manager);
  ~HitRate() override;

  void local_reduce(int local_gpu_id, RawMetricMap raw_metrics) override;
  void global_reduce(int n_nets) override;
  float finalize_metric() override;
  std::string name() const override { return "HitRate"; };

  // Public in order to use device lambda
  float _finalize_metric_per_gpu(int device_id);

 private:
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

  AUCBarrier barrier_;

  std::vector<size_t> offsets_;
  std::vector<AUCStorageOld> storage_old_;
};
*/

/*
template <typename T>
class HitRate: public Metric {
 public:
  using PredType = T;
  using LabelType = float;
  HitRate(int batch_size_per_gpu, int n_batches,
      const std::shared_ptr<ResourceManager>& resource_manager);
  ~HitRate() override;

  void local_reduce(int local_gpu_id, RawMetricMap raw_metrics) override;
  void global_reduce(int n_nets) override;
  float finalize_metric() override;
  std::string name() const override { return "HitRate"; };

  // Public in order to use device lambda
  float _finalize_metric_per_gpu(int device_id);

 private:
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

  AUCBarrier barrier_;

  std::vector<size_t> offsets_;
  std::vector<AUCStorageOld> storage_old_;
};
*/

}  // namespace metrics

}  // namespace HugeCTR