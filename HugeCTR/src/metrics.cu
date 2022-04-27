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

#include <omp.h>

#include <cub/cub.cuh>
#include <diagnose.hpp>
#include <general_buffer2.hpp>
#include <metrics.hpp>
#include <utils.cuh>

namespace HugeCTR {

namespace metrics {

namespace {

__global__ void convert_half_to_float_kernel(__half* src_ptr, float* dst_ptr, size_t num) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    dst_ptr[idx] = TypeConvertFunc<float, __half>::convert(src_ptr[idx]);
  }
}

__global__ void copy_all_kernel(float* y_pred, float* y_label, const __half* x_pred,
                                const float* x_label, int num_elems) {
  int gid_base = blockIdx.x * blockDim.x + threadIdx.x;
  for (int gid = gid_base; gid < num_elems; gid += blockDim.x * gridDim.x) {
    float pred_val = __half2float(x_pred[gid]);
    float label_val = x_label[gid];
    y_pred[gid] = pred_val;
    y_label[gid] = label_val;
  }
}

template <typename SrcType>
void copy_pred(float* y, SrcType* x, int num_elems, int num_sms, cudaStream_t stream);

template <>
void copy_pred<float>(float* y, float* x, int num_elems, int num_sms, cudaStream_t stream) {
  CK_CUDA_THROW_(
      cudaMemcpyAsync(y, x, num_elems * sizeof(float), cudaMemcpyDeviceToDevice, stream));
}

template <typename PredType>
void copy_all(float* y_pred, float* y_label, PredType* x_pred, float* x_label, int num_elems,
              int num_sms, cudaStream_t stream);

template <>
void copy_all<float>(float* y_pred, float* y_label, float* x_pred, float* x_label, int num_elems,
                     int num_sms, cudaStream_t stream) {
  copy_pred<float>(y_pred, x_pred, num_elems, num_sms, stream);
  CK_CUDA_THROW_(cudaMemcpyAsync(y_label, x_label, num_elems * sizeof(float),
                                 cudaMemcpyDeviceToDevice, stream));
}

template <>
void copy_all<__half>(float* y_pred, float* y_label, __half* x_pred, float* x_label, int num_elems,
                      int num_sms, cudaStream_t stream) {
  dim3 grid(num_sms * 2, 1, 1);
  dim3 block(1024, 1, 1);
  copy_all_kernel<<<grid, block, 0, stream>>>(y_pred, y_label, x_pred, x_label, num_elems);
}

__global__ void mock_kernel(float* predictions, int num_elems) {
  int gid_base = blockIdx.x * blockDim.x + threadIdx.x;
  for (int gid = gid_base; gid < num_elems; gid += blockDim.x * gridDim.x) {
    predictions[gid] = gid / (float)num_elems;
  }
}

__global__ void find_pivots_kernel(const CountType* bins_sum, int num_bins, CountType num_samples,
                                   int* pivots) {
  int gid_base = blockIdx.x * blockDim.x + threadIdx.x;

  for (int ibin = gid_base; ibin < num_bins - 1; ibin += blockDim.x * gridDim.x) {
    int id_start = bins_sum[ibin] / num_samples;
    int id_end = bins_sum[ibin + 1] / num_samples;
    if (ibin == 0) {
      id_start = 0;
    }
    for (int id = id_start; id < id_end; id++) {
      pivots[id] = ibin;
    }
  }
}

__global__ void find_partition_offsets_kernel(const CountType* bins_sum, const int* pivots,
                                              int num_partitions, int num_samples,
                                              CountType* offsets) {
  int gid_base = blockIdx.x * blockDim.x + threadIdx.x;

  for (int gid = gid_base; gid < num_partitions - 1; gid += blockDim.x * gridDim.x) {
    int ipart = gid;

    offsets[ipart + 1] = bins_sum[pivots[ipart]];
  }
  if (gid_base == 0) {
    offsets[0] = 0;
    offsets[num_partitions] = num_samples;
  }
}

__device__ inline CountType binsearch(int v, const int* pivots, int pivots_size) {
  int l = 0, r = pivots_size;
  while (r > l) {
    int i = (l + r) / 2;
    if (v <= pivots[i]) {
      r = i;
    } else {
      l = i + 1;
    }
  }
  return l;
}

__device__ inline int compute_ibin(float v, float minval, float maxval, int num_bins) {
  int ibin_raw = (int)((v - minval) * num_bins / (maxval - minval));
  return min(max(ibin_raw, 0), num_bins - 1);
}

__launch_bounds__(1024, 1) __global__
    void create_partitions_kernel(const float* labels, const float* predictions, const int* pivots,
                                  float pred_min, float pred_max, CountType num_samples,
                                  int num_bins, int num_partitions,
                                  const CountType* global_partition_offsets,
                                  CountType* global_partition_sizes, float* part_labels,
                                  float* part_preds) {
  int gid_base = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ extern CountType shmem[];
  CountType* local_partition_sizes = shmem;
  CountType* local_partition_offsets = shmem + num_partitions;

  for (int id = threadIdx.x; id < num_partitions; id += blockDim.x) {
    local_partition_sizes[id] = 0;
  }
  __syncthreads();

  for (CountType gid = gid_base; gid < num_samples; gid += blockDim.x * gridDim.x) {
    float value = predictions[gid];
    int ibin = compute_ibin(value, pred_min, pred_max, num_bins);
    int ipart = binsearch(ibin, pivots, num_partitions - 1);

    CountType part_offset = atomicAdd(local_partition_sizes + ipart, 1);
  }

  __syncthreads();

  for (int id = threadIdx.x; id < num_partitions; id += blockDim.x) {
    local_partition_offsets[id] = atomicAdd(global_partition_sizes + id, local_partition_sizes[id]);
    local_partition_sizes[id] = 0;
  }

  __syncthreads();

  for (CountType gid = gid_base; gid < num_samples; gid += blockDim.x * gridDim.x) {
    float value = predictions[gid];
    int ibin = compute_ibin(value, pred_min, pred_max, num_bins);
    int ipart = binsearch(ibin, pivots, num_partitions - 1);

    CountType my_glob_part_offset = global_partition_offsets[ipart] +
                                    local_partition_offsets[ipart] +
                                    atomicAdd(local_partition_sizes + ipart, 1);

    part_labels[my_glob_part_offset] = labels[gid];
    part_preds[my_glob_part_offset] = value;
  }
}

__global__ void rate_from_part_cumsum_kernel(const float* cumsum, CountType num_samples,
                                             CountType* offset, CountType* total, float* rate) {
  CountType gid_base = blockIdx.x * blockDim.x + threadIdx.x;

  for (CountType gid = gid_base; gid < num_samples; gid += blockDim.x * gridDim.x) {
    rate[gid] = (*total - (*offset + cumsum[gid])) / (float)(*total);
  }
}

__global__ void flatten_segments_kernel(const CountType* offsets, const CountType* lengths,
                                        const int* num_segments, float* tps, float* fps) {
  CountType gid_base = blockIdx.x * blockDim.x + threadIdx.x;
  CountType wid = gid_base / warpSize;
  int lane_id = gid_base % warpSize;

  int increment = (blockDim.x * gridDim.x) / warpSize;
  for (CountType seg_id = wid; seg_id < *num_segments; seg_id += increment) {
    CountType offset = offsets[seg_id];
    CountType length = lengths[seg_id];
    float value_tp = tps[offset + length - 1];
    float value_fp = fps[offset + length - 1];

    for (int el = lane_id; el < length; el += warpSize) {
      tps[offset + el] = value_tp;
      fps[offset + el] = value_fp;
    }
  }
}

__global__ void trapz_kernel(float* y, float* x, float* halo_y, float* halo_x, float* auc,
                             CountType num_samples) {
  __shared__ float s_auc;
  s_auc = 0.0f;
  __syncthreads();
  float my_area = 0.0f;
  CountType gid_base = blockIdx.x * blockDim.x + threadIdx.x;
  for (CountType gid = gid_base; gid < num_samples - 1; gid += blockDim.x * gridDim.x) {
    float a = x[gid];
    float b = x[gid + 1];
    float fa = y[gid];
    float fb = y[gid + 1];
    my_area += -0.5f * (b - a) * (fa + fb);
    if (gid == 0) {
      my_area += 0.5f * (*halo_y + fa) * (*halo_x - a);
    }
  }
  atomicAdd(&s_auc, my_area);
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(auc, s_auc);
  }
}
}  // namespace

namespace metric_comm {

template <typename T>
ncclDataType_t get_nccl_type();
template <>
ncclDataType_t get_nccl_type<int>() {
  return ncclInt32;
}
template <>
ncclDataType_t get_nccl_type<unsigned int>() {
  return ncclUint32;
}
template <>
ncclDataType_t get_nccl_type<unsigned long long>() {
  return ncclUint64;
}
template <>
ncclDataType_t get_nccl_type<float>() {
  return ncclFloat32;
}
template <>
ncclDataType_t get_nccl_type<__half>() {
  return ncclFloat16;
}

template <typename T>
void allreduce(T* srcptr, T* dstptr, int count, const GPUResource* gpu_resource) {
  auto& stream = gpu_resource->get_stream();
  CK_NCCL_THROW_(ncclAllReduce(srcptr, dstptr, count, get_nccl_type<T>(), ncclSum,
                               gpu_resource->get_nccl(), stream));
}

template <typename T>
void allgather(T* srcptr, T* dstptr, int src_count, const GPUResource* gpu_resource) {
  auto& stream = gpu_resource->get_stream();
  CK_NCCL_THROW_(ncclAllGather(srcptr, dstptr, src_count, get_nccl_type<T>(),
                               gpu_resource->get_nccl(), stream));
}

template <typename T>
void all_to_all(T* srcptr, T* dstptr, const CountType* src_offsets, const CountType* dst_offsets,
                int num_global_gpus, const GPUResource* gpu_resource) {
  auto& stream = gpu_resource->get_stream();
  auto& comm = gpu_resource->get_nccl();
  auto type = get_nccl_type<T>();

  CK_NCCL_THROW_(ncclGroupStart());
  for (int i = 0; i < num_global_gpus; i++) {
    CK_NCCL_THROW_(ncclSend(srcptr + src_offsets[i], src_offsets[i + 1] - src_offsets[i], type, i,
                            comm, stream));
    CK_NCCL_THROW_(ncclRecv(dstptr + dst_offsets[i], dst_offsets[i + 1] - dst_offsets[i], type, i,
                            comm, stream));
  }
  CK_NCCL_THROW_(ncclGroupEnd());
}

template <typename T>
void send_halo_right(T* srcptr, T* dstptr, int count, int left_neighbor, int right_neighbor,
                     const GPUResource* gpu_resource) {
  auto& stream = gpu_resource->get_stream();
  auto& comm = gpu_resource->get_nccl();
  auto type = get_nccl_type<T>();

  CK_NCCL_THROW_(ncclGroupStart());
  if (right_neighbor >= 0) {
    CK_NCCL_THROW_(ncclSend(srcptr, count, type, right_neighbor, comm, stream));
  }
  if (left_neighbor >= 0) {
    CK_NCCL_THROW_(ncclRecv(dstptr, count, type, left_neighbor, comm, stream));
  }
  CK_NCCL_THROW_(ncclGroupEnd());
}

}  // namespace metric_comm

void get_raw_metric_as_host_float_tensor(RawMetricMap metric_map, RawType raw_type,
                                         bool mixed_precision, float* rst, size_t num) {
  Tensor2<float> device_prediction_result;
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buffer_ptr =
      GeneralBuffer2<CudaAllocator>::create();

  if (mixed_precision) {
    Tensor2<__half> raw_metric_tensor = Tensor2<__half>::stretch_from(metric_map[raw_type]);
    if (raw_metric_tensor.get_num_elements() != num) {
      CK_THROW_(Error_t::WrongInput,
                "num elements: " + std::to_string(raw_metric_tensor.get_num_elements()) +
                    " not match with " + std::to_string(num));
    }
    buffer_ptr->reserve(raw_metric_tensor.get_dimensions(), &device_prediction_result);
    buffer_ptr->allocate();
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((num + blockSize.x - 1) / blockSize.x, 1, 1);
    convert_half_to_float_kernel<<<gridSize, blockSize>>>(raw_metric_tensor.get_ptr(),
                                                          device_prediction_result.get_ptr(), num);
  } else {
    device_prediction_result = Tensor2<float>::stretch_from(metric_map[raw_type]);
    if (num != device_prediction_result.get_num_elements()) {
      CK_THROW_(Error_t::WrongInput,
                "num elements: " + std::to_string(device_prediction_result.get_num_elements()) +
                    " not match with " + std::to_string(num));
    }
  }
  CK_CUDA_THROW_(cudaMemcpy(rst, device_prediction_result.get_ptr(), num * sizeof(float),
                            cudaMemcpyDeviceToHost));
}

std::unique_ptr<Metric> Metric::Create(const Type type, bool use_mixed_precision,
                                       int batch_size_eval, int n_batches,
                                       const std::shared_ptr<ResourceManager>& resource_manager) {
  std::unique_ptr<Metric> ret;
  switch (type) {
    case Type::AUC:
      if (use_mixed_precision) {
        ret.reset(new AUC<__half>(batch_size_eval, n_batches, resource_manager));
      } else {
        ret.reset(new AUC<float>(batch_size_eval, n_batches, resource_manager));
      }
      break;
    case Type::AverageLoss:
      ret.reset(new AverageLoss<float>(resource_manager));
      break;
    case Type::HitRate:
      ret.reset(new HitRate<float>(batch_size_eval, resource_manager));
      break;
  }
  return ret;
}

Metric::Metric() : current_batch_size_(0) {}
Metric::~Metric() {}

template <typename T>
AverageLoss<T>::AverageLoss(const std::shared_ptr<ResourceManager>& resource_manager)
    : Metric(),
      resource_manager_(resource_manager),
      loss_local_(std::vector<float>(resource_manager->get_local_gpu_count(), 0.0f)),
      loss_global_(0.0f),
      n_batches_(0) {}

template <typename T>
AverageLoss<T>::~AverageLoss() {}

template <typename T>
void AverageLoss<T>::local_reduce(int local_gpu_id, RawMetricMap raw_metrics) {
  float loss_host = 0.0f;
  Tensor2<T> loss_tensor = Tensor2<T>::stretch_from(raw_metrics[RawType::Loss]);
  const auto& local_gpu = resource_manager_->get_local_gpu(local_gpu_id);
  CudaDeviceContext context(local_gpu->get_device_id());
  PROFILE_RECORD("metric_local_reduce.start", local_gpu->get_stream());
  CK_CUDA_THROW_(cudaMemcpyAsync(&loss_host, loss_tensor.get_ptr(), sizeof(float),
                                 cudaMemcpyDeviceToHost, local_gpu->get_stream()));
  PROFILE_RECORD("metric_local_reduce.stop", local_gpu->get_stream());
  loss_local_[local_gpu_id] = loss_host;
}

template <typename T>
void AverageLoss<T>::global_reduce(int n_nets) {
  float loss_inter = 0.0f;
  for (auto& loss_local : loss_local_) {
    loss_inter += loss_local;
  }

#ifdef ENABLE_MPI
  if (resource_manager_->get_num_process() > 1) {
    float loss_reduced = 0.0f;
    CK_MPI_THROW_(MPI_Reduce(&loss_inter, &loss_reduced, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));
    loss_inter = loss_reduced;
  }
#endif
  loss_global_ += loss_inter / n_nets / resource_manager_->get_num_process();
  n_batches_++;
}

template <typename T>
float AverageLoss<T>::finalize_metric() {
  float ret = 0.0f;
  if (resource_manager_->is_master_process()) {
    if (n_batches_) {
      ret = loss_global_ / n_batches_;
    }
  }
#ifdef ENABLE_MPI
  CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
  CK_MPI_THROW_(MPI_Bcast(&ret, 1, MPI_FLOAT, 0, MPI_COMM_WORLD));
#endif

  loss_global_ = 0.0f;
  for (auto& loss_local : loss_local_) {
    loss_local = 0.0f;
  }
  n_batches_ = 0;
  return ret;
}

void AUCStorage::alloc_main(size_t num_local_samples, size_t num_bins, size_t num_partitions,
                            size_t num_global_gpus) {
  size_t bins_buffer_size = num_bins * sizeof(CountType);

  CK_CUDA_THROW_(cudaMalloc((void**)&(ptr_local_bins_), bins_buffer_size));
  CK_CUDA_THROW_(cudaMalloc((void**)&(ptr_global_bins_), bins_buffer_size));
  CK_CUDA_THROW_(cudaMalloc((void**)&(ptr_global_bins_sum_), bins_buffer_size));
  CK_CUDA_THROW_(cudaMalloc((void**)&(ptr_local_bins_sum_), bins_buffer_size));
  CK_CUDA_THROW_(cudaMalloc((void**)&(ptr_pivots_), num_partitions * sizeof(int)));
  CK_CUDA_THROW_(cudaMallocManaged((void**)&(ptr_partition_offsets_),
                                   (num_partitions + 1) * sizeof(CountType)));
  CK_CUDA_THROW_(cudaMallocManaged((void**)&(ptr_all_partition_offsets_),
                                   (num_partitions + 1) * num_global_gpus * sizeof(CountType)));
  CK_CUDA_THROW_(
      cudaMallocManaged((void**)&(ptr_recv_offsets_), (num_partitions + 1) * sizeof(CountType)));
  CK_CUDA_THROW_(
      cudaMalloc((void**)&(ptr_pos_per_gpu_), (num_global_gpus + 1) * sizeof(CountType)));
  CK_CUDA_THROW_(
      cudaMalloc((void**)&(ptr_neg_per_gpu_), (num_global_gpus + 1) * sizeof(CountType)));
  CK_CUDA_THROW_(cudaMalloc((void**)&(ptr_num_identical_segments_), sizeof(int)));
  CK_CUDA_THROW_(cudaMalloc((void**)&(ptr_halo_tpr_), sizeof(float)));
  CK_CUDA_THROW_(cudaMalloc((void**)&(ptr_halo_fpr_), sizeof(float)));
  CK_CUDA_THROW_(cudaMalloc((void**)&(ptr_tp_offsets_), (num_global_gpus + 1) * sizeof(CountType)));
  CK_CUDA_THROW_(cudaMalloc((void**)&(ptr_fp_offsets_), (num_global_gpus + 1) * sizeof(CountType)));
  CK_CUDA_THROW_(cudaMallocManaged((void**)&(ptr_auc_), sizeof(float)));

  CK_CUDA_THROW_(cudaMemset(ptr_pos_per_gpu_, 0, (num_global_gpus + 1) * sizeof(CountType)));
  CK_CUDA_THROW_(cudaMemset(ptr_neg_per_gpu_, 0, (num_global_gpus + 1) * sizeof(CountType)));

  realloc_redistributed(num_local_samples, 0);
}

void AUCStorage::realloc_ptr(void** ptr, size_t old_size, size_t new_size, cudaStream_t stream) {
  void* tmp;
  CK_CUDA_THROW_(cudaMalloc(&tmp, new_size));
  CK_CUDA_THROW_(cudaMemcpyAsync(tmp, *ptr, old_size, cudaMemcpyDeviceToDevice, stream));
  CK_CUDA_THROW_(cudaFree(*ptr));
  *ptr = tmp;
}

void AUCStorage::realloc_redistributed(size_t num_redistributed_samples, cudaStream_t stream) {
  if (num_redistributed_samples > num_allocated_redistributed_) {
    size_t old_size = num_allocated_redistributed_ * sizeof(float);
    num_allocated_redistributed_ = reallocate_factor_ * num_redistributed_samples;
    size_t redistributed_buffer_size = num_allocated_redistributed_ * sizeof(float);
    size_t runs_buffer_size = num_allocated_redistributed_ * sizeof(CountType) / 2;

    realloc_ptr((void**)&(ptr_preds_1_), old_size, redistributed_buffer_size, stream);
    realloc_ptr((void**)&(ptr_labels_1_), old_size, redistributed_buffer_size, stream);
    realloc_ptr((void**)&(ptr_preds_2_), old_size, redistributed_buffer_size, stream);
    realloc_ptr((void**)&(ptr_labels_2_), old_size, redistributed_buffer_size, stream);

    // These two buffers do not need to preserve their data
    CK_CUDA_THROW_(cudaFree(ptr_identical_pred_starts_));
    CK_CUDA_THROW_(cudaFree(ptr_identical_pred_lengths_));
    CK_CUDA_THROW_(cudaMalloc((void**)&ptr_identical_pred_starts_, runs_buffer_size));
    CK_CUDA_THROW_(cudaMalloc((void**)&ptr_identical_pred_lengths_, runs_buffer_size));
  }
}

void AUCStorage::realloc_workspace(size_t temp_storage) {
  if (temp_storage > allocated_temp_storage_) {
    allocated_temp_storage_ = reallocate_factor_ * temp_storage;
    // This is temporary storage, no need to preserve the data
    cudaFree(workspace_);
    CK_CUDA_THROW_(cudaMalloc((void**)&(workspace_), allocated_temp_storage_));
  }
}

void AUCStorage::free_all() {
  CK_CUDA_THROW_(cudaFree(ptr_local_bins_));
  CK_CUDA_THROW_(cudaFree(ptr_global_bins_));
  CK_CUDA_THROW_(cudaFree(ptr_global_bins_sum_));
  CK_CUDA_THROW_(cudaFree(ptr_local_bins_sum_));
  CK_CUDA_THROW_(cudaFree(ptr_pivots_));
  CK_CUDA_THROW_(cudaFree(ptr_partition_offsets_));
  CK_CUDA_THROW_(cudaFree(ptr_all_partition_offsets_));
  CK_CUDA_THROW_(cudaFree(ptr_recv_offsets_));
  CK_CUDA_THROW_(cudaFree(ptr_pos_per_gpu_));
  CK_CUDA_THROW_(cudaFree(ptr_neg_per_gpu_));
  CK_CUDA_THROW_(cudaFree(ptr_num_identical_segments_));
  CK_CUDA_THROW_(cudaFree(ptr_halo_tpr_));
  CK_CUDA_THROW_(cudaFree(ptr_halo_fpr_));
  CK_CUDA_THROW_(cudaFree(ptr_tp_offsets_));
  CK_CUDA_THROW_(cudaFree(ptr_fp_offsets_));
  CK_CUDA_THROW_(cudaFree(ptr_auc_));
  CK_CUDA_THROW_(cudaFree(ptr_preds_1_));
  CK_CUDA_THROW_(cudaFree(ptr_labels_1_));
  CK_CUDA_THROW_(cudaFree(ptr_preds_2_));
  CK_CUDA_THROW_(cudaFree(ptr_labels_2_));
  CK_CUDA_THROW_(cudaFree(ptr_identical_pred_starts_));
  CK_CUDA_THROW_(cudaFree(ptr_identical_pred_lengths_));
  CK_CUDA_THROW_(cudaFree(workspace_));
}

/// Wrapper to call CUB functions with preallocation
template <typename CUB_Func>
void CUB_allocate_and_launch(AUCStorage& st, CUB_Func func) {
  size_t requested_size = 0;
  CK_CUDA_THROW_(func(nullptr, requested_size));
  st.realloc_workspace(requested_size);
  CK_CUDA_THROW_(func(st.d_workspace(), st.temp_storage_bytes()));
}

template <typename T>
AUC<T>::AUC(int batch_size_per_gpu, int n_batches,
            const std::shared_ptr<ResourceManager>& resource_manager)
    : Metric(),
      resource_manager_(resource_manager),
      batch_size_per_gpu_(batch_size_per_gpu),
      n_batches_(n_batches),
      num_local_gpus_(resource_manager_->get_local_gpu_count()),
      num_global_gpus_(resource_manager_->get_global_gpu_count()),
      num_bins_(num_global_gpus_ * num_bins_per_gpu_),
      num_partitions_(num_global_gpus_),
      num_total_samples_(0),
      storage_(num_local_gpus_),
      offsets_(num_local_gpus_, 0) {
  size_t max_num_local_samples = (batch_size_per_gpu_ * n_batches_);

  for (int i = 0; i < num_local_gpus_; i++) {
    int device_id = resource_manager_->get_local_gpu(i)->get_device_id();
    CudaDeviceContext context(device_id);

    auto& st = storage_[i];
    st.alloc_main(max_num_local_samples, num_bins_, num_partitions_, num_global_gpus_);
    st.realloc_workspace(num_partitions_ * sizeof(CountType));
  }

  warm_up(max_num_local_samples);
}

template <typename T>
AUC<T>::~AUC() {
  for (int i = 0; i < num_local_gpus_; i++) {
    int device_id = resource_manager_->get_local_gpu(i)->get_device_id();
    CudaDeviceContext context(device_id);

    storage_[i].free_all();
  }
}

int get_num_valid_samples(int global_device_id, int current_batch_size, int batch_per_gpu) {
  int remaining = current_batch_size - global_device_id * batch_per_gpu;
  return std::max(std::min(remaining, batch_per_gpu), 0);
}

template <typename T>
void AUC<T>::local_reduce(int local_gpu_id, RawMetricMap raw_metrics) {
  Tensor2<PredType> pred_tensor = Tensor2<PredType>::stretch_from(raw_metrics[RawType::Pred]);
  Tensor2<LabelType> label_tensor = Tensor2<LabelType>::stretch_from(raw_metrics[RawType::Label]);
  int device_id = resource_manager_->get_local_gpu(local_gpu_id)->get_device_id();
  int global_device_id = resource_manager_->get_local_gpu(local_gpu_id)->get_global_id();
  const auto& st = storage_[local_gpu_id];

  // Copy the labels and predictions to the internal buffer
  size_t& offset = offsets_[local_gpu_id];
  CudaDeviceContext context(device_id);
  int num_valid_samples =
      get_num_valid_samples(global_device_id, current_batch_size_, batch_size_per_gpu_);

  copy_all<T>(st.d_preds() + offset, st.d_labels() + offset, pred_tensor.get_ptr(),
              label_tensor.get_ptr(), num_valid_samples,
              resource_manager_->get_local_gpu(local_gpu_id)->get_sm_count(),
              resource_manager_->get_local_gpu(local_gpu_id)->get_stream());

  offset += num_valid_samples;

  if (local_gpu_id == 0) {
    num_total_samples_ += current_batch_size_;
  }
}

template <typename T>
void AUC<T>::global_reduce(int n_nets) {
  // No need to do anything here
}

template <typename T>
void AUC<T>::warm_up(size_t num_local_samples) {
  dim3 grid(160, 1, 1);
  dim3 block(1024, 1, 1);

  MESSAGE_("Starting AUC NCCL warm-up");
#pragma omp parallel for num_threads(num_local_gpus_)
  for (int local_id = 0; local_id < num_local_gpus_; local_id++) {
    auto gpu_resource = resource_manager_->get_local_gpu(local_id).get();
    int device_id = gpu_resource->get_device_id();
    auto& stream = gpu_resource->get_stream();

    CudaDeviceContext context(device_id);
    auto& st = storage_[local_id];

    mock_kernel<<<grid, block, 0, stream>>>(st.d_preds(), num_local_samples);
    initialize_array<<<grid, block, 0, stream>>>(st.d_labels(), num_local_samples, 0.0f);
    offsets_[local_id] = num_local_samples;
  }
  num_total_samples_ = num_local_samples * num_global_gpus_;

  [[maybe_unused]] float dummy = finalize_metric();
  MESSAGE_("Warm-up done");
}

template <typename T>
float AUC<T>::finalize_metric() {
  float result[num_local_gpus_];
#pragma omp parallel num_threads(num_local_gpus_)
  { result[omp_get_thread_num()] = _finalize_metric_per_gpu(omp_get_thread_num()); }

  num_total_samples_ = 0;
  // All threads should have the same result here
  return result[0];
}

template <typename T>
float AUC<T>::_finalize_metric_per_gpu(int local_id) {
  dim3 grid(160, 1, 1);
  dim3 block(1024, 1, 1);

  auto gpu_resource = resource_manager_->get_local_gpu(local_id).get();
  int device_id = gpu_resource->get_device_id();
  int global_id = gpu_resource->get_global_id();
  auto& stream = gpu_resource->get_stream();

  CudaDeviceContext context(device_id);
  auto& st = storage_[local_id];
  size_t num_local_samples = offsets_[local_id];
  offsets_[local_id] = 0;

  // 1. Create local histograms of predictions
  float eps = 1e-7f;
  float loc_min = pred_min_ + eps;
  float loc_max = pred_max_ - eps;
  auto clamp = [loc_min, loc_max] __device__(const float v) {
    return fmaxf(fminf(v, loc_max), loc_min);
  };
  cub::TransformInputIterator<float, decltype(clamp), float*> d_clamped_preds(st.d_preds(), clamp);

  // (int) casting is a CUB workaround. Fixed in https://github.com/thrust/cub/pull/38
  CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
    return cub::DeviceHistogram::HistogramEven(workspace, size, d_clamped_preds, st.d_local_bins(),
                                               num_bins_ + 1, pred_min_, pred_max_,
                                               (int)num_local_samples, stream);
  });

  // 2. Allreduce histograms
  metric_comm::allreduce(st.d_local_bins(), st.d_global_bins(), num_bins_, gpu_resource);

  // 3. Find num_global_gpus_-1 pivot points
  CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
    return cub::DeviceScan::InclusiveSum(workspace, size, st.d_global_bins(),
                                         st.d_global_bins_sum(), num_bins_, stream);
  });

  find_pivots_kernel<<<grid, block, 0, stream>>>(
      st.d_global_bins_sum(), num_bins_, num_total_samples_ / num_global_gpus_, st.d_pivots());

  // 4. Partition (partially sort) local predictions into num_global_gpus_ bins
  //    separated by pivot points.
  CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
    return cub::DeviceScan::InclusiveSum(workspace, size, st.d_local_bins(), st.d_local_bins_sum(),
                                         num_bins_, stream);
  });

  find_partition_offsets_kernel<<<grid, block, 0, stream>>>(st.d_local_bins_sum(), st.d_pivots(),
                                                            num_partitions_, num_local_samples,
                                                            st.d_partition_offsets());

  initialize_array<<<grid, block, 0, stream>>>((CountType*)st.d_workspace(), num_partitions_,
                                               (CountType)0);

  size_t shmem_size = sizeof(CountType) * num_partitions_ * 2;
  create_partitions_kernel<<<grid, block, shmem_size, stream>>>(
      st.d_labels(), st.d_preds(), st.d_pivots(), pred_min_, pred_max_, num_local_samples,
      num_bins_, num_partitions_, st.d_partition_offsets(), (CountType*)st.d_workspace(),
      st.d_partitioned_labels(), st.d_partitioned_preds());

  // 5. Exchange the data such that all predicitons on GPU i are smaller than
  //    the ones on GPU i+1.
  // 5.1. Compute receiving side offsets. Also compute resulting number
  //      of elements on all the other GPUs, required to determine correct neighbors
  metric_comm::allgather(st.d_partition_offsets(), st.d_all_partition_offsets(),
                         num_partitions_ + 1, gpu_resource);

  // The following is done on the CPU, need to wait
  CK_CUDA_THROW_(cudaStreamSynchronize(stream));

  std::vector<size_t> all_num_redistributed_samples(num_global_gpus_);
  for (int dest = 0; dest < num_global_gpus_; dest++) {
    CountType sum = 0;
    for (int src = 0; src < num_global_gpus_; src++) {
      CountType size_src = st.d_all_partition_offsets()[src * (num_partitions_ + 1) + dest + 1] -
                           st.d_all_partition_offsets()[src * (num_partitions_ + 1) + dest];
      if (dest == global_id) {
        st.d_recv_offsets()[src] = sum;
      }
      sum += size_src;
    }
    all_num_redistributed_samples[dest] = sum;
  }
  st.d_recv_offsets()[num_global_gpus_] = all_num_redistributed_samples[global_id];
  const size_t num_redistributed_samples = all_num_redistributed_samples[global_id];

  // 5.2 Allocate more memory if needed
  st.realloc_redistributed(num_redistributed_samples, stream);

// 5.3 Synchronize threads before all to all to prevent hangs
#pragma omp barrier

  // 5.4 All to all
  metric_comm::all_to_all(st.d_partitioned_labels(), st.d_presorted_labels(),
                          st.d_partition_offsets(), st.d_recv_offsets(), num_global_gpus_,
                          gpu_resource);
  metric_comm::all_to_all(st.d_partitioned_preds(), st.d_presorted_preds(),
                          st.d_partition_offsets(), st.d_recv_offsets(), num_global_gpus_,
                          gpu_resource);
  CK_CUDA_THROW_(cudaStreamSynchronize(stream));

  if (num_redistributed_samples > 0) {
    // 6. Locally sort (label, pred) by pred
    CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
      return cub::DeviceRadixSort::SortPairs(workspace, size, st.d_presorted_preds(),
                                             st.d_sorted_preds(), st.d_presorted_labels(),
                                             st.d_sorted_labels(), num_redistributed_samples, 0,
                                             sizeof(float) * 8,  // begin_bit, end_bit
                                             stream);
    });

    // 7. Create TPR and FPR. Need a "global" scan
    // 7.1 Local inclusive scan to find TP and FP
    auto one_minus_val = [] __device__(const float v) { return 1.0f - v; };
    cub::TransformInputIterator<float, decltype(one_minus_val), float*> d_one_minus_labels(
        st.d_sorted_labels(), one_minus_val);

    CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
      return cub::DeviceScan::InclusiveSum(workspace, size, st.d_sorted_labels(), st.d_tp(),
                                           num_redistributed_samples, stream);
    });
    CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
      return cub::DeviceScan::InclusiveSum(workspace, size, d_one_minus_labels, st.d_fp(),
                                           num_redistributed_samples, stream);
    });

    // 7.2 'Flatten' tp and fp for cases where several consecutive predictions are identical
    CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
      return cub::DeviceRunLengthEncode::NonTrivialRuns(
          workspace, size, st.d_sorted_preds(), st.d_identical_pred_starts(),
          st.d_identical_pred_lengths(), st.d_num_identical_segments(), num_redistributed_samples,
          stream);
    });

    flatten_segments_kernel<<<grid, block, 0, stream>>>(
        st.d_identical_pred_starts(), st.d_identical_pred_lengths(), st.d_num_identical_segments(),
        st.d_tp(), st.d_fp());

    // 7.3 Allgather of the number of total positive and negative samples
    //     on each GPU and exclusive scan them
    metric_comm::allgather(st.d_tp() + num_redistributed_samples - 1, st.d_pos_per_gpu(), 1,
                           gpu_resource);
    metric_comm::allgather(st.d_fp() + num_redistributed_samples - 1, st.d_neg_per_gpu(), 1,
                           gpu_resource);

    CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
      return cub::DeviceScan::ExclusiveSum(workspace, size, st.d_pos_per_gpu(), st.d_tp_offsets(),
                                           num_global_gpus_ + 1, stream);
    });

    CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
      return cub::DeviceScan::ExclusiveSum(workspace, size, st.d_neg_per_gpu(), st.d_fp_offsets(),
                                           num_global_gpus_ + 1, stream);
    });

    // 7.4 Locally find TPR and FPR given TP, FP and global offsets
    rate_from_part_cumsum_kernel<<<grid, block, 0, stream>>>(
        st.d_tp(), num_redistributed_samples, st.d_tp_offsets() + global_id,
        st.d_tp_offsets() + num_global_gpus_, st.d_tpr());
    rate_from_part_cumsum_kernel<<<grid, block, 0, stream>>>(
        st.d_fp(), num_redistributed_samples, st.d_fp_offsets() + global_id,
        st.d_fp_offsets() + num_global_gpus_, st.d_fpr());

    // 8. Integrate TPR and FPR taking into account halo samples
    // 8.1 No need to communicate with GPUs that have 0 elements
    int left_neighbor = -1, right_neighbor = -1;
    for (int gpuid = global_id - 1; gpuid >= 0; gpuid--) {
      if (all_num_redistributed_samples[gpuid] > 0) {
        left_neighbor = gpuid;
        break;
      }
    }
    for (int gpuid = global_id + 1; gpuid < num_global_gpus_; gpuid++) {
      if (all_num_redistributed_samples[gpuid] > 0) {
        right_neighbor = gpuid;
        break;
      }
    }

    // 8.2 Send the halos
    metric_comm::send_halo_right(st.d_tpr() + num_redistributed_samples - 1, st.d_halo_tpr(), 1,
                                 left_neighbor, right_neighbor, gpu_resource);
    metric_comm::send_halo_right(st.d_fpr() + num_redistributed_samples - 1, st.d_halo_fpr(), 1,
                                 left_neighbor, right_neighbor, gpu_resource);

    // 8.3 First non-zero GPU initializes the halo to 0
    if (left_neighbor == -1) {
      initialize_array<<<grid, block, 0, stream>>>(st.d_halo_tpr(), 1, 1.0f);
      initialize_array<<<grid, block, 0, stream>>>(st.d_halo_fpr(), 1, 1.0f);
    }

    // 8.4 Integrate
    initialize_array<<<grid, block, 0, stream>>>(st.d_auc(), 1, 0.0f);
    trapz_kernel<<<grid, block, 0, stream>>>(st.d_tpr(), st.d_fpr(), st.d_halo_tpr(),
                                             st.d_halo_fpr(), st.d_auc(),
                                             num_redistributed_samples);
  } else {
    // Here we're on a GPU with no elements, need to communicate zeros where needed
    // Performance is not a concern on such GPUs
    MESSAGE_("GPU " + std::to_string(global_id) +
             " has no samples in the AUC computation "
             "due to strongly uneven distribution of the scores. "
             "This may indicate a problem in the training or an extremely accurate model.");

    initialize_array<<<grid, block, 0, stream>>>(st.d_halo_tpr(), 1, 0.0f);
    // 7.3 All GPUs need to call allgather
    metric_comm::allgather(st.d_halo_tpr(), st.d_pos_per_gpu(), 1, gpu_resource);
    metric_comm::allgather(st.d_halo_tpr(), st.d_neg_per_gpu(), 1, gpu_resource);

    // 8.4 Initialize partial auc to 0
    initialize_array<<<grid, block, 0, stream>>>(st.d_auc(), 1, 0.0f);
  }

  // 9. Finally allreduce auc
  metric_comm::allreduce(st.d_auc(), st.d_auc(), 1, gpu_resource);

  CK_CUDA_THROW_(cudaStreamSynchronize(stream));
  return *st.d_auc();
}

// HitRate Metric function implementations
template <typename T>
HitRate<T>::HitRate(int batch_size_per_gpu,
                    const std::shared_ptr<ResourceManager>& resource_manager)
    : Metric(),
      batch_size_per_gpu_(batch_size_per_gpu),
      resource_manager_(resource_manager),
      num_local_gpus_(resource_manager_->get_local_gpu_count()),
      checked_count_(resource_manager_->get_local_gpu_count()),
      hit_count_(resource_manager_->get_local_gpu_count()),
      checked_local_(std::vector<int>(resource_manager->get_local_gpu_count(), 0)),
      hits_local_(std::vector<int>(resource_manager->get_local_gpu_count(), 0)),
      hits_global_(0),
      checked_global_(0),
      n_batches_(0) {
  for (int i = 0; i < num_local_gpus_; i++) {
    int device_id = resource_manager_->get_local_gpu(i)->get_device_id();
    CudaDeviceContext context(device_id);

    CK_CUDA_THROW_(cudaMalloc((void**)(&(checked_count_[i])), sizeof(int)));
    CK_CUDA_THROW_(cudaMalloc((void**)(&(hit_count_[i])), sizeof(int)));
  }
}

template <typename T>
void HitRate<T>::free_all() {
  for (int i = 0; i < num_local_gpus_; i++) {
    int device_id = resource_manager_->get_local_gpu(i)->get_device_id();
    CudaDeviceContext context(device_id);
    CK_CUDA_THROW_(cudaFree(checked_count_[i]));
    CK_CUDA_THROW_(cudaFree(hit_count_[i]));
  }
}

template <typename T>
HitRate<T>::~HitRate() {
  free_all();
}

template <typename T>
__global__ void collect_hits(T* preds, T* labels, int num_samples, int* checked, int* hits) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_samples;
       i += blockDim.x * gridDim.x) {
    if (preds[i] > 0.8) {
      atomicAdd(checked, 1);
      if (labels[i] == 1.0) {
        atomicAdd(hits, 1);
      }
    }
  }
}

template <typename T>
void HitRate<T>::local_reduce(int local_gpu_id, RawMetricMap raw_metrics) {
  const auto& local_gpu = resource_manager_->get_local_gpu(local_gpu_id);
  CudaDeviceContext context(local_gpu->get_device_id());

  int global_device_id = resource_manager_->get_local_gpu(local_gpu_id)->get_global_id();
  int num_valid_samples =
      get_num_valid_samples(global_device_id, current_batch_size_, batch_size_per_gpu_);

  Tensor2<T> pred_tensor = Tensor2<T>::stretch_from(raw_metrics[RawType::Pred]);
  Tensor2<T> label_tensor = Tensor2<T>::stretch_from(raw_metrics[RawType::Label]);

  dim3 grid(160, 1, 1);
  dim3 block(1024, 1, 1);

  cudaMemsetAsync(checked_count_[local_gpu_id], 0, sizeof(int), local_gpu->get_stream());
  cudaMemsetAsync(hit_count_[local_gpu_id], 0, sizeof(int), local_gpu->get_stream());

  collect_hits<T><<<grid, block, 0, local_gpu->get_stream()>>>(
      pred_tensor.get_ptr(), label_tensor.get_ptr(), num_valid_samples,
      checked_count_[local_gpu_id], hit_count_[local_gpu_id]);
  int checked_host = 0;
  int hits_host = 0;
  CK_CUDA_THROW_(cudaMemcpyAsync(&checked_host, checked_count_[local_gpu_id], sizeof(int),
                                 cudaMemcpyDeviceToHost, local_gpu->get_stream()));
  checked_local_[local_gpu_id] = checked_host;
  CK_CUDA_THROW_(cudaMemcpyAsync(&hits_host, hit_count_[local_gpu_id], sizeof(int),
                                 cudaMemcpyDeviceToHost, local_gpu->get_stream()));
  hits_local_[local_gpu_id] = hits_host;
}

template <typename T>
void HitRate<T>::global_reduce(int n_nets) {
  int checked_inter = 0;
  int hits_inter = 0;

  for (auto& hits_local : hits_local_) {
    hits_inter += hits_local;
  }
  for (auto& checked_local : checked_local_) {
    checked_inter += checked_local;
  }

#ifdef ENABLE_MPI
  if (resource_manager_->get_num_process() > 1) {
    int hits_reduced = 0;
    int checked_reduced = 0;
    CK_MPI_THROW_(MPI_Reduce(&hits_inter, &hits_reduced, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
    CK_MPI_THROW_(
        MPI_Reduce(&checked_inter, &checked_reduced, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
    hits_inter = hits_reduced;
    checked_inter = checked_reduced;
  }
#endif

  hits_global_ += hits_inter;
  checked_global_ += checked_inter;

  n_batches_++;
}

template <typename T>
float HitRate<T>::finalize_metric() {
  float ret = 0.0f;
  if (resource_manager_->is_master_process()) {
    if (n_batches_) {
      ret = ((float)hits_global_) / (float)(checked_global_);
    }
  }
#ifdef ENABLE_MPI
  CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
  CK_MPI_THROW_(MPI_Bcast(&ret, 1, MPI_FLOAT, 0, MPI_COMM_WORLD));
#endif
  hits_global_ = 0;
  checked_global_ = 0;
  for (auto& hits_local : hits_local_) {
    hits_local = 0;
  }
  for (auto& checked_local : checked_local_) {
    checked_local = 0;
  }
  n_batches_ = 0;
  return ret;
}

template class AverageLoss<float>;
template class AUC<float>;
template class AUC<__half>;
template class HitRate<float>;

}  // namespace metrics

}  // namespace HugeCTR
