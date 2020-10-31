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

#include <cub/cub/cub.cuh>
#include <diagnose.hpp>
#include <metrics.hpp>
#include <utils.cuh>
#include <omp.h>

namespace HugeCTR {

namespace metrics {

namespace {

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


__global__ void find_pivots_kernel(const CountType* bins_sum, int num_bins,
                                   CountType num_samples, int* pivots) {

  int gid_base = blockIdx.x * blockDim.x + threadIdx.x;

  for (int gid = gid_base; gid < num_bins-1; gid += blockDim.x * gridDim.x) {
    int ibin = gid;

    for (int id = bins_sum[ibin]/num_samples; id < bins_sum[ibin+1]/num_samples; id++) {
      pivots[id] = ibin;
    }
  }
}

__global__ void find_partition_offsets_kernel(const CountType* bins_sum, const int* pivots,
                                              int num_partitions, int num_samples,
                                              CountType* offsets) {

  int gid_base = blockIdx.x * blockDim.x + threadIdx.x;

  for (int gid = gid_base; gid < num_partitions-1; gid += blockDim.x * gridDim.x) {
    int ipart = gid;

    offsets[ipart+1]  = bins_sum[ pivots[ipart] ];
  }
  if (gid_base == 0){
    offsets[0] = 0;
    offsets[num_partitions] = num_samples;
  }
}

__device__ inline CountType binsearch(int v, const int* pivots, int pivots_size) {
  int l = 0, r = pivots_size;
  while (r > l) {
    int i = (l+r) / 2;
    if (v <= pivots[i]) {
      r = i;
    }
    else {
      l = i+1;
    }
  }
  return l;
}

__device__ inline int compute_ibin(float v, float minval, float maxval, int num_bins) {
  int ibin_raw = (int) ((v - minval) * num_bins / (maxval - minval));
  return min(max(ibin_raw, 0), num_bins-1);
}

__launch_bounds__(1024, 1)
__global__ void create_partitions_kernel(const float* labels, const float* predictions, const int* pivots,
                                         float pred_min, float pred_max, CountType num_samples,
                                         int num_bins, int num_partitions,
                                         const CountType* global_partition_offsets,
                                         CountType* global_partition_sizes,
                                         float* part_labels, float* part_preds)
{
  int gid_base = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ extern CountType shmem[];
  CountType* local_partition_sizes   = shmem;
  CountType* local_partition_offsets = shmem + num_partitions;

  for (int id = threadIdx.x; id < num_partitions; id += blockDim.x) {
    local_partition_sizes[id] = 0;
  }
  __syncthreads();

  for (CountType gid = gid_base; gid < num_samples; gid += blockDim.x * gridDim.x) {

    float value = predictions[gid];
    int ibin  = compute_ibin(value, pred_min, pred_max, num_bins);
    int ipart = binsearch(ibin, pivots, num_partitions-1);

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
    int ibin  = compute_ibin(value, pred_min, pred_max, num_bins);
    int ipart = binsearch(ibin, pivots, num_partitions-1);

    CountType my_glob_part_offset = global_partition_offsets[ipart] +
                                    local_partition_offsets [ipart] +
                                    atomicAdd(local_partition_sizes + ipart, 1);

    part_labels[my_glob_part_offset] = labels[gid];
    part_preds [my_glob_part_offset] = value;
  }
}

__global__ void rate_from_part_cumsum_kernel(
                              const float* cumsum,
                              CountType num_samples, CountType* offset, CountType* total,
                              float* rate) {

  CountType gid_base = blockIdx.x * blockDim.x + threadIdx.x;

  for (CountType gid = gid_base; gid < num_samples; gid += blockDim.x * gridDim.x) {
    rate[gid] = (*total - (*offset + cumsum[gid])) / (float)(*total);
  }
}

__global__ void flatten_segments_kernel(const CountType* offsets, const CountType* lengths,
                                        const int* num_segments,
                                        float* tps, float* fps) {
                                          
  CountType gid_base = blockIdx.x * blockDim.x + threadIdx.x;
  CountType wid = gid_base / warpSize;
  int lane_id = gid_base % warpSize;

  int increment = (blockDim.x * gridDim.x) / warpSize;
  for (CountType seg_id = wid; seg_id < *num_segments; seg_id += increment ) {
    CountType offset = offsets[seg_id];
    CountType length = lengths[seg_id];
    float value_tp = tps[offset+length-1]; 
    float value_fp = fps[offset+length-1]; 

    for (int el = lane_id; el < length; el += warpSize) {
      tps[offset + el] = value_tp;
      fps[offset + el] = value_fp;
    }
  }
}

__global__ void trapz_kernel(float* y, float* x, float* halo_y, float* halo_x, float* auc, CountType num_samples) {
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
      my_area += 0.5f * (*halo_y+fa)*(*halo_x-a);
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

template <typename T> ncclDataType_t get_nccl_type();
template <>           ncclDataType_t get_nccl_type<int>                () { return ncclInt32;    }
template <>           ncclDataType_t get_nccl_type<unsigned int>       () { return ncclUint32;   }
template <>           ncclDataType_t get_nccl_type<unsigned long long> () { return ncclUint64;   }
template <>           ncclDataType_t get_nccl_type<float>              () { return ncclFloat32; }
template <>           ncclDataType_t get_nccl_type<__half>             () { return ncclFloat16; }

template <typename T>
void allreduce(T* srcptr, T* dstptr, int count, const GPUResource* gpu_resource)
{
  auto& stream = gpu_resource->get_stream();
  CK_NCCL_THROW_(ncclAllReduce(
    srcptr, dstptr, count, get_nccl_type<T>(), ncclSum,
    gpu_resource->get_nccl(), stream));
}

template <typename T>
void allgather(T* srcptr, T* dstptr, int src_count, const GPUResource* gpu_resource)
{
  auto& stream = gpu_resource->get_stream();
  CK_NCCL_THROW_(ncclAllGather(
    srcptr, dstptr, src_count, get_nccl_type<T>(), 
    gpu_resource->get_nccl(), stream));
}

template <typename T>
void all_to_all(T* srcptr, T* dstptr,
                const CountType* src_offsets, const CountType* dst_offsets,
                int num_global_gpus, const GPUResource* gpu_resource)
{
  auto& stream = gpu_resource->get_stream();
  auto& comm   = gpu_resource->get_nccl();
  auto type = get_nccl_type<T>();

  CK_NCCL_THROW_(ncclGroupStart());
  for (int i=0; i < num_global_gpus; i++) {
    CK_NCCL_THROW_(ncclSend(srcptr + src_offsets[i], src_offsets[i+1] - src_offsets[i], type, i, comm, stream));
    CK_NCCL_THROW_(ncclRecv(dstptr + dst_offsets[i], dst_offsets[i+1] - dst_offsets[i], type, i, comm, stream));
  }
  CK_NCCL_THROW_(ncclGroupEnd());
}

template <typename T>
void send_halo_right(T* srcptr, T* dstptr, int count, int num_global_gpus, const GPUResource* gpu_resource)
{
  auto& stream = gpu_resource->get_stream();
  auto& comm   = gpu_resource->get_nccl();
  int my_global_id = gpu_resource->get_global_gpu_id();
  auto type = get_nccl_type<T>();

  CK_NCCL_THROW_(ncclGroupStart());
  if (my_global_id < num_global_gpus-1) {
    CK_NCCL_THROW_(ncclSend(srcptr, count, type, my_global_id+1, comm, stream));
  }
  if (my_global_id > 0) {
    CK_NCCL_THROW_(ncclRecv(dstptr, count, type, my_global_id-1, comm, stream));
  }
  CK_NCCL_THROW_(ncclGroupEnd());
}

} // namespace metric_comm


std::unique_ptr<Metric> Metric::Create(const Type type, bool use_mixed_precision,
                                       int batch_size_eval, int n_batches,
                                       const std::shared_ptr<ResourceManager>& resource_manager) {
  std::unique_ptr<Metric> ret;
  switch (type) {
    case Type::AUC:
      if (use_mixed_precision) {
        ret.reset(new AUC<__half>(batch_size_eval, n_batches, resource_manager));
      } else {
        ret.reset(new AUC<float> (batch_size_eval, n_batches, resource_manager));
      }
      break;
    case Type::AverageLoss:
      ret.reset(new AverageLoss<float>(resource_manager));
      break;
  }
  return ret;
}

Metric::Metric() : num_procs_(1), pid_(0), current_batch_size_(0) {
#ifdef ENABLE_MPI
  CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &pid_));
  CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &num_procs_));
#endif
}
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
  CudaDeviceContext context(resource_manager_->get_local_gpu(local_gpu_id)->get_device_id());
  CK_CUDA_THROW_(
      cudaMemcpy(&loss_host, loss_tensor.get_ptr(), sizeof(float), cudaMemcpyDeviceToHost));
  loss_local_[local_gpu_id] = loss_host;
}

template <typename T>
void AverageLoss<T>::global_reduce(int n_nets) {
  float loss_inter = 0.0f;
  for (auto& loss_local : loss_local_) {
    loss_inter += loss_local;
  }

#ifdef ENABLE_MPI
  if (num_procs_ > 1) {
    float loss_reduced = 0.0f;
    CK_MPI_THROW_(MPI_Reduce(&loss_inter, &loss_reduced, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));
    loss_inter = loss_reduced;
  }
#endif
  loss_global_ += loss_inter / n_nets / num_procs_;
  n_batches_++;
}

template <typename T>
float AverageLoss<T>::finalize_metric() {
  float ret = 0.0f;
  if (pid_ == 0) {
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
  size_t bins_buffer_size          = num_bins                 *sizeof(CountType);

  CK_CUDA_THROW_(cudaMalloc       ((void**)&(ptr_local_bins_            ), bins_buffer_size));
  CK_CUDA_THROW_(cudaMalloc       ((void**)&(ptr_global_bins_           ), bins_buffer_size));
  CK_CUDA_THROW_(cudaMalloc       ((void**)&(ptr_global_bins_sum_       ), bins_buffer_size));
  CK_CUDA_THROW_(cudaMalloc       ((void**)&(ptr_local_bins_sum_        ), bins_buffer_size));
  CK_CUDA_THROW_(cudaMalloc       ((void**)&(ptr_pivots_                ), num_partitions*sizeof(int)));
  CK_CUDA_THROW_(cudaMallocManaged((void**)&(ptr_partition_offsets_     ), (num_partitions+1)*sizeof(CountType)));
  CK_CUDA_THROW_(cudaMallocManaged((void**)&(ptr_all_partition_offsets_ ), (num_partitions+1)*num_global_gpus*sizeof(CountType)));
  CK_CUDA_THROW_(cudaMallocManaged((void**)&(ptr_recv_offsets_          ), (num_partitions+1)*sizeof(CountType)));
  CK_CUDA_THROW_(cudaMalloc       ((void**)&(ptr_pos_per_gpu_           ), (num_global_gpus+1)*sizeof(CountType)));
  CK_CUDA_THROW_(cudaMalloc       ((void**)&(ptr_neg_per_gpu_           ), (num_global_gpus+1)*sizeof(CountType)));
  CK_CUDA_THROW_(cudaMalloc       ((void**)&(ptr_num_identical_segments_), sizeof(int)));
  CK_CUDA_THROW_(cudaMalloc       ((void**)&(ptr_halo_tpr_              ), sizeof(float)));
  CK_CUDA_THROW_(cudaMalloc       ((void**)&(ptr_halo_fpr_              ), sizeof(float)));
  CK_CUDA_THROW_(cudaMalloc       ((void**)&(ptr_tp_offsets_            ), (num_global_gpus+1)*sizeof(CountType)));
  CK_CUDA_THROW_(cudaMalloc       ((void**)&(ptr_fp_offsets_            ), (num_global_gpus+1)*sizeof(CountType)));
  CK_CUDA_THROW_(cudaMallocManaged((void**)&(ptr_auc_                   ), sizeof(float)));

  CK_CUDA_THROW_(cudaMemset(ptr_pos_per_gpu_, 0, (num_global_gpus+1)*sizeof(CountType)));
  CK_CUDA_THROW_(cudaMemset(ptr_neg_per_gpu_, 0, (num_global_gpus+1)*sizeof(CountType)));

  realloc_redistributed(num_local_samples);
}

void AUCStorage::realloc_redistributed(size_t num_redistributed_samples) {
  if (num_redistributed_samples > num_allocated_redistributed_ ) {

    free_redistributed();
    num_allocated_redistributed_ = imbalance_factor_ * num_redistributed_samples;
    size_t redistributed_buffer_size = num_allocated_redistributed_*sizeof(float);
    size_t runs_buffer_size = num_allocated_redistributed_*sizeof(CountType)/2;

    CK_CUDA_THROW_(cudaMalloc       ((void**)&(ptr_preds_1_               ), redistributed_buffer_size));
    CK_CUDA_THROW_(cudaMalloc       ((void**)&(ptr_labels_1_              ), redistributed_buffer_size));    
    CK_CUDA_THROW_(cudaMalloc       ((void**)&(ptr_preds_2_               ), redistributed_buffer_size));
    CK_CUDA_THROW_(cudaMalloc       ((void**)&(ptr_labels_2_              ), redistributed_buffer_size));
    CK_CUDA_THROW_(cudaMalloc       ((void**)&(ptr_identical_pred_starts_ ), runs_buffer_size));
    CK_CUDA_THROW_(cudaMalloc       ((void**)&(ptr_identical_pred_lengths_), runs_buffer_size));


  }
}

void AUCStorage::set_max_temp_storage_bytes(size_t new_val) {
  temp_storage_bytes_ = std::max(new_val, temp_storage_bytes_);
}

void AUCStorage::alloc_workspace() {
  CK_CUDA_THROW_(cudaMalloc       ((void**)&(workspace_), temp_storage_bytes_));
}

void AUCStorage::free_redistributed() {
  cudaFree(ptr_preds_1_ );
  cudaFree(ptr_labels_1_);
  cudaFree(ptr_preds_2_ );
  cudaFree(ptr_labels_2_);
}

void AUCStorage::free_all() {
  cudaFree(ptr_local_bins_            );
  cudaFree(ptr_global_bins_           );
  cudaFree(ptr_global_bins_sum_       );
  cudaFree(ptr_local_bins_sum_        );
  cudaFree(ptr_pivots_                );
  cudaFree(ptr_partition_offsets_     );
  cudaFree(ptr_all_partition_offsets_ );
  cudaFree(ptr_recv_offsets_          );
  cudaFree(ptr_pos_per_gpu_           );
  cudaFree(ptr_neg_per_gpu_           );
  cudaFree(ptr_num_identical_segments_);
  cudaFree(ptr_halo_tpr_              );
  cudaFree(ptr_halo_fpr_              );
  cudaFree(ptr_tp_offsets_            );
  cudaFree(ptr_fp_offsets_            );
  cudaFree(ptr_auc_                   );
  cudaFree(workspace_                 );

  free_redistributed();
}

AUCBarrier::AUCBarrier(std::size_t thread_count) : 
  threshold_(thread_count), 
  count_(thread_count), 
  generation_(0) {
}

void AUCBarrier::wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  auto gen = generation_;
  if (!--count_) {
    generation_++;
    count_ = threshold_;
    cond_.notify_all();
  } else {
    cond_.wait(lock, [this, gen] { return gen != generation_; });
  }
}

template <typename T>
AUC<T>::AUC(int batch_size_per_gpu, int n_batches,
            const std::shared_ptr<ResourceManager>& resource_manager)
    : Metric(),
      resource_manager_(resource_manager),
      batch_size_per_gpu_(batch_size_per_gpu),
      n_batches_(n_batches),
      num_local_gpus_ (resource_manager_->get_local_gpu_count()),
      num_global_gpus_(resource_manager_->get_global_gpu_count()),
      num_bins_(num_global_gpus_ * num_bins_per_gpu_),
      num_partitions_(num_global_gpus_),
      num_total_samples_(0),
      barrier_(num_local_gpus_),
      storage_(num_local_gpus_),
      offsets_(num_local_gpus_, 0) {

  size_t max_num_local_samples = (batch_size_per_gpu_ * n_batches_);
  size_t est_max_num_redistributed_samples = 2*max_num_local_samples;

  for (int i=0; i<num_local_gpus_; i++) {
    int device_id = resource_manager_->get_local_gpu(i)->get_device_id();
    CudaDeviceContext context(device_id);

    auto& st = storage_[i];
    st.alloc_main(max_num_local_samples, num_bins_, num_partitions_, num_global_gpus_);

    st.set_max_temp_storage_bytes(num_partitions_*sizeof(CountType));
    size_t new_temp_storage_bytes;

    // (int) casting is a CUB workaround. Fixed in https://github.com/thrust/cub/pull/38
    CK_CUDA_THROW_(cub::DeviceHistogram::HistogramEven(
      nullptr, new_temp_storage_bytes, st.d_preds(), st.d_local_bins(),
      num_bins_+1, pred_min_, pred_max_, (int)max_num_local_samples));
    st.set_max_temp_storage_bytes(new_temp_storage_bytes);

    CK_CUDA_THROW_(cub::DeviceScan::InclusiveSum(
      nullptr, new_temp_storage_bytes, st.d_global_bins(), st.d_global_bins_sum(), num_bins_));
    st.set_max_temp_storage_bytes(new_temp_storage_bytes);

    CK_CUDA_THROW_(cub::DeviceRadixSort::SortPairsDescending(nullptr, new_temp_storage_bytes,
                                                             st.d_presorted_preds(),  st.d_sorted_preds(),
                                                             st.d_presorted_labels(), st.d_sorted_labels(),
                                                             est_max_num_redistributed_samples));
    st.set_max_temp_storage_bytes(new_temp_storage_bytes);

    CK_CUDA_THROW_(cub::DeviceScan::InclusiveSum(
      nullptr, new_temp_storage_bytes, st.d_presorted_labels(), st.d_tp(), est_max_num_redistributed_samples));
    st.set_max_temp_storage_bytes(new_temp_storage_bytes);

    CK_CUDA_THROW_(cub::DeviceScan::ExclusiveSum(
      nullptr, new_temp_storage_bytes, st.d_pos_per_gpu(), st.d_tp_offsets(), num_global_gpus_));
    st.set_max_temp_storage_bytes(new_temp_storage_bytes);

    CK_CUDA_THROW_(cub::DeviceRunLengthEncode::NonTrivialRuns(nullptr, new_temp_storage_bytes,
                                                              st.d_sorted_preds(),
                                                              st.d_identical_pred_starts(),
                                                              st.d_identical_pred_lengths(),
                                                              st.d_num_identical_segments(),
                                                              est_max_num_redistributed_samples));
    st.set_max_temp_storage_bytes(new_temp_storage_bytes);

    st.alloc_workspace();
  }
}

template <typename T>
AUC<T>::~AUC() {
  for (int i=0; i<num_local_gpus_; i++) {
    int device_id = resource_manager_->get_local_gpu(i)->get_device_id();
    CudaDeviceContext context(device_id);

    storage_[i].free_all();
  }
}

int get_num_valid_samples(int global_device_id, int current_batch_size, int batch_per_gpu) {
  int remaining = current_batch_size - global_device_id*batch_per_gpu;
  return std::max(std::min(remaining, batch_per_gpu), 0); 
}

template <typename T>
void AUC<T>::local_reduce(int local_gpu_id, RawMetricMap raw_metrics) {
  Tensor2<PredType>   pred_tensor = Tensor2<PredType >::stretch_from(raw_metrics[RawType::Pred ]);
  Tensor2<LabelType> label_tensor = Tensor2<LabelType>::stretch_from(raw_metrics[RawType::Label]);
  int device_id        = resource_manager_->get_local_gpu(local_gpu_id)->get_device_id();
  int global_device_id = resource_manager_->get_local_gpu(local_gpu_id)->get_global_gpu_id();
  const auto& st = storage_[local_gpu_id];
 
  // Copy the labels and predictions to the internal buffer
  size_t& offset = offsets_[local_gpu_id];
  CudaDeviceContext context(device_id);
  int num_valid_samples = get_num_valid_samples(global_device_id,
                                                current_batch_size_, batch_size_per_gpu_);

  copy_all<T>(st.d_preds() + offset, st.d_labels() + offset,
              pred_tensor.get_ptr(), label_tensor.get_ptr(), num_valid_samples,
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
float AUC<T>::finalize_metric() {

  int num_local_gpus = resource_manager_->get_local_gpu_count();
  float result[num_local_gpus];
  #pragma omp parallel num_threads(num_local_gpus)
  {
    result[omp_get_thread_num()] = _finalize_metric_per_gpu(omp_get_thread_num());
  }

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
  int global_id = gpu_resource->get_global_gpu_id();
  auto& stream  = gpu_resource->get_stream();


  CudaDeviceContext context(device_id);
  auto& st = storage_[local_id];
  size_t num_local_samples = offsets_[local_id];
  offsets_[local_id] = 0;


  // 1. Create local histograms of predictions
  float eps = 1e-7f;
  float loc_min = pred_min_ + eps;
  float loc_max = pred_max_ - eps;
  auto clamp = [loc_min, loc_max] __device__ (const float v) {
    return fmaxf(fminf(v, loc_max), loc_min);
  };
  cub::TransformInputIterator<float, decltype(clamp), float*>
    d_clamped_preds(st.d_preds(), clamp);

  // (int) casting is a CUB workaround. Fixed in https://github.com/thrust/cub/pull/38
  CK_CUDA_THROW_(cub::DeviceHistogram::HistogramEven(
    st.d_workspace(), st.temp_storage_bytes(), d_clamped_preds, st.d_local_bins(),
    num_bins_+1, pred_min_, pred_max_, (int)num_local_samples, stream));


  // 2. Allreduce histograms
  metric_comm::allreduce(st.d_local_bins(), st.d_global_bins(), num_bins_, gpu_resource);


  // 3. Find num_global_gpus_-1 pivot points
  CK_CUDA_THROW_(cub::DeviceScan::InclusiveSum(
    st.d_workspace(), st.temp_storage_bytes(),
    st.d_global_bins(), st.d_global_bins_sum(),
    num_bins_, stream));

  initialize_array  <<<grid, block, 0, stream>>>(st.d_pivots(), num_partitions_-1, num_bins_-1);
  find_pivots_kernel<<<grid, block, 0, stream>>>(st.d_global_bins_sum(), num_bins_,
                                                 num_total_samples_ / num_global_gpus_, st.d_pivots());


  // 4. Partition (partially sort) local predictions into num_global_gpus_ bins
  //    separated by pivot points.
  CK_CUDA_THROW_(cub::DeviceScan::InclusiveSum(
    st.d_workspace(), st.temp_storage_bytes(),
    st.d_local_bins(), st.d_local_bins_sum(),
    num_bins_, stream));

  find_partition_offsets_kernel<<<grid, block, 0, stream>>> (
    st.d_local_bins_sum(), st.d_pivots(),
    num_partitions_, num_local_samples,
    st.d_partition_offsets());

  initialize_array<<<grid, block, 0, stream>>>((CountType*)st.d_workspace(), num_partitions_, (CountType)0);

  size_t shmem_size = sizeof(CountType) * num_partitions_ * 2;
  create_partitions_kernel<<<grid, block, shmem_size, stream>>> (
    st.d_labels(), st.d_preds(), st.d_pivots(), pred_min_, pred_max_,
    num_local_samples, num_bins_, num_partitions_,
    st.d_partition_offsets(), (CountType*)st.d_workspace(),
    st.d_partitioned_labels(), st.d_partitioned_preds());


  // 5. Exchange the data such that all predicitons on GPU i are smaller than
  //    the ones on GPU i+1. 
  // 5.1. Compute receiving side offsets.
  metric_comm::allgather(st.d_partition_offsets(), st.d_all_partition_offsets(), num_partitions_+1,
                         gpu_resource);

  // The following is done on the CPU, need to wait
  CK_CUDA_THROW_(cudaStreamSynchronize(stream));

  CountType sum = 0;
  for (int src=0; src<num_global_gpus_; src++) {
    CountType size_src = st.d_all_partition_offsets()[src*(num_partitions_+1) + global_id+1] -
                         st.d_all_partition_offsets()[src*(num_partitions_+1) + global_id  ];
    st.d_recv_offsets()[src] = sum;
    sum += size_src;
  }
  st.d_recv_offsets()[num_global_gpus_] = sum;

  const size_t num_redistributed_samples = st.d_recv_offsets()[num_global_gpus_];
  st.realloc_redistributed(num_redistributed_samples);

  // 5.2 Synchronize threads before all to all to prevent hangs
  barrier_.wait();

  // 5.3 All to all
  metric_comm::all_to_all(st.d_partitioned_labels(), st.d_presorted_labels(),
                          st.d_partition_offsets(),  st.d_recv_offsets(),
                          num_global_gpus_, gpu_resource);
  metric_comm::all_to_all(st.d_partitioned_preds(),  st.d_presorted_preds(),
                          st.d_partition_offsets(),  st.d_recv_offsets(),
                          num_global_gpus_, gpu_resource);


  // 6. Locally sort (label, pred) by pred
  CK_CUDA_THROW_(cub::DeviceRadixSort::SortPairs(st.d_workspace(), st.temp_storage_bytes(),
                                                 st.d_presorted_preds(),  st.d_sorted_preds(),
                                                 st.d_presorted_labels(), st.d_sorted_labels(),
                                                 num_redistributed_samples,
                                                 0, sizeof(float)*8, // begin_bit, end_bit
                                                 stream));


  // 7. Create TPR and FPR. Need a "global" scan
  // 7.1 Local inclusive scan to find TP and FP
  CK_CUDA_THROW_(cub::DeviceScan::InclusiveSum(
    st.d_workspace(), st.temp_storage_bytes(),
    st.d_sorted_labels(), st.d_tp(),
    num_redistributed_samples, stream));

  auto one_minus_val = [] __device__ (const float v) {
    return 1.0f-v;
  };
  cub::TransformInputIterator<float, decltype(one_minus_val), float*>
    d_one_minus_labels(st.d_sorted_labels(), one_minus_val);
  CK_CUDA_THROW_(cub::DeviceScan::InclusiveSum(
    st.d_workspace(), st.temp_storage_bytes(),
    d_one_minus_labels, st.d_fp(),
    num_redistributed_samples, stream));

  // 7.2 'Flatten' tp and fp for cases where several consecutive predictions are identical
  CK_CUDA_THROW_(cub::DeviceRunLengthEncode::NonTrivialRuns(st.d_workspace(), st.temp_storage_bytes(),
                                                            st.d_sorted_preds(),
                                                            st.d_identical_pred_starts(),
                                                            st.d_identical_pred_lengths(),
                                                            st.d_num_identical_segments(),
                                                            num_redistributed_samples,
                                                            stream));

  flatten_segments_kernel<<<grid, block, 0, stream>>> (st.d_identical_pred_starts(),
                                                       st.d_identical_pred_lengths(),
                                                       st.d_num_identical_segments(),
                                                       st.d_tp(),
                                                       st.d_fp());

  // 7.3 Allgather of the number of total positive and negative samples
  //     on each GPU and exclusive scan them
  metric_comm::allgather(st.d_tp()+num_redistributed_samples-1, st.d_pos_per_gpu(), 1, 
                         gpu_resource);
  metric_comm::allgather(st.d_fp()+num_redistributed_samples-1, st.d_neg_per_gpu(), 1,
                         gpu_resource);

  CK_CUDA_THROW_(cub::DeviceScan::ExclusiveSum(st.d_workspace(), st.temp_storage_bytes(), 
                                               st.d_pos_per_gpu(), st.d_tp_offsets(),
                                               num_global_gpus_+1, stream));

  CK_CUDA_THROW_(cub::DeviceScan::ExclusiveSum(st.d_workspace(), st.temp_storage_bytes(), 
                                               st.d_neg_per_gpu(), st.d_fp_offsets(),
                                               num_global_gpus_+1, stream));



  // 7.4 Locally find TPR and FPR given TP, FP and global offsets
  rate_from_part_cumsum_kernel<<<grid, block, 0, stream>>> (st.d_tp(), num_redistributed_samples,
                                                            st.d_tp_offsets() + global_id,
                                                            st.d_tp_offsets() + num_global_gpus_,
                                                            st.d_tpr());
  rate_from_part_cumsum_kernel<<<grid, block, 0, stream>>> (st.d_fp(), num_redistributed_samples,
                                                            st.d_fp_offsets() + global_id,
                                                            st.d_fp_offsets() + num_global_gpus_,
                                                            st.d_fpr());


  // 8. Integrate TPR and FPR taking into account halo samples
  metric_comm::send_halo_right(st.d_tpr()+num_redistributed_samples-1, st.d_halo_tpr(), 1,
                               num_global_gpus_, gpu_resource);
  metric_comm::send_halo_right(st.d_fpr()+num_redistributed_samples-1, st.d_halo_fpr(), 1, 
                               num_global_gpus_, gpu_resource);

  if ( global_id == 0 ) {
    initialize_array<<<grid, block, 0, stream>>>(st.d_halo_tpr(), 1, 1.0f);
    initialize_array<<<grid, block, 0, stream>>>(st.d_halo_fpr(), 1, 1.0f);
  }

  initialize_array<<<grid, block, 0, stream>>>(st.d_auc(), 1, 0.0f);
  trapz_kernel    <<<grid, block, 0, stream>>>(st.d_tpr(), st.d_fpr(),
                                               st.d_halo_tpr(), st.d_halo_fpr(),
                                               st.d_auc(), num_redistributed_samples);
  

  // 9. Finally allreduce auc
  metric_comm::allreduce(st.d_auc(), st.d_auc(), 1, gpu_resource);

  CK_CUDA_THROW_(cudaStreamSynchronize(stream));
  return *st.d_auc();
}

template class AverageLoss<float>;
template class AUC<float>;
template class AUC<__half>;

}  // namespace metrics

}  // namespace HugeCTR
