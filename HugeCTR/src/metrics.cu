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

#define MMAP_DEBUG(...)  // HCTR_LOG(INFO, ROOT, __VA_ARGS__)

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

__global__ void copy_pred_half_kernel(float* y_pred, const __half* x_pred, int num_elems) {
  int gid_base = blockIdx.x * blockDim.x + threadIdx.x;
  for (int gid = gid_base; gid < num_elems; gid += blockDim.x * gridDim.x) {
    float pred_val = __half2float(x_pred[gid]);
    y_pred[gid] = pred_val;
  }
}

template <typename SrcType>
void copy_pred(float* y, SrcType* x, int num_elems, int num_sms, cudaStream_t stream);

template <>
void copy_pred<float>(float* y, float* x, int num_elems, int num_sms, cudaStream_t stream) {
  HCTR_LIB_THROW(
      cudaMemcpyAsync(y, x, num_elems * sizeof(float), cudaMemcpyDeviceToDevice, stream));
}

template <>
void copy_pred<__half>(float* y, __half* x, int num_elems, int num_sms, cudaStream_t stream) {
  dim3 grid(num_sms * 2, 1, 1);
  dim3 block(1024, 1, 1);
  copy_pred_half_kernel<<<grid, block, 0, stream>>>(y, x, num_elems);
}

template <typename PredType>
void copy_all(float* y_pred, float* y_label, PredType* x_pred, float* x_label, int num_elems,
              int num_sms, cudaStream_t stream);

template <>
void copy_all<float>(float* y_pred, float* y_label, float* x_pred, float* x_label, int num_elems,
                     int num_sms, cudaStream_t stream) {
  copy_pred<float>(y_pred, x_pred, num_elems, num_sms, stream);
  HCTR_LIB_THROW(cudaMemcpyAsync(y_label, x_label, num_elems * sizeof(float),
                                 cudaMemcpyDeviceToDevice, stream));
}

template <>
void copy_all<__half>(float* y_pred, float* y_label, __half* x_pred, float* x_label, int num_elems,
                      int num_sms, cudaStream_t stream) {
  dim3 grid(num_sms * 2, 1, 1);
  dim3 block(1024, 1, 1);
  copy_all_kernel<<<grid, block, 0, stream>>>(y_pred, y_label, x_pred, x_label, num_elems);
}

__global__ void init_classes_kernel(int* classes, size_t num_valid_samples, size_t num_classes) {
  size_t tid_base = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t tid = tid_base; tid < num_valid_samples * num_classes;
       tid += blockDim.x * gridDim.x) {
    classes[tid] = tid % num_classes;
  }
}

void init_classes(int* classes, size_t num_valid_samples, size_t num_classes, int num_sms,
                  cudaStream_t stream) {
  dim3 grid(num_sms * 2, 1, 1);
  dim3 block(1024, 1, 1);
  init_classes_kernel<<<grid, block, 0, stream>>>(classes, num_valid_samples, num_classes);
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

// This version partitions both preds and labels by value of preds
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

// This version partitions labels only
__launch_bounds__(1024, 1) __global__
    void create_partitions_kernel(const float* labels, const int* pivots, float label_min,
                                  float label_max, CountType num_samples, int num_bins,
                                  int num_partitions, const CountType* global_partition_offsets,
                                  CountType* global_partition_sizes, float* part_labels) {
  int gid_base = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ extern CountType shmem[];
  CountType* local_partition_sizes = shmem;
  CountType* local_partition_offsets = shmem + num_partitions;

  for (int id = threadIdx.x; id < num_partitions; id += blockDim.x) {
    local_partition_sizes[id] = 0;
  }
  __syncthreads();

  for (CountType gid = gid_base; gid < num_samples; gid += blockDim.x * gridDim.x) {
    float value = labels[gid];
    int ibin = compute_ibin(value, label_min, label_max, num_bins);
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
    float value = labels[gid];
    int ibin = compute_ibin(value, label_min, label_max, num_bins);
    int ipart = binsearch(ibin, pivots, num_partitions - 1);

    CountType my_glob_part_offset = global_partition_offsets[ipart] +
                                    local_partition_offsets[ipart] +
                                    atomicAdd(local_partition_sizes + ipart, 1);

    part_labels[my_glob_part_offset] = value;
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
void allreduce(T* srcptr, T* dstptr, int count, const GPUResource* gpu_resource,
               const cudaStream_t& stream) {
  HCTR_LIB_THROW(ncclAllReduce(srcptr, dstptr, count, get_nccl_type<T>(), ncclSum,
                               gpu_resource->get_nccl(), stream));
}

template <typename T>
void allgather(T* srcptr, T* dstptr, int src_count, const GPUResource* gpu_resource,
               const cudaStream_t& stream) {
  HCTR_LIB_THROW(ncclAllGather(srcptr, dstptr, src_count, get_nccl_type<T>(),
                               gpu_resource->get_nccl(), stream));
}

template <typename T>
void all_to_all(T* srcptr, T* dstptr, const CountType* src_offsets, const CountType* dst_offsets,
                int num_global_gpus, const GPUResource* gpu_resource, const cudaStream_t& stream) {
  auto& comm = gpu_resource->get_nccl();
  auto type = get_nccl_type<T>();

  HCTR_LIB_THROW(ncclGroupStart());
  for (int i = 0; i < num_global_gpus; i++) {
    HCTR_LIB_THROW(ncclSend(srcptr + src_offsets[i], src_offsets[i + 1] - src_offsets[i], type, i,
                            comm, stream));
    HCTR_LIB_THROW(ncclRecv(dstptr + dst_offsets[i], dst_offsets[i + 1] - dst_offsets[i], type, i,
                            comm, stream));
  }
  HCTR_LIB_THROW(ncclGroupEnd());
}

template <typename T>
void send_halo_right(T* srcptr, T* dstptr, int count, int left_neighbor, int right_neighbor,
                     const GPUResource* gpu_resource, const cudaStream_t& stream) {
  auto& comm = gpu_resource->get_nccl();
  auto type = get_nccl_type<T>();

  HCTR_LIB_THROW(ncclGroupStart());
  if (right_neighbor >= 0) {
    HCTR_LIB_THROW(ncclSend(srcptr, count, type, right_neighbor, comm, stream));
  }
  if (left_neighbor >= 0) {
    HCTR_LIB_THROW(ncclRecv(dstptr, count, type, left_neighbor, comm, stream));
  }
  HCTR_LIB_THROW(ncclGroupEnd());
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
      std::ostringstream os;
      os << "num elements: " << raw_metric_tensor.get_num_elements() << " not match with " << num;
      HCTR_OWN_THROW(Error_t::WrongInput, os.str());
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
      std::ostringstream os;
      os << "num elements: " << device_prediction_result.get_num_elements() << " not match with "
         << num;
      HCTR_OWN_THROW(Error_t::WrongInput, os.str());
    }
  }
  HCTR_LIB_THROW(cudaMemcpy(rst, device_prediction_result.get_ptr(), num * sizeof(float),
                            cudaMemcpyDeviceToHost));
}

std::unique_ptr<Metric> Metric::Create(const Type type, bool use_mixed_precision,
                                       int batch_size_eval, int n_batches, int label_dim,
                                       const std::shared_ptr<ResourceManager>& resource_manager) {
  std::unique_ptr<Metric> ret;
  switch (type) {
    case Type::AUC:
      if (use_mixed_precision) {
        ret.reset(new AUC<__half>(batch_size_eval, n_batches, label_dim, resource_manager));
      } else {
        ret.reset(new AUC<float>(batch_size_eval, n_batches, label_dim, resource_manager));
      }
      break;
    case Type::AverageLoss:
      ret.reset(new AverageLoss<float>(resource_manager));
      break;
    case Type::HitRate:
      ret.reset(new HitRate<float>(batch_size_eval, resource_manager));
      break;
    case Type::NDCG:
      if (use_mixed_precision) {
        ret.reset(new NDCG<__half>(batch_size_eval, n_batches, resource_manager));
      } else {
        ret.reset(new NDCG<float>(batch_size_eval, n_batches, resource_manager));
      }
      break;
    case Type::SMAPE:
      ret.reset(new SMAPE<float>(batch_size_eval, resource_manager));
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
      loss_local_(std::vector<float*>(resource_manager->get_local_gpu_count(), nullptr)),
      loss_global_(0.0f),
      n_batches_(0) {
  for (size_t local_gpu_id = 0; local_gpu_id < resource_manager_->get_local_gpu_count();
       ++local_gpu_id) {
    HCTR_LIB_THROW(cudaMallocHost((void**)&loss_local_[local_gpu_id], sizeof(float)));
  }
}

template <typename T>
AverageLoss<T>::~AverageLoss() {
  for (size_t local_gpu_id = 0; local_gpu_id < resource_manager_->get_local_gpu_count();
       ++local_gpu_id) {
    HCTR_LIB_THROW(cudaFreeHost(loss_local_[local_gpu_id]));
  }
}

template <typename T>
void AverageLoss<T>::local_reduce(int local_gpu_id, RawMetricMap raw_metrics) {
  // HCTR_LOG_S(DEBUG, WORLD) << "Called local reduce" << std::endl;
  Tensor2<float> loss_tensor = Tensor2<float>::stretch_from(raw_metrics[RawType::Loss]);
  const auto& local_gpu = resource_manager_->get_local_gpu(local_gpu_id);
  auto& stream = local_gpu->get_stream();
  CudaDeviceContext context(local_gpu->get_device_id());
  HCTR_LIB_THROW(cudaMemcpy(loss_local_[local_gpu_id], loss_tensor.get_ptr(), sizeof(float),
                            cudaMemcpyDeviceToHost));
}

template <typename T>
void AverageLoss<T>::global_reduce(int n_nets) {
  float loss_inter = 0.0f;
  for (auto& ptr_loss_local : loss_local_) {
    loss_inter += *ptr_loss_local;
  }
  loss_global_ += loss_inter / n_nets / resource_manager_->get_num_process();
  n_batches_++;
}

template <typename T>
float AverageLoss<T>::finalize_metric() {
  float ret = loss_global_;
#ifdef ENABLE_MPI
  if (resource_manager_->get_num_process() > 1) {
    float loss_reduced = 0.0f;
    HCTR_MPI_THROW(MPI_Reduce(&ret, &loss_reduced, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));
    ret = loss_reduced;
  }
#endif
  if (resource_manager_->is_master_process()) {
    if (n_batches_) {
      ret = ret / n_batches_;
    }
  }
  loss_global_ = 0.0f;
  n_batches_ = 0;

  return ret;
}

void gen_access_desc(std::vector<CUmemAccessDesc>& access_desc, const std::vector<int>& peers) {
  access_desc.resize(peers.size());
  for (size_t i = 0; i < peers.size(); i++) {
    access_desc[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access_desc[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc[i].location.id = peers[i];
  }
}

void AUCStorage::alloc_main(size_t num_local_samples, size_t num_bins, size_t num_partitions,
                            size_t num_global_gpus, size_t label_dim, size_t num_streams,
                            const std::vector<int>& peers, cudaStream_t stream) {
  num_classes_ = label_dim;
  gen_access_desc(access_desc_, peers);

  num_allocated_redistributed_.resize(num_streams, 0);
  allocated_temp_storage_.resize(num_streams, 0);
  workspace_.resize(num_streams);
  finalize_storage_.resize(num_streams);

  for (auto& st : finalize_storage_) {
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();

    buf->reserve({num_bins}, &st.local_bins_);
    buf->reserve({num_bins}, &st.global_bins_);
    buf->reserve({num_bins}, &st.local_bins_sum_);
    buf->reserve({num_bins}, &st.global_bins_sum_);
    buf->reserve({num_global_gpus + 1}, &st.pos_per_gpu_);
    buf->reserve({num_global_gpus + 1}, &st.neg_per_gpu_);
    buf->reserve({num_global_gpus + 1}, &st.tp_offsets_);
    buf->reserve({num_global_gpus + 1}, &st.fp_offsets_);
    buf->reserve({num_partitions}, &st.pivots_);
    buf->reserve({1}, &st.num_identical_segments_);
    buf->reserve({1}, &st.halo_tpr_);
    buf->reserve({1}, &st.halo_fpr_);

    buf->allocate();

    std::shared_ptr<GeneralBuffer2<CudaManagedAllocator>> buf_managed =
        GeneralBuffer2<CudaManagedAllocator>::create();
    buf_managed->reserve({num_partitions + 1}, &st.partition_offsets_);
    buf_managed->reserve({num_partitions + 1, num_global_gpus}, &st.all_partition_offsets_);
    buf_managed->reserve({num_partitions + 1}, &st.recv_offsets_);
    buf_managed->reserve({1}, &st.auc_);

    buf_managed->allocate();

    HCTR_LIB_THROW(cudaMemset(st.d_pos_per_gpu(), 0, (num_global_gpus + 1) * sizeof(CountType)));
    HCTR_LIB_THROW(cudaMemset(st.d_neg_per_gpu(), 0, (num_global_gpus + 1) * sizeof(CountType)));

    st.preds_1_.init_access_desc(&this->access_desc_);
    st.labels_1_.init_access_desc(&this->access_desc_);
    st.preds_2_.init_access_desc(&this->access_desc_);
    st.labels_2_.init_access_desc(&this->access_desc_);
    st.identical_pred_starts_.init_access_desc(&this->access_desc_);
    st.identical_pred_lengths_.init_access_desc(&this->access_desc_);
  }

  if (num_classes_ > 1) {
    class_preds_.resize(num_classes_);
    class_labels_.resize(num_classes_);

    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
    for (size_t class_id = 0; class_id < num_classes_; class_id++) {
      buf->reserve({num_local_samples}, &class_preds_[class_id]);
      buf->reserve({num_local_samples}, &class_labels_[class_id]);
    }
    buf->allocate();
  }

  for (size_t stream_id = 0; stream_id < num_streams; stream_id++) {
    realloc_redistributed(num_local_samples, stream, stream_id);
  }
}

void AUCStorage::realloc_redistributed(size_t num_redistributed_samples, cudaStream_t stream,
                                       size_t stream_id) {
  size_t& num_elements = num_allocated_redistributed_[stream_id];

  if (num_redistributed_samples > num_elements) {
    num_elements = reallocate_factor_ * num_redistributed_samples;
    auto& st = finalize_storage_[stream_id];

    st.preds_1_.realloc(num_elements, stream);
    st.labels_1_.realloc(num_elements, stream);
    st.preds_2_.realloc(num_elements, stream);
    st.labels_2_.realloc(num_elements, stream);

    st.identical_pred_starts_.realloc(num_elements, stream);
    st.identical_pred_lengths_.realloc(num_elements, stream);
  }
}

void AUCStorage::realloc_workspace(size_t temp_storage, size_t stream_id) {
  if (temp_storage > allocated_temp_storage_[stream_id]) {
    allocated_temp_storage_[stream_id] = reallocate_factor_ * temp_storage;
    workspace_[stream_id].realloc(allocated_temp_storage_[stream_id]);
  }
}

bool AUCStorage::realloc_local_reduce_storage(size_t input_size) {
  if (input_size > allocated_lr_input_size_) {
    allocated_lr_input_size_ = input_size;

    lr_unsorted_preds_.realloc(input_size);
    lr_sorted_preds_.realloc(input_size);
    lr_sorted_labels_.realloc(input_size);
    lr_class_ids_.realloc(input_size);
    lr_sorted_class_ids_.realloc(input_size);

    return true;
  } else {
    return false;
  }
}

/// Wrapper to call CUB functions with preallocation
template <typename CUB_Func>
void CUB_allocate_and_launch(AUCStorage& st, size_t stream_id, CUB_Func func) {
  size_t requested_size = 0;
  HCTR_LIB_THROW(func(nullptr, requested_size));
  st.realloc_workspace(requested_size, stream_id);
  HCTR_LIB_THROW(func(st.d_workspace(stream_id), st.temp_storage_bytes(stream_id)));
}

template <typename T>
AUC<T>::AUC(int batch_size_per_gpu, int n_batches, int label_dim,
            const std::shared_ptr<ResourceManager>& resource_manager)
    : Metric(),
      resource_manager_(resource_manager),
      batch_size_per_gpu_(batch_size_per_gpu),
      n_batches_(n_batches),
      num_classes_(label_dim),
      num_local_gpus_(resource_manager_->get_local_gpu_count()),
      num_global_gpus_(resource_manager_->get_global_gpu_count()),
      num_bins_(num_global_gpus_ * num_bins_per_gpu_),
      num_partitions_(num_global_gpus_),
      num_total_samples_(0),
      storage_(num_local_gpus_),
      offsets_(num_local_gpus_, 0) {
  // HCTR_LOG(INFO, ROOT,
  //          "AUC Init batch_size_per_gpu:%d n_batches:%d num_classes:%lu\nnum_local_gpus:%d "
  //          "num_global_gpus:%d num_bins:%d num_partitions:%d\n",
  //          batch_size_per_gpu_, n_batches_, num_classes_, num_local_gpus_, num_global_gpus_,
  //          num_bins_, num_partitions_);

  // If increasing this limit, adjust end_bit in local_reduce::SortPairs call
  assert(num_classes_ <= 256);

  streams_.resize(num_local_gpus_);
  const size_t num_streams = std::min(4lu, num_classes_);
  auto& all_device_list = resource_manager_->get_local_gpu_device_id_list();

  size_t max_num_local_samples = (batch_size_per_gpu_ * n_batches_);
  for (int i = 0; i < num_local_gpus_; i++) {
    int device_id = resource_manager_->get_local_gpu(i)->get_device_id();
    CudaDeviceContext context(device_id);

    streams_[i].resize(num_streams);
    for (auto& stream : streams_[i]) {
      HCTR_LIB_THROW(cudaStreamCreate(&stream));
    }
    per_class_aucs_.resize(num_classes_);

    std::vector<int> peers;
    for (int j = 0; j < (int)all_device_list.size(); j++) {
      if (i == j or resource_manager->p2p_enabled(i, j)) {
        peers.push_back(all_device_list[j]);
      }
    }

    auto& st = storage_[i];
    auto stream = resource_manager_->get_local_gpu(i)->get_stream();
    st.alloc_main(max_num_local_samples, num_bins_, num_partitions_, num_global_gpus_, num_classes_,
                  num_streams, peers, stream);
    for (size_t stream_id = 0; stream_id < num_streams; stream_id++) {
      st.realloc_workspace(num_partitions_ * sizeof(CountType), stream_id);
    }

    if (num_classes_ > 1) {
      st.realloc_local_reduce_storage(batch_size_per_gpu * num_classes_);

      auto local_gpu = resource_manager_->get_local_gpu(i);
      init_classes(st.d_lr_class_ids(), batch_size_per_gpu, num_classes_, local_gpu->get_sm_count(),
                   stream);
    }
  }

  warm_up(max_num_local_samples);
}

template <typename T>
AUC<T>::~AUC() {}

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
  auto& st = storage_[local_gpu_id];

  // Copy the labels and predictions to the internal buffer
  size_t& offset = offsets_[local_gpu_id];
  CudaDeviceContext context(device_id);
  int num_valid_samples =
      get_num_valid_samples(global_device_id, current_batch_size_, batch_size_per_gpu_);
  auto stream = resource_manager_->get_local_gpu(local_gpu_id)->get_stream();
  int num_sms = resource_manager_->get_local_gpu(local_gpu_id)->get_sm_count();

  if (num_classes_ == 1) {
    copy_all<T>(st.fst(0).d_preds() + offset, st.fst(0).d_labels() + offset, pred_tensor.get_ptr(),
                label_tensor.get_ptr(), num_valid_samples, num_sms, stream);
  } else {
    size_t input_size = num_valid_samples * num_classes_;
    if (st.realloc_local_reduce_storage(input_size)) {
      init_classes(st.d_lr_class_ids(), num_valid_samples, num_classes_, num_sms, stream);
    }

    if (std::is_same<T, float>::value) {
      CUB_allocate_and_launch(st, 0, [&](void* workspace, size_t& size) {
        return cub::DeviceRadixSort::SortPairs(
            workspace, size, st.d_lr_class_ids(), st.d_lr_sorted_class_ids(),
            (int*)pred_tensor.get_ptr(), (int*)st.d_lr_sorted_preds(), input_size, 0,
            8,  // begin_bit, end_bit
            stream);
      });
    } else {
      copy_pred<T>(st.d_lr_unsorted_preds(), pred_tensor.get_ptr(), input_size, num_sms, stream);

      CUB_allocate_and_launch(st, 0, [&](void* workspace, size_t& size) {
        return cub::DeviceRadixSort::SortPairs(
            workspace, size, st.d_lr_class_ids(), st.d_lr_sorted_class_ids(),
            (int*)st.d_lr_unsorted_preds(), (int*)st.d_lr_sorted_preds(), input_size, 0,
            8,  // begin_bit, end_bit
            stream);
      });
    }

    CUB_allocate_and_launch(st, 0, [&](void* workspace, size_t& size) {
      return cub::DeviceRadixSort::SortPairs(
          workspace, size, st.d_lr_class_ids(), st.d_lr_sorted_class_ids(),
          (int*)label_tensor.get_ptr(), (int*)st.d_lr_sorted_labels(), input_size, 0,
          8,  // begin_bit, end_bit
          stream);
    });
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));

    size_t num_streams = streams_[local_gpu_id].size();
    for (size_t class_id = 0; class_id < num_classes_; class_id++) {
      auto& copy_stream = streams_[local_gpu_id][class_id % num_streams];
      size_t class_offset = class_id * num_valid_samples;
      HCTR_LIB_THROW(cudaMemcpyAsync(
          st.d_class_preds(class_id) + offset, st.d_lr_sorted_preds() + class_offset,
          num_valid_samples * sizeof(float), cudaMemcpyDeviceToDevice, copy_stream));
      HCTR_LIB_THROW(cudaMemcpyAsync(
          st.d_class_labels(class_id) + offset, st.d_lr_sorted_labels() + class_offset,
          num_valid_samples * sizeof(float), cudaMemcpyDeviceToDevice, copy_stream));
    }
  }

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

  HCTR_LOG(INFO, ROOT, "Starting AUC NCCL warm-up\n");
#pragma omp parallel for num_threads(num_local_gpus_)
  for (int local_id = 0; local_id < num_local_gpus_; local_id++) {
    auto gpu_resource = resource_manager_->get_local_gpu(local_id).get();
    int device_id = gpu_resource->get_device_id();
    auto& stream = gpu_resource->get_stream();

    CudaDeviceContext context(device_id);
    auto& st = storage_[local_id];

    mock_kernel<<<grid, block, 0, stream>>>(st.fst(0).d_preds(), num_local_samples);
    initialize_array<<<grid, block, 0, stream>>>(st.fst(0).d_labels(), num_local_samples, 0.0f);

    if (num_classes_ > 1) {
      for (size_t class_id = 0; class_id < num_classes_; class_id++) {
        mock_kernel<<<grid, block, 0, stream>>>(st.d_class_preds(class_id), num_local_samples);
        initialize_array<<<grid, block, 0, stream>>>(st.d_class_labels(class_id), num_local_samples,
                                                     0.0f);
      }
    }
    offsets_[local_id] = num_local_samples;
  }
  num_total_samples_ = num_local_samples * num_global_gpus_;

  [[maybe_unused]] float dummy = finalize_metric();
  HCTR_LOG(INFO, ROOT, "Warm-up done\n");
}

template <typename T>
float AUC<T>::finalize_metric() {
  float result[num_local_gpus_];
#pragma omp parallel num_threads(num_local_gpus_)
  { result[omp_get_thread_num()] = finalize_metric_per_gpu(omp_get_thread_num()); }

  num_total_samples_ = 0;
  // All threads should have the same result here
  return result[0];
}

template <typename T>
float AUC<T>::finalize_metric_per_gpu(int local_id) {
  auto& st = storage_[local_id];
  size_t num_local_samples = offsets_[local_id];
  offsets_[local_id] = 0;

  auto gpu_resource = resource_manager_->get_local_gpu(local_id).get();
  int device_id = gpu_resource->get_device_id();
  CudaDeviceContext context(device_id);

  float result = 0.0;
  if (num_classes_ == 1) {
    result = finalize_class_metric(st.fst(0).d_preds(), st.fst(0).d_labels(), local_id,
                                   num_local_samples);
  } else {
    if (streams_[local_id].size() == 1) {
      for (size_t class_id = 0; class_id < num_classes_; class_id++) {
        float class_auc = finalize_class_metric(
            st.d_class_preds(class_id), st.d_class_labels(class_id), local_id, num_local_samples);
        per_class_aucs_[class_id] = class_auc;
        result += class_auc;
      }
      result /= num_classes_;
    } else {
      result = finalize_class_metric_multi_stream(local_id, num_local_samples);
    }
  }

  return result;
}

template <typename T>
float AUC<T>::finalize_class_metric(float* d_preds, float* d_labels, int local_id,
                                    size_t num_local_samples) {
  run_finalize_step(d_preds, d_labels, local_id, num_local_samples, 0, num_finalize_steps_);
  return *(storage_[local_id].fst(0).d_auc());
}

template <typename T>
float AUC<T>::finalize_class_metric_multi_stream(int local_id, int num_local_samples) {
  auto& st = storage_[local_id];
  size_t num_streams = streams_[local_id].size();

  float result = 0.0;
  size_t class_begin = 0;
  while (class_begin < num_classes_) {
    for (size_t step_id = 0; step_id < num_finalize_steps_; step_id++) {
      // Launch all streams
      for (size_t stream_id = 0; stream_id < num_streams and class_begin + stream_id < num_classes_;
           stream_id++) {
        size_t class_id = class_begin + stream_id;
        run_finalize_step(st.d_class_preds(class_id), st.d_class_labels(class_id), local_id,
                          num_local_samples, stream_id, step_id);
      }

      // Sync all streams
      for (size_t stream_id = 0; stream_id < num_streams and class_begin + stream_id < num_classes_;
           stream_id++) {
        HCTR_LIB_THROW(cudaStreamSynchronize(streams_[local_id][stream_id]));
        if (step_id == num_finalize_steps_ - 1) {
          float class_auc = *(st.fst(stream_id).d_auc());
          per_class_aucs_[class_begin + stream_id] = class_auc;
          result += class_auc;
        }
      }
    }

    class_begin += num_streams;
  }

  return result / num_classes_;
}

template <typename T>
void AUC<T>::run_finalize_step(float* d_preds, float* d_labels, int local_id,
                               size_t num_local_samples, size_t stream_id, size_t step_id) {
  dim3 grid(160, 1, 1);
  dim3 block(1024, 1, 1);

  auto gpu_resource = resource_manager_->get_local_gpu(local_id).get();
  int global_id = gpu_resource->get_global_id();

  auto& stream = streams_[local_id].size() == 1
                     ? resource_manager_->get_local_gpu(local_id)->get_stream()
                     : streams_[local_id][stream_id];
  auto& st = storage_[local_id];
  auto& fst = st.fst(stream_id);

  HCTR_CHECK(step_id <= num_finalize_steps_);
  if (step_id == 0 or step_id == num_finalize_steps_) {
    // 1. Create local histograms of predictions
    float eps = 1e-7f;
    float loc_min = pred_min_ + eps;
    float loc_max = pred_max_ - eps;
    auto clamp = [loc_min, loc_max] __device__(const float v) {
      return fmaxf(fminf(v, loc_max), loc_min);
    };
    cub::TransformInputIterator<float, decltype(clamp), float*> d_clamped_preds(d_preds, clamp);

    // (int) casting is a CUB workaround. Fixed in https://github.com/thrust/cub/pull/38
    CUB_allocate_and_launch(st, stream_id, [&](void* workspace, size_t& size) {
      return cub::DeviceHistogram::HistogramEven(
          workspace, size, d_clamped_preds, st.fst(stream_id).d_local_bins(), num_bins_ + 1,
          pred_min_, pred_max_, (int)num_local_samples, stream);
    });

    // 2. Allreduce histograms
    metric_comm::allreduce(fst.d_local_bins(), fst.d_global_bins(), num_bins_, gpu_resource,
                           stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));

    // 3. Find num_global_gpus_-1 pivot points
    CUB_allocate_and_launch(st, stream_id, [&](void* workspace, size_t& size) {
      return cub::DeviceScan::InclusiveSum(workspace, size, st.fst(stream_id).d_global_bins(),
                                           st.fst(stream_id).d_global_bins_sum(), num_bins_,
                                           stream);
    });

    find_pivots_kernel<<<grid, block, 0, stream>>>(
        fst.d_global_bins_sum(), num_bins_, num_total_samples_ / num_global_gpus_, fst.d_pivots());

    // 4. Partition (partially sort) local predictions into num_global_gpus_ bins
    //    separated by pivot points.
    CUB_allocate_and_launch(st, stream_id, [&](void* workspace, size_t& size) {
      return cub::DeviceScan::InclusiveSum(workspace, size, fst.d_local_bins(),
                                           fst.d_local_bins_sum(), num_bins_, stream);
    });

    find_partition_offsets_kernel<<<grid, block, 0, stream>>>(
        fst.d_local_bins_sum(), fst.d_pivots(), num_partitions_, num_local_samples,
        fst.d_partition_offsets());

    initialize_array<<<grid, block, 0, stream>>>((CountType*)st.d_workspace(stream_id),
                                                 num_partitions_, (CountType)0);

    size_t shmem_size = sizeof(CountType) * num_partitions_ * 2;
    create_partitions_kernel<<<grid, block, shmem_size, stream>>>(
        d_labels, d_preds, fst.d_pivots(), pred_min_, pred_max_, num_local_samples, num_bins_,
        num_partitions_, fst.d_partition_offsets(), (CountType*)st.d_workspace(stream_id),
        fst.d_partitioned_labels(), fst.d_partitioned_preds());

    // 5. Exchange the data such that all predicitons on GPU i are smaller than
    //    the ones on GPU i+1.
    // 5.1. Compute receiving side offsets. Also compute resulting number
    //      of elements on all the other GPUs, required to determine correct neighbors
    metric_comm::allgather(fst.d_partition_offsets(), fst.d_all_partition_offsets(),
                           num_partitions_ + 1, gpu_resource, stream);
  }

  if (step_id == 1 or step_id == num_finalize_steps_) {
    if (step_id == num_finalize_steps_) {
      // The following is done on the CPU, need to wait
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    }

    fst.all_num_redistributed_samples.resize(num_global_gpus_, 0);
    for (int dest = 0; dest < num_global_gpus_; dest++) {
      CountType sum = 0;
      for (int src = 0; src < num_global_gpus_; src++) {
        CountType size_src = fst.d_all_partition_offsets()[src * (num_partitions_ + 1) + dest + 1] -
                             fst.d_all_partition_offsets()[src * (num_partitions_ + 1) + dest];
        if (dest == global_id) {
          fst.d_recv_offsets()[src] = sum;
        }
        sum += size_src;
      }
      fst.all_num_redistributed_samples[dest] = sum;
    }
    fst.d_recv_offsets()[num_global_gpus_] = fst.all_num_redistributed_samples[global_id];
    fst.num_redistributed_samples = fst.all_num_redistributed_samples[global_id];

    // 5.2 Allocate more memory if needed
    st.realloc_redistributed(fst.num_redistributed_samples, stream, stream_id);
  }

  if (step_id == 2 or step_id == num_finalize_steps_) {
// 5.3 Synchronize threads before all to all to prevent hangs
#pragma omp barrier

    // 5.4 All to all
    metric_comm::all_to_all(fst.d_partitioned_labels(), fst.d_presorted_labels(),
                            fst.d_partition_offsets(), fst.d_recv_offsets(), num_global_gpus_,
                            gpu_resource, stream);
    metric_comm::all_to_all(fst.d_partitioned_preds(), fst.d_presorted_preds(),
                            fst.d_partition_offsets(), fst.d_recv_offsets(), num_global_gpus_,
                            gpu_resource, stream);

    if (fst.num_redistributed_samples > 0) {
      // 6. Locally sort (label, pred) by pred
      CUB_allocate_and_launch(st, stream_id, [&](void* workspace, size_t& size) {
        auto& fst = st.fst(stream_id);
        return cub::DeviceRadixSort::SortPairs(
            workspace, size, fst.d_presorted_preds(), fst.d_sorted_preds(),
            fst.d_presorted_labels(), fst.d_sorted_labels(), fst.num_redistributed_samples, 0,
            sizeof(float) * 8,  // begin_bit, end_bit
            stream);
      });

      // 7. Create TPR and FPR. Need a "global" scan
      // 7.1 Local inclusive scan to find TP and FP
      auto one_minus_val = [] __device__(const float v) { return 1.0f - v; };
      cub::TransformInputIterator<float, decltype(one_minus_val), float*> d_one_minus_labels(
          fst.d_sorted_labels(), one_minus_val);

      CUB_allocate_and_launch(st, stream_id, [&](void* workspace, size_t& size) {
        auto& fst = st.fst(stream_id);
        return cub::DeviceScan::InclusiveSum(workspace, size, fst.d_sorted_labels(), fst.d_tp(),
                                             fst.num_redistributed_samples, stream);
      });
      CUB_allocate_and_launch(st, stream_id, [&](void* workspace, size_t& size) {
        auto& fst = st.fst(stream_id);
        return cub::DeviceScan::InclusiveSum(workspace, size, d_one_minus_labels, fst.d_fp(),
                                             fst.num_redistributed_samples, stream);
      });

      // 7.2 'Flatten' tp and fp for cases where several consecutive predictions are identical
      CUB_allocate_and_launch(st, stream_id, [&](void* workspace, size_t& size) {
        auto& fst = st.fst(stream_id);
        return cub::DeviceRunLengthEncode::NonTrivialRuns(
            workspace, size, fst.d_sorted_preds(), fst.d_identical_pred_starts(),
            fst.d_identical_pred_lengths(), fst.d_num_identical_segments(),
            fst.num_redistributed_samples, stream);
      });

      flatten_segments_kernel<<<grid, block, 0, stream>>>(
          fst.d_identical_pred_starts(), fst.d_identical_pred_lengths(),
          fst.d_num_identical_segments(), fst.d_tp(), fst.d_fp());

      // 7.3 Allgather of the number of total positive and negative samples
      //     on each GPU and exclusive scan them
      metric_comm::allgather(fst.d_tp() + fst.num_redistributed_samples - 1, fst.d_pos_per_gpu(), 1,
                             gpu_resource, stream);
      metric_comm::allgather(fst.d_fp() + fst.num_redistributed_samples - 1, fst.d_neg_per_gpu(), 1,
                             gpu_resource, stream);

      CUB_allocate_and_launch(st, stream_id, [&](void* workspace, size_t& size) {
        auto& fst = st.fst(stream_id);
        return cub::DeviceScan::ExclusiveSum(workspace, size, fst.d_pos_per_gpu(),
                                             fst.d_tp_offsets(), num_global_gpus_ + 1, stream);
      });

      CUB_allocate_and_launch(st, stream_id, [&](void* workspace, size_t& size) {
        auto& fst = st.fst(stream_id);
        return cub::DeviceScan::ExclusiveSum(workspace, size, fst.d_neg_per_gpu(),
                                             fst.d_fp_offsets(), num_global_gpus_ + 1, stream);
      });

      // 7.4 Locally find TPR and FPR given TP, FP and global offsets
      rate_from_part_cumsum_kernel<<<grid, block, 0, stream>>>(
          fst.d_tp(), fst.num_redistributed_samples, fst.d_tp_offsets() + global_id,
          fst.d_tp_offsets() + num_global_gpus_, fst.d_tpr());
      rate_from_part_cumsum_kernel<<<grid, block, 0, stream>>>(
          fst.d_fp(), fst.num_redistributed_samples, fst.d_fp_offsets() + global_id,
          fst.d_fp_offsets() + num_global_gpus_, fst.d_fpr());

      // 8. Integrate TPR and FPR taking into account halo samples
      // 8.1 No need to communicate with GPUs that have 0 elements
      int left_neighbor = -1, right_neighbor = -1;
      for (int gpuid = global_id - 1; gpuid >= 0; gpuid--) {
        if (fst.all_num_redistributed_samples[gpuid] > 0) {
          left_neighbor = gpuid;
          break;
        }
      }
      for (int gpuid = global_id + 1; gpuid < num_global_gpus_; gpuid++) {
        if (fst.all_num_redistributed_samples[gpuid] > 0) {
          right_neighbor = gpuid;
          break;
        }
      }

      // 8.2 Send the halos
      metric_comm::send_halo_right(fst.d_tpr() + fst.num_redistributed_samples - 1,
                                   fst.d_halo_tpr(), 1, left_neighbor, right_neighbor, gpu_resource,
                                   stream);
      metric_comm::send_halo_right(fst.d_fpr() + fst.num_redistributed_samples - 1,
                                   fst.d_halo_fpr(), 1, left_neighbor, right_neighbor, gpu_resource,
                                   stream);

      // 8.3 First non-zero GPU initializes the halo to 0
      if (left_neighbor == -1) {
        initialize_array<<<grid, block, 0, stream>>>(fst.d_halo_tpr(), 1, 1.0f);
        initialize_array<<<grid, block, 0, stream>>>(fst.d_halo_fpr(), 1, 1.0f);
      }

      // 8.4 Integrate
      initialize_array<<<grid, block, 0, stream>>>(fst.d_auc(), 1, 0.0f);
      trapz_kernel<<<grid, block, 0, stream>>>(fst.d_tpr(), fst.d_fpr(), fst.d_halo_tpr(),
                                               fst.d_halo_fpr(), fst.d_auc(),
                                               fst.num_redistributed_samples);
    } else {
      // Here we're on a GPU with no elements, need to communicate zeros where needed
      // Performance is not a concern on such GPUs
      HCTR_LOG_S(INFO, ROOT)
          << "GPU " << global_id
          << " has no samples in the AUC computation due to strongly uneven distribution of the "
             "scores. This may indicate a problem in the training or an extremely accurate model."
          << std::endl;

      initialize_array<<<grid, block, 0, stream>>>(fst.d_halo_tpr(), 1, 0.0f);
      // 7.3 All GPUs need to call allgather
      metric_comm::allgather(fst.d_halo_tpr(), fst.d_pos_per_gpu(), 1, gpu_resource, stream);
      metric_comm::allgather(fst.d_halo_tpr(), fst.d_neg_per_gpu(), 1, gpu_resource, stream);

      // 8.4 Initialize partial auc to 0
      initialize_array<<<grid, block, 0, stream>>>(fst.d_auc(), 1, 0.0f);
    }

    // 9. Finally allreduce auc
    metric_comm::allreduce(fst.d_auc(), fst.d_auc(), 1, gpu_resource, stream);

    if (step_id == num_finalize_steps_) {
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    }
  }
}

template <typename CUB_Func>
void CUB_allocate_and_launch(NDCGStorage& st, CUB_Func func) {
  size_t requested_size = 0;
  HCTR_LIB_THROW(func(nullptr, requested_size));
  st.realloc_workspace(requested_size);
  HCTR_LIB_THROW(func(st.d_workspace(), st.temp_storage_bytes()));
}

void NDCGStorage::alloc_main(size_t num_local_samples, size_t num_bins, size_t num_partitions,
                             size_t num_global_gpus, const std::vector<int>& peers) {
  num_allocated_redistributed_ = 0;
  allocated_temp_storage_ = 0;

  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();

  buf->reserve({num_bins}, &local_bins_);
  buf->reserve({num_bins}, &global_bins_);
  buf->reserve({num_bins}, &local_bins_sum_);
  buf->reserve({num_bins}, &global_bins_sum_);
  buf->reserve({num_partitions}, &pivots_);
  buf->reserve({1}, &label_count_);

  buf->allocate();

  std::shared_ptr<GeneralBuffer2<CudaManagedAllocator>> buf_managed =
      GeneralBuffer2<CudaManagedAllocator>::create();
  buf_managed->reserve({num_partitions + 1}, &partition_offsets_);
  buf_managed->reserve({num_partitions + 1, num_global_gpus}, &all_partition_offsets_);
  buf_managed->reserve({num_partitions + 1}, &recv_offsets_);
  buf_managed->reserve({1}, &dcg_);
  buf_managed->reserve({1}, &ideal_dcg_);

  buf_managed->allocate();

  gen_access_desc(access_desc_, peers);
  preds_1_.init_access_desc(&access_desc_);
  labels_1_.init_access_desc(&access_desc_);
  preds_2_.init_access_desc(&access_desc_);
  labels_2_.init_access_desc(&access_desc_);
  scaled_labels_.init_access_desc(&access_desc_);

  realloc_redistributed(num_local_samples, 0);
}

void NDCGStorage::realloc_redistributed(size_t num_redistributed_samples, cudaStream_t stream) {
  size_t& num_elements = num_allocated_redistributed_;
  if (num_redistributed_samples > num_elements) {
    num_elements = reallocate_factor_ * num_redistributed_samples;
    preds_1_.realloc(num_elements, stream);
    labels_1_.realloc(num_elements, stream);
    preds_2_.realloc(num_elements, stream);
    labels_2_.realloc(num_elements, stream);
    scaled_labels_.realloc(num_elements, stream);
  }
}

void NDCGStorage::realloc_workspace(size_t temp_storage) {
  if (temp_storage > allocated_temp_storage_) {
    allocated_temp_storage_ = reallocate_factor_ * temp_storage;
    workspace_.realloc(allocated_temp_storage_);
  }
}

__global__ void scale_labels_kernel(float* labels, float* scaled_labels, size_t offset,
                                    size_t num_samples) {
  size_t base = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = base; i < num_samples; i += blockDim.x * gridDim.x) {
    scaled_labels[i] = labels[i] / log2(2 + offset + num_samples - 1 - i);
  }
}

void scale_labels(float* labels, float* scaled_labels, size_t offset, size_t num_samples,
                  int num_sms, cudaStream_t stream) {
  dim3 grid(num_sms * 2, 1, 1);
  dim3 block(1024, 1, 1);
  scale_labels_kernel<<<grid, block, 0, stream>>>(labels, scaled_labels, offset, num_samples);
}

template <typename T>
NDCG<T>::NDCG(int batch_size_per_gpu, int n_batches,
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
  auto& all_device_list = resource_manager_->get_local_gpu_device_id_list();

  for (int i = 0; i < num_local_gpus_; i++) {
    int device_id = resource_manager_->get_local_gpu(i)->get_device_id();
    CudaDeviceContext context(device_id);

    std::vector<int> peers;
    for (int j = 0; j < (int)all_device_list.size(); j++) {
      if (i == j or resource_manager->p2p_enabled(i, j)) {
        peers.push_back(all_device_list[j]);
      }
    }

    auto& st = storage_[i];
    st.alloc_main(max_num_local_samples, num_bins_, num_partitions_, num_global_gpus_, peers);
  }

  warm_up(max_num_local_samples);
}

template <typename T>
NDCG<T>::~NDCG() {}

template <typename T>
void NDCG<T>::warm_up(size_t num_local_samples) {
  dim3 grid(160, 1, 1);
  dim3 block(1024, 1, 1);

  HCTR_LOG(INFO, ROOT, "Starting NDCG NCCL warm-up\n");
#pragma omp parallel for num_threads(num_local_gpus_)
  for (int local_id = 0; local_id < num_local_gpus_; local_id++) {
    auto gpu_resource = resource_manager_->get_local_gpu(local_id).get();
    int device_id = gpu_resource->get_device_id();
    CudaDeviceContext context(device_id);

    auto& stream = gpu_resource->get_stream();
    auto& st = storage_[local_id];

    mock_kernel<<<grid, block, 0, stream>>>(st.d_preds(), num_local_samples);
    initialize_array<<<grid, block, 0, stream>>>(st.d_labels(), num_local_samples, 0.0f);
    offsets_[local_id] = num_local_samples;
  }
  num_total_samples_ = num_local_samples * num_global_gpus_;

  [[maybe_unused]] float dummy = finalize_metric();
  HCTR_LOG(INFO, ROOT, "Warm-up done\n");
}

template <typename T>
void NDCG<T>::local_reduce(int local_gpu_id, RawMetricMap raw_metrics) {
  Tensor2<PredType> pred_tensor = Tensor2<PredType>::stretch_from(raw_metrics[RawType::Pred]);
  Tensor2<LabelType> label_tensor = Tensor2<LabelType>::stretch_from(raw_metrics[RawType::Label]);
  int device_id = resource_manager_->get_local_gpu(local_gpu_id)->get_device_id();
  int global_device_id = resource_manager_->get_local_gpu(local_gpu_id)->get_global_id();
  auto& st = storage_[local_gpu_id];

  size_t& offset = offsets_[local_gpu_id];
  CudaDeviceContext context(device_id);
  int num_valid_samples =
      get_num_valid_samples(global_device_id, current_batch_size_, batch_size_per_gpu_);
  auto stream = resource_manager_->get_local_gpu(local_gpu_id)->get_stream();
  int num_sms = resource_manager_->get_local_gpu(local_gpu_id)->get_sm_count();

  // Copy the labels and predictions to the internal buffer
  copy_all<T>(st.d_preds() + offset, st.d_labels() + offset, pred_tensor.get_ptr(),
              label_tensor.get_ptr(), num_valid_samples, num_sms, stream);

  offset += num_valid_samples;

  if (local_gpu_id == 0) {
    num_total_samples_ += current_batch_size_;
  }
}

template <typename T>
void NDCG<T>::global_reduce(int n_nets) {
  // No need to do anything here
}

template <typename T>
float NDCG<T>::finalize_metric() {
  float result[num_local_gpus_];
#pragma omp parallel num_threads(num_local_gpus_)
  { result[omp_get_thread_num()] = finalize_metric_per_gpu(omp_get_thread_num()); }

  num_total_samples_ = 0;
  // All threads should have the same result here
  return result[0];
}

template <typename T>
float NDCG<T>::finalize_metric_per_gpu(int local_id) {
  dim3 grid(160, 1, 1);
  dim3 block(1024, 1, 1);

  size_t num_local_samples = offsets_[local_id];
  offsets_[local_id] = 0;

  auto gpu_resource = resource_manager_->get_local_gpu(local_id).get();
  int device_id = gpu_resource->get_device_id();
  CudaDeviceContext context(device_id);

  int global_id = gpu_resource->get_global_id();
  auto stream = gpu_resource->get_stream();
  int num_sms = gpu_resource->get_sm_count();
  auto& st = storage_[local_id];

  ////// DCG Computation //////
  // 1. Create local histograms of predictions
  float eps = 1e-7f;
  float loc_min = pred_min_ + eps;
  float loc_max = pred_max_ - eps;
  auto clamp = [loc_min, loc_max] __device__(const float v) {
    return fmaxf(fminf(v, loc_max), loc_min);
  };

  cub::TransformInputIterator<float, decltype(clamp), float*> d_clamped_preds(st.d_preds(), clamp);

  CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
    return cub::DeviceHistogram::HistogramEven(workspace, size, d_clamped_preds, st.d_local_bins(),
                                               num_bins_ + 1, pred_min_, pred_max_,
                                               (int)num_local_samples, stream);
  });

  // 2. Allreduce histograms
  metric_comm::allreduce(st.d_local_bins(), st.d_global_bins(), num_bins_, gpu_resource, stream);

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
                         num_partitions_ + 1, gpu_resource, stream);

  // The following is done on the CPU, need to wait
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  st.all_num_redistributed_samples.resize(num_global_gpus_, 0);
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
    st.all_num_redistributed_samples[dest] = sum;
  }
  st.d_recv_offsets()[num_global_gpus_] = st.all_num_redistributed_samples[global_id];
  st.num_redistributed_samples = st.all_num_redistributed_samples[global_id];

  // 5.2 Allocate more memory if needed
  st.realloc_redistributed(st.num_redistributed_samples, stream);

// 5.3 Synchronize threads before all to all to prevent hangs
#pragma omp barrier

  // 5.4 All to all
  metric_comm::all_to_all(st.d_partitioned_labels(), st.d_presorted_labels(),
                          st.d_partition_offsets(), st.d_recv_offsets(), num_global_gpus_,
                          gpu_resource, stream);
  metric_comm::all_to_all(st.d_partitioned_preds(), st.d_presorted_preds(),
                          st.d_partition_offsets(), st.d_recv_offsets(), num_global_gpus_,
                          gpu_resource, stream);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  if (st.num_redistributed_samples > 0) {
    // 6. Locally sort (label, pred) by pred
    CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
      return cub::DeviceRadixSort::SortPairs(
          workspace, size, st.d_presorted_preds(), st.d_sorted_preds(), st.d_presorted_labels(),
          st.d_sorted_labels(), st.num_redistributed_samples, 0, sizeof(float) * 8, stream);
    });

    size_t offset = std::accumulate(st.all_num_redistributed_samples.begin() + global_id + 1,
                                    st.all_num_redistributed_samples.end(), 0u);

    scale_labels(st.d_sorted_labels(), st.d_scaled_labels(), offset, st.num_redistributed_samples,
                 num_sms, stream);

    CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
      return cub::DeviceReduce::Sum(workspace, size, st.d_scaled_labels(), st.d_dcg(),
                                    st.num_redistributed_samples, stream);
    });
  } else {
    // 6 Initialize partial dcg to 0
    initialize_array<<<grid, block, 0, stream>>>(st.d_dcg(), 1, 0.0f);
  }

  // 7. Finally allreduce dcg
  metric_comm::allreduce(st.d_dcg(), st.d_dcg(), 1, gpu_resource, stream);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  ////// Ideal DCG Computation //////
  // Labels are binary values (0 or 1) and not floating point values
  // This leads to a simpler implementation

  // Count number of ones by summing
  CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
    return cub::DeviceReduce::Sum(workspace, size, st.d_labels(), st.d_label_count(),
                                  num_local_samples, stream);
  });

  // Sort locally
  CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
    return cub::DeviceRadixSort::SortKeys(workspace, size, st.d_labels(), st.d_sorted_labels(),
                                          num_local_samples, 0, sizeof(float) * 8, stream);
  });

  // Scale down accounting for samples on other GPUs
  metric_comm::allgather(st.d_label_count(), st.d_partition_offsets(), 1, gpu_resource, stream);
  st.all_num_redistributed_samples.resize(num_global_gpus_, 0);

  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  for (int i = 0; i < num_global_gpus_; i++) {
    st.all_num_redistributed_samples[i] = st.d_partition_offsets()[i];
  }

  size_t offset = std::accumulate(st.all_num_redistributed_samples.begin() + global_id + 1,
                                  st.all_num_redistributed_samples.end(), 0u);
  scale_labels(st.d_sorted_labels(), st.d_scaled_labels(), offset, num_local_samples, num_sms,
               stream);

  // Sum scaled down values to get ideal dcg
  CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
    return cub::DeviceReduce::Sum(workspace, size, st.d_scaled_labels(), st.d_ideal_dcg(),
                                  num_local_samples, stream);
  });
  metric_comm::allreduce(st.d_ideal_dcg(), st.d_ideal_dcg(), 1, gpu_resource, stream);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  return *st.d_dcg() / *st.d_ideal_dcg();
}

// Reference implementation for a single GPU
// Avoids complexity of the multi GPU implementation
template <typename T>
float NDCG<T>::finalize_metric_single_gpu(int local_id) {
  // Pseudo code
  // 1. Descending sort (pred, label) by pred
  // 2. DCG = Compute sum(label_i / log2(i + 2))
  // 3. Descending sort label
  // 4. Ideal DCG = Compute sum(label_i / log2(i + 2))
  // 5. NDCG = DCG / Ideal DCG

  size_t num_local_samples = offsets_[local_id];
  offsets_[local_id] = 0;

  auto stream = resource_manager_->get_local_gpu(local_id)->get_stream();
  int num_sms = resource_manager_->get_local_gpu(local_id)->get_sm_count();
  auto& st = storage_[local_id];

  // Compute DCG
  CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
    return cub::DeviceRadixSort::SortPairs(workspace, size, st.d_preds(), st.d_sorted_preds(),
                                           st.d_labels(), st.d_sorted_labels(), num_local_samples,
                                           0, sizeof(LabelType) * 8, stream);
  });

  scale_labels(st.d_sorted_labels(), st.d_scaled_labels(), 0, num_local_samples, num_sms, stream);

  CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
    return cub::DeviceReduce::Sum(workspace, size, st.d_scaled_labels(), st.d_dcg(),
                                  num_local_samples, stream);
  });

  // Compute Ideal DCG
  CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
    return cub::DeviceRadixSort::SortKeys(workspace, size, st.d_sorted_labels(), st.d_labels(),
                                          num_local_samples, 0, sizeof(LabelType) * 8, stream);
  });

  scale_labels(st.d_labels(), st.d_scaled_labels(), 0, num_local_samples, num_sms, stream);

  CUB_allocate_and_launch(st, [&](void* workspace, size_t& size) {
    return cub::DeviceReduce::Sum(workspace, size, st.d_scaled_labels(), st.d_ideal_dcg(),
                                  num_local_samples, stream);
  });

  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  HCTR_LOG(INFO, ROOT, "DCG %f Ideal DCG %f\n", *st.d_dcg(), *st.d_ideal_dcg());

  return *st.d_dcg() / *st.d_ideal_dcg();
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

    HCTR_LIB_THROW(cudaMalloc((void**)(&(checked_count_[i])), sizeof(int)));
    HCTR_LIB_THROW(cudaMalloc((void**)(&(hit_count_[i])), sizeof(int)));
  }
}

template <typename T>
void HitRate<T>::free_all() {
  for (int i = 0; i < num_local_gpus_; i++) {
    int device_id = resource_manager_->get_local_gpu(i)->get_device_id();
    CudaDeviceContext context(device_id);
    HCTR_LIB_THROW(cudaFree(checked_count_[i]));
    HCTR_LIB_THROW(cudaFree(hit_count_[i]));
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

  HCTR_LIB_THROW(
      cudaMemsetAsync(checked_count_[local_gpu_id], 0, sizeof(int), local_gpu->get_stream()));
  HCTR_LIB_THROW(
      cudaMemsetAsync(hit_count_[local_gpu_id], 0, sizeof(int), local_gpu->get_stream()));

  collect_hits<T><<<grid, block, 0, local_gpu->get_stream()>>>(
      pred_tensor.get_ptr(), label_tensor.get_ptr(), num_valid_samples,
      checked_count_[local_gpu_id], hit_count_[local_gpu_id]);
  int checked_host = 0;
  int hits_host = 0;
  HCTR_LIB_THROW(cudaMemcpyAsync(&checked_host, checked_count_[local_gpu_id], sizeof(int),
                                 cudaMemcpyDeviceToHost, local_gpu->get_stream()));
  checked_local_[local_gpu_id] = checked_host;
  HCTR_LIB_THROW(cudaMemcpyAsync(&hits_host, hit_count_[local_gpu_id], sizeof(int),
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
    HCTR_MPI_THROW(MPI_Reduce(&hits_inter, &hits_reduced, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
    HCTR_MPI_THROW(
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
  HCTR_MPI_THROW(MPI_Barrier(MPI_COMM_WORLD));
  HCTR_MPI_THROW(MPI_Bcast(&ret, 1, MPI_FLOAT, 0, MPI_COMM_WORLD));
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

// sMAPE Metric function implementations
template <typename T>
SMAPE<T>::SMAPE(int batch_size_per_gpu, const std::shared_ptr<ResourceManager>& resource_manager)
    : Metric(),
      batch_size_per_gpu_(batch_size_per_gpu),
      resource_manager_(resource_manager),
      num_local_gpus_(resource_manager_->get_local_gpu_count()),
      error_(resource_manager_->get_local_gpu_count()),
      checked_local_(std::vector<int>(resource_manager->get_local_gpu_count(), 0)),
      error_local_(std::vector<float>(resource_manager->get_local_gpu_count(), 0)),
      checked_global_(0),
      error_global_(0),
      n_batches_(0) {
  for (int i = 0; i < num_local_gpus_; i++) {
    int device_id = resource_manager_->get_local_gpu(i)->get_device_id();
    CudaDeviceContext context(device_id);
    HCTR_LIB_THROW(cudaMalloc((void**)(&(error_[i])), sizeof(float)));
  }
}

template <typename T>
void SMAPE<T>::free_all() {
  for (int i = 0; i < num_local_gpus_; i++) {
    int device_id = resource_manager_->get_local_gpu(i)->get_device_id();
    CudaDeviceContext context(device_id);
    HCTR_LIB_THROW(cudaFree(error_[i]));
  }
}

template <typename T>
SMAPE<T>::~SMAPE() {
  free_all();
}

template <typename T>
__global__ void collect_error(T* preds, T* labels, int num_samples, float* error) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_samples;
       i += blockDim.x * gridDim.x) {
    T avg = (preds[i] + labels[i]) / 2;
    T err = abs(preds[i] - labels[i]);
    atomicAdd(error, 1. * err / avg);
  }
}

template <typename T>
void SMAPE<T>::local_reduce(int local_gpu_id, RawMetricMap raw_metrics) {
  const auto& local_gpu = resource_manager_->get_local_gpu(local_gpu_id);
  CudaDeviceContext context(local_gpu->get_device_id());

  int global_device_id = resource_manager_->get_local_gpu(local_gpu_id)->get_global_id();
  int num_valid_samples =
      get_num_valid_samples(global_device_id, current_batch_size_, batch_size_per_gpu_);

  Tensor2<T> pred_tensor = Tensor2<T>::stretch_from(raw_metrics[RawType::Pred]);
  Tensor2<T> label_tensor = Tensor2<T>::stretch_from(raw_metrics[RawType::Label]);

  dim3 grid(160, 1, 1);
  dim3 block(1024, 1, 1);

  cudaMemsetAsync(error_[local_gpu_id], 0, sizeof(float), local_gpu->get_stream());
  collect_error<T><<<grid, block, 0, local_gpu->get_stream()>>>(
      pred_tensor.get_ptr(), label_tensor.get_ptr(), num_valid_samples, error_[local_gpu_id]);

  checked_local_[local_gpu_id] = num_valid_samples;
  HCTR_LIB_THROW(cudaMemcpyAsync(&error_local_[local_gpu_id], error_[local_gpu_id], sizeof(float),
                                 cudaMemcpyDeviceToHost, local_gpu->get_stream()));
}

template <typename T>
void SMAPE<T>::global_reduce(int n_nets) {
  int checked_inter = 0;
  float error_inter = 0;

  for (auto& error_local : error_local_) {
    error_inter += error_local;
  }
  for (auto& checked_local : checked_local_) {
    checked_inter += checked_local;
  }

#ifdef ENABLE_MPI
  if (resource_manager_->get_num_process() > 1) {
    float error_reduced = 0;
    int checked_reduced = 0;
    HCTR_MPI_THROW(
        MPI_Reduce(&error_inter, &error_reduced, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));
    HCTR_MPI_THROW(
        MPI_Reduce(&checked_inter, &checked_reduced, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
    error_inter = error_reduced;
    checked_inter = checked_reduced;
  }
#endif

  error_global_ += error_inter;
  checked_global_ += checked_inter;

  n_batches_++;
}

template <typename T>
float SMAPE<T>::finalize_metric() {
  float ret = 0.0f;
  if (resource_manager_->is_master_process()) {
    if (n_batches_) {
      ret = ((float)error_global_) / (float)(checked_global_);
    }
  }
#ifdef ENABLE_MPI
  HCTR_MPI_THROW(MPI_Barrier(MPI_COMM_WORLD));
  HCTR_MPI_THROW(MPI_Bcast(&ret, 1, MPI_FLOAT, 0, MPI_COMM_WORLD));
#endif
  error_global_ = 0;
  checked_global_ = 0;
  for (auto& error_local : error_local_) {
    error_local = 0;
  }
  for (auto& checked_local : checked_local_) {
    checked_local = 0;
  }
  n_batches_ = 0;
  return ret;
}

template <typename T, ReallocType_t U>
ReallocBuffer<T, U>::ReallocBuffer() : num_elements_(0), ptr_(nullptr) {
  CUdevice device;
  HCTR_LIB_THROW(cudaGetDevice(&device));

  prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop_.location.id = device;
  HCTR_LIB_THROW(
      cuMemGetAllocationGranularity(&chunk_size_, &prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  access_desc_ = nullptr;
}

template <typename T, ReallocType_t U>
void ReallocBuffer<T, U>::init_access_desc(const std::vector<CUmemAccessDesc>* access_desc) {
  access_desc_ = access_desc;
}

template <typename T, ReallocType_t U>
ReallocBuffer<T, U>::~ReallocBuffer() {
  if (num_elements_ > 0 and U != ReallocType_t::MMAP) {
    HCTR_LIB_THROW(cudaFree(ptr_));
  }
  if (U == ReallocType_t::MMAP) {
    release_mmap_memory();
  }
}

template <typename T, ReallocType_t U>
void ReallocBuffer<T, U>::realloc(size_t new_num_elements, cudaStream_t stream) {
  size_t old_size = num_elements_ * sizeof(T);
  size_t new_size = new_num_elements * sizeof(T);
  if (new_size < old_size) {
    return;
  }

  if (U == ReallocType_t::NO_COPY) {
    HCTR_LIB_THROW(cudaFree(ptr_));
    HCTR_LIB_THROW(cudaMalloc(&ptr_, new_size));
  } else if (U == ReallocType_t::DEFAULT) {
    void* tmp;
    HCTR_LIB_THROW(cudaMalloc(&tmp, new_size));
    if (old_size) {
      HCTR_LIB_THROW(cudaMemcpyAsync(tmp, ptr_, old_size, cudaMemcpyDeviceToDevice, stream));
      HCTR_LIB_THROW(cudaFree(ptr_));
    }
    ptr_ = (T*)tmp;
  } else {  // MMAP
    get_aligned_size(old_size);
    get_aligned_size(new_size);

    if (new_size > old_size) {
      CUdevice device;
      HCTR_LIB_THROW(cudaGetDevice(&device));
      prop_.location.id = device;
      realloc_ptr_mmap((void**)&ptr_, old_size, new_size);
    }
  }

  num_elements_ = new_num_elements;
}

template <typename T, ReallocType_t U>
void ReallocBuffer<T, U>::realloc_ptr_mmap(void** ptr, size_t old_size, size_t new_size) {
  // Implementation based on
  // https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/

  HCTR_CHECK(access_desc_->size());
  size_t reserve_size = new_size - old_size;
  MMAP_DEBUG("Old %lu New %lu Reserve %lu bytes\n", old_size, new_size, reserve_size);

  // First step: create the neccessary virtual memory range
  // Second step: create physical memory rchunk and perform VM -> PM mapping
  // Most of the complexity is in first step when old_size != 0
  // Second step is common across different scenarios in first step

  // Physical memory handle
  CUmemGenericAllocationHandle allocHandle;
  CUdeviceptr new_ptr = 0;
  if (old_size == 0) {
    // Reserve a virtual address range
    HCTR_LIB_THROW(cuMemAddressReserve(&new_ptr, reserve_size, 0, 0, 0));
    vm_ranges_.push_back({new_ptr, reserve_size});
    *ptr = (void*)new_ptr;
  } else {
    // Try to reserve virtual memory at the end of old ptr
    const CUresult status =
        cuMemAddressReserve(&new_ptr, reserve_size, 0, (CUdeviceptr)((uint64_t)*ptr + old_size), 0);

    if ((status != CUDA_SUCCESS) || ((void*)new_ptr != (void*)((uint64_t)*ptr + old_size))) {
      // We couldn't get the address we wanted, so fall back to the slow path
      MMAP_DEBUG("Failed to extend VM range\n");

      // Don’t leak new_ptr if you got one
      if (new_ptr) {
        HCTR_LIB_THROW(cuMemAddressFree(new_ptr, reserve_size));
      }

      // Now reserve the new, bigger VA range
      HCTR_LIB_THROW(cuMemAddressReserve(&new_ptr, new_size, 0, 0, 0));

      // Map first part of new VA range to existing physical memory chunks, enabling their access
      std::vector<std::pair<CUdeviceptr, size_t>> new_mmap_ranges;
      CUdeviceptr tmp = new_ptr;
      MMAP_DEBUG("Remapping VM -> PM\n");
      for (auto handle : pm_handles_) {
        auto size = handle.second;
        HCTR_LIB_THROW(cuMemMap(tmp, size, 0, handle.first, 0));
        new_mmap_ranges.push_back({tmp, size});
        tmp += size;
      }

      // Set access permissions
      HCTR_LIB_THROW(
          cuMemSetAccess(new_ptr, old_size, &(access_desc_->at(0)), access_desc_->size()));

      // Unmap old mappings
      for (auto range : mmap_ranges_) {
        HCTR_LIB_THROW(cuMemUnmap(range.first, range.second));
      }
      // Clear old mappings, save new mappings
      mmap_ranges_ = new_mmap_ranges;

      // Free up previous VA allocations
      MMAP_DEBUG("Freeing %lu old VM allocations\n", vm_ranges_.size());
      for (auto range : vm_ranges_) {
        HCTR_LIB_THROW(cuMemAddressFree(range.first, range.second));
      }

      // Save new VA allocation information
      vm_ranges_.clear();
      vm_ranges_.push_back({new_ptr, new_size});

      *ptr = (void*)new_ptr;
      new_ptr = (CUdeviceptr)(((uint64_t)new_ptr) + old_size);
    } else {
      vm_ranges_.push_back({new_ptr, reserve_size});
    }
  }

  // Finally, create new physical memory chunk
  HCTR_LIB_THROW(cuMemCreate(&allocHandle, reserve_size, &prop_, 0));
  pm_handles_.push_back({allocHandle, reserve_size});

  // Map new_ptr to physical memory
  HCTR_LIB_THROW(cuMemMap(new_ptr, reserve_size, 0, allocHandle, 0));
  mmap_ranges_.push_back({new_ptr, reserve_size});

  // Set access permissions
  HCTR_LIB_THROW(
      cuMemSetAccess(new_ptr, reserve_size, &(access_desc_->at(0)), access_desc_->size()));
}

template <typename T, ReallocType_t U>
void ReallocBuffer<T, U>::release_mmap_memory() {
  MMAP_DEBUG("Release mmap metadata sizes: pm %lu vm %lu mmap %lu\n", pm_handles_.size(),
             vm_ranges_.size(), mmap_ranges_.size());

  CUresult status;
  // Unmap virtual memory
  for (auto range : mmap_ranges_) {
    HCTR_LIB_THROW(cuMemUnmap(range.first, range.second));
  }
  mmap_ranges_.clear();

  // Release virtual memory
  for (auto range : vm_ranges_) {
    HCTR_LIB_THROW(cuMemAddressFree(range.first, range.second));
  }
  vm_ranges_.clear();

  // Release physical memory
  for (auto handle : pm_handles_) {
    HCTR_LIB_THROW(cuMemRelease(handle.first));
  }
  pm_handles_.clear();
}

template class AverageLoss<float>;
template class AUC<float>;
template class AUC<__half>;
template class HitRate<float>;

}  // namespace metrics

}  // namespace HugeCTR
