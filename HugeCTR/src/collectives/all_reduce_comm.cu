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

#include <collectives/all_reduce_comm.hpp>
#include <collectives/ib_comm.hpp>
#include <utils.hpp>

namespace HugeCTR
{

  std::shared_ptr<AllReduceInPlaceComm> AllReduceInPlaceComm::create_nccl(
    size_t num_process, bool use_mixed_precision,
    const std::vector<std::shared_ptr<GPUResource>>& gpu_resources)
  {
    if (use_mixed_precision) {
      return std::make_shared<NCCLARInplaceComm<__half>>(num_process, gpu_resources);
    } else {
      return std::make_shared<NCCLARInplaceComm<float>>(num_process, gpu_resources);
    }
  }

  std::shared_ptr<AllReduceInPlaceComm> AllReduceInPlaceComm::create_oneshot(
    size_t num_process, bool use_mixed_precision,
    const std::vector<std::shared_ptr<GPUResource>>& gpu_resources)
  {
    if (num_process > 1) {
      CK_THROW_(Error_t::WrongInput, "Oneshot multi-node is not supported without MPI");
    }
    if (use_mixed_precision) {
      return std::make_shared<OneshotSingleARInplaceComm<__half>>(gpu_resources);
    } else {
      return std::make_shared<OneshotSingleARInplaceComm<float>>(gpu_resources);
    }
  }
      
#ifdef ENABLE_MPI

  std::shared_ptr<AllReduceInPlaceComm> AllReduceInPlaceComm::create_oneshot(
    size_t num_process, bool use_mixed_precision,
    const std::vector<std::shared_ptr<GPUResource>>& gpu_resources, IbComm* ib_comm)
  {
    if (num_process == 1) {
      if (use_mixed_precision) {
        return std::make_shared<OneshotSingleARInplaceComm<__half>>(gpu_resources);
      } else {
        return std::make_shared<OneshotSingleARInplaceComm<float>>(gpu_resources);
      }
    }
    if (use_mixed_precision) {
      return std::make_shared<OneshotMultiARInplaceComm<__half>>(ib_comm, num_process, gpu_resources);
    } else {
      return std::make_shared<OneshotMultiARInplaceComm<float>>(ib_comm, num_process, gpu_resources);
    }
  }

  std::shared_ptr<AllReduceInPlaceComm> AllReduceInPlaceComm::create(
    size_t num_process, AllReduceAlgo algo, bool use_mixed_precision,
    const std::vector<std::shared_ptr<GPUResource>>& gpu_resources, IbComm* ib_comm)
  {
    return (algo == AllReduceAlgo::ONESHOT) ? 
      create_oneshot(num_process, use_mixed_precision, gpu_resources, ib_comm) :
      create_nccl(num_process, use_mixed_precision, gpu_resources);
  }

#else
  
  std::shared_ptr<AllReduceInPlaceComm> AllReduceInPlaceComm::create(
    size_t num_process, AllReduceAlgo algo, bool use_mixed_precision,
    const std::vector<std::shared_ptr<GPUResource>>& gpu_resources)
  {
    return (algo == AllReduceAlgo::ONESHOT) ? 
      create_oneshot(num_process, use_mixed_precision, gpu_resources) :
      create_nccl(num_process, use_mixed_precision, gpu_resources);
  }

#endif

#ifdef ENABLE_MPI
  template <typename T>
  OneshotMultiARInplaceComm<T>::OneshotMultiARInplaceComm(
      IbComm* ib_comm,
      size_t num_procs,
      const std::vector<std::shared_ptr<GPUResource>>& gpu_resources):
    ib_comm_(ib_comm),
    num_procs_(num_procs),
    gpu_resources_(gpu_resources),
    num_gpus_(gpu_resources.size())
  {
  }

  template <typename T>
  AllReduceInPlaceComm::Handle OneshotMultiARInplaceComm<T>::register_coll()
  {
    ar_ctx_.emplace_back(std::make_unique<ARContext>());
    Handle handle = (Handle)(ar_ctx_.size() - 1);
    auto& ar_ctx_g = ar_ctx_[handle];
    ar_ctx_g->ctx_.resize(num_gpus_);
    ar_ctx_[handle]->ib_comm_handle_ = ib_comm_->register_ar_coll();

    return handle;
  }

  template<typename T>
  void OneshotMultiARInplaceComm<T>::set_coll_buf(Handle coll, void* ar_ptr, size_t ar_size, size_t g)
  {
    auto& ctx = ar_ctx_[coll];
    auto& ctx_g = ctx->ctx_[g];
    ctx_g.ar_ptr_ = ar_ptr;
    if ((ctx->ar_size_ != 0) && (ctx->ar_size_ != ar_size)) {
      CK_THROW_(Error_t::WrongInput, "AR size mismatch");
    }
    ctx->ar_size_ = ar_size;
    ib_comm_->set_ar_coll_buf<T>(ctx->ib_comm_handle_, ar_ptr, ar_size, g);
    // MESSAGE_("Oneshot AR size: " + std::to_string(ar_size));
  }

  template<typename T>
  void OneshotMultiARInplaceComm<T>::update_size(Handle coll, const size_t ar_size)
  {
    auto& ctx = ar_ctx_[coll];
    ctx->ar_size_ = ar_size;
    ib_comm_->update_size(ctx->ib_comm_handle_, ar_size);
    // MESSAGE_("Oneshot AR size updated to: " + std::to_string(ar_size));
  }

  template <typename T>
  void OneshotMultiARInplaceComm<T>::register_coll_buf(Handle coll)
  {
    auto& ctx = ar_ctx_[coll];
    ib_comm_->register_ar_coll_buf(ctx->ib_comm_handle_);
  }

  template <typename T>
  void OneshotMultiARInplaceComm<T>::all_reduce(AllReduceInPlaceComm::Handle coll, cudaStream_t stream, size_t g)
  {
    auto& ctx = ar_ctx_[coll];
    ib_comm_->all_reduce<T>(ctx->ib_comm_handle_, stream, g);
  }
  
  template class OneshotMultiARInplaceComm<__half>;
  template class OneshotMultiARInplaceComm<float>;
#endif

#define MAX_LOCAL_RANKS 32
#define TOTAL_FLAGS (2*MAX_LOCAL_RANKS + MAX_AR_CHANNELS)
#define MAX_AR_CHANNELS 31
#define AR_MAX_THREADS 1024
#define AR_BARRIER_FLAG_OFFSET 0
#define RS_SM_SYNC_OFFSET (RANKS)
#define AG_RANK_BCAST_OFFSET (RANKS + MAX_AR_CHANNELS)
#define UNROLL 8

template<int RANKS, typename T>
static __global__ void __launch_bounds__(AR_MAX_THREADS)
all_reduce_cuda_single(void** __restrict__ d_peer_ptrs, const int numlines,
    size_t* d_coll_cmd_, size_t** flags,
    const int device_id)
{
  // Do a barrier across all ranks
  volatile size_t* my_flag = flags[device_id];
  size_t base_count = *d_coll_cmd_;

  if (threadIdx.x < RANKS) {
    if (blockIdx.x == 0) {
      size_t* rem_flag = flags[threadIdx.x];
      rem_flag[AR_BARRIER_FLAG_OFFSET + device_id] = (base_count + 1);
    }
    while (my_flag[AR_BARRIER_FLAG_OFFSET + threadIdx.x] < (base_count + 1)) {
    }
  }

  const int warp = blockIdx.x + (threadIdx.x >> 5);
  uint4* remote_ptr[RANKS];
  for (int r = 0; r < RANKS; r++) {
    remote_ptr[r] = reinterpret_cast<uint4*>(d_peer_ptrs[(r + device_id + warp) % RANKS]);
  }
  uint4* my_ptr = reinterpret_cast<uint4*>(d_peer_ptrs[device_id]);
  __syncthreads();  // Post barrier and init sync

  /* reduce scatter */
  const int blocklines = numlines / RANKS;  // Assumption: numlines is divisible by RANKS
  const int blockstart = blocklines * device_id;

  for (int line = blockIdx.x * blockDim.x + threadIdx.x; line < blocklines;
       line += blockDim.x * gridDim.x) {
    uint4 val[RANKS];
#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      val[i] = remote_ptr[i][blockstart + line];
    }

    uint4 sum = val[0];
    T* s = reinterpret_cast<T*>(&sum);

#pragma unroll
    for (int i = 1; i < RANKS; i++) {
      T* v = reinterpret_cast<T*>(&val[i]);
#pragma unroll
      for (int j = 0; j < sizeof(uint4) / sizeof(T); j++) {
        s[j] += v[j];
      }
    }
    my_ptr[blockstart + line] = sum;
  }
  __syncthreads();

  // sync SMs --> SM 0
  if (threadIdx.x == 0) {
    __threadfence();
    if (blockIdx.x > 0) {
      my_flag[RS_SM_SYNC_OFFSET + blockIdx.x] = (base_count + 1);
    }
  } else if (blockIdx.x == 0) {
    if (threadIdx.x < gridDim.x) {
      while (((volatile size_t*)my_flag)[RS_SM_SYNC_OFFSET + threadIdx.x] < (base_count + 1)) {
      }
    }
  }
  __syncthreads();
  /* All gather flag broadcast to all ranks */
  if ((blockIdx.x == 0) && (threadIdx.x < RANKS)) {
    size_t* rem_flag = flags[threadIdx.x];
    rem_flag[AG_RANK_BCAST_OFFSET + device_id] = (base_count + 1);
    // printf("Wrote flag from %d: %llu %x\n", device_id, cachedflag, d_peer_ptrs[device_id]);
  }

  /* All gather */
  const int nwarps = ((blockDim.x) >> 5) / (RANKS - 1);
  const int myblockDim = nwarps << 5;
  const int mywarp = ((threadIdx.x) >> 5) / (RANKS - 1);
  const int maxthreadIdx = myblockDim * (RANKS - 1);
  const int mydest = (device_id + 1 + ((threadIdx.x) >> 5) % (RANKS - 1)) & (RANKS - 1);
  const int mythreadIdx = (mywarp << 5) + (threadIdx.x & 31);

  volatile size_t* flag = (volatile size_t*)&(my_flag[AG_RANK_BCAST_OFFSET + mydest]);
  uint4* dest_ptr = remote_ptr[((RANKS << 10) + mydest - device_id - warp) % RANKS];

  uint4* myptr = &my_ptr[blocklines * mydest];
  uint4* peerptr = &dest_ptr[blocklines * mydest];

  if (threadIdx.x < maxthreadIdx) {
    const int start_elem = mythreadIdx + myblockDim * blockIdx.x;
    const int end_elem = max(start_elem, blocklines);
    const int aligned_elem = ((end_elem - start_elem) / (myblockDim * gridDim.x * UNROLL)) *
                             (myblockDim * gridDim.x * UNROLL);
    const int end_aligned = start_elem + aligned_elem;

    if (mythreadIdx == 0) {
      while (*flag < (base_count + 1)) {
      }
      // printf("Gather flag received %llu %d %d %d %d %d %d %x\n", *flag, device_id, blockstart,
      // blocklines, numlines, remainder, mydest, dest_ptr);
    }
    asm volatile("bar.sync %0, %1;" ::"r"(3 + mydest), "r"(myblockDim));

    for (int line = start_elem; line < end_aligned; line += myblockDim * gridDim.x * UNROLL) {
      uint4 val[UNROLL];
#pragma unroll
      for (int i = 0; i < UNROLL; i++) {
        val[i] = peerptr[line + i * myblockDim * gridDim.x];
      }

#pragma unroll
      for (int i = 0; i < UNROLL; i++) {
        myptr[line + i * myblockDim * gridDim.x] = val[i];
      }
    }

    for (int line = end_aligned; line < end_elem; line += myblockDim * gridDim.x) {
      myptr[line] = peerptr[line];
    }
  }

  if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
    *d_coll_cmd_ = (base_count + 1);
  }
}

template <typename T>
OneshotSingleARInplaceComm<T>::OneshotSingleARInplaceComm(
    const std::vector<std::shared_ptr<GPUResource>>& gpu_resources):
  gpu_resources_(gpu_resources),
  num_gpus_(gpu_resources.size())
{
  if (getenv("ONESHOT_NCHANNELS"))   { cfg_nchannels_ = atoi(getenv("ONESHOT_NCHANNELS")); }
}

template <typename T>
AllReduceInPlaceComm::Handle OneshotSingleARInplaceComm<T>::register_coll()
{
  ar_ctx_.emplace_back(std::make_unique<ARContext>());
  Handle handle = (Handle)(ar_ctx_.size() - 1);
  auto& ar_ctx_g = ar_ctx_[handle];
  ar_ctx_g->ctx_.resize(num_gpus_);
  return handle;
}

template<typename T>
void OneshotSingleARInplaceComm<T>::set_coll_buf(Handle coll, void* ar_ptr, size_t ar_size, size_t g)
{
  auto& ctx = ar_ctx_[coll];
  auto& ctx_g = ctx->ctx_[g];
  ctx_g.ar_ptr_ = ar_ptr;
  if ((ctx->ar_size_ != 0) && (ctx->ar_size_ != ar_size)) {
    CK_THROW_(Error_t::WrongInput, "AR size mismatch");
  }
  ctx->ar_size_ = ar_size;
  // MESSAGE_("Oneshot AR size: " + std::to_string(ar_size));
}

template<typename T>
void OneshotSingleARInplaceComm<T>::update_size(Handle coll, const size_t ar_size)
{
  auto& ctx = ar_ctx_[coll];
  ctx->ar_size_ = ar_size;
  // MESSAGE_("Oneshot AR size updated to: " + std::to_string(ar_size));
}

template <typename T>
void OneshotSingleARInplaceComm<T>::register_coll_buf(Handle coll)
{ 
  auto& ctx = ar_ctx_[coll];
  // Allocations
  for (size_t g = 0; g < num_gpus_; g++) {
    CK_CUDA_THROW_(cudaSetDevice(gpu_resources_[g]->get_device_id()));
    auto& gpu_ctx = ctx->ctx_[g];
    gpu_ctx.buf_ = GeneralBuffer2<CudaAllocator>::create();
    gpu_ctx.buf_->reserve({num_gpus_}, &gpu_ctx.d_peer_ptrs_);
    gpu_ctx.buf_->reserve({1}, &gpu_ctx.d_coll_cmd_);
    gpu_ctx.buf_->reserve({TOTAL_FLAGS}, &gpu_ctx.d_flags_);
    gpu_ctx.buf_->reserve({num_gpus_}, &gpu_ctx.d_flags_ptrs_);
    gpu_ctx.buf_->allocate();
    CK_CUDA_THROW_(cudaMemset(gpu_ctx.buf_->get_ptr(), 
          0, gpu_ctx.buf_->get_size_in_bytes()));
  }

  std::vector<void*> h_peer_ptrs(num_gpus_);
  std::vector<void*> h_peer_flag_ptrs(num_gpus_);
  for (size_t g = 0; g < num_gpus_; g++) {
    auto& gpu_ctx = ctx->ctx_[g];
    h_peer_ptrs[g] = gpu_ctx.ar_ptr_;
    h_peer_flag_ptrs[g] = gpu_ctx.d_flags_.get_ptr();
  }

  for (size_t g = 0; g < num_gpus_; g++) {
    auto& gpu_ctx = ctx->ctx_[g];
    CK_CUDA_THROW_(cudaSetDevice(gpu_resources_[g]->get_device_id()));
    CK_CUDA_THROW_(cudaMemcpy(gpu_ctx.d_peer_ptrs_.get_ptr(), 
          h_peer_ptrs.data(), num_gpus_*sizeof(void*),
          cudaMemcpyHostToDevice));
    CK_CUDA_THROW_(cudaMemcpy(gpu_ctx.d_flags_ptrs_.get_ptr(),
          h_peer_flag_ptrs.data(), num_gpus_*sizeof(size_t*),
          cudaMemcpyHostToDevice));
  }

}

template <typename T>
void OneshotSingleARInplaceComm<T>::all_reduce(AllReduceInPlaceComm::Handle coll, cudaStream_t stream, size_t g)
{
  auto& ctx = ar_ctx_[coll];
  auto& gpu_ctx = ctx->ctx_[g];
  int numlines = ctx->ar_size_ / sizeof(uint4);
  int device_id_int = static_cast<int>(g);
  #define LAUNCH_KERNEL(RANKS) if(num_gpus_==RANKS) \
    all_reduce_cuda_single<RANKS, T><<<cfg_nchannels_, AR_MAX_THREADS, 0, stream>>>( \
    gpu_ctx.d_peer_ptrs_.get_ptr(), \
    numlines, \
    gpu_ctx.d_coll_cmd_.get_ptr(),  \
    gpu_ctx.d_flags_ptrs_.get_ptr(), \
    device_id_int);
  LAUNCH_KERNEL(2);
  LAUNCH_KERNEL(4);  
  LAUNCH_KERNEL(8);
}

template class OneshotSingleARInplaceComm<__half>;
template class OneshotSingleARInplaceComm<float>;

  template <typename T>
  NCCLARInplaceComm<T>::NCCLARInplaceComm(
      size_t num_procs,
      const std::vector<std::shared_ptr<GPUResource>>& gpu_resources):
    num_procs_(num_procs),
    gpu_resources_(gpu_resources),
    num_gpus_(gpu_resources.size())
  {
  }

  template <typename T>
  AllReduceInPlaceComm::Handle NCCLARInplaceComm<T>::register_coll()
  {
    ar_ctx_.emplace_back(std::make_unique<ARContext>());
    Handle handle = (Handle)(ar_ctx_.size() - 1);
    auto& ar_ctx_g = ar_ctx_[handle];
    ar_ctx_g->ctx_.resize(num_gpus_);
    
    return handle;
  }

  template<typename T>
  void NCCLARInplaceComm<T>::set_coll_buf(Handle coll, void* ar_ptr, size_t ar_size, size_t g)
  {
    auto& ctx = ar_ctx_[coll];
    auto& ctx_g = ctx->ctx_[g];
    ctx_g.ar_ptr_ = ar_ptr;
    if ((ctx->ar_size_ != 0) && (ctx->ar_size_ != ar_size)) {
      CK_THROW_(Error_t::WrongInput, "AR size mismatch");
    }
    ctx->ar_size_ = ar_size;
    // MESSAGE_("NCCL AR size: " + std::to_string(ar_size));
  }

  template<typename T>
  void NCCLARInplaceComm<T>::update_size(Handle coll, const size_t ar_size)
  {
    auto& ctx = ar_ctx_[coll];
    ctx->ar_size_ = ar_size;
    // MESSAGE_("NCCL AR size updated to: " + std::to_string(ar_size));
  }

  template<typename T>
  void NCCLARInplaceComm<T>::register_coll_buf(Handle coll)
  {
  }

  template <typename T>
  void NCCLARInplaceComm<T>::all_reduce(AllReduceInPlaceComm::Handle coll, cudaStream_t stream, size_t g)
  {
    auto& ctx = ar_ctx_[coll];
    auto& ctx_g = ctx->ctx_[g];
    CK_NCCL_THROW_(ncclAllReduce(
          (const void*) ctx_g.ar_ptr_, ctx_g.ar_ptr_,
          ctx->ar_size_ / sizeof(T),
          NcclDataType<T>::getType(),
          ncclSum,
          gpu_resources_[g]->get_nccl(),
          stream));
  }

  template class NCCLARInplaceComm<__half>;
  template class NCCLARInplaceComm<float>;
}
