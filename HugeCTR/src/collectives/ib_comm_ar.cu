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

#ifdef ENABLE_MPI
#include <infiniband/verbs.h>

#include <boost/preprocessor.hpp>
#include <collectives/ib_comm.hpp>
#include <iostream>
#include <sstream>
#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {

#define MAX_AR_CHANNELS 31

IbComm::ARCollContext::ARCollContext(IbComm* comm) {
  size_t num_gpus = comm->num_gpus_;
  num_gpus_ = num_gpus;
  std::generate_n(std::back_inserter(ctx_), num_gpus,
                  [] { return std::make_unique<ARCollContextPerGPU>(); });

  // Read config params from env
  if (getenv("ONESHOT_NBLOCKS")) {
    cfg_nblocks_ = atoi(getenv("ONESHOT_NBLOCKS"));
  }
  if (getenv("ONESHOT_ALIGN_BLOCK")) {
    cfg_align_block_ = atoi(getenv("ONESHOT_ALIGN_BLOCK"));
  }
  if (getenv("ONESHOT_MIN_BLOCK")) {
    cfg_min_block_ = atoi(getenv("ONESHOT_MIN_BLOCK"));
  }
  if (getenv("ONESHOT_NCHANNELS")) {
    cfg_nchannels_ = atoi(getenv("ONESHOT_NCHANNELS"));
  }

  PROXY_ASSERT_MSG(cfg_nchannels_ <= MAX_AR_CHANNELS, "Max oneshot channels is 31");
  PROXY_ASSERT(cfg_nblocks_ <= AR_MAX_BLOCKS);

  MESSAGE_("using oneshot nblocks: " + std::to_string(cfg_nblocks_));
  MESSAGE_("using oneshot nchannels: " + std::to_string(cfg_nchannels_));
  MESSAGE_("using oneshot min block: " + std::to_string(cfg_min_block_));
}

void IbComm::ARCollContext::update_size(size_t ar_size) {
  // calculate peerblock size
  PROXY_ASSERT_MSG((ar_size % (num_gpus_ * 16)) == 0, "AR size needs to be aligned to num_gpus*16");

  ar_size_ = ar_size;
  blocksize_ =
      (cfg_nblocks_ - 1 + (cfg_align_block_ - 1 + ar_size) / cfg_align_block_) / cfg_nblocks_;
  blocksize_ *= cfg_align_block_;
  if (blocksize_ < cfg_min_block_) {
    blocksize_ = cfg_min_block_;
  }
  peer_blocklines_ = blocksize_ / sizeof(uint4) / num_gpus_;
  num_blocks_ = (ar_size + blocksize_ - 1) / blocksize_;
  PROXY_ASSERT(num_blocks_ <= AR_MAX_BLOCKS);
}

ARCollHandle IbComm::register_ar_coll() {
  ar_coll_ctx_.emplace_back(std::make_unique<ARCollContext>(this));
  ARCollHandle coll_handle = (ARCollHandle)(ar_coll_ctx_.size() - 1);
  for (size_t g = 0; g < num_gpus_; g++) {
    M2PARCollInit coll_init_cmd_;
    coll_init_cmd_.coll_handle_ = coll_handle;
    coll_init_cmd_.cfg_nblocks_ = ar_coll_ctx_[coll_handle]->cfg_nblocks_;
    coll_init_cmd_.cfg_align_block_ = ar_coll_ctx_[coll_handle]->cfg_align_block_;
    coll_init_cmd_.cfg_min_block_ = ar_coll_ctx_[coll_handle]->cfg_min_block_;
    ARCollInitCmd cmd = std::make_pair(std::move(coll_init_cmd_), std::move(P2MNull()));
    proxy_cmd_->cmd_[g] = std::move(cmd);
  }
  proxy_cmd_->post_command();
  proxy_cmd_->wait_for_completion();
  proxy_cmd_->reset();
  return coll_handle;
}

template <>
sharp_datatype IbComm::get_sharp_dtype<int>() {
  return SHARP_DTYPE_INT;
}
template <>
sharp_datatype IbComm::get_sharp_dtype<uint32_t>() {
  return SHARP_DTYPE_UNSIGNED;
}
template <>
sharp_datatype IbComm::get_sharp_dtype<__half>() {
  return SHARP_DTYPE_FLOAT_SHORT;
}
template <>
sharp_datatype IbComm::get_sharp_dtype<float>() {
  return SHARP_DTYPE_FLOAT;
}

template <typename T>
void IbComm::set_ar_coll_buf(ARCollHandle coll, void* ar_ptr, const size_t ar_size,
                             size_t device_id) {
  PROXY_ASSERT(ar_size != 0);
  auto& coll_ctx = *ar_coll_ctx_[coll];
  if (proxy_cmd_->cmd_[device_id].which() != 0) {
    ERROR_MESSAGE_("Proxy command is already populated. Don't mix up set API");
    exit(1);
  }
  proxy_cmd_->cmd_[device_id] = ARBufInitCmd();
  ARBufInitCmd& cmd = boost::get<ARBufInitCmd>(proxy_cmd_->cmd_[device_id]);
  M2PARBufInit& buf_init = std::get<0>(cmd);

  auto& gpu_ctx = *coll_ctx.ctx_[device_id];
  gpu_ctx.d_ar_ptr_ = ar_ptr;

  buf_init.coll_handle_ = coll;
  buf_init.d_ar_ptr_ = ar_ptr;
  buf_init.ar_size_ = ar_size;
  buf_init.sharp_dtype_ = get_sharp_dtype<T>();
  buf_init.element_size_ = sizeof(T);

  if (coll_ctx.ar_size_ != 0) {
    PROXY_ASSERT(ar_size == coll_ctx.ar_size_);
  }
  coll_ctx.ar_size_ = ar_size;
  PROXY_ASSERT_MSG(((size_t)ar_ptr & 0xf) == 0, "AR pointer needs to aligned to 16B");
}

template void IbComm::set_ar_coll_buf<__half>(ARCollHandle coll, void* ar_ptr, const size_t ar_size,
                                              size_t device_id);
template void IbComm::set_ar_coll_buf<float>(ARCollHandle coll, void* ar_ptr, const size_t ar_size,
                                             size_t device_id);
template void IbComm::set_ar_coll_buf<uint32_t>(ARCollHandle coll, void* ar_ptr,
                                                const size_t ar_size, size_t device_id);

#define MAX_LOCAL_RANKS 32
#define TOTAL_FLAGS (2 * MAX_LOCAL_RANKS + MAX_AR_CHANNELS)

void IbComm::register_ar_coll_buf(ARCollHandle coll) {
  auto& coll_ctx = ar_coll_ctx_[coll];
  proxy_cmd_->post_command();
  proxy_cmd_->wait_for_completion();

  // Allocations
  for (size_t g = 0; g < num_gpus_; g++) {
    CK_CUDA_THROW_(cudaSetDevice(device_list_[g]));
    auto& gpu_ctx = *coll_ctx->ctx_[g];
    gpu_ctx.buf_ = GeneralBuffer2<CudaAllocator>::create();
    gpu_ctx.buf_->reserve({num_gpus_}, &gpu_ctx.d_peer_ptrs_);
    gpu_ctx.buf_->reserve({1}, &gpu_ctx.d_coll_cmd_);
    gpu_ctx.buf_->reserve({TOTAL_FLAGS}, &gpu_ctx.d_flags_);
    gpu_ctx.buf_->reserve({num_gpus_}, &gpu_ctx.d_flags_ptrs_);
    gpu_ctx.buf_->allocate();
    CK_CUDA_THROW_(cudaMemset(gpu_ctx.buf_->get_ptr(), 0, gpu_ctx.buf_->get_size_in_bytes()));
  }

  // Get proxy output
  std::vector<void*> h_peer_ptrs(num_gpus_);
  std::vector<void*> h_peer_flag_ptrs(num_gpus_);
  for (size_t g = 0; g < num_gpus_; g++) {
    auto& gpu_ctx = *coll_ctx->ctx_[g];
    h_peer_ptrs[g] = gpu_ctx.d_ar_ptr_;
    h_peer_flag_ptrs[g] = gpu_ctx.d_flags_.get_ptr();

    ARBufInitCmd& proxy_cmd = boost::get<ARBufInitCmd>(proxy_cmd_->cmd_[g]);
    auto& buf_init_out = std::get<1>(proxy_cmd);
    gpu_ctx.h_rs_cmd_ = buf_init_out.h_rs_cmd_;
    gpu_ctx.d_ag_cmd_ = buf_init_out.d_ag_cmd_;
  }

  for (size_t g = 0; g < num_gpus_; g++) {
    auto& gpu_ctx = *coll_ctx->ctx_[g];
    CK_CUDA_THROW_(cudaSetDevice(device_list_[g]));
    CK_CUDA_THROW_(cudaMemcpy(gpu_ctx.d_peer_ptrs_.get_ptr(), h_peer_ptrs.data(),
                              num_gpus_ * sizeof(void*), cudaMemcpyHostToDevice));
    CK_CUDA_THROW_(cudaMemcpy(gpu_ctx.d_flags_ptrs_.get_ptr(), h_peer_flag_ptrs.data(),
                              num_gpus_ * sizeof(size_t*), cudaMemcpyHostToDevice));
  }
  coll_ctx->update_size(coll_ctx->ar_size_);
  proxy_cmd_->reset();
}

void IbComm::update_size(ARCollHandle coll, const size_t ar_size) {
  auto& ctx = ar_coll_ctx_[coll];
  PROXY_ASSERT_MSG(ar_size < ctx->ar_size_, "updated AR size must be less than init size");
  for (size_t g = 0; g < num_gpus_; g++) {
    proxy_cmd_->cmd_[g] = ARUpdateSizeCmd();
    auto& cmd = boost::get<ARUpdateSizeCmd>(proxy_cmd_->cmd_[g]);
    auto& m2p_cmd = std::get<0>(cmd);
    m2p_cmd.ar_size_ = ar_size;
    m2p_cmd.coll_handle_ = coll;
  }
  proxy_cmd_->post_command();
  ctx->update_size(ar_size);
  proxy_cmd_->wait_for_completion();
  proxy_cmd_->reset();
}

// TODO: rs sync threads is max(SMS + 1, RANKS)
#define AR_MAX_THREADS 1024
#define AR_BARRIER_FLAG_OFFSET 0
#define RS_SM_SYNC_OFFSET (RANKS)
#define AG_RANK_BCAST_OFFSET (RANKS + MAX_AR_CHANNELS)
#define UNROLL 6

#define RS_SYNC_THREADS 32  // MAX of AR_CHANNELS + 1 and RANKS
#define AR_WORKER_THREADS (blockDim.x - RS_SYNC_THREADS)

template <int RANKS, typename T>
static __global__ void __launch_bounds__(AR_MAX_THREADS)
    all_reduce_cuda(void** __restrict__ d_peer_ptrs, const int numlines, size_t* d_coll_cmd_,
                    size_t* h_rs_cmd_, size_t* d_ag_cmd_, size_t** flags, const int peerblocklines,
                    const int numblocks, const int device_id) {
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

  if (threadIdx.x < RS_SYNC_THREADS) {
    __syncthreads();  // Post barrier and init sync

    /* sync across SMs and write a single RS complete flag to host */
    for (int nblock = 0; nblock < numblocks; nblock++) {
      asm volatile("bar.sync 1, %0;" ::"r"(AR_WORKER_THREADS + RS_SYNC_THREADS));

      size_t flag_count = (nblock + base_count + 1);
      if (threadIdx.x == 0) {
        __threadfence();
        if (blockIdx.x > 0) {
          my_flag[RS_SM_SYNC_OFFSET + blockIdx.x] = flag_count;
        }
      } else if (blockIdx.x == 0) {
        if (threadIdx.x < gridDim.x) {
          while (((volatile size_t*)my_flag)[RS_SM_SYNC_OFFSET + threadIdx.x] < flag_count) {
          }
        }
      }
      if (blockIdx.x == 0) {
        asm volatile("bar.sync 2, %0;" ::"r"(RS_SYNC_THREADS));
        if (threadIdx.x == 0) {
          *h_rs_cmd_ = flag_count;
        }
      }
    }

    /* All gather flag broadcast to all ranks */
    size_t cachedflag = base_count;
    if ((blockIdx.x == 0) && (threadIdx.x < RANKS)) {
      while (cachedflag < base_count + numblocks) {
        size_t newflag = *(volatile size_t*)(d_ag_cmd_);
        if (newflag == cachedflag) continue;
        cachedflag = newflag;
        size_t* rem_flag = flags[threadIdx.x];
        rem_flag[AG_RANK_BCAST_OFFSET + device_id] = cachedflag;
        // printf("Wrote flag from %d: %llu %x\n", device_id, cachedflag, d_peer_ptrs[device_id]);
      }
    }
  } else {
    constexpr int basethread = RS_SYNC_THREADS;
    const int warp = blockIdx.x + (threadIdx.x >> 5);
    uint4* remote_ptr[RANKS];
    for (int r = 0; r < RANKS; r++) {
      remote_ptr[r] = reinterpret_cast<uint4*>(d_peer_ptrs[(r + device_id + warp) % RANKS]);
    }
    uint4* my_ptr = reinterpret_cast<uint4*>(d_peer_ptrs[device_id]);
    __syncthreads();  // Post barrier and init sync

    int blocklineoffset = 0;
    while (blocklineoffset < numlines) {
      /* reduce scatter */
      const int remainder = min(numlines - blocklineoffset, peerblocklines * RANKS);
      const int blocklines = remainder / RANKS;  // Assumption: numlines is divisible by RANKS
      const int blockstart = blocklineoffset + blocklines * device_id;
      const int myThreadIdx = threadIdx.x - basethread;

      for (int line = blockIdx.x * AR_WORKER_THREADS + myThreadIdx; line < blocklines;
           line += AR_WORKER_THREADS * gridDim.x) {
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

      asm volatile("bar.sync 1, %0;" ::"r"(AR_WORKER_THREADS + RS_SYNC_THREADS));
      blocklineoffset += peerblocklines * RANKS;
    }  // Reduce scatter

    {
      /* All gather */
      const int nwarps = ((AR_WORKER_THREADS) >> 5) / (RANKS - 1);
      const int myblockDim = nwarps << 5;
      const int mywarp = ((threadIdx.x - basethread) >> 5) / (RANKS - 1);
      const int maxthreadIdx = myblockDim * (RANKS - 1) + basethread;
      const int mydest =
          (device_id + 1 + ((threadIdx.x - basethread) >> 5) % (RANKS - 1)) & (RANKS - 1);
      const int mythreadIdx = (mywarp << 5) + (threadIdx.x & 31);

      volatile size_t* flag = (volatile size_t*)&(my_flag[AG_RANK_BCAST_OFFSET + mydest]);
      uint4* dest_ptr = remote_ptr[((RANKS << 10) + mydest - device_id - warp) % RANKS];

      blocklineoffset = 0;
      int gather_count = (base_count + 1);
      while (blocklineoffset < numlines) {
        const int remainder = min(numlines - blocklineoffset, peerblocklines * RANKS);
        const int blocklines = remainder / RANKS;
        const int blockstart = blocklineoffset;

        uint4* myptr = &my_ptr[blockstart + blocklines * mydest];
        uint4* peerptr = &dest_ptr[blockstart + blocklines * mydest];

        if (threadIdx.x < maxthreadIdx) {
          const int start_elem = mythreadIdx + myblockDim * blockIdx.x;
          const int end_elem = max(start_elem, blocklines);
          const int aligned_elem = ((end_elem - start_elem) / (myblockDim * gridDim.x * UNROLL)) *
                                   (myblockDim * gridDim.x * UNROLL);
          const int end_aligned = start_elem + aligned_elem;

          if (mythreadIdx == 0) {
            while (*flag < gather_count) {
            }
            // printf("Gather flag received %llu %d %d %d %d %d %d %x\n", *flag, device_id,
            // blockstart, blocklines, numlines, remainder, mydest, dest_ptr);
            gather_count++;
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
        blocklineoffset += peerblocklines * RANKS;
      }
    }  // All-gather
  }
  if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
    *d_coll_cmd_ = (base_count + numblocks);
  }
}

template <int RANKS, typename T>
void IbComm::all_reduce(ARCollHandle coll, cudaStream_t stream, size_t device_id) const {
  auto& ctx = ar_coll_ctx_[coll];
  auto& gpu_ctx = ctx->ctx_[device_id];
  auto warps = max(RANKS, AR_MAX_THREADS / 32);
  int numlines = ctx->ar_size_ / sizeof(uint4);
  int device_id_int = static_cast<int>(device_id);

  all_reduce_cuda<RANKS, T><<<ctx->cfg_nchannels_, warps * 32, 0, stream>>>(
      gpu_ctx->d_peer_ptrs_.get_ptr(),
      numlines,  // number of 16B lines
      gpu_ctx->d_coll_cmd_.get_ptr(), gpu_ctx->h_rs_cmd_, gpu_ctx->d_ag_cmd_,
      gpu_ctx->d_flags_ptrs_.get_ptr(), ctx->peer_blocklines_, ctx->num_blocks_, device_id_int);
}

#define SUPPORTED_AR_RANKS (2)(4)(8)(16)

template <typename T>
void IbComm::all_reduce(ARCollHandle coll, cudaStream_t stream, size_t device_id) {
#define SWITCHER(r, data, p)                          \
  if (p == num_gpus_) {                               \
    return all_reduce<p, T>(coll, stream, device_id); \
  }
  BOOST_PP_SEQ_FOR_EACH(SWITCHER, "", SUPPORTED_AR_RANKS)
#undef SWITCHER
  PROXY_ASSERT_MSG(false, "Unsupported number of local GPU");
}

#define AR_METHOD(r, data, p)                                                            \
  template void IbComm::all_reduce<p, __half>(ARCollHandle, cudaStream_t, size_t) const; \
  template void IbComm::all_reduce<p, float>(ARCollHandle, cudaStream_t, size_t) const;  \
  template void IbComm::all_reduce<p, uint32_t>(ARCollHandle, cudaStream_t, size_t) const;

BOOST_PP_SEQ_FOR_EACH(AR_METHOD, "", SUPPORTED_AR_RANKS)
#undef AR_METHOD

template void IbComm::all_reduce<__half>(ARCollHandle, cudaStream_t, size_t);
template void IbComm::all_reduce<float>(ARCollHandle, cudaStream_t, size_t);
template void IbComm::all_reduce<uint32_t>(ARCollHandle, cudaStream_t, size_t);
}  // namespace HugeCTR
#endif
