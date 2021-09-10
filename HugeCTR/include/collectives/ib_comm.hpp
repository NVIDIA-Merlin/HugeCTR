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
#pragma once

#include <hwloc.h>
#include <hwloc/cudart.h>
#include <hwloc/openfabrics-verbs.h>
#include <omp.h>

#include <boost/variant.hpp>
#include <collectives/ib_proxy.hpp>
#include <general_buffer2.hpp>
#include <gpu_barrier.hpp>
#include <memory>
#include <string>
#include <tensor2.hpp>
#include <vector>

namespace HugeCTR {
class IbComm {
  // API
 public:
  int init(size_t num_procs, size_t num_gpus, size_t my_proc, const std::vector<int>& device_list);
  void finalize();

  /* Hier A2A coll api */
  HierA2ACollHandle register_hier_a2a_coll(bool skip_barrier = false);
  void set_a2a_coll_stream(HierA2ACollHandle coll, cudaStream_t stream, size_t device_id);
  void set_a2a_coll_buf(HierA2ACollHandle coll, void** send_ptrs, const size_t* send_max_size,
                        void** recv_ptrs, const size_t* recv_max_size, size_t device_id);
  void register_a2a_coll_buf(HierA2ACollHandle coll);
  void set_ready_to_transfer();
  void update_a2a_coll_buf(HierA2ACollHandle coll, const size_t* send_sizes,
                           const size_t* recv_sizes, cudaStream_t dep_stream, size_t device_id);
  template <typename T>
  void post_send_command_a2a(HierA2ACollHandle coll, cudaStream_t dep_stream, size_t device_id);

  /* Hier A2Av coll api. Requires control buffer for intra-node A2A */
  HierA2AvCollHandle register_hier_a2a_v_coll(bool skip_barrier = false);
  void set_a2a_coll_stream(HierA2AvCollHandle coll, cudaStream_t stream, size_t device_id);
  void wait_global_recv_async(HierA2ACollHandle coll, size_t device_id);

  // NOTE: Assumes contiguous buffer, no support for non contiguous buffers
  void set_a2a_coll_buf(HierA2AvCollHandle coll, void* send_ptrs, const size_t send_max_size,
                        void* recv_ptrs, const size_t recv_max_size, size_t device_id);
  void register_a2a_coll_buf(HierA2AvCollHandle coll);
  void update_a2a_coll_sizes(HierA2AvCollHandle coll, const size_t* send_sizes,
                             const size_t* recv_sizes, cudaStream_t dep_stream, size_t device_id);
  void pre_intra_update_a2a_coll_sizes(HierA2AvCollHandle coll, size_t** d_pre_intra_send_sizes,
                                       cudaStream_t dep_stream, size_t device_id);
  template <typename T>
  void post_send_command_a2a(HierA2AvCollHandle coll, cudaStream_t dep_stream, size_t device_id);
  template <typename T>
  void post_a2a_send_command(HierA2AvCollHandle coll, cudaStream_t dep_stream, size_t device_id);
  void blocking_wait(HierA2AvCollHandle coll, cudaStream_t dep_stream, size_t device_id);
  void wait_global_recv_async(HierA2AvCollHandle coll, size_t device_id);

  // AR coll api
  ARCollHandle register_ar_coll();
  template <typename T>
  void set_ar_coll_buf(ARCollHandle coll, void* ar_ptr, const size_t ar_size, size_t device_id);
  void register_ar_coll_buf(ARCollHandle coll);
  void update_size(ARCollHandle coll,
                   const size_t ar_size);  // If size is not known during buffer registration
  template <typename T>
  void all_reduce(ARCollHandle coll, cudaStream_t stream, size_t device_id);
  template <typename T>
  void all_reduce(ARCollHandle coll, size_t device_id);  // When dep_stream is same as main stream

  IbComm() = default;
  ~IbComm();

 private:
  template <int RANKS, typename T>
  void all_reduce(ARCollHandle coll, cudaStream_t stream, size_t device_id) const;
  template <typename T>
  sharp_datatype get_sharp_dtype();

  struct IbDevs {
    std::string dev_name;
    int dev_port_id;
    hwloc_obj_t hwloc_obj;
    size_t num_gpus_assigned = 0;
    std::vector<size_t> dev_affinities;
  };

  size_t num_gpus_;
  size_t num_procs_;
  size_t my_proc_;
  std::vector<int> device_list_;

  hwloc_topology_t topo_;
  std::vector<IbDevs> ib_dev_list_;
  std::vector<size_t> gpu_nic_affinity_;
  std::vector<pthread_t> proxy_thread_;
  std::unique_ptr<ProxyCommand> proxy_cmd_;
  std::vector<std::unique_ptr<IbvProxy::InitConfig>> proxy_cfg_;

  // Keep a copy local to the main thread as well, in addition to proxy thread for numa
  struct HierA2ACollContextPerGPU {
    size_t* h_send_sizes_ = NULL;
    size_t* h_recv_sizes_ = NULL;
    size_t* d_send_sizes_copy_ = NULL;
    void** d_send_ptrs_ = NULL;
    void** d_recv_ptrs_ = NULL;
    size_t h_max_send_size_ = 0;
    cudaStream_t stream_ = 0;
    cudaEvent_t event_;

    ~HierA2ACollContextPerGPU();
  };

  struct HierA2ACollContext {
    std::vector<std::unique_ptr<HierA2ACollContextPerGPU>> ctx_;

    size_t* cmd_storage_ = NULL;
    size_t* h_recv_cmd_ptr_ = NULL;
    size_t* h_send_cmd_ptr_ = NULL;
    size_t** d_send_cmd_ = NULL;
    size_t** d_ibv_atomic_ = NULL;
    size_t** d_ibv_atomic_recv_ = NULL;
    std::unique_ptr<GPUBarrier> barrier_ = NULL;
    std::unique_ptr<CollSyncHelper> sync_helper_ = NULL;

    HierA2ACollContext(IbComm* comm);
    ~HierA2ACollContext();
  };

  struct ARCollContextPerGPU {
    void* d_ar_ptr_ = NULL;
    size_t* h_rs_cmd_ = NULL;
    size_t* d_ag_cmd_ = NULL;

    Tensor2<void*> d_peer_ptrs_;
    Tensor2<size_t> d_coll_cmd_;
    Tensor2<size_t> d_flags_;
    Tensor2<size_t*> d_flags_ptrs_;
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf_ = NULL;
  };

  struct ARCollContext {
    std::vector<std::unique_ptr<ARCollContextPerGPU>> ctx_;
    size_t num_gpus_ = 0;
    size_t ar_size_ = 0;
    int blocksize_ = 0;
    int peer_blocklines_ = 0;
    int num_blocks_ = 1;

    int cfg_nblocks_ = AR_NBLOCKS;
    int cfg_align_block_ = AR_ALIGN_BLOCK;
    int cfg_min_block_ = AR_MIN_BLOCK;
    int cfg_nchannels_ = 16;

    void update_size(size_t ar_size);

    ARCollContext(IbComm* comm);
    ~ARCollContext() = default;
  };

  template <int N>
  using index_t = std::integral_constant<int, N>;  // C++14

  std::vector<std::unique_ptr<HierA2ACollContext>> hier_a2a_coll_ctx_;
  std::vector<std::unique_ptr<HierA2ACollContext>> hier_a2a_v_coll_ctx_;
  std::vector<std::unique_ptr<ARCollContext>> ar_coll_ctx_;

  bool is_initialized_ = false;
  bool is_ready_to_transfer_ = false;
  bool is_finalized_ = false;

  // Debug helpers
  void print_obj(size_t my_rank, std::string obj_name, hwloc_obj_t obj);
  void print_distance_matrix(size_t my_rank, std::vector<std::vector<size_t>>& distances);

  // Helpers
  void detect_ib_devs();
  size_t calculate_pcie_hier_distance(size_t my_rank, hwloc_obj_t obj1, hwloc_obj_t obj2);
  void calculate_gpu_nic_affinity();
  void init_proxy_threads();
};
}  // namespace HugeCTR
#endif  // ENABLE_MPI
