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
#include <api/sharp.h>
#include <infiniband/verbs.h>

#include <boost/serialization/strong_typedef.hpp>
#include <boost/variant.hpp>
#include <common.hpp>
#include <condition_variable>

#include "gdrapi.h"

#define PROXY_ASSERT(expr)                                                    \
  if (!(expr)) {                                                              \
    HCTR_LOG_S(ERROR, WORLD) << #expr << ' ' << HCTR_LOCATION() << std::endl; \
    exit(1);                                                                  \
  }
#define PROXY_ASSERT_MSG(expr, msg)                                         \
  if (!(expr)) {                                                            \
    HCTR_LOG_S(ERROR, WORLD) << msg << ' ' << HCTR_LOCATION() << std::endl; \
    exit(1);                                                                \
  }

#define MAX_IBV_DEST 1024
#define MAX_SHARP_BLOCKS 16
#define AR_NBLOCKS 8
#define AR_MAX_BLOCKS 16
#define AR_ALIGN_BLOCK 256 * 1024
#define AR_MIN_BLOCK 256 * 1024

// TODO: Limit max colls per comm to 4.

namespace HugeCTR {
class ResourceManager;
struct CollSyncHelper {
 public:
  void recv_bcast() { recv_bcast_cnt_++; }
  void wait_recv_bcast(size_t count) {
    while (*(volatile size_t*)(&recv_bcast_cnt_) < count)
      ;
  }

 private:
  volatile size_t recv_bcast_cnt_ = 1;
};

BOOST_STRONG_TYPEDEF(size_t, HierA2ACollHandle)
BOOST_STRONG_TYPEDEF(size_t, HierA2AvCollHandle)
BOOST_STRONG_TYPEDEF(size_t, ARCollHandle)

enum IbvProxyState { INIT, READY_TO_TRANSFER, DESTROY };

// Proxy Init command structures
struct M2PNull {};
struct P2MNull {};

struct M2PStateTransition {
  IbvProxyState state_;
};
struct M2PHierA2ACollInit {
  HierA2ACollHandle coll_handle_;
  CollSyncHelper* sync_helper_;
  bool skip_barrier_;
  M2PHierA2ACollInit(HierA2ACollHandle handle_, CollSyncHelper* sync_helper, bool skip_barrier)
      : coll_handle_(handle_), sync_helper_(sync_helper), skip_barrier_(skip_barrier) {}
};
struct M2PHierA2AvCollInit {
  HierA2AvCollHandle coll_handle_;
  CollSyncHelper* sync_helper_;
  bool skip_barrier_;
  size_t num_buffers_;
  M2PHierA2AvCollInit(HierA2AvCollHandle handle_, CollSyncHelper* sync_helper, bool skip_barrier,
                      size_t num_buffers)
      : coll_handle_(handle_),
        sync_helper_(sync_helper),
        skip_barrier_(skip_barrier),
        num_buffers_(num_buffers) {}
};
struct M2PARCollInit {
  ARCollHandle coll_handle_;
  int cfg_nblocks_;
  int cfg_align_block_;
  int cfg_min_block_;
};

struct M2PHierA2ABufInit {
  HierA2ACollHandle coll_handle_;
  void** d_send_ptrs_;
  void** d_recv_ptrs_;
  const size_t* h_max_send_size_;
  const size_t* h_max_recv_size_;

  size_t* h_recv_cmd_ptr_;
  size_t* d_ibv_atomic_;
  size_t* d_ibv_atomic_recv_;
};

struct M2PHierA2AvBufInit {
  HierA2AvCollHandle coll_handle_;
  void* d_send_ptrs_;
  void* d_recv_ptrs_;
  size_t h_max_send_size_;
  size_t h_max_recv_size_;

  size_t* h_recv_cmd_ptr_;
  size_t* d_ibv_atomic_;
  size_t* d_ibv_atomic_recv_;
};

struct M2PARBufInit {
  ARCollHandle coll_handle_;
  void* d_ar_ptr_;
  size_t ar_size_;
  sharp_datatype sharp_dtype_;
  int element_size_;
};

struct P2MHierA2ABufInit {
  size_t* h_send_size_;
  size_t* h_recv_size_;
};

struct P2MHierA2AvBufInit {
  size_t* h_send_size_;
  size_t* h_recv_size_;
};

struct P2MARBufInit {
  size_t* h_rs_cmd_;
  size_t* d_ag_cmd_;
};

struct M2PARUpdateSize {
  ARCollHandle coll_handle_;
  size_t ar_size_;
};

typedef std::tuple<M2PHierA2ABufInit, P2MHierA2ABufInit> HierA2ABufInitCmd;
typedef std::tuple<M2PHierA2AvBufInit, P2MHierA2AvBufInit> HierA2AvBufInitCmd;
typedef std::tuple<M2PARBufInit, P2MARBufInit> ARBufInitCmd;
typedef std::tuple<M2PHierA2ACollInit, P2MNull> HierA2ACollInitCmd;
typedef std::tuple<M2PHierA2AvCollInit, P2MNull> HierA2AvCollInitCmd;
typedef std::tuple<M2PARCollInit, P2MNull> ARCollInitCmd;
typedef std::tuple<M2PARUpdateSize, P2MNull> ARUpdateSizeCmd;
typedef std::tuple<M2PStateTransition, P2MNull> ProxyStateTransitionCmd;
typedef boost::variant<boost::blank, ARBufInitCmd, ARCollInitCmd, ARUpdateSizeCmd,
                       HierA2ABufInitCmd, HierA2ACollInitCmd, HierA2AvBufInitCmd,
                       HierA2AvCollInitCmd, ProxyStateTransitionCmd>
    ProxyCmdType;

struct ProxyCommand {
 public:
  ProxyCommand(size_t num_threads);

  std::vector<ProxyCmdType> cmd_;
  std::mutex mutex_;

  void post_command();
  // void wait_new_command(size_t thread_id, std::unique_lock<std::mutex>& lock);
  void wait_new_command(size_t thread_id);
  void post_completion(size_t thread_id);
  // void wait_for_completion(std::unique_lock<std::mutex>& lock);
  void wait_for_completion();
  void reset();
  void set_destroy() {
    destroy_ = 1;
    __sync_synchronize();
  }
  bool check_destroy() { return (*(volatile int*)(&destroy_) == 1); }

 private:
  std::condition_variable cond_;
  std::vector<size_t> last_cmd_;
  size_t cmpl_cntr_ = 0;
  size_t cmd_cntr_ = 0;
  size_t num_threads_ = 0;
  volatile int destroy_ = 0;
};

struct IbvProxy {
  struct IbQpInfo {
    uint32_t lid;
    uint8_t ib_port;
    uint32_t qpn;
    enum ibv_mtu mtu;
  };

  struct InitConfig {
    size_t device_id_;
    size_t global_id_;
    size_t proxy_id_;
    size_t num_gpus_;
    size_t num_procs_;
    size_t my_proc_;
    std::string ib_dev_;
    unsigned int ib_port_;
    ProxyCommand* proxy_cmd_;
  };

  struct HierA2AIbvContext {
    size_t num_procs_ = 1;
    size_t proxy_id_ = 0;
    bool is_finalized_ = false;

    struct ibv_context* context_ = NULL;
    struct ibv_pd* pd_ = NULL;
    struct ibv_cq** cq_ = NULL;
    struct ibv_qp** qp_ = NULL;
    struct IbQpInfo* qp_infos_ = NULL;
    struct IbQpInfo* rem_qp_infos_ = NULL;
    void init_ibv(const IbvProxy::InitConfig& cfg);
    void finalize_ibv();

#ifdef SHARP_A2A
    struct sharp_coll_context* sharp_coll_context_;
    struct sharp_coll_comm* sharp_coll_comm_;
    void init_sharp(const IbvProxy::InitConfig& cfg);
    void finalize_sharp();
#endif

    HierA2AIbvContext(const IbvProxy::InitConfig& cfg);
    ~HierA2AIbvContext();
  };

  struct HierA2ACollContext {
    enum State { BUF_INIT_PENDING, WAIT_RECV_CMD, WAIT_COMPLETION };

    State state_ = BUF_INIT_PENDING;
    CollSyncHelper* sync_helper_;

    void** send_ptrs_ = NULL;
    void** recv_ptrs_ = NULL;
    size_t* send_sizes_ = NULL;
    size_t* recv_sizes_ = NULL;
    size_t* d_ibv_atomic_ = NULL;
    size_t* d_ibv_atomic_recv_ = NULL;

    // Command pointers
    size_t* h_recv_cmd_ptr_;

    size_t last_recv_cmd_ = 1;

    // Ibv Memory regions
    struct ibv_mr** input_mr_ = NULL;
    struct ibv_mr** output_mr_ = NULL;
    struct ibv_mr* in_rem_output_mr_ = NULL;
    struct ibv_mr* rem_output_mr_ = NULL;
    struct ibv_mr* my_atomic_mr_ = NULL;
    struct ibv_mr* my_atomic_recv_mr_ = NULL;
    struct ibv_mr* rem_atomic_mr_ = NULL;

    IbvProxy* proxy_ctx_ = NULL;
    HierA2AIbvContext* ibv_ctx_ = NULL;
    size_t num_procs_ = 1;
    size_t my_proc_ = 0;

    int num_expected_send_completions_ = 0;
    int num_send_completions_ = 0;
    int num_expected_atomic_completions_ = 0;
    int num_atomic_completions_ = 0;

    bool skip_barrier_ = false;

    HierA2ACollContext(IbvProxy* proxy_ctx, HierA2AIbvContext* ibv_ctx, CollSyncHelper* sync_helper,
                       bool skip_barrier);
    void init_buf(const M2PHierA2ABufInit& in, P2MHierA2ABufInit& out);

    bool check_recv() const;
    bool check_send() const;
    void process_recv();
    void process_send();
    bool wait_send_completion();
    void stm();
  };

  struct HierA2AvCollContext {
    enum State { BUF_INIT_PENDING, WAIT_RECV_CMD, WAIT_COMPLETION };

    State state_ = BUF_INIT_PENDING;
    CollSyncHelper* sync_helper_;

    size_t h_max_send_size_per_dest_ = 0;
    size_t h_max_recv_size_per_dest_ = 0;
    size_t* send_sizes_ = NULL;
    size_t* recv_sizes_ = NULL;
    size_t* d_ibv_atomic_ = NULL;
    size_t* d_ibv_atomic_recv_ = NULL;

    // Command pointers
    size_t* h_recv_cmd_ptr_;

    size_t last_recv_cmd_ = 1;

    // Ibv WRs
    struct ibv_send_wr** wr_ = NULL;
    struct ibv_send_wr* atomic_wr_ = NULL;

    // Ibv Memory regions
    struct ibv_mr* input_mr_ = NULL;
    struct ibv_mr* output_mr_ = NULL;
    struct ibv_mr* rem_output_mr_ = NULL;
    struct ibv_mr* my_atomic_mr_ = NULL;
    struct ibv_mr* my_atomic_recv_mr_ = NULL;
    struct ibv_mr* rem_atomic_mr_ = NULL;

    IbvProxy* proxy_ctx_ = NULL;
    HierA2AIbvContext* ibv_ctx_ = NULL;
    size_t num_procs_ = 1;
    size_t my_proc_ = 0;
    size_t num_gpus_ = 1;
    size_t num_buffers_ = 1;

    int num_expected_send_completions_ = 0;
    int num_send_completions_ = 0;
    int num_expected_atomic_completions_ = 0;
    int num_atomic_completions_ = 0;

    bool skip_barrier_ = false;

    HierA2AvCollContext(IbvProxy* proxy_ctx, HierA2AIbvContext* ibv_ctx,
                        CollSyncHelper* sync_helper, size_t num_buffers, bool skip_barrier);
    ~HierA2AvCollContext();
    void init_buf(const M2PHierA2AvBufInit& in, P2MHierA2AvBufInit& out);

    bool check_recv() const;
    bool check_send() const;
    void process_recv();
    void process_send();
    bool wait_send_completion();
    void stm();
  };

  struct SharpContext {
    struct sharp_coll_context* sharp_coll_context_;
    struct sharp_coll_comm* sharp_coll_comm_;
    SharpContext(const IbvProxy::InitConfig& cfg);
    ~SharpContext();
  };

  struct ARCollContext {
    enum State { BUF_INIT_PENDING, PROCESS_SHARP };

    State state_ = BUF_INIT_PENDING;

    IbvProxy* proxy_ctx_;
    SharpContext* sharp_ctx_;

    void* handle[MAX_SHARP_BLOCKS];
    size_t* h_rs_cmd_ = NULL;
    size_t* h_gdr_ag_cmd_ = NULL;
    size_t* d_ag_cmd_ = NULL;
    size_t* d_ag_storage_ = NULL;

    gdr_t gdr_ = NULL;
    gdr_mh_t gdr_mh_;

    // For PCIE flush
    struct ibv_context* context_ = NULL;
    struct ibv_pd* pd_ = NULL;
    struct ibv_cq* cq_ = NULL;
    struct ibv_qp* qp_ = NULL;
    struct ibv_mr* h_flush_mr_ = NULL;
    struct ibv_mr* d_flush_atomic_mr_ = NULL;
    struct ibv_send_wr wr_;
    struct ibv_sge sge_;
    IbQpInfo qp_info_;

    sharp_datatype sharp_dtype_;
    struct sharp_coll_reduce_spec reduce_spec_;
    void* mem_mr = NULL;

    int cfg_nblocks_ = AR_NBLOCKS;
    int cfg_align_block_ = AR_ALIGN_BLOCK;
    int cfg_min_block_ = AR_MIN_BLOCK;

    void* d_ar_ptr_ = NULL;
    size_t num_gpus_ = 0;
    size_t proxy_id_ = 0;
    size_t ar_size_ = 0;
    int blocksize_ = 0;
    int peer_block_ = 0;
    size_t next_offset_ = 0;
    size_t next_length_ = 0;
    int elem_size_ = 2;
    int nblock_ = 0;
    int num_blocks_ = 0;
    size_t sharp_req_counter_ = 0;
    size_t sharp_cmpl_counter_ = 0;

    ARCollContext(IbvProxy* proxy_ctx, SharpContext* sharp_ctx);
    ~ARCollContext();
    void update_size(const size_t ar_size);
#ifdef AR_DISABLE_PCIE_FLUSH
    void init_gdr();
#else
    void init_pcie_flush();
    void do_pcie_flush();
#endif
    void init_buf(const M2PARBufInit& in, P2MARBufInit& out);
    void process_sharp_completions();
    void process_new_command();
    void stm();
  };

  IbvProxy(const InitConfig* cfg);
  IbvProxy(const IbvProxy&) = delete;
  IbvProxy(IbvProxy&&) = default;

  int destroy_ = 0;
  IbvProxyState state_ = INIT;
  int active_ctx_ = 0;  // TODO: support for multiple active contexts
  InitConfig cfg_;
  std::unique_ptr<HierA2AIbvContext> hier_a2a_ibv_ctx_ = NULL;
  std::vector<std::unique_ptr<HierA2ACollContext>> hier_a2a_coll_ctx_;
  std::vector<std::unique_ptr<HierA2AvCollContext>> hier_a2a_v_coll_ctx_;

  std::unique_ptr<SharpContext> sharp_ctx_ = NULL;
  std::vector<std::unique_ptr<ARCollContext>> ar_coll_ctx_;

  // Proxy command  handles
  void exec_proxy_cmd(const M2PHierA2ABufInit& in, P2MHierA2ABufInit& out);
  void exec_proxy_cmd(const M2PHierA2ACollInit& in, const P2MNull& __unused);
  void exec_proxy_cmd(const M2PHierA2AvBufInit& in, P2MHierA2AvBufInit& out);
  void exec_proxy_cmd(const M2PHierA2AvCollInit& in, const P2MNull& __unused);
  void exec_proxy_cmd(const M2PARCollInit& in, const P2MNull& __unused);
  void exec_proxy_cmd(const M2PARBufInit& in, P2MARBufInit& out);
  void exec_proxy_cmd(const M2PARUpdateSize& in, const P2MNull& __unused);
  void exec_proxy_cmd(const M2PStateTransition& in, P2MNull& __unused);

  void stm_init();
  void stm();
};

struct ProxyCommandVisitor : public boost::static_visitor<void> {
 public:
  void operator()(HierA2ABufInitCmd& cmd) const {
    proxy_->exec_proxy_cmd(std::get<0>(cmd), std::get<1>(cmd));
  }
  void operator()(HierA2ACollInitCmd& cmd) const {
    proxy_->exec_proxy_cmd(std::get<0>(cmd), std::get<1>(cmd));
  }
  void operator()(HierA2AvBufInitCmd& cmd) const {
    proxy_->exec_proxy_cmd(std::get<0>(cmd), std::get<1>(cmd));
  }
  void operator()(HierA2AvCollInitCmd& cmd) const {
    proxy_->exec_proxy_cmd(std::get<0>(cmd), std::get<1>(cmd));
  }
  void operator()(ARBufInitCmd& cmd) const {
    proxy_->exec_proxy_cmd(std::get<0>(cmd), std::get<1>(cmd));
  }
  void operator()(ARCollInitCmd& cmd) const {
    proxy_->exec_proxy_cmd(std::get<0>(cmd), std::get<1>(cmd));
  }
  void operator()(ARUpdateSizeCmd& cmd) const {
    proxy_->exec_proxy_cmd(std::get<0>(cmd), std::get<1>(cmd));
  }
  void operator()(ProxyStateTransitionCmd& cmd) const {
    proxy_->exec_proxy_cmd(std::get<0>(cmd), std::get<1>(cmd));
  }
  void operator()(boost::blank __unused) const {
    HCTR_LOG_S(ERROR, WORLD) << "Invalid proxy command " << HCTR_LOCATION() << std::endl;
    exit(1);
  }
  ProxyCommandVisitor(IbvProxy* proxy) { proxy_ = proxy; };
  IbvProxy* proxy_;
};
}  // namespace HugeCTR

#endif
