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

#ifdef ENABLE_MPI
#include <collectives/ib_proxy.hpp>
#include <immintrin.h>
#include <x86intrin.h>

namespace HugeCTR 
{
  // ProxyCommand
  ProxyCommand::ProxyCommand(size_t num_threads)
  {
    num_threads_ = num_threads;
    last_cmd_.resize(num_threads, 0);
    cmd_.resize(num_threads, boost::blank());
  }

  void ProxyCommand::post_command()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cmd_cntr_++; 
    cmpl_cntr_ = 0;
    cond_.notify_all();
  }

  void ProxyCommand::wait_new_command(size_t thread_id)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [&, this]() { return ((last_cmd_[thread_id] != cmd_cntr_) &&
          (cmpl_cntr_ == thread_id)); });
  }

  void ProxyCommand::post_completion(size_t thread_id)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    last_cmd_[thread_id] = cmd_cntr_;
    cmpl_cntr_++;
    cond_.notify_all();
  }

  // void ProxyCommand::wait_for_completion(std::unique_lock<std::mutex>& lock)
  void ProxyCommand::wait_for_completion()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [&, this]() { return (cmpl_cntr_ == num_threads_); });
  }

  void ProxyCommand::reset()
  {
    for (auto& c : cmd_) { c = boost::blank(); }
  }

  static int oob_bcast(void *comm_context, void *buf, int size, int root) {
    CK_MPI_THROW_(MPI_Bcast(buf, size, MPI_BYTE, root, MPI_COMM_WORLD));
    return 0;
  }

  static int oob_barrier(void *comm_context) {
    CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
    return 0;
  }

  static int oob_gather(void *comm_context, int root, void *sbuf, void *rbuf, int len) {
    CK_MPI_THROW_(MPI_Gather(sbuf, len, MPI_BYTE, rbuf, len, MPI_BYTE, root, MPI_COMM_WORLD));
    return 0;
  }

  void IbvProxy::HierA2AIbvContext::init_ibv(const IbvProxy::InitConfig& cfg)
  {
    size_t num_procs = cfg.num_procs_;
    num_procs_ = num_procs;

    int num_devices = 0;
    struct ibv_device** devices = ibv_get_device_list(&num_devices);
    PROXY_ASSERT_MSG(devices, "Can't get ib device list");

    // find ibv device that matches with the current device name and open
    for (int d = 0; d < num_devices; d++) {
      const char* dev_name = ibv_get_device_name(devices[d]);
      PROXY_ASSERT_MSG(dev_name, "Unable to get device name");
      if (cfg.ib_dev_ == std::string(dev_name)) {
        context_ = ibv_open_device(devices[d]);
        PROXY_ASSERT_MSG(context_, "Unable to open device");
        break;
      }
    }
    ibv_free_device_list(devices);

    // Allocate PD
    pd_ = ibv_alloc_pd(context_);
    if (!pd_) {
      ERROR_MESSAGE_("Unable to alloc protection domain for dev " + cfg.ib_dev_);
    }
    size_t num_gpus = cfg.num_gpus_;
    cq_ = (struct ibv_cq**) malloc(num_procs * sizeof(struct ibv_cq*));
    qp_ = (struct ibv_qp**) malloc(num_procs * sizeof(struct ibv_qp*));

    qp_infos_ = (struct IbQpInfo*) malloc(num_procs * sizeof(struct IbQpInfo));
    rem_qp_infos_ = (struct IbQpInfo*) malloc(num_procs * sizeof(struct IbQpInfo));

    // Create completion queue
    for (size_t n = 0; n < num_procs; n++) {
      cq_[n] = ibv_create_cq(context_, 2*num_gpus/*recv + send*/, NULL, NULL, 0); 
      if (!cq_[n]) {
        ERROR_MESSAGE_("Unable to create completion queue");
      }

      struct ibv_qp_init_attr qp_init_attr;
      memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
      qp_init_attr.send_cq = cq_[n];
      qp_init_attr.recv_cq = cq_[n];
      qp_init_attr.qp_type = IBV_QPT_RC;
      qp_init_attr.cap.max_send_wr = num_gpus*4;
      qp_init_attr.cap.max_recv_wr = 1;
      qp_init_attr.cap.max_send_sge = 1;
      qp_init_attr.cap.max_recv_sge = 1;
      qp_init_attr.cap.max_inline_data = 0;

      // Create QP
      qp_[n] = ibv_create_qp(pd_, &qp_init_attr);
      if (qp_[n] == NULL) {
        ERROR_MESSAGE_("Unable to create qp");
      }

      // QP state machine
      struct ibv_qp_attr qp_attr;
      memset(&qp_attr, 0, sizeof(ibv_qp_attr));
      qp_attr.qp_state = IBV_QPS_INIT;
      qp_attr.pkey_index = 0;
      qp_attr.port_num = cfg.ib_port_;
      qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
      if (ibv_modify_qp(qp_[n], &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
        ERROR_MESSAGE_("Unable to modify QP access attributes");
      }

      struct ibv_port_attr port_attr;
      if (ibv_query_port(context_, cfg.ib_port_, &port_attr) != 0) {
        std::cerr << "Unable to query port for port info" << std::endl;
      }

      qp_infos_[n].ib_port = cfg.ib_port_;
      qp_infos_[n].lid = port_attr.lid;
      qp_infos_[n].qpn = qp_[n]->qp_num;
      qp_infos_[n].mtu = port_attr.active_mtu;
    }

    CK_MPI_THROW_(MPI_Alltoall((void*)qp_infos_, sizeof(IbQpInfo), MPI_BYTE, 
          (void*)rem_qp_infos_, sizeof(IbQpInfo), MPI_BYTE, MPI_COMM_WORLD)); 

    // TODO: Align buffer allocation to 4KB

    for (size_t n = 0; n < cfg.num_procs_; n++) {
      // connect my QPs with corresponding node QPs.
      if (n == cfg.my_proc_) continue; // Ignore self

      // Move to RTR state
      {
        // Modify MTU to min across both
        rem_qp_infos_[n].mtu = (enum ibv_mtu)std::min(rem_qp_infos_[n].mtu, qp_infos_[n].mtu);
        struct ibv_qp_attr qp_attr;
        memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
        qp_attr.qp_state = IBV_QPS_RTR;
        qp_attr.path_mtu = rem_qp_infos_[n].mtu;
        qp_attr.dest_qp_num = rem_qp_infos_[n].qpn;
        qp_attr.rq_psn = 0;
        qp_attr.max_dest_rd_atomic = 1;
        qp_attr.min_rnr_timer = 12;
        qp_attr.ah_attr.is_global = 0;
        qp_attr.ah_attr.dlid = rem_qp_infos_[n].lid;
        qp_attr.ah_attr.sl = 0;
        qp_attr.ah_attr.src_path_bits = 0;
        qp_attr.ah_attr.port_num = rem_qp_infos_[n].ib_port;
        if (ibv_modify_qp(qp_[n], &qp_attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
          ERROR_MESSAGE_("Modify QP failed");
          exit(1);
        }
      }

      // Move to RTS state
      {
        struct ibv_qp_attr qp_attr;
        memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
        qp_attr.qp_state = IBV_QPS_RTS;
        qp_attr.timeout = 14;
        qp_attr.retry_cnt = 7;
        qp_attr.rnr_retry = 7;
        qp_attr.sq_psn = 0;
        qp_attr.max_rd_atomic = 1;
        if(ibv_modify_qp(qp_[n], &qp_attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
          ERROR_MESSAGE_("Modify QP failed RTS state");
          exit(1);
        }
      }
    }
  }

  void IbvProxy::HierA2AIbvContext::finalize_ibv()
  {
    // Destroy QPs
    for (size_t n = 0; n < num_procs_; n++) {
      if (cq_ && cq_[n]) { ibv_destroy_cq(cq_[n]); }
      if (qp_ && qp_[n]) { ibv_destroy_qp(qp_[n]); }
    }
    if (cq_) { free(cq_); }
    if (qp_) { free(qp_); }
    if (qp_infos_) { free(qp_infos_); }
    if (rem_qp_infos_) { free(rem_qp_infos_); }

    // Destroy PD
    if (pd_) { ibv_dealloc_pd(pd_); }
    if (context_) { PROXY_ASSERT(ibv_close_device(context_) == 0); }
  }

#ifdef SHARP_A2A
  void IbvProxy::HierA2AIbvContext::init_sharp(const IbvProxy::InitConfig& cfg)
  {
    if (cfg.proxy_id_ == 0) {

      MESSAGE_("using SHARP for A2A");

      struct sharp_coll_comm_init_spec comm_spec;
      struct sharp_coll_init_spec init_spec = {0};
      size_t num_procs = cfg.num_procs_;
      size_t my_proc = cfg.my_proc_;

      init_spec.progress_func  = NULL;
      init_spec.job_id = (gethostid() << 32);
      CK_MPI_THROW_(MPI_Bcast(&(init_spec.job_id), 1, MPI_LONG, 0, MPI_COMM_WORLD));
      init_spec.world_rank = my_proc;
      init_spec.world_size = num_procs;
      init_spec.world_local_rank = 0;
      init_spec.enable_thread_support = 0;
      init_spec.oob_colls.barrier = oob_barrier;
      init_spec.oob_colls.bcast = oob_bcast;
      init_spec.oob_colls.gather = oob_gather;
      init_spec.oob_ctx = (void*)this;
      init_spec.config = sharp_coll_default_config;
      std::string sharp_dev = cfg.ib_dev_ + std::string(":") + std::to_string(cfg.ib_port_);
      init_spec.config.ib_dev_list = sharp_dev.c_str();
      int ret = sharp_coll_init(&init_spec, &(sharp_coll_context_));
      PROXY_ASSERT(ret == SHARP_COLL_SUCCESS);

      comm_spec.rank = my_proc;
      comm_spec.size = num_procs;
      comm_spec.oob_ctx = (void*)this;
      comm_spec.group_world_ranks = NULL;
      ret = sharp_coll_comm_init(sharp_coll_context_, &comm_spec, &sharp_coll_comm_);
      PROXY_ASSERT(ret == SHARP_COLL_SUCCESS);
    }
  }

  void IbvProxy::HierA2AIbvContext::finalize_sharp()
  {
    if (proxy_id_ == 0) {
      if (sharp_coll_comm_) { sharp_coll_comm_destroy(sharp_coll_comm_); }
      if (sharp_coll_context_) { sharp_coll_finalize(sharp_coll_context_); }
    }
  }
#endif

  IbvProxy::HierA2AIbvContext::HierA2AIbvContext(const IbvProxy::InitConfig& cfg):
    proxy_id_(cfg.proxy_id_)
  {
    init_ibv(cfg);
#ifdef SHARP_A2A
    init_sharp(cfg);
#endif
  }

  IbvProxy::HierA2AIbvContext::~HierA2AIbvContext()
  {
    finalize_ibv();
#ifdef SHARP_A2A
    finalize_sharp();
#endif
  }

  IbvProxy::HierA2ACollContext::HierA2ACollContext(IbvProxy* _proxy_ctx,
      HierA2AIbvContext* _ibv_ctx,
      CollSyncHelper* _sync_helper,
      bool skip_barrier) :
    sync_helper_(_sync_helper),
    proxy_ctx_(_proxy_ctx),
    ibv_ctx_(_ibv_ctx),
    skip_barrier_(skip_barrier)
  {
    num_procs_ = proxy_ctx_->cfg_.num_procs_;
    my_proc_ = proxy_ctx_->cfg_.my_proc_;
  }

  void IbvProxy::HierA2ACollContext::init_buf(
      const M2PHierA2ABufInit& in, 
      P2MHierA2ABufInit& out)
  {
    /* register MR and update the size pointers for main thread access */
    PROXY_ASSERT(state_ == BUF_INIT_PENDING);

    const InitConfig& cfg = proxy_ctx_->cfg_;
    auto ctx = ibv_ctx_;
    auto num_procs = cfg.num_procs_;
    auto my_proc = cfg.my_proc_;

    // Allocates in local numa node
    send_ptrs_  = (void** )malloc(sizeof(void*  )*num_procs);
    recv_ptrs_  = (void** )malloc(sizeof(void*  )*num_procs);

    CK_CUDA_THROW_(cudaMallocHost((void**)&send_sizes_, sizeof(size_t)*num_procs));
    CK_CUDA_THROW_(cudaMallocHost((void**)&recv_sizes_, sizeof(size_t)*num_procs));

    memcpy(send_ptrs_, in.d_send_ptrs_, sizeof(void*)*num_procs);
    memcpy(recv_ptrs_, in.d_recv_ptrs_, sizeof(void*)*num_procs);
    memcpy(send_sizes_, in.h_max_send_size_, sizeof(size_t)*num_procs);
    memcpy(recv_sizes_, in.h_max_recv_size_, sizeof(size_t)*num_procs);

    out.h_send_size_ = send_sizes_;
    out.h_recv_size_ = recv_sizes_;

    // Copy command pointers
    h_recv_cmd_ptr_ = in.h_recv_cmd_ptr_;

    // Allocate MR structs
    input_mr_ = (ibv_mr**)malloc(sizeof(ibv_mr*)*num_procs);
    output_mr_ = (ibv_mr**)malloc(sizeof(ibv_mr*)*num_procs);
    in_rem_output_mr_ = (ibv_mr*)malloc(num_procs*sizeof(struct ibv_mr));
    rem_output_mr_ = (ibv_mr*)malloc(num_procs*sizeof(struct ibv_mr));

    for (size_t n = 0; n < num_procs; n++) {
      if (in.d_send_ptrs_[n]) {
        input_mr_[n] = ibv_reg_mr(ctx->pd_, in.d_send_ptrs_[n], 
            in.h_max_send_size_[n], 
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);

        if (!input_mr_[n]) {
          ERROR_MESSAGE_("Input reg mr failed");
          exit(1);
        }
      }

      if (in.d_recv_ptrs_[n]) {
        output_mr_[n] = ibv_reg_mr(ctx->pd_, in.d_recv_ptrs_[n],
            in.h_max_recv_size_[n],
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);

        if (!output_mr_[n]) {
          ERROR_MESSAGE_("Output reg mr failed");
          exit(1);
        }
        memcpy(&in_rem_output_mr_[n], output_mr_[n], sizeof(struct ibv_mr));
      }
    }

    // Get remote output MRs for RDMA write
    CK_MPI_THROW_(MPI_Alltoall(
          (void*)in_rem_output_mr_, sizeof(struct ibv_mr), MPI_BYTE,
          (void*)rem_output_mr_,    sizeof(struct ibv_mr), MPI_BYTE,
          MPI_COMM_WORLD));

    // Allocate atomics
    d_ibv_atomic_ = in.d_ibv_atomic_;
    d_ibv_atomic_recv_ = in.d_ibv_atomic_recv_;
    
    // Register atomics
    my_atomic_mr_ = ibv_reg_mr(ctx->pd_, d_ibv_atomic_, 
        MAX_IBV_DEST*sizeof(size_t), 
        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | 
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
    PROXY_ASSERT(my_atomic_mr_ != NULL);

    my_atomic_recv_mr_ = ibv_reg_mr(ctx->pd_, d_ibv_atomic_recv_, 
        MAX_IBV_DEST*sizeof(size_t), 
        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | 
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
    PROXY_ASSERT(my_atomic_recv_mr_ != NULL);

    rem_atomic_mr_ = (struct ibv_mr*)malloc(num_procs * sizeof(struct ibv_mr));
    memcpy(&rem_atomic_mr_[my_proc], my_atomic_mr_, sizeof(struct ibv_mr));

    CK_MPI_THROW_(MPI_Allgather((void*)&rem_atomic_mr_[my_proc], sizeof(struct ibv_mr), 
          MPI_BYTE, (void*)rem_atomic_mr_, sizeof(struct ibv_mr), MPI_BYTE, MPI_COMM_WORLD));

    // Set expected completions
    for (size_t n = 0; n < num_procs; n++) {

      if (my_proc == n) continue;
      if (send_sizes_[n] > 0) {
        num_expected_send_completions_++;
      }
      num_expected_atomic_completions_++;
    }

    state_ = WAIT_RECV_CMD;
  }

  inline bool IbvProxy::HierA2ACollContext::check_recv() const
  {
    return (*(volatile size_t*)(h_recv_cmd_ptr_) > last_recv_cmd_);
  }

  void IbvProxy::HierA2ACollContext::process_recv()
  {
    auto cfg = proxy_ctx_->cfg_;
    if (cfg.proxy_id_ == 0) {
      if (!skip_barrier_) {
#ifdef SHARP_A2A
        int ret = sharp_coll_do_barrier(ibv_ctx_->sharp_coll_comm_);
        PROXY_ASSERT(ret == SHARP_COLL_SUCCESS);
#else
        CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
#endif
      }
      sync_helper_->recv_bcast();
    }
    else {
      sync_helper_->wait_recv_bcast(last_recv_cmd_ + 1);
    }
  }

  void IbvProxy::HierA2ACollContext::process_send()
  {
    for (size_t i = 1; i < num_procs_; i++) {
      int n = (my_proc_ + i) % num_procs_;
      if (send_sizes_[n] > 0) {

        struct ibv_send_wr wr;
        struct ibv_sge sge;
        memset(&wr, 0, sizeof(wr));
        wr.wr_id = n;
        sge.addr = (uintptr_t)(send_ptrs_[n]);
        sge.length = send_sizes_[n];
        sge.lkey = input_mr_[n]->lkey;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED; // No need for a completion

        wr.wr.rdma.remote_addr = (uintptr_t)((void*)rem_output_mr_[n].addr);
        wr.wr.rdma.rkey = rem_output_mr_[n].rkey;

        struct ibv_send_wr atomic_wr;
        memset(&atomic_wr, 0, sizeof(atomic_wr));
        wr.next = &atomic_wr;
        atomic_wr.wr_id = n;
        struct ibv_sge atomic_sge;
        atomic_sge.addr = (uintptr_t)((size_t*)d_ibv_atomic_recv_ + n);
        atomic_sge.length = sizeof(size_t);
        atomic_sge.lkey = my_atomic_recv_mr_->lkey;
        atomic_wr.sg_list = &atomic_sge;
        atomic_wr.num_sge = 1;
        atomic_wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
        atomic_wr.send_flags = IBV_SEND_SIGNALED;
        atomic_wr.wr.atomic.remote_addr = (uintptr_t)(((size_t*)rem_atomic_mr_[n].addr) + my_proc_);
        atomic_wr.wr.atomic.compare_add = 1;
        atomic_wr.wr.atomic.rkey = rem_atomic_mr_[n].rkey;

        __sync_synchronize();
        struct ibv_send_wr* bad_wr;
        // const std::string nvtx_name = "post_send_start_" + std::to_string(cfg_.device_id);
        // nvtxRangePushA(nvtx_name.c_str()); 
        int ret = ibv_post_send(ibv_ctx_->qp_[n], &wr, &bad_wr);
        PROXY_ASSERT(ret == 0);
      }
      else {
        // Do the atomic
        struct ibv_send_wr atomic_wr;
        memset(&atomic_wr, 0, sizeof(atomic_wr));
        atomic_wr.wr_id = n;
        struct ibv_sge atomic_sge;
        atomic_sge.addr = (uintptr_t)((size_t*)d_ibv_atomic_recv_ + n);
        atomic_sge.length = sizeof(size_t);
        atomic_sge.lkey = my_atomic_recv_mr_->lkey;
        atomic_wr.sg_list = &atomic_sge;
        atomic_wr.num_sge = 1;
        atomic_wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
        atomic_wr.send_flags = IBV_SEND_SIGNALED;
        atomic_wr.wr.atomic.remote_addr = (uintptr_t)(((size_t*)rem_atomic_mr_[n].addr) + my_proc_);
        atomic_wr.wr.atomic.compare_add = 1;
        atomic_wr.wr.atomic.rkey = rem_atomic_mr_[n].rkey;

        __sync_synchronize();
        struct ibv_send_wr* bad_wr;
        // const std::string nvtx_name = "post_send_start_" + std::to_string(cfg_.device_id);
        // nvtxRangePushA(nvtx_name.c_str()); 
        int ret = ibv_post_send(ibv_ctx_->qp_[n], &atomic_wr, &bad_wr);
        PROXY_ASSERT(ret == 0);
      }

      PROXY_ASSERT_MSG((*(h_recv_cmd_ptr_) - last_recv_cmd_) <= 1, "Can't have multiple sends inflight");
      last_recv_cmd_++;
    }
  }

  bool IbvProxy::HierA2ACollContext::wait_send_completion()
  {
    for (size_t n = 0; n < num_procs_; n++) {
      if (n == my_proc_) continue;
      struct ibv_wc wcs[4];
      int done = ibv_poll_cq(ibv_ctx_->cq_[n], 4, wcs);
      PROXY_ASSERT(done >= 0);
      for (int d = 0; d < done; d++)
      {
        struct ibv_wc *wc = wcs + d;
        if (wc->opcode == IBV_WC_RDMA_WRITE) {
          num_send_completions_++;
        }
        else if (wc->opcode == IBV_WC_FETCH_ADD) {
          num_atomic_completions_++;
        }
        else {
          std::cerr << proxy_ctx_->cfg_.device_id_ << " " << my_proc_ << 
            " Unknown completion received " << wc->opcode << 
            " status: " << wc->status << std::endl;
          exit(1);
        }
      }
    }
    if ((num_send_completions_ == num_expected_send_completions_) &&
        (num_atomic_completions_ == num_expected_atomic_completions_)) {
      num_send_completions_ = 0;
      num_atomic_completions_ = 0;
      // nvtxRangePop();
      return false;
    }
    return true;
  }

  void IbvProxy::HierA2ACollContext::stm()
  {
    switch (state_)
    {
      case BUF_INIT_PENDING:
      {
        ERROR_MESSAGE_("No buffers are registered for the collective");
        exit(1);
        break;
      }
      case WAIT_RECV_CMD:
      {
        if (check_recv()) {
          process_recv();
          process_send();
          state_ = WAIT_COMPLETION;
        }
        break;
      }
      case WAIT_COMPLETION:
      {
        if (wait_send_completion()) {
          state_ = WAIT_COMPLETION;
        }
        else {
          state_ = WAIT_RECV_CMD;
        }
        break;
      }
    }
  }

  IbvProxy::HierA2AvCollContext::HierA2AvCollContext(IbvProxy* _proxy_ctx,
      HierA2AIbvContext* _ibv_ctx, CollSyncHelper* _sync_helper, bool skip_barrier) :
    sync_helper_(_sync_helper),
    proxy_ctx_(_proxy_ctx),
    ibv_ctx_(_ibv_ctx),
    skip_barrier_(skip_barrier)
  {
    num_procs_ = proxy_ctx_->cfg_.num_procs_;
    my_proc_ = proxy_ctx_->cfg_.my_proc_;
    num_gpus_ = proxy_ctx_->cfg_.num_gpus_;
  }

  void IbvProxy::HierA2AvCollContext::init_buf(
      const M2PHierA2AvBufInit& in,
      P2MHierA2AvBufInit& out)
  {
    /* register MR and update the size pointers for main thread access */
    PROXY_ASSERT(state_ == BUF_INIT_PENDING);

    // const InitConfig& cfg = proxy_ctx_->cfg_;
    auto ctx = ibv_ctx_;

    // pre-construct work requests in memory
    wr_ = (ibv_send_wr**)malloc(sizeof(ibv_send_wr*)*num_procs_);
    for (size_t n = 0; n < num_procs_; n++) {
      wr_[n] = (ibv_send_wr*)malloc(sizeof(ibv_send_wr)*num_gpus_);
      memset(wr_[n], 0, sizeof(ibv_send_wr)*num_gpus_);
      for (size_t g = 1; g < num_gpus_; g++) {
        wr_[n][g-1].next = &wr_[n][g];
      }
    }

    CK_CUDA_THROW_(cudaMallocHost((void**)&send_sizes_, sizeof(size_t)*num_procs_*num_gpus_));
    CK_CUDA_THROW_(cudaMallocHost((void**)&recv_sizes_, sizeof(size_t)*num_procs_*num_gpus_));

    PROXY_ASSERT(in.h_max_send_size_ == in.h_max_recv_size_);
    PROXY_ASSERT(in.h_max_send_size_ % (num_procs_*num_gpus_) == 0);
    
    h_max_send_size_per_dest_ = in.h_max_send_size_ / (num_procs_*num_gpus_);
    h_max_recv_size_per_dest_ = in.h_max_recv_size_ / (num_procs_*num_gpus_);

    // Initialize send/recv sizes to max
    for (size_t i = 0; i < num_procs_*num_gpus_; i++) {
      send_sizes_[i] = h_max_send_size_per_dest_;
      recv_sizes_[i] = h_max_recv_size_per_dest_;
    }

    out.h_send_size_ = send_sizes_;
    out.h_recv_size_ = recv_sizes_;

    // Copy command pointers
    h_recv_cmd_ptr_ = in.h_recv_cmd_ptr_;

    // Register MRs in PD
    if (in.d_send_ptrs_) {
      input_mr_ = ibv_reg_mr(ctx->pd_, in.d_send_ptrs_, 
          in.h_max_send_size_,
          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
      PROXY_ASSERT(input_mr_);
    }

    if (in.d_recv_ptrs_) {
      output_mr_ = ibv_reg_mr(ctx->pd_, in.d_recv_ptrs_,
          in.h_max_recv_size_,
          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
      PROXY_ASSERT(output_mr_);
    }

    // set send pointers
    for (size_t n = 0; n < num_procs_; n++) {
      for (size_t g = 0; g < num_gpus_; g++) {
        wr_[n][g].wr_id = n*num_gpus_ + g;
        wr_[n][g].sg_list = (struct ibv_sge*)malloc(sizeof(ibv_sge));
        auto& sge = wr_[n][g].sg_list[0];
        size_t offset = (n*num_gpus_ + g)*h_max_send_size_per_dest_;
        sge.addr = (uintptr_t)((char*)in.d_send_ptrs_ + offset); 
        sge.length = h_max_send_size_per_dest_;
        sge.lkey = input_mr_->lkey;
        wr_[n][g].num_sge = 1;
        wr_[n][g].opcode = IBV_WR_RDMA_WRITE;
        wr_[n][g].send_flags = IBV_SEND_SIGNALED;
      }
    }

    rem_output_mr_ = (ibv_mr*)malloc(num_procs_* sizeof(struct ibv_mr));
    memcpy(&rem_output_mr_[my_proc_], output_mr_, sizeof(struct ibv_mr));
    
    CK_MPI_THROW_(MPI_Allgather(
          (void*)&rem_output_mr_[my_proc_], sizeof(struct ibv_mr), MPI_BYTE, 
          (void*)rem_output_mr_, sizeof(struct ibv_mr), MPI_BYTE,
          MPI_COMM_WORLD));

    // Populate remote MRs
    for (size_t n = 0; n < num_procs_; n++) {
      for (size_t g = 0; g < num_gpus_; g++) {
        size_t offset = (my_proc_*num_gpus_ + g)*h_max_recv_size_per_dest_;
        wr_[n][g].wr.rdma.remote_addr = (uintptr_t)((char*)rem_output_mr_[n].addr + offset);
        wr_[n][g].wr.rdma.rkey = rem_output_mr_[n].rkey;
      }
    }

    // Allocate atomics
    d_ibv_atomic_ = in.d_ibv_atomic_;
    d_ibv_atomic_recv_ = in.d_ibv_atomic_recv_;
    
    // Register atomics
    my_atomic_mr_ = ibv_reg_mr(ctx->pd_, d_ibv_atomic_, 
        MAX_IBV_DEST*sizeof(size_t), 
        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | 
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
    PROXY_ASSERT(my_atomic_mr_ != NULL);

    my_atomic_recv_mr_ = ibv_reg_mr(ctx->pd_, d_ibv_atomic_recv_, 
        MAX_IBV_DEST*sizeof(size_t), 
        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | 
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
    PROXY_ASSERT(my_atomic_recv_mr_ != NULL);

    rem_atomic_mr_ = (struct ibv_mr*)malloc(num_procs_ * sizeof(struct ibv_mr));
    memcpy(&rem_atomic_mr_[my_proc_], my_atomic_mr_, sizeof(struct ibv_mr));

    CK_MPI_THROW_(MPI_Allgather((void*)&rem_atomic_mr_[my_proc_], sizeof(struct ibv_mr), 
          MPI_BYTE, (void*)rem_atomic_mr_, sizeof(struct ibv_mr), MPI_BYTE, MPI_COMM_WORLD));

    // // Populate atomic write requests
    atomic_wr_ = (ibv_send_wr*)(malloc(num_procs_ * sizeof(struct ibv_send_wr)));
    memset(atomic_wr_, 0, num_procs_ * sizeof(struct ibv_send_wr));
    for (size_t n = 0; n < num_procs_; n++) {
      auto& atomic_wr = atomic_wr_[n];
      atomic_wr.wr_id = n;
      atomic_wr.sg_list = (struct ibv_sge*)(malloc(sizeof(struct ibv_sge)));
      auto& sge = atomic_wr.sg_list[0];
      sge.addr = (uintptr_t)((size_t*)d_ibv_atomic_recv_ + n);
      sge.length = sizeof(size_t);
      sge.lkey = my_atomic_recv_mr_->lkey;
      atomic_wr.num_sge = 1;
      atomic_wr.next = NULL;
      atomic_wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
      atomic_wr.send_flags = IBV_SEND_SIGNALED;
      atomic_wr.wr.atomic.remote_addr = (uintptr_t)(((size_t*)rem_atomic_mr_[n].addr) + my_proc_);
      atomic_wr.wr.atomic.compare_add = 1;
      atomic_wr.wr.atomic.rkey = rem_atomic_mr_[n].rkey;
      wr_[n][num_gpus_ - 1].next = &atomic_wr_[n];
    }

    __sync_synchronize();

    // Set expected completions
    for (size_t n = 0; n < num_procs_; n++) {

      if (my_proc_ == n) continue;
      num_expected_send_completions_ += num_gpus_;
      num_expected_atomic_completions_++;
    }
    state_ = WAIT_RECV_CMD;
  }

  inline bool IbvProxy::HierA2AvCollContext::check_recv() const
  {
    return (*(volatile size_t*)(h_recv_cmd_ptr_) > last_recv_cmd_);
  }

  void IbvProxy::HierA2AvCollContext::process_recv()
  {
    auto cfg = proxy_ctx_->cfg_;
    if (cfg.proxy_id_ == 0) {
      if (!skip_barrier_) {
#ifdef SHARP_A2A
        int ret = sharp_coll_do_barrier(ibv_ctx_->sharp_coll_comm_);
        PROXY_ASSERT(ret == SHARP_COLL_SUCCESS);
#else
        CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
#endif
      }
      sync_helper_->recv_bcast();
    }
    else {
      sync_helper_->wait_recv_bcast(last_recv_cmd_ + 1);
    }
  }

  void IbvProxy::HierA2AvCollContext::process_send()
  {
    for (size_t i = 1; i < num_procs_; i++) {
      int n = (my_proc_ + i) % num_procs_;
      for (size_t g = 0; g < num_gpus_; g++) {
        volatile size_t* send_size_ptr = (volatile size_t*)&send_sizes_[n*num_gpus_ + g];
        volatile size_t send_len = *send_size_ptr;
        PROXY_ASSERT(send_len <= h_max_send_size_per_dest_);
        wr_[n][g].sg_list[0].length = send_len;
      }
      __sync_synchronize();
      struct ibv_send_wr* bad_wr;
      int ret = ibv_post_send(ibv_ctx_->qp_[n], &wr_[n][0], &bad_wr);
      PROXY_ASSERT(ret == 0);
    }
    // auto& cfg = proxy_ctx_->cfg_;
    // if (cfg.proxy_id_ == 0) { std::cout << "ibv send called " << cfg.my_proc_ << " " << last_recv_cmd_ << std::endl; }
    PROXY_ASSERT_MSG((*(h_recv_cmd_ptr_) - last_recv_cmd_) <= 2, "Can't have multiple sends inflight");
    last_recv_cmd_++;
  }

  bool IbvProxy::HierA2AvCollContext::wait_send_completion()
  {
    for (size_t n = 0; n < num_procs_; n++) {
      if (n == my_proc_) continue;
      struct ibv_wc wcs[4];
      int done = ibv_poll_cq(ibv_ctx_->cq_[n], 4, wcs);
      PROXY_ASSERT(done >= 0);
      for (int d = 0; d < done; d++)
      {
        struct ibv_wc *wc = wcs + d;
        if (wc->opcode == IBV_WC_RDMA_WRITE) {
          num_send_completions_++;
        }
        else if (wc->opcode == IBV_WC_FETCH_ADD) {
          num_atomic_completions_++;
        }
        else {
          std::cerr << proxy_ctx_->cfg_.device_id_ << " " << my_proc_ << 
            " Unknown completion received " << wc->opcode << 
            " status: " << wc->status << std::endl;
          exit(1);
        }
      }
    }
    if ((num_send_completions_ == num_expected_send_completions_) &&
        (num_atomic_completions_ == num_expected_atomic_completions_)) {
      num_send_completions_ = 0;
      num_atomic_completions_ = 0;
      // nvtxRangePop();
      return false;
    }
    return true;
  }

  void IbvProxy::HierA2AvCollContext::stm()
  {
    switch (state_)
    {
      case BUF_INIT_PENDING:
      {
        ERROR_MESSAGE_("No buffers are registered for the collective");
        exit(1);
        break;
      }
      case WAIT_RECV_CMD:
      {
        if (check_recv()) {
          process_recv();
          process_send();
          state_ = WAIT_COMPLETION;
        }
        break;
      }
      case WAIT_COMPLETION:
      {
        if (wait_send_completion()) {
          state_ = WAIT_COMPLETION;
        }
        else {
          state_ = WAIT_RECV_CMD;
        }
        break;
      }
    }
  }
  
  IbvProxy::HierA2AvCollContext::~HierA2AvCollContext()
  {
    if (atomic_wr_) {
      for (size_t n = 0; n < num_procs_; n++) {
        if (atomic_wr_[n].sg_list) { free(atomic_wr_[n].sg_list); }
      }
      free(atomic_wr_);
    }
    if (rem_atomic_mr_) { free(rem_atomic_mr_); }
    if (my_atomic_recv_mr_) { ibv_dereg_mr(my_atomic_recv_mr_); }
    if (my_atomic_mr_) { ibv_dereg_mr(my_atomic_mr_); }
    if (rem_output_mr_) { free(rem_output_mr_); }
    if (output_mr_) { ibv_dereg_mr(output_mr_); }
    if (input_mr_)  { ibv_dereg_mr(input_mr_);  }
    if (send_sizes_) { cudaFree(send_sizes_); }
    if (recv_sizes_) { cudaFree(recv_sizes_); }
    if (wr_) {
      for (size_t n = 0; n < num_procs_; n++) {
        for (size_t g = 0; g < num_gpus_; g++) { free(wr_[n][g].sg_list); }
        free(wr_[n]);
      }
      free(wr_);
    }
  }
  
  IbvProxy::IbvProxy(const InitConfig* cfg)
  {
    cfg_ = *cfg; // copy
  }

  void IbvProxy::exec_proxy_cmd(const M2PHierA2ACollInit& in, const P2MNull& __unused)
  {
    PROXY_ASSERT(state_ == INIT);
    if (!hier_a2a_ibv_ctx_) {
      hier_a2a_ibv_ctx_ = std::make_unique<HierA2AIbvContext>(cfg_);
    }
    hier_a2a_coll_ctx_.emplace_back(
        std::make_unique<HierA2ACollContext>(this, hier_a2a_ibv_ctx_.get(),
          in.sync_helper_, in.skip_barrier_));
    if ((hier_a2a_coll_ctx_.size() - 1) != in.coll_handle_) {
      ERROR_MESSAGE_("CollHandle mismatch between main and proxy threads");
      exit(1);
    }
  }

  void IbvProxy::exec_proxy_cmd(const M2PHierA2ABufInit& in, P2MHierA2ABufInit& out)
  {
    PROXY_ASSERT(state_ == INIT);
    PROXY_ASSERT_MSG(hier_a2a_ibv_ctx_, "Hier A2A context is not initialized. Register the collective before doing buffer registration");

    auto& coll_ctx = *hier_a2a_coll_ctx_[in.coll_handle_];
    coll_ctx.init_buf(in, out);
  }

  void IbvProxy::exec_proxy_cmd(const M2PStateTransition& in, P2MNull& __unused)
  {
    state_ = in.state_;
  }

  void IbvProxy::exec_proxy_cmd(const M2PHierA2AvCollInit& in, const P2MNull& __unused)
  {
    PROXY_ASSERT(state_ == INIT);
    if (!hier_a2a_ibv_ctx_) {
      hier_a2a_ibv_ctx_ = std::make_unique<HierA2AIbvContext>(cfg_);
    }
    hier_a2a_v_coll_ctx_.emplace_back(
        std::make_unique<HierA2AvCollContext>(this, hier_a2a_ibv_ctx_.get(),
          in.sync_helper_, in.skip_barrier_));
    if ((hier_a2a_v_coll_ctx_.size() - 1) != in.coll_handle_) {
      ERROR_MESSAGE_("CollHandle mismatch between main and proxy threads");
      exit(1);
    }
  }

  void IbvProxy::exec_proxy_cmd(const M2PHierA2AvBufInit& in, P2MHierA2AvBufInit& out)
  {
    PROXY_ASSERT(state_ == INIT);
    PROXY_ASSERT_MSG(hier_a2a_ibv_ctx_, "Hier A2A context is not initialized. Register the collective before doing buffer registration");

    auto& coll_ctx = *hier_a2a_v_coll_ctx_[in.coll_handle_];
    coll_ctx.init_buf(in, out);
  }

  void IbvProxy::stm_init()
  {
    cfg_.proxy_cmd_->wait_new_command(cfg_.proxy_id_);
    auto& cmd = cfg_.proxy_cmd_->cmd_[cfg_.proxy_id_];
    boost::apply_visitor(ProxyCommandVisitor(this), cmd);
    cfg_.proxy_cmd_->post_completion(cfg_.proxy_id_);
  }

  void IbvProxy::stm()
  {
    switch(state_) {
      case INIT:
      {
        stm_init();
        break;
      }
      case READY_TO_TRANSFER:
      {
        if (cfg_.proxy_cmd_->check_destroy()) {
          state_ = DESTROY;
        }

        // TODO: The lookup is O(n) in number of registered coll. Should make it O(1)
        // for (auto& a2a_ctx : hier_a2a_coll_ctx_) {
        //   a2a_ctx->stm();
        // }
        
        // Only one active context at a time
        if (hier_a2a_v_coll_ctx_.size() > 0) {
          auto& ctx = hier_a2a_v_coll_ctx_[active_ctx_];
          ctx->stm();
          if (ctx->state_ == IbvProxy::HierA2AvCollContext::WAIT_RECV_CMD) {
            active_ctx_ = (active_ctx_ + 1) % hier_a2a_v_coll_ctx_.size();
          }
        }
        for (auto& ar_ctx_ : ar_coll_ctx_) { ar_ctx_->stm(); }

        break;
      }
      case DESTROY:
      {
        destroy_ = 1;
        break;
      }
    }
  }

  // AR implementation
  IbvProxy::SharpContext::SharpContext(const IbvProxy::InitConfig& cfg)
  {
    struct sharp_coll_comm_init_spec comm_spec;
    struct sharp_coll_init_spec init_spec = {0};
    size_t num_procs = cfg.num_procs_;;
    size_t my_proc = cfg.my_proc_;

    init_spec.progress_func  = NULL;
    init_spec.job_id = (gethostid() << 32 | (cfg.proxy_id_));
    CK_MPI_THROW_(MPI_Bcast(&(init_spec.job_id), 1, MPI_LONG, 0, MPI_COMM_WORLD));
    init_spec.world_rank = my_proc;
    init_spec.world_size = num_procs;
    init_spec.world_local_rank = 0;
    init_spec.enable_thread_support = 0;
    init_spec.oob_colls.barrier = oob_barrier;
    init_spec.oob_colls.bcast = oob_bcast;
    init_spec.oob_colls.gather = oob_gather;
    init_spec.oob_ctx = (void*)this;
    init_spec.config = sharp_coll_default_config;
    std::string sharp_dev = cfg.ib_dev_ + std::string(":") + std::to_string(cfg.ib_port_);
    init_spec.config.ib_dev_list = sharp_dev.c_str();
    int ret = sharp_coll_init(&init_spec, &(sharp_coll_context_));
    PROXY_ASSERT(ret == SHARP_COLL_SUCCESS);

    comm_spec.rank = my_proc;
    comm_spec.size = num_procs;
    comm_spec.oob_ctx = (void*)this;
    comm_spec.group_world_ranks = NULL;
    ret = sharp_coll_comm_init(sharp_coll_context_, &comm_spec, &sharp_coll_comm_);
    PROXY_ASSERT(ret == SHARP_COLL_SUCCESS);
  }

  IbvProxy::SharpContext::~SharpContext()
  {
    if (sharp_coll_comm_) { sharp_coll_comm_destroy(sharp_coll_comm_); }
    if (sharp_coll_context_) { sharp_coll_finalize(sharp_coll_context_); }
  }

  IbvProxy::ARCollContext::ARCollContext(IbvProxy* proxy_ctx, SharpContext* sharp_ctx):
    proxy_ctx_(proxy_ctx),
    sharp_ctx_(sharp_ctx)
  {
    num_gpus_ = proxy_ctx_->cfg_.num_gpus_;
    proxy_id_ = proxy_ctx_->cfg_.proxy_id_;
  }

#ifdef AR_DISABLE_PCIE_FLUSH
  void IbvProxy::ARCollContext::init_gdr()
  {
    // TODO: check if this needs to be called only once per process
    const int gpu_page_shift = 16;
    const int gpu_page_size = (1UL << gpu_page_shift);

    gdr_ = gdr_open();
    PROXY_ASSERT(gdr_);
    {
      auto ret = gdr_pin_buffer(gdr_, (CUdeviceptr)(d_ag_cmd_), gpu_page_size, 0, 0, &gdr_mh_);
      PROXY_ASSERT(ret == 0);
    }
    {
      auto ret = gdr_map(gdr_, gdr_mh_, (void**)&h_gdr_ag_cmd_, gpu_page_size);
      PROXY_ASSERT(ret == 0);
    }
    *h_gdr_ag_cmd_ = 0;
  }
#else
  void IbvProxy::ARCollContext::init_pcie_flush()
  {
    auto& cfg = proxy_ctx_->cfg_;

    // For PCIe flush
    int num_devices = 0;
    struct ibv_device** devices = ibv_get_device_list(&num_devices);
    PROXY_ASSERT_MSG(devices, "Can't get ib device list");

    // find ibv device that matches with the current device name and open
    for (int d = 0; d < num_devices; d++) {
      const char* dev_name = ibv_get_device_name(devices[d]);
      PROXY_ASSERT_MSG(dev_name, "Unable to get device name");
      if (cfg.ib_dev_ == std::string(dev_name)) {
        context_ = ibv_open_device(devices[d]);
        PROXY_ASSERT_MSG(context_, "Unable to open device");
        break;
      }
    }
    ibv_free_device_list(devices);

    // Allocate PD
    pd_ = ibv_alloc_pd(context_);
    PROXY_ASSERT_MSG(pd_, "Unable to allocate protection domain for dev " + cfg.ib_dev_);

    // Create CQ
    cq_ = ibv_create_cq(context_, AR_MAX_BLOCKS, NULL, NULL, 0);
    PROXY_ASSERT_MSG(cq_, "Unable to create completion queue");
    
    // Allocate null mr on host
    h_flush_mr_ = ibv_alloc_null_mr(pd_);
    PROXY_ASSERT_MSG(h_flush_mr_, "Null MR allocation failed");

    // Allocate flush atomic mr
    const int gpu_page_shift = 16;
    const int gpu_page_size = (1UL << gpu_page_shift);
    d_flush_atomic_mr_ = ibv_reg_mr(pd_, d_ag_cmd_, gpu_page_size, 
          (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC));
    PROXY_ASSERT_MSG(d_flush_atomic_mr_, "Null MR allocation failed");

    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
    qp_init_attr.send_cq = cq_;
    qp_init_attr.recv_cq = cq_;
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.sq_sig_all = 0;
    qp_init_attr.cap.max_send_wr = AR_MAX_BLOCKS;
    qp_init_attr.cap.max_recv_wr = 1;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;
    qp_init_attr.cap.max_inline_data = 0;

    // Create QP
    qp_ = ibv_create_qp(pd_, &qp_init_attr);
    PROXY_ASSERT_MSG(qp_, "Unable to create flush QP");

    // QP state machine
    {
      struct ibv_qp_attr qp_attr;
      memset(&qp_attr, 0, sizeof(ibv_qp_attr));
      qp_attr.qp_state = IBV_QPS_INIT;
      qp_attr.pkey_index = 0;
      qp_attr.port_num = cfg.ib_port_;
      qp_attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
      int ret = ibv_modify_qp(qp_, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
      PROXY_ASSERT_MSG(ret == 0, "Modify QP failed");
    }
    
    // Get QP info
    {
      struct ibv_port_attr port_attr;
      auto ret = ibv_query_port(context_, cfg.ib_port_, &port_attr);
      PROXY_ASSERT_MSG(ret == 0, "Unable to query port for port info");

      qp_info_.ib_port = cfg.ib_port_;
      qp_info_.lid = port_attr.lid;
      qp_info_.qpn = qp_->qp_num;
      qp_info_.mtu = port_attr.active_mtu;
    }

    // Move to RTR state
    {
      struct ibv_qp_attr qp_attr;
      memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
      qp_attr.qp_state = IBV_QPS_RTR;
      qp_attr.path_mtu = qp_info_.mtu;
      qp_attr.dest_qp_num = qp_info_.qpn;
      qp_attr.rq_psn = 0;
      qp_attr.max_dest_rd_atomic = AR_MAX_BLOCKS;
      qp_attr.min_rnr_timer = 12;
      qp_attr.ah_attr.is_global = 0;
      qp_attr.ah_attr.dlid = qp_info_.lid;
      qp_attr.ah_attr.sl = 0;
      qp_attr.ah_attr.src_path_bits = 0;
      qp_attr.ah_attr.port_num = qp_info_.ib_port;

      auto ret = ibv_modify_qp(qp_, &qp_attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
      PROXY_ASSERT_MSG(ret == 0, "Modify QP failed");
    }

    // Move to RTS state
    {
      struct ibv_qp_attr qp_attr;
      memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
      qp_attr.qp_state = IBV_QPS_RTS;
      qp_attr.timeout = 14;
      qp_attr.retry_cnt = 7;
      qp_attr.rnr_retry = 7;
      qp_attr.sq_psn = 0;
      qp_attr.max_rd_atomic = AR_MAX_BLOCKS;
      auto ret = ibv_modify_qp(qp_, &qp_attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
      PROXY_ASSERT_MSG(ret == 0, "Modify QP failed RS state");
    }

    memset(&wr_, 0, sizeof(struct ibv_send_wr));
    sge_.length = sizeof(size_t);
    sge_.lkey = h_flush_mr_->lkey;
    sge_.addr = 0;
    wr_.wr.atomic.remote_addr = (uint64_t)(d_ag_cmd_);
    wr_.wr.atomic.rkey = d_flush_atomic_mr_->rkey;
    wr_.wr.atomic.compare_add = 1;
    wr_.sg_list = &(sge_);
    wr_.num_sge = 1;
    wr_.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  }
#endif

  void IbvProxy::ARCollContext::update_size(const size_t ar_size)
  {
    // calculate numblocks
    // Note: This needs to match with ib comm calculation. 
    
    ar_size_ = ar_size;
    blocksize_ = (cfg_nblocks_ - 1 + (cfg_align_block_ - 1 + ar_size_) / cfg_align_block_) / cfg_nblocks_;
    blocksize_ *= cfg_align_block_;
    if (blocksize_ < cfg_min_block_) { blocksize_ = cfg_min_block_; }
    peer_block_ = blocksize_ / num_gpus_;
    num_blocks_ = (ar_size_ + blocksize_ - 1) / blocksize_;
    peer_block_ = (num_blocks_ > 1) ? (peer_block_) : (ar_size_ / num_gpus_);
    next_length_ = peer_block_;
    next_offset_ = next_length_ * proxy_id_;
  }

  void IbvProxy::ARCollContext::init_buf(const M2PARBufInit& in, P2MARBufInit& out)
  {
    d_ar_ptr_ = in.d_ar_ptr_;
    sharp_dtype_ = in.sharp_dtype_;
    elem_size_ = in.element_size_;

    // register mr
    auto ret = sharp_coll_reg_mr(
        sharp_ctx_->sharp_coll_context_,
        in.d_ar_ptr_,
        in.ar_size_,
        &mem_mr);
    PROXY_ASSERT(ret == SHARP_COLL_SUCCESS);

    // Allocate host flags
    CK_CUDA_THROW_(cudaMallocHost(&h_rs_cmd_, sizeof(size_t)));
    *h_rs_cmd_ = 0;

    // Allocate storage for d_ag_cmd_
    const int gpu_page_shift = 16;
    const int gpu_page_size = (1UL << gpu_page_shift);
    const int gpu_page_offset = (gpu_page_size - 1);
    const int gpu_page_mask = (~gpu_page_offset);

    CK_CUDA_THROW_(cudaSetDevice(proxy_ctx_->cfg_.device_id_));
    CK_CUDA_THROW_(cudaMalloc(&d_ag_storage_, 2*gpu_page_size));
    CK_CUDA_THROW_(cudaMemset(d_ag_storage_, 0, 2*gpu_page_size));
    d_ag_cmd_ = (size_t*)(((CUdeviceptr)d_ag_storage_ + gpu_page_size - 1) & gpu_page_mask);

#ifdef AR_DISABLE_PCIE_FLUSH
    init_gdr();
#else
    init_pcie_flush();
#endif
    out.h_rs_cmd_ = h_rs_cmd_;
    out.d_ag_cmd_ = d_ag_cmd_;
    
    update_size(in.ar_size_);

    // Pre-initialize reduce spec
    reduce_spec_.sbuf_desc.buffer.mem_handle = mem_mr;
    reduce_spec_.sbuf_desc.type = SHARP_DATA_BUFFER;
    reduce_spec_.sbuf_desc.mem_type = SHARP_MEM_TYPE_CUDA;
    reduce_spec_.rbuf_desc.buffer.mem_handle = mem_mr;
    reduce_spec_.rbuf_desc.type = SHARP_DATA_BUFFER;
    reduce_spec_.rbuf_desc.mem_type = SHARP_MEM_TYPE_CUDA;
    reduce_spec_.aggr_mode = SHARP_AGGREGATION_NONE;
    
    reduce_spec_.dtype = sharp_dtype_;
    reduce_spec_.op = SHARP_OP_SUM;
    
    state_ = IbvProxy::ARCollContext::PROCESS_SHARP;
  }

#ifndef AR_DISABLE_PCIE_FLUSH
  void IbvProxy::ARCollContext::do_pcie_flush()
  {
    // Set signaling for last block
    bool signaled = ((sharp_cmpl_counter_ % num_blocks_) == (size_t)(num_blocks_ - 1));
    wr_.wr_id = sharp_cmpl_counter_;
    wr_.send_flags = (signaled) ? IBV_SEND_SIGNALED : 0;
    struct ibv_send_wr* bad_wr;
    __sync_synchronize();
    auto ret = ibv_post_send(qp_, &wr_, &bad_wr);
    PROXY_ASSERT(ret == 0);

    if (signaled) {
      // Wait for completion
      struct ibv_wc wc;
      int res = 0;
      while (res == 0) { res = ibv_poll_cq(cq_, 1, &wc); }
      PROXY_ASSERT_MSG((res > 0) && (wc.status == IBV_WC_SUCCESS), "Poll cq failed with res " + std::to_string(res) + " wc status: " + std::to_string(wc.status));
    }
  }
#endif

  void IbvProxy::ARCollContext::process_sharp_completions()
  {
    if (sharp_req_counter_ > sharp_cmpl_counter_) {
      if (sharp_coll_req_test(handle[sharp_cmpl_counter_ % MAX_SHARP_BLOCKS])) {
#ifdef AR_DISABLE_PCIE_FLUSH
        *h_gdr_ag_cmd_ = (sharp_cmpl_counter_ + 1);
        _mm_mfence();
#else
        do_pcie_flush();     
#endif
        // std::cout << "Sharp completion received" << std::endl;
        sharp_coll_req_free(handle[sharp_cmpl_counter_ % MAX_SHARP_BLOCKS]);
        sharp_cmpl_counter_++;
      }
    }
  }

  void IbvProxy::ARCollContext::process_new_command()
  {
    bool max_inflight_limit = ((sharp_req_counter_ - sharp_cmpl_counter_) < MAX_SHARP_BLOCKS);
    if (max_inflight_limit && (*(volatile size_t*)h_rs_cmd_ > sharp_req_counter_)) {
      reduce_spec_.sbuf_desc.buffer.ptr = (char*)(d_ar_ptr_) + next_offset_;
      reduce_spec_.rbuf_desc.buffer.ptr = (char*)(d_ar_ptr_) + next_offset_;
      reduce_spec_.sbuf_desc.buffer.length = next_length_;
      reduce_spec_.rbuf_desc.buffer.length = next_length_;
      reduce_spec_.length = next_length_ / elem_size_;
      
      int ret = sharp_coll_do_allreduce_nb(sharp_ctx_->sharp_coll_comm_,
          &reduce_spec_, &handle[sharp_req_counter_ % MAX_SHARP_BLOCKS]);
      // std::cout << "Sharp coll do allreduce called" << std::endl;
      PROXY_ASSERT(ret == SHARP_COLL_SUCCESS);
      
      sharp_req_counter_++;
      nblock_ = (nblock_ + 1) % num_blocks_;
      if (nblock_ == (num_blocks_-1)) {
        next_length_ = (ar_size_ - nblock_*blocksize_) / num_gpus_;
      } else {
        next_length_ = peer_block_;
      }
      next_offset_ = nblock_*blocksize_ + next_length_*proxy_id_;
    }
  }

  void IbvProxy::ARCollContext::stm()
  {
    switch (state_)
    {
      case BUF_INIT_PENDING:
      {
        ERROR_MESSAGE_("No buffers are registered for the collective");
        exit(1);
        break;
      }
      case PROCESS_SHARP:
      {
        process_sharp_completions();
        process_new_command();
      }
    }
  }

  IbvProxy::ARCollContext::~ARCollContext()
  {
    const int gpu_page_shift = 16;
    const int gpu_page_size = (1UL << gpu_page_shift);

    if (h_gdr_ag_cmd_) {
      auto ret = gdr_unmap(gdr_, gdr_mh_, (void*)h_gdr_ag_cmd_, gpu_page_size);
      PROXY_ASSERT(ret == 0);
      ret = gdr_unpin_buffer(gdr_, gdr_mh_);
      PROXY_ASSERT(ret == 0);
    }
    if (gdr_) {
      auto ret = gdr_close(gdr_);
      PROXY_ASSERT(ret == 0);
    }
    if (d_ag_storage_) { cudaFree(d_ag_storage_); }
    if (h_rs_cmd_) { cudaFree(h_rs_cmd_); }
    if (mem_mr) { sharp_coll_dereg_mr(sharp_ctx_->sharp_coll_context_, mem_mr); }
  }

  void IbvProxy::exec_proxy_cmd(const M2PARCollInit& in, const P2MNull& __unused)
  {
    PROXY_ASSERT(state_ == INIT);
    if (!sharp_ctx_) {
      sharp_ctx_ = std::make_unique<SharpContext>(cfg_);
    }
    ar_coll_ctx_.emplace_back(std::make_unique<ARCollContext>(this, sharp_ctx_.get()));
    ar_coll_ctx_.back()->cfg_nblocks_ = in.cfg_nblocks_;
    ar_coll_ctx_.back()->cfg_align_block_ = in.cfg_align_block_;
    ar_coll_ctx_.back()->cfg_min_block_ = in.cfg_min_block_;
    PROXY_ASSERT(ar_coll_ctx_.size() - 1 == in.coll_handle_);
  }

  void IbvProxy::exec_proxy_cmd(const M2PARBufInit& in, P2MARBufInit& out)
  {
    PROXY_ASSERT(state_ ==  INIT);
    PROXY_ASSERT_MSG(sharp_ctx_, "SHARP context is not initialized. Register the collective before doing buffer registration");
    auto& coll_ctx = ar_coll_ctx_[in.coll_handle_];
    coll_ctx->init_buf(in, out);
  }

  void IbvProxy::exec_proxy_cmd(const M2PARUpdateSize& in, const P2MNull& __unused)
  {
    PROXY_ASSERT(state_ == INIT);
    auto& coll_ctx = ar_coll_ctx_[in.coll_handle_];
    coll_ctx->update_size(in.ar_size_);
  }
}

#endif
