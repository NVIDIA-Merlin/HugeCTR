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
#include <linux/mempolicy.h>
#include <numaif.h>

#include <collectives/ib_comm.hpp>
#include <iostream>
#include <sstream>
#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {
static void* proxy_thread_func(void* cfg) {
  auto ibv_config = (struct IbvProxy::InitConfig*)cfg;

  // set numa allocation policy to local
  set_mempolicy(MPOL_LOCAL, NULL, 0);

  CudaCPUDeviceContext context(ibv_config->device_id_);

  IbvProxy* proxy = new IbvProxy(ibv_config);
  while (*(volatile int*)&proxy->destroy_ != 1) {
    proxy->stm();
  }
  delete (proxy);
  return NULL;
}

// Helpers
void IbComm::detect_ib_devs() {
  // Init hwloc topology
  hwloc_topology_init(&topo_);
  hwloc_topology_set_io_types_filter(topo_, HWLOC_TYPE_FILTER_KEEP_ALL);
  hwloc_topology_load(topo_);

  ibv_device** dev_list;
  int num_devices;
  dev_list = ibv_get_device_list(&num_devices);

  if ((!dev_list) || (num_devices == 0)) {
    HCTR_LOG_S(ERROR, WORLD) << "Ibv get device list failed: " << num_devices << std::endl;
    exit(-1);
  }

  // Get hwloc devices and final ib devs
  for (int d = 0; d < num_devices; d++) {
    if ((dev_list[d]->node_type != IBV_NODE_RNIC) && (dev_list[d]->node_type != IBV_NODE_CA)) {
      continue;
    }

    const char* dev_name = ibv_get_device_name(dev_list[d]);
    if (!dev_name) {
      HCTR_LOG_S(ERROR, WORLD) << "Unable to get device name" << std::endl;
      exit(-1);
    }

    ibv_context* context;
    context = ibv_open_device(dev_list[d]);
    if (!context) {
      continue;
    }

    struct ibv_device_attr dev_attr;
    memset(&dev_attr, 0, sizeof(dev_attr));
    if (ibv_query_device(context, &dev_attr) != 0) {
      HCTR_LOG_S(ERROR, WORLD) << "Unable to query device " << dev_name << std::endl;
      exit(-1);
    }

    for (int port = 1; port <= dev_attr.phys_port_cnt; port++) {
      struct ibv_port_attr port_attr;
      if (ibv_query_port(context, port, &port_attr) != 0) {
        HCTR_LOG_S(WARNING, WORLD)
            << "Unable to query port " << dev_name << ":" << port << std::endl;
        continue;
      }
      if (port_attr.state != IBV_PORT_ACTIVE) continue;
      if (port_attr.link_layer != IBV_LINK_LAYER_INFINIBAND) continue;

      // TODO: Check against user specified device list.
      ib_dev_list_.emplace_back();
      ib_dev_list_.back().dev_name = dev_name;
      ib_dev_list_.back().dev_port_id = port;
      ib_dev_list_.back().hwloc_obj = hwloc_ibv_get_device_osdev(topo_, dev_list[d]);
      if (!ib_dev_list_.back().hwloc_obj) {
        HCTR_LOG_S(ERROR, WORLD) << "unable to get hwloc obj for ib device " << dev_name
                                 << std::endl;
        exit(1);
      }
    }

    ibv_close_device(context);
  }
  ibv_free_device_list(dev_list);
}

void IbComm::print_obj(size_t my_rank, std::string obj_name, hwloc_obj_t obj) {
  if (my_rank != 0) return;
  if (!obj) {
    HCTR_LOG_S(INFO, WORLD) << obj_name << ":NULL" << std::endl;
    return;
  }
  if (obj->type == HWLOC_OBJ_PCI_DEVICE) {
    HCTR_LOG_S(INFO, WORLD) << obj_name << ":PCIeDevice " << obj->gp_index << " " << obj << " "
                            << obj->depth << " " << obj->attr->pcidev.dev << std::endl;
  } else if (obj->type == HWLOC_OBJ_OS_DEVICE) {
    HCTR_LOG_S(INFO, WORLD) << obj_name << ":OSdev " << obj->gp_index << " " << obj << " "
                            << obj->depth << " " << obj->name << " " << obj->attr->osdev.type
                            << std::endl;
  } else if (obj->type == HWLOC_OBJ_BRIDGE) {
    HCTR_LOG_S(INFO, WORLD) << obj_name << ":PCIeBridge " << obj->gp_index << " " << obj << " "
                            << obj->depth << std::endl;
  } else {
    HCTR_LOG_S(INFO, WORLD) << obj_name << ":Unknown " << obj->gp_index << " " << obj << " "
                            << obj->depth << std::endl;
  }
}

size_t IbComm::calculate_pcie_hier_distance(size_t my_rank, hwloc_obj_t obj1, hwloc_obj_t obj2) {
  size_t distance = 0;

  auto is_bridge = [](hwloc_obj_t obj) { return obj && (obj->type == HWLOC_OBJ_BRIDGE); };
  auto are_bridges = [is_bridge](hwloc_obj_t obj1, hwloc_obj_t obj2) {
    return is_bridge(obj1) && is_bridge(obj2);
  };

  while (!is_bridge(obj1)) {
    obj1 = obj1->parent;
  }
  while (!is_bridge(obj2)) {
    obj2 = obj2->parent;
  }

  while (are_bridges(obj1, obj2) && (obj1 != obj2)) {
    while (are_bridges(obj1, obj2) && (obj1->attr->bridge.depth > obj2->attr->bridge.depth)) {
      obj1 = obj1->parent;
      distance++;
    }
    while (are_bridges(obj1, obj2) && (obj2->attr->bridge.depth > obj1->attr->bridge.depth)) {
      obj2 = obj2->parent;
      distance++;
    }
    if (are_bridges(obj1, obj2) && (obj1 != obj2)) {
      obj1 = obj1->parent;
      obj2 = obj2->parent;
      distance += 2;
    }
  }

  if (obj1 != obj2) {  // No common PCIe ancestor found. Must be SYS.
    distance = std::numeric_limits<size_t>::max();
  }
  return distance;
}

void IbComm::print_distance_matrix(size_t my_rank, std::vector<std::vector<size_t>>& gpu_nic_dist) {
  // Print distance matrix
  if (my_rank == 0) {
    {
      auto log = HCTR_LOG_S(INFO, WORLD);
      for (size_t n = 0; n < ib_dev_list_.size(); n++) {
        log << std::setfill(' ') << std::setw(24) << ib_dev_list_[n].dev_name;
      }
      log << std::endl;
    }
    {
      auto log = HCTR_LOG_S(INFO, WORLD);
      for (size_t g = 0; g < num_gpus_; g++) {
        for (size_t n = 0; n < ib_dev_list_.size(); n++) {
          log << std::setfill(' ') << std::setw(24) << gpu_nic_dist[g][n];
        }
        log << std::endl;
      }
    }
  }
}

void IbComm::calculate_gpu_nic_affinity() {
  // get hwloc GPU objs
  std::vector<hwloc_obj_t> gpu_list;
  for (auto& g : device_list_) {
    auto gpu_obj = hwloc_cudart_get_device_osdev_by_index(topo_, g);
    if (!gpu_obj) {
      HCTR_LOG_S(ERROR, WORLD) << "unable to get hwloc obj for cuda device " << g << std::endl;
      exit(1);
    }
    gpu_list.push_back(gpu_obj);
  }

  // Find GPU-NIC distances
  std::vector<std::vector<size_t>> gpu_nic_dist(num_gpus_);
  for (size_t g = 0; g < num_gpus_; g++) {
    gpu_nic_dist[g].resize(ib_dev_list_.size());
    for (size_t n = 0; n < ib_dev_list_.size(); n++) {
      hwloc_obj_t gpu_obj = gpu_list[g];
      gpu_nic_dist[g][n] =
          calculate_pcie_hier_distance(my_proc_, gpu_obj, ib_dev_list_[n].hwloc_obj);
    }
  }

  // print_distance_matrix(my_proc_, gpu_nic_dist);

  // Calculate affinities. Only supports at max one NIC per GPU
  // If we need to support more than one NIC per GPU in future, we can replicate the gpu devs.
  size_t max_nics = ib_dev_list_.size();
  gpu_nic_affinity_.resize(num_gpus_, max_nics);
  if (num_gpus_ >= ib_dev_list_.size()) {
    size_t current_nic = 0;
    for (size_t assigned_gpus = 0; assigned_gpus < num_gpus_; assigned_gpus++) {
      // Greedy algorithm
      // Find unassigned gpu with min distance
      size_t min_distance = std::numeric_limits<size_t>::max();
      size_t min_gpu = 0;
      for (size_t g = 0; g < num_gpus_; g++) {
        if ((gpu_nic_affinity_[g] == max_nics) && (gpu_nic_dist[g][current_nic] <= min_distance)) {
          min_distance = gpu_nic_dist[g][current_nic];
          min_gpu = g;
        }
      }
      gpu_nic_affinity_[min_gpu] = current_nic;
      current_nic = (current_nic + 1) % ib_dev_list_.size();
    }
  } else {
    // still assigns max one NIC per GPU. Just iterate over NICs instead
    for (size_t g = 0; g < num_gpus_; g++) {
      size_t min_distance = std::numeric_limits<size_t>::max();
      size_t min_nic = 0;
      for (size_t n = 0; n < ib_dev_list_.size(); n++) {
        if ((ib_dev_list_[n].num_gpus_assigned == 0) && (gpu_nic_dist[g][n] <= min_distance)) {
          min_distance = gpu_nic_dist[g][n];
          min_nic = n;
        }
      }
      gpu_nic_affinity_[g] = min_nic;
    }
  }

  // Print gpu nic affinities that are picked;
  if (my_proc_ == 0) {
    for (size_t g = 0; g < num_gpus_; g++) {
      const auto& ib_dev = ib_dev_list_[gpu_nic_affinity_[g]];
      HCTR_LOG_S(INFO, ROOT) << "GPU-NIC affinity " << g << "-" << ib_dev.dev_name << ":"
                             << ib_dev.dev_port_id << std::endl;
    }
  }

  // Check gpu nic affinities of other nodes and warn if mismatch
  char(**gpu_nic_affinity_names)[IBV_SYSFS_NAME_MAX];
  gpu_nic_affinity_names =
      (char(**)[IBV_SYSFS_NAME_MAX])malloc(sizeof(char(*)[IBV_SYSFS_NAME_MAX]) * num_procs_);
  for (size_t r = 0; r < num_procs_; r++) {
    gpu_nic_affinity_names[r] =
        (char(*)[IBV_SYSFS_NAME_MAX])malloc(sizeof(char[IBV_SYSFS_NAME_MAX]) * num_gpus_);
  }
  for (size_t g = 0; g < num_gpus_; g++) {
    auto ib_dev = ib_dev_list_[gpu_nic_affinity_[g]];

    std::ostringstream os;
    os << ib_dev.dev_name << ":" << ib_dev.dev_port_id;
    std::string ib_name = os.str();
    ib_name = ib_name.substr(0, IBV_SYSFS_NAME_MAX);
    std::strcpy(gpu_nic_affinity_names[my_proc_][g], ib_name.c_str());
  }

  for (size_t r = 0; r < num_procs_; r++) {
    HCTR_MPI_THROW(MPI_Bcast(gpu_nic_affinity_names[r],
                             num_gpus_ * sizeof(char[IBV_SYSFS_NAME_MAX]), MPI_BYTE, r,
                             MPI_COMM_WORLD));
  }

  for (size_t r = 0; r < num_procs_; r++) {
    for (size_t g = 0; g < num_gpus_; g++) {
      std::string my_ib_name = std::string(gpu_nic_affinity_names[my_proc_][g]);
      std::string remote_ib_name = std::string(gpu_nic_affinity_names[r][g]);
      if (my_ib_name != remote_ib_name) {
        HCTR_LOG_S(WARNING, WORLD)
            << "Mismatch in mellanox dev names. " << g << " " << my_proc_ << ":" << my_ib_name
            << " " << r << ":" << remote_ib_name << std::endl;
        HCTR_LOG_S(WARNING, WORLD)
            << "Non uniform cluster detected. Performance maybe impacted" << std::endl;
      }
    }
  }

  for (size_t r = 0; r < num_procs_; r++) {
    free(gpu_nic_affinity_names[r]);
  }
  free(gpu_nic_affinity_names);
  HCTR_MPI_THROW(MPI_Barrier(MPI_COMM_WORLD));
}

void IbComm::init_proxy_threads() {
  proxy_cmd_ = std::make_unique<ProxyCommand>(num_gpus_);
  proxy_cmd_->reset();
  proxy_thread_.resize(num_gpus_);
  proxy_cfg_.resize(num_gpus_);
  for (auto& cfg : proxy_cfg_) {
    cfg = std::make_unique<IbvProxy::InitConfig>();
  }

  for (size_t g = 0; g < num_gpus_; g++) {
    size_t device_id = device_list_[g];

    auto& cfg = proxy_cfg_[g];
    cfg->device_id_ = device_id;
    cfg->global_id_ = my_proc_;
    cfg->proxy_id_ = g;
    cfg->ib_dev_ = ib_dev_list_[gpu_nic_affinity_[g]].dev_name;
    cfg->ib_port_ = ib_dev_list_[gpu_nic_affinity_[g]].dev_port_id;
    cfg->proxy_cmd_ = proxy_cmd_.get();
    cfg->num_gpus_ = num_gpus_;
    cfg->num_procs_ = num_procs_;
    cfg->my_proc_ = my_proc_;

    sched_param param;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_getschedparam(&attr, &param);
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    ;
    pthread_attr_setschedparam(&attr, &param);
    int ret = pthread_create(&proxy_thread_[g], &attr, &proxy_thread_func, cfg.get());
    PROXY_ASSERT(ret == 0);
  }
}

// API implementation
int IbComm::init(size_t num_procs, size_t num_gpus, size_t my_proc,
                 const std::vector<int>& device_list) {
  num_procs_ = num_procs;
  num_gpus_ = num_gpus;
  my_proc_ = my_proc;
  device_list_ = device_list;

  PROXY_ASSERT(num_procs > 1);
  detect_ib_devs();
  calculate_gpu_nic_affinity();
  init_proxy_threads();
  is_initialized_ = true;
  return 0;
}

IbComm::HierA2ACollContext::HierA2ACollContext(IbComm* comm) {
  HCTR_LIB_THROW(cudaMallocHost(&cmd_storage_, 2 * sizeof(size_t)));
  h_recv_cmd_ptr_ = &cmd_storage_[0];
  *h_recv_cmd_ptr_ = 1;

  size_t num_gpus = comm->num_gpus_;
  std::generate_n(std::back_inserter(ctx_), num_gpus,
                  [] { return std::make_unique<HierA2ACollContextPerGPU>(); });
  d_send_cmd_ = new size_t*[num_gpus];
  d_ibv_atomic_ = new size_t*[num_gpus];
  d_ibv_atomic_recv_ = new size_t*[num_gpus];

  for (size_t g = 0; g < num_gpus; g++) {
    HCTR_LIB_THROW(cudaSetDevice(comm->device_list_[g]));
    HCTR_LIB_THROW(cudaEventCreate(&ctx_[g]->event_));

    // TODO: collate all storage
    HCTR_LIB_THROW(cudaMalloc((void**)&d_send_cmd_[g], sizeof(size_t)));
    size_t init_value = 2;
    HCTR_LIB_THROW(cudaMemcpy(d_send_cmd_[g], &init_value, sizeof(size_t), cudaMemcpyHostToDevice));

    HCTR_LIB_THROW(cudaMalloc((void**)&d_ibv_atomic_[g], MAX_IBV_DEST * sizeof(size_t)));
    size_t atomic_init_values[MAX_IBV_DEST];
    std::fill_n(atomic_init_values, MAX_IBV_DEST, 1);
    HCTR_LIB_THROW(cudaMemcpy(d_ibv_atomic_[g], atomic_init_values, MAX_IBV_DEST * sizeof(size_t),
                              cudaMemcpyHostToDevice));

    HCTR_LIB_THROW(cudaMalloc((void**)&d_ibv_atomic_recv_[g], MAX_IBV_DEST * sizeof(size_t)));
    std::fill_n(atomic_init_values, MAX_IBV_DEST, 0);
    HCTR_LIB_THROW(cudaMemcpy(d_ibv_atomic_recv_[g], atomic_init_values,
                              MAX_IBV_DEST * sizeof(size_t), cudaMemcpyHostToDevice));
  }

  barrier_ = std::make_unique<GPUBarrier>(comm->num_gpus_, comm->device_list_);
  sync_helper_ = std::make_unique<CollSyncHelper>();
}

IbComm::HierA2ACollContext::~HierA2ACollContext() {
  size_t num_gpus = ctx_.size();
  if (d_ibv_atomic_recv_) {
    for (size_t g = 0; g < num_gpus; g++) {
      cudaFree(d_ibv_atomic_recv_[g]);
    }
    delete d_ibv_atomic_recv_;
  }
  if (d_ibv_atomic_) {
    for (size_t g = 0; g < num_gpus; g++) {
      cudaFree(d_ibv_atomic_[g]);
    }
    delete d_ibv_atomic_;
  }
  if (d_send_cmd_) {
    for (size_t g = 0; g < num_gpus; g++) {
      cudaFree(d_send_cmd_[g]);
    }
    delete d_send_cmd_;
  }
  if (cmd_storage_) {
    cudaFree(cmd_storage_);
  }
}

IbComm::HierA2ACollContextPerGPU::~HierA2ACollContextPerGPU() {
  if (d_send_ptrs_) {
    free(d_send_ptrs_);
  }
  if (d_recv_ptrs_) {
    free(d_recv_ptrs_);
  }
  if (d_send_sizes_copy_) {
    cudaFree(d_send_sizes_copy_);
  }
}

// TODO: Initialize these in the constructor for RAI

HierA2ACollHandle IbComm::register_hier_a2a_coll(bool skip_barrier) {
  // std::unique_lock<std::mutex> lock(proxy_cmd_->mutex_);
  hier_a2a_coll_ctx_.emplace_back(std::make_unique<HierA2ACollContext>(this));
  HierA2ACollHandle coll_handle = (HierA2ACollHandle)(hier_a2a_coll_ctx_.size() - 1);
  auto sync_helper = hier_a2a_v_coll_ctx_[coll_handle]->sync_helper_.get();
  M2PHierA2ACollInit coll_init_cmd_(coll_handle, sync_helper, skip_barrier);
  for (size_t g = 0; g < num_gpus_; g++) {
    M2PHierA2ACollInit coll_init_cmd_(coll_handle, sync_helper, skip_barrier);
    HierA2ACollInitCmd cmd = std::make_pair(std::move(coll_init_cmd_), std::move(P2MNull()));
    proxy_cmd_->cmd_[g] = std::move(cmd);
  }
  proxy_cmd_->post_command();
  proxy_cmd_->wait_for_completion();
  proxy_cmd_->reset();
  return coll_handle;
}

HierA2AvCollHandle IbComm::register_hier_a2a_v_coll(bool skip_barrier) {
  // std::unique_lock<std::mutex> lock(proxy_cmd_->mutex_);
  hier_a2a_v_coll_ctx_.emplace_back(std::make_unique<HierA2ACollContext>(this));
  HierA2AvCollHandle coll_handle = (HierA2AvCollHandle)(hier_a2a_v_coll_ctx_.size() - 1);
  auto sync_helper = hier_a2a_v_coll_ctx_[coll_handle]->sync_helper_.get();
  for (size_t g = 0; g < num_gpus_; g++) {
    M2PHierA2AvCollInit coll_init_cmd_(coll_handle, sync_helper, skip_barrier);
    HierA2AvCollInitCmd cmd = std::make_pair(std::move(coll_init_cmd_), std::move(P2MNull()));
    proxy_cmd_->cmd_[g] = std::move(cmd);
  }
  proxy_cmd_->post_command();
  proxy_cmd_->wait_for_completion();
  proxy_cmd_->reset();

  return coll_handle;
}

void IbComm::set_a2a_coll_stream(HierA2ACollHandle coll, cudaStream_t stream, size_t device_id) {
  hier_a2a_coll_ctx_[coll]->ctx_[device_id]->stream_ = stream;
}

void IbComm::set_a2a_coll_stream(HierA2AvCollHandle coll, cudaStream_t stream, size_t device_id) {
  hier_a2a_v_coll_ctx_[coll]->ctx_[device_id]->stream_ = stream;
}

void IbComm::set_a2a_coll_buf(HierA2ACollHandle coll, void** send_ptrs, const size_t* send_max_size,
                              void** recv_ptrs, const size_t* recv_max_size, size_t device_id) {
  auto& coll_ctx = *hier_a2a_coll_ctx_[coll];
  if (proxy_cmd_->cmd_[device_id].which() != 0) {
    HCTR_LOG_S(ERROR, WORLD) << "Proxy command is already populated. Don't mix up set API. "
                             << HCTR_LOCATION() << std::endl;
    exit(1);
  }
  proxy_cmd_->cmd_[device_id] = HierA2ABufInitCmd();
  HierA2ABufInitCmd& cmd = boost::get<HierA2ABufInitCmd>(proxy_cmd_->cmd_[device_id]);
  M2PHierA2ABufInit& buf_init = std::get<0>(cmd);

  auto& gpu_ctx = *coll_ctx.ctx_[device_id];
  gpu_ctx.d_send_ptrs_ = (void**)malloc(sizeof(void*) * num_procs_);
  gpu_ctx.d_recv_ptrs_ = (void**)malloc(sizeof(void*) * num_procs_);

  memcpy(gpu_ctx.d_send_ptrs_, send_ptrs, sizeof(void*) * num_procs_);
  memcpy(gpu_ctx.d_recv_ptrs_, recv_ptrs, sizeof(void*) * num_procs_);

  buf_init.coll_handle_ = coll;
  buf_init.d_send_ptrs_ = send_ptrs;
  buf_init.d_recv_ptrs_ = recv_ptrs;
  buf_init.h_max_send_size_ = send_max_size;
  buf_init.h_max_recv_size_ = recv_max_size;
  buf_init.h_recv_cmd_ptr_ = coll_ctx.h_recv_cmd_ptr_;
  buf_init.d_ibv_atomic_ = coll_ctx.d_ibv_atomic_[device_id];
  buf_init.d_ibv_atomic_recv_ = coll_ctx.d_ibv_atomic_recv_[device_id];
}

void IbComm::set_a2a_coll_buf(HierA2AvCollHandle coll, void* send_ptrs, const size_t send_max_size,
                              void* recv_ptrs, const size_t recv_max_size, size_t device_id) {
  auto& coll_ctx = *hier_a2a_v_coll_ctx_[coll];
  if (proxy_cmd_->cmd_[device_id].which() != 0) {
    HCTR_LOG_S(ERROR, WORLD) << "Proxy command is already populated. Don't mix up set API. "
                             << HCTR_LOCATION() << std::endl;
    exit(1);
  }
  proxy_cmd_->cmd_[device_id] = HierA2AvBufInitCmd();
  HierA2AvBufInitCmd& cmd = boost::get<HierA2AvBufInitCmd>(proxy_cmd_->cmd_[device_id]);
  M2PHierA2AvBufInit& buf_init = std::get<0>(cmd);

  auto& gpu_ctx = *coll_ctx.ctx_[device_id];
  gpu_ctx.d_send_ptrs_ = (void**)malloc(sizeof(void*));
  gpu_ctx.d_recv_ptrs_ = (void**)malloc(sizeof(void*));
  gpu_ctx.d_send_ptrs_[0] = send_ptrs;
  gpu_ctx.d_recv_ptrs_[0] = recv_ptrs;
  gpu_ctx.h_max_send_size_ = send_max_size;

  HCTR_LIB_THROW(cudaSetDevice(device_list_[device_id]));

  // Allocate A2Av send size copy storage
  HCTR_LIB_THROW(
      cudaMalloc((void**)(&gpu_ctx.d_send_sizes_copy_), sizeof(size_t) * num_gpus_ * num_procs_));
  std::vector<size_t> send_sizes(num_gpus_ * num_procs_, send_max_size / (num_gpus_ * num_procs_));
  HCTR_LIB_THROW(cudaMemcpy(gpu_ctx.d_send_sizes_copy_, send_sizes.data(),
                            sizeof(size_t) * num_gpus_ * num_procs_, cudaMemcpyHostToDevice));

  buf_init.coll_handle_ = coll;
  buf_init.d_send_ptrs_ = send_ptrs;
  buf_init.d_recv_ptrs_ = recv_ptrs;
  buf_init.h_max_send_size_ = send_max_size;
  buf_init.h_max_recv_size_ = recv_max_size;
  buf_init.h_recv_cmd_ptr_ = coll_ctx.h_recv_cmd_ptr_;
  buf_init.d_ibv_atomic_ = coll_ctx.d_ibv_atomic_[device_id];
  buf_init.d_ibv_atomic_recv_ = coll_ctx.d_ibv_atomic_recv_[device_id];
}

void IbComm::register_a2a_coll_buf(HierA2ACollHandle coll) {
  // Init command pointers
  auto& coll_ctx = *hier_a2a_coll_ctx_[coll];

  proxy_cmd_->post_command();
  proxy_cmd_->wait_for_completion();

  for (size_t g = 0; g < num_gpus_; g++) {
    HierA2ABufInitCmd& proxy_cmd = boost::get<HierA2ABufInitCmd>(proxy_cmd_->cmd_[g]);
    auto& buf_init_out = std::get<1>(proxy_cmd);
    coll_ctx.ctx_[g]->h_send_sizes_ = buf_init_out.h_send_size_;
    coll_ctx.ctx_[g]->h_recv_sizes_ = buf_init_out.h_recv_size_;
  }
  proxy_cmd_->reset();
}

void IbComm::register_a2a_coll_buf(HierA2AvCollHandle coll) {
  // Init command pointers
  auto& coll_ctx = *hier_a2a_v_coll_ctx_[coll];

  proxy_cmd_->post_command();
  proxy_cmd_->wait_for_completion();

  for (size_t g = 0; g < num_gpus_; g++) {
    HierA2AvBufInitCmd& proxy_cmd = boost::get<HierA2AvBufInitCmd>(proxy_cmd_->cmd_[g]);
    auto& buf_init_out = std::get<1>(proxy_cmd);
    coll_ctx.ctx_[g]->h_send_sizes_ = buf_init_out.h_send_size_;
    coll_ctx.ctx_[g]->h_recv_sizes_ = buf_init_out.h_recv_size_;
  }
  proxy_cmd_->reset();
}

static __global__ void update_sizes(size_t* __restrict__ h_send_sizes,
                                    size_t* __restrict__ h_recv_sizes,
                                    size_t* __restrict__ d_send_sizes_copy,
                                    const size_t* __restrict__ d_send_sizes,
                                    const size_t* __restrict__ d_recv_sizes, size_t size) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    size_t send_size = d_send_sizes[i];
    h_send_sizes[i] = send_size;
    d_send_sizes_copy[i] = send_size;
    h_recv_sizes[i] = d_recv_sizes[i];
  }
}

void IbComm::update_a2a_coll_sizes(HierA2AvCollHandle coll, const size_t* d_send_sizes,
                                   const size_t* d_recv_sizes, cudaStream_t dep_stream,
                                   size_t device_id) {
  auto& ctx = *hier_a2a_v_coll_ctx_[coll];
  auto& gpu_ctx = *ctx.ctx_[device_id];
  HCTR_LIB_THROW(cudaEventRecord(gpu_ctx.event_, dep_stream));
  HCTR_LIB_THROW(cudaStreamWaitEvent(gpu_ctx.stream_, gpu_ctx.event_));
  constexpr size_t MAX_TPB = 256;
  size_t n_blocks = ceildiv<size_t>(num_procs_ * num_gpus_, MAX_TPB);
  update_sizes<<<n_blocks, MAX_TPB, 0, gpu_ctx.stream_>>>(
      gpu_ctx.h_send_sizes_, gpu_ctx.h_recv_sizes_, gpu_ctx.d_send_sizes_copy_, d_send_sizes,
      d_recv_sizes, num_procs_ * num_gpus_);
}

// Local first distribution TODO: node first might be efficient
static __global__ void update_pre_intra_sizes(size_t* __restrict__ h_send_sizes,
                                              size_t* __restrict__ d_send_sizes,
                                              size_t** __restrict__ d_pre_intra_send_sizes,
                                              size_t my_gpu_id, size_t num_gpus, size_t num_procs) {
  // Thread blocks = num procs
  // Threads = num gpus
  int gpu_id = threadIdx.x;
  int proc_id = blockIdx.x;
  size_t send_size = d_pre_intra_send_sizes[gpu_id][proc_id * num_gpus + my_gpu_id];
  size_t send_indx = proc_id * num_gpus + gpu_id;
  h_send_sizes[send_indx] = send_size;
  d_send_sizes[send_indx] = send_size;
  // TODO: uncomment below for cuda graph support
  //  __threadfence_system();
}

void IbComm::pre_intra_update_a2a_coll_sizes(HierA2AvCollHandle coll,
                                             size_t** d_pre_intra_send_sizes,
                                             cudaStream_t dep_stream, size_t device_id) {
  auto& ctx = *hier_a2a_v_coll_ctx_[coll];
  auto& gpu_ctx = *ctx.ctx_[device_id];
  HCTR_LIB_THROW(cudaEventRecord(gpu_ctx.event_, dep_stream));
  HCTR_LIB_THROW(cudaStreamWaitEvent(gpu_ctx.stream_, gpu_ctx.event_));
  ctx.barrier_->sync_all_gpus(gpu_ctx.stream_, device_id);
  update_pre_intra_sizes<<<num_procs_, num_gpus_, 0, gpu_ctx.stream_>>>(
      gpu_ctx.h_send_sizes_, gpu_ctx.d_send_sizes_copy_, d_pre_intra_send_sizes, device_id,
      num_gpus_, num_procs_);
}

void IbComm::set_ready_to_transfer() {
  PROXY_ASSERT_MSG(!is_ready_to_transfer_, "Ready to transfer is already set")
  for (size_t g = 0; g < num_gpus_; g++) {
    proxy_cmd_->cmd_[g] = ProxyStateTransitionCmd();
    ProxyStateTransitionCmd& cmd_t = boost::get<ProxyStateTransitionCmd>(proxy_cmd_->cmd_[g]);
    M2PStateTransition& cmd = std::get<0>(cmd_t);
    cmd.state_ = IbvProxyState::READY_TO_TRANSFER;
  }
  proxy_cmd_->post_command();
  proxy_cmd_->wait_for_completion();
  proxy_cmd_->reset();
  is_ready_to_transfer_ = true;
}

template <typename T>
static __global__ void copy_local(const T* __restrict__ input_, T* __restrict__ output_,
                                  size_t size) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    output_[i] = input_[i];
  }
}

template <typename T>
static __global__ void copy_local_segmented(const T* __restrict__ input_, T* __restrict__ output_,
                                            const size_t* __restrict__ sizes, int num_segments,
                                            size_t offset) {
  for (int s = 0; s < num_segments; s++) {
    int segment_offset = s * offset;
    size_t num_elems = sizes[s] / sizeof(T);
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems;
         i += blockDim.x * gridDim.x) {
      output_[segment_offset + i] = input_[segment_offset + i];
    }
  }
}

static __global__ void wait_completion(size_t* d_ibv_cmd, size_t* atomic, int nDest, int myDest,
                                       int device_id) {
  if ((threadIdx.x < nDest) && (threadIdx.x != myDest)) {
    size_t curr_count = *(volatile size_t*)d_ibv_cmd;
    // clock_t s=clock64();
    while (*((volatile size_t*)&atomic[threadIdx.x]) < (curr_count - 1)) {
      // if (clock64()-s > 2000000000) {
      //   HCTR_LOG(INFO, WORLD, "wait completion expected: %llu %llu, got %llu from_dest %d my_dest
      //   %d %d n_dest %d\n",
      //     curr_count, (curr_count - 1), atomic[threadIdx.x], threadIdx.x, myDest, device_id,
      //     nDest);
      //   s = clock64();
      // }
    }
  }
  __syncthreads();
}

template <typename T>
void IbComm::post_send_command_a2a<T>(HierA2ACollHandle coll, cudaStream_t dep_stream,
                                      size_t device_id) {
  auto& ctx = *hier_a2a_coll_ctx_[coll];
  auto& gpu_ctx = *ctx.ctx_[device_id];
  HCTR_LIB_THROW(cudaEventRecord(gpu_ctx.event_, dep_stream));
  HCTR_LIB_THROW(cudaStreamWaitEvent(gpu_ctx.stream_, gpu_ctx.event_));
  ctx.barrier_->sync_all_gpus_report_host_and_inc(ctx.d_send_cmd_[device_id], ctx.h_recv_cmd_ptr_,
                                                  gpu_ctx.stream_, device_id);
  size_t num_elems = gpu_ctx.h_send_sizes_[my_proc_] / sizeof(T);
  // TODO: This is not capturable as we using sizes from host
  copy_local<T><<<96, 1024, 0, gpu_ctx.stream_>>>((T*)gpu_ctx.d_send_ptrs_[my_proc_],
                                                  (T*)gpu_ctx.d_recv_ptrs_[my_proc_], num_elems);
  wait_completion<<<1, 32, 0, gpu_ctx.stream_>>>(
      ctx.d_send_cmd_[device_id], ctx.d_ibv_atomic_[device_id], num_procs_, my_proc_, device_id);
}

template <typename T>
void IbComm::post_send_command_a2a<T>(HierA2AvCollHandle coll, cudaStream_t dep_stream,
                                      size_t device_id) {
  auto& ctx = *hier_a2a_v_coll_ctx_[coll];
  auto& gpu_ctx = *ctx.ctx_[device_id];
  HCTR_LIB_THROW(cudaEventRecord(gpu_ctx.event_, dep_stream));
  HCTR_LIB_THROW(cudaStreamWaitEvent(gpu_ctx.stream_, gpu_ctx.event_));
  ctx.barrier_->sync_all_gpus_report_host_and_inc(ctx.d_send_cmd_[device_id], ctx.h_recv_cmd_ptr_,
                                                  gpu_ctx.stream_, device_id);
  // TODO: Change it to use max SMs
  size_t* copy_sizes = &gpu_ctx.d_send_sizes_copy_[my_proc_ * num_gpus_];
  size_t offset = gpu_ctx.h_max_send_size_ / (num_procs_ * num_gpus_) / sizeof(T);
  // TODO: This is not good, we are reading the sizes from host, create a device copy!
  copy_local_segmented<T><<<96, 1024, 0, gpu_ctx.stream_>>>(
      (T*)gpu_ctx.d_send_ptrs_[0] + (my_proc_ * num_gpus_ * offset),
      (T*)gpu_ctx.d_recv_ptrs_[0] + (my_proc_ * num_gpus_ * offset), copy_sizes, num_gpus_, offset);
  wait_completion<<<1, 32, 0, gpu_ctx.stream_>>>(
      ctx.d_send_cmd_[device_id], ctx.d_ibv_atomic_[device_id], num_procs_, my_proc_, device_id);
}

template <typename T>
void IbComm::post_a2a_send_command<T>(HierA2AvCollHandle coll, cudaStream_t dep_stream,
                                      size_t device_id) {
  auto& ctx = *hier_a2a_v_coll_ctx_[coll];
  auto& gpu_ctx = *ctx.ctx_[device_id];
  HCTR_LIB_THROW(cudaEventRecord(gpu_ctx.event_, dep_stream));
  HCTR_LIB_THROW(cudaStreamWaitEvent(gpu_ctx.stream_, gpu_ctx.event_));
  ctx.barrier_->sync_all_gpus_report_host_and_inc(ctx.d_send_cmd_[device_id], ctx.h_recv_cmd_ptr_,
                                                  gpu_ctx.stream_, device_id);
  // TODO: Change it to use max SMs
  size_t* copy_sizes = &gpu_ctx.d_send_sizes_copy_[my_proc_ * num_gpus_];
  size_t offset = gpu_ctx.h_max_send_size_ / (num_procs_ * num_gpus_) / sizeof(T);
  // TODO: This is not good, we are reading the sizes from host, create a device copy!
  copy_local_segmented<T><<<96, 1024, 0, gpu_ctx.stream_>>>(
      (T*)gpu_ctx.d_send_ptrs_[0] + (my_proc_ * num_gpus_ * offset),
      (T*)gpu_ctx.d_recv_ptrs_[0] + (my_proc_ * num_gpus_ * offset), copy_sizes, num_gpus_, offset);
}

void IbComm::blocking_wait(HierA2AvCollHandle coll, cudaStream_t dep_stream, size_t device_id) {
  auto& ctx = *hier_a2a_v_coll_ctx_[coll];
  auto& gpu_ctx = *ctx.ctx_[device_id];
  HCTR_LIB_THROW(cudaEventRecord(gpu_ctx.event_, dep_stream));
  HCTR_LIB_THROW(cudaStreamWaitEvent(gpu_ctx.stream_, gpu_ctx.event_));

  wait_completion<<<1, 32, 0, gpu_ctx.stream_>>>(
      ctx.d_send_cmd_[device_id], ctx.d_ibv_atomic_[device_id], num_procs_, my_proc_, device_id);
}

static __global__ void wait_recv(size_t* d_ibv_cmd, size_t* atomic, int nDest, int myDest) {
  if ((threadIdx.x < nDest) && (threadIdx.x != myDest)) {
    size_t curr_count = *d_ibv_cmd;
    while (*((volatile size_t*)&atomic[threadIdx.x]) < (curr_count - 2)) {
    }
  }
  __syncthreads();
}

void IbComm::wait_global_recv_async(HierA2ACollHandle coll, size_t device_id) {
  auto& ctx = *hier_a2a_coll_ctx_[coll];
  auto& gpu_ctx = *ctx.ctx_[device_id];
  wait_recv<<<1, 32, 0, gpu_ctx.stream_>>>(ctx.d_send_cmd_[device_id],
                                           ctx.d_ibv_atomic_recv_[device_id], num_procs_, my_proc_);
}

void IbComm::wait_global_recv_async(HierA2AvCollHandle coll, size_t device_id) {
  auto& ctx = *hier_a2a_v_coll_ctx_[coll];
  auto& gpu_ctx = *ctx.ctx_[device_id];
  wait_recv<<<1, 32, 0, gpu_ctx.stream_>>>(ctx.d_send_cmd_[device_id],
                                           ctx.d_ibv_atomic_recv_[device_id], num_procs_, my_proc_);
}

template void IbComm::post_send_command_a2a<__half>(HierA2ACollHandle coll, cudaStream_t dep_stream,
                                                    size_t device_id);
template void IbComm::post_send_command_a2a<float>(HierA2ACollHandle coll, cudaStream_t dep_stream,
                                                   size_t device_id);
template void IbComm::post_send_command_a2a<uint32_t>(HierA2ACollHandle coll,
                                                      cudaStream_t dep_stream, size_t device_id);
template void IbComm::post_send_command_a2a<uint16_t>(HierA2ACollHandle coll,
                                                      cudaStream_t dep_stream, size_t device_id);

template void IbComm::post_send_command_a2a<__half>(HierA2AvCollHandle coll,
                                                    cudaStream_t dep_stream, size_t device_id);
template void IbComm::post_send_command_a2a<float>(HierA2AvCollHandle coll, cudaStream_t dep_stream,
                                                   size_t device_id);
template void IbComm::post_send_command_a2a<uint32_t>(HierA2AvCollHandle coll,
                                                      cudaStream_t dep_stream, size_t device_id);
template void IbComm::post_send_command_a2a<uint16_t>(HierA2AvCollHandle coll,
                                                      cudaStream_t dep_stream, size_t device_id);

template void IbComm::post_a2a_send_command<__half>(HierA2AvCollHandle coll,
                                                    cudaStream_t dep_stream, size_t device_id);
template void IbComm::post_a2a_send_command<float>(HierA2AvCollHandle coll, cudaStream_t dep_stream,
                                                   size_t device_id);
template void IbComm::post_a2a_send_command<uint32_t>(HierA2AvCollHandle coll,
                                                      cudaStream_t dep_stream, size_t device_id);
template void IbComm::post_a2a_send_command<uint16_t>(HierA2AvCollHandle coll,
                                                      cudaStream_t dep_stream, size_t device_id);

void IbComm::finalize() {
  if (!is_initialized_) {
    return;
  }
  if (!is_ready_to_transfer_) {
    for (size_t g = 0; g < num_gpus_; g++) {
      proxy_cmd_->cmd_[g] = ProxyStateTransitionCmd();
      ProxyStateTransitionCmd& cmd_t = boost::get<ProxyStateTransitionCmd>(proxy_cmd_->cmd_[g]);
      M2PStateTransition& cmd = std::get<0>(cmd_t);
      cmd.state_ = IbvProxyState::DESTROY;
    }
    proxy_cmd_->post_command();
    proxy_cmd_->wait_for_completion();
    proxy_cmd_->reset();
  }
  proxy_cmd_->set_destroy();
  for (size_t g = 0; g < num_gpus_; g++) {
    int ret = pthread_join(proxy_thread_[g], NULL);
    PROXY_ASSERT(ret == 0);
  }
  is_finalized_ = true;
}

IbComm::~IbComm() {
  if (!is_finalized_) {
    finalize();
  }
}
}  // namespace HugeCTR
#endif
