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

#ifdef ENABLE_MPI

#include <gtest/gtest.h>

#include <chrono>
#include <collectives/ib_comm.hpp>
#include <common.hpp>
#include <core23/mpi_init_service.hpp>
#include <general_buffer2.hpp>
#include <random>
#include <resource_managers/resource_manager_ext.hpp>
#include <tensor2.hpp>
#include <utest/test_utils.hpp>
#include <utils.hpp>

using namespace HugeCTR;

#define TIMEIT(function, bench_time)                                                       \
  {                                                                                        \
    int warmup_iters = 10;                                                                 \
    for (int i = 0; i < warmup_iters; i++) {                                               \
      function;                                                                            \
    }                                                                                      \
    stream_sync_all();                                                                     \
                                                                                           \
    int iters = 1000;                                                                      \
    auto t0 = std::chrono::high_resolution_clock::now();                                   \
    for (int i = 0; i < iters; i++) {                                                      \
      function;                                                                            \
    }                                                                                      \
    stream_sync_all();                                                                     \
    auto t1 = std::chrono::high_resolution_clock::now();                                   \
    bench_time =                                                                           \
        1.e6 * std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count(); \
    bench_time = bench_time / iters;                                                       \
  }

template <typename T>
static __global__ void intra_node_a2a(const T* __restrict__ input, T** __restrict__ output,
                                      const size_t* __restrict__ sizes, size_t max_elems_per_dest,
                                      size_t num_gpus, size_t num_procs, size_t my_gpu) {
  for (size_t g = 0; g < num_gpus; g++) {
    for (size_t n = 0; n < num_procs; n++) {
      T* output_ptr = &output[g][(n * num_gpus + my_gpu) * max_elems_per_dest];
      const T* input_ptr = &input[(n * num_gpus + g) * max_elems_per_dest];
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
           i < sizes[n * num_gpus + g] / sizeof(T); i += blockDim.x * gridDim.x) {
        output_ptr[i] = input_ptr[i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
  }
}

namespace {

template <bool is_integral, typename T>
struct uniform_distribution_selector;
template <typename T>
struct uniform_distribution_selector<true, T> {
  using type = typename std::uniform_int_distribution<T>;
};
template <typename T>
struct uniform_distribution_selector<false, T> {
  using type = typename std::uniform_real_distribution<T>;
};
template <typename T>
using uniform_distribution_t =
    typename uniform_distribution_selector<std::is_integral<T>::value, T>::type;

template <typename T>
ncclDataType_t get_nccl_type();
template <>
ncclDataType_t get_nccl_type<float>() {
  return ncclFloat32;
}
template <>
ncclDataType_t get_nccl_type<uint32_t>() {
  return ncclUint32;
}

template <typename TypeEmbeddingComp>
struct IbCommsTest {
 public:
  IbCommsTest(const std::vector<int>& device_list, size_t max_size, bool use_cuda_graph = false)
      : num_gpus_(device_list.size()),
        max_size_(max_size),
        use_cuda_graph_(use_cuda_graph),
        num_procs_{core23::MpiInitService::get().world_size()} {
    // Align max_size
    max_elems_per_dest_ = max_size_ / (num_procs_ * num_gpus_) / sizeof(TypeEmbeddingComp);
    max_size_ = (max_elems_per_dest_) * (num_procs_ * num_gpus_) * sizeof(TypeEmbeddingComp);
    max_elems_ = max_size_ / sizeof(TypeEmbeddingComp);
    max_elems_per_gpu_ = max_elems_ / device_list.size();
    max_size_per_gpu_ = max_elems_per_gpu_ * sizeof(TypeEmbeddingComp);
    max_size_ = max_size_per_gpu_ * num_gpus_;
    max_elems_per_proc_ = max_elems_ / num_procs_;

    std::vector<std::vector<int>> vvgpu;
    for (int i = 0; i < num_procs_; i++) {
      vvgpu.push_back(device_list);
    }
    resource_manager_ = ResourceManagerExt::create(vvgpu, 0, DeviceMap::LOCAL_FIRST);
    resource_manager_->init_ib_comm();
    ib_comm_ = resource_manager_->get_ib_comm();
    init_buffers();
    gen_uniform_size(max_size_);

    comm_stream_.resize(num_gpus_);
    comm_events_.resize(num_gpus_);
    for (size_t g = 0; g < num_gpus_; g++) {
      HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
      HCTR_LIB_THROW(cudaStreamCreate(&comm_stream_[g]));
      HCTR_LIB_THROW(cudaEventCreate(&comm_events_[g]));
      ib_comm_->set_a2a_coll_stream(coll_handle_, comm_stream_[g], g);
    }
  }

  ~IbCommsTest() { ib_comm_->finalize(); }

  void gen_uniform_size(size_t total_send_size) {
    size_t num_dest = num_gpus_ * num_procs_;
    size_t send_size_per_dst = total_send_size / (num_gpus_ * num_procs_);
    // Align to element type
    send_size_per_dst = (send_size_per_dst / sizeof(TypeEmbeddingComp)) * sizeof(TypeEmbeddingComp);
    auto& device_list = resource_manager_->get_local_gpu_device_id_list();

    for (size_t g = 0; g < num_gpus_; g++) {
      size_t* h_send_size_ptr = h_send_sizes_[g].get_ptr();
      size_t* h_recv_size_ptr = h_recv_sizes_[g].get_ptr();
      for (size_t d = 0; d < num_dest; d++) {
        h_send_size_ptr[d] = send_size_per_dst;
        h_recv_size_ptr[d] = send_size_per_dst;
      }
      HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
      HCTR_LIB_THROW(cudaMemcpy(d_send_sizes_[g].get_ptr(), h_send_sizes_[g].get_ptr(),
                                h_send_sizes_[g].get_num_elements() * sizeof(size_t),
                                cudaMemcpyHostToDevice));
      HCTR_LIB_THROW(cudaMemcpy(d_recv_sizes_[g].get_ptr(), h_recv_sizes_[g].get_ptr(),
                                h_recv_sizes_[g].get_num_elements() * sizeof(size_t),
                                cudaMemcpyHostToDevice));
    }
  }

  void gen_rand_size() {
    size_t num_dest = num_gpus_ * num_procs_;
    std::default_random_engine generator;
    uniform_distribution_t<size_t> distribution(1, max_elems_per_dest_);

    auto& device_list = resource_manager_->get_local_gpu_device_id_list();

    for (size_t g = 0; g < num_gpus_; g++) {
      size_t* h_send_size_ptr = h_send_sizes_[g].get_ptr();

      for (size_t d = 0; d < num_dest; d++) {
        h_send_size_ptr[d] = distribution(generator) * sizeof(TypeEmbeddingComp);
      }
    }

    for (size_t g = 0; g < num_gpus_; g++) {
      size_t* h_recv_size_ptr = h_recv_sizes_[g].get_ptr();
      std::vector<size_t> h_interim_recv_sizes;
      h_interim_recv_sizes.resize(num_dest);

      // Do intra node A2A
      for (size_t d = 0; d < num_gpus_; d++) {
        size_t* h_send_size_ptr = h_send_sizes_[d].get_ptr();
        for (int p = 0; p < num_procs_; p++) {
          h_interim_recv_sizes[p * num_gpus_ + d] = h_send_size_ptr[p * num_gpus_ + g];
        }
      }

      HCTR_MPI_THROW(MPI_Alltoall(h_interim_recv_sizes.data(), sizeof(size_t) * num_gpus_, MPI_BYTE,
                                  h_recv_size_ptr, sizeof(size_t) * num_gpus_, MPI_BYTE,
                                  MPI_COMM_WORLD));

      HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
      HCTR_LIB_THROW(cudaMemcpy(d_send_sizes_[g].get_ptr(), h_send_sizes_[g].get_ptr(),
                                h_send_sizes_[g].get_num_elements() * sizeof(size_t),
                                cudaMemcpyHostToDevice));
      HCTR_LIB_THROW(cudaMemcpy(d_recv_sizes_[g].get_ptr(), h_recv_sizes_[g].get_ptr(),
                                h_recv_sizes_[g].get_num_elements() * sizeof(size_t),
                                cudaMemcpyHostToDevice));
    }
  }

  void fill_buffers() {
    std::default_random_engine generator;
    uniform_distribution_t<TypeEmbeddingComp> distribution(1, 100);
    // reset recv buffers
    for (size_t g = 0; g < num_gpus_; g++) {
      memset(h_recv_buffs_[g].get_ptr(), 0, max_elems_ * sizeof(TypeEmbeddingComp));
      memset(h_recv_buffs_out_[g].get_ptr(), 1, max_elems_ * sizeof(TypeEmbeddingComp));
    }

    for (size_t g = 0; g < num_gpus_; g++) {
      for (size_t s = 0; s < max_elems_; s++) {
        TypeEmbeddingComp number = distribution(generator);
        *(h_send_buffs_[g].get_ptr() + s) = number;
      }
    }

    // for (size_t g = 0; g < num_gpus_; g++) {
    //   for (size_t s = 0; s < max_elems_; s++) {
    //     *(h_send_buffs_[g].get_ptr() + s) = s;
    //   }
    // }

    auto& device_list = resource_manager_->get_local_gpu_device_id_list();
    for (size_t g = 0; g < num_gpus_; g++) {
      HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
      HCTR_LIB_THROW(cudaMemcpy(d_send_buffs_[g].get_ptr(), h_send_buffs_[g].get_ptr(),
                                max_elems_ * sizeof(TypeEmbeddingComp), cudaMemcpyHostToDevice));
      HCTR_LIB_THROW(cudaMemcpy(d_recv_buffs_[g].get_ptr(), h_recv_buffs_[g].get_ptr(),
                                max_elems_ * sizeof(TypeEmbeddingComp), cudaMemcpyHostToDevice));
    }

    for (size_t g = 0; g < num_gpus_; g++) {
      HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
      ib_comm_->pre_intra_update_a2a_coll_sizes(coll_handle_, d_send_sizes_ptrs_[g].get_ptr(), 0,
                                                g);
    }
  }

  void do_intra_node_a2a_bare(size_t g, cudaStream_t stream) {
    intra_node_a2a<TypeEmbeddingComp><<<96, 256, 0, stream>>>(
        d_send_buffs_[g].get_ptr(), d_interim_send_buffs_ptrs_[g].get_ptr(),
        d_send_sizes_[g].get_ptr(), max_elems_per_dest_, num_gpus_, num_procs_, g);
  }

  void do_inter_node_a2a_bare(size_t g, cudaStream_t stream) {
    ib_comm_->post_send_command_a2a<TypeEmbeddingComp>(coll_handle_, stream, g);
    HCTR_LIB_THROW(cudaEventRecord(comm_events_[g], comm_stream_[g]));
    HCTR_LIB_THROW(cudaStreamWaitEvent(stream, comm_events_[g]));
  }

  void do_device_a2a_bare(size_t g, cudaStream_t stream) {
    do_intra_node_a2a_bare(g, stream);
    do_inter_node_a2a_bare(g, stream);
  }

  void do_device_a2a() {
    auto& device_list = resource_manager_->get_local_gpu_device_id_list();
    if (use_cuda_graph_) {
      if (!integ_graph_captured_) {
        integ_graph_captured_ = true;
        integ_graph_.resize(num_gpus_);
        integ_graph_instance_.resize(num_gpus_);
        for (size_t g = 0; g < num_gpus_; g++) {
          auto& stream = resource_manager_->get_local_gpu(g)->get_stream();
          HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
          HCTR_LIB_THROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
          do_device_a2a_bare(g, stream);
          HCTR_LIB_THROW(cudaStreamEndCapture(stream, &integ_graph_[g]));
          HCTR_LIB_THROW(
              cudaGraphInstantiate(&integ_graph_instance_[g], integ_graph_[g], NULL, NULL, 0));
        }
      }
#pragma omp parallel num_threads(num_gpus_)
      {
        int g = omp_get_thread_num();
        auto& stream = resource_manager_->get_local_gpu(g)->get_stream();
        HCTR_LIB_THROW(cudaGraphLaunch(integ_graph_instance_[g], stream));
      }
    } else {
      for (size_t g = 0; g < num_gpus_; g++) {
        auto& stream = resource_manager_->get_local_gpu(g)->get_stream();
        HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
        do_device_a2a_bare(g, stream);
      }
    }
  }

  void do_intra_a2a() {
    auto& device_list = resource_manager_->get_local_gpu_device_id_list();
    if (use_cuda_graph_) {
      auto& device_list = resource_manager_->get_local_gpu_device_id_list();
      if (!intra_graph_captured_) {
        intra_graph_captured_ = true;
        intra_graph_.resize(num_gpus_);
        intra_graph_instance_.resize(num_gpus_);
        for (size_t g = 0; g < num_gpus_; g++) {
          auto& stream = resource_manager_->get_local_gpu(g)->get_stream();
          HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
          HCTR_LIB_THROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

          do_intra_node_a2a_bare(g, stream);

          HCTR_LIB_THROW(cudaStreamEndCapture(stream, &intra_graph_[g]));
          HCTR_LIB_THROW(
              cudaGraphInstantiate(&intra_graph_instance_[g], intra_graph_[g], NULL, NULL, 0));
        }
      }
#pragma omp parallel num_threads(num_gpus_)
      {
        int g = omp_get_thread_num();
        auto& stream = resource_manager_->get_local_gpu(g)->get_stream();
        HCTR_LIB_THROW(cudaGraphLaunch(intra_graph_instance_[g], stream));
      }
    } else {
      for (size_t g = 0; g < num_gpus_; g++) {
        auto& stream = resource_manager_->get_local_gpu(g)->get_stream();
        HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
        do_intra_node_a2a_bare(g, stream);
      }
    }
  }

  void do_inter_a2a() {
    auto& device_list = resource_manager_->get_local_gpu_device_id_list();
    if (use_cuda_graph_) {
      if (!inter_graph_captured_) {
        inter_graph_captured_ = true;
        inter_graph_.resize(num_gpus_);
        inter_graph_instance_.resize(num_gpus_);
        for (size_t g = 0; g < num_gpus_; g++) {
          auto& stream = resource_manager_->get_local_gpu(g)->get_stream();
          HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
          HCTR_LIB_THROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
          do_inter_node_a2a_bare(g, stream);
          HCTR_LIB_THROW(cudaStreamEndCapture(stream, &inter_graph_[g]));
          HCTR_LIB_THROW(
              cudaGraphInstantiate(&inter_graph_instance_[g], inter_graph_[g], NULL, NULL, 0));
        }
      }
#pragma omp parallel num_threads(num_gpus_)
      {
        int g = omp_get_thread_num();
        auto& stream = resource_manager_->get_local_gpu(g)->get_stream();
        HCTR_LIB_THROW(cudaGraphLaunch(inter_graph_instance_[g], stream));
      }
    } else {
      for (size_t g = 0; g < num_gpus_; g++) {
        auto& stream = resource_manager_->get_local_gpu(g)->get_stream();
        HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
        do_inter_node_a2a_bare(g, stream);
      }
    }
  }

  void stream_sync_all() {
    auto& device_list = resource_manager_->get_local_gpu_device_id_list();
    for (size_t g = 0; g < num_gpus_; g++) {
      HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
      auto& stream = resource_manager_->get_local_gpu(g)->get_stream();
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    }
  }

  void do_nccl_a2a() {
    size_t num_dest = num_procs_ * num_gpus_;
    auto& device_list = resource_manager_->get_local_gpu_device_id_list();
    HCTR_LIB_THROW(ncclGroupStart());
    for (size_t s = 0; s < num_gpus_; s++) {
      const auto& local_gpu = resource_manager_->get_local_gpu(s);
      HCTR_LIB_THROW(cudaSetDevice(device_list[s]));
      for (size_t d = 0; d < num_dest; d++) {
        HCTR_LIB_THROW(ncclSend(d_send_buffs_[s].get_ptr() + d * max_elems_per_dest_,
                                (h_send_sizes_[s].get_ptr())[d], ncclChar, d, local_gpu->get_nccl(),
                                0));
        HCTR_LIB_THROW(ncclRecv(d_recv_buffs_[s].get_ptr() + d * max_elems_per_dest_,
                                (h_recv_sizes_[s].get_ptr())[d], ncclChar, d, local_gpu->get_nccl(),
                                0));
      }
    }
    HCTR_LIB_THROW(ncclGroupEnd());

    for (size_t g = 0; g < num_gpus_; g++) {
      HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
      HCTR_LIB_THROW(cudaMemcpy(h_recv_buffs_[g].get_ptr(), d_recv_buffs_[g].get_ptr(),
                                max_elems_ * sizeof(TypeEmbeddingComp), cudaMemcpyDeviceToHost));
      HCTR_LIB_THROW(
          cudaMemset(d_recv_buffs_[g].get_ptr(), 0, max_elems_ * sizeof(TypeEmbeddingComp)));
      HCTR_LIB_THROW(cudaDeviceSynchronize());
    }
  }

  void compare_host_and_device() {
    auto& device_list = resource_manager_->get_local_gpu_device_id_list();
    for (size_t g = 0; g < num_gpus_; g++) {
      HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
      HCTR_LIB_THROW(cudaMemcpy(h_recv_buffs_out_[g].get_ptr(), d_recv_buffs_[g].get_ptr(),
                                max_elems_ * sizeof(TypeEmbeddingComp), cudaMemcpyDeviceToHost));
    }

    for (size_t g = 0; g < num_gpus_; g++) {
      for (size_t e = 0; e < max_elems_; e++) {
        if (*(h_recv_buffs_[g].get_ptr() + e) != *(h_recv_buffs_out_[g].get_ptr() + e)) {
          size_t my_proc = resource_manager_->get_process_id();
          HCTR_LOG_S(DEBUG, WORLD)
              << my_proc << ": Data mismatch at gpu " << g << " element: " << e
              << " expected: " << *(h_recv_buffs_[g].get_ptr() + e)
              << " got: " << *(h_recv_buffs_out_[g].get_ptr() + e) << std::endl;
          exit(1);
        }
      }
    }
  }

  void do_perf_test() {
    size_t my_proc = resource_manager_->get_process_id();

    if (my_proc == 0) {
      HCTR_LOG_S(DEBUG, WORLD) << "intra A2A bench:" << std::endl;
      HCTR_LOG_S(DEBUG, WORLD) << "size(in B)  time(in us)" << std::endl;
    }
    for (size_t size = 1024; size < max_size_; size *= 2) {
      double bench_time;
      gen_uniform_size(size);
      fill_buffers();
      TIMEIT(do_intra_a2a(), bench_time);
      if (my_proc == 0) {
        HCTR_LOG_S(DEBUG, WORLD) << size << " " << bench_time << std::endl;
      }
    }

    if (my_proc == 0) {
      HCTR_LOG_S(DEBUG, WORLD) << "inter A2A bench:" << std::endl;
      HCTR_LOG_S(DEBUG, WORLD) << "size(in B)  time(in us)" << std::endl;
    }
    for (size_t size = 1024; size < max_size_; size *= 2) {
      double bench_time;
      gen_uniform_size(size);
      fill_buffers();
      TIMEIT(do_inter_a2a(), bench_time);
      if (my_proc == 0) {
        HCTR_LOG_S(DEBUG, WORLD) << size << " " << bench_time << std::endl;
      }
    }

    if (my_proc == 0) {
      HCTR_LOG_S(DEBUG, WORLD) << "intra+inter A2A bench:" << std::endl;
      HCTR_LOG_S(DEBUG, WORLD) << "size(in B)  time(in us)" << std::endl;
    }
    for (size_t size = 1024; size < max_size_; size *= 2) {
      double bench_time;
      gen_uniform_size(size);
      fill_buffers();
      TIMEIT(do_device_a2a(), bench_time);
      if (my_proc == 0) {
        HCTR_LOG_S(DEBUG, WORLD) << size << " " << bench_time << std::endl;
      }
    }
  }

 private:
  size_t num_gpus_;
  size_t max_size_;
  size_t max_elems_;
  size_t max_elems_per_gpu_;
  size_t max_elems_per_proc_;
  size_t max_elems_per_dest_;
  size_t max_size_per_gpu_;
  int num_procs_ = 1;
  bool use_cuda_graph_ = false;

  std::vector<cudaStream_t> comm_stream_;
  std::vector<cudaEvent_t> comm_events_;
  std::vector<cudaGraph_t> integ_graph_;
  std::vector<cudaGraphExec_t> integ_graph_instance_;
  bool integ_graph_captured_ = false;

  std::vector<cudaGraph_t> intra_graph_;
  std::vector<cudaGraphExec_t> intra_graph_instance_;
  bool intra_graph_captured_ = false;

  std::vector<cudaGraph_t> inter_graph_;
  std::vector<cudaGraphExec_t> inter_graph_instance_;
  bool inter_graph_captured_ = false;

  std::shared_ptr<ResourceManager> resource_manager_;
  IbComm* ib_comm_;  // TODO: Make it shared so we have only one instance of ibcomm
  HierA2AvCollHandle coll_handle_;

  std::vector<Tensor2<TypeEmbeddingComp>> h_send_buffs_;
  std::vector<Tensor2<TypeEmbeddingComp>> h_recv_buffs_;

  std::vector<Tensor2<TypeEmbeddingComp>> h_recv_buffs_out_;

  std::vector<Tensor2<TypeEmbeddingComp>> d_send_buffs_;
  std::vector<Tensor2<TypeEmbeddingComp>> d_interim_send_buffs_;
  std::vector<Tensor2<TypeEmbeddingComp*>> d_interim_send_buffs_ptrs_;
  std::vector<TypeEmbeddingComp*> h_d_interim_send_buffs_ptrs_;
  std::vector<Tensor2<TypeEmbeddingComp>> d_recv_buffs_;

  std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> dev_bufs_;
  std::vector<std::shared_ptr<GeneralBuffer2<CudaHostAllocator>>> host_bufs_;

  std::vector<Tensor2<size_t>> h_send_sizes_;
  std::vector<Tensor2<size_t>> h_recv_sizes_;

  std::vector<Tensor2<size_t>> d_send_sizes_;
  std::vector<size_t*> h_d_send_sizes_ptrs_;
  std::vector<Tensor2<size_t*>> d_send_sizes_ptrs_;
  std::vector<Tensor2<size_t>> d_recv_sizes_;

  void init_buffers() {
    coll_handle_ = ib_comm_->register_hier_a2a_v_coll();
    h_send_buffs_.resize(num_gpus_);
    h_recv_buffs_.resize(num_gpus_);
    h_recv_buffs_out_.resize(num_gpus_);
    d_send_buffs_.resize(num_gpus_);
    d_interim_send_buffs_.resize(num_gpus_);
    d_recv_buffs_.resize(num_gpus_);

    dev_bufs_.resize(num_gpus_);
    host_bufs_.resize(num_gpus_);

    h_send_sizes_.resize(num_gpus_);
    d_send_sizes_.resize(num_gpus_);
    h_recv_sizes_.resize(num_gpus_);
    d_recv_sizes_.resize(num_gpus_);

    d_send_sizes_ptrs_.resize(num_gpus_);
    h_d_send_sizes_ptrs_.resize(num_gpus_);
    d_interim_send_buffs_ptrs_.resize(num_gpus_);
    h_d_interim_send_buffs_ptrs_.resize(num_gpus_);

    CudaDeviceContext context;
    for (size_t g = 0; g < num_gpus_; g++) {
      auto& device_list = resource_manager_->get_local_gpu_device_id_list();
      context.set_device(device_list[g]);
      dev_bufs_[g] = GeneralBuffer2<CudaAllocator>::create();
      host_bufs_[g] = GeneralBuffer2<CudaHostAllocator>::create();

      dev_bufs_[g]->reserve({max_elems_}, &d_send_buffs_[g]);
      dev_bufs_[g]->reserve({max_elems_}, &d_interim_send_buffs_[g]);
      dev_bufs_[g]->reserve({max_elems_}, &d_recv_buffs_[g]);
      dev_bufs_[g]->reserve({num_gpus_ * num_procs_}, &d_send_sizes_[g]);
      dev_bufs_[g]->reserve({num_gpus_ * num_procs_}, &d_recv_sizes_[g]);
      dev_bufs_[g]->reserve({num_gpus_}, &d_send_sizes_ptrs_[g]);
      dev_bufs_[g]->reserve({num_gpus_}, &d_interim_send_buffs_ptrs_[g]);
      dev_bufs_[g]->allocate();

      h_d_send_sizes_ptrs_[g] = d_send_sizes_[g].get_ptr();
      h_d_interim_send_buffs_ptrs_[g] = d_interim_send_buffs_[g].get_ptr();

      host_bufs_[g]->reserve({max_elems_}, &h_send_buffs_[g]);
      host_bufs_[g]->reserve({max_elems_}, &h_recv_buffs_[g]);
      host_bufs_[g]->reserve({max_elems_}, &h_recv_buffs_out_[g]);
      host_bufs_[g]->reserve({num_gpus_ * num_procs_}, &h_send_sizes_[g]);
      host_bufs_[g]->reserve({num_gpus_ * num_procs_}, &h_recv_sizes_[g]);
      host_bufs_[g]->allocate();

      ib_comm_->set_a2a_coll_buf(coll_handle_, (void*)d_interim_send_buffs_[g].get_ptr(),
                                 max_elems_ * sizeof(TypeEmbeddingComp),
                                 (void*)d_recv_buffs_[g].get_ptr(),
                                 max_elems_ * sizeof(TypeEmbeddingComp), g);
    }

    for (size_t g = 0; g < num_gpus_; g++) {
      auto& device_list = resource_manager_->get_local_gpu_device_id_list();
      context.set_device(device_list[g]);
      HCTR_LIB_THROW(cudaMemcpy(d_send_sizes_ptrs_[g].get_ptr(), (void*)h_d_send_sizes_ptrs_.data(),
                                num_gpus_ * sizeof(size_t*), cudaMemcpyHostToDevice));
      HCTR_LIB_THROW(cudaMemcpy(d_interim_send_buffs_ptrs_[g].get_ptr(),
                                (void*)h_d_interim_send_buffs_ptrs_.data(),
                                num_gpus_ * sizeof(TypeEmbeddingComp*), cudaMemcpyHostToDevice));
    }

    ib_comm_->register_a2a_coll_buf(coll_handle_);
    ib_comm_->set_ready_to_transfer();
  }
};

template <typename TypeEmbeddingComp>
void test_ib_comm(const std::vector<int>& device_list, bool use_cuda_graph = false) {
  const int num_procs{core23::MpiInitService::get().world_size()};
  if (num_procs == 1) return;

  const size_t MAX_SIZE = 16 * 1024 * 1024;
  IbCommsTest<TypeEmbeddingComp> test(device_list, MAX_SIZE, use_cuda_graph);

  // Uniform size test
  for (size_t size = 1024; size < MAX_SIZE; size *= 2) {
    test.gen_uniform_size(size);
    test.fill_buffers();
    test.do_nccl_a2a();
    test.do_device_a2a();
    test.stream_sync_all();
    test.compare_host_and_device();
  }

  // Random size test
  for (int i = 0; i < 10; i++) {
    test.gen_rand_size();
    test.fill_buffers();
    test.do_nccl_a2a();
    test.do_device_a2a();
    test.stream_sync_all();
    test.compare_host_and_device();
  }
}

template <typename TypeEmbeddingComp>
void test_ib_comm_perf(const std::vector<int>& device_list, bool use_cuda_graph = false) {
  const int num_procs{core23::MpiInitService::get().world_size()};
  if (num_procs == 1) return;

  const size_t MAX_SIZE = 16 * 1024 * 1024;
  IbCommsTest<TypeEmbeddingComp> test(device_list, MAX_SIZE, use_cuda_graph);

  // Perf test
  test.do_perf_test();
}

}  // namespace

TEST(ib_comms_a2a_v_integ_test, fp_1gpu_per_node) { test_ib_comm<float>({0}); }
TEST(ib_comms_a2a_v_integ_test, fp_4gpu_per_node) { test_ib_comm<float>({0, 2, 4, 7}); }
TEST(ib_comms_a2a_v_integ_test, fp_8gpu_per_node) { test_ib_comm<float>({0, 1, 2, 3, 4, 5, 6, 7}); }
TEST(ib_comms_a2a_v_integ_test, fp_1gpu_per_node_graph) { test_ib_comm<float>({0}, true); }
TEST(ib_comms_a2a_v_integ_test, fp_4gpu_per_node_graph) { test_ib_comm<float>({0, 2, 4, 7}, true); }
TEST(ib_comms_a2a_v_integ_test, fp_8gpu_per_node_graph) {
  test_ib_comm<float>({0, 1, 2, 3, 4, 5, 6, 7}, true);
}
TEST(ib_comms_a2a_v_integ_test, fp_1gpu_per_node_perf) { test_ib_comm_perf<float>({0}, true); }
TEST(ib_comms_a2a_v_integ_test, fp_4gpu_per_node_perf) {
  test_ib_comm_perf<float>({0, 2, 4, 7}, true);
}
TEST(ib_comms_a2a_v_integ_test, fp_8gpu_per_node_perf) {
  test_ib_comm_perf<float>({0, 1, 2, 3, 4, 5, 6, 7}, true);
}
#endif
