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

#ifndef ENABLE_MPI
#include <gtest/gtest.h>

#include <collectives/collective.hpp>
#include <common.hpp>
#include <general_buffer2.hpp>
#include <random>
#include <resource_managers/resource_manager_core.hpp>
#include <tensor2.hpp>
#include <type_traits>
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
ncclDataType_t get_nccl_type<__half>() {
  return ncclFloat16;
}
template <>
ncclDataType_t get_nccl_type<uint32_t>() {
  return ncclUint32;
}

template <typename T>
bool compare_values(T a, T b, std::false_type const&) {
  T epsilon = 0.1;
  if (a > b) {
    return (a - b) < epsilon;
  } else {
    return (b - a) < epsilon;
  }
  return false;
}

template <typename T>
bool compare_values(T a, T b, std::true_type const&) {
  return (a == b);
}

template <typename T>
bool compare_values(T a, T b) {
  return compare_values<T>(a, b, std::is_integral<T>{});
}

template <typename TypeEmbeddingComp>
struct arTest {
 public:
  arTest(const std::vector<int>& device_list, size_t max_size)
      : num_gpus_(device_list.size()), max_size_(max_size) {
    max_elems_ = max_size_ / sizeof(TypeEmbeddingComp);

    std::vector<std::vector<int>> vvgpu;
    vvgpu.push_back(device_list);

    if (std::is_same<TypeEmbeddingComp, float>::value) {
      use_mixed_precision_ = false;
    } else {
      use_mixed_precision_ = true;
    }

    resource_manager_ = ResourceManagerCore::create(vvgpu, 0, DeviceMap::LOCAL_FIRST);
    collective_manager_ = std::make_shared<CollectiveManager>(resource_manager_);
    collective_manager_->set_ar_comm(AllReduceAlgo::ONESHOT, use_mixed_precision_);
    ar_comm_ = collective_manager_->get_ar_comm();
    init_buffers();
  }

 private:
  size_t num_gpus_;
  size_t max_size_;
  size_t max_elems_;
  bool use_mixed_precision_;
  AllReduceInPlaceComm* ar_comm_;
  std::shared_ptr<ResourceManager> resource_manager_;
  std::shared_ptr<CollectiveManager> collective_manager_;

  std::vector<Tensor2<TypeEmbeddingComp>> h_ar_buff_;
  std::vector<Tensor2<TypeEmbeddingComp>> d_ar_buff_;
  std::vector<Tensor2<TypeEmbeddingComp>> d_ar_buff_ref_;
  std::vector<Tensor2<TypeEmbeddingComp>> h_ar_buff_out_;
  std::vector<Tensor2<TypeEmbeddingComp>> h_ar_buff_out_ref_;

  std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> dev_bufs_;
  std::vector<std::shared_ptr<GeneralBuffer2<CudaHostAllocator>>> host_bufs_;

  std::vector<size_t> ar_sizes_;
  std::vector<AllReduceInPlaceComm::Handle> ar_handles_;

  void init_buffers() {
    dev_bufs_.resize(num_gpus_);
    host_bufs_.resize(num_gpus_);

    h_ar_buff_.resize(num_gpus_);
    d_ar_buff_.resize(num_gpus_);
    d_ar_buff_ref_.resize(num_gpus_);
    h_ar_buff_out_.resize(num_gpus_);
    h_ar_buff_out_ref_.resize(num_gpus_);

    CudaDeviceContext context;
    for (size_t g = 0; g < num_gpus_; g++) {
      auto& device_list = resource_manager_->get_local_gpu_device_id_list();
      context.set_device(device_list[g]);
      dev_bufs_[g] = GeneralBuffer2<CudaAllocator>::create();
      host_bufs_[g] = GeneralBuffer2<CudaHostAllocator>::create();

      dev_bufs_[g]->reserve({max_elems_}, &d_ar_buff_[g]);
      dev_bufs_[g]->reserve({max_elems_}, &d_ar_buff_ref_[g]);
      dev_bufs_[g]->allocate();

      host_bufs_[g]->reserve({max_elems_}, &h_ar_buff_[g]);
      host_bufs_[g]->reserve({max_elems_}, &h_ar_buff_out_[g]);
      host_bufs_[g]->reserve({max_elems_}, &h_ar_buff_out_ref_[g]);
      host_bufs_[g]->allocate();
    }
  }

  void fill_buffers() {
    // reset output buffers
    for (size_t g = 0; g < num_gpus_; g++) {
      memset(h_ar_buff_out_[g].get_ptr(), 0, max_size_);
      memset(h_ar_buff_out_ref_[g].get_ptr(), 0, max_size_);
    }

    for (size_t g = 0; g < num_gpus_; g++) {
      for (size_t s = 0; s < max_elems_; s++) {
        float tmp_num = s % 31 + 41 / (g + 1);
        *(h_ar_buff_[g].get_ptr() + s) = tmp_num;
        // *(h_ar_buff_[g].get_ptr() + s) = g;
      }
    }

    auto& device_list = resource_manager_->get_local_gpu_device_id_list();
    for (size_t g = 0; g < num_gpus_; g++) {
      HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
      HCTR_LIB_THROW(cudaMemcpy(d_ar_buff_[g].get_ptr(), h_ar_buff_[g].get_ptr(), max_size_,
                                cudaMemcpyHostToDevice));
      HCTR_LIB_THROW(cudaMemcpy(d_ar_buff_ref_[g].get_ptr(), h_ar_buff_[g].get_ptr(), max_size_,
                                cudaMemcpyHostToDevice));
    }
  }

  void gen_uniform_sizes() {
    for (size_t mysize = 1024; mysize <= max_size_; mysize = mysize * 2) {
      // make sure size is aligned to 16B* num_gpus_
      auto size_aligned = (mysize / (16 * num_gpus_)) * (16 * num_gpus_);
      ar_sizes_.push_back(size_aligned);
    }
  }

  void gen_rand_sizes() {
    int max_rand_sizes = 20;
    int rand_sizes = 0;
    while (rand_sizes < max_rand_sizes) {
      std::default_random_engine generator;
      uniform_distribution_t<size_t> distribution(1, max_size_);
      size_t size = distribution(generator);
      auto size_aligned = (size / (16 * num_gpus_)) * (16 * num_gpus_);
      if (size_aligned > 0) {
        ar_sizes_.push_back(size_aligned);
        rand_sizes++;
      }
    }
  }

  void register_buffers() {
    for (auto& size : ar_sizes_) {
      auto handle = ar_comm_->register_coll();
      ar_handles_.push_back(handle);
      for (size_t g = 0; g < num_gpus_; g++) {
        ar_comm_->set_coll_buf(handle, d_ar_buff_[g].get_ptr(), size, g);
      }
      ar_comm_->register_coll_buf(handle);
    }
  }

  void stream_sync_all() {
    auto& device_list = resource_manager_->get_local_gpu_device_id_list();
    for (size_t g = 0; g < num_gpus_; g++) {
      const auto& local_gpu = resource_manager_->get_local_gpu(g);
      HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
      HCTR_LIB_THROW(cudaStreamSynchronize(local_gpu->get_stream()));
    }
  }

  void do_nccl_ar(int i) {
    size_t size = ar_sizes_[i];
    auto& device_list = resource_manager_->get_local_gpu_device_id_list();
#pragma omp parallel for num_threads(num_gpus_)
    for (size_t g = 0; g < num_gpus_; g++) {
      const auto& local_gpu = resource_manager_->get_local_gpu(g);
      HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
      HCTR_LIB_THROW(ncclAllReduce(
          (const void*)d_ar_buff_ref_[g].get_ptr(), (void*)d_ar_buff_ref_[g].get_ptr(),
          size / sizeof(TypeEmbeddingComp), get_nccl_type<TypeEmbeddingComp>(), ncclSum,
          local_gpu->get_nccl(), local_gpu->get_stream()));
    }
  }

  void do_custom_ar(int i) {
    auto handle = ar_handles_[i];
    auto& device_list = resource_manager_->get_local_gpu_device_id_list();
#pragma omp parallel for num_threads(num_gpus_)
    for (size_t g = 0; g < num_gpus_; g++) {
      const auto& local_gpu = resource_manager_->get_local_gpu(g);
      HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
      ar_comm_->all_reduce(handle, local_gpu->get_stream(), g);
    }
  }

  void compare_outputs() {
    auto& device_list = resource_manager_->get_local_gpu_device_id_list();
    for (size_t g = 0; g < num_gpus_; g++) {
      const auto& local_gpu = resource_manager_->get_local_gpu(g);
      HCTR_LIB_THROW(cudaSetDevice(device_list[g]));
      HCTR_LIB_THROW(cudaMemcpyAsync(h_ar_buff_out_[g].get_ptr(), d_ar_buff_[g].get_ptr(),
                                     max_size_, cudaMemcpyDeviceToHost, local_gpu->get_stream()));

      HCTR_LIB_THROW(cudaMemcpyAsync(h_ar_buff_out_ref_[g].get_ptr(), d_ar_buff_ref_[g].get_ptr(),
                                     max_size_, cudaMemcpyDeviceToHost, local_gpu->get_stream()));
    }
    stream_sync_all();
    for (size_t g = 0; g < num_gpus_; g++) {
      for (size_t e = 0; e < max_elems_; e++) {
        bool match = compare_values(*(h_ar_buff_out_[g].get_ptr() + e),
                                    *(h_ar_buff_out_ref_[g].get_ptr() + e));
        if (!match) {
          size_t my_proc = resource_manager_->get_process_id();
          float expected_num = *(h_ar_buff_out_ref_[g].get_ptr() + e);
          float wrong_num = *(h_ar_buff_out_[g].get_ptr() + e);

          HCTR_LOG_S(DEBUG, WORLD)
              << my_proc << ": Data mismatch at gpu " << g << " element: " << e
              << " expected: " << expected_num << " got: " << wrong_num << std::endl;
        }
      }
    }
  }

 public:
  void test() {
    gen_uniform_sizes();
    gen_rand_sizes();
    register_buffers();
    int repeat = 2;
    for (int r = 0; r < repeat; r++) {
      for (size_t s = 0; s < ar_sizes_.size(); s++) {
        fill_buffers();
        do_nccl_ar(s);
        stream_sync_all();
        do_custom_ar(s);
        stream_sync_all();
        if (std::is_same<TypeEmbeddingComp, float>::value) {
          compare_outputs();
        }
      }
    }
  }

  void perf_test() {
    size_t my_proc = resource_manager_->get_process_id();
    gen_uniform_sizes();
    register_buffers();
    for (size_t s = 0; s < ar_sizes_.size(); s++) {
      double bench_time;
      auto size = ar_sizes_[s];
      TIMEIT(do_custom_ar(s), bench_time);
      if (my_proc == 0) {
        HCTR_LOG_S(DEBUG, WORLD) << size << " " << bench_time << std::endl;
      }
    }
  }
};

template <typename TypeEmbeddingComp>
void test_ar_comm(const std::vector<int>& device_list) {
  const size_t MAX_SIZE = 64 * 1024 * 1024;
  arTest<TypeEmbeddingComp> test(device_list, MAX_SIZE);
  test.test();
}

template <typename TypeEmbeddingComp>
void test_ar_comm_perf(const std::vector<int>& device_list) {
  const size_t MAX_SIZE = 64 * 1024 * 1024;
  arTest<TypeEmbeddingComp> test(device_list, MAX_SIZE);
  test.perf_test();
}

TEST(ar_oneshot_test, half_2gpu) { test_ar_comm<__half>({0, 1}); }
TEST(ar_oneshot_test, half_4gpu) { test_ar_comm<__half>({0, 1, 2, 3}); }
TEST(ar_oneshot_test, half_8gpu) { test_ar_comm<__half>({0, 1, 2, 3, 4, 5, 6, 7}); }
TEST(ar_oneshot_test, float_2gpu) { test_ar_comm<float>({0, 1}); }
TEST(ar_oneshot_test, float_4gpu) { test_ar_comm<float>({0, 1, 2, 3}); }
TEST(ar_oneshot_test, float_8gpu) { test_ar_comm<float>({0, 1, 2, 3, 4, 5, 6, 7}); }
TEST(ar_oneshot_perf, float_2gpu) { test_ar_comm_perf<float>({0, 1}); }
TEST(ar_oneshot_perf, float_4gpu) { test_ar_comm_perf<float>({0, 1, 2, 3}); }
TEST(ar_oneshot_perf, float_8gpu) { test_ar_comm_perf<float>({0, 1, 2, 3, 4, 5, 6, 7}); }

}  // namespace

#endif
