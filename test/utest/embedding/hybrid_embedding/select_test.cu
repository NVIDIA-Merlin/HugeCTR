#include <stdint.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "HugeCTR/include/embeddings/hybrid_embedding/select.cuh"
#include "gtest/gtest.h"
#include "utest/test_utils.h"
// #include <cub/cub.cuh>

using namespace HugeCTR;

namespace Predict {
template <typename T>
struct is_odd {
  __host__ __device__ __forceinline__ bool operator()(const T &a) const { return (a & 1); }
  is_odd() = default;
};
}  // namespace Predict

template <typename T>
void check(std::vector<T> &h_ref, std::vector<T> &h_gpu) {
  for (size_t i = 0; i < h_ref.size(); i++) {
    if (h_ref[i] != h_gpu[i]) {
      std::cerr << " error at index " << i << std::endl;
      exit(-1);
    }
  }
  std::cout << "check pass" << std::endl;
}
template <typename T, typename Pred>
struct SelectTest {
  Pred Op_;
  size_t len_;
  std::vector<T> keys_;
  std::vector<T> ref_cpu_;
  std::vector<T> ref_gpu_;
  T *d_keys_;
  T *d_output_;
  T *d_num_selected_out_;
  T ref_count_;

  void gather_if(const std::vector<T> &input, std::vector<T> &output) {
    output.clear();
    if (input.empty()) {
      for (size_t i = 0; i < len_; i++) {
        if (Op_(i)) {
          output.push_back(i);
        }
      }
    } else {
      for (auto in : input) {
        if (Op_(in)) {
          output.push_back(in);
        }
      }
    }
  }

  SelectTest(size_t len, Pred Op, bool no_input = false) : len_(len), Op_(Op), ref_count_(0) {
    if (!no_input) {
      cudaMalloc((void **)(&d_keys_), sizeof(T) * len);
      keys_.resize(len, 0);
      for (size_t i = 0; i < keys_.size(); i++) {
        keys_[i] = std::rand();
      }
      std::cout << "keys init done" << std::endl;
    } else {
      d_keys_ = nullptr;
      keys_.clear();
    }
    cudaMalloc((void **)(&d_num_selected_out_), sizeof(T));
    cudaMalloc((void **)(&d_output_), sizeof(T) * len);
  }

  void test() {
    if (d_keys_) {
      cudaMemcpy(d_keys_, keys_.data(), sizeof(T) * len_, cudaMemcpyHostToDevice);
    }
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    HugeCTR::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_keys_, d_output_,
                              d_num_selected_out_, len_, Op_);
    std::cout << "temp storage bytes\n" << temp_storage_bytes << std::endl;
    cudaMalloc((void **)&d_temp_storage, temp_storage_bytes);
    HugeCTR::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_keys_, d_output_,
                              d_num_selected_out_, len_, Op_);
    cudaDeviceSynchronize();
    cudaMemcpy(&ref_count_, d_num_selected_out_, sizeof(T), cudaMemcpyDeviceToHost);
    gather_if(keys_, ref_cpu_);
    if (ref_count_ != static_cast<T>(ref_cpu_.size())) {
      std::cerr << "seleted num mismatches\n" << std::endl;
      std::cerr << "expected: " << ref_cpu_.size() << " got " << ref_count_ << std::endl;
      exit(-1);
    }
    std::cout << "get num_selected " << ref_count_ << std::endl;
    ref_gpu_.resize(ref_count_);
    cudaMemcpy(ref_gpu_.data(), d_output_, sizeof(T) * ref_gpu_.size(), cudaMemcpyDeviceToHost);
    check(ref_cpu_, ref_gpu_);
    cudaFree(d_temp_storage);
  }
  ~SelectTest() {
    if (d_keys_) {
      cudaFree(d_keys_);
    }
    cudaFree(d_num_selected_out_);
    cudaFree(d_output_);
  }
};

TEST(select, is_odd_31) {
  SelectTest<size_t, Predict::is_odd<size_t>> select_test((1ul << 32), Predict::is_odd<size_t>());
  select_test.test();
}
TEST(select, counting) {
  SelectTest<size_t, Predict::is_odd<size_t>> select_test((1ul << 20), Predict::is_odd<size_t>(),
                                                          true);
  select_test.test();
}
TEST(select, large_counting) {
  SelectTest<size_t, Predict::is_odd<size_t>> select_test((1ul << 31), Predict::is_odd<size_t>(),
                                                          true);
  select_test.test();
}