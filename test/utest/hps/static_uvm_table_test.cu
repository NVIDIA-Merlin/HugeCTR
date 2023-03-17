#include <bits/stdc++.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <gpu_cache/include/uvm_table.hpp>

#include "key_generator.cuh"
using namespace std;

size_t GiB = 1024 * 1024 * 1024;

class Timer {
  std::chrono::steady_clock::time_point start_;

 public:
  void start() { start_ = std::chrono::steady_clock::now(); };
  float end() {
    auto end = std::chrono::steady_clock::now();
    float tmp_time =
        (float)std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count() / 1e6;
    return tmp_time;
  }
};

struct TableParam {
  size_t max_key;
  size_t num_keys;
  size_t ev_size;
};

struct LookupParam {
  size_t batch_size;
  size_t num_hotness;
};

template <typename Key_t, typename Value_t>
__global__ void distribute_values(const Key_t *keys, int num_keys, const Value_t *buffer,
                                  Value_t *dst_values, int ev_size) {
  for (size_t idx = threadIdx.x; idx < num_keys; idx += blockDim.x) {
    *(float4 *)(dst_values + idx * ev_size) = *(float4 *)(buffer + idx * ev_size);
  }
}

constexpr int num_blocks = 2;
constexpr float percent = 0.1;

template <typename Key_t, typename Value_t>
class UvmTest {
 public:
  long long total_memory_bytes{0};
  double total_time{0};

  UvmTest(TableParam table_param, LookupParam lookup_param)
      : table_param_(table_param),
        lookup_param_(lookup_param),
        num_query_keys_(lookup_param.batch_size * lookup_param.num_hotness),
        table_(table_param.num_keys * percent,
               table_param.num_keys - table_param.num_keys * percent, num_query_keys_,
               table_param.ev_size),
        key_generator_(lookup_param.batch_size, lookup_param.num_hotness, 1.2, table_param.num_keys,
                       table_param.num_keys) {
    srand(time(NULL));
    init_keys();
    CUDA_CHECK(
        cudaMallocManaged(&m_values_, sizeof(Value_t) * table_param.num_keys * table_param.ev_size))
    for (size_t i = 0; i < table_param.num_keys; i++) {
      for (size_t j = 0; j < table_param.ev_size; j++) {
        m_values_[i * table_param.ev_size + j] = h_keys_[i];
      }
    }
    CUDA_CHECK(cudaMalloc(&d_query_keys_, sizeof(*d_query_keys_) * num_query_keys_));
    for (int i = 0; i < num_blocks; i++) {
      num_query_keys_per_block_ = num_query_keys_ / num_blocks;
      CUDA_CHECK(cudaMallocHost(&h_buffers_[i],
                                sizeof(Value_t) * num_query_keys_per_block_ * table_param.ev_size));
      CUDA_CHECK(cudaMalloc(&d_buffers_[i],
                            sizeof(Value_t) * num_query_keys_per_block_ * table_param.ev_size));
      CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }
    CUDA_CHECK(cudaMalloc(&d_query_values_,
                          sizeof(*d_query_values_) * num_query_keys_ * table_param.ev_size));

    CUDA_CHECK(cudaDeviceSynchronize());

    size_t init_batch_size = table_param_.num_keys / 10;
    size_t iterations = (table_param_.num_keys - 1) / init_batch_size + 1;
    for (size_t i = 0; i < iterations; i++) {
      size_t this_batch_size = init_batch_size;
      if (i == iterations - 1) {
        this_batch_size = table_param_.num_keys - init_batch_size * i;
      }
      table_.add(h_keys_ + i * init_batch_size,
                 m_values_ + i * init_batch_size * table_param.ev_size, this_batch_size);
    }
  }

  void run(bool record_perf = false) {
    auto keys = get_random_keys();
    CUDA_CHECK(cudaMemcpy(d_query_keys_, keys.data(), sizeof(*d_query_keys_) * num_query_keys_,
                          cudaMemcpyHostToDevice));
    cudaStream_t stream = streams_[0];
    CUDA_CHECK(cudaDeviceSynchronize());

    Timer timer;
    timer.start();
    table_.query(d_query_keys_, num_query_keys_, d_query_values_, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto time_cost = timer.end();

    std::vector<Value_t> query_results(num_query_keys_ * table_param_.ev_size);

    CUDA_CHECK(cudaMemcpy(query_results.data(), d_query_values_,
                          sizeof(Value_t) * num_query_keys_ * table_param_.ev_size,
                          cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < num_query_keys_; i++) {
      bool mismatch = false;
      for (size_t j = 0; j < table_param_.ev_size; j++) {
        if (keys[i] != query_results[i * table_param_.ev_size]) {
          std::cout << keys[i] << '\t' << (size_t)query_results[i * table_param_.ev_size]
                    << std::endl;
          puts("Wrong");
          mismatch = true;
          break;
        }
      }
      ASSERT_FALSE(mismatch);
      if (mismatch) break;
    }

    total_time += time_cost;
    total_memory_bytes += get_num_bytes();
  }

  ~UvmTest() {
    CUDA_CHECK(cudaFree(m_values_));
    CUDA_CHECK(cudaFree(d_query_keys_));
    for (int i = 0; i < num_blocks; i++) {
      CUDA_CHECK(cudaFreeHost(h_buffers_[i]));
      CUDA_CHECK(cudaFree(d_buffers_[i]));
      CUDA_CHECK(cudaStreamDestroy(streams_[i]));
    }
    CUDA_CHECK(cudaFree(d_query_values_));
  }

 private:
  void init_keys() {
    h_keys_ = new Key_t[table_param_.num_keys];
    Key_t key = 0;
    for (size_t i = 0; i < table_param_.num_keys; i++) {
      key += rand() % 100 + 1;
      key %= std::numeric_limits<Key_t>::max() / 2;
      h_keys_[i] = key;
    }
  }

  size_t get_num_bytes() {
    return sizeof(Value_t) * lookup_param_.batch_size * lookup_param_.num_hotness *
               table_param_.ev_size +
           sizeof(Key_t) * lookup_param_.batch_size * lookup_param_.num_hotness;
  }

  std::vector<Key_t> get_random_keys() {
    auto powlaw_array = key_generator_.get_next_batch();
    std::vector<Key_t> array;
    for (size_t i = 0; i < lookup_param_.batch_size * lookup_param_.num_hotness; i++) {
      array.push_back(h_keys_[powlaw_array[i] % table_param_.num_keys]);
    }
    return array;
  }

  size_t num_query_keys_;
  size_t num_query_keys_per_block_;
  TableParam table_param_;
  LookupParam lookup_param_;
  Key_t *h_keys_;
  Value_t *m_values_;
  Key_t *d_query_keys_;
  Value_t *h_buffers_[num_blocks];
  Value_t *d_buffers_[num_blocks];
  cudaStream_t streams_[num_blocks];
  Value_t *d_query_values_;
  gpu_cache::UvmTable<Key_t, size_t> table_;
  KeyGenerator<Key_t> key_generator_;
};

TEST(uvm_table, uvm_table_int_float) {
  UvmTest<int, float> test({std::numeric_limits<int>::max(), 12345, 8}, {100000, 1});
  for (int i = 0; i < 100; i++) {
    test.run();
  }
}

TEST(uvm_table, uvm_table_longlong_float) {
  UvmTest<long long, float> test({std::numeric_limits<long long>::max(), 12345, 8}, {100000, 1});
  for (int i = 0; i < 100; i++) {
    test.run();
  }
}

TEST(uvm_table, uvm_table_size_t_float) {
  UvmTest<size_t, float> test({std::numeric_limits<size_t>::max(), 12345, 8}, {100000, 1});
  for (int i = 0; i < 100; i++) {
    test.run();
  }
}

TEST(uvm_table, uvm_table_int_float_bs1) {
  UvmTest<int, float> test({std::numeric_limits<int>::max(), 12345, 8}, {1, 1});
  for (int i = 0; i < 100; i++) {
    test.run();
  }
}

TEST(uvm_table, uvm_table_longlong_float_bs1) {
  UvmTest<long long, float> test({std::numeric_limits<long long>::max(), 12345, 8}, {1, 1});
  for (int i = 0; i < 100; i++) {
    test.run();
  }
}

TEST(uvm_table, uvm_table_size_t_float_bs1) {
  UvmTest<size_t, float> test({std::numeric_limits<size_t>::max(), 12345, 8}, {1, 1});
  for (int i = 0; i < 100; i++) {
    test.run();
  }
}

TEST(uvm_table, uvm_table_int_float_bs16) {
  UvmTest<int, float> test({std::numeric_limits<int>::max(), 12345, 8}, {16, 1});
  for (int i = 0; i < 100; i++) {
    test.run();
  }
}

TEST(uvm_table, uvm_table_longlong_float_bs16) {
  UvmTest<long long, float> test({std::numeric_limits<long long>::max(), 12345, 8}, {16, 1});
  for (int i = 0; i < 100; i++) {
    test.run();
  }
}

TEST(uvm_table, uvm_table_size_t_float_bs16) {
  UvmTest<size_t, float> test({std::numeric_limits<size_t>::max(), 12345, 8}, {16, 1});
  for (int i = 0; i < 100; i++) {
    test.run();
  }
}

TEST(uvm_table, uvm_table_perf) {
  UvmTest<int, float> test({std::numeric_limits<int>::max(), 8000000, 8}, {100000, 1});
  int iterations = 1000;
  int warm_up_iterations = 100;
  for (int i = 0; i < iterations; i++) {
    bool record_perf = i < warm_up_iterations ? false : true;
    test.run(record_perf);
  }
  std::cout << "Throughput: " << test.total_memory_bytes / test.total_time / GiB << " GiB/s"
            << std::endl;
  std::cout << "Average latency of each batch: "
            << test.total_time / (iterations - warm_up_iterations) * 1e6 << " us" << std::endl;
  std::cout << "Batches per second: " << 1.0 / (test.total_time / (iterations - warm_up_iterations))
            << std::endl;
}