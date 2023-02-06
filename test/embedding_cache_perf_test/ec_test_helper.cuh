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

#include <base/debug/logger.hpp>
#include <embedding_cache_perf_test/key_generator.cuh>
#include <gpu_cache/include/nv_gpu_cache.hpp>
#include <hps/inference_utils.hpp>
#include <hps/unique_op/unique_op.hpp>

template <typename key_type = long long, typename value_type = float,
          typename index_type = uint64_t>
class EcTestHelper {
 public:
  EcTestHelper(size_t max_batch_size, size_t num_hot, size_t embedding_vec_size, size_t num_sets,
               float alpha, size_t num_key_candidates, size_t key_range)
      : max_batch_size_(max_batch_size),
        num_hot_(num_hot),
        embedding_vec_size_(embedding_vec_size),
        num_sets_(num_sets),
        cache_(num_sets, embedding_vec_size),
        unique_op_(max_batch_size * num_hot),
        key_gen_(max_batch_size, num_hot, alpha, num_key_candidates, key_range) {
    int query_length = max_batch_size_ * num_hot_;
    HCTR_LIB_THROW(cudaMallocManaged(&d_keys_, query_length * sizeof(key_type)));
    HCTR_LIB_THROW(
        cudaMallocManaged(&d_values_, query_length * embedding_vec_size_ * sizeof(value_type)));
    HCTR_LIB_THROW(cudaMallocManaged(&d_values_retrieved_,
                                     query_length * embedding_vec_size_ * sizeof(value_type)));
    HCTR_LIB_THROW(cudaMallocManaged(&d_missing_index_, query_length * sizeof(index_type)));
    HCTR_LIB_THROW(cudaMallocManaged(&d_missing_keys_, query_length * sizeof(key_type)));
    HCTR_LIB_THROW(cudaMallocManaged(&d_missing_len_, sizeof(size_t)));

    HCTR_LIB_THROW(cudaMallocManaged(&d_unique_index_, query_length * sizeof(index_type)));
    HCTR_LIB_THROW(cudaMallocManaged(&d_unique_keys_, query_length * sizeof(key_type)));
    HCTR_LIB_THROW(cudaMallocManaged(&d_unique_len_, sizeof(size_t)));

    HCTR_LIB_THROW(cudaEventCreate(&event_start_));
    HCTR_LIB_THROW(cudaEventCreate(&event_end_));
  }

  void fill_cache(size_t len, cudaStream_t stream = 0) {
    for (size_t i = 0; i < len; i += max_batch_size_ * num_hot_) {
      auto keys = key_gen_.get_next_batch();
      size_t num_keys = std::min(max_batch_size_ * num_hot_, len - i);
      HCTR_LIB_THROW(cudaMemcpyAsync(d_keys_, keys.data(), sizeof(key_type) * num_keys,
                                     cudaMemcpyHostToDevice, stream));
      cache_.Replace(d_keys_, num_keys, d_values_, stream, task_per_warp_tile_);
    }
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  }

  void fill_cache_linear(size_t len, cudaStream_t stream = 0) {
    std::vector<key_type> keys;
    size_t max_num_keys = max_batch_size_ * num_hot_;
    for (size_t i = 0; i < len; i++) {
      keys.push_back(key_gen_.map_key(i));
      if (keys.size() == max_num_keys) {
        HCTR_LIB_THROW(cudaMemcpyAsync(d_keys_, keys.data(), sizeof(key_type) * max_num_keys,
                                       cudaMemcpyHostToDevice, stream));
        keys.clear();
        cache_.Replace(d_keys_, max_num_keys, d_values_, stream, task_per_warp_tile_);
      }
    }
    HCTR_LIB_THROW(cudaMemcpyAsync(d_keys_, keys.data(), sizeof(key_type) * keys.size(),
                                   cudaMemcpyHostToDevice, stream));
    cache_.Replace(d_keys_, keys.size(), d_values_, stream, task_per_warp_tile_);
  }

  void test_query(size_t batch_size, cudaStream_t stream = 0) {
    int num_query_keys = batch_size * num_hot_;
    auto keys = key_gen_.get_next_batch();
    HCTR_LIB_THROW(cudaMemcpyAsync(d_keys_, keys.data(), sizeof(key_type) * num_query_keys,
                                   cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(event_start_, stream));
    unique_op_.unique(d_keys_, num_query_keys, d_unique_index_, d_unique_keys_, d_unique_len_,
                      stream);
    ref_counter_type unique_len;
    cudaMemcpyAsync(&unique_len, d_unique_len_, sizeof(unique_len), cudaMemcpyDeviceToHost, stream);
    cache_.Query(d_unique_keys_, unique_len, d_values_, d_missing_index_, d_missing_keys_,
                 d_missing_len_, stream, task_per_warp_tile_);
    HugeCTR::decompress_emb_vec_async(d_values_retrieved_, d_unique_index_, d_values_,
                                      num_query_keys, embedding_vec_size_, 256, stream);
    unique_op_.clear(stream);
    CUDA_CHECK(cudaEventRecord(event_end_, stream));
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));

    float time_cost;
    CUDA_CHECK(cudaEventElapsedTime(&time_cost, event_start_, event_end_));
    time_recorder_.push_back(time_cost);
    missing_rate_recorder_.push_back(1.0f * d_missing_len_[0] / num_query_keys);
  }

  std::vector<float> get_missing_rate() { return missing_rate_recorder_; }
  std::vector<float> get_time_list() { return time_recorder_; }
  size_t get_memory_read_in_bytes(size_t batch_size) {
    return batch_size * num_hot_ * embedding_vec_size_ * sizeof(value_type);
  }

  void clear_results() {
    missing_rate_recorder_.clear();
    time_recorder_.clear();
  }

  ~EcTestHelper() {
    cudaFree(d_keys_);
    cudaFree(d_values_);
    cudaFree(d_values_retrieved_);
    cudaFree(d_missing_index_);
    cudaFree(d_missing_keys_);
    cudaFree(d_missing_len_);
    cudaFree(d_unique_index_);
    cudaFree(d_unique_keys_);
    cudaFree(d_unique_len_);
    cudaEventDestroy(event_start_);
    cudaEventDestroy(event_end_);
  }

  EcTestHelper(const EcTestHelper&) = delete;
  EcTestHelper& operator=(const EcTestHelper&) = delete;

 private:
  static constexpr key_type empty_key = std::numeric_limits<key_type>::max();

  static constexpr int SET_ASSOCIATIVITY_TEST = 2;
  static constexpr int SLAB_SIZE_TEST = 32;
  static constexpr int TASK_PER_WARP_TILE_TEST = 32;
  static constexpr int task_per_warp_tile_ = 32;

  using ref_counter_type = uint64_t;
  static constexpr ref_counter_type empty_index_val = std::numeric_limits<ref_counter_type>::max();

  using Cache_t = gpu_cache::gpu_cache<key_type, ref_counter_type, empty_key,
                                       SET_ASSOCIATIVITY_TEST, SLAB_SIZE_TEST>;
  using UniqueOp =
      HugeCTR::unique_op::unique_op<key_type, ref_counter_type, empty_key, empty_index_val>;

  key_type* d_keys_;
  value_type* d_values_;
  value_type* d_values_retrieved_;
  index_type* d_missing_index_;
  key_type* d_missing_keys_;
  size_t* d_missing_len_;
  index_type* d_unique_index_;
  key_type* d_unique_keys_;
  size_t* d_unique_len_;

  Cache_t cache_;
  UniqueOp unique_op_;
  KeyGenerator<key_type> key_gen_;
  size_t max_batch_size_;
  size_t num_hot_;
  size_t embedding_vec_size_;
  size_t num_sets_;

  cudaEvent_t event_start_;
  cudaEvent_t event_end_;

  std::vector<float> missing_rate_recorder_;
  std::vector<float> time_recorder_;
};