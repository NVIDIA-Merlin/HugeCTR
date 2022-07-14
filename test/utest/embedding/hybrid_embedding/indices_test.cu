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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/frequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/infrequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "hybrid_embedding_cpu.hpp"
#include "test_common.cuh"

/******************** Infrequent embedding: model indices ********************/

template <typename dtype, typename emtype>
class CalculateModelIndicesTest : public HybridEmbeddingUnitTest<dtype, emtype> {
 public:
  CalculateModelIndicesTest(const HybridEmbeddingConfig<dtype> config, size_t batch_size,
                            size_t seed = 1234ll)
      : HybridEmbeddingUnitTest<dtype, emtype>(config, batch_size, seed) {}

  void run() {
    /* Compute expected results on host */
    HybridEmbeddingCpu<dtype, emtype> cpu_embedding(this->config, this->batch_size,
                                                    this->category_location,
                                                    this->category_frequent_index, this->samples);
    cpu_embedding.calculate_infrequent_model_indices();

    /* Compute indices */
    this->build_infrequent();
    std::vector<std::vector<uint32_t>> h_model_indices(this->num_instances);
    std::vector<std::vector<uint32_t>> h_model_indices_offsets(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      this->infrequent_embeddings[i].set_current_indices(&this->infrequent_embedding_indices[i],
                                                         this->stream);
      this->infrequent_embeddings[i].indices_->calculate_model_indices(this->stream);
      download_tensor(h_model_indices[i], this->infrequent_embeddings[i].indices_->model_indices_,
                      this->stream);
      download_tensor(h_model_indices_offsets[i],
                      this->infrequent_embeddings[i].indices_->model_indices_offsets_,
                      this->stream);
    }

    /* Compare */
    for (size_t i = 0; i < this->num_instances; i++) {
      h_model_indices[i].resize(h_model_indices_offsets[i][this->num_instances]);
      EXPECT_THAT(h_model_indices[i], ::testing::ElementsAreArray(cpu_embedding.model_indices[i]));
      EXPECT_THAT(h_model_indices_offsets[i],
                  ::testing::ElementsAreArray(cpu_embedding.model_indices_offsets[i]));
    }
  }
};

/******************* Infrequent embedding: network indices *******************/

template <typename dtype, typename emtype>
class CalculateNetworkIndicesTest : public HybridEmbeddingUnitTest<dtype, emtype> {
 public:
  CalculateNetworkIndicesTest(const HybridEmbeddingConfig<dtype> config, size_t batch_size,
                              size_t seed = 1234ll)
      : HybridEmbeddingUnitTest<dtype, emtype>(config, batch_size, seed) {}

  void run() {
    /* Compute expected results on host */
    HybridEmbeddingCpu<dtype, emtype> cpu_embedding(this->config, this->batch_size,
                                                    this->category_location,
                                                    this->category_frequent_index, this->samples);
    cpu_embedding.calculate_infrequent_network_indices();

    /* Compute indices */
    this->build_infrequent();
    std::vector<std::vector<uint32_t>> h_network_indices(this->num_instances);
    std::vector<std::vector<uint32_t>> h_network_indices_offsets(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      this->infrequent_embeddings[i].set_current_indices(&this->infrequent_embedding_indices[i],
                                                         this->stream);
      this->infrequent_embeddings[i].indices_->calculate_network_indices(80, this->stream);
      download_tensor(h_network_indices[i],
                      this->infrequent_embeddings[i].indices_->network_indices_, this->stream);
      download_tensor(h_network_indices_offsets[i],
                      this->infrequent_embeddings[i].indices_->network_indices_offsets_,
                      this->stream);
    }

    /* Compare */
    for (size_t i = 0; i < this->num_instances; i++) {
      h_network_indices[i].resize(h_network_indices_offsets[i][this->num_instances]);
      EXPECT_THAT(h_network_indices[i],
                  ::testing::ElementsAreArray(cpu_embedding.network_indices[i]));
      EXPECT_THAT(h_network_indices_offsets[i],
                  ::testing::ElementsAreArray(cpu_embedding.network_indices_offsets[i]));
    }
  }
};

/**************** Frequent embedding: frequent sample indices ****************/

template <typename dtype, typename emtype>
class CalculateFrequentSampleIndicesTest : public HybridEmbeddingUnitTest<dtype, emtype> {
 public:
  CalculateFrequentSampleIndicesTest(const HybridEmbeddingConfig<dtype> config, size_t batch_size,
                                     size_t seed = 1234ll)
      : HybridEmbeddingUnitTest<dtype, emtype>(config, batch_size, seed) {}

  void run() {
    /* Compute expected results on host */
    HybridEmbeddingCpu<dtype, emtype> cpu_embedding(this->config, this->batch_size,
                                                    this->category_location,
                                                    this->category_frequent_index, this->samples);
    cpu_embedding.calculate_frequent_sample_indices();
    /* Compute indices */
    this->build_frequent();
    std::vector<std::vector<uint32_t>> h_frequent_sample_indices(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      this->frequent_embeddings[i].set_current_indices(&this->frequent_embedding_indices[i],
                                                       this->stream);
      this->frequent_embeddings[i].indices_->calculate_frequent_sample_indices(this->stream);
      download_tensor(h_frequent_sample_indices[i],
                      this->frequent_embeddings[i].indices_->frequent_sample_indices_,
                      this->stream);
    }

    /* Compare */
    for (size_t i = 0; i < this->num_instances; i++) {
      uint32_t num_frequent_sample_indices;
      HCTR_LIB_THROW(cudaMemcpyAsync(
          &num_frequent_sample_indices,
          this->frequent_embeddings[i].indices_->d_num_frequent_sample_indices_.get_ptr(),
          sizeof(uint32_t), cudaMemcpyDeviceToHost, this->stream));
      HCTR_LIB_THROW(cudaStreamSynchronize(this->stream));
      h_frequent_sample_indices[i].resize(num_frequent_sample_indices);
      EXPECT_THAT(h_frequent_sample_indices[i],
                  ::testing::ElementsAreArray(cpu_embedding.frequent_sample_indices[i]));
    }
  }
};

/****************** Frequent embedding: model cache indices ******************/

template <typename dtype, typename emtype>
class CalculateModelCacheIndicesTest : public HybridEmbeddingUnitTest<dtype, emtype> {
 public:
  CalculateModelCacheIndicesTest(const HybridEmbeddingConfig<dtype> config, size_t batch_size,
                                 size_t seed = 1234ll)
      : HybridEmbeddingUnitTest<dtype, emtype>(config, batch_size, seed) {}

  void run() {
    /* Compute expected results on host */
    HybridEmbeddingCpu<dtype, emtype> cpu_embedding(this->config, this->batch_size,
                                                    this->category_location,
                                                    this->category_frequent_index, this->samples);
    cpu_embedding.calculate_frequent_model_cache_indices();

    /* Compute indices */
    this->build_frequent();
    std::vector<std::vector<uint32_t>> h_model_cache_indices(this->num_instances);
    std::vector<std::vector<uint32_t>> h_model_cache_indices_offsets(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      this->frequent_embeddings[i].set_current_indices(&this->frequent_embedding_indices[i],
                                                       this->stream);
      this->frequent_embeddings[i].indices_->calculate_cache_masks(this->stream);
      this->frequent_embeddings[i].indices_->calculate_model_cache_indices(80, this->stream);
      download_tensor(h_model_cache_indices[i],
                      this->frequent_embeddings[i].indices_->model_cache_indices_, this->stream);
      download_tensor(h_model_cache_indices_offsets[i],
                      this->frequent_embeddings[i].indices_->model_cache_indices_offsets_,
                      this->stream);
    }

    /* Compare */
    for (size_t i = 0; i < this->num_instances; i++) {
      h_model_cache_indices[i].resize(h_model_cache_indices_offsets[i][this->num_instances]);
      EXPECT_THAT(h_model_cache_indices[i],
                  ::testing::ElementsAreArray(cpu_embedding.model_cache_indices[i]));
      EXPECT_THAT(h_model_cache_indices_offsets[i],
                  ::testing::ElementsAreArray(cpu_embedding.model_cache_indices_offsets[i]));
    }
  }
};

/***************** Frequent embedding: network cache indices *****************/

template <typename dtype, typename emtype>
class CalculateNetworkCacheIndicesTest : public HybridEmbeddingUnitTest<dtype, emtype> {
 public:
  CalculateNetworkCacheIndicesTest(const HybridEmbeddingConfig<dtype> config, size_t batch_size,
                                   size_t seed = 1234ll)
      : HybridEmbeddingUnitTest<dtype, emtype>(config, batch_size, seed) {}

  void run() {
    /* Compute expected results on host */
    HybridEmbeddingCpu<dtype, emtype> cpu_embedding(this->config, this->batch_size,
                                                    this->category_location,
                                                    this->category_frequent_index, this->samples);
    cpu_embedding.calculate_frequent_network_cache_mask();
    cpu_embedding.calculate_frequent_network_cache_indices();

    /* Compute mask and indices */
    this->build_frequent();
    std::vector<std::vector<uint8_t>> h_network_cache_mask(this->num_instances);
    std::vector<std::vector<uint32_t>> h_network_cache_indices(this->num_instances);
    std::vector<std::vector<uint32_t>> h_network_cache_indices_offsets(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      this->frequent_embeddings[i].set_current_indices(&this->frequent_embedding_indices[i],
                                                       this->stream);
      this->frequent_embeddings[i].indices_->calculate_cache_masks(this->stream);
      this->frequent_embeddings[i].indices_->calculate_network_cache_indices(this->stream);
      download_tensor(h_network_cache_indices[i],
                      this->frequent_embeddings[i].indices_->network_cache_indices_, this->stream);
      download_tensor(h_network_cache_indices_offsets[i],
                      this->frequent_embeddings[i].indices_->network_cache_indices_offsets_,
                      this->stream);
      h_network_cache_mask[i].resize(this->config.num_frequent);
      HCTR_LIB_THROW(cudaMemcpyAsync(
          h_network_cache_mask[i].data(),
          reinterpret_cast<uint8_t*>(this->frequent_embeddings[i].indices_->cache_masks_.get_ptr()),
          this->config.num_frequent, cudaMemcpyDeviceToHost, this->stream));
      HCTR_LIB_THROW(cudaStreamSynchronize(this->stream));
    }

    /* Compare */
    for (size_t i = 0; i < this->num_instances; i++) {
      h_network_cache_indices[i].resize(
          cpu_embedding.network_cache_indices_offsets[i][this->num_instances]);
      EXPECT_THAT(h_network_cache_indices[i],
                  ::testing::ElementsAreArray(cpu_embedding.network_cache_indices[i]));
      EXPECT_THAT(h_network_cache_indices_offsets[i],
                  ::testing::ElementsAreArray(cpu_embedding.network_cache_indices_offsets[i]));
      EXPECT_THAT(h_network_cache_mask[i],
                  ::testing::ElementsAreArray(cpu_embedding.network_cache_mask[i]));
    }
  }
};

/**************************** Test instantiations ****************************/

static const HybridEmbeddingConfig<uint32_t> config_uint32 = {
    4, 32, 10, 128, 1000, 128, 0.5f, CommunicationType::IB_NVLink};
static const HybridEmbeddingConfig<long long> config_int64 = {
    4, 32, 10, 128, 1000, 128, 0.5f, CommunicationType::IB_NVLink};

// Edge cases: no frequent, all frequent
static const HybridEmbeddingConfig<uint32_t> config_no_freq = {
    4, 32, 10, 128, 1000, 0, 0.5f, CommunicationType::IB_NVLink};
static const HybridEmbeddingConfig<uint32_t> config_all_freq = {
    4, 32, 10, 128, 1000, 1000, 0.5f, CommunicationType::IB_NVLink};

/* hybrid_embedding_model_indices_test */

TEST(hybrid_embedding_model_indices_test, uint32_float_64) {
  CalculateModelIndicesTest<uint32_t, float>(config_uint32, 64).run();
}

TEST(hybrid_embedding_model_indices_test, int64_float_64) {
  CalculateModelIndicesTest<long long, float>(config_int64, 64).run();
}

TEST(hybrid_embedding_model_indices_test, uint32_float_2048) {
  CalculateModelIndicesTest<uint32_t, float>(config_uint32, 2048).run();
}

TEST(hybrid_embedding_model_indices_test, int64_float_2048) {
  CalculateModelIndicesTest<long long, float>(config_int64, 2048).run();
}

TEST(hybrid_embedding_model_indices_test, uint32_float_128_no_freq) {
  CalculateModelIndicesTest<uint32_t, float>(config_no_freq, 128).run();
}

TEST(hybrid_embedding_model_indices_test, uint32_float_128_all_freq) {
  CalculateModelIndicesTest<uint32_t, float>(config_all_freq, 128).run();
}

/* hybrid_embedding_network_indices_test */

TEST(hybrid_embedding_network_indices_test, uint32_float_64) {
  CalculateNetworkIndicesTest<uint32_t, float>(config_uint32, 64).run();
}

TEST(hybrid_embedding_network_indices_test, int64_float_64) {
  CalculateNetworkIndicesTest<long long, float>(config_int64, 64).run();
}

TEST(hybrid_embedding_network_indices_test, uint32_float_2048) {
  CalculateNetworkIndicesTest<uint32_t, float>(config_uint32, 2048).run();
}

TEST(hybrid_embedding_network_indices_test, int64_float_2048) {
  CalculateNetworkIndicesTest<long long, float>(config_int64, 2048).run();
}

TEST(hybrid_embedding_network_indices_test, uint32_float_128_no_freq) {
  CalculateNetworkIndicesTest<uint32_t, float>(config_no_freq, 128).run();
}

TEST(hybrid_embedding_network_indices_test, uint32_float_128_all_freq) {
  CalculateNetworkIndicesTest<uint32_t, float>(config_all_freq, 128).run();
}

/* hybrid_embedding_frequent_sample_indices_test */

TEST(hybrid_embedding_frequent_sample_indices_test, uint32_float_64) {
  CalculateFrequentSampleIndicesTest<uint32_t, float>(config_uint32, 64).run();
}

TEST(hybrid_embedding_frequent_sample_indices_test, int64_float_64) {
  CalculateFrequentSampleIndicesTest<long long, float>(config_int64, 64).run();
}

TEST(hybrid_embedding_frequent_sample_indices_test, uint32_float_2048) {
  CalculateFrequentSampleIndicesTest<uint32_t, float>(config_uint32, 2048).run();
}

TEST(hybrid_embedding_frequent_sample_indices_test, int64_float_2048) {
  CalculateFrequentSampleIndicesTest<long long, float>(config_int64, 2048).run();
}

TEST(hybrid_embedding_frequent_sample_indices_test, uint32_float_128_no_freq) {
  CalculateFrequentSampleIndicesTest<uint32_t, float>(config_no_freq, 128).run();
}

TEST(hybrid_embedding_frequent_sample_indices_test, uint32_float_128_all_freq) {
  CalculateFrequentSampleIndicesTest<uint32_t, float>(config_all_freq, 128).run();
}

/* hybrid_embedding_model_cache_indices_test */

TEST(hybrid_embedding_model_cache_indices_test, uint32_float_64) {
  CalculateModelCacheIndicesTest<uint32_t, float>(config_uint32, 64).run();
}

TEST(hybrid_embedding_model_cache_indices_test, int64_float_64) {
  CalculateModelCacheIndicesTest<long long, float>(config_int64, 64).run();
}

TEST(hybrid_embedding_model_cache_indices_test, uint32_float_2048) {
  CalculateModelCacheIndicesTest<uint32_t, float>(config_uint32, 2048).run();
}

TEST(hybrid_embedding_model_cache_indices_test, int64_float_2048) {
  CalculateModelCacheIndicesTest<long long, float>(config_int64, 2048).run();
}

TEST(hybrid_embedding_model_cache_indices_test, uint32_float_128_no_freq) {
  CalculateModelCacheIndicesTest<uint32_t, float>(config_no_freq, 128).run();
}

TEST(hybrid_embedding_model_cache_indices_test, uint32_float_128_all_freq) {
  CalculateModelCacheIndicesTest<uint32_t, float>(config_all_freq, 128).run();
}

/* hybrid_embedding_network_cache_indices_test */

TEST(hybrid_embedding_network_cache_indices_test, uint32_float_64) {
  CalculateNetworkCacheIndicesTest<uint32_t, float>(config_uint32, 64).run();
}

TEST(hybrid_embedding_network_cache_indices_test, int64_float_64) {
  CalculateNetworkCacheIndicesTest<long long, float>(config_int64, 64).run();
}

TEST(hybrid_embedding_network_cache_indices_test, uint32_float_2048) {
  CalculateNetworkCacheIndicesTest<uint32_t, float>(config_uint32, 2048).run();
}

TEST(hybrid_embedding_network_cache_indices_test, int64_float_2048) {
  CalculateNetworkCacheIndicesTest<long long, float>(config_int64, 2048).run();
}

TEST(hybrid_embedding_network_cache_indices_test, uint32_float_128_no_freq) {
  CalculateNetworkCacheIndicesTest<uint32_t, float>(config_no_freq, 128).run();
}

TEST(hybrid_embedding_network_cache_indices_test, uint32_float_128_all_freq) {
  CalculateNetworkCacheIndicesTest<uint32_t, float>(config_all_freq, 128).run();
}
