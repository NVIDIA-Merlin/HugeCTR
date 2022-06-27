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
#include "HugeCTR/include/utils.cuh"
#include "hybrid_embedding_cpu.hpp"
#include "test_common.cuh"

/************************ Infrequent embedding update ************************/

template <typename dtype, typename emtype>
class InfrequentUpdateTest : public HybridEmbeddingUnitTest<dtype, emtype> {
 protected:
  bool single_node;

 public:
  InfrequentUpdateTest(const HybridEmbeddingConfig<dtype> config, size_t batch_size,
                       bool single_node, size_t seed = 1234ll)
      : HybridEmbeddingUnitTest<dtype, emtype>(config, batch_size, seed),
        single_node(single_node) {}

  void run() {
    uint32_t local_batch_size = ceildiv<uint32_t>(this->batch_size, this->num_instances);

    HybridEmbeddingCpu<dtype, emtype> cpu_embedding(this->config, this->batch_size,
                                                    this->category_location, this->samples);
    cpu_embedding.calculate_infrequent_model_indices();
    cpu_embedding.calculate_infrequent_network_indices();
    cpu_embedding.generate_embedding_vectors();
    cpu_embedding.generate_gradients();
    if (this->config.comm_type == CommunicationType::IB_NVLink) {
      cpu_embedding.backward_a2a_messages();
    } else if (this->config.comm_type == CommunicationType::IB_NVLink_Hier) {
      cpu_embedding.backward_a2a_messages_hier();
    }

    /* Tensors for the messages and gradients */
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
    std::vector<Tensor2<emtype>> received_messages(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      buff->reserve({this->batch_size * this->config.num_tables, this->config.embedding_vec_size},
                    &received_messages[i]);
    }
    std::vector<Tensor2<emtype>> gradients(this->num_instances);
    if (single_node) {
      for (size_t i = 0; i < this->num_instances; i++) {
        buff->reserve({this->num_instances * local_batch_size * this->config.num_tables,
                       this->config.embedding_vec_size},
                      &gradients[i]);
      }
    }
    buff->allocate();

    /* Single-node: upload gradients */
    this->build_infrequent();
    if (single_node) {
      for (size_t i = 0; i < this->num_instances; i++) {
        upload_tensor(cpu_embedding.gradients[i], gradients[i], this->stream);
      }
    }

    /* Infrequent update_model */
    std::vector<std::vector<float>> updated_vectors(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      this->infrequent_embeddings[i].set_current_indices(&this->infrequent_embedding_indices[i],
                                                         this->stream);
      upload_tensor(cpu_embedding.infrequent_embedding_vectors[i],
                    this->infrequent_embeddings[i].infrequent_embedding_vectors_, this->stream);
      this->infrequent_embeddings[i].indices_->calculate_model_indices(this->stream);
      if (single_node) {
        std::vector<const emtype *> gradients_pointers(this->num_instances);
        for (uint32_t network_id = 0; network_id < this->num_instances; network_id++)
          gradients_pointers[network_id] = gradients[network_id].get_ptr();
        HCTR_LIB_THROW(cudaMemcpyAsync(
            this->infrequent_embeddings[i].gradients_pointers_.get_ptr(), gradients_pointers.data(),
            this->num_instances * sizeof(emtype *), cudaMemcpyHostToDevice, this->stream));
        this->infrequent_embeddings[i].update_model_direct(this->dev_lr, 1.f, this->stream);
      } else {
        upload_tensor(cpu_embedding.backward_received_messages[i], received_messages[i],
                      this->stream);
        if (this->config.comm_type == CommunicationType::IB_NVLink_Hier) {
          this->infrequent_embeddings[i].hier_update_model(received_messages[i].get_ptr(),
                                                           this->dev_lr, 1.f, this->stream);
        } else {
          this->infrequent_embeddings[i].update_model(received_messages[i].get_ptr(), this->dev_lr,
                                                      1.f, this->stream);
        }
      }

      download_tensor(updated_vectors[i],
                      this->infrequent_embeddings[i].infrequent_embedding_vectors_, this->stream);
    }

    /* Reference update_model */
    cpu_embedding.infrequent_update();

    /* Compare */
    for (size_t i = 0; i < this->num_instances; i++) {
      updated_vectors[i].resize(
          ceildiv<dtype>(this->config.num_categories - this->config.num_frequent,
                         this->num_instances) *
          this->config.embedding_vec_size);
      EXPECT_THAT(updated_vectors[i],
                  ::testing::Pointwise(::testing::FloatNear(1e-2),
                                       cpu_embedding.infrequent_embedding_vectors[i]));
    }
  }
};

/************************* Frequent embedding update *************************/

template <typename dtype, typename emtype>
class FrequentUpdateTest : public HybridEmbeddingUnitTest<dtype, emtype> {
 protected:
  bool single_node;

 public:
  FrequentUpdateTest(const HybridEmbeddingConfig<dtype> config, size_t batch_size, bool single_node,
                     size_t seed = 1234ll)
      : HybridEmbeddingUnitTest<dtype, emtype>(config, batch_size, seed),
        single_node(single_node) {}

  void run() {
    uint32_t local_batch_size = ceildiv<uint32_t>(this->batch_size, this->num_instances);

    HybridEmbeddingCpu<dtype, emtype> cpu_embedding(this->config, this->batch_size,
                                                    this->category_location, this->samples);
    cpu_embedding.calculate_frequent_network_cache_indices();
    cpu_embedding.generate_embedding_vectors();
    cpu_embedding.generate_gradients();
    cpu_embedding.frequent_reduce_gradients();

    /* Tensors for the gradients (single-node) */
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
    std::vector<Tensor2<emtype>> gradients(this->num_instances);
    if (single_node) {
      for (size_t i = 0; i < this->num_instances; i++) {
        buff->reserve({local_batch_size * this->config.num_tables, this->config.embedding_vec_size},
                      &gradients[i]);
      }
    }
    buff->allocate();

    /* Frequent update_model */
    this->build_frequent();
    std::vector<std::vector<float>> updated_vectors(this->num_instances);
    std::vector<const emtype *> frequent_partial_gradients_pointers(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      this->frequent_embeddings[i].set_current_indices(&this->frequent_embedding_indices[i],
                                                       this->stream);
      upload_tensor(cpu_embedding.frequent_embedding_vectors[i],
                    this->frequent_embeddings[i].frequent_embedding_vectors_, this->stream);
      if (single_node) {
        upload_tensor(cpu_embedding.gradients[i], gradients[i], this->stream);
        frequent_partial_gradients_pointers[i] =
            this->frequent_embeddings[i].get_gradients().get_ptr();
      } else
        upload_tensor(cpu_embedding.reduced_gradients, this->frequent_embeddings[i].get_gradients(),
                      this->stream);
    }
    for (size_t i = 0; i < this->num_instances; i++) {
      if (single_node) {
        this->frequent_embeddings[i].indices_->calculate_cache_masks(this->stream);
        this->frequent_embeddings[i].indices_->calculate_network_cache_indices(this->stream);
        this->frequent_embeddings[i].indices_->calculate_model_cache_indices(80, this->stream);
        this->frequent_embeddings[i].indices_->calculate_frequent_sample_indices(this->stream);
        this->frequent_embeddings[i].local_reduce(gradients[i].get_ptr(), this->stream,
                                                  !single_node);
      } else {
        this->frequent_embeddings[i].update_model(this->dev_lr, 1.f, this->stream);
      }
    }
    for (size_t i = 0; i < this->num_instances; i++) {
      if (single_node) {
        HCTR_LIB_THROW(cudaMemcpyAsync(
            this->frequent_embeddings[i].partial_gradients_pointers_.get_ptr(),
            frequent_partial_gradients_pointers.data(), this->num_instances * sizeof(emtype *),
            cudaMemcpyHostToDevice, this->stream));
        this->frequent_embeddings[i].update_model_direct(this->dev_lr, 1.f, this->stream);
      }
      download_tensor(updated_vectors[i], this->frequent_embeddings[i].frequent_embedding_vectors_,
                      this->stream);
    }

    /* Reference update_model */
    if (single_node)
      cpu_embedding.frequent_update_single_node();
    else
      cpu_embedding.frequent_update();

    /* Compare */
    for (size_t i = 0; i < this->num_instances; i++) {
      updated_vectors[i].resize(this->config.num_frequent * this->config.embedding_vec_size);
      EXPECT_THAT(updated_vectors[i],
                  ::testing::Pointwise(::testing::FloatNear(5e-2),
                                       cpu_embedding.frequent_embedding_vectors[i]));
    }
  }
};

/**************************** Test instantiations ****************************/

static const HybridEmbeddingConfig<uint32_t> config_uint32 = {
    4, 32, 10, 128, 1000, 128, 0.5f, CommunicationType::IB_NVLink};
static const HybridEmbeddingConfig<long long> config_int64 = {
    4, 32, 10, 128, 1000, 128, 0.5f, CommunicationType::IB_NVLink};
static const HybridEmbeddingConfig<uint32_t> config_uint32_single_node = {
    1, 8, 10, 128, 1000, 128, 0.5f, CommunicationType::NVLink_SingleNode};
static const HybridEmbeddingConfig<long long> config_int64_single_node = {
    1, 8, 10, 128, 1000, 128, 0.5f, CommunicationType::NVLink_SingleNode};

// Edge cases: no frequent, all frequent
static const HybridEmbeddingConfig<uint32_t> config_no_freq = {
    4, 32, 10, 128, 1000, 0, 0.5f, CommunicationType::IB_NVLink};
static const HybridEmbeddingConfig<uint32_t> config_all_freq = {
    4, 32, 10, 128, 1000, 1000, 0.5f, CommunicationType::IB_NVLink};
static const HybridEmbeddingConfig<uint32_t> config_no_freq_single_node = {
    1, 8, 10, 128, 1000, 0, 0.5f, CommunicationType::NVLink_SingleNode};
static const HybridEmbeddingConfig<uint32_t> config_all_freq_single_node = {
    1, 8, 10, 128, 1000, 1000, 0.5f, CommunicationType::NVLink_SingleNode};

// Hierarchical A2A
static const HybridEmbeddingConfig<uint32_t> config_uint32_hier = {
    4, 32, 10, 128, 1000, 128, 0.5f, CommunicationType::IB_NVLink_Hier};
static const HybridEmbeddingConfig<long long> config_int64_hier = {
    4, 32, 10, 128, 1000, 128, 0.5f, CommunicationType::IB_NVLink_Hier};
static const HybridEmbeddingConfig<uint32_t> config_no_freq_hier = {
    4, 32, 10, 128, 1000, 0, 0.5f, CommunicationType::IB_NVLink_Hier};
static const HybridEmbeddingConfig<uint32_t> config_all_freq_hier = {
    4, 32, 10, 128, 1000, 1000, 0.5f, CommunicationType::IB_NVLink_Hier};

/* hybrid_embedding_infrequent_update_test */

TEST(hybrid_embedding_infrequent_update_test, uint32_half_64) {
  InfrequentUpdateTest<uint32_t, __half>(config_uint32, 64, false).run();
}

TEST(hybrid_embedding_infrequent_update_test, int64_half_64) {
  InfrequentUpdateTest<long long, __half>(config_int64, 64, false).run();
}

TEST(hybrid_embedding_infrequent_update_test, uint32_half_2048) {
  InfrequentUpdateTest<uint32_t, __half>(config_uint32, 2048, false).run();
}

TEST(hybrid_embedding_infrequent_update_test, int64_half_2048) {
  InfrequentUpdateTest<long long, __half>(config_int64, 2048, false).run();
}

TEST(hybrid_embedding_infrequent_update_test, uint32_float_64) {
  InfrequentUpdateTest<uint32_t, float>(config_uint32, 64, false).run();
}

TEST(hybrid_embedding_infrequent_update_test, int64_float_64) {
  InfrequentUpdateTest<long long, float>(config_int64, 64, false).run();
}

TEST(hybrid_embedding_infrequent_update_test, uint32_float_2048) {
  InfrequentUpdateTest<uint32_t, float>(config_uint32, 2048, false).run();
}

TEST(hybrid_embedding_infrequent_update_test, int64_float_2048) {
  InfrequentUpdateTest<long long, float>(config_int64, 2048, false).run();
}

TEST(hybrid_embedding_infrequent_update_test, uint32_float_128_no_freq) {
  InfrequentUpdateTest<uint32_t, float>(config_no_freq, 128, false).run();
}

TEST(hybrid_embedding_infrequent_update_test, uint32_float_128_all_freq) {
  InfrequentUpdateTest<uint32_t, float>(config_all_freq, 128, false).run();
}

/* hybrid_embedding_infrequent_update_single_node_test */

TEST(hybrid_embedding_infrequent_update_single_node_test, uint32_half_64) {
  InfrequentUpdateTest<uint32_t, __half>(config_uint32_single_node, 64, true).run();
}

TEST(hybrid_embedding_infrequent_update_single_node_test, int64_half_64) {
  InfrequentUpdateTest<long long, __half>(config_int64_single_node, 64, true).run();
}

TEST(hybrid_embedding_infrequent_update_single_node_test, uint32_half_2048) {
  InfrequentUpdateTest<uint32_t, __half>(config_uint32_single_node, 2048, true).run();
}

TEST(hybrid_embedding_infrequent_update_single_node_test, int64_half_2048) {
  InfrequentUpdateTest<long long, __half>(config_int64_single_node, 2048, true).run();
}

TEST(hybrid_embedding_infrequent_update_single_node_test, uint32_float_64) {
  InfrequentUpdateTest<uint32_t, float>(config_uint32_single_node, 64, true).run();
}

TEST(hybrid_embedding_infrequent_update_single_node_test, int64_float_64) {
  InfrequentUpdateTest<long long, float>(config_int64_single_node, 64, true).run();
}

TEST(hybrid_embedding_infrequent_update_single_node_test, uint32_float_2048) {
  InfrequentUpdateTest<uint32_t, float>(config_uint32_single_node, 2048, true).run();
}

TEST(hybrid_embedding_infrequent_update_single_node_test, int64_float_2048) {
  InfrequentUpdateTest<long long, float>(config_int64_single_node, 2048, true).run();
}

TEST(hybrid_embedding_infrequent_update_single_node_test, uint32_float_128_no_freq) {
  InfrequentUpdateTest<uint32_t, float>(config_no_freq_single_node, 128, true).run();
}

TEST(hybrid_embedding_infrequent_update_single_node_test, uint32_float_128_all_freq) {
  InfrequentUpdateTest<uint32_t, float>(config_all_freq_single_node, 128, true).run();
}

/* hybrid_embedding_infrequent_update_hier_test */

TEST(hybrid_embedding_infrequent_update_hier_test, uint32_half_64) {
  InfrequentUpdateTest<uint32_t, __half>(config_uint32_hier, 64, false).run();
}

TEST(hybrid_embedding_infrequent_update_hier_test, int64_half_64) {
  InfrequentUpdateTest<long long, __half>(config_int64_hier, 64, false).run();
}

TEST(hybrid_embedding_infrequent_update_hier_test, uint32_half_2048) {
  InfrequentUpdateTest<uint32_t, __half>(config_uint32_hier, 2048, false).run();
}

TEST(hybrid_embedding_infrequent_update_hier_test, int64_half_2048) {
  InfrequentUpdateTest<long long, __half>(config_int64_hier, 2048, false).run();
}

TEST(hybrid_embedding_infrequent_update_hier_test, uint32_float_64) {
  InfrequentUpdateTest<uint32_t, float>(config_uint32_hier, 64, false).run();
}

TEST(hybrid_embedding_infrequent_update_hier_test, int64_float_64) {
  InfrequentUpdateTest<long long, float>(config_int64_hier, 64, false).run();
}

TEST(hybrid_embedding_infrequent_update_hier_test, uint32_float_2048) {
  InfrequentUpdateTest<uint32_t, float>(config_uint32_hier, 2048, false).run();
}

TEST(hybrid_embedding_infrequent_update_hier_test, int64_float_2048) {
  InfrequentUpdateTest<long long, float>(config_int64_hier, 2048, false).run();
}

TEST(hybrid_embedding_infrequent_update_hier_test, uint32_float_128_no_freq) {
  InfrequentUpdateTest<uint32_t, float>(config_no_freq_hier, 128, false).run();
}

TEST(hybrid_embedding_infrequent_update_hier_test, uint32_float_128_all_freq) {
  InfrequentUpdateTest<uint32_t, float>(config_all_freq_hier, 128, false).run();
}

/* hybrid_embedding_frequent_update_test */

TEST(hybrid_embedding_frequent_update_test, uint32_half_64) {
  FrequentUpdateTest<uint32_t, __half>(config_uint32, 64, false).run();
}

TEST(hybrid_embedding_frequent_update_test, int64_half_64) {
  FrequentUpdateTest<long long, __half>(config_int64, 64, false).run();
}

TEST(hybrid_embedding_frequent_update_test, uint32_half_2048) {
  FrequentUpdateTest<uint32_t, __half>(config_uint32, 2048, false).run();
}

TEST(hybrid_embedding_frequent_update_test, int64_half_2048) {
  FrequentUpdateTest<long long, __half>(config_int64, 2048, false).run();
}

TEST(hybrid_embedding_frequent_update_test, uint32_float_64) {
  FrequentUpdateTest<uint32_t, float>(config_uint32, 64, false).run();
}

TEST(hybrid_embedding_frequent_update_test, int64_float_64) {
  FrequentUpdateTest<long long, float>(config_int64, 64, false).run();
}

TEST(hybrid_embedding_frequent_update_test, uint32_float_2048) {
  FrequentUpdateTest<uint32_t, float>(config_uint32, 2048, false).run();
}

TEST(hybrid_embedding_frequent_update_test, int64_float_2048) {
  FrequentUpdateTest<long long, float>(config_int64, 2048, false).run();
}

TEST(hybrid_embedding_frequent_update_test, uint32_float_128_no_freq) {
  FrequentUpdateTest<uint32_t, float>(config_no_freq, 128, false).run();
}

TEST(hybrid_embedding_frequent_update_test, uint32_float_128_all_freq) {
  FrequentUpdateTest<uint32_t, float>(config_all_freq, 128, false).run();
}

/* hybrid_embedding_frequent_update_single_node_test */

TEST(hybrid_embedding_frequent_update_single_node_test, uint32_half_64) {
  FrequentUpdateTest<uint32_t, __half>(config_uint32_single_node, 64, true).run();
}

TEST(hybrid_embedding_frequent_update_single_node_test, int64_half_64) {
  FrequentUpdateTest<long long, __half>(config_int64_single_node, 64, true).run();
}

TEST(hybrid_embedding_frequent_update_single_node_test, uint32_half_2048) {
  FrequentUpdateTest<uint32_t, __half>(config_uint32_single_node, 2048, true).run();
}

TEST(hybrid_embedding_frequent_update_single_node_test, int64_half_2048) {
  FrequentUpdateTest<long long, __half>(config_int64_single_node, 2048, true).run();
}

TEST(hybrid_embedding_frequent_update_single_node_test, uint32_float_64) {
  FrequentUpdateTest<uint32_t, float>(config_uint32_single_node, 64, true).run();
}

TEST(hybrid_embedding_frequent_update_single_node_test, int64_float_64) {
  FrequentUpdateTest<long long, float>(config_int64_single_node, 64, true).run();
}

TEST(hybrid_embedding_frequent_update_single_node_test, uint32_float_2048) {
  FrequentUpdateTest<uint32_t, float>(config_uint32_single_node, 2048, true).run();
}

TEST(hybrid_embedding_frequent_update_single_node_test, int64_float_2048) {
  FrequentUpdateTest<long long, float>(config_int64_single_node, 2048, true).run();
}

TEST(hybrid_embedding_frequent_update_single_node_test, uint32_float_128_no_freq) {
  FrequentUpdateTest<uint32_t, float>(config_no_freq_single_node, 128, true).run();
}

TEST(hybrid_embedding_frequent_update_single_node_test, uint32_float_128_all_freq) {
  FrequentUpdateTest<uint32_t, float>(config_all_freq_single_node, 128, true).run();
}
