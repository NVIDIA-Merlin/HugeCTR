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

/****************** Frequent and infrequent forward network ******************/

template <typename dtype, typename emtype>
class ForwardNetworkTest : public HybridEmbeddingUnitTest<dtype, emtype> {
 protected:
  bool single_node;

 public:
  ForwardNetworkTest(const HybridEmbeddingConfig<dtype> config, size_t batch_size, bool single_node,
                     size_t seed = 1234ll)
      : HybridEmbeddingUnitTest<dtype, emtype>(config, batch_size, seed),
        single_node(single_node) {}

  void run() {
    uint32_t local_batch_size = ceildiv<uint32_t>(this->batch_size, this->num_instances);

    /* Compute expected results on host */
    HybridEmbeddingCpu<dtype, emtype> cpu_embedding(this->config, this->batch_size,
                                                    this->category_location,
                                                    this->category_frequent_index, this->samples);
    cpu_embedding.generate_embedding_vectors();
    cpu_embedding.forward_network();
    if (!single_node) {
      cpu_embedding.calculate_infrequent_model_indices();
      if (this->config.comm_type == CommunicationType::IB_NVLink_Hier) {
        cpu_embedding.forward_a2a_messages_hier();
      } else {
        cpu_embedding.forward_a2a_messages();
      }
    }

    /* Tensors for the interaction layer input and messages */
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
    std::vector<Tensor2<emtype>> interaction_layer_input(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      buff->reserve({local_batch_size * this->config.num_tables, this->config.embedding_vec_size},
                    &interaction_layer_input[i]);
    }
    std::vector<Tensor2<emtype>> received_messages(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      buff->reserve({this->num_instances * local_batch_size * this->config.num_tables,
                     this->config.embedding_vec_size},
                    &received_messages[i]);
    }
    buff->allocate();

    /* In single-node case, make an array of the interaction mayer input pointers */
    std::vector<emtype *> interaction_layer_input_pointers_;
    if (single_node) {
      for (size_t i = 0; i < this->num_instances; i++) {
        interaction_layer_input_pointers_.push_back(interaction_layer_input[i].get_ptr());
      }
    }

    /* Frequent and infrequent forward_network */
    this->build_infrequent();
    this->build_frequent();
    for (size_t i = 0; i < this->num_instances; i++) {
      upload_tensor(cpu_embedding.frequent_embedding_vectors[i],
                    this->frequent_embeddings[i].frequent_embedding_vectors_, this->stream);
      upload_tensor(cpu_embedding.infrequent_embedding_vectors[i],
                    this->infrequent_embeddings[i].infrequent_embedding_vectors_, this->stream);
    }
    for (size_t i = 0; i < this->num_instances; i++) {
      this->  frequent_embeddings[i].set_current_indices(&this->  frequent_embedding_indices[i], this->stream);
      this->infrequent_embeddings[i].set_current_indices(&this->infrequent_embedding_indices[i], this->stream);

      if (single_node) {
        this->frequent_embeddings[i].indices_->calculate_cache_masks(this->stream);
        this->frequent_embeddings[i].indices_->calculate_model_cache_indices(80, this->stream);
        this->frequent_embeddings[i].forward_model(this->stream);
      }
    }
    for (size_t i = 0; i < this->num_instances; i++) {
      this->frequent_embeddings[i].indices_->calculate_frequent_sample_indices(this->stream);
      this->frequent_embeddings[i].forward_network(interaction_layer_input[i].get_ptr(),
                                                   single_node, this->stream);
      if (single_node) {
        this->infrequent_embeddings[i].indices_->calculate_model_indices(this->stream);
        CK_CUDA_THROW_(cudaMemcpyAsync(
            this->infrequent_embeddings[i].interaction_layer_input_pointers_train_.get_ptr(),
            interaction_layer_input_pointers_.data(), this->num_instances * sizeof(emtype *),
            cudaMemcpyHostToDevice, this->stream));
        this->infrequent_embeddings[i].forward_network_direct(true, this->stream);
      } else {
        this->infrequent_embeddings[i].indices_->calculate_network_indices(80, this->stream);
        upload_tensor(cpu_embedding.forward_received_messages[i], received_messages[i],
                      this->stream);
        if (this->config.comm_type == CommunicationType::IB_NVLink_Hier) {
          this->infrequent_embeddings[i].hier_forward_network(
              received_messages[i].get_ptr(), interaction_layer_input[i].get_ptr(), this->stream);
        } else {
          this->infrequent_embeddings[i].forward_network(
              received_messages[i].get_ptr(), interaction_layer_input[i].get_ptr(), this->stream);
        }
      }
    }

    std::vector<std::vector<emtype>> h_interaction_layer_input(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      download_tensor(h_interaction_layer_input[i], interaction_layer_input[i], this->stream);
    }

    /* Compare */
    for (size_t i = 0; i < this->num_instances; i++) {
      ASSERT_TRUE(compare_array(
          local_batch_size * this->config.num_tables * this->config.embedding_vec_size,
          h_interaction_layer_input[i].data(), cpu_embedding.interaction_layer_input[i].data(),
          1e-2));
    }
  }
};

/************** Frequent embedding forward model (single node) **************/

template <typename dtype, typename emtype>
class FrequentForwardModelTest : public HybridEmbeddingUnitTest<dtype, emtype> {
 protected:
 public:
  FrequentForwardModelTest(const HybridEmbeddingConfig<dtype> config, size_t batch_size,
                           size_t seed = 1234ll)
      : HybridEmbeddingUnitTest<dtype, emtype>(config, batch_size, seed) {}

  void run() {
    uint32_t local_batch_size = ceildiv<uint32_t>(this->batch_size, this->num_instances);

    HybridEmbeddingCpu<dtype, emtype> cpu_embedding(this->config, this->batch_size,
                                                    this->category_location,
                                                    this->category_frequent_index, this->samples);
    cpu_embedding.calculate_frequent_network_cache_indices();
    cpu_embedding.generate_embedding_vectors();
    cpu_embedding.generate_gradients();
    cpu_embedding.frequent_reduce_gradients();

    /* Tensors for the gradients */
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
    std::vector<Tensor2<emtype>> gradients(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      buff->reserve({local_batch_size * this->config.num_tables, this->config.embedding_vec_size},
                    &gradients[i]);
    }
    buff->allocate();

    /* Frequent update_model */
    this->build_frequent();
    std::vector<const emtype *> frequent_partial_gradients_pointers(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      upload_tensor(cpu_embedding.frequent_embedding_vectors[i],
                    this->frequent_embeddings[i].frequent_embedding_vectors_, this->stream);
      upload_tensor(cpu_embedding.gradients[i], gradients[i], this->stream);
      frequent_partial_gradients_pointers[i] =
          this->frequent_embeddings[i].get_gradients().get_ptr();
      this->frequent_embeddings[i].set_current_indices(&this->frequent_embedding_indices[i], this->stream);
    }
    for (size_t i = 0; i < this->num_instances; i++) {
      this->frequent_embeddings[i].indices_->calculate_cache_masks(this->stream);
      this->frequent_embeddings[i].indices_->calculate_network_cache_indices(this->stream);
      this->frequent_embeddings[i].indices_->calculate_model_cache_indices(80, this->stream);
      this->frequent_embeddings[i].indices_->calculate_frequent_sample_indices(this->stream);
      this->frequent_embeddings[i].local_reduce(gradients[i].get_ptr(), this->stream, false);
    }
    for (size_t i = 0; i < this->num_instances; i++) {
      CK_CUDA_THROW_(cudaMemcpyAsync(
          this->frequent_embeddings[i].partial_gradients_pointers_.get_ptr(),
          frequent_partial_gradients_pointers.data(), this->num_instances * sizeof(emtype *),
          cudaMemcpyHostToDevice, this->stream));
      this->frequent_embeddings[i].update_model_direct(this->dev_lr, 1.f, this->stream);
    }

    /* Set cache to zero for easy comparison with CPU version */
    if (sizeof(emtype) != sizeof(float)) {
      for (size_t i = 0; i < this->num_instances; i++) {
        CK_CUDA_THROW_(cudaMemsetAsync(
            this->frequent_embeddings[i].get_embedding_vectors_cache().get_ptr(), 0,
            this->config.num_frequent * this->config.embedding_vec_size * sizeof(emtype),
            this->stream));
      }
    }

    /* Frequent forward_model */
    for (size_t i = 0; i < this->num_instances; i++) {
      this->frequent_embeddings[i].forward_model(this->stream);
    }

    std::vector<std::vector<emtype>> updated_vectors_cache(this->num_instances);
    for (size_t i = 0; i < this->num_instances; i++) {
      download_tensor(updated_vectors_cache[i],
                      this->frequent_embeddings[i].get_embedding_vectors_cache(), this->stream);
    }

    /* Reference update_model */
    cpu_embedding.frequent_update_single_node();

    /* Reference forward_model */
    cpu_embedding.frequent_forward_model();

    /* Compare */
    for (size_t i = 0; i < this->num_instances; i++) {
      ASSERT_TRUE(compare_array(this->config.num_frequent * this->config.embedding_vec_size,
                                updated_vectors_cache[i].data(),
                                cpu_embedding.frequent_embedding_vectors_cache[i].data(), 5e-2));
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

/* hybrid_embedding_forward_network_test */

TEST(hybrid_embedding_forward_network_test, uint32_half_64) {
  ForwardNetworkTest<uint32_t, __half>(config_uint32, 64, false).run();
}

TEST(hybrid_embedding_forward_network_test, int64_half_64) {
  ForwardNetworkTest<long long, __half>(config_int64, 64, false).run();
}

TEST(hybrid_embedding_forward_network_test, uint32_half_2048) {
  ForwardNetworkTest<uint32_t, __half>(config_uint32, 2048, false).run();
}

TEST(hybrid_embedding_forward_network_test, int64_half_2048) {
  ForwardNetworkTest<long long, __half>(config_int64, 2048, false).run();
}

TEST(hybrid_embedding_forward_network_test, uint32_float_64) {
  ForwardNetworkTest<uint32_t, float>(config_uint32, 64, false).run();
}

TEST(hybrid_embedding_forward_network_test, int64_float_64) {
  ForwardNetworkTest<long long, float>(config_int64, 64, false).run();
}

TEST(hybrid_embedding_forward_network_test, uint32_float_2048) {
  ForwardNetworkTest<uint32_t, float>(config_uint32, 2048, false).run();
}

TEST(hybrid_embedding_forward_network_test, int64_float_2048) {
  ForwardNetworkTest<long long, float>(config_int64, 2048, false).run();
}

TEST(hybrid_embedding_forward_network_test, uint32_float_128_no_freq) {
  ForwardNetworkTest<uint32_t, float>(config_no_freq, 128, false).run();
}

TEST(hybrid_embedding_forward_network_test, uint32_float_128_all_freq) {
  ForwardNetworkTest<uint32_t, float>(config_all_freq, 128, false).run();
}

/* hybrid_embedding_forward_network_single_node_test */

TEST(hybrid_embedding_forward_network_single_node_test, uint32_half_64) {
  ForwardNetworkTest<uint32_t, __half>(config_uint32_single_node, 64, true).run();
}

TEST(hybrid_embedding_forward_network_single_node_test, int64_half_64) {
  ForwardNetworkTest<long long, __half>(config_int64_single_node, 64, true).run();
}

TEST(hybrid_embedding_forward_network_single_node_test, uint32_half_2048) {
  ForwardNetworkTest<uint32_t, __half>(config_uint32_single_node, 2048, true).run();
}

TEST(hybrid_embedding_forward_network_single_node_test, int64_half_2048) {
  ForwardNetworkTest<long long, __half>(config_int64_single_node, 2048, true).run();
}

TEST(hybrid_embedding_forward_network_single_node_test, uint32_float_64) {
  ForwardNetworkTest<uint32_t, float>(config_uint32_single_node, 64, true).run();
}

TEST(hybrid_embedding_forward_network_single_node_test, int64_float_64) {
  ForwardNetworkTest<long long, float>(config_int64_single_node, 64, true).run();
}

TEST(hybrid_embedding_forward_network_single_node_test, uint32_float_2048) {
  ForwardNetworkTest<uint32_t, float>(config_uint32_single_node, 2048, true).run();
}

TEST(hybrid_embedding_forward_network_single_node_test, int64_float_2048) {
  ForwardNetworkTest<long long, float>(config_int64_single_node, 2048, true).run();
}

TEST(hybrid_embedding_forward_network_single_node_test, uint32_float_128_no_freq) {
  ForwardNetworkTest<uint32_t, float>(config_no_freq_single_node, 128, true).run();
}

TEST(hybrid_embedding_forward_network_single_node_test, uint32_float_128_all_freq) {
  ForwardNetworkTest<uint32_t, float>(config_all_freq_single_node, 128, true).run();
}

/* hybrid_embedding_forward_network_hier_test */

TEST(hybrid_embedding_forward_network_hier_test, uint32_half_64) {
  ForwardNetworkTest<uint32_t, __half>(config_uint32_hier, 64, false).run();
}

TEST(hybrid_embedding_forward_network_hier_test, int64_half_64) {
  ForwardNetworkTest<long long, __half>(config_int64_hier, 64, false).run();
}

TEST(hybrid_embedding_forward_network_hier_test, uint32_half_2048) {
  ForwardNetworkTest<uint32_t, __half>(config_uint32_hier, 2048, false).run();
}

TEST(hybrid_embedding_forward_network_hier_test, int64_half_2048) {
  ForwardNetworkTest<long long, __half>(config_int64_hier, 2048, false).run();
}

TEST(hybrid_embedding_forward_network_hier_test, uint32_float_64) {
  ForwardNetworkTest<uint32_t, float>(config_uint32_hier, 64, false).run();
}

TEST(hybrid_embedding_forward_network_hier_test, int64_float_64) {
  ForwardNetworkTest<long long, float>(config_int64_hier, 64, false).run();
}

TEST(hybrid_embedding_forward_network_hier_test, uint32_float_2048) {
  ForwardNetworkTest<uint32_t, float>(config_uint32_hier, 2048, false).run();
}

TEST(hybrid_embedding_forward_network_hier_test, int64_float_2048) {
  ForwardNetworkTest<long long, float>(config_int64_hier, 2048, false).run();
}

TEST(hybrid_embedding_forward_network_hier_test, uint32_float_128_no_freq) {
  ForwardNetworkTest<uint32_t, float>(config_no_freq_hier, 128, false).run();
}

TEST(hybrid_embedding_forward_network_hier_test, uint32_float_128_all_freq) {
  ForwardNetworkTest<uint32_t, float>(config_all_freq_hier, 128, false).run();
}

/* hybrid_embedding_frequent_forward_model_test */

TEST(hybrid_embedding_frequent_forward_model_test, uint32_half_64) {
  FrequentForwardModelTest<uint32_t, __half>(config_uint32_single_node, 64).run();
}

TEST(hybrid_embedding_frequent_forward_model_test, int64_half_64) {
  FrequentForwardModelTest<long long, __half>(config_int64_single_node, 64).run();
}

TEST(hybrid_embedding_frequent_forward_model_test, uint32_half_2048) {
  FrequentForwardModelTest<uint32_t, __half>(config_uint32_single_node, 2048).run();
}

TEST(hybrid_embedding_frequent_forward_model_test, int64_half_2048) {
  FrequentForwardModelTest<long long, __half>(config_int64_single_node, 2048).run();
}

TEST(hybrid_embedding_frequent_forward_model_test, uint32_float_64) {
  FrequentForwardModelTest<uint32_t, float>(config_uint32_single_node, 64).run();
}

TEST(hybrid_embedding_frequent_forward_model_test, int64_float_64) {
  FrequentForwardModelTest<long long, float>(config_int64_single_node, 64).run();
}

TEST(hybrid_embedding_frequent_forward_model_test, uint32_float_2048) {
  FrequentForwardModelTest<uint32_t, float>(config_uint32_single_node, 2048).run();
}

TEST(hybrid_embedding_frequent_forward_model_test, int64_float_2048) {
  FrequentForwardModelTest<long long, float>(config_int64_single_node, 2048).run();
}

TEST(hybrid_embedding_frequent_forward_model_test, uint32_float_128_no_freq) {
  FrequentForwardModelTest<uint32_t, float>(config_no_freq_single_node, 128).run();
}

TEST(hybrid_embedding_frequent_forward_model_test, uint32_float_128_all_freq) {
  FrequentForwardModelTest<uint32_t, float>(config_all_freq_single_node, 128).run();
}
