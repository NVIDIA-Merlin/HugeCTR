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

#pragma once

#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/frequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/infrequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "input_generator.hpp"

using namespace HugeCTR;
using namespace HugeCTR::hybrid_embedding;

template <typename dtype, typename emtype>
class HybridEmbeddingUnitTest {
 protected:
  const HybridEmbeddingConfig<dtype> config;
  HybridEmbeddingInputGenerator<dtype> input_generator;
  const uint32_t batch_size;
  const uint32_t num_instances;
  const uint32_t embedding_vec_size;

  const std::vector<dtype> category_location;
  const std::vector<dtype> category_frequent_index;
  const std::vector<dtype> samples;
  const std::vector<size_t> table_sizes;

  cudaStream_t stream;
  // std::shared_ptr<ResourceManager> fake_resource_manager;
  GPUResource fake_resource;
  std::vector<Model<dtype>> model_list;
  std::vector<Data<dtype>> data_list;
  std::vector<FrequentEmbedding<dtype, emtype>> frequent_embeddings;
  std::vector<InfrequentEmbedding<dtype, emtype>> infrequent_embeddings;

  float* dev_lr;

 public:
  void build_model() {
    model_list.reserve(num_instances);
    std::vector<uint32_t> num_instances_per_node_list(config.num_nodes,
                                                      config.num_instances / config.num_nodes);
    for (size_t i = 0; i < num_instances; i++) {
      model_list.emplace_back(config.comm_type, i, num_instances_per_node_list,
                              config.num_categories);
      model_list[i].num_frequent = config.num_frequent;
    }

    for (size_t i = 0; i < num_instances; i++) {
      upload_tensor(category_frequent_index, model_list[i].category_frequent_index, stream);
      upload_tensor(category_location, model_list[i].category_location, stream);
    }
  }

  void build_data() {
    data_list.reserve(num_instances);
    for (size_t i = 0; i < num_instances; i++) {
      data_list.emplace_back(table_sizes, batch_size, 1);
    }

    for (size_t i = 0; i < num_instances; i++) {
      upload_tensor(samples, data_list[i].samples, stream);
      upload_tensor(samples, data_list[i].samples, stream);
    }

    CK_CUDA_THROW_(cudaMalloc(&dev_lr, sizeof(float)));
    CK_CUDA_THROW_(cudaMemcpy(dev_lr, &config.lr, sizeof(float),
                              cudaMemcpyHostToDevice));
  }

  void build_frequent() {
    frequent_embeddings.reserve(num_instances);
    for (size_t i = 0; i < num_instances; i++) {
      std::shared_ptr<BufferBlock2<emtype>> placeholder = NULL;
      frequent_embeddings.emplace_back(data_list[i], data_list[i], model_list[i], fake_resource, placeholder,
                                       embedding_vec_size, config.num_frequent);
    }

    if (config.comm_type == CommunicationType::NVLink_SingleNode) {
      std::vector<emtype *> h_vectors_cache_pointers(num_instances);
      for (uint32_t i = 0; i < num_instances; i++) {
        h_vectors_cache_pointers[i] =
            frequent_embeddings[i].get_embedding_vectors_cache().get_ptr();
      }
      for (uint32_t i = 0; i < num_instances; i++) {
        CK_CUDA_THROW_(
            cudaMemcpyAsync(frequent_embeddings[i].embedding_vectors_cache_pointers_.get_ptr(),
                            h_vectors_cache_pointers.data(), num_instances * sizeof(emtype *),
                            cudaMemcpyHostToDevice, stream));
      }
    }
  }

  void build_infrequent() {
    infrequent_embeddings.reserve(num_instances);
    for (size_t i = 0; i < num_instances; i++) {
      infrequent_embeddings.emplace_back(data_list[i], data_list[i], model_list[i], fake_resource,
                                         embedding_vec_size);
      uint32_t samples_size = data_list[i].batch_size * data_list[i].table_sizes.size();
      infrequent_embeddings[i].max_num_infrequent_per_batch_ = samples_size;
      infrequent_embeddings[i].max_num_infrequent_per_train_batch_ = samples_size;
    }
  }

  ncclComm_t get_fake_comm() {
    ncclComm_t comm;
    int device_list[1] = {0};
    ncclCommInitAll(&comm, 1, device_list);
    return comm;
  }

  HybridEmbeddingUnitTest(const HybridEmbeddingConfig<dtype> config, size_t batch_size,
                          size_t seed = 1234ll)
      : config(config),
        input_generator(config, seed),
        batch_size(batch_size),
        num_instances(config.num_instances),
        embedding_vec_size(config.embedding_vec_size),
        category_location((input_generator.generate_category_location(),
                           input_generator.get_category_location())),
        category_frequent_index(input_generator.get_category_frequent_index()),
        samples(input_generator.generate_flattened_categorical_input(batch_size)),
        table_sizes(input_generator.get_table_sizes()),
        fake_resource(0, 0, 0, seed, seed, get_fake_comm()) {
    CK_CUDA_THROW_(cudaStreamCreate(&stream));
    build_model();
    build_data();
  }
};

inline bool compare_element(float a, float b, float epsilon) {
  // compare absolute error
  if (fabs(a - b) < epsilon) return true;

  // compare relative error
  if (fabs(a) >= fabs(b))
    if (fabs((a - b) / a) < epsilon)
      return true;
    else
      return false;
  else if (fabs((a - b) / b) < epsilon)
    return true;
  else
    return false;
}

inline bool compare_array(size_t len, const float *a, const float *b, float epsilon) {
  for (size_t i = 0; i < len; i++) {
    if (!compare_element(a[i], b[i], epsilon)) {
      printf("Error in compare_array: i=%zu, a=%.8f, b=%.8f\n", i, a[i], b[i]);
      return false;
    }
  }

  return true;
}

// overload for fp16 on GPU
inline bool compare_array(size_t len, const __half *a, const __half *b, float epsilon) {
  for (size_t i = 0; i < len; i++) {
    float fa = __half2float(a[i]);
    float fb = __half2float(b[i]);
    if (!compare_element(fa, fb, epsilon)) {
      printf("Error in compare_array: i=%zu, a=%.8f, b=%.8f\n", i, fa, fb);
      return false;
    }
  }

  return true;
}
