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

#include <cuda_profiler_api.h>
#include <omp.h>

#include <algorithm>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "HugeCTR/include/inference/embedding_interface.hpp"
#include "HugeCTR/include/inference/memory_pool.hpp"
#include "HugeCTR/include/inference/session_inference.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

#define HIT_RATE_THRESHOLD 0.6
#define CACHE_SIZE_PERCENTAGE 0.5
#define UNIQUE_KNOWN_PERCENTAGE 0.1
#define UNIQUE_UNKNOWN_PERCENTAGE 0.1
#define MODEL_PATH "/workdir/test/utest/simple_inference_config.json"
#define SPARSE_MODEL_PATH "/hugectr/test/utest/0_sparse_10000.model"
#define DENSE_MODEL_PATH "/hugectr/test/utest/_dense_10000.model"
#define MODEL_NAME "DCN"
#define BATCHSIZE 1024

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

using namespace HugeCTR;
namespace {

// The random number generator
template <typename T>
class IntGenerator {
 public:
  IntGenerator() : gen_(rd_()) {}
  IntGenerator(T min, T max) : gen_(rd_()), distribution_(min, max) {}

  void fill_unique(T* data, size_t len, T empty_value) {
    if (len == 0) {
      return;
    }
    assert(distribution_.max() - distribution_.min() >= len);

    std::unordered_set<T> set;
    size_t sz = 0;
    while (sz < len) {
      T x = distribution_(gen_);
      if (x == empty_value) {
        continue;
      }
      auto res = set.insert(x);
      if (res.second) {
        data[sz++] = x;
      }
    }
    assert(sz == set.size());
    assert(sz == len);
  }

 private:
  std::random_device rd_;
  std::mt19937 gen_;
  std::uniform_int_distribution<T> distribution_;
};

// Test Model Parameter, 1 per model
struct InferenceInfo {
  std::vector<int> max_feature_num_per_sample_;
  std::vector<bool> distributed_emb_;
  std::vector<size_t> embedding_vec_size_;
  std::vector<float> default_emb_vector_value_;
  InferenceInfo(const nlohmann::json& config);
};

InferenceInfo::InferenceInfo(const nlohmann::json& config) {
  const nlohmann::json& j_layers = get_json(config, "layers");
  const nlohmann::json& j_data_layer = j_layers[0];
  const nlohmann::json& j_data_layer_sparse_layer = get_json(j_data_layer, "sparse");
  for (unsigned int i = 0; i < j_data_layer_sparse_layer.size(); i++) {
    size_t max_feature_num_per_sample = static_cast<size_t>(
        get_max_feature_num_per_sample_from_nnz_per_slot(j_data_layer_sparse_layer[i]));

    max_feature_num_per_sample_.emplace_back(max_feature_num_per_sample);
  }

  for (unsigned int i = 1; i < j_layers.size(); i++) {
    const nlohmann::json& j_single_layer = j_layers[i];
    std::string embedding_type = get_value_from_json<std::string>(j_single_layer, "type");
    if (embedding_type.compare("DistributedSlotSparseEmbeddingHash") == 0) {
      distributed_emb_.emplace_back(true);
      const nlohmann::json& embedding_hparam = get_json(j_single_layer, "sparse_embedding_hparam");
      embedding_vec_size_.emplace_back(
          get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size"));
      default_emb_vector_value_.emplace_back(
          get_value_from_json_soft<float>(embedding_hparam, "default_emb_vec_value", 0.0f));
    } else if (embedding_type.compare("LocalizedSlotSparseEmbeddingHash") == 0 ||
               embedding_type.compare("LocalizedSlotSparseEmbeddingOneHot") == 0) {
      distributed_emb_.emplace_back(false);
      const nlohmann::json& embedding_hparam = get_json(j_single_layer, "sparse_embedding_hparam");
      embedding_vec_size_.emplace_back(
          get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size"));
      default_emb_vector_value_.emplace_back(
          get_value_from_json_soft<float>(embedding_hparam, "default_emb_vec_value", 0.0f));
    } else {
      break;
    }
  }
}

// Designed for 1 model with 1 embedding table currently
template <typename TypeHashKey>
void embedding_cache_test(const std::string& config_file, const std::string& model,
                          const std::string& sparse_model_file, size_t num_of_sample,
                          int num_feature_per_sample, size_t num_of_iteration, size_t num_of_worker,
                          bool use_gpu_cache) {
  // Test will use 0# GPU
  CK_CUDA_THROW_(cudaSetDevice(0));

  InferenceInfo inference_info(read_json_file(config_file));

  size_t max_batch_size = BATCHSIZE;
  int max_feature_num_per_sample = inference_info.max_feature_num_per_sample_[0];
  float default_emb_vector_value = inference_info.default_emb_vector_value_[0];
  size_t num_emb_table = 1;
  // Check parameter
  num_of_sample = num_of_sample < max_batch_size ? num_of_sample : max_batch_size;
  num_feature_per_sample = num_feature_per_sample < max_feature_num_per_sample
                               ? num_feature_per_sample
                               : max_feature_num_per_sample;

  std::vector<size_t> num_feature(num_of_sample);
  // Generate num_of_feature for each sample if needed
  if (num_feature_per_sample == -1) {
    num_feature_per_sample = max_feature_num_per_sample;
    IntGenerator<size_t> random_gen(0, num_feature_per_sample);
    for (size_t i = 0; i < num_of_sample; i++) {
      random_gen.fill_unique(&num_feature[i], 1, num_feature_per_sample + 1);
    }
  } else {
    for (size_t i = 0; i < num_of_sample; i++) {
      num_feature[i] = num_feature_per_sample;
    }
  }

  // Prepare h_embedding_offset
  std::vector<size_t> h_embedding_offset(num_emb_table * num_of_sample + 1);
  size_t acc_offset = 0;
  for (size_t i = 0; i < num_emb_table * num_of_sample; i++) {
    h_embedding_offset[i] = acc_offset;
    acc_offset += num_feature[i];
  }
  h_embedding_offset[num_emb_table * num_of_sample] = acc_offset;
  size_t feature_per_batch = acc_offset;

  // Calculate 3 parts of features to be queried for each sample: known embeddings in emb_file,
  // unknown embeddings, and duplicated embeddings
  std::vector<size_t> num_known_embeddingcolumns(num_of_sample);
  std::vector<size_t> num_unknown_embeddingcolumns(num_of_sample);
  std::vector<size_t> num_duplicate_embeddingcolumns(num_of_sample);
  for (size_t sample_id = 0; sample_id < num_of_sample; sample_id++) {
    // If this sample doesn't need embeddings, skip
    if (num_feature[sample_id] == 0) {
      num_known_embeddingcolumns[sample_id] = 0;
      num_unknown_embeddingcolumns[sample_id] = 0;
      num_duplicate_embeddingcolumns[sample_id] = 0;
      continue;
    }
    size_t remain_feature = num_feature[sample_id];
    size_t sample_known_embeddingcolumns = UNIQUE_KNOWN_PERCENTAGE * num_feature[sample_id];
    sample_known_embeddingcolumns = std::max(sample_known_embeddingcolumns, (size_t)1);
    sample_known_embeddingcolumns = std::min(sample_known_embeddingcolumns, remain_feature);
    remain_feature -= sample_known_embeddingcolumns;
    size_t sample_unknown_embeddingcolumns = UNIQUE_UNKNOWN_PERCENTAGE * num_feature[sample_id];
    sample_unknown_embeddingcolumns = std::min(sample_unknown_embeddingcolumns, remain_feature);
    remain_feature -= sample_unknown_embeddingcolumns;
    size_t sample_duplicate_embeddingcolumns = remain_feature;
    ASSERT_TRUE(sample_known_embeddingcolumns + sample_unknown_embeddingcolumns +
                    sample_duplicate_embeddingcolumns ==
                num_feature[sample_id]);
    num_known_embeddingcolumns[sample_id] = sample_known_embeddingcolumns;
    num_unknown_embeddingcolumns[sample_id] = sample_unknown_embeddingcolumns;
    num_duplicate_embeddingcolumns[sample_id] = sample_duplicate_embeddingcolumns;
  }

  // Read all the embeddings from the model
  std::string emb_file_path = sparse_model_file;
  size_t embedding_vec_size = inference_info.embedding_vec_size_[0];

  std::string emb_file_prefix(sparse_model_file + "/");
  std::ifstream key_stream(emb_file_prefix + "key");
  std::ifstream vec_stream(emb_file_prefix + "emb_vector");
  // Check if file is opened successfully
  if (!key_stream.is_open() || !vec_stream.is_open()) {
    CK_THROW_(Error_t::WrongInput, "Error: embeddings file cannot open for reading");
  }

  // The buffers for all embeddings
  TypeHashKey* h_total_embeddingcolumns;
  float* h_total_embeddingvector;
  // The max value of all embedding ids
  TypeHashKey max_emb_id = 0;
  // The num of embeddings in the file
  size_t row_num = fs::file_size(emb_file_prefix + "key") / sizeof(long long);

  CK_CUDA_THROW_(cudaHostAlloc((void**)&h_total_embeddingcolumns, row_num * sizeof(TypeHashKey),
                               cudaHostAllocPortable));
  CK_CUDA_THROW_(cudaHostAlloc((void**)&h_total_embeddingvector,
                               row_num * embedding_vec_size * sizeof(float),
                               cudaHostAllocPortable));
  for (size_t pair = 0; pair < row_num; pair++) {
    // Read out the emb_id and emb_vec
    if (std::is_same<TypeHashKey, long long>::value) {
      key_stream.read(reinterpret_cast<char*>(h_total_embeddingcolumns + pair), sizeof(long long));
    } else {
      long long tmp_key;
      key_stream.read(reinterpret_cast<char*>(&tmp_key), sizeof(long long));
      h_total_embeddingcolumns[pair] = static_cast<TypeHashKey>(tmp_key);
    }
    vec_stream.read(reinterpret_cast<char*>(h_total_embeddingvector + pair * embedding_vec_size),
                    sizeof(float) * embedding_vec_size);
    // Calculate the max id
    max_emb_id = std::max(max_emb_id, h_total_embeddingcolumns[pair]);
  }

  // Parameter server and embedding cache shared by all workers
  std::vector<std::string> model_config_path{config_file};
  std::string dense_model{DENSE_MODEL_PATH};
  std::vector<std::string> sparse_models{SPARSE_MODEL_PATH};
  InferenceParams infer_param(model, max_batch_size, HIT_RATE_THRESHOLD, dense_model, sparse_models,
                              0, true, CACHE_SIZE_PERCENTAGE, false);
  infer_param.number_of_worker_buffers_in_pool = num_of_worker * 2;
  std::vector<InferenceParams> inference_params{infer_param};
  std::shared_ptr<HugectrUtility<TypeHashKey>> parameter_server;
  parameter_server.reset(HugectrUtility<TypeHashKey>::Create_Parameter_Server(
      INFER_TYPE::TRITON, model_config_path, inference_params));
  auto embedding_cache = parameter_server->GetEmbeddingCache(model, 0);

  // Each worker start to do inference
#pragma omp parallel default(none)                                                                \
    shared(feature_per_batch, embedding_vec_size, num_emb_table, embedding_cache, row_num,        \
           num_of_iteration, num_of_sample, num_feature, num_known_embeddingcolumns, infer_param, \
           num_unknown_embeddingcolumns, num_duplicate_embeddingcolumns, h_embedding_offset,      \
           h_total_embeddingcolumns, h_total_embeddingvector, max_emb_id,                         \
           default_emb_vector_value) num_threads(num_of_worker)
  {
    // All workers will share the #0 GPU
    CK_CUDA_THROW_(cudaSetDevice(0));
    // Get thread id
    int thread_id = omp_get_thread_num();
    int num_of_thread = omp_get_num_threads();
    if (thread_id == 0) {
      printf("Number of workers: %d.\n", num_of_thread);
    }
    // Each worker create IO buffers
    size_t* h_index;
    TypeHashKey* h_embeddingcolumns;
    float* h_shuffled_embeddingoutputvector;
    float* h_expected_shuffled_embeddingoutputvector;
    CK_CUDA_THROW_(
        cudaHostAlloc((void**)&h_index, feature_per_batch * sizeof(size_t), cudaHostAllocPortable));
    CK_CUDA_THROW_(cudaHostAlloc((void**)&h_embeddingcolumns,
                                 feature_per_batch * sizeof(TypeHashKey), cudaHostAllocPortable));
    CK_CUDA_THROW_(cudaHostAlloc((void**)&h_shuffled_embeddingoutputvector,
                                 feature_per_batch * embedding_vec_size * sizeof(float),
                                 cudaHostAllocPortable));
    CK_CUDA_THROW_(cudaHostAlloc((void**)&h_expected_shuffled_embeddingoutputvector,
                                 feature_per_batch * embedding_vec_size * sizeof(float),
                                 cudaHostAllocPortable));
    float* d_shuffled_embeddingoutputvector;
    CK_CUDA_THROW_(cudaMalloc((void**)&d_shuffled_embeddingoutputvector,
                              feature_per_batch * embedding_vec_size * sizeof(float)));
    // Each worker create CUDA stream used to look_up
    std::vector<cudaStream_t> query_streams(num_emb_table);
    for (size_t i = 0; i < num_emb_table; i++) {
      CK_CUDA_THROW_(cudaStreamCreate(&query_streams[i]));
    }
    // Apply a memory block for embedding cache look up
    MemoryBlock* memory_block = NULL;
    while (memory_block == NULL) {
      memory_block = reinterpret_cast<MemoryBlock*>(embedding_cache->get_worker_space(
          infer_param.model_name, infer_param.device_id, CACHE_SPACE_TYPE::WORKER));
    }
    // Random number generator for selecting known embedding ids from embedding table
    IntGenerator<size_t> index_gen(0, row_num - 1);

    // Each worker start to do iteration
    for (size_t iter = 0; iter < num_of_iteration; iter++) {
      // Generate embedding ids for each sample for this iteration
      for (size_t sample_id = 0; sample_id < num_of_sample; sample_id++) {
        // If this sample doesn't need embeddings, skip
        if (num_feature[sample_id] == 0) {
          continue;
        }
        size_t sample_known_embeddingcolumns = num_known_embeddingcolumns[sample_id];
        size_t sample_unknown_embeddingcolumns = num_unknown_embeddingcolumns[sample_id];
        size_t sample_duplicate_embeddingcolumns = num_duplicate_embeddingcolumns[sample_id];
        size_t sample_base_offset = h_embedding_offset[sample_id];

        index_gen.fill_unique(h_index, sample_known_embeddingcolumns, row_num);
        // Select embedding ids from embedding table
        for (size_t id = 0; id < sample_known_embeddingcolumns; id++) {
          size_t current_index = id + sample_base_offset;
          h_embeddingcolumns[current_index] = h_total_embeddingcolumns[h_index[id]];
          float* dst_emb_vec_ptr =
              h_expected_shuffled_embeddingoutputvector + current_index * embedding_vec_size;
          float* src_emb_vec_ptr = h_total_embeddingvector + h_index[id] * embedding_vec_size;
          memcpy(dst_emb_vec_ptr, src_emb_vec_ptr, embedding_vec_size * sizeof(float));
        }
        // Fill unknown embedding ids
        for (size_t id = 0; id < sample_unknown_embeddingcolumns; id++) {
          size_t current_index = id + sample_base_offset + sample_known_embeddingcolumns;
          h_embeddingcolumns[current_index] = max_emb_id + id + 1;
          for (size_t float_id = 0; float_id < embedding_vec_size; float_id++) {
            size_t emb_vec_base_offset = current_index * embedding_vec_size;
            h_expected_shuffled_embeddingoutputvector[emb_vec_base_offset + float_id] =
                default_emb_vector_value;
          }
        }

        // Fill duplicated embedding ids
        for (size_t id = 0; id < sample_duplicate_embeddingcolumns; id++) {
          size_t dst_offset = id + sample_base_offset + sample_known_embeddingcolumns +
                              sample_unknown_embeddingcolumns;
          size_t src_offset =
              (id % (sample_known_embeddingcolumns + sample_unknown_embeddingcolumns)) +
              sample_base_offset;
          h_embeddingcolumns[dst_offset] = h_embeddingcolumns[src_offset];
          float* dst_emb_vec_pos =
              h_expected_shuffled_embeddingoutputvector + dst_offset * embedding_vec_size;
          float* src_emb_vec_pos =
              h_expected_shuffled_embeddingoutputvector + src_offset * embedding_vec_size;
          memcpy(dst_emb_vec_pos, src_emb_vec_pos, embedding_vec_size * sizeof(float));
        }
      }
      // Each worker query the shared embedding cache
      embedding_cache->look_up((void*)h_embeddingcolumns, h_embedding_offset,
                               d_shuffled_embeddingoutputvector, memory_block, query_streams);

      // Each worker wait for look_up to complete
      for (size_t emb_table = 0; emb_table < num_emb_table; emb_table++) {
        CK_CUDA_THROW_(cudaStreamSynchronize(query_streams[emb_table]));
      }
      // Each worker copy look_up result back to host
      CK_CUDA_THROW_(cudaMemcpyAsync(h_shuffled_embeddingoutputvector,
                                     d_shuffled_embeddingoutputvector,
                                     feature_per_batch * embedding_vec_size * sizeof(float),
                                     cudaMemcpyDeviceToHost, query_streams[0]));
      // Each worker wait for copy to complete
      CK_CUDA_THROW_(cudaStreamSynchronize(query_streams[0]));
      // Each worker check the correctness, both buffer should be bit-indentical
      bool result_correct = true;
      for (size_t float_id = 0; float_id < feature_per_batch * embedding_vec_size; float_id++) {
        if (h_shuffled_embeddingoutputvector[float_id] !=
            h_expected_shuffled_embeddingoutputvector[float_id]) {
          result_correct = false;
          break;
        }
      }
      // ASSERT_TRUE(result_correct);
      if (result_correct == false) {
        CK_THROW_(Error_t::DataCheckError,
                  "Error: The result of embedding_cache is not as expected");
      }
    }
    // Each worker clean its buffers
    // If no features need to be queried, then don't need to free them since they are not allocated
    if (feature_per_batch != 0) {
      CK_CUDA_THROW_(cudaFreeHost(h_index));
      CK_CUDA_THROW_(cudaFreeHost(h_embeddingcolumns));
      CK_CUDA_THROW_(cudaFreeHost(h_shuffled_embeddingoutputvector));
      CK_CUDA_THROW_(cudaFreeHost(h_expected_shuffled_embeddingoutputvector));
      CK_CUDA_THROW_(cudaFree(d_shuffled_embeddingoutputvector));
    }
    for (size_t i = 0; i < num_emb_table; i++) {
      CK_CUDA_THROW_(cudaStreamDestroy(query_streams[i]));
    }
  }

  // clean up
  CK_CUDA_THROW_(cudaFreeHost(h_total_embeddingcolumns));
  CK_CUDA_THROW_(cudaFreeHost(h_total_embeddingvector));
}

}  // namespace

TEST(embedding_cache, embedding_cache_usigned_int_0_0_5_1_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 0, 0, 5, 1, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_0_0_5_1_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 0, 0, 5, 1, false);
}
TEST(embedding_cache, embedding_cache_usigned_int_16_0_5_1_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 16, 0, 5, 1, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_16_0_5_1_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 16, 0, 5, 1, false);
}
TEST(embedding_cache, embedding_cache_usigned_int_32_0_5_1_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 32, 0, 5, 1, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_32_0_5_1_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 32, 0, 5, 1, false);
}
TEST(embedding_cache, embedding_cache_usigned_int_0_16_5_1_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 0, 16, 5, 1, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_0_16_5_1_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 0, 16, 5, 1, false);
}
TEST(embedding_cache, embedding_cache_usigned_int_16_16_5_1_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 16, 16, 5, 1, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_16_16_5_1_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 16, 16, 5, 1,
                                     false);
}
TEST(embedding_cache, embedding_cache_usigned_int_32_16_5_1_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 32, 16, 5, 1, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_32_16_5_1_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 32, 16, 5, 1,
                                     false);
}
TEST(embedding_cache, embedding_cache_usigned_int_0_30_5_1_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 0, 30, 5, 1, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_0_30_5_1_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 0, 30, 5, 1, false);
}
TEST(embedding_cache, embedding_cache_usigned_int_16_30_5_1_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 16, 30, 5, 1, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_16_30_5_1_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 16, 30, 5, 1,
                                     false);
}
TEST(embedding_cache, embedding_cache_usigned_int_32_30_5_1_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 32, 30, 5, 1, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_32_30_5_1_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 32, 30, 5, 1,
                                     false);
}
TEST(embedding_cache, embedding_cache_usigned_int_32_random_5_1_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 32, -1, 5, 1, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_32_random_5_1_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 32, -1, 5, 1,
                                     false);
}

TEST(embedding_cache, embedding_cache_usigned_int_0_0_5_4_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 0, 0, 5, 4, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_0_0_5_4_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 0, 0, 5, 4, false);
}
TEST(embedding_cache, embedding_cache_usigned_int_16_0_5_4_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 16, 0, 5, 4, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_16_0_5_4_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 16, 0, 5, 4, false);
}
TEST(embedding_cache, embedding_cache_usigned_int_32_0_5_4_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 32, 0, 5, 4, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_32_0_5_4_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 32, 0, 5, 4, false);
}
TEST(embedding_cache, embedding_cache_usigned_int_0_16_5_4_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 0, 16, 5, 4, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_0_16_5_4_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 0, 16, 5, 4, false);
}
TEST(embedding_cache, embedding_cache_usigned_int_16_16_5_4_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 16, 16, 5, 4, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_16_16_5_4_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 16, 16, 5, 4,
                                     false);
}
TEST(embedding_cache, embedding_cache_usigned_int_32_16_5_4_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 32, 16, 5, 4, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_32_16_5_4_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 32, 16, 5, 4,
                                     false);
}
TEST(embedding_cache, embedding_cache_usigned_int_0_30_5_4_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 0, 30, 5, 4, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_0_30_5_4_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 0, 30, 5, 4, false);
}
TEST(embedding_cache, embedding_cache_usigned_int_16_30_5_4_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 16, 30, 5, 4, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_16_30_5_4_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 16, 30, 5, 4,
                                     false);
}
TEST(embedding_cache, embedding_cache_usigned_int_32_30_5_4_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 32, 30, 5, 4, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_32_30_5_4_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 32, 30, 5, 4,
                                     false);
}
TEST(embedding_cache, embedding_cache_usigned_int_32_random_5_4_enable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 32, -1, 5, 4, true);
}
TEST(embedding_cache, embedding_cache_usigned_int_32_random_5_4_disable) {
  embedding_cache_test<unsigned int>(MODEL_PATH, MODEL_NAME, SPARSE_MODEL_PATH, 32, -1, 5, 4,
                                     false);
}

/*TEST(embedding_cache, embedding_cache_long_long_0_0_5_1_enable) {embedding_cache_test<long
long>(MODEL_PATH, MODEL_NAME, 0, 0, 5, 1, true); } TEST(embedding_cache,
embedding_cache_long_long_0_0_5_1_disable) {embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME,
0, 0, 5, 1, false); } TEST(embedding_cache, embedding_cache_long_long_16_0_5_1_enable)
{embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME, 16, 0, 5, 1, true); } TEST(embedding_cache,
embedding_cache_long_long_16_0_5_1_disable) {embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME,
16, 0, 5, 1, false); } TEST(embedding_cache, embedding_cache_long_long_32_0_5_1_enable)
{embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME, 32, 0, 5, 1, true); } TEST(embedding_cache,
embedding_cache_long_long_32_0_5_1_disable) {embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME,
32, 0, 5, 1, false); } TEST(embedding_cache, embedding_cache_long_long_0_16_5_1_enable)
{embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME, 0, 16, 5, 1, true); } TEST(embedding_cache,
embedding_cache_long_long_0_16_5_1_disable) {embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME,
0, 16, 5, 1, false); } TEST(embedding_cache, embedding_cache_long_long_16_16_5_1_enable)
{embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME, 16, 16, 5, 1, true); }
TEST(embedding_cache, embedding_cache_long_long_16_16_5_1_disable) {embedding_cache_test<long
long>(MODEL_PATH, MODEL_NAME, 16, 16, 5, 1, false); } TEST(embedding_cache,
embedding_cache_long_long_32_16_5_1_enable) {embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME,
32, 16, 5, 1, true); } TEST(embedding_cache, embedding_cache_long_long_32_16_5_1_disable)
{embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME, 32, 16, 5, 1, false); }
TEST(embedding_cache, embedding_cache_long_long_0_30_5_1_enable) {embedding_cache_test<long
long>(MODEL_PATH, MODEL_NAME, 0, 30, 5, 1, true); } TEST(embedding_cache,
embedding_cache_long_long_0_30_5_1_disable) {embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME,
0, 30, 5, 1, false); } TEST(embedding_cache, embedding_cache_long_long_16_30_5_1_enable)
{embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME, 16, 30, 5, 1, true); }
TEST(embedding_cache, embedding_cache_long_long_16_30_5_1_disable) {embedding_cache_test<long
long>(MODEL_PATH, MODEL_NAME, 16, 30, 5, 1, false); } TEST(embedding_cache,
embedding_cache_long_long_32_30_5_1_enable) {embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME,
32, 30, 5, 1, true); } TEST(embedding_cache, embedding_cache_long_long_32_30_5_1_disable)
{embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME, 32, 30, 5, 1, false); }
TEST(embedding_cache, embedding_cache_long_long_32_random_5_1_enable) {embedding_cache_test<long
long>(MODEL_PATH, MODEL_NAME, 32, -1, 5, 1, true); } TEST(embedding_cache,
embedding_cache_long_long_32_random_5_1_disable) {embedding_cache_test<long long>(MODEL_PATH,
MODEL_NAME, 32, -1, 5, 1, false); }

TEST(embedding_cache, embedding_cache_long_long_0_0_5_4_enable) {embedding_cache_test<long
long>(MODEL_PATH, MODEL_NAME, 0, 0, 5, 4, true); } TEST(embedding_cache,
embedding_cache_long_long_0_0_5_4_disable) {embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME,
0, 0, 5, 4, false); } TEST(embedding_cache, embedding_cache_long_long_16_0_5_4_enable)
{embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME, 16, 0, 5, 4, true); } TEST(embedding_cache,
embedding_cache_long_long_16_0_5_4_disable) {embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME,
16, 0, 5, 4, false); } TEST(embedding_cache, embedding_cache_long_long_32_0_5_4_enable)
{embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME, 32, 0, 5, 4, true); } TEST(embedding_cache,
embedding_cache_long_long_32_0_5_4_disable) {embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME,
32, 0, 5, 4, false); } TEST(embedding_cache, embedding_cache_long_long_0_16_5_4_enable)
{embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME, 0, 16, 5, 4, true); } TEST(embedding_cache,
embedding_cache_long_long_0_16_5_4_disable) {embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME,
0, 16, 5, 4, false); } TEST(embedding_cache, embedding_cache_long_long_16_16_5_4_enable)
{embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME, 16, 16, 5, 4, true); }
TEST(embedding_cache, embedding_cache_long_long_16_16_5_4_disable) {embedding_cache_test<long
long>(MODEL_PATH, MODEL_NAME, 16, 16, 5, 4, false); } TEST(embedding_cache,
embedding_cache_long_long_32_16_5_4_enable) {embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME,
32, 16, 5, 4, true); } TEST(embedding_cache, embedding_cache_long_long_32_16_5_4_disable)
{embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME, 32, 16, 5, 4, false); }
TEST(embedding_cache, embedding_cache_long_long_0_30_5_4_enable) {embedding_cache_test<long
long>(MODEL_PATH, MODEL_NAME, 0, 30, 5, 4, true); } TEST(embedding_cache,
embedding_cache_long_long_0_30_5_4_disable) {embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME,
0, 30, 5, 4, false); } TEST(embedding_cache, embedding_cache_long_long_16_30_5_4_enable)
{embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME, 16, 30, 5, 4, true); }
TEST(embedding_cache, embedding_cache_long_long_16_30_5_4_disable) {embedding_cache_test<long
long>(MODEL_PATH, MODEL_NAME, 16, 30, 5, 4, false); } TEST(embedding_cache,
embedding_cache_long_long_32_30_5_4_enable) {embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME,
32, 30, 5, 4, true); } TEST(embedding_cache, embedding_cache_long_long_32_30_5_4_disable)
{embedding_cache_test<long long>(MODEL_PATH, MODEL_NAME, 32, 30, 5, 4, false); }
TEST(embedding_cache, embedding_cache_long_long_32_random_5_4_enable) {embedding_cache_test<long
long>(MODEL_PATH, MODEL_NAME, 32, -1, 5, 4, true); } TEST(embedding_cache,
embedding_cache_long_long_32_random_5_4_disable) {embedding_cache_test<long long>(MODEL_PATH,
MODEL_NAME, 32, -1, 5, 4, false); }*/
