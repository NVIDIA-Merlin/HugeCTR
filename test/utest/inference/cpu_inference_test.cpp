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

#include <cuda_profiler_api.h>
#include <gtest/gtest.h>

#include <cpu/embedding_feature_combiner_cpu.hpp>
#include <cpu/inference_session_cpu.hpp>
#include <data_generator.hpp>
#include <fstream>
#include <general_buffer2.hpp>
#include <hps/hier_parameter_server.hpp>
#include <hps/inference_utils.hpp>
#include <utest/test_utils.hpp>
#include <utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

const int RANGE[] = {0,       1460,    2018,    337396,  549106,  549411,  549431,
                     561567,  562200,  562203,  613501,  618803,  951403,  954582,
                     954609,  966800,  1268011, 1268021, 1272862, 1274948, 1274952,
                     1599225, 1599242, 1599257, 1678991, 1679087, 1737709};

std::vector<std::string>& split(const std::string& s, char delim, std::vector<std::string>& elems) {
  std::istringstream is(s);
  std::string item;
  while (std::getline(is, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

struct InferenceInfo {
  int dense_dim;
  std::vector<int> slot_num;
  std::vector<int> max_feature_num_per_sample;
  std::vector<int> embedding_vec_size;
  std::vector<HugeCTR::EmbeddingFeatureCombiner_t> combiner_type;
  InferenceInfo(const nlohmann::json& config);
};

InferenceInfo::InferenceInfo(const nlohmann::json& config) {
  auto j_layers_array = get_json(config, "layers");
  const nlohmann::json& j_data = j_layers_array[0];
  auto j_dense = get_json(j_data, "dense");
  dense_dim = get_value_from_json<int>(j_dense, "dense_dim");
  auto j_sparse_inputs = get_json(j_data, "sparse");

  for (size_t i = 0; i < j_sparse_inputs.size(); i++) {
    const nlohmann::json& j_sparse = j_sparse_inputs[0];
    slot_num.push_back(get_value_from_json<int>(j_sparse, "slot_num"));

    size_t max_feature_num_per_sample_ =
        static_cast<size_t>(get_max_feature_num_per_sample_from_nnz_per_slot(j_sparse));

    max_feature_num_per_sample.push_back(max_feature_num_per_sample_);
  }

  // get embedding params: embedding_vec_size, combiner_type
  for (size_t i = 1; i < j_layers_array.size(); i++) {
    // if not embedding then break
    const nlohmann::json& j = j_layers_array[i];
    auto embedding_name = get_value_from_json<std::string>(j, "type");
    if (embedding_name.compare("DistributedSlotSparseEmbeddingHash") != 0 &&
        embedding_name.compare("LocalizedSlotSparseEmbeddingHash") != 0 &&
        embedding_name.compare("LocalizedSlotSparseEmbeddingOneHot") != 0) {
      break;
    }
    auto j_embed_params = get_json(j, "sparse_embedding_hparam");
    auto vec_size = get_value_from_json<int>(j_embed_params, "embedding_vec_size");
    auto combiner = get_value_from_json<std::string>(j_embed_params, "combiner");
    embedding_vec_size.push_back(vec_size);
    if (combiner == "mean") {
      combiner_type.push_back(HugeCTR::EmbeddingFeatureCombiner_t::Mean);
    } else {
      combiner_type.push_back(HugeCTR::EmbeddingFeatureCombiner_t::Sum);
    }
  }
}

template <typename TypeHashKey>
void session_inference_criteo_test(const std::string& config_file, const std::string& model,
                                   const std::string& criteo_data_path, int batchsize) {
  InferenceInfo inference_info(read_json_file(config_file));
  int batch_size = batchsize;
  int dense_dim = inference_info.dense_dim;
  int slot_num = inference_info.slot_num[0];
  int max_feature_num_per_sample = inference_info.max_feature_num_per_sample[0];
  int num_samples = 0;
  std::vector<int> labels;
  std::vector<float> dense_features;
  std::vector<TypeHashKey> keys;
  std::vector<int> row_ptrs;
  HostAllocator host_allocator;
  HugeCTR::Timer timer_inference;

  // open criteo data file
  std::ifstream criteo_data_file(criteo_data_path, std::ifstream::binary);
  if (!criteo_data_file.is_open()) {
    HCTR_LOG_S(ERROR, WORLD) << "Cannot open " << criteo_data_path << std::endl;
  }

  // 4 lines: labels, dense_features, keys, row_ptrs
  for (int i = 0; i < 4; i++) {
    std::string line;
    std::getline(criteo_data_file, line);
    std::vector<std::string> vec_string;
    split(line, ' ', vec_string);
    switch (i) {
      case 0: {
        num_samples = static_cast<int>(vec_string.size());
        for (int j = 0; j < num_samples; j++) {
          int label = std::stoi(vec_string[j]);
          labels.push_back(label);
        }
        break;
      }
      case 1: {
        int dense_features_dim = static_cast<int>(vec_string.size());
        if (dense_features_dim != num_samples * dense_dim) {
          HCTR_LOG_S(ERROR, WORLD)
              << "dense_features_dim does not equal to num_samples*dense_dim" << std::endl;
        }
        for (int j = 0; j < dense_features_dim; j++) {
          float dense_feature = std::stod(vec_string[j]);
          dense_features.push_back(dense_feature);
        }
        break;
      }
      case 2: {
        int keys_dim = static_cast<int>(vec_string.size());
        if (keys_dim != num_samples * slot_num) {
          HCTR_LOG_S(ERROR, WORLD)
              << "keys_dim does not equal to num_samples*slot_num" << std::endl;
        }
        for (int j = 0; j < keys_dim; j++) {
          TypeHashKey key = static_cast<TypeHashKey>(std::stoll(vec_string[j]));
          keys.push_back(key);
        }
        break;
      }
      case 3: {
        int row_ptrs_dim = static_cast<int>(vec_string.size());
        if (row_ptrs_dim != num_samples * slot_num + 1) {
          HCTR_LOG_S(ERROR, WORLD)
              << "row_ptrs_dim does not equal to num_samples*slot_num + 1" << std::endl;
        }
        for (int j = 0; j < row_ptrs_dim; j++) {
          int row_ptr = std::stoi(vec_string[j]);
          row_ptrs.push_back(row_ptr);
        }
        break;
      }
      default: {
        assert(!"Error: Should never get here!");
      }
    }
  }

  if (batch_size == 0) {
    HCTR_OWN_THROW(Error_t::WrongInput, "batch size should not be zero!");
  }
  num_samples = num_samples < batch_size ? num_samples : batch_size;

  // h_row_ptrs
  std::vector<size_t> row_ptrs_dims = {static_cast<size_t>(batch_size * slot_num + 1)};  // 1D
  size_t row_ptrs_size = 1;
  for (auto dim : row_ptrs_dims) {
    row_ptrs_size *= dim;
  }
  size_t row_ptrs_size_samples = num_samples * slot_num + 1;
  size_t row_ptrs_size_in_bytes = row_ptrs_size * sizeof(int);
  size_t row_ptrs_size_in_bytes_samples = row_ptrs_size_samples * sizeof(int);
  int* h_row_ptrs = reinterpret_cast<int*>(host_allocator.allocate(row_ptrs_size_in_bytes));
  for (size_t i = 0; i < row_ptrs_size; i++) {
    h_row_ptrs[i] = 0;
  }

  // h_dense_features
  size_t dense_size = batch_size * dense_dim;
  size_t dense_size_samples = num_samples * dense_dim;
  size_t dense_size_in_bytes = dense_size * sizeof(float);
  size_t dense_size_in_bytes_samples = dense_size_samples * sizeof(float);
  float* h_dense_features = reinterpret_cast<float*>(host_allocator.allocate(dense_size_in_bytes));

  // h_embeddingcolumns
  size_t embeddingcolumns_size = batch_size * max_feature_num_per_sample;
  size_t embeddingcolumns_size_samples = num_samples * max_feature_num_per_sample;
  size_t embeddingcolumns_size_in_bytes = embeddingcolumns_size * sizeof(TypeHashKey);
  size_t embeddingcolumns_size_in_bytes_samples =
      embeddingcolumns_size_samples * sizeof(TypeHashKey);
  void* h_embeddingcolumns = host_allocator.allocate(embeddingcolumns_size_in_bytes);
  // TypeHashKey* h_keys = reinterpret_cast<TypeHashKey*>(h_embeddingcolumns);

  // h_output
  std::unique_ptr<float[]> h_out(new float[batch_size]);

  // memory copy
  memcpy(h_embeddingcolumns, keys.data(), embeddingcolumns_size_in_bytes_samples);
  memcpy(h_row_ptrs, row_ptrs.data(), row_ptrs_size_in_bytes_samples);
  memcpy(h_dense_features, dense_features.data(), dense_size_in_bytes_samples);

  // inference session
  std::string dense_model{"/hugectr/test/utest/_dense_10000.model"};
  std::vector<std::string> sparse_models{"/hugectr/test/utest/0_sparse_10000.model"};
  InferenceParams infer_param(model, batchsize, 0.5, dense_model, sparse_models, 0, true, 0.8,
                              false);
  std::vector<InferenceParams> inference_params{infer_param};
  std::vector<std::string> model_config_path{config_file};
  parameter_server_config ps_config{model_config_path, inference_params};
  std::shared_ptr<HierParameterServerBase> parameter_server =
      HierParameterServerBase::create(ps_config);
  InferenceSessionCPU<TypeHashKey> sess(model_config_path[0], inference_params[0],
                                        parameter_server);
  timer_inference.start();
  sess.predict(h_dense_features, h_embeddingcolumns, h_row_ptrs, h_out.get(), num_samples);
  timer_inference.stop();

  {
    auto log = HCTR_LOG_S(INFO, WORLD);
    log << "==========================labels===================" << std::endl;
    for (int i = 0; i < num_samples; i++) {
      log << labels[i] << " ";
    }
    log << std::endl;
  }
  {
    auto log = HCTR_LOG_S(INFO, WORLD);
    log << "==========================prediction result===================" << std::endl;
    for (int i = 0; i < num_samples; i++) {
      log << h_out[i] << " ";
    }
    log << std::endl;
  }
  HCTR_LOG_S(INFO, ROOT) << "Batch size: " << batch_size << ", Number samples: " << num_samples
                         << ", Time: " << timer_inference.elapsedSeconds() << "s" << std::endl;
  host_allocator.deallocate(h_embeddingcolumns);
  host_allocator.deallocate(h_dense_features);
  host_allocator.deallocate(h_row_ptrs);
}

template <typename TypeHashKey>
void session_inference_generated_test(const std::string& config_file, const std::string& model,
                                      int num_samples, int batchsize) {
  InferenceInfo inference_info(read_json_file(config_file));
  int batch_size = batchsize;
  int dense_dim = inference_info.dense_dim;
  int slot_num = inference_info.slot_num[0];
  int max_feature_num_per_sample = inference_info.max_feature_num_per_sample[0];
  int max_nnz = max_feature_num_per_sample / slot_num;
  num_samples = num_samples < batch_size ? num_samples : batch_size;
  HostAllocator host_allocator;
  HugeCTR::Timer timer_inference;

  // h_row_ptrs
  std::vector<size_t> row_ptrs_dims = {static_cast<size_t>(batch_size * slot_num + 1)};  // 1D
  size_t row_ptrs_size = 1;
  for (auto dim : row_ptrs_dims) {
    row_ptrs_size *= dim;
  }
  std::unique_ptr<int[]> h_row_ptrs(new int[row_ptrs_size]);
  std::shared_ptr<IDataSimulator<int>> ldata_sim;
  ldata_sim.reset(new IntUniformDataSimulator<int>(1, max_nnz));
  h_row_ptrs[0] = 0;
  for (size_t i = 1; i < row_ptrs_size; i++) {
    h_row_ptrs[i] = (h_row_ptrs[i - 1] + ldata_sim->get_num());
  }

  // h_dense_features
  const size_t dense_size = batch_size * dense_dim;
  std::unique_ptr<float[]> h_dense(new float[dense_size]);
  FloatUniformDataSimulator<float> fdata_sim(0, 1);
  for (size_t i = 0; i < dense_size; i++) {
    h_dense[i] = fdata_sim.get_num();
  }

  // h_embeddingcolumns
  size_t embeddingcolumns_size = batch_size * max_feature_num_per_sample;
  size_t embeddingcolumns_size_in_bytes = embeddingcolumns_size * sizeof(TypeHashKey);
  void* h_embeddingcolumns = host_allocator.allocate(embeddingcolumns_size_in_bytes);
  TypeHashKey* h_keys = reinterpret_cast<TypeHashKey*>(h_embeddingcolumns);
  for (int i = 0; i < num_samples; i++) {
    for (int j = 0; j < slot_num; j++) {
      ldata_sim.reset(new IntUniformDataSimulator<int>(RANGE[j], RANGE[j + 1] - 1));
      h_keys[i * slot_num + j] = static_cast<TypeHashKey>(ldata_sim->get_num());
    }
  }

  std::unique_ptr<float[]> h_out(new float[batch_size]);

  // inference session
  std::string dense_model{"/hugectr/test/utest/_dense_10000.model"};
  std::vector<std::string> sparse_models{"/hugectr/test/utest/0_sparse_10000.model"};
  InferenceParams infer_param(model, batchsize, 0.5, dense_model, sparse_models, 0, true, 0.8,
                              false);
  std::vector<InferenceParams> inference_params{infer_param};
  std::vector<std::string> model_config_path{config_file};
  parameter_server_config ps_config{model_config_path, inference_params};
  std::shared_ptr<HierParameterServerBase> parameter_server =
      HierParameterServerBase::create(ps_config);
  InferenceSessionCPU<TypeHashKey> sess(model_config_path[0], inference_params[0],
                                        parameter_server);
  timer_inference.start();
  sess.predict(h_dense.get(), h_embeddingcolumns, h_row_ptrs.get(), h_out.get(), num_samples);
  timer_inference.stop();

  {
    auto log = HCTR_LOG_S(INFO, WORLD);
    log << "==========================prediction result===================" << std::endl;
    for (int i = 0; i < num_samples; i++) {
      log << h_out[i] << " ";
    }
    log << std::endl;
  }
  HCTR_LOG_S(INFO, ROOT) << "Batch size: " << batch_size << ", Number samples: " << num_samples
                         << ", Time: " << timer_inference.elapsedSeconds() << "s" << std::endl;
  host_allocator.deallocate(h_embeddingcolumns);
}

}  // namespace

TEST(session_inference_cpu, criteo_dcn) {
  session_inference_criteo_test<unsigned int>("/workdir/test/utest/simple_inference_config.json",
                                              "DCN", "/hugectr/test/utest/dcn_csr.txt", 32);
}
TEST(session_inference_cpu, generated_dcn_32) {
  session_inference_generated_test<unsigned int>("/workdir/test/utest/simple_inference_config.json",
                                                 "DCN", 32, 32);
}