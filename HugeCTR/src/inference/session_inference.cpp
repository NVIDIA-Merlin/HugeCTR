/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "HugeCTR/include/inference/session_inference.hpp"

#include <iostream>
#include <vector>
namespace HugeCTR {

InferenceParser::InferenceParser(const nlohmann::json& config) {
  auto j = get_json(config, "inference");
  if (has_key_(j, "max_batchsize")) {
    max_batchsize = get_value_from_json<unsigned long long>(j, "max_batchsize");
  } else {
    max_batchsize = 1024;
  }

  bool has_dense_model_file = has_key_(j, "dense_model_file");
  bool has_sparse_model_files = has_key_(j, "sparse_model_file");
  if (!(has_dense_model_file && has_sparse_model_files)) {
    CK_THROW_(Error_t::WrongInput, "dense_model_file and sparse_model_file must be specified");
  }
  dense_model_file = get_value_from_json<std::string>(j, "dense_model_file");
  auto j_sparse_model_files = get_json(j, "sparse_model_file");
  if (j_sparse_model_files.is_array()) {
    for (auto j_embedding_tmp : j_sparse_model_files) {
      sparse_model_files.push_back(j_embedding_tmp.get<std::string>());
    }
  } else {
    sparse_model_files.push_back(get_value_from_json<std::string>(j, "sparse_model_file"));
  }
}

InferenceSession::InferenceSession(const std::string& config_file, int device_id)
    : resource_manager(ResourceManager::create({{device_id}}, 0)) {
  try {
    std::ifstream file(config_file);
    if (!file.is_open()) {
      CK_THROW_(Error_t::FileCannotOpen, "file.is_open() failed: " + config_file);
    }
    file >> config_;
    file.close();
    InferenceParser inference_parser(config_);

    size_t max_batch_size = inference_parser.max_batchsize;
    // size_t batch_size_eval = 0;
    bool use_cuda_graph = inference_parser.use_cuda_graph;

    std::vector<TensorEntry> tensor_entries;
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> input_buffer =
        GeneralBuffer2<CudaAllocator>::create();
    auto j_layers_array = get_json(config_, "layers");
    {
      const nlohmann::json& j_data = j_layers_array[0];
      auto j_dense = get_json(j_data, "dense");
      auto top_strs_dense = get_value_from_json<std::string>(j_dense, "top");
      auto dense_dim = get_value_from_json<size_t>(j_dense, "dense_dim");

      Tensor2<float> dense_input_tensor;
      input_buffer->reserve({max_batch_size, dense_dim}, &dense_input_tensor);
      tensor_entries.push_back({top_strs_dense, TensorUse::General, dense_input_tensor.shrink()});
    }

    // Add fake embedding layer input; will be deleted after merge Embedding Compute layer
    size_t slot_num = 10;
    for (unsigned int i = 1; i < j_layers_array.size(); i++) {
      const nlohmann::json& j = j_layers_array[i];
      const auto layer_type_name = get_value_from_json<std::string>(j, "type");
      if (layer_type_name.find("Embedding") != std::string::npos) {
        const std::string layer_name = get_value_from_json<std::string>(j, "name");
        const std::string layer_top = get_value_from_json<std::string>(j, "top");
        const auto& j_hparam = get_json(j, "sparse_embedding_hparam");
        size_t embedding_vec_size = get_value_from_json<size_t>(j_hparam, "embedding_vec_size");
        Tensor2<float> sparse_embedding;
        input_buffer->reserve({max_batch_size, slot_num * embedding_vec_size}, &sparse_embedding);
        tensor_entries.push_back({layer_top, TensorUse::General, sparse_embedding.shrink()});
      }
    }

    // create network
    network_ = std::move(Network::create_network(
        j_layers_array, "", tensor_entries, 1, resource_manager->get_local_cpu(),
        resource_manager->get_local_gpu(0), false, 0.f, false, use_cuda_graph, true));
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
  return;
}  // namespace HugeCTR

InferenceSession::~InferenceSession() {}

void InferenceSession::predict(float* dense, int* row, float* embeddingvector, float* output,
                               int numofsample) {}
}  // namespace HugeCTR
