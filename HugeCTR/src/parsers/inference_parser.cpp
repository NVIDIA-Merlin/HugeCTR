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
#include <inference/embedding_feature_combiner.hpp>
#include <inference/preallocated_buffer2.hpp>
#include <parser.hpp>

namespace HugeCTR {
Parser::Parser(const nlohmann::json& config)
    : config_(config),
      batch_size_(1),
      batch_size_eval_(1),
      repeat_dataset_(false),
      i64_input_key_(false),
      use_mixed_precision_(false),
      scaler_(1.0f),
      use_algorithm_search_(true),
      use_cuda_graph_(true) {}

template <typename TypeEmbeddingComp>
void Parser::create_pipeline_inference(const InferenceParser& inference_parser, Tensor2<float>& dense_input,
                                      std::vector<std::shared_ptr<Tensor2<int>>>& rows,
                                      std::vector<std::shared_ptr<Tensor2<float>>>& embeddingvecs,
                                      std::vector<size_t>& embedding_table_slot_size,
                                      std::vector<std::shared_ptr<Layer>>* embeddings,
                                      Network** network,
                                      const std::shared_ptr<ResourceManager> resource_manager) {
  //std::vector<TensorEntry> tensor_entries;

  auto j_layers_array = get_json(config_, "layers");

  auto input_buffer = GeneralBuffer2<CudaAllocator>::create();

  {
    const nlohmann::json& j_data = j_layers_array[0];
    auto j_dense = get_json(j_data, "dense");
    auto top_strs_dense = get_value_from_json<std::string>(j_dense, "top");
    auto dense_dim = get_value_from_json<size_t>(j_dense, "dense_dim");

    input_buffer->reserve({inference_parser.max_batchsize, dense_dim}, &dense_input);
    tensor_entries.push_back({top_strs_dense, TensorUse::General, dense_input.shrink()});
  }

  create_embedding<unsigned int, TypeEmbeddingComp>()(inference_parser, j_layers_array, rows, embeddingvecs, embedding_table_slot_size, &tensor_entries,
                                                    embeddings, resource_manager->get_local_gpu(0), input_buffer);
  input_buffer->allocate();

  //create network
  *network = Network::create_network(
      j_layers_array, "", tensor_entries, 1, resource_manager->get_local_cpu(),
      resource_manager->get_local_gpu(0), inference_parser.use_mixed_precision,
      false, inference_parser.scaler, false, inference_parser.use_cuda_graph, true);
}

void Parser::create_pipeline(const InferenceParser& inference_parser, Tensor2<float>& dense_input,
                             std::vector<std::shared_ptr<Tensor2<int>>>& rows,
                             std::vector<std::shared_ptr<Tensor2<float>>>& embeddingvecs,
                             std::vector<size_t>& embedding_table_slot_size,
                             std::vector<std::shared_ptr<Layer>>* embeddings, Network** network,
                             const std::shared_ptr<ResourceManager> resource_manager) {
  if (inference_parser.use_mixed_precision) {
    create_pipeline_inference<__half>(inference_parser, dense_input, rows, embeddingvecs, embedding_table_slot_size, embeddings, network,
                                     resource_manager);
  } else {
    create_pipeline_inference<float>(inference_parser, dense_input, rows, embeddingvecs, embedding_table_slot_size, embeddings, network,
                                    resource_manager);
  }
}


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

  if (has_key_(j, "mixed_precision")) {
      use_mixed_precision = true;
      int i_scaler = get_value_from_json<int>(j, "mixed_precision");
      if (i_scaler != 128 && i_scaler != 256 && i_scaler != 512 && i_scaler != 1024) {
        CK_THROW_(Error_t::WrongInput,
                  "Scaler of mixed_precision training should be either 128/256/512/1024");
      }
      scaler = i_scaler;
      std::stringstream ss;
      ss << "Mixed Precision training with scaler: " << i_scaler << " is enabled." << std::endl;
      MESSAGE_(ss.str());

  } else {
      use_mixed_precision = false;
      scaler = 1.f;
  }
    
  use_algorithm_search = get_value_from_json_soft<bool>(j, "algorithm_search", true);
  use_cuda_graph = get_value_from_json_soft<bool>(j, "cuda_graph", true);

  auto j_layers_array = get_json(config, "layers");
  const nlohmann::json& j_data = j_layers_array[0];
  auto j_sparse_data = get_json(j_data, "sparse");
  num_embedding_tables = static_cast<size_t>(j_sparse_data.size());
  slot_num = 0;
  {
    for (int i = 0; i < (int)num_embedding_tables; i++) {
      const nlohmann::json& j = j_sparse_data[i];
      auto max_feature_num_per_sample = get_value_from_json<size_t>(j, "max_feature_num_per_sample");
      auto current_slot_num = get_value_from_json<size_t>(j, "slot_num");
      max_feature_num_for_tables.push_back(max_feature_num_per_sample);
      slot_num_for_tables.push_back(current_slot_num);
      slot_num += current_slot_num;
    }
  }
  {
    for (int i = 1; i < (int)j_layers_array.size(); i++) {
      // if not embedding then break
      const nlohmann::json& j = j_layers_array[i];
      auto embedding_name = get_value_from_json<std::string>(j, "type");
      if (embedding_name.compare("DistributedSlotSparseEmbeddingHash") != 0 &&
          embedding_name.compare("LocalizedSlotSparseEmbeddingHash") != 0 &&
          embedding_name.compare("LocalizedSlotSparseEmbeddingOneHot") != 0) {
        break;
      }
      auto j_embed_params =  get_json(j, "sparse_embedding_hparam");
      auto embedding_vec_size = get_value_from_json<int>(j_embed_params, "embedding_vec_size");
      embed_vec_size_for_tables.push_back(embedding_vec_size);
    }  // for ()
  }    // get embedding params

  max_embedding_vector_size_per_sample = 0;
  for (int i = 0; i < (int)num_embedding_tables; i++) {
    max_embedding_vector_size_per_sample += (max_feature_num_for_tables[i] * embed_vec_size_for_tables[i]);
  }
}

template <typename TypeKey, typename TypeFP>
void create_embedding<TypeKey, TypeFP>::operator() (
    const InferenceParser& inference_parser, const nlohmann::json& j_layers_array,
    std::vector<std::shared_ptr<Tensor2<int>>>& rows, std::vector<std::shared_ptr<Tensor2<float>>>& embeddingvecs,
    std::vector<size_t>& embedding_table_slot_size,
    std::vector<TensorEntry>* tensor_entries,
    std::vector<std::shared_ptr<Layer>>* embeddings,
    const std::shared_ptr<GPUResource> gpu_resource,
    std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff) {
  MESSAGE_("start create embedding for inference");
  auto j_data = j_layers_array[0];
  if (!has_key_(j_data, "sparse")) {
    MESSAGE_("no sparse data input");
    return;
  }
  auto j_sparse_input = get_json(j_data, "sparse");
  std::unordered_map<std::string, std::pair<int, int>> slot_nums_map;
  for (unsigned int i = 0; i < j_sparse_input.size(); ++i) {
    auto top = get_value_from_json<std::string>(j_sparse_input[i], "top");
    auto slot_num = get_value_from_json<int>(j_sparse_input[i], "slot_num");
    auto max_feature_num_per_sample = get_value_from_json<int>(j_sparse_input[i], "max_feature_num_per_sample");
    MESSAGE_("sparse_input name " + top);
    slot_nums_map[top] = std::make_pair(slot_num,max_feature_num_per_sample);
  }
  if(j_layers_array.size() < 1){
    CK_THROW_(Error_t::WrongInput, "layer not defined in config");
  }
  for (unsigned int i = 1; i < j_layers_array.size(); i++) {
    const nlohmann::json& j = j_layers_array[i];
    auto bottom_array = get_json(j, "bottom");
    if(bottom_array.is_array()){
      continue;
    }
    std::string bottom = bottom_array.get<std::string>();;
    auto slot_nums_map_iter = slot_nums_map.find(bottom);
    if (slot_nums_map_iter == slot_nums_map.end()) {
      continue;
    }
    const std::string layer_top = get_value_from_json<std::string>(j, "top");
    int slot_num = slot_nums_map_iter->second.first;
    int max_feature_num_per_sample = slot_nums_map_iter->second.second;
    auto j_hparam = get_json(j, "sparse_embedding_hparam");
    auto combiner = get_value_from_json<int>(j_hparam, "combiner");
    EmbeddingFeatureCombiner_t feature_combiner_type;
    if (combiner == 0) {
      feature_combiner_type = EmbeddingFeatureCombiner_t::Sum;
    } else if(combiner == 1){
      feature_combiner_type = EmbeddingFeatureCombiner_t::Mean;
    } else{
      CK_THROW_(Error_t::WrongInput, "combiner need to be 0 or 1");
    }
    size_t embedding_vec_size = get_value_from_json<size_t>(j_hparam, "embedding_vec_size");

    size_t prefix_slot_num = embedding_table_slot_size.back();
    embedding_table_slot_size.push_back(prefix_slot_num + slot_num);

    std::vector<size_t> row_dims = { static_cast<size_t>(inference_parser.max_batchsize * slot_num + 1) };
    std::vector<size_t> embeddingvecs_dims = { static_cast<size_t>(inference_parser.max_batchsize * max_feature_num_per_sample),
                                               static_cast<size_t>(embedding_vec_size) };
    std::shared_ptr<Tensor2<int>> row_tensor = std::make_shared<Tensor2<int>>();
    std::shared_ptr<Tensor2<float>> embeddingvecs_tensor = std::make_shared<Tensor2<float>>();
    blobs_buff->reserve(row_dims, row_tensor.get());
    blobs_buff->reserve(embeddingvecs_dims, embeddingvecs_tensor.get());
    rows.push_back(row_tensor);
    embeddingvecs.push_back(embeddingvecs_tensor);
    Tensor2<TypeFP> embedding_output;
    embeddings->push_back(std::make_shared<EmbeddingFeatureCombiner<TypeFP>>(
        embeddingvecs[0], rows[0], embedding_output, inference_parser.max_batchsize,
        slot_num, feature_combiner_type, blobs_buff, gpu_resource));
    tensor_entries->push_back({layer_top, TensorUse::General, embedding_output.shrink()});
  }

  CudaDeviceContext context(gpu_resource->get_device_id());
  MESSAGE_("create embedding for inference success");
}

template struct create_embedding<long long, float>;
template struct create_embedding<long long, __half>;
template struct create_embedding<unsigned int, float>;
template struct create_embedding<unsigned int, __half>;
}  // namespace HugeCTR
