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
InferenceParams::InferenceParams(const std::string& model_name, const size_t max_batchsize, const float hit_rate_threshold,
                  const std::string& dense_model_file, const std::vector<std::string>& sparse_model_files,
                  const int device_id, const bool use_gpu_embedding_cache, const float cache_size_percentage,
                  const bool i64_input_key, const bool use_mixed_precision, const float scaler,
                  const bool use_algorithm_search, const bool use_cuda_graph)
  : model_name(model_name), max_batchsize(max_batchsize), hit_rate_threshold(hit_rate_threshold),
    dense_model_file(dense_model_file), sparse_model_files(sparse_model_files), device_id(device_id),
    use_gpu_embedding_cache(use_gpu_embedding_cache), cache_size_percentage(cache_size_percentage),
    i64_input_key(i64_input_key), use_mixed_precision(use_mixed_precision), scaler(scaler),
    use_algorithm_search(use_algorithm_search), use_cuda_graph(use_cuda_graph) {}

template <typename TypeEmbeddingComp>
void InferenceParser::create_pipeline_inference(const InferenceParams& inference_params,
                                      TensorBag2& dense_input_bag,
                                      std::vector<std::shared_ptr<Tensor2<int>>>& rows,
                                      std::vector<std::shared_ptr<Tensor2<float>>>& embeddingvecs,
                                      std::vector<size_t>& embedding_table_slot_size,
                                      std::vector<std::shared_ptr<Layer>>* embeddings,
                                      Network** network,
                                      const std::shared_ptr<ResourceManager> resource_manager) {
  std::vector<TensorEntry> train_tensor_entries;
  std::vector<TensorEntry> inference_tensor_entries;
  auto j_layers_array = get_json(config_, "layers");
  check_graph(tensor_active_, j_layers_array);
  
  auto input_buffer = GeneralBuffer2<CudaAllocator>::create();

  {
    const nlohmann::json& j_data = j_layers_array[0];
    auto j_dense = get_json(j_data, "dense");
    auto top_strs_dense = get_value_from_json<std::string>(j_dense, "top");
    auto dense_dim = get_value_from_json<size_t>(j_dense, "dense_dim");
    auto j_label = get_json(j_data, "label");
    auto top_strs_label = get_value_from_json<std::string>(j_label, "top");
    auto label_dim = get_value_from_json<size_t>(j_label, "label_dim");
    Tensor2<TypeEmbeddingComp> dense_input;
    Tensor2<float> label_input;
    input_buffer->reserve({inference_params.max_batchsize, dense_dim}, &dense_input);
    input_buffer->reserve({inference_params.max_batchsize, label_dim}, &label_input);
    inference_tensor_entries.push_back({top_strs_dense, dense_input.shrink()});
    inference_tensor_entries.push_back({top_strs_label, label_input.shrink()});
    dense_input_bag = dense_input.shrink();
  }

  create_embedding<unsigned int, TypeEmbeddingComp>()(
      inference_params,
      j_layers_array,
      rows,
      embeddingvecs,
      embedding_table_slot_size,
      &inference_tensor_entries,
      embeddings,
      resource_manager->get_local_gpu(0),
      input_buffer);

  input_buffer->allocate();
  // TODO: perhaps it is better to make a wrapper of this function for the inference
  // rather than passing unused parameters here.
  std::shared_ptr<ExchangeWgrad> exchange_wgrad_dummy;
  *network = Network::create_network(
      j_layers_array,
      "",
      train_tensor_entries,
      inference_tensor_entries,
      1,
      exchange_wgrad_dummy,
      resource_manager->get_local_cpu(),
      resource_manager->get_local_gpu(0),
      inference_params.use_mixed_precision,
      false,
      inference_params.scaler,
      false,
      inference_params.use_cuda_graph,
      true,
      false);
}

void InferenceParser::create_pipeline(const InferenceParams& inference_params,
                             TensorBag2& dense_input_bag,
                             std::vector<std::shared_ptr<Tensor2<int>>>& rows,
                             std::vector<std::shared_ptr<Tensor2<float>>>& embeddingvecs,
                             std::vector<size_t>& embedding_table_slot_size,
                             std::vector<std::shared_ptr<Layer>>* embeddings, Network** network,
                             const std::shared_ptr<ResourceManager> resource_manager) {
  if (inference_params.use_mixed_precision) {
    create_pipeline_inference<__half>(inference_params, dense_input_bag, rows, embeddingvecs, embedding_table_slot_size, embeddings, network,
                                     resource_manager);
  } else {
    create_pipeline_inference<float>(inference_params, dense_input_bag, rows, embeddingvecs, embedding_table_slot_size, embeddings, network,
                                    resource_manager);
  }
}


InferenceParser::InferenceParser(const nlohmann::json& config) : config_(config) {
  auto j_layers_array = get_json(config, "layers");
  const nlohmann::json& j_data = j_layers_array[0];
  auto j_label_data = get_json(j_data, "label");
  auto j_dense_data = get_json(j_data, "dense");
  auto j_sparse_data = get_json(j_data, "sparse");
  label_name = get_value_from_json<std::string>(j_label_data, "top");
  dense_name = get_value_from_json<std::string>(j_dense_data, "top");
  label_dim = get_value_from_json<size_t>(j_label_data, "label_dim");
  dense_dim = get_value_from_json<size_t>(j_dense_data, "dense_dim");
  num_embedding_tables = static_cast<size_t>(j_sparse_data.size());
  slot_num = 0;
  {
    for (int i = 0; i < (int)num_embedding_tables; i++) {
      const nlohmann::json& j = j_sparse_data[i];
      auto max_feature_num_per_sample_per_table = get_value_from_json<size_t>(j, "max_feature_num_per_sample");
      auto current_slot_num = get_value_from_json<size_t>(j, "slot_num");
      auto sparse_name = get_value_from_json<std::string>(j, "top");
      max_feature_num_for_tables.push_back(max_feature_num_per_sample_per_table);
      slot_num_for_tables.push_back(current_slot_num);
      sparse_names.push_back(sparse_name);
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
  max_feature_num_per_sample = 0;
  for (int i = 0; i < (int)num_embedding_tables; i++) {
    max_embedding_vector_size_per_sample += (max_feature_num_for_tables[i] * embed_vec_size_for_tables[i]);
    max_feature_num_per_sample += max_feature_num_for_tables[i];
  }
}

template <typename TypeKey, typename TypeFP>
void create_embedding<TypeKey, TypeFP>::operator() (
    const InferenceParams& inference_params, const nlohmann::json& j_layers_array,
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

    std::vector<size_t> row_dims = { static_cast<size_t>(inference_params.max_batchsize * slot_num + 1) };
    std::vector<size_t> embeddingvecs_dims = { static_cast<size_t>(inference_params.max_batchsize * max_feature_num_per_sample),
                                               static_cast<size_t>(embedding_vec_size) };
    std::shared_ptr<Tensor2<int>> row_tensor = std::make_shared<Tensor2<int>>();
    std::shared_ptr<Tensor2<float>> embeddingvecs_tensor = std::make_shared<Tensor2<float>>();
    blobs_buff->reserve(row_dims, row_tensor.get());
    blobs_buff->reserve(embeddingvecs_dims, embeddingvecs_tensor.get());
    rows.push_back(row_tensor);
    embeddingvecs.push_back(embeddingvecs_tensor);
    Tensor2<TypeFP> embedding_output;
    embeddings->push_back(std::make_shared<EmbeddingFeatureCombiner<TypeFP>>(
        embeddingvecs[0], rows[0], embedding_output, inference_params.max_batchsize,
        slot_num, feature_combiner_type, blobs_buff, gpu_resource));
    tensor_entries->push_back({layer_top, embedding_output.shrink()});
  }

  CudaDeviceContext context(gpu_resource->get_device_id());
  MESSAGE_("create embedding for inference success");
}

template struct create_embedding<long long, float>;
template struct create_embedding<long long, __half>;
template struct create_embedding<unsigned int, float>;
template struct create_embedding<unsigned int, __half>;
}  // namespace HugeCTR
