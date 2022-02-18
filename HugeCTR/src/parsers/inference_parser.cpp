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
#include <inference/embedding_feature_combiner.hpp>
#include <inference/preallocated_buffer2.hpp>
#include <parser.hpp>

namespace HugeCTR {

VolatileDatabaseParams::VolatileDatabaseParams(
    const DatabaseType_t type,
    // Backend specific.
    const std::string& address, const std::string& user_name, const std::string& password,
    const DatabaseHashMapAlgorithm_t algorithm, const size_t num_partitions,
    const size_t max_get_batch_size, const size_t max_set_batch_size,
    // Overflow handling related.
    const bool refresh_time_after_fetch, const size_t overflow_margin,
    const DatabaseOverflowPolicy_t overflow_policy, const double overflow_resolution_target,
    // Caching behavior related.
    const double initial_cache_rate, const bool cache_missed_embeddings,
    // Real-time update mechanism related.
    const std::vector<std::string>& update_filters)
    : type(type),
      // Backend specific.
      address(address),
      user_name(user_name),
      password(password),
      algorithm(algorithm),
      num_partitions(num_partitions),
      max_get_batch_size(max_get_batch_size),
      max_set_batch_size(max_set_batch_size),
      // Overflow handling related.
      refresh_time_after_fetch(refresh_time_after_fetch),
      overflow_margin(overflow_margin),
      overflow_policy(overflow_policy),
      overflow_resolution_target(overflow_resolution_target),
      // Caching behavior related.
      initial_cache_rate(initial_cache_rate),
      cache_missed_embeddings(cache_missed_embeddings),
      // Real-time update mechanism related.
      update_filters(update_filters) {}

PersistentDatabaseParams::PersistentDatabaseParams(const DatabaseType_t type,
                                                   // Backend specific.
                                                   const std::string& path,
                                                   const size_t num_threads, const bool read_only,
                                                   const size_t max_get_batch_size,
                                                   const size_t max_set_batch_size,
                                                   // Real-time update mechanism related.
                                                   const std::vector<std::string>& update_filters)
    : type(type),
      // Backend specific.
      path(path),
      num_threads(num_threads),
      read_only(read_only),
      max_get_batch_size(max_get_batch_size),
      max_set_batch_size(max_set_batch_size),
      // Real-time update mechanism related.
      update_filters(update_filters) {}

UpdateSourceParams::UpdateSourceParams(const UpdateSourceType_t type,
                                       // Backend specific.
                                       const std::string& brokers, const size_t poll_timeout_ms,
                                       const size_t max_receive_buffer_size,
                                       const size_t max_batch_size, const size_t failure_backoff_ms)
    : type(type),
      // Backend specific.
      brokers(brokers),
      poll_timeout_ms(poll_timeout_ms),
      max_receive_buffer_size(max_receive_buffer_size),
      max_batch_size(max_batch_size),
      failure_backoff_ms(failure_backoff_ms) {}

InferenceParams::InferenceParams(
    const std::string& model_name, const size_t max_batchsize, const float hit_rate_threshold,
    const std::string& dense_model_file, const std::vector<std::string>& sparse_model_files,
    const int device_id, const bool use_gpu_embedding_cache, const float cache_size_percentage,
    const bool i64_input_key, const bool use_mixed_precision, const float scaler,
    const bool use_algorithm_search, const bool use_cuda_graph,
    const int number_of_worker_buffers_in_pool, const int number_of_refresh_buffers_in_pool,
    const float cache_refresh_percentage_per_iteration, const std::vector<int>& deployed_devices,
    const std::vector<float>& default_value_for_each_table,
    // Database backend.
    const VolatileDatabaseParams& volatile_db, const PersistentDatabaseParams& persistent_db,
    const UpdateSourceParams& update_source)
    : model_name(model_name),
      max_batchsize(max_batchsize),
      hit_rate_threshold(hit_rate_threshold),
      dense_model_file(dense_model_file),
      sparse_model_files(sparse_model_files),
      device_id(device_id),
      use_gpu_embedding_cache(use_gpu_embedding_cache),
      cache_size_percentage(cache_size_percentage),
      i64_input_key(i64_input_key),
      use_mixed_precision(use_mixed_precision),
      scaler(scaler),
      use_algorithm_search(use_algorithm_search),
      use_cuda_graph(use_cuda_graph),
      number_of_worker_buffers_in_pool(number_of_worker_buffers_in_pool),
      number_of_refresh_buffers_in_pool(number_of_refresh_buffers_in_pool),
      cache_refresh_percentage_per_iteration(cache_refresh_percentage_per_iteration),
      deployed_devices(deployed_devices),
      default_value_for_each_table(default_value_for_each_table),
      // Database backend.
      volatile_db(volatile_db),
      persistent_db(persistent_db),
      update_source(update_source) {}

template <typename TypeEmbeddingComp>
void InferenceParser::create_pipeline_inference(
    const InferenceParams& inference_params, TensorBag2& dense_input_bag,
    std::vector<std::shared_ptr<Tensor2<int>>>& rows,
    std::vector<std::shared_ptr<Tensor2<float>>>& embeddingvecs,
    std::vector<size_t>& embedding_table_slot_size, std::vector<std::shared_ptr<Layer>>* embeddings,
    Network** network, const std::shared_ptr<ResourceManager> resource_manager) {
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
      inference_params, j_layers_array, rows, embeddingvecs, embedding_table_slot_size,
      &inference_tensor_entries, embeddings, resource_manager->get_local_gpu(0), input_buffer);

  CudaDeviceContext context(resource_manager->get_local_gpu(0)->get_device_id());
  input_buffer->allocate();
  // TODO: perhaps it is better to make a wrapper of this function for the inference
  // rather than passing unused parameters here.
  std::shared_ptr<ExchangeWgrad> exchange_wgrad_dummy;
  *network = Network::create_network(
      j_layers_array, "", train_tensor_entries, inference_tensor_entries, 1, exchange_wgrad_dummy,
      resource_manager->get_local_cpu(), resource_manager->get_local_gpu(0),
      inference_params.use_mixed_precision, false, inference_params.scaler, false,
      inference_params.use_cuda_graph, true, false);
}

void InferenceParser::create_pipeline(const InferenceParams& inference_params,
                                      TensorBag2& dense_input_bag,
                                      std::vector<std::shared_ptr<Tensor2<int>>>& rows,
                                      std::vector<std::shared_ptr<Tensor2<float>>>& embeddingvecs,
                                      std::vector<size_t>& embedding_table_slot_size,
                                      std::vector<std::shared_ptr<Layer>>* embeddings,
                                      Network** network,
                                      const std::shared_ptr<ResourceManager> resource_manager) {
  if (inference_params.use_mixed_precision) {
    create_pipeline_inference<__half>(inference_params, dense_input_bag, rows, embeddingvecs,
                                      embedding_table_slot_size, embeddings, network,
                                      resource_manager);
  } else {
    create_pipeline_inference<float>(inference_params, dense_input_bag, rows, embeddingvecs,
                                     embedding_table_slot_size, embeddings, network,
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
      size_t max_feature_num_per_sample_per_table =
          get_max_feature_num_per_sample_from_nnz_per_slot(j);
      auto current_slot_num = get_value_from_json<size_t>(j, "slot_num");
      int current_max_nnz = get_max_nnz_from_nnz_per_slot(j);
      auto sparse_name = get_value_from_json<std::string>(j, "top");
      max_feature_num_for_tables.push_back(max_feature_num_per_sample_per_table);
      slot_num_for_tables.push_back(current_slot_num);
      max_nnz_for_tables.push_back(current_max_nnz);
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
      auto j_embed_params = get_json(j, "sparse_embedding_hparam");
      auto embedding_vec_size = get_value_from_json<int>(j_embed_params, "embedding_vec_size");
      embed_vec_size_for_tables.push_back(embedding_vec_size);
    }  // for ()
  }    // get embedding params

  max_embedding_vector_size_per_sample = 0;
  max_feature_num_per_sample = 0;
  for (int i = 0; i < (int)num_embedding_tables; i++) {
    max_embedding_vector_size_per_sample +=
        (max_feature_num_for_tables[i] * embed_vec_size_for_tables[i]);
    max_feature_num_per_sample += max_feature_num_for_tables[i];
  }
}

template <typename TypeKey, typename TypeFP>
void create_embedding<TypeKey, TypeFP>::operator()(
    const InferenceParams& inference_params, const nlohmann::json& j_layers_array,
    std::vector<std::shared_ptr<Tensor2<int>>>& rows,
    std::vector<std::shared_ptr<Tensor2<float>>>& embeddingvecs,
    std::vector<size_t>& embedding_table_slot_size, std::vector<TensorEntry>* tensor_entries,
    std::vector<std::shared_ptr<Layer>>* embeddings,
    const std::shared_ptr<GPUResource> gpu_resource,
    std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff) {
  HCTR_LOG(INFO, ROOT, "start create embedding for inference\n");
  auto j_data = j_layers_array[0];
  if (!has_key_(j_data, "sparse")) {
    HCTR_LOG(INFO, ROOT, "no sparse data input\n");
    return;
  }
  auto j_sparse_input = get_json(j_data, "sparse");
  std::unordered_map<std::string, std::pair<int, int>> slot_nums_map;
  for (unsigned int i = 0; i < j_sparse_input.size(); ++i) {
    auto top = get_value_from_json<std::string>(j_sparse_input[i], "top");
    auto slot_num = get_value_from_json<int>(j_sparse_input[i], "slot_num");
    int max_feature_num_per_sample =
        get_max_feature_num_per_sample_from_nnz_per_slot(j_sparse_input[i]);
    HCTR_LOG_S(INFO, ROOT) << "sparse_input name " << top << std::endl;
    slot_nums_map[top] = std::make_pair(slot_num, max_feature_num_per_sample);
  }
  if (j_layers_array.size() < 1) {
    HCTR_OWN_THROW(Error_t::WrongInput, "layer not defined in config");
  }
  for (unsigned int i = 1; i < j_layers_array.size(); i++) {
    const nlohmann::json& j = j_layers_array[i];
    auto bottom_array = get_json(j, "bottom");
    if (bottom_array.is_array()) {
      continue;
    }
    std::string bottom = bottom_array.get<std::string>();
    ;
    auto slot_nums_map_iter = slot_nums_map.find(bottom);
    if (slot_nums_map_iter == slot_nums_map.end()) {
      continue;
    }
    const std::string layer_top = get_value_from_json<std::string>(j, "top");
    int slot_num = slot_nums_map_iter->second.first;
    int max_feature_num_per_sample = slot_nums_map_iter->second.second;
    auto j_hparam = get_json(j, "sparse_embedding_hparam");
    auto combiner_str = get_value_from_json<std::string>(j_hparam, "combiner");
    EmbeddingFeatureCombiner_t feature_combiner_type;
    if (combiner_str == "sum") {
      feature_combiner_type = EmbeddingFeatureCombiner_t::Sum;
    } else if (combiner_str == "mean") {
      feature_combiner_type = EmbeddingFeatureCombiner_t::Mean;
    } else {
      HCTR_OWN_THROW(Error_t::WrongInput, "combiner need to be sum or mean");
    }
    size_t embedding_vec_size = get_value_from_json<size_t>(j_hparam, "embedding_vec_size");

    size_t prefix_slot_num = embedding_table_slot_size.back();
    embedding_table_slot_size.push_back(prefix_slot_num + slot_num);

    std::vector<size_t> row_dims = {
        static_cast<size_t>(inference_params.max_batchsize * slot_num + 1)};
    std::vector<size_t> embeddingvecs_dims = {
        static_cast<size_t>(inference_params.max_batchsize * max_feature_num_per_sample),
        static_cast<size_t>(embedding_vec_size)};
    std::shared_ptr<Tensor2<int>> row_tensor = std::make_shared<Tensor2<int>>();
    std::shared_ptr<Tensor2<float>> embeddingvecs_tensor = std::make_shared<Tensor2<float>>();
    blobs_buff->reserve(row_dims, row_tensor.get());
    blobs_buff->reserve(embeddingvecs_dims, embeddingvecs_tensor.get());
    rows.push_back(row_tensor);
    embeddingvecs.push_back(embeddingvecs_tensor);
    Tensor2<TypeFP> embedding_output;
    embeddings->push_back(std::make_shared<EmbeddingFeatureCombiner<TypeFP>>(
        embeddingvecs.back(), rows.back(), embedding_output, inference_params.max_batchsize,
        slot_num, feature_combiner_type, blobs_buff, gpu_resource));
    tensor_entries->push_back({layer_top, embedding_output.shrink()});
  }

  HCTR_LOG(INFO, ROOT, "create embedding for inference success\n");
}

template struct create_embedding<long long, float>;
template struct create_embedding<long long, __half>;
template struct create_embedding<unsigned int, float>;
template struct create_embedding<unsigned int, __half>;
}  // namespace HugeCTR
