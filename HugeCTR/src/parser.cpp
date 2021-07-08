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

#include <parser.hpp>

namespace HugeCTR {

Parser::Parser(const std::string& configure_file, size_t batch_size, size_t batch_size_eval,
               bool repeat_dataset, bool i64_input_key, bool use_mixed_precision,
               bool enable_tf32_compute, float scaler, bool use_algorithm_search,
               bool use_cuda_graph)
    : config_(read_json_file(configure_file)),
      batch_size_(batch_size),
      batch_size_eval_(batch_size_eval),
      repeat_dataset_(repeat_dataset),
      i64_input_key_(i64_input_key),
      use_mixed_precision_(use_mixed_precision),
      enable_tf32_compute_(enable_tf32_compute),
      scaler_(scaler),
      use_algorithm_search_(use_algorithm_search),
      use_cuda_graph_(use_cuda_graph) {}

template <typename TypeKey>
void Parser::create_pipeline_internal(std::shared_ptr<IDataReader>& train_data_reader,
                                      std::shared_ptr<IDataReader>& evaluate_data_reader,
                                      std::vector<std::shared_ptr<IEmbedding>>& embeddings,
                                      std::vector<std::shared_ptr<Network>>& network,
                                      const std::shared_ptr<ResourceManager>& resource_manager) {
  try {
    std::map<std::string, SparseInput<TypeKey>> sparse_input_map;
    std::vector<TensorEntry> train_tensor_entries_list[resource_manager->get_local_gpu_count()];
    std::vector<TensorEntry> evaluate_tensor_entries_list[resource_manager->get_local_gpu_count()];
    {
      if (!network.empty()) {
        CK_THROW_(Error_t::WrongInput, "vector network is not empty");
      }

      auto j_layers_array = get_json(config_, "layers");
      auto j_optimizer = get_json(config_, "optimizer");
      check_graph(tensor_active_, j_layers_array);

      // Create Data Reader
      {
        const nlohmann::json& j = j_layers_array[0];
        create_datareader<TypeKey>()(j, sparse_input_map, train_tensor_entries_list,
                                     evaluate_tensor_entries_list, train_data_reader, evaluate_data_reader,
                                     batch_size_, batch_size_eval_, use_mixed_precision_, repeat_dataset_,
                                     resource_manager);
      }

      // Create Embedding
      {
        for (unsigned int i = 1; i < j_layers_array.size(); i++) {
          // if not embedding then break
          const nlohmann::json& j = j_layers_array[i];
          auto embedding_name = get_value_from_json<std::string>(j, "type");
          Embedding_t embedding_type;
          if (!find_item_in_map(embedding_type, embedding_name, EMBEDDING_TYPE_MAP)) {
            Layer_t layer_type;
            if (!find_item_in_map(layer_type, embedding_name, LAYER_TYPE_MAP) &&
                !find_item_in_map(layer_type, embedding_name, LAYER_TYPE_MAP_MP)) {
              CK_THROW_(Error_t::WrongInput, "No such layer: " + embedding_name);
            }
            break;
          }

          if (use_mixed_precision_) {
            create_embedding<TypeKey, __half>()(
                sparse_input_map, train_tensor_entries_list, evaluate_tensor_entries_list,
                embeddings, embedding_type, config_, resource_manager, batch_size_, batch_size_eval_,
                use_mixed_precision_, scaler_, j);
          } else {
            create_embedding<TypeKey, float>()(sparse_input_map, train_tensor_entries_list,
                                               evaluate_tensor_entries_list, embeddings,
                                               embedding_type, config_, resource_manager, batch_size_,
                                               batch_size_eval_, use_mixed_precision_, scaler_, j);
          }
        }  // for ()
      }    // Create Embedding

      // create network
      int total_gpu_count = resource_manager->get_global_gpu_count();
      if (0 != batch_size_ % total_gpu_count) {
        CK_THROW_(Error_t::WrongInput, "0 != batch_size\%total_gpu_count");
      }
      for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
        network.emplace_back(Network::create_network(
            j_layers_array, j_optimizer, train_tensor_entries_list[i],
            evaluate_tensor_entries_list[i], total_gpu_count, resource_manager->get_local_cpu(),
            resource_manager->get_local_gpu(i), use_mixed_precision_, enable_tf32_compute_, scaler_,
            use_algorithm_search_, use_cuda_graph_, false));
      }
    }

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void Parser::create_pipeline(std::shared_ptr<IDataReader>& data_reader,
                             std::shared_ptr<IDataReader>& data_reader_eval,
                             std::vector<std::shared_ptr<IEmbedding>>& embedding,
                             std::vector<std::shared_ptr<Network>>& network,
                             const std::shared_ptr<ResourceManager>& resource_manager) {
  if (i64_input_key_) {
    create_pipeline_internal<long long>(data_reader, data_reader_eval, embedding, network,
                                        resource_manager);
  } else {
    create_pipeline_internal<unsigned int>(data_reader, data_reader_eval, embedding, network,
                                           resource_manager);
  }
}
}  // namespace HugeCTR
