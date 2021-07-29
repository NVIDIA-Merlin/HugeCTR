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

#include <data_readers/async_reader/async_reader_adapter.hpp>
#include <embeddings/hybrid_sparse_embedding.hpp>
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

void Parser::create_allreduce_comm(const std::shared_ptr<ResourceManager>& resource_manager,
                                   std::shared_ptr<ExchangeWgrad>& exchange_wgrad) {
  auto ar_algo = AllReduceAlgo::NCCL;
  bool grouped_all_reduce = false;
  if (has_key_(config_, "all_reduce")) {
    auto j_all_reduce = get_json(config_, "all_reduce");
    std::string ar_algo_name = "Oneshot";
    if (has_key_(j_all_reduce, "algo")) {
      ar_algo_name = get_value_from_json<std::string>(j_all_reduce, "algo");
    }
    if (has_key_(j_all_reduce, "grouped")) {
      grouped_all_reduce = get_value_from_json<bool>(j_all_reduce, "grouped");
    }
    MESSAGE_("Using All-reduce algorithm " + ar_algo_name);
    if (!find_item_in_map(ar_algo, ar_algo_name, ALLREDUCE_ALGO_MAP)) {
      CK_THROW_(Error_t::WrongInput, "All reduce algo unknown: " + ar_algo_name);
    }
  }

  resource_manager->set_ar_comm(ar_algo, use_mixed_precision_);

  grouped_all_reduce_ = grouped_all_reduce;
  if (grouped_all_reduce_) {
    if (use_mixed_precision_) {
      exchange_wgrad = std::make_shared<GroupedExchangeWgrad<__half>>(resource_manager);
    } else {
      exchange_wgrad = std::make_shared<GroupedExchangeWgrad<float>>(resource_manager);
    }
  } else {
    if (use_mixed_precision_) {
      exchange_wgrad = std::make_shared<NetworkExchangeWgrad<__half>>(resource_manager);
    } else {
      exchange_wgrad = std::make_shared<NetworkExchangeWgrad<float>>(resource_manager);
    }
  }
}

template <typename TypeKey>
void Parser::create_pipeline_internal(std::shared_ptr<IDataReader>& init_data_reader,
                                      std::shared_ptr<IDataReader>& train_data_reader,
                                      std::shared_ptr<IDataReader>& evaluate_data_reader,
                                      std::vector<std::shared_ptr<IEmbedding>>& embeddings,
                                      std::vector<std::shared_ptr<Network>>& networks,
                                      const std::shared_ptr<ResourceManager>& resource_manager,
                                      std::shared_ptr<ExchangeWgrad>& exchange_wgrad) {
  try {
    create_allreduce_comm(resource_manager, exchange_wgrad);

    std::map<std::string, SparseInput<TypeKey>> sparse_input_map;
    std::vector<TensorEntry> train_tensor_entries_list[resource_manager->get_local_gpu_count()];
    std::vector<TensorEntry> evaluate_tensor_entries_list[resource_manager->get_local_gpu_count()];
    {
      if (!networks.empty()) {
        CK_THROW_(Error_t::WrongInput, "vector network is not empty");
      }

      auto j_layers_array = get_json(config_, "layers");
      auto j_optimizer = get_json(config_, "optimizer");
      check_graph(tensor_active_, j_layers_array);

      // Create Data Reader
      {
        // TODO: In using AsyncReader, if the overlap is disabled,
        // scheduling the data reader should be off.
        // THe scheduling needs to be generalized.
        auto j_solver = get_json(config_, "solver");
        auto enable_overlap = get_value_from_json_soft<bool>(j_solver, "enable_overlap", false);

        const nlohmann::json& j = j_layers_array[0];
        create_datareader<TypeKey>()(j, sparse_input_map, train_tensor_entries_list,
                                     evaluate_tensor_entries_list, init_data_reader,
                                     train_data_reader, evaluate_data_reader, batch_size_,
                                     batch_size_eval_, use_mixed_precision_, repeat_dataset_,
                                     enable_overlap, resource_manager);
      }  // Create Data Reader

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
                embeddings, embedding_type, config_, resource_manager, batch_size_,
                batch_size_eval_, exchange_wgrad, use_mixed_precision_, scaler_, j, use_cuda_graph_,
                grouped_all_reduce_);
          } else {
            create_embedding<TypeKey, float>()(
                sparse_input_map, train_tensor_entries_list, evaluate_tensor_entries_list,
                embeddings, embedding_type, config_, resource_manager, batch_size_,
                batch_size_eval_, exchange_wgrad, use_mixed_precision_, scaler_, j, use_cuda_graph_,
                grouped_all_reduce_);
          }
        }  // for ()
      }    // Create Embedding

      // create network
      int total_gpu_count = resource_manager->get_global_gpu_count();
      if (0 != batch_size_ % total_gpu_count) {
        CK_THROW_(Error_t::WrongInput, "0 != batch_size\%total_gpu_count");
      }
      for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
        networks.emplace_back(Network::create_network(
            j_layers_array, j_optimizer, train_tensor_entries_list[i],
            evaluate_tensor_entries_list[i], total_gpu_count, exchange_wgrad,
            resource_manager->get_local_cpu(), resource_manager->get_local_gpu(i),
            use_mixed_precision_, enable_tf32_compute_, scaler_, use_algorithm_search_,
            use_cuda_graph_, false, grouped_all_reduce_));
      }
    }
    exchange_wgrad->allocate();

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void Parser::create_pipeline(std::shared_ptr<IDataReader>& init_data_reader,
                             std::shared_ptr<IDataReader>& train_data_reader,
                             std::shared_ptr<IDataReader>& evaluate_data_reader,
                             std::vector<std::shared_ptr<IEmbedding>>& embeddings,
                             std::vector<std::shared_ptr<Network>>& networks,
                             const std::shared_ptr<ResourceManager>& resource_manager,
                             std::shared_ptr<ExchangeWgrad>& exchange_wgrad) {
  if (i64_input_key_) {
    create_pipeline_internal<long long>(init_data_reader, train_data_reader, evaluate_data_reader,
                                        embeddings, networks, resource_manager, exchange_wgrad);
  } else {
    create_pipeline_internal<unsigned int>(init_data_reader, train_data_reader,
                                           evaluate_data_reader, embeddings, networks,
                                           resource_manager, exchange_wgrad);
  }
}

// TODO: this whole function is only for HE & AysncReader. It is better to refactor or generalize
// it.
template <typename TypeKey>
void Parser::initialize_pipeline_internal(std::shared_ptr<IDataReader>& init_data_reader,
                                          std::vector<std::shared_ptr<IEmbedding>>& embedding,
                                          const std::shared_ptr<ResourceManager>& resource_manager,
                                          std::shared_ptr<ExchangeWgrad>& exchange_wgrad) {
  try {
    nlohmann::json config = config_;
    auto j_layers_array = get_json(config, "layers");
    bool use_mixed_precision = use_mixed_precision_;
    size_t embed_wgrad_size = 0;
    for (unsigned int i = 1; i < j_layers_array.size(); i++) {
      const nlohmann::json& j = j_layers_array[i];
      auto embedding_name = get_value_from_json<std::string>(j, "type");
      Embedding_t embedding_type = Embedding_t::LocalizedSlotSparseEmbeddingOneHot;
      (void)find_item_in_map(embedding_type, embedding_name, EMBEDDING_TYPE_MAP);
      if (embedding_type == Embedding_t::HybridSparseEmbedding) {
        auto init_data_reader_as =
            std::dynamic_pointer_cast<AsyncReader<TypeKey>>(init_data_reader);
        if (use_mixed_precision) {
          std::shared_ptr<HybridSparseEmbedding<TypeKey, __half>> hybrid_embedding =
              std::dynamic_pointer_cast<HybridSparseEmbedding<TypeKey, __half>>(embedding[i - 1]);

          init_data_reader_as->start();
          init_data_reader_as->read_a_batch_to_device();
          hybrid_embedding->init_model(
              // bags_to_tensors<TypeKey>(init_data_reader_as->get_value_tensors()),
              // embed_wgrad_size);
              init_data_reader_as->get_value_tensors(), embed_wgrad_size);
        } else {
          std::shared_ptr<HybridSparseEmbedding<TypeKey, float>> hybrid_embedding =
              std::dynamic_pointer_cast<HybridSparseEmbedding<TypeKey, float>>(embedding[i - 1]);
          init_data_reader_as->start();
          init_data_reader_as->read_a_batch_to_device();
          hybrid_embedding->init_model(
              // bags_to_tensors<TypeKey>(init_data_reader_as->get_value_tensors()),
              // embed_wgrad_size);
              init_data_reader_as->get_value_tensors(), embed_wgrad_size);
        }
      }
    }
    if (grouped_all_reduce_) {
      exchange_wgrad->update_embed_wgrad_size(embed_wgrad_size);
    }
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}
void Parser::initialize_pipeline(std::shared_ptr<IDataReader>& init_data_reader,
                                 std::vector<std::shared_ptr<IEmbedding>>& embedding,
                                 const std::shared_ptr<ResourceManager>& resource_manager,
                                 std::shared_ptr<ExchangeWgrad>& exchange_wgrad) {
  if (i64_input_key_) {
    initialize_pipeline_internal<long long>(init_data_reader, embedding, resource_manager,
                                            exchange_wgrad);
  } else {
    initialize_pipeline_internal<unsigned int>(init_data_reader, embedding, resource_manager,
                                               exchange_wgrad);
  }
}
}  // namespace HugeCTR
