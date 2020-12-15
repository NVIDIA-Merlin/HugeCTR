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

#include <data_readers/data_reader.hpp>
#include <embeddings/distributed_slot_sparse_embedding_hash.hpp>
#include <embeddings/localized_slot_sparse_embedding_hash.hpp>
#include <embeddings/localized_slot_sparse_embedding_one_hot.hpp>
#include <layer.hpp>
#include <layers/add_layer.hpp>
#include <layers/batch_norm_layer.hpp>
#include <layers/cast_layer.hpp>
#include <layers/concat_layer.hpp>
#include <layers/dot_product_layer.hpp>
#include <layers/dropout_cudnn_layer.hpp>
#include <layers/dropout_layer.hpp>
#include <layers/elu_layer.hpp>
#include <layers/fm_order2_layer.hpp>
#include <layers/fully_connected_layer.hpp>
#include <layers/fully_connected_layer_half.hpp>
#include <layers/fused_fully_connected_layer.hpp>
#include <layers/interaction_layer.hpp>
#include <layers/multi_cross_layer.hpp>
#include <layers/multiply_layer.hpp>
#include <layers/reduce_sum_layer.hpp>
#include <layers/relu_layer.hpp>
#include <layers/reshape_layer.hpp>
#include <layers/sigmoid_layer.hpp>
#include <layers/slice_layer.hpp>
#include <loss.hpp>
#include <metrics.hpp>
#include <optimizer.hpp>
#include <parser.hpp>
#include <regularizers/l1_regularizer.hpp>
#include <regularizers/l2_regularizer.hpp>
#include <regularizers/no_regularizer.hpp>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {
Parser::Parser(const std::string& configure_file, size_t batch_size, size_t batch_size_eval,
       bool repeat_dataset, bool i64_input_key, bool use_mixed_precision,
       float scaler, bool use_algorithm_search, bool use_cuda_graph)
    : batch_size_(batch_size),
      batch_size_eval_(batch_size_eval),
      repeat_dataset_(repeat_dataset),
      i64_input_key_(i64_input_key),
      use_mixed_precision_(use_mixed_precision),
      scaler_(scaler),
      use_algorithm_search_(use_algorithm_search),
      use_cuda_graph_(use_cuda_graph) {
  try {
    std::ifstream file(configure_file);
    if (!file.is_open()) {
      CK_THROW_(Error_t::FileCannotOpen, "file.is_open() failed: " + configure_file);
    }
    file >> config_;
    file.close();
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
  return;
}

template <typename TypeKey>
void Parser::create_pipeline_internal(std::shared_ptr<IDataReader>& data_reader,
                                      std::shared_ptr<IDataReader>& data_reader_eval,
                                      std::vector<std::shared_ptr<IEmbedding>>& embedding,
                                      std::vector<std::unique_ptr<Network>>& network,
                                      const std::shared_ptr<ResourceManager>& resource_manager,
                                      Parser& parser) {
  try {
    nlohmann::json config = parser.config_;
    size_t batch_size = parser.batch_size_;
    size_t batch_size_eval = parser.batch_size_eval_;
    bool use_mixed_precision = parser.use_mixed_precision_;
    float scaler = parser.scaler_;
    bool use_algorithm_search = parser.use_algorithm_search_;
    bool use_cuda_graph = parser.use_cuda_graph_;

    std::map<std::string, SparseInput<TypeKey>> sparse_input_map;
    std::vector<TensorEntry> tensor_entries_list[resource_manager->get_local_gpu_count()];
    {
      if (!network.empty()) {
        CK_THROW_(Error_t::WrongInput, "vector network is not empty");
      }

      auto j_layers_array = get_json(config, "layers");
      auto j_optimizer = get_json(config, "optimizer");

      // Create Data Reader
      {
        const nlohmann::json& j = j_layers_array[0];
        const auto layer_type_name = get_value_from_json<std::string>(j, "type");
        if (layer_type_name.compare("Data") != 0) {
          CK_THROW_(Error_t::WrongInput, "the first layer is not Data layer:" + layer_type_name);
        }

        const std::map<std::string, DataReaderType_t> DATA_READER_MAP = {
            {"Norm", DataReaderType_t::Norm},
            {"Raw", DataReaderType_t::Raw},
            {"Parquet", DataReaderType_t::Parquet}};

        DataReaderType_t format = DataReaderType_t::Norm;
        if (has_key_(j, "format")) {
          const auto data_format_name = get_value_from_json<std::string>(j, "format");
          if (!find_item_in_map(format, data_format_name, DATA_READER_MAP)) {
            CK_THROW_(Error_t::WrongInput, "No such data format: " + data_format_name);
          }
        }

        auto cache_eval_data = get_value_from_json_soft<int>(j, "cache_eval_data", 0);

        std::string source_data = get_value_from_json<std::string>(j, "source");

        auto j_label = get_json(j, "label");
        auto top_strs_label = get_value_from_json<std::string>(j_label, "top");
        auto label_dim = get_value_from_json<int>(j_label, "label_dim");

        auto j_dense = get_json(j, "dense");
        auto top_strs_dense = get_value_from_json<std::string>(j_dense, "top");
        auto dense_dim = get_value_from_json<int>(j_dense, "dense_dim");

        const std::map<std::string, Check_t> CHECK_TYPE_MAP = {{"Sum", Check_t::Sum},
                                                               {"None", Check_t::None}};

        Check_t check_type;
        const auto check_str = get_value_from_json<std::string>(j, "check");
        if (!find_item_in_map(check_type, check_str, CHECK_TYPE_MAP)) {
          CK_THROW_(Error_t::WrongInput, "Not supported check type: " + check_str);
        }

        std::vector<DataReaderSparseParam> data_reader_sparse_param_array;

        const std::map<std::string, DataReaderSparse_t> DATA_TYPE_MAP = {
            {"DistributedSlot", DataReaderSparse_t::Distributed},
            {"LocalizedSlot", DataReaderSparse_t::Localized},
        };

        auto j_sparse = get_json(j, "sparse");
        std::vector<std::string> sparse_names;

        for (unsigned int i = 0; i < j_sparse.size(); i++) {
          DataReaderSparseParam param;

          const nlohmann::json& js = j_sparse[i];
          const auto sparse_name = get_value_from_json<std::string>(js, "top");
          const auto data_type_name = get_value_from_json<std::string>(js, "type");
          if (!find_item_in_map(param.type, data_type_name, DATA_TYPE_MAP)) {
            CK_THROW_(Error_t::WrongInput, "Not supported data type: " + data_type_name);
          }
          param.max_feature_num = get_value_from_json<int>(js, "max_feature_num_per_sample");
          param.max_nnz = get_value_from_json_soft<int>(js, "max_nnz", param.max_feature_num);
          param.slot_num = get_value_from_json<int>(js, "slot_num");
          data_reader_sparse_param_array.push_back(param);
          SparseInput<TypeKey> sparse_input(param.slot_num, param.max_feature_num);
          sparse_input_map.emplace(sparse_name, sparse_input);
          sparse_names.push_back(sparse_name);
        }

        data_reader_eval = nullptr;
        std::string eval_source;
        FIND_AND_ASSIGN_STRING_KEY(eval_source, j);

#ifdef VAL
        const int NUM_THREADS = 1;
#else
        const int NUM_THREADS =
            format == DataReaderType_t::Parquet ? resource_manager->get_local_gpu_count() : 12;
#endif

        DataReader<TypeKey>* data_reader_tk = new DataReader<TypeKey>(
            batch_size, label_dim, dense_dim, data_reader_sparse_param_array, resource_manager,
            parser.repeat_dataset_, NUM_THREADS, use_mixed_precision, false);
        data_reader.reset(data_reader_tk);
        DataReader<TypeKey>* data_reader_eval_tk = new DataReader<TypeKey>(
            batch_size_eval, label_dim, dense_dim, data_reader_sparse_param_array, resource_manager,
            parser.repeat_dataset_, NUM_THREADS, use_mixed_precision, cache_eval_data);
        data_reader_eval.reset(data_reader_eval_tk);

        auto f = [&j]() -> std::vector<long long> {
          std::vector<long long> slot_offset;
          if (has_key_(j, "slot_size_array")) {
            auto slot_size_array = get_json(j, "slot_size_array");
            if (!slot_size_array.is_array()) {
              CK_THROW_(Error_t::WrongInput, "!slot_size_array.is_array()");
            }
            long long slot_sum = 0;
            for (auto j_slot_size : slot_size_array) {
              slot_offset.push_back(slot_sum);
              long long slot_size = j_slot_size.get<long long>();
              slot_sum += slot_size;
            }
            MESSAGE_("Vocabulary size: " + std::to_string(slot_sum));
          }
          return slot_offset;
        };

        switch (format) {
          case DataReaderType_t::Norm: {
            bool start_right_now = parser.repeat_dataset_;
            data_reader->create_drwg_norm(source_data, check_type, start_right_now);
            data_reader_eval->create_drwg_norm(eval_source, check_type, start_right_now);
            break;
          }
          case DataReaderType_t::Raw: {
            const auto num_samples = get_value_from_json<long long>(j, "num_samples");
            const auto eval_num_samples = get_value_from_json<long long>(j, "eval_num_samples");
            std::vector<long long> slot_offset = f();
            bool float_label_dense = get_value_from_json_soft<bool>(j, "float_label_dense", false);
            data_reader->create_drwg_raw(source_data, num_samples, slot_offset, float_label_dense,
                                         true, false);
            data_reader_eval->create_drwg_raw(eval_source, eval_num_samples, slot_offset,
                                              float_label_dense, false, false);

            break;
          }
          case DataReaderType_t::Parquet: {
            // @Future: Should be slot_offset here and data_reader ctor should
            // be TypeKey not long long
            std::vector<long long> slot_offset = f();
            data_reader->create_drwg_parquet(source_data, slot_offset, true);
            data_reader_eval->create_drwg_parquet(eval_source, slot_offset, true);
            break;
          }
          default: {
            assert(!"Error: no such option && should never get here!");
          }
        }

        for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
          tensor_entries_list[i].push_back(
              {top_strs_label, TensorUse::Train, data_reader_tk->get_label_tensors()[i].shrink()});
          tensor_entries_list[i].push_back({top_strs_label, TensorUse::Evaluate,
                                            data_reader_eval_tk->get_label_tensors()[i].shrink()});

          if (use_mixed_precision) {
            tensor_entries_list[i].push_back(
                {top_strs_dense, TensorUse::Train, data_reader_tk->get_dense_tensors()[i]});
            tensor_entries_list[i].push_back(
                {top_strs_dense, TensorUse::Evaluate, data_reader_eval_tk->get_dense_tensors()[i]});
          } else {
            tensor_entries_list[i].push_back(
                {top_strs_dense, TensorUse::Train, data_reader_tk->get_dense_tensors()[i]});
            tensor_entries_list[i].push_back(
                {top_strs_dense, TensorUse::Evaluate, data_reader_eval_tk->get_dense_tensors()[i]});
          }
        }

        for (unsigned int i = 0; i < j_sparse.size(); i++) {
          const auto& sparse_input = sparse_input_map.find(sparse_names[i]);
          sparse_input->second.train_row_offsets = data_reader_tk->get_row_offsets_tensors(i);
          sparse_input->second.train_values = data_reader_tk->get_value_tensors(i);
          sparse_input->second.train_nnz = data_reader_tk->get_nnz_array(i);
          sparse_input->second.evaluate_row_offsets =
              data_reader_eval_tk->get_row_offsets_tensors(i);
          sparse_input->second.evaluate_values = data_reader_eval_tk->get_value_tensors(i);
          sparse_input->second.evaluate_nnz = data_reader_eval_tk->get_nnz_array(i);
        }
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

          if (use_mixed_precision) {
            create_embedding<TypeKey, __half>()(sparse_input_map, tensor_entries_list, embedding,
                                              embedding_type, config, resource_manager, batch_size,
                                              batch_size_eval, use_mixed_precision, scaler, j);
          } else {
            create_embedding<TypeKey, float>()(sparse_input_map, tensor_entries_list, embedding,
                                             embedding_type, config, resource_manager, batch_size,
                                             batch_size_eval, use_mixed_precision, scaler, j);
          }
        }  // for ()
      }    // Create Embedding

      // create network
      int total_gpu_count = resource_manager->get_global_gpu_count();
      if (0 != batch_size % total_gpu_count) {
        CK_THROW_(Error_t::WrongInput, "0 != batch_size\%total_gpu_count");
      }
      for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
        network.emplace_back(Network::create_network(
            j_layers_array, j_optimizer, tensor_entries_list[i], total_gpu_count,
            resource_manager->get_local_cpu(), resource_manager->get_local_gpu(i),
            use_mixed_precision, scaler, use_algorithm_search, use_cuda_graph, false));
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
                             std::vector<std::unique_ptr<Network>>& network,
                             const std::shared_ptr<ResourceManager>& resource_manager) {
  if (i64_input_key_) {
    create_pipeline_internal<long long>(data_reader, data_reader_eval, embedding, network,
                                        resource_manager, *this);
  } else {
    create_pipeline_internal<unsigned int>(data_reader, data_reader_eval, embedding, network,
                                           resource_manager, *this);
  }
}


}  // namespace HugeCTR
