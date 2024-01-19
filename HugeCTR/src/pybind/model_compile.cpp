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

#include <algorithm>
#include <core/hctr_impl/hctr_backend.hpp>
#include <core23/logger.hpp>
#include <core23/mpi_init_service.hpp>
#include <core23_helper.hpp>
#include <core23_network.hpp>
#include <data_readers/multi_hot/async_data_reader.hpp>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <network_buffer_channels.hpp>
#include <pybind/model.hpp>
#include <resource_managers/resource_manager_core.hpp>
#include <sstream>
using namespace HugeCTR::MultiHot;

namespace HugeCTR {
namespace {

static std::string join(std::vector<std::string>& strs, std::string delim) {
  std::string str;
  const std::vector<std::string>::iterator itlast = strs.end() - 1;
  for (auto it = strs.begin(); it != strs.end(); it++) {
    str += *it;
    if (it != itlast) {
      str += delim;
    }
  }
  return str;
}

static std::vector<std::string>& split(const std::string& s, char delim,
                                       std::vector<std::string>& elems) {
  std::istringstream is(s);
  std::string item;
  while (std::getline(is, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

static std::string get_tensor_shape(std::string tensor_name,
                                    std::map<std::string, std::vector<size_t>> tensor_shape_info) {
  std::string shape = "";
  if (tensor_shape_info.find(tensor_name) != tensor_shape_info.end()) {
    shape += "(";
    for (unsigned int i = 0; i < tensor_shape_info[tensor_name].size(); i++) {
      shape += std::to_string(tensor_shape_info[tensor_name][i]);
      shape += ",";
    }
    shape.back() = ')';
  }
  return shape;
}

static std::string get_tensor_shape(std::string tensor_name,
                                    std::map<std::string, core23::Shape> tensor_shape_info) {
  std::stringstream ss;
  if (tensor_shape_info.find(tensor_name) != tensor_shape_info.end()) {
    ss << tensor_shape_info[tensor_name];
  }
  return ss.str();
}

}  // namespace

void Model::add(Input& input) {
  std::string label_name = input.labels_.begin()->first;
  int label_dim = input.labels_.begin()->second;

  // If multiple labels, treat them as 1 big label and add a split layer (below)
  if (input.labels_.size() > 1) {
    label_name = "combined_multi_label";
    label_dim = std::accumulate(std::begin(input.labels_), std::end(input.labels_), 0,
                                [](const int previous, const std::pair<std::string, int>& p) {
                                  return previous + p.second;
                                });
  }

  input_params_.push_back(input);
  activate_tensor(tensor_active_, label_name);
  activate_tensor(tensor_active_, input.dense_name);
  data_input_info_.push_back(label_name);
  data_input_info_.push_back(input.dense_name);
  tensor_shape_info_raw_.insert(
      std::make_pair(label_name, std::vector<int>{solver_.batchsize, label_dim}));
  tensor_shape_info_raw_.insert(
      std::make_pair(input.dense_name, std::vector<int>{solver_.batchsize, input.dense_dim}));
  if (solver_.use_embedding_collection) {
    std::vector<std::string> top_name_list;
    std::vector<int> nnz_per_slot;
    bool is_fixed_length = true;
    int num_slot = 0;
    for (size_t i = 0; i < input.data_reader_sparse_param_array.size(); ++i) {
      auto& p = input.data_reader_sparse_param_array[i];
      top_name_list.push_back(p.top_name);
      if (p.slot_num != 1) {
        HCTR_OWN_THROW(
            Error_t::WrongInput,
            "To use embedding collection, slots_num should be set to 1 in each sparse_param. "
            "Please refer to notebooks/embedding_collection.ipynb and separate your multi-slot "
            "output into multiple single-slot output");
      }
      nnz_per_slot.push_back(p.nnz_per_slot[0]);
      if (!p.is_fixed_length) is_fixed_length = false;
      num_slot += 1;
      hotness_map_.insert({p.top_name, p.max_feature_num});
    }
    std::string concat_top_name = join(top_name_list, ",");
    DataReaderSparseParam concat_data_reader_sparse_param{concat_top_name, nnz_per_slot,
                                                          is_fixed_length, num_slot};
    input.data_reader_sparse_param_array = {concat_data_reader_sparse_param};
  }
  std::vector<std::string> sparse_names;
  for (size_t i = 0; i < input.data_reader_sparse_param_array.size(); ++i) {
    sparse_names.push_back(input.data_reader_sparse_param_array[i].top_name);
    tensor_shape_info_raw_.insert(std::make_pair(
        input.data_reader_sparse_param_array[i].top_name,
        std::vector<int>{solver_.batchsize, input.data_reader_sparse_param_array[i].slot_num}));
  }
  data_input_info_.push_back(join(sparse_names, ","));
  for (unsigned int i = 0; i < input.data_reader_sparse_param_array.size(); i++) {
    activate_tensor(tensor_active_, input.data_reader_sparse_param_array[i].top_name);
  }
  if (solver_.i64_input_key) {
    add_input<long long>(input, reader_params_, sparse_input_map_64_, train_tensor_entities_list_,
                         evaluate_tensor_entities_list_, train_data_reader_, evaluate_data_reader_,
                         solver_.batchsize, solver_.batchsize_eval, solver_.use_mixed_precision,
                         solver_.repeat_dataset, solver_.train_intra_iteration_overlap,
                         solver_.num_iterations_statistics, resource_manager_);
  } else {
    add_input<unsigned int>(input, reader_params_, sparse_input_map_32_,
                            train_tensor_entities_list_, evaluate_tensor_entities_list_,
                            train_data_reader_, evaluate_data_reader_, solver_.batchsize,
                            solver_.batchsize_eval, solver_.use_mixed_precision,
                            solver_.repeat_dataset, solver_.train_intra_iteration_overlap,
                            solver_.num_iterations_statistics, resource_manager_);
  }

  if (solver_.use_embedding_collection and solver_.train_inter_iteration_overlap) {
    create_copy_ops_for_network_input(input.dense_name, label_name, true);
  }
  if (solver_.use_embedding_collection and solver_.eval_inter_iteration_overlap) {
    create_copy_ops_for_network_input(input.dense_name, label_name, false);
  }

  // Add label weights to model
  for (std::map<std::string, float>::iterator iter = input.label_weights_.begin();
       iter != input.label_weights_.end(); ++iter) {
    label_weights_.insert(std::make_pair(iter->first, iter->second));
  }

  // If multiple labels provided, add a Slice layer to handle breaking up the label
  if (input.labels_.size() > 1) {
    std::vector<std::string> label_names;
    std::vector<std::pair<int, int>> ranges;
    int idx = 0;

    for (std::map<std::string, int>::iterator iter = input.labels_.begin();
         iter != input.labels_.end(); ++iter) {
      label_names.push_back(iter->first);
      if (iter->second < 1) {
        HCTR_OWN_THROW(Error_t::WrongInput, "Each label dimension must be at lesat 1.");
      }
      ranges.push_back(std::make_pair(idx, idx + iter->second));
      idx += iter->second;
    }
    std::vector<std::string> bottom_name{"combined_multi_label"};
    DenseLayer label_slice_layer = DenseLayer(Layer_t::Slice, bottom_name, label_names);
    label_slice_layer.ranges = ranges;

    add(label_slice_layer);
  }
}

void Model::add(SparseEmbedding& sparse_embedding) {
  OptParams embedding_opt_params;
  if (!(sparse_embedding.embedding_opt_params)->initialized) {
    sparse_embedding.embedding_opt_params = opt_params_py_;
    sparse_embedding.initialize_max_vocabulary_size_per_gpu();
  }
  sparse_embedding.max_vocabulary_size_global =
      sparse_embedding.max_vocabulary_size_per_gpu * resource_manager_->get_global_gpu_count();
  sparse_embedding_params_.push_back(sparse_embedding);
  deactivate_tensor(tensor_active_, sparse_embedding.bottom_name);
  activate_tensor(tensor_active_, sparse_embedding.sparse_embedding_name);
  int slot_num = tensor_shape_info_raw_[sparse_embedding.bottom_name][1];
  tensor_shape_info_raw_.insert(
      std::make_pair(sparse_embedding.sparse_embedding_name,
                     std::vector<int>{solver_.batchsize, slot_num,
                                      static_cast<int>(sparse_embedding.embedding_vec_size)}));
  input_output_info_.push_back(
      std::make_pair(sparse_embedding.bottom_name, sparse_embedding.sparse_embedding_name));
  layer_info_.push_back(EMBEDDING_TYPE_TO_STRING[sparse_embedding.embedding_type]);

  embedding_opt_params_list_.push_back(sparse_embedding.embedding_opt_params);
  init_optimizer_params(embedding_opt_params, solver_, sparse_embedding.embedding_opt_params);
  if (solver_.i64_input_key && !solver_.use_mixed_precision) {
    add_sparse_embedding<long long, float>(
        sparse_embedding, sparse_input_map_64_, train_tensor_entities_list_,
        evaluate_tensor_entities_list_, embeddings_, resource_manager_, collective_manager_,
        solver_.batchsize, solver_.batchsize_eval, embedding_opt_params, exchange_wgrad_,
        solver_.use_cuda_graph, solver_.grouped_all_reduce, solver_.num_iterations_statistics,
        gpu_lr_sches_);
  } else if (solver_.i64_input_key && solver_.use_mixed_precision) {
    add_sparse_embedding<long long, __half>(
        sparse_embedding, sparse_input_map_64_, train_tensor_entities_list_,
        evaluate_tensor_entities_list_, embeddings_, resource_manager_, collective_manager_,
        solver_.batchsize, solver_.batchsize_eval, embedding_opt_params, exchange_wgrad_,
        solver_.use_cuda_graph, solver_.grouped_all_reduce, solver_.num_iterations_statistics,
        gpu_lr_sches_);
  } else if (!solver_.i64_input_key && !solver_.use_mixed_precision) {
    add_sparse_embedding<unsigned int, float>(
        sparse_embedding, sparse_input_map_32_, train_tensor_entities_list_,
        evaluate_tensor_entities_list_, embeddings_, resource_manager_, collective_manager_,
        solver_.batchsize, solver_.batchsize_eval, embedding_opt_params, exchange_wgrad_,
        solver_.use_cuda_graph, solver_.grouped_all_reduce, solver_.num_iterations_statistics,
        gpu_lr_sches_);
  } else {
    add_sparse_embedding<unsigned int, __half>(
        sparse_embedding, sparse_input_map_32_, train_tensor_entities_list_,
        evaluate_tensor_entities_list_, embeddings_, resource_manager_, collective_manager_,
        solver_.batchsize, solver_.batchsize_eval, embedding_opt_params, exchange_wgrad_,
        solver_.use_cuda_graph, solver_.grouped_all_reduce, solver_.num_iterations_statistics,
        gpu_lr_sches_);
  }
  embeddings_map_.insert(
      std::make_pair(sparse_embedding.sparse_embedding_name, embeddings_.back()));
  embedding_dependent_tensors_.insert(sparse_embedding.sparse_embedding_name);
}

void Model::add(DenseLayer& dense_layer) {
  for (auto& top_name : dense_layer.top_names) {
    if (tensor_shape_info_raw_.find(top_name) != tensor_shape_info_raw_.end()) {
      HCTR_OWN_THROW(Error_t::WrongInput, top_name + ", top tensor name already exists");
    }
  }
  for (auto& bottom_name : dense_layer.bottom_names) {
    if (tensor_shape_info_raw_.find(bottom_name) == tensor_shape_info_raw_.end()) {
      HCTR_OWN_THROW(Error_t::WrongInput, bottom_name + ", bottom tensor name does not exists");
    }
  }
  calculate_tensor_dimensions(tensor_shape_info_raw_, dense_layer);
  dense_layer_params_raw_.push_back(dense_layer);
}

template <typename emb_t>
void allocate_ebc_output_helper_for_feature_major(
    std::shared_ptr<ResourceManager> resource_manager_, size_t batch_size_per_gpu,
    const EmbeddingCollectionConfig& ebc_config,
    const embedding::EmbeddingCollectionParam& ebc_param,
    std::vector<std::vector<TensorEntity>>& tensor_entries_list_,
    std::vector<core23::Tensor>& ebc_output) {
  HCTR_CHECK(ebc_config.output_layout_ == embedding::EmbeddingLayout::FeatureMajor);
  int num_local_gpus = resource_manager_->get_local_gpu_count();
  for (int local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
    CudaDeviceContext context(resource_manager_->get_local_gpu(local_gpu_id)->get_device_id());
    core23::Device device(core23::DeviceType::GPU,
                          resource_manager_->get_local_gpu(local_gpu_id)->get_device_id());
    auto buffer_channel = core23::GetRandomBufferChannel();
    core23::Tensor head_tensor;
    core23::BufferParams buffer_param{.channel = buffer_channel};
    core23::TensorParams tensor_param = core23::TensorParams().buffer_params(buffer_param);
    int64_t concat_dims = 0;
    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      const embedding::LookupParam& lookup_param = ebc_param.lookup_params[lookup_id];
      std::string top_name = ebc_config.top_names_[lookup_id];
      int64_t emb_out_dims = (lookup_param.combiner == embedding::Combiner::Concat)
                                 ? lookup_param.max_hotness * lookup_param.ev_size
                                 : lookup_param.ev_size;

      core23::Tensor tmp_tensor(tensor_param.shape({(int64_t)batch_size_per_gpu, 1ll, emb_out_dims})
                                    .device(device)
                                    .data_type(core23::ToScalarType<emb_t>::value));
      concat_dims += emb_out_dims;
      tensor_entries_list_[local_gpu_id].push_back({top_name, tmp_tensor});
      if (!lookup_id) {
        head_tensor = tmp_tensor;
      }
    }
    // allocate
    void* starting_address = head_tensor.data();
    core23::Tensor continous_emb_output = core23::Tensor::bind(
        starting_address, core23::Shape({static_cast<int64_t>(batch_size_per_gpu), concat_dims}),
        core23::ToScalarType<emb_t>::value, device);
    ebc_output.push_back(continous_emb_output);
  }
}

template <typename emb_t>
void allocate_ebc_output_helper_for_batch_major(
    std::shared_ptr<ResourceManager> resource_manager_, size_t batch_size_per_gpu,
    const EmbeddingCollectionConfig& ebc_config,
    const embedding::EmbeddingCollectionParam& ebc_param,
    std::vector<std::vector<TensorEntity>>& tensor_entries_list_,
    std::vector<core23::Tensor>& ebc_output) {
  int num_local_gpus = resource_manager_->get_local_gpu_count();
  for (int local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
    CudaDeviceContext context(resource_manager_->get_local_gpu(local_gpu_id)->get_device_id());

    core23::Device device(core23::DeviceType::GPU,
                          resource_manager_->get_local_gpu(local_gpu_id)->get_device_id());
    core23::TensorParams tensor_param;
    int64_t emb_out_dims = 0;
    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      const embedding::LookupParam& lookup_param = ebc_param.lookup_params[lookup_id];

      emb_out_dims += (lookup_param.combiner == embedding::Combiner::Concat)
                          ? lookup_param.max_hotness * lookup_param.ev_size
                          : lookup_param.ev_size;
    }

    core23::Tensor continous_emb_output(
        tensor_param.shape({(int64_t)batch_size_per_gpu, emb_out_dims})
            .device(device)
            .data_type(core23::ToScalarType<emb_t>::value));
    continous_emb_output.data();
    ebc_output.push_back(continous_emb_output);

    tensor_entries_list_[local_gpu_id].push_back(
        {ebc_config.batch_major_output_name_, continous_emb_output});
  }
}

void Model::add(const EmbeddingCollectionConfig& user_ebc_config) {
  auto ebc_config = split_column_wise_sharding_config(user_ebc_config);
  TableNameToIDDict table_name_to_id_dict =
      create_table_name_to_id_dict_from_ebc_config(ebc_config);
  int global_ebc_id = static_cast<int>(ebc_list_.size());
  for (auto& [name, id] : table_name_to_id_dict) {
    HCTR_CHECK_HINT(ebc_name_to_global_id_dict_.find(name) == ebc_name_to_global_id_dict_.end(),
                    "Duplicate table name: ", name, "\n");
    ebc_name_to_global_id_dict_[name] = {global_ebc_id, id};
  }
  int num_total_gpus = resource_manager_->get_global_gpu_count();
  int num_local_gpus = resource_manager_->get_local_gpu_count();

  int num_lookup = ebc_config.lookup_configs_.size();
  core23::DataType key_type =
      solver_.i64_input_key ? core23::ScalarType::Int64 : core23::ScalarType::UInt32;
  core23::DataType index_type =
      solver_.i64_input_key ? core23::ScalarType::UInt64 : core23::ScalarType::UInt32;
  core23::DataType offset_type =
      solver_.i64_input_key ? core23::ScalarType::Int64 : core23::ScalarType::UInt32;
  core23::DataType emb_type =
      solver_.use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float;
  core23::DataType wgrad_type =
      solver_.use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float;
  embedding::EmbeddingLayout input_layout_ =
      reader_params_.data_reader_type == DataReaderType_t::RawAsync
          ? embedding::EmbeddingLayout::FeatureMajor
          : embedding::EmbeddingLayout::BatchMajor;

  std::vector<std::string> bottom_name_list;
  for (auto& bottom_name : ebc_config.bottom_names_) {
    bottom_name_list.push_back(bottom_name);
  }

  std::string bottom_name = join(bottom_name_list, ",");
  deactivate_tensor(tensor_active_, bottom_name);

  layer_info_.push_back("EmbeddingCollection" + std::to_string(ebc_list_.size()));

  auto lookup_params = create_lookup_params_from_ebc_config(table_name_to_id_dict, ebc_config);
  for (int lookup_id = 0; lookup_id < num_lookup; ++lookup_id) {
    auto b_name = ebc_config.bottom_names_[ebc_config.dr_lookup_ids_[lookup_id]];
    lookup_params[lookup_id].max_hotness = hotness_map_[b_name];
  }

  auto shard_matrix = create_shard_matrix_from_ebc_config(table_name_to_id_dict, ebc_config);

  auto grouped_emb_params =
      create_grouped_embedding_param_from_ebc_config(table_name_to_id_dict, ebc_config);

  int num_table = ebc_config.emb_table_config_list_.size();
  auto emb_table_list = create_table_params_from_ebc_config(table_name_to_id_dict, ebc_config);
  for (auto& p : emb_table_list) {
    if (p.opt_param.optimizer == Optimizer_t::NOT_INITIALIZED) {
      p.opt_param = opt_params_;
    }
  }

  embedding::AllreduceStrategy allreduce_strategy = ebc_config.allreduce_strategy_;
  if (solver_.grouped_all_reduce) {
    allreduce_strategy = embedding::AllreduceStrategy::GroupDense;
  }

  auto compression_param =
      create_compression_param_from_ebc_config(table_name_to_id_dict, ebc_config);
  embedding::EmbeddingCollectionParam ebc_param{num_table,
                                                num_lookup,
                                                lookup_params,
                                                shard_matrix,
                                                grouped_emb_params,
                                                solver_.batchsize,
                                                key_type,
                                                index_type,
                                                offset_type,
                                                emb_type,
                                                wgrad_type,
                                                input_layout_,
                                                ebc_config.output_layout_,
                                                ebc_config.sort_strategy_,
                                                ebc_config.keys_preprocess_strategy_,
                                                allreduce_strategy,
                                                ebc_config.comm_strategy_,
                                                compression_param};

  embedding::EmbeddingCollectionParam eval_ebc_param{num_table,
                                                     num_lookup,
                                                     lookup_params,
                                                     shard_matrix,
                                                     grouped_emb_params,
                                                     solver_.batchsize_eval,
                                                     key_type,
                                                     index_type,
                                                     offset_type,
                                                     emb_type,
                                                     wgrad_type,
                                                     input_layout_,
                                                     ebc_config.output_layout_,
                                                     ebc_config.sort_strategy_,
                                                     ebc_config.keys_preprocess_strategy_,
                                                     ebc_config.allreduce_strategy_,
                                                     ebc_config.comm_strategy_,
                                                     compression_param};

  std::vector<std::shared_ptr<core::CoreResourceManager>> core_list;

  for (int local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
    auto core_resource_manager =
        std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager_, local_gpu_id);
    core_list.push_back(core_resource_manager);
  }
  ebc_list_.push_back(std::make_unique<embedding::EmbeddingCollection>(
      resource_manager_, core_list, ebc_param, eval_ebc_param, emb_table_list, exchange_wgrad_));
  embedding_para_io_->add_embedding_collection((ebc_list_[ebc_list_.size() - 1]).get());

  auto prepare_ebc_input = [&](auto& sparse_input_map, bool is_longlong) {
    core23::DataType SparseType = is_longlong ? core23::DataType(core23::ScalarType::Int64)
                                              : core23::DataType(core23::ScalarType::UInt32);
    auto tensor_as_type = [&](core23::Tensor input, core23::DataType expected_type) {
      auto origin_type = input.data_type();
      HCTR_CHECK_HINT(origin_type.size() == expected_type.size(),
                      "Size not equal, cannot reinterpret type");
      return core23::Tensor::bind(input.data(), input.shape(), expected_type, input.device());
    };
    auto train_sparse_tensors = sparse_input_map[bottom_name].train_sparse_tensors;
    auto evaluate_sparse_tensors = sparse_input_map[bottom_name].evaluate_sparse_tensors;

    for (int local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
      CudaDeviceContext context(resource_manager_->get_local_gpu(local_gpu_id)->get_device_id());
      core23::Device device{core23::DeviceType::GPU,
                            static_cast<core23::DeviceIndex>(
                                resource_manager_->get_local_gpu(local_gpu_id)->get_device_id())};
      auto train_key_tensor =
          tensor_as_type(train_sparse_tensors[local_gpu_id].get_value_tensor(), SparseType);
      train_ebc_key_list_.push_back(train_key_tensor);

      auto train_bucket_range_tensor =
          tensor_as_type(train_sparse_tensors[local_gpu_id].get_rowoffset_tensor(), SparseType);
      train_ebc_bucket_range_list_.push_back(train_bucket_range_tensor);

      train_ebc_num_keys_list_.push_back(train_sparse_tensors[local_gpu_id].get_nnz_ptr().get());

      auto evaluate_key_tensor =
          tensor_as_type(evaluate_sparse_tensors[local_gpu_id].get_value_tensor(), SparseType);
      evaluate_ebc_key_list_.push_back(evaluate_key_tensor);

      auto evaluate_bucket_range_tensor =
          tensor_as_type(evaluate_sparse_tensors[local_gpu_id].get_rowoffset_tensor(), SparseType);
      evaluate_ebc_bucket_range_list_.push_back(evaluate_bucket_range_tensor);

      evaluate_ebc_num_keys_list_.push_back(
          evaluate_sparse_tensors[local_gpu_id].get_nnz_ptr().get());
    }
  };

  if (reader_params_.data_reader_type != DataReaderType_t::RawAsync) {
    if (solver_.i64_input_key) {
      prepare_ebc_input(sparse_input_map_64_, true);
    } else {
      prepare_ebc_input(sparse_input_map_32_, false);
    }
  }

  // activate_ebc_output_tensor
  size_t batch_size_per_gpu = solver_.batchsize / num_total_gpus;
  size_t eval_batch_size_per_gpu = solver_.batchsize_eval / num_total_gpus;
  if (ebc_param.output_layout_ == embedding::EmbeddingLayout::FeatureMajor) {
    std::vector<std::string> top_name_list;
    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      embedding::LookupParam& lookup_param = ebc_param.lookup_params[lookup_id];
      int emb_out_dims = (lookup_param.combiner == embedding::Combiner::Concat)
                             ? lookup_param.max_hotness * lookup_param.ev_size
                             : lookup_param.ev_size;

      std::string top_name = ebc_config.top_names_[lookup_id];
      top_name_list.push_back(top_name);

      activate_tensor(tensor_active_, top_name);
      tensor_shape_info_raw_.insert({top_name, {solver_.batchsize, 1, emb_out_dims}});
      embedding_dependent_tensors_.insert(top_name);
    }
    input_output_info_.push_back(std::make_pair(bottom_name, join(top_name_list, ",")));
    if (solver_.use_mixed_precision) {
      allocate_ebc_output_helper_for_feature_major<__half>(
          resource_manager_, batch_size_per_gpu, ebc_config, ebc_param, train_tensor_entities_list_,
          train_ebc_outptut_);
      allocate_ebc_output_helper_for_feature_major<__half>(
          resource_manager_, eval_batch_size_per_gpu, ebc_config, ebc_param,
          evaluate_tensor_entities_list_, evaluate_ebc_outptut_);
    } else {
      allocate_ebc_output_helper_for_feature_major<float>(
          resource_manager_, batch_size_per_gpu, ebc_config, ebc_param, train_tensor_entities_list_,
          train_ebc_outptut_);
      allocate_ebc_output_helper_for_feature_major<float>(
          resource_manager_, eval_batch_size_per_gpu, ebc_config, ebc_param,
          evaluate_tensor_entities_list_, evaluate_ebc_outptut_);
    }
  } else {
    int concate_out_dims = 0;
    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      embedding::LookupParam& lookup_param = ebc_param.lookup_params[lookup_id];

      int emb_out_dims = (lookup_param.combiner == embedding::Combiner::Concat)
                             ? lookup_param.max_hotness * lookup_param.ev_size
                             : lookup_param.ev_size;
      concate_out_dims += emb_out_dims;
    }

    activate_tensor(tensor_active_, ebc_config.batch_major_output_name_);
    tensor_shape_info_raw_.insert(
        {ebc_config.batch_major_output_name_, {solver_.batchsize, concate_out_dims}});
    input_output_info_.push_back(std::make_pair(bottom_name, ebc_config.batch_major_output_name_));
    embedding_dependent_tensors_.insert(ebc_config.batch_major_output_name_);

    // allocate output buffer
    if (solver_.use_mixed_precision) {
      allocate_ebc_output_helper_for_batch_major<__half>(
          resource_manager_, batch_size_per_gpu, ebc_config, ebc_param, train_tensor_entities_list_,
          train_ebc_outptut_);
      allocate_ebc_output_helper_for_batch_major<__half>(
          resource_manager_, eval_batch_size_per_gpu, ebc_config, ebc_param,
          evaluate_tensor_entities_list_, evaluate_ebc_outptut_);
    } else {
      allocate_ebc_output_helper_for_batch_major<float>(
          resource_manager_, batch_size_per_gpu, ebc_config, ebc_param, train_tensor_entities_list_,
          train_ebc_outptut_);
      allocate_ebc_output_helper_for_batch_major<float>(
          resource_manager_, eval_batch_size_per_gpu, ebc_config, ebc_param,
          evaluate_tensor_entities_list_, evaluate_ebc_outptut_);
    }
  }

  train_ddl_output_.clear();
  cache_train_ddl_output_.clear();
  evaluate_ddl_output_.clear();
  cache_evaluate_ddl_output_.clear();
  for (int local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
    train_ddl_output_.push_back(
        allocate_output_for_data_distributor(core_list[local_gpu_id], ebc_param));
    if (solver_.train_inter_iteration_overlap) {
      cache_train_ddl_output_.push_back(
          allocate_output_for_data_distributor(core_list[local_gpu_id], ebc_param));
    }
    evaluate_ddl_output_.push_back(
        allocate_output_for_data_distributor(core_list[local_gpu_id], eval_ebc_param));
    if (solver_.eval_inter_iteration_overlap) {
      cache_evaluate_ddl_output_.push_back(
          allocate_output_for_data_distributor(core_list[local_gpu_id], eval_ebc_param));
    }
  }

  // create data distributors
  train_data_distributor_ = std::make_shared<DataDistributor>(core_list, ebc_param, emb_table_list,
                                                              ebc_config.dr_lookup_ids_);
  eval_data_distributor_ = std::make_shared<DataDistributor>(
      core_list, eval_ebc_param, emb_table_list, ebc_config.dr_lookup_ids_);
}

void Model::pre_add_dense_layer(DenseLayer& dense_layer) {
  embedding_dependent_ = false;
  for (auto& bottom_name : dense_layer.bottom_names) {
    deactivate_tensor(tensor_active_, bottom_name);
    if (embedding_dependent_tensors_.find(bottom_name) != embedding_dependent_tensors_.end()) {
      embedding_dependent_ = true;
    }
  }
  for (auto& top_name : dense_layer.top_names) {
    activate_tensor(tensor_active_, top_name);
    if (embedding_dependent_) {
      embedding_dependent_tensors_.insert(top_name);
    }
  }
  std::string input_names = join(dense_layer.bottom_names, ",");
  std::string output_names = join(dense_layer.top_names, ",");
  input_output_info_.push_back(std::make_pair(input_names, output_names));
  if (solver_.use_mixed_precision) {
    layer_info_.push_back(LAYER_TYPE_TO_STRING_MP[dense_layer.layer_type]);
  } else {
    layer_info_.push_back(LAYER_TYPE_TO_STRING[dense_layer.layer_type]);
  }
}

void Model::graph_analysis() {
  HCTR_LOG(INFO, ROOT, "Graph analysis to resolve tensor dependency\n");
  std::map<std::string, unsigned int> tensor_usage;
  std::map<std::string, DenseLayer> tensor_slice_layer;
  std::map<std::string, unsigned int> tensor_slice_index;
  for (auto& dense_layer : dense_layer_params_raw_) {
    for (auto& bottom_name : dense_layer.bottom_names) {
      analyze_tensor(tensor_usage, bottom_name);
    }
  }
  for (auto iter = tensor_usage.begin(); iter != tensor_usage.end(); iter++) {
    if (iter->second > 5) {
      HCTR_OWN_THROW(Error_t::WrongInput, "The graph should not include more than 5-way branches");
    }
    if (iter->second > 1) {
      std::vector<std::string> bottom_names{iter->first};
      std::vector<std::string> top_names;
      std::vector<std::pair<int, int>> ranges;
      for (unsigned int i = 0; i < iter->second; i++) {
        top_names.push_back(iter->first + "_slice" + std::to_string(i));
        auto dims = tensor_shape_info_raw_[iter->first].size();
        ranges.emplace_back(std::make_pair(0, tensor_shape_info_raw_[iter->first][dims - 1]));
      }
      DenseLayer slice_layer(Layer_t::Slice, bottom_names, top_names);
      slice_layer.ranges = ranges;
      tensor_slice_layer.insert(std::pair<std::string, DenseLayer>(iter->first, slice_layer));
      tensor_slice_index.insert(std::pair<std::string, unsigned int>(iter->first, 0));
      HCTR_LOG(INFO, ROOT, "Add Slice layer for tensor: %s, creating %d copies\n",
               iter->first.c_str(), iter->second);
    }
  }
  for (auto& dense_layer : dense_layer_params_raw_) {
    bool flag = true;
    for (auto& bottom_name : dense_layer.bottom_names) {
      if (tensor_usage[bottom_name] > 1) {
        flag = false;
        break;
      }
    }
    if (flag) {
      dense_layer_params_.push_back(dense_layer);
    } else {
      DenseLayer new_dense_layer = dense_layer;
      for (unsigned int i = 0; i < new_dense_layer.bottom_names.size(); i++) {
        std::string old_bottom_name = new_dense_layer.bottom_names[i];
        if (tensor_slice_index.find(old_bottom_name) != tensor_slice_index.end()) {
          auto iter = tensor_slice_layer.find(old_bottom_name);
          if (tensor_slice_index[old_bottom_name] == 0) {
            dense_layer_params_.push_back(iter->second);
          }
          std::string new_bottom_name = iter->second.top_names[tensor_slice_index[old_bottom_name]];
          tensor_slice_index[old_bottom_name] += 1;
          new_dense_layer.bottom_names[i] = new_bottom_name;
        }
      }
      dense_layer_params_.push_back(new_dense_layer);
    }
  }
  add_dense_layers(dense_layer_params_);
}

// deep copy
void Model::create_copy_ops_for_network_input(const std::string& dense_name,
                                              const std::string& label_name, bool is_train) {
  auto& copy_ops = is_train ? graph_.train_copy_ops_ : graph_.evaluate_copy_ops_;
  auto& tensor_entries_list =
      is_train ? train_tensor_entities_list_ : evaluate_tensor_entities_list_;

  int num_local_gpus = resource_manager_->get_local_gpu_count();
  // copy ops for dense & label
  copy_ops.resize(2 * num_local_gpus);

  for (int id = 0; id < num_local_gpus; ++id) {
    core23::Device device(core23::DeviceType::GPU,
                          resource_manager_->get_local_gpu(id)->get_device_id());
    for (auto& tensor_entry : tensor_entries_list[id]) {
      if (tensor_entry.name == dense_name) {
        copy_ops[id].reset(
            new CopyOpImpl(resource_manager_->get_local_gpu(id), tensor_entry.tensor));
        tensor_entry.tensor = copy_ops[id]->get_tensorbag();
      } else if (tensor_entry.name == label_name) {
        copy_ops[id + num_local_gpus].reset(
            new CopyOpImpl(resource_manager_->get_local_gpu(id), tensor_entry.tensor));
        tensor_entry.tensor = copy_ops[id + num_local_gpus]->get_tensorbag();
      } else {
        HCTR_OWN_THROW(Error_t::WrongInput, "wrong tensor entry name when creating copy_op.");
      }
    }
  }
}

void Model::compile() {
  if (!graph_finalized_) {
    graph_analysis();
    graph_finalized_ = true;
  }
  if (data_input_info_.size() < 3 || layer_info_.size() < 2) {
    HCTR_OWN_THROW(Error_t::IllegalCall, "The model should include input and at least two layers");
  }
  HCTR_PRINT(INFO,
             "===================================================Model "
             "Compile===================================================\n");
  build_networks();

  // TODO: this is a WAR; need to find a way to remove the preallocation
  for (int local_gpu_id = 0; local_gpu_id < resource_manager_->get_local_gpu_count();
       ++local_gpu_id) {
    auto device_id = resource_manager_->get_local_gpu(local_gpu_id)->get_device_id();
    core23::Device device(core23::DeviceType::GPU, device_id);
    bool success = core23::AllocateBuffers(device);
    if (!success) {
      HCTR_LOG_S(DEBUG, ROOT) << "Nothing to preallocate" << std::endl;
    }
  }
  core23::Device device_h(core23::DeviceType::CPU);
  bool success = core23::AllocateBuffers(device_h);
  if (!success) {
    HCTR_LOG_S(DEBUG, ROOT) << "Nothing to preallocate" << std::endl;
  }
  initialize();
  create_metrics();
  create_pipelines();
}

void Model::update_label_weights(std::vector<std::string>& label_names,
                                 std::vector<float>& label_weights) {
  // Add implementation and support in next merge request
  if (label_names.size() != label_weights.size()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Must have the same number of label names and weights");
  }
  std::map<std::string, float>::iterator loss_lookup;
  for (size_t i = 0; i < label_names.size(); ++i) {
    loss_lookup = label_weights_.find(label_names[i]);
    if (loss_lookup == label_weights_.end()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Label name not found: " + label_names[i]);
    }
    loss_lookup->second = label_weights[i];
  }
}

void Model::compile(std::vector<std::string>& label_names, std::vector<float>& label_weights) {
  update_label_weights(label_names, label_weights);
  compile();
}

void Model::summary() {
  if (!graph_finalized_) {
    graph_analysis();
    graph_finalized_ = true;
  }
  if (data_input_info_.size() < 3 || layer_info_.size() < 2) {
    HCTR_OWN_THROW(Error_t::IllegalCall,
                   "The model should include input and at "
                   "least two layers");
  }
  for (auto tensor_entry : train_tensor_entities_list_[0]) {
    tensor_shape_info_.insert(std::make_pair(tensor_entry.name, tensor_entry.tensor.shape()));
  }
  HCTR_PRINT(INFO,
             "============================================"
             "=======Model "
             "Summary====================================="
             "==============\n");
  auto log = HCTR_LOG_S(INFO, ROOT);
  log << "Model structure on each GPU" << std::endl;
  log << std::left << std::setw(40) << std::setfill(' ') << "Label" << std::left << std::setw(30)
      << std::setfill(' ') << "Dense" << std::left << std::setw(30) << std::setfill(' ') << "Sparse"
      << std::endl;
  log << std::left << std::setw(40) << std::setfill(' ') << data_input_info_[0] << std::left
      << std::setw(30) << std::setfill(' ') << data_input_info_[1] << " " << std::left
      << std::setw(30) << std::setfill(' ') << data_input_info_[2] << std::endl;
  log << std::left << std::setw(40) << std::setfill(' ')
      << get_tensor_shape(data_input_info_[0], tensor_shape_info_) << std::left << std::setw(40)
      << std::setfill(' ') << get_tensor_shape(data_input_info_[1], tensor_shape_info_)
      << std::endl;
  log << "————————————————————————————————————————————————"
         "—————————————————————————————————"
         "—————————————————————————————————"
      << std::endl;
  log << std::left << std::setw(40) << std::setfill(' ') << "Layer Type" << std::left
      << std::setw(30) << std::setfill(' ') << "Input Name" << std::left << std::setw(30)
      << std::setfill(' ') << "Output Name" << std::left << std::setw(30) << std::setfill(' ')
      << "Output Shape" << std::endl;
  log << "————————————————————————————————————————————————"
         "—————————————————————————————————"
         "—————————————————————————————————"
      << std::endl;
  for (size_t i = 0; i < layer_info_.size(); ++i) {
    std::vector<std::string> layer_type{layer_info_[i]};
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    split(input_output_info_[i].first, ',', input_names);
    split(input_output_info_[i].second, ',', output_names);
    size_t lines =
        input_names.size() > output_names.size() ? input_names.size() : output_names.size();
    layer_type.insert(layer_type.end(), lines - 1, "");
    if (lines > input_names.size()) {
      input_names.insert(input_names.end(), lines - input_names.size(), "");
    }
    if (lines > output_names.size()) {
      output_names.insert(output_names.end(), lines - output_names.size(), "");
    }
    for (size_t j = 0; j < lines; j++) {
      log << std::left << std::setw(40) << std::setfill(' ') << layer_type[j] << std::left
          << std::setw(30) << std::setfill(' ') << input_names[j] << std::left << std::setw(30)
          << std::setfill(' ') << output_names[j] << std::left << std::setw(30) << std::setfill(' ')
          << get_tensor_shape(output_names[j], tensor_shape_info_) << std::endl;
    }
    log << "----------------------------------------------"
           "-----------------------------------"
           "---------------------------------"
        << std::endl;
  }
}

void Model::create_networks() {
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    networks_.emplace_back(new Network(resource_manager_->get_local_cpu(),
                                       resource_manager_->get_local_gpu(i),
                                       solver_.use_mixed_precision));
  }
  train_tensor_entities_list_.resize(resource_manager_->get_local_gpu_count());
  evaluate_tensor_entities_list_.resize(resource_manager_->get_local_gpu_count());
}

void Model::build_networks() {
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    networks_[i]->create_and_set_optimizer(opt_params_);
  }
  auto aligned_size = 16 * resource_manager_->get_local_gpu_count();
  core23::BufferParams bp{.channel = solver_.use_mixed_precision ? GetWgradHalfBufferChannel()
                                                                 : GetWgradBufferChannel()};
  for (int g = 0; g < resource_manager_->get_local_gpu_count(); g++) {
    auto device_id = resource_manager_->get_local_gpu(g)->get_device_id();
    core23::Device device(core23::DeviceType::GPU, device_id);
    auto wgrad_buffer = core23::GetBuffer(bp, device);
    auto wgrad_size = wgrad_buffer->reserved_size();
    size_t padded_bytes = wgrad_size % aligned_size;
    padded_bytes += aligned_size - padded_bytes;
    // alignment requirements from grouped allreduce.
    wgrad_tensor_successor_.emplace_back(core23::TensorParams()
                                             .device(device)
                                             .shape({static_cast<int64_t>(padded_bytes)})
                                             .data_type(core23::ScalarType::Char)
                                             .buffer_params(bp));
  }
  buff_allocated_ = true;
}

void Model::initialize() {
#ifndef DATA_READING_TEST

#pragma omp parallel num_threads(number_of_networks())
  {
    size_t id = omp_get_thread_num();
    networks_[id]->initialize();
    if (solver_.use_algorithm_search) {
      networks_[id]->search_algorithm();
    }
    HCTR_LIB_THROW(cudaStreamSynchronize(resource_manager_->get_local_gpu(id)->get_stream()));
  }

  int num_gpus = resource_manager_->get_local_gpu_count();
  std::vector<void*> wgrad_buffer_ptrs;
  size_t wgrad_buffer_size{};
  core23::BufferParams bp{.channel = solver_.use_mixed_precision ? GetWgradHalfBufferChannel()
                                                                 : GetWgradBufferChannel()};
  for (int g = 0; g < num_gpus; g++) {
    auto device_id = resource_manager_->get_local_gpu(g)->get_device_id();
    core23::Device device(core23::DeviceType::GPU, device_id);
    auto wgrad_buffer = core23::GetBuffer(bp, device);
    auto [ptr_, size_] = wgrad_buffer->decay();
    wgrad_buffer_size = size_;
    HCTR_CHECK_HINT(size_ && ptr_, "wgrad is null or it's a confederal buffer");
    wgrad_buffer_ptrs.push_back(ptr_);
  }
  exchange_wgrad_->init_ar_comm(wgrad_buffer_ptrs, wgrad_buffer_size);
#endif
  init_params_for_dense_();
  if (solver_.perf_logging) {
    for (size_t i = 0; i < dense_layer_params_.size(); i++) {
      bool is_trainable =
          TRAINABLE_LAYERS.find(dense_layer_params_[i].layer_type) != TRAINABLE_LAYERS.end();
      if (is_trainable) {
        std::string output_names = join(dense_layer_params_[i].top_names, "-");
        HCTR_LOG_ARGS(timer_log.elapsedMilliseconds(), "weights_initialization", output_names);
      }
    }
  }
  init_params_for_sparse_();
}
void Model::create_metrics() {
  int num_total_gpus = resource_manager_->get_global_gpu_count();
  int label_dim = input_params_[0].labels_.begin()->second;
  if (input_params_[0].labels_.size() > 1) {
    auto labs = input_params_[0].labels_;
    label_dim = std::accumulate(std::begin(labs), std::end(labs), 0,
                                [](const int previous, const std::pair<std::string, int>& p) {
                                  return previous + p.second;
                                });
  }

  auto num_metrics = [&]() { return networks_[0]->get_raw_metrics_all().size(); };
  for (const auto& metric : solver_.metrics_spec) {
    // Only AUC is currently supported for models with more than one loss layer
    if ((metric.first != metrics::Type::AUC) && num_metrics() > 1) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Metrics besides AUC are not supported for multi-task models.");
    }

    metrics_.emplace_back(std::move(metrics::Metric::Create(
        metric.first, solver_.use_mixed_precision, solver_.batchsize_eval / num_total_gpus,
        solver_.max_eval_batches, label_dim, resource_manager_)));
  }
}

void Model::create_pipelines() {
  // TODO: currently it is only for HE
  if (embeddings_.size() == 1) {
    auto lr_scheds = embeddings_[0]->get_learning_rate_schedulers();
    for (size_t i = 0; i < lr_scheds.size(); i++) {
      networks_[i]->set_learning_rate_scheduler(lr_scheds[i]);
    }
  }

  if (solver_.use_embedding_collection) {
    create_train_pipeline_with_ebc(networks_);
    create_evaluate_pipeline_with_ebc(networks_);
  } else {
    // will create pipeline for dense network.
    create_train_network_pipeline(networks_);
    create_eval_network_pipeline(networks_);
  }

  if (solver_.perf_logging) {
    HCTR_LOG_ARGS(timer_log.elapsedMilliseconds(), "init_stop");
    HCTR_LOG_ARGS(timer_log.elapsedMilliseconds(), "run_start");
  }

  if (solver_.perf_logging) {
    for (size_t i = 0; i < sparse_embedding_params_.size(); i++) {
      HCTR_LOG_ARGS(timer_log.elapsedMilliseconds(), "weights_initialization",
                    sparse_embedding_params_[i].sparse_embedding_name);
    }
  }

#ifdef ENABLE_MPI
  if (resource_manager_->get_num_process() > 1) {
    collective_manager_->set_ready_to_transfer();
  }
#endif
}

}  // namespace HugeCTR