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

#include <data_readers/data_reader.hpp>
#include <loss.hpp>
#include <metrics.hpp>
#include <optimizer.hpp>
#include <parser.hpp>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {
template <typename TypeKey>
void create_datareader<TypeKey>::operator()(
    const nlohmann::json& j, std::map<std::string, SparseInput<TypeKey>>& sparse_input_map,
    std::vector<TensorEntry>* train_tensor_entries_list, std::vector<TensorEntry>* evaluate_tensor_entries_list,std::shared_ptr<IDataReader>& train_data_reader,
    std::shared_ptr<IDataReader>& evaluate_data_reader, size_t batch_size, size_t batch_size_eval,
    bool use_mixed_precision, bool repeat_dataset_,
    const std::shared_ptr<ResourceManager> resource_manager) {
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

#ifdef VAL
  const int num_workers = 1;
#else
  int num_workers_default =
      format == DataReaderType_t::Parquet ? resource_manager->get_local_gpu_count() : 2 * resource_manager->get_local_gpu_count();
  const int num_workers = get_value_from_json_soft<int>(j, "num_workers", num_workers_default);
#endif
  MESSAGE_("num of DataReader workers: " + std::to_string(num_workers));

  std::vector<DataReaderSparseParam> data_reader_sparse_param_array;

  auto j_sparse = get_json(j, "sparse");
  std::vector<std::string> sparse_names;

  for (unsigned int i = 0; i < j_sparse.size(); i++) {
    const nlohmann::json& js = j_sparse[i];
    const auto sparse_name = get_value_from_json<std::string>(js, "top");
    bool is_fixed_length = get_value_from_json<int>(js, "is_fixed_length");
    int slot_num = get_value_from_json<int>(js, "slot_num");
    auto nnz_per_slot = get_json(js, "nnz_per_slot");
    std::vector<int> nnz_per_slot_vec;

    if(nnz_per_slot.is_array()) {
      if(nnz_per_slot.size() != static_cast<size_t>(slot_num)) {
        CK_THROW_(Error_t::WrongInput, "nnz_per_slot.size() != slot_num");
      }
      for(int slot_id = 0; slot_id < slot_num; ++slot_id){
        nnz_per_slot_vec.push_back(nnz_per_slot[slot_id].get<int>());
      }  
    }else {
      // max nnz for each slot is the same
      int max_nnz = nnz_per_slot.get<int>();
      for(int slot_id = 0; slot_id < slot_num; ++slot_id){
        nnz_per_slot_vec.push_back(max_nnz);
      }
    }
    DataReaderSparseParam param{sparse_name, nnz_per_slot_vec, is_fixed_length, slot_num};
    
    // DataReaderSparseParam param;

    // const nlohmann::json& js = j_sparse[i];
    // const auto sparse_name = get_value_from_json<std::string>(js, "top");
    // const auto data_type_name = get_value_from_json<std::string>(js, "type");
    // if (!find_item_in_map(param.type, data_type_name, DATA_TYPE_MAP)) {
    //   CK_THROW_(Error_t::WrongInput, "Not supported data type: " + data_type_name);
    // }
    // param.max_feature_num = get_value_from_json<int>(js, "max_feature_num_per_sample");
    // param.max_nnz = get_value_from_json_soft<int>(js, "max_nnz", param.max_feature_num);
    // param.slot_num = get_value_from_json<int>(js, "slot_num");
    data_reader_sparse_param_array.push_back(param);
    SparseInput<TypeKey> sparse_input(param.slot_num, param.max_feature_num);
    sparse_input_map.emplace(sparse_name, sparse_input);
    sparse_names.push_back(sparse_name);
  }

  evaluate_data_reader = nullptr;
  std::string eval_source;
  FIND_AND_ASSIGN_STRING_KEY(eval_source, j);

  DataReader<TypeKey>* data_reader_tk = new DataReader<TypeKey>(
      batch_size, label_dim, dense_dim, data_reader_sparse_param_array, resource_manager,
      repeat_dataset_, num_workers, use_mixed_precision);
  train_data_reader.reset(data_reader_tk);
  DataReader<TypeKey>* data_reader_eval_tk = new DataReader<TypeKey>(
      batch_size_eval, label_dim, dense_dim, data_reader_sparse_param_array, resource_manager,
      repeat_dataset_, num_workers, use_mixed_precision);
  evaluate_data_reader.reset(data_reader_eval_tk);
  
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
      bool start_right_now = repeat_dataset_;
      train_data_reader->create_drwg_norm(source_data, check_type, start_right_now);
      evaluate_data_reader->create_drwg_norm(eval_source, check_type, start_right_now);
      break;
    }
    case DataReaderType_t::Raw: {
      const auto num_samples = get_value_from_json<long long>(j, "num_samples");
      const auto eval_num_samples = get_value_from_json<long long>(j, "eval_num_samples");
      bool float_label_dense = get_value_from_json_soft<bool>(j, "float_label_dense", false);
      train_data_reader->create_drwg_raw(source_data, num_samples, float_label_dense,
                                         true, false);
      evaluate_data_reader->create_drwg_raw(eval_source, eval_num_samples,
                                            float_label_dense, false, false);

      break;
    }
    case DataReaderType_t::Parquet: {
      
      // @Future: Should be slot_offset here and data_reader ctor should
      // be TypeKey not long long
      std::vector<long long> slot_offset = f();
      train_data_reader->create_drwg_parquet(source_data,slot_offset, true);
      evaluate_data_reader->create_drwg_parquet(eval_source,slot_offset, true);
      break;
    }
    default: {
      assert(!"Error: no such option && should never get here!");
    }
  }

  for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
    train_tensor_entries_list[i].push_back(
        {top_strs_label, data_reader_tk->get_label_tensors()[i]});
    evaluate_tensor_entries_list[i].push_back(
        {top_strs_label, data_reader_eval_tk->get_label_tensors()[i]});

    if (use_mixed_precision) {
      train_tensor_entries_list[i].push_back(
          {top_strs_dense, data_reader_tk->get_dense_tensors()[i]});
      evaluate_tensor_entries_list[i].push_back(
          {top_strs_dense, data_reader_eval_tk->get_dense_tensors()[i]});
    } else {
      train_tensor_entries_list[i].push_back(
          {top_strs_dense, data_reader_tk->get_dense_tensors()[i]});
      evaluate_tensor_entries_list[i].push_back(
          {top_strs_dense, data_reader_eval_tk->get_dense_tensors()[i]});
    }
  }

  for (unsigned int i = 0; i < j_sparse.size(); i++) {
    const std::string &sparse_name = sparse_names[i];
    const auto& sparse_input = sparse_input_map.find(sparse_name);

    auto copy = [](const std::vector<SparseTensorBag> &tensorbags, SparseTensors<TypeKey> &sparse_tensors) {
      sparse_tensors.resize(tensorbags.size());
      for(size_t j = 0; j < tensorbags.size(); ++j){
        sparse_tensors[j] = SparseTensor<TypeKey>::stretch_from(tensorbags[j]);
      }
    };
    copy(data_reader_tk->get_sparse_tensors(sparse_name), sparse_input->second.train_sparse_tensors);
    copy(data_reader_eval_tk->get_sparse_tensors(sparse_name), sparse_input->second.evaluate_sparse_tensors);
  }
}

template <typename TypeKey>
void create_datareader<TypeKey>::operator()(const InferenceParams& inference_params,
                  const InferenceParser& inference_parser,
                  std::shared_ptr<IDataReader>& data_reader,
                  const std::shared_ptr<ResourceManager> resource_manager,
                  std::map<std::string, SparseInput<TypeKey>>& sparse_input_map,
                  std::map<std::string, TensorBag2>& label_dense_map,
                  const std::string& source, const DataReaderType_t data_reader_type,
                  const Check_t check_type, const std::vector<long long>& slot_size_array,
                  const bool repeat_dataset) {
  // TO DO：support multi-hot
  long long slot_sum = 0;
  std::vector<long long> slot_offset;
  for (auto slot_size:slot_size_array) {
    slot_offset.push_back(slot_sum);
    slot_sum += slot_size;
  }

  std::vector<DataReaderSparseParam> data_reader_sparse_param_array;
  for (size_t i = 0; i < inference_parser.slot_num_for_tables.size(); i++) {
    DataReaderSparseParam param;
    param.top_name = inference_parser.sparse_names[i];
    param.max_feature_num = inference_parser.max_feature_num_for_tables[i];
    param.slot_num = inference_parser.slot_num_for_tables[i];
    param.max_nnz = 1;
    data_reader_sparse_param_array.push_back(param);
  }

  for (unsigned int i = 0; i < inference_parser.sparse_names.size(); i++) {
    DataReaderSparseParam param = data_reader_sparse_param_array[i];
    std::string sparse_name = inference_parser.sparse_names[i];
    SparseInput<TypeKey> sparse_input(param.slot_num, param.max_feature_num);
    sparse_input_map.emplace(sparse_name, sparse_input);
  }

  DataReader<TypeKey>* data_reader_tk = new DataReader<TypeKey>(
      inference_params.max_batchsize, inference_parser.label_dim, inference_parser.dense_dim, 
      data_reader_sparse_param_array, resource_manager,
      true, 1, false);
  data_reader.reset(data_reader_tk);
  
  switch (data_reader_type) {
    case DataReaderType_t::Norm: {
      bool start_right_now = repeat_dataset;
      data_reader->create_drwg_norm(source, check_type, start_right_now);
      break;
    }
    case DataReaderType_t::Parquet: {
      data_reader->create_drwg_parquet(source,slot_offset, true);
      MESSAGE_("Vocabulary size: " + std::to_string(slot_sum));
      break;
    }
    default: {
      assert(!"Error: no such option && should never get here!");
    }
  }
  
  label_dense_map.emplace(inference_parser.label_name, data_reader_tk->get_label_tensors()[0]);
  label_dense_map.emplace(inference_parser.dense_name, data_reader_tk->get_dense_tensors()[0]);

  for (unsigned int i = 0; i < inference_parser.sparse_names.size(); i++) {
    const std::string &sparse_name = inference_parser.sparse_names[i];
    const auto& sparse_input = sparse_input_map.find(sparse_name);

    auto copy = [](const std::vector<SparseTensorBag> &tensorbags, SparseTensors<TypeKey> &sparse_tensors) {
      sparse_tensors.resize(tensorbags.size());
      for(size_t j = 0; j < tensorbags.size(); ++j){
        sparse_tensors[j] = SparseTensor<TypeKey>::stretch_from(tensorbags[j]);
      }
    };
    copy(data_reader_tk->get_sparse_tensors(sparse_name), sparse_input->second.evaluate_sparse_tensors);
  }
}

template struct create_datareader<long long>;
template struct create_datareader<unsigned int>;

}  // namespace HugeCTR
