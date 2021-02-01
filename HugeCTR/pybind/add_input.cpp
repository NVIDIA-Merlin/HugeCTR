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

#include <HugeCTR/include/data_readers/data_reader.hpp>
#include <HugeCTR/pybind/model.hpp>

namespace HugeCTR {

template <typename TypeKey>
void add_input(Input& input,
            std::map<std::string, SparseInput<TypeKey>>& sparse_input_map,
            std::vector<std::vector<TensorEntry>>& train_tensor_entries_list,
            std::vector<std::vector<TensorEntry>>& evaluate_tensor_entries_list,
            std::shared_ptr<IDataReader>& train_data_reader,
            std::shared_ptr<IDataReader>& evaluate_data_reader, size_t batch_size,
            size_t batch_size_eval, bool use_mixed_precision, bool repeat_dataset,
            const std::shared_ptr<ResourceManager> resource_manager) {

  DataReaderType_t format = input.data_reader_type;
  Check_t check_type = input.check_type;
  int cache_eval_data = input.cache_eval_data;
  std::string source_data = input.source;
  std::string eval_source =  input.eval_source;
  std::string top_strs_label = input.label_name;
  int label_dim = input.label_dim;
  std::string top_strs_dense = input.dense_name;
  int dense_dim = input.dense_dim;
  long long num_samples = input.num_samples;
  long long eval_num_samples = input.eval_num_samples;
  bool float_label_dense = input.float_label_dense;

#ifdef VAL
  const int num_workers = 1;
#else
  const int num_workers = format == DataReaderType_t::Parquet ? resource_manager->get_local_gpu_count() : input.num_workers;
#endif
  MESSAGE_("num of DataReader workers: " + std::to_string(num_workers));

  for (unsigned int i = 0; i < input.sparse_names.size(); i++) {
    DataReaderSparseParam param = input.data_reader_sparse_param_array[i];
    std::string sparse_name = input.sparse_names[i];
    SparseInput<TypeKey> sparse_input(param.slot_num, param.max_feature_num);
    sparse_input_map.emplace(sparse_name, sparse_input);
  }

  DataReader<TypeKey>* data_reader_tk = new DataReader<TypeKey>(
      batch_size, label_dim, dense_dim, input.data_reader_sparse_param_array, resource_manager,
      repeat_dataset, num_workers, use_mixed_precision, false);
  train_data_reader.reset(data_reader_tk);
  DataReader<TypeKey>* data_reader_eval_tk = new DataReader<TypeKey>(
      batch_size_eval, label_dim, dense_dim, input.data_reader_sparse_param_array, resource_manager,
      repeat_dataset, num_workers, use_mixed_precision, cache_eval_data);
  evaluate_data_reader.reset(data_reader_eval_tk);

  long long slot_sum = 0;
  std::vector<long long> slot_offset;
  for (auto slot_size:input.slot_size_array) {
    slot_offset.push_back(slot_sum);
    slot_sum += slot_size;
  }

  switch (format) {
    case DataReaderType_t::Norm: {
      bool start_right_now = repeat_dataset;
      train_data_reader->create_drwg_norm(source_data, check_type, start_right_now);
      evaluate_data_reader->create_drwg_norm(eval_source, check_type, start_right_now);
      break;
    }
    case DataReaderType_t::Raw: {
      train_data_reader->create_drwg_raw(source_data, num_samples, slot_offset, float_label_dense,
                                         true, false);
      evaluate_data_reader->create_drwg_raw(eval_source, eval_num_samples, slot_offset,
                                            float_label_dense, false, false);
      MESSAGE_("Vocabulary size: " + std::to_string(slot_sum));
      break;
    }
    case DataReaderType_t::Parquet: {
      train_data_reader->create_drwg_parquet(source_data, slot_offset, true);
      evaluate_data_reader->create_drwg_parquet(eval_source, slot_offset, true);
      MESSAGE_("Vocabulary size: " + std::to_string(slot_sum));
      break;
    }
    default: {
      assert(!"Error: no such option && should never get here!");
    }
  }

  for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
    train_tensor_entries_list[i].push_back(
        {top_strs_label, data_reader_tk->get_label_tensors()[i].shrink()});
    evaluate_tensor_entries_list[i].push_back(
        {top_strs_label, data_reader_eval_tk->get_label_tensors()[i].shrink()});

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

  for (unsigned int i = 0; i < input.sparse_names.size(); i++) {
    const auto& sparse_input = sparse_input_map.find(input.sparse_names[i]);
    sparse_input->second.train_row_offsets = data_reader_tk->get_row_offsets_tensors(i);
    sparse_input->second.train_values = data_reader_tk->get_value_tensors(i);
    sparse_input->second.train_nnz = data_reader_tk->get_nnz_array(i);
    sparse_input->second.evaluate_row_offsets = data_reader_eval_tk->get_row_offsets_tensors(i);
    sparse_input->second.evaluate_values = data_reader_eval_tk->get_value_tensors(i);
    sparse_input->second.evaluate_nnz = data_reader_eval_tk->get_nnz_array(i);
  }
}


template void add_input<long long>(Input&,
            std::map<std::string, SparseInput<long long>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::shared_ptr<IDataReader>&,
            std::shared_ptr<IDataReader>&, size_t,
            size_t, bool, bool,
            const std::shared_ptr<ResourceManager>);
template void add_input<unsigned int>(Input&,
            std::map<std::string, SparseInput<unsigned int>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::shared_ptr<IDataReader>&,
            std::shared_ptr<IDataReader>&, size_t,
            size_t, bool, bool,
            const std::shared_ptr<ResourceManager>);

} // namespace HugeCTR