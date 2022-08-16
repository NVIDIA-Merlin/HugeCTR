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
#include <data_readers/data_reader.hpp>
#include <loss.hpp>
#include <metrics.hpp>
#include <optimizer.hpp>
#include <parser.hpp>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {
// Create data reader for InferenceSession (internal use)
template <typename TypeKey>
void create_datareader<TypeKey>::operator()(
    const InferenceParams& inference_params, const InferenceParser& inference_parser,
    std::shared_ptr<IDataReader>& data_reader,
    const std::shared_ptr<ResourceManager> resource_manager,
    std::map<std::string, SparseInput<TypeKey>>& sparse_input_map,
    std::map<std::string, TensorBag2>& label_dense_map, const std::string& source,
    const DataReaderType_t data_reader_type, const Check_t check_type,
    const std::vector<long long>& slot_size_array, const bool repeat_dataset,
    const long long num_samples) {
  // TO DO：support multi-hot
  long long slot_sum = 0;
  std::vector<long long> slot_offset;
  for (auto slot_size : slot_size_array) {
    slot_offset.push_back(slot_sum);
    slot_sum += slot_size;
  }

  std::vector<DataReaderSparseParam> data_reader_sparse_param_array;
  for (size_t i = 0; i < inference_parser.slot_num_for_tables.size(); i++) {
    DataReaderSparseParam param;
    param.top_name = inference_parser.sparse_names[i];
    param.max_feature_num = inference_parser.max_feature_num_for_tables[i];
    param.slot_num = inference_parser.slot_num_for_tables[i];
    param.max_nnz = inference_parser.max_nnz_for_tables[i];
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
      data_reader_sparse_param_array, resource_manager, true, 1, false);
  data_reader.reset(data_reader_tk);

  switch (data_reader_type) {
    case DataReaderType_t::Norm: {
      bool start_right_now = repeat_dataset;
      data_reader->create_drwg_norm(source, check_type, start_right_now);
      break;
    }
    case DataReaderType_t::Raw: {
      data_reader->create_drwg_raw(source, num_samples, false, false, true);
      break;
    }
    case DataReaderType_t::Parquet: {
#ifdef DISABLE_CUDF
      HCTR_OWN_THROW(Error_t::WrongInput, "Parquet is not supported under DISABLE_CUDF");
#else
      data_reader->create_drwg_parquet(source, false, slot_offset, true);
      HCTR_LOG_S(INFO, ROOT) << "Vocabulary size: " << slot_sum << std::endl;
#endif
      break;
    }
    default: {
      assert(!"Error: no such option && should never get here!");
    }
  }

  label_dense_map.emplace(inference_parser.label_name, data_reader_tk->get_label_tensors()[0]);
  label_dense_map.emplace(inference_parser.dense_name, data_reader_tk->get_dense_tensors()[0]);

  for (unsigned int i = 0; i < inference_parser.sparse_names.size(); i++) {
    const std::string& sparse_name = inference_parser.sparse_names[i];
    const auto& sparse_input = sparse_input_map.find(sparse_name);

    auto copy = [](const std::vector<SparseTensorBag>& tensorbags,
                   SparseTensors<TypeKey>& sparse_tensors) {
      sparse_tensors.resize(tensorbags.size());
      for (size_t j = 0; j < tensorbags.size(); ++j) {
        sparse_tensors[j] = SparseTensor<TypeKey>::stretch_from(tensorbags[j]);
      }
    };
    copy(data_reader_tk->get_sparse_tensors(sparse_name),
         sparse_input->second.evaluate_sparse_tensors);
  }
}

// Create data reader for InferenceModel (multi-GPU offline inference use)
template <typename TypeKey>
void create_datareader<TypeKey>::operator()(
    const InferenceParams& inference_params, const InferenceParser& inference_parser,
    std::shared_ptr<IDataReader>& data_reader,
    const std::shared_ptr<ResourceManager> resource_manager,
    std::map<std::string, SparseInput<TypeKey>>& sparse_input_map,
    std::vector<TensorBag2>& label_tensor_list, std::vector<TensorBag2>& dense_tensor_list,
    const std::string& source, const DataReaderType_t data_reader_type, const Check_t check_type,
    const std::vector<long long>& slot_size_array, const bool repeat_dataset) {
  HCTR_CHECK_HINT(label_tensor_list.size() == 0,
                  "label tensor list should be empty before creating data reader");
  HCTR_CHECK_HINT(dense_tensor_list.size() == 0,
                  "dense tensor list should be empty before creating data reader");
  HCTR_CHECK_HINT(repeat_dataset, "repeat dataset should be true for inference");
  HCTR_LOG_S(INFO, ROOT) << "Create inference data reader on "
                         << resource_manager->get_local_gpu_count() << " GPU(s)" << std::endl;
  long long slot_sum = 0;
  std::vector<long long> slot_offset;
  for (auto slot_size : slot_size_array) {
    slot_offset.push_back(slot_sum);
    slot_sum += slot_size;
  }

  std::vector<DataReaderSparseParam> data_reader_sparse_param_array;
  for (size_t i = 0; i < inference_parser.slot_num_for_tables.size(); i++) {
    DataReaderSparseParam param;
    param.top_name = inference_parser.sparse_names[i];
    param.max_feature_num = inference_parser.max_feature_num_for_tables[i];
    param.slot_num = inference_parser.slot_num_for_tables[i];
    param.max_nnz = inference_parser.max_nnz_for_tables[i];
    data_reader_sparse_param_array.push_back(param);
  }

  for (unsigned int i = 0; i < inference_parser.sparse_names.size(); i++) {
    DataReaderSparseParam param = data_reader_sparse_param_array[i];
    std::string sparse_name = inference_parser.sparse_names[i];
    SparseInput<TypeKey> sparse_input(param.slot_num, param.max_feature_num);
    sparse_input_map.emplace(sparse_name, sparse_input);
  }

  const int num_workers =
      data_reader_type == DataReaderType_t::Parquet ? resource_manager->get_local_gpu_count() : 12;
  HCTR_LOG_S(INFO, ROOT) << "num of DataReader workers: " << num_workers << std::endl;

  DataReader<TypeKey>* data_reader_tk = new DataReader<TypeKey>(
      inference_params.max_batchsize, inference_parser.label_dim, inference_parser.dense_dim,
      data_reader_sparse_param_array, resource_manager, repeat_dataset, num_workers,
      false);  // use_mixed_precision = false
  data_reader.reset(data_reader_tk);

  switch (data_reader_type) {
    case DataReaderType_t::Norm: {
      bool start_right_now = repeat_dataset;
      data_reader->create_drwg_norm(source, check_type, start_right_now);
      break;
    }
    case DataReaderType_t::Parquet: {
#ifdef DISABLE_CUDF
      HCTR_OWN_THROW(Error_t::WrongInput, "Parquet is not supported under DISABLE_CUDF");
#else
      // read_file_sequentially = True, start_reading_from_beginning = True
      data_reader->create_drwg_parquet(source, true, slot_offset, true);
      HCTR_LOG_S(INFO, ROOT) << "Vocabulary size: " << slot_sum << std::endl;
#endif
      break;
    }
    default: {
      assert(!"Error: no such option && should never get here!");
    }
  }

  for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
    label_tensor_list.push_back(data_reader_tk->get_label_tensors()[i]);
    dense_tensor_list.push_back(data_reader_tk->get_dense_tensors()[i]);
  }

  for (unsigned int i = 0; i < inference_parser.sparse_names.size(); i++) {
    const std::string& sparse_name = inference_parser.sparse_names[i];
    const auto& sparse_input = sparse_input_map.find(sparse_name);

    auto copy = [](const std::vector<SparseTensorBag>& tensorbags,
                   SparseTensors<TypeKey>& sparse_tensors) {
      sparse_tensors.resize(tensorbags.size());
      for (size_t j = 0; j < tensorbags.size(); ++j) {
        sparse_tensors[j] = SparseTensor<TypeKey>::stretch_from(tensorbags[j]);
      }
    };
    copy(data_reader_tk->get_sparse_tensors(sparse_name),
         sparse_input->second.evaluate_sparse_tensors);
  }
}

template struct create_datareader<long long>;
template struct create_datareader<unsigned int>;

}  // namespace HugeCTR
