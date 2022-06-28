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
#include <pybind/model.hpp>

namespace HugeCTR {

Input get_input_from_json(const nlohmann::json& j_input) {
  auto type_name = get_value_from_json<std::string>(j_input, "type");
  if (type_name.compare("Data") != 0) {
    HCTR_OWN_THROW(Error_t::WrongInput, "the first layer is not Data layer:" + type_name);
  }
  auto j_label = get_json(j_input, "label");
  auto label_name = get_value_from_json<std::string>(j_label, "top");
  auto label_dim = get_value_from_json<int>(j_label, "label_dim");

  auto j_dense = get_json(j_input, "dense");
  auto dense_name = get_value_from_json<std::string>(j_dense, "top");
  auto dense_dim = get_value_from_json<int>(j_dense, "dense_dim");

  std::vector<DataReaderSparseParam> data_reader_sparse_param_array;
  auto j_sparse = get_json(j_input, "sparse");
  for (unsigned int i = 0; i < j_sparse.size(); i++) {
    const nlohmann::json& js = j_sparse[i];
    const auto sparse_name = get_value_from_json<std::string>(js, "top");
    bool is_fixed_length = get_value_from_json<int>(js, "is_fixed_length");
    int slot_num = get_value_from_json<int>(js, "slot_num");
    auto nnz_per_slot = get_json(js, "nnz_per_slot");
    std::vector<int> nnz_per_slot_vec;

    if (nnz_per_slot.is_array()) {
      if (nnz_per_slot.size() != static_cast<size_t>(slot_num)) {
        HCTR_OWN_THROW(Error_t::WrongInput, "nnz_per_slot.size() != slot_num");
      }
      for (int slot_id = 0; slot_id < slot_num; ++slot_id) {
        nnz_per_slot_vec.push_back(nnz_per_slot[slot_id].get<int>());
      }
    } else {
      // max nnz for each slot is the same
      int max_nnz = nnz_per_slot.get<int>();
      for (int slot_id = 0; slot_id < slot_num; ++slot_id) {
        nnz_per_slot_vec.push_back(max_nnz);
      }
    }
    DataReaderSparseParam param{sparse_name, nnz_per_slot_vec, is_fixed_length, slot_num};

    data_reader_sparse_param_array.push_back(param);
  }
  Input input = Input(label_dim, label_name, dense_dim, dense_name, data_reader_sparse_param_array);
  return input;
}

template <typename TypeKey>
void add_input(Input& input, DataReaderParams& reader_params,
               std::map<std::string, SparseInput<TypeKey>>& sparse_input_map,
               std::vector<std::vector<TensorEntry>>& train_tensor_entries_list,
               std::vector<std::vector<TensorEntry>>& evaluate_tensor_entries_list,
               std::shared_ptr<IDataReader>& train_data_reader,
               std::shared_ptr<IDataReader>& evaluate_data_reader,
               std::shared_ptr<IDataReader>& init_data_reader, size_t batch_size,
               size_t batch_size_eval, bool use_mixed_precision, bool repeat_dataset,
               bool enable_overlap, size_t num_iterations_statistics,
               const std::shared_ptr<ResourceManager> resource_manager) {
  DataReaderType_t format = reader_params.data_reader_type;
  Check_t check_type = reader_params.check_type;
  std::string source_data = reader_params.source[0];
  std::string eval_source = reader_params.eval_source;
  long long num_samples = reader_params.num_samples;
  long long eval_num_samples = reader_params.eval_num_samples;
  bool float_label_dense = reader_params.float_label_dense;
  // TODO - changes structures to support multiple labels
  std::string top_strs_dense = input.dense_name;
  int dense_dim = input.dense_dim;

  std::string top_strs_label = input.labels_.begin()->first;
  int total_label_dim = std::accumulate(
      std::begin(input.labels_), std::end(input.labels_), 0,
      [](const int previous, const std::pair<std::string, int>& p) { return previous + p.second; });

  if (input.labels_.size() > 1) {
    top_strs_label = "combined_multi_label";
  }

  for (unsigned int i = 0; i < input.data_reader_sparse_param_array.size(); i++) {
    DataReaderSparseParam param = input.data_reader_sparse_param_array[i];
    std::string sparse_name = param.top_name;
    SparseInput<TypeKey> sparse_input(param.slot_num, param.max_feature_num);
    sparse_input_map.emplace(sparse_name, sparse_input);
  }

  if ((format == DataReaderType_t::RawAsync)) {
    if (!repeat_dataset) {
      HCTR_OWN_THROW(
          Error_t::WrongInput,
          "Epoch mode cannot be used with RawAsync reader, please set repeat_dataset as true");
    }
    Alignment_t aligned_type = reader_params.async_param.aligned_type;
    int num_threads = reader_params.async_param.num_threads;
    int num_batches_per_thread = reader_params.async_param.num_batches_per_thread;
    int io_block_size = reader_params.async_param.io_block_size;
    int io_depth = reader_params.async_param.io_depth;
    int io_alignment = reader_params.async_param.io_alignment;
    bool shuffle = reader_params.async_param.shuffle;

    HCTR_LOG_S(INFO, ROOT) << "AsyncReader: num_threads = " << num_threads << std::endl;
    HCTR_LOG_S(INFO, ROOT) << "AsyncReader: num_batches_per_thread = " << num_batches_per_thread
                           << std::endl;
    HCTR_LOG_S(INFO, ROOT) << "AsyncReader: io_block_size = " << io_block_size << std::endl;
    HCTR_LOG_S(INFO, ROOT) << "AsyncReader: io_depth = " << io_depth << std::endl;
    HCTR_LOG_S(INFO, ROOT) << "AsyncReader: io_alignment = " << io_alignment << std::endl;
    HCTR_LOG_S(INFO, ROOT) << "AsyncReader: shuffle = " << (shuffle ? "ON" : "OFF") << std::endl;
    HCTR_LOG_S(INFO, ROOT) << "AsyncReader: num_iterations_statistics = "
                           << num_iterations_statistics << std::endl;

    train_data_reader.reset(new AsyncReader<TypeKey>(
        source_data, batch_size, total_label_dim, dense_dim, input.data_reader_sparse_param_array,
        use_mixed_precision, resource_manager, num_threads, num_batches_per_thread, io_block_size,
        io_depth, io_alignment, shuffle, enable_overlap, aligned_type));

    // If we want to cache eval, make sure we have enough buffers
    auto eval_num_batches_per_thread = num_batches_per_thread;
    int cache_eval_data = reader_params.cache_eval_data;
    if (cache_eval_data > num_threads * num_batches_per_thread) {
      eval_num_batches_per_thread = (cache_eval_data + num_threads - 1) / num_threads;
      HCTR_LOG_S(INFO, ROOT) << "AsyncReader: eval reader increased batches per thread to "
                             << eval_num_batches_per_thread << " to accommodate for the caching"
                             << std::endl;
    }
    // Small IO block may lead to too many AIO requests which hang,
    // so use a larger one for eval and init which are typically larger than train
    evaluate_data_reader.reset(new AsyncReader<TypeKey>(
        eval_source, batch_size_eval, total_label_dim, dense_dim,
        input.data_reader_sparse_param_array, use_mixed_precision, resource_manager, num_threads,
        eval_num_batches_per_thread, io_block_size * 8, io_depth, io_alignment, false, false,
        aligned_type));

    init_data_reader.reset(new AsyncReader<TypeKey>(
        source_data, num_iterations_statistics * batch_size, total_label_dim, dense_dim,
        input.data_reader_sparse_param_array, use_mixed_precision, resource_manager, 1, 1,
        io_block_size * 8, 4, io_alignment, false, false, aligned_type));

    auto train_data_reader_as = std::dynamic_pointer_cast<AsyncReader<TypeKey>>(train_data_reader);
    auto evaluate_data_reader_as =
        std::dynamic_pointer_cast<AsyncReader<TypeKey>>(evaluate_data_reader);

    for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
      train_tensor_entries_list[i].push_back(
          {top_strs_label, train_data_reader_as->get_label_tensors()[i]});
      evaluate_tensor_entries_list[i].push_back(
          {top_strs_label, evaluate_data_reader_as->get_label_tensors()[i]});

      if (use_mixed_precision) {
        train_tensor_entries_list[i].push_back(
            {top_strs_dense, train_data_reader_as->get_dense_tensors()[i]});
        evaluate_tensor_entries_list[i].push_back(
            {top_strs_dense, evaluate_data_reader_as->get_dense_tensors()[i]});
      } else {
        train_tensor_entries_list[i].push_back(
            {top_strs_dense, train_data_reader_as->get_dense_tensors()[i]});
        evaluate_tensor_entries_list[i].push_back(
            {top_strs_dense, evaluate_data_reader_as->get_dense_tensors()[i]});
      }
    }
    if (input.data_reader_sparse_param_array.size() > 1) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Only one sparse input is supported.");
    }
    const auto& sparse_input =
        sparse_input_map.find(input.data_reader_sparse_param_array[0].top_name);
    sparse_input->second.train_sparse_tensors = train_data_reader_as->get_value_tensors();
    sparse_input->second.evaluate_sparse_tensors = evaluate_data_reader_as->get_value_tensors();
    return;

  } else {
    int num_workers_train = reader_params.num_workers;
    int num_workers_eval = reader_params.num_workers;
    int local_gpu_count = resource_manager->get_local_gpu_count();

    if (format == DataReaderType_t::Parquet) {
      // if parallelism granularity is file, num_files should be greater than num of workers
      if (!reader_params.read_file_sequentially) {
        {
          std::ifstream read_stream(eval_source, std::ifstream::in);
          if (!read_stream.is_open()) {
            HCTR_OWN_THROW(Error_t::FileCannotOpen, "file list open failed: " + eval_source);
          }
          std::string buff;
          std::getline(read_stream, buff);
          int num_of_files = std::stoi(buff);
          read_stream.close();
          num_workers_eval = std::min(num_workers_eval, num_of_files);
        }
        std::vector<std::string> train_sources = reader_params.source;
        int min_num_files = 0;
        // there may exist multiple training sources 
        for (const auto &file_list_name : train_sources) {
          std::ifstream read_stream(file_list_name, std::ifstream::in);
          if (!read_stream.is_open()) {
            HCTR_OWN_THROW(Error_t::FileCannotOpen, "file list open failed: " + file_list_name);
          }

          std::string buff;
          std::getline(read_stream, buff);
          int num_of_files = std::stoi(buff);
          if (!min_num_files || num_of_files < min_num_files) min_num_files = num_of_files;
          read_stream.close();
        }
        num_workers_train = std::min(num_workers_train, min_num_files);
      }
      num_workers_train = std::min(local_gpu_count, num_workers_train);
      num_workers_eval = std::min(local_gpu_count, num_workers_eval);
    }

    HCTR_LOG_S(INFO, ROOT) << "num of DataReader workers for train: " << num_workers_train << std::endl;
    HCTR_LOG_S(INFO, ROOT) << "num of DataReader workers for eval: " << num_workers_eval << std::endl;

    DataReader<TypeKey>* data_reader_tk = new DataReader<TypeKey>(
        batch_size, total_label_dim, dense_dim, input.data_reader_sparse_param_array,
        resource_manager, repeat_dataset, num_workers_train, use_mixed_precision);
    train_data_reader.reset(data_reader_tk);
    DataReader<TypeKey>* data_reader_eval_tk = new DataReader<TypeKey>(
        batch_size_eval, total_label_dim, dense_dim, input.data_reader_sparse_param_array,
        resource_manager, repeat_dataset, num_workers_eval, use_mixed_precision);
    evaluate_data_reader.reset(data_reader_eval_tk);

    long long slot_sum = 0;
    std::vector<long long> slot_offset;
    for (auto slot_size : reader_params.slot_size_array) {
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
        train_data_reader->create_drwg_raw(source_data, num_samples, float_label_dense, true,
                                           false);
        evaluate_data_reader->create_drwg_raw(eval_source, eval_num_samples, float_label_dense,
                                              false, false);
        break;
      }
      case DataReaderType_t::Parquet: {
#ifdef DISABLE_CUDF
        HCTR_OWN_THROW(Error_t::WrongInput, "Parquet is not supported under DISABLE_CUDF");
#else
        train_data_reader->create_drwg_parquet(source_data, reader_params.read_file_sequentially, slot_offset, repeat_dataset);
        evaluate_data_reader->create_drwg_parquet(eval_source, reader_params.read_file_sequentially, slot_offset, repeat_dataset);
        HCTR_LOG_S(INFO, ROOT) << "Vocabulary size: " << slot_sum << std::endl;
#endif
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

    for (unsigned int i = 0; i < input.data_reader_sparse_param_array.size(); i++) {
      auto& top_name = input.data_reader_sparse_param_array[i].top_name;
      const auto& sparse_input = sparse_input_map.find(top_name);

      auto copy = [](const std::vector<SparseTensorBag>& tensorbags,
                     SparseTensors<TypeKey>& sparse_tensors) {
        sparse_tensors.resize(tensorbags.size());
        for (size_t j = 0; j < tensorbags.size(); ++j) {
          sparse_tensors[j] = SparseTensor<TypeKey>::stretch_from(tensorbags[j]);
        }
      };
      copy(data_reader_tk->get_sparse_tensors(top_name), sparse_input->second.train_sparse_tensors);
      copy(data_reader_eval_tk->get_sparse_tensors(top_name),
           sparse_input->second.evaluate_sparse_tensors);
    }
  }  // end of else. not AsynRaw Reader
}

template void add_input<long long>(Input&, DataReaderParams&,
                                   std::map<std::string, SparseInput<long long>>&,
                                   std::vector<std::vector<TensorEntry>>&,
                                   std::vector<std::vector<TensorEntry>>&,
                                   std::shared_ptr<IDataReader>&, std::shared_ptr<IDataReader>&,
                                   std::shared_ptr<IDataReader>&, size_t, size_t, bool, bool, bool,
                                   size_t, const std::shared_ptr<ResourceManager>);
template void add_input<unsigned int>(Input&, DataReaderParams&,
                                      std::map<std::string, SparseInput<unsigned int>>&,
                                      std::vector<std::vector<TensorEntry>>&,
                                      std::vector<std::vector<TensorEntry>>&,
                                      std::shared_ptr<IDataReader>&, std::shared_ptr<IDataReader>&,
                                      std::shared_ptr<IDataReader>&, size_t, size_t, bool, bool,
                                      bool, size_t, const std::shared_ptr<ResourceManager>);

}  // namespace HugeCTR
