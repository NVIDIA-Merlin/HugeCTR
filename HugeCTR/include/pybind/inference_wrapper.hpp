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
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <hps/embedding_cache.hpp>
#include <hps/hier_parameter_server.hpp>
#include <inference/inference_session.hpp>
#include <pybind/inference_model.hpp>
#include <utils.hpp>

namespace HugeCTR {

namespace python_lib {

/**
 * @brief Main InferenceSessionPy class
 *
 * This is a class supporting HugeCTR inference in Python, which includes predict32 and
 predict64.
 * To support dynamic batch size during inference, this class need to be modified in the future.
 */
class InferenceSessionPy : public InferenceSession {
 public:
  InferenceSessionPy(const std::string& model_config_path, const InferenceParams& inference_params,
                     const std::shared_ptr<EmbeddingCacheBase>& embedding_cache);
  InferenceSessionPy(const std::string& model_config_path, const InferenceParams& inference_params,
                     const std::shared_ptr<EmbeddingCacheBase>& embedding_cache,
                     const std::shared_ptr<HierParameterServerBase>& parameter_server);
  ~InferenceSessionPy();

  float evaluate(size_t num_batches, const std::string& source, DataReaderType_t data_reader_type,
                 Check_t check_type, const std::vector<long long>& slot_size_array);
  pybind11::array_t<float> predict(size_t num_batches, const std::string& source,
                                   DataReaderType_t data_reader_type, Check_t check_type,
                                   const std::vector<long long>& slot_size_array);
  std::vector<float>& predict(const std::vector<float>& dense,
                              const std::vector<long long>& embeddingcolumns,
                              const std::vector<int>& row_ptrs);
  void refresh_embedding_cache();

 private:
  void initialize();

  template <typename TypeKey>
  void load_data(const std::string& source, DataReaderType_t data_reader_type, Check_t check_type,
                 const std::vector<long long>& slot_size_array);

  template <typename TypeKey>
  float evaluate_(size_t num_batches, const std::string& source, DataReaderType_t data_reader_type,
                  Check_t check_type, const std::vector<long long>& slot_size_array);

  template <typename TypeKey>
  pybind11::array_t<float> predict_(size_t num_batches, const std::string& source,
                                    DataReaderType_t data_reader_type, Check_t check_type,
                                    const std::vector<long long>& slot_size_array);

  template <typename TypeKey>
  void predict_(const std::vector<float>& dense, const std::vector<TypeKey>& embeddingcolumns,
                const std::vector<int>& row_ptrs);

  std::vector<unsigned int> embeddingcolumns_u32_;
  std::vector<float> output_;

  // Allocate in this class
  float* d_dense_;
  void* h_embeddingcolumns_;
  int* d_row_ptrs_;
  float* d_output_;

  // Derived from data reader output tensor
  float* d_reader_dense_;
  std::vector<void*> d_reader_keys_list_;
  std::vector<void*> d_reader_row_ptrs_list_;
  TensorBag2 label_tensor_;

  // Parameter Server
  std::shared_ptr<HierParameterServerBase> parameter_server_;
};

void InferenceSessionPy::initialize() {
  CudaDeviceContext context(resource_manager_->get_local_gpu(0)->get_device_id());
  HCTR_LIB_THROW(cudaMalloc((void**)&d_dense_, inference_params_.max_batchsize *
                                                   inference_parser_.dense_dim * sizeof(float)));
  HCTR_LIB_THROW(cudaMalloc((void**)&d_row_ptrs_,
                            (inference_params_.max_batchsize * inference_parser_.slot_num +
                             inference_parser_.num_embedding_tables) *
                                sizeof(int)));
  HCTR_LIB_THROW(cudaMalloc((void**)&d_output_, inference_params_.max_batchsize *
                                                    inference_parser_.label_dim * sizeof(float)));
  if (inference_params_.i64_input_key) {
    HCTR_LIB_THROW(cudaHostAlloc((void**)&h_embeddingcolumns_,
                                 inference_params_.max_batchsize *
                                     inference_parser_.max_feature_num_per_sample *
                                     sizeof(long long),
                                 cudaHostAllocPortable));
  } else {
    HCTR_LIB_THROW(cudaHostAlloc((void**)&h_embeddingcolumns_,
                                 inference_params_.max_batchsize *
                                     inference_parser_.max_feature_num_per_sample *
                                     sizeof(unsigned int),
                                 cudaHostAllocPortable));
  }
}

InferenceSessionPy::InferenceSessionPy(const std::string& model_config_path,
                                       const InferenceParams& inference_params,
                                       const std::shared_ptr<EmbeddingCacheBase>& embedding_cache)
    : InferenceSession(model_config_path, inference_params, embedding_cache) {
  initialize();
}

InferenceSessionPy::InferenceSessionPy(
    const std::string& model_config_path, const InferenceParams& inference_params,
    const std::shared_ptr<EmbeddingCacheBase>& embedding_cache,
    const std::shared_ptr<HierParameterServerBase>& parameter_server)
    : InferenceSession(model_config_path, inference_params, embedding_cache),
      parameter_server_(parameter_server) {
  initialize();
}

InferenceSessionPy::~InferenceSessionPy() {
  CudaDeviceContext context(resource_manager_->get_local_gpu(0)->get_device_id());
  cudaFree(d_dense_);
  cudaFreeHost(h_embeddingcolumns_);
  cudaFree(d_row_ptrs_);
  cudaFree(d_output_);
}

template <typename TypeKey>
void InferenceSessionPy::predict_(const std::vector<float>& dense,
                                  const std::vector<TypeKey>& embeddingcolumns,
                                  const std::vector<int>& row_ptrs) {
  if (inference_parser_.slot_num == 0) {
    HCTR_OWN_THROW(Error_t::WrongInput, "The number of slots should not be zero");
  }
  size_t num_samples =
      (row_ptrs.size() - inference_parser_.num_embedding_tables) / inference_parser_.slot_num;
  if (num_samples > inference_params_.max_batchsize) {
    HCTR_OWN_THROW(Error_t::WrongInput, "The number of samples should not exceed max_batchsize");
  }
  if (num_samples * inference_parser_.dense_dim != dense.size()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "The dimension of dense features is not consistent");
  }
  if (num_samples * inference_parser_.slot_num + inference_parser_.num_embedding_tables !=
      row_ptrs.size()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "The dimension of row pointers is not consistent");
  }
  if (num_samples * inference_parser_.max_feature_num_per_sample < embeddingcolumns.size()) {
    HCTR_OWN_THROW(
        Error_t::WrongInput,
        "The dimension of embedding keys is greater than num_samples*max_feature_num_per_sample");
  }
  size_t num_embeddingcolumns = 0;
  size_t row_ptr_offset = 0;
  for (int j = 0; j < static_cast<int>(inference_parser_.num_embedding_tables); j++) {
    num_embeddingcolumns +=
        row_ptrs[num_samples * inference_parser_.slot_num_for_tables[j] + row_ptr_offset];
    row_ptr_offset += num_samples * inference_parser_.slot_num_for_tables[j] + 1;
  }
  if (embeddingcolumns.size() != num_embeddingcolumns) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "The dimension of embedding keys is not consistent with row pointers");
  }
  CudaDeviceContext context(resource_manager_->get_local_gpu(0)->get_device_id());
  output_.resize(num_samples);
  size_t num_keys = embeddingcolumns.size();
  HCTR_LIB_THROW(cudaMemcpyAsync(
      d_dense_, dense.data(), num_samples * inference_parser_.dense_dim * sizeof(float),
      cudaMemcpyHostToDevice, resource_manager_->get_local_gpu(0)->get_stream()));
  HCTR_LIB_THROW(cudaMemcpyAsync(
      d_row_ptrs_, row_ptrs.data(),
      (num_samples * inference_parser_.slot_num + inference_parser_.num_embedding_tables) *
          sizeof(int),
      cudaMemcpyHostToDevice, resource_manager_->get_local_gpu(0)->get_stream()));
  memcpy(h_embeddingcolumns_, embeddingcolumns.data(), num_keys * sizeof(TypeKey));
  InferenceSession::predict(d_dense_, h_embeddingcolumns_, d_row_ptrs_, d_output_,
                            static_cast<int>(num_samples));
  HCTR_LIB_THROW(cudaMemcpyAsync(
      output_.data(), d_output_, num_samples * inference_parser_.label_dim * sizeof(float),
      cudaMemcpyDeviceToHost, resource_manager_->get_local_gpu(0)->get_stream()));
  HCTR_LIB_THROW(cudaStreamSynchronize(resource_manager_->get_local_gpu(0)->get_stream()));
}

std::vector<float>& InferenceSessionPy::predict(const std::vector<float>& dense,
                                                const std::vector<long long>& embeddingcolumns,
                                                const std::vector<int>& row_ptrs) {
  if (inference_params_.i64_input_key) {
    predict_<long long>(dense, embeddingcolumns, row_ptrs);
  } else {
    std::vector<unsigned int>().swap(embeddingcolumns_u32_);
    std::transform(embeddingcolumns.begin(), embeddingcolumns.end(),
                   std::back_inserter(embeddingcolumns_u32_),
                   [](const long long& v) -> unsigned int { return static_cast<unsigned int>(v); });
    predict_<unsigned int>(dense, embeddingcolumns_u32_, row_ptrs);
  }
  return output_;
}

template <typename TypeKey>
void InferenceSessionPy::load_data(const std::string& source,
                                   const DataReaderType_t data_reader_type,
                                   const Check_t check_type,
                                   const std::vector<long long>& slot_size_array) {
  CudaDeviceContext context(resource_manager_->get_local_gpu(0)->get_device_id());
  bool repeat_dataset = true;
  std::map<std::string, SparseInput<TypeKey>> sparse_input_map;
  std::map<std::string, TensorBag2> label_dense_map;
  // force the data reader to not use mixed precision
  create_datareader<TypeKey>()(inference_params_, inference_parser_, data_reader_,
                               resource_manager_, sparse_input_map, label_dense_map, source,
                               data_reader_type, check_type, slot_size_array, repeat_dataset);
  if (data_reader_->is_started() == false) {
    HCTR_OWN_THROW(Error_t::IllegalCall, "Start the data reader first before evaluation");
  }
  TensorBag2 dense_tensor;
  if (!find_item_in_map(label_tensor_, inference_parser_.label_name, label_dense_map)) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Cannot find " + inference_parser_.label_name);
  }
  if (!find_item_in_map(dense_tensor, inference_parser_.dense_name, label_dense_map)) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Cannot find " + inference_parser_.dense_name);
  }
  d_reader_dense_ = reinterpret_cast<float*>(dense_tensor.get_ptr());
  d_reader_keys_list_.clear();
  d_reader_row_ptrs_list_.clear();
  for (size_t i = 0; i < inference_parser_.num_embedding_tables; i++) {
    SparseInput<TypeKey> sparse_input;
    if (!find_item_in_map(sparse_input, inference_parser_.sparse_names[i], sparse_input_map)) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Cannot find " + inference_parser_.sparse_names[i]);
    }
    d_reader_keys_list_.push_back(
        reinterpret_cast<void*>(sparse_input.evaluate_sparse_tensors[0].get_value_ptr()));
    d_reader_row_ptrs_list_.push_back(
        reinterpret_cast<void*>(sparse_input.evaluate_sparse_tensors[0].get_rowoffset_ptr()));
  }
}

template <typename TypeKey>
float InferenceSessionPy::evaluate_(const size_t num_batches, const std::string& source,
                                    const DataReaderType_t data_reader_type,
                                    const Check_t check_type,
                                    const std::vector<long long>& slot_size_array) {
  CudaDeviceContext context(resource_manager_->get_local_gpu(0)->get_device_id());
  load_data<TypeKey>(source, data_reader_type, check_type, slot_size_array);
  std::vector<size_t> keys_elements_list(inference_parser_.num_embedding_tables);
  std::vector<size_t> row_ptr_elements_list(inference_parser_.num_embedding_tables);
  for (size_t i = 0; i < inference_parser_.num_embedding_tables; i++) {
    keys_elements_list[i] =
        inference_params_.max_batchsize * inference_parser_.max_feature_num_for_tables[i];
    row_ptr_elements_list[i] =
        inference_params_.max_batchsize * inference_parser_.slot_num_for_tables[i] + 1;
  }
  std::vector<size_t> pred_dims = {inference_params_.max_batchsize, inference_parser_.label_dim};
  std::shared_ptr<TensorBuffer2> pred_buff =
      PreallocatedBuffer2<float>::create(d_output_, pred_dims);
  Tensor2<float> pred_tensor(pred_dims, pred_buff);
  std::shared_ptr<metrics::AUC<float>> metric = std::make_shared<metrics::AUC<float>>(
      inference_params_.max_batchsize, num_batches, inference_parser_.label_dim, resource_manager_);
  metrics::RawMetricMap metric_maps = {{metrics::RawType::Pred, pred_tensor.shrink()},
                                       {metrics::RawType::Label, label_tensor_}};
  for (size_t batch = 0; batch < num_batches; batch++) {
    long long current_batchsize = data_reader_->read_a_batch_to_device();
    size_t keys_offset = 0;
    size_t row_ptrs_offset = 0;
    std::vector<TypeKey> h_reader_keys(inference_params_.max_batchsize *
                                       inference_parser_.max_feature_num_per_sample);
    std::vector<std::vector<TypeKey>> h_reader_row_ptrs_list;
    for (size_t i = 0; i < inference_parser_.num_embedding_tables; i++) {
      std::vector<TypeKey> h_reader_row_ptrs(row_ptr_elements_list[i]);
      convert_array_on_device(
          d_row_ptrs_ + row_ptrs_offset, reinterpret_cast<TypeKey*>(d_reader_row_ptrs_list_[i]),
          row_ptr_elements_list[i], resource_manager_->get_local_gpu(0)->get_stream());
      HCTR_LIB_THROW(cudaMemcpyAsync(h_reader_row_ptrs.data(), d_reader_row_ptrs_list_[i],
                                     row_ptr_elements_list[i] * sizeof(TypeKey),
                                     cudaMemcpyDeviceToHost,
                                     resource_manager_->get_local_gpu(0)->get_stream()));
      HCTR_LIB_THROW(cudaStreamSynchronize(resource_manager_->get_local_gpu(0)->get_stream()));
      size_t num_keys = h_reader_row_ptrs.back() - h_reader_row_ptrs.front();
      h_reader_row_ptrs_list.push_back(h_reader_row_ptrs);
      HCTR_LIB_THROW(cudaMemcpyAsync(h_reader_keys.data() + keys_offset, d_reader_keys_list_[i],
                                     num_keys * sizeof(TypeKey), cudaMemcpyDeviceToHost,
                                     resource_manager_->get_local_gpu(0)->get_stream()));
      keys_offset += num_keys;
      row_ptrs_offset += row_ptr_elements_list[i];
    }
    distribute_keys_for_inference(reinterpret_cast<TypeKey*>(h_embeddingcolumns_),
                                  h_reader_keys.data(), current_batchsize, h_reader_row_ptrs_list,
                                  inference_parser_.slot_num_for_tables);
    InferenceSession::predict(d_reader_dense_, h_embeddingcolumns_, d_row_ptrs_, d_output_,
                              current_batchsize);
    metric->set_current_batch_size(current_batchsize);
    metric->local_reduce(0, metric_maps);
  }
  float auc_value = metric->finalize_metric();
  return auc_value;
}

template <typename TypeKey>
pybind11::array_t<float> InferenceSessionPy::predict_(
    const size_t num_batches, const std::string& source, const DataReaderType_t data_reader_type,
    const Check_t check_type, const std::vector<long long>& slot_size_array) {
  CudaDeviceContext context(resource_manager_->get_local_gpu(0)->get_device_id());
  load_data<TypeKey>(source, data_reader_type, check_type, slot_size_array);
  std::vector<size_t> keys_elements_list(inference_parser_.num_embedding_tables);
  std::vector<size_t> row_ptr_elements_list(inference_parser_.num_embedding_tables);
  for (size_t i = 0; i < inference_parser_.num_embedding_tables; i++) {
    keys_elements_list[i] =
        inference_params_.max_batchsize * inference_parser_.max_feature_num_for_tables[i];
    row_ptr_elements_list[i] =
        inference_params_.max_batchsize * inference_parser_.slot_num_for_tables[i] + 1;
  }
  std::vector<size_t> pred_size;
  if (inference_parser_.label_dim == 1) {
    pred_size = {inference_params_.max_batchsize * num_batches};
  } else {
    pred_size = {inference_params_.max_batchsize * num_batches, inference_parser_.label_dim};
  }
  auto pred = pybind11::array_t<float>(pred_size);
  pybind11::buffer_info pred_array_buff = pred.request();
  float* pred_ptr = static_cast<float*>(pred_array_buff.ptr);
  size_t pred_ptr_offset = 0;
  for (size_t batch = 0; batch < num_batches; batch++) {
    long long current_batchsize = data_reader_->read_a_batch_to_device();
    size_t keys_offset = 0;
    size_t row_ptrs_offset = 0;
    std::vector<TypeKey> h_reader_keys(inference_params_.max_batchsize *
                                       inference_parser_.max_feature_num_per_sample);
    std::vector<std::vector<TypeKey>> h_reader_row_ptrs_list;
    for (size_t i = 0; i < inference_parser_.num_embedding_tables; i++) {
      std::vector<TypeKey> h_reader_row_ptrs(row_ptr_elements_list[i]);
      convert_array_on_device(
          d_row_ptrs_ + row_ptrs_offset, reinterpret_cast<TypeKey*>(d_reader_row_ptrs_list_[i]),
          row_ptr_elements_list[i], resource_manager_->get_local_gpu(0)->get_stream());
      HCTR_LIB_THROW(cudaMemcpyAsync(h_reader_row_ptrs.data(), d_reader_row_ptrs_list_[i],
                                     row_ptr_elements_list[i] * sizeof(TypeKey),
                                     cudaMemcpyDeviceToHost,
                                     resource_manager_->get_local_gpu(0)->get_stream()));
      HCTR_LIB_THROW(cudaStreamSynchronize(resource_manager_->get_local_gpu(0)->get_stream()));
      size_t num_keys = h_reader_row_ptrs.back() - h_reader_row_ptrs.front();
      h_reader_row_ptrs_list.push_back(h_reader_row_ptrs);
      HCTR_LIB_THROW(cudaMemcpyAsync(h_reader_keys.data() + keys_offset, d_reader_keys_list_[i],
                                     num_keys * sizeof(TypeKey), cudaMemcpyDeviceToHost,
                                     resource_manager_->get_local_gpu(0)->get_stream()));
      keys_offset += num_keys;
      row_ptrs_offset += row_ptr_elements_list[i];
    }
    distribute_keys_for_inference(reinterpret_cast<TypeKey*>(h_embeddingcolumns_),
                                  h_reader_keys.data(), current_batchsize, h_reader_row_ptrs_list,
                                  inference_parser_.slot_num_for_tables);
    InferenceSession::predict(d_reader_dense_, h_embeddingcolumns_, d_row_ptrs_, d_output_,
                              current_batchsize);
    HCTR_LIB_THROW(cudaMemcpyAsync(
        pred_ptr + pred_ptr_offset, d_output_,
        inference_params_.max_batchsize * inference_parser_.label_dim * sizeof(float),
        cudaMemcpyDeviceToHost, resource_manager_->get_local_gpu(0)->get_stream()));
    pred_ptr_offset += inference_params_.max_batchsize * inference_parser_.label_dim;
  }
  HCTR_LIB_THROW(cudaStreamSynchronize(resource_manager_->get_local_gpu(0)->get_stream()));
  return pred;
}

float InferenceSessionPy::evaluate(const size_t num_batches, const std::string& source,
                                   const DataReaderType_t data_reader_type,
                                   const Check_t check_type,
                                   const std::vector<long long>& slot_size_array) {
  float auc_value;
  if (inference_params_.i64_input_key) {
    auc_value =
        evaluate_<long long>(num_batches, source, data_reader_type, check_type, slot_size_array);
  } else {
    auc_value =
        evaluate_<unsigned int>(num_batches, source, data_reader_type, check_type, slot_size_array);
  }
  return auc_value;
}

pybind11::array_t<float> InferenceSessionPy::predict(
    const size_t num_batches, const std::string& source, const DataReaderType_t data_reader_type,
    const Check_t check_type, const std::vector<long long>& slot_size_array) {
  if (inference_params_.i64_input_key) {
    return predict_<long long>(num_batches, source, data_reader_type, check_type, slot_size_array);
  } else {
    return predict_<unsigned int>(num_batches, source, data_reader_type, check_type,
                                  slot_size_array);
  }
}

void InferenceSessionPy::refresh_embedding_cache() {
  parameter_server_->refresh_embedding_cache(inference_params_.model_name,
                                             inference_params_.device_id);
}

std::shared_ptr<InferenceSessionPy> CreateInferenceSession(
    const std::string& model_config_path, const InferenceParams& inference_params) {
  std::vector<std::string> model_config_path_array{model_config_path};
  std::vector<InferenceParams> inference_params_array{inference_params};
  parameter_server_config ps_config{model_config_path_array, inference_params_array};

  auto parameter_server = HierParameterServerBase::create(ps_config, inference_params_array);
  auto embedding_cache = parameter_server->get_embedding_cache(inference_params.model_name,
                                                               inference_params.device_id);
  std::shared_ptr<InferenceSessionPy> inference_session = std::make_shared<InferenceSessionPy>(
      model_config_path, inference_params, embedding_cache, parameter_server);
  return inference_session;
}

void InferencePybind(pybind11::module& m) {
  pybind11::module infer = m.def_submodule("inference", "inference submodule of hugectr");

  pybind11::class_<HugeCTR::VolatileDatabaseParams,
                   std::shared_ptr<HugeCTR::VolatileDatabaseParams>>(infer,
                                                                     "VolatileDatabaseParams")
      .def(pybind11::init<DatabaseType_t,
                          // Backend specific.
                          const std::string&, const std::string&, const std::string&,
                          DatabaseHashMapAlgorithm_t, size_t, size_t, size_t,
                          // Overflow handling related.
                          bool, size_t, DatabaseOverflowPolicy_t, double,
                          // Caching behavior related.
                          double, bool,
                          // Real-time update mechanism related.
                          const std::vector<std::string>&>(),
           pybind11::arg("type") = DatabaseType_t::ParallelHashMap,
           // Backend specific.
           pybind11::arg("address") = "127.0.0.1:7000", pybind11::arg("user_name") = "default",
           pybind11::arg("password") = "",
           pybind11::arg("algorithm") = DatabaseHashMapAlgorithm_t::PHM,
           pybind11::arg("num_partitions") = std::min(16u, std::thread::hardware_concurrency()),
           pybind11::arg("max_get_batch_size") = 10'000,
           pybind11::arg("max_set_batch_size") = 10'000,
           // Overflow handling related.
           pybind11::arg("refresh_time_after_fetch") = false,
           pybind11::arg("overflow_margin") = std::numeric_limits<size_t>::max(),
           pybind11::arg("overflow_policy") = DatabaseOverflowPolicy_t::EvictOldest,
           pybind11::arg("overflow_resolution_target") = 0.8,
           // Caching behavior related.
           pybind11::arg("initial_cache_rate") = 1.0,
           pybind11::arg("cache_missed_embeddings") = false,
           // Real-time update mechanism related.
           pybind11::arg("update_filters") = std::vector<std::string>{"^hps_.+$"});

  pybind11::class_<HugeCTR::PersistentDatabaseParams,
                   std::shared_ptr<HugeCTR::PersistentDatabaseParams>>(infer,
                                                                       "PersistentDatabaseParams")
      .def(pybind11::init<DatabaseType_t,
                          // Backend specific.
                          const std::string&, size_t, bool, size_t, size_t,
                          // Real-time update mechanism related.
                          const std::vector<std::string>&>(),
           pybind11::arg("backend") = DatabaseType_t::Disabled,
           // Backend specific.
           pybind11::arg("path") = (std::filesystem::temp_directory_path() / "rocksdb").string(),
           pybind11::arg("num_threads") = 16, pybind11::arg("read_only") = false,
           pybind11::arg("max_get_batch_size") = 10'000,
           pybind11::arg("max_set_batch_size") = 10'000,
           // Real-time update mechanism related.
           pybind11::arg("update_filters") = std::vector<std::string>{"^hps_.+$"});

  pybind11::class_<HugeCTR::UpdateSourceParams, std::shared_ptr<HugeCTR::UpdateSourceParams>>(
      infer, "UpdateSourceParams")
      .def(pybind11::init<UpdateSourceType_t,
                          // Backend specific.
                          const std::string&, size_t, size_t, size_t, size_t, size_t, size_t>(),
           pybind11::arg("type") = UpdateSourceType_t::Null,
           // Backend specific.
           pybind11::arg("brokers") = "127.0.0.1:9092",
           pybind11::arg("metadata_refresh_interval_ms") = 30'000,
           pybind11::arg("receive_buffer_size") = 256 * 1024,
           pybind11::arg("poll_timeout_ms") = 500, pybind11::arg("max_batch_size") = 8 * 1024,
           pybind11::arg("failure_backoff_ms") = 50, pybind11::arg("max_commit_interval") = 32);

  pybind11::class_<HugeCTR::InferenceParams, std::shared_ptr<HugeCTR::InferenceParams>>(
      infer, "InferenceParams")
      .def(pybind11::init<const std::string&, const size_t, const float, const std::string&,
                          const std::vector<std::string>&, const int, const bool, const float,
                          const bool, const bool, const float, const bool, const bool,
                          // HugeCTR::DATABASE_TYPE, const std::string&, const std::string&,
                          // const float,
                          const int, const int, const float, const std::vector<int>&,
                          const std::vector<float>&, const VolatileDatabaseParams&,
                          const PersistentDatabaseParams&, const UpdateSourceParams&, const int,
                          const float, const float, const std::vector<size_t>&,
                          const std::vector<size_t>&, const std::vector<std::string>&>(),

           pybind11::arg("model_name"), pybind11::arg("max_batchsize"),
           pybind11::arg("hit_rate_threshold"), pybind11::arg("dense_model_file"),
           pybind11::arg("sparse_model_files"), pybind11::arg("device_id") = 0,
           pybind11::arg("use_gpu_embedding_cache"), pybind11::arg("cache_size_percentage"),
           pybind11::arg("i64_input_key"), pybind11::arg("use_mixed_precision") = false,
           pybind11::arg("scaler") = 1.0, pybind11::arg("use_algorithm_search") = true,
           pybind11::arg("use_cuda_graph") = true,
           pybind11::arg("number_of_worker_buffers_in_pool") = 2,
           pybind11::arg("number_of_refresh_buffers_in_pool") = 1,
           pybind11::arg("cache_refresh_percentage_per_iteration") = 0.1,
           pybind11::arg("deployed_devices") = std::vector<int>{0},
           pybind11::arg("default_value_for_each_table") = std::vector<float>{0.0f},
           // Database backend.
           pybind11::arg("volatile_db") = VolatileDatabaseParams{},
           pybind11::arg("persistent_db") = PersistentDatabaseParams{},
           pybind11::arg("update_source") = UpdateSourceParams{},
           // HPS required
           pybind11::arg("maxnum_des_feature_per_sample") = 20,
           pybind11::arg("refresh_delay") = 0.0f, pybind11::arg("refresh_interval") = 0.0f,
           pybind11::arg("maxnum_catfeature_query_per_table_per_sample") = std::vector<int>{26},
           pybind11::arg("embedding_vecsize_per_table") = std::vector<int>{128},
           pybind11::arg("embedding_table_names") = std::vector<std::string>{""});

  infer.def("CreateInferenceSession", &HugeCTR::python_lib::CreateInferenceSession,
            pybind11::arg("model_config_path"), pybind11::arg("inference_params"));

  pybind11::class_<HugeCTR::python_lib::InferenceSessionPy,
                   std::shared_ptr<HugeCTR::python_lib::InferenceSessionPy>>(infer,
                                                                             "InferenceSession")
      .def("evaluate", &HugeCTR::python_lib::InferenceSessionPy::evaluate,
           pybind11::arg("num_batches"), pybind11::arg("source"), pybind11::arg("data_reader_type"),
           pybind11::arg("check_type"), pybind11::arg("slot_size_array") = std::vector<long long>())
      .def("predict",
           pybind11::overload_cast<size_t, const std::string&, DataReaderType_t, Check_t,
                                   const std::vector<long long>&>(
               &HugeCTR::python_lib::InferenceSessionPy::predict),
           pybind11::arg("num_batches"), pybind11::arg("source"), pybind11::arg("data_reader_type"),
           pybind11::arg("check_type"), pybind11::arg("slot_size_array") = std::vector<long long>())
      .def("predict",
           pybind11::overload_cast<const std::vector<float>&, const std::vector<long long>&,
                                   const std::vector<int>&>(
               &HugeCTR::python_lib::InferenceSessionPy::predict),
           pybind11::arg("dense_feature"), pybind11::arg("embeddingcolumns"),
           pybind11::arg("row_ptrs"))
      .def("refresh_embedding_cache",
           &HugeCTR::python_lib::InferenceSessionPy::refresh_embedding_cache);
  pybind11::class_<HugeCTR::InferenceModel, std::shared_ptr<HugeCTR::InferenceModel>>(
      infer, "InferenceModel")
      .def(pybind11::init<const std::string&, const InferenceParams&>(),
           pybind11::arg("model_config_path"), pybind11::arg("inference_params"))
      .def(
          "predict",
          [](HugeCTR::InferenceModel& self, size_t num_batches, const std::string& source,
             DataReaderType_t data_reader_type, Check_t check_type,
             const std::vector<long long>& slot_size_array) {
            auto& inference_params = self.get_inference_params();
            auto& inference_parser = self.get_inference_parser();
            float* pred_output = new float[num_batches * inference_params.max_batchsize *
                                           inference_parser.label_dim];
            self.predict(pred_output, num_batches, source, data_reader_type, check_type,
                         slot_size_array);
            auto pred_output_capsule = pybind11::capsule(pred_output, [](void* v) {
              float* vv = reinterpret_cast<float*>(v);
              delete[] vv;
            });
            pybind11::array_t<float> pred_array(
                {num_batches * inference_params.max_batchsize, inference_parser.label_dim},
                pred_output, pred_output_capsule);
            return pred_array;
          },
          pybind11::arg("num_batches"), pybind11::arg("source"), pybind11::arg("data_reader_type"),
          pybind11::arg("check_type"), pybind11::arg("slot_size_array") = std::vector<long long>())
      .def("evaluate", &HugeCTR::InferenceModel::evaluate, pybind11::arg("num_batches"),
           pybind11::arg("source"), pybind11::arg("data_reader_type"), pybind11::arg("check_type"),
           pybind11::arg("slot_size_array") = std::vector<long long>());
}

}  // namespace python_lib

}  // namespace HugeCTR