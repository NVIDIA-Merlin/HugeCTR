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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <HugeCTR/include/inference/parameter_server.hpp>
#include <HugeCTR/include/inference/embedding_cache.hpp>
#include <HugeCTR/include/inference/session_inference.hpp>
#include <HugeCTR/include/utils.hpp>

namespace HugeCTR {

namespace python_lib {

/**
 * @brief Main InferenceSessionPy class
 *
 * This is a class supporting HugeCTR inference in Python, which includes predict32 and predict64.
 * To support dynamic batch size during inference, this class need to be modified in the future.
 */
class InferenceSessionPy : public InferenceSession {
public:
  InferenceSessionPy(const std::string& model_config_path,
                    const InferenceParams& inference_params,
                    std::shared_ptr<embedding_interface>& embedding_cache);
  
  ~InferenceSessionPy();

  float evaluate(const size_t num_batches, const std::string& source, const DataReaderType_t data_reader_type,
              const Check_t check_type, const std::vector<long long>& slot_size_array);
  pybind11::array_t<float> predict(const size_t num_batches, const std::string& source, const DataReaderType_t data_reader_type,
              const Check_t check_type, const std::vector<long long>& slot_size_array);
  std::vector<float>& predict(std::vector<float>& dense, std::vector<long long>& embeddingcolumns, std::vector<int>& row_ptrs);

private:
  template <typename TypeKey>
  void load_data_(const std::string& source, const DataReaderType_t data_reader_type,
                  const Check_t check_type, const std::vector<long long>& slot_size_array);
  
  template <typename TypeKey>
  float evaluate_(const size_t num_batches, const std::string& source, const DataReaderType_t data_reader_type,
                const Check_t check_type, const std::vector<long long>& slot_size_array);

  template <typename TypeKey>
  pybind11::array_t<float> predict_(const size_t num_batches, const std::string& source, const DataReaderType_t data_reader_type,
                const Check_t check_type, const std::vector<long long>& slot_size_array);
  
  template <typename TypeKey>
  void predict_(std::vector<float>& dense, std::vector<TypeKey>& embeddingcolumns, std::vector<int>& row_ptrs);
  
  std::vector<unsigned int> embeddingcolumns_u32_;
  std::vector<float> output_;
  float* d_dense_;
  void* h_embeddingcolumns_;
  int* d_row_ptrs_;
  float* d_output_;
  void* d_reader_keys_;
  void* d_reader_row_ptrs_;
  TensorBag2 label_tensor_;
};

InferenceSessionPy::InferenceSessionPy(const std::string& model_config_path,
                  const InferenceParams& inference_params,
                  std::shared_ptr<embedding_interface>& embedding_cache)
  : InferenceSession(model_config_path, inference_params, embedding_cache) {
  CudaDeviceContext context(resource_manager_->get_local_gpu(0)->get_device_id());
  CK_CUDA_THROW_(cudaMalloc((void**)&d_dense_, inference_params_.max_batchsize *  inference_parser_.dense_dim * sizeof(float)));
  CK_CUDA_THROW_(cudaMalloc((void**)&d_row_ptrs_, (inference_params_.max_batchsize *  inference_parser_.slot_num + 1) * sizeof(int)));
  CK_CUDA_THROW_(cudaMalloc((void**)&d_output_, inference_params_.max_batchsize * inference_parser_.label_dim * sizeof(float)));
  if (inference_params_.i64_input_key) {
    CK_CUDA_THROW_(cudaHostAlloc((void**)&h_embeddingcolumns_, inference_params_.max_batchsize *  inference_parser_.max_feature_num_per_sample * sizeof(long long), cudaHostAllocPortable));
  } else {
    CK_CUDA_THROW_(cudaHostAlloc((void**)&h_embeddingcolumns_, inference_params_.max_batchsize *  inference_parser_.max_feature_num_per_sample * sizeof(unsigned int), cudaHostAllocPortable));
  }
}

InferenceSessionPy::~InferenceSessionPy() {
  CudaDeviceContext context(resource_manager_->get_local_gpu(0)->get_device_id());
  cudaFree(d_dense_);
  cudaFreeHost(h_embeddingcolumns_);
  cudaFree(d_row_ptrs_);
  cudaFree(d_output_);
}

template <typename TypeKey>
void InferenceSessionPy::predict_(std::vector<float>& dense, std::vector<TypeKey>& embeddingcolumns, std::vector<int>& row_ptrs) {
  if (inference_parser_.slot_num == 0) {
    CK_THROW_(Error_t::WrongInput, "The number of slots should not be zero");
  }
  size_t num_samples = (row_ptrs.size() - 1) / inference_parser_.slot_num;
  if (num_samples > inference_params_.max_batchsize) {
    CK_THROW_(Error_t::WrongInput, "The number of samples should not exceed max_batchsize");
  }
  if (num_samples * inference_parser_.dense_dim != dense.size()) {
    CK_THROW_(Error_t::WrongInput, "The dimension of dense features is not consistent");
  }
  if (num_samples * inference_parser_.slot_num + inference_parser_.num_embedding_tables != row_ptrs.size()) {
    CK_THROW_(Error_t::WrongInput, "The dimension of row pointers is not consistent");
  }
  if (num_samples * inference_parser_.max_feature_num_per_sample < embeddingcolumns.size()) {
    CK_THROW_(Error_t::WrongInput, "The dimension of embedding keys is greater than num_samples*max_feature_num_per_sample");
  }
  size_t num_embeddingcolumns=0;
  size_t row_ptr_offset=0;
  for (int j = 0; j < static_cast<int>(inference_parser_.num_embedding_tables); j++) {
      num_embeddingcolumns += row_ptrs[num_samples * inference_parser_.slot_num_for_tables[j] + row_ptr_offset];
      row_ptr_offset += num_samples * inference_parser_.slot_num_for_tables[j]+1;
    }
  if (embeddingcolumns.size() != num_embeddingcolumns) {
    CK_THROW_(Error_t::WrongInput, "The dimension of embedding keys is not consistent with row pointers");
  }
  CudaDeviceContext context(resource_manager_->get_local_gpu(0)->get_device_id());
  output_.resize(num_samples);
  size_t num_keys = embeddingcolumns.size();
  CK_CUDA_THROW_(cudaMemcpy(d_dense_, dense.data(), num_samples*inference_parser_.dense_dim*sizeof(float), cudaMemcpyHostToDevice));
  CK_CUDA_THROW_(cudaMemcpy(d_row_ptrs_, row_ptrs.data(), (num_samples*inference_parser_.slot_num+inference_parser_.num_embedding_tables)*sizeof(int), cudaMemcpyHostToDevice)); 
  memcpy(h_embeddingcolumns_, embeddingcolumns.data(), num_keys * sizeof(TypeKey));
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  InferenceSession::predict(d_dense_, h_embeddingcolumns_, d_row_ptrs_, d_output_, static_cast<int>(num_samples));
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  CK_CUDA_THROW_(cudaMemcpy(output_.data(), d_output_, num_samples*inference_parser_.label_dim*sizeof(float), cudaMemcpyDeviceToHost));
}

std::vector<float>& InferenceSessionPy::predict(std::vector<float>& dense, std::vector<long long>& embeddingcolumns, std::vector<int>& row_ptrs) {
  if (inference_params_.i64_input_key) {
    predict_<long long>(dense, embeddingcolumns, row_ptrs);
  } else {
    std::vector<unsigned int>().swap(embeddingcolumns_u32_);
    std::transform(embeddingcolumns.begin(), embeddingcolumns.end(), std::back_inserter(embeddingcolumns_u32_),
                  [](long long& v)->unsigned int { return static_cast<unsigned int>(v); });
    predict_<unsigned int>(dense, embeddingcolumns_u32_, row_ptrs);
  }
  return output_;
}

template <typename TypeKey>
void InferenceSessionPy::load_data_(const std::string& source, const DataReaderType_t data_reader_type,
                                const Check_t check_type, const std::vector<long long>& slot_size_array) {
  CudaDeviceContext context(resource_manager_->get_local_gpu(0)->get_device_id());
  bool repeat_dataset = true;
  std::map<std::string, SparseInput<TypeKey>> sparse_input_map;
  std::map<std::string, TensorBag2> label_dense_map;
  // force the data reader to not use mixed precision
  create_datareader<TypeKey>()(inference_params_, inference_parser_, data_reader_, resource_manager_,
                              sparse_input_map, label_dense_map,
                              source, data_reader_type, check_type, slot_size_array, repeat_dataset);
  if (data_reader_->is_started() == false) {
    CK_THROW_(Error_t::IllegalCall, "Start the data reader first before evaluation");
  }
  SparseInput<TypeKey> sparse_input;
  TensorBag2 dense_tensor;
  if (!find_item_in_map(sparse_input, inference_parser_.sparse_names[0], sparse_input_map)) {
    CK_THROW_(Error_t::WrongInput, "Cannot find " + inference_parser_.sparse_names[0]);
  }
  if (!find_item_in_map(label_tensor_, inference_parser_.label_name, label_dense_map)) {
    CK_THROW_(Error_t::WrongInput, "Cannot find " + inference_parser_.label_name);
  }
  if (!find_item_in_map(dense_tensor, inference_parser_.dense_name, label_dense_map)) {
    CK_THROW_(Error_t::WrongInput, "Cannot find " + inference_parser_.dense_name);
  }
  d_dense_ = reinterpret_cast<float*>(dense_tensor.get_ptr());
  d_reader_keys_ = reinterpret_cast<void*>(sparse_input.evaluate_sparse_tensors[0].get_value_ptr());
  d_reader_row_ptrs_ = reinterpret_cast<void*>(sparse_input.evaluate_sparse_tensors[0].get_rowoffset_ptr());
}

template <typename TypeKey>
float InferenceSessionPy::evaluate_(const size_t num_batches, const std::string& source, const DataReaderType_t data_reader_type,
                                const Check_t check_type, const std::vector<long long>& slot_size_array) {
  CudaDeviceContext context(resource_manager_->get_local_gpu(0)->get_device_id());
  load_data_<TypeKey>(source, data_reader_type, check_type, slot_size_array);
  size_t keys_elements = inference_params_.max_batchsize *  inference_parser_.max_feature_num_per_sample;
  size_t row_ptr_elements =  inference_params_.max_batchsize *  inference_parser_.slot_num + 1;
  std::vector<size_t> pred_dims = {1, inference_params_.max_batchsize};
  std::shared_ptr<TensorBuffer2> pred_buff = PreallocatedBuffer2<float>::create(d_output_, pred_dims);
  Tensor2<float> pred_tensor(pred_dims, pred_buff);
  std::shared_ptr<metrics::AUC<float>> metric = std::make_shared<metrics::AUC<float>>(inference_params_.max_batchsize, num_batches, resource_manager_);
  metrics::RawMetricMap metric_maps = {{metrics::RawType::Pred, pred_tensor.shrink()}, {metrics::RawType::Label,  label_tensor_}};
  for (size_t batch = 0; batch < num_batches; batch++) {
    long long current_batchsize = data_reader_->read_a_batch_to_device();
    CK_CUDA_THROW_(cudaMemcpy(h_embeddingcolumns_, d_reader_keys_,  keys_elements*sizeof(TypeKey), cudaMemcpyDeviceToHost));
    convert_array_on_device(d_row_ptrs_, reinterpret_cast<TypeKey*>(d_reader_row_ptrs_), row_ptr_elements, resource_manager_->get_local_gpu(0)->get_stream());
    InferenceSession::predict(d_dense_, h_embeddingcolumns_, d_row_ptrs_, d_output_, current_batchsize);
    metric->set_current_batch_size(current_batchsize);
    metric->local_reduce(0, metric_maps);
  }
  float auc_value = metric->finalize_metric();
  return auc_value;
}

template <typename TypeKey>
pybind11::array_t<float> InferenceSessionPy::predict_(const size_t num_batches, const std::string& source,
                                                    const DataReaderType_t data_reader_type,
                                                    const Check_t check_type, const std::vector<long long>& slot_size_array) {
  CudaDeviceContext context(resource_manager_->get_local_gpu(0)->get_device_id());
  load_data_<TypeKey>(source, data_reader_type, check_type, slot_size_array);
  size_t keys_elements = inference_params_.max_batchsize *  inference_parser_.max_feature_num_per_sample;
  size_t row_ptr_elements =  inference_params_.max_batchsize *  inference_parser_.slot_num + 1;
  auto pred = pybind11::array_t<float>(inference_params_.max_batchsize * num_batches);
  pybind11::buffer_info pred_array_buff = pred.request();
  float* pred_ptr = static_cast<float*>(pred_array_buff.ptr);
  size_t pred_ptr_offset = 0;
  for (size_t batch = 0; batch < num_batches; batch++) {
    long long current_batchsize = data_reader_->read_a_batch_to_device();
    CK_CUDA_THROW_(cudaMemcpy(h_embeddingcolumns_, d_reader_keys_,  keys_elements*sizeof(TypeKey), cudaMemcpyDeviceToHost));
    convert_array_on_device(d_row_ptrs_, reinterpret_cast<TypeKey*>(d_reader_row_ptrs_), row_ptr_elements, resource_manager_->get_local_gpu(0)->get_stream());

    InferenceSession::predict(d_dense_, h_embeddingcolumns_, d_row_ptrs_, d_output_, current_batchsize);
    CK_CUDA_THROW_(cudaMemcpy(pred_ptr + pred_ptr_offset, d_output_, inference_params_.max_batchsize * sizeof(float), cudaMemcpyDeviceToHost));
    pred_ptr_offset += inference_params_.max_batchsize;
  }
  return pred;
}


float InferenceSessionPy::evaluate(const size_t num_batches, const std::string& source, const DataReaderType_t data_reader_type,
  const Check_t check_type, const std::vector<long long>& slot_size_array) {
  float auc_value;
  if (inference_params_.i64_input_key) {
    auc_value = evaluate_<long long>(num_batches, source, data_reader_type, check_type, slot_size_array);
  } else {
    auc_value = evaluate_<unsigned int>(num_batches, source, data_reader_type, check_type, slot_size_array);
  }
  return auc_value;
}

pybind11::array_t<float> InferenceSessionPy::predict(const size_t num_batches, const std::string& source, 
                                                    const DataReaderType_t data_reader_type,
                                                    const Check_t check_type, const std::vector<long long>& slot_size_array) {
  if (inference_params_.i64_input_key) {
    return predict_<long long>(num_batches, source, data_reader_type, check_type, slot_size_array);
  } else {
    return predict_<unsigned int>(num_batches, source, data_reader_type, check_type, slot_size_array);
  }
}

std::shared_ptr<InferenceSessionPy> CreateInferenceSession(const std::string& model_config_path,
                                                        const InferenceParams& inference_params) {
  HugectrUtility<long long>* ps64;
  HugectrUtility<unsigned int>* ps32;
  std::shared_ptr<embedding_interface> ec;
  std::shared_ptr<InferenceSessionPy> sess;
  std::vector<std::string> model_config_path_array{model_config_path};
  std::vector<InferenceParams> inference_params_array{inference_params};
  if (inference_params.i64_input_key) {
    ps64 = new parameter_server<long long>("Other", model_config_path_array, inference_params_array);
    ec.reset(new embedding_cache<long long>(model_config_path, inference_params, ps64));
  } else {
    ps32 = new parameter_server<unsigned int>("Other", model_config_path_array, inference_params_array);
    ec.reset(new embedding_cache<unsigned int>(model_config_path, inference_params, ps32));
  }
  sess.reset(new InferenceSessionPy(model_config_path, inference_params, ec));
  return sess;
}

void InferencePybind(pybind11::module &m) {
  pybind11::module infer = m.def_submodule("inference", "inference submodule of hugectr");

  pybind11::class_<HugeCTR::InferenceParams, std::shared_ptr<HugeCTR::InferenceParams>>(infer, "InferenceParams")
    .def(pybind11::init<const std::string&, const size_t, const float,
                  const std::string&, const std::vector<std::string>&,
                  const int, const bool, const float, const bool,
                  const bool, const float, const bool, const bool>(),
      pybind11::arg("model_name"),
      pybind11::arg("max_batchsize"),
      pybind11::arg("hit_rate_threshold"),
      pybind11::arg("dense_model_file"),
      pybind11::arg("sparse_model_files"),
		  pybind11::arg("device_id"),
		  pybind11::arg("use_gpu_embedding_cache"),
      pybind11::arg("cache_size_percentage"),
      pybind11::arg("i64_input_key"),
      pybind11::arg("use_mixed_precision") = false,
      pybind11::arg("scaler") = 1.0,
      pybind11::arg("use_algorithm_search") = true,
      pybind11::arg("use_cuda_graph") = true);

  infer.def("CreateInferenceSession", &HugeCTR::python_lib::CreateInferenceSession,
    pybind11::arg("model_config_path"),
    pybind11::arg("inference_params"));
  
  pybind11::class_<HugeCTR::python_lib::InferenceSessionPy, std::shared_ptr<HugeCTR::python_lib::InferenceSessionPy>>(infer, "InferenceSession")
    .def("evaluate", &HugeCTR::python_lib::InferenceSessionPy::evaluate,
      pybind11::arg("num_batches"),
      pybind11::arg("source"),
      pybind11::arg("data_reader_type"),
      pybind11::arg("check_type"),
      pybind11::arg("slot_size_array") = std::vector<long long>())
    .def("predict",
      pybind11::overload_cast<const size_t, const std::string&,
                              const DataReaderType_t, const Check_t,
                              const std::vector<long long>&>(
        &HugeCTR::python_lib::InferenceSessionPy::predict),
      pybind11::arg("num_batches"),
      pybind11::arg("source"),
      pybind11::arg("data_reader_type"),
      pybind11::arg("check_type"),
      pybind11::arg("slot_size_array") = std::vector<long long>())
    .def("predict",
      pybind11::overload_cast<std::vector<float>&, std::vector<long long>&, std::vector<int>&>(
        &HugeCTR::python_lib::InferenceSessionPy::predict),
      pybind11::arg("dense_feature"),
      pybind11::arg("embeddingcolumns"),
      pybind11::arg("row_ptrs"));
}

} // namespace python_lib

} // namespace HugeCTR