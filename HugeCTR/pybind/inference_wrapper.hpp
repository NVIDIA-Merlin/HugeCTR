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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <HugeCTR/include/inference/session_inference.hpp>
#include <HugeCTR/include/inference/parameter_server.hpp>
#include <HugeCTR/include/inference/embedding_cache.hpp>

namespace HugeCTR {

namespace python_lib {
  
std::shared_ptr<parameter_server_base> CreateParameterServer(const std::vector<std::string>& model_config_path_array,
                                                          const std::vector<InferenceParams>& inference_params_array) {
  std::shared_ptr<parameter_server_base> ps;
  if (inference_params_array[0].i64_input_key) {
    ps.reset(new parameter_server<long long>("Other", model_config_path_array, inference_params_array));
  } else {
    ps.reset(new parameter_server<unsigned int>("Other", model_config_path_array, inference_params_array));
  }
  return ps;
}

std::shared_ptr<embedding_interface> CreateEmbeddingCache(const std::string& model_config_path,
                                                        const InferenceParams& inference_params,
                                                        std::shared_ptr<parameter_server_base>& parameter_server) {
  std::shared_ptr<embedding_interface> ec;
  if (inference_params.i64_input_key) {
    ec.reset(new embedding_cache<long long>(model_config_path, inference_params, reinterpret_cast<HugectrUtility<long long>*>(parameter_server.get())));
  } else {
    ec.reset(new embedding_cache<unsigned int>(model_config_path, inference_params, reinterpret_cast<HugectrUtility<unsigned int>*>(parameter_server.get())));
  }
  return ec;
}

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
                    std::shared_ptr<embedding_interface>& embedding_cache)
    : InferenceSession(model_config_path, inference_params, embedding_cache) {
    CK_CUDA_THROW_(cudaMalloc((void**)&d_dense_, inference_params_.max_batchsize *  inference_parser_.dense_dim * sizeof(float)));
    CK_CUDA_THROW_(cudaMalloc((void**)&d_row_ptrs_, (inference_params_.max_batchsize *  inference_parser_.slot_num + 1) * sizeof(int)));
    CK_CUDA_THROW_(cudaMalloc((void**)&d_output_, inference_params_.max_batchsize * inference_parser_.label_dim * sizeof(float)));
    if (inference_params_.i64_input_key) {
      CK_CUDA_THROW_(cudaHostAlloc((void**)&h_embeddingcolumns_, inference_params_.max_batchsize *  inference_parser_.max_feature_num_per_sample * sizeof(long long), cudaHostAllocPortable));
    } else {
      CK_CUDA_THROW_(cudaHostAlloc((void**)&h_embeddingcolumns_, inference_params_.max_batchsize *  inference_parser_.max_feature_num_per_sample * sizeof(unsigned int), cudaHostAllocPortable));
    }
  }
  
  ~InferenceSessionPy() {
    cudaFree(d_dense_);
    cudaFreeHost(h_embeddingcolumns_);
    cudaFree(d_row_ptrs_);
    cudaFree(d_output_);
  }

  std::vector<float>& predict32(std::vector<float>& dense, std::vector<unsigned int>& embeddingcolumns, std::vector<int>& row_ptrs) {
    if (inference_parser_.slot_num == 0) {
      CK_THROW_(Error_t::WrongInput, "The number of slots should not be zero");
    }
    size_t num_samples = (row_ptrs.size() - 1) / inference_parser_.slot_num;
    if (num_samples > inference_params_.max_batchsize ||
        num_samples * inference_parser_.dense_dim != dense.size() ||
        num_samples * inference_parser_.max_feature_num_per_sample < embeddingcolumns.size() ||
        num_samples * inference_parser_.slot_num + 1 != row_ptrs.size() ||
        embeddingcolumns.size() != static_cast<size_t>(row_ptrs.back()) ||
        inference_params_.i64_input_key == true) {
      CK_THROW_(Error_t::WrongInput, "Input size is not consistent!");
    }
    output_.resize(num_samples);
    size_t num_keys = embeddingcolumns.size();
    CK_CUDA_THROW_(cudaMemcpy(d_dense_, dense.data(), num_samples*inference_parser_.dense_dim*sizeof(float), cudaMemcpyHostToDevice));
    CK_CUDA_THROW_(cudaMemcpy(d_row_ptrs_, row_ptrs.data(), (num_samples*inference_parser_.slot_num+1)*sizeof(int), cudaMemcpyHostToDevice)); 
    memcpy(h_embeddingcolumns_, embeddingcolumns.data(), num_keys * sizeof(unsigned int));
    CK_CUDA_THROW_(cudaDeviceSynchronize());
    predict(d_dense_, h_embeddingcolumns_, d_row_ptrs_, d_output_, static_cast<int>(num_samples));
    CK_CUDA_THROW_(cudaDeviceSynchronize());
    CK_CUDA_THROW_(cudaMemcpy(output_.data(), d_output_, num_samples*inference_parser_.label_dim*sizeof(float), cudaMemcpyDeviceToHost));
    return output_;
  };

  std::vector<float>& predict64(std::vector<float>& dense, std::vector<long long>& embeddingcolumns, std::vector<int>& row_ptrs, bool i64_input_key) {
    if (inference_parser_.slot_num == 0) {
      CK_THROW_(Error_t::WrongInput, "The number of slots should not be zero");
    }
    size_t num_samples = (row_ptrs.size() - 1) / inference_parser_.slot_num;
    if (num_samples > inference_params_.max_batchsize ||
        num_samples * inference_parser_.dense_dim != dense.size() ||
        num_samples * inference_parser_.max_feature_num_per_sample < embeddingcolumns.size() ||
        num_samples * inference_parser_.slot_num + 1 != row_ptrs.size() ||
        embeddingcolumns.size() != static_cast<size_t>(row_ptrs.back()) ||
        i64_input_key == false ||
        inference_params_.i64_input_key == false) {
      CK_THROW_(Error_t::WrongInput, "Input size is not consistent!");
    }
    output_.resize(num_samples);
    size_t num_keys = embeddingcolumns.size();
    CK_CUDA_THROW_(cudaMemcpy(d_dense_, dense.data(), num_samples*inference_parser_.dense_dim*sizeof(float), cudaMemcpyHostToDevice));
    CK_CUDA_THROW_(cudaMemcpy(d_row_ptrs_, row_ptrs.data(), (num_samples*inference_parser_.slot_num+1)*sizeof(int), cudaMemcpyHostToDevice)); 
    memcpy(h_embeddingcolumns_, embeddingcolumns.data(), num_keys * sizeof(long long));
    predict(d_dense_, h_embeddingcolumns_, d_row_ptrs_, d_output_, static_cast<int>(num_samples));
    CK_CUDA_THROW_(cudaMemcpy(output_.data(), d_output_, num_samples*inference_parser_.label_dim*sizeof(float), cudaMemcpyDeviceToHost));
    return output_;
  };

private:
  std::vector<float> output_;
  float* d_dense_;
  void* h_embeddingcolumns_;
  int* d_row_ptrs_;
  float* d_output_;
};

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
  
  pybind11::enum_<HugeCTR::INFER_TYPE>(infer, "InferType")
    .value("Triton", HugeCTR::INFER_TYPE::TRITON)
    .value("Other", HugeCTR::INFER_TYPE::OTHER)
    .export_values();
  
  pybind11::class_<HugeCTR::parameter_server_base, std::shared_ptr<HugeCTR::parameter_server_base>>(infer, "ParameterServerBase");
  
  pybind11::class_<HugeCTR::embedding_interface, std::shared_ptr<HugeCTR::embedding_interface>>(infer, "EmbeddingCacheInterface");
  
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
      pybind11::arg("use_cuda_graph") = true)
    .def_readonly("use_gpu_embedding_cache", &HugeCTR::InferenceParams::use_gpu_embedding_cache)
    .def_readonly("i64_input_key", &HugeCTR::InferenceParams::i64_input_key)
    .def_readonly("use_mixed_precision", &HugeCTR::InferenceParams::use_mixed_precision);
  
  infer.def("CreateParameterServer", &HugeCTR::python_lib::CreateParameterServer,
    pybind11::arg("model_config_path_array"),
    pybind11::arg("inference_params_array"));
  
  infer.def("CreateEmbeddingCache", &HugeCTR::python_lib::CreateEmbeddingCache,
    pybind11::arg("model_config_path"),
    pybind11::arg("inference_params"),
    pybind11::arg("parameter_server"));

  infer.def("CreateInferenceSession", &HugeCTR::python_lib::CreateInferenceSession,
    pybind11::arg("model_config_path"),
    pybind11::arg("inference_params"));
  
  pybind11::class_<HugeCTR::python_lib::InferenceSessionPy, std::shared_ptr<HugeCTR::python_lib::InferenceSessionPy>>(infer, "InferenceSession")
    .def(pybind11::init<const std::string&, const InferenceParams&, std::shared_ptr<embedding_interface>&>(),
      pybind11::arg("model_config_path"),
		  pybind11::arg("inference_params"),
		  pybind11::arg("embedding_cache"))
    .def("predict",
      pybind11::overload_cast<std::vector<float>&, std::vector<unsigned int>&, std::vector<int>&>(
        &HugeCTR::python_lib::InferenceSessionPy::predict32),
      pybind11::arg("dense_feature"),
      pybind11::arg("embeddingcolumns"),
      pybind11::arg("row_ptrs"))
    .def("predict",
      pybind11::overload_cast<std::vector<float>&, std::vector<long long>&, std::vector<int>&, bool>(
        &HugeCTR::python_lib::InferenceSessionPy::predict64),
      pybind11::arg("dense_feature"),
      pybind11::arg("embeddingcolumns"),
      pybind11::arg("row_ptrs"),
      pybind11::arg("i64_input_key"));
}

}

}
