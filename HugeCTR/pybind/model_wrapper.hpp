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
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <HugeCTR/pybind/model.hpp>

namespace HugeCTR {

namespace python_lib {

void ModelPybind(pybind11::module &m) {
  pybind11::class_<HugeCTR::Input, std::shared_ptr<HugeCTR::Input>>(m, "Input")
    .def(pybind11::init<DataReaderType_t,
       std::string, std::string, Check_t,
       int, int, std::string, int, std::string,
       long long, long long, bool, int, std::vector<long long>&,
       std::vector<DataReaderSparseParam>&, std::vector<std::string>&>(),
	     pybind11::arg("data_reader_type"),
       pybind11::arg("source"),
       pybind11::arg("eval_source"),
       pybind11::arg("check_type"),
       pybind11::arg("cache_eval_data") = 0,
       pybind11::arg("label_dim"),
       pybind11::arg("label_name"),
       pybind11::arg("dense_dim"),
       pybind11::arg("dense_name"),
       pybind11::arg("num_samples") = 0,
       pybind11::arg("eval_num_samples") = 0,
       pybind11::arg("float_label_dense") = true,
       pybind11::arg("num_workers") = 12,
       pybind11::arg("slot_size_array") = std::vector<long long>(),
       pybind11::arg("data_reader_sparse_param_array"),
       pybind11::arg("sparse_names"));
  pybind11::class_<HugeCTR::SparseEmbedding, std::shared_ptr<HugeCTR::SparseEmbedding>>(m, "SparseEmbedding")
    .def(pybind11::init<Embedding_t,
       size_t, size_t, int, std::string, std::string, std::vector<size_t>&>(),
	     pybind11::arg("embedding_type"),
       pybind11::arg("max_vocabulary_size_per_gpu"),
       pybind11::arg("embedding_vec_size"),
       pybind11::arg("combiner"),
       pybind11::arg("sparse_embedding_name"),
       pybind11::arg("bottom_name"),
       pybind11::arg("slot_size_array") = std::vector<size_t>());
  pybind11::class_<HugeCTR::DenseLayer, std::shared_ptr<HugeCTR::DenseLayer>>(m, "DenseLayer")
    .def(pybind11::init<Layer_t,
            std::vector<std::string>&, std::vector<std::string>&, float,
            float, Initializer_t, Initializer_t,
            float, float, size_t,
            Initializer_t, Initializer_t, int,
            size_t, bool, std::vector<int>&,
            std::vector<std::pair<int, int>>&, std::vector<size_t>&, size_t,
            int, std::vector<float>&, bool,
            Regularizer_t, float>(),
	     pybind11::arg("layer_type"),
       pybind11::arg("bottom_names"),
       pybind11::arg("top_names"),
	     pybind11::arg("factor") = 1.0,
       pybind11::arg("eps") =  0.00001,
       pybind11::arg("gamma_init_type") = Initializer_t::Default,
       pybind11::arg("beta_init_type") = Initializer_t::Default,
       pybind11::arg("dropout_rate") = 0.5,
       pybind11::arg("elu_alpha") = 1.0,
       pybind11::arg("num_output") = 1,
	     pybind11::arg("weight_init_type") = Initializer_t::Default,
       pybind11::arg("bias_init_type") = Initializer_t::Default,
       pybind11::arg("num_layers") = 0,
       pybind11::arg("leading_dim") = 1,
       pybind11::arg("selected") = false,
       pybind11::arg("selected_slots") = std::vector<int>(),
       pybind11::arg("ranges") = std::vector<std::pair<int, int>>(),
       pybind11::arg("weight_dims") = std::vector<size_t>(),
       pybind11::arg("out_dim") = 0,
       pybind11::arg("axis") = 1,
	     pybind11::arg("target_weight_vec") = std::vector<float>(),
       pybind11::arg("use_regularizer") = false,
       pybind11::arg("regularizer_type") = Regularizer_t::L1,
       pybind11::arg("lambda") = 0);
  pybind11::class_<HugeCTR::Model, std::shared_ptr<HugeCTR::Model>>(m, "Model")
    .def(pybind11::init<const SolverParser&, std::shared_ptr<OptParamsBase>&>(),
		   pybind11::arg("solver_parser"),
       pybind11::arg("opt_params"))
    .def("compile", &HugeCTR::Model::compile)
    .def("summary", &HugeCTR::Model::summary)
    .def("fit", &HugeCTR::Model::fit)
    .def("add", pybind11::overload_cast<Input&>(&HugeCTR::Model::add),
           pybind11::arg("input"))
    .def("add", pybind11::overload_cast<SparseEmbedding&>(&HugeCTR::Model::add),
           pybind11::arg("sparse_embedding"))
    .def("add", pybind11::overload_cast<DenseLayer&>(&HugeCTR::Model::add),
           pybind11::arg("dense_layer"));
}

} // namespace python_lib

} // namespace HugeCTR
