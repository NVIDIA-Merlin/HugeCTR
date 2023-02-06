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

#include "embeddings/embedding_collection.hpp"

namespace HugeCTR {

namespace python_lib {

void EmbeddingCollectionPybind(pybind11::module &m) {
  pybind11::class_<EmbeddingTableConfig, std::shared_ptr<EmbeddingTableConfig>>(
      m, "EmbeddingTableConfig")
      .def(pybind11::init<const std::string &, int, int, std::optional<OptParams>,
                          std::optional<embedding::InitParams>>(),
           pybind11::arg("name"), pybind11::arg("max_vocabulary_size"), pybind11::arg("ev_size"),
           pybind11::arg("opt_params_or_empty") = std::nullopt,
           pybind11::arg("init_param_or_empty") = std::nullopt);
  pybind11::class_<HugeCTR::EmbeddingCollectionConfig,
                   std::shared_ptr<HugeCTR::EmbeddingCollectionConfig>>(m,
                                                                        "EmbeddingCollectionConfig")
      .def(pybind11::init<const std::string &, bool>(),
           pybind11::arg("output_layout") = "feature_major", pybind11::arg("indices_only") = false)
      .def("embedding_lookup", &HugeCTR::EmbeddingCollectionConfig::embedding_lookup,
           pybind11::arg("table_config"), pybind11::arg("bottom_name"), pybind11::arg("top_name"),
           pybind11::arg("combiner"))
      .def("shard", &HugeCTR::EmbeddingCollectionConfig::shard, pybind11::arg("shard_matrix"),
           pybind11::arg("shard_strategy"));
}
}  // namespace python_lib
}  // namespace HugeCTR