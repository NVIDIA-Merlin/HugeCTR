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
  pybind11::class_<embedding::EmbeddingTableParam, std::shared_ptr<embedding::EmbeddingTableParam>>(
      m, "EmbeddingTableConfig")
      .def(pybind11::init<int, int, int, std::optional<OptParams>, Initializer_t, float, float>(),
           pybind11::arg("table_id"), pybind11::arg("max_vocabulary_size"),
           pybind11::arg("ev_size"), pybind11::arg("opt_params_py") = std::nullopt,
           pybind11::arg("initializer_type") = Initializer_t::Default,
           pybind11::arg("up_bound") = -1, pybind11::arg("max_sequence_len") = -1);
  pybind11::class_<HugeCTR::EmbeddingPlanner, std::shared_ptr<HugeCTR::EmbeddingPlanner>>(
      m, "EmbeddingPlanner")
      .def(pybind11::init())
      .def("embedding_lookup", &HugeCTR::EmbeddingPlanner::embedding_lookup,
           pybind11::arg("table_config"), pybind11::arg("bottom_name"), pybind11::arg("top_name"),
           pybind11::arg("combiner"))
      .def("create_embedding_collection", &HugeCTR::EmbeddingPlanner::create_embedding_collection,
           pybind11::arg("shard_matrix"), pybind11::arg("emb_table_group_strategy"),
           pybind11::arg("emb_table_placement_strategy"));
}
}  // namespace python_lib
}  // namespace HugeCTR