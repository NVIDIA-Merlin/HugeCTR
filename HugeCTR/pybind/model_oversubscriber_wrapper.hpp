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
#include <HugeCTR/include/model_oversubscriber/model_oversubscriber.hpp>
#include <HugeCTR/pybind/model.hpp>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

namespace HugeCTR {

namespace python_lib {

namespace {

auto remove_prefix(const std::string& path) {
  size_t found = path.rfind("/");
  if (found != std::string::npos)
    return std::string(path, found + 1);
  else
    return path;
}

}

std::shared_ptr<ModelOversubscriberParams> CreateMOS(bool train_from_scratch,
                                                  std::vector<std::string>& trained_sparse_models,
                                                  std::vector<std::string>& dest_sparse_models) {
  std::shared_ptr<ModelOversubscriberParams> mos_params;
  if (train_from_scratch) {
    if (dest_sparse_models.empty()) {
      CK_THROW_(Error_t::WrongInput, "no destination provided for model oversubscriber to save sparse models");
    }
    std::for_each(dest_sparse_models.begin(), dest_sparse_models.end(),
      [](const std::string& sparse_model) {
        if (fs::exists(sparse_model) && fs::is_directory(sparse_model) && !fs::is_empty(sparse_model)) {
          if(fs::file_size(sparse_model + "/" + remove_prefix(sparse_model) + ".key") != 0) {
            CK_THROW_(Error_t::WrongInput, sparse_model + " exist and not empty, please use another name");
          } else {
            fs::remove_all(sparse_model);
          }
        }
    });
  } else {
    if (trained_sparse_models.empty()) {
      CK_THROW_(Error_t::WrongInput, "no trained sparse models provided for model oversubscriber");
    }
    std::for_each(trained_sparse_models.begin(), trained_sparse_models.end(),
      [](const std::string& sparse_model) {
        if (!fs::exists(sparse_model) || fs::is_empty(sparse_model))
          CK_THROW_(Error_t::WrongInput,
              sparse_model + " not exist or empty, but train_from_scratch is set to false");
    });
  }
  mos_params.reset(new ModelOversubscriberParams(train_from_scratch, trained_sparse_models, dest_sparse_models));
  return mos_params;
}

void ModelOversubscriberPybind(pybind11::module &m) {
  m.def("CreateMOS", &HugeCTR::python_lib::CreateMOS,
    pybind11::arg("train_from_scratch"),
    pybind11::arg("trained_sparse_models") = std::vector<std::string>(),
    pybind11::arg("dest_sparse_models") = std::vector<std::string>());
  pybind11::class_<HugeCTR::ModelOversubscriberParams, std::shared_ptr<HugeCTR::ModelOversubscriberParams>>(m, "ModelOversubscriberParams");
  pybind11::class_<HugeCTR::ModelOversubscriber, std::shared_ptr<HugeCTR::ModelOversubscriber>>(m, "ModelOversubscriber")
   .def("store", &HugeCTR::ModelOversubscriber::store)
   .def("update", pybind11::overload_cast<std::string&>(&HugeCTR::ModelOversubscriber::update),
        pybind11::arg("keyset_file"))
   .def("update", pybind11::overload_cast<std::vector<std::string>&>(&HugeCTR::ModelOversubscriber::update),
        pybind11::arg("keyset_file_list"));
}

}  //  namespace python_lib

}  //  namespace HugeCTR

