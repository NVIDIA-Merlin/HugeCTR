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

#include <HugeCTR/include/model_oversubscriber/model_oversubscriber.hpp>
#include <HugeCTR/pybind/model.hpp>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

namespace HugeCTR {

namespace python_lib {

namespace {

auto check_sparse_models = [](std::vector<std::string>& sparse_models) {
  if (sparse_models.empty()) {
    CK_THROW_(Error_t::WrongInput,
              "Must provide the path of sparse models, if:\n\
        \tTrain from scratch: please provide name(s) for the generated model after train;\n\
        \tTrain with existing model: please provide the path of existing sparse model(s);");
  }
  auto check_integrity = [](std::string sparse_model) {
    auto key_file{sparse_model + "/key"};
    auto vec_file{sparse_model + "/emb_vector"};
    if (fs::exists(key_file) && fs::exists(vec_file)) {
      if (fs::file_size(key_file) != 0 && fs::file_size(vec_file) != 0) {
        return true;
      } else {
        MESSAGE_(std::string("Wrong File: key or emb_vector is empty in ") + sparse_model);
        return false;
      }
    } else {
      MESSAGE_(std::string("Wrong File: key and emb_vector must exists in ") + sparse_model);
      return false;
    }
  };

  for (auto const& sparse_model : sparse_models) {
    if (fs::exists(sparse_model) && fs::is_directory(sparse_model) && !fs::is_empty(sparse_model)) {
      if (!check_integrity(sparse_model)) {
        CK_THROW_(Error_t::BrokenFile, std::string("Please check ") + sparse_model);
      }
      MESSAGE_(std::string("Use existing embedding: ") + sparse_model);
    } else {
      if (fs::exists(sparse_model) && !fs::is_directory(sparse_model)) {
        CK_THROW_(Error_t::BrokenFile,
                  std::string("File with the same name ") + sparse_model + " exists");
      }
      if (fs::exists(sparse_model) && fs::is_directory(sparse_model) &&
          fs::is_empty(sparse_model)) {
        fs::remove_all(sparse_model);
      }
      MESSAGE_(std::string("Empty embedding, trained table will be stored in ") + sparse_model);
    }
  }
};

}

HMemCacheConfig CreateHMemCache(size_t num_blocks, double target_hit_rate, size_t max_num_evict) {
  return HMemCacheConfig(num_blocks, target_hit_rate, max_num_evict);
}

std::shared_ptr<ModelOversubscriberParams> CreateMOS(std::vector<TrainPSType_t>& ps_types,
                                                     std::vector<std::string>& sparse_models,
                                                     std::vector<std::string>& local_paths,
                                                     std::vector<HMemCacheConfig>& hcache_configs) {
  std::shared_ptr<ModelOversubscriberParams> mos_params;
  check_sparse_models(sparse_models);

  size_t num_cache(std::count(ps_types.begin(), ps_types.end(), TrainPSType_t::Cached));
  if (num_cache != 0) {
    if (hcache_configs.size() == 1) {
      std::vector<HMemCacheConfig> tmp_hc_configs(ps_types.size());
      for (size_t i{0}; i < ps_types.size(); i++) {
        if (ps_types[i] == TrainPSType_t::Cached) {
          tmp_hc_configs[i] = hcache_configs[0];
        }
      }
      hcache_configs = std::move(tmp_hc_configs);
    } else if (ps_types.size() != hcache_configs.size()) {
      if (hcache_configs.size() != num_cache) {
        CK_THROW_(Error_t::WrongInput,
                  "hcache_configs.size() > 1: hcache_configs.size() should be equal to the num of "
                  "Cached PS");
      }

      auto cnt{0};
      std::vector<HMemCacheConfig> tmp_hc_configs(ps_types.size());
      for (size_t i{0}; i < ps_types.size(); i++) {
        if (ps_types[i] == TrainPSType_t::Cached) {
          tmp_hc_configs[i] = hcache_configs[cnt++];
        }
      }
      hcache_configs = std::move(tmp_hc_configs);
    } else if (num_cache == ps_types.size() && num_cache == hcache_configs.size()) {
      // do nothing
    } else {
      std::stringstream ss;
      ss << "Wrong hcache_configs:\n"
         << "  If hcache_configs.size() == 1, all Cached PS will use the same HMemCacheConfig\n"
         << "  Otherwise, hcache_configs.size() should be equal to the num of Cached PS";
      CK_THROW_(Error_t::WrongInput, ss.str());
    }
  }

  mos_params.reset(
      new ModelOversubscriberParams(ps_types, sparse_models, local_paths, hcache_configs));
  return mos_params;
}

void ModelOversubscriberPybind(pybind11::module& m) {
  m.def("CreateHMemCache", &HugeCTR::python_lib::CreateHMemCache, pybind11::arg("num_blocks"),
        pybind11::arg("target_hit_rate"), pybind11::arg("max_num_evict"));
  pybind11::class_<HugeCTR::HMemCacheConfig, std::shared_ptr<HugeCTR::HMemCacheConfig>>(
      m, "HMemCacheConfig");
  m.def("CreateMOS", &HugeCTR::python_lib::CreateMOS, pybind11::arg("ps_types"),
        pybind11::arg("sparse_models") = std::vector<std::string>(),
        pybind11::arg("local_paths") = std::vector<std::string>(),
        pybind11::arg("hmem_cache_configs") = std::vector<HMemCacheConfig>());
  pybind11::class_<HugeCTR::ModelOversubscriberParams,
                   std::shared_ptr<HugeCTR::ModelOversubscriberParams>>(
      m, "ModelOversubscriberParams");
  pybind11::class_<HugeCTR::ModelOversubscriber, std::shared_ptr<HugeCTR::ModelOversubscriber>>(
      m, "ModelOversubscriber")
      .def("update", pybind11::overload_cast<std::string&>(&HugeCTR::ModelOversubscriber::update),
           pybind11::arg("keyset_file"))
      .def(
          "update",
          pybind11::overload_cast<std::vector<std::string>&>(&HugeCTR::ModelOversubscriber::update),
          pybind11::arg("keyset_file_list"));
}

}  //  namespace python_lib

}  //  namespace HugeCTR
