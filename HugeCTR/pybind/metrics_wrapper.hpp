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
#include <HugeCTR/include/metrics.hpp>

namespace HugeCTR {

namespace python_lib {

void MetricsPybind(pybind11::module& m) {
  pybind11::enum_<HugeCTR::metrics::RawType>(m, "MetricsRawType")
      .value("Loss", HugeCTR::metrics::RawType::Loss)
      .value("Pred", HugeCTR::metrics::RawType::Pred)
      .value("Label", HugeCTR::metrics::RawType::Label)
      .export_values();
  pybind11::enum_<HugeCTR::metrics::Type>(m, "MetricsType")
      .value("AUC", HugeCTR::metrics::Type::AUC)
      .value("AverageLoss", HugeCTR::metrics::Type::AverageLoss)
      .export_values();
}

}  // namespace python_lib

}  // namespace HugeCTR
