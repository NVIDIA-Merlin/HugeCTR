/*
 * @Author: your name
 * @Date: 2020-11-06 15:00:46
 * @LastEditTime: 2020-11-06 16:24:53
 * @LastEditors: your name
 * @Description: In User Settings Edit
 * @FilePath: /hugectr/HugeCTR/pybind/module_main.cpp
 */
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
#include <HugeCTR/pybind/common_wrapper.hpp>
#include <HugeCTR/pybind/third_party_wrapper.hpp>
#include <HugeCTR/pybind/utils_wrapper.hpp>
#include <HugeCTR/pybind/csr_chunk_wrapper.hpp>
#include <HugeCTR/pybind/device_map_wrapper.hpp>
#include <HugeCTR/pybind/learning_rate_scheduler_wrapper.hpp>
#include <HugeCTR/pybind/metrics_wrapper.hpp>
#include <HugeCTR/pybind/mmap_offset_wrapper.hpp>
#include <HugeCTR/pybind/resource_manager_wrapper.hpp>
#include <HugeCTR/pybind/solver_parser_wrapper.hpp>
#include <HugeCTR/pybind/data_reader_wrapper.hpp>
#include <HugeCTR/pybind/model_oversubscriber_wrapper.hpp>
#include <HugeCTR/pybind/session_wrapper.hpp>
using namespace HugeCTR::python_lib;

PYBIND11_MODULE(hugectr, m) {
  m.doc() = "hugectr python interface";
  CommonPybind(m);
  ThirdPartyPybind(m);
  UtilsPybind(m);
  CSRChunkPybind(m);
  DeviceMapPybind(m);
  LearningRateSchedulerPybind(m);
  MetricsPybind(m);
  MmapOffsetPybind(m);
  ResourceManagerPybind(m);
  SolverParserPybind(m);
  DataReaderPybind(m);
  ModelOversubscriberPybind(m);
  SessionPybind(m);
}
