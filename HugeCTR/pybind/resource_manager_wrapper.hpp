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
#include <HugeCTR/include/cpu_resource.hpp>
#include <HugeCTR/include/gpu_resource.hpp>
#include <HugeCTR/include/resource_manager.hpp>

namespace HugeCTR {

namespace python_lib {

void ResourceManagerPybind(pybind11::module& m) {
  pybind11::class_<HugeCTR::GPUResource, std::shared_ptr<HugeCTR::GPUResource>>(m, "GPUResource")
      .def(pybind11::init<int, size_t, unsigned long long>(), pybind11::arg("device_id"),
           pybind11::arg("global_gpu_id"), pybind11::arg("seed"));
  pybind11::class_<HugeCTR::CPUResource, std::shared_ptr<HugeCTR::CPUResource>>(m, "CPUResource")
      .def(pybind11::init<unsigned long long, size_t>(), pybind11::arg("seed"),
           pybind11::arg("thread_num"));
  pybind11::class_<HugeCTR::ResourceManager, std::shared_ptr<HugeCTR::ResourceManager>>(
      m, "ResourceManager")
      .def_static("create", &HugeCTR::ResourceManager::create, pybind11::arg("visible_devices"),
                  pybind11::arg("seed"))
      .def("get_num_process", &HugeCTR::ResourceManager::get_num_process)
      .def("get_pid", &HugeCTR::ResourceManager::get_pid)
      .def("get_local_cpu", &HugeCTR::ResourceManager::get_local_cpu)
      .def("get_local_gpu", &HugeCTR::ResourceManager::get_local_gpu, pybind11::arg("local_gpu_id"))
      .def("get_local_gpu_device_id_list", &HugeCTR::ResourceManager::get_local_gpu_device_id_list)
      .def("get_local_gpu_count", &HugeCTR::ResourceManager::get_local_gpu_count)
      .def("get_global_gpu_count", &HugeCTR::ResourceManager::get_global_gpu_count)
      .def("get_pid_fron_gpu_global_id", &HugeCTR::ResourceManager::get_pid_from_gpu_global_id,
           pybind11::arg("global_gpu_id"))
      .def("get_gpu_local_id_from_global_id",
           &HugeCTR::ResourceManager::get_gpu_local_id_from_global_id,
           pybind11::arg("global_gpu_id"))
      .def("p2p_enabled", &HugeCTR::ResourceManager::p2p_enabled, pybind11::arg("src_dev"),
           pybind11::arg("dst_dev"))
      .def("all_p2p_enabled", &HugeCTR::ResourceManager::all_p2p_enabled);
}

}  // namespace python_lib

}  // namespace HugeCTR
