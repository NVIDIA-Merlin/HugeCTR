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
#include <hps/dlpack.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <hps/hps_dlpack.hpp>

// this convertor will:
// 1) take a Tensor object and wrap it in the DLPack tensor
// 2) take a dlpack tensor and convert it to the HPS Tensor

namespace HugeCTR {
namespace python_lib {
HPSTensor fromDLPack(pybind11::capsule& dlpack_tensor) {
  DLManagedTensor* dlmt = static_cast<DLManagedTensor*>(dlpack_tensor);

  HCTR_CHECK_HINT(dlmt,
                  "from_dlpack received an invalid capsule. "
                  "Note that DLTensor capsules can be consumed only once, "
                  "so you might have already constructed a tensor from it once.");

  // atensor steals the ownership of the underlying storage. It also passes a
  // destructor function that will be called when the underlying storage goes
  // out of scope. When the destructor is called, the dlMTensor is destructed
  // too.
  auto atensor = HugeCTR::fromDLPack(dlmt);
  return atensor;
}

}  // namespace python_lib
}  // namespace HugeCTR