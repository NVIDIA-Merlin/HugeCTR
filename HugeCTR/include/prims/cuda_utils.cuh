/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <math_constants.h>
#include <stdint.h>
#include <cuml/common/utils.hpp>

namespace MLCommon {

/** helper macro for device inlined functions */
#define DI inline __device__

template <typename ReduceLambda>
DI void myAtomicReduce(__half *address, __half val, ReduceLambda op) {
  // float *address_f = address;
  float val_f = val;
  unsigned int *address_as_uint = (unsigned int *)address;
  unsigned int old = *address_as_uint, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_uint, assumed, __float_as_uint(op(val_f, __uint_as_float(assumed))));
  } while (assumed != old);
}

}  // namespace MLCommon