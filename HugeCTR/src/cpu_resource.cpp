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

#include <common.hpp>
#include <cpu_resource.hpp>
#include <utils.hpp>

namespace HugeCTR {
CPUResource::CPUResource(unsigned long long seed, size_t thread_num) {
  CK_CURAND_THROW_(curandCreateGeneratorHost(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  CK_CURAND_THROW_(curandSetPseudoRandomGeneratorSeed(curand_generator_, seed));
}

CPUResource::~CPUResource() { curandDestroyGenerator(curand_generator_); }
}  // namespace HugeCTR