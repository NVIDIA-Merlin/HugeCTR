/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>

#include <common.hpp>
#include <core23/shape.hpp>
#include <utest/regularizers/regularizer_test_common.hpp>
#include <vector>

using namespace HugeCTR;

TEST(l1_regularizer_layer, 32x64_64x1) {
  test::regularizer_test_common(32, {{64, 1}}, 0.001, Regularizer_t::L1);
}

TEST(l1_regularizer_layer, 1024x64_64x256_256x1) {
  test::regularizer_test_common(1024, {{64, 256}, {256, 1}}, 0.001, Regularizer_t::L1);
}