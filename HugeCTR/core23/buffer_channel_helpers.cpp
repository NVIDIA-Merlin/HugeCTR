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

#include <algorithm>
#include <core23/buffer_channel_helpers.hpp>
#include <core23/logger.hpp>
#include <random>
#include <string>

namespace HugeCTR {
namespace core23 {

std::string GetRandomBufferChannelName() {
  static const char alpha[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz"
      "0123456789";

  static std::random_device r;
  static std::default_random_engine e(r());
  static std::uniform_int_distribution<int> length_dist(32, 64);
  static std::uniform_int_distribution<int> alpha_dist(0, sizeof(alpha) - 1);

  std::string name(length_dist(e), 0);
  std::generate_n(name.begin(), name.length(), []() { return alpha[alpha_dist(e)]; });

  return name;
}

BufferChannel GetRandomBufferChannel() { return GetRandomBufferChannelName(); }

}  // namespace core23
}  // namespace HugeCTR
