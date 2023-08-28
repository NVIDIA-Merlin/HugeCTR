/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may no use this file except in compliance with the License.
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

#include <core23/buffer_channel_helpers.hpp>
#include <core23/logger.hpp>
#include <network_buffer_channels.hpp>
#include <unordered_map>
namespace HugeCTR {

namespace {

static std::unordered_map<NetworkBufferChannelType, std::string> g_type_to_name = {
    {NetworkBufferChannelType::Blobs, "BLOBS"},   {NetworkBufferChannelType::Weight, "WEIGHT"},
    {NetworkBufferChannelType::WeightHalf, "WH"}, {NetworkBufferChannelType::Wgrad, "WG"},
    {NetworkBufferChannelType::WgradHalf, "WGH"}, {NetworkBufferChannelType::OptState, "OPT"},
};

}  // namespace
std::string SetNetworkBufferChannel(NetworkBufferChannelType type, const std::string& new_name) {
  if (g_type_to_name.find(type) == g_type_to_name.end()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "There is no such BufferChannel type");
  }
  auto original = g_type_to_name[type];
  g_type_to_name[type] = new_name;
  return original;
}

std::string GetNetworkBufferChannel(NetworkBufferChannelType type) {
  if (g_type_to_name.find(type) == g_type_to_name.end()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "There is no such BufferChannel type");
  }

  return g_type_to_name[type];
}

core23::BufferChannel GetBlobsBufferChannel() {
  return GetNetworkBufferChannel(NetworkBufferChannelType::Blobs);
}
core23::BufferChannel GetWeightBufferChannel() {
  return GetNetworkBufferChannel(NetworkBufferChannelType::Weight);
}

core23::BufferChannel GetWeightHalfBufferChannel() {
  return GetNetworkBufferChannel(NetworkBufferChannelType::WeightHalf);
}

core23::BufferChannel GetWgradBufferChannel() {
  return GetNetworkBufferChannel(NetworkBufferChannelType::Wgrad);
}

core23::BufferChannel GetWgradHalfBufferChannel() {
  auto ret = GetNetworkBufferChannel(NetworkBufferChannelType::WgradHalf);
  return ret;
}
core23::BufferChannel GetOptStateBufferChannnel() {
  return GetNetworkBufferChannel(NetworkBufferChannelType::OptState);
}

}  // namespace HugeCTR
