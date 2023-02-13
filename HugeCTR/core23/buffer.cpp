/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <base/debug/logger.hpp>
#include <core23/buffer.hpp>
#include <core23/buffer_client.hpp>
#include <core23/device_guard.hpp>
#include <core23/offsetted_buffer.hpp>

namespace HugeCTR {

namespace core23 {

Buffer::~Buffer() {
  HCTR_THROW_IF(served_client_requirements_.size() != 0 || new_client_requirements_.size() != 0,
                HugeCTR::Error_t::IllegalCall,
                "There must be no remaining clients in destructing a Buffer");
  HCTR_THROW_IF(client_offsets_.size() != 0, HugeCTR::Error_t::IllegalCall,
                "The ClientOffset must be empty in destructing a Buffer");
}

void Buffer::subscribe(BufferClient* client, BufferRequirements requirements) {
  if (subscribable()) {
    if (served_client_requirements_.find(client) != served_client_requirements_.end()) {
      HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                     "The client is already subscribing the buffer.");
    }
    new_client_requirements_[client] = requirements;
    post_subscribe(client, requirements);
    client->on_subscribe(std::shared_ptr<OffsettedBuffer>(
        new OffsettedBuffer(shared_from_this(), {}), [client, this](OffsettedBuffer* ob) {
          unsubscribe(client);
          delete ob;
        }));
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "The buffer cannot be subscribed.");
  }
}
void Buffer::unsubscribe(BufferClient* client) {
  if (served_client_requirements_.size() == 0 && new_client_requirements_.size() == 0) {
    HCTR_LOG_S(WARNING, ROOT) << "The buffer has never been subscribed." << std::endl;
    return;
  }

  auto it0 = served_client_requirements_.find(client);
  if (it0 != served_client_requirements_.end()) {
    auto requirements = it0->second;
    served_client_requirements_.erase(it0);

    auto it_co = client_offsets_.find(client);
    auto offset = it_co->second;
    client_offsets_.erase(it_co);
    post_unsubscribe(client, requirements, offset);
  } else {
    auto it1 = new_client_requirements_.find(client);
    if (it1 != new_client_requirements_.end()) {
      auto requirements = it1->second;
      new_client_requirements_.erase(it1);
    } else {
      HCTR_LOG_S(WARNING, ROOT) << "The client has never subscribed the buffer." << std::endl;
    }
  }
  client->on_unsubscribe(OffsettedBuffer(nullptr, {}));
}

void Buffer::allocate() {
  DeviceGuard device_guard(device_);

  if (!allocatable()) {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "The buffer is not at a state to do allocate().");
  }

  auto new_client_offsets = do_allocate(allocator_, new_client_requirements_);
  for (auto [client, offset] : new_client_offsets) {
    if (client) {
      client->on_allocate(OffsettedBuffer(shared_from_this(), offset, {}));
    }
  }

  client_offsets_.merge(new_client_offsets);
  served_client_requirements_.merge(new_client_requirements_);
}

}  // namespace core23

}  // namespace HugeCTR
