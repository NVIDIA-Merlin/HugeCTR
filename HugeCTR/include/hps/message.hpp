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

#include <common.hpp>
#include <functional>

namespace HugeCTR {

/**
 * Each instance represents an emitter link to a theoretically infinitely sized message queue..
 *
 * @tparam TKey Data-type to be used for keys in this message queue.
 */
template <typename TKey>
class MessageSink {
 public:
  DISALLOW_COPY_AND_MOVE(MessageSink);

  MessageSink() = default;

  virtual ~MessageSink() = default;

  /**
   * Emit a message to append a key/value pair to the queue.
   *
   * @param tag A tag under which the key/value pair should be filed.
   * @param key The value of the key.
   * @param value Pointer to the value.
   * @param value_size The size of the value in bytes.
   */
  virtual void post(const std::string& tag, const TKey& key, const char* value,
                    size_t value_size) = 0;

  /**
   * Emit multiple key/value pairs to the queue.
   *
   * @param tag A tag under which the key/value pairs should be filed.
   * @param num_pairs The number of \p keys and \p values .
   * @param keys Pointer to the keys.
   * @param values Pointer to the values.
   * @param value_size The size of each value in bytes.
   */
  virtual void post(const std::string& tag, size_t num_pairs, const TKey* keys, const char* values,
                    size_t value_size);
};

#define HCTR_MESSAGE_SOURCE_CALLBACK \
  bool(const std::string&, const size_t, const TKey*, const char*, const size_t)

/**
 * Each instance represents an consumer link to a theoretically infinitely sized message queue..
 *
 * @tparam TKey Data-type to be used for keys in this message queue.
 */
template <typename TKey>
class MessageSource {
 public:
  DISALLOW_COPY_AND_MOVE(MessageSource);

  MessageSource() = default;

  virtual ~MessageSource() = default;

  /**
   * Start listening to the message queue and invoke function for each message received.
   *
   * @param callback Callback function be invoked for each received message.
   */
  virtual void enable(std::function<HCTR_MESSAGE_SOURCE_CALLBACK> callback) = 0;
};

}  // namespace HugeCTR