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
#pragma once

#include <atomic>

template <typename T>
struct AtomicWrapper {
  std::atomic<T> raw;

  AtomicWrapper() : raw() {}

  AtomicWrapper(const std::atomic<T> &a) : raw(a.load()) {}

  AtomicWrapper(const AtomicWrapper &other) : raw(other.raw.load()) {}

  AtomicWrapper &operator=(const AtomicWrapper &other) {
    raw.store(other.raw.load());
    return *this;
  }

  AtomicWrapper &operator=(T value) {
    raw.store(value);
    return *this;
  }

  AtomicWrapper &operator++() {
    raw++;
    return *this;
  }

  AtomicWrapper operator++(int)  // post-increment
  {
    AtomicWrapper<size_t> copy(*this);
    ++(*this);  // pre-increment here seems logical
    return copy;
  }
};