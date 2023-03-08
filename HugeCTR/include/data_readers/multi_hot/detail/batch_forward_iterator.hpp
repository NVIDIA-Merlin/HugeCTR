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

#include <iterator>

namespace HugeCTR {

class IBatchLocations;
class BatchDescriptor;

class BatchForwardIterator {
 public:
  using value_type = BatchDescriptor;
  using reference = value_type&;
  using const_reference = const value_type&;
  using iterator_category = std::forward_iterator_tag;
  using difference_type = size_t;

  explicit BatchForwardIterator(IBatchLocations* locations, size_t current_batch);

  value_type operator*() const;

  // Pre- and post-incrementable
  BatchForwardIterator& operator++();
  BatchForwardIterator operator++(int);

  // Equality / inequality
  bool operator==(const BatchForwardIterator& rhs);
  bool operator!=(const BatchForwardIterator& rhs);

  difference_type operator-(const BatchForwardIterator& rhs);

 private:
  IBatchLocations* locations_;
  size_t current_;
};
}  // namespace HugeCTR