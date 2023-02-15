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

#include <data_readers/multi_hot/detail/batch_forward_iterator.hpp>
#include <data_readers/multi_hot/detail/batch_locations.hpp>

namespace HugeCTR {

BatchForwardIterator::BatchForwardIterator(IBatchLocations *locations, size_t current_batch)
    : locations_(locations), current_(current_batch) {}

BatchForwardIterator::value_type BatchForwardIterator::operator*() const {
  return locations_->at(current_);
}

BatchForwardIterator &BatchForwardIterator::operator++() {
  ++current_;
  return *this;
}

BatchForwardIterator BatchForwardIterator::operator++(int) {
  BatchForwardIterator tmp = *this;
  ++current_;
  return tmp;
}

bool BatchForwardIterator::operator==(const BatchForwardIterator &rhs) {
  return current_ == rhs.current_;
}

bool BatchForwardIterator::operator!=(const BatchForwardIterator &rhs) { return !(*this == rhs); }

BatchForwardIterator::difference_type BatchForwardIterator::operator-(
    const BatchForwardIterator &rhs) {
  return current_ - rhs.current_;
}

}  // namespace HugeCTR