#include "data_readers/multi_hot/detail/batch_forward_iterator.hpp"
#include "data_readers/multi_hot/detail/batch_locations.hpp"

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