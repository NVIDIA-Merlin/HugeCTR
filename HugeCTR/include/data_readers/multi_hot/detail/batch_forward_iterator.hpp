#pragma once

#include <iterator>

namespace HugeCTR {

class IBatchLocations;
class BatchFileLocation;

class BatchForwardIterator {
 public:
  using value_type = BatchFileLocation;
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