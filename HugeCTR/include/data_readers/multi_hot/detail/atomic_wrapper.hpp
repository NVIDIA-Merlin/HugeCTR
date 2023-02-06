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