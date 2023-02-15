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

#include <cstdlib>
#include <limits>
#include <memory>
#include <new>

/**
 * Will allocate aligned memory in std::vectors, std::unique_ptr and such.
 *
 * Originally, inspired by mxmlnkn's answer to
 * https://stackoverflow.com/questions/60169819/modern-approach-to-making-stdvector-allocate-aligned-memory
 * , but contains quite a couple of extensions.
 *
 * By default, will align to x86 cache lines, which are 64 bytes long.
 */
template <typename T, std::size_t ALIGNMENT = 64>
struct AlignedAllocator {
  using value_type = T;
  using size_type = std::size_t;

  static constexpr size_type alignment{ALIGNMENT};
  static_assert(alignment >=
                alignof(value_type));  // Allocation alignment must exeed value alignment.
  static_assert(!(alignment & (alignment - 1)));  // Ensure alignment is power of 2.

  using unique_ptr_type = std::unique_ptr<value_type, AlignedAllocator<value_type, alignment>>;
  using shared_ptr_type = std::shared_ptr<value_type>;

  /**
   * This is necessary because AlignedAllocator has a second template argument for the alignment
   * that will make the default std::allocator_traits implementation fail during compilation
   * otherwise.
   * @see https://stackoverflow.com/a/48062758/2191065
   */
  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, alignment>;
  };

  constexpr AlignedAllocator() noexcept = default;

  // Copy construction.
  constexpr AlignedAllocator(const AlignedAllocator&) noexcept = default;
  constexpr AlignedAllocator& operator=(const AlignedAllocator&) noexcept = default;
  template <typename U>
  constexpr AlignedAllocator(const AlignedAllocator<U, alignment>&) noexcept {}

  // Move construction.
  constexpr AlignedAllocator(AlignedAllocator&&) noexcept = default;
  constexpr AlignedAllocator& operator=(AlignedAllocator&&) noexcept = default;
  template <typename U>
  constexpr AlignedAllocator(AlignedAllocator<U, alignment>&&) noexcept {}

  /**
   * Allocation function used by std::vector.
   */
  [[nodiscard]] inline static value_type* allocate(size_type n = 1) {
    // Must allocate multipel of alignment.
    n = next_aligned_boundary(n * sizeof(value_type));
    if (n > std::numeric_limits<size_type>::max() / alignment) {
      throw std::bad_array_new_length();
    }
    n *= alignment;

    auto p{static_cast<value_type*>(std::aligned_alloc(alignment, n))};
    if (p) {
      return p;
    }

    throw std::bad_alloc();
  }

  /**
   * Deallocation function used by std::vector.
   */
  inline static void deallocate(value_type* p, size_type n) noexcept { std::free(p); }

  /**
   * Optional function suggested by standard.
   */
  inline static size_type max_size() {
    return (std::numeric_limits<size_type>::max() - alignment + 1) / sizeof(value_type);
  }

  inline static size_type next_aligned_boundary(const size_type n) {
    return (n + alignment - 1) / alignment;
  }

  /**
   * Optional function suggested by standard. Means `this` aligned allocator can deallocate memory
   * that was originally allocated by another aligned allocator.
   */
  template <typename U, size_type OTHER_ALIGNMENT>
  inline bool operator==(const AlignedAllocator<U, OTHER_ALIGNMENT>&) {
    return true;
  }
  template <typename U, size_type OTHER_ALIGNMENT>
  inline bool operator!=(const AlignedAllocator<U, OTHER_ALIGNMENT>&) {
    return false;
  }

  /**
   * Deallocation function used by std::unique_ptr.
   */
  inline void operator()(value_type* p) noexcept { std::free(p); }

  /**
   * Wraps the aligned memory pointer in a std::unique_ptr.
   */
  inline static unique_ptr_type make_unique(size_type n = 1) {
    return unique_ptr_type{allocate(n)};
  }

  /**
   * Wraps the aligned memory pointer in a std:shared_ptr
   */
  inline static shared_ptr_type make_shared(size_type n = 1) { return {allocate(n), std::free}; }
};
