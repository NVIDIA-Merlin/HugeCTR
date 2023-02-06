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

#include <hps/database_backend_detail.hpp>
#include <hps/inference_utils.hpp>
#include <thread_pool.hpp>
#include <type_traits>

namespace HugeCTR {

// TODO: Remove me!
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wconversion"

/**
 * HashMap Backend / Contains
 */
#ifdef HCTR_HPS_HASH_MAP_CONTAINS_
#error HCTR_HPS_HASH_MAP_CONTAINS_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_HASH_MAP_CONTAINS_(MODE)                                               \
  [&]() {                                                                               \
    HCTR_HPS_DB_APPLY_(MODE, hit_count += part.entries.find(*k) != part.entries.end()); \
    return true;                                                                        \
  }()

/**
 * HashMap Backend / Evict
 */
#ifdef HCTR_HPS_HASH_MAP_EVICT_K_
#error HCTR_HPS_HASH_MAP_EVICT_K_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_HASH_MAP_EVICT_K_()                                \
  do {                                                              \
    static_assert(std::is_same_v<decltype(num_deletions), size_t>); \
    static_assert(std::is_same_v<decltype(k), const Key*> ||        \
                  std::is_same_v<decltype(k), const Key* const>);   \
                                                                    \
    const auto& it{part.entries.find(*k)};                          \
    if (it != part.entries.end()) {                                 \
      const Payload& payload{it->second};                           \
                                                                    \
      /* Stash pointer and reference in map. */                     \
      part.value_slots.emplace_back(payload.value);                 \
      part.entries.erase(it);                                       \
      ++num_deletions;                                              \
    }                                                               \
  } while (0)

#ifdef HCTR_HPS_HASH_MAP_EVICT_
#error HCTR_HPS_HASH_MAP_EVICT_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_HASH_MAP_EVICT_(MODE)                      \
  [&]() {                                                   \
    HCTR_HPS_DB_APPLY_(MODE, HCTR_HPS_HASH_MAP_EVICT_K_()); \
    return true;                                            \
  }()

/**
 * HashMap Backend / Fetch
 */
#ifdef HCTR_HPS_HASH_MAP_FETCH_IMPL_
#error HCTR_HPS_HASH_MAP_FETCH_IMPL_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_HASH_MAP_FETCH_IMPL_(...)                                                   \
  do {                                                                                       \
    static_assert(std::is_same_v<decltype(miss_count), size_t>);                             \
    static_assert(std::is_invocable_v<decltype(on_miss), size_t>);                           \
    static_assert(std::is_same_v<decltype(value_stride), const size_t>);                     \
    static_assert(std::is_same_v<decltype(k), const Key*> ||                                 \
                  std::is_same_v<decltype(k), const Key* const>);                            \
    static_assert(std::is_same_v<decltype(values), char* const>);                            \
                                                                                             \
    const auto& it{part.entries.find(*k)};                                                   \
    if (it != part.entries.end()) {                                                          \
      Payload& payload{it->second};                                                          \
                                                                                             \
      /* Race-conditions here are deliberately ignored because insignificant in practice. */ \
      __VA_ARGS__;                                                                           \
      std::copy_n(payload.value, part.value_size, &values[(k - keys) * value_stride]);       \
    } else {                                                                                 \
      on_miss(k - keys);                                                                     \
      ++miss_count;                                                                          \
    }                                                                                        \
  } while (0)

#ifdef HCTR_HPS_HASH_MAP_FETCH_
#error HCTR_HPS_HASH_MAP_FETCH_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_HASH_MAP_FETCH_(MODE)                                                        \
  [&]() {                                                                                     \
    static_assert(std::is_same_v<decltype(overflow_policy), const DatabaseOverflowPolicy_t>); \
                                                                                              \
    switch (overflow_policy) {                                                                \
      case DatabaseOverflowPolicy_t::EvictRandom: {                                           \
        HCTR_HPS_DB_APPLY_(MODE, HCTR_HPS_HASH_MAP_FETCH_IMPL_());                            \
      } break;                                                                                \
      case DatabaseOverflowPolicy_t::EvictLeastUsed: {                                        \
        HCTR_HPS_DB_APPLY_(MODE, HCTR_HPS_HASH_MAP_FETCH_IMPL_(++payload.access_count));      \
      } break;                                                                                \
      case DatabaseOverflowPolicy_t::EvictOldest: {                                           \
        const time_t now{std::time(nullptr)};                                                 \
        HCTR_HPS_DB_APPLY_(MODE, HCTR_HPS_HASH_MAP_FETCH_IMPL_(payload.last_access = now));   \
      } break;                                                                                \
    }                                                                                         \
    return true;                                                                              \
  }()

/**
 * HashMap Backend / Insert
 */
#ifdef HCTR_HPS_HASH_MAP_INSERT_IMPL_
#error HCTR_HPS_HASH_MAP_INSERT_IMPL_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_HASH_MAP_INSERT_IMPL_(...)                                                  \
  do {                                                                                       \
    static_assert(std::is_same_v<decltype(num_inserts), size_t>);                            \
    static_assert(std::is_same_v<decltype(value_size), const uint32_t>);                     \
    static_assert(std::is_same_v<decltype(value_stride), const size_t>);                     \
    static_assert(std::is_same_v<decltype(k), const Key*> ||                                 \
                  std::is_same_v<decltype(k), const Key* const>);                            \
    static_assert(std::is_same_v<decltype(values), const char* const>);                      \
                                                                                             \
    const auto& res{part.entries.try_emplace(*k)};                                           \
    Payload& payload{res.first->second};                                                     \
                                                                                             \
    __VA_ARGS__;                                                                             \
                                                                                             \
    /* If new insertion. */                                                                  \
    if (res.second) {                                                                        \
      /* If no free space, allocate another buffer, and fill pointer queue. */               \
      if (part.value_slots.empty()) {                                                        \
        const size_t stride{(value_size + value_page_alignment - 1) / value_page_alignment * \
                            value_page_alignment};                                           \
        const size_t num_values{part.allocation_rate / stride};                              \
        HCTR_CHECK(num_values > 0);                                                          \
                                                                                             \
        /* Get more memory. */                                                               \
        part.value_pages.emplace_back(num_values* stride, char_allocator_);                  \
        ValuePage& value_page{part.value_pages.back()};                                      \
                                                                                             \
        /* Stock up slot references. */                                                      \
        part.value_slots.reserve(part.value_slots.size() + num_values);                      \
        for (auto it{value_page.end()}; it != value_page.begin();) {                         \
          it -= stride;                                                                      \
          part.value_slots.emplace_back(&*it);                                               \
        }                                                                                    \
      }                                                                                      \
                                                                                             \
      /* Fetch storage slot. */                                                              \
      payload.value = part.value_slots.back();                                               \
      part.value_slots.pop_back();                                                           \
      ++num_inserts;                                                                         \
    }                                                                                        \
                                                                                             \
    std::copy_n(&values[(k - keys) * value_stride], value_size, payload.value);              \
  } while (0)

/**
 * HashMap Backend / Insert
 */
#ifdef HCTR_HPS_HASH_MAP_INSERT_
#error HCTR_HPS_HASH_MAP_INSERT_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_HASH_MAP_INSERT_(MODE)                                                       \
  [&]() {                                                                                     \
    static_assert(std::is_same_v<decltype(overflow_policy), const DatabaseOverflowPolicy_t>); \
                                                                                              \
    switch (overflow_policy) {                                                                \
      case DatabaseOverflowPolicy_t::EvictRandom: {                                           \
        HCTR_HPS_DB_APPLY_(MODE, HCTR_HPS_HASH_MAP_INSERT_IMPL_());                           \
      } break;                                                                                \
      case DatabaseOverflowPolicy_t::EvictLeastUsed: {                                        \
        HCTR_HPS_DB_APPLY_(MODE, HCTR_HPS_HASH_MAP_INSERT_IMPL_(payload.access_count = 0));   \
      } break;                                                                                \
      case DatabaseOverflowPolicy_t::EvictOldest: {                                           \
        const time_t now{std::time(nullptr)};                                                 \
        HCTR_HPS_DB_APPLY_(MODE, HCTR_HPS_HASH_MAP_INSERT_IMPL_(payload.last_access = now));  \
      } break;                                                                                \
    }                                                                                         \
    return true;                                                                              \
  }()

// TODO: Remove me!
#pragma GCC diagnostic pop

}  // namespace HugeCTR