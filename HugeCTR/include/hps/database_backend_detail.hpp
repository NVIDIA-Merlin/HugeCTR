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

#include <rocksdb/db.h>

#include <cstdint>
#include <type_traits>

namespace HugeCTR {

/**
 * Catch-all as described in https://en.cppreference.com/w/cpp/language/if .
 */
template <typename>
inline constexpr bool dependent_false_v{false};

// TODO: Remove me!
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wconversion"

/**
 * 64 bit left shift operation.
 */
inline uint64_t rotl64(const uint64_t x, const int n) {
  return (x << n) | (x >> (8 * sizeof(uint64_t) - n));
}

/**
 * 64 bit right shift operation.
 */
inline uint64_t rotr64(const uint64_t x, const int n) {
  return (x >> n) | (x << (8 * sizeof(uint64_t) - n));
}

/**
 * A fairly strong but simple public domain numeric mixer by Pelle Evensen.
 * https://mostlymangling.blogspot.com/2019/01/better-stronger-mixer-and-test-procedure.html
 */
inline uint64_t rrxmrrxmsx_0(uint64_t x) {
  x ^= rotr64(x, 25) ^ rotr64(x, 50);
  x *= UINT64_C(0xA24BAED4963EE407);
  x ^= rotr64(x, 24) ^ rotr64(x, 49);
  x *= UINT64_C(0x9FB21C651E98DF25);
  return x ^ x >> 28;
}

#ifdef HCTR_HPS_KEY_TO_PART_INDEX_
#error HCTR_HPS_KEY_TO_PART_INDEX_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_KEY_TO_PART_INDEX_(KEY) (rrxmrrxmsx_0(KEY) % num_partitions)

/**
 * Time budget checking and resolution.
 */
#ifdef HCTR_HPS_DB_CHECK_TIME_BUDGET_
#error HCTR_HPS_DB_EVAL_TIME_BUDGET_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_DB_CHECK_TIME_BUDGET_(MODE, MISS_OP)                          \
  do {                                                                         \
    if (time_budget != std::chrono::nanoseconds::zero()) {                     \
      elapsed = std::chrono::high_resolution_clock::now() - begin;             \
      if (elapsed >= time_budget) {                                            \
        HCTR_LOG_C(WARNING, WORLD, get_name(), " backend; Table ", table_name, \
                   ": Timeout = ", elapsed.count(), " ns!\n");                 \
                                                                               \
        HCTR_HPS_DB_HANDLE_TIMEOUT_##MODE##_(MISS_OP);                         \
      }                                                                        \
    }                                                                          \
  } while (0)

#ifdef HCTR_HPS_DB_HANDLE_TIMEOUT_SEQUENTIAL_DIRECT_
#error HCTR_HPS_DB_HANDLE_TIMEOUT_SEQUENTIAL_DIRECT_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_DB_HANDLE_TIMEOUT_SEQUENTIAL_DIRECT_(MISS_OP)        \
  do {                                                                \
    skip_count += keys_end - k;                                       \
                                                                      \
    if constexpr (std::is_invocable_v<decltype(MISS_OP), size_t>) {   \
      /* Called by `fetch` functions. */                              \
      for (; k != keys_end; ++k) {                                    \
        MISS_OP(k - keys);                                            \
      }                                                               \
    } else if constexpr (std::is_null_pointer_v<decltype(MISS_OP)>) { \
      /* Called by `contains` functions. */                           \
      /* Do nothing ;-) */                                            \
    } else {                                                          \
      static_assert(dependent_false_v<decltype(MISS_OP)>);            \
    }                                                                 \
  } while (0)

#ifdef HCTR_HPS_DB_HANDLE_TIMEOUT_SEQUENTIAL_INDIRECT_
#error HCTR_HPS_DB_HANDLE_TIMEOUT_SEQUENTIAL_INDIRECT_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_DB_HANDLE_TIMEOUT_SEQUENTIAL_INDIRECT_(MISS_OP)      \
  do {                                                                \
    skip_count += indices_end - i;                                    \
                                                                      \
    if constexpr (std::is_invocable_v<decltype(MISS_OP), size_t>) {   \
      /* Called by `fetch` functions. */                              \
      for (; i != indices_end; ++i) {                                 \
        MISS_OP(*i);                                                  \
      }                                                               \
    } else if constexpr (std::is_null_pointer_v<decltype(MISS_OP)>) { \
      /* Called by `contains` functions. */                           \
      /* Do nothing ;-) */                                            \
    } else {                                                          \
      static_assert(dependent_false_v<decltype(MISS_OP)>);            \
    }                                                                 \
  } while (0)

#ifdef HCTR_HPS_DB_HANDLE_TIMEOUT_PARALLEL_DIRECT_
#error HCTR_HPS_DB_HANDLE_TIMEOUT_PARALLEL_DIRECT_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_DB_HANDLE_TIMEOUT_PARALLEL_DIRECT_(MISS_OP)              \
  do {                                                                    \
    size_t skip_count{0};                                                 \
                                                                          \
    for (; k != keys_end; ++k) {                                          \
      if (HCTR_HPS_KEY_TO_PART_INDEX_(*k) == part_index) {                \
        if constexpr (std::is_invocable_v<decltype(MISS_OP), size_t>) {   \
          /* Called by `fetch` functions. */                              \
          MISS_OP(k - keys);                                              \
        } else if constexpr (std::is_null_pointer_v<decltype(MISS_OP)>) { \
          /* Called by `contains` functions. */                           \
          /* Do nothing ;-) */                                            \
        } else {                                                          \
          static_assert(dependent_false_v<decltype(MISS_OP)>);            \
        }                                                                 \
                                                                          \
        ++skip_count;                                                     \
      }                                                                   \
    }                                                                     \
                                                                          \
    joint_skip_count += skip_count;                                       \
  } while (0)

#ifdef HCTR_HPS_DB_HANDLE_TIMEOUT_PARALLEL_INDIRECT_
#error HCTR_HPS_DB_HANDLE_TIMEOUT_PARALLEL_INDIRECT_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_DB_HANDLE_TIMEOUT_PARALLEL_INDIRECT_(MISS_OP)            \
  do {                                                                    \
    size_t skip_count{0};                                                 \
                                                                          \
    for (; i != indices_end; ++i) {                                       \
      if (HCTR_HPS_KEY_TO_PART_INDEX_(keys[*i]) == part_index) {          \
        if constexpr (std::is_invocable_v<decltype(MISS_OP), size_t>) {   \
          /* Called by `fetch` functions. */                              \
          MISS_OP(*i);                                                    \
        } else if constexpr (std::is_null_pointer_v<decltype(MISS_OP)>) { \
          /* Called by `contains` functions. */                           \
          /* Do nothing ;-) */                                            \
        } else {                                                          \
          static_assert(dependent_false_v<decltype(MISS_OP)>);            \
        }                                                                 \
                                                                          \
        ++skip_count;                                                     \
      }                                                                   \
    }                                                                     \
                                                                          \
    joint_skip_count += skip_count;                                       \
  } while (0)

/**
 * Iterators used in Database backend implementations.
 */
#ifdef HCTR_HPS_DB_APPLY_
#error HCTR_HPS_DB_APPLY_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_DB_APPLY_(MODE, ...) HCTR_HPS_DB_APPLY_##MODE##_(__VA_ARGS__)

#ifdef HCTR_HPS_DB_APPLY_SEQUENTIAL_DIRECT_
#error HCTR_HPS_DB_APPLY_SEQUENTIAL_DIRECT_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_DB_APPLY_SEQUENTIAL_DIRECT_(...)                           \
  do {                                                                      \
    static_assert(std::is_same_v<decltype(batch_size), const size_t>);      \
    static_assert(std::is_same_v<decltype(k), const Key*>);                 \
                                                                            \
    for (const Key* const batch_end{&k[batch_size]}; k != batch_end; ++k) { \
      __VA_ARGS__;                                                          \
    }                                                                       \
  } while (0)

#ifdef HCTR_HPS_DB_APPLY_SEQUENTIAL_INDIRECT_
#error HCTR_HPS_DB_APPLY_SEQUENTIAL_INDIRECT_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_DB_APPLY_SEQUENTIAL_INDIRECT_(...)                        \
  do {                                                                     \
    static_assert(std::is_same_v<decltype(batch_size), const size_t>);     \
    static_assert(std::is_same_v<decltype(i), const size_t*>);             \
                                                                           \
    for (const size_t* const batch_end{&i[batch_size]}; i != batch_end;) { \
      const Key* const k{&keys[*i++]};                                     \
      __VA_ARGS__;                                                         \
    }                                                                      \
  } while (0)

#ifdef HCTR_HPS_DB_APPLY_PARALLEL_DIRECT_
#error HCTR_HPS_DB_APPLY_PARALLEL_DIRECT_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_DB_APPLY_PARALLEL_DIRECT_(...)                            \
  do {                                                                     \
    static_assert(std::is_same_v<decltype(batch_size), size_t>);           \
    static_assert(std::is_same_v<decltype(max_batch_size), const size_t>); \
    static_assert(std::is_same_v<decltype(k), const Key*>);                \
                                                                           \
    for (; k != keys_end; ++k) {                                           \
      if (HCTR_HPS_KEY_TO_PART_INDEX_(*k) != part_index) {                 \
        continue;                                                          \
      }                                                                    \
                                                                           \
      __VA_ARGS__;                                                         \
      if (++batch_size >= max_batch_size) {                                \
        ++k;                                                               \
        break;                                                             \
      }                                                                    \
    }                                                                      \
    if (!batch_size) {                                                     \
      return false;                                                        \
    }                                                                      \
  } while (0)

#ifdef HCTR_HPS_DB_APPLY_PARALLEL_INDIRECT_
#error HCTR_HPS_DB_APPLY_PARALLEL_INDIRECT_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_DB_APPLY_PARALLEL_INDIRECT_(...)                          \
  do {                                                                     \
    static_assert(std::is_same_v<decltype(batch_size), size_t>);           \
    static_assert(std::is_same_v<decltype(max_batch_size), const size_t>); \
    static_assert(std::is_same_v<decltype(i), const size_t*>);             \
                                                                           \
    while (i != indices_end) {                                             \
      const Key* const k{&keys[*i++]};                                     \
      if (HCTR_HPS_KEY_TO_PART_INDEX_(*k) != part_index) {                 \
        continue;                                                          \
      }                                                                    \
                                                                           \
      __VA_ARGS__;                                                         \
      if (++batch_size >= max_batch_size) {                                \
        break;                                                             \
      }                                                                    \
    }                                                                      \
    if (!batch_size) {                                                     \
      return false;                                                        \
    }                                                                      \
  } while (0)

/**
 * Uses default ThreadPool to parallelize execution accross parts of a DB backend.
 */
#ifdef HCTR_HPS_DB_PARALLEL_FOR_EACH_PART_
#error HCTR_HPS_DB_PARALLEL_FOR_EACH_PART_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_DB_PARALLEL_FOR_EACH_PART_(...)                                        \
  do {                                                                                  \
    std::vector<std::future<void>> tasks;                                               \
    tasks.reserve(num_partitions);                                                      \
                                                                                        \
    for (size_t part_index{0}; part_index < num_partitions; ++part_index) {             \
      tasks.emplace_back(ThreadPool::get().submit([&, part_index]() { __VA_ARGS__; })); \
    }                                                                                   \
    ThreadPool::await(tasks.begin(), tasks.end());                                      \
  } while (0)

/**
 * Since SST writing needs to be supported by all backends, we need this macro everywhere too. Hence
 * the reason why it is defined here.
 */
#ifdef HCTR_ROCKSDB_CHECK
#error HCTR_ROCKSDB_CHECK is already defined. This could lead to unpredictable behavior!
#endif
#define HCTR_ROCKSDB_CHECK(EXPR)                                                  \
  do {                                                                            \
    const rocksdb::Status status{(EXPR)};                                         \
    HCTR_CHECK_HINT(status.ok(), "RocksDB error: %s", status.ToString().c_str()); \
  } while (0)

// TODO: Remove me!
#pragma GCC diagnostic pop

}  // namespace HugeCTR