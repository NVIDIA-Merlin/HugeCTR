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

#include <cstdint>

namespace HugeCTR {

inline uint64_t rotl64(const uint64_t x, const int n) {
  return (x << n) | (x >> (8 * sizeof(uint64_t) - n));
}

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

#ifdef HCTR_KEY_TO_DB_PART_INDEX
#error "HCTR_KEY_TO_DB_PART_INDEX is already defined. This could lead to unpredictable behavior!"
#else
#define HCTR_KEY_TO_DB_PART_INDEX(KEY) (rrxmrrxmsx_0(KEY) % num_partitions_)
#endif

}  // namespace HugeCTR