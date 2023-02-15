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

#include <rocksdb/db.h>

#include <algorithm>
#include <hps/database_backend_detail.hpp>
#include <string>
#include <vector>

namespace HugeCTR {

// TODO: Remove me!
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wconversion"

/**
 * RocksDB Backend / Fetch
 */
#ifdef HCTR_HPS_ROCKSDB_FETCH_
#error HCTR_HPS_ROCKSDB_FETCH_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_ROCKSDB_FETCH_(MODE)                                                              \
  [&]() {                                                                                          \
    static_assert(std::is_same_v<decltype(miss_count), size_t>);                                   \
    static_assert(std::is_same_v<decltype(k_views), std::vector<rocksdb::Slice>>);                 \
    static_assert(std::is_same_v<decltype(v_views), std::vector<std::string>>);                    \
                                                                                                   \
    k_views.clear();                                                                               \
    HCTR_HPS_DB_APPLY_(MODE, k_views.emplace_back(reinterpret_cast<const char*>(k), sizeof(Key))); \
    col_handles.resize(k_views.size(), ch);                                                        \
                                                                                                   \
    v_views.clear();                                                                               \
    v_views.reserve(col_handles.size());                                                           \
    const std::vector<rocksdb::Status>& statuses{                                                  \
        db_->MultiGet(read_options_, col_handles, k_views, &v_views)};                             \
                                                                                                   \
    for (size_t idx{0}; idx < batch_size; ++idx) {                                                 \
      const Key* const k{reinterpret_cast<const Key*>(k_views[idx].data())};                       \
      const rocksdb::Status& s{statuses[idx]};                                                     \
      if (s.ok()) {                                                                                \
        const std::string& v_view{v_views[idx]};                                                   \
        HCTR_CHECK(v_view.size() <= value_stride);                                                 \
        std::copy(v_view.begin(), v_view.end(), &values[(k - keys) * value_stride]);               \
      } else if (s.IsNotFound()) {                                                                 \
        on_miss(k - keys);                                                                         \
        ++miss_count;                                                                              \
      } else {                                                                                     \
        HCTR_ROCKSDB_CHECK(s);                                                                     \
      }                                                                                            \
    }                                                                                              \
                                                                                                   \
    return true;                                                                                   \
  }()

// TODO: Remove me!
#pragma GCC diagnostic pop

}  // namespace HugeCTR