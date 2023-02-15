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

#include <vector>

namespace HugeCTR {

namespace hybrid_embedding {

template <typename dtype>
void generate_reference_stats(const std::vector<dtype> &data, std::vector<dtype> &samples,
                              std::vector<size_t> &categories_stats,
                              std::vector<size_t> &counts_stats,
                              const std::vector<size_t> &table_sizes, const size_t batch_size);

}  // namespace hybrid_embedding

}  // namespace HugeCTR
