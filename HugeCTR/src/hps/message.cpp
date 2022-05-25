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

#include <hps/message.hpp>

namespace HugeCTR {

template <typename TKey>
void MessageSink<TKey>::post(const std::string& tag, const size_t num_pairs, const TKey* const keys,
                             const char* const values, const uint32_t value_size) {
  num_posts_++;
  num_pairs_posted_ += num_pairs;
}

template <typename TKey>
void MessageSink<TKey>::flush() {
  num_flushes_++;
}

template class MessageSink<unsigned int>;
template class MessageSink<long long>;

template class MessageSource<unsigned int>;
template class MessageSource<long long>;

}  // namespace HugeCTR