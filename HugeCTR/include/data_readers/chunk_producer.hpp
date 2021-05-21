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

namespace HugeCTR {

/**
 * @brief An interface to get a free chunk and commit it after the initialization
 */
template <typename T>
class ChunkProducer {
 public:
  virtual T* checkout_free_chunk(unsigned int ch_id) = 0;
  virtual void commit_data_chunk(unsigned int ch_id, bool is_nop) = 0;
};

}  // namespace HugeCTR
