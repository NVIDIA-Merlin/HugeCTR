/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#ifndef GROUP_LOOKUP_H
#define GROUP_LOOKUP_H

#include <vector>

namespace sok {

template <typename KeyType, typename DataType>
struct LookupTask {
  const float *input;
  const void *key;
  int32_t dimension;
  int32_t num_keys;
  void *output;
};

template <typename KeyType, typename DataType>
class LookupLauncher {
 public:
  LookupLauncher();
  ~LookupLauncher();

  void initialize(size_t num_tasks);

  void operator()(std::vector<LookupTask<KeyType, DataType>> &h_tasks, cudaStream_t stream = 0);

 private:
  size_t num_tasks_;
  LookupTask<KeyType, DataType> *d_tasks_;

  int sm_count_;
};

}  // namespace sok

#endif
