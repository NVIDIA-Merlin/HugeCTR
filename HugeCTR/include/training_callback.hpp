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

namespace HugeCTR {

struct TrainingCallback {
  virtual ~TrainingCallback() = default;
  virtual void on_training_start() = 0;
  virtual void on_training_end(int curent_iter) = 0;
  virtual bool on_eval_start(int current_iter) = 0;
  virtual bool on_eval_end(int current_iter, const std::map<std::string, float>& eval_results) = 0;
};

}  // namespace HugeCTR