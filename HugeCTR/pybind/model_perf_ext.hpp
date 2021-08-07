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
#include <HugeCTR/pybind/model.hpp>

namespace HugeCTR {

class ModelPerfExt final : public Model {
 public:
  ModelPerfExt(const Solver& solver, const DataReaderParams& reader_params,
               std::shared_ptr<OptParamsPy>& opt_params,
               std::shared_ptr<ModelOversubscriberParams>& mos_params);
  ~ModelPerfExt() override {}
  bool train() override;
  bool eval(int eval_batch = -1) override;
  void fit(int num_epochs, int max_iter, int display, int eval_interval, int snapshot,
           std::string snapshot_prefix) override;

 private:
  void train_overlapped() override;
  void exchange_wgrad(size_t device_id) override;
};

}  // namespace HugeCTR
