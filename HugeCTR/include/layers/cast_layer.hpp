/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <layer.hpp>

namespace HugeCTR {

template <typename From, typename To>
class CastLayer : public Layer {
  Tensor2<From> bottom_tensor_;
  Tensor2<To> top_tensor_;

 public:
  CastLayer(const Tensor2<From>& bottom_tensor, const Tensor2<To>& top_tensor,
            const std::shared_ptr<GPUResource>& gpu_resource);
  void fprop(bool is_train) override;
  void bprop() override;
};

}  // namespace HugeCTR
