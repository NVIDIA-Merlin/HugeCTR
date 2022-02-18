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

#include <vector>

#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

enum class HybridEmbeddingType { Distributed, Unknown };
enum class CommunicationType { IB_NVLink_Hier, IB_NVLink, NVLink_SingleNode, Unknown };
enum class CommunicationDirection { CommunicationForward, CommunicationBackward };

template <typename dtype>
void download_tensor(std::vector<dtype>& h_tensor, const Tensor2<dtype> tensor,
                     cudaStream_t stream);

template <typename dtype>
void upload_tensor(const std::vector<dtype>& h_tensor, Tensor2<dtype> tensor, cudaStream_t stream);

}  // namespace hybrid_embedding

}  // namespace HugeCTR
