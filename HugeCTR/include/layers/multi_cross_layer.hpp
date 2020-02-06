/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <functional>
#include <vector>
#include "HugeCTR/include/general_buffer.hpp"
#include "HugeCTR/include/layer.hpp"


namespace HugeCTR {
class MultiCrossLayer : public Layer {
private:
  const int num_layers_;
  std::shared_ptr<GeneralBuffer<float>> blobs_buff_;  /**< internal blobs' general buffer */
  Tensors<float> blob_tensors_;                            /**< vector of internal blobs' tensors */
  
  const int TMP_MATS{2};
  const int TMP_VECS{2};
  Tensors<float> tmp_mat_tensors_; //[h,w]
  Tensors<float> tmp_vec_tensors_; //[h,1]
public:
  /**
   * forward pass
   */
  void fprop(cudaStream_t stream) final;
  /**
   * backward pass
   */
  void bprop(cudaStream_t stream) final;


  MultiCrossLayer( const std::shared_ptr<GeneralBuffer<float>>& weight_buff,
		   const std::shared_ptr<GeneralBuffer<float>>& wgrad_buff,
		   const std::shared_ptr<Tensor<float>>& in_tensor,
		   const std::shared_ptr<Tensor<float>>& out_tensor,
		   int num_layers,
		   int device_id);
  MultiCrossLayer(const FullyConnectedLayer& C) = delete;
 private:
  /**
   * Use Gaussian initialization.
   */
  std::vector<float> get_initializer() override;
};
} //namespace HugeCTR
