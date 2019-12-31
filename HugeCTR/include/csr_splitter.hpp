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

template <typename TypeKey>
class CsrSplitter {
  std::shared<GeneralBuffer<TypeKey>> csr_buffer_;
  Tensors<float> label_tensors_;        /**< Label tensors for the usage of loss */
  Tensors<float> dense_tensors_;        /**< Label tensors for the usage of loss */
public: 
  CsrSplitter();
  split();

};
