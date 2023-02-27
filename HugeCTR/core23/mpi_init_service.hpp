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

#include <core/macro.hpp>

namespace HugeCTR {
namespace core23 {

class MpiInitService {
 public:
  HCTR_DISALLOW_COPY_AND_MOVE(MpiInitService);

  virtual ~MpiInitService();

  /**
   * Rank of this process in MPI world
   */
  inline int world_rank() const { return world_rank_; }
  inline int world_size() const { return world_size_; }

  /**
   * Double check if MPI is initialized.
   */
  bool is_initialized() const;

  static MpiInitService& get();

 private:
  MpiInitService();

 private:
  int was_preinitialized_{};
  int world_rank_{0};
  int world_size_{1};
};

}  // namespace core23
}  // namespace HugeCTR