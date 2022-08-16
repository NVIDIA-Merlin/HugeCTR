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

#ifdef ENABLE_MPI
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace HugeCTR;

class MPIEnvironment : public ::testing::Environment {
 protected:
  virtual void SetUp() {
    test::mpi_init();
    HCTR_LIB_THROW(nvmlInit_v2());
  };
  virtual void TearDown() { test::mpi_finalize(); }
  virtual ~MPIEnvironment(){};
};

::testing::Environment* const mpi_env = ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
#endif
