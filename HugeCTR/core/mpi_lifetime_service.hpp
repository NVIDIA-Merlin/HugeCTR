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

#ifdef ENABLE_MPI
#include <mpi.h>

namespace HugeCTR {

class MPILifetimeService {
 private:
  MPILifetimeService() {
    MPI_Initialized(&mpi_preinitialized_);
    // TODO: logs must be reactivated after resolving the lib dependency
    if (mpi_preinitialized_) {
      // HCTR_LOG(WARNING, ROOT,
      //          "MPI was already initialized somewhere elese. Lifetime service disabled.\n");
    } else {
      MPI_Init(nullptr, nullptr);
      // HCTR_LOG(INFO, ROOT, "MPI initialized from native backend.\n");
      // HCTR_LOG(WARNING, ROOT,
      //          "\nMPI can only be initialized once per process. If HugeCTR is run on top of\n"
      //          "Python, unexpected things might happen if other other packages are used that\n"
      //          "rely on MPI. If you experience problems, try adding \"from mpi4py import MPI\"\n"
      //          "in your Python script before importing HugeCTR.\n");
    }
  }

  MPILifetimeService(const MPILifetimeService&) = delete;
  MPILifetimeService& operator=(const MPILifetimeService&) = delete;

 public:
  virtual ~MPILifetimeService() {
    if (!mpi_preinitialized_) {
      HCTR_MPI_THROW(MPI_Finalize());
      HCTR_LOG(INFO, ROOT, "MPI finalization done.\n");
    }
  }

  static void init() {
    static std::unique_ptr<MPILifetimeService> instance;
    static std::once_flag once_flag;
    std::call_once(once_flag, []() { instance.reset(new MPILifetimeService()); });
  }

 private:
  int mpi_preinitialized_ = 0;
};

}  // namespace HugeCTR

#endif
