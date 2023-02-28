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

#include <core23/mpi_init_service.hpp>
#include <iostream>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

#ifdef HCTR_CODE_REFERENCE_
#error HCTR_CODE_REFERENCE_ already defined. Potential naming conflict!
#endif
#define HCTR_CODE_REFERENCE_(EXPR) \
  CodeReference { __FILE__, __LINE__, __func__, #EXPR }

#ifdef HCTR_MPI_CHECK_
#error HCTR_MPI_CHECK_ already defined. Potential naming conflict!
#endif
#define HCTR_MPI_CHECK_(EXPR)                                                                  \
  do {                                                                                         \
    int err{(EXPR)};                                                                           \
    if (err != MPI_SUCCESS) {                                                                  \
      std::cerr << "MpiInitService: " << HCTR_CODE_REFERENCE_(EXPR) << "\nError code: " << err \
                << "\nReason: ";                                                               \
      char err_msg_buffer[MPI_MAX_ERROR_STRING];                                               \
      int err_len{MPI_MAX_ERROR_STRING};                                                       \
      err = MPI_Error_string(err, err_msg_buffer, &err_len);                                   \
      std::cerr << (err == MPI_SUCCESS ? err_msg_buffer : "Unknown MPI error!") << std::endl;  \
      std::abort();                                                                            \
    }                                                                                          \
  } while (0)

namespace HugeCTR {
namespace core23 {

/**
 * TODO: Hotfix; will be moved to core23 after merging logger PR.
 */
struct CodeReference {
  const char* const file;
  const size_t line;
  const char* function;
  const char* expression;
};

inline std::ostream& operator<<(std::ostream& os, const CodeReference& ref) {
  os << ref.expression << " at " << ref.function << " (" << ref.file << ':' << ref.line << ')';
  return os;
}

MpiInitService::MpiInitService() {
#ifdef ENABLE_MPI
  HCTR_MPI_CHECK_(MPI_Initialized(&was_preinitialized_));
  if (was_preinitialized_) {
    std::cerr << "MpiInitService: MPI was already initialized by another (non-HugeCTR) mechanism."
              << std::endl;
  } else {
    HCTR_MPI_CHECK_(MPI_Init(nullptr, nullptr));
    HCTR_MPI_CHECK_(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_));
    HCTR_MPI_CHECK_(MPI_Comm_size(MPI_COMM_WORLD, &world_size_));
    std::cerr << "MpiInitService: Initialized!" << std::endl;
  }
#endif
}

MpiInitService::~MpiInitService() {
  if (!was_preinitialized_ && is_initialized()) {
#ifdef ENABLE_MPI
    HCTR_MPI_CHECK_(MPI_Finalize());
#endif
  }
}

bool MpiInitService::is_initialized() const {
  int initialized{};
#ifdef ENABLE_MPI
  HCTR_MPI_CHECK_(MPI_Initialized(&initialized));
#endif
  return initialized != 0;
}

MpiInitService& MpiInitService::get() {
  static MpiInitService instance;
  return instance;
}

}  // namespace core23
}  // namespace HugeCTR