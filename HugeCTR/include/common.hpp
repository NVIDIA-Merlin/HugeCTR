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

#include <assert.h>
#include <cublas_v2.h>
#include <curand.h>
#include <config.hpp>
#include <ctime>
#include <exception>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <utility>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

#define PYTORCH_INIT

namespace HugeCTR {

#define HUGECTR_VERSION_MAJOR 3
#define HUGECTR_VERSION_MINOR 0
#define HUGECTR_VERSION_PATCH 0

#define WARP_SIZE 32

//#define DATA_READING_TEST

enum class Error_t {
  Success,
  FileCannotOpen,
  BrokenFile,
  OutOfMemory,
  OutOfBound,
  WrongInput,
  IllegalCall,
  NotInitialized,
  UnSupportedFormat,
  InvalidEnv,
  MpiError,
  CublasError,
  CudnnError,
  CudaError,
  NcclError,
  DataCheckError,
  UnspecificError,
  EndOfFile
};

enum class Check_t { Sum, None };

enum class DataReaderSparse_t { Distributed, Localized };

enum class DataReaderType_t { Norm, Raw, Parquet };

enum class SourceType_t { FileList, Mmap, Parquet };

struct DataReaderSparseParam {
  DataReaderSparse_t type;
  int max_feature_num;
  int max_nnz;
  int slot_num;
};

struct NameID {
  std::string file_name;
  unsigned int id;
};

/**
 * An internal exception to carry the error code.
 * This exception inherits std::runtime_error and
 * adds HugeCTR specific text prefix to method what()
 * On the boundary of subsystem: session will return the
 * error code instead of throwing exceptions.
 */
class internal_runtime_error : public std::runtime_error {
 private:
  const Error_t err_;

 public:
  /**
   * Get the error code from exception.
   * @return error
   **/
  Error_t get_error() const { return err_; }
  /**
   * Ctor
   */
  internal_runtime_error(Error_t err, std::string str)
      : runtime_error("[HCDEBUG][ERROR] " + str), err_(err) {}
};

enum class LrPolicy_t { fixed };

enum class Optimizer_t { Adam, AdaGrad, MomentumSGD, Nesterov, SGD };

enum class Update_t { Local, Global, LazyGlobal };

enum class Regularizer_t { L1, L2 };

enum class Layer_t {
  BatchNorm,
  BinaryCrossEntropyLoss,
  Reshape,
  Concat,
  CrossEntropyLoss,
  Dropout,
  ELU,
  InnerProduct,
  FusedInnerProduct,
  Interaction,
  MultiCrossEntropyLoss,
  ReLU,
  ReLUHalf,
  Sigmoid,
  Slice,
  WeightMultiply,
  FmOrder2,
  Add,
  ReduceSum,
  MultiCross,
  Cast,
  DotProduct
};

enum class Embedding_t {
  DistributedSlotSparseEmbeddingHash,
  LocalizedSlotSparseEmbeddingHash,
  LocalizedSlotSparseEmbeddingOneHot
};

enum class Initializer_t { Default, Uniform, XavierNorm, XavierUniform, Zero };

typedef struct DataSetHeader_ {
  long long error_check;        // 0: no error check; 1: check_num
  long long number_of_records;  // the number of samples in this data file
  long long label_dim;          // dimension of label
  long long dense_dim;          // dimension of dense feature
  long long slot_num;           // slot_num for each embedding
  long long reserved[3];        // reserved for future use
} DataSetHeader;

#ifdef ENABLE_MPI
#define CK_MPI_THROW_(cmd)                                                                       \
  do {                                                                                           \
    auto retval = (cmd);                                                                         \
    if (retval != MPI_SUCCESS) {                                                                 \
      throw internal_runtime_error(                                                              \
          Error_t::MpiError, std::string("MPI Runtime error: ") + std::to_string(retval) + " " + \
                                 __FILE__ + ":" + std::to_string(__LINE__) + " \n");             \
    }                                                                                            \
  } while (0)

#endif

#define CK_(err)                                                                       \
  do {                                                                                 \
    Error_t retval = (err);                                                            \
    if (retval != Error_t::Success) {                                                  \
      std::cerr << "[HCDEBUG][ERROR] Return Error: at " << __FILE__ << ":" << __LINE__ \
                << std::endl;                                                          \
    }                                                                                  \
  } while (0)

#ifdef ENABLE_MPI
#define ERROR_MESSAGE_(msg)                                                                   \
  do {                                                                                        \
    int __PID(-1), __NUM_PROCS(-1);                                                           \
    MPI_Comm_rank(MPI_COMM_WORLD, &__PID);                                                    \
    MPI_Comm_size(MPI_COMM_WORLD, &__NUM_PROCS);                                              \
    std::string str = (msg);                                                                  \
    std::cerr << "[HCDEBUG][ERROR] " << str << " " << __FILE__ << ":" << __LINE__             \
              << " in pid: " << __PID << " of " << __NUM_PROCS << " processes." << std::endl; \
  } while (0)
#else
#define ERROR_MESSAGE_(msg)                                                                     \
  do {                                                                                          \
    std::string str = (msg);                                                                    \
    std::cerr << "[HCDEBUG][ERROR] " << str << " " << __FILE__ << ":" << __LINE__ << std::endl; \
  } while (0)
#endif

#define CK_THROW_(x, msg)                                                                       \
  do {                                                                                          \
    Error_t retval = (x);                                                                       \
    if (retval != Error_t::Success) {                                                           \
      throw internal_runtime_error(x, std::string("Runtime error: ") + (msg) + " " + __FILE__ + \
                                          ":" + std::to_string(__LINE__) + " \n");              \
    }                                                                                           \
  } while (0)

#define CK_RETURN_(x, msg)                                                         \
  do {                                                                             \
    Error_t retval = (x);                                                          \
    if (retval != Error_t::Success) {                                              \
      std::cerr << std::string("Runtime error: ") + (msg) + " " + __FILE__ + ":" + \
                       std::to_string(__LINE__) + " \n";                           \
      return x;                                                                    \
    }                                                                              \
  } while (0)

inline void MESSAGE_(const std::string msg, bool per_process=false) {
#ifdef ENABLE_MPI
  int __PID(-1);
  MPI_Comm_rank(MPI_COMM_WORLD, &__PID);
  if (__PID && !per_process) return;
#endif
  std::time_t time_instance = std::time(0);
  std::tm* time_now = std::localtime(&time_instance);
  std::string str = (std::move(msg));
  std::cout.fill('0');
  std::cout << "[" << std::setw(2) << time_now->tm_mday << "d" << std::setw(2)
            << time_now->tm_hour << "h" << std::setw(2) << time_now->tm_min << "m"
            << std::setw(2) << time_now->tm_sec << "s"
            << "][HUGECTR][INFO]: " << str << std::endl;
}

#define CK_CUDA_THROW_(x)                                                                          \
  do {                                                                                             \
    cudaError_t retval = (x);                                                                      \
    if (retval != cudaSuccess) {                                                                   \
      throw internal_runtime_error(Error_t::CudaError,                                             \
                                   std::string("Runtime error: ") + (cudaGetErrorString(retval)) + \
                                       " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");   \
    }                                                                                              \
  } while (0)

#define CK_CUDA_RETURN_BOOL_(x)  \
  do {                           \
    cudaError_t retval = (x);    \
    if (retval != cudaSuccess) { \
      return false;              \
    }                            \
  } while (0)

#ifdef ENABLE_MPI
#define PRINT_FUNC_NAME_()                                                            \
  do {                                                                                \
    int __PID(-1), __NUM_PROCS(-1);                                                   \
    MPI_Comm_rank(MPI_COMM_WORLD, &__PID);                                            \
    MPI_Comm_size(MPI_COMM_WORLD, &__NUM_PROCS);                                      \
    std::cout << "[HCDEBUG][CALL] " << __FUNCTION__ << " in pid: " << __PID << " of " \
              << __NUM_PROCS << " processes." << std::endl;                           \
  } while (0)
#else
#define PRINT_FUNC_NAME_()                                               \
  do {                                                                   \
    std::cout << "[HCDEBUG][CALL] " << __FUNCTION__ << " " << std::endl; \
  } while (0)
#endif

#define CK_CUBLAS_THROW_(x)                                                                        \
  do {                                                                                             \
    cublasStatus_t retval = (x);                                                                   \
    if (retval == CUBLAS_STATUS_NOT_INITIALIZED) {                                                 \
      throw internal_runtime_error(Error_t::CublasError, std::string("Runtime error: ") +          \
                                                             ("cublas_status_not_initialized ") +  \
                                                             __FILE__ + ":" +                      \
                                                             std::to_string(__LINE__) + " \n");    \
    }                                                                                              \
    if (retval == CUBLAS_STATUS_ARCH_MISMATCH) {                                                   \
      throw internal_runtime_error(Error_t::CublasError, std::string("Runtime error: ") +          \
                                                             ("cublas_status_arch_mismatch ") +    \
                                                             __FILE__ + ":" +                      \
                                                             std::to_string(__LINE__) + " \n");    \
    }                                                                                              \
    if (retval == CUBLAS_STATUS_NOT_SUPPORTED) {                                                   \
      throw internal_runtime_error(Error_t::CublasError, std::string("Runtime error: ") +          \
                                                             ("cublas_status_not_supported ") +    \
                                                             __FILE__ + ":" +                      \
                                                             std::to_string(__LINE__) + " \n");    \
    }                                                                                              \
    if (retval == CUBLAS_STATUS_INVALID_VALUE) {                                                   \
      throw internal_runtime_error(Error_t::CublasError, std::string("Runtime error: ") +          \
                                                             ("cublas_status_invalid_value ") +    \
                                                             __FILE__ + ":" +                      \
                                                             std::to_string(__LINE__) + " \n");    \
    }                                                                                              \
    if (retval == CUBLAS_STATUS_EXECUTION_FAILED) {                                                \
      throw internal_runtime_error(Error_t::CublasError, std::string("Runtime error: ") +          \
                                                             ("cublas_status_execution_failed ") + \
                                                             __FILE__ + ":" +                      \
                                                             std::to_string(__LINE__) + " \n");    \
    }                                                                                              \
  } while (0)

#define CK_NCCL_THROW_(cmd)                                                                        \
  do {                                                                                             \
    ncclResult_t r = (cmd);                                                                        \
    if (r != ncclSuccess) {                                                                        \
      throw internal_runtime_error(Error_t::NcclError, std::string("Runtime error: NCCL Error ") + \
                                                           std::string(ncclGetErrorString(r)) +    \
                                                           " " + __FILE__ + ":" +                  \
                                                           std::to_string(__LINE__) + " \n");      \
    }                                                                                              \
  } while (0)

#define CK_CUDNN_THROW_(cmd)                                                                      \
  do {                                                                                            \
    cudnnStatus_t retval = (cmd);                                                                 \
    if (retval != CUDNN_STATUS_SUCCESS) {                                                         \
      throw internal_runtime_error(                                                               \
          Error_t::CudnnError, std::string("CUDNN Runtime error: ") +                             \
                                   std::string(cudnnGetErrorString(cmd)) + " " + __FILE__ + ":" + \
                                   std::to_string(__LINE__) + " \n");                             \
    }                                                                                             \
  } while (0)

#define CK_CURAND_THROW_(cmd)                                                                   \
  do {                                                                                          \
    curandStatus_t retval = (cmd);                                                              \
    if (retval != CURAND_STATUS_SUCCESS) {                                                      \
      throw internal_runtime_error(                                                             \
          Error_t::CudnnError, std::string("CURAND Runtime error: ") + std::to_string(retval) + \
                                   " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");    \
    }                                                                                           \
  } while (0)

template <typename T>
inline void print_func(T& t) {
  std::cout << t << ", ";
  return;
}

template <typename... Args>
inline void LOG(const Args&... args) {
  std::cout << "[";
  std::initializer_list<char>{(print_func(args), 'a')...};
  std::cout << "]" << std::endl;

  return;
}

}  // namespace HugeCTR
