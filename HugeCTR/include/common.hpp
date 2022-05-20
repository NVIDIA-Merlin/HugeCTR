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

#include <assert.h>
#include <cublas_v2.h>
#include <curand.h>
#include <nvml.h>

#include <algorithm>
#include <base/debug/logger.hpp>
#include <config.hpp>
#include <ctime>
#include <exception>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef ENABLE_MPI
#include <limits.h>
#include <mpi.h>
#include <stdint.h>

#if SIZE_MAX == UCHAR_MAX
#define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
#error "no suitable MPI type for size_t"
#endif

#endif

#define PYTORCH_INIT

namespace HugeCTR {

#define HUGECTR_VERSION_MAJOR 3
#define HUGECTR_VERSION_MINOR 6
#define HUGECTR_VERSION_PATCH 0

#define WARP_SIZE 32

namespace hybrid_embedding {

enum class HybridEmbeddingType;
enum class CommunicationType;

}  // namespace hybrid_embedding

enum class Check_t { Sum, None, Unknown };

enum class DataReaderSparse_t { Distributed, Localized };

enum class DataReaderType_t { Norm, Raw, Parquet, RawAsync };

enum class SourceType_t { FileList, Mmap, Parquet };

enum class TrainPSType_t { Staged, Cached };

struct NameID {
  std::string file_name;
  unsigned int id;
};

/**
 * An internal exception, as a child of std::runtime_error, to carry the error code.
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
  internal_runtime_error(Error_t err, std::string str) : runtime_error(str), err_(err) {}
};

enum class LrPolicy_t { fixed };

enum class Optimizer_t { Adam, AdaGrad, MomentumSGD, Nesterov, SGD, DEFAULT };

enum class Update_t { Local, Global, LazyGlobal };

// TODO: Consider to move them into a separate file
enum class Activation_t { Relu, None };

enum class FcPosition_t { None, Head, Body, Tail, Isolated };

enum class Regularizer_t { L1, L2, None };

enum class Alignment_t { Auto, None };

enum class GroupLayer_t { GroupFusedInnerProduct };

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
  GRU,
  MatrixMultiply,
  Scale,
  FusedReshapeConcat,
  FusedReshapeConcatGeneral,
  Softmax,
  PReLU_Dice,
  ReduceMean,
  Sub,
  Gather,
  Sigmoid,
  Slice,
  WeightMultiply,
  FmOrder2,
  Add,
  ReduceSum,
  MultiCross,
  Cast,
  ElementwiseMultiply,
  Unknown
};

enum class Embedding_t {
  DistributedSlotSparseEmbeddingHash,
  LocalizedSlotSparseEmbeddingHash,
  LocalizedSlotSparseEmbeddingOneHot,
  HybridSparseEmbedding,
  None
};

enum class Initializer_t { Default, Uniform, XavierNorm, XavierUniform, Zero };

enum class TrainState_t {
  Init,
  BottomMLPFprop,
  TopMLPFprop,
  BottomMLPBprop,
  TopMLPBprop,
  MLPExchangeWgrad,
  MLPUpdate,
  Finalize
};

enum class Distribution_t { Uniform, PowerLaw };

enum class PowerLaw_t { Long, Medium, Short, Specific };

// TODO: Consider to move them into a separate file
struct TrainState {
  TrainState_t state = TrainState_t::Init;
  cudaEvent_t* event = nullptr;
};

struct AsyncParam {
  int num_threads;
  int num_batches_per_thread;
  int io_block_size;
  int io_depth;
  int io_alignment;
  bool shuffle;
  Alignment_t aligned_type;
};

struct HybridEmbeddingParam {
  size_t max_num_frequent_categories;
  int64_t max_num_infrequent_samples;
  double p_dup_max;
  double max_all_reduce_bandwidth;
  double max_all_to_all_bandwidth;
  double efficiency_bandwidth_ratio;
  bool use_train_precompute_indices, use_eval_precompute_indices;
  hybrid_embedding::CommunicationType communication_type;
  hybrid_embedding::HybridEmbeddingType hybrid_embedding_type;
};

typedef struct DataSetHeader_ {
  long long error_check;        // 0: no error check; 1: check_sum
  long long number_of_records;  // the number of samples in this data file
  long long label_dim;          // dimension of label
  long long dense_dim;          // dimension of dense feature
  long long slot_num;           // slot_num for each embedding
  long long reserved[3];        // reserved for future use
} DataSetHeader;

#define DISALLOW_COPY(ClassName)        \
  ClassName(const ClassName&) = delete; \
  ClassName& operator=(const ClassName&) = delete;

#define DISALLOW_MOVE(ClassName)   \
  ClassName(ClassName&&) = delete; \
  ClassName& operator=(ClassName&&) = delete;

#define DISALLOW_COPY_AND_MOVE(ClassName) \
  DISALLOW_COPY(ClassName)                \
  DISALLOW_MOVE(ClassName)

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

template <typename T>
inline void print_func(T const& t) {
  std::cout << t << ", ";
}

// Set precision for double type
template <>
inline void print_func<double>(double const& t) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(2) << t << ", ";
  std::cout << os.str();
}

template <typename... Args>
inline void LOG(const Args&... args) {
  if (Logger::get().get_rank() == 0) {
    std::cout << '[';
    std::initializer_list<char>{(print_func(args), 'a')...};
    std::cout << ']' << std::endl;
  }
}

struct DataReaderSparseParam {
  std::string top_name;
  std::vector<int> nnz_per_slot;
  bool is_fixed_length;
  int slot_num;

  DataReaderSparse_t type;
  int max_feature_num;
  int max_nnz;

  DataReaderSparseParam() {}
  DataReaderSparseParam(const std::string& top_name_, const std::vector<int>& nnz_per_slot_,
                        bool is_fixed_length_, int slot_num_)
      : top_name(top_name_),
        nnz_per_slot(nnz_per_slot_),
        is_fixed_length(is_fixed_length_),
        slot_num(slot_num_),
        type(DataReaderSparse_t::Distributed) {
    if (static_cast<size_t>(slot_num_) != nnz_per_slot_.size()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "slot num != nnz_per_slot.size().");
    }
    max_feature_num = std::accumulate(nnz_per_slot.begin(), nnz_per_slot.end(), 0);
    max_nnz = *std::max_element(nnz_per_slot.begin(), nnz_per_slot.end());
  }

  DataReaderSparseParam(const std::string& top_name_, const int nnz_per_slot_,
                        bool is_fixed_length_, int slot_num_)
      : top_name(top_name_),
        nnz_per_slot(slot_num_, nnz_per_slot_),
        is_fixed_length(is_fixed_length_),
        slot_num(slot_num_),
        type(DataReaderSparse_t::Distributed) {
    max_feature_num = std::accumulate(nnz_per_slot.begin(), nnz_per_slot.end(), 0);
    max_nnz = *std::max_element(nnz_per_slot.begin(), nnz_per_slot.end());
  }
};

struct DenseLayerSwitchs {
  bool fuse_wb;
  DenseLayerSwitchs(bool fuse_wb_ = false) : fuse_wb(fuse_wb_) {}
};

}  // namespace HugeCTR

#include <profiler.hpp>
