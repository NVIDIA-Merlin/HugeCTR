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

#include <atomic>
#include <common.hpp>
// TODO remove me
#include <data_reader.hpp>
#include <sparse_tensor.hpp>
#include <tensor2.hpp>
#include <vector>

namespace HugeCTR {

enum class BufferState : int { FileEOF, Reading, ReadyForRead, Writing, ReadyForWrite };

// TODO remove me
struct ThreadBuffer {
  std::vector<SparseTensorBag> device_sparse_buffers;  // same number as embedding number
  std::vector<unsigned char> is_fixed_length;          // same number as embedding number
  TensorBag2 device_dense_buffers;
  std::atomic<BufferState> state;
  long long current_batch_size;
  int batch_size;
  size_t param_num;
  int label_dim;
  int dense_dim;
  int batch_size_start_idx;  // dense buffer
  int batch_size_end_idx;
};
struct ThreadBuffer23 {
  std::vector<SparseTensor23> device_sparse_buffers;  // same number as embedding number
  std::vector<unsigned char> is_fixed_length;         // same number as embedding number
  core23::Tensor device_dense_buffers;
  std::atomic<BufferState> state;
  long long current_batch_size;
  int batch_size;
  size_t param_num;
  int label_dim;
  int dense_dim;
  int batch_size_start_idx;  // dense buffer
  int batch_size_end_idx;
};

// TODO remove me
struct BroadcastBuffer {
  std::vector<SparseTensorBag>
      sparse_buffers;  // same number as (embedding number * local device number)
  std::vector<unsigned char> is_fixed_length;        // same number as embedding number
  std::vector<TensorBag2> dense_tensors;             // same number as local device number
  std::vector<cudaEvent_t> finish_broadcast_events;  // same number as local device number
  std::atomic<BufferState> state;
  long long current_batch_size;
  size_t param_num;
};

struct BroadcastBuffer23 {
  std::vector<SparseTensor23>
      sparse_buffers;  // same number as (embedding number * local device number)
  std::vector<unsigned char> is_fixed_length;        // same number as embedding number
  std::vector<core23::Tensor> dense_tensors;         // same number as local device number
  std::vector<cudaEvent_t> finish_broadcast_events;  // same number as local device number
  std::atomic<BufferState> state;
  long long current_batch_size;
  size_t param_num;
};

// TODO remove me
struct DataReaderOutput {
  std::map<std::string, std::vector<SparseTensorBag>> sparse_tensors_map;
  std::vector<std::string> sparse_name_vec;
  std::vector<TensorBag2> label_tensors;
  std::vector<TensorBag2> dense_tensors;
  bool use_mixed_precision;
  int label_dense_dim;
};
struct DataReaderOutput23 {
  std::map<std::string, std::vector<SparseTensor23>> sparse_tensors_map;
  std::vector<std::string> sparse_name_vec;
  std::vector<core23::Tensor> label_tensors;
  std::vector<core23::Tensor> dense_tensors;
  bool use_mixed_precision;
  int label_dense_dim;
};

}  // namespace HugeCTR
