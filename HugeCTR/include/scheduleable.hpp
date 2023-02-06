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

#include <data_reader.hpp>
#include <embedding.hpp>

namespace HugeCTR {
/**
 * @brief Data Reader that enables scheduling of various
 * computation to enable better overlap within the pipeline.
 */
class SchedulableDataReader : public IDataReader {
 public:
  // TODO: remove, use get_value_tensors() instead
  virtual bool is_batch_cached() const = 0;
  virtual size_t get_current_inflight_id() const = 0;
  virtual cudaStream_t get_split_3_way_stream(int raw_device_id) const = 0;
  virtual cudaStream_t get_d2d_stream(int raw_device_id) const = 0;
  virtual void set_schedule_streams(cudaStream_t s3w_stream, cudaStream_t d2d_stream,
                                    int raw_device_id) = 0;
  virtual void schedule_split_3_way_here(cudaStream_t stream, int raw_device_id,
                                         bool from_graph) = 0;
  virtual void schedule_d2d_here(cudaStream_t stream, int raw_device_id, bool from_graph) = 0;
  virtual void schedule_here(cudaStream_t stream, int raw_device_id) = 0;
  virtual void schedule_here_graph(cudaStream_t stream, int raw_device_id) = 0;
  virtual void update_schedule_graph(int raw_device_id) = 0;
  virtual void stream_wait_sparse_tensors(cudaStream_t stream, int raw_device_id,
                                          bool from_graph) = 0;
  virtual void stream_wait_dense_tensors(cudaStream_t stream, int raw_device_id,
                                         bool from_graph) = 0;
  virtual std::vector<TensorBag2> get_label_tensors() const = 0;
  virtual std::vector<TensorBag2> get_dense_tensors() const = 0;
};

class SchedulableEmbeding : public IEmbedding {
 public:
  virtual ~SchedulableEmbeding() = default;

  virtual void assign_input_tensors(bool is_train, size_t batch_size, size_t inflight_id,
                                    bool cached) = 0;
  virtual void index_calculation(bool is_train, int i) = 0;
  virtual void freq_forward(bool is_train, int i, bool is_first_eval_batch = true) = 0;
  virtual void freq_backward(int i) = 0;
  virtual void freq_update_params(int i) = 0;
  virtual void infreq_model_forward(int i) = 0;
  virtual void infreq_network_forward(bool is_train, int i) = 0;
  virtual void global_barrier(bool is_train, int i) = 0;
  virtual void infreq_network_backward(int i) = 0;
  virtual void infreq_model_backward(int i) = 0;
};

}  // namespace HugeCTR