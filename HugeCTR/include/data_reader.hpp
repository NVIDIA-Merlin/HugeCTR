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

#include <atomic>
#include <common.hpp>
#include <fstream>
#include <gpu_resource.hpp>
#include <utils.hpp>
#include <tensor2.hpp>
#include <vector>

namespace HugeCTR {

/**
 * @brief Data reading controller.
 *
 * Control the data reading from data set to embedding.
 * An instance of DataReader will maintain independent
 * threads for data reading (IDataReaderWorker)
 * from dataset to heap. Meanwhile one independent
 * thread consumes the data (DataCollector),
 * and copy the data to GPU buffer.
 */
class IDataReader {
 public:
  virtual ~IDataReader() {}

  virtual TensorScalarType get_scalar_type() const = 0;

  virtual long long read_a_batch_to_device() = 0;
  virtual long long read_a_batch_to_device_delay_release() = 0;
  virtual long long get_current_batchsize_per_device(size_t local_id) = 0;
  virtual long long get_full_batchsize() const = 0;
  virtual void ready_to_collect() = 0;
  virtual bool is_started() const = 0;
  virtual void start() = 0;

  virtual void create_drwg_norm(std::string file_list, 
                        Check_t check_type,
                        bool start_reading_from_beginning = true) = 0;
  virtual void create_drwg_raw( std::string file_name, 
                        long long num_samples,
                        bool float_label_dense,
                        bool data_shuffle, 
                        bool start_reading_from_beginning = true) = 0;

  virtual void create_drwg_parquet( std::string file_list,const std::vector<long long> slot_offset,
                            bool start_reading_from_beginning = true) = 0;

  // TODO(xiaoleis, 01182021): add SourceType_t to allow user to change the type
  virtual void set_source(std::string file_name = std::string()) = 0;
};

class IDataReaderWithScheduling : public IDataReader {
 public:
  virtual void schedule_here(cudaStream_t stream, int raw_device_id) = 0;
  virtual void schedule_here_graph(cudaStream_t stream, int raw_device_id) = 0;
  virtual void update_schedule_graph(int raw_device_id) = 0;
};

}  // namespace HugeCTR