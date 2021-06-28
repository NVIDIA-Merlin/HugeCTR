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

#include <data_readers/data_reader_worker_group.hpp>
#include <data_readers/data_reader_worker_raw.hpp>

namespace HugeCTR {

template <typename TypeKey>
class DataReaderWorkerGroupRaw : public DataReaderWorkerGroup {
  std::shared_ptr<RawOffsetList> file_offset_list_;
  bool create_offset_{true};
  long long num_samples_;
  long long stride_;
  long long batchsize_;
  bool data_shuffle_;

  std::shared_ptr<Source> create_source(size_t worker_id, size_t num_worker,
      const std::string& file_name, bool repeat) override {

    std::shared_ptr<RawOffsetList> file_offset_list;
    if (!worker_id && create_offset_) {
      file_offset_list_.reset(new RawOffsetList(file_name, num_samples_, stride_, batchsize_,
                                                data_shuffle_, num_worker, repeat));
      create_offset_ = false;
    }
    file_offset_list = file_offset_list_;
    create_offset_ = (worker_id == num_worker - 1) ? true : create_offset_;

    return std::make_shared<MmapSource>(file_offset_list, worker_id);
  }

 public:
  // Ctor
  DataReaderWorkerGroupRaw(std::shared_ptr<HeapEx<CSRChunk<TypeKey>>> csr_heap,
                           std::string file_name, long long num_samples, bool repeat,
                           const std::vector<DataReaderSparseParam> params,
                           const std::vector<long long> slot_offset, int label_dim, int dense_dim,
                           int batchsize, bool float_label_dense, bool data_shuffle = false,
                           bool start_reading_from_beginning = true)
      : DataReaderWorkerGroup(start_reading_from_beginning, DataReaderType_t::Raw),
        num_samples_(num_samples),
        batchsize_(batchsize),
        data_shuffle_(data_shuffle) {
    // todo param check
    if (file_name.empty()) {
      CK_THROW_(Error_t::WrongInput, "file_name.empty()");
    }

    {
      int slots = 0;
      for (auto& param : params) {
        slots += param.slot_num;
      }
      size_t stride = slots * sizeof(int) +
                      (label_dim + dense_dim) * (float_label_dense ? sizeof(float) : sizeof(int));
      file_offset_list_.reset(new RawOffsetList(file_name, num_samples, stride, batchsize,
                                                data_shuffle, csr_heap->get_size(), repeat));
      stride_ = stride;
    }

    for (int i = 0; i < csr_heap->get_size(); i++) {
      std::shared_ptr<IDataReaderWorker> data_reader(new DataReaderWorkerRaw<TypeKey>(
          i, csr_heap->get_size(), file_offset_list_, csr_heap, repeat, params, slot_offset,
          label_dim, float_label_dense));
      data_readers_.push_back(data_reader);
    }
    create_data_reader_threads();
  }
};
}  // namespace HugeCTR
