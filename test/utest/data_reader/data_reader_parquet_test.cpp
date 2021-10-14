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

#include <HugeCTR/include/resource_managers/resource_manager_ext.hpp>
#include <fstream>

#include "HugeCTR/include/data_generator.hpp"
#include "HugeCTR/include/data_readers/data_reader.hpp"
#include "HugeCTR/include/data_readers/file_list.hpp"
#include "HugeCTR/include/data_readers/parquet_data_reader_worker.hpp"
#include "HugeCTR/include/resource_managers/resource_manager_ext.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <rmm/device_buffer.hpp>
#pragma GCC diagnostic pop

using namespace HugeCTR;

const std::vector<long long> slot_size = {
    39884406, 39043,    17289,    7420,     20263,  3,     7120, 1543, 63,
    38532951, 2953546,  403346,   10,       2208,   11938, 155,  4,    976,
    14,       39979771, 25641295, 39664984, 585935, 12972, 108,  36};
// const std::vector<long long> slot_size(26, 0);
std::vector<bool> is_mhot(26, false);

const int max_nnz = 5;
const int slot_num = 26;
const int label_dim = 1;
const int dense_dim = 13;
using LABEL_TYPE = float;
using DENSE_TYPE = float;
using CAT_TYPE = int64_t;

const Check_t CHK = Check_t::None;
const std::string prefix("./data_reader_parquet_test_data/");
const std::string file_list_name("data_reader_parquet_file_list.txt");
using CVector = std::vector<std::unique_ptr<cudf::column>>;
typedef long long T;
// generate parquet files and return data in the form of `row-major` stored in vector<>
void generate_parquet_input_files(int num_files, int sample_per_file, std::vector<bool>& is_mhot,
                                  std ::vector<LABEL_TYPE>& labels, std::vector<DENSE_TYPE>& denses,
                                  std::vector<int32_t>& row_offsets,
                                  std::vector<CAT_TYPE>& sparse_values) {
  check_make_dir(prefix);
  denses.resize(num_files * sample_per_file * dense_dim);
  row_offsets.resize(num_files * sample_per_file * slot_num, 0);
  size_t dense_feature_per_file = dense_dim * sample_per_file;
  for (int file_num = 0; file_num < num_files; file_num++) {
    CVector cols;
    // create label vector
    for (int i = 0; i < label_dim; i++) {
      std::vector<LABEL_TYPE> label_vector(sample_per_file, 0);
      for (auto& c : label_vector) {
        auto val = LABEL_TYPE(std::rand() % 2);
        c = val;  // 0,1 value
        labels.push_back(val);
      }

      rmm::device_buffer dev_buffer(label_vector.data(), sizeof(LABEL_TYPE) * sample_per_file, rmm::cuda_stream_default);
      auto pcol = std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<LABEL_TYPE>()}, cudf::size_type(sample_per_file), std::move(dev_buffer));
      cols.emplace_back(std::move(pcol));
    }
    // create dense vector
    for (int i = 0; i < dense_dim; i++) {
      std::vector<DENSE_TYPE> dense_vector(sample_per_file, 0);
      for (int sample = 0; sample < sample_per_file; sample++) {
        auto val = DENSE_TYPE(std::rand());
        dense_vector[sample] = val;
        denses[file_num * dense_feature_per_file + sample * dense_dim + i] = val;
      };

      rmm::device_buffer dev_buffer(dense_vector.data(), sizeof(DENSE_TYPE) * sample_per_file, rmm::cuda_stream_default);
      auto pcol = std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<DENSE_TYPE>()}, cudf::size_type(sample_per_file), std::move(dev_buffer));
      cols.emplace_back(std::move(pcol));
    }
    std::vector<std::vector<CAT_TYPE>> slot_vectors(slot_num);
    std::vector<std::vector<int32_t>> row_off(slot_num,
                                              std::vector<int32_t>(sample_per_file + 1, 0));
    for (int sample = 0; sample < sample_per_file; sample++) {
      for (int slot = 0; slot < slot_num; slot++) {
        int nnz_cur_slot = std::rand() % max_nnz;
        nnz_cur_slot = std::max(1, nnz_cur_slot);
        if (!is_mhot[slot]) nnz_cur_slot = 1;
        row_off[slot][sample + 1] = nnz_cur_slot + row_off[slot][sample];
        row_offsets[file_num * sample_per_file * slot_num + sample * slot_num + slot] =
            nnz_cur_slot;
        for (int nnz = 0; nnz < nnz_cur_slot; nnz++) {
          auto val = CAT_TYPE(std::rand());
          slot_vectors[slot].push_back(val);
          sparse_values.push_back(val);
        }
      }
    }

    constexpr size_t bitmask_bits = cudf::detail::size_in_bits<cudf::bitmask_type>();
    size_t bits = (sample_per_file + bitmask_bits - 1) / bitmask_bits;
    std::vector<cudf::bitmask_type> null_mask(bits, 0);

    // construct columns for all slots
    for (int i = 0; i < slot_num; i++) {
      auto& slot_vector = slot_vectors[i];
      auto& row_off_vector = row_off[i];

      if (!is_mhot[i]) {
        rmm::device_buffer dev_buffer(slot_vector.data(), sizeof(CAT_TYPE) * slot_vector.size(), rmm::cuda_stream_default);
        cols.emplace_back(
            std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<CAT_TYPE>()},
                                           cudf::size_type(slot_vector.size()), std::move(dev_buffer)));
      } else {
        rmm::device_buffer dev_buffer_0(slot_vector.data(), sizeof(CAT_TYPE) * slot_vector.size(), rmm::cuda_stream_default);
        auto child =
            std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<CAT_TYPE>()},
                                           cudf::size_type(slot_vector.size()), std::move(dev_buffer_0));
        rmm::device_buffer dev_buffer_1(row_off_vector.data(),
                                        sizeof(int32_t) * row_off_vector.size(), rmm::cuda_stream_default);
        auto row_off =
            std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<int32_t>()},
                                           cudf::size_type(row_off_vector.size()), std::move(dev_buffer_1));
        // auto cur_col =
        auto null_mask_df = rmm::device_buffer(null_mask.data(), null_mask.size() * sizeof(cudf::bitmask_type), rmm::cuda_stream_default);
        cols.emplace_back(cudf::make_lists_column(
            sample_per_file, std::move(row_off), std::move(child), cudf::UNKNOWN_NULL_COUNT,
            std::move(null_mask_df), rmm::cuda_stream_default));
      }
    }
    cudf::table input_table(std::move(cols));

    std::string filepath = prefix + std::to_string(file_num) + std::string(".parquet");
    cudf::io::parquet_writer_options writer_args =
        cudf_io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input_table.view());
    cudf::io::write_parquet(writer_args);
  }

  std::ofstream output_file_stream;
  output_file_stream.open(file_list_name, std::ofstream::out);
  output_file_stream << num_files << std::endl;
  for (int i = 0; i < num_files; i++) {
    std::string filepath = prefix + "./" + std::to_string(i) + std::string(".parquet");
    output_file_stream << filepath << std::endl;
  }
  output_file_stream.close();

  // also write metadata
  std::stringstream metadata;
  metadata << "{ \"file_stats\": [";
  for (int i = 0; i < num_files - 1; i++) {
    std::string filepath = std::to_string(i) + std::string(".parquet");
    metadata << "{\"file_name\": \"" << filepath << "\", "
             << "\"num_rows\":" << sample_per_file << "}, ";
  }
  std::string filepath = std::to_string(num_files - 1) + std::string(".parquet");
  metadata << "{\"file_name\": \"" << filepath << "\", "
           << "\"num_rows\":" << sample_per_file << "} ";
  metadata << "], ";
  metadata << "\"labels\": [";
  for (int i = 0; i < label_dim; i++) {
    metadata << "{\"col_name\": \"label\", "
             << "\"index\":" << i << "} ";
  }
  metadata << "], ";

  metadata << "\"conts\": [";
  for (int i = label_dim; i < (label_dim + dense_dim - 1); i++) {
    metadata << "{\"col_name\": \"C" << i << "\", "
             << "\"index\":" << i << "}, ";
  }
  metadata << "{\"col_name\": \"C" << (label_dim + dense_dim - 1) << "\", "
           << "\"index\":" << (label_dim + dense_dim - 1) << "} ";
  metadata << "], ";

  metadata << "\"cats\": [";
  for (int i = label_dim + dense_dim; i < (label_dim + dense_dim + slot_num - 1); i++) {
    metadata << "{\"col_name\": \"C" << i << "\", "
             << "\"index\":" << i << "}, ";
  }
  metadata << "{\"col_name\": \"C" << (label_dim + dense_dim + slot_num - 1) << "\", "
           << "\"index\":" << (label_dim + dense_dim + slot_num - 1) << "} ";
  metadata << "] ";
  metadata << "}";

  std::ofstream metadata_file_stream;
  metadata_file_stream.open(prefix + "_metadata.json", std::ofstream::out);
  metadata_file_stream << metadata.rdbuf();
  metadata_file_stream.close();
}

TEST(data_reader_parquet_worker, data_reader_parquet_worker_single_worker_iter) {
  auto p_mr = rmm::mr::get_current_device_resource();
  std::vector<LABEL_TYPE> labels;
  std::vector<LABEL_TYPE> denses;
  // dim is (total_sample + 1)
  std::vector<int32_t> row_offsets;
  std::vector<CAT_TYPE> sparse_values;
  is_mhot[0] = true;
  is_mhot[3] = true;
  is_mhot[5] = true;
  int sample_per_file = 2048;
  int num_files = 3;
  // size_t total_samples = sample_per_file * num_files;
  generate_parquet_input_files(num_files, sample_per_file, is_mhot, labels, denses, row_offsets,
                               sparse_values);

  int numprocs = 1;
  std::vector<std::vector<int>> vvgpu;
  std::vector<int> device_list = {0};
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  auto gpu_resource_group = ResourceManagerExt::create(vvgpu, 0);
  // const int num_devices = 1;
  const int batchsize = 1024;
  const DataReaderSparseParam param = {"localized", std::vector<int>(slot_num, max_nnz), false,
                                       slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  CudaDeviceContext context(0);
  auto buff = GeneralBuffer2<CudaAllocator>::create();
  // create buffer for a reader worker
  std::shared_ptr<ThreadBuffer> thread_buffer = std::make_shared<ThreadBuffer>();
  // readytowrite
  thread_buffer->state.store(BufferState::ReadyForWrite);
  thread_buffer->batch_size = batchsize;
  thread_buffer->param_num = params.size();
  thread_buffer->label_dim = label_dim;
  thread_buffer->dense_dim = dense_dim;

  int batch_start = 5;
  int batch_end = 10;
  int batch_current_worker = batch_end - batch_start;
  thread_buffer->batch_size_start_idx = batch_start;
  thread_buffer->batch_size_end_idx = batch_end;
  for (size_t i = 0; i < params.size(); ++i) {
    auto& param = params[i];
    thread_buffer->is_fixed_length.push_back(params[i].is_fixed_length);
    SparseTensor<T> sparse_tensor;
    buff->reserve({(size_t)batchsize, (size_t)param.max_feature_num}, param.slot_num,
                  &sparse_tensor);
    thread_buffer->device_sparse_buffers.push_back(sparse_tensor.shrink());
  }
  Tensor2<float> label_dense_tensor;
  buff->reserve({(size_t)batchsize, (size_t)(label_dim + dense_dim)}, &label_dense_tensor);
  thread_buffer->device_dense_buffers = label_dense_tensor.shrink();
  buff->allocate();

  std::vector<long long> slot_offset(slot_size.size(), 0);
  for (unsigned int i = 1; i < slot_size.size(); i++) {
    slot_offset[i] = slot_offset[i - 1] + slot_size[i - 1];
  }
  int loop_flag = 1;
  // setup a data reader
  ParquetDataReaderWorker<T> data_reader(0, 1, gpu_resource_group->get_local_gpu(0), &loop_flag,
                                         thread_buffer, file_list_name, true , params,
                                         slot_offset, 0, gpu_resource_group);

  int iter = 7;
  size_t sample_offset = 0;
  size_t nnz_offset = 0;
  size_t total_samples = num_files * sample_per_file;
  size_t total_nnz = std::accumulate(row_offsets.begin(), row_offsets.end(), 0);
  for (int i = 0; i < iter; i++) {
    data_reader.read_a_batch();

    auto sparse_tensorbag = thread_buffer->device_sparse_buffers[0];
    auto sparse_tensor = SparseTensor<T>::stretch_from(sparse_tensorbag);
    size_t nnz = sparse_tensor.nnz();
    std::unique_ptr<T[]> keys(new T[nnz]);
    std::unique_ptr<T[]> row_offset_read_a_batch(new T[batchsize * slot_num + 1]);
    CK_CUDA_THROW_(cudaMemcpy(keys.get(), sparse_tensor.get_value_ptr(), nnz * sizeof(T),
                              cudaMemcpyDeviceToHost));
    CK_CUDA_THROW_(cudaMemcpy(row_offset_read_a_batch.get(), sparse_tensor.get_rowoffset_ptr(),
                              (batchsize * slot_num + 1) * sizeof(T), cudaMemcpyDeviceToHost));
    for (int nnz_id = 0; nnz_id < batchsize * slot_num; ++nnz_id) {
      T expected = row_offsets[(sample_offset * slot_num + nnz_id) % (total_samples * slot_num)];
      T value = row_offset_read_a_batch[nnz_id + 1] - row_offset_read_a_batch[nnz_id];
      ASSERT_TRUE(value == expected) << " iter: " << i << " idx: " << nnz_id << " value: " <<
      value
                                     << " expected: " << expected;
      for (T start = row_offset_read_a_batch[nnz_id]; start < row_offset_read_a_batch[nnz_id +
      1];
           start++) {
        int slot_id = nnz_id % slot_num;
        // std::cout<< "idx:" << start<<" slot_id " <<slot_id
        // <<" slot_offset "<<slot_offset[slot_id]
        // <<" read :"<< sparse_values[(nnz_offset + start) % total_nnz] <<" vs "<<keys[start] -
        // slot_offset[slot_id]<<std::endl;
        ASSERT_TRUE(sparse_values[(nnz_offset + start) % total_nnz] ==
                    keys[start] - slot_offset[slot_id])
            << "idx:" << start;
      }
    }
    // for (size_t nnz_id = 0; nnz_id < nnz; ++nnz_id) {
    //   ASSERT_TRUE(sparse_values[(nnz_offset + nnz_id) % total_nnz] == keys[nnz_id])
    //       << "idx:" << nnz_id;
    // }
    int label_dense_dim = label_dim + dense_dim;
    auto dense_tensor_bag = thread_buffer->device_dense_buffers;
    auto dense_tensor = Tensor2<DENSE_TYPE>::stretch_from(dense_tensor_bag);

    std::unique_ptr<DENSE_TYPE[]> dense(new DENSE_TYPE[batchsize * (label_dim + dense_dim)]);
    CK_CUDA_THROW_(cudaMemcpy(dense.get(), dense_tensor.get_ptr(),
                              (batch_end - batch_start) * label_dense_dim * sizeof(DENSE_TYPE),
                              cudaMemcpyDeviceToHost));
    for (int sample = 0; sample < batch_current_worker; ++sample) {
      // use != for float, because there's no computation
      if (dense[sample * label_dense_dim] !=
          labels[(sample_offset + sample + batch_start) % (total_samples)]) {
        std::cout << "sample " << sample << " label error " << std::endl;
        exit(-1);
      }

      for (int d = 0; d < dense_dim; d++) {
        if (dense[sample * label_dense_dim + d + 1] !=
            denses[((sample_offset + sample + batch_start) * dense_dim + d) %
                   (total_samples * dense_dim)]) {
          std::cout << "sample " << i << " dense error " << std::endl;
        }
      }
    }
    sample_offset += batchsize;
    nnz_offset += nnz;
    auto expected = BufferState::ReadyForRead;
    while (thread_buffer->state.compare_exchange_weak(expected, BufferState::ReadyForWrite)) {
      expected = BufferState::ReadyForRead;
    }
  }
  rmm::mr::set_current_device_resource(p_mr);
}

TEST(data_reader_group_test, data_reader_group_test_3files_1worker_iter) {
  auto p_mr = rmm::mr::get_current_device_resource();

  std::vector<LABEL_TYPE> labels;
  std::vector<LABEL_TYPE> denses;
  std::vector<int32_t> row_offsets;
  std::vector<CAT_TYPE> sparse_values;
  is_mhot[0] = true;
  is_mhot[3] = true;
  is_mhot[5] = true;
  int sample_per_file = 2048;
  int num_files = 3;
  generate_parquet_input_files(num_files, sample_per_file, is_mhot, labels, denses, row_offsets,
                               sparse_values);

  const int batchsize = 1024;
  int numprocs = 1;
  std::vector<int> device_list = {0};
  std::vector<std::vector<int>> vvgpu;
  const size_t batchsize_per_gpu = batchsize / device_list.size();
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  const auto& resource_manager = ResourceManagerExt::create(vvgpu, 0);
  const DataReaderSparseParam param = {"distributed", std::vector<int>(slot_num, max_nnz), false,
                                       slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  CudaDeviceContext context(0);
  auto buff = GeneralBuffer2<CudaAllocator>::create();
  // create buffer for a reader worker
  std::shared_ptr<ThreadBuffer> thread_buffer = std::make_shared<ThreadBuffer>();
  // readytowrite
  thread_buffer->state.store(BufferState::ReadyForWrite);
  thread_buffer->batch_size = batchsize;
  thread_buffer->param_num = params.size();
  thread_buffer->label_dim = label_dim;
  thread_buffer->dense_dim = dense_dim;

  int batch_start = 5;
  int batch_end = 10;
  thread_buffer->batch_size_start_idx = batch_start;
  thread_buffer->batch_size_end_idx = batch_end;
  for (size_t i = 0; i < params.size(); ++i) {
    auto& param = params[i];
    thread_buffer->is_fixed_length.push_back(params[i].is_fixed_length);
    SparseTensor<T> sparse_tensor;
    buff->reserve({(size_t)batchsize, (size_t)param.max_feature_num}, param.slot_num,
                  &sparse_tensor);
    thread_buffer->device_sparse_buffers.push_back(sparse_tensor.shrink());
  }
  Tensor2<float> label_dense_tensor;
  buff->reserve({(size_t)batchsize, (size_t)(label_dim + dense_dim)}, &label_dense_tensor);
  thread_buffer->device_dense_buffers = label_dense_tensor.shrink();
  buff->allocate();

  std::vector<long long> slot_offset(slot_size.size(), 0);
  for (unsigned int i = 1; i < slot_size.size(); i++) {
    slot_offset[i] = slot_offset[i - 1] + slot_size[i - 1];
  }
  size_t local_gpu_count = resource_manager->get_local_gpu_count();

  DataReader<T> data_reader(batchsize, label_dim, dense_dim, params, resource_manager, true,
                            device_list.size(), false);
  auto& sparse_tensorbag = data_reader.get_sparse_tensors("distributed");
  auto& label_tensorbag = data_reader.get_label_tensors();
  auto& dense_tensorbag = data_reader.get_dense_tensors();
  data_reader.create_drwg_parquet(file_list_name, slot_offset, true);
  int iter = 6;
  size_t nnz_offset = 0;
  size_t sample_offset = 0;
  size_t total_nnz = std::accumulate(row_offsets.begin(), row_offsets.end(), 0);
  size_t total_samples = num_files * sample_per_file;
  for (int i = 0; i < iter; i++) {
    // std::cout << "iter " << i << std::endl;
    long long current_batchsize = data_reader.read_a_batch_to_device();
    ASSERT_TRUE(current_batchsize == batchsize);
    for (size_t gpu = 0; gpu < local_gpu_count; ++gpu) {
      auto sparse_tensor = SparseTensor<T>::stretch_from(sparse_tensorbag[gpu]);
      auto label_tensor = Tensor2<LABEL_TYPE>::stretch_from(label_tensorbag[gpu]);
      auto dense_tensor = Tensor2<DENSE_TYPE>::stretch_from(dense_tensorbag[gpu]);
      size_t label_size = label_tensor.get_num_elements();
      size_t dense_size = dense_tensor.get_num_elements();
      // std::cout << "label size " << label_size << " dense size " << dense_size << std::endl;
      ASSERT_TRUE(label_size == batchsize_per_gpu * label_dim &&
                  dense_size == batchsize_per_gpu * dense_dim);
      std::unique_ptr<LABEL_TYPE[]> label_read(new LABEL_TYPE[label_size]);
      std::unique_ptr<DENSE_TYPE[]> dense_read(new DENSE_TYPE[dense_size]);

      CK_CUDA_THROW_(cudaMemcpy(label_read.get(), label_tensor.get_ptr(),
                                label_size * sizeof(LABEL_TYPE), cudaMemcpyDeviceToHost));
      CK_CUDA_THROW_(cudaMemcpy(dense_read.get(), dense_tensor.get_ptr(),
                                dense_size * sizeof(DENSE_TYPE), cudaMemcpyDeviceToHost));

      int batch_starting = gpu * batchsize_per_gpu;
      for (size_t sample = 0; sample < batchsize_per_gpu; sample++) {
        ASSERT_TRUE(labels[sample + batch_starting + sample_offset] == label_read[sample]);
        for (int d = 0; d < dense_dim; d++) {
          ASSERT_TRUE(denses[(sample + batch_starting + sample_offset) * dense_dim + d] ==
                      dense_read[sample * dense_dim + d]);
        }
      }
      size_t nnz = sparse_tensor.nnz();
      std::unique_ptr<CAT_TYPE[]> keys(new CAT_TYPE[nnz]);
      CK_CUDA_THROW_(cudaMemcpy(keys.get(), sparse_tensor.get_value_ptr(), nnz *
      sizeof(CAT_TYPE),
                                cudaMemcpyDeviceToHost));

      std::unique_ptr<T[]> rowoffsets(new T[batchsize * slot_num + 1]);
      CK_CUDA_THROW_(cudaMemcpy(rowoffsets.get(), sparse_tensor.get_rowoffset_ptr(),
                                (1 + batchsize * slot_num) * sizeof(T), cudaMemcpyDeviceToHost));
      for (int nnz_id = 0; nnz_id < batchsize * slot_num; ++nnz_id) {
        T expected = row_offsets[(sample_offset * slot_num + nnz_id) % (total_samples *
        slot_num)]; T value = rowoffsets[nnz_id + 1] - rowoffsets[nnz_id]; ASSERT_TRUE(value ==
        expected) << " iter: " << i << " idx: " << nnz_id
                                       << " value: " << value << " expected: " << expected;
        for (T start = rowoffsets[nnz_id]; start < rowoffsets[nnz_id + 1];start++) {
          int slot_id = nnz_id % slot_num;
          ASSERT_TRUE(sparse_values[(nnz_offset + start) % total_nnz] ==
                      keys[start] - slot_offset[slot_id])
              << "idx:" << start;
        }
      }
      if (gpu == local_gpu_count - 1) nnz_offset += nnz;
      T generated_nnz = rowoffsets[batchsize * slot_num];
      if (gpu == local_gpu_count - 1) sample_offset += batchsize;
      ASSERT_TRUE(nnz == static_cast<size_t>(generated_nnz))
          << "nnz is " << generated_nnz << " expected " << nnz;
    }
  }
  rmm::mr::set_current_device_resource(p_mr);
}
// TODO when num_files is not a multiple of workers, reference can be wrong,
// but the woker itself can work

// say 3 files but 2 workers are specified, where a file contains 2 batches
// batch 0 -> file 0 (worker 0)
// batch 1 -> file 1 (worker 1)
// batch 2 -> file 0 (worker 0)
// batch 3 -> file 1 (worker 1)
// batch 4 -> file 2 (worker 0)
// batch 5 -> file 0 (worker 1, repeated)
// batch 6 -> file 2 (worker 0)
// batch 7 -> file 0 (worker 1,repeated)
TEST(data_reader_test, data_reader_group_test_3files_3workers_iter) {
  auto p_mr = rmm::mr::get_current_device_resource();
  // store the reference dataset
  std::vector<LABEL_TYPE> labels;
  std::vector<LABEL_TYPE> denses;
  std::vector<int32_t> row_offsets;
  std::vector<CAT_TYPE> sparse_values;

  is_mhot[0] = true;
  is_mhot[3] = true;
  is_mhot[5] = true;
  int sample_per_file = 2048;
  int num_files = 3;
  generate_parquet_input_files(num_files, sample_per_file, is_mhot, labels, denses, row_offsets,
                               sparse_values);

  const int batchsize = 1026;
  int numprocs = 1;
  std::vector<int> device_list = {0, 1, 3};
  std::vector<std::vector<int>> vvgpu;
  size_t batchsize_per_gpu = batchsize / device_list.size();
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }

  //! assume num_files is a multiple of #workers so that each file is only confined to one worker
  ASSERT_TRUE(num_files % device_list.size() == 0);
  int files_per_worker = num_files / device_list.size();

  const auto& resource_manager = ResourceManagerExt::create(vvgpu, 0);
  const DataReaderSparseParam param = {"distributed", std::vector<int>(slot_num, max_nnz), false,
                                       slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  std::vector<long long> slot_offset(slot_size.size(), 0);
  for (unsigned int i = 1; i < slot_size.size(); i++) {
    slot_offset[i] = slot_offset[i - 1] + slot_size[i - 1];
  }
  size_t local_gpu_count = resource_manager->get_local_gpu_count();
  DataReader<T> data_reader(batchsize, label_dim, dense_dim, params, resource_manager, true,
                            device_list.size(), false);
  auto& sparse_tensorbag = data_reader.get_sparse_tensors("distributed");
  auto& label_tensorbag = data_reader.get_label_tensors();
  auto& dense_tensorbag = data_reader.get_dense_tensors();
  data_reader.create_drwg_parquet(file_list_name,slot_offset, true);

  std::vector<size_t> nnz_offset(local_gpu_count, 0);
  std::vector<size_t> sample_offset(local_gpu_count, 0);
  std::vector<size_t> total_nnz(local_gpu_count, 0);
  std::vector<size_t> total_slots(local_gpu_count, 0);
  size_t total_nnz_files = 0;
  const size_t slots_per_file = sample_per_file * slot_num;
  std::vector<std::vector<int32_t>> row_offsets_per_worker(local_gpu_count);
  std::vector<std::vector<CAT_TYPE>> sparse_value_per_worker(local_gpu_count);
  std::vector<std::vector<LABEL_TYPE>> label_per_worker(local_gpu_count);
  std::vector<std::vector<DENSE_TYPE>> dense_per_worker(local_gpu_count);

  for (auto& dense_vec : dense_per_worker) {
    dense_vec.resize(files_per_worker * sample_per_file * dense_dim);
  }
  for (auto& label_vec : label_per_worker) {
    label_vec.resize(files_per_worker * sample_per_file * label_dim);
  }

  // pack data for reference
  for (int i = 0; i < num_files; i++) {
    size_t cur_file_nnz = std::accumulate(row_offsets.begin() + slots_per_file * i,
                                          row_offsets.begin() + slots_per_file * (i + 1), 0);
    int worker_id = i % local_gpu_count;
    int round = i / local_gpu_count;
    total_nnz[worker_id] += cur_file_nnz;
    total_slots[worker_id] += slots_per_file;
    row_offsets_per_worker[worker_id].resize(total_slots[worker_id], 0);
    sparse_value_per_worker[worker_id].resize(total_nnz[worker_id], 0);
    std::copy(row_offsets.begin() + slots_per_file * i,
              row_offsets.begin() + slots_per_file * (i + 1),
              row_offsets_per_worker[worker_id].begin() + total_slots[worker_id] -
              slots_per_file);
    std::copy(sparse_values.begin() + total_nnz_files,
              sparse_values.begin() + total_nnz_files + cur_file_nnz,
              sparse_value_per_worker[worker_id].begin() + total_nnz[worker_id] - cur_file_nnz);
    std::copy(labels.begin() + label_dim * i * sample_per_file,
              labels.begin() + label_dim * (i + 1) * sample_per_file,
              label_per_worker[worker_id].begin() + round * label_dim * sample_per_file);
    std::copy(denses.begin() + dense_dim * i * sample_per_file,
              denses.begin() + dense_dim * (i + 1) * sample_per_file,
              dense_per_worker[worker_id].begin() + round * dense_dim * sample_per_file);
    total_nnz_files += cur_file_nnz;
  }
  int iter = 12;
  for (int i = 0; i < iter; i++) {
    int worker_id = i % local_gpu_count;
    int round = i / local_gpu_count;
    long long current_batchsize = data_reader.read_a_batch_to_device();
    ASSERT_TRUE(current_batchsize == batchsize);
    for (size_t gpu = 0; gpu < local_gpu_count; ++gpu) {
      auto sparse_tensor = SparseTensor<T>::stretch_from(sparse_tensorbag[gpu]);
      auto label_tensor = Tensor2<LABEL_TYPE>::stretch_from(label_tensorbag[gpu]);
      auto dense_tensor = Tensor2<DENSE_TYPE>::stretch_from(dense_tensorbag[gpu]);
      size_t label_size = label_tensor.get_num_elements();
      size_t dense_size = dense_tensor.get_num_elements();
      // std::cout << "label size ??? " << label_size << " dense size " << dense_size <<
      // std::endl;
      ASSERT_TRUE(label_size == batchsize_per_gpu * label_dim &&
                  dense_size == batchsize_per_gpu * dense_dim);
      std::unique_ptr<LABEL_TYPE[]> label_read(new LABEL_TYPE[label_size]);
      std::unique_ptr<DENSE_TYPE[]> dense_read(new DENSE_TYPE[dense_size]);

      CK_CUDA_THROW_(cudaMemcpy(label_read.get(), label_tensor.get_ptr(),
                                label_size * sizeof(LABEL_TYPE), cudaMemcpyDeviceToHost));
      CK_CUDA_THROW_(cudaMemcpy(dense_read.get(), dense_tensor.get_ptr(),
                                dense_size * sizeof(DENSE_TYPE), cudaMemcpyDeviceToHost));

      int batch_starting = gpu * batchsize_per_gpu;
      for (size_t sample = 0; sample < batchsize_per_gpu; sample++) {
        for (int l = 0; l < label_dim; l++) {
          size_t label_idx = ((sample + batch_starting + round * batchsize) * label_dim + l) %
                             label_per_worker[worker_id].size();
          ASSERT_TRUE(label_per_worker[worker_id][label_idx] == label_read[sample * label_dim +
          l]);
        }

        for (int d = 0; d < dense_dim; d++) {
          size_t dense_idx = ((sample + batch_starting + round * batchsize) * dense_dim + d) %
                             dense_per_worker[worker_id].size();
          ASSERT_TRUE(dense_per_worker[worker_id][dense_idx] == dense_read[sample * dense_dim +
          d]);
        }
      }

      size_t nnz = sparse_tensor.nnz();
      std::unique_ptr<CAT_TYPE[]> keys(new CAT_TYPE[nnz]);
      CK_CUDA_THROW_(cudaMemcpy(keys.get(), sparse_tensor.get_value_ptr(), nnz *
      sizeof(CAT_TYPE),
                                cudaMemcpyDeviceToHost));

      std::unique_ptr<T[]> rowoffsets(new T[batchsize * slot_num + 1]);
      CK_CUDA_THROW_(cudaMemcpy(rowoffsets.get(), sparse_tensor.get_rowoffset_ptr(),
                                (1 + batchsize * slot_num) * sizeof(T), cudaMemcpyDeviceToHost));
      for (int nnz_id = 0; nnz_id < batchsize * slot_num; ++nnz_id) {
        T expected =
            row_offsets_per_worker[worker_id][(sample_offset[worker_id] * slot_num + nnz_id) %
                                              (total_slots[worker_id])];
        T value = rowoffsets[nnz_id + 1] - rowoffsets[nnz_id];
        ASSERT_TRUE(value == expected)
            << " iter: " << i << " idx: " << nnz_id << " worker " << worker_id << "sample_offset"
            << sample_offset[worker_id] << " value: " << value << " expected: " << expected
            << " expected idx "
            << (sample_offset[worker_id] * slot_num + nnz_id) % (total_slots[worker_id]);
        for (T start = rowoffsets[nnz_id]; start < rowoffsets[nnz_id + 1]; ++start) {
          int slot_id = nnz_id % slot_num;
          ASSERT_TRUE(sparse_value_per_worker[worker_id][(nnz_offset[worker_id] + start) %
                                                        total_nnz[worker_id]] ==
                                                        keys[start]-slot_offset[slot_id])
              << "idx: " << nnz_id << " "
              << sparse_value_per_worker[worker_id]
                                        [(nnz_offset[worker_id] + start) % total_nnz[worker_id]]
              << " (ref) vs " << keys[start] - slot_offset[slot_id]<< " read" << std::endl;
        }
      }
      T generated_nnz = rowoffsets[batchsize * slot_num];
      if (gpu == local_gpu_count - 1) sample_offset[worker_id] += batchsize;
      if (gpu == local_gpu_count - 1) nnz_offset[worker_id] += nnz;
      ASSERT_TRUE(nnz == static_cast<size_t>(generated_nnz))
          << "nnz is " << generated_nnz << " expected " << nnz;
    }
  }
  rmm::mr::set_current_device_resource(p_mr);
}

void data_reader_epoch_test_impl(int num_files,const int batchsize, std::vector<int> device_list, int epochs ){
  auto p_mr = rmm::mr::get_current_device_resource();
  // store the reference dataset
  std::vector<LABEL_TYPE> labels;
  std::vector<LABEL_TYPE> denses;
  std::vector<int32_t> row_offsets;
  std::vector<CAT_TYPE> sparse_values;

  is_mhot[0] = true;
  is_mhot[3] = true;
  is_mhot[5] = true;
  int sample_per_file = 2048;
  generate_parquet_input_files(num_files, sample_per_file, is_mhot, labels, denses, row_offsets,
                               sparse_values);
  int worker_num = device_list.size();
  int numprocs = 1;
  std::vector<std::vector<int>> vvgpu;
  size_t batchsize_per_gpu = batchsize / worker_num;
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }

  const auto& resource_manager = ResourceManagerExt::create(vvgpu, 0);
  const DataReaderSparseParam param = {"distributed", std::vector<int>(slot_num, max_nnz), false,
                                       slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  std::vector<long long> slot_offset(slot_size.size(), 0);
  for (unsigned int i = 1; i < slot_size.size(); i++) {
    slot_offset[i] = slot_offset[i - 1] + slot_size[i - 1];
  }
  size_t local_gpu_count = resource_manager->get_local_gpu_count();
  DataReader<T> data_reader(batchsize, label_dim, dense_dim, params, resource_manager, false,
                            device_list.size(), false);
  auto& sparse_tensorbag = data_reader.get_sparse_tensors("distributed");
  auto& label_tensorbag = data_reader.get_label_tensors();
  auto& dense_tensorbag = data_reader.get_dense_tensors();
  data_reader.create_drwg_parquet(file_list_name, slot_offset, false);

  std::vector<size_t> nnz_offset(local_gpu_count, 0);
  std::vector<size_t> sample_offset(local_gpu_count, 0);
  std::vector<size_t> total_nnz(local_gpu_count, 0);
  std::vector<size_t> total_slots(local_gpu_count, 0);
  size_t total_nnz_files = 0;
  const size_t slots_per_file = sample_per_file * slot_num;
  std::vector<std::vector<int32_t>> row_offsets_per_worker(local_gpu_count);
  std::vector<std::vector<CAT_TYPE>> sparse_value_per_worker(local_gpu_count);
  std::vector<std::vector<LABEL_TYPE>> label_per_worker(local_gpu_count);
  std::vector<std::vector<DENSE_TYPE>> dense_per_worker(local_gpu_count);
  std::vector<int> batch_per_worker;
  for(int worker_id = 0 ;worker_id < worker_num; worker_id++)
  {
    int files_cur_worker = num_files / worker_num;
    if(worker_id < num_files % worker_num)
      files_cur_worker++;
    dense_per_worker[worker_id].resize(files_cur_worker * sample_per_file * dense_dim);
    label_per_worker[worker_id].resize(files_cur_worker * sample_per_file * label_dim);
    if(files_cur_worker > 0)
      batch_per_worker.push_back((files_cur_worker * sample_per_file - 1) / batchsize + 1);
    else
      batch_per_worker.push_back(0);
  }
  // Given any iter, return worker_id that is in charge of the batch 
  std::vector<int> iter_worker_map;
  {
    int max_batch = *std::max_element(batch_per_worker.begin(),batch_per_worker.end());
    std::vector<int> batch_worker_tmp(worker_num,0);
    for(int b = 0; b < max_batch;b++){
      for(int worker = 0 ;worker < worker_num;worker++){
        if(batch_worker_tmp[worker] >= batch_per_worker[worker])
        {
          continue;
        }
        batch_worker_tmp[worker]++;
        iter_worker_map.push_back(worker);
      }
    }
  }
  // pack data for reference
  for (int i = 0; i < num_files; i++) {
    size_t cur_file_nnz = std::accumulate(row_offsets.begin() + slots_per_file * i,
                                          row_offsets.begin() + slots_per_file * (i + 1), 0);
    int worker_id = i % local_gpu_count;
    int round = i / local_gpu_count;
    total_nnz[worker_id] += cur_file_nnz;
    total_slots[worker_id] += slots_per_file;
    row_offsets_per_worker[worker_id].resize(total_slots[worker_id], 0);
    sparse_value_per_worker[worker_id].resize(total_nnz[worker_id], 0);
    std::copy(row_offsets.begin() + slots_per_file * i,
              row_offsets.begin() + slots_per_file * (i + 1),
              row_offsets_per_worker[worker_id].begin() + total_slots[worker_id] - slots_per_file);
    std::copy(sparse_values.begin() + total_nnz_files,
              sparse_values.begin() + total_nnz_files + cur_file_nnz,
              sparse_value_per_worker[worker_id].begin() + total_nnz[worker_id] - cur_file_nnz);
    std::copy(labels.begin() + label_dim * i * sample_per_file,
              labels.begin() + label_dim * (i + 1) * sample_per_file,
              label_per_worker[worker_id].begin() + round * label_dim * sample_per_file);
    std::copy(denses.begin() + dense_dim * i * sample_per_file,
              denses.begin() + dense_dim * (i + 1) * sample_per_file,
              dense_per_worker[worker_id].begin() + round * dense_dim * sample_per_file);
    total_nnz_files += cur_file_nnz;
  }
  for (int e = 0; e < epochs; e++) {
    std::cout << "epoch " << e << std::endl;
    data_reader.set_source(file_list_name);
    std::vector<int> round_per_worker(worker_num,0);
    for (int i = 0;; i++) {
      long long current_batchsize = data_reader.read_a_batch_to_device();
      if (current_batchsize == 0) break;
      int worker_id = iter_worker_map[i];
      int round = round_per_worker[worker_id];
      std::cout << "iter " << i << " batchsize "<<current_batchsize<<" from worker "<< worker_id <<" round "<<round<<std::endl;
      for (size_t gpu = 0; gpu < local_gpu_count ; ++gpu) {
        auto sparse_tensor = SparseTensor<T>::stretch_from(sparse_tensorbag[gpu]);
        auto label_tensor = Tensor2<LABEL_TYPE>::stretch_from(label_tensorbag[gpu]);
        auto dense_tensor = Tensor2<DENSE_TYPE>::stretch_from(dense_tensorbag[gpu]);
        size_t label_size = label_tensor.get_num_elements();
        size_t dense_size = dense_tensor.get_num_elements();
        // std::cout << "label size ??? " << label_size << " dense size " << dense_size <<
        // std::endl;
        ASSERT_TRUE(label_size == batchsize_per_gpu * label_dim &&
                    dense_size == batchsize_per_gpu * dense_dim);
        std::unique_ptr<LABEL_TYPE[]> label_read(new LABEL_TYPE[label_size]);
        std::unique_ptr<DENSE_TYPE[]> dense_read(new DENSE_TYPE[dense_size]);

        CK_CUDA_THROW_(cudaMemcpy(label_read.get(), label_tensor.get_ptr(),
                                  label_size * sizeof(LABEL_TYPE), cudaMemcpyDeviceToHost));
        CK_CUDA_THROW_(cudaMemcpy(dense_read.get(), dense_tensor.get_ptr(),
                                  dense_size * sizeof(DENSE_TYPE), cudaMemcpyDeviceToHost));
        size_t batch_starting = std::min(static_cast<size_t>(gpu * batchsize_per_gpu),
                                         static_cast<size_t>(current_batchsize));
        size_t batch_ending = std::min(static_cast<size_t>((gpu + 1) * batchsize_per_gpu),
                                       static_cast<size_t>(current_batchsize));
        size_t sample_current_gpu = batch_ending - batch_starting;
        for (size_t sample = 0; sample < sample_current_gpu; sample++) {
          for (int l = 0; l < label_dim; l++) {
            size_t label_idx = ((sample + batch_starting + round * batchsize) * label_dim + l) %
                               label_per_worker[worker_id].size();
            ASSERT_TRUE(label_per_worker[worker_id][label_idx] ==
                        label_read[sample * label_dim + l])<<"sample "<<sample<<" "<<(sample + batch_starting + round * batchsize);
          }

          for (int d = 0; d < dense_dim; d++) {
            size_t dense_idx = ((sample + batch_starting + round * batchsize) * dense_dim + d) %
                               dense_per_worker[worker_id].size();
            ASSERT_TRUE(dense_per_worker[worker_id][dense_idx] ==
                        dense_read[sample * dense_dim + d]);
          }
        }

        size_t nnz = sparse_tensor.nnz();
        std::unique_ptr<CAT_TYPE[]> keys(new CAT_TYPE[nnz]);
        CK_CUDA_THROW_(cudaMemcpy(keys.get(), sparse_tensor.get_value_ptr(), nnz * sizeof(CAT_TYPE),
                                  cudaMemcpyDeviceToHost));

        std::unique_ptr<T[]> rowoffsets(new T[current_batchsize * slot_num + 1]);
        CK_CUDA_THROW_(cudaMemcpy(rowoffsets.get(), sparse_tensor.get_rowoffset_ptr(),
                                  (1 + current_batchsize * slot_num) * sizeof(T), cudaMemcpyDeviceToHost));
        for (int nnz_id = 0; nnz_id < current_batchsize * slot_num; ++nnz_id) {
          T expected =
              row_offsets_per_worker[worker_id][(sample_offset[worker_id] * slot_num + nnz_id) %
                                                (total_slots[worker_id])];
          T value = rowoffsets[nnz_id + 1] - rowoffsets[nnz_id];
          ASSERT_TRUE(value == expected)
              << " iter: " << i << " idx: " << nnz_id << " worker " << worker_id << " sample_offset"
              << sample_offset[worker_id] << " value: " << value << " expected: " << expected
              << " expected idx "
              << (sample_offset[worker_id] * slot_num + nnz_id) % (total_slots[worker_id]);
          for (T start = rowoffsets[nnz_id]; start < rowoffsets[nnz_id + 1]; ++start) {
            int slot_id = nnz_id % slot_num;
            ASSERT_TRUE(sparse_value_per_worker[worker_id][(nnz_offset[worker_id] + start) %
                                                           total_nnz[worker_id]] ==
                        keys[start] - slot_offset[slot_id])
                << "idx: " << nnz_id << " "
                << sparse_value_per_worker[worker_id]
                                          [(nnz_offset[worker_id] + start) % total_nnz[worker_id]]
                << " (ref) vs " << keys[start] - slot_offset[slot_id] << " read" << std::endl;
          }
        }
        T generated_nnz = rowoffsets[current_batchsize * slot_num];
        if (gpu == local_gpu_count - 1) sample_offset[worker_id] += current_batchsize;
        if (gpu == local_gpu_count - 1) nnz_offset[worker_id] += nnz;
        ASSERT_TRUE(nnz == static_cast<size_t>(generated_nnz))
            << "nnz is " << generated_nnz << " expected " << nnz;
      }
      round_per_worker[worker_id]++;
    }
  }
  rmm::mr::set_current_device_resource(p_mr);
}

TEST(data_reader_test, data_reader_group_test_3files_1worker_epoch) {
  data_reader_epoch_test_impl(3,1026,{0},10);
}
TEST(data_reader_test, data_reader_group_test_3files_2worker_epoch) {
  data_reader_epoch_test_impl(3,1026,{0,1},10);
}
TEST(data_reader_test, data_reader_group_test_3files_4worker_epoch) {
  data_reader_epoch_test_impl(3,1028,{0,1,2,3},10);
}
TEST(data_reader_test, data_reader_group_test_1files_3worker_epoch) {
  data_reader_epoch_test_impl(1,1026,{0,1,2},10);
}

