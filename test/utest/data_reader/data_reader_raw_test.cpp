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

#include <gtest/gtest.h>

#include <data_generator.hpp>
#include <data_readers/data_reader.hpp>
#include <fstream>
#include <thread>
#include <utest/test_utils.hpp>

using namespace HugeCTR;

const std::vector<size_t> slot_size = {
    39884406, 39043,    17289,    7420,     20263,  3,     7120, 1543, 63,
    38532951, 2953546,  403346,   10,       2208,   11938, 155,  4,    976,
    14,       39979771, 25641295, 39664984, 585935, 12972, 108,  36};

const std::vector<long long> slot_offset = {
    0,        39884406, 39923449,  39940738,  39948158,  39968421,  39968424,  39975544, 39977087,
    39977150, 78510101, 81463647,  81866993,  81867003,  81869211,  81881149,  81881304, 81881308,
    81882284, 81882298, 121862069, 147503364, 187168348, 187754283, 187767255, 187767363};

const int batchsize = 4;
const long long num_samples = batchsize * 2 + 1;
const int max_nnz = 1;
const int slot_num = 26;
const int label_dim = 1;
const int dense_dim = 13;
typedef unsigned int T;
const Check_t CHK = Check_t::None;
const std::string file_name = "./train_data.bin";

struct DensePreprocess {
  bool float_label_dense;

  float operator()(float value) { return float_label_dense ? value : log(value + 1.f); }
};

void data_reader_worker_raw_test_impl(bool float_label_dense, bool repeat) {
  std::vector<T> generated_sparse_data;
  std::vector<float> generated_dense_data;
  std::vector<float> generated_label_data;

  // data generation
  data_generation_for_raw(file_name, num_samples, label_dim, dense_dim, float_label_dense,
                          slot_size, std::vector<int>(), false, 0.0, &generated_sparse_data,
                          &generated_dense_data, &generated_label_data);
  ASSERT_TRUE(generated_sparse_data.size() == num_samples * slot_num * max_nnz);
  ASSERT_TRUE(generated_dense_data.size() == num_samples * dense_dim);
  ASSERT_TRUE(generated_label_data.size() == num_samples * label_dim);

  auto resource_manager = ResourceManagerExt::create({{0}}, 0);
  auto local_gpu = resource_manager->get_local_gpu(0);
  const DataReaderSparseParam param = {"distributed", std::vector<int>(slot_num, max_nnz), false,
                                       slot_num};

  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  std::shared_ptr<ThreadBuffer> thread_buffer = std::make_shared<ThreadBuffer>();

  CudaDeviceContext context(0);
  auto buff = GeneralBuffer2<CudaAllocator>::create();

  thread_buffer->state.store(BufferState::ReadyForWrite);
  thread_buffer->batch_size = batchsize;
  thread_buffer->param_num = params.size();
  thread_buffer->label_dim = label_dim;
  thread_buffer->dense_dim = dense_dim;
  thread_buffer->batch_size_start_idx = 0;
  thread_buffer->batch_size_end_idx = batchsize;
  for (size_t i = 0; i < params.size(); ++i) {
    auto &param = params[i];
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

  // setup a data reader
  auto file_offset_list = std::make_shared<MmapOffsetList>(
      file_name, num_samples, (label_dim + dense_dim + slot_num) * sizeof(int), batchsize, false, 1,
      repeat);

  std::shared_ptr<std::atomic<bool>> loop_flag = std::make_shared<std::atomic<bool>>(1);
  DataReaderWorkerRaw<T> data_reader(0, 1, local_gpu, loop_flag, thread_buffer, file_offset_list,
                                     repeat, params, float_label_dense);

  int round = (num_samples - 1) / batchsize + 1;

  DensePreprocess dense_preprocess{float_label_dense};
  for (int iter = 0; iter < 12; ++iter) {
    // call read a batch
    data_reader.read_a_batch();
    long long current_batch_size = thread_buffer->current_batch_size;

    if (iter % round == round - 1) {
      ASSERT_TRUE(current_batch_size == num_samples % batchsize);
    } else {
      ASSERT_TRUE(current_batch_size == batchsize);
    }

    auto sparse_tensorbag = thread_buffer->device_sparse_buffers[0];
    auto sparse_tensor = SparseTensor<T>::stretch_from(sparse_tensorbag);
    std::unique_ptr<T[]> keys(new T[current_batch_size * slot_num]);
    HCTR_LIB_THROW(cudaMemcpy(keys.get(), sparse_tensor.get_value_ptr(),
                              current_batch_size * slot_num * sizeof(T), cudaMemcpyDeviceToHost));

    for (int i = 0; i < current_batch_size * slot_num; ++i) {
      ASSERT_TRUE(generated_sparse_data[(iter % round) * batchsize * slot_num + i] == keys[i])
          << "idx:" << i;
    }

    // rowoffset always have (1 + batchsize * slot_num) number
    std::unique_ptr<T[]> rowoffsets(new T[1 + batchsize * slot_num]);
    HCTR_LIB_THROW(cudaMemcpy(rowoffsets.get(), sparse_tensor.get_rowoffset_ptr(),
                              (1 + batchsize * slot_num) * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < batchsize * slot_num + 1; ++i) {
      if (i < 1 + current_batch_size * slot_num) {
        ASSERT_TRUE(rowoffsets[i] == static_cast<T>(i)) << "idx:" << i;
      } else {
        ASSERT_TRUE(rowoffsets[i] == static_cast<T>(current_batch_size * slot_num)) << "idx:" << i;
      }
    }

    ASSERT_TRUE(sparse_tensor.nnz() == static_cast<size_t>(current_batch_size * slot_num));

    auto dense_tensorbag = thread_buffer->device_dense_buffers;
    auto label_dense_tensor = Tensor2<float>::stretch_from(dense_tensorbag);

    std::unique_ptr<float[]> label_dense_vec(
        new float[current_batch_size * (dense_dim + label_dim)]);
    HCTR_LIB_THROW(cudaMemcpy(label_dense_vec.get(), label_dense_tensor.get_ptr(),
                              current_batch_size * (dense_dim + label_dim) * sizeof(float),
                              cudaMemcpyDeviceToHost));

    for (int i = 0; i < current_batch_size; ++i) {
      int expected_label = generated_label_data[(iter % round) * batchsize + i];
      ASSERT_FLOAT_EQ(label_dense_vec[i * (label_dim + dense_dim)], expected_label);
      for (int j = 0; j < dense_dim; ++j) {
        float generated_dense = dense_preprocess(
            generated_dense_data[(iter % round) * batchsize * dense_dim + i * dense_dim + j]);
        ASSERT_FLOAT_EQ(label_dense_vec[i * (label_dim + dense_dim) + 1 + j], generated_dense);
      }
    }
    assert(thread_buffer->state.load() == BufferState::ReadyForRead);
    thread_buffer->state.store(BufferState::ReadyForWrite);
  }
}

void data_reader_raw_test_impl(const std::vector<int> &device_list, int num_threads,
                               bool float_label_dense, bool repeat, bool use_mixed_precision) {
  // data generation
  std::vector<T> generated_sparse_data;
  std::vector<float> generated_dense_data;
  std::vector<float> generated_label_data;

  data_generation_for_raw(file_name, num_samples, label_dim, dense_dim, float_label_dense,
                          slot_size, std::vector<int>(), false, 0.0, &generated_sparse_data,
                          &generated_dense_data, &generated_label_data);
  ASSERT_TRUE(generated_sparse_data.size() == num_samples * slot_num * max_nnz);
  ASSERT_TRUE(generated_dense_data.size() == num_samples * dense_dim);
  ASSERT_TRUE(generated_label_data.size() == num_samples * label_dim);

  std::vector<std::vector<int>> vvgpu{device_list};

  auto resource_manager = ResourceManagerExt::create(vvgpu, 0);
  size_t local_gpu_count = resource_manager->get_local_gpu_count();
  size_t total_gpu_count = resource_manager->get_global_gpu_count();
  int batch_size_per_gpu = batchsize / total_gpu_count;

  const DataReaderSparseParam param = {"localized", std::vector<int>(slot_num, max_nnz), true,
                                       slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  HugeCTR::DataReader<T> data_reader(batchsize, label_dim, dense_dim, params, resource_manager,
                                     repeat, num_threads, use_mixed_precision);

  auto &sparse_tensorbag = data_reader.get_sparse_tensors("localized");

  data_reader.create_drwg_raw(file_name, num_samples, float_label_dense, false, true);

  int round = (num_samples - 1) / batchsize + 1;

  size_t batch_size_start_idx = resource_manager->get_gpu_global_id_from_local_id(0);
  DensePreprocess dense_preprocess{float_label_dense};

  for (int iter = 0; iter < 12; ++iter) {
    long long current_batch_size = data_reader.read_a_batch_to_device();
    HCTR_LOG_S(DEBUG, WORLD) << "current_batch_size:" << current_batch_size << std::endl;
    if (current_batch_size == 0) return;
    if (iter % round == round - 1) {
      ASSERT_TRUE(current_batch_size == num_samples % batchsize);
    } else {
      ASSERT_TRUE(current_batch_size == batchsize);
    }

    for (size_t local_id = 0; local_id < local_gpu_count; ++local_id) {
      auto sparse_tensor = SparseTensor<T>::stretch_from(sparse_tensorbag[local_id]);

      ASSERT_TRUE(sparse_tensor.nnz() == static_cast<size_t>(current_batch_size * slot_num));

      std::unique_ptr<T[]> keys(new T[current_batch_size * slot_num]);
      HCTR_LIB_THROW(cudaMemcpy(keys.get(), sparse_tensor.get_value_ptr(),
                                current_batch_size * slot_num * sizeof(T), cudaMemcpyDeviceToHost));
      // HCTR_LOG_S(DEBUG, WORLD) << "iter:" << iter << " keys:" << keys[0] << std::endl;
      for (int i = 0; i < current_batch_size * slot_num; ++i) {
        ASSERT_TRUE(keys[i] == generated_sparse_data[batchsize * slot_num * (iter % round) + i])
            << "idx:" << i << ",a:" << keys[i]
            << ",b:" << generated_sparse_data[batchsize * slot_num * (iter % round) + i];
      }

      std::unique_ptr<T[]> rowoffsets(new T[1 + batchsize * slot_num]);
      HCTR_LIB_THROW(cudaMemcpy(rowoffsets.get(), sparse_tensor.get_rowoffset_ptr(),
                                (1 + batchsize * slot_num) * sizeof(T), cudaMemcpyDeviceToHost));
      for (int i = 0; i < batchsize * slot_num + 1; ++i) {
        if (i < 1 + current_batch_size * slot_num) {
          ASSERT_TRUE(rowoffsets[i] == static_cast<T>(i))
              << "idx:" << i << ",a:" << rowoffsets[i] << ",b:" << i;
        } else {
          ASSERT_TRUE(rowoffsets[i] == static_cast<T>(current_batch_size * slot_num))
              << "idx:" << i << ",a:" << rowoffsets[i] << ",b:" << current_batch_size * slot_num;
        }
      }

      auto dense_tensorbag = data_reader.get_dense_tensors()[local_id];
      if (use_mixed_precision) {
        auto dense_tensor = Tensor2<__half>::stretch_from(dense_tensorbag);

        std::unique_ptr<__half[]> dense(new __half[batch_size_per_gpu * dense_dim]);
        HCTR_LIB_THROW(cudaMemcpy(dense.get(), dense_tensor.get_ptr(),
                                  batch_size_per_gpu * dense_dim * sizeof(__half),
                                  cudaMemcpyDeviceToHost));

        int batch = batch_size_start_idx + local_id * batch_size_per_gpu;
        for (int i = 0; i < batch_size_per_gpu * dense_dim; ++i) {
          float val = TypeConvert<float, __half>::convert(dense[i]);
          float expected;
          if (batch + i / dense_dim < current_batch_size) {
            expected = dense_preprocess(
                generated_dense_data[((iter % round) * batchsize + batch) * dense_dim + i]);
            // without the following conversion, check will not pass
            __half expected_half = TypeConvert<__half, float>::convert(expected);
            expected = TypeConvert<float, __half>::convert(expected_half);
          } else {
            expected = 0.f;
          }
          ASSERT_FLOAT_EQ(val, expected);
        }
      } else {
        auto dense_tensor = Tensor2<float>::stretch_from(dense_tensorbag);

        std::unique_ptr<float[]> dense(new float[batch_size_per_gpu * dense_dim]);
        HCTR_LIB_THROW(cudaMemcpy(dense.get(), dense_tensor.get_ptr(),
                                  batch_size_per_gpu * dense_dim * sizeof(float),
                                  cudaMemcpyDeviceToHost));

        int batch = batch_size_start_idx + local_id * batch_size_per_gpu;
        for (int i = 0; i < batch_size_per_gpu * dense_dim; ++i) {
          float val = dense[i];
          float expected;
          if (batch + i / dense_dim < current_batch_size) {
            expected = dense_preprocess(
                generated_dense_data[((iter % round) * batchsize + batch) * dense_dim + i]);
          } else {
            expected = 0.f;
          }
          ASSERT_FLOAT_EQ(val, expected);
        }
      }

      auto label_tensorbag = data_reader.get_label_tensors()[local_id];
      {
        auto label_tensor = Tensor2<float>::stretch_from(label_tensorbag);

        std::unique_ptr<float[]> label(new float[batch_size_per_gpu * label_dim]);
        HCTR_LIB_THROW(cudaMemcpy(label.get(), label_tensor.get_ptr(),
                                  batch_size_per_gpu * label_dim * sizeof(float),
                                  cudaMemcpyDeviceToHost));

        int batch = batch_size_start_idx + local_id * batch_size_per_gpu;
        for (int i = 0; i < batch_size_per_gpu * label_dim; ++i) {
          float val = label[i];
          float expected;
          if (batch + i / label_dim < current_batch_size) {
            expected = generated_label_data[((iter % round) * batchsize + batch) * label_dim + i];
          } else {
            expected = 0.f;
          }
          ASSERT_FLOAT_EQ(val, expected);
        }
      }
    }
  }
}

TEST(data_reader_raw, data_reader_worker_raw_float_test) {
  data_reader_worker_raw_test_impl(true, true);
}
TEST(data_reader_raw, data_reader_worker_raw_int_test) {
  data_reader_worker_raw_test_impl(false, true);
}

TEST(data_reader_raw, float_test_1) { data_reader_raw_test_impl({0}, 1, true, true, false); }
TEST(data_reader_raw, float_test_2) { data_reader_raw_test_impl({0}, 2, true, true, false); }
TEST(data_reader_raw, float_test_3) { data_reader_raw_test_impl({0, 1}, 2, true, true, false); }
TEST(data_reader_raw, float_test_4) { data_reader_raw_test_impl({0, 1}, 4, true, true, false); }

TEST(data_reader_raw_epoch, float_test_1) { data_reader_raw_test_impl({0}, 1, true, false, false); }
TEST(data_reader_raw_epoch, float_test_2) { data_reader_raw_test_impl({0}, 2, true, false, false); }
TEST(data_reader_raw_epoch, float_test_3) {
  data_reader_raw_test_impl({0, 1}, 2, true, false, false);
}
TEST(data_reader_raw_epoch, float_test_4) {
  data_reader_raw_test_impl({0, 1}, 4, true, false, false);
}

TEST(data_reader_raw_half, float_test_1) { data_reader_raw_test_impl({0}, 1, true, true, true); }
TEST(data_reader_raw_half, float_test_2) { data_reader_raw_test_impl({0}, 2, true, true, true); }
TEST(data_reader_raw_half, float_test_3) { data_reader_raw_test_impl({0, 1}, 2, true, true, true); }
TEST(data_reader_raw_half, float_test_4) { data_reader_raw_test_impl({0, 1}, 4, true, true, true); }

TEST(data_reader_raw, int_test_1) { data_reader_raw_test_impl({0}, 1, false, true, false); }
TEST(data_reader_raw, int_test_2) { data_reader_raw_test_impl({0}, 2, false, true, false); }
TEST(data_reader_raw, int_test_3) { data_reader_raw_test_impl({0, 1}, 2, false, true, false); }
TEST(data_reader_raw, int_test_4) { data_reader_raw_test_impl({0, 1}, 4, false, true, false); }

TEST(data_reader_raw_half, int_test_1) { data_reader_raw_test_impl({0}, 1, false, true, true); }
TEST(data_reader_raw_half, int_test_2) { data_reader_raw_test_impl({0}, 2, false, true, true); }
TEST(data_reader_raw_half, int_test_3) { data_reader_raw_test_impl({0, 1}, 2, false, true, true); }
TEST(data_reader_raw_half, int_test_4) { data_reader_raw_test_impl({0, 1}, 4, false, true, true); }
