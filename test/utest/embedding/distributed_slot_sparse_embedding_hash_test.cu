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

#include <sys/time.h>
#include <fstream>
#include <functional>
#include <unordered_set>
#include "HugeCTR/include/data_generator.hpp"
#include "HugeCTR/include/data_readers/data_reader.hpp"
#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include "gtest/gtest.h"
#include "nvToolsExt.h"
#include "utest/embedding/embedding_test_utils.hpp"
#include "utest/embedding/sparse_embedding_hash_cpu.hpp"
#include "utest/test_utils.h"

using namespace HugeCTR;
using namespace embedding_test;

namespace {
//---------------------------------------------------------------------------------------
// global params for all testing
const int train_batch_num = 10;  // can not more than 32
const int test_batch_num = 1;
const int train_batchsize = 1024;
const int test_batchsize = 2560;
const int slot_num = 26;
const int max_nnz_per_slot = 1;
const int max_feature_num = max_nnz_per_slot * slot_num;  // max_feature_num in a sample
const long long vocabulary_size = 100000;
const int embedding_vec_size = 64;
const int combiner = 0;  // 0-sum, 1-mean
const long long label_dim = 1;
const long long dense_dim = 0;
typedef long long T;

const float scaler = 1.0f;  // used in mixed precision training

// In order to not allocate the total size of hash table on each GPU, the users need to set the
// size of max_vocabulary_size_per_gpu, which should be more than vocabulary_size/gpu_count,
// eg: 1.25x of that.

const int num_chunk_threads = 1;  // must be 1 for CPU and GPU results comparation
const int num_files = 1;
const Check_t CHK = Check_t::Sum;  // Check_t::Sum
const char *train_file_list_name = "train_file_list.txt";
const char *test_file_list_name = "test_file_list.txt";
const char *prefix = "./data_reader_test_data/temp_dataset_";
const char *hash_table_file_name = "distributed_hash_table.bin";
//-----------------------------------------------------------------------------------------

template <typename TypeEmbeddingComp>
void train_and_test(const std::vector<int> &device_list, const Optimizer_t &optimizer,
                    const Update_t &update_type) {
  OptHyperParams<TypeEmbeddingComp> hyper_params;
  hyper_params.adam.beta1 = 0.9f;
  hyper_params.adam.beta2 = 0.999f;
  float tolerance;
  if (std::is_same<TypeEmbeddingComp, __half>::value) {
    hyper_params.adam.epsilon = 1e-4f;
    tolerance = 5e-3f;
  } else {
    hyper_params.adam.epsilon = 1e-7f;
    tolerance = 1e-4f;
  }
  hyper_params.momentum.factor = 0.9f;
  hyper_params.nesterov.mu = 0.9f;

  const float lr = optimizer == Optimizer_t::Adam ? 0.001f : 0.01f;

  const OptParams<TypeEmbeddingComp> opt_params = {optimizer, lr, hyper_params, update_type,
                                                   scaler};

  int numprocs = 1, pid = 0;
  std::vector<std::vector<int>> vvgpu;
  test::mpi_init();
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  const auto &resource_manager = ResourceManager::create(vvgpu, 0);

  if (pid == 0) {
    // re-generate the dataset files
    {
      std::ifstream fs(train_file_list_name);
      if (fs.good()) {
        std::remove(train_file_list_name);
      }
    }
    {
      std::ifstream fs(test_file_list_name);
      if (fs.good()) {
        std::remove(test_file_list_name);
      }
    }
    // data generation
    HugeCTR::data_generation_for_test<T, CHK>(
        train_file_list_name, prefix, num_files, train_batch_num * train_batchsize, slot_num,
        vocabulary_size, label_dim, dense_dim, max_nnz_per_slot);
    HugeCTR::data_generation_for_test<T, CHK>(
        test_file_list_name, prefix, num_files, test_batch_num * test_batchsize, slot_num,
        vocabulary_size, label_dim, dense_dim, max_nnz_per_slot);
  }

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // setup a data reader
  const DataReaderSparseParam param = {DataReaderSparse_t::Distributed, max_nnz_per_slot * slot_num,
                                       max_nnz_per_slot, slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  std::unique_ptr<DataReader<T>> train_data_reader(new DataReader<T>(
      train_batchsize, label_dim, dense_dim, params, resource_manager, true, num_chunk_threads));

  train_data_reader->create_drwg_norm(train_file_list_name, CHK);

  std::unique_ptr<DataReader<T>> test_data_reader(new DataReader<T>(
      test_batchsize, label_dim, dense_dim, params, resource_manager, true, num_chunk_threads));

  test_data_reader->create_drwg_norm(test_file_list_name, CHK);

  // init hash table file
  if (pid == 0) {
    std::ofstream fs(hash_table_file_name);
    if (!fs.is_open()) {
      ERROR_MESSAGE_("Error: file not open for writing");
    }
    test::UniformDataSimulator fdata_sim;
    std::unique_ptr<float[]> buf(new float[embedding_vec_size]);
    for (long long i = 0; i < vocabulary_size; i++) {
      T key = (T)i;
      // T key = ldata_sim.get_num();
      // CAUSION: can not set random keys here, because we need to ensure that:
      // 1) we can find keys in the data file from this hash table
      // 2) there are no repeated keys
      fs.write((char *)&key, sizeof(T));

      fdata_sim.fill(buf.get(), embedding_vec_size, -0.1f, 0.1f);
      fs.write(reinterpret_cast<const char *>(buf.get()), embedding_vec_size * sizeof(float));
    }
    fs.close();
  }

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  const SparseEmbeddingHashParams<TypeEmbeddingComp> embedding_params = {
      train_batchsize, test_batchsize, vocabulary_size, {},        embedding_vec_size,
      max_feature_num, slot_num,       combiner,        opt_params};

  std::unique_ptr<Embedding<T, TypeEmbeddingComp>> embedding(
      new DistributedSlotSparseEmbeddingHash<T, TypeEmbeddingComp>(
          train_data_reader->get_row_offsets_tensors(), train_data_reader->get_value_tensors(),
          train_data_reader->get_nnz_array(), test_data_reader->get_row_offsets_tensors(),
          test_data_reader->get_value_tensors(), test_data_reader->get_nnz_array(),
          embedding_params, resource_manager));

  {
    // upload hash table to device
    std::ifstream fs(hash_table_file_name);
    embedding->load_parameters(fs);
    fs.close();
  }

  // for SparseEmbeddingCpu
  std::unique_ptr<SparseEmbeddingHashCpu<T, TypeEmbeddingComp>> embedding_cpu(
      new SparseEmbeddingHashCpu<T, TypeEmbeddingComp>(
          train_batchsize, max_feature_num, vocabulary_size, embedding_vec_size, slot_num,
          label_dim, dense_dim, CHK, train_batch_num * train_batchsize, combiner, opt_params,
          train_file_list_name, hash_table_file_name, SparseEmbedding_t::Distributed));

  TypeEmbeddingComp *embedding_feature_from_cpu = embedding_cpu->get_forward_results();
  TypeEmbeddingComp *wgrad_from_cpu = embedding_cpu->get_backward_results();
  T *hash_table_key_from_cpu = embedding_cpu->get_hash_table_key_ptr();
  float *hash_table_value_from_cpu = embedding_cpu->get_hash_table_value_ptr();

  // for results check
  std::shared_ptr<GeneralBuffer2<HostAllocator>> buf = GeneralBuffer2<HostAllocator>::create();

  Tensor2<TypeEmbeddingComp> embedding_feature_from_gpu;
  buf->reserve({train_batchsize * slot_num * embedding_vec_size}, &embedding_feature_from_gpu);

  Tensor2<TypeEmbeddingComp> wgrad_from_gpu;
  buf->reserve({train_batchsize * slot_num * embedding_vec_size}, &wgrad_from_gpu);

  Tensor2<T> hash_table_key_from_gpu;
  buf->reserve({vocabulary_size}, &hash_table_key_from_gpu);

  Tensor2<float> hash_table_value_from_gpu;
  buf->reserve({vocabulary_size * embedding_vec_size}, &hash_table_value_from_gpu);

  Tensor2<TypeEmbeddingComp> embedding_feature_from_gpu_eval;
  buf->reserve({test_batchsize * slot_num * embedding_vec_size}, &embedding_feature_from_gpu_eval);

  buf->allocate();

  typedef struct TypeHashValue_ { float data[embedding_vec_size]; } TypeHashValue;

  for (int i = 0; i < train_batch_num; i++) {
    printf("Rank%d: Round %d start training:\n", pid, i);

    // call read a batch
    printf("Rank%d: data_reader->read_a_batch_to_device()\n", pid);
    train_data_reader->read_a_batch_to_device();

    // GPU forward
    printf("Rank%d: embedding->forward()\n", pid);
    embedding->forward(true);

    // check the result of forward
    printf("Rank%d: embedding->get_forward_results()\n", pid);
    embedding->get_forward_results(true, embedding_feature_from_gpu);  // memcpy from GPU to CPU

    if (pid == 0) {
      // CPU forward
      printf("Rank0: embedding_cpu->forward()\n");
      embedding_cpu->forward();

      printf("Rank0: check forward results\n");
      ASSERT_TRUE(compare_embedding_feature(train_batchsize * slot_num * embedding_vec_size,
                                            embedding_feature_from_gpu.get_ptr(),
                                            embedding_feature_from_cpu, tolerance));
    }

#ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // GPU backward
    printf("Rank%d: embedding->backward()\n", pid);
    embedding->backward();

    // check the result of backward
    printf("Rank%d: embedding->get_backward_results()\n", pid);
    embedding->get_backward_results(wgrad_from_gpu, 0);

    if (pid == 0) {
      // CPU backward
      printf("Rank0: embedding_cpu->backward()\n");
      embedding_cpu->backward();

      printf("Rank0: check backward results: GPU and CPU\n");
      ASSERT_TRUE(compare_wgrad(train_batchsize * slot_num * embedding_vec_size,
                                wgrad_from_gpu.get_ptr(), wgrad_from_cpu, tolerance));
    }

#ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // GPU update_params
    printf("Rank%d: embedding->update_params()\n", pid);
    embedding->update_params();

    // check the results of update params
    printf("Rank%d: embedding->get_update_params_results()\n", pid);
    embedding->get_update_params_results(hash_table_key_from_gpu,
                                         hash_table_value_from_gpu);  // memcpy from GPU to CPU

    if (pid == 0) {
      // CPU update_params
      printf("Rank0: embedding_cpu->update_params()\n");
      embedding_cpu->update_params();

      printf("Rank0: check update_params results\n");
      ASSERT_TRUE(compare_hash_table(
          vocabulary_size, hash_table_key_from_gpu.get_ptr(),
          reinterpret_cast<TypeHashValue *>(hash_table_value_from_gpu.get_ptr()),
          hash_table_key_from_cpu, reinterpret_cast<TypeHashValue *>(hash_table_value_from_cpu),
          tolerance));
    }

#ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    printf("Rank%d: Round %d end:\n\n", pid, i);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // create new obj for eval()

  {
    std::ofstream fs(hash_table_file_name);
    embedding->dump_parameters(fs);
    fs.close();
  }

  // for SparseEmbeddingCpu eval
  std::unique_ptr<SparseEmbeddingHashCpu<T, TypeEmbeddingComp>> test_embedding_cpu(
      new SparseEmbeddingHashCpu<T, TypeEmbeddingComp>(
          test_batchsize, max_feature_num, vocabulary_size, embedding_vec_size, slot_num, label_dim,
          dense_dim, CHK, test_batch_num * test_batchsize, combiner, opt_params,
          test_file_list_name, hash_table_file_name, SparseEmbedding_t::Distributed));

  TypeEmbeddingComp *embedding_feature_from_cpu_eval = test_embedding_cpu->get_forward_results();

  {
    // eval
    printf("\nRank%d: start eval:\n", pid);

    // call read a batch
    printf("Rank%d: data_reader_eval->read_a_batch_to_device()\n", pid);
    test_data_reader->read_a_batch_to_device();

    // GPU forward
    printf("Rank%d: embedding_eval->forward()\n", pid);
    embedding->forward(false);

    // check the result of forward
    printf("Rank%d: embedding_eval->get_forward_results()\n", pid);
    embedding->get_forward_results(false,
                                   embedding_feature_from_gpu_eval);  // memcpy from GPU to CPU

    if (pid == 0) {
      // CPU forward
      printf("Rank0: embedding_cpu_eval->forward()\n");
      test_embedding_cpu->forward();

      printf("Rank0: check forward results\n");
      ASSERT_TRUE(compare_embedding_feature(test_batchsize * slot_num * embedding_vec_size,
                                            embedding_feature_from_gpu_eval.get_ptr(),
                                            embedding_feature_from_cpu_eval, tolerance));
    }
  }

  test::mpi_finalize();
}

template <typename TypeEmbeddingComp>
void load_and_dump(const std::vector<int> &device_list, const Optimizer_t &optimizer,
                   const Update_t &update_type) {
  OptHyperParams<TypeEmbeddingComp> hyper_params;
  hyper_params.adam.beta1 = 0.9f;
  hyper_params.adam.beta2 = 0.999f;
  if (std::is_same<TypeEmbeddingComp, __half>::value) {
    hyper_params.adam.epsilon = 1e-4f;
  } else {
    hyper_params.adam.epsilon = 1e-7f;
  }
  hyper_params.momentum.factor = 0.9f;
  hyper_params.nesterov.mu = 0.9f;

  const float lr = optimizer == Optimizer_t::Adam ? 0.001f : 0.01f;

  const OptParams<TypeEmbeddingComp> opt_params = {optimizer, lr, hyper_params, update_type,
                                                   scaler};

  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back(device_list);
  const auto &resource_manager = ResourceManager::create(vvgpu, 0);

  // re-generate the dataset files
  {
    std::ifstream fs(train_file_list_name);
    if (fs.good()) {
      std::remove(train_file_list_name);
    }
  }

  // data generation
  HugeCTR::data_generation_for_test<T, CHK>(
      train_file_list_name, prefix, num_files, train_batch_num * train_batchsize, slot_num,
      vocabulary_size, label_dim, dense_dim, max_nnz_per_slot);

  // setup a data reader
  const DataReaderSparseParam param = {DataReaderSparse_t::Distributed, max_nnz_per_slot * slot_num,
                                       max_nnz_per_slot, slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  std::unique_ptr<DataReader<T>> train_data_reader(new DataReader<T>(
      train_batchsize, label_dim, dense_dim, params, resource_manager, true, num_chunk_threads));

  train_data_reader->create_drwg_norm(train_file_list_name, CHK);

  // init hash table file
  std::ofstream fs(hash_table_file_name);
  if (!fs.is_open()) {
    ERROR_MESSAGE_("Error: file not open for writing");
  }
  test::UniformDataSimulator fdata_sim;
  std::unique_ptr<float[]> buf(new float[embedding_vec_size]);
  for (long long i = 0; i < vocabulary_size; i++) {
    T key = (T)i;
    // T key = ldata_sim.get_num();
    // CAUSION: can not set random keys here, because we need to ensure that:
    // 1) we can find keys in the data file from this hash table
    // 2) there are no repeated keys
    fs.write((char *)&key, sizeof(T));

    fdata_sim.fill(buf.get(), embedding_vec_size, -0.1f, 0.1f);
    fs.write(reinterpret_cast<const char *>(buf.get()), embedding_vec_size * sizeof(float));
  }
  fs.close();

  const SparseEmbeddingHashParams<TypeEmbeddingComp> embedding_params = {
      train_batchsize, test_batchsize, vocabulary_size, {},        embedding_vec_size,
      max_feature_num, slot_num,       combiner,        opt_params};

  std::unique_ptr<Embedding<T, TypeEmbeddingComp>> embedding(
      new DistributedSlotSparseEmbeddingHash<T, TypeEmbeddingComp>(
          train_data_reader->get_row_offsets_tensors(), train_data_reader->get_value_tensors(),
          train_data_reader->get_nnz_array(), train_data_reader->get_row_offsets_tensors(),
          train_data_reader->get_value_tensors(), train_data_reader->get_nnz_array(),
          embedding_params, resource_manager));

  {
    // upload hash table to device
    std::ifstream fs(hash_table_file_name);
    embedding->load_parameters(fs);
    fs.close();
  }

  printf("max_vocabulary_size=%zu, vocabulary_size=%zu\n", embedding->get_max_vocabulary_size(),
         embedding->get_vocabulary_size());

  std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> blobs_buff =
      GeneralBuffer2<CudaHostAllocator>::create();

  Tensor2<T> keys;
  blobs_buff->reserve({embedding->get_max_vocabulary_size()}, &keys);

  Tensor2<float> embeddings;
  blobs_buff->reserve({embedding->get_max_vocabulary_size(), embedding_vec_size}, &embeddings);

  blobs_buff->allocate();

  BufferBag buf_bag;
  buf_bag.keys = keys.shrink();
  buf_bag.embedding = embeddings;

  size_t dump_size;
  embedding->dump_parameters(buf_bag, &dump_size);

  printf("dump_size=%zu, max_vocabulary_size=%zu, vocabulary_size=%zu\n", dump_size,
         embedding->get_max_vocabulary_size(), embedding->get_vocabulary_size());

  embedding->dump_parameters(buf_bag, &dump_size);

  printf("dump_size=%zu, max_vocabulary_size=%zu, vocabulary_size=%zu\n", dump_size,
         embedding->get_max_vocabulary_size(), embedding->get_vocabulary_size());

  embedding->reset();

  printf("max_vocabulary_size=%zu, vocabulary_size=%zu\n", embedding->get_max_vocabulary_size(),
         embedding->get_vocabulary_size());

  embedding->load_parameters(buf_bag, dump_size);

  printf("max_vocabulary_size=%zu, vocabulary_size=%zu\n", embedding->get_max_vocabulary_size(),
         embedding->get_vocabulary_size());

  embedding->dump_parameters(buf_bag, &dump_size);

  printf("dump_size=%zu, max_vocabulary_size=%zu, vocabulary_size=%zu\n", dump_size,
         embedding->get_max_vocabulary_size(), embedding->get_vocabulary_size());
}

}  // namespace

TEST(distributed_sparse_embedding_hash_test, fp32_sgd_1gpu) {
  train_and_test<float>({0}, Optimizer_t::SGD, Update_t::Local);
}

TEST(distributed_sparse_embedding_hash_test, fp32_sgd_8gpu) {
  train_and_test<float>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::SGD, Update_t::Local);
}

TEST(distributed_sparse_embedding_hash_test, fp32_sgd_global_update_1gpu) {
  train_and_test<float>({0}, Optimizer_t::SGD, Update_t::Global);
}

TEST(distributed_sparse_embedding_hash_test, fp32_sgd_global_update_8gpu) {
  train_and_test<float>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::SGD, Update_t::Global);
}

TEST(distributed_sparse_embedding_hash_test, fp16_sgd_1gpu) {
  train_and_test<__half>({0}, Optimizer_t::SGD, Update_t::Local);
}

TEST(distributed_sparse_embedding_hash_test, fp16_sgd_8gpu) {
  train_and_test<__half>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::SGD, Update_t::Local);
}

TEST(distributed_sparse_embedding_hash_test, fp16_sgd_global_update_1gpu) {
  train_and_test<__half>({0}, Optimizer_t::SGD, Update_t::Global);
}

TEST(distributed_sparse_embedding_hash_test, fp16_sgd_global_update_8gpu) {
  train_and_test<__half>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::SGD, Update_t::Global);
}

TEST(distributed_sparse_embedding_hash_test, fp32_adam_1gpu) {
  train_and_test<float>({0}, Optimizer_t::Adam, Update_t::Local);
}

TEST(distributed_sparse_embedding_hash_test, fp32_adam_8gpu) {
  train_and_test<float>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::Adam, Update_t::Local);
}

TEST(distributed_sparse_embedding_hash_test, fp32_adam_global_update_1gpu) {
  train_and_test<float>({0}, Optimizer_t::Adam, Update_t::Global);
}

TEST(distributed_sparse_embedding_hash_test, fp32_adam_global_update_8gpu) {
  train_and_test<float>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::Adam, Update_t::Global);
}

TEST(distributed_sparse_embedding_hash_test, fp32_adam_lazyglobal_update_1gpu) {
  train_and_test<float>({0}, Optimizer_t::Adam, Update_t::LazyGlobal);
}

TEST(distributed_sparse_embedding_hash_test, fp32_adam_lazyglobal_update_8gpu) {
  train_and_test<float>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::Adam, Update_t::LazyGlobal);
}

TEST(distributed_sparse_embedding_hash_test, fp16_adam_1gpu) {
  train_and_test<__half>({0}, Optimizer_t::Adam, Update_t::Local);
}

TEST(distributed_sparse_embedding_hash_test, fp16_adam_8gpu) {
  train_and_test<__half>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::Adam, Update_t::Local);
}

TEST(distributed_sparse_embedding_hash_test, fp16_adam_global_update_1gpu) {
  train_and_test<__half>({0}, Optimizer_t::Adam, Update_t::Global);
}

TEST(distributed_sparse_embedding_hash_test, fp16_adam_global_update_8gpu) {
  train_and_test<__half>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::Adam, Update_t::Global);
}

TEST(distributed_sparse_embedding_hash_test, fp16_adam_lazyglobal_update_1gpu) {
  train_and_test<__half>({0}, Optimizer_t::Adam, Update_t::LazyGlobal);
}

TEST(distributed_sparse_embedding_hash_test, fp16_adam_lazyglobal_update_8gpu) {
  train_and_test<__half>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::Adam, Update_t::LazyGlobal);
}

TEST(distributed_sparse_embedding_hash_test, load_and_dump) {
  load_and_dump<float>({0}, Optimizer_t::SGD, Update_t::Global);
}
