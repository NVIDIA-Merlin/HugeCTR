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
#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/data_reader.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp"
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
const long long vocabulary_size = slot_num * 100;
const int embedding_vec_size = 128;
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

#ifndef NCCl_A2A
const std::string plan_file(PROJECT_HOME_ + "utest/all2all_plan_dgx_{0,1,2,3,4,5,6,7}.json");
#else
const std::string plan_file = "";
#endif

const char *hash_table_file_name = "localized_hash_table.bin";

std::vector<size_t> slot_sizes;  // null means use vocabulary_size/gpu_count/load_factor as
                                 // max_vocabulary_size_per_gpu

// CAUSION: must match vocabulary_size
// std::vector<size_t> slot_sizes = {39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,
//   2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295,39664984,585935,12972,108,36}; //
//   for cretio dataset
// std::vector<size_t> slot_sizes =
// {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100};
// // just for verify

//-----------------------------------------------------------------------------------------

template <typename TypeEmbeddingComp>
void train_and_test(const std::vector<int> &device_list, const Optimizer_t &optimizer,
                    bool global_update) {
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

  const OptParams<TypeEmbeddingComp> opt_params = {optimizer, lr, hyper_params, global_update,
                                                   scaler};

  int numprocs = 1, pid = 0;
  std::vector<std::vector<int>> vvgpu;
  test::mpi_init();
#ifdef ENABLE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
#endif

  // if there are multi-node, we assume each node has the same gpu device_list
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  std::shared_ptr<DeviceMap> device_map(new DeviceMap(vvgpu, pid));
  std::shared_ptr<GPUResourceGroup> gpu_resource_group(new GPUResourceGroup(device_map));
  std::shared_ptr<rmm::mr::device_memory_resource> memory_resource;
  if (pid == 0) {
    std::cout << "rank " << pid << " is generating data" << std::endl;
    // re-generate the dataset files
    {
      std::ifstream file(train_file_list_name);
      if (file.good()) {
        std::remove(train_file_list_name);
      }
    }
    {
      std::ifstream file(test_file_list_name);
      if (file.good()) {
        std::remove(test_file_list_name);
      }
    }
    // data generation: key's corresponding slot_id=(key%slot_num)
    if (slot_sizes.size() > 0) {
      HugeCTR::data_generation_for_localized_test<T, CHK>(
          train_file_list_name, prefix, num_files, train_batchsize * train_batch_num, slot_num,
          vocabulary_size, label_dim, dense_dim, max_nnz_per_slot, slot_sizes);
      HugeCTR::data_generation_for_localized_test<T, CHK>(
          test_file_list_name, prefix, num_files, test_batchsize * test_batch_num, slot_num,
          vocabulary_size, label_dim, dense_dim, max_nnz_per_slot, slot_sizes);
    } else {
      HugeCTR::data_generation_for_localized_test<T, CHK>(
          train_file_list_name, prefix, num_files, train_batchsize * train_batch_num, slot_num,
          vocabulary_size, label_dim, dense_dim, max_nnz_per_slot);
      HugeCTR::data_generation_for_localized_test<T, CHK>(
          test_file_list_name, prefix, num_files, test_batchsize * test_batch_num, slot_num,
          vocabulary_size, label_dim, dense_dim, max_nnz_per_slot);
    }
  }

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "This is rank: " << pid << std::endl;
#endif

  // setup a data reader
  const DataReaderSparseParam param = {DataReaderSparse_t::Localized, max_nnz_per_slot * slot_num,
                                       max_nnz_per_slot, slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  std::unique_ptr<DataReader<T>> train_data_reader(
      new DataReader<T>(train_file_list_name, train_batchsize, label_dim, dense_dim, CHK, params,
                        gpu_resource_group, memory_resource, num_chunk_threads));

  std::unique_ptr<DataReader<T>> test_data_reader(
      new DataReader<T>(test_file_list_name, test_batchsize, label_dim, dense_dim, CHK, params,
                        gpu_resource_group, memory_resource, num_chunk_threads));

  slot_sizes.clear();  // don't init hashtable when doing training correctness checking.
                       // Because we will upload hashtable to GPUs.

  // generate hashtable
  if (pid == 0) {
    std::cout << "Init hash table";
    // init hash table file: <key, solt_id, value>
    std::ofstream fs(hash_table_file_name);
    if (!fs.is_open()) {
      ERROR_MESSAGE_("Error: file not open for writing");
    }
    // UnifiedDataSimulator<T> ldata_sim(0, slot_num-1); // for slot_id
    UnifiedDataSimulator<float> fdata_sim(-0.1f, 0.1f);  // for value
    for (long long i = 0; i < vocabulary_size; i++) {
      T key = (T)i;
      // T key = ldata_sim.get_num();
      // CAUSION: can not set random keys here, because we need to ensure that:
      // 1) we can find keys in the data file from this hash table
      // 2) there are no repeated keys
      fs.write((char *)&key, sizeof(T));
      T slot_id;
      if (slot_sizes.size() == 0) {
        slot_id = key % slot_num;  // CAUSION: need to dedicate the slot_id for each key for
                                   // correctness verification
      } else {
        size_t offset = 0;
        for (size_t j = 0; j < slot_sizes.size(); j++) {
          if ((key >= static_cast<T>(offset)) && (key < static_cast<T>(offset + slot_sizes[j]))) {
            slot_id = (T)j;
            break;
          }
          offset += slot_sizes[j];
        }
      }
      fs.write((char *)&slot_id, sizeof(T));
      float val = fdata_sim.get_num();
      for (int j = 0; j < embedding_vec_size; j++) {
        fs.write((char *)&val, sizeof(float));
      }
    }
    fs.close();
    std::cout << " Done" << std::endl;
  }

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  const SparseEmbeddingHashParams<TypeEmbeddingComp> embedding_params = {
      train_batchsize, test_batchsize, vocabulary_size, {},        embedding_vec_size,
      max_feature_num, slot_num,       combiner,        opt_params};

  std::unique_ptr<Embedding<T, TypeEmbeddingComp>> embedding(
      new LocalizedSlotSparseEmbeddingHash<T, TypeEmbeddingComp>(
          train_data_reader->get_row_offsets_tensors(), train_data_reader->get_value_tensors(),
          train_data_reader->get_nnz_array(), test_data_reader->get_row_offsets_tensors(),
          test_data_reader->get_value_tensors(), test_data_reader->get_nnz_array(),
          embedding_params, plan_file, gpu_resource_group));

  {
    // upload hash table to device
    std::ifstream fs(hash_table_file_name);
    embedding->upload_params_to_device(fs);
    fs.close();
  }

  // for SparseEmbeddingCpu
  std::unique_ptr<SparseEmbeddingHashCpu<T, TypeEmbeddingComp>> embedding_cpu(
      new SparseEmbeddingHashCpu<T, TypeEmbeddingComp>(
          train_batchsize, max_feature_num, vocabulary_size, embedding_vec_size, slot_num,
          label_dim, dense_dim, CHK, train_batch_num * train_batchsize, combiner, opt_params,
          train_file_list_name, hash_table_file_name, SparseEmbedding_t::Localized));

  TypeEmbeddingComp *embedding_feature_from_cpu = embedding_cpu->get_forward_results();
  TypeEmbeddingComp *wgrad_from_cpu = embedding_cpu->get_backward_results();
  T *hash_table_key_from_cpu = embedding_cpu->get_hash_table_key_ptr();
  float *hash_table_value_from_cpu = embedding_cpu->get_hash_table_value_ptr();

  // for results check
  std::unique_ptr<TypeEmbeddingComp[]> embedding_feature_from_gpu(
      new TypeEmbeddingComp[train_batchsize * slot_num * embedding_vec_size]);
  std::unique_ptr<TypeEmbeddingComp[]> wgrad_from_gpu(
      new TypeEmbeddingComp[train_batchsize * slot_num * embedding_vec_size]);
  std::unique_ptr<T[]> hash_table_key_from_gpu(new T[vocabulary_size]);
  std::unique_ptr<float[]> hash_table_value_from_gpu(
      new float[vocabulary_size * embedding_vec_size]);

  typedef struct TypeHashValue_ {
    float data[embedding_vec_size];
  } TypeHashValue;

  embedding->train();
  for (int i = 0; i < train_batch_num; i++) {
    printf("Rank%d: Round %d start training:\n", pid, i);

    // call read a batch
    printf("Rank%d: data_reader->read_a_batch_to_device()\n", pid);
    train_data_reader->read_a_batch_to_device();

    // GPU forward
    printf("Rank%d: embedding->forward()\n", pid);
    embedding->forward();

    // check the result of forward
    printf("Rank%d: embedding->get_forward_results()\n", pid);
    embedding->get_forward_results(embedding_feature_from_gpu.get());  // memcpy from GPU to CPU

    if (pid == 0) {
      // CPU forward
      printf("Rank0: embedding_cpu->forward()\n");
      embedding_cpu->forward();

      printf("Rank0: check forward results\n");
      ASSERT_EQ(true, compare_embedding_feature(train_batchsize * slot_num * embedding_vec_size,
                                                embedding_feature_from_gpu.get(),
                                                embedding_feature_from_cpu));
    }

#ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // GPU backward
    printf("Rank%d: embedding->backward()\n", pid);
    embedding->backward();

    // check the result of backward
    printf("Rank%d: embedding->get_backward_results()\n", pid);
    embedding->get_backward_results(wgrad_from_gpu.get(), 0);

    if (pid == 0) {
      // CPU backward
      printf("Rank0: embedding_cpu->backward()\n");
      embedding_cpu->backward();

      printf("Rank0: check backward results: GPU and CPU\n");
      ASSERT_EQ(true, compare_wgrad(train_batchsize * slot_num * embedding_vec_size,
                                    wgrad_from_gpu.get(), wgrad_from_cpu));
    }

#ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // GPU update_params
    printf("Rank%d: embedding->update_params()\n", pid);
    embedding->update_params();

    // check the results of update params
    printf("Rank%d: embedding->get_update_params_results()\n", pid);
    embedding->get_update_params_results(
        hash_table_key_from_gpu.get(),
        hash_table_value_from_gpu.get());  // memcpy from GPU to CPU

    if (pid == 0) {
      // CPU update_params
      printf("Rank0: embedding_cpu->update_params()\n");
      embedding_cpu->update_params();

      printf("Rank0: check update_params results\n");
      bool rtn = compare_hash_table<T, TypeHashValue>(
          vocabulary_size, hash_table_key_from_gpu.get(),
          reinterpret_cast<TypeHashValue *>(hash_table_value_from_gpu.get()),
          hash_table_key_from_cpu, reinterpret_cast<TypeHashValue *>(hash_table_value_from_cpu));
      ASSERT_EQ(true, rtn);
    }

#ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // create new obj for eval()
  {
    std::ofstream fs(hash_table_file_name);
    embedding->download_params_to_host(fs);
    fs.close();
  }

  // for SparseEmbeddingCpu eval
  std::unique_ptr<SparseEmbeddingHashCpu<T, TypeEmbeddingComp>> test_embedding_cpu(
      new SparseEmbeddingHashCpu<T, TypeEmbeddingComp>(
          test_batchsize, max_feature_num, vocabulary_size, embedding_vec_size, slot_num, label_dim,
          dense_dim, CHK, test_batch_num * test_batchsize, combiner, opt_params,
          test_file_list_name, hash_table_file_name, SparseEmbedding_t::Localized));

  TypeEmbeddingComp *embedding_feature_from_cpu_eval = test_embedding_cpu->get_forward_results();

  // for results check
  std::unique_ptr<TypeEmbeddingComp[]> embedding_feature_from_gpu_eval(
      new TypeEmbeddingComp[test_batchsize * slot_num * embedding_vec_size]);

  /////////////////////////////////////////////////////////////////////////////////////////////
  // eval
  embedding->evaluate();
  {
    printf("\nRank%d: Round start eval:\n", pid);

    // call read a batch
    printf("Rank%d: data_reader_eval->read_a_batch_to_device()\n", pid);
    test_data_reader->read_a_batch_to_device();

    // GPU forward
    printf("Rank%d: embedding_eval->forward()\n", pid);
    embedding->forward();

    // check the result of forward
    printf("Rank%d: embedding_eval->get_forward_results()\n", pid);
    embedding->get_forward_results(
        embedding_feature_from_gpu_eval.get());  // memcpy from GPU to CPU

    if (pid == 0) {
      // CPU forward
      printf("Rank0: embedding_cpu_eval->forward()\n");
      test_embedding_cpu->forward();

      printf("Rank0: check forward results\n");
      ASSERT_EQ(true, compare_embedding_feature(test_batchsize * slot_num * embedding_vec_size,
                                                embedding_feature_from_gpu_eval.get(),
                                                embedding_feature_from_cpu_eval));
    }

#ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    printf("Rank%d: Round end:\n", pid);
  }

  test::mpi_finialize();
}

}  // namespace

TEST(localized_sparse_embedding_hash_test, fp32_sgd_1gpu) {
  train_and_test<float>({0}, Optimizer_t::SGD, false);
}

TEST(localized_sparse_embedding_hash_test, fp32_sgd_8gpu) {
  train_and_test<float>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::SGD, false);
}

TEST(localized_sparse_embedding_hash_test, fp32_sgd_global_update_1gpu) {
  train_and_test<float>({0}, Optimizer_t::SGD, true);
}

TEST(localized_sparse_embedding_hash_test, fp32_sgd_global_update_8gpu) {
  train_and_test<float>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::SGD, true);
}

TEST(localized_sparse_embedding_hash_test, fp16_sgd_1gpu) {
  train_and_test<__half>({0}, Optimizer_t::SGD, false);
}

TEST(localized_sparse_embedding_hash_test, fp16_sgd_8gpu) {
  train_and_test<__half>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::SGD, false);
}

TEST(localized_sparse_embedding_hash_test, fp16_sgd_global_update_1gpu) {
  train_and_test<__half>({0}, Optimizer_t::SGD, true);
}

TEST(localized_sparse_embedding_hash_test, fp16_sgd_global_update_8gpu) {
  train_and_test<__half>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::SGD, true);
}

TEST(localized_sparse_embedding_hash_test, fp32_adam_1gpu) {
  train_and_test<float>({0}, Optimizer_t::Adam, false);
}

TEST(localized_sparse_embedding_hash_test, fp32_adam_8gpu) {
  train_and_test<float>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::Adam, false);
}

TEST(localized_sparse_embedding_hash_test, fp32_adam_global_update_1gpu) {
  train_and_test<float>({0}, Optimizer_t::Adam, true);
}

TEST(localized_sparse_embedding_hash_test, fp32_adam_global_update_8gpu) {
  train_and_test<float>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::Adam, true);
}

TEST(localized_sparse_embedding_hash_test, fp16_adam_1gpu) {
  train_and_test<__half>({0}, Optimizer_t::Adam, false);
}

TEST(localized_sparse_embedding_hash_test, fp16_adam_8gpu) {
  train_and_test<__half>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::Adam, false);
}

TEST(localized_sparse_embedding_hash_test, fp16_adam_global_update_1gpu) {
  train_and_test<__half>({0}, Optimizer_t::Adam, true);
}

TEST(localized_sparse_embedding_hash_test, fp16_adam_global_update_8gpu) {
  train_and_test<__half>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::Adam, true);
}