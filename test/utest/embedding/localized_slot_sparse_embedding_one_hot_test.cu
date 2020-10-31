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

#include <cuda_profiler_api.h>
#include <sys/time.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <random>
#include "HugeCTR/include/data_generator.hpp"
#include "HugeCTR/include/data_readers/data_reader.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_one_hot.hpp"
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
const float lr = 0.01f;

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

// std::vector<size_t> slot_sizes; // null means use vocabulary_size/gpu_count/load_factor as
// max_vocabulary_size_per_gpu

// CAUSION: must match vocabulary_size
// std::vector<size_t> slot_sizes = {39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,
//   2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295,39664984,585935,12972,108,36}; //
//   for cretio dataset
std::vector<size_t> slot_sizes = {100, 100, 100, 100, 100, 100, 100, 100, 100,
                                  100, 100, 100, 100, 100, 100, 100, 100, 100,
                                  100, 100, 100, 100, 100, 100, 100, 100};  // just for verify

//-----------------------------------------------------------------------------------------

template <typename TypeEmbeddingComp>
void train_and_test(const std::vector<int> &device_list, const Optimizer_t &optimizer,
                    const Update_t &update_type) {
  OptHyperParams<TypeEmbeddingComp> hyper_params;
  hyper_params.sgd.atomic_update = true;
  const OptParams<TypeEmbeddingComp> opt_params = {optimizer, lr, hyper_params, update_type,
                                                   scaler};
  float tolerance;
  if (std::is_same<TypeEmbeddingComp, __half>::value) {
    tolerance = 5e-3f;
  } else {
    tolerance = 1e-4f;
  }

  test::mpi_init();
  int numprocs = 1;
#ifdef ENABLE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
#endif

  // if there are multi-node, we assume each node has the same gpu device_list
  std::vector<std::vector<int>> vvgpu;
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  const auto &resource_manager = ResourceManager::create(vvgpu, 0);

  if (resource_manager->get_pid() == 0) {
    std::cout << "rank " << resource_manager->get_pid() << " is generating data" << std::endl;
    {
      // re-generate the dataset files
      std::ifstream file(train_file_list_name);
      if (file.good()) {
        std::remove(train_file_list_name);
      }
    }
    {
      // re-generate the dataset files
      std::ifstream file(test_file_list_name);
      if (file.good()) {
        std::remove(test_file_list_name);
      }
    }
    // data generation: key's corresponding slot_id=(key%slot_num)
    if (slot_sizes.size() > 0) {
      HugeCTR::data_generation_for_localized_test<T, CHK>(
          train_file_list_name, prefix, num_files, train_batch_num * train_batchsize, slot_num,
          vocabulary_size, label_dim, dense_dim, max_nnz_per_slot, slot_sizes);
      HugeCTR::data_generation_for_localized_test<T, CHK>(
          test_file_list_name, prefix, num_files, test_batch_num * test_batchsize, slot_num,
          vocabulary_size, label_dim, dense_dim, max_nnz_per_slot, slot_sizes);
    } else {
      CK_THROW_(
          Error_t::WrongInput,
          "Must set slot_sizes since there is no hashtable in LocalizedSlotSpasrseEmbeddingOneHot");
    }
  }

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "This is rank: " << resource_manager->get_pid() << std::endl;
#endif

  // setup a data reader
  const DataReaderSparseParam param = {DataReaderSparse_t::Localized, max_nnz_per_slot * slot_num,
                                       max_nnz_per_slot, slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  std::unique_ptr<DataReader<T>> train_data_reader(new DataReader<T>(
      train_batchsize, label_dim, dense_dim, params, resource_manager, true, num_chunk_threads));

  train_data_reader->create_drwg_norm(train_file_list_name, CHK);

  std::unique_ptr<DataReader<T>> test_data_reader(new DataReader<T>(
      test_batchsize, label_dim, dense_dim, params, resource_manager, true, num_chunk_threads));

  test_data_reader->create_drwg_norm(test_file_list_name, CHK);

  // generate hashtable
  if (resource_manager->get_pid() == 0) {
    std::cout << "Init hash table";
    // init hash table file: <key, solt_id, value>
    std::ofstream weight_stream(hash_table_file_name);
    if (!weight_stream.is_open()) {
      ERROR_MESSAGE_("Error: file not open for writing");
    }
    // UnifiedDataSimulator<T> ldata_sim(0, slot_num-1); // for slot_id
    test::UniformDataSimulator fdata_sim;  // for value
    std::unique_ptr<float[]> buf(new float[embedding_vec_size]);
    for (long long i = 0; i < vocabulary_size; i++) {
      T key = (T)i;
      // T key = ldata_sim.get_num();
      // CAUSION: can not set random keys here, because we need to ensure that:
      // 1) we can find keys in the data file from this hash table
      // 2) there are no repeated keys
      weight_stream.write((char *)&key, sizeof(T));
      T slot_id;
      if (slot_sizes.size() == 0) {
        // slot_id = key % slot_num;  // CAUSION: need to dedicate the slot_id for each key for
        //                            // correctness verification
        CK_THROW_(Error_t::WrongInput,
                  "Must set slot_sizes since there is no hashtable in "
                  "LocalizedSlotSpasrseEmbeddingOneHot");
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
      weight_stream.write((char *)&slot_id, sizeof(T));
      // float val = (float)i;
      // float val = 0.1f;
      fdata_sim.fill(buf.get(), embedding_vec_size, -0.1f, 0.1f);
      weight_stream.write(reinterpret_cast<const char *>(buf.get()),
                          embedding_vec_size * sizeof(float));
    }
    weight_stream.close();
    std::cout << " Done" << std::endl;
  }

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  const SparseEmbeddingHashParams<TypeEmbeddingComp> embedding_params = {
      train_batchsize, test_batchsize, 0,        slot_sizes, embedding_vec_size,
      max_feature_num, slot_num,       combiner, opt_params};

  std::unique_ptr<Embedding<T, TypeEmbeddingComp>> embedding(
      new LocalizedSlotSparseEmbeddingOneHot<T, TypeEmbeddingComp>(
          train_data_reader->get_row_offsets_tensors(), train_data_reader->get_value_tensors(),
          train_data_reader->get_nnz_array(), test_data_reader->get_row_offsets_tensors(),
          test_data_reader->get_value_tensors(), test_data_reader->get_nnz_array(),
          embedding_params, plan_file, resource_manager));

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
  std::shared_ptr<GeneralBuffer2<HostAllocator>> buf = GeneralBuffer2<HostAllocator>::create();

  Tensor2<TypeEmbeddingComp> embedding_feature_from_gpu;
  buf->reserve({train_batchsize * slot_num * embedding_vec_size}, &embedding_feature_from_gpu);
  Tensor2<TypeEmbeddingComp> wgrad_from_gpu;
  buf->reserve({train_batchsize * slot_num * embedding_vec_size}, &wgrad_from_gpu);
  Tensor2<TypeEmbeddingComp> embedding_feature_from_gpu_eval;
  buf->reserve({test_batchsize * slot_num * embedding_vec_size}, &embedding_feature_from_gpu_eval);

  buf->allocate();

  typedef struct TypeHashValue_ {
    float data[embedding_vec_size];
  } TypeHashValue;

  for (int i = 0; i < train_batch_num; i++) {
    printf("Rank%d: Round %d start training:\n", resource_manager->get_pid(), i);

    // call read a batch
    printf("Rank%d: data_reader->read_a_batch_to_device()\n", resource_manager->get_pid());
    train_data_reader->read_a_batch_to_device();

    // GPU forward
    printf("Rank%d: embedding->forward()\n", resource_manager->get_pid());
    embedding->forward(true);

    // check the result of forward
    printf("Rank%d: embedding->get_forward_results()\n", resource_manager->get_pid());
    embedding->get_forward_results(true, embedding_feature_from_gpu);  // memcpy from GPU to CPU

    if (resource_manager->get_pid() == 0) {
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
    printf("Rank%d: embedding->backward()\n", resource_manager->get_pid());
    embedding->backward();

    // check the result of backward
    printf("Rank%d: embedding->get_backward_results()\n", resource_manager->get_pid());
    embedding->get_backward_results(wgrad_from_gpu, 0);

    if (resource_manager->get_pid() == 0) {
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
    printf("Rank%d: embedding->update_params()\n", resource_manager->get_pid());
    embedding->update_params();

    if (resource_manager->get_pid() == 0) {
      // CPU update_params
      printf("Rank0: embedding_cpu->update_params()\n");
      embedding_cpu->update_params();
    }

#ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    printf("Rank%d: Round %d end:\n", resource_manager->get_pid(), i);
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

  {
    /////////////////////////////////////////////////////////////////////////////////////////////
    // eval
    printf("\nRank%d: Round start eval:\n", resource_manager->get_pid());

    // call read a batch
    printf("Rank%d: data_reader_eval->read_a_batch_to_device()\n", resource_manager->get_pid());
    test_data_reader->read_a_batch_to_device();

    // GPU forward
    printf("Rank%d: embedding_eval->forward()\n", resource_manager->get_pid());
    embedding->forward(false);

    // check the result of forward
    printf("Rank%d: embedding_eval->get_forward_results()\n", resource_manager->get_pid());
    embedding->get_forward_results(false,
                                   embedding_feature_from_gpu_eval);  // memcpy from GPU to CPU

    if (resource_manager->get_pid() == 0) {
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

}  // namespace

TEST(localized_sparse_embedding_one_hot_test, fp32_sgd_1gpu) {
  train_and_test<float>({0}, Optimizer_t::SGD, Update_t::Local);
}

TEST(localized_sparse_embedding_one_hot_test, fp32_sgd_4gpu) {
  train_and_test<float>({0, 1, 2, 3}, Optimizer_t::SGD, Update_t::Local);
}

TEST(localized_sparse_embedding_one_hot_test, fp32_sgd_global_update_1gpu) {
  train_and_test<float>({0}, Optimizer_t::SGD, Update_t::Global);
}

TEST(localized_sparse_embedding_one_hot_test, fp32_sgd_global_update_4gpu) {
  train_and_test<float>({0, 1, 2, 3}, Optimizer_t::SGD, Update_t::Global);
}

TEST(localized_sparse_embedding_one_hot_test, fp16_sgd_1gpu) {
  train_and_test<__half>({0}, Optimizer_t::SGD, Update_t::Local);
}

TEST(localized_sparse_embedding_one_hot_test, fp16_sgd_4gpu) {
  train_and_test<__half>({0, 1, 2, 3}, Optimizer_t::SGD, Update_t::Local);
}

TEST(localized_sparse_embedding_one_hot_test, fp16_sgd_global_update_1gpu) {
  train_and_test<__half>({0}, Optimizer_t::SGD, Update_t::Global);
}

TEST(localized_sparse_embedding_one_hot_test, fp16_sgd_global_update_4gpu) {
  train_and_test<__half>({0, 1, 2, 3}, Optimizer_t::SGD, Update_t::Global);
}
