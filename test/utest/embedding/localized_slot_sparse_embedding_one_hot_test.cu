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

#include <experimental/filesystem>

using namespace HugeCTR;
using namespace embedding_test;
namespace fs = std::experimental::filesystem;

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

const char *sparse_model_file = "localized_hash_table";

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
void init_sparse_model(const char *sparse_model) {
  std::cout << "Init hash table";
  // init hash table file: <key, solt_id, value>
  if (!fs::exists(sparse_model)) {
    fs::create_directory(sparse_model);
  }
  const std::string key_file = std::string(sparse_model) + "/" + sparse_model + ".key";
  const std::string slot_file = std::string(sparse_model) + "/" + sparse_model + ".slot";
  const std::string vec_file = std::string(sparse_model) + "/" + sparse_model + ".vec";
  std::ofstream fs_key(key_file);
  std::ofstream fs_slot(slot_file);
  std::ofstream fs_vec(vec_file);
  if (!fs_key.is_open() || !fs_slot.is_open() || !fs_vec.is_open()) {
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
    fs_key.write((char *)&key, sizeof(T));
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
    fs_slot.write((char *)&slot_id, sizeof(T));
    // float val = (float)i;
    // float val = 0.1f;
    fdata_sim.fill(buf.get(), embedding_vec_size, -0.1f, 0.1f);
    fs_vec.write(reinterpret_cast<const char *>(buf.get()),
                        embedding_vec_size * sizeof(float));
  }
  std::cout << " Done" << std::endl;
}

template <typename TypeEmbeddingComp>
void train_and_test(const std::vector<int> &device_list, const Optimizer_t &optimizer,
                    const Update_t &update_type,
                    const DeviceMap::Layout layout = DeviceMap::LOCAL_FIRST) {
  OptHyperParams hyper_params;
  hyper_params.sgd.atomic_update = true;
  const OptParams opt_params = {optimizer, lr, hyper_params, update_type,
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
  const auto &resource_manager = ResourceManager::create(vvgpu, 0, layout);

  if (resource_manager->is_master_process()) {
    std::cout << "rank " << resource_manager->get_process_id() << " is generating data"
              << std::endl;
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
  std::cout << "This is rank: " << resource_manager->get_process_id() << std::endl;
#endif

  // setup a data reader
  const DataReaderSparseParam param = {DataReaderSparse_t::Localized, max_nnz_per_slot * slot_num,
                                       max_nnz_per_slot, slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  std::unique_ptr<DataReader<T>> train_data_reader(new DataReader<T>(
      train_batchsize, label_dim, dense_dim, params, resource_manager, true, num_chunk_threads, false, 0));

  train_data_reader->create_drwg_norm(train_file_list_name, CHK);

  std::unique_ptr<DataReader<T>> test_data_reader(new DataReader<T>(
      test_batchsize, label_dim, dense_dim, params, resource_manager, true, num_chunk_threads, false, 0));

  test_data_reader->create_drwg_norm(test_file_list_name, CHK);

  // generate hashtable
  if (resource_manager->is_master_process()) {
    init_sparse_model(sparse_model_file);
  }

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  const SparseEmbeddingHashParams embedding_params = {
      train_batchsize, test_batchsize, 0,        slot_sizes, embedding_vec_size,
      max_feature_num, slot_num,       combiner, opt_params};

  std::unique_ptr<Embedding<T, TypeEmbeddingComp>> embedding(
      new LocalizedSlotSparseEmbeddingOneHot<T, TypeEmbeddingComp>(
          bags_to_tensors<T>(train_data_reader->get_row_offsets_tensors()),
          bags_to_tensors<T>(train_data_reader->get_value_tensors()),
          train_data_reader->get_nnz_array(),
          bags_to_tensors<T>(test_data_reader->get_row_offsets_tensors()),
          bags_to_tensors<T>(test_data_reader->get_value_tensors()),
          test_data_reader->get_nnz_array(),
          embedding_params,
          resource_manager));

  // upload hash table to device
  embedding->load_parameters(sparse_model_file);

  // for SparseEmbeddingCpu
  std::unique_ptr<SparseEmbeddingHashCpu<T, TypeEmbeddingComp>> embedding_cpu(
      new SparseEmbeddingHashCpu<T, TypeEmbeddingComp>(
          train_batchsize, max_feature_num, vocabulary_size, embedding_vec_size, slot_num,
          label_dim, dense_dim, CHK, train_batch_num * train_batchsize, combiner, opt_params,
          train_file_list_name, sparse_model_file, SparseEmbedding_t::Localized));

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
    printf("Rank%d: Round %d start training:\n", resource_manager->get_process_id(), i);

    // call read a batch
    printf("Rank%d: data_reader->read_a_batch_to_device()\n", resource_manager->get_process_id());
    train_data_reader->read_a_batch_to_device();

    // GPU forward
    printf("Rank%d: embedding->forward()\n", resource_manager->get_process_id());
    embedding->forward(true);

    // check the result of forward
    printf("Rank%d: embedding->get_forward_results()\n", resource_manager->get_process_id());
    embedding->get_forward_results(true, embedding_feature_from_gpu);  // memcpy from GPU to CPU

    if (resource_manager->is_master_process()) {
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
    printf("Rank%d: embedding->backward()\n", resource_manager->get_process_id());
    for (size_t g = 0; g < resource_manager->get_local_gpu_count(); g++) {
      const auto &local_gpu = resource_manager->get_local_gpu(g);
      local_gpu->set_compute_event_sync(local_gpu->get_stream());
      local_gpu->wait_on_compute_event(local_gpu->get_comp_overlap_stream());
    }
    embedding->backward();
    for (size_t g = 0; g < resource_manager->get_local_gpu_count(); g++) {
      const auto &local_gpu = resource_manager->get_local_gpu(g);
      local_gpu->set_compute2_event_sync(local_gpu->get_comp_overlap_stream());
      local_gpu->wait_on_compute2_event(local_gpu->get_stream());
    }

    // check the result of backward
    printf("Rank%d: embedding->get_backward_results()\n", resource_manager->get_process_id());
    embedding->get_backward_results(wgrad_from_gpu, 0);

    if (resource_manager->is_master_process()) {
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
    printf("Rank%d: embedding->update_params()\n", resource_manager->get_process_id());
    for (size_t g = 0; g < resource_manager->get_local_gpu_count(); g++) {
      const auto &local_gpu = resource_manager->get_local_gpu(g);
      local_gpu->set_compute_event_sync(local_gpu->get_stream());
      local_gpu->wait_on_compute_event(local_gpu->get_comp_overlap_stream());
    }
    embedding->update_params();
    for (size_t g = 0; g < resource_manager->get_local_gpu_count(); g++) {
      const auto &local_gpu = resource_manager->get_local_gpu(g);
      local_gpu->set_compute2_event_sync(local_gpu->get_comp_overlap_stream());
      local_gpu->wait_on_compute2_event(local_gpu->get_stream());
    }

    if (resource_manager->is_master_process()) {
      // CPU update_params
      printf("Rank0: embedding_cpu->update_params()\n");
      embedding_cpu->update_params();
    }

#ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    printf("Rank%d: Round %d end:\n", resource_manager->get_process_id(), i);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // create new obj for eval()
  embedding->dump_parameters(sparse_model_file);

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // for SparseEmbeddingCpu eval
  std::unique_ptr<SparseEmbeddingHashCpu<T, TypeEmbeddingComp>> test_embedding_cpu(
      new SparseEmbeddingHashCpu<T, TypeEmbeddingComp>(
          test_batchsize, max_feature_num, vocabulary_size, embedding_vec_size, slot_num, label_dim,
          dense_dim, CHK, test_batch_num * test_batchsize, combiner, opt_params,
          test_file_list_name, sparse_model_file, SparseEmbedding_t::Localized));

  TypeEmbeddingComp *embedding_feature_from_cpu_eval = test_embedding_cpu->get_forward_results();

  {
    /////////////////////////////////////////////////////////////////////////////////////////////
    // eval
    printf("\nRank%d: Round start eval:\n", resource_manager->get_process_id());

    // call read a batch
    printf("Rank%d: data_reader_eval->read_a_batch_to_device()\n",
           resource_manager->get_process_id());
    test_data_reader->read_a_batch_to_device();

    // GPU forward
    printf("Rank%d: embedding_eval->forward()\n", resource_manager->get_process_id());
    embedding->forward(false);

    // check the result of forward
    printf("Rank%d: embedding_eval->get_forward_results()\n", resource_manager->get_process_id());
    embedding->get_forward_results(false,
                                   embedding_feature_from_gpu_eval);  // memcpy from GPU to CPU

    if (resource_manager->is_master_process()) {
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
  OptHyperParams hyper_params;
  hyper_params.sgd.atomic_update = true;
  const OptParams opt_params = {optimizer, lr, hyper_params, update_type,
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

  // setup a data reader
  const DataReaderSparseParam param = {DataReaderSparse_t::Localized, max_nnz_per_slot * slot_num,
                                       max_nnz_per_slot, slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  std::unique_ptr<DataReader<T>> train_data_reader(new DataReader<T>(
      train_batchsize, label_dim, dense_dim, params, resource_manager, true, num_chunk_threads, false, 0));

  train_data_reader->create_drwg_norm(train_file_list_name, CHK);

  // generate hashtable
  init_sparse_model(sparse_model_file);

  const SparseEmbeddingHashParams embedding_params = {
      train_batchsize, test_batchsize, 0,        slot_sizes, embedding_vec_size,
      max_feature_num, slot_num,       combiner, opt_params};

  std::unique_ptr<Embedding<T, TypeEmbeddingComp>> embedding(
      new LocalizedSlotSparseEmbeddingOneHot<T, TypeEmbeddingComp>(
          bags_to_tensors<T>(train_data_reader->get_row_offsets_tensors()),
          bags_to_tensors<T>(train_data_reader->get_value_tensors()),
          train_data_reader->get_nnz_array(),
          bags_to_tensors<T>(train_data_reader->get_row_offsets_tensors()),
          bags_to_tensors<T>(train_data_reader->get_value_tensors()),
          train_data_reader->get_nnz_array(),
          embedding_params,
          resource_manager));

  // upload hash table to device
  embedding->load_parameters(sparse_model_file);

  printf("max_vocabulary_size=%zu, vocabulary_size=%zu\n", embedding->get_max_vocabulary_size(),
         embedding->get_vocabulary_size());

  std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> blobs_buff =
      GeneralBuffer2<CudaHostAllocator>::create();

  Tensor2<T> keys;
  blobs_buff->reserve({embedding->get_max_vocabulary_size()}, &keys);

  Tensor2<size_t> slot_id;
  blobs_buff->reserve({embedding->get_max_vocabulary_size()}, &slot_id);

  Tensor2<float> embeddings;
  blobs_buff->reserve({embedding->get_max_vocabulary_size(), embedding_vec_size}, &embeddings);

  blobs_buff->allocate();

  BufferBag buf_bag;
  buf_bag.keys = keys.shrink();
  buf_bag.slot_id = slot_id.shrink();
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

template <typename TypeEmbeddingComp>
void load_and_dump_file(const std::vector<int> &device_list, const Optimizer_t &optimizer,
                   const Update_t &update_type) {
  std::string sparse_model_src("sparse_model_src");
  std::string sparse_model_dst("sparse_model_dst");

  float tolerance = 1e-4f;
  OptHyperParams hyper_params;
  hyper_params.sgd.atomic_update = true;
  const OptParams opt_params = {optimizer, lr, hyper_params, update_type, scaler};

  int numprocs = 1, pid = 0;
  std::vector<std::vector<int>> vvgpu;
  test::mpi_init();
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  const auto &resource_manager = ResourceManager::create(vvgpu, 0);

  if (pid == 0) {
    // re-generate the dataset files
    if (fs::exists(train_file_list_name)) {
      fs::remove(train_file_list_name);
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
#endif

  // setup a data reader
  const DataReaderSparseParam param = {DataReaderSparse_t::Localized, max_nnz_per_slot * slot_num,
                                       max_nnz_per_slot, slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  std::unique_ptr<DataReader<T>> train_data_reader(new DataReader<T>(
      train_batchsize, label_dim, dense_dim, params, resource_manager, true, num_chunk_threads, false, 0));

  train_data_reader->create_drwg_norm(train_file_list_name, CHK);

  const SparseEmbeddingHashParams embedding_params = {
      train_batchsize, test_batchsize, 0,        slot_sizes, embedding_vec_size,
      max_feature_num, slot_num,       combiner, opt_params};

  std::unique_ptr<Embedding<T, TypeEmbeddingComp>> embedding(
      new LocalizedSlotSparseEmbeddingOneHot<T, TypeEmbeddingComp>(
          bags_to_tensors<T>(train_data_reader->get_row_offsets_tensors()),
          bags_to_tensors<T>(train_data_reader->get_value_tensors()),
          train_data_reader->get_nnz_array(),
          bags_to_tensors<T>(train_data_reader->get_row_offsets_tensors()),
          bags_to_tensors<T>(train_data_reader->get_value_tensors()),
          train_data_reader->get_nnz_array(),
          embedding_params,
          resource_manager));

  // init hash table file
  if (pid == 0) {
    init_sparse_model(sparse_model_src.c_str());
  }

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // upload hash table to device
  embedding->load_parameters(sparse_model_src);

  if (pid == 0) {
    printf("max_vocabulary_size=%zu, vocabulary_size=%zu\n", embedding->get_max_vocabulary_size(),
      embedding->get_vocabulary_size());
  }

  // dump sparse model to file
  embedding->dump_parameters(sparse_model_dst);

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  auto load_sparse_model_to_map = [](std::vector<T>& key_vec, std::vector<T>& slot_vec,
      std::vector<float>& vec_vec, const std::string& sparse_model) {
    const std::string key_file(sparse_model + "/" + sparse_model + ".key");
    const std::string slot_file(sparse_model + "/" + sparse_model + ".slot");
    const std::string vec_file(sparse_model + "/" + sparse_model + ".vec");

    std::ifstream fs_key(key_file, std::ifstream::binary);
    std::ifstream fs_slot(slot_file, std::ifstream::binary);
    std::ifstream fs_vec(vec_file, std::ifstream::binary);

    const size_t key_file_size_in_B = fs::file_size(key_file);
    const size_t slot_file_size_in_B = fs::file_size(slot_file);
    const size_t vec_file_size_in_B = fs::file_size(vec_file);
    const long long num_key = key_file_size_in_B / sizeof(T);
    const long long num_slot = slot_file_size_in_B / sizeof(T);
    const long long num_vec = vec_file_size_in_B / (sizeof(float) * embedding_vec_size);

    if (num_key != num_vec || num_key != num_slot || num_key != vocabulary_size) {
      CK_THROW_(Error_t::BrokenFile, "num_key != num_vec (num_slot) || num_key != vocabulary_size");
    }

    key_vec.clear();
    key_vec.reserve(num_key);
    slot_vec.clear();
    slot_vec.reserve(num_key);
    vec_vec.clear();
    vec_vec.reserve(num_vec * embedding_vec_size);

    fs_key.read(reinterpret_cast<char *>(key_vec.data()), key_file_size_in_B);
    fs_slot.read(reinterpret_cast<char *>(slot_vec.data()), slot_file_size_in_B);
    fs_vec.read(reinterpret_cast<char *>(vec_vec.data()), vec_file_size_in_B);
  };

  std::vector<T> hash_table_key_from_cpu;
  std::vector<T> slot_id_from_cpu;
  std::vector<float> hash_table_value_from_cpu;
  load_sparse_model_to_map(hash_table_key_from_cpu, slot_id_from_cpu, hash_table_value_from_cpu, sparse_model_src);

  std::vector<T> hash_table_key_from_gpu;
  std::vector<T> slot_id_from_gpu;
  std::vector<float> hash_table_value_from_gpu;
  load_sparse_model_to_map(hash_table_key_from_gpu, slot_id_from_gpu, hash_table_value_from_gpu, sparse_model_dst);

  typedef struct TypeHashValue_ { float data[embedding_vec_size]; } TypeHashValue;

  ASSERT_TRUE(compare_hash_table(vocabulary_size,
    hash_table_key_from_gpu.data(), reinterpret_cast<TypeHashValue *>(hash_table_value_from_gpu.data()),
    hash_table_key_from_cpu.data(), reinterpret_cast<TypeHashValue *>(hash_table_value_from_cpu.data()),
    tolerance));

  ASSERT_TRUE(compare_key_slot(vocabulary_size,
    hash_table_key_from_gpu.data(), slot_id_from_gpu.data(),
    hash_table_key_from_cpu.data(), slot_id_from_cpu.data()));

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

TEST(localized_sparse_embedding_one_hot_test, load_and_dump) {
  load_and_dump<float>({0}, Optimizer_t::SGD, Update_t::Global);
}

TEST(localized_sparse_embedding_one_hot_test, load_and_dump_file_1gpu) {
  load_and_dump_file<float>({0}, Optimizer_t::SGD, Update_t::Global);
}

TEST(localized_sparse_embedding_one_hot_test, load_and_dump_file_4gpu) {
  load_and_dump_file<float>({0, 1, 2, 3}, Optimizer_t::SGD, Update_t::Global);
}
TEST(localized_sparse_embedding_one_hot_test, fp32_sgd_1gpu_nf) {
  train_and_test<float>({0}, Optimizer_t::SGD, Update_t::Local, DeviceMap::NODE_FIRST);
}

TEST(localized_sparse_embedding_one_hot_test, fp32_sgd_4gpu_nf) {
  train_and_test<float>({0, 1, 2, 3}, Optimizer_t::SGD, Update_t::Local, DeviceMap::NODE_FIRST);
}

TEST(localized_sparse_embedding_one_hot_test, fp32_sgd_global_update_1gpu_nf) {
  train_and_test<float>({0}, Optimizer_t::SGD, Update_t::Global, DeviceMap::NODE_FIRST);
}

TEST(localized_sparse_embedding_one_hot_test, fp32_sgd_global_update_4gpu_nf) {
  train_and_test<float>({0, 1, 2, 3}, Optimizer_t::SGD, Update_t::Global, DeviceMap::NODE_FIRST);
}

TEST(localized_sparse_embedding_one_hot_test, fp16_sgd_1gpu_nf) {
  train_and_test<__half>({0}, Optimizer_t::SGD, Update_t::Local, DeviceMap::NODE_FIRST);
}

TEST(localized_sparse_embedding_one_hot_test, fp16_sgd_4gpu_nf) {
  train_and_test<__half>({0, 1, 2, 3}, Optimizer_t::SGD, Update_t::Local, DeviceMap::NODE_FIRST);
}

TEST(localized_sparse_embedding_one_hot_test, fp16_sgd_global_update_1gpu_nf) {
  train_and_test<__half>({0}, Optimizer_t::SGD, Update_t::Global, DeviceMap::NODE_FIRST);
}

TEST(localized_sparse_embedding_one_hot_test, fp16_sgd_global_update_4gpu_nf) {
  train_and_test<__half>({0, 1, 2, 3}, Optimizer_t::SGD, Update_t::Global, DeviceMap::NODE_FIRST);
}
