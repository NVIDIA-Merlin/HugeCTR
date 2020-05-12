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

#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/data_reader.hpp"
//#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include <nccl.h>
#include <sys/time.h>
#include <fstream>
#include <functional>
#include <unordered_set>
#include "HugeCTR/include/embedding.hpp"
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
// const std::vector<int> device_list = {0};
// const std::vector<int> device_list = {0,1};
// const std::vector<int> device_list = {0,1,2,3};
const std::vector<int> device_list = {0, 1, 2, 3, 4, 5, 6, 7};
const int batch_num = 2;  // can not more than 32
const int batchsize = 1024;
const long long num_records = batchsize * batch_num;
const int slot_num = 26;
const int max_nnz_per_slot = 4;
const int max_feature_num = max_nnz_per_slot * slot_num;  // max_feature_num in a sample
const long long vocabulary_size = 1000;
const int embedding_vec_size = 64;
const int combiner = 0;   // 0-sum, 1-mean
const int optimizer = 0;  // 0-adam, 1-momentum_sgd, 2-nesterov
const bool global_update =
    true;  // true-embedding table global update; fase-embedding table local update
// const bool global_update = false;
const float scaler = 1.0f;  // used in mixed precision training
const float lr = 0.01;
const long long label_dim = 1;
const long long dense_dim = 0;
typedef long long T;

// In order to not allocate the total size of hash table on each GPU, the users need to set the
// size of max_vocabulary_size_per_gpu, which should be more than vocabulary_size/gpu_count,
// eg: 1.25x of that.
const float load_factor = 0.75;  // CAUSION: this is a very important param for performance

const int num_chunk_threads = 1;  // must be 1 for CPU and GPU results comparation
const int num_files = 1;
const Check_t CHK = Check_t::Sum;  // Check_t::Sum
const std::string file_list_name("sample_file_list.txt");
const std::string prefix("./data_reader_test_data/temp_dataset_");

const char *hash_table_file_name = "distributed_hash_table.bin";
bool init_hash_table = true;  // true: init hash_table and upload_to_device
                              // false: don't init hash_table or upload_to_device, just use an
                              // empty hash_table to train
//-----------------------------------------------------------------------------------------

#if 0
// distributed_sparse_embedding_hash upload_params() and download_params() testing
TEST(distributed_sparse_embedding_hash_test, upload_and_download_params) {

  const SparseEmbeddingHashParams embedding_params = {
      batchsize, vocabulary_size, load_factor, embedding_vec_size, 
      max_feature_num, slot_num, 0, 0, 1.f};

  int numprocs = 1, pid = 0;
  std::vector<std::vector<int>> vvgpu;
#ifdef ENABLE_MPI
  test::mpi_init();
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
#endif
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  std::shared_ptr<DeviceMap> device_map(new DeviceMap(vvgpu, pid));
  std::shared_ptr<GPUResourceGroup> gpu_resource_group(new GPUResourceGroup(device_map));

  if(pid == 0) {
#if 1
    // re-generate the dataset files 
    std::ifstream file(file_list_name);
    if(file.good()) {
      std::remove(file_list_name.c_str());
    }
#endif 
    // data generation
    HugeCTR::data_generation<T, CHK>(file_list_name, prefix, num_files, num_records, slot_num,
        vocabulary_size, label_dim, dense_dim, max_nnz_per_slot);
  }

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "This is rank: " << pid << std::endl;
#endif 

  //setup a data reader
  const DataReaderSparseParam param = {DataReaderSparse_t::Distributed, max_nnz_per_slot*slot_num, slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);
  DataReader<T> * data_reader = new DataReader<T>(file_list_name, batchsize, label_dim, dense_dim, CHK, params, 
                            gpu_resource_group, num_chunk_threads);

  // define object
  // Embedding<T>* embedding = new DistributedSlotSparseEmbeddingHash<T>(data_reader->get_row_offsets_tensors(), data_reader->get_value_tensors(), embedding_params, gpu_resource_group);

  Embedding<T> *embedding = EmbeddingCreator::create_distributed_sparse_embedding_hash(
                                data_reader->get_row_offsets_tensors(),
                                data_reader->get_value_tensors(),
                                embedding_params, gpu_resource_group);

  const std::string hash_table_upload("distributed_hash_table_upload.bin");
  const std::string hash_table_download("distributed_hash_table_download.bin");         

  if(pid == 0) {
    // init hash table file
    std::ofstream weight_stream(hash_table_upload);
    if(!weight_stream.is_open()) {
      ERROR_MESSAGE_("Error: file not open for writing");
    }
    UnifiedDataSimulator<T> ldata_sim(0, vocabulary_size-1);
    UnifiedDataSimulator<float> fdata_sim(0, vocabulary_size-1);
    T * p_key = (T *)malloc(vocabulary_size * sizeof(T));
    UnorderedKeyGenerator<T> unorderedKey;
    unorderedKey.fill_unique(p_key, vocabulary_size);
    for(int i = 0; i < vocabulary_size; i++) {
      //T key = (T)i;
      //T key = ldata_sim.get_num(); // CAUSION: can not get correct results when testing by the case with duplicated keys
      //weight_stream.write((char *)&key, sizeof(T));
      weight_stream.write((char *)&p_key[i], sizeof(T));
      //float val = (float)i;
      float val = fdata_sim.get_num();
      for(int j = 0; j < embedding_vec_size; j++) {
        weight_stream.write((char *)&val, sizeof(float));
      }
    }
    weight_stream.close();
    free(p_key);
  }

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "This is rank: " << pid << std::endl;
#endif 

  // upload data from host to device
  std::ifstream i_weight_stream(hash_table_upload);
  printf("start updaload_params_to_device()\n");
  embedding->upload_params_to_device(i_weight_stream);
  i_weight_stream.close();

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "This is rank: " << pid << std::endl;
#endif 

  // download data from device to host
  std::ofstream o_weight_stream(hash_table_download);
  printf("start download_params_to_host()\n");
  embedding->download_params_to_host(o_weight_stream);
  o_weight_stream.close();

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif 

  // comapre the read file with the written file
  typedef struct TypeHashValue_{
  	float data[embedding_vec_size];
  } TypeHashValue;

  printf("start compare_distributed_hash_table_files()\n");
  bool rtn = compare_distributed_hash_table_files<T, TypeHashValue>(hash_table_upload, hash_table_download);
  ASSERT_EQ(true, rtn);
}
#endif

#if 1
// distributed_sparse_embedding_hash correctness testing: forward->backward->update_params
TEST(distributed_sparse_embedding_hash_test, training_correctness) {
  OptHyperParams hyper_params;
  hyper_params.adam.beta1 = 0.9f;
  hyper_params.adam.beta2 = 0.999f;
  hyper_params.adam.epsilon = 1e-8f;
  hyper_params.momentum.factor = 0.9f;
  hyper_params.nesterov.mu = 0.9f;

  const OptParams opt_params = {optimizer, lr, hyper_params, global_update};

  const SparseEmbeddingHashParams embedding_params = {
      batchsize, vocabulary_size, load_factor, embedding_vec_size, max_feature_num, slot_num,
      combiner,  opt_params,      scaler};

  int numprocs = 1, pid = 0;
  std::vector<std::vector<int>> vvgpu;
#ifdef ENABLE_MPI
  test::mpi_init();
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
#endif
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  std::shared_ptr<DeviceMap> device_map(new DeviceMap(vvgpu, pid));
  std::shared_ptr<GPUResourceGroup> gpu_resource_group(new GPUResourceGroup(device_map));

  if (pid == 0) {
#if 1
    // re-generate the dataset files
    std::ifstream file(file_list_name);
    if (file.good()) {
      std::remove(file_list_name.c_str());
    }
#endif
    // data generation
    HugeCTR::data_generation<T, CHK>(file_list_name, prefix, num_files, num_records, slot_num,
                                     vocabulary_size, label_dim, dense_dim, max_nnz_per_slot);
  }

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "This is rank: " << pid << std::endl;
#endif

  // setup a data reader
  const DataReaderSparseParam param = {DataReaderSparse_t::Distributed, max_nnz_per_slot * slot_num,
                                       slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);
  DataReader<T> *data_reader =
      new DataReader<T>(file_list_name, batchsize, label_dim, dense_dim, CHK, params,
                        gpu_resource_group, num_chunk_threads);

  // Embedding<T> *embedding = new
  // DistributedSlotSparseEmbeddingHash<T>(data_reader->get_row_offsets_tensors(),
  //                                                      data_reader->get_value_tensors(),
  //                                                      embedding_params, gpu_resource_group);

  Embedding<T> *embedding = EmbeddingCreator::create_distributed_sparse_embedding_hash(
      data_reader->get_row_offsets_tensors(), data_reader->get_value_tensors(), embedding_params,
      gpu_resource_group);

  // init hash table file
  if (init_hash_table) {
    if (pid == 0) {
      std::ofstream weight_stream(hash_table_file_name);
      if (!weight_stream.is_open()) {
        ERROR_MESSAGE_("Error: file not open for writing");
      }
      UnifiedDataSimulator<float> fdata_sim(-0.1f, 0.1f);
      for (long long i = 0; i < vocabulary_size; i++) {
        T key = (T)i;
        // T key = ldata_sim.get_num();
        // CAUSION: can not set random keys here, because we need to ensure that:
        // 1) we can find keys in the data file from this hash table
        // 2) there are no repeated keys
        weight_stream.write((char *)&key, sizeof(T));
        // float val = (float)i;
        // float val = 1.0f;
        float val = fdata_sim.get_num();
        for (int j = 0; j < embedding_vec_size; j++) {
          weight_stream.write((char *)&val, sizeof(float));
        }
      }
      weight_stream.close();
    }

#ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // upload hash table to device
    std::ifstream i_weight_stream(hash_table_file_name);
    embedding->upload_params_to_device(i_weight_stream);
    i_weight_stream.close();
  }

  // for SparseEmbeddingCpu
  SparseEmbeddingHashCpu<T> *embedding_cpu = new SparseEmbeddingHashCpu<T>(
      batchsize, max_feature_num, vocabulary_size, embedding_vec_size, slot_num, label_dim,
      dense_dim, CHK, num_records, combiner, optimizer, lr, file_list_name, hash_table_file_name,
      SparseEmbedding_t::Distributed, global_update, scaler);

  // for results check
  float *embedding_feature_from_gpu =
      (float *)malloc(batchsize * slot_num * embedding_vec_size * sizeof(float));
  float *embedding_feature_from_cpu = embedding_cpu->get_forward_results();
  float *wgrad_from_gpu[device_list.size()];
  for (unsigned int i = 0; i < device_list.size(); i++) {
    wgrad_from_gpu[i] = (float *)malloc(batchsize * slot_num * embedding_vec_size * sizeof(float));
  }
  float *wgrad_from_cpu = embedding_cpu->get_backward_results();
  T *hash_table_key_from_gpu = (T *)malloc(vocabulary_size * sizeof(T));
  float *hash_table_value_from_gpu =
      (float *)malloc(vocabulary_size * (long long)embedding_vec_size * sizeof(float));
  T *hash_table_key_from_cpu = embedding_cpu->get_hash_table_key_ptr();
  float *hash_table_value_from_cpu = embedding_cpu->get_hash_table_value_ptr();

  typedef struct TypeHashValue_ {
    float data[embedding_vec_size];
  } TypeHashValue;

  for (int i = 0; i < batch_num; i++) {
    printf("Rank%d: Round %d start:\n", pid, i);

    // call read a batch
    printf("Rank%d: data_reader->read_a_batch_to_device()\n", pid);
    data_reader->read_a_batch_to_device();

    // GPU forward
    printf("Rank%d: embedding->forward()\n", pid);
    embedding->forward();

    // check the result of forward
    printf("Rank%d: embedding->get_forward_results()\n", pid);
    embedding->get_forward_results(embedding_feature_from_gpu);  // memcpy from GPU to CPU

    if (pid == 0) {
      // CPU forward
      printf("Rank0: embedding_cpu->forward()\n");
      embedding_cpu->forward();

      printf("Rank0: check forward results\n");
      ASSERT_EQ(true,
                compare_embedding_feature(batchsize * slot_num * embedding_vec_size,
                                          embedding_feature_from_gpu, embedding_feature_from_cpu));
    }

#ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // GPU backward
    printf("Rank%d: embedding->backward()\n", pid);
    embedding->backward();

    // check the result of backward
    printf("Rank%d: embedding->get_backward_results()\n", pid);
    embedding->get_backward_results(wgrad_from_gpu[0], 0);

    if (pid == 0) {
      // CPU backward
      printf("Rank0: embedding_cpu->backward()\n");
      embedding_cpu->backward();

      printf("Rank0: check backward results: GPU and CPU\n");
      ASSERT_EQ(true, compare_wgrad(batchsize * slot_num * embedding_vec_size, wgrad_from_gpu[0],
                                    wgrad_from_cpu));
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
      bool rtn = compare_hash_table<T, TypeHashValue>(
          vocabulary_size, (T *)hash_table_key_from_gpu, (TypeHashValue *)hash_table_value_from_gpu,
          (T *)hash_table_key_from_cpu, (TypeHashValue *)hash_table_value_from_cpu);
      ASSERT_EQ(true, rtn);
    }

#ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    printf("Rank%d: Round %d end:\n", pid, i);
  }

  // release resources
  free(embedding_feature_from_gpu);
  for (unsigned int i = 0; i < device_list.size(); i++) {
    free(wgrad_from_gpu[i]);
  }
  free(hash_table_value_from_gpu);
  free(hash_table_key_from_gpu);
}
#endif

}  // namespace
