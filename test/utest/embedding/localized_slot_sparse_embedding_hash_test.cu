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
//#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp"
#include <sys/time.h>
#include <fstream>
#include <functional>
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
// const std::vector<int> device_list = {0,3};
// const std::vector<int> device_list = {0,1,2,3};
const std::vector<int> device_list = {0, 1, 2, 3, 4, 5, 6, 7};
const int batch_num = 2;  // can not more than 32
const int batchsize = 1024;
const int batchsize_eval = 2048;
const long long num_records = batchsize * batch_num;
const int slot_num = 26;
const int max_nnz_per_slot = 1;
const int max_feature_num = max_nnz_per_slot * slot_num;  // max_feature_num in a sample
// const long long vocabulary_size = 187767399;   // for cretio dataset
const long long vocabulary_size = slot_num * 100;
const int embedding_vec_size = 128;
const int combiner = 0;  // 0-sum, 1-mean
// const Optimizer_t optimizer = Optimizer_t::SGD;
const Optimizer_t optimizer = Optimizer_t::Adam;
const bool global_update =
    true;  // true-embedding table global update; fase-embedding table local update
// const bool global_update = false;
const long long label_dim = 1;
const long long dense_dim = 0;
typedef long long T;
typedef float TypeEmbeddingComp;  // fp32 test
// typedef __half TypeEmbeddingComp; // fp16 test

const float scaler = 1.0f;  // used in mixed precision training
const float lr = 0.01f;

// In order to not allocate the total size of hash table on each GPU, the users need to set the
// size of max_vocabulary_size_per_gpu, which should be more than vocabulary_size/gpu_count,
// eg: 1.25x of that.

const int num_chunk_threads = 1;  // must be 1 for CPU and GPU results comparation
const int num_files = 1;
const Check_t CHK = Check_t::Sum;  // Check_t::Sum
const std::string file_list_name("sample_file_list.txt");
const std::string prefix("./data_reader_test_data/temp_dataset_");

#ifndef NCCl_A2A
// const std::string plan_file(PROJECT_HOME_ + "utest/all2all_plan_dgx_{0}.json");
// const std::string plan_file(PROJECT_HOME_ + "utest/all2all_plan_dgx_{0,1}.json");
// const std::string plan_file(PROJECT_HOME_ + "utest/all2all_plan_dgx_{0,3}.json");
// const std::string plan_file(PROJECT_HOME_ + "utest/all2all_plan_dgx_{0,1,2,3}.json");
const std::string plan_file(PROJECT_HOME_ + "utest/all2all_plan_dgx_{0,1,2,3,4,5,6,7}.json");
#else
const std::string plan_file = "";
#endif

const char *hash_table_file_name = "localized_hash_table.bin";
bool init_hash_table =
    true;  // true: init hash_table and upload_to_device
           // false: don't init hash_table or upload_to_device, just use an
           //        empty hash_table to train
           // CAUSION: for training_correctness checking, must set this flag to true

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
#if 0  // init embedding in LocalizedEmbedding ctor
TEST(localized_sparse_embedding_hash_test, init_embedding) {

  OptHyperParams hyper_params;
  const OptParams<float> opt_params = {optimizer, lr, hyper_params, global_update, scaler};
  const SparseEmbeddingHashParams<float> embedding_params = {
      batchsize, vocabulary_size, {}, embedding_vec_size, max_feature_num, slot_num,
      combiner,  opt_params};

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
#if 0
    // re-generate the dataset files 
    std::ifstream file(file_list_name);
    if(file.good()) {
      std::remove(file_list_name.c_str());
    }
#endif 
    // data generation: key's corresponding slot_id=(key%slot_num)
    HugeCTR::data_generation_for_localized_test<T, CHK>(file_list_name, prefix, num_files, num_records, slot_num,
                                                        vocabulary_size, label_dim, dense_dim, max_nnz_per_slot);
  }

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "This is rank: " << pid << std::endl;
#endif 

  //setup a data reader
  const DataReaderSparseParam param = {DataReaderSparse_t::Localized, max_nnz_per_slot*slot_num, 
                                      max_nnz_per_slot, slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);
  DataReader<T> * data_reader = new DataReader<T>(file_list_name, batchsize, label_dim, dense_dim, CHK, params, 
                            gpu_resource_group, num_chunk_threads);
  
  Embedding<T, float> *embedding = EmbeddingCreator::create_localized_sparse_embedding_hash(data_reader->get_row_offsets_tensors(),
        data_reader->get_value_tensors(), embedding_params, plan_file, gpu_resource_group, slot_sizes);

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

#if 0  // just for verify
  // download data from device to host
  std::ofstream o_weight_stream("hash_table_init.bin");
  printf("start download_params_to_host()\n");
  embedding->download_params_to_host(o_weight_stream);
  o_weight_stream.close();

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif 

  std::ifstream i_weight_stream("hash_table_init.bin");
  if(!i_weight_stream.is_open()) {
    ERROR_MESSAGE_("Error: file not open for reading");
  }
  i_weight_stream.seekg(0, i_weight_stream.beg);
  long long start_pos = i_weight_stream.tellg();
  i_weight_stream.seekg(0, i_weight_stream.end);
  long long end_pos = i_weight_stream.tellg();
  long long file_size = end_pos - start_pos;
  std::cout << "file size=" << file_size << std::endl;
  i_weight_stream.seekg(0, i_weight_stream.beg);
  size_t pair_size_in_B = sizeof(T) + sizeof(T) + sizeof(float)*embedding_vec_size;
  std::cout << "pair_size_in_B=" << pair_size_in_B << std::endl;
  std::unique_ptr<char> buf(new char[pair_size_in_B]);
  T * key;
  T * slot_id;
  float * value;
  while (i_weight_stream.peek() != EOF) {
    i_weight_stream.read(buf.get(), pair_size_in_B);
    key = (T *)(buf.get());
    slot_id = (T *)(buf.get() + sizeof(T));
    value = (float *)(buf.get() + sizeof(T) * 2);  // including slot_id and value

    std::cout << "key=" << *key << ", slot_id=" << *slot_id << ", value=[";
    for(int i = 0; i < embedding_vec_size; i++) {
      std::cout << value[i] << " ";
    }
    std::cout << "]" << std::endl;
  } 
  i_weight_stream.close();
#endif 
}

// #if 0  // upload_params time measurement
// TEST(localized_sparse_embedding_hash_test, upload_params) {

  // OptHyperParams hyper_params;
  // const OptParams<TypeEmbeddingComp> opt_params = {optimizer, lr, hyper_params, global_update, scaler};
  // const SparseEmbeddingHashParams<TypeEmbeddingComp> embedding_params = {
  //     batchsize, vocabulary_size, embedding_vec_size, max_feature_num, slot_num,
  //     combiner,  opt_params};

//   int numprocs = 1, pid = 0;
//   std::vector<std::vector<int>> vvgpu;
// #ifdef ENABLE_MPI
//   test::mpi_init();
//   MPI_Comm_rank(MPI_COMM_WORLD, &pid);
//   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
// #endif
//   for (int i = 0; i < numprocs; i++) {
//     vvgpu.push_back(device_list);
//   }
//   std::shared_ptr<DeviceMap> device_map(new DeviceMap(vvgpu, pid));
//   std::shared_ptr<GPUResourceGroup> gpu_resource_group(new GPUResourceGroup(device_map));

//   if(pid == 0) {
// #if 0
//     // re-generate the dataset files 
//     std::ifstream file(file_list_name);
//     if(file.good()) {
//       std::remove(file_list_name.c_str());
//     }
// #endif 
//     // data generation: key's corresponding slot_id=(key%slot_num)
//     HugeCTR::data_generation_for_localized_test<T, CHK>(file_list_name, prefix, num_files, num_records, slot_num,
//                                                        vocabulary_size, label_dim, dense_dim, max_nnz_per_slot);
//   }

// #ifdef ENABLE_MPI
//   MPI_Barrier(MPI_COMM_WORLD);
//   std::cout << "This is rank: " << pid << std::endl;
// #endif 

//   //setup a data reader
//   const DataReaderSparseParam param = {DataReaderSparse_t::Localized, max_nnz_per_slot*slot_num, max_nnz_per_slot, slot_num};
//   std::vector<DataReaderSparseParam> params;
//   params.push_back(param);
//   DataReader<T> * data_reader = new DataReader<T>(file_list_name, batchsize, label_dim, dense_dim, CHK, params, 
//                             gpu_resource_group, num_chunk_threads);

//   // define object
//   // Embedding<T>* embedding = new LocalizedSlotSparseEmbeddingHash<T>(
//   //     data_reader->get_row_offsets_tensors(), data_reader->get_value_tensors(), 
//   //     embedding_params, plan_file, gpu_resource_group);

//   Embedding<T> *embedding = EmbeddingCreator::create_localized_sparse_embedding_hash(data_reader->get_row_offsets_tensors(),
//         data_reader->get_value_tensors(), embedding_params, plan_file, gpu_resource_group, slot_sizes);

//   const std::string hash_table_upload("../tools//model_generation/embedding_table_64.bin");

//   // upload data from host to device
//   std::ifstream i_weight_stream(hash_table_upload);
//   printf("start updaload_params_to_device()\n");
//   embedding->upload_params_to_device(i_weight_stream);
//   i_weight_stream.close();

// #ifdef ENABLE_MPI
//   MPI_Barrier(MPI_COMM_WORLD);
// #endif 
// }
// #endif
#endif

#if 0
// localized_sparse_embedding_hash upload_params() and download_params() testing
TEST(localized_sparse_embedding_hash_test, upload_and_download_params) {

  OptHyperParams hyper_params;
  hyper_params.adam.alpha_t = 0.0f;
  hyper_params.adam.beta1 = 0.9f;
  hyper_params.adam.beta2 = 0.999f;
  hyper_params.adam.epsilon = 1e-8f;
  hyper_params.momentum.factor = 0.9f;
  hyper_params.nesterov.mu = 0.9f;

  const OptParams<TypeEmbeddingComp> opt_params = {optimizer, lr, hyper_params, global_update, scaler};

  const SparseEmbeddingHashParams<TypeEmbeddingComp> embedding_params = {
      batchsize, vocabulary_size, {}, embedding_vec_size, max_feature_num, slot_num,
      combiner,  opt_params};

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
    if(file.good()) 
      std::remove(file_list_name.c_str());
    }
#endif 
    // data generation: key's corresponding slot_id=(key%slot_num)
    HugeCTR::data_generation_for_localized_test<T, CHK>(file_list_name, prefix, num_files, num_records, slot_num,
                                                        vocabulary_size, label_dim, dense_dim, max_nnz_per_slot);
  }

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "This is rank: " << pid << std::endl;
#endif 

  //setup a data reader
  const DataReaderSparseParam param = {DataReaderSparse_t::Localized, max_nnz_per_slot*slot_num, 
                                      max_nnz_per_slot, slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);
  DataReader<T> * data_reader = new DataReader<T>(file_list_name, batchsize, label_dim, dense_dim, CHK, params, 
                            gpu_resource_group, num_chunk_threads);

  // define object
  // Embedding<T>* embedding = new LocalizedSlotSparseEmbeddingHash<T>(
  //     data_reader->get_row_offsets_tensors(), data_reader->get_value_tensors(), 
  //     embedding_params, plan_file, gpu_resource_group);

  Embedding<T, TypeEmbeddingComp> *embedding = EmbeddingCreator::create_localized_sparse_embedding_hash(
    data_reader->get_row_offsets_tensors(), data_reader->get_value_tensors(), embedding_params,
    plan_file, gpu_resource_group, slot_sizes);

  // init hash table file
  const std::string hash_table_upload("localized_hash_table_upload.bin");
  const std::string hash_table_download("localized_hash_table_download.bin");       

  if(pid == 0) {
    std::ofstream weight_stream(hash_table_upload);
    if(!weight_stream.is_open()) {
      ERROR_MESSAGE_("Error: file not open for writing");
    }
    //UnifiedDataSimulator<T> ldata_sim(0, vocabulary_size-1); // for key 
    UnifiedDataSimulator<T> ldata_sim(0, slot_num-1); // for slot_id
    UnifiedDataSimulator<float> fdata_sim(0, vocabulary_size-1); // for value
    T * p_key = (T *)malloc(vocabulary_size * sizeof(T));
    UnorderedKeyGenerator<T> unorderedKey;
    unorderedKey.fill_unique(p_key, vocabulary_size);
    // key + slot_id + value
    for(int i = 0; i < vocabulary_size; i++) {
      //T key = (T)i;
      //T key = ldata_sim.get_num(); // CAUSION: can not get correct results when testing by the case with duplicated keys
      //weight_stream.write((char *)&key, sizeof(T));
      weight_stream.write((char *)&p_key[i], sizeof(T));
      T slot_id = ldata_sim.get_num();
      weight_stream.write((char *)&slot_id, sizeof(T));
      //float val = (float)i;
      float val = fdata_sim.get_num();
      for(int j = 0; j < embedding_vec_size; j++) {
        weight_stream.write((char *)&val, sizeof(float));
      }

      // just for debug 
      // std::cout << "i=" << i << ":key=" << p_key[i] << " slot_id=" << slot_id << " val=" << val << std::endl;
    }
    weight_stream.close();
    free(p_key);
  }

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif 

  // upload data from host to device
  std::ifstream i_weight_stream(hash_table_upload);
  printf("start updaload_params_to_device()\n");
  embedding->upload_params_to_device(i_weight_stream);
  i_weight_stream.close();

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
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

  printf("start compare_localized_hash_table_files()\n");
  bool rtn = compare_localized_hash_table_files<T, T, TypeHashValue>(hash_table_upload, hash_table_download);
  ASSERT_EQ(true, rtn);
}
#endif

#if 1
// localized_sparse_embedding_hash correctness testing: forward->backward->update_params
TEST(localized_sparse_embedding_hash_test, training_correctness) {
  OptHyperParams hyper_params;
  hyper_params.adam.alpha_t = 0.0f;
  hyper_params.adam.beta1 = 0.9f;
  hyper_params.adam.beta2 = 0.999f;
  hyper_params.adam.epsilon = 1e-8f;
  hyper_params.momentum.factor = 0.9f;
  hyper_params.nesterov.mu = 0.9f;

  const OptParams<TypeEmbeddingComp> opt_params = {optimizer, lr, hyper_params, global_update,
                                                   scaler};

  const SparseEmbeddingHashParams<TypeEmbeddingComp> embedding_params = {
      batchsize,       vocabulary_size, {}, embedding_vec_size,
      max_feature_num, slot_num,        combiner,    opt_params};

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

  if (pid == 0) {
    std::cout << "rank " << pid << " is generating data" << std::endl;
#if 1
    // re-generate the dataset files
    std::ifstream file(file_list_name);
    if (file.good()) {
      std::remove(file_list_name.c_str());
    }
#endif
    // data generation: key's corresponding slot_id=(key%slot_num)
    if (slot_sizes.size() > 0) {
      HugeCTR::data_generation_for_localized_test<T, CHK>(
          file_list_name, prefix, num_files, num_records, slot_num, vocabulary_size, label_dim,
          dense_dim, max_nnz_per_slot, slot_sizes);
    } else {
      HugeCTR::data_generation_for_localized_test<T, CHK>(file_list_name, prefix, num_files,
                                                          num_records, slot_num, vocabulary_size,
                                                          label_dim, dense_dim, max_nnz_per_slot);
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
  DataReader<T> *data_reader =
      new DataReader<T>(file_list_name, batchsize, label_dim, dense_dim, CHK, params,
                        gpu_resource_group, num_chunk_threads);

  // Embedding<T> *embedding = new
  // LocalizedSlotSparseEmbeddingHash<T>(data_reader->get_row_offsets_tensors(),
  //                                                      data_reader->get_value_tensors(),
  //                                                      embedding_params, plan_file,
  //                                                      gpu_resource_group);

  slot_sizes.clear();  // don't init hashtable when doing training correctness checking.
                       // Because we will upload hashtable to GPUs.
  Embedding<T, TypeEmbeddingComp> *embedding =
      EmbeddingCreator::create_localized_sparse_embedding_hash(
          data_reader->get_row_offsets_tensors(), data_reader->get_value_tensors(),
          embedding_params, plan_file, gpu_resource_group);

  if (init_hash_table) {
    // generate hashtable
    if (pid == 0) {
      std::cout << "Init hash table";
      // init hash table file: <key, solt_id, value>
      std::ofstream weight_stream(hash_table_file_name);
      if (!weight_stream.is_open()) {
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
        weight_stream.write((char *)&key, sizeof(T));
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
        weight_stream.write((char *)&slot_id, sizeof(T));
        // float val = (float)i;
        // float val = 0.1f;
        float val = fdata_sim.get_num();
        for (int j = 0; j < embedding_vec_size; j++) {
          weight_stream.write((char *)&val, sizeof(float));
        }
      }
      weight_stream.close();
      std::cout << " Done" << std::endl;
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
      SparseEmbedding_t::Localized, global_update, scaler);

  float *embedding_feature_from_cpu = embedding_cpu->get_forward_results();
  float *wgrad_from_cpu = embedding_cpu->get_backward_results();
  T *hash_table_key_from_cpu = embedding_cpu->get_hash_table_key_ptr();
  float *hash_table_value_from_cpu = embedding_cpu->get_hash_table_value_ptr();

  // for results check
  TypeEmbeddingComp *embedding_feature_from_gpu = (TypeEmbeddingComp *)malloc(
      batchsize * slot_num * embedding_vec_size * sizeof(TypeEmbeddingComp));
  TypeEmbeddingComp *wgrad_from_gpu = (TypeEmbeddingComp *)malloc(
      batchsize * slot_num * embedding_vec_size * sizeof(TypeEmbeddingComp));
  T *hash_table_key_from_gpu = (T *)malloc(vocabulary_size * sizeof(T));
  float *hash_table_value_from_gpu =
      (float *)malloc(vocabulary_size * (long long)embedding_vec_size * sizeof(float));

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
    embedding->get_backward_results(wgrad_from_gpu, 0);

    if (pid == 0) {
      // CPU backward
      printf("Rank0: embedding_cpu->backward()\n");
      embedding_cpu->backward();

      printf("Rank0: check backward results: GPU and CPU\n");
      ASSERT_EQ(true, compare_wgrad(batchsize * slot_num * embedding_vec_size, wgrad_from_gpu,
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

  test::mpi_finialize();

  // release resources
  free(embedding_feature_from_gpu);
  free(wgrad_from_gpu);
  free(hash_table_value_from_gpu);
  free(hash_table_key_from_gpu);
}
#endif

#if 1
TEST(localized_sparse_embedding_hash_test, train_eval_correctness) {
  OptHyperParams hyper_params;
  hyper_params.adam.alpha_t = 0.0f;
  hyper_params.adam.beta1 = 0.9f;
  hyper_params.adam.beta2 = 0.999f;
  hyper_params.adam.epsilon = 1e-8f;
  hyper_params.momentum.factor = 0.9f;
  hyper_params.nesterov.mu = 0.9f;

  const OptParams<TypeEmbeddingComp> opt_params = {optimizer, lr, hyper_params, global_update,
                                                   scaler};

  const SparseEmbeddingHashParams<TypeEmbeddingComp> embedding_params = {
      batchsize,       vocabulary_size, {}, embedding_vec_size,
      max_feature_num, slot_num,        combiner,    opt_params};

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

  if (pid == 0) {
    std::cout << "rank " << pid << " is generating data" << std::endl;
#if 1
    // re-generate the dataset files
    std::ifstream file(file_list_name);
    if (file.good()) {
      std::remove(file_list_name.c_str());
    }
#endif
    // data generation: key's corresponding slot_id=(key%slot_num)
    if (slot_sizes.size() > 0) {
      HugeCTR::data_generation_for_localized_test<T, CHK>(
          file_list_name, prefix, num_files, num_records, slot_num, vocabulary_size, label_dim,
          dense_dim, max_nnz_per_slot, slot_sizes);
    } else {
      HugeCTR::data_generation_for_localized_test<T, CHK>(file_list_name, prefix, num_files,
                                                          num_records, slot_num, vocabulary_size,
                                                          label_dim, dense_dim, max_nnz_per_slot);
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
  DataReader<T> *data_reader =
      new DataReader<T>(file_list_name, batchsize, label_dim, dense_dim, CHK, params,
                        gpu_resource_group, num_chunk_threads);

  // Embedding<T> *embedding = new
  // LocalizedSlotSparseEmbeddingHash<T>(data_reader->get_row_offsets_tensors(),
  //                                                      data_reader->get_value_tensors(),
  //                                                      embedding_params, plan_file,
  //                                                      gpu_resource_group);

  slot_sizes.clear();  // don't init hashtable when doing training correctness checking.
                       // Because we will upload hashtable to GPUs.
  Embedding<T, TypeEmbeddingComp> *embedding =
      EmbeddingCreator::create_localized_sparse_embedding_hash(
          data_reader->get_row_offsets_tensors(), data_reader->get_value_tensors(),
          embedding_params, plan_file, gpu_resource_group);

  if (init_hash_table) {
    // generate hashtable
    if (pid == 0) {
      std::cout << "Init hash table";
      // init hash table file: <key, solt_id, value>
      std::ofstream weight_stream(hash_table_file_name);
      if (!weight_stream.is_open()) {
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
        weight_stream.write((char *)&key, sizeof(T));
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
        weight_stream.write((char *)&slot_id, sizeof(T));
        // float val = (float)i;
        // float val = 0.1f;
        float val = fdata_sim.get_num();
        for (int j = 0; j < embedding_vec_size; j++) {
          weight_stream.write((char *)&val, sizeof(float));
        }
      }
      weight_stream.close();
      std::cout << " Done" << std::endl;
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
      SparseEmbedding_t::Localized, global_update, scaler);

  float *embedding_feature_from_cpu = embedding_cpu->get_forward_results();
  float *wgrad_from_cpu = embedding_cpu->get_backward_results();
  T *hash_table_key_from_cpu = embedding_cpu->get_hash_table_key_ptr();
  float *hash_table_value_from_cpu = embedding_cpu->get_hash_table_value_ptr();

  // for results check
  TypeEmbeddingComp *embedding_feature_from_gpu = (TypeEmbeddingComp *)malloc(
      batchsize * slot_num * embedding_vec_size * sizeof(TypeEmbeddingComp));
  TypeEmbeddingComp *wgrad_from_gpu = (TypeEmbeddingComp *)malloc(
      batchsize * slot_num * embedding_vec_size * sizeof(TypeEmbeddingComp));
  T *hash_table_key_from_gpu = (T *)malloc(vocabulary_size * sizeof(T));
  float *hash_table_value_from_gpu =
      (float *)malloc(vocabulary_size * (long long)embedding_vec_size * sizeof(float));

  typedef struct TypeHashValue_ {
    float data[embedding_vec_size];
  } TypeHashValue;

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // create new obj for eval()
  DataReader<T> *data_reader_eval =
      new DataReader<T>(file_list_name, batchsize_eval, label_dim, dense_dim, CHK, params,
                        gpu_resource_group, num_chunk_threads);
  Embedding<T, TypeEmbeddingComp> *embedding_eval = embedding->clone_eval(
      data_reader_eval->get_row_offsets_tensors(), data_reader_eval->get_value_tensors(),
      batchsize_eval, gpu_resource_group);

  // for SparseEmbeddingCpu eval
  SparseEmbeddingHashCpu<T> *embedding_cpu_eval = new SparseEmbeddingHashCpu<T>(
      batchsize_eval, max_feature_num, vocabulary_size, embedding_vec_size, slot_num, label_dim,
      dense_dim, CHK, num_records, combiner, optimizer, lr, file_list_name, hash_table_file_name,
      SparseEmbedding_t::Localized, global_update, scaler);

  float *embedding_feature_from_cpu_eval = embedding_cpu_eval->get_forward_results();

  // for results check
  TypeEmbeddingComp *embedding_feature_from_gpu_eval = (TypeEmbeddingComp *)malloc(
      batchsize_eval * slot_num * embedding_vec_size * sizeof(TypeEmbeddingComp));

  for (int i = 0; i < batch_num; i++) {
    printf("Rank%d: Round %d start training:\n", pid, i);

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
    embedding->get_backward_results(wgrad_from_gpu, 0);

    if (pid == 0) {
      // CPU backward
      printf("Rank0: embedding_cpu->backward()\n");
      embedding_cpu->backward();

      printf("Rank0: check backward results: GPU and CPU\n");
      ASSERT_EQ(true, compare_wgrad(batchsize * slot_num * embedding_vec_size, wgrad_from_gpu,
                                    wgrad_from_cpu));
    }

#ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    /* do not do update_params since cpu code can not support eval mode (2 embedding obj share one
    hashtable)

        // GPU update_params
        printf("Rank%d: embedding->update_params()\n", pid);
        embedding->update_params();

    #if 1 // can not check update_params since one-hot has no hashtable
        if (pid == 0) {
          // CPU update_params
          printf("Rank0: embedding_cpu->update_params()\n");
          embedding_cpu->update_params();
        }
    #else
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
              vocabulary_size, (T *)hash_table_key_from_gpu, (TypeHashValue
    *)hash_table_value_from_gpu, (T *)hash_table_key_from_cpu, (TypeHashValue
    *)hash_table_value_from_cpu); ASSERT_EQ(true, rtn);
        }

    #endif

    #ifdef ENABLE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
    #endif

    */

    /////////////////////////////////////////////////////////////////////////////////////////////
    // eval
    printf("\nRank%d: Round %d start eval:\n", pid, i);

    // call read a batch
    printf("Rank%d: data_reader_eval->read_a_batch_to_device()\n", pid);
    data_reader_eval->read_a_batch_to_device();

    // GPU forward
    printf("Rank%d: embedding_eval->forward()\n", pid);
    embedding_eval->forward();

    // check the result of forward
    printf("Rank%d: embedding_eval->get_forward_results()\n", pid);
    embedding_eval->get_forward_results(embedding_feature_from_gpu_eval);  // memcpy from GPU to CPU

    if (pid == 0) {
      // CPU forward
      printf("Rank0: embedding_cpu_eval->forward()\n");
      embedding_cpu_eval->forward();

      printf("Rank0: check forward results\n");
      ASSERT_EQ(true, compare_embedding_feature(batchsize_eval * slot_num * embedding_vec_size,
                                                embedding_feature_from_gpu_eval,
                                                embedding_feature_from_cpu_eval));
    }

#ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    printf("Rank%d: Round %d end:\n", pid, i);
  }

  test::mpi_finialize();

  // release resources
  free(embedding_feature_from_gpu);
  free(wgrad_from_gpu);
  free(hash_table_value_from_gpu);
  free(hash_table_key_from_gpu);
  free(embedding_feature_from_gpu_eval);
}
#endif

#if 0
TEST(localized_sparse_embedding_hash_test, profile) {
  OptHyperParams hyper_params;
  hyper_params.adam.alpha_t = 0.0f;
  hyper_params.adam.beta1 = 0.9f;
  hyper_params.adam.beta2 = 0.999f;
  hyper_params.adam.epsilon = 1e-8f;
  hyper_params.momentum.factor = 0.9f;
  hyper_params.nesterov.mu = 0.9f;

  const OptParams<TypeEmbeddingComp> opt_params = {optimizer, lr, hyper_params, global_update, scaler};

  const SparseEmbeddingHashParams<TypeEmbeddingComp> embedding_params = {
      batchsize, vocabulary_size, {}, embedding_vec_size, max_feature_num, slot_num,
      combiner,  opt_params};

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

  if (pid == 0) {
    std::cout << "rank " << pid << " is generating data" << std::endl;
#if 1
    // re-generate the dataset files
    std::ifstream file(file_list_name);
    if (file.good()) {
      std::remove(file_list_name.c_str());
    }
#endif
    // data generation: key's corresponding slot_id=(key%slot_num)
    HugeCTR::data_generation_for_localized_test<T, CHK>(file_list_name, prefix, num_files,
                                                        num_records, slot_num, vocabulary_size,
                                                        label_dim, dense_dim, max_nnz_per_slot);
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
  DataReader<T> *data_reader =
      new DataReader<T>(file_list_name, batchsize, label_dim, dense_dim, CHK, params,
                        gpu_resource_group, num_chunk_threads);

  // Embedding<T> *embedding = new
  // LocalizedSlotSparseEmbeddingHash<T>(data_reader->get_row_offsets_tensors(),
  //                                                      data_reader->get_value_tensors(),
  //                                                      embedding_params, plan_file,
  //                                                      gpu_resource_group);

  Embedding<T, TypeEmbeddingComp> *embedding = EmbeddingCreator::create_localized_sparse_embedding_hash(
      data_reader->get_row_offsets_tensors(), data_reader->get_value_tensors(), embedding_params,
      plan_file, gpu_resource_group, slot_sizes);

  HugeCTR::Timer timer;

  for (int i = 0; i < batch_num; i++) {

    data_reader->read_a_batch_to_device();

    timer.start();
  
    embedding->forward();
    embedding->backward();
    embedding->update_params();

    timer.stop();
    std::cout << "Time per iter: " << timer.elapsedSeconds() << "s" << std::endl;
  }

  test::mpi_finialize();
}
#endif

}  // namespace
