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

#include <sys/time.h>

#include <experimental/filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <unordered_set>

#include "HugeCTR/include/data_generator.hpp"
#include "HugeCTR/include/data_readers/data_reader.hpp"
#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/resource_managers/resource_manager_ext.hpp"
#include "gtest/gtest.h"
#include "nvToolsExt.h"
#include "utest/embedding/embedding_test_utils.hpp"
#include "utest/embedding/sparse_embedding_hash_cpu.hpp"
#include "utest/test_utils.h"

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

const int num_threads = 1;  // must be 1 to keep data reading consistency with cpu and gpu
const int num_files = 1;
const Check_t CHK = Check_t::Sum;  // Check_t::Sum
const char *train_file_list_name = "train_file_list.txt";
const char *test_file_list_name = "test_file_list.txt";
const char *prefix = "./data_reader_test_data/temp_dataset_";
const char *sparse_model_file = "distributed_hash_table";
const char *opt_file_name = "distributed_opt.bin";
//-----------------------------------------------------------------------------------------

auto load_sparse_model_to_map = [](std::vector<T> &key_vec, std::vector<float> &vec_vec,
                                   const std::string &sparse_model) {
  const std::string key_file(sparse_model + "/key");
  const std::string vec_file(sparse_model + "/emb_vector");

  std::ifstream fs_key(key_file, std::ifstream::binary);
  std::ifstream fs_vec(vec_file, std::ifstream::binary);

  const size_t key_file_size_in_B = fs::file_size(key_file);
  const size_t vec_file_size_in_B = fs::file_size(vec_file);
  const long long num_key = key_file_size_in_B / sizeof(long long);
  const long long num_vec = vec_file_size_in_B / (sizeof(float) * embedding_vec_size);

  if (num_key != num_vec || num_key != vocabulary_size) {
    CK_THROW_(Error_t::BrokenFile, "num_key != num_vec || num_key != vocabulary_size");
  }

  key_vec.clear();
  key_vec.resize(num_key);
  vec_vec.clear();
  vec_vec.resize(num_vec * embedding_vec_size);

  using TypeKey = typename std::decay<decltype(*key_vec.begin())>::type;
  if (std::is_same<TypeKey, long long>::value) {
    fs_key.read(reinterpret_cast<char *>(key_vec.data()), key_file_size_in_B);
  } else {
    std::vector<long long> i64_key_vec(num_key, 0);
    fs_key.read(reinterpret_cast<char *>(i64_key_vec.data()), key_file_size_in_B);
    std::transform(i64_key_vec.begin(), i64_key_vec.end(), key_vec.begin(),
                   [](long long key) { return static_cast<unsigned>(key); });
  }
  fs_vec.read(reinterpret_cast<char *>(vec_vec.data()), vec_file_size_in_B);
};

void init_sparse_model(const char *sparse_model) {
  if (!fs::exists(sparse_model)) {
    fs::create_directories(sparse_model);
  }
  const std::string key_file = std::string(sparse_model) + "/key";
  const std::string vec_file = std::string(sparse_model) + "/emb_vector";

  std::ofstream fs_key(key_file);
  std::ofstream fs_vec(vec_file);
  if (!fs_key.is_open() || !fs_vec.is_open()) {
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
    fs_key.write((char *)&key, sizeof(T));

    fdata_sim.fill(buf.get(), embedding_vec_size, -0.1f, 0.1f);
    fs_vec.write(reinterpret_cast<const char *>(buf.get()), embedding_vec_size * sizeof(float));
  }
}

template <typename TypeEmbeddingComp>
void train_and_test(const std::vector<int> &device_list, const Optimizer_t &optimizer,
                    const Update_t &update_type) {
  OptHyperParams hyper_params;
  hyper_params.adam.beta1 = 0.9f;
  hyper_params.adam.beta2 = 0.999f;
  float tolerance;
  if (std::is_same<TypeEmbeddingComp, __half>::value) {
    hyper_params.adam.epsilon = 1e-4f;
    hyper_params.adagrad.epsilon = 1e-4f;
    tolerance = 5e-3f;
  } else {
    hyper_params.adam.epsilon = 1e-7f;
    hyper_params.adagrad.epsilon = 1e-6f;
    tolerance = 1e-4f;
  }
  hyper_params.momentum.factor = 0.9f;
  hyper_params.nesterov.mu = 0.9f;
  hyper_params.adagrad.initial_accu_value = 0.f;

  const float lr = optimizer == Optimizer_t::Adam ? 0.001f : 0.01f;

  const OptParams opt_params = {optimizer, lr, hyper_params, update_type, scaler};

  int numprocs = 1, pid = 0;
  std::vector<std::vector<int>> vvgpu;
  test::mpi_init();
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  const auto &resource_manager = ResourceManagerExt::create(vvgpu, 0);

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
  const DataReaderSparseParam param = {"distributed", max_nnz_per_slot, false, slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  std::unique_ptr<DataReader<T>> train_data_reader(new DataReader<T>(
      train_batchsize, label_dim, dense_dim, params, resource_manager, true, num_threads, false));

  train_data_reader->create_drwg_norm(train_file_list_name, CHK);

  std::unique_ptr<DataReader<T>> test_data_reader(new DataReader<T>(
      test_batchsize, label_dim, dense_dim, params, resource_manager, true, num_threads, false));

  test_data_reader->create_drwg_norm(test_file_list_name, CHK);

  // init hash table file
  if (pid == 0) {
    init_sparse_model(sparse_model_file);
  }

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  const SparseEmbeddingHashParams embedding_params = {train_batchsize,
                                                      test_batchsize,
                                                      vocabulary_size,
                                                      {},
                                                      embedding_vec_size,
                                                      max_feature_num,
                                                      slot_num,
                                                      combiner,
                                                      opt_params,
                                                      true,
                                                      false};

  auto copy = [](const std::vector<SparseTensorBag> &tensorbags, SparseTensors<T> &sparse_tensors) {
    sparse_tensors.resize(tensorbags.size());
    for (size_t j = 0; j < tensorbags.size(); ++j) {
      sparse_tensors[j] = SparseTensor<T>::stretch_from(tensorbags[j]);
    }
  };
  SparseTensors<T> train_input;
  copy(train_data_reader->get_sparse_tensors("distributed"), train_input);
  SparseTensors<T> test_input;
  copy(test_data_reader->get_sparse_tensors("distributed"), test_input);

  std::unique_ptr<DistributedSlotSparseEmbeddingHash<T, TypeEmbeddingComp>> embedding(
      new DistributedSlotSparseEmbeddingHash<T, TypeEmbeddingComp>(
          train_input, test_input, embedding_params, resource_manager));

  // upload hash table to device
  embedding->load_parameters(sparse_model_file);

  // for SparseEmbeddingCpu
  std::unique_ptr<SparseEmbeddingHashCpu<T, TypeEmbeddingComp>> embedding_cpu(
      new SparseEmbeddingHashCpu<T, TypeEmbeddingComp>(
          train_batchsize, max_feature_num, vocabulary_size, embedding_vec_size, slot_num,
          label_dim, dense_dim, CHK, train_batch_num * train_batchsize, combiner, opt_params,
          train_file_list_name, sparse_model_file, SparseEmbedding_t::Distributed));

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

  typedef struct TypeHashValue_ {
    float data[embedding_vec_size];
  } TypeHashValue;

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
  embedding->dump_parameters(sparse_model_file);

  {
    printf("Rank%d: embedding->dump_opt_states()\n", pid);
    std::ofstream fs(opt_file_name);
    embedding->dump_opt_states(fs);
    fs.close();
  }

  {
    printf("Rank%d: embedding->load_opt_states()\n", pid);
    std::ifstream fs(opt_file_name);
    embedding->load_opt_states(fs);
    fs.close();
  }

  // for SparseEmbeddingCpu eval
  std::unique_ptr<SparseEmbeddingHashCpu<T, TypeEmbeddingComp>> test_embedding_cpu(
      new SparseEmbeddingHashCpu<T, TypeEmbeddingComp>(
          test_batchsize, max_feature_num, vocabulary_size, embedding_vec_size, slot_num, label_dim,
          dense_dim, CHK, test_batch_num * test_batchsize, combiner, opt_params,
          test_file_list_name, sparse_model_file, SparseEmbedding_t::Distributed));

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
  using TypeKey = T;
  OptHyperParams hyper_params;
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
  if (std::is_same<TypeEmbeddingComp, __half>::value) {
    hyper_params.adagrad.epsilon = 1e-4f;
    hyper_params.adam.epsilon = 1e-4f;
  } else {
    hyper_params.adagrad.epsilon = 1e-7f;
    hyper_params.adam.epsilon = 1e-7f;
  }
  hyper_params.momentum.factor = 0.9f;
  hyper_params.nesterov.mu = 0.9f;
  hyper_params.adagrad.initial_accu_value = 0.9f;

  const float lr = optimizer == Optimizer_t::Adam ? 0.001f : 0.01f;

  const OptParams opt_params = {optimizer, lr, hyper_params, update_type, scaler};

  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back(device_list);
  const auto &resource_manager = ResourceManagerExt::create(vvgpu, 0);

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
  const DataReaderSparseParam param = {"distributed", max_nnz_per_slot, false, slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  std::unique_ptr<DataReader<T>> train_data_reader(new DataReader<T>(
      train_batchsize, label_dim, dense_dim, params, resource_manager, true, num_threads, false));

  train_data_reader->create_drwg_norm(train_file_list_name, CHK);

  // init hash table file
  init_sparse_model(sparse_model_file);

  const SparseEmbeddingHashParams embedding_params = {train_batchsize,
                                                      test_batchsize,
                                                      vocabulary_size,
                                                      {},
                                                      embedding_vec_size,
                                                      max_feature_num,
                                                      slot_num,
                                                      combiner,
                                                      opt_params,
                                                      true,
                                                      false};

  auto copy = [](const std::vector<SparseTensorBag> &tensorbags, SparseTensors<T> &sparse_tensors) {
    sparse_tensors.resize(tensorbags.size());
    for (size_t j = 0; j < tensorbags.size(); ++j) {
      sparse_tensors[j] = SparseTensor<T>::stretch_from(tensorbags[j]);
    }
  };
  SparseTensors<T> train_input;
  copy(train_data_reader->get_sparse_tensors("distributed"), train_input);

  std::unique_ptr<DistributedSlotSparseEmbeddingHash<T, TypeEmbeddingComp>> embedding(
      new DistributedSlotSparseEmbeddingHash<T, TypeEmbeddingComp>(
          train_input, train_input, embedding_params, resource_manager));

  // upload hash table to device
  embedding->load_parameters(sparse_model_file);

  printf("max_vocabulary_size=%zu, vocabulary_size=%zu\n", embedding->get_max_vocabulary_size(),
         embedding->get_vocabulary_size());

  BufferBag buf_bag;
  {
    size_t buffer_size = embedding->get_max_vocabulary_size();
    size_t max_voc_size_per_gpu = embedding_params.max_vocabulary_size_per_gpu;

    auto host_blobs_buff = GeneralBuffer2<CudaHostAllocator>::create();

    Tensor2<TypeKey> tensor_keys;
    Tensor2<size_t> tensor_slot_id;
    host_blobs_buff->reserve({buffer_size}, &tensor_keys);
    host_blobs_buff->reserve({buffer_size}, &tensor_slot_id);
    host_blobs_buff->reserve({buffer_size, embedding_vec_size}, &(buf_bag.embedding));

    buf_bag.keys = tensor_keys.shrink();
    buf_bag.slot_id = tensor_slot_id.shrink();

    const size_t local_gpu_count = resource_manager->get_local_gpu_count();

    for (size_t id = 0; id < local_gpu_count; id++) {
      Tensor2<float> tensor;
      host_blobs_buff->reserve({max_voc_size_per_gpu, embedding_vec_size}, &tensor);
      buf_bag.h_value_tensors.push_back(tensor);
    }
    host_blobs_buff->allocate();

    CudaDeviceContext context;
    for (size_t id = 0; id < local_gpu_count; id++) {
      context.set_device(resource_manager->get_local_gpu(id)->get_device_id());
      {
        auto uvm_blobs_buff = GeneralBuffer2<CudaManagedAllocator>::create();
        Tensor2<TypeKey> tensor;
        uvm_blobs_buff->reserve({max_voc_size_per_gpu}, &tensor);
        buf_bag.uvm_key_tensor_bags.push_back(tensor.shrink());
        uvm_blobs_buff->allocate();
      }
      {
        auto hbm_blobs_buff = GeneralBuffer2<CudaAllocator>::create();
        Tensor2<size_t> tensor;
        hbm_blobs_buff->reserve({max_voc_size_per_gpu}, &tensor);
        buf_bag.d_value_index_tensors.push_back(tensor);
        hbm_blobs_buff->allocate();
      }
    }
  }

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

  std::string tmp_sparse_model_file{"tmp_sparse_model"};
  embedding->dump_parameters(tmp_sparse_model_file);

  std::vector<T> hash_table_key_from_cpu;
  std::vector<float> hash_table_value_from_cpu;
  load_sparse_model_to_map(hash_table_key_from_cpu, hash_table_value_from_cpu, sparse_model_file);

  std::vector<T> hash_table_key_from_gpu;
  std::vector<float> hash_table_value_from_gpu;
  load_sparse_model_to_map(hash_table_key_from_gpu, hash_table_value_from_gpu,
                           tmp_sparse_model_file);

  typedef struct TypeHashValue_ {
    float data[embedding_vec_size];
  } TypeHashValue;

  ASSERT_TRUE(compare_hash_table(
      vocabulary_size, hash_table_key_from_gpu.data(),
      reinterpret_cast<TypeHashValue *>(hash_table_value_from_gpu.data()),
      hash_table_key_from_cpu.data(),
      reinterpret_cast<TypeHashValue *>(hash_table_value_from_cpu.data()), tolerance));
}

template <typename TypeEmbeddingComp>
void load_and_dump_file(const std::vector<int> &device_list, const Optimizer_t &optimizer,
                        const Update_t &update_type) {
  std::string sparse_model_src("sparse_model_src");
  std::string sparse_model_dst("sparse_model_dst");

  OptHyperParams hyper_params;
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
  const OptParams opt_params = {optimizer, lr, hyper_params, update_type, scaler};

  int numprocs = 1, pid = 0;
  std::vector<std::vector<int>> vvgpu;
  test::mpi_init();
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  const auto &resource_manager = ResourceManagerExt::create(vvgpu, 0);

  if (pid == 0) {
    // re-generate the dataset files
    if (fs::exists(train_file_list_name)) {
      fs::remove(train_file_list_name);
    }

    // data generation
    HugeCTR::data_generation_for_test<T, CHK>(
        train_file_list_name, prefix, num_files, train_batch_num * train_batchsize, slot_num,
        vocabulary_size, label_dim, dense_dim, max_nnz_per_slot);
  }

#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // setup a data reader
  const DataReaderSparseParam param = {"distributed", max_nnz_per_slot, false, slot_num};
  std::vector<DataReaderSparseParam> params;
  params.push_back(param);

  std::unique_ptr<DataReader<T>> train_data_reader(new DataReader<T>(
      train_batchsize, label_dim, dense_dim, params, resource_manager, true, num_threads, false));

  train_data_reader->create_drwg_norm(train_file_list_name, CHK);

  const SparseEmbeddingHashParams embedding_params = {train_batchsize,
                                                      test_batchsize,
                                                      vocabulary_size,
                                                      {},
                                                      embedding_vec_size,
                                                      max_feature_num,
                                                      slot_num,
                                                      combiner,
                                                      opt_params,
                                                      true,
                                                      false};

  auto copy = [](const std::vector<SparseTensorBag> &tensorbags, SparseTensors<T> &sparse_tensors) {
    sparse_tensors.resize(tensorbags.size());
    for (size_t j = 0; j < tensorbags.size(); ++j) {
      sparse_tensors[j] = SparseTensor<T>::stretch_from(tensorbags[j]);
    }
  };
  SparseTensors<T> train_input;
  copy(train_data_reader->get_sparse_tensors("distributed"), train_input);

  std::unique_ptr<DistributedSlotSparseEmbeddingHash<T, TypeEmbeddingComp>> embedding(
      new DistributedSlotSparseEmbeddingHash<T, TypeEmbeddingComp>(
          train_input, train_input, embedding_params, resource_manager));

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

  std::vector<T> hash_table_key_from_cpu;
  std::vector<float> hash_table_value_from_cpu;
  load_sparse_model_to_map(hash_table_key_from_cpu, hash_table_value_from_cpu, sparse_model_src);

  std::vector<T> hash_table_key_from_gpu;
  std::vector<float> hash_table_value_from_gpu;
  load_sparse_model_to_map(hash_table_key_from_gpu, hash_table_value_from_gpu, sparse_model_dst);

  typedef struct TypeHashValue_ {
    float data[embedding_vec_size];
  } TypeHashValue;

  ASSERT_TRUE(compare_hash_table(
      vocabulary_size, hash_table_key_from_gpu.data(),
      reinterpret_cast<TypeHashValue *>(hash_table_value_from_gpu.data()),
      hash_table_key_from_cpu.data(),
      reinterpret_cast<TypeHashValue *>(hash_table_value_from_cpu.data()), tolerance));

  test::mpi_finalize();
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

TEST(distributed_sparse_embedding_hash_test, fp32_adagrad_1gpu) {
  train_and_test<float>({0}, Optimizer_t::AdaGrad, Update_t::Local);
}

TEST(distributed_sparse_embedding_hash_test, fp32_adagrad_8gpu) {
  train_and_test<float>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::AdaGrad, Update_t::Local);
}

TEST(distributed_sparse_embedding_hash_test, fp16_adagrad_1gpu) {
  train_and_test<__half>({0}, Optimizer_t::AdaGrad, Update_t::Local);
}

TEST(distributed_sparse_embedding_hash_test, fp16_adagrad_8gpu) {
  train_and_test<__half>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::AdaGrad, Update_t::Local);
}

TEST(distributed_sparse_embedding_hash_test, load_and_dump_1gpu) {
  load_and_dump<float>({0}, Optimizer_t::SGD, Update_t::Global);
}

TEST(distributed_sparse_embedding_hash_test, load_and_dump_8gpu) {
  load_and_dump<float>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::SGD, Update_t::Global);
}

TEST(distributed_sparse_embedding_hash_test, load_and_dump_file_1gpu) {
  load_and_dump_file<float>({0}, Optimizer_t::SGD, Update_t::Global);
}

TEST(distributed_sparse_embedding_hash_test, load_and_dump_file_8gpu) {
  load_and_dump_file<float>({0, 1, 2, 3, 4, 5, 6, 7}, Optimizer_t::SGD, Update_t::Global);
}
