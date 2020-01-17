/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <nccl.h>
#include <sys/time.h>
#include <fstream>
#include <functional>
#include <unordered_set>
#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/data_reader.hpp"
#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp"
#include "gtest/gtest.h"
#include "nvToolsExt.h"
#include "utest/embedding/sparse_embedding_hash_cpu.hpp"
#include "utest/test_utils.h"

#define EPSILON 1e-4

//#define PRINT_DEBUG 1
#ifdef PRINT_DEBUG
#define PRINTF printf
#else
#define PRINTF(...)
#endif

using namespace HugeCTR;

bool compare_float(float a, float b) {
  // compare absolute error
  if (fabs(a - b) < EPSILON) return true;

  // compare relative error
  if (fabs(a) >= fabs(b))
    if (fabs((a - b) / a) < EPSILON)
      return true;
    else
      return false;
  else if (fabs((a - b) / b) < EPSILON)
    return true;
  else
    return false;
}

bool compare_float_array(float *a, float *b, size_t len) {
  bool rtn = true;

  for (size_t i = 0; i < len; i++) {
    if (compare_float(a[i], b[i]) != true) {
      printf("Error in compare_float_array: i=%d, a=%.8f, n=%.8f\n", (int)i, a[i], b[i]);
      rtn = false;
      break;
    }
  }

  return rtn;
}

bool compare_float_files(std::string file1, std::string file2) {
  std::ifstream file_stream1(file1);
  std::ifstream file_stream2(file2);

  if (!file_stream1.is_open() || !file_stream2.is_open()) {
    ERROR_MESSAGE_("Error: file open failed");
    return false;
  }

  long long start_pos = file_stream1.tellg();
  file_stream1.seekg(0, file_stream1.end);
  long long end_pos = file_stream1.tellg();
  long long file_size1 = end_pos - start_pos;

  file_stream2.seekg(0, file_stream1.beg);
  start_pos = file_stream2.tellg();
  file_stream2.seekg(0, file_stream2.end);
  long long file_size2 = end_pos - start_pos;

  if (file_size1 != file_size2) {
    ERROR_MESSAGE_("Error: files size is not same");
    file_stream1.close();
    file_stream2.close();
    return false;
  }

  file_stream1.seekg(0, file_stream1.beg);
  file_stream2.seekg(0, file_stream2.beg);

  bool rtn = true;
  while (file_stream1.peek() != EOF) {
    float val1, val2;
    file_stream1.read((char *)&val1, sizeof(float));
    file_stream2.read((char *)&val2, sizeof(float));
    if (!compare_float(val1, val2)) {
      rtn = false;
      break;
    }
  }

  file_stream1.close();
  file_stream2.close();

  return rtn;
}

// hash table files have same keys and values, but they may be unordered
template <typename TypeHashKey, typename TypeHashValue>
bool compare_distributed_hash_table_files(std::string file1, std::string file2) {
  bool rtn = true;

  std::ifstream file_stream1(file1);
  std::ifstream file_stream2(file2);

  if (!file_stream1.is_open() || !file_stream2.is_open()) {
    ERROR_MESSAGE_("Error: file open failed");
    return false;
  }

  long long start_pos = file_stream1.tellg();
  file_stream1.seekg(0, file_stream1.end);
  long long end_pos = file_stream1.tellg();
  long long file_size1 = end_pos - start_pos;

  file_stream2.seekg(0, file_stream1.beg);
  start_pos = file_stream2.tellg();
  file_stream2.seekg(0, file_stream2.end);
  long long file_size2 = end_pos - start_pos;

  if (file_size1 != file_size2) {
    ERROR_MESSAGE_("Error: files size is not same");
    file_stream1.close();
    file_stream2.close();
    return false;
  }

  file_stream1.seekg(0, file_stream1.beg);
  file_stream2.seekg(0, file_stream2.beg);

  size_t pair_size_in_B = sizeof(TypeHashKey) + sizeof(TypeHashValue);
  long long pair_num = file_size1 / pair_size_in_B;

  // CAUSION: file_stream1 is ordered, while file_stream2 is unordered
  // So, firstly, we read <key,value> pairs from file_stream2, and insert it into a hash table.
  char *buf = (char *)malloc(pair_size_in_B);
  TypeHashKey *key;
  TypeHashValue *value;
  HashTableCpu<TypeHashKey, TypeHashValue> *hash_table =
      new HashTableCpu<TypeHashKey, TypeHashValue>();
  while (file_stream2.peek() != EOF) {
    file_stream2.read(buf, pair_size_in_B);
    key = (TypeHashKey *)buf;
    value = (TypeHashValue *)(buf + sizeof(TypeHashKey));
    hash_table->insert(key, value, 1);
  }
  file_stream2.close();

  if (hash_table->get_size() != pair_num) {
    ERROR_MESSAGE_(
        "Error: The number of <key,value> pair inserting into hash table is not equal to hash "
        "table file size\n");
    return false;
  }

  // Then, we read <key,value1> pairs from file_stream1, and get(key,value2) from hash table, and
  // compare value1 and value2.
  TypeHashValue *value1;
  TypeHashValue *value2 = (TypeHashValue *)malloc(sizeof(TypeHashValue));
  size_t value_len = sizeof(TypeHashValue) / sizeof(float);
  while (file_stream1.peek() != EOF) {
    file_stream1.read(buf, pair_size_in_B);
    key = (TypeHashKey *)buf;
    value1 = (TypeHashValue *)(buf + sizeof(TypeHashKey));
    hash_table->get(key, value2, 1);
    if (!compare_float_array((float *)value1, (float *)value2, value_len)) {
      rtn = false;
      break;
    }
  }
  file_stream1.close();

  free(value2);

  return rtn;
}

// hash table files have same keys and values, but they may be unordered
template <typename TypeHashKey, typename TypeHashValue>
bool compare_localized_hash_table_files(std::string file1, std::string file2) {
  bool rtn = true;

  std::ifstream file_stream1(file1);
  std::ifstream file_stream2(file2);

  if (!file_stream1.is_open() || !file_stream2.is_open()) {
    ERROR_MESSAGE_("Error: file open failed");
    return false;
  }

  long long start_pos = file_stream1.tellg();
  file_stream1.seekg(0, file_stream1.end);
  long long end_pos = file_stream1.tellg();
  long long file_size1 = end_pos - start_pos;

  file_stream2.seekg(0, file_stream1.beg);
  start_pos = file_stream2.tellg();
  file_stream2.seekg(0, file_stream2.end);
  long long file_size2 = end_pos - start_pos;

  if (file_size1 != file_size2) {
    ERROR_MESSAGE_("Error: files size is not same");
    std::cout << "file_size1=" << file_size1 << ", file_size2="\
         << file_size2 << std::endl;
    file_stream1.close();
    file_stream2.close();
    return false;
  }

  file_stream1.seekg(0, file_stream1.beg);
  file_stream2.seekg(0, file_stream2.beg);

  size_t pair_size_in_B = sizeof(TypeHashKey) + sizeof(TypeHashValue);
  size_t pair_num = file_size1 / pair_size_in_B;

#ifndef NDEBUG
  std::cout << "pair_size_in_B=" << pair_size_in_B << std::endl;
  std::cout << "pair_num=" << pair_num << std::endl; 
#endif 

  // CAUSION: file_stream1 is ordered, while file_stream2 is unordered
  // So, firstly, we read <key,value> pairs from file_stream2, and insert it into a hash table.
  char *buf = (char *)malloc(pair_size_in_B);
  TypeHashKey *key;
  TypeHashValue *value;
  HashTableCpu<TypeHashKey, TypeHashValue> *hash_table =
      new HashTableCpu<TypeHashKey, TypeHashValue>();
  while (file_stream2.peek() != EOF) {
    file_stream2.read(buf, pair_size_in_B);
    key = (TypeHashKey *)buf;
    value = (TypeHashValue *)(buf + sizeof(TypeHashKey)); // including slot_id and value
    hash_table->insert(key, value, 1);
  }
  file_stream2.close();

  size_t hash_table_size = hash_table->get_size();
  if (hash_table_size != pair_num) {
    ERROR_MESSAGE_(
        "Error: The number of <key,value> pair inserting into CPU hash table is not equal to hash "
        "table file size\n");
    std::cout << "CPU hash_table_size=" << hash_table_size << std::endl;
    return false;
  }

  // Then, we read <key,value1> pairs from file_stream1, and get(key,value2) from hash_table, and
  // compare value1 and value2.
  TypeHashValue *value1;
  TypeHashValue *value2 = (TypeHashValue *)malloc(sizeof(TypeHashValue));
  size_t value_len = sizeof(TypeHashValue) / sizeof(float);
  while (file_stream1.peek() != EOF) {
    file_stream1.read(buf, pair_size_in_B);
    key = (TypeHashKey *)buf;
    value1 = (TypeHashValue *)(buf + sizeof(TypeHashKey));
    hash_table->get(key, value2, 1);
    if (!compare_float_array((float *)value1, (float *)value2, value_len)) {
      rtn = false;
      break;
    }
  }
  file_stream1.close();

  free(value2);

  return rtn;
}

bool compare_embedding_feature(int num, float *embedding_feature_from_gpu,
                               float *embedding_feature_from_cpu) {
  bool rtn = true;
  // int err = 0;

  for (int i = 0; i < num; i++) {
    if (!compare_float(embedding_feature_from_gpu[i], embedding_feature_from_cpu[i])) {
      rtn = false;
      break;

      //      err++;
      //      if(err > 256) {
      //        break;
      //      }
      //      printf("Error: i=%d, embedding_feature_from_gpu=%.8f,
      //      embedding_feature_from_cpu=%.8f\n", i, embedding_feature_from_gpu[i],
      //      embedding_feature_from_cpu[i]);
    }
  }

  return rtn;
}

bool compare_wgrad(int num, float *wgrad_from_gpu, float *wgrad_from_cpu) {
  bool rtn = true;
  //  int err = 0;

  for (int i = 0; i < num; i++) {
    if (!compare_float(wgrad_from_gpu[i], wgrad_from_cpu[i])) {
      rtn = false;
      break;

      //      err++;
      //      if(err > 256) {
      //        break;
      //      }
      //      printf("Error: i=%d, wgrad_from_gpu=%.8f, wgrad_from_cpu=%.8f\n", i,
      //      wgrad_from_gpu[i], wgrad_from_cpu[i]);
    }
  }

  return rtn;
}

bool compare_embedding_table(long long num, float *embedding_table_from_gpu,
                             float *embedding_table_from_cpu) {
  bool rtn = true;
  int err = 0;

  for (long long i = 0; i < num; i++) {
    if (!compare_float(embedding_table_from_gpu[i], embedding_table_from_cpu[i])) {
      rtn = false;
      //      break;

      err++;
      if (err > 256) {
        break;
      }
      printf("Error: i=%lld, embedding_table_from_gpu=%.8f, embedding_table_from_cpu=%.8f\n", i,
             embedding_table_from_gpu[i], embedding_table_from_cpu[i]);
    }
  }

  return rtn;
}

template <typename TypeHashKey, typename TypeHashValue>
bool compare_hash_table(long long capacity, TypeHashKey *hash_table_key_from_gpu,
                        TypeHashValue *hash_table_value_from_gpu,
                        TypeHashKey *hash_table_key_from_cpu,
                        TypeHashValue *hash_table_value_from_cpu) {
  bool rtn = true;

  //	// just for debug
  //	for(long long i = 0; i < capacity; i++) {
  //		printf("i=%d, key_from_gpu=%d, key_from_cpu=%d \n", i, hash_table_key_from_gpu[i],
  // hash_table_key_from_cpu[i]);
  //	}

  // Since the <key1,value1> and <key2,value2> is not the same ordered, we need to insert <key1,
  // value1> into a hash_table, then compare value1=hash_table->get(key2) with value2
  HashTableCpu<TypeHashKey, TypeHashValue> *hash_table =
      new HashTableCpu<TypeHashKey, TypeHashValue>();
  hash_table->insert(hash_table_key_from_gpu, hash_table_value_from_gpu, capacity);

  TypeHashKey *key;
  TypeHashValue *value1 = (TypeHashValue *)malloc(sizeof(TypeHashValue));
  TypeHashValue *value2;
  size_t value_len = sizeof(TypeHashValue) / sizeof(float);
  for (long long i = 0; i < capacity; i++) {
    key = hash_table_key_from_cpu + i;
    value2 = hash_table_value_from_cpu + i;

    hash_table->get(key, value1, 1);
    if (!compare_float_array((float *)value1, (float *)value2, value_len)) {
      rtn = false;
      break;
    }
  }

  free(value1);

  return rtn;
}

template <typename T>
class UnorderedKeyGenerator {
 public:
  UnorderedKeyGenerator() : gen_(rd_()) {}
  UnorderedKeyGenerator(T min, T max) : gen_(rd_()), dis_(min, max) {}

  // generate unduplicated dataset
  void fill_unique(T *data, size_t len) {
    if (len == 0) {
      return;
    }
    assert(dis_.max() - dis_.min() >= len - 1);

    std::unordered_set<T> set;
    size_t sz = 0;
    while (sz < len) {
      T x = dis_(gen_);
      auto res = set.insert(x);
      if (res.second) {
        data[sz++] = x;
      }
    }
    assert(sz == set.size());
    assert(sz == len);
  }

 private:
  std::random_device rd_;
  std::mt19937 gen_;
  std::uniform_int_distribution<T> dis_;
};

#if 0
// sparse_embedding_hash upload_params() and download_params() testing
TEST(distributed_sparse_embedding_hash_test, upload_and_download_params) {
  test::mpi_init();
  
  // influential params for this test
  const long long vocabulary_size = 50010;
  //const long long vocabulary_size = 20;
  //const long long vocabulary_size = 1010;
  const int embedding_vec_size = 64;
  //const int embedding_vec_size = 1;
  std::vector<int> device_list = {0};
  //std::vector<int> device_list = {0,1};
  int num_devices = device_list.size();
  const char * hash_table_file_name = "distributed_hash_table.bin";
  const char * hash_table_check_file_name = "hash_table_check.bin";

  // uninfluential params
  const int slot_num = 2;
  const int max_feature_num = 2*slot_num;
  const int batchsize = 2;
  const int batch_num = 1; // can not more than 32
  const long long num_records = batchsize * batch_num;
  const long long label_dim = 1;
  typedef long long T;

  // In order to not allocate the total size of hash table on each GPU, the users need to set the size of max_vocabulary_size_per_gpu,
	// which should be more than vocabulary_size/gpu_count, eg: (1/0.75)x of that.
  float load_factor = 0.75; // CAUSION: this is a very important param for hash_table get() performance
  //long long max_vocabulary_size_per_gpu = (long long)((double)(vocabulary_size) / num_devices / load_factor);

  const SparseEmbeddingHashParams embedding_params = {
  	batchsize,
    vocabulary_size,
    load_factor,
    embedding_vec_size,
    max_feature_num,
    slot_num,
    0,              //combiner: 0-sum, 1-mean
    0               //optimizer: 0-adam
  };

  // CUASION: the data will be used in this test case, but we still need to let the data file non-empty since the DataRead requiring
  // generate input data
  const std::string tmp_file_name("temp_dataset_embedding.data");
  const std::string file_list_name("file_list_embedding.txt");
  {
    //data generation;
    std::ofstream out_stream(tmp_file_name, std::ofstream::binary);
    DataSetHeader header = {num_records, label_dim, slot_num, 0};
    out_stream.write(reinterpret_cast<char*>(&header), sizeof(DataSetHeader));
    for(int i=0; i<num_records; i++){
      UnifiedDataSimulator<int> idata_sim(0, max_feature_num/slot_num-1); //both inclusive
      UnifiedDataSimulator<T> ldata_sim(0,vocabulary_size-1);
      for(int j=0; j<label_dim; j++){
        int label = idata_sim.get_num();
        out_stream.write(reinterpret_cast<char*>(&label), sizeof(int));
      }
      for(int k=0; k<slot_num; k++){
        int nnz = idata_sim.get_num();
        //nnz = 10; // just for test 20181211
        nnz = (int)(max_feature_num/slot_num); // just for test 20181221
        out_stream.write(reinterpret_cast<char*>(&nnz), sizeof(int));
        for(int j=0; j<nnz; j++){
          T value = ldata_sim.get_num();
          out_stream.write(reinterpret_cast<char*>(&value), sizeof(T));
        }
        //std::cout << std::endl; // just for test 20181211
      }
    }
    out_stream.close();
    std::ofstream file_list_stream(file_list_name, std::ofstream::out);
    file_list_stream << (std::to_string(1) + "\n");
    file_list_stream << (tmp_file_name + "\n");
    file_list_stream.close();
  }

  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back(device_list);
  std::shared_ptr<DeviceMap> device_map(new DeviceMap(vvgpu, 0));
  std::shared_ptr<GPUResourceGroup> gpu_resource_group(new GPUResourceGroup(device_map));

  //setup a data reader
  DataReader<T>* data_reader = new DataReader<T>(file_list_name, batchsize, \
    label_dim, slot_num, max_feature_num, gpu_resource_group, 1, 1);

  // define object
  Embedding<T>* embedding = new DistributedSlotSparseEmbeddingHash<T>(data_reader->get_row_offsets_tensors(), data_reader->get_value_tensors(), embedding_params, gpu_resource_group);

  // init hash table file
  std::ofstream weight_stream(hash_table_file_name);
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

  // upload data from host to device
  std::ifstream i_weight_stream(hash_table_file_name);
  printf("start updaload_params_to_device()\n");
  embedding->upload_params_to_device(i_weight_stream);
  i_weight_stream.close();

  // download data from device to host
  std::ofstream o_weight_stream(hash_table_check_file_name);
  printf("start download_params_to_host()\n");
  embedding->download_params_to_host(o_weight_stream);
  o_weight_stream.close();

  // comapre the read file with the written file
  typedef struct TypeHashValue_{
  	float data[embedding_vec_size];
  } TypeHashValue;
  //ASSERT_EQ(true, compare_distributed_hash_table_files<T, TypeHashValue>(hash_table_file_name, hash_table_check_file_name));
  printf("start compare_distributed_hash_table_files()\n");
  bool rtn = compare_distributed_hash_table_files<T, TypeHashValue>(hash_table_file_name, hash_table_check_file_name);
  ASSERT_EQ(true, rtn);
}
#endif

#if 0
// sparse_embedding_hash correctness testing: forward->backward->update_params
TEST(distributed_sparse_embedding_hash_test, training_correctness) {
  test::mpi_init();

  constexpr int batch_num = 4;  // can not more than 32
  // constexpr int batch_num = 1;
  constexpr int batchsize = 4096;
  // constexpr int batchsize = 2;
  constexpr long long num_records = batchsize * batch_num;
  // constexpr int slot_num = 1;
  constexpr int slot_num = 2;
  constexpr int max_feature_num = 10 * slot_num;  // max_feature_num in a sample
  // constexpr int max_feature_num = 2*slot_num;
  constexpr long long vocabulary_size = 55000;
  // constexpr long long vocabulary_size = 10;
  constexpr int embedding_vec_size = 128;
  // constexpr int embedding_vec_size = 1;
  constexpr int combiner = 0;   // 0-sum, 1-mean
  constexpr int optimizer = 0;  // 0-adam, 1-momentum_sgd, 2-nesterov
  constexpr float lr = 0.01;
  // std::vector<int> device_list = {0,1};
  std::vector<int> device_list = {0};
  int num_devices = device_list.size();
  constexpr long long label_dim = 1;
  typedef long long T;

  // In order to not allocate the total size of hash table on each GPU, the users need to set the
  // size of max_vocabulary_size_per_gpu, which should be more than vocabulary_size/gpu_count,
  // eg: 1.25x of that.
  float load_factor = 0.75;  // CAUSION: this is a very important param for performance
  // long long max_vocabulary_size_per_gpu = (long long)((double)(vocabulary_size) / num_devices /
  // load_factor);

  const char *hash_table_file_name = "distributed_hash_table.bin";
  const char *data_file_name = "temp_dataset_embedding.data";

  bool init_hash_table = true;  // true: init hash_table and upload_to_device
                                // false: don't init hash_table or upload_to_device, just use an
                                // empty hash_table to train

  // set up params
  OptHyperParams hyper_params;
  hyper_params.adam.beta1 = 0.9f;
  hyper_params.adam.beta2 = 0.999f;
  hyper_params.adam.epsilon = 1e-8f;
  hyper_params.momentum.factor = 0.9f;
  hyper_params.nesterov.mu = 0.9f;

  OptParams opt_params = {optimizer, lr, hyper_params};

  const SparseEmbeddingHashParams embedding_params = {
      batchsize, vocabulary_size, load_factor, embedding_vec_size, max_feature_num, slot_num,
      combiner,  // combiner: 0-sum, 1-mean
      opt_params};

  // generate input data
  const std::string tmp_file_name(data_file_name);
  const std::string file_list_name("file_list_embedding.txt");
  {
    // data generation;
    std::ofstream out_stream(tmp_file_name, std::ofstream::binary);
    DataSetHeader header = {num_records, label_dim, slot_num, 0};
    out_stream.write(reinterpret_cast<char *>(&header), sizeof(DataSetHeader));
    for (int i = 0; i < num_records; i++) {
      UnifiedDataSimulator<int> idata_sim(0, max_feature_num / slot_num - 1);  // both inclusive
      UnifiedDataSimulator<T> ldata_sim(0, vocabulary_size - 1);
      for (int j = 0; j < label_dim; j++) {
        int label = idata_sim.get_num();
        out_stream.write(reinterpret_cast<char *>(&label), sizeof(int));
      }
      for (int k = 0; k < slot_num; k++) {
        int nnz = idata_sim.get_num();
        nnz = (int)(max_feature_num / slot_num);  // just for test
        out_stream.write(reinterpret_cast<char *>(&nnz), sizeof(int));
        for (int j = 0; j < nnz; j++) {
          T value = ldata_sim.get_num();
          // T value = k*nnz+j; // just for test, 20190625
          out_stream.write(reinterpret_cast<char *>(&value), sizeof(T));
        }
        // std::cout << std::endl; // just for test 20181211
      }
    }
    out_stream.close();
    std::ofstream file_list_stream(file_list_name, std::ofstream::out);
    file_list_stream << (std::to_string(1) + "\n");
    file_list_stream << (tmp_file_name + "\n");
    file_list_stream.close();
  }

  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back(device_list);
  std::shared_ptr<DeviceMap> device_map(new DeviceMap(vvgpu, 0));
  std::shared_ptr<GPUResourceGroup> gpu_resource_group(new GPUResourceGroup(device_map));

  // setup a data reader
  DataReader<T> *data_reader = new DataReader<T>(file_list_name, batchsize, label_dim, slot_num,
                                                 max_feature_num, gpu_resource_group, 1, 1);

  Embedding<T> *embedding = new DistributedSlotSparseEmbeddingHash<T>(data_reader->get_row_offsets_tensors(),
                                                       data_reader->get_value_tensors(),
                                                       embedding_params, gpu_resource_group);

  if (init_hash_table) {
    // init hash table file
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

    // upload hash table to device
    std::ifstream i_weight_stream(hash_table_file_name);
    embedding->upload_params_to_device(i_weight_stream);
    i_weight_stream.close();
  }

  // for SparseEmbeddingCpu
  std::ifstream weight_stream_cpu(hash_table_file_name);
  std::ifstream csr_stream_cpu(data_file_name);
  SparseEmbeddingHashCpu<T> *embedding_cpu = new SparseEmbeddingHashCpu<T>(
      batchsize, max_feature_num, vocabulary_size, embedding_vec_size, slot_num, combiner,
      optimizer, lr, weight_stream_cpu, csr_stream_cpu, label_dim);

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
    printf("Round %d start:\n", i);

    // call read a batch
    data_reader->read_a_batch_to_device();

    // GPU forward
    embedding->forward();

    // CPU forward
    embedding_cpu->forward();

    // check the result of forward
    embedding->get_forward_results(embedding_feature_from_gpu);  // memcpy from GPU to CPU
    ASSERT_EQ(true,
              compare_embedding_feature(batchsize * slot_num * embedding_vec_size,
                                        embedding_feature_from_gpu, embedding_feature_from_cpu));

    // GPU backward
    embedding->backward();

    // CPU backward
    embedding_cpu->backward();

    // check the result of backward
    embedding->get_backward_results(wgrad_from_gpu[0], 0);
    // check the result on multi GPUs first
    if (device_list.size() > 1) {
      for (unsigned int j = 1; j < device_list.size(); j++) {
        embedding->get_backward_results(wgrad_from_gpu[j], j);  // memcpy from GPU to CPU
        // printf("\ncompare GPU[%d] and GPU[%d]\n", 0, j);
        ASSERT_EQ(true, compare_wgrad(batchsize * slot_num * embedding_vec_size, wgrad_from_gpu[0],
                                      wgrad_from_gpu[j]));
      }
    }
    // printf("\ncompare GPU0 and CPU\n");
    ASSERT_EQ(true, compare_wgrad(batchsize * slot_num * embedding_vec_size, wgrad_from_gpu[0],
                                  wgrad_from_cpu));

    // GPU update_params
    embedding->update_params();

    // CPU update_params
    embedding_cpu->update_params();

    // check the results of update params
    embedding->get_update_params_results(hash_table_key_from_gpu,
                                  hash_table_value_from_gpu);  // memcpy from GPU to CPU
    // ASSERT_EQ(true, compare_embedding_table(vocabulary_size*embedding_vec_size,
    // hash_table_value_from_gpu, hash_table_value_from_cpu));
    bool rtn = compare_hash_table<T, TypeHashValue>(
        vocabulary_size, (T *)hash_table_key_from_gpu, (TypeHashValue *)hash_table_value_from_gpu,
        (T *)hash_table_key_from_cpu, (TypeHashValue *)hash_table_value_from_cpu);
    ASSERT_EQ(true, rtn);
    printf("Round %d end:\n", i);
  }

  // release resources
  free(embedding_feature_from_gpu);
  for (int i = 0; i < num_devices; i++) {
    free(wgrad_from_gpu[i]);
  }
  free(hash_table_value_from_gpu);
  free(hash_table_key_from_gpu);
}
#endif

#if 0
// sparse_embedding_hash performance profiling: forward()/backward()/update_params()
// 1. complie this app as release version
// 2. use nvprof / nvvp to run this app
TEST(distributed_sparse_embedding_hash_test, perf_profiling) {
  test::mpi_init();
  constexpr int batch_num = 10;  // can not more than 32
  // constexpr int batch_num = 1;
  constexpr int batchsize = 40960;
  // constexpr int batchsize = 2;
  constexpr long long num_records = batchsize * batch_num;
  // constexpr int slot_num = 1;
  constexpr int slot_num = 10;
  constexpr int max_feature_num = 10 * slot_num;
  // constexpr int max_feature_num = 2*slot_num;
  constexpr long long vocabulary_size = 55000;
  // constexpr long long vocabulary_size = 100;
  constexpr int embedding_vec_size = 128;
  // constexpr int embedding_vec_size = 1;
  constexpr int combiner = 0;   // 0-sum, 1-mean
  constexpr int optimizer = 0;  // 0-adam, 1-momentum_sgd, 2-nesterov
  constexpr float lr = 0.01;
  // std::vector<int> device_list = {0,1};
  std::vector<int> device_list = {0};
  int num_devices = device_list.size();
  constexpr long long label_dim = 1;
  typedef long long T;

  // In order to not allocate the total size of hash table on each GPU, the users need to set the
  // size of max_vocabulary_size_per_gpu, which should be more than vocabulary_size/gpu_count, eg:
  // (1/0.75)x of that.
  float load_factor =
      0.75;  // CAUSION: this is a very important param for hash_table get() performance
  // long long max_vocabulary_size_per_gpu = (long long)((double)(vocabulary_size) / num_devices /
  // load_factor);

  const char *hash_table_file_name = "distributed_hash_table.bin";
  const char *data_file_name = "temp_dataset_embedding.data";

  bool init_hash_table = true;  // true: init hash_table and upload_to_device
                                // false: don't init hash_table or upload_to_device, just use an
                                // empty hash_table to train

  // set up params
  OptHyperParams hyper_params;
  hyper_params.adam.beta1 = 0.9f;
  hyper_params.adam.beta2 = 0.999f;
  hyper_params.adam.epsilon = 1e-8f;
  hyper_params.momentum.factor = 0.9f;
  hyper_params.nesterov.mu = 0.9f;

  OptParams opt_params = {optimizer, lr, hyper_params};

  const SparseEmbeddingHashParams embedding_params = {
      batchsize, vocabulary_size, load_factor, embedding_vec_size, max_feature_num, slot_num,
      combiner,  // combiner: 0-sum, 1-mean
      opt_params};

  // generate input data
  const std::string tmp_file_name(data_file_name);
  const std::string file_list_name("file_list_embedding.txt");
  {
    // data generation;
    std::ofstream out_stream(tmp_file_name, std::ofstream::binary);
    DataSetHeader header = {num_records, label_dim, slot_num, 0};
    out_stream.write(reinterpret_cast<char *>(&header), sizeof(DataSetHeader));
    for (int i = 0; i < num_records; i++) {
      UnifiedDataSimulator<int> idata_sim(0, max_feature_num / slot_num - 1);  // both inclusive
      UnifiedDataSimulator<T> ldata_sim(0, vocabulary_size - 1);
      for (int j = 0; j < label_dim; j++) {
        int label = idata_sim.get_num();
        out_stream.write(reinterpret_cast<char *>(&label), sizeof(int));
      }
      for (int k = 0; k < slot_num; k++) {
        int nnz = idata_sim.get_num();
        // nnz = 10; // just for test 20181211
        nnz = (int)(max_feature_num / slot_num);  // just for test 20181221
        out_stream.write(reinterpret_cast<char *>(&nnz), sizeof(int));
        for (int j = 0; j < nnz; j++) {
          T value = ldata_sim.get_num();
          out_stream.write(reinterpret_cast<char *>(&value), sizeof(T));
        }
        // std::cout << std::endl; // just for test 20181211
      }
    }
    out_stream.close();
    std::ofstream file_list_stream(file_list_name, std::ofstream::out);
    file_list_stream << (std::to_string(1) + "\n");
    file_list_stream << (tmp_file_name + "\n");
    file_list_stream.close();
  }
  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back(device_list);
  std::shared_ptr<DeviceMap> device_map(new DeviceMap(vvgpu, 0));
  std::shared_ptr<GPUResourceGroup> gpu_resource_group(new GPUResourceGroup(device_map));

  // setup a data reader
  DataReader<T> *data_reader = new DataReader<T>(file_list_name, batchsize, label_dim, slot_num,
                                                 max_feature_num, gpu_resource_group, 1, 1);

  Embedding<T> *embedding = new DistributedSlotSparseEmbeddingHash<T>(data_reader->get_row_offsets_tensors(),
                                                       data_reader->get_value_tensors(),
                                                       embedding_params, gpu_resource_group);

  if (init_hash_table) {
    // init hash table file
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

    // upload hash table to device
    std::ifstream i_weight_stream(hash_table_file_name);
    embedding->upload_params_to_device(i_weight_stream);
    i_weight_stream.close();
  }

  struct timeval start;
  struct timeval end;
  unsigned long time = 0, time_forward = 0, time_backward = 0, time_update_params = 0;
  unsigned long total_time = 0;

  for (int i = 0; i < batch_num; i++) {
    nvtxRangePushA("read_a_batch");

    // call read a batch
    data_reader->read_a_batch_to_device();

    nvtxRangePop();

    nvtxRangePushA("forward");

    total_time = 0;
    gettimeofday(&start, NULL);

    // GPU forward
    embedding->forward();

    gettimeofday(&end, NULL);

    nvtxRangePop();

    time = 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec);
    printf("Round[%d]: forward time %lu us \n", i, time);
    time_forward += time;
    total_time += time;

    nvtxRangePushA("backward");

    gettimeofday(&start, NULL);

    // GPU backward
    embedding->backward();

    gettimeofday(&end, NULL);

    nvtxRangePop();

    time = 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec);
    printf("Round[%d]: backward time %lu us \n", i, time);
    time_backward += time;
    total_time += time;

    nvtxRangePushA("update_params");

    gettimeofday(&start, NULL);

    // GPU update_params
    embedding->update_params();

    gettimeofday(&end, NULL);

    nvtxRangePop();

    time = 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec);
    printf("Round[%d]: update_params time %lu us \n", i, time);
    total_time += time;
    time_update_params += time;

    printf("Round[%d]: total time %lu us \n", i, total_time);
  }

  printf("Average time of forward: %lu us\n", (unsigned long)(time_forward / batch_num));
  printf("Average time of backward: %lu us\n", (unsigned long)(time_backward / batch_num));
  printf("Average time of update_params: %lu us\n",
         (unsigned long)(time_update_params / batch_num));
  printf("Average time of total_time: %lu us\n",
         (unsigned long)((time_update_params + time_forward + time_backward) / batch_num));
}
#endif

TEST(localized_sparse_embedding_hash_test, reorder) {
  int local_gpu_count = 4; // 4,2 pass 
  int embedding_vec_size = 4;
  int batch_size = 16; // 8,16 pass 
  int samples_per_gpu = batch_size / local_gpu_count;
  int slot_num = 10; // 8,10 pass 
  int slots_per_sample = (slot_num + local_gpu_count - 1) / local_gpu_count; 
  int size_per_gpu = batch_size * slots_per_sample * embedding_vec_size;

  float * h_src, * d_src, * h_dst, * d_dst;
  cudaMallocHost(&h_src, size_per_gpu*sizeof(float));
  cudaMallocHost(&h_dst, size_per_gpu*sizeof(float));
  cudaMalloc(&d_src, size_per_gpu*sizeof(float));
  cudaMalloc(&d_dst, size_per_gpu*sizeof(float));

  int stride = samples_per_gpu * slots_per_sample * embedding_vec_size;
  for(int i = 0; i < samples_per_gpu; i++) {
    int offset = i * slots_per_sample * embedding_vec_size;
    for(int j = 0; j < slot_num; j++) {
      int addr = offset + (j/local_gpu_count) * embedding_vec_size + (j%local_gpu_count) * stride;
      //printf("sample_id=%d, slot_id=%d, addr=%d\n", i, j, addr);
      for(int k = 0; k < embedding_vec_size; k++) {
        h_src[addr+k] = (float)j;
      }
    }
  }

  for(int i = 0; i < batch_size; i++) {
    for(int j = 0; j < slots_per_sample; j++) {
      for(int k = 0; k < embedding_vec_size; k++) {
        int addr = i*slots_per_sample*embedding_vec_size+j*embedding_vec_size+k;
        //std::cout << "addr[" << addr << "]=" << h_src[addr] << ", ";
        std::cout << h_src[addr] << ", ";
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  dim3 blockSize(embedding_vec_size, 1, 1);
  dim3 gridSize(batch_size/local_gpu_count, 1, 1);

  cudaMemcpy(d_src, h_src, size_per_gpu * sizeof(float), cudaMemcpyHostToDevice);

  reorder_kernel<float><<<gridSize, blockSize>>>(batch_size,
                                                  slot_num,
                                                  embedding_vec_size,
                                                  local_gpu_count,
                                                  d_src,
                                                  d_dst);

  cudaMemcpy(h_dst, d_dst, size_per_gpu * sizeof(float), cudaMemcpyDeviceToHost);

  for(int i = 0; i < samples_per_gpu; i++) {
    std::cout << "sample " << i << ":" << std::endl;
    for(int j = 0; j < slot_num; j++) {
      for(int k = 0; k < embedding_vec_size; k++) {
        int addr = i*slot_num*embedding_vec_size+j*embedding_vec_size+k;
        std::cout << h_dst[addr] << ", ";
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;         

  // check results 
  bool results = true;
  for(int i = 0; i < samples_per_gpu; i++) {
    for(int j = 0; j < slot_num; j++) {
      for(int k = 0; k < embedding_vec_size; k++) {
        int addr = i*slot_num*embedding_vec_size+j*embedding_vec_size+k;
        if(!compare_float(h_dst[addr], float(j))) {
          results = false;
          j = slot_num;
          i = samples_per_gpu;
          break;
        }
      }
    }
  }

  ASSERT_EQ(results, true);
  
  cudaFreeHost(h_src);
  cudaFreeHost(h_dst);
  cudaFree(d_src);
  cudaFree(d_dst);
}

#if 0
TEST(localized_sparse_embedding_hash_test, all2all_reorder_single_node) {
  std::vector<int> device_list = {0,1,2,3}; // 4,8 gpus pass
  int local_gpu_count = device_list.size();
  int embedding_vec_size = 4;
  int batch_size = 8; // 8,16 pass
  int samples_per_gpu = batch_size / local_gpu_count;
  int slot_num = 10;  // 8,10 pass
  int slots_per_sample = (slot_num + local_gpu_count - 1) / local_gpu_count; 
  int size_per_gpu = batch_size * slots_per_sample * embedding_vec_size;

  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back(device_list);
  std::shared_ptr<DeviceMap> device_map(new DeviceMap(vvgpu, 0));
  std::shared_ptr<GPUResourceGroup> gpu_resource_group(new GPUResourceGroup(device_map));
  CudaDeviceContext context((*gpu_resource_group)[0]->get_device_id());
  
  SparseEmbeddingHashFunctors functors;

  std::vector<float *> h_src(local_gpu_count);
  std::vector<float *> h_mid(local_gpu_count);
  std::vector<float *> h_dst(local_gpu_count);
  for(int id = 0; id < local_gpu_count; id++) {
    cudaMallocHost(&h_src[id], size_per_gpu*sizeof(float));
    cudaMallocHost(&h_mid[id], size_per_gpu*sizeof(float));
    cudaMallocHost(&h_dst[id], size_per_gpu*sizeof(float));
  }

  Tensors<float> d_src;
  Tensors<float> d_mid;
  Tensors<float> d_dst;
  GeneralBuffers<float> buf;
  for (int i = 0; i < local_gpu_count; i++) {
    int cur_device = (*gpu_resource_group)[i]->get_device_id();
    context.set_device(cur_device);
    std::cout << "GPU " << cur_device << std::endl;

    buf.emplace_back(new GeneralBuffer<float>());

    std::vector<int> dims = {batch_size, slots_per_sample, embedding_vec_size};
    std::cout << "\tdims[" << dims[0] << " " << dims[1] << " " << dims[2] << "]" << std::endl;

    d_src.emplace_back(new Tensor<float>(dims, buf.back(), TensorFormat_t::HSW));
    d_mid.emplace_back(new Tensor<float>(dims, buf.back(), TensorFormat_t::HSW));
    d_dst.emplace_back(new Tensor<float>(dims, buf.back(), TensorFormat_t::HSW));

    buf.back()->init(cur_device);
    std::cout << "\tbuf size:" << buf.back()->get_size() << std::endl;
  }

  // init src
  for(int id = 0; id < local_gpu_count; id++) {
    for(int sample_id = 0; sample_id < batch_size; sample_id++) {
      for(int slot_id = 0; slot_id < slots_per_sample; slot_id++) {
        int index = sample_id * slots_per_sample + slot_id;
        int value = id + slot_id * local_gpu_count;
        if(value < slot_num) {
          for(int k = 0; k < embedding_vec_size; k++) {
            h_src[id][index * embedding_vec_size + k] = value;
          }
        }
      }
    }
  }

  // show src 
  for(int id = 0; id < local_gpu_count; id++) {
    std::cout << "gpu " << id << ": " << std::endl;
    for(int sample_id = 0; sample_id < batch_size; sample_id++) {
      std::cout << "\tsample " << sample_id << ": ";
      for(int slot_id = 0; slot_id < slots_per_sample; slot_id++) {
        int index = sample_id * slots_per_sample + slot_id;
        for(int k = 0; k < embedding_vec_size; k++) {
          std::cout << h_src[id][index * embedding_vec_size + k] << ", ";
        }
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  // memcpy from CPU to GPU
  std::cout << "memcpy from CPU to GPU:" << std::endl;
  for(int id = 0; id < local_gpu_count; id++) {
    int cur_device = (*gpu_resource_group)[id]->get_device_id();
    context.set_device(cur_device);

    cudaMemcpyAsync(d_src[id]->get_ptr(), h_src[id], size_per_gpu * sizeof(float), \
      cudaMemcpyHostToDevice, (*gpu_resource_group)[id]->get_stream());
  }
  functors.sync_all_gpus(gpu_resource_group, context);

  // all2all 
  using comm_handler_traits = FasterGossipComm::FasterGossipCommAll2AllTraits<float>;
  using comm_handler = FasterGossipComm::FasterGossipComm<float, comm_handler_traits>;
  std::unique_ptr<comm_handler> all2all;
  const std::string plan_file = "./bin/all2all_plan.json";
  const size_t element_per_send = samples_per_gpu * slots_per_sample * embedding_vec_size;
  std::cout << "all2all init" << std::endl;
  functors.all2all_init(all2all, plan_file, element_per_send, d_src, d_mid, gpu_resource_group);
  std::cout << "all2all async" << std::endl;
  functors.all2all_async(all2all);
  std::cout << "sync" << std::endl;
  functors.sync_all_gpus(gpu_resource_group, context);

  // check results of all2all
  for(int id = 0; id < local_gpu_count; id++) {
    int cur_device = (*gpu_resource_group)[id]->get_device_id();
    context.set_device(cur_device);

    cudaMemcpyAsync(h_mid[id], d_mid[id]->get_ptr(), size_per_gpu * sizeof(float), \
      cudaMemcpyDeviceToHost, (*gpu_resource_group)[id]->get_stream());
  }
  functors.sync_all_gpus(gpu_resource_group, context);
  for(int id = 0; id < local_gpu_count; id++) {
    std::cout << "gpu " << id << ": " << std::endl;
    for(int sample_id = 0; sample_id < batch_size; sample_id++) {
      std::cout << "\t";
      for(int slot_id = 0; slot_id < slots_per_sample; slot_id++) {
        int index = sample_id * slots_per_sample + slot_id;
        for(int k = 0; k < embedding_vec_size; k++) {
          std::cout << h_mid[id][index * embedding_vec_size + k] << ", ";
        }
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }   

  // reorder
  std::cout << "reorder" << std::endl;
  dim3 blockSize(embedding_vec_size, 1, 1);
  dim3 gridSize(batch_size/local_gpu_count, 1, 1);
  for(int id = 0; id < local_gpu_count; id++) {
    context.set_device((*gpu_resource_group)[id]->get_device_id());
    reorder_kernel<float><<<gridSize, blockSize, 0, (*gpu_resource_group)[id]->get_stream()>>>(batch_size,
                                                    slot_num,
                                                    embedding_vec_size,
                                                    local_gpu_count,
                                                    d_mid[id]->get_ptr(),
                                                    d_dst[id]->get_ptr());
  }

  // memcpy from GPU to CPU
  std::cout << "memcpy from GPU to CPU" << std::endl;
  for(int id = 0; id < local_gpu_count; id++) {
    int cur_device = (*gpu_resource_group)[id]->get_device_id();
    context.set_device(cur_device);

    cudaMemcpyAsync(h_dst[id], d_dst[id]->get_ptr(), size_per_gpu * sizeof(float), \
      cudaMemcpyDeviceToHost, (*gpu_resource_group)[id]->get_stream());
  }
  functors.sync_all_gpus(gpu_resource_group, context);

  // show dst
  for(int id = 0; id < local_gpu_count; id++) {
    std::cout << "gpu " << id << ": " << std::endl;
    for(int sample_id = 0; sample_id < samples_per_gpu; sample_id++) {
      std::cout << "\tsample " << id*samples_per_gpu+sample_id << ": ";
      for(int slot_id = 0; slot_id < slot_num; slot_id++) {
        int index = sample_id * slot_num + slot_id;
        for(int k = 0; k < embedding_vec_size; k++) {
          std::cout << h_dst[id][index * embedding_vec_size + k] << ", ";
        }
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }     

  // check results 
  bool results = true;
  for(int id = 0; id < local_gpu_count; id++) {
    for(int sample_id = 0; sample_id < samples_per_gpu; sample_id++) {
      for(int slot_id = 0; slot_id < slot_num; slot_id++) {
        int index = sample_id * slot_num + slot_id;
        for(int k = 0; k < embedding_vec_size; k++) {
          if(!compare_float(h_dst[id][index * embedding_vec_size + k], float(slot_id))) {
            results = false;
            id = local_gpu_count;
            sample_id = samples_per_gpu;
            slot_id = slot_num;
            break;
          }
        }
      }
    }
  } 

  ASSERT_EQ(results, true);
  
  for(int id = 0; id < local_gpu_count; id++) {
    cudaFreeHost(h_src[id]);
    cudaFreeHost(h_mid[id]);
    cudaFreeHost(h_dst[id]);
  }
}
#endif 

#if 1
// sparse_embedding_hash upload_params() and download_params() testing
TEST(localized_sparse_embedding_hash_test, upload_and_download_params) {
  test::mpi_init();
  
  // influential params for this test
  const long long vocabulary_size = 50010;
  //const long long vocabulary_size = 20;
  //const long long vocabulary_size = 1010;
  const int embedding_vec_size = 64;
  //const int embedding_vec_size = 1;
  std::vector<int> device_list = {0};
  //std::vector<int> device_list = {0,1};
  int num_devices = device_list.size();
  const char * hash_table_file_name = "localized_hash_table.bin";
  const char * hash_table_check_file_name = "localized_hash_table_check.bin";
  const std::string plan_file = "./bin/all2all_plan.json";
  
  // uninfluential params
  const int slot_num = 2;
  const int max_feature_num = 2*slot_num;
  const int batchsize = 2;
  const int batch_num = 1; // can not more than 32
  const long long num_records = batchsize * batch_num;
  const long long label_dim = 1;
  typedef long long T;

  // In order to not allocate the total size of hash table on each GPU, the users need to set the size of max_vocabulary_size_per_gpu,
	// which should be more than vocabulary_size/gpu_count, eg: (1/0.75)x of that.
  float load_factor = 0.75; // CAUSION: this is a very important param for hash_table get() performance
  //long long max_vocabulary_size_per_gpu = (long long)((double)(vocabulary_size) / num_devices / load_factor);

  const SparseEmbeddingHashParams embedding_params = {
  	batchsize,
    vocabulary_size,
    load_factor,
    embedding_vec_size,
    max_feature_num,
    slot_num,
    0,              //combiner: 0-sum, 1-mean
    0               //optimizer: 0-adam
  };

  // CUASION: the dataset will not be used in this test case, but we still need to let the data file non-empty since the DataRead requiring
  // generate input data
  const std::string tmp_file_name("temp_dataset_embedding.data");
  const std::string file_list_name("file_list_embedding.txt");
  {
    //data generation;
    std::ofstream out_stream(tmp_file_name, std::ofstream::binary);
    DataSetHeader header = {num_records, label_dim, slot_num, 0};
    out_stream.write(reinterpret_cast<char*>(&header), sizeof(DataSetHeader));
    for(int i=0; i<num_records; i++){
      UnifiedDataSimulator<int> idata_sim(0, max_feature_num/slot_num-1); //both inclusive
      UnifiedDataSimulator<T> ldata_sim(0,vocabulary_size-1);
      for(int j=0; j<label_dim; j++){
        int label = idata_sim.get_num();
        out_stream.write(reinterpret_cast<char*>(&label), sizeof(int));
      }
      for(int k=0; k<slot_num; k++){
        int nnz = idata_sim.get_num();
        //nnz = 10; // just for test 20181211
        nnz = (int)(max_feature_num/slot_num); // just for test 20181221
        out_stream.write(reinterpret_cast<char*>(&nnz), sizeof(int));
        for(int j=0; j<nnz; j++){
          T value = ldata_sim.get_num();
          out_stream.write(reinterpret_cast<char*>(&value), sizeof(T));
        }
        //std::cout << std::endl; // just for test 20181211
      }
    }
    out_stream.close();
    std::ofstream file_list_stream(file_list_name, std::ofstream::out);
    file_list_stream << (std::to_string(1) + "\n");
    file_list_stream << (tmp_file_name + "\n");
    file_list_stream.close();
  }

  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back(device_list);
  std::shared_ptr<DeviceMap> device_map(new DeviceMap(vvgpu, 0));
  std::shared_ptr<GPUResourceGroup> gpu_resource_group(new GPUResourceGroup(device_map));

  //setup a data reader
  DataReader<T>* data_reader = new DataReader<T>(file_list_name, batchsize, \
    label_dim, slot_num, max_feature_num, gpu_resource_group, 1, 1);

  // define object
  Embedding<T>* embedding = new LocalizedSlotSparseEmbeddingHash<T>(\
      data_reader->get_row_offsets_tensors(), data_reader->get_value_tensors(), \
      embedding_params, plan_file, gpu_resource_group);

  // init hash table file
  std::ofstream weight_stream(hash_table_file_name);
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
  }
  weight_stream.close();
  free(p_key);

  // upload data from host to device
  std::ifstream i_weight_stream(hash_table_file_name);
  printf("start updaload_params_to_device()\n");
  embedding->upload_params_to_device(i_weight_stream);
  i_weight_stream.close();

  // download data from device to host
  std::ofstream o_weight_stream(hash_table_check_file_name);
  printf("start download_params_to_host()\n");
  embedding->download_params_to_host(o_weight_stream);
  o_weight_stream.close();

  // comapre the read file with the written file
  typedef struct TypeHashValue_{
    T slot_id;
  	float data[embedding_vec_size];
  } TypeHashValue;
  //ASSERT_EQ(true, compare_localized_hash_table_files<T, TypeHashValue>(hash_table_file_name, hash_table_check_file_name));
  printf("start compare_localized_hash_table_files()\n");
  bool rtn = compare_localized_hash_table_files<T, TypeHashValue>(hash_table_file_name, hash_table_check_file_name);
  ASSERT_EQ(true, rtn);
}
#endif 

#if 1
// localized_sparse_embedding_hash correctness testing: forward->backward->update_params
TEST(localized_sparse_embedding_hash_test, training_correctness) {
  test::mpi_init();

  constexpr int batch_num = 4;  // can not more than 32
  // constexpr int batch_num = 1;
  constexpr int batchsize = 4096;
  // constexpr int batchsize = 2;
  constexpr long long num_records = batchsize * batch_num;
  // constexpr int slot_num = 1;
  constexpr int slot_num = 2;
  constexpr int max_feature_num = 10 * slot_num;  // max_feature_num in a sample
  // constexpr int max_feature_num = 2*slot_num;
  constexpr long long vocabulary_size = 55000;
  // constexpr long long vocabulary_size = 10;
  constexpr int embedding_vec_size = 128;
  // constexpr int embedding_vec_size = 1;
  constexpr int combiner = 0;   // 0-sum, 1-mean
  constexpr int optimizer = 0;  // 0-adam, 1-momentum_sgd, 2-nesterov
  constexpr float lr = 0.01;
  // std::vector<int> device_list = {0,1};
  std::vector<int> device_list = {0};
  int num_devices = device_list.size();
  constexpr long long label_dim = 1;
  typedef long long T;

  // In order to not allocate the total size of hash table on each GPU, the users need to set the
  // size of max_vocabulary_size_per_gpu, which should be more than vocabulary_size/gpu_count,
  // eg: 1.25x of that.
  float load_factor = 0.75;  // CAUSION: this is a very important param for performance
  // long long max_vocabulary_size_per_gpu = (long long)((double)(vocabulary_size) / num_devices /
  // load_factor);

  const char *hash_table_file_name = "localized_hash_table.bin";
  const char *data_file_name = "temp_dataset_embedding.data";
  const std::string plan_file = "./bin/all2all_plan.json";

  bool init_hash_table = true;  // true: init hash_table and upload_to_device
                                // false: don't init hash_table or upload_to_device, just use an
                                //        empty hash_table to train

  // set up params
  OptHyperParams hyper_params;
  hyper_params.adam.beta1 = 0.9f;
  hyper_params.adam.beta2 = 0.999f;
  hyper_params.adam.epsilon = 1e-8f;
  hyper_params.momentum.factor = 0.9f;
  hyper_params.nesterov.mu = 0.9f;

  OptParams opt_params = {optimizer, lr, hyper_params};

  const SparseEmbeddingHashParams embedding_params = {
      batchsize, vocabulary_size, load_factor, embedding_vec_size, max_feature_num, slot_num,
      combiner,  // combiner: 0-sum, 1-mean
      opt_params};

  // generate input data
  const std::string tmp_file_name(data_file_name);
  const std::string file_list_name("file_list_embedding.txt");
  {
    // data generation;
    std::ofstream out_stream(tmp_file_name, std::ofstream::binary);
    DataSetHeader header = {num_records, label_dim, slot_num, 0};
    out_stream.write(reinterpret_cast<char *>(&header), sizeof(DataSetHeader));
    for (int i = 0; i < num_records; i++) {
      UnifiedDataSimulator<int> idata_sim(0, max_feature_num / slot_num - 1);  // both inclusive
      UnifiedDataSimulator<T> ldata_sim(0, vocabulary_size - 1);
      for (int j = 0; j < label_dim; j++) {
        int label = idata_sim.get_num();
        out_stream.write(reinterpret_cast<char *>(&label), sizeof(int));
      }
      for (int k = 0; k < slot_num; k++) {
        int nnz = idata_sim.get_num();
        nnz = (int)(max_feature_num / slot_num);  // just for test
        out_stream.write(reinterpret_cast<char *>(&nnz), sizeof(int));
        for (int j = 0; j < nnz; j++) {
          T value = ldata_sim.get_num();
          // T value = k*nnz+j; // just for test, 20190625
          out_stream.write(reinterpret_cast<char *>(&value), sizeof(T));
        }
        // std::cout << std::endl; // just for test 20181211
      }
    }
    out_stream.close();
    std::ofstream file_list_stream(file_list_name, std::ofstream::out);
    file_list_stream << (std::to_string(1) + "\n");
    file_list_stream << (tmp_file_name + "\n");
    file_list_stream.close();
  }

  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back(device_list);
  std::shared_ptr<DeviceMap> device_map(new DeviceMap(vvgpu, 0));
  std::shared_ptr<GPUResourceGroup> gpu_resource_group(new GPUResourceGroup(device_map));

  // setup a data reader
  DataReader<T> *data_reader = new DataReader<T>(file_list_name, batchsize, label_dim, slot_num,
                                                 max_feature_num, gpu_resource_group, 1, 1);

  Embedding<T> *embedding = new LocalizedSlotSparseEmbeddingHash<T>(data_reader->get_row_offsets_tensors(),
                                                       data_reader->get_value_tensors(),
                                                       embedding_params, plan_file, gpu_resource_group);

  if (init_hash_table) {
    // init hash table file: <key, solt_id, value>
    std::ofstream weight_stream(hash_table_file_name);
    if (!weight_stream.is_open()) {
      ERROR_MESSAGE_("Error: file not open for writing");
    }
    UnifiedDataSimulator<T> ldata_sim(0, slot_num-1); // for slot_id
    UnifiedDataSimulator<float> fdata_sim(-0.1f, 0.1f); // for value
    for (long long i = 0; i < vocabulary_size; i++) {
      T key = (T)i;
      // T key = ldata_sim.get_num();
      // CAUSION: can not set random keys here, because we need to ensure that:
      // 1) we can find keys in the data file from this hash table
      // 2) there are no repeated keys
      weight_stream.write((char *)&key, sizeof(T));
      T slot_id = ldata_sim.get_num();
      weight_stream.write((char *)&slot_id, sizeof(T));
      // float val = (float)i;
      // float val = 1.0f;
      float val = fdata_sim.get_num();
      for (int j = 0; j < embedding_vec_size; j++) {
        weight_stream.write((char *)&val, sizeof(float));
      }
    }
    weight_stream.close();

    // upload hash table to device
    std::ifstream i_weight_stream(hash_table_file_name);
    embedding->upload_params_to_device(i_weight_stream);
    i_weight_stream.close();
  }

  // for SparseEmbeddingCpu
  std::ifstream weight_stream_cpu(hash_table_file_name);
  std::ifstream csr_stream_cpu(data_file_name);
  SparseEmbeddingHashCpu<T> *embedding_cpu = new SparseEmbeddingHashCpu<T>(
      batchsize, max_feature_num, vocabulary_size, embedding_vec_size, slot_num, combiner,
      optimizer, lr, weight_stream_cpu, csr_stream_cpu, label_dim);

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
    printf("Round %d start:\n", i);

    // call read a batch
    data_reader->read_a_batch_to_device();

    printf("finish data_reader->read_a_batch_to_device()\n");

    // GPU forward
    embedding->forward();

    printf("finish finish embedding->forward()\n");

    // CPU forward
    embedding_cpu->forward();

    printf("finish embedding_cpu->forward()\n");

    // check the result of forward
    embedding->get_forward_results(embedding_feature_from_gpu);  // memcpy from GPU to CPU
    ASSERT_EQ(true,
              compare_embedding_feature(batchsize * slot_num * embedding_vec_size,
                                        embedding_feature_from_gpu, embedding_feature_from_cpu));

    printf("finish embedding->get_forward_results()\n");

    // GPU backward
    embedding->backward();

    printf("finish embedding->backward()\n");

    // CPU backward
    embedding_cpu->backward();

    printf("finish embedding_cpu->backward()\n");

    // check the result of backward
    embedding->get_backward_results(wgrad_from_gpu[0], 0);

    printf("finish embedding->get_backward_results()\n");

    // check the result on multi GPUs first
    if (device_list.size() > 1) {
      for (unsigned int j = 1; j < device_list.size(); j++) {
        embedding->get_backward_results(wgrad_from_gpu[j], j);  // memcpy from GPU to CPU
        // printf("\ncompare GPU[%d] and GPU[%d]\n", 0, j);
        ASSERT_EQ(true, compare_wgrad(batchsize * slot_num * embedding_vec_size, wgrad_from_gpu[0],
                                      wgrad_from_gpu[j]));
      }
    }
    // printf("\ncompare GPU0 and CPU\n");
    ASSERT_EQ(true, compare_wgrad(batchsize * slot_num * embedding_vec_size, wgrad_from_gpu[0],
                                  wgrad_from_cpu));

    // GPU update_params
    embedding->update_params();

    printf("finish embedding->update_params()\n");

    // CPU update_params
    embedding_cpu->update_params();

    printf("finish embedding_cpu->update_params()\n");

    // check the results of update params
    embedding->get_update_params_results(hash_table_key_from_gpu,
                                  hash_table_value_from_gpu);  // memcpy from GPU to CPU
    
    printf("finish embedding->get_update_params_results()\n");

    // ASSERT_EQ(true, compare_embedding_table(vocabulary_size*embedding_vec_size,
    // hash_table_value_from_gpu, hash_table_value_from_cpu));
    bool rtn = compare_hash_table<T, TypeHashValue>(
        vocabulary_size, (T *)hash_table_key_from_gpu, (TypeHashValue *)hash_table_value_from_gpu,
        (T *)hash_table_key_from_cpu, (TypeHashValue *)hash_table_value_from_cpu);
    ASSERT_EQ(true, rtn);
    printf("Round %d end:\n", i);
  }

  // release resources
  free(embedding_feature_from_gpu);
  for (int i = 0; i < num_devices; i++) {
    free(wgrad_from_gpu[i]);
  }
  free(hash_table_value_from_gpu);
  free(hash_table_key_from_gpu);
}
#endif