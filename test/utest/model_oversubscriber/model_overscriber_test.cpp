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

#include "HugeCTR/include/model_oversubscriber/model_oversubscriber.hpp"
#include "HugeCTR/include/data_generator.hpp"
#include "HugeCTR/include/data_readers/data_reader.hpp"
#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/parser.hpp"
#include "HugeCTR/include/utils.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

#include <fstream>
#include <set>

using namespace HugeCTR;

namespace {

const char* prefix = "./model_oversubscriber_test_data/tmp_";
const char* file_list_name_train = "file_list_train.txt";
const char* file_list_name_eval = "file_list_eval.txt";
const char* snapshot_src_file = "distributed_snapshot_src.bin";
const char* snapshot_dst_file = "distributed_snapshot_dst.bin";
const char* keyset_file_name_postfix = "_keyset_file.bin";
const char* temp_embedding_dir = "./";

const int batchsize = 4096;
const long long label_dim = 1;
const long long dense_dim = 0;
// const int embedding_vector_size = 128;
const int slot_num = 128;
const int max_nnz_per_slot = 1;
const int max_feature_num = max_nnz_per_slot * slot_num;
const long long vocabulary_size = 100000;
const int combiner = 0;
const float scaler = 1.0f;
const int num_workers = 1;
const int num_files = 1;

const Check_t check = Check_t::Sum;

// const int batch_num_train = 10;
const int batch_num_eval = 1;
const size_t pass_size = 2;
// const size_t num_total_passes = batch_num_train / pass_size;

const Update_t update_type = Update_t::Local;

template <typename KeyType>
void create_keyset_files(std::vector<KeyType> keys, size_t batch_num_train,
                         size_t num_total_passes) {
  // just get a handful of keys to test the functions
  std::vector<std::ofstream> keyset_files;
  for (size_t p = 0; p < num_total_passes; p++) {
    std::string keyset_file_name = std::to_string(p) + keyset_file_name_postfix;
    keyset_files.emplace_back(std::ofstream(keyset_file_name));
  }

  for (size_t k = 0; k < batch_num_train; k++) {
    size_t kfi = (k / pass_size) % num_total_passes;
    keyset_files[kfi].write((char*)(keys.data() + k), sizeof(KeyType));
  }
}

std::vector<char> load_to_vector(std::ifstream& stream) {
  stream.seekg(0, stream.end);
  size_t file_size_in_byte = stream.tellg();
  stream.seekg(0, stream.beg);

  std::vector<char> vec(file_size_in_byte);
  stream.read(vec.data(), file_size_in_byte);
  return vec;
}

template <typename KeyType, typename EmbeddingCompType>
void do_upload_and_download_snapshot(size_t batch_num_train, size_t embedding_vector_size) {
  const size_t num_total_passes = batch_num_train / pass_size;
  // create a resource manager for a single GPU
  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back({0});

  const auto resource_manager = ResourceManager::create(vvgpu, 0);

  // generate train/test datasets
  {
    std::ifstream fs(file_list_name_train);
    if (fs.good()) {
      std::remove(file_list_name_train);
    }
  }
  {
    std::ifstream fs(file_list_name_eval);
    if (fs.good()) {
      std::remove(file_list_name_eval);
    }
  }
  // data generation
  HugeCTR::data_generation_for_test<KeyType, check>(
      file_list_name_train, prefix, num_files, batch_num_train * batchsize, slot_num,
      vocabulary_size, label_dim, dense_dim, max_nnz_per_slot);
  HugeCTR::data_generation_for_test<KeyType, check>(
      file_list_name_eval, prefix, num_files, batch_num_eval * batchsize, slot_num, vocabulary_size,
      label_dim, dense_dim, max_nnz_per_slot);

  // create train/eval data readers
  const DataReaderSparseParam param = {DataReaderSparse_t::Distributed, max_feature_num,
                                       max_nnz_per_slot, slot_num};
  std::vector<DataReaderSparseParam> data_reader_params;
  data_reader_params.push_back(param);

  std::unique_ptr<DataReader<KeyType>> data_reader_train(new DataReader<KeyType>(
      batchsize, label_dim, dense_dim, data_reader_params, resource_manager, num_workers));
  std::unique_ptr<DataReader<KeyType>> data_reader_eval(new DataReader<KeyType>(
      batchsize, label_dim, dense_dim, data_reader_params, resource_manager, num_workers));

  data_reader_train->create_drwg_norm(file_list_name_train, check);
  data_reader_eval->create_drwg_norm(file_list_name_eval, check);

  // create an embedding
  OptHyperParams<EmbeddingCompType> hyper_params;
  hyper_params.adam.beta1 = 0.9f;
  hyper_params.adam.beta2 = 0.999f;
  if (std::is_same<EmbeddingCompType, __half>::value) {
    hyper_params.adam.epsilon = 1e-4f;
  } else {
    hyper_params.adam.epsilon = 1e-7f;
  }
  hyper_params.momentum.factor = 0.9f;
  hyper_params.nesterov.mu = 0.9f;

  const OptParams<EmbeddingCompType> opt_params = {Optimizer_t::Adam, 0.001f, hyper_params,
                                                   update_type, scaler};

  const SparseEmbeddingHashParams<EmbeddingCompType> embedding_param = {
      batchsize,       batchsize, vocabulary_size, {},        embedding_vector_size,
      max_feature_num, slot_num,  combiner,        opt_params};

  std::shared_ptr<IEmbedding> embedding(
      new DistributedSlotSparseEmbeddingHash<KeyType, EmbeddingCompType>(
          data_reader_train->get_row_offsets_tensors(), data_reader_train->get_value_tensors(),
          data_reader_train->get_nnz_array(), data_reader_eval->get_row_offsets_tensors(),
          data_reader_eval->get_value_tensors(), data_reader_eval->get_nnz_array(), embedding_param,
          resource_manager));
  embedding->init_params();

  // train the embedding
  data_reader_train->read_a_batch_to_device();
  embedding->forward(true);
  embedding->backward();
  embedding->update_params();

  // store the snapshot from the embedding
  {
    std::ofstream fs(snapshot_src_file, std::ofstream::binary);
    embedding->dump_parameters(fs);
  }

  // Create a ParameterServer
  ParameterServer<KeyType, EmbeddingCompType> parameter_server(embedding_param, snapshot_src_file,
                                                               temp_embedding_dir);

  SolverParser solver_config;
  solver_config.embedding_files.push_back(snapshot_src_file);
  solver_config.i64_input_key = true;
  solver_config.enable_tf32_compute = false;

  // Create a ModelOversubscriber
  std::vector<std::shared_ptr<IEmbedding>> embeddings;
  embeddings.push_back(embedding);
  std::vector<SparseEmbeddingHashParams<EmbeddingCompType>> embedding_params;
  std::string temp_dir = temp_embedding_dir;

  std::shared_ptr<ModelOversubscriber> model_oversubscriber(
      new ModelOversubscriber(embeddings, embedding_params, solver_config, temp_dir));

  // Make a synthetic keyset files
  auto keys = parameter_server.get_keys_from_hash_table();
  create_keyset_files<KeyType>(keys, batch_num_train, num_total_passes);

  std::string keyset_file_name = std::to_string(0) + keyset_file_name_postfix;
  std::vector<std::string> keyset_file_list;
  keyset_file_list.push_back(keyset_file_name);
  std::vector<std::string> snapshot_file_list;
  snapshot_file_list.push_back(snapshot_dst_file);

  Timer timer_ps;
  timer_ps.start();

  model_oversubscriber->update(keyset_file_list);
  // transfer the internal embedding table to the snapshot
  model_oversubscriber->store(snapshot_file_list);

  MESSAGE_("Batch_num: " + std::to_string(batch_num_train) + ", embedding_vec_size: " +
           std::to_string(embedding_vector_size) + ", elapsed time: " +
           std::to_string(timer_ps.elapsedSeconds()) + "s");

  // Check if the result is correct
  std::ifstream fs_src(snapshot_src_file);
  std::ifstream fs_dst(snapshot_dst_file);

  std::vector<char> vec_src = load_to_vector(fs_src);
  std::vector<char> vec_dst = load_to_vector(fs_dst);

  size_t len_src = vec_src.size();
  size_t len_dst = vec_src.size();
  ASSERT_EQ(true, len_src == len_dst);
  ASSERT_TRUE(test::compare_array_approx<char>(vec_src.data(), vec_dst.data(), len_src, 0));
}

TEST(model_oversubscriber_test, long_long_float) {
  do_upload_and_download_snapshot<long long, float>(10, 64);
}

}  // namespace
