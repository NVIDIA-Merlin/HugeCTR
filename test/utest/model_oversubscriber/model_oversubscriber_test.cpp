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

#include "utest/model_oversubscriber/mos_test_utils.hpp"
#include "HugeCTR/include/data_generator.hpp"
#include "HugeCTR/include/data_readers/data_reader.hpp"
#include "HugeCTR/include/model_oversubscriber/model_oversubscriber.hpp"
#include "HugeCTR/include/parser.hpp"
#include "HugeCTR/include/utils.hpp"
#include "gtest/gtest.h"

using namespace HugeCTR;
using namespace mos_test;

namespace {

const char* prefix = "./model_oversubscriber_test_data/tmp_";
const char* file_list_name_train = "file_list_train.txt";
const char* file_list_name_eval = "file_list_eval.txt";
const char* snapshot_src_file = "distributed_snapshot_src";
const char* snapshot_dst_file = "distributed_snapshot_dst";
const char* keyset_file_name = "keyset_file.bin";

const int batchsize = 4096;
const long long label_dim = 1;
const long long dense_dim = 0;
const int slot_num = 128;
const int max_nnz_per_slot = 1;
const int max_feature_num = max_nnz_per_slot * slot_num;
const long long vocabulary_size = 100000;
const int combiner = 0;
const float scaler = 1.0f;
const int num_workers = 1;
const int num_files = 1;

const Check_t check = Check_t::Sum;
const Update_t update_type = Update_t::Local;

// const int batch_num_train = 10;
const int batch_num_eval = 1;

template <typename KeyType, typename EmbeddingCompType>
void do_upload_and_download_snapshot(size_t batch_num_train, size_t embedding_vector_size,
    Embedding_t embedding_type) {
  // create a resource manager for a single GPU
  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back({0});

  const auto resource_manager = ResourceManager::create(vvgpu, 0);

  // generate train/test datasets
  if (fs::exists(file_list_name_train)) {
    fs::remove(file_list_name_train);
  }

  if (fs::exists(file_list_name_eval)) {
    fs::remove(file_list_name_eval);
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
      batchsize, label_dim, dense_dim, data_reader_params, resource_manager, true, num_workers, false, 0));
  std::unique_ptr<DataReader<KeyType>> data_reader_eval(new DataReader<KeyType>(
      batchsize, label_dim, dense_dim, data_reader_params, resource_manager, true, num_workers, false, 0));

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

  std::shared_ptr<Embedding<KeyType, EmbeddingCompType>> embedding = init_embedding(
          data_reader_train->get_row_offsets_tensors(), data_reader_train->get_value_tensors(),
          data_reader_train->get_nnz_array(), data_reader_eval->get_row_offsets_tensors(),
          data_reader_eval->get_value_tensors(), data_reader_eval->get_nnz_array(),
          embedding_param, resource_manager, embedding_type);
  embedding->init_params();

  // train the embedding
  data_reader_train->read_a_batch_to_device();
  embedding->forward(true);
  embedding->backward();
  embedding->update_params();

  // store the snapshot from the embedding
  embedding->dump_parameters(snapshot_src_file);
  copy_sparse_model(snapshot_src_file, snapshot_dst_file);

  {
    // Create a ParameterServer
    ParameterServer<KeyType, EmbeddingCompType> parameter_server(embedding_param,
        snapshot_src_file, embedding_type);

    // Make a synthetic keyset files
    auto keys = parameter_server.get_keys_from_hash_table();
    std::ofstream key_stream(keyset_file_name, std::ofstream::binary | std::ofstream::trunc);
    key_stream.write(reinterpret_cast<char *>(keys.data()), keys.size() * sizeof(KeyType));
  }

  std::vector<std::string> keyset_file_list;
  keyset_file_list.push_back(keyset_file_name);
  std::vector<std::string> snapshot_file_list;
  snapshot_file_list.push_back(snapshot_dst_file);

  // Create a ModelOversubscriber
  std::vector<std::shared_ptr<IEmbedding>> embeddings;
  embeddings.push_back(embedding);
  std::vector<SparseEmbeddingHashParams<EmbeddingCompType>> embedding_params;

  Solver solver;
  std::vector<std::string> sparse_embedding_files;
  sparse_embedding_files.push_back(snapshot_dst_file);
  solver.i64_input_key = true;
  solver.enable_tf32_compute = false;

  std::shared_ptr<ModelOversubscriber> model_oversubscriber(
      new ModelOversubscriber(embeddings, embedding_params, sparse_embedding_files, solver));

  Timer timer_ps;
  timer_ps.start();

  // upload embedding table from disk according to keyset
  model_oversubscriber->update(keyset_file_list);
  model_oversubscriber->store();

  MESSAGE_("Batch_num: " + std::to_string(batch_num_train) + ", embedding_vec_size: " +
           std::to_string(embedding_vector_size) + ", elapsed time: " +
           std::to_string(timer_ps.elapsedSeconds()) + "s");

  // Check if the result is correct
  ASSERT_TRUE(check_vector_equality(snapshot_src_file, snapshot_dst_file, "key"));
  ASSERT_TRUE(check_vector_equality(snapshot_src_file, snapshot_dst_file, "vec"));
  if (embedding_type != Embedding_t::DistributedSlotSparseEmbeddingHash) {
    ASSERT_TRUE(check_vector_equality(snapshot_src_file, snapshot_dst_file, "slot"));
  }
}

TEST(model_oversubscriber_distributed_embedding_test, long_long_float) {
  const Embedding_t dis_embedding = Embedding_t::DistributedSlotSparseEmbeddingHash;
  do_upload_and_download_snapshot<long long, float>(10, 128, dis_embedding);
}

TEST(model_oversubscriber_localized_embedding_test, long_long_float) {
  const Embedding_t loc_embedding = Embedding_t::LocalizedSlotSparseEmbeddingHash;
  do_upload_and_download_snapshot<long long, float>(20, 64, loc_embedding);
}

TEST(model_oversubscriber_localized_embedding_one_hot_test, long_long_float) {
  const Embedding_t loc_oh_embedding = Embedding_t::LocalizedSlotSparseEmbeddingOneHot;
  do_upload_and_download_snapshot<long long, float>(30, 32, loc_oh_embedding);
}

}  // namespace
