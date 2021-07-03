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

#pragma once

#include "HugeCTR/include/utils.hpp"
#include "HugeCTR/include/data_generator.hpp"
#include "HugeCTR/include/data_readers/data_reader.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include "utest/embedding/embedding_test_utils.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"
#include "utest/test_utils.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

namespace HugeCTR {

namespace mos_test {

inline std::vector<char> load_to_vector(const std::string& file_name) {
  std::ifstream stream(file_name, std::ifstream::binary);
  if (!stream.is_open()) {
    CK_THROW_(Error_t::FileCannotOpen, "Can not open " + file_name);
  }

  size_t file_size_in_byte = fs::file_size(file_name);

  std::vector<char> vec(file_size_in_byte);
  stream.read(vec.data(), file_size_in_byte);
  return vec;
}

template <typename TypeKey, typename EmbeddingCompType>
std::shared_ptr<Embedding<TypeKey, EmbeddingCompType>> init_embedding(
    const Tensors2<TypeKey> &train_row_offsets_tensors,
    const Tensors2<TypeKey> &train_value_tensors,
    const std::vector<std::shared_ptr<size_t>> &train_nnz_array,
    const Tensors2<TypeKey> &evaluate_row_offsets_tensors,
    const Tensors2<TypeKey> &evaluate_value_tensors,
    const std::vector<std::shared_ptr<size_t>> &evaluate_nnz_array,
    const SparseEmbeddingHashParams &embedding_params,
    const std::shared_ptr<ResourceManager> &resource_manager,
    const Embedding_t embedding_type) {
  if (embedding_type == Embedding_t::DistributedSlotSparseEmbeddingHash) {
    std::shared_ptr<Embedding<TypeKey, EmbeddingCompType>> embedding(
        new DistributedSlotSparseEmbeddingHash<TypeKey, EmbeddingCompType>(
            train_row_offsets_tensors, train_value_tensors, train_nnz_array,
            evaluate_row_offsets_tensors, evaluate_value_tensors,
            evaluate_nnz_array, embedding_params, resource_manager));
    return embedding;
  } else {
    std::shared_ptr<Embedding<TypeKey, EmbeddingCompType>> embedding(
        new LocalizedSlotSparseEmbeddingHash<TypeKey, EmbeddingCompType>(
            train_row_offsets_tensors, train_value_tensors, train_nnz_array,
            evaluate_row_offsets_tensors, evaluate_value_tensors,
            evaluate_nnz_array, embedding_params, resource_manager));
    return embedding;
  }
}

inline void copy_sparse_model(
    std::string sparse_model_src, std::string sparse_model_dst) {
  if (fs::exists(sparse_model_dst)) {
    fs::remove_all(sparse_model_dst);
  }
  fs::copy(sparse_model_src, sparse_model_dst, fs::copy_options::recursive);
  std::string src_prefix = std::string(sparse_model_dst) + "/" + sparse_model_src;
  std::string dst_prefix = std::string(sparse_model_dst) + "/" + sparse_model_dst;
  fs::rename((src_prefix + ".key").c_str(), (dst_prefix + ".key").c_str());
  fs::rename((src_prefix + ".vec").c_str(), (dst_prefix + ".vec").c_str());

  if (fs::exists(src_prefix + ".slot")) {
    fs::rename((src_prefix + ".slot").c_str(), (dst_prefix + ".slot").c_str());
  }
}

inline bool check_vector_equality(const char *sparse_model_src,
    const char *sparse_model_dst, const char *ext) {
  const std::string src_file(
      std::string(sparse_model_src) + "/" + sparse_model_src + "." + ext);
  const std::string dst_file(
      std::string(sparse_model_dst) + "/" + sparse_model_dst + "." + ext);

  std::vector<char> vec_src = load_to_vector(src_file);
  std::vector<char> vec_dst = load_to_vector(dst_file);

  size_t len_src = vec_src.size();
  size_t len_dst = vec_src.size();

  if (len_src != len_dst) {
    return false;
  }
  bool flag = test::compare_array_approx<char>(vec_src.data(), vec_dst.data(), len_src, 0);
  return flag;
}

template <typename TypeKey, Check_t check>
inline void generate_sparse_model_impl(std::string sparse_model_name,
    std::string file_list_name_train, std::string file_list_name_eval,
    std::string prefix, int num_files, long long label_dim, long long dense_dim,
    size_t slot_num, int max_nnz_per_slot, size_t max_feature_num,
    size_t vocabulary_size, size_t emb_vec_size, int combiner, float scaler,
    int num_workers, size_t batchsize, int batch_num_train, int batch_num_eval,
    Update_t update_type, std::shared_ptr<ResourceManager> resource_manager) {
  // generate train/test datasets
  if (fs::exists(file_list_name_train)) {
    fs::remove(file_list_name_train);
  }

  if (fs::exists(file_list_name_eval)) {
    fs::remove(file_list_name_eval);
  }

  // data generation
  HugeCTR::data_generation_for_test<TypeKey, check>(file_list_name_train,
      prefix, num_files, batch_num_train * batchsize, slot_num,
      vocabulary_size, label_dim, dense_dim, max_nnz_per_slot);
  HugeCTR::data_generation_for_test<TypeKey, check>(file_list_name_eval,
      prefix, num_files, batch_num_eval * batchsize, slot_num,
      vocabulary_size, label_dim, dense_dim, max_nnz_per_slot);

  // create train/eval data readers
  const DataReaderSparseParam param = {
      DataReaderSparse_t::Distributed,
      static_cast<int>(max_feature_num),
      max_nnz_per_slot,
      static_cast<int>(slot_num)
  };
  std::vector<DataReaderSparseParam> data_reader_params;
  data_reader_params.push_back(param);

  std::unique_ptr<DataReader<TypeKey>> data_reader_train(
      new DataReader<TypeKey>(batchsize, label_dim, dense_dim,
          data_reader_params, resource_manager, true, num_workers, false, 0));
  std::unique_ptr<DataReader<TypeKey>> data_reader_eval(
      new DataReader<TypeKey>(batchsize, label_dim, dense_dim,
          data_reader_params, resource_manager, true, num_workers, false, 0));

  data_reader_train->create_drwg_norm(file_list_name_train, check);
  data_reader_eval->create_drwg_norm(file_list_name_eval, check);

  Embedding_t embedding_type = Embedding_t::LocalizedSlotSparseEmbeddingHash;

  // create an embedding
  OptHyperParams hyper_params;
  hyper_params.adam.beta1 = 0.9f;
  hyper_params.adam.beta2 = 0.999f;
  hyper_params.adam.epsilon = 1e-7f;
  hyper_params.momentum.factor = 0.9f;
  hyper_params.nesterov.mu = 0.9f;

  const OptParams opt_params = {Optimizer_t::Adam, 0.001f, hyper_params,
                                                   update_type, scaler};

  const SparseEmbeddingHashParams embedding_params = {
      batchsize,       batchsize, vocabulary_size, {},         emb_vec_size,
      max_feature_num, slot_num,  combiner,        opt_params};

  auto embedding = init_embedding<TypeKey, float>(
      data_reader_train->get_row_offsets_tensors(),
      data_reader_train->get_value_tensors(),
      data_reader_train->get_nnz_array(),
      data_reader_eval->get_row_offsets_tensors(),
      data_reader_eval->get_value_tensors(),
      data_reader_eval->get_nnz_array(),
      embedding_params, resource_manager, embedding_type);
  embedding->init_params();

  // train the embedding
  data_reader_train->read_a_batch_to_device();
  embedding->forward(true);
  embedding->backward();
  embedding->update_params();

  // store the snapshot from the embedding
  embedding->dump_parameters(sparse_model_name);
}

template <typename TypeKey, Check_t check>
inline void generate_sparse_model(
    std::string snapshot_src_file,  std::string snapshot_dst_file,
    std::string sparse_model_unsigned, std::string sparse_model_longlong,
    std::string file_list_name_train, std::string file_list_name_eval,
    std::string prefix, int num_files, long long label_dim, long long dense_dim,
    size_t slot_num, int max_nnz_per_slot, size_t max_feature_num,
    size_t vocabulary_size, size_t emb_vec_size, int combiner, float scaler,
    int num_workers, size_t batchsize, int batch_num_train, int batch_num_eval,
    Update_t update_type, std::shared_ptr<ResourceManager> resource_manager) {
  if (std::is_same<TypeKey, unsigned>::value) {
    if (fs::exists(sparse_model_unsigned)) {
      if (fs::exists(snapshot_src_file)) fs::remove_all(snapshot_src_file);
      if (fs::exists(snapshot_dst_file)) fs::remove_all(snapshot_dst_file);
      copy_sparse_model(sparse_model_unsigned, snapshot_src_file);
    } else {
      generate_sparse_model_impl<unsigned, check>(snapshot_src_file,
          file_list_name_train, file_list_name_eval, prefix, num_files, label_dim,
          dense_dim, slot_num, max_nnz_per_slot, max_feature_num,
          vocabulary_size, emb_vec_size, combiner, scaler, num_workers, batchsize,
          batch_num_train, batch_num_eval, update_type, resource_manager);
      copy_sparse_model(snapshot_src_file, sparse_model_unsigned);
    }
  } else {
    if (fs::exists(sparse_model_longlong)) {
      if (fs::exists(snapshot_src_file)) fs::remove_all(snapshot_src_file);
      if (fs::exists(snapshot_dst_file)) fs::remove_all(snapshot_dst_file);
      copy_sparse_model(sparse_model_longlong, snapshot_src_file);
    } else {
      generate_sparse_model_impl<long long, check>(snapshot_src_file,
          file_list_name_train, file_list_name_eval, prefix, num_files, label_dim,
          dense_dim, slot_num, max_nnz_per_slot, max_feature_num,
          vocabulary_size, emb_vec_size, combiner, scaler, num_workers, batchsize,
          batch_num_train, batch_num_eval, update_type, resource_manager);
      copy_sparse_model(snapshot_src_file, sparse_model_longlong);
    }
  }
}

}

}