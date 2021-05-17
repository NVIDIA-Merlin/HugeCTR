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

#pragma once

#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include "utest/test_utils.h"

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

template <typename KeyType, typename EmbeddingCompType>
std::unique_ptr<Embedding<KeyType, EmbeddingCompType>> init_embedding(
    const Tensors2<KeyType> &train_row_offsets_tensors,
    const Tensors2<KeyType> &train_value_tensors,
    const std::vector<std::shared_ptr<size_t>> &train_nnz_array,
    const Tensors2<KeyType> &evaluate_row_offsets_tensors,
    const Tensors2<KeyType> &evaluate_value_tensors,
    const std::vector<std::shared_ptr<size_t>> &evaluate_nnz_array,
    const SparseEmbeddingHashParams &embedding_params,
    const std::shared_ptr<ResourceManager> &resource_manager,
    const Embedding_t embedding_type) {
  if (embedding_type == Embedding_t::DistributedSlotSparseEmbeddingHash) {
    std::unique_ptr<Embedding<KeyType, EmbeddingCompType>> embedding(
        new DistributedSlotSparseEmbeddingHash<KeyType, EmbeddingCompType>(
            train_row_offsets_tensors, train_value_tensors, train_nnz_array,
            evaluate_row_offsets_tensors, evaluate_value_tensors, evaluate_nnz_array,
            embedding_params, resource_manager));
    return embedding;
  } else {
    std::unique_ptr<Embedding<KeyType, EmbeddingCompType>> embedding(
        new LocalizedSlotSparseEmbeddingHash<KeyType, EmbeddingCompType>(
            train_row_offsets_tensors, train_value_tensors, train_nnz_array,
            evaluate_row_offsets_tensors, evaluate_value_tensors, evaluate_nnz_array,
            embedding_params, resource_manager));
    return embedding;
  }
}

inline void copy_sparse_model(std::string sparse_model_src, std::string sparse_model_dst) {
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

inline bool check_vector_equality(const char *sparse_model_src, const char *sparse_model_dst, const char *ext) {
  const std::string src_file(std::string(sparse_model_src) + "/" + sparse_model_src + "." + ext);
  const std::string dst_file(std::string(sparse_model_dst) + "/" + sparse_model_dst + "." + ext);

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

}

}