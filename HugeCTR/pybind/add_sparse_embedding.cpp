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

#include <HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp>
#include <HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp>
#include <HugeCTR/include/embeddings/localized_slot_sparse_embedding_one_hot.hpp>
#include <loss.hpp>
#include <optimizer.hpp>
#include <HugeCTR/pybind/model.hpp>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {

template <typename TypeKey, typename TypeFP>
void add_sparse_embedding(SparseEmbedding& sparse_embedding,
            std::map<std::string, SparseInput<TypeKey>>& sparse_input_map,
            std::vector<std::vector<TensorEntry>>& train_tensor_entries_list,
            std::vector<std::vector<TensorEntry>>& evaluate_tensor_entries_list,
            std::vector<std::shared_ptr<IEmbedding>>& embeddings,
            const std::shared_ptr<ResourceManager>& resource_manager,
            size_t batch_size, size_t batch_size_eval,
            OptParams<TypeFP>& embedding_opt_params) {
#ifdef ENABLE_MPI
  int num_procs = 1, pid = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif

  Embedding_t embedding_type = sparse_embedding.embedding_type;
  std::string bottom_name = sparse_embedding.bottom_name;
  std::string top_name = sparse_embedding.sparse_embedding_name;
  size_t max_vocabulary_size_per_gpu = sparse_embedding.max_vocabulary_size_per_gpu;
  size_t embedding_vec_size = sparse_embedding.embedding_vec_size;
  int combiner = sparse_embedding.combiner;

  SparseInput<TypeKey> sparse_input;
  if (!find_item_in_map(sparse_input, bottom_name, sparse_input_map)) {
    CK_THROW_(Error_t::WrongInput, "Cannot find bottom");
  }
  //embedding_opt_params.scaler = scaler;
  switch (embedding_type) {
    case Embedding_t::DistributedSlotSparseEmbeddingHash: {
      const SparseEmbeddingHashParams<TypeFP> embedding_params = {
          batch_size,
          batch_size_eval,
          max_vocabulary_size_per_gpu,
          {},
          embedding_vec_size,
          sparse_input.max_feature_num_per_sample,
          sparse_input.slot_num,
          combiner,  // combiner: 0-sum, 1-mean
          embedding_opt_params};
      embeddings.emplace_back(new DistributedSlotSparseEmbeddingHash<TypeKey, TypeFP>(
          sparse_input.train_row_offsets, sparse_input.train_values, sparse_input.train_nnz,
          sparse_input.evaluate_row_offsets, sparse_input.evaluate_values,
          sparse_input.evaluate_nnz, embedding_params, resource_manager));
      break;
    }
    case Embedding_t::LocalizedSlotSparseEmbeddingHash: {
      std::string plan_file = "";
      const SparseEmbeddingHashParams<TypeFP> embedding_params = {
          batch_size,
          batch_size_eval,
          max_vocabulary_size_per_gpu,
          sparse_embedding.slot_size_array,
          embedding_vec_size,
          sparse_input.max_feature_num_per_sample,
          sparse_input.slot_num,
          combiner,  // combiner: 0-sum, 1-mean
          embedding_opt_params};
      embeddings.emplace_back(new LocalizedSlotSparseEmbeddingHash<TypeKey, TypeFP>(
          sparse_input.train_row_offsets, sparse_input.train_values, sparse_input.train_nnz,
          sparse_input.evaluate_row_offsets, sparse_input.evaluate_values,
          sparse_input.evaluate_nnz, embedding_params, plan_file, resource_manager));
      break;
    }
    case Embedding_t::LocalizedSlotSparseEmbeddingOneHot: {
      std::string plan_file = "";
      const SparseEmbeddingHashParams<TypeFP> embedding_params = {
          batch_size,
          batch_size_eval,
          0,
          sparse_embedding.slot_size_array,
          embedding_vec_size,
          sparse_input.max_feature_num_per_sample,
          sparse_input.slot_num,
          combiner,  // combiner: 0-sum, 1-mean
          embedding_opt_params};
      embeddings.emplace_back(new LocalizedSlotSparseEmbeddingOneHot<TypeKey, TypeFP>(
          sparse_input.train_row_offsets, sparse_input.train_values, sparse_input.train_nnz,
          sparse_input.evaluate_row_offsets, sparse_input.evaluate_values,
          sparse_input.evaluate_nnz, embedding_params, plan_file, resource_manager));
      break;
    }
  }  // switch

  for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
    train_tensor_entries_list[i].push_back(
        {top_name, (embeddings.back()->get_train_output_tensors())[i]});
    evaluate_tensor_entries_list[i].push_back(
        {top_name, (embeddings.back()->get_evaluate_output_tensors())[i]});
  }            
}

template void add_sparse_embedding<long long, float>(SparseEmbedding&,
            std::map<std::string, SparseInput<long long>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::vector<std::shared_ptr<IEmbedding>>&,
            const std::shared_ptr<ResourceManager>&,
            size_t, size_t, OptParams<float>&);
template void add_sparse_embedding<long long, __half>(SparseEmbedding&,
            std::map<std::string, SparseInput<long long>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::vector<std::shared_ptr<IEmbedding>>&,
            const std::shared_ptr<ResourceManager>&,
            size_t, size_t, OptParams<__half>&);
template void add_sparse_embedding<unsigned int, float>(SparseEmbedding&,
            std::map<std::string, SparseInput<unsigned int>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::vector<std::shared_ptr<IEmbedding>>&,
            const std::shared_ptr<ResourceManager>&,
            size_t, size_t, OptParams<float>&);
template void add_sparse_embedding<unsigned int, __half>(SparseEmbedding&,
            std::map<std::string, SparseInput<unsigned int>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::vector<std::vector<TensorEntry>>&,
            std::vector<std::shared_ptr<IEmbedding>>&,
            const std::shared_ptr<ResourceManager>&,
            size_t, size_t, OptParams<__half>&);
} // namespace HugeCTR
