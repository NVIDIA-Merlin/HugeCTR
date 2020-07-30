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

#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_one_hot.hpp"

namespace HugeCTR {

Embedding<EmbeddingCreator::TYPE_1, float>*
EmbeddingCreator::create_distributed_sparse_embedding_hash(
    const TensorPtrs<TYPE_1>& train_row_offsets_tensors,
    const TensorPtrs<TYPE_1>& train_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& train_nnz_array,
    const TensorPtrs<TYPE_1>& test_row_offsets_tensors,
    const TensorPtrs<TYPE_1>& test_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& test_nnz_array,
    const SparseEmbeddingHashParams<float>& embedding_params,
    const GPUResourceGroupPtr& gpu_resource_group) {
  return new DistributedSlotSparseEmbeddingHash<TYPE_1, float>(
      train_row_offsets_tensors, train_value_tensors, train_nnz_array, test_row_offsets_tensors,
      test_value_tensors, test_nnz_array, embedding_params, gpu_resource_group);
}

Embedding<EmbeddingCreator::TYPE_2, float>*
EmbeddingCreator::create_distributed_sparse_embedding_hash(
    const TensorPtrs<TYPE_2>& train_row_offsets_tensors,
    const TensorPtrs<TYPE_2>& train_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& train_nnz_array,
    const TensorPtrs<TYPE_2>& test_row_offsets_tensors,
    const TensorPtrs<TYPE_2>& test_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& test_nnz_array,
    const SparseEmbeddingHashParams<float>& embedding_params,
    const GPUResourceGroupPtr& gpu_resource_group) {
  return new DistributedSlotSparseEmbeddingHash<TYPE_2, float>(
      train_row_offsets_tensors, train_value_tensors, train_nnz_array, test_row_offsets_tensors,
      test_value_tensors, test_nnz_array, embedding_params, gpu_resource_group);
}

Embedding<EmbeddingCreator::TYPE_1, __half>*
EmbeddingCreator::create_distributed_sparse_embedding_hash(
    const TensorPtrs<TYPE_1>& train_row_offsets_tensors,
    const TensorPtrs<TYPE_1>& train_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& train_nnz_array,
    const TensorPtrs<TYPE_1>& test_row_offsets_tensors,
    const TensorPtrs<TYPE_1>& test_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& test_nnz_array,
    const SparseEmbeddingHashParams<__half>& embedding_params,
    const GPUResourceGroupPtr& gpu_resource_group) {
  return new DistributedSlotSparseEmbeddingHash<TYPE_1, __half>(
      train_row_offsets_tensors, train_value_tensors, train_nnz_array, test_row_offsets_tensors,
      test_value_tensors, test_nnz_array, embedding_params, gpu_resource_group);
}

Embedding<EmbeddingCreator::TYPE_2, __half>*
EmbeddingCreator::create_distributed_sparse_embedding_hash(
    const TensorPtrs<TYPE_2>& train_row_offsets_tensors,
    const TensorPtrs<TYPE_2>& train_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& train_nnz_array,
    const TensorPtrs<TYPE_2>& test_row_offsets_tensors,
    const TensorPtrs<TYPE_2>& test_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& test_nnz_array,
    const SparseEmbeddingHashParams<__half>& embedding_params,
    const GPUResourceGroupPtr& gpu_resource_group) {
  return new DistributedSlotSparseEmbeddingHash<TYPE_2, __half>(
      train_row_offsets_tensors, train_value_tensors, train_nnz_array, test_row_offsets_tensors,
      test_value_tensors, test_nnz_array, embedding_params, gpu_resource_group);
}

Embedding<EmbeddingCreator::TYPE_1, float>*
EmbeddingCreator::create_localized_sparse_embedding_hash(
    const TensorPtrs<TYPE_1>& train_row_offsets_tensors,
    const TensorPtrs<TYPE_1>& train_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& train_nnz_array,
    const TensorPtrs<TYPE_1>& test_row_offsets_tensors,
    const TensorPtrs<TYPE_1>& test_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& test_nnz_array,
    const SparseEmbeddingHashParams<float>& embedding_params, const std::string& plan_file,
    const GPUResourceGroupPtr& gpu_resource_group) {
  return new LocalizedSlotSparseEmbeddingHash<TYPE_1, float>(
      train_row_offsets_tensors, train_value_tensors, train_nnz_array, test_row_offsets_tensors,
      test_value_tensors, test_nnz_array, embedding_params, plan_file, gpu_resource_group);
}

Embedding<EmbeddingCreator::TYPE_2, float>*
EmbeddingCreator::create_localized_sparse_embedding_hash(
    const TensorPtrs<TYPE_2>& train_row_offsets_tensors,
    const TensorPtrs<TYPE_2>& train_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& train_nnz_array,
    const TensorPtrs<TYPE_2>& test_row_offsets_tensors,
    const TensorPtrs<TYPE_2>& test_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& test_nnz_array,
    const SparseEmbeddingHashParams<float>& embedding_params, const std::string& plan_file,
    const GPUResourceGroupPtr& gpu_resource_group) {
  return new LocalizedSlotSparseEmbeddingHash<TYPE_2, float>(
      train_row_offsets_tensors, train_value_tensors, train_nnz_array, test_row_offsets_tensors,
      test_value_tensors, test_nnz_array, embedding_params, plan_file, gpu_resource_group);
}

Embedding<EmbeddingCreator::TYPE_1, __half>*
EmbeddingCreator::create_localized_sparse_embedding_hash(
    const TensorPtrs<TYPE_1>& train_row_offsets_tensors,
    const TensorPtrs<TYPE_1>& train_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& train_nnz_array,
    const TensorPtrs<TYPE_1>& test_row_offsets_tensors,
    const TensorPtrs<TYPE_1>& test_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& test_nnz_array,
    const SparseEmbeddingHashParams<__half>& embedding_params, const std::string& plan_file,
    const GPUResourceGroupPtr& gpu_resource_group) {
  return new LocalizedSlotSparseEmbeddingHash<TYPE_1, __half>(
      train_row_offsets_tensors, train_value_tensors, train_nnz_array, test_row_offsets_tensors,
      test_value_tensors, test_nnz_array, embedding_params, plan_file, gpu_resource_group);
}

Embedding<EmbeddingCreator::TYPE_2, __half>*
EmbeddingCreator::create_localized_sparse_embedding_hash(
    const TensorPtrs<TYPE_2>& train_row_offsets_tensors,
    const TensorPtrs<TYPE_2>& train_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& train_nnz_array,
    const TensorPtrs<TYPE_2>& test_row_offsets_tensors,
    const TensorPtrs<TYPE_2>& test_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& test_nnz_array,
    const SparseEmbeddingHashParams<__half>& embedding_params, const std::string& plan_file,
    const GPUResourceGroupPtr& gpu_resource_group) {
  return new LocalizedSlotSparseEmbeddingHash<TYPE_2, __half>(
      train_row_offsets_tensors, train_value_tensors, train_nnz_array, test_row_offsets_tensors,
      test_value_tensors, test_nnz_array, embedding_params, plan_file, gpu_resource_group);
}

Embedding<EmbeddingCreator::TYPE_1, float>*
EmbeddingCreator::create_localized_sparse_embedding_one_hot(
    const TensorPtrs<TYPE_1>& train_row_offsets_tensors,
    const TensorPtrs<TYPE_1>& train_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& train_nnz_array,
    const TensorPtrs<TYPE_1>& test_row_offsets_tensors,
    const TensorPtrs<TYPE_1>& test_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& test_nnz_array,
    const SparseEmbeddingHashParams<float>& embedding_params, const std::string& plan_file,
    const GPUResourceGroupPtr& gpu_resource_group) {
  return new LocalizedSlotSparseEmbeddingOneHot<TYPE_1, float>(
      train_row_offsets_tensors, train_value_tensors, train_nnz_array, test_row_offsets_tensors,
      test_value_tensors, test_nnz_array, embedding_params, plan_file, gpu_resource_group);
}

Embedding<EmbeddingCreator::TYPE_2, float>*
EmbeddingCreator::create_localized_sparse_embedding_one_hot(
    const TensorPtrs<TYPE_2>& train_row_offsets_tensors,
    const TensorPtrs<TYPE_2>& train_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& train_nnz_array,
    const TensorPtrs<TYPE_2>& test_row_offsets_tensors,
    const TensorPtrs<TYPE_2>& test_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& test_nnz_array,
    const SparseEmbeddingHashParams<float>& embedding_params, const std::string& plan_file,
    const GPUResourceGroupPtr& gpu_resource_group) {
  return new LocalizedSlotSparseEmbeddingOneHot<TYPE_2, float>(
      train_row_offsets_tensors, train_value_tensors, train_nnz_array, test_row_offsets_tensors,
      test_value_tensors, test_nnz_array, embedding_params, plan_file, gpu_resource_group);
}

Embedding<EmbeddingCreator::TYPE_1, __half>*
EmbeddingCreator::create_localized_sparse_embedding_one_hot(
    const TensorPtrs<TYPE_1>& train_row_offsets_tensors,
    const TensorPtrs<TYPE_1>& train_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& train_nnz_array,
    const TensorPtrs<TYPE_1>& test_row_offsets_tensors,
    const TensorPtrs<TYPE_1>& test_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& test_nnz_array,
    const SparseEmbeddingHashParams<__half>& embedding_params, const std::string& plan_file,
    const GPUResourceGroupPtr& gpu_resource_group) {
  return new LocalizedSlotSparseEmbeddingOneHot<TYPE_1, __half>(
      train_row_offsets_tensors, train_value_tensors, train_nnz_array, test_row_offsets_tensors,
      test_value_tensors, test_nnz_array, embedding_params, plan_file, gpu_resource_group);
}

Embedding<EmbeddingCreator::TYPE_2, __half>*
EmbeddingCreator::create_localized_sparse_embedding_one_hot(
    const TensorPtrs<TYPE_2>& train_row_offsets_tensors,
    const TensorPtrs<TYPE_2>& train_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& train_nnz_array,
    const TensorPtrs<TYPE_2>& test_row_offsets_tensors,
    const TensorPtrs<TYPE_2>& test_value_tensors,
    const std::vector<std::shared_ptr<size_t>>& test_nnz_array,
    const SparseEmbeddingHashParams<__half>& embedding_params, const std::string& plan_file,
    const GPUResourceGroupPtr& gpu_resource_group) {
  return new LocalizedSlotSparseEmbeddingOneHot<TYPE_2, __half>(
      train_row_offsets_tensors, train_value_tensors, train_nnz_array, test_row_offsets_tensors,
      test_value_tensors, test_nnz_array, embedding_params, plan_file, gpu_resource_group);
}

}  // namespace HugeCTR
