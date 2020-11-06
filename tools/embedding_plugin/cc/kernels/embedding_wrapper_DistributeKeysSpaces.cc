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


#include "embedding_wrapper.h"
#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_one_hot.hpp"
#include "embedding_utils.hpp"

namespace HugeCTR {
namespace Version1 {


template <typename TypeKey, typename TypeFP>
EmbeddingWrapper<TypeKey, TypeFP>::DistributeKeysSpaces::DistributeKeysSpaces(const size_t gpu_count)
    : allocated_(false), gpu_count_(gpu_count), batch_size_(0), slot_num_(0), max_nnz_(0) {

    input_keys_copies_.insert(input_keys_copies_.begin(), gpu_count, nullptr);

    csr_values_.insert(csr_values_.begin(), gpu_count, nullptr);
    csr_row_offsets_.insert(csr_row_offsets_.begin(), gpu_count, nullptr);
    csr_row_offsets_casts_.insert(csr_row_offsets_casts_.begin(), gpu_count, nullptr);
    csr_col_indices_.insert(csr_col_indices_.begin(), gpu_count, nullptr);
    csr_nnz_rows_.insert(csr_nnz_rows_.begin(), gpu_count, nullptr);
    input_keys_transposes_.insert(input_keys_transposes_.begin(), gpu_count, nullptr);
    total_nnzs_.insert(total_nnzs_.begin(), gpu_count, 0);
}

template <typename TypeKey, typename TypeFP>
EmbeddingWrapper<TypeKey, TypeFP>::DistributeKeysSpaces::~DistributeKeysSpaces() {
    for (size_t i = 0; i < gpu_count_; ++i) {
        // if (cusparse_handles_[i]) cusparseDestroy(cusparse_handles_[i]);
        // if (cublas_handles_[i]) cublasDestroy(cublas_handles_[i]);
        if (cusparse_mat_descs_[i]) cusparseDestroyMatDescr(cusparse_mat_descs_[i]);
        if (cuda_streams_[i]) cudaStreamDestroy(cuda_streams_[i]);
    }
    for (auto handle : cusparse_handles_) {
        if (handle) cusparseDestroy(handle);
    }
    for (auto handle : cublas_handles_) {
        if (handle) cublasDestroy(handle);
    }
}

template EmbeddingWrapper<long long, float>::DistributeKeysSpaces::DistributeKeysSpaces(const size_t gpu_count);
template EmbeddingWrapper<long long, __half>::DistributeKeysSpaces::DistributeKeysSpaces(const size_t gpu_count);
template EmbeddingWrapper<unsigned int, float>::DistributeKeysSpaces::DistributeKeysSpaces(const size_t gpu_count);
template EmbeddingWrapper<unsigned int, __half>::DistributeKeysSpaces::DistributeKeysSpaces(const size_t gpu_count);

template EmbeddingWrapper<long long, float>::DistributeKeysSpaces::~DistributeKeysSpaces();
template EmbeddingWrapper<long long, __half>::DistributeKeysSpaces::~DistributeKeysSpaces();
template EmbeddingWrapper<unsigned int, float>::DistributeKeysSpaces::~DistributeKeysSpaces();
template EmbeddingWrapper<unsigned int, __half>::DistributeKeysSpaces::~DistributeKeysSpaces();

} // namespace Version1
} // namespace HugeCTR