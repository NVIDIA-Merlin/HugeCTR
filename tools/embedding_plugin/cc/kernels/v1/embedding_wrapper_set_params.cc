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

/** This function is used to set embedding params.
* and store those params into a struct.
*/
template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::set_embedding_params(const std::string& name, 
                const HugeCTR::Embedding_t& embedding_type, const HugeCTR::Optimizer_t& optimizer_type,
                const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
                const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, 
                const bool atomic_update, const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
                const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner) {
    std::shared_ptr<EmbeddingParams> embedding_params = std::make_shared<EmbeddingParams>();

    embedding_params->embedding_type_ = embedding_type;
    embedding_params->optimizer_type_ = optimizer_type;
    embedding_params->max_vocabulary_size_per_gpu_ = max_vocabulary_size_per_gpu;
    embedding_params->slot_size_array_ = slot_size_array;
    embedding_params->opt_hparams_ = opt_hparams;
    embedding_params->update_type_ = update_type;
    embedding_params->atomic_update_ = atomic_update;
    embedding_params->scaler_ = scaler;
    embedding_params->slot_num_ = slot_num;
    embedding_params->max_nnz_ = max_nnz;
    embedding_params->max_feature_num_ = max_feature_num;
    embedding_params->embedding_vec_size_ = embedding_vec_size;
    embedding_params->combiner_ = combiner;

    auto pair = embedding_params_.emplace(std::make_pair(name, embedding_params));
    if (pair.second != true) return tensorflow::errors::Aborted(__FILE__, ": ", __LINE__, " emplace embedding_params for ", 
                                                                name, " failed.");
    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::set_embedding_params(const std::string& name, 
                const HugeCTR::Embedding_t& embedding_type, const HugeCTR::Optimizer_t& optimizer_type,
                const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
                const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, 
                const bool atomic_update, const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
                const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::set_embedding_params(const std::string& name, 
                const HugeCTR::Embedding_t& embedding_type, const HugeCTR::Optimizer_t& optimizer_type,
                const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
                const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, 
                const bool atomic_update, const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
                const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner);
template tensorflow::Status EmbeddingWrapper<long long, __half>::set_embedding_params(const std::string& name, 
                const HugeCTR::Embedding_t& embedding_type, const HugeCTR::Optimizer_t& optimizer_type,
                const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
                const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, 
                const bool atomic_update, const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
                const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::set_embedding_params(const std::string& name, 
                const HugeCTR::Embedding_t& embedding_type, const HugeCTR::Optimizer_t& optimizer_type,
                const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
                const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, 
                const bool atomic_update, const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
                const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner);

} // namespace Version1
} // namespace HugeCTR