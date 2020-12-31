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


/** This function is used to store initial values to file.
* @param init_value, shape = [vocabulary, embedding_vec_size].
*/
template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::save_initial_to_file(const std::string& embedding_name, 
                            const tensorflow::Tensor* const init_value, const std::string& save_name,
                            const bool on_gpu) {
    std::shared_ptr<EmbeddingParams> params = get_embedding_params(embedding_name);
    if (!params) return tensorflow::errors::NotFound(__FILE__, ": ", __LINE__, " Not found embedding params for ", embedding_name);

    std::ofstream embedding_stream(save_name, std::ofstream::binary);
    if (!embedding_stream.is_open()) return tensorflow::errors::Unavailable(__FILE__, ": ", __LINE__, " embedding_stream is not open.");

    if (init_value->dims() != 2) return tensorflow::errors::Unavailable(__FILE__, ": ", __LINE__, " init_value's dims should be 2.");
    if (static_cast<size_t>(init_value->dim_size(0)) > 
        (params->max_vocabulary_size_per_gpu_ * resource_manager_->get_global_gpu_count()))
        return tensorflow::errors::Unavailable(__FILE__, ": ", __LINE__, 
                    " init_value's vocabulary is larger than total vocabulary size.");
    if (init_value->dim_size(1) != params->embedding_vec_size_) return tensorflow::errors::Unavailable(__FILE__, ": ", __LINE__, 
                    " init_value's embedding_vec_size_ is not equal to which is initialized.");

    auto init_value_flat = init_value->flat<float>();
    for (long int row = 0; row < init_value->dim_size(0); ++row) { // each row
        TypeKey key = static_cast<TypeKey>(row); // key
        embedding_stream.write(reinterpret_cast<char*>(&key), sizeof(TypeKey));

        switch (params->embedding_type_) {
            case Embedding_t::DistributedSlotSparseEmbeddingHash: {
                // embedding vector values
                if (on_gpu) {
                    std::unique_ptr<float []> temp_init_value(new float[params->embedding_vec_size_]());
                    WRAPPER_CUDA_CHECK(cudaMemcpy(temp_init_value.get(), 
                                                   init_value_flat.data() + row * params->embedding_vec_size_,
                                                   sizeof(float) * params->embedding_vec_size_,
                                                   cudaMemcpyDeviceToHost));
                    embedding_stream.write(reinterpret_cast<char*>(temp_init_value.get()), 
                                           sizeof(float) * params->embedding_vec_size_);
                } else { // on cpu
                    embedding_stream.write(reinterpret_cast<const char*>(init_value_flat.data() + row * params->embedding_vec_size_),
                                            sizeof(float) * params->embedding_vec_size_); 
                }
                break;
            }
            case Embedding_t::LocalizedSlotSparseEmbeddingOneHot:
            case Embedding_t::LocalizedSlotSparseEmbeddingHash: {
                size_t slot_id = key % params->slot_num_; // slot_id
                embedding_stream.write(reinterpret_cast<char*>(&slot_id), sizeof(size_t));

                // embedding vector values
                if (on_gpu) {
                    std::unique_ptr<float []> temp_init_value(new float[params->embedding_vec_size_]());
                    WRAPPER_CUDA_CHECK(cudaMemcpy(temp_init_value.get(), 
                                                   init_value_flat.data() + row * params->embedding_vec_size_,
                                                   sizeof(float) * params->embedding_vec_size_,
                                                   cudaMemcpyDeviceToHost));
                    embedding_stream.write(reinterpret_cast<char*>(temp_init_value.get()), 
                                           sizeof(float) * params->embedding_vec_size_);
                } else { // on cpu
                    embedding_stream.write(reinterpret_cast<const char*>(init_value_flat.data() + row * params->embedding_vec_size_),
                                       sizeof(float) * params->embedding_vec_size_);
                }
                break;
            }
            default: {
                return tensorflow::errors::Unavailable(__FILE__, ": ", __LINE__, " Do not support such embedding type.");
            }
        } // switch (params->embedding_type_)
    } // for row
    embedding_stream.close();

    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::save_initial_to_file(
                            const std::string& embedding_name, 
                            const tensorflow::Tensor* const init_value, const std::string& save_name,
                            const bool on_gpu);
template tensorflow::Status EmbeddingWrapper<long long, __half>::save_initial_to_file(
                            const std::string& embedding_name, 
                            const tensorflow::Tensor* const init_value, const std::string& save_name,
                            const bool on_gpu);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::save_initial_to_file(
                            const std::string& embedding_name, 
                            const tensorflow::Tensor* const init_value, const std::string& save_name,
                            const bool on_gpu);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::save_initial_to_file(
                            const std::string& embedding_name, 
                            const tensorflow::Tensor* const init_value, const std::string& save_name,
                            const bool on_gpu);



} // namespace Version1
} // namespace HugeCTR