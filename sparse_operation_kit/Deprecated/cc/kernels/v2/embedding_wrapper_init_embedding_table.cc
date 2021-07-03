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


#include "embedding_wrapper.h"
#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_one_hot.hpp"

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

namespace HugeCTR {
namespace Version2 {

template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::save_init_value_to_file(
            const std::string& embedding_name, 
            const tensorflow::Tensor* const init_value, const std::string& save_name) {
    const bool on_gpu = true;

    std::shared_ptr<EmbeddingHyperParams> hyper_params = get_item_from_map(embedding_hyper_params_, embedding_name);
    if (!hyper_params) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                    "Cannot find hyper params of ", embedding_name);

    std::ofstream file_stream(save_name, std::ofstream::binary);
    if (!file_stream.is_open()) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                    "file_stream of ", embedding_name, " init_value open failed.");
    if (init_value->dims() != 2) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                    "init_value's dims should be 2.");
    if (static_cast<size_t>(init_value->dim_size(0)) > 
        (hyper_params->max_vocabulary_size_per_gpu_ * resource_manager_->get_local_gpu_count())) {
        return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                    "vocabulary_size of init_value is larger than the allocated memory space.");
    }
    
    auto init_value_flat = init_value->flat<float>();
    for (long long row = 0; row < init_value->dim_size(0); ++row) { // each row
        TypeKey key = static_cast<TypeKey>(row); // key
        file_stream.write(reinterpret_cast<char*>(&key), sizeof(TypeKey));

        switch (hyper_params->embedding_type_) {
            case Embedding_t::DistributedSlotSparseEmbeddingHash: {
                // embedding vector values
                if (on_gpu) {
                    std::unique_ptr<float []> temp_init_value(new float[hyper_params->embedding_vec_size_]());
                    WRAPPER_CUDA_CHECK(cudaMemcpy(temp_init_value.get(), 
                                                   init_value_flat.data() + row * hyper_params->embedding_vec_size_,
                                                   sizeof(float) * hyper_params->embedding_vec_size_,
                                                   cudaMemcpyDeviceToHost));
                    file_stream.write(reinterpret_cast<char*>(temp_init_value.get()), 
                                           sizeof(float) * hyper_params->embedding_vec_size_);
                } else { // on cpu
                    file_stream.write(reinterpret_cast<const char*>(init_value_flat.data() + row * hyper_params->embedding_vec_size_),
                                            sizeof(float) * hyper_params->embedding_vec_size_); 
                }
                break;
            }
            case Embedding_t::LocalizedSlotSparseEmbeddingOneHot:
            case Embedding_t::LocalizedSlotSparseEmbeddingHash: {
                size_t slot_id = key % hyper_params->slot_num_; // slot_id
                file_stream.write(reinterpret_cast<char*>(&slot_id), sizeof(size_t));

                // embedding vector values
                if (on_gpu) {
                    std::unique_ptr<float []> temp_init_value(new float[hyper_params->embedding_vec_size_]());
                    WRAPPER_CUDA_CHECK(cudaMemcpy(temp_init_value.get(), 
                                                   init_value_flat.data() + row * hyper_params->embedding_vec_size_,
                                                   sizeof(float) * hyper_params->embedding_vec_size_,
                                                   cudaMemcpyDeviceToHost));
                    file_stream.write(reinterpret_cast<char*>(temp_init_value.get()), 
                                           sizeof(float) * hyper_params->embedding_vec_size_);
                } else { // on cpu
                    file_stream.write(reinterpret_cast<const char*>(init_value_flat.data() + row * hyper_params->embedding_vec_size_),
                                       sizeof(float) * hyper_params->embedding_vec_size_);
                }
                break;
            }
            default: {
                return tensorflow::errors::Unavailable(__FILE__, ": ", __LINE__, " Do not support such embedding type.");
            }
        } // switch (hyper_params->embedding_type_)
    } // for row
    file_stream.close();

    return tensorflow::Status::OK();
}

template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::init_embedding_table(
            const std::string& embedding_name, 
            const tensorflow::Tensor* init_value) {
    std::shared_ptr<IEmbedding> embedding = get_item_from_map(embeddings_, embedding_name);
    if (!embedding) return tensorflow::errors::NotFound(__FILE__, ":", __LINE__, " ",
                    "Cannot find embedding instance of ", embedding_name);

    try {
        if (init_value->dtype() == tensorflow::DT_BOOL) {
            embedding->init_params();
        } else {
            /*save initial values to file*/
            const std::string sparse_embedding_model = "TMP_EMBEDDING_INITIAL_VALUES";
            WRAPPER_REQUIRE_OK(save_init_value_to_file(embedding_name, init_value, sparse_embedding_model));

            /*load initial values to memory*/
            embedding->load_parameters(sparse_embedding_model);

            /*delete sparse_embedding_model*/
            if (fs::remove_all(sparse_embedding_model) == 0)
                return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                                "Cannot delete ", sparse_embedding_model);
        }

    } catch (const HugeCTR::internal_runtime_error& rt_err) {
        return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                    rt_err.what());
    }

    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::init_embedding_table(
            const std::string& embedding_name, 
            const tensorflow::Tensor* init_value);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::init_embedding_table(
            const std::string& embedding_name, 
            const tensorflow::Tensor* init_value);
template tensorflow::Status EmbeddingWrapper<long long, __half>::init_embedding_table(
            const std::string& embedding_name, 
            const tensorflow::Tensor* init_value);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::init_embedding_table(
            const std::string& embedding_name, 
            const tensorflow::Tensor* init_value);

} // namespace Version2
} // namespace HugeCTR