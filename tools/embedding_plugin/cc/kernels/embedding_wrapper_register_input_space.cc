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

/** This function is used to register input tensor spaces. And not allocate true GPU memory.
* @param embedding_type, which type of this embedding instance.
* @param slot_num, how many slots (feature fields) for this embedding instance.
* @param max_nnz, how many valid features in a slot.
* @param max_feature_num, how many features in a sample.
* @param name, register input space for which embedding instance.
*/
template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::register_input_space(const HugeCTR::Embedding_t& embedding_type,
                        const size_t& slot_num, const size_t& batchsize, const size_t& max_nnz,
                        const size_t& max_feature_num, const std::string& embedding_name, const std::string& space_name
                        ){
    size_t local_gpu_count = resource_manager_->get_local_gpu_count();
    size_t total_gpu_count = resource_manager_->get_global_gpu_count();
    auto buff = get_buff(embedding_name);
    if (buff.empty()) return tensorflow::errors::Aborted(__FILE__, ": ", __LINE__, ": buffs for ", embedding_name, " is empty.");

    std::shared_ptr<InputSpace> tmp_space = std::make_shared<InputSpace>();

    /*create row offset and value tensors*/
    for (size_t i = 0; i < local_gpu_count; ++i) {
        int slots = 0;
        if (HugeCTR::Embedding_t::DistributedSlotSparseEmbeddingHash == embedding_type) {
            slots = slot_num;
        } else if (HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingHash == embedding_type || 
                   HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingOneHot == embedding_type) {
            size_t mod_slots = static_cast<size_t>(slot_num) % total_gpu_count;
            size_t global_id = resource_manager_->get_local_gpu(i)->get_global_gpu_id();
            if (global_id < mod_slots) {
                slots = slot_num / total_gpu_count + 1;
            } else {
                slots = slot_num / total_gpu_count;
            }
        } else {
            return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " Unsupported embedding type.");
        }

        std::shared_ptr<BufferBlock2<TypeKey>> blockbuff = buff[i]->template create_block<TypeKey>();

        std::vector<size_t> num_rows_dim = {1, batchsize * slots + 1};
        Tensor2<TypeKey> tmp_row_offset;
        blockbuff->reserve(num_rows_dim, &tmp_row_offset);

        size_t num_max_value = (max_nnz * slots) <= max_feature_num
                                ? (max_nnz * slots * batchsize)
                                : (max_feature_num * batchsize);

        std::vector<size_t> num_max_value_dim = {1, num_max_value};

        Tensor2<TypeKey> tmp_value;
        blockbuff->reserve(num_max_value_dim, &tmp_value);

        tmp_space->row_offsets_tensors_.emplace_back(tmp_row_offset);
        tmp_space->value_tensors_.emplace_back(tmp_value);
        tmp_space->nnz_array_.emplace_back(new size_t);
        tmp_space->csr_buffers_.emplace_back(blockbuff->as_tensor());
    }

    auto pair = input_spaces_.emplace(std::make_pair(space_name, tmp_space));
    if (pair.second != true) return tensorflow::errors::Aborted("emplace input_spaces_ faild.");

    return tensorflow::Status::OK();
}


template tensorflow::Status EmbeddingWrapper<long long, float>::register_input_space(
                        const HugeCTR::Embedding_t& embedding_type,
                        const size_t& slot_num, const size_t& batchsize, const size_t& max_nnz,
                        const size_t& max_feature_num, const std::string& embedding_name, const std::string& space_name
                        );
template tensorflow::Status EmbeddingWrapper<long long, __half>::register_input_space(
                        const HugeCTR::Embedding_t& embedding_type,
                        const size_t& slot_num, const size_t& batchsize, const size_t& max_nnz,
                        const size_t& max_feature_num, const std::string& embedding_name, const std::string& space_name
                        );
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::register_input_space(
                        const HugeCTR::Embedding_t& embedding_type,
                        const size_t& slot_num, const size_t& batchsize, const size_t& max_nnz,
                        const size_t& max_feature_num, const std::string& embedding_name, const std::string& space_name
                        );
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::register_input_space(
                        const HugeCTR::Embedding_t& embedding_type,
                        const size_t& slot_num, const size_t& batchsize, const size_t& max_nnz,
                        const size_t& max_feature_num, const std::string& embedding_name, const std::string& space_name
                        );

} // namespace Version1
} // namespace HugeCTR