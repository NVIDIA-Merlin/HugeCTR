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
#include "../v1/embedding_utils.hpp"

namespace HugeCTR {
namespace Version2 {

/*save embedding layer's hyper params*/
template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::save_embedding_hyper_params(const std::string& name, 
        const HugeCTR::Embedding_t& embedding_type, 
        const HugeCTR::Optimizer_t& optimizer_type,
        const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
        const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, const bool atomic_update,
        const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
        const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner) {
    std::shared_ptr<EmbeddingHyperParams> embedding_hyper_params = std::make_shared<EmbeddingHyperParams>();

    embedding_hyper_params->embedding_type_ = embedding_type;
    embedding_hyper_params->optimizer_type_ = optimizer_type;
    embedding_hyper_params->max_vocabulary_size_per_gpu_ = max_vocabulary_size_per_gpu;
    embedding_hyper_params->slot_size_array_ = slot_size_array;
    embedding_hyper_params->opt_hparams_ = opt_hparams;
    embedding_hyper_params->update_type_ = update_type;
    embedding_hyper_params->atomic_update_ = atomic_update;
    embedding_hyper_params->scaler_ = scaler;
    embedding_hyper_params->slot_num_ = slot_num;
    embedding_hyper_params->max_nnz_ = max_nnz;
    embedding_hyper_params->max_feature_num_ = max_feature_num;
    embedding_hyper_params->embedding_vec_size_ = embedding_vec_size;
    embedding_hyper_params->combiner_ = combiner;

    auto pair = embedding_hyper_params_.emplace(std::make_pair(name, embedding_hyper_params));
    if (pair.second != true) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                " emplace embedding hyper params failed.");

    return tensorflow::Status::OK();
}

template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::register_input_space(
            const HugeCTR::Embedding_t& embedding_type,
            const size_t& slot_num, const size_t& batchsize, const size_t& max_nnz,
            const size_t& max_feature_num, const std::string& embedding_name, const std::string& space_name) {
    size_t local_gpu_count = resource_manager_->get_local_gpu_count();
    size_t total_gpu_count = resource_manager_->get_global_gpu_count();

    auto input_buff = get_item_from_map(input_buffs_, embedding_name);
    if (input_buff.empty()) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                        "Cannot find input_buffer for ", embedding_name);

    std::shared_ptr<InputSpace> tmp_input_space = std::make_shared<InputSpace>();
    /*create row offset, value tensors and nnz_array*/
    for (size_t dev_id = 0; dev_id < local_gpu_count; ++dev_id) {
        int slots = 0;
        if (HugeCTR::Embedding_t::DistributedSlotSparseEmbeddingHash == embedding_type) {
            slots = slot_num;
        } else if (HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingHash == embedding_type || 
                   HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingOneHot == embedding_type) {
            size_t mod_slots = static_cast<size_t>(slot_num) % total_gpu_count;
            size_t device_id = resource_manager_->get_local_gpu(dev_id)->get_device_id();
            if (device_id < mod_slots) {
                slots = slot_num / total_gpu_count + 1;
            } else {
                slots = slot_num / total_gpu_count;
            }
        }

        std::shared_ptr<BufferBlock2<TypeKey>> blockbuff = input_buff[dev_id]->template create_block<TypeKey>();

        std::vector<size_t> num_rows_dim = {1, batchsize * slots + 1};
        Tensor2<TypeKey> tmp_row_offset;
        blockbuff->reserve(num_rows_dim, &tmp_row_offset);

        size_t num_max_value = (max_nnz * slots) <= max_feature_num
                                ? (max_nnz * slots * batchsize)
                                : (max_feature_num * batchsize);
        std::vector<size_t> num_max_value_dim = {1, num_max_value};
        Tensor2<TypeKey> tmp_value;
        blockbuff->reserve(num_max_value_dim, &tmp_value);

        tmp_input_space->row_offsets_tensors_.emplace_back(tmp_row_offset);
        tmp_input_space->value_tensors_.emplace_back(tmp_value);
        tmp_input_space->nnz_array_.emplace_back(new size_t);
        tmp_input_space->csr_buffers_.emplace_back(blockbuff->as_tensor());
    } // for dev_id

    auto pair = input_spaces_.emplace(std::make_pair(space_name, tmp_input_space));
    if (pair.second != true) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                    "create input_space for ", space_name, " failed.");

    return tensorflow::Status::OK();
}


template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::create_embedding(
            std::string& name, const HugeCTR::Embedding_t& embedding_type, 
            const HugeCTR::Optimizer_t& optimizer_type,
            const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
            const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, const bool atomic_update,
            const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
            const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner){
    /*decide embedding instance name*/
    std::string embedding_name = name;
    if (embeddings_.find(embedding_name) != embeddings_.end()) {
        auto name_s = Utils::split(embedding_name, "_");
        int num = Utils::string2num(*(name_s.rbegin()));
        if (-1 == num) {
            embedding_name = name + "_" + std::to_string(1);
        } else {
            *(name_s.rbegin()) = std::to_string(num + 1);
            embedding_name = Utils::strs_concat(name_s, "_");
        }
    }

    /*store embedding hyper params*/
    WRAPPER_REQUIRE_OK(save_embedding_hyper_params(embedding_name, embedding_type, optimizer_type,
                        max_vocabulary_size_per_gpu, slot_size_array, opt_hparams, update_type,
                        atomic_update, scaler, slot_num, max_nnz, max_feature_num, embedding_vec_size,
                        combiner));

    /*create input buffer and register input space*/
    size_t local_gpu_count = resource_manager_->get_local_gpu_count();
    std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> tmp_buff;
    for (size_t dev_id = 0; dev_id < local_gpu_count; ++dev_id) {
        tmp_buff.push_back(GeneralBuffer2<CudaAllocator>::create());
    }
    input_buffs_.emplace(std::make_pair(embedding_name, tmp_buff));
    WRAPPER_REQUIRE_OK(register_input_space(embedding_type, slot_num, batch_size_, max_nnz, max_feature_num,
                        embedding_name, embedding_name + "_train"));
    WRAPPER_REQUIRE_OK(register_input_space(embedding_type, slot_num, batch_size_eval_, max_nnz, max_feature_num,
                        embedding_name, embedding_name + "_eval"));

    try {
        /*check vocabulary size*/
        if (HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingHash == embedding_type) {
            if (0 >= max_vocabulary_size_per_gpu && 0 == slot_size_array.size()) {
                return tensorflow::errors::InvalidArgument(__FILE__, ":", __LINE__, " ",
                        "One of max_vocabulary_size_per_gpu or slot_size_array must be set.");
            }
        }

        /*create optimizer*/
        HugeCTR::OptParams<TypeFP> embedding_opt_params;
        HugeCTR::OptHyperParams<TypeFP> opt_hyper_params;
        memset(&opt_hyper_params, 0, sizeof(opt_hyper_params));
        switch (optimizer_type) {
            case HugeCTR::Optimizer_t::Adam:{ // opt_hprams = {lr, beta1, beta2, epsilon}
                if (opt_hparams.size() != 4) {
                    return tensorflow::errors::Unavailable("opt_hparams should be [lr, beta1, beta2, epsilon] when using Adam.");
                }
                opt_hyper_params.adam.beta1 = opt_hparams[1]; // beta1
                opt_hyper_params.adam.beta2 = opt_hparams[2]; // beta2
                opt_hyper_params.adam.epsilon = opt_hparams[3]; // epsilon
                embedding_opt_params = {HugeCTR::Optimizer_t::Adam, opt_hparams[0]/*learning rate*/, opt_hyper_params, update_type};
                break;
            }
            case HugeCTR::Optimizer_t::MomentumSGD: { // opt_hprams = {lr, momentum_factor}
                if (opt_hparams.size() != 2) {
                    return tensorflow::errors::Unavailable("opt_hparams should be [lr, momentum_factor] when using MomentumSGD.");
                }
                opt_hyper_params.momentum.factor = opt_hparams[1]; // momentum_factor
                embedding_opt_params = {HugeCTR::Optimizer_t::MomentumSGD, opt_hparams[0]/*learning rate*/, opt_hyper_params, update_type};
                break;
            }
            case HugeCTR::Optimizer_t::Nesterov:{  // opt_hprams = {lr, momentum_factor}
                if (opt_hparams.size() != 2) {
                    return tensorflow::errors::Unavailable("opt_hparams should be [lr, momentum_factor] when using Nesterov.");
                }
                opt_hyper_params.nesterov.mu = opt_hparams[1]; // momentum_fator
                embedding_opt_params = {HugeCTR::Optimizer_t::Nesterov, opt_hparams[0]/*learning rate*/, opt_hyper_params, update_type};
                break;
            }
            case HugeCTR::Optimizer_t::SGD: { // opt_hprams = {lr}
                if (opt_hparams.size() != 1) {
                    return tensorflow::errors::Unavailable("opt_hparams should be [lr] when using SGD.");
                }
                opt_hyper_params.sgd.atomic_update = atomic_update;
                embedding_opt_params = {HugeCTR::Optimizer_t::SGD, opt_hparams[0]/*learning rate*/, opt_hyper_params, update_type};
                break;
            }
            default: {
                return tensorflow::errors::InvalidArgument(__FILE__, ": ", __LINE__, " No such optimizer type.");
            }
        }
        embedding_opt_params.scaler = scaler;

        /*create embedding layer instance*/
        auto train_space = get_item_from_map(input_spaces_, embedding_name + "_train");
        if (!train_space) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                        "input_train_space for ", embedding_name, " not found.");
        auto eval_space = get_item_from_map(input_spaces_, embedding_name + "_eval");
        if (!eval_space) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                        "input_eval_space for ", embedding_name, " not found.");

        std::shared_ptr<IEmbedding> embedding;
        switch(embedding_type) {
            case HugeCTR::Embedding_t::DistributedSlotSparseEmbeddingHash: {
                const HugeCTR::SparseEmbeddingHashParams<TypeFP> embedding_params = {
                    static_cast<size_t>(batch_size_),
                    static_cast<size_t>(batch_size_eval_),
                    max_vocabulary_size_per_gpu, 
                    {},
                    embedding_vec_size,
                    max_feature_num, 
                    slot_num, 
                    combiner,
                    embedding_opt_params};
                embedding.reset(new HugeCTR::DistributedSlotSparseEmbeddingHash<TypeKey, TypeFP>(
                    train_space->row_offsets_tensors_, train_space->value_tensors_, train_space->nnz_array_,
                    eval_space->row_offsets_tensors_, eval_space->value_tensors_, eval_space->nnz_array_,
                    embedding_params, resource_manager_));
                break;
            }
            case HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingHash: {
                const HugeCTR::SparseEmbeddingHashParams<TypeFP> embedding_params = {
                    static_cast<size_t>(batch_size_),
                    static_cast<size_t>(batch_size_eval_),
                    max_vocabulary_size_per_gpu,
                    slot_size_array,
                    embedding_vec_size,
                    max_feature_num,
                    slot_num,
                    combiner,
                    embedding_opt_params};
                std::string plan_file = "";
                embedding.reset(new LocalizedSlotSparseEmbeddingHash<TypeKey, TypeFP>(
                    train_space->row_offsets_tensors_, train_space->value_tensors_, train_space->nnz_array_,
                    eval_space->row_offsets_tensors_, eval_space->value_tensors_, eval_space->nnz_array_,
                    embedding_params, plan_file, resource_manager_));
                break;
            }
            case HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingOneHot:{
                std::string plan_file = "";
                const HugeCTR::SparseEmbeddingHashParams<TypeFP> embedding_params = {
                    static_cast<size_t>(batch_size_),
                    static_cast<size_t>(batch_size_eval_),
                    0,
                    slot_size_array,
                    embedding_vec_size,
                    max_feature_num,
                    slot_num,
                    combiner,
                    embedding_opt_params};
                embedding.reset(new LocalizedSlotSparseEmbeddingOneHot<TypeKey, TypeFP>(
                    train_space->row_offsets_tensors_, train_space->value_tensors_, train_space->nnz_array_,
                    eval_space->row_offsets_tensors_, eval_space->value_tensors_, eval_space->nnz_array_,
                    embedding_params, plan_file, resource_manager_));
                break;
            }
            default: {
                return tensorflow::errors::InvalidArgument("Not supported embedding_type.");
            }
        } // switch embedding_type
        embeddings_.emplace(std::make_pair(embedding_name, embedding));

        /*do allocate embedding memory spaces*/
        auto input_buff = get_item_from_map(input_buffs_, embedding_name);
        if (input_buff.empty()) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                    "Cannot find input_buff for ", embedding_name);
        for (size_t dev_id = 0; dev_id < local_gpu_count; ++dev_id) {
            CudaDeviceContext context(resource_manager_->get_local_gpu(dev_id)->get_device_id());
            input_buff[dev_id]->allocate();
        }

        /*create cudaEvents for this embedding*/
        std::vector<cudaEvent_t> fprop_events(local_gpu_count, nullptr);
        std::vector<cudaEvent_t> bprop_events(local_gpu_count, nullptr);
        std::vector<cudaEvent_t> to_csr_events(local_gpu_count, nullptr);
        for (size_t dev_id = 0; dev_id < local_gpu_count; ++dev_id) {
            CudaDeviceContext context(resource_manager_->get_local_gpu(dev_id)->get_device_id());

            WRAPPER_CUDA_CHECK(cudaEventCreateWithFlags(&(fprop_events[dev_id]), cudaEventDisableTiming));
            WRAPPER_CUDA_CHECK(cudaEventCreateWithFlags(&(bprop_events[dev_id]), cudaEventDisableTiming));
            WRAPPER_CUDA_CHECK(cudaEventCreateWithFlags(&(to_csr_events[dev_id]), cudaEventDisableTiming));
        } // for dev_id
        fprop_events_.emplace(std::make_pair(embedding_name, std::move(fprop_events)));
        bprop_events_.emplace(std::make_pair(embedding_name, std::move(bprop_events)));
        to_csr_events_.emplace(std::make_pair(embedding_name, std::move(to_csr_events)));

        /*allocate internel spaces for distributing keys to each GPU*/
        std::shared_ptr<DistributeKeysInternelSpaces> distribute_keys_spaces = 
                        std::make_shared<DistributeKeysInternelSpaces>();
        WRAPPER_REQUIRE_OK(DistributeKeysInternelSpaces::create(resource_manager_, embedding_type, batch_size_, 
                        batch_size_eval_, slot_num, max_nnz, distribute_keys_spaces));
        emb_distribute_keys_internel_spaces_.emplace(std::make_pair(embedding_name, distribute_keys_spaces));

        /*distribute functor*/
        distribute_keys_gpu_func_type distribute_keys_func;
        switch (embedding_type) {
            case HugeCTR::Embedding_t::DistributedSlotSparseEmbeddingHash:{
                distribute_keys_func = &EmbeddingWrapper<TypeKey, TypeFP>::distribute_keys_gpu_distributed;
                break;
            }
            case HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingOneHot:
            case HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingHash:{
                distribute_keys_func = &EmbeddingWrapper<TypeKey, TypeFP>::distribute_keys_gpu_localized;
                break;
            }
        }
        distribute_keys_on_gpu_func_.emplace(std::make_pair(embedding_name, distribute_keys_func));

        /*modify output name*/
        name = embedding_name;

    } catch (const std::exception& rt_error){
        return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ", rt_error.what());
    }
    
    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::create_embedding(
            std::string& name, const HugeCTR::Embedding_t& embedding_type, 
            const HugeCTR::Optimizer_t& optimizer_type,
            const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
            const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, const bool atomic_update,
            const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
            const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner);
template tensorflow::Status EmbeddingWrapper<long long, __half>::create_embedding(
            std::string& name, const HugeCTR::Embedding_t& embedding_type, 
            const HugeCTR::Optimizer_t& optimizer_type,
            const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
            const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, const bool atomic_update,
            const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
            const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::create_embedding(
            std::string& name, const HugeCTR::Embedding_t& embedding_type, 
            const HugeCTR::Optimizer_t& optimizer_type,
            const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
            const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, const bool atomic_update,
            const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
            const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::create_embedding(
            std::string& name, const HugeCTR::Embedding_t& embedding_type, 
            const HugeCTR::Optimizer_t& optimizer_type,
            const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
            const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, const bool atomic_update,
            const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
            const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner);

} // namespace Version2
} // namespace HugeCTR