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


/** This function is used to create a embedding instance.
* @param name, specified name for this embedding instance, which will be modified in this function to be globally unique.
* @param embedding_type, type of this embedding instance.
* @param optimizer_type, as name illustrated.
* @param max_vocabulary_size_per_gpu, as name illustrated, used to allocate spaces for embedding parameters.
* @param slot_size_array, used to dedicatly specify memory distribution for this embedding instance.
* @param opt_hprams, hyper params for optimizer
* @param update_type, as name illustrated, can be Local, Global, LazyGlobal
* @param atomic_update, used in SGD optimizer
* @param scaler, used in mixed_precision training.
* @param slot_num, how many slots (feature fields) in this embedding instance.
* @param max_nnz, max valid number of features in a slot.
* @param max_feature_num, max feature number of features in a sample.
* @param embedding_vec_size, embedding vector size of this embedding instance
* @param combiner, how to combine embedding forward results in a slot, 0:sum, 1:mean
*/
template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::create_embedding(std::string& name, 
                const HugeCTR::Embedding_t& embedding_type, 
                const HugeCTR::Optimizer_t& optimizer_type,
                const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
                const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, const bool atomic_update,
                const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
                const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner) {
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

    /*store embedding params*/
    tensorflow::Status status;
    status = set_embedding_params(embedding_name, embedding_type, optimizer_type, max_vocabulary_size_per_gpu, 
                                  slot_size_array, opt_hparams, update_type, atomic_update,
                                  scaler, slot_num, max_nnz, max_feature_num, embedding_vec_size, combiner);
    if (status != tensorflow::Status::OK()) return status;

    /*create buffer for this embedding instance*/
    size_t local_gpu_count = resource_manager_->get_local_gpu_count();
    std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> tmp_buff;
    for (size_t i = 0; i < local_gpu_count; ++i) {
        tmp_buff.push_back(GeneralBuffer2<CudaAllocator>::create());
    }
    auto pair = buffs_.emplace(std::make_pair(embedding_name, tmp_buff));
    if (pair.second != true) return tensorflow::errors::Aborted(__FILE__, ": ", __LINE__, "buffs_ emplace for ", 
                                                                embedding_name, " failed.");

    /*check vocabulary size*/
    if (HugeCTR::Embedding_t::DistributedSlotSparseEmbeddingHash == embedding_type) {
        if (0 >= max_vocabulary_size_per_gpu) {
            return tensorflow::errors::InvalidArgument("Attr max_vocabulary_size_per_gpu should be >= 0, but get ", 
                        max_vocabulary_size_per_gpu);
        }
    } else if (HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingHash == embedding_type) {
        if (0 >= max_vocabulary_size_per_gpu && 0 == slot_size_array.size()) {
            return tensorflow::errors::InvalidArgument("Not specify Attr max_vocabulary_size_per_gpu or slot_size_array.",
                        " Must set one.");
        } 
    }

    /*set optimizer*/
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

    /*register input spaces*/
    status = register_input_space(embedding_type, slot_num, batch_size_, max_nnz, max_feature_num, 
                                    embedding_name, embedding_name + "_train");
    if (status != tensorflow::Status::OK()) return status;
    status = register_input_space(embedding_type, slot_num, batch_size_eval_, max_nnz, max_feature_num, 
                                    embedding_name, embedding_name + "_eval");
    if (status != tensorflow::Status::OK()) return status;

    std::shared_ptr<InputSpace> train_space = get_input_space(embedding_name + "_train");
    std::shared_ptr<InputSpace> eval_space = get_input_space(embedding_name + "_eval");
    if (!train_space) {
        return tensorflow::errors::NotFound("Did not find ", embedding_name + "_train", " in input_spaces.");
    }
    if (!eval_space) {
        return tensorflow::errors::NotFound("Did not find ", embedding_name + "_eval", " in input_spaces.");
    }

    /*create embedding layer*/
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
    }
    embeddings_.emplace(std::make_pair(embedding_name, embedding));

    /*allocate GPU spaces*/
    auto buff = get_buff(embedding_name);
    if (buff.empty()) return tensorflow::errors::Aborted(__FILE__, ": ", __LINE__, ": buffs for ", embedding_name, " is empty.");
    for (size_t i = 0; i < local_gpu_count; ++i) {
        CudaDeviceContext context(resource_manager_->get_local_gpu(i)->get_device_id());
        buff[i]->allocate();
    }

    /*modify output name*/
    name = embedding_name;
    return tensorflow::Status::OK();
}


template tensorflow::Status EmbeddingWrapper<long long, float>::create_embedding(std::string& name, 
                const HugeCTR::Embedding_t& embedding_type, 
                const HugeCTR::Optimizer_t& optimizer_type,
                const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
                const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, const bool atomic_update,
                const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
                const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner);
template tensorflow::Status EmbeddingWrapper<long long, __half>::create_embedding(std::string& name, 
                const HugeCTR::Embedding_t& embedding_type, 
                const HugeCTR::Optimizer_t& optimizer_type,
                const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
                const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, const bool atomic_update,
                const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
                const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::create_embedding(std::string& name, 
                const HugeCTR::Embedding_t& embedding_type, 
                const HugeCTR::Optimizer_t& optimizer_type,
                const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
                const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, const bool atomic_update,
                const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
                const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::create_embedding(std::string& name, 
                const HugeCTR::Embedding_t& embedding_type, 
                const HugeCTR::Optimizer_t& optimizer_type,
                const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
                const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, const bool atomic_update,
                const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
                const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner);

} // namespace Version1
} // namespace HugeCTR