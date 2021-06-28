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

#include "parameters/raw_param.h"
#include "tensor_buffer/tensor2_wrapper.h"
#include "embeddings/embedding_layer.h"
#include "common.h"
#include <system_error>
#include <fstream>

namespace SparseOperationKit {

RawParam::RawParam(const std::string& initializer, const std::vector<size_t> shape,
                   const std::shared_ptr<ResourcesManager>& resource_mgr,
                   const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
                   const std::string var_name, const bool trainable)
: resource_mgr_(resource_mgr), 
hashtables_(resource_mgr->get_local_gpu_count()),
max_vocabulary_size_per_gpu_(shape[0]), embedding_vector_size_(shape[1]),
var_name_(var_name), trainable_(trainable), initializer_(Initializer::Get(initializer)),
has_hashtable_(true)
{
    emb_table_tensors_.reserve(resource_mgr_->get_local_gpu_count());
    emb_table_tensors_interface_.reserve(resource_mgr_->get_local_gpu_count());

    HugeCTR::CudaDeviceContext device_context;
    for (size_t dev_id = 0; dev_id < resource_mgr->get_local_gpu_count(); ++dev_id) {
        device_context.set_device(resource_mgr_->get_local_gpu(dev_id)->get_local_device_id());
        // reserve spaces for embedding table
        {
            Tensor2<float> tensor;
            buffers[dev_id]->reserve(shape, &tensor);
            emb_table_tensors_.push_back(tensor);
            emb_table_tensors_interface_.push_back(Tensor2Wrapper<float>::create(tensor));
        }
        
        // construct hashtable
        {
            hashtables_[dev_id].reset(new NvHashTable(max_vocabulary_size_per_gpu_));
        }
    } // for dev_id

    if (emb_table_tensors_.size() != emb_table_tensors_interface_.size())
        throw std::runtime_error(ErrorBase + "The size of embedding table tensors and its interface if not equal.");
}

RawParam::~RawParam() {}

std::shared_ptr<RawParam> RawParam::create(const std::string& initializer, const bool use_hashtable,
                                            const std::vector<size_t> shape,
                                            const std::shared_ptr<ResourcesManager>& resource_mgr,
            const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
                                            const std::string var_name, const bool trainable) {
    if (use_hashtable)
        return std::shared_ptr<RawParam>(new RawParam(initializer, shape, resource_mgr, buffers, var_name, trainable));
    else 
        throw std::runtime_error(ErrorBase + "Not implemented yet.");
}

size_t RawParam::get_max_vocabulary_size_per_gpu() const {
    return max_vocabulary_size_per_gpu_;
}

size_t RawParam::get_embedding_vec_size() const {
    return embedding_vector_size_;
}

void RawParam::init(const size_t global_replica_id) {
    const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
    MESSAGE("Variable: " + var_name_ + " on global_replica_id: " + 
            std::to_string(global_replica_id) + " start initialization");
    if (local_replica_id >= emb_table_tensors_.size()) 
        throw std::runtime_error(ErrorBase + "local_replica_id is out of the range of emb_table_tensors.size().");

    const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

    initializer_->fill(emb_table_tensors_interface_[local_replica_id],
                       local_gpu->get_sm_count(),
                       local_gpu->get_variant_curand_gen(),
                       local_gpu->get_stream());

    resource_mgr_->sync_gpu(local_replica_id);

    MESSAGE("Variable: " + var_name_ + " on global_replica_id: " + 
            std::to_string(global_replica_id) + " initialization done.");
}


bool RawParam::trainable() const {
    return trainable_;
}

void RawParam::set_user(std::shared_ptr<EmbeddingLayer>& embedding) {
    user_ = embedding;
}

auto RawParam::get_hashtable(const size_t local_replica_id) -> std::shared_ptr<NvHashTable>&  {
    if (has_hashtable_) return hashtables_[local_replica_id];
    else throw std::runtime_error(ErrorBase + "Hashtable is not valid.");
}

std::shared_ptr<Tensor>& RawParam::get_embedding_table_tensor(const size_t local_replica_id) {
    if (local_replica_id >= emb_table_tensors_.size())
        throw std::runtime_error(ErrorBase + "local_replica_id is out of the range of emb_table_tensors.size().");

    return emb_table_tensors_interface_[local_replica_id];
}

std::string RawParam::get_var_name() const {
    return var_name_;
}

void RawParam::dump_to_file(const std::string filename) {
    MESSAGE("Saving " + var_name_ + " to " + filename);

    /* for RawParam, the content need to be dumped to file
    *  is related to the details of embedding, so delegate
    *  this job to embedding layer.*/
    try {
        /*only chief node need to write file.*/
        if (resource_mgr_->get_worker_id() == 0) {
            std::ofstream param_stream(filename, std::ofstream::binary);
            user_->dump_to_file(param_stream);
            param_stream.close();
        } else {
            std::ofstream param_stream;
            user_->dump_to_file(param_stream);
        }

    } catch (const std::system_error& error) {
        throw std::runtime_error(ErrorBase + error.what());
    }

    MESSAGE("Saved.");
}

void RawParam::restore_from_file(const std::string filename) {
    MESSAGE("Restoring " + var_name_ + " from " + filename);

    /* for RawParam, the content need to be restored from file
    *  is related to the details of embedding, so delegate
    *  this job to embedding layer.*/
    try {
        /*all nodes reads the file simultaneously.*/
        std::ifstream param_stream(filename, std::ifstream::binary);
        user_->restore_from_file(param_stream);
        param_stream.close();
    } catch (const std::system_error& error) {
        throw std::runtime_error(ErrorBase + error.what());
    }

    MESSAGE("Restored.");
}

void RawParam::load_tensors_to_memory(const std::vector<std::shared_ptr<Tensor>>& tensors) {
    /*for RawParam, how to load tensors to GPU memory 
    * is related to the details of embedding, so delegate
    * this job to embedding layer*/
    MESSAGE("Loading tensors to GPU memory.");

    user_->load_tensors_to_memory(tensors);

    MESSAGE("Loaded.");
}

} // namespace SparseOperationKit