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

#ifndef EMBEDDING_WRAPPER_H
#define EMBEDDING_WRAPPER_H

#include "HugeCTR/include/embeddings/embedding.hpp"
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/utils.hpp"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "cuda_utils.h"

#include <string>
#include <map>
#include <vector>
#include <utility>

namespace HugeCTR {
inline namespace Version1 {

/*wrapper base class*/
class Wrapper {
public:
    virtual tensorflow::Status create_embedding(std::string& name, const HugeCTR::Embedding_t& embedding_type, 
                const HugeCTR::Optimizer_t& optimizer_type,
                const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
                const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, const bool atomic_update,
                const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
                const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner) = 0;
    virtual tensorflow::Status init_embedding_params(const std::string& embedding_name, const tensorflow::Tensor* init_value,
                                                     const bool on_gpu) = 0;
    virtual tensorflow::Status fprop(const tensorflow::Tensor* sparse_indices, 
                        const tensorflow::Tensor* values, const tensorflow::Tensor* dense_shape, 
                        const std::string& embedding_name, const bool is_training,
                        tensorflow::Tensor* const forward_result, const bool on_gpu) = 0;
    virtual tensorflow::Status fprop_v2(const tensorflow::OpInputList& row_offsets, const tensorflow::OpInputList& value_tensors,
                    const tensorflow::Tensor* nnz_array,
                    const std::string& embedding_name, const bool is_training,
                    tensorflow::Tensor* const forward_result) = 0;
    virtual tensorflow::Status fprop_v3(const tensorflow::Tensor* row_offsets, const tensorflow::Tensor* value_tensors,
                    const tensorflow::Tensor* nnz_array,
                    const std::string& embedding_name, const bool is_training,
                    const cudaStream_t& tf_stream,
                    tensorflow::Tensor* const forward_result) = 0;
    virtual tensorflow::Status fprop_v4(const tensorflow::Tensor* row_indices, 
                                        const tensorflow::Tensor* values,
                                        const std::string& embedding_name,
                                        const bool is_training,
                                        const cudaStream_t& tf_stream,
                                        tensorflow::Tensor* const forward_result) = 0;
    virtual tensorflow::Status distribute_keys_gpu(const tensorflow::Tensor* row_indices,
                                                   const tensorflow::Tensor* values,
                                                   const std::string& embedding_name, 
                                                   const bool is_training, 
                                                   tensorflow::Tensor* row_offsets_output,
                                                   tensorflow::Tensor* value_tensors_output,
                                                   tensorflow::Tensor* nnz_array_output) = 0;
    virtual tensorflow::Status get_output_tensor_shape(const std::string& embedding_name, const bool is_training,
                                                tensorflow::TensorShape& shape) = 0;
    virtual tensorflow::Status bprop(const std::string& embedding_name, const tensorflow::Tensor* top_gradients,
                                     const bool on_gpu, const cudaStream_t& tf_stream) = 0;
    virtual tensorflow::Status save(const std::string& embedding_name, const std::string& save_name) = 0;
    virtual tensorflow::Status restore(const std::string& embedding_name, const std::string& file_name) = 0;
};


/* This class is a wrapper of HugeCTR's embedding layer. */
template <typename TypeKey, typename TypeFP>
class EmbeddingWrapper : public Wrapper {
private:
    /*This class is a wrapper of the input tensors for an embedding layer.*/
    struct InputSpace {
        Tensors2<TypeKey> csr_buffers_;
        Tensors2<TypeKey> row_offsets_tensors_;
        Tensors2<TypeKey> value_tensors_;
        std::vector<std::shared_ptr<size_t>> nnz_array_;
    };

    /*This class is a wrapper of embedding hyper params.*/
    struct EmbeddingParams {
        Embedding_t embedding_type_;
        Optimizer_t optimizer_type_;
        long long max_vocabulary_size_per_gpu_;
        std::vector<size_t> slot_size_array_;
        std::vector<float> opt_hparams_;
        HugeCTR::Update_t update_type_;
        bool atomic_update_;
        float scaler_;
        long long slot_num_;
        long long max_nnz_;
        long long max_feature_num_;
        long long embedding_vec_size_;
        int combiner_;
    };

    /*This class is a wrapper of internel spaces used in distribute keys on GPU*/
    struct DistributeKeysInternelSpaces {
        static tensorflow::Status create(const std::shared_ptr<HugeCTR::ResourceManager>& resource_manager, 
                    const HugeCTR::Embedding_t& embedding_type,
                    const size_t& batch_size, const size_t& batch_size_eval,
                    const size_t& slot_num, const size_t& max_nnz,
                    std::shared_ptr<DistributeKeysInternelSpaces>& distribute_keys_space); // create an instance of this class.

        tensorflow::Status reset();

        HugeCTR::Tensors2<bool> binary_flags_;
        std::vector<size_t> cub_temp_storage_bytes_;
        std::vector<void*, CudaUtils::CudaAllocator<void*>> cub_d_temp_storage_;
        HugeCTR::Tensors2<int> cub_coo_indices_output_;
        HugeCTR::Tensors2<TypeKey> cub_values_output_;
        std::vector<size_t*, CudaUtils::CudaHostAllocator<size_t*>> cub_host_num_selected_;
        std::vector<size_t*, CudaUtils::CudaAllocator<size_t*>> cub_dev_num_selected_;
        std::vector<cusparseHandle_t> cusparse_handles_;
        HugeCTR::Tensors2<int> cusparse_csr_row_offsets_output_;
        HugeCTR::Tensors2<TypeKey> csr_row_offsets_cast_; 
        HugeCTR::Tensors2<long long> copy_input_row_indices_;
        HugeCTR::Tensors2<TypeKey> copy_input_values_;
        std::vector<size_t> dev_slot_num_;

        ~DistributeKeysInternelSpaces();
    
    private:
        std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>> internel_buff_;
        std::shared_ptr<HugeCTR::ResourceManager> resource_manager_;
    };

public:
    EmbeddingWrapper(const std::vector<std::vector<int>>& vvgpu, unsigned long long seed, 
                     long long batch_size, long long batch_size_eval);
    EmbeddingWrapper() = delete;
    virtual ~EmbeddingWrapper();
    EmbeddingWrapper(EmbeddingWrapper&) = delete;
    EmbeddingWrapper& operator=(const EmbeddingWrapper&) = delete;

    tensorflow::Status create_embedding(std::string& name, const HugeCTR::Embedding_t& embedding_type, 
                const HugeCTR::Optimizer_t& optimizer_type,
                const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
                const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, const bool atomic_update,
                const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
                const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner) override;
    tensorflow::Status init_embedding_params(const std::string& embedding_name, const tensorflow::Tensor* init_value,
                                             const bool on_gpu) override;
    tensorflow::Status fprop(const tensorflow::Tensor* sparse_indices, 
                        const tensorflow::Tensor* values, const tensorflow::Tensor* dense_shape, 
                        const std::string& embedding_name, const bool is_training,
                        tensorflow::Tensor* const forward_result, const bool on_gpu) override;
    tensorflow::Status fprop_v2(const tensorflow::OpInputList& row_offsets, const tensorflow::OpInputList& value_tensors,
                    const tensorflow::Tensor* nnz_array,
                    const std::string& embedding_name, const bool is_training,
                    tensorflow::Tensor* const forward_result) override;
    tensorflow::Status fprop_v3(const tensorflow::Tensor* row_offsets, const tensorflow::Tensor* value_tensors,
                    const tensorflow::Tensor* nnz_array,
                    const std::string& embedding_name, const bool is_training,
                    const cudaStream_t& tf_stream,
                    tensorflow::Tensor* const forward_result) override;
    tensorflow::Status fprop_v4(const tensorflow::Tensor* row_indices, 
                                const tensorflow::Tensor* values,
                                const std::string& embedding_name,
                                const bool is_training,
                                const cudaStream_t& tf_stream,
                                tensorflow::Tensor* const forward_result) override;
    tensorflow::Status distribute_keys_gpu(const tensorflow::Tensor* row_indices,
                                            const tensorflow::Tensor* values,
                                            const std::string& embedding_name, 
                                            const bool is_training, 
                                            tensorflow::Tensor* row_offsets_output,
                                            tensorflow::Tensor* value_tensors_output,
                                            tensorflow::Tensor* nnz_array_output);                            
    tensorflow::Status get_output_tensor_shape(const std::string& embedding_name, const bool is_training,
                                                tensorflow::TensorShape& shape) override;

    tensorflow::Status bprop(const std::string& embedding_name, const tensorflow::Tensor* top_gradients,
                             const bool on_gpu, const cudaStream_t& tf_stream) override;
    tensorflow::Status save(const std::string& embedding_name, const std::string& save_name) override;
    tensorflow::Status restore(const std::string& embedding_name, const std::string& file_name) override;

    void evaluate();
    
private:
    std::map<std::string, std::shared_ptr<IEmbedding>> embeddings_; // <embedding_instance_name, embedding>
    std::map<std::string, std::shared_ptr<InputSpace>> input_spaces_; // <input_space_name, input_space>
    std::shared_ptr<HugeCTR::ResourceManager> resource_manager_; 
    std::map<std::string, std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>> buffs_; // <embedding_instance_name, buff>
    std::map<std::string, std::shared_ptr<EmbeddingParams>> embedding_params_; // <embedding_instance_name, EmbeddingParams>
    std::map<std::string, std::vector<cudaEvent_t>> fprop_events_;
    std::map<std::string, std::vector<cudaEvent_t>> bprop_events_;
    std::map<std::string, std::shared_ptr<DistributeKeysInternelSpaces>> emb_distribute_keys_internel_spaces_;

    using distribute_keys_gpu_func_type = tensorflow::Status(EmbeddingWrapper<TypeKey, TypeFP>::*)(
                                                                const tensorflow::Tensor*,
                                                                const tensorflow::Tensor*,
                                                                const std::string&,
                                                                const bool,
                                                                std::shared_ptr<InputSpace>&);
    std::map<std::string, distribute_keys_gpu_func_type> distribute_keys_on_gpu_func_;

    tensorflow::Status set_embedding_params(const std::string& name, const HugeCTR::Embedding_t& embedding_type, 
                const HugeCTR::Optimizer_t& optimizer_type,
                const size_t& max_vocabulary_size_per_gpu, const std::vector<size_t>& slot_size_array,
                const std::vector<float>& opt_hparams, const HugeCTR::Update_t& update_type, const bool atomic_update,
                const float& scaler, const size_t& slot_num, const size_t& max_nnz, 
                const size_t& max_feature_num, const size_t& embedding_vec_size, const int& combiner);
    std::shared_ptr<EmbeddingParams> get_embedding_params(const std::string& name);
    tensorflow::Status register_input_space(const HugeCTR::Embedding_t& embedding_type,
                        const size_t& slot_num, const size_t& batchsize, const size_t& max_nnz,
                        const size_t& max_feature_num, const std::string& embedding_name, const std::string& space_name);
    std::shared_ptr<InputSpace> get_input_space(const std::string& space_name);
    std::shared_ptr<IEmbedding> get_embedding(const std::string& embedding_name);
    std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> get_buff(const std::string& embedding_name);

    template <typename Item>
    Item get_item_from_map(const std::map<std::string, Item>& map, 
                            const std::string& map_key);

    tensorflow::Status distribute_keys(const tensorflow::Tensor* sparse_indices, 
            const tensorflow::Tensor* values, const tensorflow::Tensor* dense_shape, 
            const std::string& embedding_name, const bool is_training,
            const HugeCTR::Embedding_t& embedding_type, const bool on_gpu); // distribute input tensors for each GPU as CSR format.
    
    /*helper functions for distribute keys on GPU*/
    tensorflow::Status distribute_keys_gpu_distributed(const tensorflow::Tensor* row_indices,
                                                       const tensorflow::Tensor* values,
                                                       const std::string& embedding_name,
                                                       const bool is_training,
                                                       std::shared_ptr<InputSpace>& input_space);
    tensorflow::Status distribute_keys_gpu_localized(const tensorflow::Tensor* row_indices,
                                                    const tensorflow::Tensor* values,
                                                    const std::string& embedding_name,
                                                    const bool is_training,
                                                    std::shared_ptr<InputSpace>& input_space); 
    
    tensorflow::Status save_initial_to_file(const std::string& embedding_name, 
                            const tensorflow::Tensor* const init_value, const std::string& save_name, const bool on_gpu);
    tensorflow::Status get_event(const unsigned int dev_id, cudaEvent_t& event);
    
private:
    long long batch_size_;
    long long batch_size_eval_;
    ncclDataType_t nccl_type_;
}; // class EmbeddingWrapper

} // namespace Version1
} // namespace HugeCTR


#endif 