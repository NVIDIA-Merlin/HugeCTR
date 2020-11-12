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
                    tensorflow::Tensor* const forward_result) = 0;
    virtual tensorflow::Status get_output_tensor_shape(const std::string& embedding_name, const bool is_training,
                                                tensorflow::TensorShape& shape) = 0;
    virtual tensorflow::Status bprop(const std::string& embedding_name, const tensorflow::Tensor* top_gradients,
                                     const bool on_gpu) = 0;
    virtual tensorflow::Status try_allocate_distributing_spaces(const std::string& space_name, 
                                                                const HugeCTR::Embedding_t& embedding_type,
                                                                const long long& batch_size,
                                                                const long long& slot_num,
                                                                const long long& max_nnz) = 0;
    virtual tensorflow::Status do_distributing_keys(const std::string& space_name, 
                                        const tensorflow::Tensor* input_keys,
                                        std::vector<tensorflow::Tensor*>& row_offset_output,
                                        std::vector<tensorflow::Tensor*>& value_tensor_output,
                                        tensorflow::Tensor* nnz_array_output) = 0;
    virtual tensorflow::Status save(const std::string& embedding_name, const std::string& save_name) = 0;
    virtual tensorflow::Status restore(const std::string& embedding_name, const std::string& file_name) = 0;
    virtual tensorflow::Status get_events(std::vector<cudaEvent_t>& events) = 0;
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

    /*This class is a wrapper for spaces used in the process of distributing keys.*/
    struct DistributeKeysSpaces{
        DistributeKeysSpaces(const size_t gpu_count); 
        ~DistributeKeysSpaces();
        bool allocated_; // flag, indicate whether those spaces have been allocated GPU spaces.
        size_t gpu_count_;
        long long batch_size_;
        long long slot_num_;
        long long max_nnz_;
        HugeCTR::Embedding_t embedding_type_;

        std::vector<TypeKey*, CudaUtils::CudaAllocator<TypeKey*>> input_keys_copies_;
        std::vector<cusparseHandle_t> cusparse_handles_;
        std::vector<cublasHandle_t> cublas_handles_;

        std::vector<TypeKey*, CudaUtils::CudaAllocator<TypeKey*>> csr_values_;
        std::vector<int*, CudaUtils::CudaAllocator<int*>> csr_row_offsets_;
        std::vector<long long*, CudaUtils::CudaAllocator<long long*>> csr_row_offsets_casts_;
        std::vector<int*, CudaUtils::CudaAllocator<int*>> csr_col_indices_;
        std::vector<int*, CudaUtils::CudaAllocator<int*>> csr_nnz_rows_;
        std::vector<TypeKey*, CudaUtils::CudaAllocator<TypeKey*>> input_keys_transposes_;
        std::vector<cusparseMatDescr_t> cusparse_mat_descs_;
        std::vector<long long> total_nnzs_;
        std::vector<cudaStream_t> cuda_streams_;
    };

    struct DoDistributeKeysFunctor {
        virtual tensorflow::Status operator()(EmbeddingWrapper<TypeKey, TypeFP>* const wrapper,
                                            std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
                                            const tensorflow::Tensor* input_keys,
                                            std::vector<tensorflow::Tensor*> row_offset_output,
                                            std::vector<tensorflow::Tensor*> value_tensor_output,
                                            tensorflow::Tensor* nnz_array_output) = 0;
    };

    struct DoDistributedDistributeKeysFunctor : public DoDistributeKeysFunctor {
        tensorflow::Status operator()(EmbeddingWrapper<TypeKey, TypeFP>* const wrapper,
                                    std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
                                    const tensorflow::Tensor* input_keys,
                                    std::vector<tensorflow::Tensor*> row_offset_output,
                                    std::vector<tensorflow::Tensor*> value_tensor_output,
                                    tensorflow::Tensor* nnz_array_output) override;
    };

    struct DoLocalizedDistributeKeysFunctor : public DoDistributeKeysFunctor {
        tensorflow::Status operator()(EmbeddingWrapper<TypeKey, TypeFP>* const wrapper,
                                    std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
                                    const tensorflow::Tensor* input_keys,
                                    std::vector<tensorflow::Tensor*> row_offset_output,
                                    std::vector<tensorflow::Tensor*> value_tensor_output,
                                    tensorflow::Tensor* nnz_array_output) override;
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
                    tensorflow::Tensor* const forward_result) override;
    tensorflow::Status get_output_tensor_shape(const std::string& embedding_name, const bool is_training,
                                                tensorflow::TensorShape& shape) override;

    tensorflow::Status bprop(const std::string& embedding_name, const tensorflow::Tensor* top_gradients,
                             const bool on_gpu) override;
    tensorflow::Status try_allocate_distributing_spaces(const std::string& space_name, 
                                                        const HugeCTR::Embedding_t& embedding_type,
                                                        const long long& batch_size,
                                                        const long long& slot_num,
                                                        const long long& max_nnz) override;
    tensorflow::Status do_distributing_keys(const std::string& space_name, const tensorflow::Tensor* input_keys,
                                            std::vector<tensorflow::Tensor*>& row_offset_output,
                                            std::vector<tensorflow::Tensor*>& value_tensor_output,
                                            tensorflow::Tensor* nnz_array_output) override;
    tensorflow::Status save(const std::string& embedding_name, const std::string& save_name) override;
    tensorflow::Status restore(const std::string& embedding_name, const std::string& file_name) override;
    tensorflow::Status get_events(std::vector<cudaEvent_t>& events) override;

    void evaluate();
    
private:
    std::map<std::string, std::shared_ptr<IEmbedding>> embeddings_; // <embedding_instance_name, embedding>
    std::map<std::string, std::shared_ptr<InputSpace>> input_spaces_; // <input_space_name, input_space>
    std::shared_ptr<HugeCTR::ResourceManager> resource_manager_; 
    std::map<std::string, std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>> buffs_; // <embedding_instance_name, buff>
    std::map<std::string, std::shared_ptr<EmbeddingParams>> embedding_params_; // <embedding_instance_name, EmbeddingParams>
    std::map<std::string, std::shared_ptr<DistributeKeysSpaces>> distribute_keys_spaces_; // <space_name, DistributeKeysSpaces>
    std::unique_ptr<DoDistributeKeysFunctor> do_distribute_keys_functor_; // this should have multiple instance ??
    std::unique_ptr<CudaUtils::ConvertDenseToCSRFunctor> convert_dense_to_csr_functor_;
    std::vector<cudaEvent_t> events_; // events for each GPU

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
    tensorflow::Status distribute_keys(const tensorflow::Tensor* sparse_indices, 
            const tensorflow::Tensor* values, const tensorflow::Tensor* dense_shape, 
            const std::string& embedding_name, const bool is_training,
            const HugeCTR::Embedding_t& embedding_type, const bool on_gpu); // distribute input tensors for each GPU as CSR format.
    tensorflow::Status save_initial_to_file(const std::string& embedding_name, 
                            const tensorflow::Tensor* const init_value, const std::string& save_name, const bool on_gpu);
    std::shared_ptr<DistributeKeysSpaces> get_distribute_keys_spaces(const std::string& embedding_name);
    tensorflow::Status allocate_distribute_keys_spaces_helper(const std::string& space_name, 
                    const HugeCTR::Embedding_t& embedding_type, const long long& batch_size, 
                    const long long& slot_num, const long long& max_nnz);
    tensorflow::Status distributed_embedding_distribute_keys_helper(
                                                        std::shared_ptr<DistributeKeysSpaces>& distribute_keys_space,
                                                        const tensorflow::Tensor* input_keys,
                                                        std::vector<tensorflow::Tensor*>& row_offset_output,
                                                        std::vector<tensorflow::Tensor*>& value_tensor_output,
                                                        tensorflow::Tensor* nnz_array_output);
    tensorflow::Status localized_embedding_distribute_keys_helper();
    tensorflow::Status get_event(const unsigned int dev_id, cudaEvent_t& event);
    
private:
    long long batch_size_;
    long long batch_size_eval_;
}; // class EmbeddingWrapper

} // namespace Version1
} // namespace HugeCTR


#endif 