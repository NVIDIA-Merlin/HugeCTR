#include "embedding_wrapper.h"
#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_one_hot.hpp"

#ifdef PLUGIN_NVTX
#include <nvToolsExt.h> 
#endif

namespace HugeCTR {
namespace Version2 {

template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::fprop_v2(
            const std::string& embedding_name, const bool is_training, 
            const tensorflow::Tensor* replica_id, const cudaStream_t& tf_stream, 
            tensorflow::Tensor* replica_forward_result) {
#ifdef PLUGIN_NVTX
    nvtxRangeId_t fprop_id = nvtxRangeStartA("plugin_fprop");
#endif
    /*get host replica_id*/
    size_t gpu_count = resource_manager_->get_local_gpu_count();
    int host_replica_id = 0;
    WRAPPER_CUDA_CHECK(cudaMemcpyAsync(&host_replica_id, replica_id->data(), sizeof(int) * 1,
                    cudaMemcpyDeviceToHost, tf_stream));
    WRAPPER_CUDA_CHECK(cudaStreamSynchronize(tf_stream));
    if (host_replica_id < 0 || host_replica_id >= static_cast<long long>(gpu_count)) 
                    return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                        "replica_id should be in range of [0, gpu_count)");

    /*do forward propagation once*/
    try {
        call_once(&EmbeddingWrapper<TypeKey, TypeFP>::forward_helper, this, embedding_name, is_training);
    } catch (const std::exception& error) {
        return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                error.what());
    }

    /*get forward result*/
    WRAPPER_REQUIRE_OK(copy_from_output_tensor(embedding_name, host_replica_id, is_training, replica_forward_result));

    /*synchronize plugin stream with tf stream*/
    std::vector<cudaEvent_t> fprop_events = get_item_from_map(fprop_events_, embedding_name);
    if (fprop_events.empty()) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                    "Cannot find fprop cudaEvent_t for embedding: ", embedding_name);
    WRAPPER_CUDA_CHECK(cudaStreamWaitEvent(tf_stream, fprop_events[host_replica_id], 0));

#ifdef PLUGIN_NVTX
    nvtxRangeEnd(fprop_id);
#endif
    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::fprop_v2(
            const std::string& embedding_name, const bool is_training, 
            const tensorflow::Tensor* replica_id, const cudaStream_t& tf_stream, 
            tensorflow::Tensor* replica_forward_result);
template tensorflow::Status EmbeddingWrapper<long long, __half>::fprop_v2(
            const std::string& embedding_name, const bool is_training, 
            const tensorflow::Tensor* replica_id, const cudaStream_t& tf_stream, 
            tensorflow::Tensor* replica_forward_result);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::fprop_v2(
            const std::string& embedding_name, const bool is_training, 
            const tensorflow::Tensor* replica_id, const cudaStream_t& tf_stream, 
            tensorflow::Tensor* replica_forward_result);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::fprop_v2(
            const std::string& embedding_name, const bool is_training, 
            const tensorflow::Tensor* replica_id, const cudaStream_t& tf_stream, 
            tensorflow::Tensor* replica_forward_result);

} // namespace Version2 
} // namespace HugeCTR