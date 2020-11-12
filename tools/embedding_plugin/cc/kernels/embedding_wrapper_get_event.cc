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

template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::get_event(const unsigned int dev_id, 
                                                                cudaEvent_t& event) {
    if (dev_id >= events_.size()) {
        return tensorflow::errors::OutOfRange(__FILE__, ":", __LINE__, " ",
                                              "dev_id is out of range of valid events.");
    }

    event = events_[dev_id];
    return tensorflow::Status::OK();
}

template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::get_events(std::vector<cudaEvent_t>& events) {
    if (0 == events_.size()) {
        return tensorflow::errors::Unavailable(__FILE__, ":", __LINE__, " ",
                                              "wrapper events.size == 0");
    }
    events = events_;
    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::get_event(const unsigned int, cudaEvent_t&);
template tensorflow::Status EmbeddingWrapper<long long, __half>::get_event(const unsigned int, cudaEvent_t&);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::get_event(const unsigned int, cudaEvent_t&);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::get_event(const unsigned int, cudaEvent_t&);

template tensorflow::Status EmbeddingWrapper<long long, float>::get_events(std::vector<cudaEvent_t>&);
template tensorflow::Status EmbeddingWrapper<long long, __half>::get_events(std::vector<cudaEvent_t>&);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::get_events(std::vector<cudaEvent_t>&);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::get_events(std::vector<cudaEvent_t>&);

} // namespace Version1
} // namespace HugeCTR