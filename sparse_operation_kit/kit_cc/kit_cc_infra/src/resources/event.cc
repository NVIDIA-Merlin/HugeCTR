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

#include "resources/event.h"
#include "common.h"

namespace SparseOperationKit {

Event::Event() 
: mu_(), in_use_(true)
{
    CK_CUDA(cudaEventCreateWithFlags(&cuda_event_, cudaEventDisableTiming));
}

Event::~Event() {
    try {
        CK_CUDA(cudaEventDestroy(cuda_event_));
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
    }
}

std::shared_ptr<Event> Event::create() {
    return std::shared_ptr<Event>(new Event());
}

void Event::Record(cudaStream_t& stream) {
    CK_CUDA(cudaEventRecord(cuda_event_, stream));
}

bool Event::IsReady() const {
    cudaError_t error = cudaEventQuery(cuda_event_);
    if (cudaSuccess == error) {
        return true;
    } else if (cudaErrorNotReady == error) {
        return false;
    } else {
        CK_CUDA(error);
        return false;
    }
}

void Event::TillReady(cudaStream_t& stream) {
    CK_CUDA(cudaStreamWaitEvent(stream, cuda_event_, cudaEventWaitDefault));
    std::lock_guard<std::mutex> lock(mu_);
    in_use_ = false;
}

void Event::TillReady() {
    CK_CUDA(cudaEventSynchronize(cuda_event_));
    std::lock_guard<std::mutex> lock(mu_);
    in_use_ = false;
}

bool Event::IsInUse() const {
    std::lock_guard<std::mutex> lock(mu_);    
    return in_use_ || !IsReady();
}

void Event::Reset() {
    std::lock_guard<std::mutex> lock(mu_);
    in_use_ = true;
}

} // namespace SparseOperationKit