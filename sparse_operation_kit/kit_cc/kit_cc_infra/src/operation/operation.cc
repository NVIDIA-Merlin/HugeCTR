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

#include "operation/operation.h"
#include "common.h"

namespace SparseOperationKit {

Operation::Operation(ConstructionContext_t context)
: base_context_(context) 
{}

void Operation::AllocateForwardSpaces(size_t const global_batch_size) {
    allocate_forward_spaces(global_batch_size);
    if (next_op_) next_op_->AllocateForwardSpaces(global_batch_size);
}

void Operation::AllocateBackwardSpaces(size_t const global_batch_size) {
    allocate_backward_spaces(global_batch_size);
    if (next_op_) next_op_->AllocateBackwardSpaces(global_batch_size);
}

void Operation::Forward(const Context_t &replica_context, const bool training) {
    forward(replica_context, training);
    if (next_op_) next_op_->Forward(replica_context, training);
}

void Operation::Backward(const Context_t &replica_context) {
    if (next_op_) next_op_->Backward(replica_context);
    backward(replica_context);
}

void Operation::set_next(std::shared_ptr<Operation> operation) {
    if (nullptr == next_op_) { // next_op_ is invalid
        next_op_ = operation;
        return;
    } else { // next_op_ is valid, then link it to its next_op
        return next_op_->set_next(operation);
    }
}

ConstructionContext_t Operation::base_context() const {
    return base_context_;
}

} // namespace SparseOperationKit